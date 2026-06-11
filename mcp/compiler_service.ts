/**
 * compiler_service.ts — the TS compiler behind the Lean engine.
 *
 * Phase 2 of the Lean port: this service is **stateless pure compile**.
 * The Lean engine owns the session, the native runtime (plan loading,
 * slot control), the DAC, and params; this service owns what hasn't
 * been ported yet — `partitionKernel` to tropical_plan_5 JSON (the
 * Lean engine runs the session lowering, the elaboration, AND — Phase
 * 5 stage 6b — every production strata call; this side decodes shipped
 * `tropical_resolved_1` payloads and partitions/registers them), and
 * save/export/load/merge (they need the v2 ingest and the compiler).
 * It shrinks phase by phase until Phase 6 deletes it.
 *
 * Phase 4 stage 4b + Phase 5 stage 6b: the production paths perform NO
 * raise, NO elaboration, and NO strata here. The Lean engine raises,
 * elaborates, relinks, and stratas, driving registration one program
 * at a time — `register_program` receives `{name, parsed, resolved}`
 * (ParsedProgram JSON + encoded POST-STRATA resolved IR for concrete
 * programs; raw templates for generics) and shrinks to typeDef
 * registration + decode + makeCompiled; `resolve_type` receives
 * engine-specialized resolved per `Type<N=8>` key; `register_lifted`
 * receives the engine's lifted `__wire_N` programs (raw + post-strata);
 * `compile` receives per-instance resolved and instantiates verbatim.
 * The service also boots EMPTY: the engine registers the stdlib from
 * the pre-parsed bridge (`stdlib/parsed/`) through register_program.
 *
 * Protocol: newline JSON-RPC on stdio, same framing as ir_service.ts.
 * Tool-level failures return `{result: {error: <ErrorEnvelope>}}` so the
 * Lean side maps them onto its own envelope type; JSON-RPC `error` is
 * reserved for transport/internal faults.
 *
 * The one stateful contract: `compile` rebuilds this service's session
 * from the Lean engine's authoritative snapshot (instances + canonical
 * pre-extraction wires + outputs + params), compiles it, and returns
 * the plan JSON string for the Lean side to load over its own FFI.
 * Every other graph-reading method (`save`, `export_program`) operates
 * on the session as of the last `compile`/`load`/`merge`.
 */

import { readFileSync } from 'node:fs'
import {
  makeSession, loadJSON, instantiate,
  reconstructWireDelays,
  normalizeProgramFile, v2NodeToFile,
  inputNames, outputNames, registerNames,
  inputPortTypes, outputPortTypes, registerPortTypes,
  rawInputDefaults,
  type SessionState, type ExprNode, type Instance,
} from '../compiler/session.js'
import {
  loadProgramAsType, exportSessionAsProgram, saveProgramFromSession,
  mergeProgramIntoSession, instanceDecls,
} from '../compiler/program.js'
import { compileSession } from '../compiler/ir/compile_session.js'
import { compileSessionSlottedFromResolved } from '../compiler/ir/compile_session_slotted.js'
import { decodeResolved, encodeResolved } from '../compiler/ir/resolved_codec.js'
import { makeCompiled } from '../compiler/program_types.js'
import { assertSessionAcyclic } from '../compiler/ir/lowering/session_cycle_check.js'
import { toWirePlan } from '../compiler/flat_plan.js'
import { Param } from '../compiler/runtime/param.js'
import { portTypeToString } from '../compiler/ir/port_type.js'
import { parseWireKey, wireKey } from '../compiler/ir/branded_names.js'
import type { PortType, ResolvedProgram } from '../compiler/ir/nodes.js'
import { failBare, toEnvelope, type ErrorEnvelope } from './envelope.js'
import { RESOURCES, PROMPTS, readResourceText, getPromptMessages } from './resources.js'

// Phase 4 stage 4b: the service boots EMPTY. The Lean engine drives all
// registration — stdlib (from the pre-parsed bridge) and define_program
// alike — through `register_program`.
const session: SessionState = makeSession()

const portTypeOrNull = (t: PortType | undefined): string | null =>
  t === undefined ? null : portTypeToString(t)

// ─── Catalog entries ─────────────────────────────────────────────────────────
// The port-metadata shape the Lean engine mirrors. `type` is the display
// string (list_programs), `type_obj` the structured PortType (get_info echo
// + wire type-checking). `resolved` is the program's post-strata IR in
// `tropical_resolved_1` encoding — the Lean engine adopts it into its typed
// store so its own `compile` elaboration can LINK instances against it
// (Phase 4 stage 4a). Generic templates carry `resolved: null` (the engine
// only needs concrete specializations, which arrive via `resolve_type`).

function concreteEntry(name: string, type: Instance | NonNullable<ReturnType<typeof loadProgramAsType>>) {
  const defaultsMap = rawInputDefaults(type) as Record<string, unknown>
  const ipt = inputPortTypes(type)
  const opt = outputPortTypes(type)
  const rpt = registerPortTypes(type)
  const prog = 'compiled' in type ? type.compiled.prog : type.prog
  return {
    program_name: name,
    generic: false,
    type_params: null,
    resolved: encodeResolved(prog),
    inputs: inputNames(type).map((n, i) => ({
      name: n, type: portTypeOrNull(ipt[i]), type_obj: ipt[i] ?? null, default: defaultsMap[n] ?? null,
    })),
    outputs: outputNames(type).map((n, i) => ({
      name: n, type: portTypeOrNull(opt[i]), type_obj: opt[i] ?? null,
    })),
    registers: registerNames(type).map((n, i) => ({
      name: n, type: portTypeOrNull(rpt[i]), type_obj: rpt[i] ?? null,
    })),
  }
}

function genericEntry(name: string, prog: ResolvedProgram) {
  return {
    program_name: name,
    generic: true,
    resolved: null,
    type_params: Object.fromEntries(prog.typeParams.map(tp => {
      const entry: { type: 'int'; default?: number } = { type: 'int' }
      if (tp.default !== undefined) entry.default = tp.default
      return [tp.name, entry]
    })),
    inputs: prog.ports.inputs.map(i => ({ name: i.name, type: null, type_obj: null, default: null })),
    outputs: prog.ports.outputs.map(o => ({ name: o.name, type: null, type_obj: null })),
    registers: [] as unknown[],
  }
}

/** Entry for a registered program by name (concrete or generic). */
function entryFor(name: string) {
  const concrete = session.typeRegistry.get(name)
  if (concrete) return concreteEntry(name, concrete)
  const prog = session.programs.get(name)
  if (prog && prog.typeParams.length > 0) return genericEntry(name, prog)
  if (prog) return undefined // concrete prog without typeRegistry pair — loader invariant violation
  return undefined
}

/** Names registered since `before`, plus the explicitly (re)defined names. */
function newEntries(before: ReadonlySet<string>, defined: string[]) {
  const names = [...session.programs.keys()].filter(k => !before.has(k))
  for (const d of defined) if (!names.includes(d) && session.programs.has(d)) names.push(d)
  return names.map(entryFor).filter(e => e !== undefined)
}

// ─── State dump (after load / merge) ─────────────────────────────────────────
// Lean adopts this as its authoritative session mirror. Wires are emitted in
// canonical authored form (delay() restored from the slot registry).

function dumpState(beforePrograms: ReadonlySet<string>) {
  return {
    instances: [...session.instanceRegistry.entries()].map(([name, inst]) => ({
      name,
      program: inst.baseTypeName,
      type_args: inst.typeArgs ?? null,
      entry: concreteEntry(inst.baseTypeName, inst),
    })),
    wires: [...session.inputExprNodes.entries()].map(([key, expr]) => ({
      key,
      expr: reconstructWireDelays(expr, session.delaySlotRegistry),
    })),
    graph_outputs: session.graphOutputs.map(o => ({ instance: o.instance, output: o.output })),
    params: [...session.paramRegistry.entries()].map(([name, p]) => ({ name, value: p.value })),
    entries: newEntries(beforePrograms, []),
  }
}

// ─── Method handlers ─────────────────────────────────────────────────────────

type CompilePayload = {
  /** The session's synthetic root, elaborated by the Lean engine and
   *  shipped as `tropical_resolved_1` JSON — the Phase 4 stage-4a seam
   *  payload (the elaborate step moved engine-side; this service just
   *  decodes and partitions). */
  resolved_root: unknown
  /** Each instance carries its post-strata resolved IR (Phase 5 stage
   *  6b) — the engine's adopted snapshot, instantiated here verbatim;
   *  no name-based re-resolution and no strata on this path. */
  instances: Array<{
    name: string
    program: string
    type_args?: Record<string, number> | null
    resolved: unknown
  }>
  /** Post-extraction wires — consumed by the input-alias checks and the
   *  defensive acyclicity tripwire, not re-lowered. */
  wires: Array<{ key: string; expr: ExprNode }>
  graph_outputs: Array<{ instance: string; output: string }>
  params: Array<{ name: string; value: number }>
  /** Slot bookkeeping the Lean lowering allocated (canonical order:
   *  params, then instance outputs, then extraction delay slots). */
  slot_count: number
  param_slots: Array<[string, number]>
  output_slots: Array<[string, number]>
  output_port_meta: Array<[string, unknown]>
  io_array: { count: number; sizes: number[]; names: string[] }
  delay_slots: unknown[]
}

function handleCompile(p: CompilePayload) {
  // Reset graph state; keep type registries (programs persist across
  // compiles) and paramRegistry (live Param handles).
  session.instanceRegistry.clear()
  session.inputExprNodes.clear()
  session.graphOutputs.length = 0
  session.outputSlotRegistry.clear()
  session.paramSlotRegistry.clear()
  session.outputPortMeta.clear()
  session.inputSlotRegistry.clear()
  session.inputPortMeta.clear()
  session.inputExprs.clear()

  for (const inst of p.instances) {
    const type = makeCompiled(decodeResolved(inst.resolved))
    const instance = instantiate(type, inst.name, {
      baseTypeName: inst.program,
      typeArgs: inst.type_args ?? undefined,
    })
    session.instanceRegistry.set(inst.name, instance)
  }
  for (const w of p.wires) {
    session.inputExprNodes.set(wireKey(parseWireKey(w.key)), w.expr)
  }
  for (const o of p.graph_outputs) session.graphOutputs.push({ instance: o.instance, output: o.output })
  for (const pr of p.params) {
    // The Lean engine's mirror is authoritative — adopt its values so
    // slot_defaults (and the params echo) reflect set_param calls made
    // since the last compile.
    const existing = session.paramRegistry.get(pr.name)
    if (existing) existing.value = pr.value
    else session.paramRegistry.set(pr.name, new Param(pr.value))
  }

  // Adopt the Lean lowering's slot allocation verbatim. partitionKernel
  // continues allocation from slot_count for anything the mirror cannot
  // see (nested fractal ports).
  session.slotCount = p.slot_count
  for (const [name, idx] of p.param_slots) session.paramSlotRegistry.set(name, idx)
  for (const [key, idx] of p.output_slots) {
    session.outputSlotRegistry.set(key as never, idx)
  }
  for (const [key, meta] of p.output_port_meta) {
    session.outputPortMeta.set(key as never, meta as never)
  }
  session.ioArraySlotCount = p.io_array.count
  session.ioArraySlotSizes = [...p.io_array.sizes]
  session.ioArraySlotNames = [...p.io_array.names]
  session.delaySlotRegistry = p.delay_slots as never

  // Defensive tripwire (the Lean lowering already checked).
  assertSessionAcyclic(session)

  // Decode the Lean-elaborated root. A malformed resolved_root is an
  // engine bug, not a user error — `decodeResolved` throws
  // `ResolvedCodecError`, which the stdio loop's `toEnvelope` maps to
  // `internal_error` with the decoder's message.
  const root = decodeResolved(p.resolved_root)
  const plan = compileSessionSlottedFromResolved(session, root, 'fused')
  const planJson = JSON.stringify(toWirePlan(plan))
  return {
    plan_json: planJson,
    params: [...session.paramRegistry.entries()].map(([name, prm]) => ({ name, value: prm.value })),
  }
}

/** Phase 5 stage 6b: the engine lifts array-literal wires to anonymous
 *  `__wire_N` programs itself (build + strata + wire rewrite,
 *  `lean/Tropical/Ir/WireProgram.lean`); this residue mirrors exactly
 *  what `liftWiresToInstances` stored in the registries: the
 *  post-strata Compiled in typeRegistry, and the RAW lifted resolved
 *  in `session.programs` — the form `export_program`'s
 *  loadProgramAsType resolver expects to elaborate against. The
 *  instance itself arrives with the next `compile` payload. */
function handleRegisterLifted(p: { name: string; raw: unknown; resolved: unknown }) {
  const compiled = makeCompiled(decodeResolved(p.resolved))
  session.typeRegistry.set(p.name, compiled)
  session.programs.set(p.name, decodeResolved(p.raw))
  return { entry: concreteEntry(p.name, compiled) }
}

// ─── register_program (Phase 4 stage 4b; strata engine-side at 6b) ───────────
// One registration per call, driven by the Lean engine in batch order
// (nested programDecls depth-first, the root last). The engine already
// raised, elaborated, relinked, AND ran strata (Phase 5 stage 6b);
// this side performs only the registry residue: typeDef registry
// mutations, decode, `makeCompiled`, registry inserts.

/** The ParsedProgram wire shape of `ports.type_defs` (alias base is a
 *  NameRef node there, unlike the v2 string form). */
type ParsedTypeDefs = {
  ports?: {
    type_defs?: Array<
      | { kind: 'alias'; name: string; base: { op: 'nameRef'; name: string } }
      | { kind: 'sum'; name: string; variants: Array<{
          name: string
          payload: Array<{ name: string; scalar_type: 'float' | 'int' | 'bool' }>
        }> }
      | { kind: 'struct'; name: string; fields: Array<{
          name: string; scalar_type: 'float' | 'int' | 'bool'
        }> }
    >
  }
}

function handleRegisterProgram(p: { name: string; parsed: ParsedTypeDefs; resolved: unknown }) {
  // TypeDef registration — the registry mutations loadProgramAsType
  // performed, fed from the ParsedProgram wire shape.
  for (const td of p.parsed.ports?.type_defs ?? []) {
    if (td.kind === 'alias') {
      session.typeAliasRegistry.set(td.name, { base: td.base.name })
    } else if (td.kind === 'sum') {
      session.sumTypeRegistry.set(td.name, {
        name: td.name,
        variants: td.variants.map(v => ({
          name: v.name,
          payload: v.payload.map(f => ({ name: f.name, scalar: f.scalar_type })),
        })),
      })
    } else {
      session.structTypeRegistry.set(td.name, {
        fields: td.fields.map(f => ({ name: f.name, scalar: f.scalar_type })),
      })
    }
  }

  const decoded = decodeResolved(p.resolved)
  if (decoded.typeParams.length > 0) {
    // Generic template: stored raw, exactly as loadProgramAsType /
    // loadStdlibFromResolved stash it (the engine ships generics
    // unstrata'd and nothing ever relinks them).
    session.programs.set(p.name, decoded)
  } else {
    // Phase 5 stage 6b: the engine ships POST-STRATA resolved — it ran
    // the relink (loadStdlibFromResolved's `processedByName` step) and
    // the strata pipeline itself. Wrap verbatim.
    const type = makeCompiled(decoded)
    session.typeRegistry.set(p.name, type)
    session.programs.set(p.name, type.prog)   // canonical post-strata
  }
  return { entry: entryFor(p.name) ?? null }
}

function handleMethod(method: string, params: Record<string, unknown>): unknown {
  switch (method) {
    case 'boot':
      // Handshake only — the engine boots the stdlib itself from the
      // pre-parsed bridge, through register_program.
      return {}

    case 'register_program':
      return handleRegisterProgram(params as unknown as Parameters<typeof handleRegisterProgram>[0])

    case 'resolve_type': {
      // Phase 5 stage 6b: the engine resolved the type args and ran
      // the specialization (strata) itself; this side decodes, caches
      // the Compiled under the engine's key, and renders the catalog
      // entry (port metadata stays service-side until Phase 6).
      const key = params.key as string
      const type = makeCompiled(decodeResolved(params.resolved), { displayName: key })
      session.specializationCache.set(key, type)
      return { key, type_args: params.type_args ?? null, entry: concreteEntry(key, type) }
    }

    case 'compile':
      return handleCompile(params as unknown as CompilePayload)

    case 'register_lifted':
      return handleRegisterLifted(params as unknown as Parameters<typeof handleRegisterLifted>[0])

    case 'save': {
      const { node, topLevel } = saveProgramFromSession(session)
      return { program: v2NodeToFile(node as unknown as ExprNode, topLevel) }
    }

    case 'export_program': {
      const name = params.name as string
      const exportedNode = exportSessionAsProgram(session, {
        name,
        inputs: (params.inputs ?? {}) as Record<string, string>,
        outputs: params.outputs as Record<string, { instance: string; output: string }>,
      })
      const exportedInstanceNames = [...instanceDecls(exportedNode)].map(d => d.name)
      const before = new Set(session.programs.keys())
      const type = loadProgramAsType(exportedNode, session)
      if (!type) {
        failBare({ code: 'internal_error', message: 'export_program produced a generic program — not supported' })
      }
      session.typeRegistry.set(name, type)
      return {
        program: v2NodeToFile(exportedNode as unknown as ExprNode),
        exported_instances: exportedInstanceNames,
        inputs: inputNames(type),
        outputs: outputNames(type),
        entries: newEntries(before, [name]),
      }
    }

    case 'load': {
      let raw: unknown
      if (params.path) {
        raw = JSON.parse(readFileSync(params.path as string, 'utf-8'))
      } else {
        raw = params.program
      }
      const before = new Set(session.programs.keys())
      const t0 = performance.now()
      loadJSON(raw as { schema: string }, session)
      const wall_ms = performance.now() - t0
      // Re-emit the compiled plan for the Lean side to load over its own
      // FFI (loadJSON compiled into this service's inert runtime; the
      // session is post-extraction, so this recompile is idempotent).
      const planJson = JSON.stringify(toWirePlan(compileSession(session)))
      return { state: dumpState(before), plan_json: planJson, timing: { wall_ms } }
    }

    case 'merge': {
      const { node, topLevel } = normalizeProgramFile(params.program as { schema?: string; [k: string]: unknown })
      const before = new Set(session.programs.keys())
      mergeProgramIntoSession(node, topLevel, session)
      const planJson = JSON.stringify(toWirePlan(compileSession(session)))
      return { state: dumpState(before), plan_json: planJson }
    }

    // ── MCP non-tool surface ────────────────────────────────────────────────

    case 'resources/list': return { resources: RESOURCES }
    case 'resources/read': return { text: readResourceText(String(params.uri ?? ''), session) }
    case 'prompts/list':   return { prompts: PROMPTS }
    case 'prompts/get':    return getPromptMessages(String(params.name ?? ''))

    default:
      throw new Error(`compiler_service: unknown method '${method}'`)
  }
}

// ─── Stdio loop (same framing as ir_service.ts) ──────────────────────────────

function send(obj: unknown): void {
  process.stdout.write(JSON.stringify(obj) + '\n')
}

process.stdin.setEncoding('utf8')
let buf = ''

for await (const chunk of process.stdin) {
  buf += String(chunk)
  let nl: number
  while ((nl = buf.indexOf('\n')) !== -1) {
    const line = buf.slice(0, nl).trim()
    buf = buf.slice(nl + 1)
    if (line === '') continue

    let id: unknown = null
    try {
      const req = JSON.parse(line) as { id?: unknown; method: string; params?: unknown }
      id = req.id ?? null
      let result: unknown
      try {
        result = handleMethod(req.method, (req.params ?? {}) as Record<string, unknown>)
      } catch (e) {
        result = { error: toEnvelope(e) satisfies ErrorEnvelope }
      }
      send({ jsonrpc: '2.0', id, result })
    } catch (e) {
      send({
        jsonrpc: '2.0',
        id,
        error: { code: -32603, message: e instanceof Error ? e.message : String(e) },
      })
    }
  }
}
