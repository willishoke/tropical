/**
 * compiler_service.ts — the TS compiler behind the Lean engine.
 *
 * Phase 2 of the Lean port: this service is **stateless pure compile**.
 * The Lean engine owns the session, the native runtime (plan loading,
 * slot control), the DAC, and params; this service owns what hasn't
 * been ported yet — program registration (raise → elaborate → strata),
 * compilation to tropical_plan_5 JSON (Phase 3: from the Lean engine's
 * own session lowering — a ParsedProgram + slot bookkeeping; this side
 * just elaborates and partitions), wire lifting (it needs strata), and
 * save/export/load/merge (they need the compiler). It shrinks phase by
 * phase until Phase 6 deletes it.
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
  makeSession, loadJSON, resolveProgramType, instantiate,
  allocateOutputSlots, setWireExpr, reconstructWireDelays,
  normalizeProgramFile, v2NodeToFile,
  inputNames, outputNames, registerNames,
  inputPortTypes, outputPortTypes, registerPortTypes,
  rawInputDefaults,
  type SessionState, type ExprNode, type Instance,
} from '../compiler/session.js'
import {
  loadProgramAsType, exportSessionAsProgram, saveProgramFromSession,
  mergeProgramIntoSession, loadStdlib, instanceDecls,
} from '../compiler/program.js'
import { compileSession } from '../compiler/ir/compile_session.js'
import { compileSessionSlottedFromParsed } from '../compiler/ir/compile_session_slotted.js'
import { liftWiresToInstances } from '../compiler/ir/lift_wires.js'
import { assertSessionAcyclic } from '../compiler/ir/lowering/session_cycle_check.js'
import { toWirePlan } from '../compiler/flat_plan.js'
import { Param } from '../compiler/runtime/param.js'
import { portTypeToString } from '../compiler/ir/port_type.js'
import { parseWireKey, wireKey, portRef, instanceName as toInstanceName } from '../compiler/ir/branded_names.js'
import type { PortType } from '../compiler/ir/nodes.js'
import { failBare, toEnvelope, type ErrorEnvelope } from './envelope.js'
import { RESOURCES, PROMPTS, readResourceText, getPromptMessages } from './resources.js'

const session: SessionState = makeSession()
loadStdlib(session)

const portTypeOrNull = (t: PortType | undefined): string | null =>
  t === undefined ? null : portTypeToString(t)

// ─── Catalog entries ─────────────────────────────────────────────────────────
// The port-metadata shape the Lean engine mirrors. `type` is the display
// string (list_programs), `type_obj` the structured PortType (get_info echo
// + wire type-checking).

function concreteEntry(name: string, type: Instance | NonNullable<ReturnType<typeof loadProgramAsType>>) {
  const defaultsMap = rawInputDefaults(type) as Record<string, unknown>
  const ipt = inputPortTypes(type)
  const opt = outputPortTypes(type)
  const rpt = registerPortTypes(type)
  return {
    program_name: name,
    generic: false,
    type_params: null,
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

function genericEntry(name: string, prog: import('../compiler/ir/nodes.js').ResolvedProgram) {
  return {
    program_name: name,
    generic: true,
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

/** Catalog in session.programs insertion order (= list_programs semantics:
 *  typeRegistry order for concrete is the same registration order). */
function fullCatalog() {
  return [...session.programs.keys()].map(entryFor).filter(e => e !== undefined)
}

/** Names registered since `before`, plus the explicitly (re)defined names. */
function newEntries(before: ReadonlySet<string>, defined: string[]) {
  const names = [...session.programs.keys()].filter(k => !before.has(k))
  for (const d of defined) if (!names.includes(d) && session.programs.has(d)) names.push(d)
  return names.map(entryFor).filter(e => e !== undefined)
}

/** All program names declared by a def (itself + inline programDecls). */
function declaredNames(node: unknown): string[] {
  const names: string[] = []
  const walk = (n: unknown): void => {
    if (typeof n !== 'object' || n === null) return
    if (Array.isArray(n)) { for (const e of n) walk(e); return }
    const obj = n as Record<string, unknown>
    if (obj.op === 'programDecl' && typeof obj.name === 'string') names.push(obj.name)
    for (const v of Object.values(obj)) walk(v)
  }
  const root = node as { name?: string }
  if (typeof root.name === 'string') names.push(root.name)
  walk((node as { body?: unknown }).body)
  return names
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
  /** The session serialized to a ParsedProgram by the Lean engine
   *  (post-lift, post-extraction) — the Phase 3 seam payload. */
  parsed_program: unknown
  instances: Array<{ name: string; program: string; type_args?: Record<string, number> | null }>
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
    const { type, typeArgs } = resolveProgramType(session, inst.program, inst.type_args ?? undefined, undefined)
    const instance = instantiate(type, inst.name, { baseTypeName: inst.program, typeArgs })
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

  const plan = compileSessionSlottedFromParsed(
    session,
    p.parsed_program as Parameters<typeof compileSessionSlottedFromParsed>[1],
    'fused',
  )
  const planJson = JSON.stringify(toWirePlan(plan))
  return {
    plan_json: planJson,
    params: [...session.paramRegistry.entries()].map(([name, prm]) => ({ name, value: prm.value })),
  }
}

/** Lift array-literal wires to anonymous `__wire_N` instances. The Lean
 *  engine detects lift-needing wires (pure check), sends its canonical
 *  wire set + its `__wire` name counter, and adopts the rewritten wires
 *  and new instances durably — same observable behavior the TS engine's
 *  in-session lift had. The lifted Compiled types persist in this
 *  service's typeRegistry for later `compile` calls to instantiate. */
function handleLiftWires(p: { wires: Array<{ key: string; expr: ExprNode }>; counter_base: number }) {
  session.instanceRegistry.clear()
  session.inputExprNodes.clear()
  for (const w of p.wires) {
    session.inputExprNodes.set(wireKey(parseWireKey(w.key)), w.expr)
  }
  session.nameCounters.set('__wire', p.counter_base)
  liftWiresToInstances(session)
  return {
    wires: [...session.inputExprNodes.entries()].map(([key, expr]) => ({ key, expr })),
    instances: [...session.instanceRegistry.entries()].map(([name, inst]) => ({
      name,
      program: inst.baseTypeName,
      entry: concreteEntry(inst.baseTypeName, inst),
    })),
    counter: session.nameCounters.get('__wire') ?? p.counter_base,
  }
}

function handleMethod(method: string, params: Record<string, unknown>): unknown {
  switch (method) {
    case 'boot':
      return { catalog: fullCatalog() }

    case 'register_program': {
      const before = new Set(session.programs.keys())
      const { node } = normalizeProgramFile(params.def as { schema?: string; [k: string]: unknown })
      const type = loadProgramAsType(node, session)
      const result = type
        ? { program_name: type.displayName, inputs: inputNames(type), outputs: outputNames(type) }
        : {
            program_name: node.name,
            inputs: (node.ports?.inputs ?? []).map(i => typeof i === 'string' ? i : i.name),
            outputs: (node.ports?.outputs ?? []).map(o => typeof o === 'string' ? o : o.name),
            type_params: node.type_params,
          }
      return { result, entries: newEntries(before, declaredNames(node)) }
    }

    case 'resolve_type': {
      const program = params.program as string
      const typeArgs = params.type_args as Record<string, number> | undefined
      const { type, typeArgs: resolved } = resolveProgramType(session, program, typeArgs, undefined)
      const key = type.displayName
      return { key, type_args: resolved ?? null, entry: concreteEntry(key, type) }
    }

    case 'compile':
      return handleCompile(params as unknown as CompilePayload)

    case 'lift_wires':
      return handleLiftWires(params as unknown as Parameters<typeof handleLiftWires>[0])

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
