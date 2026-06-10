/**
 * compiler_service.ts — the TS compiler behind the Lean engine.
 *
 * Phase 1 of the Lean port: session *state* and tool semantics live in
 * Lean (lean/Tropical/Engine.lean); this service owns what hasn't been
 * ported yet — program registration (raise → elaborate → strata),
 * session compilation (compileSession → tropical_plan_5 →
 * runtime.loadPlan), the native runtime/DAC/param FFI, and
 * save/export/load/merge (they need the compiler). It shrinks phase by
 * phase until Phase 6 deletes it.
 *
 * Protocol: newline JSON-RPC on stdio, same framing as ir_service.ts.
 * Tool-level failures return `{result: {error: <ErrorEnvelope>}}` so the
 * Lean side maps them onto its own envelope type; JSON-RPC `error` is
 * reserved for transport/internal faults.
 *
 * The one stateful contract: `sync` rebuilds this service's session from
 * the Lean engine's authoritative snapshot (instances + canonical
 * pre-extraction wires + outputs + params) and compiles it. Every other
 * graph-reading method (`save`, `export_program`) operates on the
 * session as of the last `sync`/`load`/`merge`.
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
import { applyFlatPlan } from '../compiler/apply_plan.js'
import { DAC } from '../compiler/runtime/audio.js'
import { Param } from '../compiler/runtime/param.js'
import { portTypeToString } from '../compiler/ir/port_type.js'
import { parseWireKey, wireKey, portRef, instanceName as toInstanceName } from '../compiler/ir/branded_names.js'
import type { PortType } from '../compiler/ir/nodes.js'
import { failEnum, failBare, toEnvelope, type ErrorEnvelope } from './envelope.js'
import { RESOURCES, PROMPTS, readResourceText, getPromptMessages } from './resources.js'

const DEFAULT_SAMPLE_RATE = 44100
const DEFAULT_DAC_CHANNELS = 2

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

function validatePositiveInt(value: unknown, param: string, defaultValue: number): number {
  if (value === undefined || value === null) return defaultValue
  if (typeof value !== 'number' || !Number.isInteger(value) || value <= 0) {
    failBare({
      code:    'invalid_value',
      message: `${param} must be a positive integer; got ${JSON.stringify(value)}`,
      param,
      value,
    })
  }
  return value as number
}

type SyncPayload = {
  instances: Array<{ name: string; program: string; type_args?: Record<string, number> | null }>
  wires: Array<{ key: string; expr: ExprNode }>
  graph_outputs: Array<{ instance: string; output: string }>
  params: Array<{ name: string; value: number }>
}

function handleSync(p: SyncPayload) {
  // Reset graph state; keep type registries (programs persist across syncs)
  // and paramRegistry (live Param handles are referenced by the loaded
  // kernel — recreating them would orphan the pointers the JIT baked in).
  session.instanceRegistry.clear()
  session.inputExprNodes.clear()
  session.graphOutputs.length = 0
  session.outputSlotRegistry.clear()
  session.paramSlotRegistry.clear()
  session.outputPortMeta.clear()
  session.inputSlotRegistry.clear()
  session.inputPortMeta.clear()
  session.inputExprs.clear()
  session.slotCount = 0
  session.ioArraySlotCount = 0
  session.ioArraySlotSizes.length = 0
  session.ioArraySlotNames.length = 0
  session.delaySlotRegistry = []

  for (const inst of p.instances) {
    const { type, typeArgs } = resolveProgramType(session, inst.program, inst.type_args ?? undefined, undefined)
    const instance = instantiate(type, inst.name, { baseTypeName: inst.program, typeArgs })
    session.instanceRegistry.set(inst.name, instance)
    allocateOutputSlots(session, toInstanceName(inst.name), type)
  }
  for (const w of p.wires) {
    // Wires arrive in stored form (the Lean engine already applied the
    // auto-delay wrap where it applies); set verbatim, no re-wrapping.
    session.inputExprNodes.set(wireKey(parseWireKey(w.key)), w.expr)
  }
  for (const o of p.graph_outputs) session.graphOutputs.push({ instance: o.instance, output: o.output })
  for (const pr of p.params) {
    if (!session.paramRegistry.has(pr.name)) session.paramRegistry.set(pr.name, new Param(pr.value))
  }

  applyFlatPlan(session, session.runtime)
  return { params: [...session.paramRegistry.entries()].map(([name, prm]) => ({ name, value: prm.value })) }
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

    case 'sync':
      return handleSync(params as unknown as SyncPayload)

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
      if (session.dac?.isRunning) session.dac.stop()
      const before = new Set(session.programs.keys())
      const t0 = performance.now()
      loadJSON(raw as { schema: string }, session)
      const wall_ms = performance.now() - t0
      return { state: dumpState(before), timing: { wall_ms } }
    }

    case 'merge': {
      const { node, topLevel } = normalizeProgramFile(params.program as { schema?: string; [k: string]: unknown })
      const before = new Set(session.programs.keys())
      mergeProgramIntoSession(node, topLevel, session)
      return { state: dumpState(before) }
    }

    // ── Audio / params (runtime ownership moves to Lean in Phase 2) ─────────

    case 'start_audio': {
      if (!session.dac) {
        const sampleRate = validatePositiveInt(params.sample_rate, 'sample_rate', DEFAULT_SAMPLE_RATE)
        const channels   = validatePositiveInt(params.channels,    'channels',    DEFAULT_DAC_CHANNELS)
        session.dac = DAC.fromRuntime(session.runtime._h, sampleRate, channels)
      }
      const deviceName = params.device_name as string | undefined
      if (deviceName) {
        const devices = DAC.listDevices()
        const match = devices.find(d => d.name.toLowerCase().includes(deviceName.toLowerCase()))
        if (!match)
          failEnum({
            code:    'unknown_device',
            param:   'device_name',
            value:   deviceName,
            options: devices.map(d => d.name),
          })
        if (session.dac.isRunning) {
          session.dac.switchDevice(match.id)
        } else {
          session.dac.start()
          session.dac.switchDevice(match.id)
        }
        return { is_running: session.dac.isRunning, device: match.name }
      }
      if (!session.dac.isRunning) session.dac.start()
      return { is_running: session.dac.isRunning }
    }

    case 'stop_audio': {
      if (!session.dac)
        failBare({ code: 'invalid_state', message: 'DAC has not been created yet.' })
      session.dac.stop()
      return { is_running: session.dac.isRunning }
    }

    case 'audio_status': {
      if (!session.dac) return { is_running: false }
      return {
        is_running:      session.dac.isRunning,
        is_reconnecting: session.dac.isReconnecting,
        stats:           session.dac.callbackStats(),
      }
    }

    case 'set_param': {
      const paramName = params.name as string
      const value     = params.value as number
      const p = session.paramRegistry.get(paramName)
      if (!p)
        failEnum({
          code:    'unknown_param',
          param:   'name',
          value:   paramName,
          options: [...session.paramRegistry.keys()],
        })
      p.value = value
      return { name: paramName, value: p.value }
    }

    case 'list_params':
      return [...session.paramRegistry.entries()].map(([name, p]) => ({ name, value: p.value }))

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
