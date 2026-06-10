/**
 * Tropical IR engine — transport-agnostic session logic.
 *
 * Owns the SessionState and exposes `handleTool(name, args)`: the JSON-in /
 * JSON-out entry point into the IR. Both the MCP server (server.ts) and the
 * plain JSON-RPC IR service (ir_service.ts) call into this; it has no
 * transport of its own.
 */

import { readFileSync }        from 'node:fs'
import {
  makeSession, nextName, loadJSON,
  prettyExpr, resolveProgramType, SessionState, ExprNode,
  type DelaySlotEntry,
  normalizeProgramFile, v2NodeToFile,
  type Instance,
  instantiate,
  inputNames, outputNames, registerNames,
  inputPortTypes, outputPortTypes, registerPortTypes,
  inputPortType, outputPortType, registerPortType,
  rawInputDefaults,
  allocateOutputSlots,
  setWireExpr, unwrapDelay, reconstructWireDelays,
} from '../compiler/session.js'
import {
  saveProgramFromSession, loadProgramAsType, mergeProgramIntoSession,
  exportSessionAsProgram, loadStdlib as loadBuiltins, instanceDecls,
} from '../compiler/program.js'
import { DAC }                 from '../compiler/runtime/audio.js'
import { applyFlatPlan }  from '../compiler/apply_plan.js'
import { checkArrayConnection } from '../compiler/array_wiring.js'
import { validateExpr }         from '../compiler/expr.js'
import { exprDependencies }     from '../compiler/compiler.js'
import { portTypeToString } from '../compiler/ir/port_type.js'
import {
  parseWireKey, portRef, wireKey,
  instanceName as toInstanceName,
  portName as toPortName,
} from '../compiler/ir/branded_names.js'
import type { PortType, InputDecl, OutputDecl, TypeParamDecl } from '../compiler/ir/nodes.js'
import { PROGRAM_FORMAT_EXAMPLE } from './program_format_example.js'

const portTypeOrNull = (t: PortType | undefined): string | null =>
  t === undefined ? null : portTypeToString(t)

/** JSON.stringify with sorted object keys — for display labels that must
 *  not depend on the client's key order (the Lean engine's JSON layer
 *  canonicalizes key order, so labels must too). */
function stableStringify(v: unknown): string {
  if (Array.isArray(v)) return `[${v.map(stableStringify).join(',')}]`
  if (typeof v === 'object' && v !== null) {
    const entries = Object.entries(v as Record<string, unknown>)
      .filter(([, val]) => val !== undefined)
      .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
    return `{${entries.map(([k, val]) => `${JSON.stringify(k)}:${stableStringify(val)}`).join(',')}}`
  }
  return JSON.stringify(v)
}

// ─── Session ──────────────────────────────────────────────────────────────────

// Defaults for DAC configuration. Host config — describes the audio device,
// not anything semantic about the program. Overridable per-call on
// `start_audio`; once the DAC is created the values are fixed for the
// runtime's lifetime.
const DEFAULT_SAMPLE_RATE = 44100
const DEFAULT_DAC_CHANNELS = 2

const session: SessionState = makeSession()
loadBuiltins(session)

// Reserved instance name for the audio output boundary leaf. Wires whose
// destination has `instance === DAC_INSTANCE_NAME` route to session.graphOutputs
// instead of session.inputExprNodes. Multiple wires accumulate (mix-bus sum
// semantics, matching the C++ kernel's output_targets summing).
const DAC_INSTANCE_NAME = 'dac'
const DAC_OUT_PORT = 'out'

// ─── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Type-check an input expression against a destination port type, inserting
 * a broadcast_to wrapper when the shapes are compatible but differ (e.g. scalar→array).
 * Throws if the connection is incompatible (e.g. array→scalar, shape mismatch).
 * Returns the (possibly wrapped) expression and the resultShape if broadcasting occurred.
 */
function adaptInputExpr(
  node: ExprNode,
  dstType: PortType | undefined,
  instanceName: string,
  inputName: string,
): { expr: ExprNode; resultShape?: number[] } {
  let srcType: PortType | undefined

  if (typeof node === 'number') {
    srcType = { kind: 'scalar', scalar: 'float' }
  } else if (typeof node === 'boolean') {
    srcType = { kind: 'scalar', scalar: 'bool' }
  } else if (Array.isArray(node)) {
    srcType = { kind: 'array', element: 'float', shape: [(node as unknown[]).length] }
  } else if (typeof node === 'object' && node !== null) {
    const obj = node as Record<string, unknown>
    if (obj.op === 'ref') {
      const srcInst = session.instanceRegistry.get(obj.instance as string)
      if (srcInst) {
        const outName = obj.output as string | number
        const outIdx = typeof outName === 'number' ? outName : outputNames(srcInst).indexOf(String(outName))
        if (outIdx !== -1) srcType = outputPortType(srcInst, outIdx)
      }
    }
  }

  if (srcType === undefined) return { expr: node }

  const check = checkArrayConnection(srcType, dstType, node)
  if (!check.compatible) {
    failPredicate({
      code:      'type_mismatch',
      param:     'expr',
      value:     node,
      predicate: 'type_compatible',
      expected:  dstType,
      got:       srcType,
      message:   `Type mismatch on '${instanceName}'.${inputName}: ${check.error}`,
    })
  }
  return { expr: check.broadcastExpr ?? node, resultShape: check.resultShape }
}

// ─── Error envelope ───────────────────────────────────────────────────────────
// Spec: mcp/ERRORS.md

type ErrorCode =
  | 'unknown_program'
  | 'unknown_instance'
  | 'unknown_input'
  | 'unknown_output'
  | 'unknown_param'
  | 'unknown_device'
  | 'instance_exists'
  | 'invalid_type_args'
  | 'type_mismatch'
  | 'shape_mismatch'
  | 'length_mismatch'
  | 'arity_error'
  | 'missing_argument'
  | 'invalid_value'
  | 'invalid_state'
  | 'compile_failed'
  | 'audio_error'
  | 'internal_error'

type FieldSpec = {
  type:     'int' | 'float' | 'string' | 'bool'
  required: boolean
  min?:     number
  max?:     number
  options?: string[]
}

type Valid =
  | { kind: 'enum';      options: string[] }
  | { kind: 'record';    fields: Record<string, FieldSpec> }
  | { kind: 'predicate'; predicate: string; expected: unknown; got: unknown }

type ErrorEnvelope = {
  code:        ErrorCode
  message:     string
  retryable:   boolean
  param?:      string
  value?:      unknown
  valid?:      Valid
  suggestion?: unknown
}

class ToolError extends Error {
  envelope: ErrorEnvelope
  constructor(envelope: ErrorEnvelope) {
    super(envelope.message)
    this.envelope = envelope
  }
}

function nearestMatch(value: string, candidates: string[]): string | undefined {
  if (candidates.length === 0 || typeof value !== 'string') return undefined
  const dist = (a: string, b: string): number => {
    const m = a.length, n = b.length
    const dp: number[][] = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0))
    for (let i = 0; i <= m; i++) dp[i][0] = i
    for (let j = 0; j <= n; j++) dp[0][j] = j
    for (let i = 1; i <= m; i++)
      for (let j = 1; j <= n; j++)
        dp[i][j] = a[i - 1] === b[j - 1]
          ? dp[i - 1][j - 1]
          : 1 + Math.min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]
  }
  let best = candidates[0], bestD = dist(value, best)
  for (const c of candidates.slice(1)) {
    const d = dist(value, c)
    if (d < bestD) { best = c; bestD = d }
  }
  return bestD <= Math.max(2, Math.floor(value.length / 3)) ? best : undefined
}

function failEnum(opts: {
  code: ErrorCode; param: string; value: unknown;
  options: string[]; message?: string;
}): never {
  const suggestion = typeof opts.value === 'string'
    ? nearestMatch(opts.value, opts.options)
    : undefined
  const message = opts.message
    ?? `Invalid ${opts.param}: ${JSON.stringify(opts.value)}${suggestion ? `. Did you mean '${suggestion}'?` : ''}`
  throw new ToolError({
    code:      opts.code,
    message,
    retryable: false,
    param:     opts.param,
    value:     opts.value,
    valid:     { kind: 'enum', options: opts.options },
    ...(suggestion !== undefined ? { suggestion } : {}),
  })
}

function failRecord(opts: {
  code: ErrorCode; param: string; value: unknown;
  fields: Record<string, FieldSpec>; message?: string; suggestion?: unknown;
}): never {
  throw new ToolError({
    code:      opts.code,
    message:   opts.message ?? `Invalid ${opts.param}`,
    retryable: false,
    param:     opts.param,
    value:     opts.value,
    valid:     { kind: 'record', fields: opts.fields },
    ...(opts.suggestion !== undefined ? { suggestion: opts.suggestion } : {}),
  })
}

function failPredicate(opts: {
  code: ErrorCode; param: string; value: unknown;
  predicate: string; expected: unknown; got: unknown;
  suggestion?: unknown; message?: string;
}): never {
  throw new ToolError({
    code:      opts.code,
    message:   opts.message ?? `${opts.predicate} failed on ${opts.param}`,
    retryable: false,
    param:     opts.param,
    value:     opts.value,
    valid:     { kind: 'predicate', predicate: opts.predicate, expected: opts.expected, got: opts.got },
    ...(opts.suggestion !== undefined ? { suggestion: opts.suggestion } : {}),
  })
}

function failBare(opts: {
  code: ErrorCode; message: string; retryable?: boolean;
  param?: string; value?: unknown;
}): never {
  throw new ToolError({
    code:      opts.code,
    message:   opts.message,
    retryable: opts.retryable ?? false,
    ...(opts.param !== undefined ? { param: opts.param } : {}),
    ...(opts.value !== undefined ? { value: opts.value } : {}),
  })
}

const ok  = (data: unknown) =>
  ({ content: [{ type: 'text' as const, text: JSON.stringify({ status: 'ok', data }) }] })

const fail = (e: unknown) => {
  const envelope: ErrorEnvelope = e instanceof ToolError
    ? e.envelope
    : { code: 'internal_error', message: e instanceof Error ? e.message : String(e), retryable: false }
  return {
    content: [{ type: 'text' as const, text: JSON.stringify({ status: 'error', error: envelope }) }],
    isError: true as const,
  }
}

function wrap(fn: () => unknown) {
  try   { return ok(fn())         }
  catch (e) { return fail(e) }
}

function resolveProgramTypeOrFail(
  programName: string,
  typeArgs: Record<string, number> | undefined,
  programParam: string,
) {
  try {
    return resolveProgramType(session, programName, typeArgs, undefined)
  } catch (e) {
    const registered = session.typeRegistry.has(programName) || session.programs.has(programName)
    if (!registered) {
      failEnum({
        code:    'unknown_program',
        param:   programParam,
        value:   programName,
        options: [
          ...session.typeRegistry.keys(),
          ...session.programs.keys(),
        ],
      })
    }
    failBare({
      code:    'invalid_type_args',
      message: e instanceof Error ? e.message : String(e),
      param:   'type_args',
      value:   typeArgs,
    })
  }
}

function requireInstance(name: string, param: string) {
  const inst = session.instanceRegistry.get(name)
  if (!inst)
    failEnum({
      code:    'unknown_instance',
      param,
      value:   name,
      options: [...session.instanceRegistry.keys()],
    })
  return inst
}

/** Validate an optional positive integer arg; fall back to `defaultValue` when
 *  the arg is undefined. Throws an `invalid_value` envelope on bad input. */
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

function resolveOutputIdx(inst: Instance, nameOrIdx: string | number): number {
  if (typeof nameOrIdx === 'number') return nameOrIdx
  if (typeof nameOrIdx === 'string' && /^\d+$/.test(nameOrIdx)) return parseInt(nameOrIdx, 10)
  const idx = outputNames(inst).indexOf(nameOrIdx)
  if (idx === -1)
    failEnum({
      code:    'unknown_output',
      param:   'output',
      value:   nameOrIdx,
      options: outputNames(inst),
    })
  return idx
}

function resolveInputIdx(inst: Instance, nameOrIdx: string | number): number {
  if (typeof nameOrIdx === 'number') return nameOrIdx
  if (typeof nameOrIdx === 'string' && /^\d+$/.test(nameOrIdx)) return parseInt(nameOrIdx, 10)
  const idx = inputNames(inst).indexOf(nameOrIdx)
  if (idx === -1)
    failEnum({
      code:    'unknown_input',
      param:   'input',
      value:   nameOrIdx,
      options: inputNames(inst),
    })
  return idx
}

function instanceSummary(name: string) {
  const inst = session.instanceRegistry.get(name)!
  return {
    name,
    type_name: inst.baseTypeName,
    type_args: inst.typeArgs ?? null,
    inputs: inputNames(inst),
    outputs: outputNames(inst),
  }
}

/** Apply wiring via FlatRuntime.
 *
 *  Canonicalizes wires and rebuilds the slot state before compiling, so
 *  recompiles are idempotent (the same reset compiler_service.ts performs
 *  per sync). Without this, re-wiring an already-compiled port left a
 *  stale delaySlotRegistry entry behind and the next compile died with
 *  "duplicate reg/delay '__autodelay:…'". */
function wire(): {} {
  const wires = [...session.inputExprNodes.entries()].map(
    ([k, e]) => [k, reconstructWireDelays(e, session.delaySlotRegistry)] as const)
  session.inputExprNodes.clear()
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
  for (const [name, inst] of session.instanceRegistry) {
    allocateOutputSlots(session, toInstanceName(name), inst.compiled)
  }
  for (const [k, e] of wires) session.inputExprNodes.set(k, e)
  applyFlatPlan(session, session.runtime)
  return {}
}
// ─── Tool handlers ────────────────────────────────────────────────────────────

// ─── Shared handler logic ─────────────────────────────────────────────────────

function handleDefineProgram(args: Record<string, unknown>) {
  return wrap(() => {
    const { node } = normalizeProgramFile(args.def as { schema?: string; [k: string]: unknown })
    const type = loadProgramAsType(node, session)
    if (type) {
      return { program_name: type.displayName, inputs: inputNames(type), outputs: outputNames(type) }
    }
    // Generic — template only, no concrete ports until instantiation.
    return {
      program_name: node.name,
      inputs: (node.ports?.inputs ?? []).map(i => typeof i === 'string' ? i : i.name),
      outputs: (node.ports?.outputs ?? []).map(o => typeof o === 'string' ? o : o.name),
      type_params: node.type_params,
    }
  })
}

function handleAddInstance(
  programName: string,
  instanceName: string,
  typeArgs?: Record<string, number>,
) {
  return wrap(() => {
    if (instanceName === DAC_INSTANCE_NAME)
      failBare({
        code:    'invalid_value',
        message: `'${DAC_INSTANCE_NAME}' is a reserved instance name (audio output boundary). Choose a different name.`,
        param:   'instance_name',
        value:   instanceName,
      })
    if (session.instanceRegistry.has(instanceName))
      failBare({
        code:    'instance_exists',
        message: `Instance '${instanceName}' already exists.`,
        param:   'instance_name',
        value:   instanceName,
      })
    const { type, typeArgs: resolved } = resolveProgramTypeOrFail(programName, typeArgs, 'program')
    const inst = instantiate(type, instanceName, { baseTypeName: programName, typeArgs: resolved })
    session.instanceRegistry.set(instanceName, inst)
    // Slot model (M3, additive): populate the output slot registry
    // alongside the legacy instanceRegistry. Nothing consumes these
    // yet; M4 wires them into the new compile path.
    allocateOutputSlots(session, toInstanceName(instanceName), type)
    return instanceSummary(instanceName)
  })
}

function handleReplicate(
  programName: string,
  count: number,
  namePrefix?: string,
  typeArgs?: Record<string, number>,
) {
  return wrap(() => {
    if (!Number.isInteger(count) || count < 1)
      failRecord({
        code:    'invalid_value',
        param:   'count',
        value:   count,
        message: `count must be a positive integer, got ${count}`,
        fields:  { count: { type: 'int', required: true, min: 1 } },
      })

    const prefix = namePrefix ?? programName.toLowerCase()
    if (prefix === DAC_INSTANCE_NAME)
      failBare({
        code:    'invalid_value',
        message: `'${DAC_INSTANCE_NAME}' is a reserved instance name (audio output boundary). Choose a different name_prefix.`,
        param:   'name_prefix',
        value:   prefix,
      })
    const created = []
    for (let i = 0; i < count; i++) {
      const name = nextName(session, prefix)
      if (session.instanceRegistry.has(name))
        failBare({
          code:    'instance_exists',
          message: `Instance '${name}' already exists — pick a different name_prefix`,
          param:   'name_prefix',
          value:   namePrefix,
        })
      const { type, typeArgs: resolved } = resolveProgramTypeOrFail(programName, typeArgs, 'program')
      const inst = instantiate(type, name, { baseTypeName: programName, typeArgs: resolved })
      session.instanceRegistry.set(name, inst)
      allocateOutputSlots(session, toInstanceName(name), type)
      created.push(instanceSummary(name))
    }
    return { created }
  })
}

/** Resolve an output name/index on an instance and return the canonical name string. */
function resolveOutputName(inst: Instance, nameOrIdx: string | number): string {
  const idx = resolveOutputIdx(inst, nameOrIdx)
  return outputNames(inst)[idx]
}

/** Resolve an input name/index on an instance and return the canonical name string. */
function resolveInputName(inst: Instance, nameOrIdx: string | number): string {
  const idx = resolveInputIdx(inst, nameOrIdx)
  return inputNames(inst)[idx]
}

function handleWireChain(args: Record<string, unknown>) {
  return wrap(() => {
    const instanceNames = args.instances as string[]
    const outputPort    = args.output as string | number
    const inputPort     = args.input  as string | number
    const initialExpr   = args.initial_expr as import('../compiler/expr.js').ExprNode | undefined

    if (instanceNames.length < 2 && initialExpr === undefined)
      failBare({
        code: 'arity_error',
        message: 'wire_chain needs at least 2 instances, or 1 instance with initial_expr',
        param: 'instances',
        value: instanceNames,
      })

    // Validate all instances upfront
    const insts = instanceNames.map(n => requireInstance(n, 'instances'))

    // Optionally set first instance's input
    if (initialExpr !== undefined) {
      const firstName  = instanceNames[0]
      const firstInst  = insts[0]
      const inputName  = resolveInputName(firstInst, inputPort)
      const { expr }   = adaptInputExpr(initialExpr, inputPortType(firstInst, inputNames(firstInst).indexOf(inputName)), firstName, inputName)
      setWireExpr(session, portRef(toInstanceName(firstName), toPortName(inputName)), expr)
    }

    // Wire instances[i].output → instances[i+1].input
    const linked = []
    for (let i = 0; i < instanceNames.length - 1; i++) {
      const srcInst  = insts[i]
      const dstInst  = insts[i + 1]
      const srcName  = instanceNames[i]
      const dstName  = instanceNames[i + 1]
      const outName  = resolveOutputName(srcInst, outputPort)
      const inName   = resolveInputName(dstInst, inputPort)
      const refExpr  = { op: 'ref' as const, instance: srcName, output: outName }
      const { expr } = adaptInputExpr(refExpr, inputPortType(dstInst, inputNames(dstInst).indexOf(inName)), dstName, inName)
      setWireExpr(session, portRef(toInstanceName(dstName), toPortName(inName)), expr)
      linked.push(`${srcName}.${outName} → ${dstName}.${inName}`)
    }

    return { linked, ...wire() }
  })
}

function handleWireZip(args: Record<string, unknown>) {
  return wrap(() => {
    const sources = args.sources as Array<{ instance: string; output: string | number }>
    const targets = args.targets as Array<{ instance: string; input:  string | number }>

    if (sources.length !== targets.length)
      failBare({
        code: 'length_mismatch',
        message: `sources and targets must be the same length (got ${sources.length} vs ${targets.length})`,
        param: 'sources',
        value: { sources: sources.length, targets: targets.length },
      })

    const linked = []
    for (let i = 0; i < sources.length; i++) {
      const src     = sources[i]
      const dst     = targets[i]
      const srcInst = requireInstance(src.instance, 'sources[].instance')
      const dstInst = requireInstance(dst.instance, 'targets[].instance')

      const outName  = resolveOutputName(srcInst, src.output)
      const inName   = resolveInputName(dstInst, dst.input)
      const refExpr  = { op: 'ref' as const, instance: src.instance, output: outName }
      const { expr } = adaptInputExpr(refExpr, inputPortType(dstInst, inputNames(dstInst).indexOf(inName)), dst.instance, inName)
      setWireExpr(session, portRef(toInstanceName(dst.instance), toPortName(inName)), expr)
      linked.push(`${src.instance}.${outName} → ${dst.instance}.${inName}`)
    }

    return { linked, ...wire() }
  })
}

function handleFanOut(args: Record<string, unknown>) {
  return wrap(() => {
    const rawSource = args.source as { instance?: string; output?: string | number } | ExprNode
    const targets   = args.targets as Array<{ instance: string; input: string | number }>

    // Resolve source to a canonical ExprNode plus a display label
    let sourceExpr: ExprNode
    let sourceLabel: string
    if (
      rawSource !== null &&
      typeof rawSource === 'object' &&
      !Array.isArray(rawSource) &&
      typeof (rawSource as Record<string, unknown>).instance === 'string' &&
      (rawSource as Record<string, unknown>).output !== undefined
    ) {
      const s       = rawSource as { instance: string; output: string | number }
      const srcInst = requireInstance(s.instance, 'source.instance')
      const outName = resolveOutputName(srcInst, s.output)
      sourceExpr  = { op: 'ref' as const, instance: s.instance, output: outName }
      sourceLabel = `${s.instance}.${outName}`
    } else {
      sourceExpr  = rawSource as ExprNode
      sourceLabel = stableStringify(sourceExpr)
    }

    const linked = []
    for (const dst of targets) {
      const dstInst = requireInstance(dst.instance, 'targets[].instance')
      const inName   = resolveInputName(dstInst, dst.input)
      const { expr } = adaptInputExpr(sourceExpr, inputPortType(dstInst, inputNames(dstInst).indexOf(inName)), dst.instance, inName)
      setWireExpr(session, portRef(toInstanceName(dst.instance), toPortName(inName)), expr)
      linked.push(`${sourceLabel} → ${dst.instance}.${inName}`)
    }

    return { linked, ...wire() }
  })
}

function handleFanIn(args: Record<string, unknown>) {
  return wrap(() => {
    const sources = args.sources as Array<{ instance: string; output: string | number; gain?: number }>
    const target  = args.target  as { instance: string; input: string | number }

    if (sources.length === 0)
      failBare({ code: 'arity_error', message: 'sources must be non-empty', param: 'sources', value: sources })

    const dstInst = requireInstance(target.instance, 'target.instance')

    // Build one term per source: ref, optionally scaled
    const terms: ExprNode[] = sources.map(src => {
      const srcInst = requireInstance(src.instance, 'sources[].instance')
      const outName = resolveOutputName(srcInst, src.output)
      const ref: ExprNode = { op: 'ref' as const, instance: src.instance, output: outName }
      return src.gain !== undefined
        ? { op: 'mul' as const, args: [ref, src.gain] }
        : ref
    })

    // Left-fold into a binary add tree
    const sumExpr = terms.slice(1).reduce<ExprNode>(
      (acc, t) => ({ op: 'add' as const, args: [acc, t] }),
      terms[0],
    )

    const inName   = resolveInputName(dstInst, target.input)
    const { expr } = adaptInputExpr(sumExpr, inputPortType(dstInst, inputNames(dstInst).indexOf(inName)), target.instance, inName)
    setWireExpr(session, portRef(toInstanceName(target.instance), toPortName(inName)), expr)

    return { mixed: sources.length, target: `${target.instance}.${inName}`, ...wire() }
  })
}

function handleFeedback(args: Record<string, unknown>) {
  return wrap(() => {
    // Post-auto-delay: every MCP wire is implicitly one-sample-delayed
    // via `setWireExpr`. `feedback` becomes a `wire` alias that exposes
    // two extra knobs — `init` (initial value of the underlying delay
    // slot) and `delay_id` (stable identifier for hot-swap state
    // transfer). Both are preserved by passing them through to
    // setWireExpr's `WireExprOpts`.
    const from    = args.from     as { instance: string; output: string | number }
    const to      = args.to       as { instance: string; input:  string | number }
    const init    = (args.init    as number | undefined) ?? 0
    const delayId = args.delay_id as string | undefined

    const srcInst = requireInstance(from.instance, 'from.instance')
    const dstInst = requireInstance(to.instance, 'to.instance')

    const outName = resolveOutputName(srcInst, from.output)
    const inName  = resolveInputName(dstInst, to.input)

    const refExpr: ExprNode = { op: 'ref' as const, instance: from.instance, output: outName }
    validateExpr(refExpr, `${to.instance}.${inName}`)
    const { expr } = adaptInputExpr(refExpr, inputPortType(dstInst, inputNames(dstInst).indexOf(inName)), to.instance, inName)
    setWireExpr(session, portRef(toInstanceName(to.instance), toPortName(inName)), expr, { init, id: delayId })

    return {
      feedback: `${from.instance}.${outName} →[delay init=${init}]→ ${to.instance}.${inName}`,
      ...wire(),
    }
  })
}

function handleExportProgram(args: Record<string, unknown>) {
  return wrap(() => {
    const name      = args.name as string
    const inputs    = (args.inputs ?? {}) as Record<string, string>
    const outputs   = args.outputs as Record<string, { instance: string; output: string }>
    const removeExported = (args.remove_exported as boolean) ?? false

    if (!name) failBare({ code: 'missing_argument', message: 'name is required', param: 'name' })
    if (!outputs || Object.keys(outputs).length === 0)
      failBare({ code: 'missing_argument', message: 'outputs is required (at least one)', param: 'outputs' })

    const exportedNode = exportSessionAsProgram(session, { name, inputs, outputs })
    const exportedInstanceNames = [...instanceDecls(exportedNode)].map(d => d.name)

    // Register as a usable type (exported programs are never generic)
    const type = loadProgramAsType(exportedNode, session)
    if (!type)
      failBare({
        code:    'internal_error',
        message: 'export_program produced a generic program — not supported',
      })
    session.typeRegistry.set(name, type)

    // Optionally clean up exported instances from the session
    if (removeExported) {
      const exportedInstances = new Set(exportedInstanceNames)
      for (const instName of exportedInstances) {
        session.instanceRegistry.delete(instName)
        for (const key of [...session.inputExprNodes.keys()]) {
          if (key.startsWith(`${instName}:`)) session.inputExprNodes.delete(key)
        }
        session.graphOutputs = session.graphOutputs.filter(o => o.instance !== instName)
      }
      // Clean up dangling refs in remaining wiring
      for (const [key, expr] of [...session.inputExprNodes.entries()]) {
        if ([...exprDependencies(expr)].some(dep => exportedInstances.has(dep))) {
          session.inputExprNodes.delete(key)
        }
      }
      if (session.instanceRegistry.size > 0 || session.graphOutputs.length > 0) {
        wire()
      }
    }

    return {
      program_name: name,
      inputs: inputNames(type),
      outputs: outputNames(type),
      instances_included: exportedInstanceNames,
      program: v2NodeToFile(exportedNode as unknown as ExprNode),
    }
  })
}

/** Collect the hoisted-delay slot indices a wire reads, split by class
 *  (scalar `sessionSlot` vs array `sessionArraySlot`). */
function collectSlotRefs(expr: ExprNode, scalar: Set<number>, array: Set<number>): void {
  if (typeof expr !== 'object' || expr === null) return
  if (Array.isArray(expr)) { for (const e of expr) collectSlotRefs(e, scalar, array); return }
  const n = expr as { op?: string; index?: unknown; [k: string]: unknown }
  if (n.op === 'sessionSlot' && typeof n.index === 'number') { scalar.add(n.index); return }
  if (n.op === 'sessionArraySlot' && typeof n.index === 'number') { array.add(n.index); return }
  for (const v of Object.values(n)) {
    if (typeof v === 'object' && v !== null) collectSlotRefs(v as ExprNode, scalar, array)
  }
}

/** All instances a wire transitively references, resolving hoisted
 *  `sessionSlot` / `sessionArraySlot` reads back through the delay
 *  registry. After a compile, `extractSessionDelays` has rewritten
 *  auto-delayed wires (e.g. a feedback ring) into slot reads whose real
 *  source lives in `delaySlotRegistry`, so a plain `exprDependencies`
 *  would miss the instances the delayed source touches. Cycle-safe via
 *  the visited slot sets (feedback rings chain slot → source → slot). */
function resolvedWireDeps(
  expr: ExprNode,
  registry: readonly DelaySlotEntry[],
  acc: Set<string> = new Set(),
  seenScalar: Set<number> = new Set(),
  seenArray: Set<number> = new Set(),
): Set<string> {
  for (const d of exprDependencies(expr)) acc.add(d)
  const scalar = new Set<number>()
  const array = new Set<number>()
  collectSlotRefs(expr, scalar, array)
  for (const idx of scalar) {
    if (seenScalar.has(idx)) continue
    seenScalar.add(idx)
    const e = registry.find(r => !r.isArray && r.slotIdx === idx)
    if (e) resolvedWireDeps(e.sourceExpr, registry, acc, seenScalar, seenArray)
  }
  for (const idx of array) {
    if (seenArray.has(idx)) continue
    seenArray.add(idx)
    const e = registry.find(r => r.isArray && r.arraySlot === idx)
    if (e) resolvedWireDeps(e.sourceExpr, registry, acc, seenScalar, seenArray)
  }
  return acc
}

function handleRemoveInstance(instanceName: string) {
  return wrap(() => {
    requireInstance(instanceName, 'instance_name')
    session.instanceRegistry.delete(instanceName)

    const reg = session.delaySlotRegistry

    // Hoisted-delay slots whose (resolved) source touches the removed
    // instance — e.g. a feedback ring's `lfo_i.freq = delay(… ref(lfoN) …)`
    // that compiled to `sessionSlot` with the ref now living in the
    // registry. Both their backing wires AND their registry entries must
    // go, or the post-removal recompile materializes a ref to a missing
    // instance (the dangling-wiring bug).
    const taintedScalar = new Set<number>()
    const taintedArray = new Set<number>()
    for (const e of reg) {
      if (!resolvedWireDeps(e.sourceExpr, reg).has(instanceName)) continue
      if (e.isArray) { if (e.arraySlot !== undefined) taintedArray.add(e.arraySlot) }
      else if (e.slotIdx !== undefined) taintedScalar.add(e.slotIdx)
    }

    for (const [key, expr] of [...session.inputExprNodes.entries()]) {
      // A wire goes if it feeds the removed instance, or its source —
      // resolved through any hoisted delay slots — references it.
      if (key.startsWith(`${instanceName}:`) || resolvedWireDeps(expr, reg).has(instanceName)) {
        session.inputExprNodes.delete(key)
      }
    }

    // Drop the tainted registry entries. `sessionSlot` reads index by the
    // entry's `slotIdx` / `arraySlot` field (not array position), so
    // filtering the array out is safe for the surviving slots.
    session.delaySlotRegistry = reg.filter(e =>
      e.isArray
        ? !(e.arraySlot !== undefined && taintedArray.has(e.arraySlot))
        : !(e.slotIdx !== undefined && taintedScalar.has(e.slotIdx)))

    session.graphOutputs = session.graphOutputs.filter(o => o.instance !== instanceName)
    return { removed: instanceName, ...wire() }
  })
}

function handleListPrograms() {
  return wrap(() => {
    const concrete = [...session.typeRegistry.entries()].map(([typeName, type]) => {
      const defaultsMap = rawInputDefaults(type) as Record<string, unknown>
      const ipt = inputPortTypes(type)
      const opt = outputPortTypes(type)
      const rpt = registerPortTypes(type)
      return {
        program_name: typeName,
        inputs:    inputNames(type).map((n, i) => ({
          name: n,
          type: portTypeOrNull(ipt[i]),
          default: defaultsMap[n] ?? null,
        })),
        outputs: outputNames(type).map((n, i) => ({
          name: n,
          type: portTypeOrNull(opt[i]),
        })),
        registers: registerNames(type).map((n, i) => ({ name: n, type: portTypeOrNull(rpt[i]) })),
        type_params: null as Record<string, { type: 'int'; default?: number }> | null,
      }
    })

    // Generic templates live in session.programs alongside concrete
    // programs; distinguish by typeParams.length > 0. Concrete entries
    // are surfaced via typeRegistry above; filter to generics here.
    const generic = [...session.programs.entries()]
      .filter(([_typeName, prog]) => prog.typeParams.length > 0)
      .map(([typeName, prog]) => ({
        program_name: typeName,
        inputs: prog.ports.inputs.map((i: InputDecl) => ({
          name: i.name,
          type: null,
          default: null,
        })),
        outputs: prog.ports.outputs.map((o: OutputDecl) => ({
          name: o.name,
          type: null,
        })),
        registers: [] as Array<{ name: string; type: string | null }>,
        type_params: prog.typeParams.length > 0
          ? Object.fromEntries(prog.typeParams.map((tp: TypeParamDecl) => {
              const entry: { type: 'int'; default?: number } = { type: 'int' }
              if (tp.default !== undefined) entry.default = tp.default
              return [tp.name, entry]
            }))
          : null,
      }))

    return [...concrete, ...generic]
  })
}

function handleListInstances() {
  return wrap(() =>
    [...session.instanceRegistry.keys()].map(instanceSummary),
  )
}

function handleGetInfo(instanceName: string) {
  return wrap(() => {
    const inst = requireInstance(instanceName, 'instance_name')
    return {
      name: instanceName,
      program: inst.baseTypeName,
      type_args: inst.typeArgs ?? null,
      inputs:  inputNames(inst).map((n, i) => {
        const key = wireKey(portRef(toInstanceName(instanceName), toPortName(n)))
        const expr = session.inputExprNodes.get(key)
        return {
          name: n, index: i,
          type: inputPortType(inst, i) ?? null,
          // Canonical authored form, not the compile-internal `sessionSlot`
          // rewrite a prior compile may have left in the wire map — same
          // canonicalization `save` applies.
          expr: expr !== undefined ? reconstructWireDelays(expr, session.delaySlotRegistry) : null,
          pretty: expr !== undefined
            ? prettyExpr(expr, session.instanceRegistry, session.delaySlotRegistry)
            : null,
        }
      }),
      outputs: outputNames(inst).map((n, i) => ({
        name: n, index: i,
        type: outputPortType(inst, i) ?? null,
      })),
      registers: registerNames(inst).map((n, i) => ({
        name: n, index: i, type: registerPortType(inst, i) ?? null,
      })),
    }
  })
}

/** Resolve a wire-to-dac.out source expression to an {instance, output} pair.
 *  Today's session.graphOutputs stores only named instance outputs (matching the
 *  C++ kernel's output_targets). The `expr` must be a `ref`-shaped node;
 *  arbitrary expression outputs land in a later phase (A4). */
function resolveDacSource(expr: ExprNode): { instance: string; output: string } {
  if (typeof expr !== 'object' || expr === null || Array.isArray(expr)) {
    failBare({
      code:    'invalid_value',
      message: `dac.${DAC_OUT_PORT} requires a ref-shaped expression (use refExpr or {op:'ref',instance,output}). Got literal/array.`,
      param:   'expr',
      value:   expr,
    })
  }
  const e = expr as { op?: string; instance?: unknown; output?: unknown }
  if (e.op !== 'ref') {
    failBare({
      code:    'invalid_value',
      message: `dac.${DAC_OUT_PORT} requires expr.op === 'ref'. Got op='${String(e.op)}'.`,
      param:   'expr',
      value:   expr,
    })
  }
  if (typeof e.instance !== 'string') {
    failBare({
      code:    'invalid_value',
      message: `dac.${DAC_OUT_PORT}: ref.instance must be a string`,
      param:   'instance',
      value:   e.instance,
    })
  }
  const inst = requireInstance(e.instance, 'instance')
  let outputName: string
  if (typeof e.output === 'number') {
    if (e.output < 0 || e.output >= outputNames(inst).length) {
      failEnum({
        code:    'unknown_output',
        param:   'output',
        value:   e.output,
        options: outputNames(inst),
      })
    }
    outputName = outputNames(inst)[e.output]
  } else if (typeof e.output === 'string') {
    if (!outputNames(inst).includes(e.output)) {
      failEnum({
        code:    'unknown_output',
        param:   'output',
        value:   e.output,
        options: outputNames(inst),
      })
    }
    outputName = e.output
  } else {
    failBare({
      code:    'invalid_value',
      message: `dac.${DAC_OUT_PORT}: ref.output must be a number or string`,
      param:   'output',
      value:   e.output,
    })
  }
  return { instance: e.instance, output: outputName }
}

function handleWire(args: Record<string, unknown>) {
  return wrap(() => {
    const setOps = (args.set ?? []) as Array<{ instance: string; input: string | number; expr: ExprNode; combine?: string }>
    const removeOps = (args.remove ?? []) as Array<{ instance: string; input: string | number }>

    // Process removes first
    let dacRemoved = 0
    for (const r of removeOps) {
      if (r.instance === DAC_INSTANCE_NAME) {
        if (r.input !== DAC_OUT_PORT) {
          failBare({
            code:    'unknown_output',
            message: `dac has only one output port: '${DAC_OUT_PORT}'. Got '${r.input}'.`,
            param:   'remove[].input',
            value:   r.input,
          })
        }
        // Mass-clear all dac wires; per-wire identity is not encoded.
        dacRemoved += session.graphOutputs.length
        session.graphOutputs.length = 0
        continue
      }
      const inst = requireInstance(r.instance, 'remove[].instance')
      const inputId = resolveInputIdx(inst, r.input)
      const resolvedName = inputNames(inst)[inputId] ?? String(inputId)
      session.inputExprNodes.delete(wireKey(portRef(toInstanceName(r.instance), toPortName(resolvedName))))
    }

    // Process sets
    const results = []
    const dacWires = []
    for (const s of setOps) {
      if (s.instance === DAC_INSTANCE_NAME) {
        if (s.input !== DAC_OUT_PORT) {
          failBare({
            code:    'unknown_output',
            message: `dac has only one output port: '${DAC_OUT_PORT}'. Got '${s.input}'.`,
            param:   'set[].input',
            value:   s.input,
          })
        }
        validateExpr(s.expr, `${DAC_INSTANCE_NAME}.${DAC_OUT_PORT}`)
        const resolved = resolveDacSource(s.expr)
        session.graphOutputs.push(resolved)
        dacWires.push({ instance: s.instance, input: s.input, expr: s.expr })
        continue
      }
      const inst = requireInstance(s.instance, 'set[].instance')
      const inputId = resolveInputIdx(inst, s.input)
      const resolvedName = inputNames(inst)[inputId] ?? String(inputId)
      validateExpr(s.expr, `${s.instance}.${resolvedName}`)
      const { expr } = adaptInputExpr(s.expr, inputPortType(inst, inputId), s.instance, resolvedName)
      // M9d: fan-in support. If the input already has a wired
      // expression and the caller provided `combine`, wrap both in
      // {op:combine, args:[existing_raw, new]}. The auto-delay on
      // stored wires means `existing` carries an outer `delay(...)`;
      // strip it with `unwrapDelay` so the combined expression has a
      // single outer delay (applied by `setWireExpr`) rather than a
      // nested-delay-inside-delay shape that would give the existing
      // source two samples of latency.
      const ref = portRef(toInstanceName(s.instance), toPortName(resolvedName))
      const existing = session.inputExprNodes.get(wireKey(ref))
      let toStore: ExprNode = expr
      if (existing !== undefined && s.combine !== undefined) {
        // Resolve the stored wire back to its authored form first: after a
        // compile, the map holds a bare `sessionSlot` read that
        // `unwrapDelay` cannot strip, and combining against it would give
        // the existing source a second sample of latency.
        toStore = { op: s.combine, args: [unwrapDelay(reconstructWireDelays(existing, session.delaySlotRegistry)), expr] } as ExprNode
      }
      setWireExpr(session, ref, toStore)
      results.push({ instance: s.instance, input: resolvedName, expr: toStore })
    }

    return {
      set: results,
      ...(dacWires.length ? { dac: dacWires } : {}),
      removed: removeOps.length,
      ...(dacRemoved ? { dacRemoved } : {}),
      ...wire(),
    }
  })
}

function handleListWiring(filterInstance?: string) {
  return wrap(() => {
    const results: Array<{ instance: string; input: string; expr: string }> = []
    for (const [key, node] of session.inputExprNodes) {
      const { instance, port } = parseWireKey(key)
      if (filterInstance && instance !== filterInstance) continue
      results.push({ instance, input: port, expr: prettyExpr(node, session.instanceRegistry, session.delaySlotRegistry) })
    }
    return results
  })
}

function handleLoad(args: Record<string, unknown>) {
  return wrap(() => {
    let raw: unknown
    if (args.path) {
      raw = JSON.parse(readFileSync(args.path as string, 'utf-8'))
    } else if (args.program) {
      raw = args.program
    } else {
      failBare({ code: 'missing_argument', message: 'Provide either path (file) or program (inline JSON).' })
    }

    if (session.dac?.isRunning) session.dac.stop()
    const t0 = performance.now()
    loadJSON(raw as { schema: string }, session)
    const wall_ms = performance.now() - t0
    return {
      instances:   [...session.instanceRegistry.keys()],
      wiring:      session.inputExprNodes.size,
      outputs:     session.graphOutputs.length,
      params:      [...session.paramRegistry.keys()],
      timing: { wall_ms },
    }
  })
}

function handleSave() {
  return wrap(() => {
    const { node, topLevel } = saveProgramFromSession(session)
    return { program: v2NodeToFile(node as unknown as ExprNode, topLevel) }
  })
}

function handleMerge(args: Record<string, unknown>) {
  return wrap(() => {
    const raw = args.program ?? args.patch
    if (!raw) failBare({ code: 'missing_argument', message: 'Provide a program or patch object.' })
    const { node, topLevel } = normalizeProgramFile(raw as { schema?: string; [k: string]: unknown })
    mergeProgramIntoSession(node, topLevel, session)
    return {
      instances:   [...session.instanceRegistry.keys()],
      wiring:      session.inputExprNodes.size,
      outputs:     session.graphOutputs.length,
      params:      [...session.paramRegistry.keys()],
    }
  })
}

// ─── Tool dispatcher ─────────────────────────────────────────────────────────

function handleTool(name: string, args: Record<string, unknown>) {
  switch (name) {

    // ── New unified tools ─────────────────────────────────────────────────

    case 'define_program':
      return handleDefineProgram(args)

    case 'add_instance':
      return handleAddInstance(
        args.program as string,
        args.instance_name as string,
        args.type_args as Record<string, number> | undefined,
      )

    case 'remove_instance':
      return handleRemoveInstance(args.instance_name as string)

    case 'replicate':
      return handleReplicate(
        args.program as string,
        args.count as number,
        args.name_prefix as string | undefined,
        args.type_args as Record<string, number> | undefined,
      )

    case 'wire_chain':
      return handleWireChain(args)

    case 'wire_zip':
      return handleWireZip(args)

    case 'fan_out':
      return handleFanOut(args)

    case 'fan_in':
      return handleFanIn(args)

    case 'feedback':
      return handleFeedback(args)

    case 'export_program':
      return handleExportProgram(args)

    case 'list_programs':
      return handleListPrograms()

    case 'list_instances':
      return handleListInstances()

    case 'get_info':
      return handleGetInfo(args.instance_name as string)

    case 'wire':
      return handleWire(args)

    case 'list_wiring':
      return handleListWiring(args.instance as string | undefined)

    case 'load':
      return handleLoad(args)

    case 'save':
      return handleSave()

    case 'merge':
      return handleMerge(args)

    // ── Audio / params (unchanged) ────────────────────────────────────────

    case 'start_audio': return wrap(() => {
      if (!session.dac) {
        const sampleRate = validatePositiveInt(args.sample_rate, 'sample_rate', DEFAULT_SAMPLE_RATE)
        const channels   = validatePositiveInt(args.channels,    'channels',    DEFAULT_DAC_CHANNELS)
        session.dac = DAC.fromRuntime(session.runtime._h, sampleRate, channels)
      }

      const deviceName = args.device_name as string | undefined
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
    })

    case 'stop_audio': return wrap(() => {
      if (!session.dac)
        failBare({
          code:    'invalid_state',
          message: 'DAC has not been created yet.',
        })
      session.dac.stop()
      return { is_running: session.dac.isRunning }
    })

    case 'audio_status': return wrap(() => {
      if (!session.dac) return { is_running: false }
      return {
        is_running:      session.dac.isRunning,
        is_reconnecting: session.dac.isReconnecting,
        stats:           session.dac.callbackStats(),
      }
    })

    case 'set_param': return wrap(() => {
      const paramName = args.name as string
      const value     = args.value as number
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
    })

    case 'list_params': return wrap(() => {
      return [...session.paramRegistry.entries()].map(([name, p]) => ({
        name, value: p.value,
      }))
    })

    default:
      return fail(`Unknown tool: '${name}'`)
  }
}

// ─── Resources & prompts (the MCP non-tool surface) ─────────────────────────

const RESOURCES = [
  {
    uri:         'tropical://programs',
    name:        'Program catalog',
    description: 'Markdown catalog of all registered program types with inputs, outputs, and default values.',
    mimeType:    'text/markdown',
  },
  {
    uri:         'tropical://program-format',
    name:        'Program format',
    description: 'Reference doc for the tropical_program_2 schema.',
    mimeType:    'text/markdown',
  },
]

function renderProgramCatalog(): string {
  const lines: string[] = ['# tropical program catalog\n']
  for (const [typeName, type] of session.typeRegistry) {
    lines.push(`## ${typeName}`)
    const defaultsMap = rawInputDefaults(type) as Record<string, unknown>
    const inputParts = inputNames(type).map(n => {
      const val = defaultsMap[n]
      return val !== undefined ? `${n}=${JSON.stringify(val)}` : n
    })
    lines.push(`Inputs:  ${inputParts.join(', ')}`)
    lines.push(`Outputs: ${outputNames(type).join(', ')}`)
    lines.push('')
  }
  return lines.join('\n')
}

const PROGRAM_FORMAT_DOC = `# tropical program format

Programs are the unified representation for all DSP in tropical. A program is an
expression with declared ports and a body that declares regs/delays/instances
and assigns outputs and next-tick register updates.

## tropical_program_2

The body is a \`block\` of \`decls\` (reg_decl, delay_decl, instance_decl,
program_decl) and \`assigns\` (output_assign, next_update). Ports, type_params,
and breaks_cycles sit alongside the body. **Audio output is an \`output_assign\`
in the body with name \`"dac.out"\`** — wire the signal you want heard to it.
Session metadata — \`params\`, \`config\` — is top-level.

${JSON.stringify(PROGRAM_FORMAT_EXAMPLE, null, 2)}

Key fields: schema, name, ports (inputs/outputs/type_defs), body (block),
type_params, sample_rate, breaks_cycles, and top-level session metadata
(params, config). Send a signal to the speakers with a body \`output_assign\`
named \`"dac.out"\`. (File-root \`audio_outputs\` is deprecated — don't use it.)

Decl node shapes:
- reg_decl:      { op, name, init, type? }
- delay_decl:    { op, name, update, init? }
- instance_decl: { op, name, program, inputs?, type_args? }
- program_decl:  { op, name, program: <program node> }

Assign node shapes:
- output_assign: { op, name, expr }
- next_update:   { op, target: { kind: "reg"|"delay", name }, expr }

## Expression node format (ExprNode)

Used in instance input wiring and inline program process definitions.

- **Literal**: \`3.14\` or \`true\`
- **Reference**: \`{ "op": "ref", "instance": "osc", "output": "out" }\` — routes another instance's output to this input
- **Input port**: \`{ "op": "input", "name": "freq" }\` — (inside program definitions only)
- **Param**: \`{ "op": "param", "name": "cutoff" }\`
- **Binary**: \`{ "op": "mul", "args": [<expr>, <expr>] }\` — add, sub, mul, div, floor_div, mod, pow; lt, lte, gt, gte, eq, neq; bit_and, bit_or, bit_xor, lshift, rshift
- **Unary**: \`{ "op": "neg", "args": [<expr>] }\` — neg, abs, not, bit_not
- **Clamp / Select / Array / Index / Delay / Builtins**: see the program catalog and stdlib for worked examples.
`

const PROMPTS = [
  {
    name:        'build-patch',
    description: 'Three-tiered workflow guidance for building and editing tropical patches efficiently.',
  },
]

const BUILD_PATCH_PROMPT = `# build-patch workflow

Before writing any program, always fetch both resources:
- \`tropical://programs\` — full catalog of available program types with inputs, defaults, and outputs
- \`tropical://program-format\` — tropical_program_2 schema reference with a worked example

## Choose the right tool for the job

### New program (starting from scratch)
Use \`load\` with a **complete** tropical_program_2 JSON object in a single call.
Do not call \`add_instance\` and \`wire\` one-by-one — that requires many round
trips and recompiles the JIT kernel on every change.

### Extending an existing program (adding new instances)
Use \`merge\` with a partial program containing only the new instances and wiring.
Then use \`wire\` to connect the new instances to existing ones.

### Targeted edits (changing a value, tweaking a param)
Use \`wire\` with a single set entry, or \`set_param\` for control parameters.

## Writing programs

- Use named ports (e.g. \`"output": "out"\`) rather than integer indices.
- Wire everything in one program object where possible; minimize round trips.
- Audio output goes through a body \`output_assign\` named \`"dac.out"\` (or the
  \`wire\` tool with \`instance: "dac", input: "out"\`).
`

/** List the MCP resources this server exposes. */
export function listResources() { return RESOURCES }

/** Read a resource's text content by uri. */
export function readResource(uri: string): string {
  if (uri === 'tropical://programs' || uri === 'tropical://modules') return renderProgramCatalog()
  if (uri === 'tropical://program-format' || uri === 'tropical://patch-format') return PROGRAM_FORMAT_DOC
  throw new Error(`Unknown resource: ${uri}`)
}

/** List the MCP prompts this server exposes. */
export function listPrompts() { return PROMPTS }

/** Get a prompt's messages by name. */
export function getPrompt(name: string): { messages: unknown[] } {
  if (name === 'build-patch') {
    return { messages: [{ role: 'user', content: { type: 'text', text: BUILD_PATCH_PROMPT } }] }
  }
  throw new Error(`Unknown prompt: ${name}`)
}

// ─── Public engine surface ──────────────────────────────────────────────────
export { session, handleTool }
