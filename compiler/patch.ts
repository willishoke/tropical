/**
 * JSON patch / module definition schema, expression AST builder, and load/save.
 * Replaces tropical/yaml_schema.py with a clean-break JSON format.
 */

import {
  type SignalExpr, type ExprCoercible, coerce, type ExprNode,
} from './expr.js'
import {
  ModuleType, ModuleInstance,
  type ProgramDef, type NestedCall, type ValueCoercible,
} from './module.js'
import { Param, Trigger } from './runtime/param.js'
import { applyFlatPlan } from './apply_plan.js'
import { Runtime } from './runtime/runtime.js'
import { loadProgramAsSession, type ProgramJSON } from './program.js'

// ─────────────────────────────────────────────────────────────
// JSON schema types
// ─────────────────────────────────────────────────────────────

// ExprNode is defined in expr.ts and re-exported here for backward compatibility.
export type { ExprNode } from './expr.js'
import type { ExprNode } from './expr.js'


export interface NestedModuleJSON {
  type: string
  inputs: Record<string, ExprNode>
}

export interface ModuleDefJSON {
  name: string
  inputs: (string | { name: string; type: string })[]
  outputs: (string | { name: string; type: string })[]
  regs?: Record<string, number | boolean | number[] | number[][] | { init: number | boolean | number[] | number[][]; type: string }>
  /** Named delay nodes, declared before any expression that references them. */
  delays?: Record<string, { update: ExprNode; init?: number }>
  /** Named nested sub-module instances. */
  nested?: Record<string, NestedModuleJSON>
  sample_rate?: number
  input_defaults?: Record<string, ExprNode>
  process: {
    outputs: Record<string, ExprNode>
    next_regs?: Record<string, ExprNode>
  }
  /** When true, outputs depend only on previous-sample state — allows feedback cycles. */
  breaks_cycles?: boolean
}

export interface TypeDefFieldJSON {
  name: string
  scalar_type: number
}

export interface StructTypeDefJSON {
  kind: 'struct'
  name: string
  fields: TypeDefFieldJSON[]
}

export interface SumVariantJSON {
  name: string
  payload: TypeDefFieldJSON[]
}

export interface SumTypeDefJSON {
  kind: 'sum'
  name: string
  variants: SumVariantJSON[]
}

export type TypeDefJSON = StructTypeDefJSON | SumTypeDefJSON

export interface PatchJSON {
  schema: 'tropical_patch_1'
  config?: {
    buffer_length?: number
  }
  /** Inline ADT type definitions (loaded before module instantiation). */
  type_defs?: TypeDefJSON[]
  /** Inline module type definitions (loaded before instantiation). */
  module_defs?: ModuleDefJSON[]
  modules: Array<{ type: string; name?: string }>
  connections?: Array<{
    src: string; src_output: string | number
    dst: string; dst_input: string | number
  }>
  /** Graph-level mix outputs. */
  outputs?: Array<
    | { module: string; output: string | number }
    | { expr: ExprNode }
  >
  params?: Array<{
    name: string
    value?: number
    time_const?: number
    type?: 'param' | 'trigger'
  }>
  input_exprs?: Array<{
    module: string
    input: string | number
    expr: ExprNode
  }>
}

// ─────────────────────────────────────────────────────────────
// Session state (shared by patch load/save and MCP server)
// ─────────────────────────────────────────────────────────────

export interface SessionState {
  bufferLength: number
  dac: import('./runtime/audio.js').DAC | null  // lazy type import to avoid circular dep
  typeRegistry: Map<string, ModuleType>
  instanceRegistry: Map<string, ModuleInstance>
  graphOutputs: Array<{ module: string; output: string }>
  paramRegistry: Map<string, Param>
  triggerRegistry: Map<string, Trigger>
  /** Canonical input wiring: key is `${module}:${input}`, value is the ExprNode for round-trip save. */
  inputExprNodes: Map<string, ExprNode>  // key: `${module}:${input}`
  /** FlatRuntime — all audio goes through this. */
  runtime: Runtime
  /** Thin proxy over runtime that matches the old Graph interface for tests and legacy callers. */
  graph: { primeJit(): void; process(): void; readonly outputBuffer: Float64Array; dispose(): void }
  /** Name counter for auto-generated instance names. */
  _nameCounters: Map<string, number>
}

export function makeSession(bufferLength = 512): SessionState {
  const runtime = new Runtime(bufferLength)
  return {
    bufferLength,
    dac: null,
    typeRegistry: new Map(),
    instanceRegistry: new Map(),
    graphOutputs: [],
    paramRegistry: new Map(),
    triggerRegistry: new Map(),
    inputExprNodes: new Map(),
    runtime,
    graph: {
      primeJit: () => {},
      process: () => runtime.process(),
      get outputBuffer() { return runtime.outputBuffer },
      dispose: () => runtime.dispose(),
    },
    _nameCounters: new Map(),
  }
}

/** Generate a unique instance name from a type prefix. */
export function nextName(session: SessionState, prefix: string): string {
  const count = (session._nameCounters.get(prefix) ?? 0) + 1
  session._nameCounters.set(prefix, count)
  return `${prefix}${count}`
}

// ─────────────────────────────────────────────────────────────
// Op name sets (used by pretty-printer)
// ─────────────────────────────────────────────────────────────

const BINARY_OPS = new Set([
  'add', 'sub', 'mul', 'div', 'floor_div', 'mod', 'pow',
  'lt', 'lte', 'gt', 'gte', 'eq', 'neq',
  'bit_and', 'bit_or', 'bit_xor', 'lshift', 'rshift',
])

const UNARY_OPS = new Set([
  'neg', 'abs', 'sin', 'cos', 'exp', 'log', 'tanh', 'not', 'bit_not',
])

// ─────────────────────────────────────────────────────────────
// Module loader
// ─────────────────────────────────────────────────────────────
// Module loader
// ─────────────────────────────────────────────────────────────

/**
 * Convert a name-based JSON ExprNode to a slot-based ExprNode.
 * Pure tree walk — no side effects, no context stack.
 */
function slottifyExpr(
  node: ExprNode,
  inputNames: string[],
  regNames: string[],
  delayNameToId: Map<string, number>,
  nestedAliasToId: Map<string, number>,
  nestedAliasDef: Map<string, ProgramDef>,
): ExprNode {
  if (typeof node === 'number' || typeof node === 'boolean') return node
  if (Array.isArray(node)) return node.map(n => slottifyExpr(n, inputNames, regNames, delayNameToId, nestedAliasToId, nestedAliasDef))

  const obj = node as Record<string, unknown>
  const op = obj.op as string
  const recurse = (n: ExprNode) => slottifyExpr(n, inputNames, regNames, delayNameToId, nestedAliasToId, nestedAliasDef)

  if (op === 'input') {
    const name = obj.name as string | undefined
    if (name !== undefined) {
      const id = inputNames.indexOf(name)
      if (id === -1) throw new Error(`Unknown input '${name}'. Available: ${inputNames.join(', ')}`)
      return { op: 'input', id }
    }
    return node // already slot-based
  }

  if (op === 'reg') {
    const name = obj.name as string | undefined
    if (name !== undefined) {
      const id = regNames.indexOf(name)
      if (id === -1) throw new Error(`Unknown register '${name}'. Available: ${regNames.join(', ')}`)
      return { op: 'reg', id }
    }
    return node
  }

  if (op === 'delay_ref') {
    const id = obj.id as string
    const nodeId = delayNameToId.get(id)
    if (nodeId === undefined) throw new Error(`delay_ref: no delay with id '${id}'.`)
    return { op: 'delay_value', node_id: nodeId }
  }

  if (op === 'nested_out') {
    const ref = obj.ref as string
    const nodeId = nestedAliasToId.get(ref)
    if (nodeId === undefined) throw new Error(`nested_out: no nested program named '${ref}'.`)
    const nestedDef = nestedAliasDef.get(ref)!
    const output = obj.output as string | number
    const outputId = typeof output === 'number' ? output : nestedDef.outputNames.indexOf(output)
    if (outputId === -1) throw new Error(`nested_out: unknown output '${output}' on '${ref}'.`)
    return { op: 'nested_output', node_id: nodeId, output_id: outputId }
  }

  // Generic op with args — recurse
  if ('args' in obj) {
    const args = (obj.args as ExprNode[]).map(recurse)
    const result: Record<string, unknown> = {}
    for (const [k, v] of Object.entries(obj)) {
      if (k === 'args') continue
      result[k] = v
    }
    result.args = args
    return result as ExprNode
  }

  // Handle array-like containers that aren't in 'args'
  if (op === 'array') {
    const items = (obj.items as ExprNode[]).map(recurse)
    return { ...obj, items } as ExprNode
  }

  // Combinator forms with nested expr fields
  if (op === 'let') {
    const bind = obj.bind as Record<string, ExprNode>
    const converted: Record<string, ExprNode> = {}
    for (const [k, v] of Object.entries(bind)) converted[k] = recurse(v)
    return { ...obj, bind: converted, in: recurse(obj.in as ExprNode) } as ExprNode
  }
  if (op === 'generate' || op === 'chain' || op === 'iterate') {
    return { ...obj, ...(obj.init !== undefined ? { init: recurse(obj.init as ExprNode) } : {}), body: recurse(obj.body as ExprNode) } as ExprNode
  }
  if (op === 'fold' || op === 'scan') {
    return { ...obj, over: recurse(obj.over as ExprNode), init: recurse(obj.init as ExprNode), body: recurse(obj.body as ExprNode) } as ExprNode
  }
  if (op === 'map2') {
    return { ...obj, over: recurse(obj.over as ExprNode), body: recurse(obj.body as ExprNode) } as ExprNode
  }
  if (op === 'zip_with') {
    return { ...obj, a: recurse(obj.a as ExprNode), b: recurse(obj.b as ExprNode), body: recurse(obj.body as ExprNode) } as ExprNode
  }

  // ADT ops
  if (op === 'construct_struct') {
    return { ...obj, fields: (obj.fields as ExprNode[]).map(recurse) } as ExprNode
  }
  if (op === 'field_access') {
    return { ...obj, struct_expr: recurse(obj.struct_expr as ExprNode) } as ExprNode
  }
  if (op === 'construct_variant') {
    return { ...obj, payload: (obj.payload as ExprNode[]).map(recurse) } as ExprNode
  }
  if (op === 'match_variant') {
    return { ...obj, scrutinee: recurse(obj.scrutinee as ExprNode), branches: (obj.branches as ExprNode[]).map(recurse) } as ExprNode
  }

  // Leaf ops (sample_rate, sample_index, binding, float, int, bool, matrix, etc.)
  return node
}

export function loadModuleFromJSON(
  def: ModuleDefJSON,
  session: Pick<SessionState, 'typeRegistry' | 'instanceRegistry' | 'paramRegistry' | 'triggerRegistry'>,
): ModuleType {
  const inputSpecs  = def.inputs
  const outputSpecs = def.outputs
  const inputNames  = inputSpecs.map(i => typeof i === 'string' ? i : i.name)
  const outputNames = outputSpecs.map(o => typeof o === 'string' ? o : o.name)
  const inputPortTypes  = inputSpecs.map(i => typeof i === 'string' ? undefined : i.type)
  const outputPortTypes = outputSpecs.map(o => typeof o === 'string' ? undefined : o.type)
  const regsRaw     = def.regs ?? {}
  const delaysRaw   = def.delays ?? {}
  const nestedRaw   = def.nested ?? {}

  // ── Parse registers ──
  const regNames: string[] = []
  const regInitValues: ValueCoercible[] = []
  const regPortTypes: (string | undefined)[] = []
  for (const [name, val] of Object.entries(regsRaw)) {
    regNames.push(name)
    if (typeof val === 'object' && val !== null && !Array.isArray(val) && 'init' in val) {
      const typed = val as { init: ValueCoercible; type: string }
      regInitValues.push(typed.init)
      regPortTypes.push(typed.type)
    } else {
      regInitValues.push(val as ValueCoercible)
      regPortTypes.push(undefined)
    }
  }

  // ── Assign delay IDs ──
  const delayNames = Object.keys(delaysRaw)
  const delayNameToId = new Map(delayNames.map((name, i) => [name, i]))
  const delayInitValues = delayNames.map(name => delaysRaw[name].init ?? 0)

  // ── Assign nested call IDs ──
  const nestedAliases = Object.keys(nestedRaw)
  const nestedAliasToId = new Map(nestedAliases.map((alias, i) => [alias, i]))
  const nestedAliasDef = new Map<string, ProgramDef>()
  const nestedCalls: NestedCall[] = []

  for (const alias of nestedAliases) {
    const spec = nestedRaw[alias]
    const type = session.typeRegistry.get(spec.type)
    if (!type) throw new Error(`Unknown module type '${spec.type}' in nested.`)
    nestedAliasDef.set(alias, type._def)

    // Build call arg nodes — order inputs by the nested type's input order
    const callArgNodes: ExprNode[] = type._def.inputNames.map((name, idx) => {
      if (name in spec.inputs) {
        return slottifyExpr(spec.inputs[name], inputNames, regNames, delayNameToId, nestedAliasToId, nestedAliasDef)
      }
      const defaultExpr = type._def.inputDefaults[idx]
      if (defaultExpr) return defaultExpr._node
      throw new Error(`Missing input '${name}' for nested module '${alias}'.`)
    })

    nestedCalls.push({ programDef: type._def, callArgNodes })
  }

  // ── Convert delay update expressions ──
  const delayUpdateNodes = delayNames.map(name =>
    slottifyExpr(delaysRaw[name].update, inputNames, regNames, delayNameToId, nestedAliasToId, nestedAliasDef)
  )

  // ── Convert output expressions ──
  const outputExprNodes = outputNames.map(name => {
    const node = def.process.outputs[name]
    if (node === undefined) throw new Error(`Output '${name}' missing from process.outputs.`)
    return slottifyExpr(node, inputNames, regNames, delayNameToId, nestedAliasToId, nestedAliasDef)
  })

  // ── Convert register update expressions ──
  const registerExprNodes: (ExprNode | null)[] = regNames.map(name => {
    const node = def.process.next_regs?.[name]
    if (node === undefined) return null
    return slottifyExpr(node, inputNames, regNames, delayNameToId, nestedAliasToId, nestedAliasDef)
  })

  // ── Parse input defaults ──
  const rawInputDefaults: Record<string, ExprNode> = {}
  const inputDefaults: (SignalExpr | null)[] = new Array(inputNames.length).fill(null)
  if (def.input_defaults) {
    for (const [k, v] of Object.entries(def.input_defaults)) {
      rawInputDefaults[k] = v as ExprNode
      const idx = inputNames.indexOf(k)
      if (idx !== -1) inputDefaults[idx] = coerce(v as ExprCoercible)
    }
  }

  const programDef: ProgramDef = {
    typeName: def.name,
    inputNames,
    outputNames,
    inputPortTypes,
    outputPortTypes,
    registerNames: regNames,
    registerPortTypes: regPortTypes,
    registerInitValues: regInitValues,
    sampleRate: def.sample_rate ?? 44100.0,
    rawInputDefaults,
    inputDefaults,
    delayInitValues,
    outputExprNodes,
    registerExprNodes,
    delayUpdateNodes,
    nestedCalls,
    breaksCycles: def.breaks_cycles ?? false,
  }

  return new ModuleType(programDef)
}

// ─────────────────────────────────────────────────────────────
// Expression pretty-printer
// ─────────────────────────────────────────────────────────────

/** Infix symbols for binary ops. Ops in BINARY_OPS but absent here fall back to `op(l, r)`. */
const BINARY_INFIX: Record<string, string> = {
  add: '+', sub: '-', mul: '*', div: '/', floor_div: '//', mod: '%', pow: '**', matmul: '@',
  lt: '<', lte: '<=', gt: '>', gte: '>=', eq: '==', neq: '!=',
  bit_and: '&', bit_or: '|', bit_xor: '^', lshift: '<<', rshift: '>>',
}

/** Prefix symbols for unary ops. Ops in UNARY_OPS but absent here use `op(x)` notation. */
const UNARY_PREFIX: Record<string, string> = { neg: '-' }

/**
 * Render an ExprNode as a human-readable string.
 * Refs appear as `Module.output`; math appears as infix expressions.
 * instanceRegistry is used to resolve numeric output indices to port names.
 */
export function prettyExpr(
  node: ExprNode,
  instanceRegistry: Map<string, ModuleInstance>,
): string {
  if (typeof node === 'number') return String(node)
  if (typeof node === 'boolean') return String(node)
  if (Array.isArray(node)) return `[${node.map(n => prettyExpr(n, instanceRegistry)).join(', ')}]`

  const n = node as { op: string; [k: string]: unknown }
  const op = n.op
  const args = (n.args as ExprNode[] | undefined) ?? []

  if (op === 'ref') {
    const mod = n.module as string
    const out = n.output
    const inst = instanceRegistry.get(mod)
    const outName = inst && typeof out === 'number' ? (inst.outputNames[out] ?? String(out)) : String(out)
    return `${mod}.${outName}`
  }
  if (op === 'input')     return `input(${n.name})`
  if (op === 'param')     return `param(${n.name})`
  if (op === 'trigger')   return `trigger(${n.name})`
  if (op === 'sample_rate')  return 'sample_rate'
  if (op === 'sample_index') return 'sample_index'
  if (op === 'float' || op === 'int')  return String(n.value)
  if (op === 'bool')  return String(n.value)

  if (BINARY_OPS.has(op)) {
    const sym = BINARY_INFIX[op]
    const l = prettyExpr(args[0], instanceRegistry)
    const r = prettyExpr(args[1], instanceRegistry)
    return sym ? `(${l} ${sym} ${r})` : `${op}(${l}, ${r})`
  }
  if (UNARY_OPS.has(op)) {
    const pfx = UNARY_PREFIX[op]
    const x = prettyExpr(args[0], instanceRegistry)
    return pfx ? `${pfx}${x}` : `${op}(${x})`
  }

  if (op === 'clamp')  return `clamp(${args.map(a => prettyExpr(a, instanceRegistry)).join(', ')})`
  if (op === 'select') return `select(${args.map(a => prettyExpr(a, instanceRegistry)).join(', ')})`
  if (op === 'index')  return `${prettyExpr(args[0], instanceRegistry)}[${prettyExpr(args[1], instanceRegistry)}]`
  if (op === 'array_set') return `array_set(${args.map(a => prettyExpr(a, instanceRegistry)).join(', ')})`
  if (op === 'array') return `[${(n.items as ExprNode[]).map(i => prettyExpr(i, instanceRegistry)).join(', ')}]`
  if (op === 'matrix') return `matrix(${JSON.stringify(n.rows)})`
  if (op === 'delay') return `delay(${prettyExpr(args[0], instanceRegistry)}, ${n.init ?? 0})`
  if (op === 'delay_ref') return `delay_ref(${n.id})`
  if (op === 'nested_out') return `${n.ref}.${n.output}`
  if (op === 'construct_struct') {
    const fields = (n.fields as ExprNode[]).map(f => prettyExpr(f, instanceRegistry))
    return `${n.type_name}{${fields.join(', ')}}`
  }
  if (op === 'field_access') {
    return `${prettyExpr(n.struct_expr as ExprNode, instanceRegistry)}.field[${n.field_index}]`
  }
  if (op === 'construct_variant') {
    const payload = (n.payload as ExprNode[]).map(p => prettyExpr(p, instanceRegistry))
    return `${n.type_name}::${n.variant_tag}(${payload.join(', ')})`
  }
  if (op === 'match_variant') {
    const branches = (n.branches as ExprNode[]).map(b => prettyExpr(b, instanceRegistry))
    return `match(${prettyExpr(n.scrutinee as ExprNode, instanceRegistry)}){${branches.join(', ')}}`
  }

  // Should never reach here given the finite op set, but keep a safe fallback
  throw new Error(`prettyExpr: unhandled op '${op}'`)
}

// ─────────────────────────────────────────────────────────────
// Patch loader
// ─────────────────────────────────────────────────────────────

/**
 * Load any supported JSON schema into a session.
 * Detects schema version and delegates to the appropriate loader.
 */
export function loadJSON(json: { schema: string; [k: string]: unknown }, session: SessionState): void {
  if (json.schema === 'tropical_program_1') {
    loadProgramAsSession(json as unknown as ProgramJSON, session, loadPatchFromJSON)
    return
  }
  if (json.schema === 'tropical_patch_1') {
    loadPatchFromJSON(json as unknown as PatchJSON, session)
    return
  }
  throw new Error(`Unknown schema '${json.schema}'. Expected 'tropical_program_1' or 'tropical_patch_1'.`)
}

export function loadPatchFromJSON(json: PatchJSON, session: SessionState): void {
  session.dac = null
  session.instanceRegistry.clear()
  session.graphOutputs.length = 0
  session.paramRegistry.clear()
  session.triggerRegistry.clear()
  session.inputExprNodes.clear()
  session._nameCounters.clear()

  // Load inline module type definitions
  for (const def of json.module_defs ?? []) {
    const type = loadModuleFromJSON(def, session)
    session.typeRegistry.set(type.name, type)
  }

  // Create params and triggers before modules (modules may reference them)
  for (const p of json.params ?? []) {
    if (p.type === 'trigger') {
      session.triggerRegistry.set(p.name, new Trigger())
    } else {
      session.paramRegistry.set(p.name, new Param(p.value ?? 0.0, p.time_const ?? 0.005))
    }
  }

  // Instantiate modules (TS-only — no C API calls)
  for (const entry of json.modules) {
    const type = session.typeRegistry.get(entry.type)
    if (!type) throw new Error(`Unknown module type '${entry.type}'.`)
    const name = entry.name ?? nextName(session, type.name)
    const inst = type.instantiateAs(name)
    session.instanceRegistry.set(inst.name, inst)
  }

  // Populate wiring state — connections, input expressions, outputs
  for (const conn of json.connections ?? []) {
    const srcInst = session.instanceRegistry.get(conn.src)
    const dstInst = session.instanceRegistry.get(conn.dst)
    if (!srcInst) throw new Error(`Connection src module '${conn.src}' not found.`)
    if (!dstInst) throw new Error(`Connection dst module '${conn.dst}' not found.`)
    const srcId = typeof conn.src_output === 'number' ? conn.src_output : srcInst.outputIndex(conn.src_output)
    const dstId = typeof conn.dst_input  === 'number' ? conn.dst_input  : dstInst.inputIndex(conn.dst_input)
    const srcOutName = srcInst.outputNames[srcId]
    const dstInName  = dstInst.inputNames[dstId]
    session.inputExprNodes.set(`${conn.dst}:${dstInName}`, { op: 'ref', module: conn.src, output: srcOutName })
  }

  for (const ie of json.input_exprs ?? []) {
    const inst = session.instanceRegistry.get(ie.module)
    if (!inst) throw new Error(`input_expr module '${ie.module}' not found.`)
    session.inputExprNodes.set(`${ie.module}:${ie.input}`, ie.expr)
  }

  // Ensure module input defaults are included in the plan
  for (const [name, inst] of session.instanceRegistry) {
    const defaults = inst._def.rawInputDefaults as Record<string, ExprNode>
    for (const [inputName, value] of Object.entries(defaults)) {
      const key = `${name}:${inputName}`
      if (!session.inputExprNodes.has(key)) {
        session.inputExprNodes.set(key, value)
      }
    }
  }

  for (const out of json.outputs ?? []) {
    if ('expr' in out) {
      throw new Error('Output expressions not supported in plan-based path. Use module output refs instead.')
    }
    const inst = session.instanceRegistry.get(out.module)
    if (!inst) throw new Error(`Output module '${out.module}' not found.`)
    session.graphOutputs.push({ module: out.module, output: String(out.output) })
  }

  // Apply wiring via FlatRuntime
  applyFlatPlan(session, session.runtime)
}

// ─────────────────────────────────────────────────────────────
// Patch merger (additive — no session teardown)
// ─────────────────────────────────────────────────────────────

export function mergePatchFromJSON(json: PatchJSON, session: SessionState): void {
  // Fail fast on name collisions before touching state
  for (const entry of json.modules) {
    if (entry.name && session.instanceRegistry.has(entry.name))
      throw new Error(`merge_patch collision: module '${entry.name}' already exists.`)
  }
  for (const p of json.params ?? []) {
    if (session.paramRegistry.has(p.name) || session.triggerRegistry.has(p.name))
      throw new Error(`merge_patch collision: param/trigger '${p.name}' already exists.`)
  }

  // Load inline type definitions
  for (const def of json.module_defs ?? []) {
    const type = loadModuleFromJSON(def, session)
    session.typeRegistry.set(type.name, type)
  }

  // Create params and triggers
  for (const p of json.params ?? []) {
    if (p.type === 'trigger') {
      session.triggerRegistry.set(p.name, new Trigger())
    } else {
      session.paramRegistry.set(p.name, new Param(p.value ?? 0.0, p.time_const ?? 0.005))
    }
  }

  // Instantiate modules (TS-only)
  for (const entry of json.modules) {
    const type = session.typeRegistry.get(entry.type)
    if (!type) throw new Error(`Unknown module type '${entry.type}'.`)
    const name = entry.name ?? nextName(session, type.name)
    const inst = type.instantiateAs(name)
    session.instanceRegistry.set(inst.name, inst)
  }

  // Populate wiring state — connections, input expressions, outputs
  for (const conn of json.connections ?? []) {
    const srcInst = session.instanceRegistry.get(conn.src)
    const dstInst = session.instanceRegistry.get(conn.dst)
    if (!srcInst) throw new Error(`Connection src module '${conn.src}' not found.`)
    if (!dstInst) throw new Error(`Connection dst module '${conn.dst}' not found.`)
    const srcId = typeof conn.src_output === 'number' ? conn.src_output : srcInst.outputIndex(conn.src_output)
    const dstId = typeof conn.dst_input  === 'number' ? conn.dst_input  : dstInst.inputIndex(conn.dst_input)
    const srcOutName = srcInst.outputNames[srcId]
    const dstInName  = dstInst.inputNames[dstId]
    session.inputExprNodes.set(`${conn.dst}:${dstInName}`, { op: 'ref', module: conn.src, output: srcOutName })
  }

  for (const ie of json.input_exprs ?? []) {
    const inst = session.instanceRegistry.get(ie.module)
    if (!inst) throw new Error(`input_expr module '${ie.module}' not found.`)
    session.inputExprNodes.set(`${ie.module}:${ie.input}`, ie.expr)
  }

  // Ensure module input defaults are included in the plan
  for (const [name, inst] of session.instanceRegistry) {
    const defaults = inst._def.rawInputDefaults as Record<string, ExprNode>
    for (const [inputName, value] of Object.entries(defaults)) {
      const key = `${name}:${inputName}`
      if (!session.inputExprNodes.has(key)) {
        session.inputExprNodes.set(key, value)
      }
    }
  }

  for (const out of json.outputs ?? []) {
    if ('expr' in out) {
      throw new Error('Output expressions not supported in plan-based path. Use module output refs instead.')
    }
    const inst = session.instanceRegistry.get(out.module)
    if (!inst) throw new Error(`Output module '${out.module}' not found.`)
    session.graphOutputs.push({ module: out.module, output: String(out.output) })
  }

  // Apply wiring via FlatRuntime
  applyFlatPlan(session, session.runtime)
}

// ─────────────────────────────────────────────────────────────
// Patch saver
// ─────────────────────────────────────────────────────────────

export function savePatchToJSON(session: SessionState): PatchJSON {
  const modules: PatchJSON['modules'] = []
  for (const [name, inst] of session.instanceRegistry) {
    modules.push({ type: inst.typeName, name })
  }

  const params: NonNullable<PatchJSON['params']> = []
  for (const [name, p] of session.paramRegistry) {
    params.push({ name, value: p.value, time_const: 0.005 })
  }
  for (const [name] of session.triggerRegistry) {
    params.push({ name, type: 'trigger' })
  }

  const outputs: NonNullable<PatchJSON['outputs']> = session.graphOutputs.map(o => ({
    module: o.module, output: o.output,
  }))

  const inputExprs: NonNullable<PatchJSON['input_exprs']> = []
  for (const [key, node] of session.inputExprNodes) {
    const [module, input] = key.split(':')
    inputExprs.push({ module, input, expr: node })
  }

  const patch: PatchJSON = {
    schema: 'tropical_patch_1',
    modules,
  }
  if (params.length)      patch.params      = params
  if (outputs.length)     patch.outputs     = outputs
  if (inputExprs.length)  patch.input_exprs = inputExprs

  return patch
}
