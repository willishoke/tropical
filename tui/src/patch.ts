/**
 * JSON patch / module definition schema, expression AST builder, and load/save.
 * Replaces egress/yaml_schema.py with a clean-break JSON format.
 */

import * as b from './bindings.js'
import {
  SignalExpr, ExprCoercible, coerce,
  add, sub, mul, div, floorDiv, mod, pow_, matmul,
  lt, lte, gt, gte, eq, neq,
  bitAnd, bitOr, bitXor, lshift, rshift, bitNot,
  neg, abs_, sin, log, logicalNot,
  clamp, select, arrayPack, arraySet, matrix,
  inputExpr, registerExpr, refExpr, sampleRate, sampleIndex,
  paramExpr, triggerParamExpr,
  constructStruct, fieldAccess, constructVariant, matchVariant,
} from './expr.js'
import {
  defineModule, ModuleType, ModuleInstance, delay,
  valueHandle, SymbolMap, ValueCoercible, RegInit,
} from './module.js'
import { Graph } from './graph.js'
import { DAC } from './audio.js'
import { Param, Trigger } from './param.js'
import { applySessionWiring, applyFlatPlan } from './apply_plan.js'

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
  inputs: string[]
  outputs: string[]
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
  schema: 'egress_patch_1'
  config?: {
    buffer_length?: number
    worker_count?: number
    fusion_enabled?: boolean
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
  graph: Graph
  dac: DAC | null
  typeRegistry: Map<string, ModuleType>
  instanceRegistry: Map<string, ModuleInstance>
  graphOutputs: Array<{ module: string; output: string }>
  paramRegistry: Map<string, Param>
  triggerRegistry: Map<string, Trigger>
  /** Canonical input wiring: key is `${module}:${input}`, value is the ExprNode for round-trip save. */
  inputExprNodes: Map<string, ExprNode>  // key: `${module}:${input}`
  /** Optional FlatRuntime — when set, wiring goes through applyFlatPlan instead of Graph. */
  runtime?: import('./runtime.js').Runtime
}

export function makeSession(bufferLength = 512): SessionState {
  return {
    graph: new Graph(bufferLength),
    dac: null,
    typeRegistry: new Map(),
    instanceRegistry: new Map(),
    graphOutputs: [],
    paramRegistry: new Map(),
    triggerRegistry: new Map(),
    inputExprNodes: new Map(),
  }
}

// ─────────────────────────────────────────────────────────────
// Op dispatch tables
// ─────────────────────────────────────────────────────────────

const BINARY_OPS: Record<string, (l: ExprCoercible, r: ExprCoercible) => SignalExpr> = {
  add, sub, mul, div, floor_div: floorDiv, mod, pow: pow_, matmul,
  lt, lte, gt, gte, eq, neq,
  bit_and: bitAnd, bit_or: bitOr, bit_xor: bitXor, lshift, rshift,
}

const UNARY_OPS: Record<string, (x: ExprCoercible) => SignalExpr> = {
  neg, abs: abs_, sin, log, not: logicalNot, bit_not: bitNot,
}

// ─────────────────────────────────────────────────────────────
// Build context
// ─────────────────────────────────────────────────────────────

interface BuildCtx {
  inputNames: string[]
  regNames: string[]
  paramRegistry: Map<string, Param>
  triggerRegistry: Map<string, Trigger>
  instanceRegistry: Map<string, ModuleInstance>
  typeRegistry: Map<string, ModuleType>
  /** Named delay value exprs, populated during the delay pre-pass. */
  delayRefs: Map<string, SignalExpr>
  /** Named nested sub-module output exprs, populated during the nested pre-pass. */
  nestedAliases: Map<string, SignalExpr | SignalExpr[]>
}

// ─────────────────────────────────────────────────────────────
// Expression builder
// ─────────────────────────────────────────────────────────────

/**
 * Recursively build a SignalExpr from a JSON ExprNode.
 * Throws with a descriptive message on invalid input (validates as it builds).
 */
export function buildExpr(node: ExprNode, ctx: BuildCtx): SignalExpr {
  // ── Scalar shorthand ──
  if (typeof node === 'number' || typeof node === 'boolean') return coerce(node)
  if (Array.isArray(node)) return arrayPack(node.map(n => buildExpr(n, ctx)))

  const { op, args: rawArgs = [] } = node as { op: string; args?: ExprNode[]; [k: string]: unknown }
  const args = rawArgs as ExprNode[]

  // ── References ──
  if (op === 'input') {
    const n = node as { op: string; name?: string; id?: number }
    if (n.name !== undefined) {
      const idx = ctx.inputNames.indexOf(n.name)
      if (idx === -1) throw new Error(`Unknown input '${n.name}'. Available: ${ctx.inputNames.join(', ')}`)
      return inputExpr(idx)
    }
    return inputExpr(n.id!)
  }

  if (op === 'reg') {
    const n = node as { op: string; name?: string; id?: number }
    if (n.name !== undefined) {
      const idx = ctx.regNames.indexOf(n.name)
      if (idx === -1) throw new Error(`Unknown register '${n.name}'. Available: ${ctx.regNames.join(', ')}`)
      return registerExpr(idx)
    }
    return registerExpr(n.id!)
  }

  if (op === 'ref') {
    const n = node as { op: string; module: string; output: string | number }
    const inst = ctx.instanceRegistry.get(n.module)
    if (!inst) throw new Error(`Unknown module instance '${n.module}' in ref expression.`)
    const outputId = typeof n.output === 'number' ? n.output : inst.outputIndex(n.output)
    return refExpr(n.module, outputId)
  }

  if (op === 'param') {
    const n = node as { op: string; name: string }
    const p = ctx.paramRegistry.get(n.name)
    if (!p) throw new Error(`Unknown param '${n.name}'.`)
    return SignalExpr.fromHandle(b.check(b.egress_expr_param(p._h), 'expr_param'), { op: 'smoothed_param', name: n.name, _ptr: true })
  }

  if (op === 'trigger') {
    const n = node as { op: string; name: string }
    const t = ctx.triggerRegistry.get(n.name)
    if (!t) throw new Error(`Unknown trigger '${n.name}'.`)
    return SignalExpr.fromHandle(b.check(b.egress_expr_trigger_param(t._h), 'expr_trigger_param'), { op: 'trigger_param', name: n.name, _ptr: true })
  }

  if (op === 'sample_rate')  return sampleRate()
  if (op === 'sample_index') return sampleIndex()

  // ── Delay ──
  if (op === 'delay') {
    const n = node as { op: string; args: ExprNode[]; init?: number; id?: string }
    const updateExpr = buildExpr(n.args[0], ctx)
    const result = delay(updateExpr, n.init ?? 0.0)
    if (n.id !== undefined) ctx.delayRefs.set(n.id, result)
    return result
  }

  if (op === 'delay_ref') {
    const n = node as { op: string; id: string }
    const ref = ctx.delayRefs.get(n.id)
    if (!ref) throw new Error(`delay_ref: no delay with id '${n.id}'. Declare it in 'delays' before use.`)
    return ref
  }

  // ── Nested module output reference ──
  if (op === 'nested_out') {
    const n = node as { op: string; ref: string; output: string | number }
    const alias = ctx.nestedAliases.get(n.ref)
    if (!alias) throw new Error(`nested_out: no nested module named '${n.ref}'.`)
    if (Array.isArray(alias)) {
      const idx = typeof n.output === 'number' ? n.output : null
      if (idx === null) {
        // Look up by output name in the type
        const type = ctx.typeRegistry.get(
          (ctx.instanceRegistry.get(n.ref) ?? null)?.typeName ?? ''
        )
        throw new Error(`nested_out: use numeric index or provide type context for '${n.ref}.${n.output}'`)
      }
      return alias[idx]
    }
    return alias as SignalExpr
  }

  // ── Binary ops ──
  if (op in BINARY_OPS) {
    if (args.length !== 2) throw new Error(`Op '${op}' requires exactly 2 args, got ${args.length}.`)
    return BINARY_OPS[op](buildExpr(args[0], ctx), buildExpr(args[1], ctx))
  }

  // ── Unary ops ──
  if (op in UNARY_OPS) {
    if (args.length !== 1) throw new Error(`Op '${op}' requires exactly 1 arg, got ${args.length}.`)
    return UNARY_OPS[op](buildExpr(args[0], ctx))
  }

  // ── Multi-arg ──
  if (op === 'clamp') {
    if (args.length !== 3) throw new Error(`'clamp' requires 3 args.`)
    return clamp(buildExpr(args[0], ctx), buildExpr(args[1], ctx), buildExpr(args[2], ctx))
  }
  if (op === 'select') {
    if (args.length !== 3) throw new Error(`'select' requires 3 args.`)
    return select(buildExpr(args[0], ctx), buildExpr(args[1], ctx), buildExpr(args[2], ctx))
  }
  if (op === 'index') {
    if (args.length !== 2) throw new Error(`'index' requires 2 args.`)
    return buildExpr(args[0], ctx).at(buildExpr(args[1], ctx))
  }
  if (op === 'array_set') {
    if (args.length !== 3) throw new Error(`'array_set' requires 3 args.`)
    return arraySet(buildExpr(args[0], ctx), buildExpr(args[1], ctx), buildExpr(args[2], ctx))
  }
  if (op === 'array') {
    const n = node as { op: string; items: ExprNode[] }
    return arrayPack(n.items.map(i => buildExpr(i, ctx)))
  }
  if (op === 'matrix') {
    const n = node as { op: string; rows: number[][] }
    return matrix(n.rows)
  }

  // ── ADT operations ──
  if (op === 'construct_struct') {
    const n = node as { op: string; type_name: string; fields: ExprNode[] }
    return constructStruct(n.type_name, n.fields.map(f => buildExpr(f, ctx)))
  }
  if (op === 'field_access') {
    const n = node as { op: string; type_name: string; struct_expr: ExprNode; field_index: number }
    return fieldAccess(n.type_name, buildExpr(n.struct_expr, ctx), n.field_index)
  }
  if (op === 'construct_variant') {
    const n = node as { op: string; type_name: string; variant_tag: number; payload: ExprNode[] }
    return constructVariant(n.type_name, n.variant_tag, n.payload.map(p => buildExpr(p, ctx)))
  }
  if (op === 'match_variant') {
    const n = node as { op: string; type_name: string; scrutinee: ExprNode; branches: ExprNode[] }
    return matchVariant(n.type_name, buildExpr(n.scrutinee, ctx), n.branches.map(b => buildExpr(b, ctx)))
  }

  // ── Explicit literal forms (rarely needed, bare numbers are preferred) ──
  if (op === 'float') return coerce((node as { op: string; value: number }).value)
  if (op === 'int')   return coerce((node as { op: string; value: number }).value)
  if (op === 'bool')  return coerce((node as { op: string; value: boolean }).value)

  throw new Error(`Unknown expr op: '${op}'.`)
}

// ─────────────────────────────────────────────────────────────
// Validator (structural only — no C API calls)
// ─────────────────────────────────────────────────────────────

export type ExprType = 'scalar' | 'array' | 'bool' | 'unknown'

interface ValidateCtx {
  inputNames: string[]
  regNames: string[]
  paramNames: Set<string>
  triggerNames: Set<string>
  instanceNames: Set<string>
  delayIds: Set<string>
  nestedIds: Set<string>
}

export function validateExpr(node: ExprNode, ctx: ValidateCtx): ExprType {
  if (typeof node === 'number')  return 'scalar'
  if (typeof node === 'boolean') return 'bool'
  if (Array.isArray(node))       return 'array'

  const { op } = node as { op: string; [k: string]: unknown }

  if (op === 'input') {
    const n = node as { op: string; name?: string; id?: number }
    if (n.name !== undefined && !ctx.inputNames.includes(n.name))
      throw new Error(`Unknown input '${n.name}'.`)
    return 'scalar'
  }
  if (op === 'reg') {
    const n = node as { op: string; name?: string; id?: number }
    if (n.name !== undefined && !ctx.regNames.includes(n.name))
      throw new Error(`Unknown register '${n.name}'.`)
    return 'unknown'
  }
  if (op === 'ref') {
    const n = node as { op: string; module: string }
    if (!ctx.instanceNames.has(n.module))
      throw new Error(`Unknown module instance '${n.module}' in ref.`)
    return 'scalar'
  }
  if (op === 'param') {
    const n = node as { op: string; name: string }
    if (!ctx.paramNames.has(n.name)) throw new Error(`Unknown param '${n.name}'.`)
    return 'scalar'
  }
  if (op === 'trigger') {
    const n = node as { op: string; name: string }
    if (!ctx.triggerNames.has(n.name)) throw new Error(`Unknown trigger '${n.name}'.`)
    return 'scalar'
  }
  if (op === 'sample_rate' || op === 'sample_index') return 'scalar'
  if (op === 'delay') {
    const n = node as { op: string; id?: string }
    if (n.id !== undefined) ctx.delayIds.add(n.id)
    return 'scalar'
  }
  if (op === 'delay_ref') {
    const n = node as { op: string; id: string }
    if (!ctx.delayIds.has(n.id)) throw new Error(`delay_ref '${n.id}' has no matching delay declaration.`)
    return 'scalar'
  }
  if (op === 'nested_out') {
    const n = node as { op: string; ref: string }
    if (!ctx.nestedIds.has(n.ref)) throw new Error(`nested_out '${n.ref}' has no matching nested declaration.`)
    return 'scalar'
  }
  if (op in BINARY_OPS || op in UNARY_OPS || ['clamp','select','index','array_set'].includes(op))
    return 'scalar'
  if (op === 'array' || op === 'array_set') return 'array'
  if (op === 'matrix') return 'array'
  if (op === 'float' || op === 'int') return 'scalar'
  if (op === 'bool') return 'bool'
  if (op === 'construct_struct')  return 'unknown'
  if (op === 'field_access')      return 'scalar'
  if (op === 'construct_variant') return 'unknown'
  if (op === 'match_variant')     return 'unknown'
  throw new Error(`Unknown op '${op}'.`)
}

// ─────────────────────────────────────────────────────────────
// Module loader
// ─────────────────────────────────────────────────────────────

export function loadModuleFromJSON(
  def: ModuleDefJSON,
  session: Pick<SessionState, 'typeRegistry' | 'instanceRegistry' | 'paramRegistry' | 'triggerRegistry'>,
): ModuleType {
  const inputNames  = def.inputs
  const outputNames = def.outputs
  const regsRaw     = def.regs ?? {}
  const delaysRaw   = def.delays ?? {}
  const nestedRaw   = def.nested ?? {}

  // Convert regs to defineModule format (supports bare values and { init, type })
  const regsForDefine: Record<string, RegInit> = {}
  for (const [name, val] of Object.entries(regsRaw)) {
    if (typeof val === 'object' && val !== null && !Array.isArray(val) && 'init' in val) {
      regsForDefine[name] = { init: (val as { init: ValueCoercible; type: string }).init, type: (val as { init: ValueCoercible; type: string }).type }
    } else {
      regsForDefine[name] = val as ValueCoercible
    }
  }

  // Parse input defaults (build each default through the expr builder so they become SignalExpr)
  const inputDefaultsForDefine: Record<string, ExprCoercible> | undefined = def.input_defaults
    ? Object.fromEntries(
        Object.entries(def.input_defaults).map(([k, v]) => {
          const ctx0: BuildCtx = {
            inputNames: [], regNames: [],
            paramRegistry: session.paramRegistry,
            triggerRegistry: session.triggerRegistry,
            instanceRegistry: session.instanceRegistry,
            typeRegistry: session.typeRegistry,
            delayRefs: new Map(), nestedAliases: new Map(),
          }
          return [k, buildExpr(v, ctx0)]
        })
      )
    : undefined

  return defineModule(
    def.name,
    inputNames,
    outputNames,
    regsForDefine,
    (inputsMap: SymbolMap, regsMap: SymbolMap) => {
      const regNames = regsMap.names()
      const ctx: BuildCtx = {
        inputNames,
        regNames,
        paramRegistry: session.paramRegistry,
        triggerRegistry: session.triggerRegistry,
        instanceRegistry: session.instanceRegistry,
        typeRegistry: session.typeRegistry,
        delayRefs: new Map(),
        nestedAliases: new Map(),
      }

      // ── Nested pre-pass ──
      for (const [alias, spec] of Object.entries(nestedRaw)) {
        const type = session.typeRegistry.get(spec.type)
        if (!type) throw new Error(`Unknown module type '${spec.type}' in nested.`)
        // Order inputs by position
        const orderedInputs = type._def.inputNames.map((name) => {
          if (name in spec.inputs) return buildExpr(spec.inputs[name], ctx)
          const idx = type._def.inputNames.indexOf(name)
          const def = type._def.inputDefaults[idx]
          if (def) return def
          throw new Error(`Missing input '${name}' for nested module '${alias}'.`)
        })
        const result = type.call(...orderedInputs)
        ctx.nestedAliases.set(alias, result)
      }

      // ── Delay pre-pass ──
      for (const [id, ds] of Object.entries(delaysRaw)) {
        const updateExpr = buildExpr(ds.update, ctx)
        const result = delay(updateExpr, ds.init ?? 0.0)
        ctx.delayRefs.set(id, result)
      }

      // ── Build output expressions ──
      const outputs: Record<string, ExprCoercible> = {}
      for (const outName of outputNames) {
        const node = def.process.outputs[outName]
        if (node === undefined) throw new Error(`Output '${outName}' missing from process.outputs.`)
        outputs[outName] = buildExpr(node, ctx)
      }

      // ── Build next_regs expressions ──
      const nextRegs: Record<string, ExprCoercible> = {}
      for (const [regName, node] of Object.entries(def.process.next_regs ?? {})) {
        nextRegs[regName] = buildExpr(node, ctx)
      }

      return { outputs, nextRegs }
    },
    def.sample_rate ?? 44100.0,
    inputDefaultsForDefine,
  )
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

  if (op in BINARY_OPS) {
    const sym = BINARY_INFIX[op]
    const l = prettyExpr(args[0], instanceRegistry)
    const r = prettyExpr(args[1], instanceRegistry)
    return sym ? `(${l} ${sym} ${r})` : `${op}(${l}, ${r})`
  }
  if (op in UNARY_OPS) {
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

export function loadPatchFromJSON(json: PatchJSON, session: SessionState): void {
  // Rebuild from a clean slate (same graph object, wiped state)
  const bufLen = json.config?.buffer_length ?? session.graph.bufferLength
  session.graph = new Graph(bufLen)
  if (json.config?.worker_count !== undefined) session.graph.workerCount = json.config.worker_count
  if (json.config?.fusion_enabled !== undefined) session.graph.fusionEnabled = json.config.fusion_enabled

  session.dac = null
  session.instanceRegistry.clear()
  session.graphOutputs.length = 0
  session.paramRegistry.clear()
  session.triggerRegistry.clear()
  session.inputExprNodes.clear()

  // Load ADT type definitions
  for (const td of json.type_defs ?? []) {
    if (td.kind === 'struct') {
      session.graph.defineStruct(td.name, td.fields)
    } else {
      session.graph.defineSumType(td.name, td.variants)
    }
  }

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

  // Batch module instantiation + wiring into a single C++ rebuild.
  // addModule() and setInputExpr() skip rebuilds while a batch is active;
  // loadPlan() detects the active batch and defers its own end_update().
  // The single endUpdate() at the end triggers one fused JIT compile.
  session.graph.beginUpdate()
  try {
    // Instantiate modules (rebuilds deferred)
    for (const entry of json.modules) {
      const type = session.typeRegistry.get(entry.type)
      if (!type) throw new Error(`Unknown module type '${entry.type}'.`)
      const inst = entry.name
        ? type.instantiateAs(session.graph, entry.name)
        : type.instantiate(session.graph)
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

    // Ensure module input defaults are included in the plan so they survive clearWiring()
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

    // Apply wiring: FlatRuntime if available, otherwise Graph pipeline
    if (session.runtime) applyFlatPlan(session, session.runtime)
    else applySessionWiring(session)

    // Single rebuild for all module additions + wiring
    session.graph.endUpdate()
  } catch (e) {
    try { session.graph.endUpdate() } catch {}
    throw e
  }
}

// ─────────────────────────────────────────────────────────────
// Patch merger (additive — no session teardown)
// ─────────────────────────────────────────────────────────────

export function mergePatchFromJSON(json: PatchJSON, session: SessionState): void {
  // Fail fast on name collisions before touching the graph
  for (const entry of json.modules) {
    if (entry.name && session.instanceRegistry.has(entry.name))
      throw new Error(`merge_patch collision: module '${entry.name}' already exists.`)
  }
  for (const p of json.params ?? []) {
    if (session.paramRegistry.has(p.name) || session.triggerRegistry.has(p.name))
      throw new Error(`merge_patch collision: param/trigger '${p.name}' already exists.`)
  }

  // Load ADT type definitions
  for (const td of json.type_defs ?? []) {
    if (td.kind === 'struct') {
      session.graph.defineStruct(td.name, td.fields)
    } else {
      session.graph.defineSumType(td.name, td.variants)
    }
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

  // Batch module instantiation + wiring into a single C++ rebuild.
  session.graph.beginUpdate()
  try {
    // Instantiate modules (rebuilds deferred)
    for (const entry of json.modules) {
      const type = session.typeRegistry.get(entry.type)
      if (!type) throw new Error(`Unknown module type '${entry.type}'.`)
      const inst = entry.name
        ? type.instantiateAs(session.graph, entry.name)
        : type.instantiate(session.graph)
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

    // Ensure module input defaults are included in the plan so they survive clearWiring()
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

    // Apply wiring: FlatRuntime if available, otherwise Graph pipeline
    if (session.runtime) applyFlatPlan(session, session.runtime)
    else applySessionWiring(session)

    // Single rebuild for all module additions + wiring
    session.graph.endUpdate()
  } catch (e) {
    try { session.graph.endUpdate() } catch {}
    throw e
  }
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
    schema: 'egress_patch_1',
    modules,
  }
  if (params.length)      patch.params      = params
  if (outputs.length)     patch.outputs     = outputs
  if (inputExprs.length)  patch.input_exprs = inputExprs

  return patch
}
