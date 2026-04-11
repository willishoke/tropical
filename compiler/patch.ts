/**
 * Session state, expression pretty-printer, JSON loading, and ProgramJSON → ProgramDef builder.
 */

import {
  type SignalExpr, type ExprCoercible, coerce, type ExprNode, validateExpr,
} from './expr.js'
import {
  ModuleType, ModuleInstance,
  type ProgramDef, type NestedCall, type ValueCoercible,
} from './module.js'
import { Runtime } from './runtime/runtime.js'
import { loadProgramAsSession, type ProgramJSON } from './program.js'
import { Param, Trigger } from './runtime/param.js'

// ─────────────────────────────────────────────────────────────
// JSON schema types
// ─────────────────────────────────────────────────────────────

// ExprNode is defined in expr.ts and re-exported here for backward compatibility.
export type { ExprNode } from './expr.js'



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
    return { ...obj, items } as unknown as ExprNode
  }

  // Combinator forms with nested expr fields
  if (op === 'let') {
    const bind = obj.bind as Record<string, ExprNode>
    const converted: Record<string, ExprNode> = {}
    for (const [k, v] of Object.entries(bind)) converted[k] = recurse(v)
    return { ...obj, bind: converted, in: recurse(obj.in as ExprNode) } as unknown as ExprNode
  }
  if (op === 'generate' || op === 'chain' || op === 'iterate') {
    return { ...obj, ...(obj.init !== undefined ? { init: recurse(obj.init as ExprNode) } : {}), body: recurse(obj.body as ExprNode) } as unknown as ExprNode
  }
  if (op === 'fold' || op === 'scan') {
    return { ...obj, over: recurse(obj.over as ExprNode), init: recurse(obj.init as ExprNode), body: recurse(obj.body as ExprNode) } as unknown as ExprNode
  }
  if (op === 'map2') {
    return { ...obj, over: recurse(obj.over as ExprNode), body: recurse(obj.body as ExprNode) } as unknown as ExprNode
  }
  if (op === 'zip_with') {
    return { ...obj, a: recurse(obj.a as ExprNode), b: recurse(obj.b as ExprNode), body: recurse(obj.body as ExprNode) } as unknown as ExprNode
  }

  // ADT ops
  if (op === 'construct_struct') {
    return { ...obj, fields: (obj.fields as ExprNode[]).map(recurse) } as unknown as ExprNode
  }
  if (op === 'field_access') {
    return { ...obj, struct_expr: recurse(obj.struct_expr as ExprNode) } as unknown as ExprNode
  }
  if (op === 'construct_variant') {
    return { ...obj, payload: (obj.payload as ExprNode[]).map(recurse) } as unknown as ExprNode
  }
  if (op === 'match_variant') {
    return { ...obj, scrutinee: recurse(obj.scrutinee as ExprNode), branches: (obj.branches as ExprNode[]).map(recurse) } as unknown as ExprNode
  }

  // Leaf ops (sample_rate, sample_index, binding, float, int, bool, matrix, etc.)
  return node
}

export function loadProgramDef(
  def: ProgramJSON,
  session: Pick<SessionState, 'typeRegistry' | 'instanceRegistry' | 'paramRegistry' | 'triggerRegistry'>,
): ModuleType {
  const inputSpecs  = def.inputs ?? []
  const outputSpecs = def.outputs ?? []
  const inputNames  = inputSpecs.map(i => typeof i === 'string' ? i : i.name)
  const outputNames = outputSpecs.map(o => typeof o === 'string' ? o : o.name)
  const inputPortTypes  = inputSpecs.map(i => typeof i === 'string' ? undefined : i.type)
  const outputPortTypes = outputSpecs.map(o => typeof o === 'string' ? undefined : o.type)
  const regsRaw     = def.regs ?? {}
  const delaysRaw   = def.delays ?? {}
  const nestedRaw   = def.instances ?? {}

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
    const type = session.typeRegistry.get(spec.program)
    if (!type) throw new Error(`Unknown program type '${spec.program}' in instances.`)
    nestedAliasDef.set(alias, type._def)

    // Build call arg nodes — order inputs by the nested type's input order
    const callArgNodes: ExprNode[] = type._def.inputNames.map((name, idx) => {
      if (name in (spec.inputs ?? {})) {
        return slottifyExpr((spec.inputs ?? {})[name], inputNames, regNames, delayNameToId, nestedAliasToId, nestedAliasDef)
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
  const process = def.process ?? { outputs: {} }
  const outputExprNodes = outputNames.map(name => {
    const node = process.outputs[name]
    if (node === undefined) throw new Error(`Output '${name}' missing from process.outputs.`)
    return slottifyExpr(node, inputNames, regNames, delayNameToId, nestedAliasToId, nestedAliasDef)
  })

  // ── Convert register update expressions ──
  const registerExprNodes: (ExprNode | null)[] = regNames.map(name => {
    const node = process.next_regs?.[name]
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
    loadProgramAsSession(json as unknown as ProgramJSON, session)
    return
  }
  throw new Error(`Unknown schema '${json.schema}'. Expected 'tropical_program_1'.`)
}

