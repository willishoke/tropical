/**
 * SignalExpr — symbolic expression wrapper. Port of egress/expr.py.
 *
 * TypeScript has no operator overloading, so all operations are named
 * free functions: add(a, b), mul(a, b), sin(x), etc.
 *
 * SignalExpr is a pure wrapper around an ExprNode (JSON-serializable tree).
 * No C handles — all expression evaluation happens via FlatRuntime's plan JSON.
 */

// ---------- ExprNode (JSON-serializable expression tree) ----------

/** An expression node — bare scalar, inline array, or a named op object. */
export type ExprNode =
  | number
  | boolean
  | ExprNode[]
  | { op: string; [key: string]: unknown }

// ---------- SignalExpr ----------

export class SignalExpr {
  /** JSON-serializable expression tree. */
  _node: ExprNode

  private constructor(node: ExprNode) {
    this._node = node
  }

  static fromNode(node: ExprNode): SignalExpr {
    return new SignalExpr(node)
  }

  /** @deprecated Alias for fromNode — old code used fromHandle(handle, node). */
  static fromHandle(_handle: unknown, node: ExprNode): SignalExpr {
    return new SignalExpr(node)
  }

  /** Index into an array expression: expr[idx] */
  at(idx: ExprCoercible): SignalExpr {
    const i = coerce(idx)
    return new SignalExpr({ op: 'index', args: [this._node, i._node] })
  }
}

// ---------- Coercion ----------

export type ExprCoercible = SignalExpr | number | boolean | ExprCoercible[]

/** Convert a scalar, boolean, array, or SignalExpr to a SignalExpr. */
export function coerce(value: ExprCoercible): SignalExpr {
  if (value instanceof SignalExpr) return value
  if (typeof value === 'boolean') return SignalExpr.fromNode(value)
  if (typeof value === 'number') return SignalExpr.fromNode(value)
  if (Array.isArray(value)) return arrayPack(value)
  throw new TypeError(`Cannot coerce ${typeof value} to SignalExpr`)
}

// ---------- Internal helpers ----------

function binary(opName: string, lhs: ExprCoercible, rhs: ExprCoercible): SignalExpr {
  const l = coerce(lhs)
  const r = coerce(rhs)
  return SignalExpr.fromNode({ op: opName, args: [l._node, r._node] })
}

function unary(opName: string, operand: ExprCoercible): SignalExpr {
  const o = coerce(operand)
  return SignalExpr.fromNode({ op: opName, args: [o._node] })
}

// ---------- Arithmetic ----------

export const add      = (lhs: ExprCoercible, rhs: ExprCoercible) => binary('add',       lhs, rhs)
export const sub      = (lhs: ExprCoercible, rhs: ExprCoercible) => binary('sub',       lhs, rhs)
export const mul      = (lhs: ExprCoercible, rhs: ExprCoercible) => binary('mul',       lhs, rhs)
export const div      = (lhs: ExprCoercible, rhs: ExprCoercible) => binary('div',       lhs, rhs)
export const floorDiv = (lhs: ExprCoercible, rhs: ExprCoercible) => binary('floor_div', lhs, rhs)
export const mod      = (lhs: ExprCoercible, rhs: ExprCoercible) => binary('mod',       lhs, rhs)
export const pow_     = (lhs: ExprCoercible, rhs: ExprCoercible) => binary('pow',       lhs, rhs)
export const matmul   = (lhs: ExprCoercible, rhs: ExprCoercible) => binary('matmul',    lhs, rhs)

// ---------- Comparison ----------

export const lt  = (lhs: ExprCoercible, rhs: ExprCoercible) => binary('lt',  lhs, rhs)
export const lte = (lhs: ExprCoercible, rhs: ExprCoercible) => binary('lte', lhs, rhs)
export const gt  = (lhs: ExprCoercible, rhs: ExprCoercible) => binary('gt',  lhs, rhs)
export const gte = (lhs: ExprCoercible, rhs: ExprCoercible) => binary('gte', lhs, rhs)
export const eq  = (lhs: ExprCoercible, rhs: ExprCoercible) => binary('eq',  lhs, rhs)
export const neq = (lhs: ExprCoercible, rhs: ExprCoercible) => binary('neq', lhs, rhs)

// ---------- Bitwise ----------

export const bitAnd  = (lhs: ExprCoercible, rhs: ExprCoercible) => binary('bit_and', lhs, rhs)
export const bitOr   = (lhs: ExprCoercible, rhs: ExprCoercible) => binary('bit_or',  lhs, rhs)
export const bitXor  = (lhs: ExprCoercible, rhs: ExprCoercible) => binary('bit_xor', lhs, rhs)
export const lshift  = (lhs: ExprCoercible, rhs: ExprCoercible) => binary('lshift',  lhs, rhs)
export const rshift  = (lhs: ExprCoercible, rhs: ExprCoercible) => binary('rshift',  lhs, rhs)
export const bitNot  = (operand: ExprCoercible) => unary('bit_not', operand)

// ---------- Unary / math ----------

export const neg        = (operand: ExprCoercible) => unary('neg', operand)
export const abs_       = (operand: ExprCoercible) => unary('abs', operand)
export const sin        = (operand: ExprCoercible) => unary('sin', operand)
export const log        = (operand: ExprCoercible) => unary('log', operand)
export const logicalNot = (operand: ExprCoercible) => unary('not', operand)

export function clamp(value: ExprCoercible, lo: ExprCoercible, hi: ExprCoercible): SignalExpr {
  const v = coerce(value)
  const l = coerce(lo)
  const h = coerce(hi)
  return SignalExpr.fromNode({ op: 'clamp', args: [v._node, l._node, h._node] })
}

export function select(cond: ExprCoercible, thenVal: ExprCoercible, elseVal: ExprCoercible): SignalExpr {
  const c = coerce(cond)
  const t = coerce(thenVal)
  const e = coerce(elseVal)
  return SignalExpr.fromNode({ op: 'select', args: [c._node, t._node, e._node] })
}

// ---------- Array operations ----------

export function arrayPack(values: ExprCoercible[]): SignalExpr {
  const items = values.map(coerce)
  return SignalExpr.fromNode(items.map(e => e._node))
}

export function arraySet(arrExpr: ExprCoercible, idx: ExprCoercible, val: ExprCoercible): SignalExpr {
  const a = coerce(arrExpr)
  const i = coerce(idx)
  const v = coerce(val)
  return SignalExpr.fromNode({ op: 'array_set', args: [a._node, i._node, v._node] })
}

/** Build a matrix literal expression from a row-major 2D array of numbers. */
export function matrix(rows: number[][]): SignalExpr {
  return SignalExpr.fromNode({ op: 'matrix', rows })
}

// ---------- Function expressions ----------

export function exprFunction(paramCount: number, body: SignalExpr): SignalExpr {
  return SignalExpr.fromNode({ op: 'function', param_count: paramCount, body: body._node })
}

export function exprCall(fn: SignalExpr, args: ExprCoercible[]): SignalExpr {
  const coerced = args.map(coerce)
  return SignalExpr.fromNode({ op: 'call', callee: fn._node, args: coerced.map(e => e._node) })
}

// ---------- ADT expression builders ----------

export function constructStruct(typeName: string, fieldExprs: ExprCoercible[]): SignalExpr {
  const items = fieldExprs.map(coerce)
  return SignalExpr.fromNode({ op: 'construct_struct', type_name: typeName, fields: items.map(e => e._node) })
}

export function fieldAccess(typeName: string, structExpr: ExprCoercible, fieldIndex: number): SignalExpr {
  const s = coerce(structExpr)
  return SignalExpr.fromNode({ op: 'field_access', type_name: typeName, struct_expr: s._node, field_index: fieldIndex })
}

export function constructVariant(typeName: string, variantTag: number, payloadExprs: ExprCoercible[]): SignalExpr {
  const items = payloadExprs.map(coerce)
  return SignalExpr.fromNode({ op: 'construct_variant', type_name: typeName, variant_tag: variantTag, payload: items.map(e => e._node) })
}

export function matchVariant(typeName: string, scrutinee: ExprCoercible, branchExprs: ExprCoercible[]): SignalExpr {
  const s = coerce(scrutinee)
  const items = branchExprs.map(coerce)
  return SignalExpr.fromNode({ op: 'match_variant', type_name: typeName, scrutinee: s._node, branches: items.map(e => e._node) })
}

// ---------- Leaf node constructors ----------

export function sampleRate(): SignalExpr {
  return SignalExpr.fromNode({ op: 'sample_rate' })
}

export function sampleIndex(): SignalExpr {
  return SignalExpr.fromNode({ op: 'sample_index' })
}

export function inputExpr(inputId: number): SignalExpr {
  return SignalExpr.fromNode({ op: 'input', id: inputId })
}

export function registerExpr(regId: number): SignalExpr {
  return SignalExpr.fromNode({ op: 'reg', id: regId })
}

export function refExpr(moduleName: string, outputId: number): SignalExpr {
  return SignalExpr.fromNode({ op: 'ref', module: moduleName, output: outputId })
}

export function nestedOutputExpr(nodeId: number, outputId: number): SignalExpr {
  return SignalExpr.fromNode({ op: 'nested_output', node_id: nodeId, output_id: outputId })
}

export function delayValueExpr(nodeId: number): SignalExpr {
  return SignalExpr.fromNode({ op: 'delay_value', node_id: nodeId })
}

/** Create a smoothed-param expression node for use in wiring expressions. */
export function paramExpr(paramHandle: unknown): SignalExpr {
  return SignalExpr.fromNode({ op: 'smoothed_param', _ptr: true, _handle: paramHandle })
}

/** Create a trigger-param expression node for use in wiring expressions. */
export function triggerParamExpr(paramHandle: unknown): SignalExpr {
  return SignalExpr.fromNode({ op: 'trigger_param', _ptr: true, _handle: paramHandle })
}
