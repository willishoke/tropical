/**
 * SignalExpr — symbolic expression wrapper. Port of tropical/expr.py.
 *
 * TypeScript has no operator overloading, so all operations are named
 * free functions: add(a, b), mul(a, b), sin(x), etc.
 *
 * SignalExpr is a pure wrapper around an ExprNode (JSON-serializable tree).
 * No C handles — all expression evaluation happens via FlatRuntime's plan JSON.
 */

import { broadcastShapes } from './term.js'

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
  /** Static shape, if known at module-definition time. undefined = scalar or unknown. */
  readonly shape: number[] | undefined

  private constructor(node: ExprNode, shape?: number[]) {
    this._node = node
    this.shape = shape
  }

  static fromNode(node: ExprNode, shape?: number[]): SignalExpr {
    return new SignalExpr(node, shape)
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

  /** Reshape this array to a new shape. */
  reshape(newShape: number[]): SignalExpr {
    return new SignalExpr({ op: 'reshape', args: [this._node], shape: newShape })
  }

  /** Transpose a 2D array. */
  transpose(): SignalExpr {
    return new SignalExpr({ op: 'transpose', args: [this._node] })
  }

  /** Slice along an axis: [start, end). */
  slice(axis: number, start: number, end: number): SignalExpr {
    return new SignalExpr({ op: 'slice', args: [this._node], axis, start, end })
  }

  /** Reduce along an axis with an associative op ('add', 'mul', 'min', 'max'). */
  reduce(axis: number, reduceOp: string): SignalExpr {
    return new SignalExpr({ op: 'reduce', args: [this._node], axis, reduce_op: reduceOp })
  }

  /** Sum all elements (reduce over axis 0 with 'add'). */
  sum(axis = 0): SignalExpr {
    return this.reduce(axis, 'add')
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

function propagateBinaryShape(l: SignalExpr, r: SignalExpr): number[] | undefined {
  if (l.shape !== undefined && r.shape !== undefined) {
    return broadcastShapes(l.shape, r.shape) ?? undefined
  }
  if (l.shape !== undefined) return l.shape
  if (r.shape !== undefined) return r.shape
  return undefined
}

function binary(opName: string, lhs: ExprCoercible, rhs: ExprCoercible): SignalExpr {
  const l = coerce(lhs)
  const r = coerce(rhs)
  return SignalExpr.fromNode({ op: opName, args: [l._node, r._node] }, propagateBinaryShape(l, r))
}

function unary(opName: string, operand: ExprCoercible): SignalExpr {
  const o = coerce(operand)
  return SignalExpr.fromNode({ op: opName, args: [o._node] }, o.shape)
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
export const cos        = (operand: ExprCoercible) => unary('cos', operand)
export const exp        = (operand: ExprCoercible) => unary('exp', operand)
export const log        = (operand: ExprCoercible) => unary('log', operand)
export const tanh       = (operand: ExprCoercible) => unary('tanh', operand)
export const logicalNot = (operand: ExprCoercible) => unary('not', operand)

export function clamp(value: ExprCoercible, lo: ExprCoercible, hi: ExprCoercible): SignalExpr {
  const v = coerce(value)
  const l = coerce(lo)
  const h = coerce(hi)
  const shape = propagateBinaryShape(v, l) ?? v.shape ?? l.shape ?? h.shape
  return SignalExpr.fromNode({ op: 'clamp', args: [v._node, l._node, h._node] }, shape)
}

export function select(cond: ExprCoercible, thenVal: ExprCoercible, elseVal: ExprCoercible): SignalExpr {
  const c = coerce(cond)
  const t = coerce(thenVal)
  const e = coerce(elseVal)
  const shape = propagateBinaryShape(t, e) ?? t.shape ?? e.shape ?? c.shape
  return SignalExpr.fromNode({ op: 'select', args: [c._node, t._node, e._node] }, shape)
}

// ---------- Array operations ----------

export function arrayPack(values: ExprCoercible[]): SignalExpr {
  const items = values.map(coerce)
  return SignalExpr.fromNode(items.map(e => e._node), [items.length])
}

export function arraySet(arrExpr: ExprCoercible, idx: ExprCoercible, val: ExprCoercible): SignalExpr {
  const a = coerce(arrExpr)
  const i = coerce(idx)
  const v = coerce(val)
  return SignalExpr.fromNode({ op: 'array_set', args: [a._node, i._node, v._node] }, a.shape)
}

/** Build a matrix literal expression from a row-major 2D array of numbers. */
export function matrix(rows: number[][]): SignalExpr {
  return SignalExpr.fromNode({ op: 'matrix', rows })
}

// ---------- First-class array operations (static shapes) ----------

/**
 * Typed array literal with explicit shape.
 * Values are provided in row-major order and must match product(shape).
 */
export function arrayLiteral(shape: number[], values: ExprCoercible[]): SignalExpr {
  const items = values.map(v => coerce(v)._node)
  return SignalExpr.fromNode({ op: 'array_literal', shape, values: items }, shape)
}

/** Create an array filled with zeros. */
export function zeros(shape: number[]): SignalExpr {
  return SignalExpr.fromNode({ op: 'zeros', shape }, shape)
}

/** Create an array filled with ones. */
export function ones(shape: number[]): SignalExpr {
  return SignalExpr.fromNode({ op: 'ones', shape }, shape)
}

/** Create an array filled with a constant value. */
export function fill(shape: number[], value: ExprCoercible): SignalExpr {
  return SignalExpr.fromNode({ op: 'fill', shape, value: coerce(value)._node }, shape)
}

/** Reshape an array to a new shape (total elements must match). */
export function reshape(arr: ExprCoercible, newShape: number[]): SignalExpr {
  return SignalExpr.fromNode({ op: 'reshape', args: [coerce(arr)._node], shape: newShape })
}

/** Transpose a 2D array (swap axes). */
export function transpose(arr: ExprCoercible): SignalExpr {
  return unary('transpose', arr)
}

/**
 * Slice an array along a dimension.
 * Returns elements [start, end) along the given axis.
 */
export function slice(arr: ExprCoercible, axis: number, start: number, end: number): SignalExpr {
  return SignalExpr.fromNode({ op: 'slice', args: [coerce(arr)._node], axis, start, end })
}

/**
 * Reduce an array along an axis using an associative operation.
 * reduceOp is one of: 'add', 'mul', 'min', 'max'.
 */
export function reduce(arr: ExprCoercible, axis: number, reduceOp: string): SignalExpr {
  return SignalExpr.fromNode({ op: 'reduce', args: [coerce(arr)._node], axis, reduce_op: reduceOp })
}

/** Explicitly broadcast an array to a target shape. */
export function broadcastTo(arr: ExprCoercible, shape: number[]): SignalExpr {
  return SignalExpr.fromNode({ op: 'broadcast_to', args: [coerce(arr)._node], shape }, shape)
}

/** Map a function over array elements: map(fn, arr) applies fn to each element. */
export function mapArray(fn: (elem: SignalExpr) => SignalExpr, arr: ExprCoercible): SignalExpr {
  // Build as: map(function(1, body), arr) — function takes 1 param (the element)
  const param = inputExpr(0)
  const body = fn(param)
  return SignalExpr.fromNode({
    op: 'map',
    callee: { op: 'function', param_count: 1, body: body._node },
    args: [coerce(arr)._node],
  })
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
