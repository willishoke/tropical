/**
 * SignalExpr — symbolic expression wrapper. Port of egress/expr.py.
 *
 * TypeScript has no operator overloading, so all operations are named
 * free functions: add(a, b), mul(a, b), sin(x), etc.
 * SignalExpr holds a koffi opaque pointer; FinalizationRegistry calls
 * egress_expr_free when the object is collected.
 */

import * as b from './bindings.js'

// ---------- FinalizationRegistry ----------

const _registry = new FinalizationRegistry((handle: unknown) => {
  b.egress_expr_free(handle)
})

// ---------- SignalExpr ----------

export class SignalExpr {
  _h: unknown

  private constructor(handle: unknown) {
    this._h = handle
    _registry.register(this, handle, this)
  }

  static fromHandle(handle: unknown): SignalExpr {
    return new SignalExpr(handle)
  }

  /** Explicitly free the underlying C expression before GC. */
  dispose(): void {
    if (this._h !== null) {
      _registry.unregister(this)
      b.egress_expr_free(this._h)
      this._h = null
    }
  }

  /** Index into an array expression: expr[idx] */
  at(idx: ExprCoercible): SignalExpr {
    const i = coerce(idx)
    return SignalExpr.fromHandle(b.check(b.egress_expr_index(this._h, i._h), 'expr_index'))
  }
}

// ---------- Coercion ----------

export type ExprCoercible = SignalExpr | number | boolean | ExprCoercible[]

/** Convert a scalar, boolean, array, or SignalExpr to a SignalExpr. */
export function coerce(value: ExprCoercible): SignalExpr {
  if (value instanceof SignalExpr) return value
  if (typeof value === 'boolean') {
    return SignalExpr.fromHandle(b.check(b.egress_expr_literal_bool(value), 'literal_bool'))
  }
  if (typeof value === 'number') {
    return SignalExpr.fromHandle(b.check(b.egress_expr_literal_float(value), 'literal_float'))
  }
  if (Array.isArray(value)) return arrayPack(value)
  throw new TypeError(`Cannot coerce ${typeof value} to SignalExpr`)
}

// ---------- Internal helpers ----------

function binary(kind: number, lhs: ExprCoercible, rhs: ExprCoercible): SignalExpr {
  const l = coerce(lhs)
  const r = coerce(rhs)
  return SignalExpr.fromHandle(b.check(b.egress_expr_binary(kind, l._h, r._h), 'binary_expr'))
}

function unary(kind: number, operand: ExprCoercible): SignalExpr {
  const op = coerce(operand)
  return SignalExpr.fromHandle(b.check(b.egress_expr_unary(kind, op._h), 'unary_expr'))
}

// ---------- Arithmetic ----------

export const add      = (lhs: ExprCoercible, rhs: ExprCoercible) => binary(b.EXPR_ADD,       lhs, rhs)
export const sub      = (lhs: ExprCoercible, rhs: ExprCoercible) => binary(b.EXPR_SUB,       lhs, rhs)
export const mul      = (lhs: ExprCoercible, rhs: ExprCoercible) => binary(b.EXPR_MUL,       lhs, rhs)
export const div      = (lhs: ExprCoercible, rhs: ExprCoercible) => binary(b.EXPR_DIV,       lhs, rhs)
export const floorDiv = (lhs: ExprCoercible, rhs: ExprCoercible) => binary(b.EXPR_FLOOR_DIV, lhs, rhs)
export const mod      = (lhs: ExprCoercible, rhs: ExprCoercible) => binary(b.EXPR_MOD,       lhs, rhs)
export const pow_     = (lhs: ExprCoercible, rhs: ExprCoercible) => binary(b.EXPR_POW,       lhs, rhs)
export const matmul   = (lhs: ExprCoercible, rhs: ExprCoercible) => binary(b.EXPR_MATMUL,    lhs, rhs)

// ---------- Comparison ----------

export const lt  = (lhs: ExprCoercible, rhs: ExprCoercible) => binary(b.EXPR_LESS,          lhs, rhs)
export const lte = (lhs: ExprCoercible, rhs: ExprCoercible) => binary(b.EXPR_LESS_EQUAL,    lhs, rhs)
export const gt  = (lhs: ExprCoercible, rhs: ExprCoercible) => binary(b.EXPR_GREATER,       lhs, rhs)
export const gte = (lhs: ExprCoercible, rhs: ExprCoercible) => binary(b.EXPR_GREATER_EQUAL, lhs, rhs)
export const eq  = (lhs: ExprCoercible, rhs: ExprCoercible) => binary(b.EXPR_EQUAL,         lhs, rhs)
export const neq = (lhs: ExprCoercible, rhs: ExprCoercible) => binary(b.EXPR_NOT_EQUAL,     lhs, rhs)

// ---------- Bitwise ----------

export const bitAnd  = (lhs: ExprCoercible, rhs: ExprCoercible) => binary(b.EXPR_BIT_AND, lhs, rhs)
export const bitOr   = (lhs: ExprCoercible, rhs: ExprCoercible) => binary(b.EXPR_BIT_OR,  lhs, rhs)
export const bitXor  = (lhs: ExprCoercible, rhs: ExprCoercible) => binary(b.EXPR_BIT_XOR, lhs, rhs)
export const lshift  = (lhs: ExprCoercible, rhs: ExprCoercible) => binary(b.EXPR_LSHIFT,  lhs, rhs)
export const rshift  = (lhs: ExprCoercible, rhs: ExprCoercible) => binary(b.EXPR_RSHIFT,  lhs, rhs)
export const bitNot  = (operand: ExprCoercible) => unary(b.EXPR_BIT_NOT, operand)

// ---------- Unary / math ----------

export const neg        = (operand: ExprCoercible) => unary(b.EXPR_NEG, operand)
export const abs_       = (operand: ExprCoercible) => unary(b.EXPR_ABS, operand)
export const sin        = (operand: ExprCoercible) => unary(b.EXPR_SIN, operand)
export const log        = (operand: ExprCoercible) => unary(b.EXPR_LOG, operand)
export const logicalNot = (operand: ExprCoercible) => unary(b.EXPR_NOT, operand)

export function clamp(value: ExprCoercible, lo: ExprCoercible, hi: ExprCoercible): SignalExpr {
  const v = coerce(value)
  const l = coerce(lo)
  const h = coerce(hi)
  return SignalExpr.fromHandle(b.check(b.egress_expr_clamp(v._h, l._h, h._h), 'clamp'))
}

export function select(cond: ExprCoercible, thenVal: ExprCoercible, elseVal: ExprCoercible): SignalExpr {
  const c = coerce(cond)
  const t = coerce(thenVal)
  const e = coerce(elseVal)
  return SignalExpr.fromHandle(b.check(b.egress_expr_select(c._h, t._h, e._h), 'select'))
}

// ---------- Array operations ----------

export function arrayPack(values: ExprCoercible[]): SignalExpr {
  const items = values.map(coerce)
  const handles = items.map(e => e._h)
  return SignalExpr.fromHandle(b.check(b.egress_expr_array_pack(handles, handles.length), 'array_pack'))
}

export function arraySet(arrExpr: ExprCoercible, idx: ExprCoercible, val: ExprCoercible): SignalExpr {
  const a = coerce(arrExpr)
  const i = coerce(idx)
  const v = coerce(val)
  return SignalExpr.fromHandle(b.check(b.egress_expr_array_set(a._h, i._h, v._h), 'array_set'))
}

/** Build a matrix literal expression from a row-major 2D array of numbers. */
export function matrix(rows: number[][]): SignalExpr {
  const nRows = rows.length
  const nCols = rows[0]?.length ?? 0
  const valueHandles: unknown[] = []
  for (const row of rows) {
    for (const item of row) {
      valueHandles.push(b.check(b.egress_value_float(item), 'value_float'))
    }
  }
  const valH = b.check(b.egress_value_matrix(valueHandles, nRows, nCols), 'value_matrix')
  for (const h of valueHandles) b.egress_value_free(h)
  const exprH = b.check(b.egress_expr_literal_value(valH), 'literal_value')
  b.egress_value_free(valH)
  return SignalExpr.fromHandle(exprH)
}

// ---------- Function expressions ----------

export function exprFunction(paramCount: number, body: SignalExpr): SignalExpr {
  return SignalExpr.fromHandle(b.check(b.egress_expr_function(paramCount, body._h), 'expr_function'))
}

export function exprCall(fn: SignalExpr, args: ExprCoercible[]): SignalExpr {
  const coerced = args.map(coerce)
  const handles = coerced.map(e => e._h)
  return SignalExpr.fromHandle(b.check(b.egress_expr_call(fn._h, handles, handles.length), 'expr_call'))
}

// ---------- Leaf node constructors ----------

export function sampleRate(): SignalExpr {
  return SignalExpr.fromHandle(b.check(b.egress_expr_sample_rate(), 'sample_rate'))
}

export function sampleIndex(): SignalExpr {
  return SignalExpr.fromHandle(b.check(b.egress_expr_sample_index(), 'sample_index'))
}

export function inputExpr(inputId: number): SignalExpr {
  return SignalExpr.fromHandle(b.check(b.egress_expr_input(inputId), 'expr_input'))
}

export function registerExpr(regId: number): SignalExpr {
  return SignalExpr.fromHandle(b.check(b.egress_expr_register(regId), 'expr_register'))
}

export function refExpr(moduleName: string, outputId: number): SignalExpr {
  return SignalExpr.fromHandle(b.check(b.egress_expr_ref(moduleName, outputId), 'expr_ref'))
}

export function nestedOutputExpr(nodeId: number, outputId: number): SignalExpr {
  return SignalExpr.fromHandle(b.check(b.egress_expr_nested_output(nodeId, outputId), 'nested_output'))
}

export function delayValueExpr(nodeId: number): SignalExpr {
  return SignalExpr.fromHandle(b.check(b.egress_expr_delay_value(nodeId), 'delay_value'))
}

/** Wrap an egress_param_t handle in a smoothed-param expression. */
export function paramExpr(paramHandle: unknown): SignalExpr {
  return SignalExpr.fromHandle(b.check(b.egress_expr_param(paramHandle), 'expr_param'))
}

/** Wrap an egress_param_t trigger handle in a trigger-param expression. */
export function triggerParamExpr(paramHandle: unknown): SignalExpr {
  return SignalExpr.fromHandle(b.check(b.egress_expr_trigger_param(paramHandle), 'expr_trigger_param'))
}
