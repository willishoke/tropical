/**
 * interpret.ts — ExprNode tree interpreter for differential testing.
 *
 * Recursively evaluates post-flatten, post-lower ExprNode trees using
 * JavaScript Math.* for transcendentals. Serves as a reference oracle
 * against the LLVM JIT path.
 *
 * The interpreter handles the ~40 ops that survive array lowering and
 * combinator expansion. It does NOT handle higher-level ops like
 * generate, fold, let, zeros, reshape — those are expanded by
 * lowerArrayOps before the interpreter sees them.
 *
 * No FFI, no C++ dependency. Pure TS, fully deterministic.
 */

import type { ExprNode } from './expr.js'
import type { FlatExpressions } from './flatten.js'

// ─────────────────────────────────────────────────────────────
// Environment
// ─────────────────────────────────────────────────────────────

export interface InterpretEnv {
  sampleRate: number
  sampleIndex: number
  registers: (number | boolean | number[])[]
  inputs: number[]
  params: Map<number, number>
}

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────

/** Coerce a value to a JS number (booleans → 0/1). */
function toNum(v: number | boolean): number {
  return typeof v === 'boolean' ? (v ? 1 : 0) : v
}

/** Coerce a value to an integer for bitwise ops. */
function toInt(v: number | boolean): number {
  return Math.trunc(toNum(v))
}

/** Coerce a value to boolean (0 → false, nonzero → true). */
function toBool(v: number | boolean): boolean {
  return typeof v === 'boolean' ? v : v !== 0
}

type Value = number | boolean | number[]

/** Apply a scalar binary op elementwise, broadcasting scalar args. */
function binOp(a: Value, b: Value, fn: (x: number, y: number) => number | boolean): Value {
  const aArr = Array.isArray(a)
  const bArr = Array.isArray(b)
  if (!aArr && !bArr) return fn(toNum(a as number | boolean), toNum(b as number | boolean))
  if (aArr && bArr) {
    const aa = a as number[], bb = b as number[]
    const len = Math.max(aa.length, bb.length)
    const out = new Array<number>(len)
    for (let i = 0; i < len; i++) out[i] = toNum(fn(aa[i % aa.length], bb[i % bb.length]))
    return out
  }
  if (aArr) {
    const aa = a as number[], bv = toNum(b as number | boolean)
    return aa.map(x => toNum(fn(x, bv)))
  }
  const bb = b as number[], av = toNum(a as number | boolean)
  return bb.map(y => toNum(fn(av, y)))
}

/** Apply a scalar unary op elementwise. */
function unOp(a: Value, fn: (x: number) => number | boolean): Value {
  if (Array.isArray(a)) return (a as number[]).map(x => toNum(fn(x)))
  return fn(toNum(a as number | boolean))
}

// ─────────────────────────────────────────────────────────────
// Core evaluator
// ─────────────────────────────────────────────────────────────

export function evalExpr(node: ExprNode, env: InterpretEnv): Value {
  if (typeof node === 'number') return node
  if (typeof node === 'boolean') return node
  if (Array.isArray(node)) return (node as ExprNode[]).map(n => toNum(evalExpr(n, env) as number | boolean))

  const obj = node as Record<string, unknown>
  const op = obj.op as string
  const args = obj.args as ExprNode[] | undefined

  switch (op) {
    // ── Terminals ──────────────────────────────────────────
    case 'input':
      return env.inputs[(obj.id ?? obj.slot) as number] ?? 0
    case 'reg':
      return env.registers[(obj.id ?? obj.slot) as number] ?? 0
    case 'sample_rate':
      return env.sampleRate
    case 'sample_index':
      return env.sampleIndex
    case 'smoothed_param':
    case 'trigger_param': {
      const handle = (obj._handle ?? obj.ptr) as number
      return env.params.get(handle) ?? 0
    }

    // ── Binary arithmetic ─────────────────────────────────
    case 'add':       return binOp(evalExpr(args![0], env), evalExpr(args![1], env), (a, b) => a + b)
    case 'sub':       return binOp(evalExpr(args![0], env), evalExpr(args![1], env), (a, b) => a - b)
    case 'mul':       return binOp(evalExpr(args![0], env), evalExpr(args![1], env), (a, b) => a * b)
    case 'div':       return binOp(evalExpr(args![0], env), evalExpr(args![1], env), (a, b) => b !== 0 ? a / b : 0)
    case 'mod':       return binOp(evalExpr(args![0], env), evalExpr(args![1], env), (a, b) => b !== 0 ? a % b : 0)
    case 'pow':       return binOp(evalExpr(args![0], env), evalExpr(args![1], env), (a, b) => Math.pow(a, b))
    case 'ldexp':     return binOp(evalExpr(args![0], env), evalExpr(args![1], env), (a, b) => a * Math.pow(2, Math.trunc(b)))
    case 'floor_div':
    case 'floorDiv':  return binOp(evalExpr(args![0], env), evalExpr(args![1], env), (a, b) => b !== 0 ? Math.floor(a / b) : 0)

    // ── Binary comparison ─────────────────────────────────
    case 'lt':  return binOp(evalExpr(args![0], env), evalExpr(args![1], env), (a, b) => a < b)
    case 'lte': return binOp(evalExpr(args![0], env), evalExpr(args![1], env), (a, b) => a <= b)
    case 'gt':  return binOp(evalExpr(args![0], env), evalExpr(args![1], env), (a, b) => a > b)
    case 'gte': return binOp(evalExpr(args![0], env), evalExpr(args![1], env), (a, b) => a >= b)
    case 'eq':  return binOp(evalExpr(args![0], env), evalExpr(args![1], env), (a, b) => a === b)
    case 'neq': return binOp(evalExpr(args![0], env), evalExpr(args![1], env), (a, b) => a !== b)

    // ── Binary bitwise ────────────────────────────────────
    case 'bit_and':
    case 'bitAnd':  return binOp(evalExpr(args![0], env), evalExpr(args![1], env), (a, b) => toInt(a) & toInt(b))
    case 'bit_or':
    case 'bitOr':   return binOp(evalExpr(args![0], env), evalExpr(args![1], env), (a, b) => toInt(a) | toInt(b))
    case 'bit_xor':
    case 'bitXor':  return binOp(evalExpr(args![0], env), evalExpr(args![1], env), (a, b) => toInt(a) ^ toInt(b))
    case 'lshift':  return binOp(evalExpr(args![0], env), evalExpr(args![1], env), (a, b) => toInt(a) << toInt(b))
    case 'rshift':  return binOp(evalExpr(args![0], env), evalExpr(args![1], env), (a, b) => toInt(a) >> toInt(b))

    // ── Binary logical ────────────────────────────────────
    case 'and': return binOp(evalExpr(args![0], env), evalExpr(args![1], env), (a, b) => toBool(a) && toBool(b))
    case 'or':  return binOp(evalExpr(args![0], env), evalExpr(args![1], env), (a, b) => toBool(a) || toBool(b))

    // ── Unary math ────────────────────────────────────────
    case 'neg':   return unOp(evalExpr(args![0], env), x => -x)
    case 'abs':   return unOp(evalExpr(args![0], env), x => Math.abs(x))
    case 'sin':   return unOp(evalExpr(args![0], env), x => Math.sin(x))
    case 'cos':   return unOp(evalExpr(args![0], env), x => Math.cos(x))
    case 'tanh':  return unOp(evalExpr(args![0], env), x => Math.tanh(x))
    case 'log':   return unOp(evalExpr(args![0], env), x => Math.log(x))
    case 'exp':   return unOp(evalExpr(args![0], env), x => Math.exp(x))
    case 'sqrt':  return unOp(evalExpr(args![0], env), x => Math.sqrt(x))
    case 'floor': return unOp(evalExpr(args![0], env), x => Math.floor(x))
    case 'ceil':  return unOp(evalExpr(args![0], env), x => Math.ceil(x))
    case 'round': return unOp(evalExpr(args![0], env), x => Math.round(x))
    case 'float_exponent': return unOp(evalExpr(args![0], env), x => {
      if (x === 0 || !isFinite(x)) return 0
      return Math.floor(Math.log2(Math.abs(x)))
    })

    // ── Unary logical/bitwise ─────────────────────────────
    case 'not':     return unOp(evalExpr(args![0], env), x => !toBool(x))
    case 'bit_not': return unOp(evalExpr(args![0], env), x => ~toInt(x))

    // ── Ternary ───────────────────────────────────────────
    case 'select': {
      const cond = evalExpr(args![0], env)
      const then_ = evalExpr(args![1], env)
      const else_ = evalExpr(args![2], env)
      if (Array.isArray(cond)) {
        const c = cond as number[], t = then_ as number[] | number, e = else_ as number[] | number
        return c.map((cv, i) => toBool(cv)
          ? toNum(Array.isArray(t) ? t[i] : t as number | boolean)
          : toNum(Array.isArray(e) ? e[i] : e as number | boolean))
      }
      return toBool(cond as number | boolean) ? then_ : else_
    }
    case 'clamp': {
      const v = evalExpr(args![0], env)
      const lo = evalExpr(args![1], env)
      const hi = evalExpr(args![2], env)
      if (Array.isArray(v)) {
        return (v as number[]).map((x, i) => {
          const l = toNum(Array.isArray(lo) ? lo[i] : lo as number | boolean)
          const h = toNum(Array.isArray(hi) ? hi[i] : hi as number | boolean)
          return Math.min(Math.max(x, l), h)
        })
      }
      return Math.min(Math.max(toNum(v as number | boolean), toNum(lo as number | boolean)), toNum(hi as number | boolean))
    }

    // ── Array ops ─────────────────────────────────────────
    case 'array': {
      const items = obj.items as ExprNode[]
      return items.map(item => toNum(evalExpr(item, env) as number | boolean))
    }
    case 'index': {
      const arr = evalExpr(args![0], env)
      const idx = toInt(evalExpr(args![1], env) as number | boolean)
      if (Array.isArray(arr)) return (arr as number[])[idx] ?? 0
      return arr // scalar indexing returns the scalar
    }
    case 'array_set': {
      const arr = evalExpr(args![0], env)
      const idx = toInt(evalExpr(args![1], env) as number | boolean)
      const val = toNum(evalExpr(args![2], env) as number | boolean)
      if (Array.isArray(arr)) {
        const copy = [...arr as number[]]
        if (idx >= 0 && idx < copy.length) copy[idx] = val
        return copy
      }
      return val // scalar case
    }
    case 'broadcast_to': {
      const src = evalExpr(args![0], env)
      const shape = obj.shape as number[]
      const len = shape.reduce((a, b) => a * b, 1)
      if (Array.isArray(src)) {
        const s = src as number[]
        const out = new Array<number>(len)
        for (let i = 0; i < len; i++) out[i] = s[i % s.length]
        return out
      }
      return new Array<number>(len).fill(toNum(src as number | boolean))
    }
    case 'matrix': {
      const rows = obj.rows as ExprNode[][]
      const flat: number[] = []
      for (const row of rows) {
        for (const cell of row) flat.push(toNum(evalExpr(cell, env) as number | boolean))
      }
      return flat
    }

    default:
      throw new Error(`interpret: unsupported op '${op}'`)
  }
}

// ─────────────────────────────────────────────────────────────
// Sample runner
// ─────────────────────────────────────────────────────────────

/**
 * Interpret a flattened program for nSamples, returning the audio output buffer.
 *
 * Mirrors FlatRuntime execution order:
 * 1. Evaluate output expressions → accumulate sum
 * 2. Evaluate register update expressions → store in new values
 * 3. Write back all registers atomically (sample-delay semantics)
 * 4. Write sum(selected outputs) / 20.0 to output buffer
 * 5. Advance sampleIndex
 */
export function interpretSamples(
  flat: FlatExpressions,
  nSamples: number,
  params?: Map<number, number>,
): Float64Array {
  const output = new Float64Array(nSamples)
  const registers: (number | boolean | number[])[] = flat.stateInit.map(
    v => Array.isArray(v) ? [...v] : v,
  )

  const env: InterpretEnv = {
    sampleRate: flat.sampleRate,
    sampleIndex: 0,
    registers,
    inputs: [],
    params: params ?? new Map(),
  }

  for (let s = 0; s < nSamples; s++) {
    env.sampleIndex = s

    // Evaluate all output expressions
    const outputValues: Value[] = flat.outputExprs.map(expr => evalExpr(expr, env))

    // Evaluate all register update expressions
    const newRegisters: (number | boolean | number[])[] = flat.registerExprs.map(
      expr => evalExpr(expr, env) as number | boolean | number[],
    )

    // Atomic writeback (sample-delay semantics)
    for (let i = 0; i < newRegisters.length; i++) {
      const v = newRegisters[i]
      env.registers[i] = Array.isArray(v) ? [...v] : v
    }

    // Mix selected outputs and scale
    let mixed = 0
    for (const idx of flat.outputIndices) {
      const v = outputValues[idx]
      mixed += toNum(Array.isArray(v) ? v[0] : v as number | boolean)
    }
    output[s] = mixed / 20.0
  }

  return output
}
