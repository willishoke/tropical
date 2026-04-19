/**
 * transcendentals.test.ts — Differential accuracy tests for stdlib transcendentals.
 *
 * Flattens each stdlib program (Sin, Cos, Tanh, Exp, Log, Pow) with a literal
 * input, interprets the resulting ExprNode tree, and compares against
 * JavaScript Math.*. Pure TS, no FFI — runs without libtropical.dylib.
 *
 * These tests pin the polynomial approximations shipped in stdlib/*.json
 * to documented accuracy thresholds. Changing a coefficient without
 * updating the corresponding program should trip a threshold here.
 */

import { describe, test, expect } from 'bun:test'
import { loadStdlib } from './program'
import { flattenExpressions } from './flatten'
import { evalExpr, type InterpretEnv } from './interpret'
import type { ProgramType, ProgramInstance, Bounds } from './program_types'
import type { ExprNode } from './expr'
import type { SessionState } from './session'
import { Param, Trigger } from './runtime/param'

// ─────────────────────────────────────────────────────────────
// Setup — load stdlib once, reuse types across all evaluations
// ─────────────────────────────────────────────────────────────

const typeRegistry = new Map<string, ProgramType>()
const typeAliasRegistry = new Map<string, { base: string; bounds: Bounds }>()
loadStdlib({
  typeRegistry,
  typeAliasRegistry,
  instanceRegistry: new Map<string, ProgramInstance>(),
  paramRegistry: new Map<string, Param>(),
  triggerRegistry: new Map<string, Trigger>(),
})

/**
 * Evaluate `programName(inputs…) → outputName` at given numeric input values.
 * Uses flattenExpressions + evalExpr — no JIT, no FFI.
 *
 * graphOutputs is intentionally empty: the audio safety clamp only applies
 * to outputs routed to graphOutputs, and we want raw polynomial values
 * (especially for Exp/Log which produce out-of-[-1,1] results).
 */
function evalProgram(
  programName: string,
  inputs: Record<string, number>,
  outputName = 'out',
): number {
  const type = typeRegistry.get(programName)
  if (!type) throw new Error(`Unknown program: ${programName}`)
  const inst = type.instantiateAs('it')

  const inputExprNodes = new Map<string, ExprNode>()
  for (const [k, v] of Object.entries(inputs)) inputExprNodes.set(`it:${k}`, v)

  const fullSession = {
    typeRegistry,
    typeAliasRegistry,
    instanceRegistry: new Map([['it', inst]]),
    paramRegistry: new Map(),
    triggerRegistry: new Map(),
    bufferLength: 1,
    dac: null,
    graphOutputs: [],
    inputExprNodes,
    runtime: null as any,
    graph: null as any,
    _nameCounters: new Map<string, number>(),
  } as unknown as SessionState

  const flat = flattenExpressions(fullSession)
  const outIdx = type._def.outputNames.indexOf(outputName)
  if (outIdx < 0) throw new Error(`Unknown output '${outputName}' on ${programName}`)

  const env: InterpretEnv = {
    sampleRate: 44100,
    sampleIndex: 0,
    registers: flat.stateInit,
    inputs: [],
    params: new Map(),
  }
  return evalExpr(flat.outputExprs[outIdx], env) as number
}

/** Max absolute error of `ours(x)` vs `ref(x)` across a linear sweep. */
function sweepMaxAbsError(
  f: (x: number) => number,
  ref: (x: number) => number,
  lo: number,
  hi: number,
  n: number,
): number {
  let worst = 0
  for (let i = 0; i <= n; i++) {
    const x = lo + (i / n) * (hi - lo)
    const err = Math.abs(f(x) - ref(x))
    if (err > worst) worst = err
  }
  return worst
}

/** Max relative error over a sweep — for values with large dynamic range (exp, pow). */
function sweepMaxRelError(
  f: (x: number) => number,
  ref: (x: number) => number,
  lo: number,
  hi: number,
  n: number,
): number {
  let worst = 0
  for (let i = 0; i <= n; i++) {
    const x = lo + (i / n) * (hi - lo)
    const r = ref(x)
    const denom = Math.max(Math.abs(r), 1e-300)
    const err = Math.abs(f(x) - r) / denom
    if (err > worst) worst = err
  }
  return worst
}

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

describe('stdlib transcendentals vs Math.*', () => {
  test('Sin — 7th-order odd minimax, ≤ 5e-7 over [-4π, 4π]', () => {
    const err = sweepMaxAbsError(
      x => evalProgram('Sin', { x }),
      Math.sin,
      -4 * Math.PI, 4 * Math.PI,
      400,
    )
    expect(err).toBeLessThan(5e-7)
  })

  test('Sin — exact at 0', () => {
    expect(evalProgram('Sin', { x: 0 })).toBeCloseTo(0, 10)
  })

  test('Cos — matches Math.cos, ≤ 5e-7 over [-4π, 4π]', () => {
    const err = sweepMaxAbsError(
      x => evalProgram('Cos', { x }),
      Math.cos,
      -4 * Math.PI, 4 * Math.PI,
      400,
    )
    expect(err).toBeLessThan(5e-7)
  })

  test('Cos — exact at 0', () => {
    expect(evalProgram('Cos', { x: 0 })).toBeCloseTo(1, 6)
  })

  test('Tanh — Padé approximation, ≤ 0.03 over [-3, 3]', () => {
    // tanh(x) ≈ x * (27 + x²) / (27 + 9x²), clamped to [-3, 3].
    // This is the classic cheap audio-rate Padé approximation — max abs error ≈ 0.0235
    // near |x| ≈ 1.5. Intended as a waveshaper, not a precision tanh. If tight accuracy
    // is ever required, replace the polynomial; nothing else in stdlib depends on it.
    const err = sweepMaxAbsError(
      x => evalProgram('Tanh', { x }),
      Math.tanh,
      -3, 3,
      200,
    )
    expect(err).toBeLessThan(0.03)
  })

  test('Tanh — clamps saturate to ±1 outside [-3, 3]', () => {
    // Outside the clamp the approximation returns the ±3 endpoint value, which is
    // close to ±tanh(3) ≈ ±0.995. Differs from Math.tanh (which approaches ±1 fully).
    expect(evalProgram('Tanh', { x: 10 })).toBeGreaterThan(0.99)
    expect(evalProgram('Tanh', { x: -10 })).toBeLessThan(-0.99)
  })

  test('Exp — Cody-Waite + Horner, ≤ 5e-7 relative over [-10, 10]', () => {
    const err = sweepMaxRelError(
      x => evalProgram('Exp', { x }),
      Math.exp,
      -10, 10,
      400,
    )
    expect(err).toBeLessThan(5e-7)
  })

  test('Exp — exact at 0', () => {
    expect(evalProgram('Exp', { x: 0 })).toBeCloseTo(1, 9)
  })

  test('Log — Remez approximation, ≤ 5e-7 over [0.01, 100]', () => {
    const err = sweepMaxAbsError(
      x => evalProgram('Log', { x }),
      Math.log,
      0.01, 100,
      400,
    )
    expect(err).toBeLessThan(5e-7)
  })

  test('Log — exact at 1', () => {
    expect(evalProgram('Log', { x: 1 })).toBeCloseTo(0, 9)
  })

  test('Log — safe sentinel at x ≤ 0', () => {
    // Log clamps non-positive inputs to 1e-45 before the polynomial —
    // returns a large negative number, not NaN / -Inf.
    expect(Number.isFinite(evalProgram('Log', { x: 0 }))).toBe(true)
    expect(Number.isFinite(evalProgram('Log', { x: -1 }))).toBe(true)
    expect(evalProgram('Log', { x: 0 })).toBeLessThan(-100)
  })

  test('Pow — exp(y · log(x)), ≤ 1e-5 relative for x∈[0.5,5], y∈[-2,2]', () => {
    // Pow composes Log then Exp; error is roughly the sum of their errors
    // plus amplification from the multiply. Looser threshold than either alone.
    let worst = 0
    for (let xi = 0; xi <= 20; xi++) {
      const x = 0.5 + (xi / 20) * 4.5
      for (let yi = 0; yi <= 20; yi++) {
        const y = -2 + (yi / 20) * 4
        const ours = evalProgram('Pow', { x, y })
        const ref = Math.pow(x, y)
        const err = Math.abs(ours - ref) / Math.max(Math.abs(ref), 1e-300)
        if (err > worst) worst = err
      }
    }
    expect(worst).toBeLessThan(1e-5)
  })

  test('Pow — x^0 = 1', () => {
    expect(evalProgram('Pow', { x: 2.5, y: 0 })).toBeCloseTo(1, 6)
  })

  test('Pow — x^1 ≈ x', () => {
    expect(evalProgram('Pow', { x: 3.7, y: 1 })).toBeCloseTo(3.7, 4)
  })
})
