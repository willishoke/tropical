/**
 * transcendentals.test.ts — Differential accuracy tests for stdlib transcendentals.
 *
 * Routes each stdlib program (Sin, Cos, Tanh, Exp, Log, Pow) through the
 * resolved-IR pipeline + the pure-TS interpreter, comparing against
 * JavaScript Math.*. Loads libtropical.dylib via makeSession (used for
 * its session shell, not its JIT — we never call process()).
 *
 * These tests pin the polynomial approximations shipped in stdlib/*.trop
 * to documented accuracy thresholds. Changing a coefficient without
 * updating the corresponding program should trip a threshold here.
 */

import { describe, test, expect } from 'bun:test'
import { makeSession, resolveProgramType, instantiate } from './session'
import { loadStdlib } from './program'
import { renderFramesJit } from './test_utils/audio'
import type { ExprNode } from './expr'
import { wireKey, portRef, instanceName, portName } from './ir/branded_names.js'

const wk = (i: string, p: string) => wireKey(portRef(instanceName(i), portName(p)))

// A fresh session per evaluation. Now that accuracy sweeps render as a
// single JIT kernel each (`renderSweep`), the whole suite is only a
// dozen-odd evaluations, so re-loading stdlib per call is cheap — and a
// fresh session keeps the JIT slot-model state clean between renders (a
// shared session accumulates slot allocations across applyFlatPlan calls,
// which the old interpreter sidestepped by not using the slot model).
function freshSession() {
  const s = makeSession(512)
  loadStdlib(s)
  return s
}

/**
 * Evaluate `programName(inputs…) → outputName` at given numeric input values.
 *
 * Routes through `renderFramesJit` (the JIT), which mixes audio outputs into
 * a single scalar with a /20 gain compensation — so we wire the program's
 * target output as the sole `dac.out` and undo the /20 scale post-hoc.
 */
function evalProgram(
  programName: string,
  inputs: Record<string, number>,
  outputName = 'out',
): number {
  const session = freshSession()
  const { type } = resolveProgramType(session, programName, undefined, undefined)
  const inst = instantiate(type, 'it', { baseTypeName: programName })
  session.instanceRegistry.set('it', inst)
  for (const [k, v] of Object.entries(inputs)) session.inputExprNodes.set(wk(`it`, k), v)
  session.graphOutputs.push({ instance: 'it', output: outputName })
  const buf = renderFramesJit(session, 1)
  return buf[0] * 20.0   // undo the /20 audio mix scaling
}

/** Wire expression for the i-th sweep point as a function of sampleIndex:
 *  x(i) = lo + i * step, step = (hi - lo) / n, i = sampleIndex = 0..n. A
 *  precomputed float step keeps the kernel multiply in floating point (no
 *  integer-division ambiguity). The matching JS value is `sweepX`. */
function rampExpr(lo: number, hi: number, n: number): ExprNode {
  return { op: 'add', args: [lo, { op: 'mul', args: [{ op: 'sampleIndex' }, (hi - lo) / n] }] }
}
function sweepX(lo: number, hi: number, n: number, i: number): number {
  return lo + i * ((hi - lo) / n)
}

/** Render a program over a `sampleIndex`-parameterized sweep in ONE JIT
 *  kernel: wire each input to a `sampleIndex`-derived expression and render
 *  `nSamples` samples. Returns the output at each sample (the /20 mix gain
 *  undone). One compile per sweep instead of one per point — the per-point
 *  path recompiled the kernel every call. */
function renderSweep(
  programName: string,
  inputExprs: Record<string, ExprNode>,
  nSamples: number,
  outputName = 'out',
): Float64Array {
  const session = freshSession()
  const { type } = resolveProgramType(session, programName, undefined, undefined)
  const inst = instantiate(type, 'it', { baseTypeName: programName })
  session.instanceRegistry.set('it', inst)
  for (const [k, e] of Object.entries(inputExprs)) session.inputExprNodes.set(wk('it', k), e)
  session.graphOutputs.push({ instance: 'it', output: outputName })

  const buf = renderFramesJit(session, nSamples)
  const out = new Float64Array(nSamples)
  for (let i = 0; i < nSamples; i++) out[i] = buf[i] * 20.0   // undo the /20 mix scale
  return out
}

/** Max absolute error of precomputed sweep outputs `ours[i]` vs `ref(x_i)`. */
function sweepMaxAbsError(
  ours: Float64Array,
  ref: (x: number) => number,
  lo: number,
  hi: number,
  n: number,
): number {
  let worst = 0
  for (let i = 0; i <= n; i++) {
    const err = Math.abs(ours[i] - ref(sweepX(lo, hi, n, i)))
    if (err > worst) worst = err
  }
  return worst
}

/** Max relative error over a sweep — for large-dynamic-range fns (exp, pow). */
function sweepMaxRelError(
  ours: Float64Array,
  ref: (x: number) => number,
  lo: number,
  hi: number,
  n: number,
): number {
  let worst = 0
  for (let i = 0; i <= n; i++) {
    const r = ref(sweepX(lo, hi, n, i))
    const err = Math.abs(ours[i] - r) / Math.max(Math.abs(r), 1e-300)
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
      renderSweep('Sin', { x: rampExpr(-4 * Math.PI, 4 * Math.PI, 400) }, 401),
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
      renderSweep('Cos', { x: rampExpr(-4 * Math.PI, 4 * Math.PI, 400) }, 401),
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
      renderSweep('Tanh', { x: rampExpr(-3, 3, 200) }, 201),
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
      renderSweep('Exp', { x: rampExpr(-10, 10, 400) }, 401),
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
      renderSweep('Log', { x: rampExpr(0.01, 100, 400) }, 401),
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
    // 21×21 grid rendered in ONE kernel: sample s → (xi = s ÷ 21, yi = s mod 21).
    const N = 21
    const xStep = 4.5 / (N - 1), yStep = 4 / (N - 1)
    const ours = renderSweep('Pow', {
      x: { op: 'add', args: [0.5, { op: 'mul', args: [{ op: 'floorDiv', args: [{ op: 'sampleIndex' }, N] }, xStep] }] },
      y: { op: 'add', args: [-2,  { op: 'mul', args: [{ op: 'mod',      args: [{ op: 'sampleIndex' }, N] }, yStep] }] },
    }, N * N)
    let worst = 0
    for (let xi = 0; xi < N; xi++) {
      const x = 0.5 + xi * xStep
      for (let yi = 0; yi < N; yi++) {
        const y = -2 + yi * yStep
        const ref = Math.pow(x, y)
        const err = Math.abs(ours[xi * N + yi] - ref) / Math.max(Math.abs(ref), 1e-300)
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
