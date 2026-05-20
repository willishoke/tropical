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
import { interpretSession } from './interpret_resolved'
import { wireKey, portRef, instanceName, portName } from './ir/branded_names.js'

const wk = (i: string, p: string) => wireKey(portRef(instanceName(i), portName(p)))

// One shared session for the whole suite — cuts ~400× repeated stdlib
// loads down to one. evalProgram resets the per-call state (instance
// + wiring + graph outputs); the type registry survives.
const sharedSession = makeSession(1)
loadStdlib(sharedSession)

/**
 * Evaluate `programName(inputs…) → outputName` at given numeric input values.
 *
 * Routes through `interpretSession`, which mixes audio outputs into a single
 * scalar with a /20 gain compensation — so we wire the program's target
 * output as the sole `dac.out` and undo the /20 scale post-hoc.
 */
function evalProgram(
  programName: string,
  inputs: Record<string, number>,
  outputName = 'out',
): number {
  // Clear per-call state. typeRegistry / resolvedRegistry / etc. survive.
  sharedSession.instanceRegistry.clear()
  sharedSession.inputExprNodes.clear()
  sharedSession.graphOutputs.length = 0

  const { type } = resolveProgramType(sharedSession, programName, undefined, undefined)
  const inst = instantiate(type, 'it', { baseTypeName: programName })
  sharedSession.instanceRegistry.set('it', inst)
  for (const [k, v] of Object.entries(inputs)) sharedSession.inputExprNodes.set(wk(`it`, k), v)
  sharedSession.graphOutputs.push({ instance: 'it', output: outputName })
  const buf = interpretSession(sharedSession, 1)
  return buf[0] * 20.0   // undo interpretSession's /20 audio mix scaling
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
