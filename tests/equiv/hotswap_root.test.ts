/**
 * hotswap_root.test.ts — hot-swap state transfer survives a topology
 * edit on the root-program (default) lowering.
 *
 * This is the end-to-end gate for Option A's naming transparency. The
 * whole point of `ROOT_INSTANCE_PATH` keeping children/registers at
 * BARE names is that an instance's state slot keeps the SAME name when
 * the session topology changes — so the engine's load-time
 * state-transfer-by-name (`FlatRuntime::load_plan`) carries it across a
 * recompile without a click. If naming transparency regressed (e.g. the
 * root prefixed `osc1`'s phase register), adding a second instance would
 * rename `osc1`'s register, the transfer would miss, and `osc1`'s phase
 * would reset to 0 — a discontinuity this test would catch.
 *
 * `loadPlan` does NOT fade (fade is a separate explicit call), so the
 * post-swap output is directly comparable to a continuously-running
 * reference.
 *
 * Requires libtropical.dylib (`make build`).
 */

import { describe, test, expect } from 'bun:test'
import { makeSession, resolveProgramType, instantiate } from '../../compiler/session.js'
import { loadStdlib } from '../../compiler/program.js'
import { applyFlatPlan } from '../../compiler/apply_plan.js'
import { wireKey, portRef, instanceName, portName } from '../../compiler/ir/branded_names.js'

const wk = (i: string, p: string) => wireKey(portRef(instanceName(i), portName(p)))
const BUFLEN = 256

function addSinOsc(session: ReturnType<typeof makeSession>, name: string, freq: number) {
  const { type } = resolveProgramType(session, 'SinOsc', undefined, undefined)
  session.instanceRegistry.set(name, instantiate(type, name, { baseTypeName: 'SinOsc' }))
  session.inputExprNodes.set(wk(name, 'freq'), freq)
}

/** Run `nBuffers` and return the last buffer. */
function runN(session: ReturnType<typeof makeSession>, nBuffers: number): Float64Array {
  let last = new Float64Array(BUFLEN)
  for (let i = 0; i < nBuffers; i++) {
    session.runtime.process()
    last = Float64Array.from(session.runtime.outputBuffer.subarray(0, BUFLEN))
  }
  return last
}

function maxAbsDiff(a: Float64Array, b: Float64Array): number {
  let m = 0
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]))
  return m
}

const peak = (a: Float64Array): number => a.reduce((m, v) => Math.max(m, Math.abs(v)), 0)

describe('hot-swap state transfer on the root path', () => {
  // `osc1`'s phase is a Phasor register (accumulator), so its waveform
  // depends ENTIRELY on transferred state — there is no sampleIndex
  // fallback. If the transfer works, the post-swap buffer equals the
  // buffer a never-swapped reference produces at the same sample count.
  const M = 6

  test('oscillator phase survives ADDING an instance', () => {
    // Reference: osc1 alone, run M buffers straight through.
    const ref = makeSession(BUFLEN)
    loadStdlib(ref)
    addSinOsc(ref, 'osc1', 220)
    ref.graphOutputs.push({ instance: 'osc1', output: 'sine' })
    applyFlatPlan(ref, ref.runtime)
    const refBuf = runN(ref, M)

    // Test: osc1 alone for M-1 buffers, then ADD osc2 (topology edit) and
    // recompile — a genuine hot-swap — then run the M-th buffer.
    const t = makeSession(BUFLEN)
    loadStdlib(t)
    addSinOsc(t, 'osc1', 220)
    t.graphOutputs.push({ instance: 'osc1', output: 'sine' })
    applyFlatPlan(t, t.runtime)
    runN(t, M - 1)

    addSinOsc(t, 'osc2', 330)          // topology change → register set grows
    applyFlatPlan(t, t.runtime)        // recompile + load = hot-swap by name
    const testBuf = runN(t, 1)

    // dac is wired to osc1 only, so output is osc1's sine. It must
    // continue exactly — phase carried across the swap.
    expect(peak(refBuf)).toBeGreaterThan(0.01) // not a vacuous (silent) pass
    expect(maxAbsDiff(testBuf, refBuf)).toBeLessThan(1e-9)
  })

  test('survivor oscillator phase survives REMOVING an instance', () => {
    // Reference: osc1 alone (the survivor), M buffers straight.
    const ref = makeSession(BUFLEN)
    loadStdlib(ref)
    addSinOsc(ref, 'osc1', 220)
    ref.graphOutputs.push({ instance: 'osc1', output: 'sine' })
    applyFlatPlan(ref, ref.runtime)
    const refBuf = runN(ref, M)

    // Test: osc1 + osc2 for M-1 buffers, then REMOVE osc2 and recompile.
    const t = makeSession(BUFLEN)
    loadStdlib(t)
    addSinOsc(t, 'osc1', 220)
    addSinOsc(t, 'osc2', 330)
    t.graphOutputs.push({ instance: 'osc1', output: 'sine' })
    applyFlatPlan(t, t.runtime)
    runN(t, M - 1)

    t.instanceRegistry.delete('osc2')
    t.inputExprNodes.delete(wk('osc2', 'freq'))
    applyFlatPlan(t, t.runtime)        // hot-swap; osc1 phase transfers by name
    const testBuf = runN(t, 1)

    expect(peak(refBuf)).toBeGreaterThan(0.01) // not a vacuous (silent) pass
    expect(maxAbsDiff(testBuf, refBuf)).toBeLessThan(1e-9)
  })
})
