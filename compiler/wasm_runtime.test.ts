/**
 * wasm_runtime.test.ts — exercise WasmRuntime (the worklet-side class) in
 * isolation. No AudioWorklet needed; we drive `process()` directly across
 * many 128-sample blocks and check that sustained audio is produced.
 *
 * This is the test the browser bug will show up in first, since the
 * runtime (state init, fade envelope, mono output fill) is the only layer
 * between a known-good WASM module and the Web Audio output channel.
 */

import { describe, test, expect } from 'bun:test'
import type { FlatPlan } from './flatten'
import { flattenSession } from './flatten'
import { emitWasm } from './emit_wasm'
import { WasmRuntime, type LoadedPlan } from '../web/worklet/runtime'
import { makeSession, loadJSON } from './session'
import { loadStdlib } from './program'

/** Build the pure-sine-440 plan inline rather than depending on the
 *  web/dist/patches/ artifact (which would require running `bun web/build_patches.ts`
 *  before tests). Mirrors the patch definition in web/build_patches.ts. */
function buildPureSine440Plan(): FlatPlan {
  const session = makeSession(128)
  loadStdlib(session)
  loadJSON({
    schema: 'tropical_program_2',
    name: 'pure_sine_440',
    body: { op: 'block', decls: [
      { op: 'instance_decl', name: 'osc', program: 'SinOsc', inputs: { freq: 440 } },
    ]},
    audio_outputs: [{ instance: 'osc', output: 'sine' }],
  }, session)
  return flattenSession(session)
}

async function compile(plan: FlatPlan, maxBlockSize: number): Promise<LoadedPlan> {
  const { bytes, layout, paramPtrs } = emitWasm(plan, { maxBlockSize })
  return {
    bytes,
    layout,
    paramPtrs,
    stateInit: plan.state_init,
    registerTypes: plan.register_types,
    registerNames: plan.register_names,
    arraySlotNames: plan.array_slot_names,
  }
}

describe('WasmRuntime — block-driven render', () => {
  test('pure-sine-440: sustained non-silent output across blocks', async () => {
    const plan = buildPureSine440Plan()
    const loaded = await compile(plan, 128)

    // Fake shared param buffer (unused by this plan).
    const fakeShared = new ArrayBuffer(16 * 2 * 8) // enough for 16 params
    const rt = new WasmRuntime(new Float64Array(fakeShared), 16)
    await rt.loadPlan(loaded)
    // Do NOT beginFadeIn — test steady-state output without fade logic first.

    const blockSize = 128
    const blocks = 40 // ~0.1s at 48k; enough to be well past fade
    const out = new Float32Array(blockSize * blocks)
    for (let b = 0; b < blocks; b++) {
      const block = new Float32Array(blockSize)
      const rendered = rt.process(block, blockSize)
      expect(rendered).toBe(blockSize)
      out.set(block, b * blockSize)
    }

    // After fade-in completes (2048 samples ≈ 16 blocks), output should be
    // a 440 Hz sine at peak 0.05.
    let peakAfter = 0, nzAfter = 0
    for (let i = 2048; i < out.length; i++) {
      const v = Math.abs(out[i]!)
      if (v > 1e-6) nzAfter++
      peakAfter = Math.max(peakAfter, v)
    }
    const postFadeSamples = out.length - 2048
    // eslint-disable-next-line no-console
    console.log(`  sustained: peak=${peakAfter.toFixed(4)}, nonzero=${nzAfter}/${postFadeSamples}`)
    expect(peakAfter).toBeGreaterThan(0.02)
    expect(nzAfter).toBeGreaterThan(postFadeSamples / 2)
  })

  test('loadPlan auto-fades from zero to avoid click; first block is quiet', async () => {
    const plan = buildPureSine440Plan()
    const loaded = await compile(plan, 128)

    const fakeShared = new ArrayBuffer(16 * 2 * 8)
    const rt = new WasmRuntime(new Float64Array(fakeShared), 16)
    await rt.loadPlan(loaded)

    const block = new Float32Array(128)
    rt.process(block, 128)

    let peak = 0
    for (let i = 0; i < 128; i++) peak = Math.max(peak, Math.abs(block[i]!))
    // Documented behavior: loadPlan sets fadeInRem=2048 so the first 128
    // samples are the start of a smoothstep ramp from 0.
    expect(peak).toBeLessThan(0.01)
    expect(peak).toBeGreaterThan(0) // but not bit-zero — some ramp has started
  })

  test('explicit fade-in produces smooth ramp from 0', async () => {
    const plan = buildPureSine440Plan()
    const loaded = await compile(plan, 128)

    const fakeShared = new ArrayBuffer(16 * 2 * 8)
    const rt = new WasmRuntime(new Float64Array(fakeShared), 16)
    await rt.loadPlan(loaded)
    rt.beginFadeIn()

    const blockSize = 128
    const blocks = 32
    const out = new Float32Array(blockSize * blocks)
    for (let b = 0; b < blocks; b++) {
      const block = new Float32Array(blockSize)
      rt.process(block, blockSize)
      out.set(block, b * blockSize)
    }

    // First sample must be 0 (fade at t=0).
    expect(out[0]!).toBe(0)
    // After fade completes (sample 2048), should hit the 0.05 peak.
    let peakPost = 0
    for (let i = 2048; i < out.length; i++) peakPost = Math.max(peakPost, Math.abs(out[i]!))
    expect(peakPost).toBeGreaterThan(0.02)
  })
})
