/**
 * render.test.ts — End-to-end integration tests using the buffer backend.
 *
 * These tests exercise the full TS compiler pipeline:
 *   makeSession + loadStdlib
 *   → loadJSON (tropical_program_1 schema)
 *   → applyFlatPlan (flatten + emit + JIT compile)
 *   → renderFrames (synchronous audio rendering, no audio device)
 *   → peak / rms / dominantFrequency (signal-level assertions)
 *
 * Requires build/libtropical.dylib — run `make build` first.
 * Like apply_plan.test.ts, run with: bun test compiler/render.test.ts
 */

import { describe, test, expect, afterEach } from 'bun:test'
import { statSync, existsSync, unlinkSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { makeSession, loadJSON } from './session'
import { loadStdlib as loadBuiltins, type ProgramJSON } from './program'
import { applySessionWiring } from './apply_plan'
import { flattenExpressions } from './flatten'
import { interpretSamples } from './interpret'
import {
  renderFrames,
  peak,
  rms,
  dominantFrequency,
  writeWav,
} from './test_utils/audio'

// ─── helpers ──────────────────────────────────────────────────────────────────

/** Build and compile a single-VCO session outputting the named waveform. */
function vcoSession(freq: number, output: 'saw' | 'sin' | 'tri' | 'sqr', bufferLength = 256) {
  const session = makeSession(bufferLength)
  loadBuiltins(session.typeRegistry)
  loadJSON({
    schema: 'tropical_program_1',
    name: 'test',
    instances: { osc: { program: 'VCO', inputs: { freq } } },
    audio_outputs: [{ instance: 'osc', output }],
  } as ProgramJSON, session)
  return session
}

// ─── tests ────────────────────────────────────────────────────────────────────

describe('renderFrames / buffer backend', () => {
  test('sawtooth peak and RMS are in expected range', () => {
    // VCO sawtooth maps phase [0,1) → [-1,1).  The JIT kernel divides all
    // outputs by 20.0 (OrcJitEngine.cpp), so the output buffer is in [-0.05, 0.05].
    // After ~40 full cycles (440 Hz × 4096/44100 s ≈ 40.8) peak should be near 1/20.
    const session = vcoSession(440, 'saw')
    const samples = renderFrames(session.runtime, 16)  // 16 × 256 = 4096 samples

    // Expect peak ≈ 1.0/20 = 0.05 (PolyBLEP smoothing may trim it slightly)
    expect(peak(samples)).toBeGreaterThan(0.03)
    expect(peak(samples)).toBeLessThan(0.07)
    // Theoretical RMS of sawtooth ≈ 0.577/20 ≈ 0.029
    expect(rms(samples)).toBeGreaterThan(0.015)

    session.graph.dispose()
  })

  test('sine dominant frequency matches configured VCO frequency', () => {
    // Use the sin output — no harmonics, so the peak FFT bin maps cleanly to
    // the fundamental.  174 × 256 = 44544 samples ≈ 1 second at 44100 Hz,
    // giving sub-Hz resolution after zero-pad to 65536.
    const session = vcoSession(440, 'sin')
    const samples = renderFrames(session.runtime, 174)  // ~1 s

    const freq = dominantFrequency(samples, 44100)
    expect(Math.abs(freq - 440)).toBeLessThan(15)  // ±15 Hz tolerance

    session.graph.dispose()
  })

  test('hot-swap updates frequency while preserving phase state', () => {
    // Start at 220 Hz, advance state, hot-swap to 440 Hz, then verify the
    // output frequency changed.  Phase register persistence means no reset to 0.
    const session = makeSession(256)
    loadBuiltins(session.typeRegistry)
    loadJSON({
      schema: 'tropical_program_1',
      name: 'test',
      instances: { osc: { program: 'VCO', inputs: { freq: 220 } } },
      audio_outputs: [{ instance: 'osc', output: 'sin' }],
    } as ProgramJSON, session)

    // Advance phase state (8 buffer frames worth of 220 Hz oscillation)
    renderFrames(session.runtime, 8)

    // Hot-swap to 440 Hz — applySessionWiring recompiles and atomically swaps
    session.inputExprNodes.set('osc:freq', 440)
    applySessionWiring(session)

    // Render ~1 second and verify dominant frequency is now 440, not 220
    const samples = renderFrames(session.runtime, 174)
    const freq = dominantFrequency(samples, 44100)
    expect(Math.abs(freq - 440)).toBeLessThan(15)
    expect(Math.abs(freq - 220)).toBeGreaterThan(15)

    session.graph.dispose()
  })

  test('WAV file is written with correct byte size', async () => {
    // 16 × 256 = 4096 samples → file = 46-byte header + 4096 × 4 bytes = 16430 bytes
    const session = vcoSession(440, 'saw')
    const samples = renderFrames(session.runtime, 16)

    const path = join(tmpdir(), 'tropical_render_test.wav')
    await writeWav(path, samples, 44100)

    expect(existsSync(path)).toBe(true)
    const expectedBytes = 46 + samples.length * 4  // header + float32 data
    expect(statSync(path).size).toBe(expectedBytes)

    unlinkSync(path)
    session.graph.dispose()
  })

  test('sample count equals nCalls * bufferLength regardless of buffer size', () => {
    // Two sessions with different buffer sizes but the same program and initial
    // state should produce identical output — the JIT kernel is per-sample and
    // cannot be vectorized across samples due to state register updates.
    const prog: ProgramJSON = {
      schema: 'tropical_program_1',
      name: 'test',
      instances: { osc: { program: 'VCO', inputs: { freq: 440 } } },
      audio_outputs: [{ instance: 'osc', output: 'sin' }],
    }

    const s32 = makeSession(32)
    loadBuiltins(s32.typeRegistry)
    loadJSON(prog, s32)
    const a = renderFrames(s32.runtime, 16)  // 16 × 32 = 512

    const s512 = makeSession(512)
    loadBuiltins(s512.typeRegistry)
    loadJSON(prog, s512)
    const b = renderFrames(s512.runtime, 1)  // 1 × 512 = 512

    expect(a.length).toBe(512)
    expect(b.length).toBe(512)
    for (let i = 0; i < 512; i++) {
      expect(a[i]).toBeCloseTo(b[i], 10)
    }

    s32.graph.dispose()
    s512.graph.dispose()
  })
})

// ─── differential tests: interpreter vs JIT ──────────────────────────────────

/** Max absolute difference between two buffers. */
function maxDiff(a: Float64Array, b: Float64Array): number {
  let d = 0
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    d = Math.max(d, Math.abs(a[i] - b[i]))
  }
  return d
}

describe('interpreter vs JIT differential', () => {
  test('VCO sawtooth matches within epsilon', () => {
    const session = vcoSession(440, 'saw')
    const flat = flattenExpressions(session)
    const nSamples = 16 * 256  // 4096
    const interp = interpretSamples(flat, nSamples)
    const jit = renderFrames(session.runtime, 16)

    expect(interp.length).toBe(nSamples)
    expect(jit.length).toBe(nSamples)
    expect(maxDiff(interp, jit)).toBeLessThan(1e-6)

    session.graph.dispose()
  })

  test('VCO sine matches within epsilon', () => {
    // Wider tolerance: JIT uses 7th-order minimax sin approximation that
    // diverges from Math.sin by up to ~0.004 in the output range.
    const session = vcoSession(440, 'sin')
    const flat = flattenExpressions(session)
    const nSamples = 16 * 256
    const interp = interpretSamples(flat, nSamples)
    const jit = renderFrames(session.runtime, 16)

    expect(maxDiff(interp, jit)).toBeLessThan(0.005)

    session.graph.dispose()
  })

  test('VCO triangle matches within epsilon', () => {
    const session = vcoSession(440, 'tri')
    const flat = flattenExpressions(session)
    const nSamples = 16 * 256
    const interp = interpretSamples(flat, nSamples)
    const jit = renderFrames(session.runtime, 16)

    expect(maxDiff(interp, jit)).toBeLessThan(1e-6)

    session.graph.dispose()
  })

  test('VCO square matches within epsilon', () => {
    const session = vcoSession(440, 'sqr')
    const flat = flattenExpressions(session)
    const nSamples = 16 * 256
    const interp = interpretSamples(flat, nSamples)
    const jit = renderFrames(session.runtime, 16)

    expect(maxDiff(interp, jit)).toBeLessThan(1e-6)

    session.graph.dispose()
  })

  test('VCO+VCA chain matches within epsilon', () => {
    const session = makeSession(256)
    loadBuiltins(session.typeRegistry)
    loadJSON({
      schema: 'tropical_program_1',
      name: 'test',
      instances: {
        osc: { program: 'VCO', inputs: { freq: 440 } },
        amp: { program: 'VCA', inputs: {
          audio: { op: 'ref', instance: 'osc', output: 'saw' },
          cv: 1.0,
        }},
      },
      audio_outputs: [{ instance: 'amp', output: 'out' }],
    } as ProgramJSON, session)

    const flat = flattenExpressions(session)
    const nSamples = 16 * 256
    const interp = interpretSamples(flat, nSamples)
    const jit = renderFrames(session.runtime, 16)

    expect(maxDiff(interp, jit)).toBeLessThan(1e-6)

    session.graph.dispose()
  })

  test('Clock module matches within epsilon', () => {
    const session = makeSession(256)
    loadBuiltins(session.typeRegistry)
    loadJSON({
      schema: 'tropical_program_1',
      name: 'test',
      instances: {
        clk: { program: 'Clock', inputs: { freq: 1.0, ratios_in: [1.0] } },
      },
      audio_outputs: [{ instance: 'clk', output: 'output' }],
    } as ProgramJSON, session)

    const flat = flattenExpressions(session)
    const nSamples = 16 * 256
    const interp = interpretSamples(flat, nSamples)
    const jit = renderFrames(session.runtime, 16)

    expect(maxDiff(interp, jit)).toBeLessThan(1e-6)

    session.graph.dispose()
  })
})
