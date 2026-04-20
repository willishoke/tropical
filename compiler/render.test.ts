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

import { describe, test, expect } from 'bun:test'
import { statSync, existsSync, unlinkSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { makeSession, loadJSON } from './session'
import { loadStdlib as loadBuiltins, loadProgramAsType } from './program'
import type { ProgramNode, ProgramFile } from './program'
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

/** Minimal test oscillator — naive saw + sin, phase accumulator. */
const TEST_OSC: ProgramNode = {
  op: 'program',
  name: 'TestOsc',
  ports: {
    inputs: [{ name: 'freq', type: 'freq', default: 440 }],
    outputs: [
      { name: 'saw', type: 'signal' },
      { name: 'sin', type: 'signal' },
    ],
  },
  body: { op: 'block',
    decls: [
      { op: 'reg_decl', name: 'phase', init: 0 },
      { op: 'instance_decl', name: 'sin1', program: 'Sin', inputs: {
        x: { op: 'mul', args: [6.283185307179586, { op: 'reg', name: 'phase' }] },
      }},
    ],
    assigns: [
      { op: 'output_assign', name: 'saw', expr: { op: 'sub', args: [{ op: 'mul', args: [2, { op: 'reg', name: 'phase' }] }, 1] } },
      { op: 'output_assign', name: 'sin', expr: { op: 'nested_out', ref: 'sin1', output: 'out' } },
      { op: 'next_update', target: { kind: 'reg', name: 'phase' }, expr: { op: 'mod', args: [
        { op: 'add', args: [
          { op: 'reg', name: 'phase' },
          { op: 'div', args: [{ op: 'input', name: 'freq' }, { op: 'sample_rate' }] },
        ]},
        1,
      ]}},
    ],
  },
}

/** Build and compile a single-oscillator session outputting the named waveform. */
function oscSession(freq: number, output: 'saw' | 'sin', bufferLength = 256) {
  const session = makeSession(bufferLength)
  loadBuiltins(session.typeRegistry)
  session.typeRegistry.set('TestOsc', loadProgramAsType(TEST_OSC,session))
  loadJSON({
    schema: 'tropical_program_2',
    name: 'test',
    body: { op: 'block', decls: [
      { op: 'instance_decl', name: 'osc', program: 'TestOsc', inputs: { freq } },
    ]},
    audio_outputs: [{ instance: 'osc', output }],
  } as ProgramFile, session)
  return session
}

// ─── tests ────��───────────────────────────────────────────────────────────────

describe('renderFrames / buffer backend', () => {
  test('sawtooth peak and RMS are in expected range', () => {
    const session = oscSession(440, 'saw')
    const samples = renderFrames(session.runtime, 16)  // 16 × 256 = 4096 samples

    // Naive saw ranges [-1, 1), JIT divides by 20 → peak ≈ 0.05
    expect(peak(samples)).toBeGreaterThan(0.03)
    expect(peak(samples)).toBeLessThan(0.07)
    // Theoretical RMS of sawtooth ≈ 0.577/20 ≈ 0.029
    expect(rms(samples)).toBeGreaterThan(0.015)

    session.graph.dispose()
  })

  test('sine dominant frequency matches configured oscillator frequency', () => {
    const session = oscSession(440, 'sin')
    const samples = renderFrames(session.runtime, 174)  // ~1 s

    const freq = dominantFrequency(samples, 44100)
    expect(Math.abs(freq - 440)).toBeLessThan(15)  // ±15 Hz tolerance

    session.graph.dispose()
  })

  test('hot-swap updates frequency while preserving phase state', () => {
    const session = makeSession(256)
    loadBuiltins(session.typeRegistry)
    session.typeRegistry.set('TestOsc', loadProgramAsType(TEST_OSC,session))
    loadJSON({
      schema: 'tropical_program_2',
      name: 'test',
      body: { op: 'block', decls: [
        { op: 'instance_decl', name: 'osc', program: 'TestOsc', inputs: { freq: 220 } },
      ]},
      audio_outputs: [{ instance: 'osc', output: 'sin' }],
    } as ProgramFile, session)

    renderFrames(session.runtime, 8)

    session.inputExprNodes.set('osc:freq', 440)
    applySessionWiring(session)

    const samples = renderFrames(session.runtime, 174)
    const freq = dominantFrequency(samples, 44100)
    expect(Math.abs(freq - 440)).toBeLessThan(15)
    expect(Math.abs(freq - 220)).toBeGreaterThan(15)

    session.graph.dispose()
  })

  test('WAV file is written with correct byte size', async () => {
    const session = oscSession(440, 'saw')
    const samples = renderFrames(session.runtime, 16)

    const path = join(tmpdir(), 'tropical_render_test.wav')
    await writeWav(path, samples, 44100)

    expect(existsSync(path)).toBe(true)
    const expectedBytes = 46 + samples.length * 4
    expect(statSync(path).size).toBe(expectedBytes)

    unlinkSync(path)
    session.graph.dispose()
  })

  test('sample count equals nCalls * bufferLength regardless of buffer size', () => {
    const prog: ProgramFile = {
      schema: 'tropical_program_2',
      name: 'test',
      body: { op: 'block', decls: [
        { op: 'instance_decl', name: 'osc', program: 'TestOsc', inputs: { freq: 440 } },
      ]},
      audio_outputs: [{ instance: 'osc', output: 'sin' }],
    }

    const s32 = makeSession(32)
    loadBuiltins(s32.typeRegistry)
    s32.typeRegistry.set('TestOsc', loadProgramAsType(TEST_OSC,s32))
    loadJSON(prog, s32)
    const a = renderFrames(s32.runtime, 16)  // 16 × 32 = 512

    const s512 = makeSession(512)
    loadBuiltins(s512.typeRegistry)
    s512.typeRegistry.set('TestOsc', loadProgramAsType(TEST_OSC,s512))
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
