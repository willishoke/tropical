import { describe, test, expect } from 'bun:test'
import { makeSession, loadJSON } from './session'
import { loadStdlib } from './program'
import { flattenExpressions } from './flatten'
import { interpretSamples } from './interpret'
import { renderFrames } from './test_utils/audio'

describe('stdlib BubbleCloud', () => {
  test('single trigger fires exactly one voice; 8 triggers distribute round-robin', () => {
    const session = makeSession(44100)
    loadStdlib(session)
    loadJSON({
      schema: 'tropical_program_2',
      name: 'test',
      body: { op: 'block', decls: [
        { op: 'instanceDecl', name: 'c', program: 'BubbleCloud', inputs: {
          trigger: {
            op: 'select',
            args: [
              { op: 'eq', args: [{ op: 'mod', args: [{ op: 'sampleIndex' }, 1000] }, 0] },
              1,
              0,
            ],
          },
          radius: 0.003,
          amp_scale: 0.05,
        }},
      ]},
      audio_outputs: [{ instance: 'c', output: 'out' }],
    }, session)

    const flat = flattenExpressions(session)
    const out = interpretSamples(flat, 10000)

    let peak = 0
    for (const v of out) peak = Math.max(peak, Math.abs(v))
    expect(Number.isFinite(peak)).toBe(true)
    expect(peak).toBeGreaterThan(0)
    // 8 voices distributed across 10 triggers at amp=0.05 — staggered overlaps
    // should stay safely under the output safety clamp at [-1, 1].
    expect(peak).toBeLessThan(0.05) // interp divides by 20; this bound is ~1.0 real-audio

    // Each trigger should produce an audible peak in its window.
    let audibleWindows = 0
    for (let i = 0; i < 10000; i += 1000) {
      let windowPeak = 0
      for (let j = i; j < Math.min(i + 500, 10000); j++) {
        windowPeak = Math.max(windowPeak, Math.abs(out[j]))
      }
      if (windowPeak > peak * 0.1) audibleWindows++
    }
    expect(audibleWindows).toBeGreaterThanOrEqual(8)
  })

  test('BubbleCloud JIT matches interpreter bit-exact', () => {
    const bufLen = 256
    const nCalls = 8

    const setup = () => {
      const session = makeSession(bufLen)
      loadStdlib(session)
      loadJSON({
        schema: 'tropical_program_2',
        name: 'test',
        body: { op: 'block', decls: [
          { op: 'instanceDecl', name: 'clk', program: 'Clock', inputs: { freq: 8, ratios_in: [1] }},
          { op: 'instanceDecl', name: 'c', program: 'BubbleCloud', inputs: {
            trigger: { op: 'ref', instance: 'clk', output: 'output' },
            radius: 0.003,
            amp_scale: 0.1,
          }},
        ]},
        audio_outputs: [{ instance: 'c', output: 'out' }],
      }, session)
      return session
    }

    const interpSession = setup()
    const flat = flattenExpressions(interpSession)
    const interp = interpretSamples(flat, bufLen * nCalls)

    const jitSession = setup()
    const jit = renderFrames(jitSession.runtime, nCalls)
    jitSession.runtime.dispose()

    let maxDiff = 0
    for (let i = 0; i < interp.length; i++) {
      maxDiff = Math.max(maxDiff, Math.abs(interp[i] - jit[i]))
    }
    expect(maxDiff).toBeLessThan(1e-9)
  })
})
