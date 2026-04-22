import { describe, test, expect } from 'bun:test'
import { makeSession, loadJSON } from './session'
import { loadStdlib } from './program'
import { flattenExpressions } from './flatten'
import { interpretSamples } from './interpret'

describe('stdlib Bubble', () => {
  test('single trigger produces decaying output with audible ringing', () => {
    const session = makeSession(44100)
    loadStdlib(session)
    loadJSON({
      schema: 'tropical_program_2',
      name: 'test',
      body: { op: 'block', decls: [
        { op: 'instance_decl', name: 'b', program: 'Bubble', inputs: {
          trigger: {
            op: 'select',
            args: [
              { op: 'eq', args: [{ op: 'sample_index' }, 100] },
              1,
              0,
            ],
          },
          radius: 0.003,
        }},
      ]},
      audio_outputs: [{ instance: 'b', output: 'out' }],
    }, session)

    const flat = flattenExpressions(session)
    const out = interpretSamples(flat, 8000)

    let maxAbs = 0
    for (const v of out) maxAbs = Math.max(maxAbs, Math.abs(v))
    expect(Number.isFinite(maxAbs)).toBe(true)
    expect(maxAbs).toBeLessThan(100)

    let preTrigAbs = 0
    for (let i = 0; i < 100; i++) preTrigAbs = Math.max(preTrigAbs, Math.abs(out[i]))
    expect(preTrigAbs).toBeLessThan(1e-9)

    let peakAbs = 0
    for (let i = 100; i < 500; i++) peakAbs = Math.max(peakAbs, Math.abs(out[i]))
    expect(peakAbs).toBeGreaterThan(0)

    let tailAbs = 0
    for (let i = 7000; i < 8000; i++) tailAbs = Math.max(tailAbs, Math.abs(out[i]))
    expect(tailAbs).toBeLessThan(peakAbs * 0.1)

    let zeroCrossings = 0
    for (let i = 101; i < 2000; i++) {
      if ((out[i - 1] >= 0) !== (out[i] >= 0)) zeroCrossings++
    }
    expect(zeroCrossings).toBeGreaterThan(10)
  })

  test('radius controls pitch — smaller radius produces more zero crossings per unit time', () => {
    function runWithRadius(r: number): number {
      const session = makeSession(44100)
      loadStdlib(session)
      loadJSON({
        schema: 'tropical_program_2',
        name: 'test',
        body: { op: 'block', decls: [
          { op: 'instance_decl', name: 'b', program: 'Bubble', inputs: {
            trigger: {
              op: 'select',
              args: [
                { op: 'eq', args: [{ op: 'sample_index' }, 100] },
                1,
                0,
              ],
            },
            radius: r,
            sigma: 0,
          }},
        ]},
        audio_outputs: [{ instance: 'b', output: 'out' }],
      }, session)

      const flat = flattenExpressions(session)
      const out = interpretSamples(flat, 2000)
      let zc = 0
      for (let i = 101; i < 2000; i++) {
        if ((out[i - 1] >= 0) !== (out[i] >= 0)) zc++
      }
      return zc
    }

    const zcSmall = runWithRadius(0.001)
    const zcLarge = runWithRadius(0.01)
    expect(zcSmall).toBeGreaterThan(zcLarge * 2)
  })
})
