import { describe, test, expect } from 'bun:test'
import { makeSession, loadJSON } from './session'
import { loadStdlib } from './program'
import { flattenExpressions } from './flatten'
import { interpretSamples } from './interpret'

describe('stdlib SVF', () => {
  test('flattens and produces a decaying impulse response on bp output', () => {
    const session = makeSession(44100)
    loadStdlib(session)
    loadJSON({
      schema: 'tropical_program_2',
      name: 'test',
      body: { op: 'block', decls: [
        { op: 'instanceDecl', name: 'svf', program: 'SVF', inputs: {
          input: {
            op: 'select',
            args: [
              { op: 'eq', args: [{ op: 'sampleIndex' }, 0] },
              1,
              0,
            ],
          },
          cutoff: 1000,
          q: 10,
        }},
      ]},
      audio_outputs: [{ instance: 'svf', output: 'bp' }],
    }, session)

    const flat = flattenExpressions(session)
    const out = interpretSamples(flat, 2000)

    let earlyPeak = 0
    for (let i = 0; i < 200; i++) earlyPeak = Math.max(earlyPeak, Math.abs(out[i]))
    let latePeak = 0
    for (let i = 1500; i < 2000; i++) latePeak = Math.max(latePeak, Math.abs(out[i]))

    expect(earlyPeak).toBeGreaterThan(0)
    expect(latePeak).toBeLessThan(earlyPeak * 0.1)

    let maxAbs = 0
    for (const v of out) maxAbs = Math.max(maxAbs, Math.abs(v))
    expect(Number.isFinite(maxAbs)).toBe(true)
    expect(maxAbs).toBeLessThan(10)
  })
})
