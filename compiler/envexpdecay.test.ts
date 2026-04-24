import { describe, test, expect } from 'bun:test'
import { makeSession, loadJSON } from './session'
import { loadStdlib } from './program'
import { flattenExpressions } from './flatten'
import { interpretSamples } from './interpret'

describe('stdlib EnvExpDecay', () => {
  test('resets to 1 on trigger and decays exponentially', () => {
    const session = makeSession(44100)
    loadStdlib(session)
    loadJSON({
      schema: 'tropical_program_2',
      name: 'test',
      body: { op: 'block', decls: [
        { op: 'instance_decl', name: 'env', program: 'EnvExpDecay', inputs: {
          trigger: {
            op: 'select',
            args: [
              { op: 'eq', args: [{ op: 'sample_index' }, 10] },
              1,
              0,
            ],
          },
          decay: 0.99,
        }},
      ]},
      audio_outputs: [{ instance: 'env', output: 'env' }],
    }, session)

    const flat = flattenExpressions(session)
    const out = interpretSamples(flat, 200)

    const peak = out[11] * 20.0
    expect(peak).toBeCloseTo(1.0, 3)

    const late = out[150] * 20.0
    expect(late).toBeGreaterThan(0)
    expect(late).toBeLessThan(peak * 0.5)
  })
})
