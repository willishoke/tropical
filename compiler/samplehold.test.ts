import { describe, test, expect } from 'bun:test'
import { makeSession, loadJSON } from './session'
import { loadStdlib } from './program'
import { flattenExpressions } from './flatten'
import { interpretSamples } from './interpret'

describe('stdlib SampleHold', () => {
  test('latches on trigger rising edge', () => {
    const session = makeSession(44100)
    loadStdlib(session)
    loadJSON({
      schema: 'tropical_program_2',
      name: 'test',
      body: { op: 'block', decls: [
        { op: 'instance_decl', name: 'sh', program: 'SampleHold', inputs: {
          trigger: {
            op: 'select',
            args: [
              { op: 'eq', args: [{ op: 'sample_index' }, 100] },
              1,
              0,
            ],
          },
          input: {
            op: 'mul',
            args: [{ op: 'sample_index' }, 0.01],
          },
        }},
      ]},
      audio_outputs: [{ instance: 'sh', output: 'value' }],
    }, session)

    const flat = flattenExpressions(session)
    const out = interpretSamples(flat, 300)

    expect(out[50] / 20.0 / 0.05).toBeCloseTo(0, 5)
    const heldValue = out[200] * 20.0
    expect(heldValue).toBeCloseTo(1.0, 1)
  })
})
