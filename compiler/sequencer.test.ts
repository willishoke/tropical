import { describe, test, expect } from 'bun:test'
import { makeSession, loadJSON, resolveProgramType } from './session'
import { loadStdlib } from './program'
import { flattenSession } from './flatten'
import { Float, Int, ArrayType, portTypeEqual } from './term'

describe('stdlib Sequencer<N>', () => {
  test('Sequencer<N> monomorphizes values input shape to [N]', () => {
    const session = makeSession()
    loadStdlib(session)
    const { type: t4 } = resolveProgramType(session, 'Sequencer', { N: 4 }, undefined)
    const { type: t8 } = resolveProgramType(session, 'Sequencer', { N: 8 }, undefined)
    // values is the second input
    expect(t4._def.inputPortTypes[1]).toEqual(ArrayType(Float, [4]))
    expect(t8._def.inputPortTypes[1]).toEqual(ArrayType(Float, [8]))
    expect(t4).not.toBe(t8)
  })

  test('Sequencer uses declared default N=8 when no type_args provided', () => {
    const session = makeSession()
    loadStdlib(session)
    const { type, typeArgs } = resolveProgramType(session, 'Sequencer', undefined, undefined)
    expect(typeArgs).toEqual({ N: 8 })
    expect(type._def.inputPortTypes[1]).toEqual(ArrayType(Float, [8]))
  })

  test('Sequencer<4> flattens end-to-end with an arrayPack input', () => {
    const session = makeSession(64)
    loadStdlib(session)
    loadJSON({
      schema: 'tropical_program_2',
      name: 'test',
      body: { op: 'block', decls: [
        { op: 'instanceDecl', name: 'clk', program: 'Clock', inputs: { freq: 4, ratios_in: [1] } },
        { op: 'instanceDecl', name: 'seq', program: 'Sequencer', type_args: { N: 4 }, inputs: {
          clock: { op: 'ref', instance: 'clk', output: 'output' },
          values: [110, 220, 330, 440],
        }},
      ]},
      audio_outputs: [{ instance: 'seq', output: 'value' }],
    }, session)
    const plan = flattenSession(session)
    expect(plan).toBeDefined()
    const inst = session.instanceRegistry.get('seq')!
    expect(inst.typeName).toBe('Sequencer')
    expect(inst.typeArgs).toEqual({ N: 4 })
  })
})
