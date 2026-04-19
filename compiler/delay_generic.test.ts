import { describe, test, expect } from 'bun:test'
import { makeSession, loadJSON, resolveProgramType } from './session'
import { loadStdlib } from './program'
import { flattenSession } from './flatten'
import { Float, ArrayType } from './term'

describe('stdlib Delay<N>', () => {
  test('Delay with N=8 resolves to a distinct type from Delay with N=44100', () => {
    const session = makeSession()
    loadStdlib(session)
    const a = resolveProgramType(session, 'Delay', { N: 8 }, undefined)
    const b = resolveProgramType(session, 'Delay', { N: 44100 }, undefined)
    expect(a.type).not.toBe(b.type)
    expect(a.type._def.registerPortTypes[0]).toEqual(ArrayType(Float, [8]))
    expect(b.type._def.registerPortTypes[0]).toEqual(ArrayType(Float, [44100]))
  })

  test('Delay with default N=44100 matches explicit N=44100', () => {
    const session = makeSession()
    loadStdlib(session)
    const def = resolveProgramType(session, 'Delay', undefined, undefined)
    const exp = resolveProgramType(session, 'Delay', { N: 44100 }, undefined)
    expect(def.type).toBe(exp.type)
    expect(def.typeArgs).toEqual({ N: 44100 })
  })

  test('Delay<N=8> flattens in a small patch without error', () => {
    const session = makeSession(64)
    loadStdlib(session)
    loadJSON({
      schema: 'tropical_program_1',
      name: 'test',
      instances: {
        d: { program: 'Delay', type_args: { N: 8 }, inputs: { x: 0.5 } },
      },
      audio_outputs: [{ instance: 'd', output: 'y' }],
    }, session)
    const plan = flattenSession(session)
    expect(plan).toBeDefined()
    const inst = session.instanceRegistry.get('d')!
    expect(inst.typeName).toBe('Delay')
    expect(inst.typeArgs).toEqual({ N: 8 })
  })
})
