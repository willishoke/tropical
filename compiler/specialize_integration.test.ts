import { describe, test, expect } from 'bun:test'
import { makeSession, resolveProgramType } from './session.js'
import { loadProgramAsType } from './program.js'
import type { ProgramJSON } from './program.js'
import { Float, ArrayType } from './term.js'

function genericDelay(): ProgramJSON {
  return {
    schema: 'tropical_program_1',
    name: 'Delay',
    type_params: { N: { type: 'int', default: 44100 } },
    inputs: ['x'],
    outputs: ['y'],
    regs: { buf: { zeros: { type_param: 'N' } } as any },
    input_defaults: { x: 0 },
    breaks_cycles: true,
    process: {
      outputs: {
        y: {
          op: 'index',
          args: [
            { op: 'reg', name: 'buf' },
            { op: 'mod', args: [{ op: 'sample_index' }, { op: 'type_param', name: 'N' }] },
          ],
        },
      },
      next_regs: {
        buf: {
          op: 'array_set',
          args: [
            { op: 'reg', name: 'buf' },
            { op: 'mod', args: [{ op: 'sample_index' }, { op: 'type_param', name: 'N' }] },
            { op: 'input', name: 'x' },
          ],
        },
      },
    },
  }
}

describe('resolveProgramType — generic instantiation', () => {
  test('monomorphizes and caches a generic type', () => {
    const session = makeSession()
    loadProgramAsType(genericDelay(), session)
    // Generic programs go into genericTemplates, not typeRegistry
    expect(session.typeRegistry.has('Delay')).toBe(false)
    expect(session.genericTemplates.has('Delay')).toBe(true)

    const { type: t1, typeArgs: a1 } = resolveProgramType(session, 'Delay', { N: 8 }, undefined)
    const { type: t2, typeArgs: a2 } = resolveProgramType(session, 'Delay', { N: 16 }, undefined)
    const { type: t3, typeArgs: a3 } = resolveProgramType(session, 'Delay', { N: 8 }, undefined)

    expect(t1).not.toBe(t2)
    expect(t1).toBe(t3)  // cache hit
    expect(a1).toEqual({ N: 8 })
    expect(a2).toEqual({ N: 16 })
    expect(a3).toEqual({ N: 8 })

    // Specialized regs have concrete sizes
    expect(t1._def.registerPortTypes[0]).toEqual(ArrayType(Float, [8]))
    expect(t2._def.registerPortTypes[0]).toEqual(ArrayType(Float, [16]))
  })

  test('applies declared default when type_args absent', () => {
    const session = makeSession()
    loadProgramAsType(genericDelay(), session)
    const { type, typeArgs } = resolveProgramType(session, 'Delay', undefined, undefined)
    expect(typeArgs).toEqual({ N: 44100 })
    expect(type._def.registerPortTypes[0]).toEqual(ArrayType(Float, [44100]))
  })

  test('rejects type_args on non-generic programs', () => {
    const session = makeSession()
    const onePole: ProgramJSON = {
      schema: 'tropical_program_1',
      name: 'Identity',
      inputs: ['x'],
      outputs: ['y'],
      process: { outputs: { y: { op: 'input', name: 'x' } } },
    }
    loadProgramAsType(onePole, session)
    expect(() => resolveProgramType(session, 'Identity', { N: 8 }, undefined)).toThrow(/does not declare type_params/)
  })

  test('errors on unknown program', () => {
    const session = makeSession()
    expect(() => resolveProgramType(session, 'NoSuchThing', undefined, undefined)).toThrow(/Unknown program type/)
  })

  test('non-generic programs still resolve through typeRegistry', () => {
    const session = makeSession()
    const p: ProgramJSON = {
      schema: 'tropical_program_1',
      name: 'Passthrough',
      inputs: ['x'],
      outputs: ['y'],
      process: { outputs: { y: { op: 'input', name: 'x' } } },
    }
    loadProgramAsType(p, session)
    const { type, typeArgs } = resolveProgramType(session, 'Passthrough', undefined, undefined)
    expect(type).toBeDefined()
    expect(typeArgs).toBeUndefined()
  })

  test('type_param refs in array port shapes become concrete PortTypes after instantiation', () => {
    const prog: ProgramJSON = {
      schema: 'tropical_program_1',
      name: 'Bus',
      type_params: { N: { type: 'int', default: 4 } },
      inputs: [
        { name: 'values', type: { kind: 'array', element: 'float', shape: [{ op: 'type_param', name: 'N' }] } },
      ],
      outputs: [
        { name: 'out', type: { kind: 'array', element: 'float', shape: [{ op: 'type_param', name: 'N' }] } },
      ],
      process: { outputs: { out: { op: 'input', name: 'values' } } },
    }
    const session = makeSession()
    loadProgramAsType(prog, session)

    const { type: t8 } = resolveProgramType(session, 'Bus', { N: 8 }, undefined)
    expect(t8._def.inputPortTypes[0]).toEqual(ArrayType(Float, [8]))
    expect(t8._def.outputPortTypes[0]).toEqual(ArrayType(Float, [8]))

    const { type: tdef } = resolveProgramType(session, 'Bus', undefined, undefined)
    expect(tdef._def.inputPortTypes[0]).toEqual(ArrayType(Float, [4]))
  })
})
