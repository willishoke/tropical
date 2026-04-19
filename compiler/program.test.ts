/**
 * program.test.ts — Tests for ProgramJSON schema validation and export.
 */

import { describe, test, expect } from 'bun:test'
import { parseProgram } from './schema'
import { makeSession, loadJSON } from './session'
import {
  loadStdlib as loadBuiltins, loadProgramAsType, loadProgramAsSession,
  saveProgramFromSession, exportSessionAsProgram, type ProgramJSON,
} from './program'
import { resolveProgramType } from './session'
import { validateExpr } from './expr'

// ─────────────────────────────────────────────────────────────
// Schema validation
// ─────────────────────────────────────────────────────────────

describe('parseProgram', () => {
  test('validates a minimal program', () => {
    const raw = {
      schema: 'tropical_program_1',
      name: 'Test',
      outputs: ['out'],
      process: { outputs: { out: 42 } },
    }
    const prog = parseProgram(raw)
    expect(prog.name).toBe('Test')
  })

  test('validates a graph program', () => {
    const raw = {
      schema: 'tropical_program_1',
      name: 'TestPatch',
      instances: {
        VCO1: { program: 'VCO', inputs: { freq: 440 } },
      },
      audio_outputs: [{ instance: 'VCO1', output: 'sin' }],
    }
    const prog = parseProgram(raw)
    expect(prog.instances!['VCO1'].program).toBe('VCO')
  })

  test('validates nested programs', () => {
    const raw = {
      schema: 'tropical_program_1',
      name: 'Composite',
      programs: {
        MyOsc: {
          schema: 'tropical_program_1',
          name: 'MyOsc',
          inputs: ['freq'],
          outputs: ['out'],
          process: { outputs: { out: { op: 'input', name: 'freq' } } },
        },
      },
      instances: { o1: { program: 'MyOsc', inputs: { freq: 440 } } },
      audio_outputs: [{ instance: 'o1', output: 'out' }],
    }
    const prog = parseProgram(raw)
    expect(prog.programs!['MyOsc'].name).toBe('MyOsc')
  })

  test('rejects invalid schema', () => {
    expect(() => parseProgram({ schema: 'wrong', name: 'X' })).toThrow('Invalid program')
  })

  test('rejects missing name', () => {
    expect(() => parseProgram({ schema: 'tropical_program_1' })).toThrow('Invalid program')
  })
})

// ─────────────────────────────────────────────────────────────
// exportSessionAsProgram
// ─────────────────────────────────────────────────────────────

function makeTestSession() {
  const session = makeSession(256)
  loadBuiltins(session)
  return session
}

describe('exportSessionAsProgram — port type round-trip', () => {
  test('emits typed inputs/outputs matching source instance port types', () => {
    const session = makeSession(256)
    // Register a leaf type with typed inputs and outputs
    const typedLeaf: ProgramJSON = {
      schema: 'tropical_program_1',
      name: 'TypedLeaf',
      inputs: [{ name: 'a', type: { kind: 'array', element: 'float', shape: [4] } }],
      outputs: [{ name: 'out', type: { kind: 'array', element: 'float', shape: [4] } }],
      process: { outputs: { out: { op: 'input', name: 'a' } } },
    }
    loadProgramAsType(typedLeaf, session)
    const { type } = resolveProgramType(session, 'TypedLeaf', undefined, undefined)
    session.instanceRegistry.set('t1', type.instantiateAs('t1', { baseTypeName: 'TypedLeaf' }))

    const exported = exportSessionAsProgram(session, {
      name: 'Exported',
      inputs: { a: 't1:a' },
      outputs: { out: { instance: 't1', output: 'out' } },
    })

    // Re-parse through the schema to confirm it's well-formed
    const reparsed = parseProgram(exported) as ProgramJSON
    const inputEntry = reparsed.inputs![0]
    const outputEntry = reparsed.outputs![0]
    expect(typeof inputEntry === 'object' && inputEntry.type).toEqual({ kind: 'array', element: 'float', shape: [4] })
    expect(typeof outputEntry === 'object' && outputEntry.type).toEqual({ kind: 'array', element: 'float', shape: [4] })

    // Load the exported program into a fresh session and check port types survive.
    // The nested TypedLeaf must be registered before the exported composite that uses it.
    const session2 = makeSession(256)
    loadProgramAsType(typedLeaf, session2)
    loadProgramAsType(exported, session2)
    const { type: exportedType } = resolveProgramType(session2, 'Exported', undefined, undefined)
    const srcPt = exportedType._def.inputPortTypes[0]
    const dstPt = exportedType._def.outputPortTypes[0]
    expect(srcPt?.tag).toBe('array')
    expect(dstPt?.tag).toBe('array')
    if (srcPt?.tag === 'array') expect(srcPt.shape).toEqual([4])
    if (dstPt?.tag === 'array') expect(dstPt.shape).toEqual([4])
  })

  test('emits bare string names when no type or bounds are declared', () => {
    const session = makeSession(256)
    const plain: ProgramJSON = {
      schema: 'tropical_program_1',
      name: 'Plain',
      inputs: ['x'],
      outputs: ['y'],
      process: { outputs: { y: { op: 'input', name: 'x' } } },
    }
    loadProgramAsType(plain, session)
    const { type } = resolveProgramType(session, 'Plain', undefined, undefined)
    session.instanceRegistry.set('p1', type.instantiateAs('p1', { baseTypeName: 'Plain' }))

    const exported = exportSessionAsProgram(session, {
      name: 'Exported',
      inputs: { x: 'p1:x' },
      outputs: { y: { instance: 'p1', output: 'y' } },
    })
    expect(exported.inputs![0]).toBe('x')
    expect(exported.outputs![0]).toBe('y')
  })
})

describe('exportSessionAsProgram', () => {
  test('errors on unknown instance in outputs', () => {
    const session = makeTestSession()
    expect(() => exportSessionAsProgram(session, {
      name: 'Bad',
      inputs: {},
      outputs: { out: { instance: 'nope', output: 'out' } },
    })).toThrow("unknown instance 'nope'")
  })

  test('errors on malformed input target', () => {
    const session = makeTestSession()
    expect(() => exportSessionAsProgram(session, {
      name: 'Bad',
      inputs: { x: 'nocolon' },
      outputs: {},
    })).toThrow('must be "instance:port"')
  })
})

// ─────────────────────────────────────────────────────────────
// On-demand type resolution
// ─────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────
// Generic programs — save/load round-trip
// ─────────────────────────────────────────────────────────────

describe('generic programs round-trip', () => {
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

  test('saveProgramFromSession emits type_args on instance entries', () => {
    const session = makeSession()
    loadProgramAsType(genericDelay(), session)
    const { type, typeArgs } = resolveProgramType(session, 'Delay', { N: 8 }, undefined)
    const inst = type.instantiateAs('d1', { baseTypeName: 'Delay', typeArgs })
    session.instanceRegistry.set('d1', inst)

    const saved = saveProgramFromSession(session)
    expect(saved.instances!['d1'].program).toBe('Delay')
    expect(saved.instances!['d1'].type_args).toEqual({ N: 8 })
  })

  test('saveProgramFromSession omits type_args on non-generic instances', () => {
    const session = makeSession()
    const p: ProgramJSON = {
      schema: 'tropical_program_1',
      name: 'Passthrough',
      inputs: ['x'],
      outputs: ['y'],
      process: { outputs: { y: { op: 'input', name: 'x' } } },
    }
    loadProgramAsType(p, session)
    const { type } = resolveProgramType(session, 'Passthrough', undefined, undefined)
    session.instanceRegistry.set('p1', type.instantiateAs('p1', { baseTypeName: 'Passthrough' }))

    const saved = saveProgramFromSession(session)
    expect(saved.instances!['p1'].program).toBe('Passthrough')
    expect(saved.instances!['p1'].type_args).toBeUndefined()
  })
})

describe('typeResolver', () => {
  test('circular stdlib dependency throws with cycle path', () => {
    const session = makeSession()

    const fakeTypes = new Map<string, ProgramJSON>([
      ['CycleA', {
        schema: 'tropical_program_1', name: 'CycleA',
        inputs: [], outputs: ['out'],
        instances: { b: { program: 'CycleB' } },
        process: { outputs: { out: 0 } },
      } as ProgramJSON],
      ['CycleB', {
        schema: 'tropical_program_1', name: 'CycleB',
        inputs: [], outputs: ['out'],
        instances: { a: { program: 'CycleA' } },
        process: { outputs: { out: 0 } },
      } as ProgramJSON],
    ])

    const loading = new Set<string>()
    session.typeResolver = (name: string) => {
      const existing = session.typeRegistry.get(name)
      if (existing) return existing
      if (loading.has(name))
        throw new Error(`Circular stdlib dependency: ${[...loading, name].join(' → ')}`)
      const prog = fakeTypes.get(name)
      if (!prog) return undefined
      loading.add(name)
      const type = loadProgramAsType(prog, session)
      session.typeRegistry.set(name, type)
      loading.delete(name)
      return type
    }

    expect(() => session.typeResolver!('CycleA')).toThrow('Circular stdlib dependency: CycleA → CycleB → CycleA')

    session.graph.dispose()
  })
})
