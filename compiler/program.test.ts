/**
 * program.test.ts — Tests for program schema validation and export.
 */

import { describe, test, expect } from 'bun:test'
import { parseProgramV2 } from './schema'
import { makeSession, loadJSON, v2NodeToFile, type ExprNode } from './session'
import {
  loadStdlib as loadBuiltins, loadProgramAsType,
  saveProgramFromSession, exportSessionAsProgram, instanceDecls,
  type ProgramNode, type ProgramPortSpec,
} from './program'
import { resolveProgramType } from './session'

// ─────────────────────────────────────────────────────────────
// Schema validation
// ─────────────────────────────────────────────────────────────

describe('parseProgramV2', () => {
  test('validates a minimal program', () => {
    const raw = {
      schema: 'tropical_program_2',
      name: 'Test',
      ports: { outputs: ['out'] },
      body: { op: 'block', assigns: [{ op: 'output_assign', name: 'out', expr: 42 }] },
    }
    const prog = parseProgramV2(raw)
    expect(prog.name).toBe('Test')
  })

  test('validates a graph program', () => {
    const raw = {
      schema: 'tropical_program_2',
      name: 'TestPatch',
      body: { op: 'block', decls: [
        { op: 'instance_decl', name: 'VCO1', program: 'VCO', inputs: { freq: 440 } },
      ]},
      audio_outputs: [{ instance: 'VCO1', output: 'sin' }],
    }
    const prog = parseProgramV2(raw)
    const vco1 = [...instanceDecls(prog as unknown as ProgramNode)].find(d => d.name === 'VCO1')!
    expect(vco1.program).toBe('VCO')
  })

  test('validates nested programs', () => {
    const raw = {
      schema: 'tropical_program_2',
      name: 'Composite',
      body: { op: 'block', decls: [
        { op: 'program_decl', name: 'MyOsc', program: {
          op: 'program',
          name: 'MyOsc',
          ports: { inputs: ['freq'], outputs: ['out'] },
          body: { op: 'block',
            assigns: [{ op: 'output_assign', name: 'out', expr: { op: 'input', name: 'freq' } }],
          },
        }},
        { op: 'instance_decl', name: 'o1', program: 'MyOsc', inputs: { freq: 440 } },
      ]},
      audio_outputs: [{ instance: 'o1', output: 'out' }],
    }
    const prog = parseProgramV2(raw)
    expect(prog.name).toBe('Composite')
  })

  test('rejects invalid schema', () => {
    expect(() => parseProgramV2({ schema: 'wrong', name: 'X' })).toThrow('Invalid program')
  })

  test('rejects missing name', () => {
    expect(() => parseProgramV2({ schema: 'tropical_program_2' })).toThrow('Invalid program')
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
    const typedLeaf: ProgramNode = {
      op: 'program',
      name: 'TypedLeaf',
      ports: {
        inputs: [{ name: 'a', type: { kind: 'array', element: 'float', shape: [4] } }],
        outputs: [{ name: 'out', type: { kind: 'array', element: 'float', shape: [4] } }],
      },
      body: { op: 'block',
        assigns: [{ op: 'output_assign', name: 'out', expr: { op: 'input', name: 'a' } }],
      },
    }
    loadProgramAsType(typedLeaf, session)
    const { type } = resolveProgramType(session, 'TypedLeaf', undefined, undefined)
    session.instanceRegistry.set('t1', type.instantiateAs('t1', { baseTypeName: 'TypedLeaf' }))

    const exported = exportSessionAsProgram(session, {
      name: 'Exported',
      inputs: { a: 't1:a' },
      outputs: { out: { instance: 't1', output: 'out' } },
    })

    const reparsed = parseProgramV2(v2NodeToFile(exported as unknown as ExprNode))
    const inputEntry = (reparsed.ports?.inputs ?? [])[0] as ProgramPortSpec
    const outputEntry = (reparsed.ports?.outputs ?? [])[0] as ProgramPortSpec
    expect(typeof inputEntry === 'object' && inputEntry.type).toEqual({ kind: 'array', element: 'float', shape: [4] })
    expect(typeof outputEntry === 'object' && outputEntry.type).toEqual({ kind: 'array', element: 'float', shape: [4] })

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
    const plain: ProgramNode = {
      op: 'program',
      name: 'Plain',
      ports: { inputs: ['x'], outputs: ['y'] },
      body: { op: 'block',
        assigns: [{ op: 'output_assign', name: 'y', expr: { op: 'input', name: 'x' } }],
      },
    }
    loadProgramAsType(plain, session)
    const { type } = resolveProgramType(session, 'Plain', undefined, undefined)
    session.instanceRegistry.set('p1', type.instantiateAs('p1', { baseTypeName: 'Plain' }))

    const exported = exportSessionAsProgram(session, {
      name: 'Exported',
      inputs: { x: 'p1:x' },
      outputs: { y: { instance: 'p1', output: 'y' } },
    })
    expect(exported.ports?.inputs?.[0]).toBe('x')
    expect(exported.ports?.outputs?.[0]).toBe('y')
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
// Generic programs — save/load round-trip
// ─────────────────────────────────────────────────────────────

describe('generic programs round-trip', () => {
  function genericDelay(): ProgramNode {
    return {
      op: 'program',
      name: 'Delay',
      type_params: { N: { type: 'int', default: 44100 } },
      breaks_cycles: true,
      ports: {
        inputs: [{ name: 'x', default: 0 }],
        outputs: ['y'],
      },
      body: { op: 'block',
        decls: [
          { op: 'reg_decl', name: 'buf', init: { zeros: { type_param: 'N' } } as any },
        ],
        assigns: [
          { op: 'output_assign', name: 'y', expr: {
            op: 'index',
            args: [
              { op: 'reg', name: 'buf' },
              { op: 'mod', args: [{ op: 'sample_index' }, { op: 'type_param', name: 'N' }] },
            ],
          }},
          { op: 'next_update', target: { kind: 'reg', name: 'buf' }, expr: {
            op: 'array_set',
            args: [
              { op: 'reg', name: 'buf' },
              { op: 'mod', args: [{ op: 'sample_index' }, { op: 'type_param', name: 'N' }] },
              { op: 'input', name: 'x' },
            ],
          }},
        ],
      },
    }
  }

  test('saveProgramFromSession emits type_args on instance entries', () => {
    const session = makeSession()
    loadProgramAsType(genericDelay(), session)
    const { type, typeArgs } = resolveProgramType(session, 'Delay', { N: 8 }, undefined)
    const inst = type.instantiateAs('d1', { baseTypeName: 'Delay', typeArgs })
    session.instanceRegistry.set('d1', inst)

    const { node: saved } = saveProgramFromSession(session)
    const d1 = [...instanceDecls(saved)].find(d => d.name === 'd1')!
    expect(d1.program).toBe('Delay')
    expect(d1.type_args).toEqual({ N: 8 })
  })

  test('saveProgramFromSession omits type_args on non-generic instances', () => {
    const session = makeSession()
    const p: ProgramNode = {
      op: 'program',
      name: 'Passthrough',
      ports: { inputs: ['x'], outputs: ['y'] },
      body: { op: 'block',
        assigns: [{ op: 'output_assign', name: 'y', expr: { op: 'input', name: 'x' } }],
      },
    }
    loadProgramAsType(p, session)
    const { type } = resolveProgramType(session, 'Passthrough', undefined, undefined)
    session.instanceRegistry.set('p1', type.instantiateAs('p1', { baseTypeName: 'Passthrough' }))

    const { node: saved } = saveProgramFromSession(session)
    const p1 = [...instanceDecls(saved)].find(d => d.name === 'p1')!
    expect(p1.program).toBe('Passthrough')
    expect(p1.type_args).toBeUndefined()
  })
})

describe('typeResolver', () => {
  test('circular stdlib dependency throws with cycle path', () => {
    const session = makeSession()

    const fakeTypes = new Map<string, ProgramNode>([
      ['CycleA', {
        op: 'program', name: 'CycleA',
        ports: { inputs: [], outputs: ['out'] },
        body: { op: 'block', decls: [
          { op: 'instance_decl', name: 'b', program: 'CycleB' },
        ], assigns: [{ op: 'output_assign', name: 'out', expr: 0 }] },
      }],
      ['CycleB', {
        op: 'program', name: 'CycleB',
        ports: { inputs: [], outputs: ['out'] },
        body: { op: 'block', decls: [
          { op: 'instance_decl', name: 'a', program: 'CycleA' },
        ], assigns: [{ op: 'output_assign', name: 'out', expr: 0 }] },
      }],
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
