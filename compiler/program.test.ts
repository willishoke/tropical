/**
 * program.test.ts — Tests for program schema validation and export.
 */

import { describe, test, expect } from 'bun:test'
import { parseProgramV2 } from './schema'
import { makeSession, loadJSON, v2NodeToFile, type ExprNode } from './session'
import {
  loadStdlib as loadBuiltins, loadProgramAsType, loadProgramAsSession,
  saveProgramFromSession, exportSessionAsProgram, instanceDecls,
  type ProgramNode, type ProgramPortSpec, type ProgramTopLevel,
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
      body: { op: 'block', assigns: [{ op: 'outputAssign', name: 'out', expr: 42 }] },
    }
    const prog = parseProgramV2(raw)
    expect(prog.name).toBe('Test')
  })

  test('validates a graph program', () => {
    const raw = {
      schema: 'tropical_program_2',
      name: 'TestPatch',
      body: { op: 'block', decls: [
        { op: 'instanceDecl', name: 'VCO1', program: 'VCO', inputs: { freq: 440 } },
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
        { op: 'programDecl', name: 'MyOsc', program: {
          op: 'program',
          name: 'MyOsc',
          ports: { inputs: ['freq'], outputs: ['out'] },
          body: { op: 'block',
            assigns: [{ op: 'outputAssign', name: 'out', expr: { op: 'input', name: 'freq' } }],
          },
        }},
        { op: 'instanceDecl', name: 'o1', program: 'MyOsc', inputs: { freq: 440 } },
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
        assigns: [{ op: 'outputAssign', name: 'out', expr: { op: 'input', name: 'a' } }],
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
        assigns: [{ op: 'outputAssign', name: 'y', expr: { op: 'input', name: 'x' } }],
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
          { op: 'regDecl', name: 'buf', init: { zeros: { typeParam: 'N' } } as any },
        ],
        assigns: [
          { op: 'outputAssign', name: 'y', expr: {
            op: 'index',
            args: [
              { op: 'reg', name: 'buf' },
              { op: 'mod', args: [{ op: 'sampleIndex' }, { op: 'typeParam', name: 'N' }] },
            ],
          }},
          { op: 'nextUpdate', target: { kind: 'reg', name: 'buf' }, expr: {
            op: 'arraySet',
            args: [
              { op: 'reg', name: 'buf' },
              { op: 'mod', args: [{ op: 'sampleIndex' }, { op: 'typeParam', name: 'N' }] },
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
        assigns: [{ op: 'outputAssign', name: 'y', expr: { op: 'input', name: 'x' } }],
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

  test('gateable round-trips through schema, loader, and saver (Phase 3)', () => {
    const session = makeSession()
    const passthrough: ProgramNode = {
      op: 'program',
      name: 'Passthrough',
      ports: { inputs: ['x'], outputs: ['y'] },
      body: { op: 'block',
        assigns: [{ op: 'outputAssign', name: 'y', expr: { op: 'input', name: 'x' } }],
      },
    }
    loadProgramAsType(passthrough, session)

    const raw = {
      schema: 'tropical_program_2',
      name: 'Patch',
      body: { op: 'block', decls: [
        { op: 'instanceDecl', name: 'voice_0', program: 'Passthrough',
          inputs: { x: 1.0 }, gateable: true, gate_input: true },
      ]},
      audio_outputs: [{ instance: 'voice_0', output: 'y' }],
    }

    // Pass through v2 schema validation (pass-through Zod + validateExpr on body).
    const prog = parseProgramV2(raw)
    const decl = (prog.body as { decls?: Array<Record<string, unknown>> }).decls!
      .find(d => d.op === 'instanceDecl' && d.name === 'voice_0')!
    expect(decl.gateable).toBe(true)

    // Load into a session, verify the ProgramInstance carries the flag.
    loadProgramAsSession(prog as unknown as ProgramNode, {
      audio_outputs: (raw as { audio_outputs: ProgramTopLevel['audio_outputs'] }).audio_outputs,
    }, session)
    const inst = session.instanceRegistry.get('voice_0')!
    expect(inst.gateable).toBe(true)
    expect(inst.gateInput).toBe(true)

    // Save back — gateable and gate_input should survive on the instance_decl.
    const { node: savedNode } = saveProgramFromSession(session)
    const savedDecl = [...instanceDecls(savedNode)].find(d => d.name === 'voice_0')!
    expect(savedDecl.gateable).toBe(true)
    expect(savedDecl.gate_input).toBe(true)

    session.graph.dispose()
  })

  test('gateable=true without gate_input is rejected at load (Phase 3)', () => {
    const session = makeSession()
    const passthrough: ProgramNode = {
      op: 'program',
      name: 'Passthrough',
      ports: { inputs: ['x'], outputs: ['y'] },
      body: { op: 'block',
        assigns: [{ op: 'outputAssign', name: 'y', expr: { op: 'input', name: 'x' } }],
      },
    }
    loadProgramAsType(passthrough, session)

    const bad: ProgramNode = {
      op: 'program',
      name: 'Patch',
      body: { op: 'block', decls: [
        // gate_input intentionally missing
        { op: 'instanceDecl', name: 'voice_0', program: 'Passthrough',
          inputs: { x: 1.0 }, gateable: true } as unknown as ExprNode,
      ]},
    }
    expect(() => loadProgramAsSession(bad, {
      audio_outputs: [{ instance: 'voice_0', output: 'y' }],
    }, session)).toThrow(/gate_input/)
    session.graph.dispose()
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
          { op: 'instanceDecl', name: 'b', program: 'CycleB' },
        ], assigns: [{ op: 'outputAssign', name: 'out', expr: 0 }] },
      }],
      ['CycleB', {
        op: 'program', name: 'CycleB',
        ports: { inputs: [], outputs: ['out'] },
        body: { op: 'block', decls: [
          { op: 'instanceDecl', name: 'a', program: 'CycleA' },
        ], assigns: [{ op: 'outputAssign', name: 'out', expr: 0 }] },
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

// ─────────────────────────────────────────────────────────────
// paramDecl — body-decl form for params/triggers (Phase A3)
// ─────────────────────────────────────────────────────────────

describe('paramDecl in body decls (Phase A3)', () => {
  test('loadProgramAsSession populates Param/Trigger registry from body paramDecls', () => {
    const session = makeTestSession()
    const prog: ProgramNode = {
      op: 'program',
      name: 'Patch',
      body: {
        op: 'block',
        decls: [
          { op: 'paramDecl', name: 'cutoff', value: 1234.0, time_const: 0.01 } as unknown as ExprNode,
          { op: 'paramDecl', name: 'gate', type: 'trigger' } as unknown as ExprNode,
        ],
      },
    }
    loadProgramAsSession(prog, {}, session)

    expect(session.paramRegistry.has('cutoff')).toBe(true)
    expect(session.triggerRegistry.has('gate')).toBe(true)
    const p = session.paramRegistry.get('cutoff')!
    expect(p.value).toBeCloseTo(1234.0)

    session.graph.dispose()
  })

  test('saveProgramFromSession emits paramDecls in body, no topLevel.params', () => {
    const session = makeTestSession()
    const prog: ProgramNode = {
      op: 'program',
      name: 'Patch',
      body: {
        op: 'block',
        decls: [
          { op: 'paramDecl', name: 'freq', value: 440.0, time_const: 0.005 } as unknown as ExprNode,
          { op: 'paramDecl', name: 'fire', type: 'trigger' } as unknown as ExprNode,
        ],
      },
    }
    loadProgramAsSession(prog, {}, session)

    const { node, topLevel } = saveProgramFromSession(session)
    expect(topLevel.params).toBeUndefined()

    const decls = (node.body.decls ?? []) as Array<Record<string, unknown>>
    const paramDeclEntries = decls.filter(d => d.op === 'paramDecl')
    expect(paramDeclEntries).toHaveLength(2)
    const byName = new Map(paramDeclEntries.map(d => [d.name as string, d]))
    expect(byName.get('freq')?.value).toBeCloseTo(440.0)
    expect(byName.get('fire')?.type).toBe('trigger')

    session.graph.dispose()
  })

  test('round-trip: save then load preserves param values and trigger names', () => {
    const session = makeTestSession()
    const prog: ProgramNode = {
      op: 'program',
      name: 'Patch',
      body: {
        op: 'block',
        decls: [
          { op: 'paramDecl', name: 'q', value: 0.7 } as unknown as ExprNode,
          { op: 'paramDecl', name: 'reset', type: 'trigger' } as unknown as ExprNode,
        ],
      },
    }
    loadProgramAsSession(prog, {}, session)
    // Mutate to verify round-trip preserves the live value.
    session.paramRegistry.get('q')!.value = 0.42

    const { node, topLevel } = saveProgramFromSession(session)

    const session2 = makeTestSession()
    loadProgramAsSession(node, topLevel, session2)
    expect(session2.paramRegistry.get('q')?.value).toBeCloseTo(0.42)
    expect(session2.triggerRegistry.has('reset')).toBe(true)

    session.graph.dispose()
    session2.graph.dispose()
  })

  test('legacy topLevel.params still loads (deprecated fallback)', () => {
    const session = makeTestSession()
    const prog: ProgramNode = {
      op: 'program',
      name: 'Patch',
      body: { op: 'block', decls: [] },
    }
    const topLevel: ProgramTopLevel = {
      params: [
        { name: 'legacy', value: 7.5, time_const: 0.02 },
        { name: 'tap', type: 'trigger' },
      ],
    }
    loadProgramAsSession(prog, topLevel, session)
    expect(session.paramRegistry.get('legacy')?.value).toBeCloseTo(7.5)
    expect(session.triggerRegistry.has('tap')).toBe(true)
    session.graph.dispose()
  })

  test('body paramDecl wins over duplicate topLevel.params entry', () => {
    const session = makeTestSession()
    const prog: ProgramNode = {
      op: 'program',
      name: 'Patch',
      body: {
        op: 'block',
        decls: [{ op: 'paramDecl', name: 'shared', value: 100.0 } as unknown as ExprNode],
      },
    }
    const topLevel: ProgramTopLevel = {
      params: [{ name: 'shared', value: 999.0 }],
    }
    loadProgramAsSession(prog, topLevel, session)
    expect(session.paramRegistry.get('shared')?.value).toBeCloseTo(100.0)
    session.graph.dispose()
  })
})

// ─────────────────────────────────────────────────────────────
// dac.out body wires — outputAssign(name='dac.out') (Phase A4)
// ─────────────────────────────────────────────────────────────

describe('dac.out body wires (Phase A4)', () => {
  test('loadProgramAsSession reads body outputAssign(name="dac.out")', () => {
    const session = makeTestSession()
    const prog: ProgramNode = {
      op: 'program',
      name: 'Patch',
      body: {
        op: 'block',
        decls: [
          { op: 'instanceDecl', name: 'osc', program: 'BlepSaw', inputs: { freq: 220.0 } } as unknown as ExprNode,
        ],
        assigns: [
          { op: 'outputAssign', name: 'dac.out',
            expr: { op: 'ref', instance: 'osc', output: 0 } } as unknown as ExprNode,
        ],
      },
    }
    loadProgramAsSession(prog, {}, session)
    expect(session.graphOutputs).toEqual([{ instance: 'osc', output: 'saw' }])
    session.graph.dispose()
  })

  test('saveProgramFromSession emits dac.out outputAssigns in body, no topLevel.audio_outputs', () => {
    const session = makeTestSession()
    const prog: ProgramNode = {
      op: 'program',
      name: 'Patch',
      body: {
        op: 'block',
        decls: [
          { op: 'instanceDecl', name: 'a', program: 'BlepSaw', inputs: { freq: 110.0 } } as unknown as ExprNode,
          { op: 'instanceDecl', name: 'b', program: 'BlepSaw', inputs: { freq: 220.0 } } as unknown as ExprNode,
        ],
        assigns: [
          { op: 'outputAssign', name: 'dac.out',
            expr: { op: 'ref', instance: 'a', output: 0 } } as unknown as ExprNode,
          { op: 'outputAssign', name: 'dac.out',
            expr: { op: 'ref', instance: 'b', output: 0 } } as unknown as ExprNode,
        ],
      },
    }
    loadProgramAsSession(prog, {}, session)

    const { node, topLevel } = saveProgramFromSession(session)
    expect(topLevel.audio_outputs).toBeUndefined()

    const assigns = (node.body.assigns ?? []) as Array<Record<string, unknown>>
    const dacAssigns = assigns.filter(a => a.op === 'outputAssign' && a.name === 'dac.out')
    expect(dacAssigns).toHaveLength(2)
    const refs = dacAssigns.map(a => a.expr as { op: string; instance: string; output: number })
    expect(refs.some(r => r.op === 'ref' && r.instance === 'a')).toBe(true)
    expect(refs.some(r => r.op === 'ref' && r.instance === 'b')).toBe(true)

    session.graph.dispose()
  })

  test('round-trip: save then load preserves dac wires', () => {
    const session = makeTestSession()
    const prog: ProgramNode = {
      op: 'program',
      name: 'Patch',
      body: {
        op: 'block',
        decls: [
          { op: 'instanceDecl', name: 'osc', program: 'BlepSaw', inputs: { freq: 440.0 } } as unknown as ExprNode,
        ],
        assigns: [
          { op: 'outputAssign', name: 'dac.out',
            expr: { op: 'ref', instance: 'osc', output: 0 } } as unknown as ExprNode,
        ],
      },
    }
    loadProgramAsSession(prog, {}, session)
    const { node, topLevel } = saveProgramFromSession(session)

    const session2 = makeTestSession()
    loadProgramAsSession(node, topLevel, session2)
    expect(session2.graphOutputs).toEqual([{ instance: 'osc', output: 'saw' }])

    session.graph.dispose()
    session2.graph.dispose()
  })

  test('legacy topLevel.audio_outputs still loads (deprecated fallback)', () => {
    const session = makeTestSession()
    const prog: ProgramNode = {
      op: 'program',
      name: 'Patch',
      body: {
        op: 'block',
        decls: [
          { op: 'instanceDecl', name: 'osc', program: 'BlepSaw', inputs: { freq: 100.0 } } as unknown as ExprNode,
        ],
      },
    }
    const topLevel: ProgramTopLevel = {
      audio_outputs: [{ instance: 'osc', output: 'saw' }],
    }
    loadProgramAsSession(prog, topLevel, session)
    expect(session.graphOutputs).toEqual([{ instance: 'osc', output: 'saw' }])
    session.graph.dispose()
  })

  test('body wires + legacy topLevel both contribute (body first, fallback after)', () => {
    const session = makeTestSession()
    const prog: ProgramNode = {
      op: 'program',
      name: 'Patch',
      body: {
        op: 'block',
        decls: [
          { op: 'instanceDecl', name: 'a', program: 'BlepSaw', inputs: { freq: 110.0 } } as unknown as ExprNode,
          { op: 'instanceDecl', name: 'b', program: 'BlepSaw', inputs: { freq: 220.0 } } as unknown as ExprNode,
        ],
        assigns: [
          { op: 'outputAssign', name: 'dac.out',
            expr: { op: 'ref', instance: 'a', output: 0 } } as unknown as ExprNode,
        ],
      },
    }
    const topLevel: ProgramTopLevel = {
      audio_outputs: [{ instance: 'b', output: 'saw' }],
    }
    loadProgramAsSession(prog, topLevel, session)
    expect(session.graphOutputs).toEqual([
      { instance: 'a', output: 'saw' },
      { instance: 'b', output: 'saw' },
    ])
    session.graph.dispose()
  })

  test('non-ref expression in dac.out outputAssign throws', () => {
    const session = makeTestSession()
    const prog: ProgramNode = {
      op: 'program',
      name: 'Patch',
      body: {
        op: 'block',
        decls: [
          { op: 'instanceDecl', name: 'osc', program: 'BlepSaw', inputs: { freq: 100.0 } } as unknown as ExprNode,
        ],
        assigns: [
          { op: 'outputAssign', name: 'dac.out', expr: 0.5 as unknown as ExprNode } as unknown as ExprNode,
        ],
      },
    }
    expect(() => loadProgramAsSession(prog, {}, session)).toThrow(/ref-shaped/)
    session.graph.dispose()
  })

  test('dac.out with unknown instance throws', () => {
    const session = makeTestSession()
    const prog: ProgramNode = {
      op: 'program',
      name: 'Patch',
      body: {
        op: 'block',
        assigns: [
          { op: 'outputAssign', name: 'dac.out',
            expr: { op: 'ref', instance: 'nope', output: 0 } } as unknown as ExprNode,
        ],
      },
    }
    expect(() => loadProgramAsSession(prog, {}, session)).toThrow(/unknown instance/)
    session.graph.dispose()
  })
})

// ─────────────────────────────────────────────────────────────
// Source-level config field removed (Phase A5)
// ─────────────────────────────────────────────────────────────

describe('source-level `config` removed (Phase A5)', () => {
  test('files with legacy `config` field still parse — Zod strips it silently', () => {
    const raw = {
      schema: 'tropical_program_2',
      name: 'OldPatch',
      body: { op: 'block', decls: [], assigns: [] },
      config: { sample_rate: 96000, buffer_length: 1024 },
    }
    const parsed = parseProgramV2(raw) as Record<string, unknown>
    expect(parsed.config).toBeUndefined()
    expect(parsed.name).toBe('OldPatch')
  })

  test('saveProgramFromSession does not emit a config field', () => {
    const session = makeTestSession()
    const prog: ProgramNode = {
      op: 'program',
      name: 'Patch',
      body: { op: 'block' },
    }
    loadProgramAsSession(prog, {}, session)
    const { topLevel } = saveProgramFromSession(session)
    expect((topLevel as Record<string, unknown>).config).toBeUndefined()
    session.graph.dispose()
  })
})
