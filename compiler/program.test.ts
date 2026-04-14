/**
 * program.test.ts — Tests for ProgramJSON schema validation and export.
 */

import { describe, test, expect } from 'bun:test'
import { parseProgram } from './schema'
import { makeSession, loadJSON } from './session'
import {
  loadStdlib as loadBuiltins, loadProgramAsType,
  exportSessionAsProgram, type ProgramJSON,
} from './program'
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

describe('exportSessionAsProgram', () => {
  test('exports a simple VCO→VCA chain', () => {
    const session = makeTestSession()

    // Build a session: VCO → VCA → output
    const vcoType = session.typeRegistry.get('VCO')!
    const vcaType = session.typeRegistry.get('VCA')!
    session.instanceRegistry.set('osc', vcoType.instantiateAs('osc'))
    session.instanceRegistry.set('amp', vcaType.instantiateAs('amp'))
    session.inputExprNodes.set('osc:freq', 440)
    session.inputExprNodes.set('amp:audio', { op: 'ref', instance: 'osc', output: 'saw' })
    session.inputExprNodes.set('amp:cv', 0.5)

    const prog = exportSessionAsProgram(session, {
      name: 'SimpleVoice',
      inputs: {
        freq: 'osc:freq',
        volume: 'amp:cv',
      },
      outputs: {
        out: { instance: 'amp', output: 'out' },
      },
    })

    expect(prog.schema).toBe('tropical_program_1')
    expect(prog.name).toBe('SimpleVoice')
    expect(prog.inputs).toEqual(['freq', 'volume'])
    expect(prog.outputs).toEqual(['out'])
    // osc.freq should be rewritten to {op:"input", name:"freq"}
    expect(prog.instances!['osc'].inputs!['freq']).toEqual({ op: 'input', name: 'freq' })
    // amp.cv should be rewritten to {op:"input", name:"volume"} (port name stays 'cv')
    expect(prog.instances!['amp'].inputs!['cv']).toEqual({ op: 'input', name: 'volume' })
    // amp.audio should be a nested_out ref (internal instance, not a session-level ref)
    expect(prog.instances!['amp'].inputs!['audio']).toEqual({ op: 'nested_out', ref: 'osc', output: 'saw' })
    // Current wiring becomes input_defaults
    expect(prog.input_defaults!['freq']).toBe(440)
    expect(prog.input_defaults!['volume']).toBe(0.5)
    // No audio_outputs — this is a composite, not a top-level graph
    expect(prog.audio_outputs).toBeUndefined()
  })

  test('only includes reachable instances', () => {
    const session = makeTestSession()

    const vcoType = session.typeRegistry.get('VCO')!
    const vcaType = session.typeRegistry.get('VCA')!
    const clockType = session.typeRegistry.get('Clock')!
    session.instanceRegistry.set('osc', vcoType.instantiateAs('osc'))
    session.instanceRegistry.set('amp', vcaType.instantiateAs('amp'))
    session.instanceRegistry.set('clk', clockType.instantiateAs('clk')) // not connected to output chain
    session.inputExprNodes.set('osc:freq', 440)
    session.inputExprNodes.set('amp:audio', { op: 'ref', instance: 'osc', output: 'saw' })
    session.inputExprNodes.set('clk:freq', 2)

    const prog = exportSessionAsProgram(session, {
      name: 'Voice',
      inputs: { freq: 'osc:freq' },
      outputs: { out: { instance: 'amp', output: 'out' } },
    })

    // clk should not appear — it's unreachable from the output
    expect(prog.instances!['clk']).toBeUndefined()
    expect(Object.keys(prog.instances!).sort()).toEqual(['amp', 'osc'])
  })

  test('errors on dangling reference outside subgraph', () => {
    const session = makeTestSession()

    const vcoType = session.typeRegistry.get('VCO')!
    const vcaType = session.typeRegistry.get('VCA')!
    const clockType = session.typeRegistry.get('Clock')!
    session.instanceRegistry.set('osc', vcoType.instantiateAs('osc'))
    session.instanceRegistry.set('amp', vcaType.instantiateAs('amp'))
    session.instanceRegistry.set('lfo', vcoType.instantiateAs('lfo'))
    // osc.freq references lfo, which is NOT reachable from amp.out
    // but IS referenced by osc (which IS reachable)
    session.inputExprNodes.set('osc:freq', {
      op: 'mul', args: [{ op: 'ref', instance: 'lfo', output: 'sin' }, 100],
    })
    session.inputExprNodes.set('amp:audio', { op: 'ref', instance: 'osc', output: 'saw' })

    // lfo IS reachable through osc's wiring, so this should succeed
    const prog = exportSessionAsProgram(session, {
      name: 'Voice',
      inputs: {},
      outputs: { out: { instance: 'amp', output: 'out' } },
    })
    expect(prog.instances!['lfo']).toBeDefined()
  })

  test('errors on unknown instance in outputs', () => {
    const session = makeTestSession()
    expect(() => exportSessionAsProgram(session, {
      name: 'Bad',
      inputs: {},
      outputs: { out: { instance: 'nope', output: 'out' } },
    })).toThrow("unknown instance 'nope'")
  })

  test('errors on unknown port in inputs', () => {
    const session = makeTestSession()
    const vcoType = session.typeRegistry.get('VCO')!
    session.instanceRegistry.set('osc', vcoType.instantiateAs('osc'))
    expect(() => exportSessionAsProgram(session, {
      name: 'Bad',
      inputs: { x: 'osc:nope' },
      outputs: { out: { instance: 'osc', output: 'saw' } },
    })).toThrow("has no input 'nope'")
  })

  test('errors on malformed input target', () => {
    const session = makeTestSession()
    expect(() => exportSessionAsProgram(session, {
      name: 'Bad',
      inputs: { x: 'nocolon' },
      outputs: {},
    })).toThrow('must be "instance:port"')
  })

  test('exported program can be registered and instantiated', () => {
    const session = makeTestSession()

    const vcoType = session.typeRegistry.get('VCO')!
    const vcaType = session.typeRegistry.get('VCA')!
    session.instanceRegistry.set('osc', vcoType.instantiateAs('osc'))
    session.instanceRegistry.set('amp', vcaType.instantiateAs('amp'))
    session.inputExprNodes.set('osc:freq', 440)
    session.inputExprNodes.set('amp:audio', { op: 'ref', instance: 'osc', output: 'saw' })
    session.inputExprNodes.set('amp:cv', 0.8)

    const prog = exportSessionAsProgram(session, {
      name: 'TestVoice',
      inputs: { freq: 'osc:freq', vol: 'amp:cv' },
      outputs: { out: { instance: 'amp', output: 'out' } },
    })

    // Register as a type and instantiate
    const type = loadProgramAsType(prog, session)
    session.typeRegistry.set(prog.name, type)

    expect(type.name).toBe('TestVoice')
    expect(type._def.inputNames).toEqual(['freq', 'vol'])
    expect(type._def.outputNames).toEqual(['out'])

    const inst = type.instantiateAs('voice1')
    expect(inst.inputNames).toEqual(['freq', 'vol'])
    expect(inst.outputNames).toEqual(['out'])
  })

  test('round-trip: export → load → flatten', () => {
    const session = makeTestSession()

    // Build a VCO → VCA session
    const vcoType = session.typeRegistry.get('VCO')!
    const vcaType = session.typeRegistry.get('VCA')!
    session.instanceRegistry.set('osc', vcoType.instantiateAs('osc'))
    session.instanceRegistry.set('amp', vcaType.instantiateAs('amp'))
    session.inputExprNodes.set('osc:freq', 440)
    session.inputExprNodes.set('amp:audio', { op: 'ref', instance: 'osc', output: 'saw' })
    session.inputExprNodes.set('amp:cv', 0.8)

    // Export it
    const prog = exportSessionAsProgram(session, {
      name: 'RoundTripVoice',
      inputs: { freq: 'osc:freq' },
      outputs: { out: { instance: 'amp', output: 'out' } },
    })

    // Load it into a fresh session as a type, then use it
    const session2 = makeTestSession()
    const type = loadProgramAsType(prog, session2)
    session2.typeRegistry.set(prog.name, type)
    const inst = type.instantiateAs('v1')
    session2.instanceRegistry.set('v1', inst)
    session2.inputExprNodes.set('v1:freq', 880)
    session2.graphOutputs.push({ instance: 'v1', output: 'out' })

    // This should flatten without errors — validates the full pipeline
    const { flattenSession } = require('./flatten')
    const plan = flattenSession(session2)
    expect(plan).toBeDefined()
    expect(plan.outputs.length).toBeGreaterThan(0)
  })
})
