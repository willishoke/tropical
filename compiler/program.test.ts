/**
 * program.test.ts — Tests for unified ProgramJSON schema and conversions.
 */

import { describe, test, expect } from 'bun:test'
import { readFileSync } from 'fs'
import { join } from 'path'
import {
  convertPatchToProgram,
  convertModuleDefToProgram,
  convertProgramToPatch,
  loadProgramAsType,
  type ProgramJSON,
} from './program'
import type { PatchJSON, ModuleDefJSON, SessionState } from './patch'
import { makeSession } from './patch'
import { parseProgram } from './schema'
import { loadBuiltins, vca } from './module_library'
import { flattenPatch } from './flatten'

// ─────────────────────────────────────────────────────────────
// Conversion: PatchJSON → ProgramJSON
// ─────────────────────────────────────────────────────────────

describe('convertPatchToProgram', () => {
  test('converts minimal patch', () => {
    const patch: PatchJSON = {
      schema: 'tropical_patch_1',
      modules: [{ type: 'VCO', name: 'VCO1' }],
      outputs: [{ module: 'VCO1', output: 'sin' }],
      input_exprs: [{ module: 'VCO1', input: 'freq', expr: 440 }],
    }
    const prog = convertPatchToProgram(patch)
    expect(prog.schema).toBe('tropical_program_1')
    expect(prog.instances).toEqual({
      VCO1: { program: 'VCO', inputs: { freq: 440 } },
    })
    expect(prog.audio_outputs).toEqual([
      { instance: 'VCO1', output: 'sin' },
    ])
  })

  test('converts connections to instance inputs', () => {
    const patch: PatchJSON = {
      schema: 'tropical_patch_1',
      modules: [
        { type: 'VCO', name: 'VCO1' },
        { type: 'VCA', name: 'VCA1' },
      ],
      connections: [
        { src: 'VCO1', src_output: 'sin', dst: 'VCA1', dst_input: 'audio' },
      ],
      outputs: [{ module: 'VCA1', output: 'out' }],
    }
    const prog = convertPatchToProgram(patch)
    expect(prog.instances!['VCA1'].inputs).toEqual({
      audio: { op: 'ref', module: 'VCO1', output: 'sin' },
    })
  })

  test('merges connections and input_exprs for same instance', () => {
    const patch: PatchJSON = {
      schema: 'tropical_patch_1',
      modules: [
        { type: 'VCO', name: 'VCO1' },
        { type: 'VCA', name: 'VCA1' },
      ],
      connections: [
        { src: 'VCO1', src_output: 'sin', dst: 'VCA1', dst_input: 'audio' },
      ],
      input_exprs: [
        { module: 'VCA1', input: 'cv', expr: 0.5 },
        { module: 'VCO1', input: 'freq', expr: 220 },
      ],
      outputs: [{ module: 'VCA1', output: 'out' }],
    }
    const prog = convertPatchToProgram(patch)
    expect(prog.instances!['VCA1'].inputs).toEqual({
      audio: { op: 'ref', module: 'VCO1', output: 'sin' },
      cv: 0.5,
    })
    expect(prog.instances!['VCO1'].inputs).toEqual({ freq: 220 })
  })

  test('converts module_defs to programs', () => {
    const patch: PatchJSON = {
      schema: 'tropical_patch_1',
      module_defs: [{
        name: 'MyOsc',
        inputs: ['freq'],
        outputs: ['out'],
        process: { outputs: { out: { op: 'input', name: 'freq' } } },
      }],
      modules: [{ type: 'MyOsc', name: 'Osc1' }],
      outputs: [{ module: 'Osc1', output: 'out' }],
    }
    const prog = convertPatchToProgram(patch)
    expect(prog.programs).toBeDefined()
    expect(prog.programs!['MyOsc']).toBeDefined()
    expect(prog.programs!['MyOsc'].name).toBe('MyOsc')
    expect(prog.programs!['MyOsc'].inputs).toEqual(['freq'])
  })

  test('preserves params', () => {
    const patch: PatchJSON = {
      schema: 'tropical_patch_1',
      modules: [],
      params: [{ name: 'cutoff', value: 1000, time_const: 0.01 }],
    }
    const prog = convertPatchToProgram(patch)
    expect(prog.params).toEqual([{ name: 'cutoff', value: 1000, time_const: 0.01 }])
  })
})

// ─────────────────────────────────────────────────────────────
// Conversion: ModuleDefJSON → ProgramJSON
// ─────────────────────────────────────────────────────────────

describe('convertModuleDefToProgram', () => {
  test('converts basic module', () => {
    const def: ModuleDefJSON = {
      name: 'Gain',
      inputs: ['input', 'gain'],
      outputs: ['out'],
      process: {
        outputs: {
          out: { op: 'mul', args: [{ op: 'input', name: 'input' }, { op: 'input', name: 'gain' }] },
        },
      },
    }
    const prog = convertModuleDefToProgram(def)
    expect(prog.schema).toBe('tropical_program_1')
    expect(prog.name).toBe('Gain')
    expect(prog.inputs).toEqual(['input', 'gain'])
    expect(prog.outputs).toEqual(['out'])
    expect(prog.process).toBeDefined()
  })

  test('converts module with regs and delays', () => {
    const def: ModuleDefJSON = {
      name: 'Counter',
      inputs: ['clock'],
      outputs: ['count'],
      regs: { n: 0 },
      delays: { prev: { update: { op: 'input', name: 'clock' }, init: 0 } },
      process: {
        outputs: { count: { op: 'reg', name: 'n' } },
        next_regs: { n: { op: 'add', args: [{ op: 'reg', name: 'n' }, 1] } },
      },
    }
    const prog = convertModuleDefToProgram(def)
    expect(prog.regs).toEqual({ n: 0 })
    expect(prog.delays).toBeDefined()
  })

  test('converts nested modules to instances', () => {
    const def: ModuleDefJSON = {
      name: 'WithNested',
      inputs: ['x'],
      outputs: ['y'],
      nested: {
        sub: { type: 'VCO', inputs: { freq: { op: 'input', name: 'x' } } },
      },
      process: {
        outputs: { y: { op: 'nested_out', ref: 'sub', output: 'sin' } },
      },
    }
    const prog = convertModuleDefToProgram(def)
    expect(prog.instances).toEqual({
      sub: { program: 'VCO', inputs: { freq: { op: 'input', name: 'x' } } },
    })
  })
})

// ─────────────────────────────────────────────────────────────
// Round-trip: PatchJSON → ProgramJSON → PatchJSON
// ─────────────────────────────────────────────────────────────

describe('round-trip conversion', () => {
  test('patch → program → patch preserves structure', () => {
    const original: PatchJSON = {
      schema: 'tropical_patch_1',
      modules: [
        { type: 'VCO', name: 'VCO1' },
        { type: 'VCA', name: 'VCA1' },
      ],
      input_exprs: [
        { module: 'VCO1', input: 'freq', expr: 440 },
        { module: 'VCA1', input: 'audio', expr: { op: 'ref', module: 'VCO1', output: 'sin' } },
        { module: 'VCA1', input: 'cv', expr: 0.5 },
      ],
      outputs: [{ module: 'VCA1', output: 'out' }],
      params: [{ name: 'vol', value: 0.8 }],
    }

    const prog = convertPatchToProgram(original)
    const roundTripped = convertProgramToPatch(prog)

    expect(roundTripped.schema).toBe('tropical_patch_1')
    expect(roundTripped.modules).toHaveLength(2)
    expect(roundTripped.modules.map(m => m.name).sort()).toEqual(['VCA1', 'VCO1'])
    expect(roundTripped.outputs).toEqual([{ module: 'VCA1', output: 'out' }])
    expect(roundTripped.params).toEqual([{ name: 'vol', value: 0.8 }])
    // input_exprs should contain all wiring
    expect(roundTripped.input_exprs).toHaveLength(3)
  })
})

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
// stdlib: ProgramJSON produces identical plans to TypeScript defs
// ─────────────────────────────────────────────────────────────

describe('loadProgramAsType', () => {
  function makePlan(session: SessionState) {
    return flattenPatch(session)
  }

  test('VCA.json produces identical plan to TypeScript VCA', () => {
    // TypeScript-defined VCA
    const tsSession = makeSession()
    loadBuiltins(tsSession.typeRegistry)
    const tsVca = tsSession.typeRegistry.get('VCA')!
    tsVca.instantiateAs('VCA1').name // instantiate
    tsSession.instanceRegistry.set('VCA1', tsVca.instantiateAs('VCA1'))
    tsSession.inputExprNodes.set('VCA1:audio', 1.0)
    tsSession.inputExprNodes.set('VCA1:cv', 0.5)
    tsSession.graphOutputs.push({ module: 'VCA1', output: 'out' })
    const tsPlan = makePlan(tsSession)

    // JSON-defined VCA
    const jsonSession = makeSession()
    loadBuiltins(jsonSession.typeRegistry) // load other builtins in case of deps
    const vcaJson = JSON.parse(
      readFileSync(join(__dirname, '../stdlib/VCA.json'), 'utf-8')
    ) as ProgramJSON
    const jsonVca = loadProgramAsType(vcaJson, jsonSession)
    jsonSession.typeRegistry.set('VCA_JSON', jsonVca)
    jsonSession.instanceRegistry.set('VCA1', jsonVca.instantiateAs('VCA1'))
    jsonSession.inputExprNodes.set('VCA1:audio', 1.0)
    jsonSession.inputExprNodes.set('VCA1:cv', 0.5)
    jsonSession.graphOutputs.push({ module: 'VCA1', output: 'out' })
    const jsonPlan = makePlan(jsonSession)

    // Compare the plans — instructions should be identical
    expect(jsonPlan.instructions).toEqual(tsPlan.instructions)
    expect(jsonPlan.register_count).toEqual(tsPlan.register_count)
    expect(jsonPlan.output_targets).toEqual(tsPlan.output_targets)
  })
})
