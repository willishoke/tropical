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

describe('loadProgramAsType — stdlib equivalence', () => {
  /**
   * Load a stdlib JSON file as a ModuleType, verify it loads and flattens
   * without error, and has the same structure (register count, output count,
   * state init) as the TypeScript version.
   *
   * Instruction-level identity isn't guaranteed because JSON-parsed trees
   * lack object-identity sharing across outputs/registers (TypeScript
   * shares via local variables). The plans are semantically equivalent
   * but may differ in instruction count due to redundant sub-expressions.
   */
  function compareStdlib(
    typeName: string,
    jsonFile: string,
    wiring: Record<string, number | boolean>,
    outputName: string,
  ) {
    // TypeScript-defined version
    const tsSession = makeSession()
    loadBuiltins(tsSession.typeRegistry)
    const tsType = tsSession.typeRegistry.get(typeName)!
    tsSession.instanceRegistry.set('inst', tsType.instantiateAs('inst'))
    for (const [k, v] of Object.entries(wiring)) {
      tsSession.inputExprNodes.set(`inst:${k}`, v)
    }
    tsSession.graphOutputs.push({ module: 'inst', output: outputName })
    const tsPlan = flattenPatch(tsSession)

    // JSON-defined version
    const jsonSession = makeSession()
    loadBuiltins(jsonSession.typeRegistry)
    const prog = JSON.parse(
      readFileSync(join(__dirname, `../stdlib/${jsonFile}`), 'utf-8')
    ) as ProgramJSON
    const jsonType = loadProgramAsType(prog, jsonSession)
    jsonSession.typeRegistry.set(typeName, jsonType) // override the TS version
    jsonSession.instanceRegistry.set('inst', jsonType.instantiateAs('inst'))
    for (const [k, v] of Object.entries(wiring)) {
      jsonSession.inputExprNodes.set(`inst:${k}`, v)
    }
    jsonSession.graphOutputs.push({ module: 'inst', output: outputName })
    const jsonPlan = flattenPatch(jsonSession)

    // Structural equivalence: same ports, same state shape
    expect(jsonType._def.inputNames).toEqual(tsType._def.inputNames)
    expect(jsonType._def.outputNames).toEqual(tsType._def.outputNames)
    expect(jsonType._def.registerNames).toEqual(tsType._def.registerNames)
    expect(jsonType._def.registerInitValues).toEqual(tsType._def.registerInitValues)
    expect(jsonType._def.delayInitValues).toEqual(tsType._def.delayInitValues)

    // Plan compiles and has correct output count
    expect(jsonPlan.output_targets.length).toBe(tsPlan.output_targets.length)
    expect(jsonPlan.schema).toBe('tropical_plan_4')

    // For modules without cross-output sharing, instructions should match exactly
    // (VCA, Clock have no shared sub-expressions across outputs/registers)
  }

  /** Strict comparison: plans must be instruction-identical. */
  function compareStdlibStrict(
    typeName: string,
    jsonFile: string,
    wiring: Record<string, number | boolean>,
    outputName: string,
  ) {
    const tsSession = makeSession()
    loadBuiltins(tsSession.typeRegistry)
    const tsType = tsSession.typeRegistry.get(typeName)!
    tsSession.instanceRegistry.set('inst', tsType.instantiateAs('inst'))
    for (const [k, v] of Object.entries(wiring)) {
      tsSession.inputExprNodes.set(`inst:${k}`, v)
    }
    tsSession.graphOutputs.push({ module: 'inst', output: outputName })
    const tsPlan = flattenPatch(tsSession)

    const jsonSession = makeSession()
    loadBuiltins(jsonSession.typeRegistry)
    const prog = JSON.parse(
      readFileSync(join(__dirname, `../stdlib/${jsonFile}`), 'utf-8')
    ) as ProgramJSON
    const jsonType = loadProgramAsType(prog, jsonSession)
    jsonSession.typeRegistry.set(typeName, jsonType)
    jsonSession.instanceRegistry.set('inst', jsonType.instantiateAs('inst'))
    for (const [k, v] of Object.entries(wiring)) {
      jsonSession.inputExprNodes.set(`inst:${k}`, v)
    }
    jsonSession.graphOutputs.push({ module: 'inst', output: outputName })
    const jsonPlan = flattenPatch(jsonSession)

    expect(jsonPlan.instructions).toEqual(tsPlan.instructions)
    expect(jsonPlan.register_count).toEqual(tsPlan.register_count)
    expect(jsonPlan.output_targets).toEqual(tsPlan.output_targets)
  }

  test('VCA.json matches TypeScript VCA (strict)', () => {
    compareStdlibStrict('VCA', 'VCA.json', { audio: 1.0, cv: 0.5 }, 'out')
  })

  test('Clock.json matches TypeScript Clock (strict)', () => {
    compareStdlibStrict('Clock', 'Clock.json', { freq: 2.0 }, 'output')
  })

  test('BitCrusher.json matches TypeScript BitCrusher', () => {
    compareStdlib('BitCrusher', 'BitCrusher.json', { audio: 0.5, bit_depth: 8.0, sample_rate_hz: 22050.0 }, 'output')
  })

  test('NoiseLFSR.json matches TypeScript NoiseLFSR', () => {
    compareStdlib('NoiseLFSR', 'NoiseLFSR.json', { clock: 1.0 }, 'out')
  })

  test('Delay8.json matches TypeScript Delay8 (strict)', () => {
    compareStdlibStrict('Delay8', 'Delay8.json', { x: 1.0 }, 'y')
  })

  test('VCO.json matches TypeScript VCO (saw)', () => {
    compareStdlib('VCO', 'VCO.json', { freq: 440.0, fm: 0.0, fm_index: 5.0 }, 'saw')
  })

  test('VCO.json matches TypeScript VCO (sin)', () => {
    compareStdlib('VCO', 'VCO.json', { freq: 440.0, fm: 0.0, fm_index: 5.0 }, 'sin')
  })

  test('VCO.json matches TypeScript VCO (sqr)', () => {
    compareStdlib('VCO', 'VCO.json', { freq: 440.0, fm: 0.0, fm_index: 5.0 }, 'sqr')
  })

  test('VCO.json matches TypeScript VCO (tri)', () => {
    compareStdlib('VCO', 'VCO.json', { freq: 440.0, fm: 0.0, fm_index: 5.0 }, 'tri')
  })

  test('Compressor.json matches TypeScript Compressor', () => {
    compareStdlib('Compressor', 'Compressor.json', { input: 1.0, sidechain: 0.5, threshold: -12.0, ratio: 4.0, attack_ms: 10.0, release_ms: 100.0, makeup: 1.0 }, 'output')
  })

  test('BassDrum.json matches TypeScript BassDrum', () => {
    compareStdlib('BassDrum', 'BassDrum.json', { gate: 1.0, freq: 60.0, punch: 0.5, decay: 0.35, tone: 8.0 }, 'output')
  })

  test('ADEnvelope.json matches TypeScript ADEnvelope', () => {
    compareStdlib('ADEnvelope', 'ADEnvelope.json', { gate: 1.0, attack: 0.01, decay: 0.3 }, 'env')
  })

  test('ADSREnvelope.json matches TypeScript ADSREnvelope', () => {
    compareStdlib('ADSREnvelope', 'ADSREnvelope.json', { gate: 1.0, attack: 0.01, decay: 0.1, sustain: 0.7, release: 0.3 }, 'env')
  })

  test('LadderFilter.json matches TypeScript LadderFilter', () => {
    compareStdlib('LadderFilter', 'LadderFilter.json', { input: 1.0, cutoff: 1000.0, resonance: 0.5, drive: 1.0 }, 'lp')
  })

  test('Phaser.json matches TypeScript Phaser', () => {
    compareStdlib('Phaser', 'Phaser.json', { input: 1.0, feedback: 0.4, lfo_speed: 0.2 }, 'output')
  })

  test('Phaser16.json matches TypeScript Phaser16', () => {
    compareStdlib('Phaser16', 'Phaser16.json', { input: 1.0, feedback: 0.4, lfo_speed: 0.2 }, 'output')
  })

  test('Reverb.json matches TypeScript Reverb', () => {
    compareStdlib('Reverb', 'Reverb.json', { input: 1.0, mix: 0.35, decay: 0.84, damp: 0.4 }, 'output')
  })

  test('Delay16.json matches TypeScript Delay16 (strict)', () => {
    compareStdlibStrict('Delay16', 'Delay16.json', { x: 1.0 }, 'y')
  })

  test('Delay512.json matches TypeScript Delay512 (strict)', () => {
    compareStdlibStrict('Delay512', 'Delay512.json', { x: 1.0 }, 'y')
  })

  test('Delay4410.json matches TypeScript Delay4410 (strict)', () => {
    compareStdlibStrict('Delay4410', 'Delay4410.json', { x: 1.0 }, 'y')
  })

  test('Delay44100.json matches TypeScript Delay44100 (strict)', () => {
    compareStdlibStrict('Delay44100', 'Delay44100.json', { x: 1.0 }, 'y')
  })

  test('TopoWaveguide.json matches TypeScript TopoWaveguide', () => {
    compareStdlib('TopoWaveguide', 'TopoWaveguide.json', {
      g: 0.035, decay: 0.9997, brightness: 0.88,
    }, 'out')
  })
})
