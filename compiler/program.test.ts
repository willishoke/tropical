/**
 * program.test.ts — Tests for ProgramJSON schema validation.
 */

import { describe, test, expect } from 'bun:test'
import { parseProgram } from './schema'

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
