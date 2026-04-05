/**
 * plan.test.ts — Tests for execution plan generation and validation.
 */

import { describe, test, expect } from 'bun:test'
import {
  generatePlan,
  validatePlan,
  planToJSON,
  planStats,
  PlanValidationError,
  type ExecutionPlan,
} from './plan'
import {
  compilePatch,
  type CompilerInput,
  type ModuleInfo,
} from './compiler'
import { portTypeFromString } from './compiler'
import { Float, Int, Bool, product } from './term'
import type { ExprNode } from './patch'
import type { PatchJSON } from './patch'

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────

function modInfo(
  name: string,
  opts: {
    typeName?: string
    inputs?: string[]
    outputs?: string[]
    registers?: string[]
  } = {},
): ModuleInfo {
  const inputs = opts.inputs ?? ['in']
  const outputs = opts.outputs ?? ['out']
  const registers = opts.registers ?? []
  return {
    name,
    typeName: opts.typeName ?? name,
    inputNames: inputs,
    outputNames: outputs,
    registerNames: registers,
    inputTypes: inputs.map(() => Float),
    outputTypes: outputs.map(() => Float),
    registerTypes: registers.map(() => Float),
  }
}

function compileAndPlan(input: CompilerInput, config?: { sample_rate?: number; buffer_length?: number }): ExecutionPlan {
  const patch = compilePatch(input)
  return generatePlan(patch, config)
}

// ─────────────────────────────────────────────────────────────
// Basic plan generation
// ─────────────────────────────────────────────────────────────

describe('generatePlan', () => {
  test('empty patch', () => {
    const plan = compileAndPlan({
      modules: new Map(),
      inputExprNodes: new Map(),
      graphOutputs: [],
    })
    expect(plan.schema).toBe('egress_plan_1')
    expect(plan.kernels).toEqual([])
    expect(plan.wiring).toEqual([])
    expect(plan.outputs).toEqual([])
  })

  test('single module', () => {
    const plan = compileAndPlan({
      modules: new Map([
        ['Gain1', modInfo('Gain1', { inputs: ['in'], outputs: ['out'] })],
      ]),
      inputExprNodes: new Map([['Gain1:in', 1.0]]),
      graphOutputs: [{ module: 'Gain1', output: 'out' }],
    })

    expect(plan.kernels.length).toBe(1)
    expect(plan.kernels[0].name).toBe('Gain1')
    expect(plan.kernels[0].id).toBe(0)
    expect(plan.kernels[0].group).toBe(0)
    expect(plan.kernels[0].inputs).toEqual(['in'])
    expect(plan.kernels[0].outputs).toEqual(['out'])

    expect(plan.wiring.length).toBe(1)
    expect(plan.wiring[0].kernel).toBe(0)
    expect(plan.wiring[0].input).toBe(0)
    expect(plan.wiring[0].expr).toBe(1.0)

    expect(plan.outputs.length).toBe(1)
    expect(plan.outputs[0].kernel).toBe(0)
    expect(plan.outputs[0].output).toBe(0)
  })

  test('two-module chain: kernel IDs in topological order', () => {
    const plan = compileAndPlan({
      modules: new Map([
        ['A', modInfo('A', { inputs: ['freq'], outputs: ['out'] })],
        ['B', modInfo('B', { inputs: ['in'], outputs: ['out'] })],
      ]),
      inputExprNodes: new Map<string, ExprNode>([
        ['A:freq', 440],
        ['B:in', { op: 'ref', module: 'A', output: 'out' }],
      ]),
      graphOutputs: [{ module: 'B', output: 'out' }],
    })

    expect(plan.kernels.length).toBe(2)
    // A comes first (no deps)
    expect(plan.kernels[0].name).toBe('A')
    expect(plan.kernels[0].group).toBe(0)
    // B comes second (depends on A)
    expect(plan.kernels[1].name).toBe('B')
    expect(plan.kernels[1].group).toBe(1)

    // Wiring: A.freq=440, B.in=ref(A, out)
    expect(plan.wiring.length).toBe(2)
    const bWiring = plan.wiring.find(w => w.kernel === 1)!
    expect(bWiring.input_name).toBe('in')
    expect((bWiring.expr as { op: string; module: string }).module).toBe('A')
  })

  test('parallel modules share group', () => {
    const plan = compileAndPlan({
      modules: new Map([
        ['A', modInfo('A', { inputs: ['freq'], outputs: ['out'] })],
        ['B', modInfo('B', { inputs: ['freq'], outputs: ['out'] })],
      ]),
      inputExprNodes: new Map([
        ['A:freq', 440],
        ['B:freq', 880],
      ]),
      graphOutputs: [],
    })

    expect(plan.kernels.length).toBe(2)
    expect(plan.kernels[0].group).toBe(0)
    expect(plan.kernels[1].group).toBe(0)
  })

  test('diamond: A → B,C → D', () => {
    const plan = compileAndPlan({
      modules: new Map([
        ['A', modInfo('A', { inputs: ['freq'], outputs: ['out'] })],
        ['B', modInfo('B', { inputs: ['in'], outputs: ['out'] })],
        ['C', modInfo('C', { inputs: ['in'], outputs: ['out'] })],
        ['D', modInfo('D', { inputs: ['x', 'y'], outputs: ['out'] })],
      ]),
      inputExprNodes: new Map<string, ExprNode>([
        ['A:freq', 100],
        ['B:in', { op: 'ref', module: 'A', output: 'out' }],
        ['C:in', { op: 'ref', module: 'A', output: 'out' }],
        ['D:x', { op: 'ref', module: 'B', output: 'out' }],
        ['D:y', { op: 'ref', module: 'C', output: 'out' }],
      ]),
      graphOutputs: [{ module: 'D', output: 'out' }],
    })

    expect(plan.kernels.length).toBe(4)
    // Group 0: A
    expect(plan.kernels[0].group).toBe(0)
    // Group 1: B, C (parallel)
    expect(plan.kernels[1].group).toBe(1)
    expect(plan.kernels[2].group).toBe(1)
    // Group 2: D
    expect(plan.kernels[3].group).toBe(2)
  })

  test('stateful module has registers and state_init', () => {
    const plan = compileAndPlan({
      modules: new Map([
        ['Osc1', modInfo('Osc1', {
          inputs: ['freq'],
          outputs: ['out'],
          registers: ['phase', 'state'],
        })],
      ]),
      inputExprNodes: new Map([['Osc1:freq', 440]]),
      graphOutputs: [],
    })

    expect(plan.kernels[0].registers).toEqual(['phase', 'state'])
    expect(plan.kernels[0].state_init).toEqual([0, 0])
  })

  test('config override', () => {
    const plan = compileAndPlan(
      { modules: new Map(), inputExprNodes: new Map(), graphOutputs: [] },
      { sample_rate: 48000, buffer_length: 1024 },
    )
    expect(plan.config.sample_rate).toBe(48000)
    expect(plan.config.buffer_length).toBe(1024)
  })

  test('default config', () => {
    const plan = compileAndPlan({
      modules: new Map(),
      inputExprNodes: new Map(),
      graphOutputs: [],
    })
    expect(plan.config.sample_rate).toBe(44100)
    expect(plan.config.buffer_length).toBe(512)
  })
})

// ─────────────────────────────────────────────────────────────
// Plan validation
// ─────────────────────────────────────────────────────────────

describe('validatePlan', () => {
  test('valid plan passes', () => {
    const plan = compileAndPlan({
      modules: new Map([
        ['A', modInfo('A', { inputs: ['freq'], outputs: ['out'] })],
        ['B', modInfo('B', { inputs: ['in'], outputs: ['out'] })],
      ]),
      inputExprNodes: new Map<string, ExprNode>([
        ['A:freq', 440],
        ['B:in', { op: 'ref', module: 'A', output: 'out' }],
      ]),
      graphOutputs: [{ module: 'B', output: 'out' }],
    })
    // Should not throw
    validatePlan(plan)
  })

  test('duplicate kernel name throws', () => {
    const plan: ExecutionPlan = {
      schema: 'egress_plan_1',
      config: { sample_rate: 44100, buffer_length: 512 },
      kernels: [
        { id: 0, name: 'A', module_type: 'X', group: 0, inputs: [], outputs: ['out'], registers: [], state_init: [] },
        { id: 1, name: 'A', module_type: 'X', group: 0, inputs: [], outputs: ['out'], registers: [], state_init: [] },
      ],
      wiring: [],
      outputs: [],
    }
    expect(() => validatePlan(plan)).toThrow(PlanValidationError)
  })

  test('non-sequential kernel IDs throw', () => {
    const plan: ExecutionPlan = {
      schema: 'egress_plan_1',
      config: { sample_rate: 44100, buffer_length: 512 },
      kernels: [
        { id: 0, name: 'A', module_type: 'X', group: 0, inputs: [], outputs: [], registers: [], state_init: [] },
        { id: 5, name: 'B', module_type: 'X', group: 0, inputs: [], outputs: [], registers: [], state_init: [] },
      ],
      wiring: [],
      outputs: [],
    }
    expect(() => validatePlan(plan)).toThrow(PlanValidationError)
  })

  test('wiring to unknown kernel throws', () => {
    const plan: ExecutionPlan = {
      schema: 'egress_plan_1',
      config: { sample_rate: 44100, buffer_length: 512 },
      kernels: [
        { id: 0, name: 'A', module_type: 'X', group: 0, inputs: ['in'], outputs: [], registers: [], state_init: [] },
      ],
      wiring: [
        { kernel: 99, input: 0, input_name: 'in', expr: 0 },
      ],
      outputs: [],
    }
    expect(() => validatePlan(plan)).toThrow(/unknown kernel/)
  })

  test('wiring to out-of-range input throws', () => {
    const plan: ExecutionPlan = {
      schema: 'egress_plan_1',
      config: { sample_rate: 44100, buffer_length: 512 },
      kernels: [
        { id: 0, name: 'A', module_type: 'X', group: 0, inputs: ['in'], outputs: [], registers: [], state_init: [] },
      ],
      wiring: [
        { kernel: 0, input: 5, input_name: 'bad', expr: 0 },
      ],
      outputs: [],
    }
    expect(() => validatePlan(plan)).toThrow(/invalid input/)
  })

  test('output referencing unknown kernel throws', () => {
    const plan: ExecutionPlan = {
      schema: 'egress_plan_1',
      config: { sample_rate: 44100, buffer_length: 512 },
      kernels: [],
      wiring: [],
      outputs: [{ kernel: 0, output: 0, output_name: 'out' }],
    }
    expect(() => validatePlan(plan)).toThrow(/unknown kernel/)
  })

  test('mismatched state_init length throws', () => {
    const plan: ExecutionPlan = {
      schema: 'egress_plan_1',
      config: { sample_rate: 44100, buffer_length: 512 },
      kernels: [
        { id: 0, name: 'A', module_type: 'X', group: 0, inputs: [], outputs: [], registers: ['r1', 'r2'], state_init: [0] },
      ],
      wiring: [],
      outputs: [],
    }
    expect(() => validatePlan(plan)).toThrow(/state_init length/)
  })

  test('non-monotonic group ordering throws', () => {
    const plan: ExecutionPlan = {
      schema: 'egress_plan_1',
      config: { sample_rate: 44100, buffer_length: 512 },
      kernels: [
        { id: 0, name: 'A', module_type: 'X', group: 1, inputs: [], outputs: [], registers: [], state_init: [] },
        { id: 1, name: 'B', module_type: 'X', group: 0, inputs: [], outputs: [], registers: [], state_init: [] },
      ],
      wiring: [],
      outputs: [],
    }
    expect(() => validatePlan(plan)).toThrow(/group/)
  })
})

// ─────────────────────────────────────────────────────────────
// Serialization
// ─────────────────────────────────────────────────────────────

describe('planToJSON', () => {
  test('produces valid JSON', () => {
    const plan = compileAndPlan({
      modules: new Map([
        ['A', modInfo('A', { inputs: ['freq'], outputs: ['out'] })],
      ]),
      inputExprNodes: new Map([['A:freq', 440]]),
      graphOutputs: [{ module: 'A', output: 'out' }],
    })
    const json = planToJSON(plan)
    const parsed = JSON.parse(json)
    expect(parsed.schema).toBe('egress_plan_1')
    expect(parsed.kernels.length).toBe(1)
  })

  test('round-trips through JSON', () => {
    const plan = compileAndPlan({
      modules: new Map([
        ['A', modInfo('A', { inputs: ['freq'], outputs: ['out'] })],
        ['B', modInfo('B', { inputs: ['in'], outputs: ['out'] })],
      ]),
      inputExprNodes: new Map<string, ExprNode>([
        ['A:freq', 440],
        ['B:in', { op: 'ref', module: 'A', output: 'out' }],
      ]),
      graphOutputs: [{ module: 'B', output: 'out' }],
    })
    const json = planToJSON(plan)
    const parsed = JSON.parse(json) as ExecutionPlan
    expect(parsed.kernels.length).toBe(plan.kernels.length)
    expect(parsed.wiring.length).toBe(plan.wiring.length)
    expect(parsed.outputs.length).toBe(plan.outputs.length)
    // Validate the round-tripped plan
    validatePlan(parsed)
  })
})

// ─────────────────────────────────────────────────────────────
// Plan statistics
// ─────────────────────────────────────────────────────────────

describe('planStats', () => {
  test('diamond patch stats', () => {
    const plan = compileAndPlan({
      modules: new Map([
        ['A', modInfo('A', { inputs: ['freq'], outputs: ['out'] })],
        ['B', modInfo('B', { inputs: ['in'], outputs: ['out'] })],
        ['C', modInfo('C', { inputs: ['in'], outputs: ['out'] })],
        ['D', modInfo('D', { inputs: ['x', 'y'], outputs: ['out'] })],
      ]),
      inputExprNodes: new Map<string, ExprNode>([
        ['A:freq', 100],
        ['B:in', { op: 'ref', module: 'A', output: 'out' }],
        ['C:in', { op: 'ref', module: 'A', output: 'out' }],
        ['D:x', { op: 'ref', module: 'B', output: 'out' }],
        ['D:y', { op: 'ref', module: 'C', output: 'out' }],
      ]),
      graphOutputs: [{ module: 'D', output: 'out' }],
    })
    const stats = planStats(plan)
    expect(stats.kernel_count).toBe(4)
    expect(stats.group_count).toBe(3)
    expect(stats.max_parallelism).toBe(2) // B and C in parallel
    expect(stats.output_count).toBe(1)
    expect(stats.total_inputs).toBe(5)   // freq + in + in + x + y
    expect(stats.total_outputs).toBe(4)  // 4 modules × 1 output each
  })

  test('empty patch stats', () => {
    const plan = compileAndPlan({
      modules: new Map(),
      inputExprNodes: new Map(),
      graphOutputs: [],
    })
    const stats = planStats(plan)
    expect(stats.kernel_count).toBe(0)
    expect(stats.max_parallelism).toBe(0)
  })
})

// ─────────────────────────────────────────────────────────────
// Integration: compile → plan → validate
// ─────────────────────────────────────────────────────────────

describe('end-to-end: compile → plan → validate', () => {
  test('VCO-like module chain', () => {
    const input: CompilerInput = {
      modules: new Map([
        ['VCO1', modInfo('VCO1', {
          typeName: 'VCO',
          inputs: ['freq', 'fm', 'fm_index'],
          outputs: ['saw', 'tri', 'sin', 'sqr'],
          registers: ['phase', 'tri_state'],
        })],
        ['VCA1', modInfo('VCA1', {
          typeName: 'VCA',
          inputs: ['audio', 'cv'],
          outputs: ['out'],
        })],
        ['Env1', modInfo('Env1', {
          typeName: 'ADEnvelope',
          inputs: ['gate', 'attack', 'decay'],
          outputs: ['env'],
          registers: ['state', 'level'],
        })],
        ['Clock1', modInfo('Clock1', {
          typeName: 'Clock',
          inputs: ['freq'],
          outputs: ['output'],
          registers: ['phase'],
        })],
      ]),
      inputExprNodes: new Map<string, ExprNode>([
        ['VCO1:freq', 440],
        ['VCO1:fm', 0],
        ['VCO1:fm_index', 0],
        ['Clock1:freq', 2.0],
        ['Env1:gate', { op: 'ref', module: 'Clock1', output: 'output' }],
        ['Env1:attack', 0.01],
        ['Env1:decay', 0.3],
        ['VCA1:audio', { op: 'mul', args: [{ op: 'ref', module: 'VCO1', output: 'sin' }, 0.5] }],
        ['VCA1:cv', { op: 'ref', module: 'Env1', output: 'env' }],
      ]),
      graphOutputs: [{ module: 'VCA1', output: 'out' }],
    }

    const patch = compilePatch(input)
    const plan = generatePlan(patch, { sample_rate: 48000 })

    // Validate structural integrity
    validatePlan(plan)

    // Check structure
    expect(plan.kernels.length).toBe(4)
    expect(plan.config.sample_rate).toBe(48000)

    // VCO1 and Clock1 should be in group 0 (no deps)
    const group0 = plan.kernels.filter(k => k.group === 0).map(k => k.name).sort()
    expect(group0).toEqual(['Clock1', 'VCO1'])

    // Env1 in group 1 (depends on Clock1)
    const group1 = plan.kernels.filter(k => k.group === 1).map(k => k.name)
    expect(group1).toEqual(['Env1'])

    // VCA1 in group 2 (depends on VCO1 and Env1)
    const group2 = plan.kernels.filter(k => k.group === 2).map(k => k.name)
    expect(group2).toEqual(['VCA1'])

    // Output should reference VCA1
    expect(plan.outputs.length).toBe(1)
    const vcaKernel = plan.kernels.find(k => k.name === 'VCA1')!
    expect(plan.outputs[0].kernel).toBe(vcaKernel.id)

    // Stateful kernels should have registers
    const vcoKernel = plan.kernels.find(k => k.name === 'VCO1')!
    expect(vcoKernel.registers).toEqual(['phase', 'tri_state'])
    expect(vcoKernel.state_init.length).toBe(2)

    // JSON round-trip
    const json = planToJSON(plan)
    const parsed = JSON.parse(json)
    validatePlan(parsed)

    const stats = planStats(plan)
    expect(stats.kernel_count).toBe(4)
    expect(stats.group_count).toBe(3)
    expect(stats.max_parallelism).toBe(2)
    expect(stats.total_registers).toBe(5) // VCO:2 + Env:2 + Clock:1
  })
})

// Integration tests that tested the Graph → C++ plan loading path have been
// removed. The Graph class no longer exists. End-to-end audio tests are now
// in tests/test_module_process.cpp using the FlatRuntime C API.
