/**
 * flatten_wiring.test.ts — Tests for flatten-time wiring type validation.
 */

import { describe, test, expect } from 'bun:test'
import { normalizeWiringTypes, FlattenError, flattenSession, flattenExpressions } from './flatten'
import type { InstanceInfo } from './compiler'
import { Float, ArrayType } from './term'
import type { ExprNode } from './expr'
import { loadProgramDef } from './session'
import type { ProgramType, ProgramInstance, Bounds } from './program_types'
import type { ProgramJSON } from './program'
import { Param, Trigger } from './runtime/param'

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────

function makeInfo(
  name: string,
  inputTypes: InstanceInfo['inputTypes'],
  inputNames: string[],
  outputTypes: InstanceInfo['outputTypes'],
  outputNames: string[],
): InstanceInfo {
  return {
    name, typeName: name,
    inputNames, outputNames, registerNames: [],
    inputTypes, outputTypes, registerTypes: [],
  }
}

const ref = (instance: string, output: string): ExprNode =>
  ({ op: 'ref', instance, output })

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

describe('normalizeWiringTypes', () => {
  test('passes through compatible float→float ref', () => {
    const infos = new Map([
      ['src', makeInfo('src', [], [], [Float], ['out'])],
      ['dst', makeInfo('dst', [Float], ['in'], [], [])],
    ])
    const exprs = new Map([['dst:in', ref('src', 'out')]])
    const result = normalizeWiringTypes(infos, exprs)
    expect(result.get('dst:in')).toEqual(ref('src', 'out'))
  })

  test('passes through float literal on float input', () => {
    const infos = new Map([
      ['dst', makeInfo('dst', [Float], ['freq'], [], [])],
    ])
    const exprs = new Map<string, ExprNode>([['dst:freq', 440]])
    const result = normalizeWiringTypes(infos, exprs)
    expect(result.get('dst:freq')).toBe(440)
  })

  test('inserts broadcast_to for scalar ref into array input', () => {
    const infos = new Map([
      ['src', makeInfo('src', [], [], [Float], ['out'])],
      ['dst', makeInfo('dst', [ArrayType(Float, [4])], ['in'], [], [])],
    ])
    const exprs = new Map([['dst:in', ref('src', 'out')]])
    const result = normalizeWiringTypes(infos, exprs)
    const node = result.get('dst:in') as Record<string, unknown>
    expect(node.op).toBe('broadcast_to')
    expect(node.shape).toEqual([4])
  })

  test('inserts broadcast_to for scalar literal into array input', () => {
    const infos = new Map([
      ['dst', makeInfo('dst', [ArrayType(Float, [3])], ['gain'], [], [])],
    ])
    const exprs = new Map<string, ExprNode>([['dst:gain', 0.5]])
    const result = normalizeWiringTypes(infos, exprs)
    const node = result.get('dst:gain') as Record<string, unknown>
    expect(node.op).toBe('broadcast_to')
    expect(node.shape).toEqual([3])
  })

  test('inserts broadcast_to for [1] array ref into [4] array input', () => {
    const infos = new Map([
      ['src', makeInfo('src', [], [], [ArrayType(Float, [1])], ['out'])],
      ['dst', makeInfo('dst', [ArrayType(Float, [4])], ['in'], [], [])],
    ])
    const exprs = new Map([['dst:in', ref('src', 'out')]])
    const result = normalizeWiringTypes(infos, exprs)
    const node = result.get('dst:in') as Record<string, unknown>
    expect(node.op).toBe('broadcast_to')
    expect(node.shape).toEqual([4])
  })

  test('throws FlattenError for array[4] ref into scalar input', () => {
    const infos = new Map([
      ['src', makeInfo('src', [], [], [ArrayType(Float, [4])], ['out'])],
      ['dst', makeInfo('dst', [Float], ['in'], [], [])],
    ])
    const exprs = new Map([['dst:in', ref('src', 'out')]])
    expect(() => normalizeWiringTypes(infos, exprs)).toThrow(FlattenError)
    expect(() => normalizeWiringTypes(infos, exprs)).toThrow("'dst'.in")
  })

  test('throws FlattenError for incompatible array shapes [3] → [4]', () => {
    const infos = new Map([
      ['src', makeInfo('src', [], [], [ArrayType(Float, [3])], ['out'])],
      ['dst', makeInfo('dst', [ArrayType(Float, [4])], ['in'], [], [])],
    ])
    const exprs = new Map([['dst:in', ref('src', 'out')]])
    expect(() => normalizeWiringTypes(infos, exprs)).toThrow(FlattenError)
  })

  test('passes through compatible array[4]→array[4] ref without wrapping', () => {
    const infos = new Map([
      ['src', makeInfo('src', [], [], [ArrayType(Float, [4])], ['out'])],
      ['dst', makeInfo('dst', [ArrayType(Float, [4])], ['in'], [], [])],
    ])
    const exprs = new Map([['dst:in', ref('src', 'out')]])
    const result = normalizeWiringTypes(infos, exprs)
    expect(result.get('dst:in')).toEqual(ref('src', 'out'))
  })

  test('skips expressions whose type cannot be inferred (e.g. arithmetic)', () => {
    const infos = new Map([
      ['dst', makeInfo('dst', [Float], ['in'], [], [])],
    ])
    const arith: ExprNode = { op: 'add', args: [1, 2] }
    const exprs = new Map<string, ExprNode>([['dst:in', arith]])
    // Should not throw — arithmetic node type is unknown, so pass through
    const result = normalizeWiringTypes(infos, exprs)
    expect(result.get('dst:in')).toEqual(arith)
  })

  test('skips unknown modules (no crash on stale keys)', () => {
    const infos = new Map<string, InstanceInfo>()
    const exprs = new Map([['ghost:in', ref('src', 'out')]])
    expect(() => normalizeWiringTypes(infos, exprs)).not.toThrow()
  })

  test('handles 2D array broadcast [1,4] → [3,4]', () => {
    const infos = new Map([
      ['src', makeInfo('src', [], [], [ArrayType(Float, [1, 4])], ['out'])],
      ['dst', makeInfo('dst', [ArrayType(Float, [3, 4])], ['in'], [], [])],
    ])
    const exprs = new Map([['dst:in', ref('src', 'out')]])
    const result = normalizeWiringTypes(infos, exprs)
    const node = result.get('dst:in') as Record<string, unknown>
    expect(node.op).toBe('broadcast_to')
    expect(node.shape).toEqual([3, 4])
  })
})

// ─────────────────────────────────────────────────────────────
// Feedback cycle auto-delay
// ─────────────────────────────────────────────────────────────

describe('feedback cycle auto-delay', () => {
  /** Minimal session with no FFI. */
  function mockSession() {
    return {
      typeRegistry: new Map<string, ProgramType>(),
      typeAliasRegistry: new Map<string, { base: string; bounds: Bounds }>(),
      instanceRegistry: new Map<string, ProgramInstance>(),
      paramRegistry: new Map<string, Param>(),
      triggerRegistry: new Map<string, Trigger>(),
    }
  }

  /** Simple passthrough program: output = input. */
  function passthrough(): ProgramJSON {
    return {
      schema: 'tropical_program_1',
      name: 'Pass',
      inputs: ['in'],
      outputs: ['out'],
      input_defaults: { in: 0 },
      process: { outputs: { out: { op: 'input', name: 'in' } } },
    } as ProgramJSON
  }

  test('A → B → A feedback cycle flattens without throwing', () => {
    const session = mockSession()

    const type = loadProgramDef(passthrough(), session)
    session.typeRegistry.set('Pass', type)

    const a = type.instantiateAs('a')
    const b = type.instantiateAs('b')
    session.instanceRegistry.set('a', a)
    session.instanceRegistry.set('b', b)

    const fullSession = {
      ...session,
      bufferLength: 1,
      dac: null,
      // A.in ← B.out, B.in ← A.out — direct feedback loop, no delay
      inputExprNodes: new Map<string, ExprNode>([
        ['a:in', { op: 'ref', instance: 'b', output: 'out' }],
        ['b:in', { op: 'ref', instance: 'a', output: 'out' }],
      ]),
      graphOutputs: [{ instance: 'a', output: 'out' }],
      runtime: null as any,
      graph: null as any,
      _nameCounters: new Map<string, number>(),
    }

    // Should NOT throw — the flattener should auto-insert a one-sample delay
    // to break the cycle, just like hardware propagation delay.
    const plan = flattenSession(fullSession)
    expect(plan.instructions.length).toBeGreaterThan(0)
  })

  test('self-feedback (A.out → A.in) flattens without throwing', () => {
    const session = mockSession()

    const type = loadProgramDef(passthrough(), session)
    session.typeRegistry.set('Pass', type)

    const a = type.instantiateAs('a')
    session.instanceRegistry.set('a', a)

    const fullSession = {
      ...session,
      bufferLength: 1,
      dac: null,
      inputExprNodes: new Map<string, ExprNode>([
        ['a:in', { op: 'ref', instance: 'a', output: 'out' }],
      ]),
      graphOutputs: [{ instance: 'a', output: 'out' }],
      runtime: null as any,
      graph: null as any,
      _nameCounters: new Map<string, number>(),
    }

    const plan = flattenSession(fullSession)
    expect(plan.instructions.length).toBeGreaterThan(0)
  })
})

// ─────────────────────────────────────────────────────────────
// Gateable subgraphs (Phase 4)
// ─────────────────────────────────────────────────────────────

describe('gateable instances wrap outputs in source_tag', () => {
  function mockSession() {
    return {
      typeRegistry: new Map<string, ProgramType>(),
      typeAliasRegistry: new Map<string, { base: string; bounds: Bounds }>(),
      instanceRegistry: new Map<string, ProgramInstance>(),
      paramRegistry: new Map<string, Param>(),
      triggerRegistry: new Map<string, Trigger>(),
    }
  }

  // Passthrough with a one-register state to exercise both output wrapping
  // and register-update wrapping (the latter uses on_skip for hold-on-skip).
  function passthroughWithState(): ProgramJSON {
    return {
      schema: 'tropical_program_1',
      name: 'Pass',
      inputs: ['in'],
      outputs: ['out'],
      regs: { last: 0 },
      input_defaults: { in: 0 },
      process: {
        outputs: { out: { op: 'input', name: 'in' } },
        next_regs: { last: { op: 'input', name: 'in' } },
      },
    } as ProgramJSON
  }

  test('gateable instance emits source_tag on outputs and register updates', () => {
    const session = mockSession()

    const type = loadProgramDef(passthroughWithState(), session)
    session.typeRegistry.set('Pass', type)

    const voice = type.instantiateAs('voice_0')
    voice.gateable = true
    voice.gateInput = true  // always-on gate (constant true)
    session.instanceRegistry.set('voice_0', voice)

    const fullSession = {
      ...session,
      bufferLength: 1,
      dac: null,
      inputExprNodes: new Map<string, ExprNode>([
        ['voice_0:in', 1.0 as ExprNode],
      ]),
      graphOutputs: [{ instance: 'voice_0', output: 'out' }],
      runtime: null as unknown as import('./runtime/runtime').Runtime,
      graph: null as unknown as ReturnType<typeof import('./session').makeSession>['graph'],
      _nameCounters: new Map<string, number>(),
    }

    const flat = flattenExpressions(fullSession)

    // Output expression is wrapped in source_tag. An outer safety clamp
    // (added for graph outputs) sits around it — unwrap once.
    expect(flat.outputExprs.length).toBe(1)
    const outer = flat.outputExprs[0] as { op: string; args: unknown[] }
    expect(outer.op).toBe('clamp')  // safety clamp
    const tag = outer.args[0] as { op: string; source_instance: string; gate_expr: unknown }
    expect(tag.op).toBe('source_tag')
    expect(tag.source_instance).toBe('voice_0')
    expect(tag.gate_expr).toBe(true)

    // Register update is also wrapped, with on_skip = reg(id). No safety
    // clamp on the register path.
    expect(flat.registerExprs.length).toBe(1)
    const reg = flat.registerExprs[0] as { op: string; on_skip: { op: string; id: number } }
    expect(reg.op).toBe('source_tag')
    expect(reg.on_skip.op).toBe('reg')
    expect(reg.on_skip.id).toBe(0)  // the only register's flat id
  })

  test('gateable=true without gate_input rejected at flatten time', () => {
    const session = mockSession()

    const type = loadProgramDef(passthroughWithState(), session)
    session.typeRegistry.set('Pass', type)

    const voice = type.instantiateAs('voice_0')
    voice.gateable = true
    // gateInput left undefined on purpose
    session.instanceRegistry.set('voice_0', voice)

    const fullSession = {
      ...session,
      bufferLength: 1,
      dac: null,
      inputExprNodes: new Map<string, ExprNode>(),
      graphOutputs: [{ instance: 'voice_0', output: 'out' }],
      runtime: null as unknown as import('./runtime/runtime').Runtime,
      graph: null as unknown as ReturnType<typeof import('./session').makeSession>['graph'],
      _nameCounters: new Map<string, number>(),
    }

    expect(() => flattenExpressions(fullSession)).toThrow(FlattenError)
  })

  test('non-gateable instances emit outputs without source_tag wrapping', () => {
    const session = mockSession()

    const type = loadProgramDef(passthroughWithState(), session)
    session.typeRegistry.set('Pass', type)

    const voice = type.instantiateAs('voice_0')
    // gateable left false
    session.instanceRegistry.set('voice_0', voice)

    const fullSession = {
      ...session,
      bufferLength: 1,
      dac: null,
      inputExprNodes: new Map<string, ExprNode>([
        ['voice_0:in', 1.0 as ExprNode],
      ]),
      graphOutputs: [{ instance: 'voice_0', output: 'out' }],
      runtime: null as unknown as import('./runtime/runtime').Runtime,
      graph: null as unknown as ReturnType<typeof import('./session').makeSession>['graph'],
      _nameCounters: new Map<string, number>(),
    }

    const flat = flattenExpressions(fullSession)
    const out = flat.outputExprs[0] as { op: string }
    expect(out.op).not.toBe('source_tag')
  })
})
