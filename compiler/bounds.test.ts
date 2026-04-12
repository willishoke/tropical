/**
 * bounds.test.ts — Tests for bounded types and automatic clamp enforcement.
 */

import { describe, test, expect } from 'bun:test'
import { applyBounds, flattenSession } from './flatten'
import { loadProgramDef, resolveBounds, resolveBaseType, BOUNDED_TYPE_ALIASES } from './session'
import type { ExprNode } from './expr'
import type { ProgramJSON } from './program'
import type { ProgramType, ProgramInstance, Bounds } from './program_types'
import { Param, Trigger } from './runtime/param'

// ─────────────────────────────────────────────────────────────
// Helpers
// ────────���────────────────────────────────────────────────────

/** Minimal session for loadProgramDef (no FFI). */
function mockSession() {
  return {
    typeRegistry: new Map<string, ProgramType>(),
    instanceRegistry: new Map<string, ProgramInstance>(),
    paramRegistry: new Map<string, Param>(),
    triggerRegistry: new Map<string, Trigger>(),
  }
}

/** Leaf program that computes output = input * 2. */
function leafProgram(overrides: Partial<ProgramJSON> = {}): ProgramJSON {
  return {
    schema: 'tropical_program_1',
    name: 'TestLeaf',
    inputs: [{ name: 'x', type: 'float' }],
    outputs: [{ name: 'out', type: 'float' }],
    input_defaults: { x: 0 },
    process: {
      outputs: { out: { op: 'mul', args: [{ op: 'input', name: 'x' }, 2] } },
    },
    ...overrides,
  }
}

/** Check whether an expression tree contains a node with the given op. */
function containsOp(node: ExprNode, op: string): boolean {
  if (typeof node === 'number' || typeof node === 'boolean') return false
  if (Array.isArray(node)) return node.some(n => containsOp(n, op))
  if (typeof node !== 'object' || node === null) return false
  const obj = node as { op: string; [k: string]: unknown }
  if (obj.op === op) return true
  for (const [k, v] of Object.entries(obj)) {
    if (k === 'op') continue
    if (Array.isArray(v)) { if (v.some(n => containsOp(n as ExprNode, op))) return true }
    else if (typeof v === 'object' && v !== null) { if (containsOp(v as ExprNode, op)) return true }
  }
  return false
}

// ──────���────────────────────────────────────���─────────────────
// applyBounds (unit)
// ────────────────────────────────���────────────────────────────

describe('applyBounds', () => {
  const x: ExprNode = { op: 'input', id: 0 }

  test('two-sided bounds produce clamp', () => {
    const result = applyBounds(x, [-1, 1])
    expect(result).toEqual({ op: 'clamp', args: [x, -1, 1] })
  })

  test('lower bound only produces select+gt (max)', () => {
    const result = applyBounds(x, [0, null])
    expect(result).toEqual({ op: 'select', args: [{ op: 'gt', args: [x, 0] }, x, 0] })
  })

  test('upper bound only produces select+lt (min)', () => {
    const result = applyBounds(x, [null, 10])
    expect(result).toEqual({ op: 'select', args: [{ op: 'lt', args: [x, 10] }, x, 10] })
  })

  test('both null returns expression unchanged', () => {
    const result = applyBounds(x, [null, null])
    expect(result).toBe(x)
  })
})

// ───────���─────────────────────────────────────────────────────
// resolveBounds / resolveBaseType
// ───────────��───────────────────────────────��─────────────────

describe('resolveBounds', () => {
  test('string spec returns null', () => {
    expect(resolveBounds('x')).toBeNull()
  })

  test('explicit bounds are returned', () => {
    expect(resolveBounds({ name: 'out', bounds: [-2, 2] })).toEqual([-2, 2])
  })

  test('named alias extracts bounds', () => {
    expect(resolveBounds({ name: 'out', type: 'signal' })).toEqual([-1, 1])
    expect(resolveBounds({ name: 'out', type: 'unipolar' })).toEqual([0, 1])
    expect(resolveBounds({ name: 'out', type: 'freq' })).toEqual([0, null])
    expect(resolveBounds({ name: 'out', type: 'phase' })).toEqual([0, 1])
  })

  test('explicit bounds override alias', () => {
    expect(resolveBounds({ name: 'out', type: 'signal', bounds: [-2, 2] })).toEqual([-2, 2])
  })

  test('unknown type returns null', () => {
    expect(resolveBounds({ name: 'out', type: 'float' })).toBeNull()
  })

  test('no type no bounds returns null', () => {
    expect(resolveBounds({ name: 'out' })).toBeNull()
  })
})

describe('resolveBaseType', () => {
  test('alias resolves to base type', () => {
    expect(resolveBaseType('signal')).toBe('float')
    expect(resolveBaseType('freq')).toBe('float')
    expect(resolveBaseType('phase')).toBe('float')
  })

  test('non-alias passes through', () => {
    expect(resolveBaseType('float')).toBe('float')
    expect(resolveBaseType('int')).toBe('int')
    expect(resolveBaseType('float[4]')).toBe('float[4]')
  })

  test('undefined passes through', () => {
    expect(resolveBaseType(undefined)).toBeUndefined()
  })
})

// ─────────────────────────────────────────────────────────────
// loadProgramDef bounds extraction
// ────────────────────────────────────────────────────────────���

describe('loadProgramDef bounds', () => {
  test('extracts explicit output bounds', () => {
    const session = mockSession()
    const type = loadProgramDef(leafProgram({
      outputs: [{ name: 'out', type: 'float', bounds: [-1, 1] }],
    }), session)
    expect(type._def.outputBounds).toEqual([[-1, 1]])
  })

  test('extracts bounds from named type alias', () => {
    const session = mockSession()
    const type = loadProgramDef(leafProgram({
      outputs: [{ name: 'out', type: 'signal' }],
    }), session)
    expect(type._def.outputBounds).toEqual([[-1, 1]])
    // Base type resolved to float
    expect(type._def.outputPortTypes).toEqual(['float'])
  })

  test('extracts explicit input bounds', () => {
    const session = mockSession()
    const type = loadProgramDef(leafProgram({
      inputs: [{ name: 'x', type: 'float', bounds: [0, null] }],
    }), session)
    expect(type._def.inputBounds).toEqual([[0, null]])
  })

  test('no bounds yields null entries', () => {
    const session = mockSession()
    const type = loadProgramDef(leafProgram(), session)
    expect(type._def.outputBounds).toEqual([null])
    expect(type._def.inputBounds).toEqual([null])
  })

  test('string-shorthand input has null bounds', () => {
    const session = mockSession()
    const type = loadProgramDef(leafProgram({
      inputs: ['x'],
    }), session)
    expect(type._def.inputBounds).toEqual([null])
  })
})

// ───────���──────────────���─────────────────────────��────────────
// flattenSession with bounds
// ───────────────────���─────────────────────────────────────────

describe('flattenSession output bounds', () => {
  test('output bounds inject clamp into flattened expression', () => {
    const session = mockSession()

    // Register a leaf type with bounded output
    const type = loadProgramDef(leafProgram({
      outputs: [{ name: 'out', type: 'float', bounds: [-1, 1] }],
    }), session)
    session.typeRegistry.set('TestLeaf', type)

    // Create instance
    const inst = type.instantiateAs('a')
    session.instanceRegistry.set('a', inst)

    // Build a minimal full session for flattenSession
    const fullSession = {
      ...session,
      bufferLength: 1,
      dac: null,
      graphOutputs: [{ instance: 'a', output: 'out' }],
      inputExprNodes: new Map<string, ExprNode>(),
      runtime: null as any,
      graph: null as any,
      _nameCounters: new Map<string, number>(),
    }

    const plan = flattenSession(fullSession)
    // The plan should contain a Clamp instruction
    const hasClamp = plan.instructions.some(
      (instr: any) => instr.tag === 'Clamp'
    )
    expect(hasClamp).toBe(true)
  })

  test('unbounded output still gets audio safety clamp when routed to graph output', () => {
    const session = mockSession()

    const type = loadProgramDef(leafProgram(), session)
    session.typeRegistry.set('TestLeaf', type)

    const inst = type.instantiateAs('a')
    session.instanceRegistry.set('a', inst)

    const fullSession = {
      ...session,
      bufferLength: 1,
      dac: null,
      graphOutputs: [{ instance: 'a', output: 'out' }],
      inputExprNodes: new Map<string, ExprNode>(),
      runtime: null as any,
      graph: null as any,
      _nameCounters: new Map<string, number>(),
    }

    const plan = flattenSession(fullSession)
    // Audio safety clamp injects Clamp even with no declared bounds
    const hasClamp = plan.instructions.some(
      (instr: any) => instr.tag === 'Clamp'
    )
    expect(hasClamp).toBe(true)
  })

  test('one-sided lower bound produces Select (max) plus audio safety Clamp', () => {
    const session = mockSession()

    const type = loadProgramDef(leafProgram({
      outputs: [{ name: 'out', type: 'float', bounds: [0, null] }],
    }), session)
    session.typeRegistry.set('TestLeaf', type)

    const inst = type.instantiateAs('a')
    session.instanceRegistry.set('a', inst)

    const fullSession = {
      ...session,
      bufferLength: 1,
      dac: null,
      graphOutputs: [{ instance: 'a', output: 'out' }],
      inputExprNodes: new Map<string, ExprNode>(),
      runtime: null as any,
      graph: null as any,
      _nameCounters: new Map<string, number>(),
    }

    const plan = flattenSession(fullSession)
    // Select from output bounds [0, null] (max pattern), plus safety Clamp [-1, 1]
    const hasSelect = plan.instructions.some((instr: any) => instr.tag === 'Select')
    const hasClamp = plan.instructions.some((instr: any) => instr.tag === 'Clamp')
    expect(hasSelect).toBe(true)
    expect(hasClamp).toBe(true)
  })
})

describe('flattenSession input bounds', () => {
  test('input bounds clamp wiring expression', () => {
    const session = mockSession()

    // Source: unbounded output
    const srcType = loadProgramDef(leafProgram({
      name: 'Source',
      inputs: [],
      input_defaults: {},
      process: { outputs: { out: 999 } },
    }), session)
    session.typeRegistry.set('Source', srcType)

    // Dest: bounded input [0, 1]
    const dstType = loadProgramDef(leafProgram({
      name: 'Dest',
      inputs: [{ name: 'x', type: 'float', bounds: [0, 1] }],
    }), session)
    session.typeRegistry.set('Dest', dstType)

    const srcInst = srcType.instantiateAs('src')
    const dstInst = dstType.instantiateAs('dst')
    session.instanceRegistry.set('src', srcInst)
    session.instanceRegistry.set('dst', dstInst)

    const fullSession = {
      ...session,
      bufferLength: 1,
      dac: null,
      graphOutputs: [{ instance: 'dst', output: 'out' }],
      inputExprNodes: new Map<string, ExprNode>([
        ['dst:x', { op: 'ref', instance: 'src', output: 'out' }],
      ]),
      runtime: null as any,
      graph: null as any,
      _nameCounters: new Map<string, number>(),
    }

    const plan = flattenSession(fullSession)
    // The input wiring should be clamped
    const hasClamp = plan.instructions.some((instr: any) => instr.tag === 'Clamp')
    expect(hasClamp).toBe(true)
  })
})

describe('flattenSession cross-instance bounded output', () => {
  test('bounded output shared across consumers', () => {
    const session = mockSession()

    // Source with bounded output
    const srcType = loadProgramDef(leafProgram({
      name: 'Source',
      inputs: [],
      input_defaults: {},
      outputs: [{ name: 'out', type: 'float', bounds: [-1, 1] }],
      process: { outputs: { out: 999 } },
    }), session)
    session.typeRegistry.set('Source', srcType)

    // Two consumers
    const dstType = loadProgramDef(leafProgram({
      name: 'Dest',
    }), session)
    session.typeRegistry.set('Dest', dstType)

    session.instanceRegistry.set('src', srcType.instantiateAs('src'))
    session.instanceRegistry.set('d1', dstType.instantiateAs('d1'))
    session.instanceRegistry.set('d2', dstType.instantiateAs('d2'))

    const fullSession = {
      ...session,
      bufferLength: 1,
      dac: null,
      graphOutputs: [
        { instance: 'd1', output: 'out' },
        { instance: 'd2', output: 'out' },
      ],
      inputExprNodes: new Map<string, ExprNode>([
        ['d1:x', { op: 'ref', instance: 'src', output: 'out' }],
        ['d2:x', { op: 'ref', instance: 'src', output: 'out' }],
      ]),
      runtime: null as any,
      graph: null as any,
      _nameCounters: new Map<string, number>(),
    }

    const plan = flattenSession(fullSession)
    // Source's bounded output: 1 Clamp (shared via DAG).
    // d1 and d2 are unbounded graph outputs: each gets 1 audio safety Clamp.
    // Total: 3 Clamps.
    const clampCount = plan.instructions.filter((instr: any) => instr.tag === 'Clamp').length
    expect(clampCount).toBe(3)
  })
})

// ─────────────────────────────────────────────────────────────
// Audio output safety clamp
// ─────────────────────────────────────────────────────────────

describe('audio output safety clamp', () => {
  function makeFullSession(session: ReturnType<typeof mockSession>, graphOutputs: Array<{ instance: string; output: string }>) {
    return {
      ...session,
      bufferLength: 1,
      dac: null,
      graphOutputs,
      inputExprNodes: new Map<string, ExprNode>(),
      runtime: null as any,
      graph: null as any,
      _nameCounters: new Map<string, number>(),
    }
  }

  test('unbounded audio output gets safety clamp to [-1, 1]', () => {
    const session = mockSession()
    const type = loadProgramDef(leafProgram({
      process: { outputs: { out: 999 } },
    }), session)
    session.typeRegistry.set('TestLeaf', type)
    session.instanceRegistry.set('a', type.instantiateAs('a'))

    const plan = flattenSession(makeFullSession(session, [{ instance: 'a', output: 'out' }]))
    const hasClamp = plan.instructions.some((instr: any) => instr.tag === 'Clamp')
    expect(hasClamp).toBe(true)
  })

  test('output bounded [-1, 1] gets no extra safety clamp', () => {
    const session = mockSession()
    const type = loadProgramDef(leafProgram({
      outputs: [{ name: 'out', type: 'float', bounds: [-1, 1] }],
      process: { outputs: { out: { op: 'input', name: 'x' } } },
    }), session)
    session.typeRegistry.set('TestLeaf', type)
    session.instanceRegistry.set('a', type.instantiateAs('a'))

    const plan = flattenSession(makeFullSession(session, [{ instance: 'a', output: 'out' }]))
    // One Clamp from the output bounds, no extra from safety
    const clampCount = plan.instructions.filter((instr: any) => instr.tag === 'Clamp').length
    expect(clampCount).toBe(1)
  })

  test('output bounded [0, 1] (tighter) gets no extra safety clamp', () => {
    const session = mockSession()
    const type = loadProgramDef(leafProgram({
      outputs: [{ name: 'out', type: 'float', bounds: [0, 1] }],
      process: { outputs: { out: { op: 'input', name: 'x' } } },
    }), session)
    session.typeRegistry.set('TestLeaf', type)
    session.instanceRegistry.set('a', type.instantiateAs('a'))

    const plan = flattenSession(makeFullSession(session, [{ instance: 'a', output: 'out' }]))
    // One Clamp from the output bounds, no extra from safety
    const clampCount = plan.instructions.filter((instr: any) => instr.tag === 'Clamp').length
    expect(clampCount).toBe(1)
  })

  test('output bounded [-5, 5] (wider) gets additional safety clamp', () => {
    const session = mockSession()
    const type = loadProgramDef(leafProgram({
      outputs: [{ name: 'out', type: 'float', bounds: [-5, 5] }],
      process: { outputs: { out: { op: 'input', name: 'x' } } },
    }), session)
    session.typeRegistry.set('TestLeaf', type)
    session.instanceRegistry.set('a', type.instantiateAs('a'))

    const plan = flattenSession(makeFullSession(session, [{ instance: 'a', output: 'out' }]))
    // Two Clamps: one from output bounds [-5, 5], one from safety [-1, 1]
    const clampCount = plan.instructions.filter((instr: any) => instr.tag === 'Clamp').length
    expect(clampCount).toBe(2)
  })

  test('one-sided output bounds [0, null] gets safety clamp', () => {
    const session = mockSession()
    const type = loadProgramDef(leafProgram({
      outputs: [{ name: 'out', type: 'float', bounds: [0, null] }],
      process: { outputs: { out: { op: 'input', name: 'x' } } },
    }), session)
    session.typeRegistry.set('TestLeaf', type)
    session.instanceRegistry.set('a', type.instantiateAs('a'))

    const plan = flattenSession(makeFullSession(session, [{ instance: 'a', output: 'out' }]))
    // Select from output bounds (max), plus Clamp from safety [-1, 1]
    const hasSelect = plan.instructions.some((instr: any) => instr.tag === 'Select')
    const hasClamp = plan.instructions.some((instr: any) => instr.tag === 'Clamp')
    expect(hasSelect).toBe(true)
    expect(hasClamp).toBe(true)
  })
})
