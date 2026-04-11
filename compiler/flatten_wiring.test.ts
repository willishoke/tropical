/**
 * flatten_wiring.test.ts — Tests for flatten-time wiring type validation.
 */

import { describe, test, expect } from 'bun:test'
import { normalizeWiringTypes, FlattenError } from './flatten'
import type { InstanceInfo } from './compiler'
import { Float, ArrayType } from './term'
import type { ExprNode } from './expr'

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
