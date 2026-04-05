/**
 * lower_arrays.test.ts — Tests for array operation lowering.
 */

import { describe, test, expect } from 'bun:test'
import { lowerArrayOps } from './lower_arrays'
import type { ExprNode } from './expr'

describe('lowerArrayOps', () => {
  test('passes through scalars', () => {
    expect(lowerArrayOps(42)).toBe(42)
    expect(lowerArrayOps(true)).toBe(true)
  })

  test('passes through non-array ops', () => {
    const node: ExprNode = { op: 'add', args: [1, 2] }
    expect(lowerArrayOps(node)).toEqual({ op: 'add', args: [1, 2] })
  })

  test('lowers zeros to array of 0s', () => {
    const node: ExprNode = { op: 'zeros', shape: [4] }
    expect(lowerArrayOps(node)).toEqual([0, 0, 0, 0])
  })

  test('lowers ones to array of 1s', () => {
    const node: ExprNode = { op: 'ones', shape: [2, 2] }
    expect(lowerArrayOps(node)).toEqual([1, 1, 1, 1])
  })

  test('lowers fill to repeated value', () => {
    const node: ExprNode = { op: 'fill', shape: [3], value: 0.5 }
    expect(lowerArrayOps(node)).toEqual([0.5, 0.5, 0.5])
  })

  test('lowers array_literal to inline array', () => {
    const node: ExprNode = { op: 'array_literal', shape: [2, 2], values: [1, 2, 3, 4] }
    expect(lowerArrayOps(node)).toEqual([1, 2, 3, 4])
  })

  test('lowers reshape to identity (flat data unchanged)', () => {
    const node: ExprNode = { op: 'reshape', args: [[1, 2, 3, 4, 5, 6]], shape: [3, 2] }
    expect(lowerArrayOps(node)).toEqual([1, 2, 3, 4, 5, 6])
  })

  test('lowers slice to Index ops', () => {
    const node: ExprNode = { op: 'slice', args: [[10, 20, 30, 40, 50]], axis: 0, start: 1, end: 4 }
    const result = lowerArrayOps(node) as ExprNode[]
    expect(Array.isArray(result)).toBe(true)
    expect(result.length).toBe(3)
    // Each element should be an index op
    for (let i = 0; i < 3; i++) {
      const r = result[i] as Record<string, unknown>
      expect(r.op).toBe('index')
    }
  })

  test('lowers reduce on inline array to tree reduction', () => {
    const node: ExprNode = { op: 'reduce', args: [[1, 2, 3, 4]], axis: 0, reduce_op: 'add' }
    const result = lowerArrayOps(node)
    // Should be a tree of add ops, not an array
    expect(Array.isArray(result)).toBe(false)
    const r = result as Record<string, unknown>
    expect(r.op).toBe('add')
  })

  test('lowers reduce of single element to the element', () => {
    const node: ExprNode = { op: 'reduce', args: [[42]], axis: 0, reduce_op: 'add' }
    expect(lowerArrayOps(node)).toBe(42)
  })

  test('lowers broadcast_to scalar', () => {
    const node: ExprNode = { op: 'broadcast_to', args: [5], shape: [4] }
    expect(lowerArrayOps(node)).toEqual([5, 5, 5, 5])
  })

  test('lowers broadcast_to [1] to [N]', () => {
    const node: ExprNode = { op: 'broadcast_to', args: [[7]], shape: [3] }
    expect(lowerArrayOps(node)).toEqual([7, 7, 7])
  })

  test('lowers map on inline array', () => {
    const callee: ExprNode = {
      op: 'function', param_count: 1,
      body: { op: 'mul', args: [{ op: 'input', id: 0 }, 2] },
    }
    const node: ExprNode = { op: 'map', callee, args: [[1, 2, 3]] }
    const result = lowerArrayOps(node) as ExprNode[]
    expect(Array.isArray(result)).toBe(true)
    expect(result.length).toBe(3)
    // Each element should be a call
    for (const r of result) {
      expect((r as Record<string, unknown>).op).toBe('call')
    }
  })

  test('recursively lowers nested array ops', () => {
    // add(zeros([2]), ones([2])) should lower zeros and ones inside
    const node: ExprNode = {
      op: 'add',
      args: [
        { op: 'zeros', shape: [2] },
        { op: 'ones', shape: [2] },
      ],
    }
    const result = lowerArrayOps(node) as Record<string, unknown>
    expect(result.op).toBe('add')
    const args = result.args as ExprNode[]
    expect(args[0]).toEqual([0, 0])
    expect(args[1]).toEqual([1, 1])
  })
})
