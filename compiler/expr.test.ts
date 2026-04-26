/**
 * expr.test.ts — Tests for first-class array expression DSL.
 */

import { describe, test, expect } from 'bun:test'
import {
  SignalExpr,
  coerce,
  add, mul, sub, matmul,
  arrayPack, arraySet, arrayLiteral,
  zeros, ones, fill, reshape, transpose,
  slice, reduce, broadcastTo, mapArray,
} from './expr'

describe('array construction', () => {
  test('arrayLiteral produces correct node', () => {
    const expr = arrayLiteral([2, 2], [1, 2, 3, 4])
    const node = expr._node as Record<string, unknown>
    expect(node.op).toBe('arrayLiteral')
    expect(node.shape).toEqual([2, 2])
    expect(node.values).toEqual([1, 2, 3, 4])
  })

  test('zeros produces correct node', () => {
    const expr = zeros([4])
    const node = expr._node as Record<string, unknown>
    expect(node.op).toBe('zeros')
    expect(node.shape).toEqual([4])
  })

  test('ones produces correct node', () => {
    const expr = ones([3, 3])
    const node = expr._node as Record<string, unknown>
    expect(node.op).toBe('ones')
    expect(node.shape).toEqual([3, 3])
  })

  test('fill produces correct node', () => {
    const expr = fill([8], 0.5)
    const node = expr._node as Record<string, unknown>
    expect(node.op).toBe('fill')
    expect(node.shape).toEqual([8])
    expect(node.value).toBe(0.5)
  })
})

describe('array manipulation', () => {
  test('reshape', () => {
    const arr = arrayLiteral([2, 3], [1, 2, 3, 4, 5, 6])
    const r = reshape(arr, [3, 2])
    const node = r._node as Record<string, unknown>
    expect(node.op).toBe('reshape')
    expect(node.shape).toEqual([3, 2])
  })

  test('transpose', () => {
    const arr = arrayLiteral([2, 3], [1, 2, 3, 4, 5, 6])
    const t = transpose(arr)
    const node = t._node as Record<string, unknown>
    expect(node.op).toBe('transpose')
  })

  test('slice', () => {
    const arr = arrayLiteral([8], [0, 1, 2, 3, 4, 5, 6, 7])
    const s = slice(arr, 0, 2, 5)
    const node = s._node as Record<string, unknown>
    expect(node.op).toBe('slice')
    expect(node.axis).toBe(0)
    expect(node.start).toBe(2)
    expect(node.end).toBe(5)
  })

  test('reduce', () => {
    const arr = arrayLiteral([4], [1, 2, 3, 4])
    const s = reduce(arr, 0, 'add')
    const node = s._node as Record<string, unknown>
    expect(node.op).toBe('reduce')
    expect(node.axis).toBe(0)
    expect(node.reduce_op).toBe('add')
  })

  test('broadcastTo', () => {
    const arr = arrayLiteral([1, 4], [1, 2, 3, 4])
    const b = broadcastTo(arr, [3, 4])
    const node = b._node as Record<string, unknown>
    expect(node.op).toBe('broadcastTo')
    expect(node.shape).toEqual([3, 4])
  })

  test('mapArray', () => {
    const arr = arrayLiteral([4], [1, 2, 3, 4])
    const m = mapArray(x => mul(x, 2), arr)
    const node = m._node as Record<string, unknown>
    expect(node.op).toBe('map')
    expect((node.callee as Record<string, unknown>).op).toBe('function')
  })
})

describe('SignalExpr array methods', () => {
  test('reshape method', () => {
    const arr = coerce([1, 2, 3, 4, 5, 6])
    const r = arr.reshape([2, 3])
    const node = r._node as Record<string, unknown>
    expect(node.op).toBe('reshape')
    expect(node.shape).toEqual([2, 3])
  })

  test('transpose method', () => {
    const arr = coerce([1, 2, 3, 4])
    const t = arr.transpose()
    const node = t._node as Record<string, unknown>
    expect(node.op).toBe('transpose')
  })

  test('slice method', () => {
    const arr = coerce([1, 2, 3, 4, 5])
    const s = arr.slice(0, 1, 3)
    const node = s._node as Record<string, unknown>
    expect(node.op).toBe('slice')
    expect(node.axis).toBe(0)
    expect(node.start).toBe(1)
    expect(node.end).toBe(3)
  })

  test('reduce method', () => {
    const arr = coerce([1, 2, 3, 4])
    const s = arr.reduce(0, 'add')
    const node = s._node as Record<string, unknown>
    expect(node.op).toBe('reduce')
  })

  test('sum method', () => {
    const arr = coerce([1, 2, 3, 4])
    const s = arr.sum()
    const node = s._node as Record<string, unknown>
    expect(node.op).toBe('reduce')
    expect(node.reduce_op).toBe('add')
    expect(node.axis).toBe(0)
  })
})

describe('shape-polymorphic arithmetic', () => {
  test('add arrays produces add node', () => {
    const a = arrayLiteral([4], [1, 2, 3, 4])
    const b = arrayLiteral([4], [5, 6, 7, 8])
    const r = add(a, b)
    const node = r._node as Record<string, unknown>
    expect(node.op).toBe('add')
    // args should be the two array_literal nodes
    const args = node.args as unknown[]
    expect((args[0] as Record<string, unknown>).op).toBe('arrayLiteral')
    expect((args[1] as Record<string, unknown>).op).toBe('arrayLiteral')
  })

  test('mul scalar by array produces mul node', () => {
    const a = arrayLiteral([4], [1, 2, 3, 4])
    const r = mul(2.0, a)
    const node = r._node as Record<string, unknown>
    expect(node.op).toBe('mul')
  })

  test('matmul produces matmul node with shapes and result shape [M,N]', () => {
    const a = arrayLiteral([2, 3], [1, 2, 3, 4, 5, 6])
    const b = arrayLiteral([3, 2], [1, 2, 3, 4, 5, 6])
    const r = matmul(a, b, [2, 3], [3, 2])
    const node = r._node as Record<string, unknown>
    expect(node.op).toBe('matmul')
    expect(node.shape_a).toEqual([2, 3])
    expect(node.shape_b).toEqual([3, 2])
    expect(node.element_type).toBe('float')
    expect(r.shape).toEqual([2, 2])
  })

  test('matmul with bool element_type embeds element_type on node', () => {
    const a = arrayLiteral([2, 2], [1, 0, 0, 1])
    const b = arrayLiteral([2, 2], [1, 1, 0, 1])
    const r = matmul(a, b, [2, 2], [2, 2], 'bool')
    const node = r._node as Record<string, unknown>
    expect(node.element_type).toBe('bool')
  })

  test('matmul throws on inner dimension mismatch', () => {
    const a = arrayLiteral([2, 3], [1, 2, 3, 4, 5, 6])
    const b = arrayLiteral([2, 2], [1, 2, 3, 4])
    expect(() => matmul(a, b, [2, 3], [2, 2])).toThrow('inner dimensions must match')
  })
})
