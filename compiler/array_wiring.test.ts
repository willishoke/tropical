/**
 * array_wiring.test.ts — Tests for inter-module array connection validation.
 */

import { describe, test, expect } from 'bun:test'
import { checkArrayConnection } from './array_wiring'
import type { ExprNode } from './expr'

const ref: ExprNode = { op: 'ref', module: 'VCO1', output: 'saw' }

describe('checkArrayConnection', () => {
  test('identical scalar types are compatible', () => {
    const check = checkArrayConnection('float', 'float', ref)
    expect(check.compatible).toBe(true)
    expect(check.broadcastExpr).toBeUndefined()
  })

  test('undefined types are compatible (backwards compat)', () => {
    const check = checkArrayConnection(undefined, undefined, ref)
    expect(check.compatible).toBe(true)
  })

  test('identical string types are compatible', () => {
    const check = checkArrayConnection('float[4]', 'float[4]', ref)
    expect(check.compatible).toBe(true)
    expect(check.broadcastExpr).toBeUndefined()
  })

  test('different scalar types are incompatible', () => {
    const check = checkArrayConnection('float', 'bool', ref)
    expect(check.compatible).toBe(false)
    expect(check.error).toBeDefined()
  })

  test('scalar to array auto-broadcasts', () => {
    const check = checkArrayConnection('float', 'float[4]', ref)
    expect(check.compatible).toBe(true)
    expect(check.broadcastExpr).toBeDefined()
    const node = check.broadcastExpr as Record<string, unknown>
    expect(node.op).toBe('broadcast_to')
    expect(node.shape).toEqual([4])
    expect(check.resultShape).toEqual([4])
  })

  test('array to scalar is incompatible', () => {
    const check = checkArrayConnection('float[4]', 'float', ref)
    expect(check.compatible).toBe(false)
    expect(check.error).toContain('reduce or index')
  })

  test('compatible array shapes pass through', () => {
    const check = checkArrayConnection('float[4]', 'float[4]', ref)
    expect(check.compatible).toBe(true)
    expect(check.broadcastExpr).toBeUndefined()
  })

  test('incompatible array shapes fail', () => {
    const check = checkArrayConnection('float[3]', 'float[4]', ref)
    expect(check.compatible).toBe(false)
    expect(check.error).toContain('not broadcast-compatible')
  })

  test('broadcastable array shapes insert broadcast_to', () => {
    const check = checkArrayConnection('float[1]', 'float[4]', ref)
    expect(check.compatible).toBe(true)
    expect(check.broadcastExpr).toBeDefined()
    const node = check.broadcastExpr as Record<string, unknown>
    expect(node.op).toBe('broadcast_to')
    expect(node.shape).toEqual([4])
  })

  test('2D array shape compatibility', () => {
    const check = checkArrayConnection('float[4,4]', 'float[4,4]', ref)
    expect(check.compatible).toBe(true)
    expect(check.broadcastExpr).toBeUndefined()
  })

  test('2D array shape broadcast', () => {
    const check = checkArrayConnection('float[1,4]', 'float[3,4]', ref)
    expect(check.compatible).toBe(true)
    expect(check.broadcastExpr).toBeDefined()
  })

  test('unknown type names without registry default to float (compatible)', () => {
    const check = checkArrayConnection('MyStruct', 'OtherStruct', ref)
    expect(check.compatible).toBe(true)  // both resolve to Float without registry
  })

  test('same type names are compatible', () => {
    const check = checkArrayConnection('MyStruct', 'MyStruct', ref)
    expect(check.compatible).toBe(true)
  })
})
