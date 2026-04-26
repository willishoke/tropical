/**
 * array_wiring.test.ts — Tests for inter-module array connection validation.
 */

import { describe, test, expect } from 'bun:test'
import { checkArrayConnection } from './array_wiring'
import { Float, Int, Bool, ArrayType, StructType } from './term'
import type { ExprNode } from './expr'

const ref: ExprNode = { op: 'ref', instance: 'VCO1', output: 'saw' }

describe('checkArrayConnection', () => {
  test('identical scalar types are compatible', () => {
    const check = checkArrayConnection(Float, Float, ref)
    expect(check.compatible).toBe(true)
    expect(check.broadcastExpr).toBeUndefined()
  })

  test('undefined types default to float and are compatible', () => {
    const check = checkArrayConnection(undefined, undefined, ref)
    expect(check.compatible).toBe(true)
  })

  test('identical array types are compatible', () => {
    const check = checkArrayConnection(ArrayType(Float, [4]), ArrayType(Float, [4]), ref)
    expect(check.compatible).toBe(true)
    expect(check.broadcastExpr).toBeUndefined()
  })

  test('different scalar types are incompatible', () => {
    const check = checkArrayConnection(Float, Bool, ref)
    expect(check.compatible).toBe(false)
    expect(check.error).toBeDefined()
  })

  test('scalar to array auto-broadcasts', () => {
    const check = checkArrayConnection(Float, ArrayType(Float, [4]), ref)
    expect(check.compatible).toBe(true)
    expect(check.broadcastExpr).toBeDefined()
    const node = check.broadcastExpr as Record<string, unknown>
    expect(node.op).toBe('broadcastTo')
    expect(node.shape).toEqual([4])
    expect(check.resultShape).toEqual([4])
  })

  test('array to scalar is incompatible', () => {
    const check = checkArrayConnection(ArrayType(Float, [4]), Float, ref)
    expect(check.compatible).toBe(false)
    expect(check.error).toContain('reduce or index')
  })

  test('compatible array shapes pass through', () => {
    const check = checkArrayConnection(ArrayType(Float, [4]), ArrayType(Float, [4]), ref)
    expect(check.compatible).toBe(true)
    expect(check.broadcastExpr).toBeUndefined()
  })

  test('incompatible array shapes fail', () => {
    const check = checkArrayConnection(ArrayType(Float, [3]), ArrayType(Float, [4]), ref)
    expect(check.compatible).toBe(false)
    expect(check.error).toContain('not broadcast-compatible')
  })

  test('broadcastable array shapes insert broadcast_to', () => {
    const check = checkArrayConnection(ArrayType(Float, [1]), ArrayType(Float, [4]), ref)
    expect(check.compatible).toBe(true)
    expect(check.broadcastExpr).toBeDefined()
    const node = check.broadcastExpr as Record<string, unknown>
    expect(node.op).toBe('broadcastTo')
    expect(node.shape).toEqual([4])
  })

  test('2D array shape compatibility', () => {
    const check = checkArrayConnection(ArrayType(Float, [4, 4]), ArrayType(Float, [4, 4]), ref)
    expect(check.compatible).toBe(true)
    expect(check.broadcastExpr).toBeUndefined()
  })

  test('2D array shape broadcast', () => {
    const check = checkArrayConnection(ArrayType(Float, [1, 4]), ArrayType(Float, [3, 4]), ref)
    expect(check.compatible).toBe(true)
    expect(check.broadcastExpr).toBeDefined()
  })

  test('struct type mismatch', () => {
    const check = checkArrayConnection(StructType('MyStruct'), StructType('OtherStruct'), ref)
    expect(check.compatible).toBe(false)
  })

  test('same struct types are compatible', () => {
    const check = checkArrayConnection(StructType('MyStruct'), StructType('MyStruct'), ref)
    expect(check.compatible).toBe(true)
  })

  // ── Scalar-kind widening lattice (Phase 4) ───────────────────
  //
  // Implicit widening chain: bool → int → float. Narrowing
  // (float → int, float → bool, int → bool) must be rejected with
  // an actionable error telling the user to wrap the source in a cast.

  test('int scalar widens to float scalar', () => {
    const check = checkArrayConnection(Int, Float, ref)
    expect(check.compatible).toBe(true)
  })

  test('bool scalar widens to int scalar', () => {
    const check = checkArrayConnection(Bool, Int, ref)
    expect(check.compatible).toBe(true)
  })

  test('float scalar to int scalar rejected with to_int hint', () => {
    const check = checkArrayConnection(Float, Int, ref)
    expect(check.compatible).toBe(false)
    expect(check.error).toMatch(/to_int|narrow|lossy/i)
  })

  test('float scalar to bool scalar rejected', () => {
    const check = checkArrayConnection(Float, Bool, ref)
    expect(check.compatible).toBe(false)
    expect(check.error).toMatch(/to_bool|narrow|lossy/i)
  })

  test('int scalar to bool scalar rejected', () => {
    const check = checkArrayConnection(Int, Bool, ref)
    expect(check.compatible).toBe(false)
    expect(check.error).toMatch(/to_bool|narrow|lossy/i)
  })

  test('same-shape int array widens to float array', () => {
    const check = checkArrayConnection(ArrayType(Int, [4]), ArrayType(Float, [4]), ref)
    expect(check.compatible).toBe(true)
  })

  test('same-shape float array to int array rejected', () => {
    const check = checkArrayConnection(ArrayType(Float, [4]), ArrayType(Int, [4]), ref)
    expect(check.compatible).toBe(false)
    expect(check.error).toMatch(/to_int|narrow|lossy/i)
  })
})
