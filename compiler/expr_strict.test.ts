/**
 * Phase 1 — verify the closed parametric-arity discriminated union actually
 * narrows. These tests construct typed ExprOpNodeStrict values and assert that
 * TypeScript's type-narrowing produces typed field access without `as` casts.
 *
 * Most of the value is at the type level — failures show up as compile errors,
 * not runtime test failures. The runtime checks here just exercise the types.
 */

import { describe, expect, test } from 'bun:test'
import type {
  ExprNode,
  ExprOpNodeStrict,
  Op,
  AddNode,
  BinaryNode,
  BinaryTag,
  UnaryNode,
  TernaryNode,
  TagNode,
  MatchNode,
  MatchArmStrict,
  ReshapeNode,
  ZerosNode,
  LeafNode,
  InputNode,
  DelayDeclNode,
} from './expr.js'

// ─────────────────────────────────────────────────────────────
// Type-level shape tests. These compile-or-fail.
// If any type assertion below errors, Phase 1 is broken at the type level.
// ─────────────────────────────────────────────────────────────

describe('Op<N, Tag> tuple shape', () => {
  test('Op<2, _> args is a tuple of length 2', () => {
    // Type-level check: AddNode is `Op<2, 'add'>`, so args is [ExprNode, ExprNode].
    const n: AddNode = { op: 'add', args: [1, 2] }
    expect(n.args).toHaveLength(2)
    // Narrowing on `op` gives precise field types.
    if (n.op === 'add') {
      // n.args is statically [ExprNode, ExprNode] — direct destructuring.
      const [a, b] = n.args
      expect(a).toBe(1)
      expect(b).toBe(2)
    }
  })

  test('Op<1, _> args is a tuple of length 1', () => {
    const n: UnaryNode = { op: 'neg', args: [5] }
    expect(n.args).toHaveLength(1)
  })

  test('Op<3, _> args is a tuple of length 3', () => {
    const n: TernaryNode = { op: 'select', args: [true, 1, 0] }
    expect(n.args).toHaveLength(3)
  })

  test('BinaryTag union covers all binary op kinds', () => {
    // Every binary tag is assignable to BinaryTag.
    const tags: BinaryTag[] = [
      'add', 'sub', 'mul', 'div', 'mod', 'floorDiv', 'ldexp',
      'lt', 'lte', 'gt', 'gte', 'eq', 'neq',
      'bitAnd', 'bitOr', 'bitXor', 'lshift', 'rshift',
      'and', 'or',
    ]
    expect(tags).toHaveLength(20)
  })

  test('BinaryNode covers any BinaryTag op', () => {
    // A BinaryNode value can carry any BinaryTag.
    const adds: BinaryNode = { op: 'add', args: [1, 2] }
    const lts: BinaryNode = { op: 'lt', args: [1, 2] }
    const ands: BinaryNode = { op: 'and', args: [true, false] }
    expect(adds.op).toBe('add')
    expect(lts.op).toBe('lt')
    expect(ands.op).toBe('and')
  })
})

describe('Op<N> with extras narrows correctly', () => {
  test('ReshapeNode has args + shape', () => {
    const n: ReshapeNode = { op: 'reshape', args: [{ op: 'sampleRate' }], shape: [2, 3] }
    if (n.op === 'reshape') {
      // Both fields are statically typed.
      expect(n.args[0]).toEqual({ op: 'sampleRate' })
      expect(n.shape).toEqual([2, 3])
    }
  })

  test('ZerosNode has shape but no args', () => {
    const n: ZerosNode = { op: 'zeros', shape: [4, 4] }
    expect(n.shape).toEqual([4, 4])
    // Type-level: `'args' in n` is statically false; we don't attempt access.
  })
})

describe('Named-children ops narrow correctly', () => {
  test('TagNode payload is Record<string, ExprNode>', () => {
    const n: TagNode = {
      op: 'tag', type: 'Env', variant: 'Decaying',
      payload: { level: 1.0 },
    }
    if (n.op === 'tag') {
      // n.payload is Record<string, ExprNode> | undefined — direct field access.
      expect(n.payload?.level).toBe(1.0)
    }
  })

  test('MatchNode arms are Record<string, MatchArmStrict>', () => {
    const arm: MatchArmStrict = { bind: 'level', body: { op: 'binding', name: 'level' } }
    const n: MatchNode = {
      op: 'match', type: 'Env',
      scrutinee: { op: 'sampleRate' },
      arms: { Idle: { body: 0 }, Decaying: arm },
    }
    if (n.op === 'match') {
      expect(n.arms.Idle.body).toBe(0)
      expect(n.arms.Decaying.bind).toBe('level')
    }
  })
})

describe('Leaf nodes narrow correctly', () => {
  test('InputNode has optional id and name', () => {
    const post: InputNode = { op: 'input', id: 0 }
    const pre:  InputNode = { op: 'input', name: 'freq' }
    expect(post.id).toBe(0)
    expect(pre.name).toBe('freq')
  })
})

describe('Decl nodes narrow correctly', () => {
  test('DelayDeclNode has init/update/type fields', () => {
    const n: DelayDeclNode = {
      op: 'delayDecl',
      name: 'state',
      type: 'Env',
      init: { op: 'tag', type: 'Env', variant: 'Idle' },
    }
    if (n.op === 'delayDecl') {
      expect(n.name).toBe('state')
      expect(n.type).toBe('Env')
    }
  })
})

describe('ExprOpNodeStrict union assignability', () => {
  test('every strict variant is assignable to ExprOpNodeStrict', () => {
    // Mostly a type-level test. If any of these don't compile, the union is
    // missing a variant.
    const nodes: ExprOpNodeStrict[] = [
      { op: 'add', args: [1, 2] },
      { op: 'neg', args: [3] },
      { op: 'select', args: [true, 1, 0] },
      { op: 'array', args: [1, 2, 3] },
      { op: 'reshape', args: [0], shape: [2, 2] },
      { op: 'zeros', shape: [4] },
      { op: 'tag', type: 'T', variant: 'V' },
      { op: 'match', type: 'T', scrutinee: 0, arms: { V: { body: 0 } } },
      { op: 'sampleRate' },
      { op: 'input', id: 0 },
      { op: 'delayDecl', name: 's' },
    ]
    expect(nodes).toHaveLength(11)
  })

  test('ExprOpNodeStrict is assignable to the broader ExprNode', () => {
    // Strict union values flow into existing ExprNode-typed code without casts.
    const strict: ExprOpNodeStrict = { op: 'add', args: [1, 2] }
    const broad: ExprNode = strict
    expect(broad).toBe(strict)
  })
})

// ─────────────────────────────────────────────────────────────
// Synthetic-op test: prove the closed union catches missing variants.
// ─────────────────────────────────────────────────────────────
//
// This test demonstrates that a hypothetical new op like 'foobar' is NOT
// assignable to ExprOpNodeStrict — it would be a compile error. We can't
// directly test the negative at runtime; the comment captures the property.
//
// To verify manually:
//   const fake: ExprOpNodeStrict = { op: 'foobar', args: [1] }
//   // ^ should produce TS error: Type '"foobar"' is not assignable...
