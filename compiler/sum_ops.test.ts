/**
 * Phase 2 — sum-typed wiring expressions: tag and match.
 *
 * Validates the new ExprNode ops at the parse boundary, the scope-aware
 * substitution behavior of match arms, and the pretty-printer / slottifier
 * traversal hooks. Lowering of tag/match to scalar bundle ops happens later
 * (Phase 3) — these tests exercise only the structural-level concerns.
 */

import { describe, expect, test } from 'bun:test'
import { tag, match, validateExpr, type ExprNode, type SignalExpr } from './expr.js'
import { prettyExpr } from './session.js'

// ─────────────────────────────────────────────────────────────
// Construction
// ─────────────────────────────────────────────────────────────

describe('tag builder', () => {
  test('produces a well-shaped op node for a nullary variant', () => {
    const e = tag('Env', 'Idle')
    expect(e._node).toEqual({ op: 'tag', type: 'Env', variant: 'Idle' })
  })

  test('produces a well-shaped op node for a payloaded variant', () => {
    const e = tag('Env', 'Decaying', { level: 1.0 })
    expect(e._node).toEqual({
      op: 'tag', type: 'Env', variant: 'Decaying',
      payload: { level: 1.0 },
    })
  })

  test('coerces payload values from numbers and SignalExprs', () => {
    const inner = tag('Inner', 'A', { x: 5 })
    const e = tag('Env', 'Decaying', { level: inner })
    const node = e._node as { payload: Record<string, ExprNode> }
    expect(node.payload.level).toEqual(inner._node)
  })
})

describe('match builder', () => {
  test('produces a well-shaped op node with arms', () => {
    const e = match('Env', tag('Env', 'Idle'), {
      Idle:     { body: 0 },
      Decaying: { bind: 'level', body: 1 },
    })
    const node = e._node as Record<string, unknown>
    expect(node.op).toBe('match')
    expect(node.type).toBe('Env')
    expect(node.scrutinee).toEqual({ op: 'tag', type: 'Env', variant: 'Idle' })
    const arms = node.arms as Record<string, { bind?: string; body: ExprNode }>
    expect(arms.Idle).toEqual({ body: 0 })
    expect(arms.Decaying).toEqual({ bind: 'level', body: 1 })
  })

  test('preserves multi-name bindings as arrays', () => {
    const e = match('Pair', 0, {
      Two: { bind: ['a', 'b'], body: 5 },
    })
    const arms = (e._node as Record<string, unknown>).arms as Record<string, { bind?: string | string[] }>
    expect(arms.Two.bind).toEqual(['a', 'b'])
  })

  test('omits bind for nullary arms', () => {
    const e = match('Env', tag('Env', 'Idle'), {
      Idle: { body: 0 },
    })
    const arms = (e._node as Record<string, unknown>).arms as Record<string, { bind?: string; body: ExprNode }>
    expect(arms.Idle).toEqual({ body: 0 })
    expect('bind' in arms.Idle).toBe(false)
  })
})

// ─────────────────────────────────────────────────────────────
// Validation
// ─────────────────────────────────────────────────────────────

describe('validateExpr — tag', () => {
  test('accepts a well-formed tag with payload', () => {
    expect(() => validateExpr({
      op: 'tag', type: 'Env', variant: 'Decaying', payload: { level: 1.0 },
    })).not.toThrow()
  })

  test('accepts a nullary variant (no payload key)', () => {
    expect(() => validateExpr({ op: 'tag', type: 'Env', variant: 'Idle' })).not.toThrow()
  })

  test('rejects missing type', () => {
    expect(() => validateExpr({ op: 'tag', variant: 'Idle' } as ExprNode))
      .toThrow(/'tag' requires type/)
  })

  test('rejects missing variant', () => {
    expect(() => validateExpr({ op: 'tag', type: 'Env' } as ExprNode))
      .toThrow(/'tag' requires variant/)
  })

  test('rejects payload that is not an object', () => {
    expect(() => validateExpr({
      op: 'tag', type: 'Env', variant: 'Decaying', payload: [1, 2, 3],
    } as unknown as ExprNode)).toThrow(/payload must be an object/)
  })

  test('recursively validates payload field expressions', () => {
    expect(() => validateExpr({
      op: 'tag', type: 'Env', variant: 'Decaying',
      payload: { level: { op: 'add' } },  // missing args
    } as ExprNode)).toThrow(/'add' requires 'args'/)
  })
})

describe('validateExpr — match', () => {
  const okScrutinee: ExprNode = { op: 'tag', type: 'Env', variant: 'Idle' }

  test('accepts a well-formed match', () => {
    expect(() => validateExpr({
      op: 'match', type: 'Env', scrutinee: okScrutinee, arms: {
        Idle:     { body: 0 },
        Decaying: { bind: 'level', body: 1 },
      },
    })).not.toThrow()
  })

  test('rejects missing type', () => {
    expect(() => validateExpr({ op: 'match', scrutinee: 0, arms: { A: { body: 0 } } } as ExprNode))
      .toThrow(/'match' requires type/)
  })

  test('rejects missing scrutinee', () => {
    expect(() => validateExpr({ op: 'match', type: 'T', arms: { A: { body: 0 } } } as ExprNode))
      .toThrow(/'match' requires scrutinee/)
  })

  test('rejects non-object arms', () => {
    expect(() => validateExpr({
      op: 'match', type: 'T', scrutinee: 0, arms: [],
    } as unknown as ExprNode)).toThrow(/arms must be an object/)
  })

  test('rejects empty arms', () => {
    expect(() => validateExpr({ op: 'match', type: 'T', scrutinee: 0, arms: {} } as ExprNode))
      .toThrow(/at least one arm/)
  })

  test('rejects an arm missing body', () => {
    expect(() => validateExpr({
      op: 'match', type: 'T', scrutinee: 0, arms: { A: { bind: 'x' } },
    } as unknown as ExprNode)).toThrow(/missing required 'body'/)
  })

  test('rejects bind that is not string or string[]', () => {
    expect(() => validateExpr({
      op: 'match', type: 'T', scrutinee: 0, arms: { A: { bind: 5, body: 0 } },
    } as unknown as ExprNode)).toThrow(/bind: must be string or string\[\]/)
  })

  test('rejects bind array containing non-string', () => {
    expect(() => validateExpr({
      op: 'match', type: 'T', scrutinee: 0, arms: { A: { bind: ['x', 5], body: 0 } },
    } as unknown as ExprNode)).toThrow(/bind\[1\]: must be a string/)
  })

  test('recursively validates arm body expressions', () => {
    expect(() => validateExpr({
      op: 'match', type: 'T', scrutinee: 0, arms: {
        A: { body: { op: 'add' } },  // missing args
      },
    } as ExprNode)).toThrow(/'add' requires 'args'/)
  })
})

// ─────────────────────────────────────────────────────────────
// Scope-aware substitution (BINDER_OPS extension via match)
// ─────────────────────────────────────────────────────────────
//
// substituteBindings is internal to lower_arrays.ts, but its behavior is
// observable via the lowerArrayOps pipeline. We test by constructing a let
// that wraps a match whose arm shadows the let-bound name.

import { lowerArrayOps } from './lower_arrays.js'

describe('match arm bindings shield outer let bindings', () => {
  test('arm-bound name shadows outer let when used inside arm body', () => {
    // let level = 99 in match scrutinee {
    //   Decaying bind level => $level    ← should refer to ARM's level, not 99
    // }
    const expr: ExprNode = {
      op: 'let',
      bind: { level: 99 },
      in: {
        op: 'match', type: 'Env', scrutinee: { op: 'tag', type: 'Env', variant: 'Idle' },
        arms: {
          Decaying: { bind: 'level', body: { op: 'binding', name: 'level' } },
        },
      },
    }
    const lowered = lowerArrayOps(expr) as Record<string, unknown>
    // After let lowering: binding under arm should remain unresolved (the
    // arm-bound name shielded the outer let from substituting it). The lowered
    // body of the arm should still be {op:'binding', name:'level'}.
    expect(lowered.op).toBe('match')
    const arms = lowered.arms as Record<string, { body: ExprNode }>
    expect(arms.Decaying.body).toEqual({ op: 'binding', name: 'level' })
  })

  test('outer let binding flows into arm body when arm does NOT bind that name', () => {
    // let x = 99 in match scrutinee {
    //   Decaying bind level => $x   ← should refer to OUTER x = 99
    // }
    const expr: ExprNode = {
      op: 'let',
      bind: { x: 99 },
      in: {
        op: 'match', type: 'Env', scrutinee: { op: 'tag', type: 'Env', variant: 'Idle' },
        arms: {
          Decaying: { bind: 'level', body: { op: 'binding', name: 'x' } },
        },
      },
    }
    const lowered = lowerArrayOps(expr) as Record<string, unknown>
    expect(lowered.op).toBe('match')
    const arms = lowered.arms as Record<string, { body: ExprNode }>
    expect(arms.Decaying.body).toBe(99)
  })

  test('outer let flows into scrutinee even when arms shadow same name', () => {
    // let level = 99 in match $level {
    //   Decaying bind level => $level
    // }
    // Scrutinee should see 99; arm body should see arm-bound (unresolved).
    const expr: ExprNode = {
      op: 'let',
      bind: { level: 99 },
      in: {
        op: 'match', type: 'Env', scrutinee: { op: 'binding', name: 'level' },
        arms: {
          Decaying: { bind: 'level', body: { op: 'binding', name: 'level' } },
        },
      },
    }
    const lowered = lowerArrayOps(expr) as Record<string, unknown>
    expect(lowered.op).toBe('match')
    expect(lowered.scrutinee).toBe(99)
    const arms = lowered.arms as Record<string, { body: ExprNode }>
    expect(arms.Decaying.body).toEqual({ op: 'binding', name: 'level' })
  })

  test('multi-name bind shields all of its names', () => {
    const expr: ExprNode = {
      op: 'let',
      bind: { a: 10, b: 20, c: 30 },
      in: {
        op: 'match', type: 'Pair', scrutinee: 0,
        arms: {
          Two: {
            bind: ['a', 'b'],
            body: {
              op: 'add',
              args: [
                { op: 'binding', name: 'a' },  // shielded — stays unresolved
                { op: 'binding', name: 'c' },  // not shielded — substitutes to 30
              ],
            },
          },
        },
      },
    }
    const lowered = lowerArrayOps(expr) as Record<string, unknown>
    const arms = lowered.arms as Record<string, { body: { args: ExprNode[] } }>
    expect(arms.Two.body.args[0]).toEqual({ op: 'binding', name: 'a' })
    expect(arms.Two.body.args[1]).toBe(30)
  })
})

// ─────────────────────────────────────────────────────────────
// Pretty-printer (smoke check, exercises slottify path indirectly)
// ─────────────────────────────────────────────────────────────

describe('prettyExpr — sum ops', () => {
  test('renders a tag with payload', () => {
    const node: ExprNode = { op: 'tag', type: 'Env', variant: 'Decaying', payload: { level: 1 } }
    expect(prettyExpr(node, new Map() as never)).toBe('Env::Decaying{level: 1}')
  })

  test('renders a tag without payload', () => {
    const node: ExprNode = { op: 'tag', type: 'Env', variant: 'Idle' }
    expect(prettyExpr(node, new Map() as never)).toBe('Env::Idle')
  })

  test('renders a match with bind names', () => {
    const node: ExprNode = {
      op: 'match', type: 'Env', scrutinee: { op: 'tag', type: 'Env', variant: 'Idle' },
      arms: {
        Idle:     { body: 0 },
        Decaying: { bind: 'level', body: { op: 'binding', name: 'level' } },
      },
    }
    const s = prettyExpr(node, new Map() as never)
    expect(s).toContain('match(')
    expect(s).toContain('type=Env')
    expect(s).toContain('Idle: 0')
    expect(s).toContain('Decaying bind level')
  })
})
