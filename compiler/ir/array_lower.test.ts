/**
 * array_lower.test.ts — coverage for the Phase C6 combinator unrolling
 * and array-op lowering pass on the resolved IR.
 *
 * Each test elaborates a `.trop`-style fixture, runs `arrayLower`, and
 * checks structural properties of the output:
 *
 *   - No combinator ops survive (`let`, `fold`, `scan`, `generate`,
 *     `iterate`, `chain`, `map2`, `zipWith`).
 *   - No `bindingRef` survives (every binder introduced by a
 *     combinator/let has been substituted away).
 *   - `zeros{N}` with a numeric `N` collapses to an inline `[0, ..., 0]`.
 *
 * Cases ported from `compiler/lower_arrays.test.ts` cover every
 * combinator in the resolved IR; the legacy file's tests for ops not
 * present in the resolved IR (`ones`, `fill`, `arrayLiteral`, `reshape`,
 * `transpose`, `slice`, `reduce`, `broadcastTo`, `map`, `matmul`,
 * `expandDeclGenerators`) are skipped — those ops never reach the
 * resolved IR (the parser/elaborator either doesn't accept them or
 * lowers them to other shapes).
 */

import { describe, test, expect } from 'bun:test'
import { parseProgram } from '../parse/declarations.js'
import { elaborate } from './elaborator.js'
import { arrayLower } from './array_lower.js'
import type { ResolvedProgram, ResolvedExpr } from './nodes.js'

function elab(src: string): ResolvedProgram {
  return elaborate(parseProgram(src))
}

/** Walk a resolved-program graph (cycle-safe via WeakSet) and return
 *  every `op` value matching the targets. */
function findOps(prog: ResolvedProgram, targets: string[]): string[] {
  const set = new Set(targets)
  const out: string[] = []
  const seen = new WeakSet<object>()
  const visitExpr = (e: ResolvedExpr): void => {
    if (e === null || typeof e !== 'object') return
    if (seen.has(e as object)) return
    seen.add(e as object)
    if (Array.isArray(e)) { e.forEach(visitExpr); return }
    if (set.has(e.op)) out.push(e.op)
    walkExprChildren(e, visitExpr)
  }
  for (const d of prog.body.decls) {
    if (d.op === 'regDecl') {
      visitExpr(d.init)
      if (d.update !== undefined) visitExpr(d.update)
    } else if (d.op === 'instanceDecl') {
      for (const i of d.inputs) visitExpr(i.value)
    }
  }
  for (const a of prog.body.assigns) visitExpr(a.expr)
  return out
}

function walkExprChildren(
  node: Exclude<ResolvedExpr, number | boolean | ResolvedExpr[]>,
  visit: (e: ResolvedExpr) => void,
): void {
  switch (node.op) {
    case 'inputRef': case 'regRef': case 'paramRef':
    case 'typeParamRef': case 'bindingRef': case 'nestedOut':
    case 'sampleRate': case 'sampleIndex':
      return
    case 'add': case 'sub': case 'mul': case 'div': case 'mod':
    case 'lt': case 'lte': case 'gt': case 'gte': case 'eq': case 'neq':
    case 'and': case 'or':
    case 'bitAnd': case 'bitOr': case 'bitXor': case 'lshift': case 'rshift':
    case 'floorDiv': case 'ldexp':
    case 'neg': case 'not': case 'bitNot':
    case 'sqrt': case 'abs': case 'floor': case 'ceil': case 'round':
    case 'floatExponent': case 'toInt': case 'toBool': case 'toFloat':
    case 'clamp': case 'select': case 'index': case 'arraySet':
      node.args.forEach(visit); return
    case 'zeros':
      visit(node.count); return
    case 'tag':
      node.payload.forEach(p => visit(p.value)); return
    case 'match':
      visit(node.scrutinee); node.arms.forEach(a => visit(a.body)); return
    case 'let':
      node.binders.forEach(b => visit(b.value)); visit(node.in); return
    case 'fold': case 'scan':
      visit(node.over); visit(node.init); visit(node.body); return
    case 'generate':
      visit(node.count); visit(node.body); return
    case 'iterate': case 'chain':
      visit(node.count); visit(node.init); visit(node.body); return
    case 'map2':
      visit(node.over); visit(node.body); return
    case 'zipWith':
      visit(node.a); visit(node.b); visit(node.body); return
  }
}

/** Pull the (single) output expression for assertion. */
function outputExpr(prog: ResolvedProgram, name = 'out'): ResolvedExpr {
  for (const a of prog.body.assigns) {
    if (a.op !== 'outputAssign') continue
    // Post-migration: a.target is OutputIdx (a number) or {kind:'dac'}.
    if (typeof a.target === 'number' && prog.ports.outputs[a.target]?.name === name) return a.expr
  }
  throw new Error(`outputExpr: no assign for output '${name}'`)
}

const COMBINATORS = ['let', 'fold', 'scan', 'generate', 'iterate', 'chain', 'map2', 'zipWith'] as const

// ─────────────────────────────────────────────────────────────
// Identity on programs that exercise no combinators
// ─────────────────────────────────────────────────────────────

describe('arrayLower — identity on plain programs', () => {
  test('no combinators, no array ops → input returned by identity', () => {
    const p = elab('program X(a: float) -> (out: float) { out = a + 1 }')
    expect(arrayLower(p)).toBe(p)
  })

  test('arithmetic-only register update returns input by identity', () => {
    const p = elab(`
      program X(a: float) -> (out: float) {
        reg s: float = 0
        out = s + a
        next s = a * 0.5
      }
    `)
    expect(arrayLower(p)).toBe(p)
  })
})

// ─────────────────────────────────────────────────────────────
// let — sequential let* binding
// ─────────────────────────────────────────────────────────────

describe('arrayLower — let', () => {
  test('single binding substitutes value into body', () => {
    const p = elab(`
      program X(a: float) -> (out: float) {
        out = let { y: a + 1 } in y * 2
      }
    `)
    const out = arrayLower(p)
    expect(findOps(out, [...COMBINATORS, 'bindingRef'])).toEqual([])
  })

  test('let* sequential: later binding sees earlier (Tanh-style)', () => {
    // Tanh's body: let { c: clamp(x,-3,3); c2: c*c } in c * (27 + c2) / (27 + 9*c2)
    const p = elab(`
      program X(x: float) -> (out: float) {
        out = let { c: clamp(x, -3, 3); c2: c * c } in c * (27 + c2) / (27 + 9 * c2)
      }
    `)
    const out = arrayLower(p)
    expect(findOps(out, [...COMBINATORS, 'bindingRef'])).toEqual([])
  })

  test('nested let with shadowing inner binder name (decl identity wins)', () => {
    // Outer `y` binds `a`; inner `y` binds `99`. The body's `y` must see
    // the inner binder, not the outer. Decl-identity substitution makes
    // shadowing structurally impossible to get wrong.
    const p = elab(`
      program X(a: float) -> (out: float) {
        out = let { y: a } in let { y: 99 } in y
      }
    `)
    const out = arrayLower(p)
    // After lowering, the output should be the literal 99 — both `y`
    // binders are eliminated and the inner-most one wins.
    expect(outputExpr(out)).toBe(99)
  })

  test('let inside a delay update lowers correctly', () => {
    // AllpassDelay: `next s = let { y: coeff * input + s } in input - coeff * y`
    const p = elab(`
      program X(input: float, coeff: float = 0.5) -> (out: float) {
        reg s: float = 0
        out = coeff * input + s
        next s = let { y: coeff * input + s } in input - coeff * y
      }
    `)
    const out = arrayLower(p)
    expect(findOps(out, [...COMBINATORS, 'bindingRef'])).toEqual([])
  })
})

// ─────────────────────────────────────────────────────────────
// fold — left fold to scalar
// ─────────────────────────────────────────────────────────────

describe('arrayLower — fold', () => {
  test('fold over inline array reduces to a scalar tree', () => {
    const p = elab(`
      program X() -> (out: float) {
        out = fold([1.0, 2.0, 3.0], 0, (a, c) => a + c)
      }
    `)
    const out = arrayLower(p)
    expect(findOps(out, [...COMBINATORS, 'bindingRef'])).toEqual([])
    // Result should be a tree of `add` ops over the literals.
    const e = outputExpr(out)
    expect(typeof e).toBe('object')
    if (typeof e === 'object' && !Array.isArray(e) && e !== null) {
      expect(e.op).toBe('add')
    }
  })

  test('fold inside let inside a body resolves cleanly (Sin-style)', () => {
    // Sin's body uses `fold([...], 0, (a, c) => c + a * r2)` inside a
    // let binding. This is the most-nested combinator pattern in the
    // stdlib.
    const p = elab(`
      program X(x: float) -> (out: float) {
        out = let { r2: x * x; poly: fold([1.0, 2.0, 3.0], 0, (a, c) => c + a * r2) } in poly
      }
    `)
    const out = arrayLower(p)
    expect(findOps(out, [...COMBINATORS, 'bindingRef'])).toEqual([])
  })

  test('fold of single element gives initial body application', () => {
    const p = elab(`
      program X() -> (out: float) {
        out = fold([42.0], 0, (a, c) => a + c)
      }
    `)
    const out = arrayLower(p)
    expect(findOps(out, [...COMBINATORS, 'bindingRef'])).toEqual([])
    // Result: 0 + 42 → an add(0, 42) tree (legacy doesn't constant-fold).
    const e = outputExpr(out)
    if (typeof e === 'object' && !Array.isArray(e) && e !== null) {
      expect(e.op).toBe('add')
    }
  })

  test('fold with two-element array yields nested add', () => {
    const p = elab(`
      program X() -> (out: float) {
        out = fold([1.0, 2.0], 5, (a, c) => a + c)
      }
    `)
    const out = arrayLower(p)
    expect(findOps(out, [...COMBINATORS, 'bindingRef'])).toEqual([])
  })
})

// ─────────────────────────────────────────────────────────────
// scan
// ─────────────────────────────────────────────────────────────

describe('arrayLower — scan', () => {
  test('scan over inline array produces an array of intermediates', () => {
    const p = elab(`
      program X() -> (out: float[3]) {
        out = scan([1.0, 2.0, 3.0], 0, (a, c) => a + c)
      }
    `)
    const out = arrayLower(p)
    expect(findOps(out, [...COMBINATORS, 'bindingRef'])).toEqual([])
    // Result is an array literal of length 3.
    const e = outputExpr(out)
    expect(Array.isArray(e)).toBe(true)
    if (Array.isArray(e)) expect(e.length).toBe(3)
  })
})

// ─────────────────────────────────────────────────────────────
// generate
// ─────────────────────────────────────────────────────────────

describe('arrayLower — generate', () => {
  test('generate with constant count emits an inline array', () => {
    const p = elab(`
      program X() -> (out: float[4]) {
        out = generate(4, (i) => i * 2.0)
      }
    `)
    const out = arrayLower(p)
    expect(findOps(out, [...COMBINATORS, 'bindingRef'])).toEqual([])
    const e = outputExpr(out)
    expect(Array.isArray(e)).toBe(true)
    if (Array.isArray(e)) expect(e.length).toBe(4)
  })

  test('count = 0 produces an empty array', () => {
    const p = elab(`
      program X() -> (out: float[0]) {
        out = generate(0, (i) => i)
      }
    `)
    const out = arrayLower(p)
    const e = outputExpr(out)
    expect(Array.isArray(e)).toBe(true)
    if (Array.isArray(e)) expect(e.length).toBe(0)
  })
})

// ─────────────────────────────────────────────────────────────
// iterate
// ─────────────────────────────────────────────────────────────

describe('arrayLower — iterate', () => {
  test('iterate emits [init, f(init), f(f(init)), ...]', () => {
    const p = elab(`
      program X() -> (out: float[3]) {
        out = iterate(3, 1.0, (x) => x * 2)
      }
    `)
    const out = arrayLower(p)
    expect(findOps(out, [...COMBINATORS, 'bindingRef'])).toEqual([])
    const e = outputExpr(out)
    expect(Array.isArray(e)).toBe(true)
    if (Array.isArray(e)) {
      expect(e.length).toBe(3)
      // First element is the init unchanged.
      expect(e[0]).toBe(1)
    }
  })
})

// ─────────────────────────────────────────────────────────────
// chain
// ─────────────────────────────────────────────────────────────

describe('arrayLower — chain', () => {
  test('chain of n applications threads through an accumulator (scalar result)', () => {
    const p = elab(`
      program X(a: float) -> (out: float) {
        out = chain(3, a, (x) => x + 1)
      }
    `)
    const out = arrayLower(p)
    expect(findOps(out, [...COMBINATORS, 'bindingRef'])).toEqual([])
    // Result is a scalar (not an array); should be an add tree.
    const e = outputExpr(out)
    expect(Array.isArray(e)).toBe(false)
    if (typeof e === 'object' && !Array.isArray(e) && e !== null) {
      expect(e.op).toBe('add')
    }
  })
})

// ─────────────────────────────────────────────────────────────
// map2
// ─────────────────────────────────────────────────────────────

describe('arrayLower — map2', () => {
  test('map2 over inline array produces same-length output', () => {
    const p = elab(`
      program X() -> (out: float[3]) {
        out = map2([1.0, 2.0, 3.0], (e) => e * 2)
      }
    `)
    const out = arrayLower(p)
    expect(findOps(out, [...COMBINATORS, 'bindingRef'])).toEqual([])
    const e = outputExpr(out)
    expect(Array.isArray(e)).toBe(true)
    if (Array.isArray(e)) expect(e.length).toBe(3)
  })
})

// ─────────────────────────────────────────────────────────────
// zipWith
// ─────────────────────────────────────────────────────────────

describe('arrayLower — zipWith', () => {
  test('zipWith of two same-length arrays produces same-length output', () => {
    const p = elab(`
      program X() -> (out: float[3]) {
        out = zipWith([1.0, 2.0, 3.0], [10.0, 20.0, 30.0], (x, y) => x + y)
      }
    `)
    const out = arrayLower(p)
    expect(findOps(out, [...COMBINATORS, 'bindingRef'])).toEqual([])
    const e = outputExpr(out)
    expect(Array.isArray(e)).toBe(true)
    if (Array.isArray(e)) expect(e.length).toBe(3)
  })

  test('zipWith of different-length arrays uses min length', () => {
    const p = elab(`
      program X() -> (out: float[2]) {
        out = zipWith([1.0, 2.0, 3.0], [10.0, 20.0], (x, y) => x + y)
      }
    `)
    const out = arrayLower(p)
    const e = outputExpr(out)
    expect(Array.isArray(e)).toBe(true)
    if (Array.isArray(e)) expect(e.length).toBe(2)
  })
})

// ─────────────────────────────────────────────────────────────
// zeros
// ─────────────────────────────────────────────────────────────

describe('arrayLower — zeros', () => {
  test('zeros{N} with literal N expands to inline [0, ..., 0]', () => {
    const p = elab(`
      program X() -> (out: float) {
        reg buf = zeros(4)
        out = buf[0]
        next buf = buf
      }
    `)
    const out = arrayLower(p)
    // The reg's init should now be an inline array of four zeros.
    const reg = out.body.decls.find(d => d.op === 'regDecl')
    expect(reg?.op).toBe('regDecl')
    if (reg?.op === 'regDecl') {
      expect(reg.init).toEqual([0, 0, 0, 0])
    }
  })
})

// ─────────────────────────────────────────────────────────────
// arraySet — preserved (not lowered)
// ─────────────────────────────────────────────────────────────

describe('arrayLower — arraySet survives', () => {
  test('arraySet in a delay update survives; its children are lowered', () => {
    // Delay-style: `next buf = arraySet(buf, sampleIndex() % 4, x)`.
    // The resulting program should still contain an `arraySet` op
    // (we don't lower it; load.ts emits it directly).
    const p = elab(`
      program X(x: float) -> (out: float) {
        reg buf = zeros(4)
        out = buf[sampleIndex() % 4]
        next buf = arraySet(buf, sampleIndex() % 4, x)
      }
    `)
    const out = arrayLower(p)
    expect(findOps(out, [...COMBINATORS, 'bindingRef'])).toEqual([])
    expect(findOps(out, ['arraySet']).length).toBeGreaterThan(0)
  })
})

// ─────────────────────────────────────────────────────────────
// Sharing / DAG preservation
// ─────────────────────────────────────────────────────────────

describe('arrayLower — sharing discipline', () => {
  test('two output assignments referencing the same shared input do not exponentially blow up', () => {
    // A combinator whose body uses an outer-scope expression (the input
    // `a`) multiple times. Without memoization, chained combinators
    // duplicate the input subtree per iteration; with the memo, the
    // subtree is preserved by reference where possible. We don't probe
    // identity here (combinator unrolling does produce fresh nodes per
    // iteration by design); we just check the pass terminates and
    // produces a well-formed program.
    const p = elab(`
      program X(a: float) -> (out: float) {
        out = chain(8, a, (x) => x + a)
      }
    `)
    const out = arrayLower(p)
    expect(findOps(out, [...COMBINATORS, 'bindingRef'])).toEqual([])
  })

  test('Phaser16-style chain pattern lowers without exponential blowup', () => {
    // A let-bound expression used inside a generate body — the kind of
    // pattern that exposes shadowing/sharing bugs in the legacy.
    const p = elab(`
      program X(a: float) -> (out: float[16]) {
        out = generate(16, (i) => let { y: a + i } in y * y)
      }
    `)
    const out = arrayLower(p)
    expect(findOps(out, [...COMBINATORS, 'bindingRef'])).toEqual([])
    const e = outputExpr(out)
    expect(Array.isArray(e)).toBe(true)
    if (Array.isArray(e)) expect(e.length).toBe(16)
  })
})

// ─────────────────────────────────────────────────────────────
// Combinators inside instance inputs and delay updates
// ─────────────────────────────────────────────────────────────

describe('arrayLower — combinators on every site', () => {
  test('combinator inside register init lowers', () => {
    const p = elab(`
      program X() -> (out: float) {
        reg s: float = let { x: 1; y: 2 } in x + y
        out = s
        next s = s
      }
    `)
    const out = arrayLower(p)
    expect(findOps(out, [...COMBINATORS, 'bindingRef'])).toEqual([])
  })

  test('combinator inside delay init and update lowers', () => {
    // delay <name> = <update_expr> init <init_expr>
    const p = elab(`
      program X() -> (out: float) {
        delay d = let { x: 1 } in x init 0
        out = d
      }
    `)
    const out = arrayLower(p)
    expect(findOps(out, [...COMBINATORS, 'bindingRef'])).toEqual([])
  })
})
