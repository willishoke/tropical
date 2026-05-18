/**
 * strata.test.ts — scaffolding correctness for the C1 strata stubs.
 *
 * Each stub passes a "trivial" program through unchanged (the program
 * doesn't exercise the feature the stub eventually lowers) and throws
 * on programs that require its full implementation. These tests
 * sanity-check both branches.
 */

import { describe, test, expect } from 'bun:test'
import { parseProgram } from '../parse/declarations.js'
import { elaborate } from './elaborator.js'
import { specializeProgram } from './specialize.js'
import { sumLower } from './sum_lower.js'
import { traceCycles } from './trace_cycles.js'
import { inlineInstances } from './inline_instances.js'
import { arrayLower } from './array_lower.js'
import { strataPipeline } from './strata.js'
import type { ResolvedProgram, TypeParamDecl } from './nodes.js'

function elab(src: string): ResolvedProgram {
  return elaborate(parseProgram(src))
}

/** Walk a resolved object graph (cycle-safe) and return every `op` field
 *  matching `targets`. Used to assert that a pass eliminated certain ops
 *  (e.g. sumLower drops every `tag`/`match`). */
function findOps(root: unknown, targets: string[]): string[] {
  const set = new Set(targets)
  const out: string[] = []
  const seen = new WeakSet<object>()
  const walk = (v: unknown): void => {
    if (v === null || typeof v !== 'object') return
    if (seen.has(v as object)) return
    seen.add(v as object)
    if (Array.isArray(v)) { v.forEach(walk); return }
    const o = v as Record<string, unknown>
    if (typeof o.op === 'string' && set.has(o.op)) out.push(o.op)
    for (const k of Object.keys(o)) walk(o[k])
  }
  walk(root)
  return out
}

const TRIVIAL = 'program X(a: float) -> (out: float) { reg s: float = 0  out = s + a  next s = a }'

describe('strata — pass-through on trivial programs', () => {
  test('specialize: empty type-args returns input unchanged', () => {
    const p = elab(TRIVIAL)
    expect(specializeProgram(p, new Map())).toBe(p)
  })

  test('sumLower: no sums returns input unchanged', () => {
    const p = elab(TRIVIAL)
    expect(sumLower(p)).toBe(p)
  })

  test('traceCycles: stub passes through unchanged', () => {
    const p = elab(TRIVIAL)
    expect(traceCycles(p)).toBe(p)
  })

  test('inlineInstances: no instances returns input unchanged', () => {
    const p = elab(TRIVIAL)
    expect(inlineInstances(p)).toBe(p)
  })

  test('arrayLower: no combinators or arrays returns input unchanged', () => {
    const p = elab(TRIVIAL)
    expect(arrayLower(p)).toBe(p)
  })

  test('strataPipeline: trivial program threads through all five strata', () => {
    const p = elab(TRIVIAL)
    expect(strataPipeline(p)).toBe(p)
  })
})

describe('strata — throws on unsupported features', () => {
  test('specialize: type-arg keyed by an undeclared TypeParamDecl throws', () => {
    // C3 implementation: passing an arg whose key is not one of the
    // program's declared type-params is an error (replaces the C1 stub
    // behavior of "any non-empty args throws").
    const p = elab(TRIVIAL)
    const fakeDecl: TypeParamDecl = { op: 'typeParamDecl', name: 'N' }
    const args = new Map<TypeParamDecl, number>([[fakeDecl, 4]])
    expect(() => specializeProgram(p, args)).toThrow(/not a declared type-param/)
  })

  test('sumLower: program declaring an unused sum type returns unchanged', () => {
    // A sum type that isn't used by any delay/match/tag is an identity
    // for sumLower: nothing to decompose, no expressions to rewrite.
    const p = elab(`
      program X(t: float) -> (out: float) {
        enum S { A, B }
        out = 0
      }
    `)
    expect(sumLower(p)).toBe(p)
  })

  test('sumLower: lowers a Toggle-style match over a sum-typed delay', () => {
    const p = elab(`
      program Toggle() -> (value: float) {
        enum St { Off, On }
        delay state: St = match state { Off => On { }, On => Off { } } init Off { }
        value = match state { Off => 0.0, On => 1.0 }
      }
    `)
    const out = sumLower(p)
    // No tag/match expressions remain anywhere in the rewritten body.
    expect(findOps(out, ['match', 'tag'])).toEqual([])
    // The sum-typed delay decomposes into a single tag-slot delay
    // (no payload variants → just the discriminator).
    expect(out.body.decls.length).toBe(1)
    const d = out.body.decls[0]
    expect(d.op).toBe('regDecl')
    if (d.op === 'regDecl') expect(d.name).toBe('state#tag')
  })

  test('arrayLower: program with let combinator returns a let-free program', () => {
    // Post-C6: arrayLower unrolls the let into the body, no `let` op
    // and no `bindingRef` remain.
    const p = elab(`
      program X(a: float) -> (out: float) {
        out = let { y: a + 1 } in y * 2
      }
    `)
    const out = arrayLower(p)
    expect(findOps(out, ['let', 'bindingRef'])).toEqual([])
  })

  test('arrayLower: program with fold over inline array returns a fold-free program', () => {
    // Post-C6: arrayLower unrolls the fold; no `fold` op and no
    // `bindingRef` remain. The inline array literal `[1, 2, 3]` is
    // consumed by the fold during unrolling — the resulting body is
    // a chain of `add` ops over numeric literals.
    const p = elab(`
      program X() -> (out: float) {
        out = fold([1.0, 2.0, 3.0], 0, (a, c) => a + c)
      }
    `)
    const out = arrayLower(p)
    expect(findOps(out, ['fold', 'bindingRef'])).toEqual([])
  })

  test('inlineInstances: program with InstanceDecl returns an instance-free program', () => {
    // Post-C5: inlineInstances splices the inner body into the outer
    // and removes the InstanceDecl. The previously-skip path now
    // exercises the inliner end-to-end on a one-deep program.
    const p = elab(`
      program X(a: float) -> (out: float) {
        program Inner(x: float) -> (y: float) { y = x + 1 }
        inst = Inner(x: a)
        out = inst.y
      }
    `)
    const out = inlineInstances(p)
    // No InstanceDecl survives.
    for (const d of out.body.decls) expect(d.op).not.toBe('instanceDecl')
  })
})
