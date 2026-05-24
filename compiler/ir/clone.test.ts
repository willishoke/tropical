/**
 * clone.test.ts — coverage for the resolved-program graph cloner.
 *
 * The cloner is foundation for Phase C3 (specialize) and C5
 * (inlineInstances). These tests verify three properties:
 *   1. Decls are cloned exactly once (Map<old,new> dedup works).
 *   2. Within the clone, every Ref points at the cloned decl.
 *   3. Sum-type defs and variants are SHARED, not cloned, so
 *      `MatchArm.variant` retains `===` identity across the clone
 *      boundary.
 */

import { describe, test, expect } from 'bun:test'
import { parseProgram } from '../parse/declarations.js'
import { elaborate } from './elaborator.js'
import { cloneResolvedProgram } from './clone.js'
import type {
  ResolvedProgram, RegDecl, InputDecl, OutputDecl,
  RegRef, InputRef,
  SumTypeDef, Match, OutputAssign,
} from './nodes.js'

function clab(src: string): { orig: ResolvedProgram; copy: ResolvedProgram } {
  const orig = elaborate(parseProgram(src))
  const copy = cloneResolvedProgram(orig)
  return { orig, copy }
}

describe('clone — basic structure', () => {
  test('empty-body program clones to a deep copy', () => {
    const { orig, copy } = clab('program X(a: float) -> (out: float) { out = a }')
    expect(copy).not.toBe(orig)
    expect(copy.name).toBe('X')
    expect(copy.ports.inputs).not.toBe(orig.ports.inputs)
    expect(copy.ports.inputs.length).toBe(1)
    expect(copy.ports.inputs[0]).not.toBe(orig.ports.inputs[0])
    expect(copy.ports.inputs[0].name).toBe('a')
    expect(copy.body).not.toBe(orig.body)
  })

  test('clone is idempotent for trivial programs', () => {
    const { orig } = clab('program X(a: float) -> (out: float) { out = a }')
    const c1 = cloneResolvedProgram(orig)
    const c2 = cloneResolvedProgram(c1)
    expect(JSON.stringify(stripIdentity(c2))).toBe(JSON.stringify(stripIdentity(c1)))
  })
})

describe('clone — reference identity (within-clone)', () => {
  test('OutputAssign expr (InputRef) points at cloned InputDecl', () => {
    const { copy } = clab('program X(a: float) -> (out: float) { out = a }')
    const inputDecl = copy.ports.inputs[0]
    const assign = copy.body.assigns[0] as OutputAssign
    const ref = assign.expr as InputRef
    expect(ref.op).toBe('inputRef')
    expect(copy.ports.inputs[ref.idx]).toBe(inputDecl)
  })

  test('self-referential RegDecl: update on decl points at the regDecl in the clone', () => {
    // Post-Phase-0a: next-update folds into RegDecl.update at elaboration.
    const { copy } = clab(`
      program X() -> (out: float) {
        reg s: float = 0
        out = s
        next s = s
      }
    `)
    const regDecl = copy.body.decls[0] as RegDecl
    const out  = copy.body.assigns[0] as OutputAssign
    expect(out.expr).toMatchObject({ op: 'regRef' })
    expect(copy.regs[(out.expr as RegRef).idx]).toBe(regDecl)
    expect(regDecl.update).toBeDefined()
    expect(copy.regs[(regDecl.update as RegRef).idx]).toBe(regDecl)
  })

  test('multiple RegRefs to same RegDecl all point at the same cloned decl', () => {
    const { copy } = clab(`
      program X(a: float) -> (out: float) {
        reg s: float = 0
        out = s + s + s
        next s = a
      }
    `)
    const regDecl = copy.body.decls[0] as RegDecl
    const out = copy.body.assigns[0] as OutputAssign
    // Walk the (s + s + s) tree; collect every regRef.idx → resolve to RegDecl.
    const refs: RegDecl[] = []
    function walk(n: unknown): void {
      if (typeof n !== 'object' || n === null) return
      if (Array.isArray(n)) { n.forEach(walk); return }
      const o = n as { op?: string; idx?: number; args?: unknown }
      if (o.op === 'regRef') { refs.push(copy.regs[o.idx as number]); return }
      if (Array.isArray(o.args)) o.args.forEach(walk)
    }
    walk(out.expr)
    expect(refs.length).toBe(3)
    for (const r of refs) expect(r).toBe(regDecl)
  })

  test('delay-form RegDecl clone preserves RegRef identity', () => {
    // Post-Phase-0a: `delay z = u init v` desugars to a RegDecl with
    // update populated. Reads of z are RegRefs (by idx into copy.regs).
    const { copy } = clab(`
      program X(x: float) -> (out: float) {
        delay z = x init 0
        out = z
      }
    `)
    const regDecl = copy.body.decls[0] as RegDecl
    const out = copy.body.assigns[0] as OutputAssign
    const ref = out.expr as RegRef
    expect(ref.op).toBe('regRef')
    expect(copy.regs[ref.idx]).toBe(regDecl)
    expect(regDecl.update).toBeDefined()
  })

  test('clone breaks decl identity vs original', () => {
    const { orig, copy } = clab(`
      program X(a: float) -> (out: float) {
        reg s: float = 0
        out = s
        next s = a
      }
    `)
    const origReg = orig.body.decls[0] as RegDecl
    const copyReg = copy.body.decls[0] as RegDecl
    expect(copyReg).not.toBe(origReg)
    const copyOut = copy.body.assigns[0] as OutputAssign
    expect((copyOut.expr as RegRef).decl).not.toBe(origReg)
  })
})

describe('clone — sum-type sharing', () => {
  test('SumTypeDef and SumVariant are shared (===) across clone boundary', () => {
    const src = `
      program X(trig: float) -> (out: float) {
        enum S { A, B(v: float) }
        delay s: S = match s { A => B { v: 1 }, B { v: w } => A { } } init A { }
        out = match s { A => 0, B { v: w } => w }
      }
    `
    const { orig, copy } = clab(src)
    const origSum = orig.ports.typeDefs[0] as SumTypeDef
    const copySum = copy.ports.typeDefs[0] as SumTypeDef
    expect(copySum).toBe(origSum)
    expect(copySum.variants[0]).toBe(origSum.variants[0])
    expect(copySum.variants[1]).toBe(origSum.variants[1])

    // The MatchArm.variant must still === the original variants.
    const out = copy.body.assigns[0] as OutputAssign
    const m = out.expr as Match
    expect(m.op).toBe('match')
    expect(m.type).toBe(origSum)
    expect(m.arms[0].variant).toBe(origSum.variants[0])
    expect(m.arms[1].variant).toBe(origSum.variants[1])
  })
})

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────

/** Strip Decl identity from a tree for structural equality assertions:
 *  replace every decl object with a `{__decl: name}` placeholder. */
function stripIdentity(prog: ResolvedProgram): unknown {
  const seen = new WeakMap<object, unknown>()
  function strip(v: unknown): unknown {
    if (v === null || typeof v !== 'object') return v
    if (seen.has(v as object)) return seen.get(v as object)
    if (Array.isArray(v)) {
      const out: unknown[] = []
      seen.set(v as object, out)
      v.forEach(x => out.push(strip(x)))
      return out
    }
    const obj = v as Record<string, unknown>
    const out: Record<string, unknown> = {}
    seen.set(v as object, out)
    for (const [k, val] of Object.entries(obj)) {
      out[k] = strip(val)
    }
    return out
  }
  return strip(prog)
}
