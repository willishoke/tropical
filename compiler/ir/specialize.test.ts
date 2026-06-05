/**
 * specialize.test.ts — coverage for the Phase C3 clone-and-rewrite
 * specializer.
 *
 * Properties tested:
 *   1. Default fill-in / empty-args / extra-args / missing-required.
 *   2. Single type-param substitution: shape dim and TypeParamRef in
 *      expression position both collapse to the same integer.
 *   3. Decl identity is per-specialization: two specializations of the
 *      same template produce distinct RegDecl objects.
 *   4. Sum-type defs and variants are SHARED across the clone boundary
 *      (Phase C4 sum_lower compares by `===`).
 *   5. InstanceDecl.typeArgs survive the clone (literal values pass
 *      through; the param ref points at the cloned target's typeParam).
 *   6. The result has `typeParams: []` and no TypeParamRef referencing
 *      the original outer params remains.
 */

import { describe, test, expect } from 'bun:test'
import { parseProgram } from '../parse/declarations.js'
import { elaborate, type ExternalProgramResolver } from './elaborator.js'
import { specializeProgram } from './specialize.js'
import type {
  ResolvedProgram, TypeParamDecl,
  RegDecl, InstanceDecl,
  SumTypeDef, Match, OutputAssign,
} from './nodes.js'

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────

function elab(src: string, resolver?: ExternalProgramResolver): ResolvedProgram {
  return elaborate(parseProgram(src), resolver)
}

/** Build a typeArgs map by param name (more readable than threading
 *  TypeParamDecl objects through the test). */
function args(prog: ResolvedProgram, byName: Record<string, number>): Map<TypeParamDecl, number> {
  const m = new Map<TypeParamDecl, number>()
  for (const [name, value] of Object.entries(byName)) {
    const decl = prog.typeParams.find(p => p.name === name)
    if (!decl) throw new Error(`test: program '${prog.name}' has no type-param '${name}'`)
    m.set(decl, value)
  }
  return m
}

/** Walk a resolved expression / decl tree and return every TypeParamRef
 *  encountered (its decl). Used to assert "no outer-typeParam refs
 *  remain post-specialize." */
function collectTypeParamRefs(prog: ResolvedProgram): TypeParamDecl[] {
  const refs: TypeParamDecl[] = []
  const seen = new WeakSet<object>()
  function walk(v: unknown): void {
    if (v === null || typeof v !== 'object') return
    if (seen.has(v as object)) return
    seen.add(v as object)
    if (Array.isArray(v)) { v.forEach(walk); return }
    const o = v as { op?: string; decl?: unknown }
    if (o.op === 'typeParamRef') {
      refs.push(o.decl as TypeParamDecl)
      return
    }
    for (const k of Object.keys(v as Record<string, unknown>)) {
      walk((v as Record<string, unknown>)[k])
    }
  }
  walk(prog)
  return refs
}

// ─────────────────────────────────────────────────────────────
// Validation: empty / extra / missing
// ─────────────────────────────────────────────────────────────

describe('specialize — input validation', () => {
  test('empty typeArgs on a non-generic program returns input by identity', () => {
    const p = elab('program X(a: float) -> (out: float) { out = a }')
    expect(specializeProgram(p, new Map())).toBe(p)
  })

  test('extra typeArg (not declared on the program) throws', () => {
    const p = elab('program X(a: float) -> (out: float) { out = a }')
    const fakeDecl: TypeParamDecl = { op: 'typeParamDecl', name: 'N' }
    const m = new Map<TypeParamDecl, number>([[fakeDecl, 4]])
    expect(() => specializeProgram(p, m)).toThrow(/not a declared type-param/)
  })

  test('missing required typeArg with no default throws', () => {
    const p = elab(`program X<N: int>(x: float) -> (out: float) { out = x }`)
    expect(() => specializeProgram(p, new Map())).toThrow(/missing required type-arg 'N'/)
  })

  test('default fill-in: missing typeArg uses declared default', () => {
    const p = elab(`
      program X<N: int = 8>(x: float) -> (out: float) {
        reg buf = zeros(N)
        out = buf[0]
        next buf = arraySet(buf, 0, x)
      }
    `)
    const c = specializeProgram(p, new Map())
    expect(c.typeParams).toEqual([])
    const reg = c.body.decls[0] as RegDecl
    // RegDecl.init is `{op: 'zeros', count: <number>}`. The count must
    // have collapsed to the default (8).
    const init = reg.init as { op: 'zeros'; count: number }
    expect(init.op).toBe('zeros')
    expect(init.count).toBe(8)
  })
})

// ─────────────────────────────────────────────────────────────
// Substitution: shape dims + TypeParamRef in expr position
// ─────────────────────────────────────────────────────────────

describe('specialize — substitution', () => {
  const SRC = `
    program Delay<N: int = 44100>(x = 0) -> (y) {
      reg buf = zeros(N)
      y = buf[sampleIndex() % N]
      next buf = arraySet(buf, sampleIndex() % N, x)
    }
  `

  test('TypeParamRef in expression position becomes a numeric literal', () => {
    const p = elab(SRC)
    const c = specializeProgram(p, args(p, { N: 8 }))
    // No outer-typeParam refs should remain anywhere in the clone.
    expect(collectTypeParamRefs(c)).toEqual([])
  })

  test('shape dim TypeParamDecl collapses to integer in RegDecl.init zeros count', () => {
    const p = elab(SRC)
    const c = specializeProgram(p, args(p, { N: 8 }))
    const reg = c.body.decls[0] as RegDecl
    const init = reg.init as { op: 'zeros'; count: number }
    expect(init.op).toBe('zeros')
    expect(init.count).toBe(8)
  })

  test('cloned program has empty typeParams', () => {
    const p = elab(SRC)
    expect(p.typeParams.length).toBe(1)
    const c = specializeProgram(p, args(p, { N: 8 }))
    expect(c.typeParams).toEqual([])
  })
})

// ─────────────────────────────────────────────────────────────
// Decl identity: per-specialization freshness
// ─────────────────────────────────────────────────────────────

describe('specialize — decl identity per specialization', () => {
  const SRC = `
    program Delay<N: int = 44100>(x = 0) -> (y) {
      reg buf = zeros(N)
      y = buf[0]
      next buf = arraySet(buf, 0, x)
    }
  `

  test('same template specialized twice with same N produces distinct programs and decls', () => {
    const p = elab(SRC)
    const a = specializeProgram(p, args(p, { N: 8 }))
    const b = specializeProgram(p, args(p, { N: 8 }))
    expect(a).not.toBe(b)
    expect(a.body.decls[0]).not.toBe(b.body.decls[0])
    // Caching is the loader's job (Phase C7); this function must produce
    // a fresh clone every call.
  })

  test('different N values produce structurally distinct register shapes', () => {
    const p = elab(SRC)
    const a = specializeProgram(p, args(p, { N: 8 }))
    const b = specializeProgram(p, args(p, { N: 4 }))
    const aReg = a.body.decls[0] as RegDecl
    const bReg = b.body.decls[0] as RegDecl
    expect(aReg).not.toBe(bReg)
    const aInit = aReg.init as { op: 'zeros'; count: number }
    const bInit = bReg.init as { op: 'zeros'; count: number }
    expect(aInit.count).toBe(8)
    expect(bInit.count).toBe(4)
  })
})

// ─────────────────────────────────────────────────────────────
// Sum-type sharing: variants must remain `===`
// ─────────────────────────────────────────────────────────────

describe('specialize — sum-type variant identity preserved', () => {
  test('SumTypeDef and SumVariant are shared (===) between original and specialization', () => {
    // Generic program with a sum type. The match arm's `variant` must
    // still === the original variant after specialization, otherwise
    // Phase C4 sum_lower (which compares by `===`) breaks.
    const src = `
      program X<N: int = 4>(t: float) -> (out: float) {
        enum S { A, B(v: float) }
        delay s: S = match s { A => B { v: 1 }, B { v: w } => A { } } init A { }
        out = match s { A => 0, B { v: w } => w }
      }
    `
    const orig = elab(src)
    const copy = specializeProgram(orig, args(orig, { N: 8 }))
    const origSum = orig.ports.typeDefs[0] as SumTypeDef
    const copySum = copy.ports.typeDefs[0] as SumTypeDef
    expect(copySum).toBe(origSum)
    expect(copySum.variants[0]).toBe(origSum.variants[0])
    expect(copySum.variants[1]).toBe(origSum.variants[1])

    // The Match in the assign points at the same SumTypeDef and
    // its arms still reference the original variants by `===`.
    const out = copy.body.assigns[0] as OutputAssign
    const m = out.expr as Match
    expect(m.op).toBe('match')
    expect(m.type).toBe(origSum)
    expect(m.arms[0].variant).toBe(origSum.variants[0])
    expect(m.arms[1].variant).toBe(origSum.variants[1])
  })
})

// ─────────────────────────────────────────────────────────────
// InstanceDecl.typeArgs: numeric values pass through; param refs
// point at the cloned (or shared) target's TypeParamDecl.
// ─────────────────────────────────────────────────────────────

describe('specialize — InstanceDecl.typeArgs survive substitution', () => {
  test('outer-program instantiates a generic Delay; instance.typeArgs literal preserved', () => {
    // Today the parser only admits numeric literals for type-args at
    // the call site (no `<N=N>` forwarding syntax in the surface language). So the
    // test exercises the literal pass-through case explicitly.
    const delaySrc = `
      program Delay<N: int = 44100>(x = 0) -> (y) {
        reg buf = zeros(N)
        y = buf[0]
        next buf = arraySet(buf, 0, x)
      }
    `
    const delay = elab(delaySrc)
    const resolver: ExternalProgramResolver = name => name === 'Delay' ? delay : undefined
    const outerSrc = `
      program Outer<M: int = 2>(s: float) -> (out: float) {
        del = Delay<N=8>(x: s)
        out = del.y
      }
    `
    const outer = elab(outerSrc, resolver)
    const c = specializeProgram(outer, args(outer, { M: 4 }))
    const inst = c.body.decls[0] as InstanceDecl
    expect(inst.op).toBe('instanceDecl')
    expect(inst.typeArgs.length).toBe(1)
    expect(inst.typeArgs[0].value).toBe(8)
    // The param ref on the typeArg is a TypeParamIdx into the
    // instance's referenced program's typeParams[]. Resolution of
    // the target type goes through the enclosing program's
    // programRegistry by typeKey (Phase 4b: no `.type` pointer).
    const instType = c.programRegistry.get(inst.typeKey)!
    expect(instType.typeParams.length).toBe(1)
    expect(instType.typeParams[inst.typeArgs[0].param]).toBe(instType.typeParams[0])
    expect(instType.typeParams[inst.typeArgs[0].param].name).toBe('N')
  })
})
