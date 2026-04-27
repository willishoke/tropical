/**
 * elaborator.test.ts — coverage for the parser → resolved-IR elaborator.
 *
 * The elaborator's job is to turn the parsed tree (with NameRefNode
 * placeholders) into a graph where every reference is a direct decl
 * reference. These tests verify both:
 *   - resolution: each reference resolves to the correct decl
 *   - reference identity: regRef.decl === <the actual RegDecl> (not just
 *     a structurally-equal copy)
 */

import { describe, test, expect } from 'bun:test'
import { parseProgram } from '../parse/declarations.js'
import { elaborate } from './elaborator.js'
import { ElaborationError } from './nodes.js'
import type {
  ResolvedProgram, ResolvedExpr, ResolvedExprOpNode,
  RegRef, InputRef, DelayRef, ParamRef, TypeParamRef, BindingRef,
  NestedOut, BinaryOpNode, ClampNode, SelectNode,
  TagExpr, MatchExpr, LetExpr, FoldExpr, GenerateExpr,
  RegDecl, DelayDecl, ParamDecl, InputDecl, TypeParamDecl,
  InstanceDecl, ProgramDecl, OutputAssign, NextUpdate,
  SumTypeDef, AliasTypeDef,
} from './nodes.js'

function elabSrc(src: string): ResolvedProgram {
  return elaborate(parseProgram(src))
}

// ─────────────────────────────────────────────────────────────
// References resolve to decl objects (with `===` reference identity)
// ─────────────────────────────────────────────────────────────

describe('elaborator — value references', () => {
  test('input ref: nameRef resolves to InputRef.decl === the input port', () => {
    const p = elabSrc('program X(freq: signal) -> (out: signal) { out = freq }')
    const inputDecl = p.ports.inputs[0]
    const outputAssign = p.body.assigns[0] as OutputAssign
    const expr = outputAssign.expr as InputRef
    expect(expr.op).toBe('inputRef')
    expect(expr.decl).toBe(inputDecl)  // reference identity
  })

  test('reg ref: nameRef resolves to RegRef.decl === the regDecl', () => {
    const p = elabSrc(`
      program X() -> (out: signal) {
        reg s: float = 0
        out = s
        next s = s
      }
    `)
    const regDecl = p.body.decls[0] as RegDecl
    const out = p.body.assigns[0] as OutputAssign
    const next = p.body.assigns[1] as NextUpdate
    const outRef = out.expr as RegRef
    const nextRef = next.expr as RegRef
    expect(outRef.op).toBe('regRef')
    expect(outRef.decl).toBe(regDecl)
    expect(nextRef.decl).toBe(regDecl)
    // Both refs point at the same object — graph edge identity.
    expect(outRef.decl).toBe(nextRef.decl)
    // nextUpdate.target also points at the same RegDecl.
    expect(next.target).toBe(regDecl)
  })

  test('delay ref: nameRef resolves to DelayRef.decl', () => {
    const p = elabSrc(`
      program X(x: signal) -> (out: signal) {
        delay z = x init 0
        out = z
      }
    `)
    const delayDecl = p.body.decls[0] as DelayDecl
    const out = p.body.assigns[0] as OutputAssign
    const ref = out.expr as DelayRef
    expect(ref.op).toBe('delayRef')
    expect(ref.decl).toBe(delayDecl)
  })

  test('param ref: nameRef resolves to ParamRef.decl', () => {
    const p = elabSrc(`
      program X() -> (out: signal) {
        param cutoff: smoothed = 1000
        out = cutoff
      }
    `)
    const paramDecl = p.body.decls[0] as ParamDecl
    const out = p.body.assigns[0] as OutputAssign
    const ref = out.expr as ParamRef
    expect(ref.op).toBe('paramRef')
    expect(ref.decl).toBe(paramDecl)
  })

  test('type-param ref: nameRef resolves to TypeParamRef.decl', () => {
    const p = elabSrc(`
      program X<N: int = 4>() -> (out: signal) {
        out = N
      }
    `)
    const typeParam = p.typeParams[0]
    const out = p.body.assigns[0] as OutputAssign
    const ref = out.expr as TypeParamRef
    expect(ref.op).toBe('typeParamRef')
    expect(ref.decl).toBe(typeParam)
  })

  test('unknown name: clear error', () => {
    expect(() => elabSrc('program X() -> (out: signal) { out = nope }'))
      .toThrow(/unknown name 'nope'/)
  })
})

describe('elaborator — sentinel calls', () => {
  test('sample_rate() resolves to SampleRate node', () => {
    const p = elabSrc(`
      program X() -> (out: signal) { out = sample_rate() }
    `)
    const out = p.body.assigns[0] as OutputAssign
    expect((out.expr as ResolvedExprOpNode).op).toBe('sampleRate')
  })

  test('sample_index() resolves to SampleIndex node', () => {
    const p = elabSrc(`
      program X() -> (out: signal) { out = sample_index() }
    `)
    const out = p.body.assigns[0] as OutputAssign
    expect((out.expr as ResolvedExprOpNode).op).toBe('sampleIndex')
  })

  test('sample_rate(arg) errors — nullary only', () => {
    expect(() => elabSrc(`
      program X() -> (out: signal) { out = sample_rate(0) }
    `)).toThrow(/no arguments/)
  })
})

describe('elaborator — builtin calls', () => {
  test('clamp(v, lo, hi) resolves to ClampNode', () => {
    const p = elabSrc(`
      program X(v: signal) -> (out: signal) { out = clamp(v, -1, 1) }
    `)
    const out = p.body.assigns[0] as OutputAssign
    const c = out.expr as ClampNode
    expect(c.op).toBe('clamp')
    expect(c.args.length).toBe(3)
  })

  test('select(c, t, e) resolves to SelectNode', () => {
    const p = elabSrc(`
      program X(g: bool) -> (out: signal) { out = select(g, 1, 0) }
    `)
    const out = p.body.assigns[0] as OutputAssign
    const s = out.expr as SelectNode
    expect(s.op).toBe('select')
  })

  test('sqrt(x) resolves to UnaryOpNode with sqrt op', () => {
    const p = elabSrc(`
      program X(x: signal) -> (out: signal) { out = sqrt(x) }
    `)
    const out = p.body.assigns[0] as OutputAssign
    const u = out.expr as { op: string }
    expect(u.op).toBe('sqrt')
  })

  test('unknown function call errors', () => {
    expect(() => elabSrc(`
      program X(x: signal) -> (out: signal) { out = mystery(x) }
    `)).toThrow(/unknown function 'mystery'/)
  })

  test('clamp wrong arity errors', () => {
    expect(() => elabSrc(`
      program X(v: signal) -> (out: signal) { out = clamp(v, -1) }
    `)).toThrow(/'clamp' takes 3 arguments/)
  })
})

// ─────────────────────────────────────────────────────────────
// Binders (let / combinator / match arm) become decl objects
// ─────────────────────────────────────────────────────────────

describe('elaborator — binders are decl objects with refs', () => {
  test('let binders become BinderDecl, body refs hold the decl', () => {
    const p = elabSrc(`
      program X(a: signal) -> (out: signal) {
        out = let { x: a } in x + x
      }
    `)
    const out = p.body.assigns[0] as OutputAssign
    const letExpr = out.expr as LetExpr
    expect(letExpr.op).toBe('let')
    expect(letExpr.binders.length).toBe(1)
    const xBinder = letExpr.binders[0].binder
    const body = letExpr.in as BinaryOpNode
    expect(body.op).toBe('add')
    const left = body.args[0] as BindingRef
    const right = body.args[1] as BindingRef
    expect(left.op).toBe('bindingRef')
    expect(left.decl).toBe(xBinder)
    expect(right.decl).toBe(xBinder)
    // Both occurrences share the same binder decl — graph edge identity.
    expect(left.decl).toBe(right.decl)
  })

  test('shadowing: inner let binder shadows outer; refs distinguish', () => {
    const p = elabSrc(`
      program X() -> (out: signal) {
        out = let { x: 1 } in let { x: 2 } in x
      }
    `)
    const out = p.body.assigns[0] as OutputAssign
    const outer = out.expr as LetExpr
    const inner = outer.in as LetExpr
    const outerX = outer.binders[0].binder
    const innerX = inner.binders[0].binder
    expect(outerX).not.toBe(innerX)  // distinct decl objects
    const innerRef = inner.in as BindingRef
    expect(innerRef.decl).toBe(innerX)  // refers to inner, not outer
  })

  test('fold binders (acc, elem) become two BinderDecls', () => {
    const p = elabSrc(`
      program X() -> (out: signal) {
        out = fold([1, 2, 3], 0, (acc, e) => acc + e)
      }
    `)
    const out = p.body.assigns[0] as OutputAssign
    const fold = out.expr as FoldExpr
    expect(fold.op).toBe('fold')
    const accBinder = fold.acc
    const elemBinder = fold.elem
    expect(accBinder.name).toBe('acc')
    expect(elemBinder.name).toBe('e')
    const body = fold.body as BinaryOpNode
    const lhs = body.args[0] as BindingRef
    const rhs = body.args[1] as BindingRef
    expect(lhs.decl).toBe(accBinder)
    expect(rhs.decl).toBe(elemBinder)
  })

  test('generate binder', () => {
    const p = elabSrc(`
      program X() -> (out: signal) {
        out = generate(4, (i) => i * i)
      }
    `)
    const out = p.body.assigns[0] as OutputAssign
    const gen = out.expr as GenerateExpr
    const iterBinder = gen.iter
    const body = gen.body as BinaryOpNode
    const lhs = body.args[0] as BindingRef
    const rhs = body.args[1] as BindingRef
    expect(lhs.decl).toBe(iterBinder)
    expect(rhs.decl).toBe(iterBinder)
  })

  test('binders do not leak outside their parent', () => {
    expect(() => elabSrc(`
      program X() -> (out: signal) {
        out = (let { x: 1 } in x) + x
      }
    `)).toThrow(/unknown name 'x'/)
  })
})

// ─────────────────────────────────────────────────────────────
// ADTs: tag construction and match elimination
// ─────────────────────────────────────────────────────────────

describe('elaborator — tags', () => {
  test('tag construction: variant resolves, payload field decl-keyed', () => {
    const p = elabSrc(`
      program X() -> (out: signal) {
        enum Maybe { Some(value: float), None }
        out = match Some { value: 42 } {
          Some { value: v } => v,
          None => 0
        }
      }
    `)
    const matchExpr = (p.body.assigns[0] as OutputAssign).expr as MatchExpr
    const tag = matchExpr.scrutinee as TagExpr
    expect(tag.op).toBe('tag')
    expect(tag.variant.name).toBe('Some')
    // Variant carries a back-pointer to its parent SumTypeDef
    expect(tag.variant.parent.name).toBe('Maybe')
    // Payload field references the variant's StructField decl
    expect(tag.payload.length).toBe(1)
    expect(tag.payload[0].field.name).toBe('value')
    expect(tag.payload[0].field).toBe(tag.variant.payload[0])
  })

  test('tag with unknown variant errors', () => {
    expect(() => elabSrc(`
      program X() -> (out: signal) {
        enum Mode { On, Off }
        out = match Bogus { } {
          On => 1,
          Off => 0
        }
      }
    `)).toThrow(/unknown variant 'Bogus'/)
  })

  test('tag missing required payload field errors', () => {
    expect(() => elabSrc(`
      program X() -> (out: signal) {
        enum Pair { P(a: float, b: float) }
        out = match P { a: 1 } { P { a: x, b: y } => x + y }
      }
    `)).toThrow(/missing payload field 'b'/)
  })

  test('tag with extra payload field errors', () => {
    expect(() => elabSrc(`
      program X() -> (out: signal) {
        enum Pair { P(a: float) }
        out = match P { a: 1, b: 2 } { P { a: x } => x }
      }
    `)).toThrow(/unknown payload field/)
  })
})

describe('elaborator — match', () => {
  test('match scrutinee + arms resolve; type back-pointer set', () => {
    const p = elabSrc(`
      program X(v: signal) -> (out: signal) {
        enum Color { Red, Green, Blue }
        out = match v {
          Red => 1,
          Green => 2,
          Blue => 3
        }
      }
    `)
    const matchExpr = (p.body.assigns[0] as OutputAssign).expr as MatchExpr
    expect(matchExpr.op).toBe('match')
    expect(matchExpr.type.name).toBe('Color')
    expect(matchExpr.arms.length).toBe(3)
    // Each arm holds a SumVariant, not a name string.
    expect(matchExpr.arms[0].variant.name).toBe('Red')
    expect(matchExpr.arms[0].variant.parent).toBe(matchExpr.type)
  })

  test('match arm payload binders become BinderDecls in scope of arm body', () => {
    const p = elabSrc(`
      program X(v: signal) -> (out: signal) {
        enum N { Hz(freq: float, gain: float), Off }
        out = match v {
          Hz { freq: f, gain: g } => f * g,
          Off => 0
        }
      }
    `)
    const matchExpr = (p.body.assigns[0] as OutputAssign).expr as MatchExpr
    const hzArm = matchExpr.arms.find(a => a.variant.name === 'Hz')!
    expect(hzArm.binders.length).toBe(2)
    expect(hzArm.binders[0].name).toBe('f')
    expect(hzArm.binders[1].name).toBe('g')
    const body = hzArm.body as BinaryOpNode
    const lhs = body.args[0] as BindingRef
    const rhs = body.args[1] as BindingRef
    expect(lhs.decl).toBe(hzArm.binders[0])
    expect(rhs.decl).toBe(hzArm.binders[1])
  })

  test('non-exhaustive match errors with missing variant name', () => {
    expect(() => elabSrc(`
      program X(v: signal) -> (out: signal) {
        enum Color { Red, Green, Blue }
        out = match v { Red => 1, Green => 2 }
      }
    `)).toThrow(/non-exhaustive: missing variant 'Blue'/)
  })

  test('arm with wrong variant for the inferred sum type errors', () => {
    expect(() => elabSrc(`
      program X(v: signal) -> (out: signal) {
        enum A { On }
        enum B { Off }
        out = match v { On => 1, Off => 0 }
      }
    `)).toThrow(/'Off' is not a member of sum type 'A'/)
  })

  test('duplicate arm errors', () => {
    expect(() => elabSrc(`
      program X(v: signal) -> (out: signal) {
        enum Color { Red, Green }
        out = match v { Red => 1, Green => 2, Red => 3 }
      }
    `)).toThrow(/duplicate arm for variant 'Red'/)
  })

  test('arm binder count must match payload arity', () => {
    expect(() => elabSrc(`
      program X(v: signal) -> (out: signal) {
        enum E { Two(a: float, b: float) }
        out = match v { Two { a: x } => x }
      }
    `)).toThrow(/expected 2 binder/)
  })
})

// ─────────────────────────────────────────────────────────────
// Instances + nested programs
// ─────────────────────────────────────────────────────────────

describe('elaborator — instances + nested programs', () => {
  test('instance refs the nested program decl by reference; ports keyed by InputDecl', () => {
    const p = elabSrc(`
      program Outer() -> (out: signal) {
        program Inner(x: signal = 0) -> (y: signal) { y = x }
        i = Inner(x: 1)
        out = i.y
      }
    `)
    // The nested ProgramDecl
    const progDecl = p.body.decls[0] as ProgramDecl
    expect(progDecl.op).toBe('programDecl')
    // The instance
    const instDecl = p.body.decls[1] as InstanceDecl
    expect(instDecl.op).toBe('instanceDecl')
    expect(instDecl.type).toBe(progDecl.program)  // shared reference
    // Input wire is keyed by the actual InputDecl
    expect(instDecl.inputs.length).toBe(1)
    expect(instDecl.inputs[0].port).toBe(progDecl.program.ports.inputs[0])
    // The output assign uses NestedOut with OutputDecl reference
    const out = p.body.assigns[0] as OutputAssign
    const nest = out.expr as NestedOut
    expect(nest.op).toBe('nestedOut')
    expect(nest.instance).toBe(instDecl)
    expect(nest.output).toBe(progDecl.program.ports.outputs[0])
  })

  test('instance with unknown program type errors', () => {
    expect(() => elabSrc(`
      program X() -> (out: signal) {
        osc = NotDeclared(freq: 440)
        out = 0
      }
    `)).toThrow(/program type 'NotDeclared' is not a nested program/)
  })

  test('instance with unknown input port errors with helpful list', () => {
    expect(() => elabSrc(`
      program X() -> (out: signal) {
        program P(a: signal) -> (out: signal) { out = a }
        i = P(b: 1)
        out = i.out
      }
    `)).toThrow(/input 'b' is not a declared port of 'P' \(have: a\)/)
  })

  test('instance with unknown output port errors', () => {
    expect(() => elabSrc(`
      program X() -> (out: signal) {
        program P() -> (y: signal) { y = 0 }
        i = P()
        out = i.z
      }
    `)).toThrow(/has no output 'z'/)
  })

  test('type-args resolve to the program type\'s TypeParamDecl', () => {
    const p = elabSrc(`
      program Outer() -> (out: signal) {
        program Inner<N: int = 4>() -> (y: signal) { y = N }
        i = Inner<N=8>()
        out = i.y
      }
    `)
    const progDecl = p.body.decls[0] as ProgramDecl
    const instDecl = p.body.decls[1] as InstanceDecl
    expect(instDecl.typeArgs.length).toBe(1)
    expect(instDecl.typeArgs[0].param).toBe(progDecl.program.typeParams[0])
    expect(instDecl.typeArgs[0].value).toBe(8)
  })
})

// ─────────────────────────────────────────────────────────────
// Output assigns + dac.out
// ─────────────────────────────────────────────────────────────

describe('elaborator — output assigns', () => {
  test('outputAssign target is the actual OutputDecl', () => {
    const p = elabSrc('program X() -> (out: signal) { out = 1 }')
    const out = p.body.assigns[0] as OutputAssign
    if (out.target !== null && typeof out.target === 'object' && 'kind' in out.target) {
      throw new Error('expected OutputDecl, got dac sentinel')
    }
    expect(out.target).toBe(p.ports.outputs[0])
  })

  test('dac.out target is the dac sentinel', () => {
    const p = elabSrc(`
      program Patch() {
        program Osc() -> (out: signal) { out = 0 }
        o = Osc()
        dac.out = o.out
      }
    `)
    // Find the assign for dac.out
    const dacAssign = p.body.assigns.find(a =>
      a.op === 'outputAssign'
      && (a.target as { kind?: string }).kind === 'dac',
    ) as OutputAssign
    expect((dacAssign.target as { kind: string }).kind).toBe('dac')
  })

  test('outputAssign to undeclared output errors', () => {
    expect(() => elabSrc(`
      program X() -> (out: signal) {
        unknown = 0
        out = 0
      }
    `)).toThrow(/unknown output port 'unknown'/)
  })
})

// ─────────────────────────────────────────────────────────────
// Type defs
// ─────────────────────────────────────────────────────────────

describe('elaborator — type defs', () => {
  test('alias type-def: base resolves to ScalarKind', () => {
    const p = elabSrc(`
      program X() -> (out: signal) {
        type Bipolar = float in [-1, 1]
        out = 0
      }
    `)
    const alias = p.ports.typeDefs[0] as AliasTypeDef
    expect(alias.op).toBe('aliasTypeDef')
    expect(alias.base).toBe('float')
    expect(alias.bounds).toEqual([-1, 1])
  })

  test('struct type-def: fields preserve scalar kinds', () => {
    const p = elabSrc(`
      program X() -> (out: signal) {
        struct Pair { a: float, b: int }
        out = 0
      }
    `)
    const td = p.ports.typeDefs[0] as { op: string; fields: Array<{ type: string }> }
    expect(td.op).toBe('structTypeDef')
    expect(td.fields[0].type).toBe('float')
    expect(td.fields[1].type).toBe('int')
  })

  test('sum variant carries back-pointer to parent type', () => {
    const p = elabSrc(`
      program X() -> (out: signal) {
        enum Mode { On, Off }
        out = 0
      }
    `)
    const sum = p.ports.typeDefs[0] as SumTypeDef
    expect(sum.variants[0].parent).toBe(sum)
    expect(sum.variants[1].parent).toBe(sum)
  })

  test('variant name conflict across sum types errors', () => {
    expect(() => elabSrc(`
      program X() -> (out: signal) {
        enum A { Same, A1 }
        enum B { Same, B1 }
        out = 0
      }
    `)).toThrow(/variant 'Same' is declared in multiple sum types/)
  })
})

// ─────────────────────────────────────────────────────────────
// Port types and shape dims
// ─────────────────────────────────────────────────────────────

describe('elaborator — port types', () => {
  test('scalar port type resolves to {kind:scalar, scalar}', () => {
    const p = elabSrc('program X(x: float) -> (out: signal) { out = x }')
    const portType = p.ports.inputs[0].type
    expect(portType).toEqual({ kind: 'scalar', scalar: 'float' })
  })

  test('array port type with type-param shape dim resolves dim to TypeParamDecl', () => {
    const p = elabSrc(`
      program X<N: int = 4>(buf: float[N]) -> (out: signal) { out = 0 }
    `)
    const inputType = p.ports.inputs[0].type
    if (!inputType || inputType.kind !== 'array') throw new Error('expected array')
    expect(inputType.shape[0]).toBe(p.typeParams[0])  // === reference identity
  })

  test('shape dim references undeclared type-param errors', () => {
    expect(() => elabSrc(`
      program X(buf: float[K]) -> (out: signal) { out = 0 }
    `)).toThrow(/'K' is not a declared type-param/)
  })
})

// ─────────────────────────────────────────────────────────────
// Graph integrity invariants
// ─────────────────────────────────────────────────────────────

describe('elaborator — graph integrity', () => {
  test('a single declaration produces exactly one decl object', () => {
    const p = elabSrc(`
      program X() -> (out: signal) {
        reg s: float = 0
        out = s + s + s
        next s = s
      }
    `)
    const regDecl = p.body.decls[0] as RegDecl
    // Walk all references; every regRef.decl is the same object.
    const seen: unknown[] = []
    walk(p, (obj) => {
      if ((obj as { op?: string }).op === 'regRef') {
        seen.push((obj as { decl: unknown }).decl)
      }
    })
    expect(seen.length).toBeGreaterThan(0)
    for (const s of seen) {
      expect(s).toBe(regDecl)  // strict reference identity
    }
  })

  test('elaborating the same source twice produces structurally identical graphs', () => {
    const src = `
      program X(x: signal) -> (out: signal) {
        reg s: float = 0
        out = x + s
        next s = s + 1
      }
    `
    const a = elabSrc(src)
    const b = elabSrc(src)
    // Same shape, but DIFFERENT decl objects (the two elaborations
    // produce distinct graphs). Internal reference identity within each
    // graph is preserved; cross-graph identity is not.
    expect(a).not.toBe(b)
    expect(a.body.decls[0]).not.toBe(b.body.decls[0])  // distinct RegDecls
    // Structurally equal: same names, same shapes
    const aReg = a.body.decls[0] as RegDecl
    const bReg = b.body.decls[0] as RegDecl
    expect(aReg.name).toBe(bReg.name)
    expect(aReg.init).toEqual(bReg.init)
  })

  test('feedback through a register is a graph cycle (decl referenced via next)', () => {
    const p = elabSrc(`
      program X(x: signal) -> (out: signal) {
        reg s: float = 0
        out = s
        next s = s + x
      }
    `)
    const regDecl = p.body.decls[0] as RegDecl
    const next = p.body.assigns[1] as NextUpdate
    // The reg's nextUpdate target points at the same RegDecl as
    // p.body.decls[0]. The expr graph contains a regRef that also points
    // at it. This is a cycle in the graph (decl ↔ ref ↔ decl).
    expect(next.target).toBe(regDecl)
    const expr = next.expr as BinaryOpNode
    expect((expr.args[0] as RegRef).decl).toBe(regDecl)
  })
})

// ─────────────────────────────────────────────────────────────
// Helper: walk every plain object in a value, calling visit
// ─────────────────────────────────────────────────────────────
function walk(value: unknown, visit: (obj: Record<string, unknown>) => void, seen = new WeakSet()): void {
  if (Array.isArray(value)) {
    for (const v of value) walk(v, visit, seen)
    return
  }
  if (value !== null && typeof value === 'object') {
    if (seen.has(value as object)) return
    seen.add(value as object)
    visit(value as Record<string, unknown>)
    for (const v of Object.values(value as Record<string, unknown>)) {
      walk(v, visit, seen)
    }
  }
}
