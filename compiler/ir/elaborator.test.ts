/**
 * elaborator.test.ts — coverage for the parser → resolved-IR elaborator.
 *
 * The elaborator's job is to turn the parsed tree (with NameRef
 * placeholders) into a graph where every reference is a direct decl
 * reference. These tests verify both:
 *   - resolution: each reference resolves to the correct decl
 *   - reference identity: regRef.decl === <the actual RegDecl> (not just
 *     a structurally-equal copy)
 */

import { describe, test, expect } from 'bun:test'
import { parseProgram } from '../parse/declarations.js'
import { elaborate, type ExternalProgramResolver } from './elaborator.js'
import { ElaborationError } from './nodes.js'
import type {
  ResolvedProgram, ResolvedExpr, ResolvedExprOp,
  RegRef, InputRef, ParamRef, TypeParamRef, BindingRef,
  NestedOut, BinaryOp, Clamp, Select,
  Tag, Match, Let, Fold, Generate,
  RegDecl, ParamDecl, InputDecl, TypeParamDecl,
  InstanceDecl, ProgramDecl, OutputAssign,
  SumTypeDef, AliasTypeDef,
  Zeros, ArraySet,
} from './nodes.js'

function elabSrc(src: string): ResolvedProgram {
  return elaborate(parseProgram(src))
}

// ─────────────────────────────────────────────────────────────
// References resolve to decl objects (with `===` reference identity)
// ─────────────────────────────────────────────────────────────

describe('elaborator — value references', () => {
  test('input ref: nameRef resolves to InputRef.idx === the input port position', () => {
    const p = elabSrc('program X(freq: float) -> (out: float) { out = freq }')
    const inputDecl = p.ports.inputs[0]
    const outputAssign = p.body.assigns[0] as OutputAssign
    const expr = outputAssign.expr as InputRef
    expect(expr.op).toBe('inputRef')
    expect(p.ports.inputs[expr.idx]).toBe(inputDecl)
  })

  test('reg ref: nameRef resolves to RegRef.decl === the regDecl', () => {
    // Post-Phase-0a: `next s = e` folds into RegDecl.update at
    // elaboration; the resolved IR has no NextUpdate body-assign.
    const p = elabSrc(`
      program X() -> (out: float) {
        reg s: float = 0
        out = s
        next s = s
      }
    `)
    const regDecl = p.body.decls[0] as RegDecl
    const out = p.body.assigns[0] as OutputAssign
    const outRef = out.expr as RegRef
    expect(outRef.op).toBe('regRef')
    expect(p.regs[outRef.idx]).toBe(regDecl)
    // The folded update reads s — same decl identity (resolved via idx).
    expect(regDecl.update).toBeDefined()
    expect(p.regs[(regDecl.update as RegRef).idx]).toBe(regDecl)
  })

  test('delay-form RegDecl: nameRef resolves to RegRef on the decl', () => {
    // Post-Phase-0a: `delay z = u init v` desugars to a RegDecl with
    // update populated. Reads of z are RegRefs.
    const p = elabSrc(`
      program X(x: float) -> (out: float) {
        delay z = x init 0
        out = z
      }
    `)
    const regDecl = p.body.decls[0] as RegDecl
    expect(regDecl.op).toBe('regDecl')
    expect(regDecl.name).toBe('z')
    expect(regDecl.update).toBeDefined()
    expect(regDecl.init).toBe(0)
    const out = p.body.assigns[0] as OutputAssign
    const ref = out.expr as RegRef
    expect(ref.op).toBe('regRef')
    expect(p.regs[ref.idx]).toBe(regDecl)
  })

  test('param ref: nameRef resolves to ParamRef.idx', () => {
    const p = elabSrc(`
      program X() -> (out: float) {
        param cutoff: smoothed = 1000
        out = cutoff
      }
    `)
    const paramDecl = p.body.decls[0] as ParamDecl
    const out = p.body.assigns[0] as OutputAssign
    const ref = out.expr as ParamRef
    expect(ref.op).toBe('paramRef')
    expect(p.params[ref.idx]).toBe(paramDecl)
  })

  test('type-param ref: nameRef resolves to TypeParamRef.idx', () => {
    const p = elabSrc(`
      program X<N: int = 4>() -> (out: float) {
        out = N
      }
    `)
    const typeParam = p.typeParams[0]
    const out = p.body.assigns[0] as OutputAssign
    const ref = out.expr as TypeParamRef
    expect(ref.op).toBe('typeParamRef')
    expect(p.typeParams[ref.idx]).toBe(typeParam)
  })

  test('unknown name: clear error', () => {
    expect(() => elabSrc('program X() -> (out: float) { out = nope }'))
      .toThrow(/unknown name 'nope'/)
  })
})

describe('elaborator — sentinel calls', () => {
  test('sample_rate() resolves to SampleRate node', () => {
    const p = elabSrc(`
      program X() -> (out: float) { out = sample_rate() }
    `)
    const out = p.body.assigns[0] as OutputAssign
    expect((out.expr as ResolvedExprOp).op).toBe('sampleRate')
  })

  test('sample_index() resolves to SampleIndex node', () => {
    const p = elabSrc(`
      program X() -> (out: float) { out = sample_index() }
    `)
    const out = p.body.assigns[0] as OutputAssign
    expect((out.expr as ResolvedExprOp).op).toBe('sampleIndex')
  })

  test('sample_rate(arg) errors — nullary only', () => {
    expect(() => elabSrc(`
      program X() -> (out: float) { out = sample_rate(0) }
    `)).toThrow(/no arguments/)
  })
})

describe('elaborator — builtin calls', () => {
  test('clamp(v, lo, hi) resolves to Clamp', () => {
    const p = elabSrc(`
      program X(v: float) -> (out: float) { out = clamp(v, -1, 1) }
    `)
    const out = p.body.assigns[0] as OutputAssign
    const c = out.expr as Clamp
    expect(c.op).toBe('clamp')
    expect(c.args.length).toBe(3)
  })

  test('select(c, t, e) resolves to Select', () => {
    const p = elabSrc(`
      program X(g: bool) -> (out: float) { out = select(g, 1, 0) }
    `)
    const out = p.body.assigns[0] as OutputAssign
    const s = out.expr as Select
    expect(s.op).toBe('select')
  })

  test('sqrt(x) resolves to UnaryOp with sqrt op', () => {
    const p = elabSrc(`
      program X(x: float) -> (out: float) { out = sqrt(x) }
    `)
    const out = p.body.assigns[0] as OutputAssign
    const u = out.expr as { op: string }
    expect(u.op).toBe('sqrt')
  })

  test('unknown function call errors', () => {
    expect(() => elabSrc(`
      program X(x: float) -> (out: float) { out = mystery(x) }
    `)).toThrow(/unknown function 'mystery'/)
  })

  test('clamp wrong arity errors', () => {
    expect(() => elabSrc(`
      program X(v: float) -> (out: float) { out = clamp(v, -1) }
    `)).toThrow(/'clamp' takes 3 arguments/)
  })
})

// ─────────────────────────────────────────────────────────────
// Binders (let / combinator / match arm) become decl objects
// ─────────────────────────────────────────────────────────────

describe('elaborator — binders are decl objects with refs', () => {
  test('let binders become BinderDecl, body refs hold the decl', () => {
    const p = elabSrc(`
      program X(a: float) -> (out: float) {
        out = let { x: a } in x + x
      }
    `)
    const out = p.body.assigns[0] as OutputAssign
    const letExpr = out.expr as Let
    expect(letExpr.op).toBe('let')
    expect(letExpr.binders.length).toBe(1)
    const xBinder = letExpr.binders[0].binder
    const body = letExpr.in as BinaryOp
    expect(body.op).toBe('add')
    const left = body.args[0] as BindingRef
    const right = body.args[1] as BindingRef
    expect(left.op).toBe('bindingRef')
    expect(left.idx).toBe(xBinder.idx)
    expect(right.idx).toBe(xBinder.idx)
    // Both occurrences share the same binder idx — graph edge identity.
    expect(left.idx).toBe(right.idx)
  })

  test('shadowing: inner let binder shadows outer; refs distinguish', () => {
    const p = elabSrc(`
      program X() -> (out: float) {
        out = let { x: 1 } in let { x: 2 } in x
      }
    `)
    const out = p.body.assigns[0] as OutputAssign
    const outer = out.expr as Let
    const inner = outer.in as Let
    const outerX = outer.binders[0].binder
    const innerX = inner.binders[0].binder
    expect(outerX.idx).not.toBe(innerX.idx)  // distinct binder idx
    const innerRef = inner.in as BindingRef
    expect(innerRef.idx).toBe(innerX.idx)  // refers to inner, not outer
  })

  test('fold binders (acc, elem) become two BinderDecls', () => {
    const p = elabSrc(`
      program X() -> (out: float) {
        out = fold([1, 2, 3], 0, (acc, e) => acc + e)
      }
    `)
    const out = p.body.assigns[0] as OutputAssign
    const fold = out.expr as Fold
    expect(fold.op).toBe('fold')
    const accBinder = fold.acc
    const elemBinder = fold.elem
    expect(accBinder.name).toBe('acc')
    expect(elemBinder.name).toBe('e')
    const body = fold.body as BinaryOp
    const lhs = body.args[0] as BindingRef
    const rhs = body.args[1] as BindingRef
    expect(lhs.idx).toBe(accBinder.idx)
    expect(rhs.idx).toBe(elemBinder.idx)
  })

  test('generate binder', () => {
    const p = elabSrc(`
      program X() -> (out: float) {
        out = generate(4, (i) => i * i)
      }
    `)
    const out = p.body.assigns[0] as OutputAssign
    const gen = out.expr as Generate
    const iterBinder = gen.iter
    const body = gen.body as BinaryOp
    const lhs = body.args[0] as BindingRef
    const rhs = body.args[1] as BindingRef
    expect(lhs.idx).toBe(iterBinder.idx)
    expect(rhs.idx).toBe(iterBinder.idx)
  })

  test('binders do not leak outside their parent', () => {
    expect(() => elabSrc(`
      program X() -> (out: float) {
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
      program X() -> (out: float) {
        enum Maybe { Some(value: float), None }
        out = match Some { value: 42 } {
          Some { value: v } => v,
          None => 0
        }
      }
    `)
    const matchExpr = (p.body.assigns[0] as OutputAssign).expr as Match
    const tag = matchExpr.scrutinee as Tag
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
      program X() -> (out: float) {
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
      program X() -> (out: float) {
        enum Pair { P(a: float, b: float) }
        out = match P { a: 1 } { P { a: x, b: y } => x + y }
      }
    `)).toThrow(/missing payload field 'b'/)
  })

  test('tag with extra payload field errors', () => {
    expect(() => elabSrc(`
      program X() -> (out: float) {
        enum Pair { P(a: float) }
        out = match P { a: 1, b: 2 } { P { a: x } => x }
      }
    `)).toThrow(/unknown payload field/)
  })
})

describe('elaborator — match', () => {
  test('match scrutinee + arms resolve; type back-pointer set', () => {
    const p = elabSrc(`
      program X(v: float) -> (out: float) {
        enum Color { Red, Green, Blue }
        out = match v {
          Red => 1,
          Green => 2,
          Blue => 3
        }
      }
    `)
    const matchExpr = (p.body.assigns[0] as OutputAssign).expr as Match
    expect(matchExpr.op).toBe('match')
    expect(matchExpr.type.name).toBe('Color')
    expect(matchExpr.arms.length).toBe(3)
    // Each arm holds a SumVariant, not a name string.
    expect(matchExpr.arms[0].variant.name).toBe('Red')
    expect(matchExpr.arms[0].variant.parent).toBe(matchExpr.type)
  })

  test('match arm payload binders become BinderDecls in scope of arm body', () => {
    const p = elabSrc(`
      program X(v: float) -> (out: float) {
        enum N { Hz(freq: float, gain: float), Off }
        out = match v {
          Hz { freq: f, gain: g } => f * g,
          Off => 0
        }
      }
    `)
    const matchExpr = (p.body.assigns[0] as OutputAssign).expr as Match
    const hzArm = matchExpr.arms.find(a => a.variant.name === 'Hz')!
    expect(hzArm.binders.length).toBe(2)
    expect(hzArm.binders[0].name).toBe('f')
    expect(hzArm.binders[1].name).toBe('g')
    const body = hzArm.body as BinaryOp
    const lhs = body.args[0] as BindingRef
    const rhs = body.args[1] as BindingRef
    expect(lhs.idx).toBe(hzArm.binders[0].idx)
    expect(rhs.idx).toBe(hzArm.binders[1].idx)
  })

  test('non-exhaustive match errors with missing variant name', () => {
    expect(() => elabSrc(`
      program X(v: float) -> (out: float) {
        enum Color { Red, Green, Blue }
        out = match v { Red => 1, Green => 2 }
      }
    `)).toThrow(/non-exhaustive: missing variant 'Blue'/)
  })

  test('arm with wrong variant for the inferred sum type errors', () => {
    expect(() => elabSrc(`
      program X(v: float) -> (out: float) {
        enum A { On }
        enum B { Off }
        out = match v { On => 1, Off => 0 }
      }
    `)).toThrow(/'Off' is not a member of sum type 'A'/)
  })

  test('duplicate arm errors', () => {
    expect(() => elabSrc(`
      program X(v: float) -> (out: float) {
        enum Color { Red, Green }
        out = match v { Red => 1, Green => 2, Red => 3 }
      }
    `)).toThrow(/duplicate arm for variant 'Red'/)
  })

  test('arm binder count must match payload arity', () => {
    expect(() => elabSrc(`
      program X(v: float) -> (out: float) {
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
      program Outer() -> (out: float) {
        program Inner(x: float = 0) -> (y: float) { y = x }
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
    // Input wire is keyed by the input port's position in the target
    expect(instDecl.inputs.length).toBe(1)
    expect(progDecl.program.ports.inputs[instDecl.inputs[0].port]).toBe(progDecl.program.ports.inputs[0])
    // The output assign uses NestedOut with instance + output indices
    const out = p.body.assigns[0] as OutputAssign
    const nest = out.expr as NestedOut
    expect(nest.op).toBe('nestedOut')
    expect(p.instances[nest.instance]).toBe(instDecl)
    expect(progDecl.program.ports.outputs[nest.output]).toBe(progDecl.program.ports.outputs[0])
  })

  test('instance with unknown program type errors', () => {
    expect(() => elabSrc(`
      program X() -> (out: float) {
        osc = NotDeclared(freq: 440)
        out = 0
      }
    `)).toThrow(/program type 'NotDeclared' is not a nested program/)
  })

  test('instance with unknown input port errors with helpful list', () => {
    expect(() => elabSrc(`
      program X() -> (out: float) {
        program P(a: float) -> (out: float) { out = a }
        i = P(b: 1)
        out = i.out
      }
    `)).toThrow(/input 'b' is not a declared port of 'P' \(have: a\)/)
  })

  test('instance with unknown output port errors', () => {
    expect(() => elabSrc(`
      program X() -> (out: float) {
        program P() -> (y: float) { y = 0 }
        i = P()
        out = i.z
      }
    `)).toThrow(/has no output 'z'/)
  })

  test('type-args resolve to the program type\'s TypeParamDecl', () => {
    const p = elabSrc(`
      program Outer() -> (out: float) {
        program Inner<N: int = 4>() -> (y: float) { y = N }
        i = Inner<N=8>()
        out = i.y
      }
    `)
    const progDecl = p.body.decls[0] as ProgramDecl
    const instDecl = p.body.decls[1] as InstanceDecl
    expect(instDecl.typeArgs.length).toBe(1)
    expect(progDecl.program.typeParams[instDecl.typeArgs[0].param]).toBe(progDecl.program.typeParams[0])
    expect(instDecl.typeArgs[0].value).toBe(8)
  })
})

// ─────────────────────────────────────────────────────────────
// Output assigns + dac.out
// ─────────────────────────────────────────────────────────────

describe('elaborator — output assigns', () => {
  test('outputAssign target is the actual OutputDecl', () => {
    const p = elabSrc('program X() -> (out: float) { out = 1 }')
    const out = p.body.assigns[0] as OutputAssign
    if (typeof out.target !== 'number') {
      throw new Error('expected OutputIdx, got dac sentinel')
    }
    expect(p.ports.outputs[out.target]).toBe(p.ports.outputs[0])
  })

  test('dac.out target is the dac sentinel', () => {
    const p = elabSrc(`
      program Patch() {
        program Osc() -> (out: float) { out = 0 }
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
      program X() -> (out: float) {
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
      program X() -> (out: float) {
        type Bipolar = float
        out = 0
      }
    `)
    const alias = p.ports.typeDefs[0] as AliasTypeDef
    expect(alias.op).toBe('aliasTypeDef')
    expect(alias.base).toBe('float')
  })

  test('struct type-def: fields preserve scalar kinds', () => {
    const p = elabSrc(`
      program X() -> (out: float) {
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
      program X() -> (out: float) {
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
      program X() -> (out: float) {
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
    const p = elabSrc('program X(x: float) -> (out: float) { out = x }')
    const portType = p.ports.inputs[0].type
    expect(portType).toEqual({ kind: 'scalar', scalar: 'float' })
  })

  test('array port type with type-param shape dim resolves dim to TypeParamDecl', () => {
    const p = elabSrc(`
      program X<N: int = 4>(buf: float[N]) -> (out: float) { out = 0 }
    `)
    const inputType = p.ports.inputs[0].type
    if (!inputType || inputType.kind !== 'array') throw new Error('expected array')
    expect(inputType.shape[0]).toBe(p.typeParams[0])  // === reference identity
  })

  test('shape dim references undeclared type-param errors', () => {
    expect(() => elabSrc(`
      program X(buf: float[K]) -> (out: float) { out = 0 }
    `)).toThrow(/'K' is not a declared type-param/)
  })
})

// ─────────────────────────────────────────────────────────────
// Graph integrity invariants
// ─────────────────────────────────────────────────────────────

describe('elaborator — graph integrity', () => {
  test('a single declaration produces exactly one decl object', () => {
    const p = elabSrc(`
      program X() -> (out: float) {
        reg s: float = 0
        out = s + s + s
        next s = s
      }
    `)
    const regDecl = p.body.decls[0] as RegDecl
    // Walk all references; every regRef.idx resolves to the same RegDecl.
    const seen: unknown[] = []
    walk(p, (obj) => {
      if ((obj as { op?: string }).op === 'regRef') {
        seen.push(p.regs[(obj as { idx: number }).idx])
      }
    })
    expect(seen.length).toBeGreaterThan(0)
    for (const s of seen) {
      expect(s).toBe(regDecl)  // strict reference identity (via idx → decl)
    }
  })

  test('elaborating the same source twice produces structurally identical graphs', () => {
    const src = `
      program X(x: float) -> (out: float) {
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

  test('feedback through a register is a graph cycle (decl referenced via update)', () => {
    // Post-Phase-0a: next-update folds into RegDecl.update; the cycle
    // is decl ↔ update.regRef ↔ decl.
    const p = elabSrc(`
      program X(x: float) -> (out: float) {
        reg s: float = 0
        out = s
        next s = s + x
      }
    `)
    const regDecl = p.body.decls[0] as RegDecl
    expect(regDecl.update).toBeDefined()
    const expr = regDecl.update as BinaryOp
    expect(p.regs[(expr.args[0] as RegRef).idx]).toBe(regDecl)
  })
})

// ─────────────────────────────────────────────────────────────
// External program resolver (Phase C1.5)
// ─────────────────────────────────────────────────────────────

describe('elaborator — external program resolver', () => {
  test('resolver supplies a program type that is not nested', () => {
    // Elaborate Inner first, then feed it to a sibling program that
    // instantiates it without declaring it as a nested programDecl.
    const inner = elabSrc(`
      program Inner(x: float = 0) -> (y: float) { y = x }
    `)
    const resolver: ExternalProgramResolver = name =>
      name === 'Inner' ? inner : undefined
    const outer = elaborate(parseProgram(`
      program Outer() -> (out: float) {
        i = Inner(x: 1)
        out = i.y
      }
    `), resolver)
    const inst = outer.body.decls[0] as InstanceDecl
    expect(inst.op).toBe('instanceDecl')
    // The instance's program type IS the externally-supplied object.
    expect(inst.type).toBe(inner)
    expect(inner.ports.inputs[inst.inputs[0].port]).toBe(inner.ports.inputs[0])
  })

  test('resolver returns undefined → instance error mentions resolver', () => {
    const resolver: ExternalProgramResolver = () => undefined
    expect(() => elaborate(parseProgram(`
      program Outer() -> (out: float) {
        i = Missing(x: 1)
        out = i.y
      }
    `), resolver)).toThrow(/no external resolver provided/)
  })

  test('without resolver, external instance still errors as before', () => {
    expect(() => elabSrc(`
      program Outer() -> (out: float) {
        i = NotDeclared()
        out = 0
      }
    `)).toThrow(/program type 'NotDeclared' is not a nested program/)
  })

  test('nested programs inherit the resolver from the enclosing scope', () => {
    // The inner program also gets the same resolver — instances inside
    // a nested programDecl can reach external program types.
    const sib = elabSrc(`
      program Sib(x: float = 0) -> (y: float) { y = x }
    `)
    const resolver: ExternalProgramResolver = name =>
      name === 'Sib' ? sib : undefined
    const outer = elaborate(parseProgram(`
      program Outer() -> (out: float) {
        program Wrap(z: float = 0) -> (w: float) {
          inner = Sib(x: z)
          w = inner.y
        }
        wrapped = Wrap(z: 1)
        out = wrapped.w
      }
    `), resolver)
    const wrap = (outer.body.decls[0] as ProgramDecl).program
    const innerInst = wrap.body.decls[0] as InstanceDecl
    expect(innerInst.type).toBe(sib)
  })
})

// ─────────────────────────────────────────────────────────────
// Sequential let* (Phase C1.5)
// ─────────────────────────────────────────────────────────────

describe('elaborator — sequential let* binding', () => {
  test('a later binder may reference an earlier one in the same let', () => {
    // This shape is used by stdlib (Tanh, Exp, Sin); the older parallel
    // semantics rejected it as `unknown name 'c'`.
    const p = elabSrc(`
      program X(x: float) -> (out: float) {
        out = let { c: x + 1; c2: c * c } in c2
      }
    `)
    const letExpr = (p.body.assigns[0] as OutputAssign).expr as Let
    const cBinder = letExpr.binders[0].binder
    const c2Value = letExpr.binders[1].value as BinaryOp
    // c2's value is `c * c` — both operands reference the c binder.
    expect(c2Value.op).toBe('mul')
    const lhs = c2Value.args[0] as BindingRef
    const rhs = c2Value.args[1] as BindingRef
    expect(lhs.idx).toBe(cBinder.idx)
    expect(rhs.idx).toBe(cBinder.idx)
  })

  test('the body sees every binder in the let block', () => {
    const p = elabSrc(`
      program X(x: float) -> (out: float) {
        out = let { a: x; b: a + 1; c: b + 1 } in a + b + c
      }
    `)
    const letExpr = (p.body.assigns[0] as OutputAssign).expr as Let
    expect(letExpr.binders.length).toBe(3)
    // No throw on `a`, `b`, `c` in the body — that's the assertion.
    expect(letExpr.in).toBeDefined()
  })

  test('binder shadowing in let* still restores the prior binder afterwards', () => {
    // Construct a scenario where the same name is used in nested lets
    // and verify the inner binder is distinct from the outer.
    const p = elabSrc(`
      program X(x: float) -> (out: float) {
        out = let { a: x } in let { a: a + 1 } in a
      }
    `)
    const outer = (p.body.assigns[0] as OutputAssign).expr as Let
    const outerA = outer.binders[0].binder
    const inner = outer.in as Let
    const innerA = inner.binders[0].binder
    // The inner let's value reads `a` — the outer binder, since the
    // inner binder isn't in scope until after its value is resolved.
    const innerValue = inner.binders[0].value as BinaryOp
    const ref = innerValue.args[0] as BindingRef
    expect(ref.idx).toBe(outerA.idx)
    // The body of the inner let sees the inner binder.
    const body = inner.in as BindingRef
    expect(body.idx).toBe(innerA.idx)
  })
})

// ─────────────────────────────────────────────────────────────
// New builtins (Phase C1.5)
// ─────────────────────────────────────────────────────────────

describe('elaborator — new builtin calls', () => {
  test('camelCase sampleRate / sampleIndex resolve to the same nodes as snake_case', () => {
    const p = elabSrc(`
      program X() -> (out: float) { out = sampleRate() + sampleIndex() }
    `)
    const expr = (p.body.assigns[0] as OutputAssign).expr as BinaryOp
    expect(expr.op).toBe('add')
    const argOps = expr.args.map(a => (a as ResolvedExprOp).op).sort()
    expect(argOps).toEqual(['sampleIndex', 'sampleRate'])
  })

  test('floatExponent (camelCase) resolves to UnaryOp', () => {
    const p = elabSrc(`
      program X(x: float) -> (out: float) { out = floatExponent(x) }
    `)
    const expr = (p.body.assigns[0] as OutputAssign).expr as { op: string }
    expect(expr.op).toBe('floatExponent')
  })

  test('floorDiv / ldexp resolve to BinaryOpNodes', () => {
    const p = elabSrc(`
      program X(a: float, b: float) -> (out: float) {
        out = floorDiv(a, b) + ldexp(a, b)
      }
    `)
    // out = (floorDiv + ldexp); walk for the two op tags.
    const tags: string[] = []
    walk((p.body.assigns[0] as OutputAssign).expr, (obj) => {
      const op = (obj as { op?: string }).op
      if (op === 'floorDiv' || op === 'ldexp') tags.push(op)
    })
    expect(tags.sort()).toEqual(['floorDiv', 'ldexp'])
  })

  test('zeros(n) resolves to Zeros with the count expr', () => {
    const p = elabSrc(`
      program X<N: int = 4>() -> (out: float[N]) {
        reg buf: float = 0
        out = 0
        next buf = zeros(N)
      }
    `)
    const regDecl = p.body.decls[0] as RegDecl
    const z = regDecl.update as Zeros
    expect(z.op).toBe('zeros')
    const ref = z.count as TypeParamRef
    expect(ref.op).toBe('typeParamRef')
    expect(p.typeParams[ref.idx]).toBe(p.typeParams[0])
  })

  test('arraySet(arr, idx, val) resolves to ArraySet with three args', () => {
    const p = elabSrc(`
      program X<N: int = 4>(x = 0) -> (y) {
        reg buf: float = 0
        y = 0
        next buf = arraySet(buf, sampleIndex() % N, x)
      }
    `)
    const regDecl = p.body.decls[0] as RegDecl
    const a = regDecl.update as ArraySet
    expect(a.op).toBe('arraySet')
    expect(a.args.length).toBe(3)
  })

  test('binary builtin wrong arity errors', () => {
    expect(() => elabSrc(`
      program X(a: float) -> (out: float) { out = ldexp(a) }
    `)).toThrow(/'ldexp' takes 2 arguments/)
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
