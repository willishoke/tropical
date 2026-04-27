/**
 * adts.test.ts — ADT type defs (struct/enum/type alias) + match/tag
 * expressions (Phase B5).
 */

import { describe, test, expect } from 'bun:test'
import { parseProgram, type ProgramNode, type StructTypeDef, type SumTypeDef, type AliasTypeDef } from './declarations.js'
import { parseExpr, ParseError } from './expressions.js'

// ─────────────────────────────────────────────────────────────
// Struct decls
// ─────────────────────────────────────────────────────────────

describe('adts — struct decls', () => {
  test('empty-program with single struct', () => {
    const p = parseProgram(`
      program X() {
        struct Pair { a: float, b: int }
      }
    `)
    expect(p.ports?.type_defs).toEqual([
      { kind: 'struct', name: 'Pair', fields: [
        { name: 'a', scalar_type: 'float' },
        { name: 'b', scalar_type: 'int' },
      ]},
    ])
  })

  test('multiple struct fields with all scalar kinds', () => {
    const p = parseProgram(`
      program X() {
        struct Sample { left: float, right: float, gain: float, on: bool, count: int }
      }
    `)
    const td = (p.ports!.type_defs![0] as StructTypeDef)
    expect(td.fields.map(f => f.scalar_type)).toEqual(['float', 'float', 'float', 'bool', 'int'])
  })

  test('empty struct (no fields)', () => {
    const p = parseProgram(`program X() { struct Unit {} }`)
    expect((p.ports!.type_defs![0] as StructTypeDef).fields).toEqual([])
  })

  test('non-scalar field type rejected', () => {
    expect(() => parseProgram(`program X() { struct S { a: signal } }`))
      .toThrow(/float\/int\/bool/)
  })

  test('duplicate field name in struct rejected', () => {
    expect(() => parseProgram(`program X() { struct S { a: float, a: int } }`))
      .toThrow(/duplicate field 'a'/)
  })
})

// ─────────────────────────────────────────────────────────────
// Enum (sum) decls
// ─────────────────────────────────────────────────────────────

describe('adts — enum decls', () => {
  test('all-nullary enum', () => {
    const p = parseProgram(`
      program X() {
        enum Color { Red, Green, Blue }
      }
    `)
    expect(p.ports?.type_defs).toEqual([
      { kind: 'sum', name: 'Color', variants: [
        { name: 'Red', payload: [] },
        { name: 'Green', payload: [] },
        { name: 'Blue', payload: [] },
      ]},
    ])
  })

  test('mixed nullary and payload variants', () => {
    const p = parseProgram(`
      program X() {
        enum Maybe { Some(value: float), None }
      }
    `)
    const td = p.ports!.type_defs![0] as SumTypeDef
    expect(td.variants).toEqual([
      { name: 'Some', payload: [{ name: 'value', scalar_type: 'float' }] },
      { name: 'None', payload: [] },
    ])
  })

  test('multi-field variant payload', () => {
    const p = parseProgram(`
      program X() {
        enum Note { Hz(freq: float, gain: float), Off }
      }
    `)
    const td = p.ports!.type_defs![0] as SumTypeDef
    expect(td.variants[0]).toEqual({
      name: 'Hz',
      payload: [
        { name: 'freq', scalar_type: 'float' },
        { name: 'gain', scalar_type: 'float' },
      ],
    })
  })

  test('duplicate variant name rejected', () => {
    expect(() => parseProgram(`program X() { enum E { A, A } }`))
      .toThrow(/duplicate variant 'A'/)
  })

  test('duplicate field within a variant rejected', () => {
    expect(() => parseProgram(`program X() { enum E { V(a: float, a: int) } }`))
      .toThrow(/duplicate field 'a'/)
  })
})

// ─────────────────────────────────────────────────────────────
// Type aliases
// ─────────────────────────────────────────────────────────────

describe('adts — type aliases', () => {
  test('alias with positive bounds', () => {
    const p = parseProgram(`
      program X() {
        type Freq = float in [0, 20000]
      }
    `)
    expect(p.ports?.type_defs).toEqual([
      { kind: 'alias', name: 'Freq', base: 'float', bounds: [0, 20000] },
    ])
  })

  test('alias with negative bound', () => {
    const p = parseProgram(`
      program X() {
        type Sig = float in [-1, 1]
      }
    `)
    expect((p.ports!.type_defs![0] as AliasTypeDef).bounds).toEqual([-1, 1])
  })

  test('alias with null bounds', () => {
    const p = parseProgram(`
      program X() {
        type AnyFloat = float in [null, null]
      }
    `)
    expect((p.ports!.type_defs![0] as AliasTypeDef).bounds).toEqual([null, null])
  })

  test('alias missing `in` rejected', () => {
    expect(() => parseProgram(`program X() { type T = float }`)).toThrow(ParseError)
  })
})

// ─────────────────────────────────────────────────────────────
// Multiple type defs in the same program
// ─────────────────────────────────────────────────────────────

describe('adts — multiple type defs', () => {
  test('struct + enum + alias side by side, source order preserved', () => {
    const p = parseProgram(`
      program X() {
        struct Pair { a: float, b: float }
        enum Tag { A, B }
        type Freq = float in [0, 20000]
      }
    `)
    expect(p.ports?.type_defs).toHaveLength(3)
    expect((p.ports!.type_defs![0] as StructTypeDef).kind).toBe('struct')
    expect((p.ports!.type_defs![1] as SumTypeDef).kind).toBe('sum')
    expect((p.ports!.type_defs![2] as AliasTypeDef).kind).toBe('alias')
  })

  test('type defs and regular decls coexist in a body', () => {
    const p = parseProgram(`
      program X() -> (out: signal) {
        struct Pair { a: float, b: float }
        reg s: float = 0
        out = s
      }
    `)
    expect(p.ports?.type_defs).toHaveLength(1)
    expect(p.body.decls).toHaveLength(1)  // just the regDecl
    expect(p.body.assigns).toHaveLength(1)
  })
})

// ─────────────────────────────────────────────────────────────
// Tag construction in expression position
// ─────────────────────────────────────────────────────────────

describe('expressions — tag construction', () => {
  test('nullary tag uses bare ident (no `{}`)', () => {
    // `Variant` alone parses as a nameRef placeholder; the elaborator
    // converts to a tag if Variant is a known nullary sum variant.
    expect(parseExpr('Foo')).toEqual({ op: 'nameRef', name: 'Foo' })
  })

  test('tag with single-field payload', () => {
    expect(parseExpr('Some { value: 42 }')).toEqual({
      op: 'tag',
      variant: 'Some',
      payload: { value: 42 },
    })
  })

  test('tag with multi-field payload', () => {
    expect(parseExpr('Hz { freq: 440, gain: 0.5 }')).toEqual({
      op: 'tag',
      variant: 'Hz',
      payload: { freq: 440, gain: 0.5 },
    })
  })

  test('empty-payload tag via `Variant { }`', () => {
    expect(parseExpr('Empty { }')).toEqual({
      op: 'tag',
      variant: 'Empty',
    })
  })

  test('payload field can be a complex expression', () => {
    expect(parseExpr('Vec { x: a + b, y: c * 2 }')).toEqual({
      op: 'tag',
      variant: 'Vec',
      payload: {
        x: { op: 'add', args: [{ op: 'nameRef', name: 'a' }, { op: 'nameRef', name: 'b' }] },
        y: { op: 'mul', args: [{ op: 'nameRef', name: 'c' }, 2] },
      },
    })
  })

  test('lowercase ident with `{` does NOT trigger tag construction', () => {
    // `let { x: 1 } in x` is a let, not a tag
    const parsed = parseExpr('let { x: 1 } in x') as { op: string }
    expect(parsed.op).toBe('let')
  })

  test('duplicate payload field rejected', () => {
    expect(() => parseExpr('Foo { a: 1, a: 2 }'))
      .toThrow(/duplicate payload field 'a'/)
  })
})

// ─────────────────────────────────────────────────────────────
// Match expressions
// ─────────────────────────────────────────────────────────────

describe('expressions — match', () => {
  test('all-nullary match', () => {
    expect(parseExpr(`
      match v {
        Red => 1,
        Green => 2,
        Blue => 3
      }
    `)).toEqual({
      op: 'match',
      scrutinee: { op: 'nameRef', name: 'v' },
      arms: {
        Red:   { body: 1 },
        Green: { body: 2 },
        Blue:  { body: 3 },
      },
    })
  })

  test('match with single-field bindings', () => {
    expect(parseExpr(`
      match v {
        Some { value: x } => x + 1,
        None => 0
      }
    `)).toEqual({
      op: 'match',
      scrutinee: { op: 'nameRef', name: 'v' },
      arms: {
        Some: {
          bind: 'x',
          body: { op: 'add', args: [{ op: 'binding', name: 'x' }, 1] },
        },
        None: { body: 0 },
      },
    })
  })

  test('match with multi-field bindings emits bind as array', () => {
    expect(parseExpr(`
      match v {
        Hz { freq: f, gain: g } => f * g
      }
    `)).toEqual({
      op: 'match',
      scrutinee: { op: 'nameRef', name: 'v' },
      arms: {
        Hz: {
          bind: ['f', 'g'],
          body: {
            op: 'mul',
            args: [{ op: 'binding', name: 'f' }, { op: 'binding', name: 'g' }],
          },
        },
      },
    })
  })

  test('match arm bindings shadow outer scope', () => {
    // Outer `x` via let; inner `Some { value: x }` shadows it.
    const parsed = parseExpr(`
      let { x: 1 } in
      match v {
        Some { value: x } => x,
        None => x
      }
    ` ) as {
      in: {
        arms: {
          Some: { body: ExprNode }
          None: { body: ExprNode }
        }
      }
    }
    // Inner arm's `x` refers to the bound payload (shadowing the outer let).
    expect(parsed.in.arms.Some.body).toEqual({ op: 'binding', name: 'x' })
    // Outer arm's `x` refers to the outer let binding (still bound, since
    // `withScope` doesn't double-add an existing name).
    expect(parsed.in.arms.None.body).toEqual({ op: 'binding', name: 'x' })
  })

  test('duplicate arm rejected', () => {
    expect(() => parseExpr('match v { A => 1, A => 2 }'))
      .toThrow(/duplicate arm for variant 'A'/)
  })

  test('trailing comma after last arm allowed', () => {
    // commaList tolerates a trailing comma
    expect(() => parseExpr('match v { A => 1, B => 2, }')).not.toThrow()
  })

  test('match scrutinee can be a complex expression', () => {
    const parsed = parseExpr('match a + b { A => 0 }') as { scrutinee: { op: string } }
    expect(parsed.scrutinee.op).toBe('add')
  })
})

// ─────────────────────────────────────────────────────────────
// Full-program integration with ADTs
// ─────────────────────────────────────────────────────────────

describe('adts — program integration', () => {
  test('program with type defs, regs, instances, and a match expression', () => {
    const p = parseProgram(`
      program Synth(freq: freq = 220) -> (out: signal) {
        enum Mode { Sine, Saw }
        struct Pair { a: float, b: float }
        type Bipolar = float in [-1, 1]

        reg mode: Mode = Sine

        out = match mode {
          Sine => 1,
          Saw => 0
        }
      }
    `)
    expect(p.name).toBe('Synth')
    expect(p.ports?.type_defs).toHaveLength(3)
    expect(p.body.decls).toHaveLength(1)  // regDecl 'mode'
    expect(p.body.assigns).toHaveLength(1)
    const matchExpr = (p.body.assigns[0] as { expr: { op: string } }).expr
    expect(matchExpr.op).toBe('match')
  })

  test('struct/enum/type are forbidden when no body opts allow them', () => {
    // The body parser used by `parseBody` (statements.ts) without opts
    // raises a clear error rather than silently mistreating the keyword.
    // We check by parsing a program — declarations.ts wires the
    // typeDefHandler in, so the program path itself succeeds.
    expect(() => parseProgram(`program X() { struct S { a: float } }`)).not.toThrow()
  })
})

// Top-level type defs in tests need this import for ExprNode.
type ExprNode =
  | number
  | boolean
  | ExprNode[]
  | { op: string; [k: string]: unknown }
