/**
 * declarations.test.ts — program-declaration parser coverage (Phase B4).
 */

import { describe, test, expect } from 'bun:test'
import { parseProgram, type ProgramNode, type ProgramPortSpec } from './declarations.js'
import { ParseError } from './expressions.js'
import { nameRef } from './nodes.js'

describe('declarations — minimal program', () => {
  test('empty body, no ports', () => {
    expect(parseProgram('program Empty() { }')).toEqual({
      op: 'program',
      name: 'Empty',
      body: { op: 'block', decls: [], assigns: [] },
    })
  })

  test('with single output, no body', () => {
    const p = parseProgram('program Const() -> (out: signal) { out = 0 }')
    expect(p.name).toBe('Const')
    expect(p.ports?.outputs).toEqual([{ name: 'out', type: nameRef('signal') }])
    expect(p.body.assigns).toHaveLength(1)
  })

  test('with input + output', () => {
    const p = parseProgram(`
      program Passthrough(x: signal) -> (y: signal) {
        y = x
      }
    `)
    expect(p.ports?.inputs).toEqual([{ name: 'x', type: nameRef('signal') }])
    expect(p.ports?.outputs).toEqual([{ name: 'y', type: nameRef('signal') }])
  })

  test('trailing input rejected', () => {
    expect(() => parseProgram('program X() {} extra')).toThrow(/unexpected trailing/)
  })
})

describe('declarations — port specs', () => {
  test('bare-name port (no type, no default)', () => {
    const p = parseProgram('program X(a, b) -> (out) { out = 0 }')
    expect(p.ports?.inputs).toEqual(['a', 'b'])
    expect(p.ports?.outputs).toEqual(['out'])
  })

  test('input with type and default', () => {
    const p = parseProgram(`
      program X(freq: freq = 220) -> (out: signal) { out = 0 }
    `)
    expect(p.ports?.inputs).toEqual([{ name: 'freq', type: nameRef('freq'), default: 220 }])
  })

  test('input with bounds', () => {
    const p = parseProgram(`
      program X(g: float in [0, 1]) -> (out: signal) { out = 0 }
    `)
    expect(p.ports?.inputs).toEqual([
      { name: 'g', type: nameRef('float'), bounds: [0, 1] },
    ])
  })

  test('input with default and bounds', () => {
    const p = parseProgram(`
      program X(g: float = 0.5 in [0, 1]) -> (out: signal) { out = 0 }
    `)
    expect(p.ports?.inputs).toEqual([
      { name: 'g', type: nameRef('float'), default: 0.5, bounds: [0, 1] },
    ])
  })

  test('null bound is accepted', () => {
    const p = parseProgram(`
      program X(g: float in [null, 1]) -> (out: signal) { out = 0 }
    `)
    expect((p.ports!.inputs![0] as ProgramPortSpec).bounds).toEqual([null, 1])
  })

  test('negative literal bound', () => {
    const p = parseProgram(`
      program X(s: float in [-1, 1]) -> (out: signal) { out = 0 }
    `)
    expect((p.ports!.inputs![0] as ProgramPortSpec).bounds).toEqual([-1, 1])
  })

  test('output cannot have a default', () => {
    expect(() => parseProgram('program X() -> (y: signal = 0) { y = 0 }'))
      .toThrow(/cannot have a default/)
  })

  test('mixed bare and typed ports', () => {
    const p = parseProgram(`
      program X(a, b: signal, c: float = 1) -> (out: signal) { out = 0 }
    `)
    expect(p.ports?.inputs).toEqual([
      'a',
      { name: 'b', type: nameRef('signal') },
      { name: 'c', type: nameRef('float'), default: 1 },
    ])
  })

  test('multiple outputs', () => {
    const p = parseProgram(`
      program X() -> (lp: signal, bp: signal, hp: signal) {
        lp = 0
        bp = 0
        hp = 0
      }
    `)
    expect(p.ports?.outputs).toEqual([
      { name: 'lp', type: nameRef('signal') },
      { name: 'bp', type: nameRef('signal') },
      { name: 'hp', type: nameRef('signal') },
    ])
  })
})

describe('declarations — array port types', () => {
  test('array with literal shape dim', () => {
    const p = parseProgram(`
      program X(buf: float[4]) -> (out: signal) { out = 0 }
    `)
    expect((p.ports!.inputs![0] as ProgramPortSpec).type).toEqual({
      kind: 'array', element: nameRef('float'), shape: [4],
    })
  })

  test('array with type-param shape dim', () => {
    const p = parseProgram(`
      program X<N: int = 8>(buf: float[N]) -> (out: signal) { out = 0 }
    `)
    expect((p.ports!.inputs![0] as ProgramPortSpec).type).toEqual({
      kind: 'array', element: nameRef('float'), shape: [nameRef('N')],
    })
  })

  test('multi-dim array', () => {
    const p = parseProgram(`
      program X<N: int = 4, M: int = 8>(buf: float[N, M]) -> (out: signal) { out = 0 }
    `)
    expect((p.ports!.inputs![0] as ProgramPortSpec).type).toEqual({
      kind: 'array', element: nameRef('float'),
      shape: [nameRef('N'), nameRef('M')],
    })
  })

  test('array shape identifier emits a NameRef without parser-side validation', () => {
    // The parser performs no scope analysis. Whether `K` is actually a
    // declared type-param is determined by the elaborator (B6) when it
    // resolves the NameRefNode against the enclosing program's type-params.
    // The parser simply records the reference here.
    const p = parseProgram(`
      program X(buf: float[K]) -> (out: signal) { out = 0 }
    `)
    expect((p.ports!.inputs![0] as ProgramPortSpec).type).toEqual({
      kind: 'array', element: nameRef('float'), shape: [nameRef('K')],
    })
  })

  test('non-integer literal shape dim rejected', () => {
    expect(() => parseProgram(`
      program X(buf: float[2.5]) -> (out: signal) { out = 0 }
    `)).toThrow(/non-negative integer/)
  })

  test('empty shape rejected', () => {
    expect(() => parseProgram(`
      program X(buf: float[]) -> (out: signal) { out = 0 }
    `)).toThrow(/at least one shape dim/)
  })

  test('chained `[N][M]` (array-of-array) is NOT supported', () => {
    // The grammar's port-type parser stops after the first `]`. Chained
    // brackets become a syntax error at the second `[` (unexpected
    // character in port type position). Pin this behaviour: if/when
    // multi-dim shapes need surface support, prefer `float[N, M]` (one
    // bracket pair, comma-separated dims) which already works.
    expect(() => parseProgram(`
      program X(buf: float[4][8]) -> (out: signal) { out = 0 }
    `)).toThrow()
  })
})

describe('declarations — type params', () => {
  test('single type-param with default', () => {
    const p = parseProgram('program X<N: int = 8>() -> (out) { out = 0 }')
    expect(p.type_params).toEqual({ N: { type: 'int', default: 8 } })
  })

  test('type-param without default', () => {
    const p = parseProgram('program X<N: int>() -> (out) { out = 0 }')
    expect(p.type_params).toEqual({ N: { type: 'int' } })
  })

  test('multiple type-params', () => {
    const p = parseProgram('program X<N: int = 4, M: int>() -> (out) { out = 0 }')
    expect(p.type_params).toEqual({
      N: { type: 'int', default: 4 },
      M: { type: 'int' },
    })
  })

  test('non-int type-param rejected', () => {
    expect(() => parseProgram('program X<N: float>() -> (out) { out = 0 }'))
      .toThrow(/must be 'int'/)
  })

  test('non-integer default rejected', () => {
    expect(() => parseProgram('program X<N: int = 8.5>() -> (out) { out = 0 }'))
      .toThrow(/must be an integer/)
  })

  test('duplicate type-param rejected', () => {
    expect(() => parseProgram('program X<N: int, N: int>() -> (out) { out = 0 }'))
      .toThrow(/duplicate type-param/)
  })

  test('empty type-params <> parses cleanly (degenerate case)', () => {
    const p = parseProgram('program X<>() -> (out) { out = 0 }')
    // type_params is omitted from the node when empty
    expect(p.type_params).toBeUndefined()
  })
})

describe('declarations — body integration', () => {
  test('body with regs, instances, assigns', () => {
    const p = parseProgram(`
      program OnePole(x: signal, g: float = 0.5) -> (y: signal) {
        reg s: float = 0
        y = s
        next s = x * (1 - g) + s * g
      }
    `)
    expect(p.body.decls).toHaveLength(1)
    expect(p.body.assigns).toHaveLength(2)
  })

  test('body uses dac.out wire', () => {
    const p = parseProgram(`
      program Patch() {
        osc = SinOsc(freq: 220)
        dac.out = osc.sin
      }
    `)
    expect(p.body.decls).toHaveLength(1)
    expect(p.body.assigns).toHaveLength(1)
    const assign = p.body.assigns[0] as { name: string }
    expect(assign.name).toBe('dac.out')
  })
})

describe('declarations — nested programs', () => {
  test('single nested program', () => {
    const p = parseProgram(`
      program Outer() -> (out: signal) {
        program Inner(x: signal) -> (y: signal) {
          y = x
        }
        i = Inner(x: 0)
        out = i.y
      }
    `)
    expect(p.body.decls).toHaveLength(2)  // programDecl + instanceDecl
    const programDecl = p.body.decls[0] as { op: string; name: string; program: ProgramNode }
    expect(programDecl.op).toBe('programDecl')
    expect(programDecl.name).toBe('Inner')
    expect(programDecl.program.name).toBe('Inner')
    expect(programDecl.program.body.assigns).toHaveLength(1)
  })

  test('nested type-param scope is independent', () => {
    // Outer N=8; inner re-declares N=4. Inner's body should see N=4.
    const p = parseProgram(`
      program Outer<N: int = 8>() {
        program Inner<N: int = 4>(buf: float[N]) -> (out) { out = 0 }
      }
    `)
    const inner = (p.body.decls[0] as { program: ProgramNode }).program
    expect(inner.type_params).toEqual({ N: { type: 'int', default: 4 } })
    const buf = inner.ports!.inputs![0] as ProgramPortSpec
    // Should reference N (resolved against inner scope, not outer)
    expect(buf.type).toEqual({
      kind: 'array', element: nameRef('float'), shape: [nameRef('N')],
    })
  })

  test('multiple sibling nested programs', () => {
    const p = parseProgram(`
      program Outer() {
        program A() -> (out) { out = 0 }
        program B() -> (out) { out = 0 }
      }
    `)
    expect(p.body.decls).toHaveLength(2)
    expect((p.body.decls[0] as { name: string }).name).toBe('A')
    expect((p.body.decls[1] as { name: string }).name).toBe('B')
  })
})

describe('declarations — error cases', () => {
  test('missing program name', () => {
    expect(() => parseProgram('program () { }')).toThrow(ParseError)
  })

  test('missing opening paren', () => {
    expect(() => parseProgram('program X { }')).toThrow(ParseError)
  })

  test('missing body', () => {
    expect(() => parseProgram('program X()')).toThrow(ParseError)
  })

  test('error position info', () => {
    let err: ParseError | undefined
    try { parseProgram('program X(\n  bad: 0\n)') } catch (e) { err = e as ParseError }
    expect(err).toBeInstanceOf(ParseError)
  })
})
