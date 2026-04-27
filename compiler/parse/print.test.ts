/**
 * print.test.ts — pretty-printer round-trip tests (Phase B7).
 *
 * Round-trip property:
 *
 *     parseProgram(printProgram(parseProgram(text))) === parseProgram(text)
 *
 * The printer doesn't try to preserve original whitespace, ordering of
 * synonymous separators, or comments. It produces *canonical* `.trop`.
 * Two textually-distinct sources that parse to the same tree should
 * print to the same text.
 */

import { describe, test, expect } from 'bun:test'
import { parseProgram } from './declarations.js'
import { parseExpr } from './expressions.js'
import { parseBody } from './statements.js'
import { extractMarkdown } from './markdown.js'
import { printProgram, printExpr, printProgramDecl } from './print.js'

/** Strip the markdown wrapper to get just the program-decl text. */
function unwrap(printed: string): string {
  const ext = extractMarkdown(printed)
  if (ext.blocks.length !== 1) {
    throw new Error(`expected 1 tropical block, got ${ext.blocks.length}`)
  }
  return ext.blocks[0].source
}

/** Round-trip helper: parse, print, parse again, return both trees. */
function roundTrip(src: string): { first: ReturnType<typeof parseProgram>; second: ReturnType<typeof parseProgram> } {
  const first = parseProgram(src)
  const printed = printProgram(first)
  const reparseSource = unwrap(printed)
  const second = parseProgram(reparseSource)
  return { first, second }
}

/** Idempotent-printing helper: parse, print, print-of-reparse, assert
 *  the second print equals the first. */
function idempotent(src: string): void {
  const first = parseProgram(src)
  const printed1 = printProgram(first)
  const second = parseProgram(unwrap(printed1))
  const printed2 = printProgram(second)
  expect(printed2).toBe(printed1)
}

// ─────────────────────────────────────────────────────────────
// Basic round-trips
// ─────────────────────────────────────────────────────────────

describe('printer — basic programs', () => {
  test('empty program', () => {
    const { first, second } = roundTrip('program X() { }')
    expect(second).toEqual(first)
  })

  test('with single output', () => {
    const { first, second } = roundTrip(`
      program Const() -> (out: signal) { out = 0 }
    `)
    expect(second).toEqual(first)
  })

  test('input + output passthrough', () => {
    const { first, second } = roundTrip(`
      program P(x: signal) -> (y: signal) { y = x }
    `)
    expect(second).toEqual(first)
  })

  test('printed output is a literate-program markdown block', () => {
    const printed = printProgram(parseProgram('program X() { }'))
    expect(printed).toContain('```tropical')
    expect(printed).toContain('```\n')
  })
})

// ─────────────────────────────────────────────────────────────
// Port specs
// ─────────────────────────────────────────────────────────────

describe('printer — ports', () => {
  test('typed inputs and outputs', () => {
    idempotent(`
      program X(freq: freq = 220, x: signal) -> (out: signal, lp: signal) {
        out = x
        lp = x
      }
    `)
  })

  test('bare-name ports', () => {
    idempotent(`
      program X(a, b) -> (out) { out = a }
    `)
  })

  test('bounds on inputs', () => {
    idempotent(`
      program X(g: float in [0, 1]) -> (out: signal) { out = 0 }
    `)
  })

  test('null-bound', () => {
    idempotent(`
      program X(g: float in [null, 1]) -> (out: signal) { out = 0 }
    `)
  })

  test('negative-literal bound', () => {
    idempotent(`
      program X(s: float in [-1, 1]) -> (out: signal) { out = 0 }
    `)
  })

  test('default + bounds', () => {
    idempotent(`
      program X(g: float = 0.5 in [0, 1]) -> (out: signal) { out = 0 }
    `)
  })

  test('array port type with type-param', () => {
    idempotent(`
      program X<N: int = 4>(buf: float[N]) -> (out: signal) { out = 0 }
    `)
  })

  test('multi-dim array', () => {
    idempotent(`
      program X<N: int = 2, M: int = 3>(buf: float[N, M]) -> (out: signal) { out = 0 }
    `)
  })
})

// ─────────────────────────────────────────────────────────────
// Type params
// ─────────────────────────────────────────────────────────────

describe('printer — type params', () => {
  test('single, with default', () => {
    idempotent(`program X<N: int = 8>() -> (out) { out = 0 }`)
  })

  test('multiple, mixed defaults', () => {
    idempotent(`program X<N: int = 4, M: int>() -> (out) { out = 0 }`)
  })
})

// ─────────────────────────────────────────────────────────────
// Body decls
// ─────────────────────────────────────────────────────────────

describe('printer — body decls', () => {
  test('regDecl with type', () => {
    idempotent(`
      program X() -> (out: signal) {
        reg s: float = 0
        out = s
        next s = s + 1
      }
    `)
  })

  test('regDecl without type', () => {
    idempotent(`
      program X() -> (out: signal) {
        reg s = 0
        out = s
      }
    `)
  })

  test('delayDecl', () => {
    idempotent(`
      program X(x: signal) -> (out: signal) {
        delay z = x init 0
        out = z
      }
    `)
  })

  test('paramDecl smoothed with default', () => {
    idempotent(`
      program X() -> (out: signal) {
        param cutoff: smoothed = 1000
        out = cutoff
      }
    `)
  })

  test('paramDecl trigger', () => {
    idempotent(`
      program X() -> (out: signal) {
        param fire: trigger
        out = fire
      }
    `)
  })

  test('instanceDecl with type-args + inputs', () => {
    idempotent(`
      program Outer() -> (out: signal) {
        program Inner<N: int = 4>(x: signal = 0) -> (y: signal) { y = x }
        i = Inner<N=8>(x: 1)
        out = i.y
      }
    `)
  })
})

// ─────────────────────────────────────────────────────────────
// Body assigns
// ─────────────────────────────────────────────────────────────

describe('printer — body assigns', () => {
  test('outputAssign + nextUpdate', () => {
    idempotent(`
      program X(x: signal) -> (out: signal) {
        reg s: float = 0
        out = s + x
        next s = s
      }
    `)
  })

  test('dac.out wire', () => {
    idempotent(`
      program Patch() {
        program Osc() -> (out: signal) { out = 0 }
        o = Osc()
        dac.out = o.out
      }
    `)
  })
})

// ─────────────────────────────────────────────────────────────
// Expression-level round-trips
// ─────────────────────────────────────────────────────────────

describe('printer — expressions', () => {
  function exprIdempotent(src: string): void {
    const a = parseExpr(src)
    const printed = printExpr(a)
    const b = parseExpr(printed)
    expect(b).toEqual(a)
  }

  test('arithmetic with precedence', () => {
    exprIdempotent('a + b * c')
    exprIdempotent('(a + b) * c')
    exprIdempotent('a + b + c')   // left-assoc: a+b first
    exprIdempotent('a - b - c')
    exprIdempotent('-a * b')
  })

  test('comparison + logical', () => {
    exprIdempotent('a < b && c > d')
    exprIdempotent('a == b || c != d')
  })

  test('bitwise', () => {
    exprIdempotent('a & b | c')
    exprIdempotent('a << 2 + 1')
  })

  test('unary', () => {
    exprIdempotent('-x')
    exprIdempotent('!flag')
    exprIdempotent('~bits')
    exprIdempotent('-(a + b)')
  })

  test('dotted port refs', () => {
    exprIdempotent('osc.sin')
    exprIdempotent('a.b + c.d')
  })

  test('indexing', () => {
    exprIdempotent('a[i]')
    exprIdempotent('osc.out[0]')
  })

  test('function calls', () => {
    exprIdempotent('clamp(x, 0, 1)')
    exprIdempotent('sample_rate()')
    exprIdempotent('sqrt(x * x)')
  })

  test('let binding', () => {
    exprIdempotent('let { x: 1 } in x + x')
    exprIdempotent('let { x: 1; y: 2 } in x + y')
  })

  test('combinators', () => {
    exprIdempotent('fold(a, 0, (acc, e) => acc + e)')
    exprIdempotent('scan(a, 0, (acc, e) => acc + e)')
    exprIdempotent('generate(8, (i) => i * i)')
    exprIdempotent('iterate(4, 1, (x) => x * 2)')
    exprIdempotent('chain(3, 0, (x) => x + 1)')
    exprIdempotent('map2(a, (e) => e * 2)')
    exprIdempotent('zipWith(a, b, (x, y) => x + y)')
  })

  test('nested combinators', () => {
    exprIdempotent('fold([1, 2, 3], 0, (a, e) => a + e * e)')
  })

  test('array literal', () => {
    exprIdempotent('[1, 2, 3]')
    exprIdempotent('[1, -0.5, 0.25]')
  })

  test('tag construction', () => {
    exprIdempotent('Some { value: 42 }')
    exprIdempotent('Hz { freq: 440, gain: 0.5 }')
    exprIdempotent('Empty { }')
  })

  test('match — all-nullary', () => {
    exprIdempotent('match v { Red => 1, Green => 2, Blue => 3 }')
  })

  test('match — with bindings', () => {
    exprIdempotent('match v { Some { value: x } => x, None => 0 }')
  })

  test('boolean literals', () => {
    exprIdempotent('true')
    exprIdempotent('false')
    exprIdempotent('!true')
  })
})

// ─────────────────────────────────────────────────────────────
// Type defs
// ─────────────────────────────────────────────────────────────

describe('printer — type defs', () => {
  test('struct', () => {
    idempotent(`
      program X() -> (out: signal) {
        struct Pair { a: float, b: int }
        out = 0
      }
    `)
  })

  test('enum with mixed nullary/payload', () => {
    idempotent(`
      program X() -> (out: signal) {
        enum Maybe { Some(value: float), None }
        out = 0
      }
    `)
  })

  test('alias with bounds', () => {
    idempotent(`
      program X() -> (out: signal) {
        type Bipolar = float in [-1, 1]
        out = 0
      }
    `)
  })

  test('multiple type defs side by side', () => {
    idempotent(`
      program X() -> (out: signal) {
        struct Pair { a: float, b: float }
        enum Mode { On, Off }
        type Freq = float in [0, 20000]
        out = 0
      }
    `)
  })
})

// ─────────────────────────────────────────────────────────────
// Nested programs
// ─────────────────────────────────────────────────────────────

describe('printer — nested programs', () => {
  test('single nested program', () => {
    idempotent(`
      program Outer() -> (out: signal) {
        program Inner(x: signal) -> (y: signal) { y = x }
        i = Inner(x: 0)
        out = i.y
      }
    `)
  })

  test('nested with type-params', () => {
    idempotent(`
      program Outer<N: int = 4>() -> (out: signal) {
        program Inner<M: int = 8>(buf: float[M]) -> (out) { out = 0 }
        i = Inner<M=2>(buf: [0, 0])
        out = i.out
      }
    `)
  })
})

// ─────────────────────────────────────────────────────────────
// Realistic stdlib-shaped programs (round-trip)
// ─────────────────────────────────────────────────────────────

describe('printer — stdlib-shaped patterns', () => {
  test('OnePole-shaped', () => {
    idempotent(`
      program OnePole(x: signal, g: float = 0.5) -> (y: signal) {
        reg s: float = 0
        y = s
        next s = x * (1 - g) + s * g
      }
    `)
  })

  test('LadderFilter-shaped (instance composition)', () => {
    idempotent(`
      program LadderFilter(cutoff: signal, x: signal) -> (out: signal) {
        program OnePole(x: signal, g: signal) -> (y: signal) {
          reg s: float = 0
          y = s
          next s = x * (1 - g) + s * g
        }
        delay z = x init 0
        lp1 = OnePole(x: x - 4 * z, g: cutoff)
        lp2 = OnePole(x: lp1.y, g: cutoff)
        out = lp2.y
      }
    `)
  })

  test('patch with dac.out', () => {
    idempotent(`
      program Patch() {
        program Osc() -> (out: signal) { out = 0 }
        o = Osc()
        dac.out = o.out
      }
    `)
  })
})

// ─────────────────────────────────────────────────────────────
// Canonicalization: textually-different sources that parse-equal
// should print-equal.
// ─────────────────────────────────────────────────────────────

describe('printer — canonicalization', () => {
  test('extra whitespace is normalized', () => {
    const a = parseProgram('program  X (  ) ->  (out: signal) {  out = 0  }')
    const b = parseProgram('program X() -> (out: signal) { out = 0 }')
    expect(printProgram(a)).toBe(printProgram(b))
  })

  test('semicolons vs newlines normalize identically', () => {
    const a = parseProgram(`
      program X() -> (out: signal) { reg s = 0; out = s; next s = s }
    `)
    const b = parseProgram(`
      program X() -> (out: signal) {
        reg s = 0
        out = s
        next s = s
      }
    `)
    expect(printProgram(a)).toBe(printProgram(b))
  })
})

// ─────────────────────────────────────────────────────────────
// printProgramDecl (no markdown wrapper)
// ─────────────────────────────────────────────────────────────

describe('printer — printProgramDecl', () => {
  test('produces the program-decl text without markdown fences', () => {
    const prog = parseProgram('program X() -> (out: signal) { out = 0 }')
    const inner = printProgramDecl(prog, 0)
    expect(inner).not.toContain('```')
    expect(inner.startsWith('program X')).toBe(true)
  })
})
