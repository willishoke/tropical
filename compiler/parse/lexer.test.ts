/**
 * lexer.test.ts — token-level coverage for the .trop lexer (Phase B1).
 */

import { describe, test, expect } from 'bun:test'
import { tokenize, formatTok, LexError, type Tok, type TokKind } from './lexer.js'

function kinds(toks: Tok[]): TokKind[] {
  return toks.map(t => t.kind)
}

function values(toks: Tok[]): Array<unknown> {
  return toks.map(t => (t.value !== undefined ? t.value : t.kind))
}

describe('lexer — literals', () => {
  test('integer', () => {
    const toks = tokenize('42')
    expect(kinds(toks)).toEqual(['num', 'eof'])
    expect(toks[0].value).toBe(42)
  })

  test('float', () => {
    const toks = tokenize('3.14')
    expect(toks[0].value).toBeCloseTo(3.14)
  })

  test('exponent (lowercase e, no sign)', () => {
    const toks = tokenize('1e6')
    expect(toks[0].value).toBe(1e6)
  })

  test('exponent (uppercase E, signed)', () => {
    const toks = tokenize('1.5E-3')
    expect(toks[0].value).toBeCloseTo(1.5e-3)
  })

  test('malformed exponent (missing digits) errors', () => {
    expect(() => tokenize('1e')).toThrow(LexError)
  })

  test('boolean literals tokenize as their own kinds', () => {
    expect(kinds(tokenize('true'))).toEqual(['true', 'eof'])
    expect(kinds(tokenize('false'))).toEqual(['false', 'eof'])
  })

  test('double-quoted string', () => {
    const toks = tokenize('"hello"')
    expect(toks[0].kind).toBe('string')
    expect(toks[0].value).toBe('hello')
  })

  test('single-quoted string', () => {
    expect(tokenize("'world'")[0].value).toBe('world')
  })

  test('escape sequences', () => {
    expect(tokenize('"a\\nb"')[0].value).toBe('a\nb')
    expect(tokenize('"a\\tb"')[0].value).toBe('a\tb')
    expect(tokenize('"a\\\\b"')[0].value).toBe('a\\b')
  })

  test('unterminated string errors', () => {
    expect(() => tokenize('"unfinished')).toThrow(LexError)
  })

  test('newline inside string errors', () => {
    expect(() => tokenize('"line1\nline2"')).toThrow(LexError)
  })
})

describe('lexer — identifiers and keywords', () => {
  test('identifier', () => {
    const toks = tokenize('foo_bar')
    expect(toks[0].kind).toBe('ident')
    expect(toks[0].value).toBe('foo_bar')
  })

  test('identifier starting with underscore', () => {
    expect(tokenize('_x')[0].value).toBe('_x')
  })

  test('all declaration keywords', () => {
    const kws = ['program', 'reg', 'delay', 'param', 'next',
                 'let', 'in', 'if', 'else', 'match',
                 'struct', 'enum', 'type']
    for (const kw of kws) {
      const toks = tokenize(kw)
      expect(toks[0].kind).toBe(kw as TokKind)
      expect(toks[0].value).toBeUndefined()
    }
  })

  test('keyword as part of identifier is identifier', () => {
    expect(tokenize('programs')[0].kind).toBe('ident')
    expect(tokenize('let_in')[0].kind).toBe('ident')
  })
})

describe('lexer — operators', () => {
  test('arithmetic', () => {
    expect(kinds(tokenize('a + b - c * d / e % f')))
      .toEqual(['ident', '+', 'ident', '-', 'ident', '*', 'ident', '/', 'ident', '%', 'ident', 'eof'])
  })

  test('comparison (longest-match wins for two-char)', () => {
    expect(kinds(tokenize('< <= > >= == !=')))
      .toEqual(['<', '<=', '>', '>=', '==', '!=', 'eof'])
  })

  test('arrow tokens disambiguated from = and >', () => {
    expect(kinds(tokenize('=>'))).toEqual(['=>', 'eof'])
    expect(kinds(tokenize('->'))).toEqual(['->', 'eof'])
    expect(kinds(tokenize('= >'))).toEqual(['=', '>', 'eof'])
    expect(kinds(tokenize('- >'))).toEqual(['-', '>', 'eof'])
  })

  test('bitwise vs logical', () => {
    expect(kinds(tokenize('&& & || | ^ ~ !')))
      .toEqual(['&&', '&', '||', '|', '^', '~', '!', 'eof'])
  })

  test('shifts', () => {
    expect(kinds(tokenize('<< >>'))).toEqual(['<<', '>>', 'eof'])
  })
})

describe('lexer — punctuation', () => {
  test('parens, brackets, braces', () => {
    expect(kinds(tokenize('([{}])'))).toEqual(['(', '[', '{', '}', ']', ')', 'eof'])
  })

  test('separators and dot', () => {
    expect(kinds(tokenize('a, b. c; d : e')))
      .toEqual(['ident', ',', 'ident', '.', 'ident', ';', 'ident', ':', 'ident', 'eof'])
  })
})

describe('lexer — comments', () => {
  test('line comment', () => {
    expect(kinds(tokenize('a // tail comment\nb')))
      .toEqual(['ident', 'ident', 'eof'])
  })

  test('block comment', () => {
    expect(kinds(tokenize('a /* skip me */ b')))
      .toEqual(['ident', 'ident', 'eof'])
  })

  test('block comment spans newlines', () => {
    const toks = tokenize('a\n/* line1\nline2 */\nb')
    expect(kinds(toks)).toEqual(['ident', 'ident', 'eof'])
    // Position of `b` should reflect the lines consumed by the comment
    expect(toks[1].line).toBe(4)
  })

  test('unterminated block comment errors', () => {
    expect(() => tokenize('a /* incomplete')).toThrow(LexError)
  })
})

describe('lexer — position info', () => {
  test('line and column tracking', () => {
    const src = 'a\n  b\n   c'
    const toks = tokenize(src)
    expect(toks[0]).toMatchObject({ kind: 'ident', line: 1, col: 1 })
    expect(toks[1]).toMatchObject({ kind: 'ident', line: 2, col: 3 })
    expect(toks[2]).toMatchObject({ kind: 'ident', line: 3, col: 4 })
  })

  test('eof position is past last token', () => {
    const toks = tokenize('foo')
    expect(toks.at(-1)?.kind).toBe('eof')
    expect(toks.at(-1)?.pos).toBe(3)
  })

  test('CR is treated as whitespace (not a column reset)', () => {
    const toks = tokenize('a\r\nb')
    expect(toks[1]).toMatchObject({ kind: 'ident', line: 2, col: 1 })
  })
})

describe('lexer — error position info on bad input', () => {
  test('unexpected character carries line/col', () => {
    let err: LexError | undefined
    try { tokenize('a\n  @b') } catch (e) { err = e as LexError }
    expect(err).toBeInstanceOf(LexError)
    expect(err?.line).toBe(2)
    expect(err?.col).toBe(3)
  })
})

describe('lexer — sample expressions', () => {
  test('simple infix expression', () => {
    expect(kinds(tokenize('3 * x + 1')))
      .toEqual(['num', '*', 'ident', '+', 'num', 'eof'])
  })

  test('dotted port reference', () => {
    expect(kinds(tokenize('osc.sin')))
      .toEqual(['ident', '.', 'ident', 'eof'])
  })

  test('array literal', () => {
    const toks = tokenize('[1, 2.5, 3]')
    expect(kinds(toks)).toEqual(['[', 'num', ',', 'num', ',', 'num', ']', 'eof'])
  })

  test('lambda-style combinator', () => {
    expect(kinds(tokenize('fold(a, 0, (acc, e) => acc + e)')))
      .toEqual(['ident', '(', 'ident', ',', 'num', ',', '(', 'ident', ',', 'ident', ')', '=>', 'ident', '+', 'ident', ')', 'eof'])
  })

  test('declaration header', () => {
    expect(kinds(tokenize('program OnePole<N: int = 8>(x: signal) -> (y: signal)')))
      .toEqual([
        'program', 'ident', '<', 'ident', ':', 'ident', '=', 'num', '>',
        '(', 'ident', ':', 'ident', ')', '->',
        '(', 'ident', ':', 'ident', ')', 'eof',
      ])
  })

  test('let binding', () => {
    expect(kinds(tokenize('let x = 1 in x + x')))
      .toEqual(['let', 'ident', '=', 'num', 'in', 'ident', '+', 'ident', 'eof'])
  })

  test('match expression', () => {
    expect(kinds(tokenize('match v { Foo => 1, Bar => 2 }')))
      .toEqual(['match', 'ident', '{', 'ident', '=>', 'num', ',', 'ident', '=>', 'num', '}', 'eof'])
  })
})

describe('lexer — formatTok', () => {
  test('value-bearing tokens get value in format', () => {
    expect(formatTok(tokenize('foo')[0])).toBe('ident("foo")')
    expect(formatTok(tokenize('42')[0])).toBe('num(42)')
    expect(formatTok(tokenize('"hi"')[0])).toBe('string("hi")')
  })

  test('keyword tokens format as their kind', () => {
    expect(formatTok(tokenize('let')[0])).toBe('let')
    expect(formatTok(tokenize('=>')[0])).toBe('=>')
  })
})
