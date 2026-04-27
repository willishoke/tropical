/**
 * shared.ts — parser plumbing common to expressions / statements / declarations.
 *
 * The three parser layers all walk the same `Tok[]` stream with the same
 * lookahead idioms (peek, consume, eat) and the same comma-separated-list
 * shapes. This module gives them one set of helpers so each layer stays
 * focused on its grammar.
 */

import type { Tok, TokKind } from './lexer.js'

export class ParseError extends Error {
  constructor(message: string, public tok: Tok) {
    super(`${tok.line}:${tok.col}: ${message}`)
  }
}

/** Minimal cursor that all parser contexts extend. Helpers operate on this
 *  surface only — layer-specific state (binders, type-params, etc.) lives
 *  on the concrete Ctx and is invisible here. */
export interface Cursor {
  toks: Tok[]
  i: number
}

export const peek = (c: Cursor, offset = 0): Tok =>
  c.toks[Math.min(c.i + offset, c.toks.length - 1)]

export function consume(c: Cursor, kind: TokKind, what?: string): Tok {
  const t = c.toks[c.i]
  if (t.kind !== kind) {
    throw new ParseError(`expected ${what ?? kind}, got ${formatTok(t)}`, t)
  }
  c.i++
  return t
}

export function eat(c: Cursor, kind: TokKind): Tok | null {
  const t = c.toks[c.i]
  if (t.kind !== kind) return null
  c.i++
  return t
}

export function formatTok(t: Tok): string {
  if (t.kind === 'eof') return 'end of input'
  if (t.value !== undefined) return `${t.kind}(${JSON.stringify(t.value)})`
  return `'${t.kind}'`
}

/** Parse a comma-separated list, stopping at `terminator`. Does not consume
 *  the terminator. Tolerates an empty list and a trailing comma. */
export function commaList<T>(c: Cursor, terminator: TokKind, parse: () => T): T[] {
  const items: T[] = []
  if (peek(c).kind === terminator) return items
  items.push(parse())
  while (eat(c, ',')) {
    if (peek(c).kind === terminator) break
    items.push(parse())
  }
  return items
}

/** Run `body` with `names` added to a scope set, then remove only the names
 *  this call actually added (so nested binders nest cleanly). Replaces the
 *  manual `added: string[]` + try/finally idiom. */
export function withScope<T>(scope: Set<string>, names: Iterable<string>, body: () => T): T {
  const added: string[] = []
  for (const n of names) {
    if (!scope.has(n)) {
      scope.add(n)
      added.push(n)
    }
  }
  try {
    return body()
  } finally {
    for (const n of added) scope.delete(n)
  }
}

/** True iff `t` is an `ident` token whose value equals `name` — for
 *  contextual keywords like `init`, `null`, `dac`, `smoothed`, `trigger`. */
export const isContextualKw = (t: Tok, name: string): boolean =>
  t.kind === 'ident' && t.value === name
