/**
 * lexer.ts — tokenizer for the `.trop` surface language.
 *
 * Scope: tokens for stages 1+2+3+4 from `design/surface-syntax.md`:
 *   - Expressions (numbers, idents, infix ops, function calls, dot, indexing)
 *   - Statements (=, : type annotations, next, separators)
 *   - Program declarations (program, reg, delay, param, ports, generics)
 *   - ADTs (struct, enum, type, match)
 *
 * Output is a flat token stream; the parser layers above (expressions,
 * statements, declarations) consume it via lookahead.
 *
 * Design: an unfold over `(src, ctx)`. Each step picks the first matching
 * rule from a static table, advances the context immutably, and (sometimes)
 * yields a token. `tokenize` is just `[...lex(src)]` — collection comes from
 * spread, not push. Position tracking (line/col, 1-indexed) is recovered
 * from the consumed span by counting newlines; only whitespace and block
 * comments can ever cross lines.
 */

export type TokKind =
  // Literals + identifiers
  | 'num' | 'ident' | 'string'
  // Boolean keywords (lexed as their own kinds for parser convenience)
  | 'true' | 'false'
  // Binders + control flow
  | 'let' | 'in' | 'if' | 'else' | 'match'
  // Declarations
  | 'program' | 'reg' | 'delay' | 'param' | 'next'
  // ADTs
  | 'struct' | 'enum' | 'type'
  // Punctuation: parens, brackets, braces
  | '(' | ')' | '[' | ']' | '{' | '}'
  // Punctuation: separators, accessors
  | ',' | '.' | ';' | ':'
  // Assignment + arrows
  | '=' | '=>' | '->'
  // Arithmetic
  | '+' | '-' | '*' | '/' | '%'
  // Comparison
  | '<' | '<=' | '>' | '>=' | '==' | '!='
  // Bitwise
  | '<<' | '>>' | '&' | '|' | '^' | '~'
  // Logical
  | '&&' | '||' | '!'
  // End of stream
  | 'eof'

/** A lexed token. `value` is set for `num` (number), `ident` (name string),
 *  and `string` (the unquoted, escape-processed contents). */
export interface Tok {
  kind: TokKind
  value?: string | number
  pos: number   // byte offset into the source
  line: number  // 1-indexed
  col: number   // 1-indexed
}

const KEYWORDS: Record<string, TokKind> = {
  true: 'true', false: 'false',
  let: 'let', in: 'in',
  if: 'if', else: 'else', match: 'match',
  program: 'program',
  reg: 'reg', delay: 'delay', param: 'param',
  next: 'next',
  struct: 'struct', enum: 'enum', type: 'type',
}

const ESCAPES: Record<string, string> = {
  n: '\n', t: '\t', r: '\r', '\\': '\\', "'": "'", '"': '"',
}

export class LexError extends Error {
  constructor(message: string, public pos: number, public line: number, public col: number) {
    super(`${line}:${col}: ${message}`)
  }
}

interface LexCtx { i: number; line: number; lineStart: number }
type Emit = Pick<Tok, 'kind' | 'value'>
type Match = { length: number; tok?: Emit }
type Rule = (src: string, ctx: LexCtx) => Match | null

function errAt(msg: string, ctx: LexCtx, offset = ctx.i): never {
  throw new LexError(msg, offset, ctx.line, offset - ctx.lineStart + 1)
}

/** Wraps a sticky regex into a Rule. If `emit` is omitted the match is skipped. */
const re = (pattern: RegExp, emit?: (m: string) => Emit): Rule => {
  if (!pattern.sticky) throw new Error(`lexer rule regex must be sticky: ${pattern}`)
  return (src, ctx) => {
    pattern.lastIndex = ctx.i
    const m = pattern.exec(src)
    if (!m || m.index !== ctx.i) return null
    return emit ? { length: m[0].length, tok: emit(m[0]) } : { length: m[0].length }
  }
}

// --- skip rules (whitespace / comments emit no token) ---------------------

const skipSpace = re(/[ \t\r\n]+/y)
const skipLineComment = re(/\/\/[^\n]*/y)

const skipBlockComment: Rule = (src, ctx) => {
  if (src[ctx.i] !== '/' || src[ctx.i + 1] !== '*') return null
  const end = src.indexOf('*/', ctx.i + 2)
  if (end < 0) errAt('unterminated block comment', ctx)
  return { length: end - ctx.i + 2 }
}

// --- value-bearing rules --------------------------------------------------

// Greedy: matches `1e` (with empty exponent digits) so we can report a
// targeted error instead of letting the trailing `e` re-tokenize as an ident.
const NUM_RE = /(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)(?:[eE][+-]?[0-9]*)?/y

const number: Rule = (src, ctx) => {
  NUM_RE.lastIndex = ctx.i
  const m = NUM_RE.exec(src)
  if (!m || m.index !== ctx.i) return null
  const text = m[0]
  const eAt = text.search(/[eE]/)
  if (eAt >= 0 && !/[0-9]$/.test(text)) {
    errAt(`malformed number (exponent missing digits): ${text}`, ctx)
  }
  return { length: text.length, tok: { kind: 'num', value: Number(text) } }
}

const identOrKeyword = re(/[A-Za-z_][A-Za-z0-9_]*/y, m =>
  KEYWORDS[m] ? { kind: KEYWORDS[m] } : { kind: 'ident', value: m },
)

// Body: any backslash + char (escape), or any non-quote/backslash/newline char.
const STRING_RE = /"((?:\\[^]|[^"\\\n])*)"|'((?:\\[^]|[^'\\\n])*)'/y

const stringLit: Rule = (src, ctx) => {
  const c = src[ctx.i]
  if (c !== '"' && c !== "'") return null
  STRING_RE.lastIndex = ctx.i
  const m = STRING_RE.exec(src)
  if (!m || m.index !== ctx.i) errAt('unterminated string literal', ctx)
  const body = (m[1] ?? m[2])!
  // Body offset = ctx.i + 1 (skip opening quote); used to point escape errors precisely.
  const value = body.replace(/\\(.)/g, (_, ch: string, idx: number) => {
    const replacement = ESCAPES[ch]
    if (replacement === undefined) errAt(`unknown escape: \\${ch}`, ctx, ctx.i + 1 + idx)
    return replacement
  })
  return { length: m[0].length, tok: { kind: 'string', value } }
}

const punct2 = re(/<=|>=|==|!=|<<|>>|&&|\|\||=>|->/y, m => ({ kind: m as TokKind }))
const punct1 = re(/[()\[\]{},.;:=+\-*/%<>&|^~!]/y, m => ({ kind: m as TokKind }))

// Order: skips first, then literals, then ident/keyword, then strings, then
// longest-match punctuation before single-char.
const RULES: ReadonlyArray<Rule> = [
  skipSpace,
  skipLineComment,
  skipBlockComment,
  number,
  identOrKeyword,
  stringLit,
  punct2,
  punct1,
]

/** First-non-null map: applies `f` and returns the first defined result. */
function firstMatch(src: string, ctx: LexCtx): Match | null {
  for (const rule of RULES) {
    const m = rule(src, ctx)
    if (m) return m
  }
  return null
}

/** Advance the context by `length` chars, immutably. Recomputes line/lineStart
 *  from the consumed span — cheap because newlines only occur in skip rules. */
function advance(src: string, ctx: LexCtx, length: number): LexCtx {
  const span = src.slice(ctx.i, ctx.i + length)
  const last = span.lastIndexOf('\n')
  if (last < 0) return { i: ctx.i + length, line: ctx.line, lineStart: ctx.lineStart }
  const newlines = (span.match(/\n/g) as string[]).length
  return {
    i: ctx.i + length,
    line: ctx.line + newlines,
    lineStart: ctx.i + last + 1,
  }
}

const at = (ctx: LexCtx, tok: Emit): Tok => ({
  ...tok,
  pos: ctx.i,
  line: ctx.line,
  col: ctx.i - ctx.lineStart + 1,
})

const eof = (ctx: LexCtx): Tok => ({ kind: 'eof', pos: ctx.i, line: ctx.line, col: ctx.i - ctx.lineStart + 1 })

/** The unfold: `(src, ctx) → Maybe (Tok, nextCtx)`, repeated until EOF. */
function* lex(src: string): Generator<Tok> {
  let ctx: LexCtx = { i: 0, line: 1, lineStart: 0 }
  while (ctx.i < src.length) {
    const match = firstMatch(src, ctx)
    if (!match) errAt(`unexpected character: ${JSON.stringify(src[ctx.i])}`, ctx)
    if (match.tok) yield at(ctx, match.tok)
    ctx = advance(src, ctx, match.length)
  }
  yield eof(ctx)
}

export const tokenize = (src: string): Tok[] => [...lex(src)]

/** Pretty-format a token for diagnostic messages. */
export function formatTok(t: Tok): string {
  if (t.kind === 'num' || t.kind === 'ident' || t.kind === 'string') {
    return `${t.kind}(${JSON.stringify(t.value)})`
  }
  return t.kind
}
