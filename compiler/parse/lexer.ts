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
 * Position info: every token carries `pos` (byte offset into the source)
 * and `line`/`col` (1-indexed). Suitable for error reporting once mapped
 * back through the markdown extractor's line offsets.
 */

export type TokKind =
  // Literals + identifiers
  | 'num' | 'ident' | 'string'
  // Boolean keywords (lexed as their own kinds for parser convenience)
  | 'true' | 'false'
  // Binders + control flow
  | 'let' | 'in' | 'if' | 'else' | 'match'
  // Declarations
  | 'program' | 'reg' | 'delay' | 'param' | 'next' | 'out'
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
 *  and `string` (the unquoted contents). All other kinds carry their kind only. */
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
  next: 'next', out: 'out',
  struct: 'struct', enum: 'enum', type: 'type',
}

export class LexError extends Error {
  constructor(message: string, public pos: number, public line: number, public col: number) {
    super(`${line}:${col}: ${message}`)
  }
}

export function tokenize(src: string): Tok[] {
  const toks: Tok[] = []
  let i = 0
  let line = 1
  let lineStart = 0  // byte offset of the start of the current line

  const colAt = (offset: number) => offset - lineStart + 1
  const push = (kind: TokKind, pos: number, value?: string | number) =>
    toks.push({ kind, pos, line, col: colAt(pos), ...(value !== undefined ? { value } : {}) })

  while (i < src.length) {
    const c = src[i]

    // Whitespace
    if (c === ' ' || c === '\t' || c === '\r') { i++; continue }
    if (c === '\n') { i++; line++; lineStart = i; continue }

    // Line comment
    if (c === '/' && src[i + 1] === '/') {
      while (i < src.length && src[i] !== '\n') i++
      continue
    }

    // Block comment /* ... */ (non-nesting)
    if (c === '/' && src[i + 1] === '*') {
      i += 2
      while (i < src.length && !(src[i] === '*' && src[i + 1] === '/')) {
        if (src[i] === '\n') { line++; lineStart = i + 1 }
        i++
      }
      if (i >= src.length) throw new LexError('unterminated block comment', i, line, colAt(i))
      i += 2
      continue
    }

    const start = i

    // Number: optional leading dot, digits, optional fractional, optional exponent.
    if (/[0-9]/.test(c) || (c === '.' && /[0-9]/.test(src[i + 1] ?? ''))) {
      let j = i
      while (j < src.length && /[0-9]/.test(src[j])) j++
      if (src[j] === '.' && /[0-9]/.test(src[j + 1] ?? '')) {
        j++
        while (j < src.length && /[0-9]/.test(src[j])) j++
      }
      if (src[j] === 'e' || src[j] === 'E') {
        j++
        if (src[j] === '+' || src[j] === '-') j++
        if (!/[0-9]/.test(src[j] ?? '')) {
          throw new LexError(`malformed number (exponent missing digits): ${src.slice(start, j)}`, start, line, colAt(start))
        }
        while (j < src.length && /[0-9]/.test(src[j])) j++
      }
      const text = src.slice(i, j)
      const n = Number(text)
      if (!Number.isFinite(n)) throw new LexError(`invalid number: ${text}`, start, line, colAt(start))
      push('num', start, n)
      i = j
      continue
    }

    // Identifier or keyword
    if (/[A-Za-z_]/.test(c)) {
      let j = i
      while (j < src.length && /[A-Za-z0-9_]/.test(src[j])) j++
      const text = src.slice(i, j)
      const kw = KEYWORDS[text]
      if (kw) push(kw, start)
      else push('ident', start, text)
      i = j
      continue
    }

    // String literal (single or double quoted; same semantics)
    if (c === '"' || c === "'") {
      const quote = c
      let j = i + 1
      let buf = ''
      while (j < src.length && src[j] !== quote) {
        if (src[j] === '\n') throw new LexError('unterminated string literal', start, line, colAt(start))
        if (src[j] === '\\') {
          const next = src[j + 1]
          if (next === undefined) throw new LexError('unterminated escape', j, line, colAt(j))
          const esc: Record<string, string> = { n: '\n', t: '\t', r: '\r', '\\': '\\', "'": "'", '"': '"' }
          if (!(next in esc)) throw new LexError(`unknown escape: \\${next}`, j, line, colAt(j))
          buf += esc[next]
          j += 2
          continue
        }
        buf += src[j]
        j++
      }
      if (j >= src.length) throw new LexError('unterminated string literal', start, line, colAt(start))
      push('string', start, buf)
      i = j + 1
      continue
    }

    // Three-char punctuation (none currently — reserved for future "..." or "==>")
    // Two-char punctuation (longest-match wins)
    const two = src.slice(i, i + 2)
    if (two === '<=' || two === '>=' || two === '==' || two === '!=' ||
        two === '<<' || two === '>>' || two === '&&' || two === '||' ||
        two === '=>' || two === '->') {
      push(two as TokKind, start)
      i += 2
      continue
    }

    // Single-char punctuation
    if ('()[]{},.;:=+-*/%<>&|^~!'.includes(c)) {
      push(c as TokKind, start)
      i++
      continue
    }

    throw new LexError(`unexpected character: ${JSON.stringify(c)}`, start, line, colAt(start))
  }

  toks.push({ kind: 'eof', pos: i, line, col: colAt(i) })
  return toks
}

/** Pretty-format a token for diagnostic messages. */
export function formatTok(t: Tok): string {
  if (t.kind === 'num' || t.kind === 'ident' || t.kind === 'string') {
    return `${t.kind}(${JSON.stringify(t.value)})`
  }
  return t.kind
}
