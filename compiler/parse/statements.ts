/**
 * statements.ts — body-level statement parser (Phase B3, Stage 2).
 *
 * Consumes tokens from the lexer and produces a `BlockNode`-shaped value
 * with `decls` and `assigns` arrays. Body items are dispatched by their
 * leading token:
 *
 *   reg      → regDecl     (regDecl(name, init, type?))
 *   delay    → delayDecl   (delayDecl(name, update, init))
 *   param    → paramDecl   (paramDecl(name, type='param'|'trigger', value?))
 *   next     → nextUpdate  (target.kind='reg' always — delays carry their
 *                           update inside delayDecl, not via next)
 *   dac      → outputAssign(name='dac.out', expr) — boundary leaf wire
 *   Identifier "=" Capitalized(...)  → instanceDecl
 *   Identifier "=" expr              → outputAssign(name, expr)
 *
 * Statements are separated by optional `;` (or just whitespace — the leading
 * tokens of each statement are unambiguous).
 *
 * Out of scope (later sub-phases):
 *  - Nested `program` decls (B4)
 *  - Full port-type grammar with bounds, arrays, generics (B4)
 *  - Type args on instance decls beyond simple `<N=4>` form (B4)
 *  - ADTs / match (B5)
 */

import { tokenize, type Tok, type TokKind } from './lexer.js'
import { parseExprFromTokens, type ExprNode, ParseError } from './expressions.js'

// ─────────────────────────────────────────────────────────────
// BlockNode shape — kept loose to avoid a hard dependency on compiler/expr.ts
// ─────────────────────────────────────────────────────────────

export interface BlockNode {
  op: 'block'
  decls: ExprNode[]
  assigns: ExprNode[]
}

// ─────────────────────────────────────────────────────────────
// Parser context
// ─────────────────────────────────────────────────────────────

interface Ctx {
  toks: Tok[]
  i: number
}

function peek(ctx: Ctx, offset = 0): Tok {
  return ctx.toks[Math.min(ctx.i + offset, ctx.toks.length - 1)]
}

function consume(ctx: Ctx, kind: TokKind, what?: string): Tok {
  const t = ctx.toks[ctx.i]
  if (t.kind !== kind) {
    throw new ParseError(`expected ${what ?? kind}, got ${formatTok(t)}`, t)
  }
  ctx.i++
  return t
}

function eat(ctx: Ctx, kind: TokKind): Tok | null {
  const t = ctx.toks[ctx.i]
  if (t.kind !== kind) return null
  ctx.i++
  return t
}

function formatTok(t: Tok): string {
  if (t.kind === 'eof') return 'end of input'
  if (t.value !== undefined) return `${t.kind}(${JSON.stringify(t.value)})`
  return `'${t.kind}'`
}

function isCapitalized(name: string): boolean {
  return /^[A-Z]/.test(name)
}

// ─────────────────────────────────────────────────────────────
// Public entry: parse a body block
// ─────────────────────────────────────────────────────────────

/** Parse a brace-delimited body block from source text, e.g.
 *  `{ reg s = 0; out = s; next s = s + 1 }`. Returns a BlockNode. */
export function parseBody(src: string): BlockNode {
  const toks = tokenize(src)
  const ctx: Ctx = { toks, i: 0 }
  consume(ctx, '{', 'opening `{` of body')
  const block = parseBodyItems(ctx)
  consume(ctx, '}', 'closing `}` of body')
  const trailing = ctx.toks[ctx.i]
  if (trailing.kind !== 'eof') {
    throw new ParseError(`unexpected trailing input: ${formatTok(trailing)}`, trailing)
  }
  return block
}

/** Parse the contents of a body block from a token stream, starting at the
 *  position immediately after the opening `{`. Stops at the matching `}`
 *  (left for the caller to consume). Used by upper-layer parsers (B4
 *  declarations) that share a token stream. */
export function parseBodyFromTokens(toks: Tok[], startIdx: number): { block: BlockNode; nextIdx: number } {
  const ctx: Ctx = { toks, i: startIdx }
  const block = parseBodyItems(ctx)
  return { block, nextIdx: ctx.i }
}

function parseBodyItems(ctx: Ctx): BlockNode {
  const decls: ExprNode[] = []
  const assigns: ExprNode[] = []
  while (peek(ctx).kind !== '}' && peek(ctx).kind !== 'eof') {
    const item = parseBodyItem(ctx)
    if (item.kind === 'decl') decls.push(item.node)
    else assigns.push(item.node)
    // Optional `;` separator
    eat(ctx, ';')
  }
  return { op: 'block', decls, assigns }
}

// ─────────────────────────────────────────────────────────────
// Body-item dispatch
// ─────────────────────────────────────────────────────────────

type BodyItem =
  | { kind: 'decl';   node: ExprNode }
  | { kind: 'assign'; node: ExprNode }

function parseBodyItem(ctx: Ctx): BodyItem {
  const t = peek(ctx)

  if (t.kind === 'reg')   return { kind: 'decl',   node: parseRegDecl(ctx) }
  if (t.kind === 'delay') return { kind: 'decl',   node: parseDelayDecl(ctx) }
  if (t.kind === 'param') return { kind: 'decl',   node: parseParamDecl(ctx) }
  if (t.kind === 'next')  return { kind: 'assign', node: parseNextUpdate(ctx) }

  if (t.kind === 'ident') {
    const name = t.value as string
    if (name === 'dac') return { kind: 'assign', node: parseDacOutAssign(ctx) }
    // Lookahead: `name = ...` is either instanceDecl or outputAssign
    return parseAssignOrInstance(ctx)
  }

  throw new ParseError(`expected body item, got ${formatTok(t)}`, t)
}

// ─────────────────────────────────────────────────────────────
// Decls
// ─────────────────────────────────────────────────────────────

/** `reg name [: type] = init` */
function parseRegDecl(ctx: Ctx): ExprNode {
  consume(ctx, 'reg', 'reg keyword')
  const name = consume(ctx, 'ident', 'reg name').value as string
  let type: string | undefined
  if (eat(ctx, ':')) {
    type = consume(ctx, 'ident', 'reg type name').value as string
  }
  consume(ctx, '=', 'reg `=` before init')
  const init = parseExpr(ctx)
  const out: Record<string, unknown> = { op: 'regDecl', name, init }
  if (type !== undefined) out.type = type
  return out as ExprNode
}

/** `delay name = update_expr init init_value` —
 *  `init` is a contextual keyword (parsed as an ident token). */
function parseDelayDecl(ctx: Ctx): ExprNode {
  consume(ctx, 'delay', 'delay keyword')
  const name = consume(ctx, 'ident', 'delay name').value as string
  consume(ctx, '=', 'delay `=` before update expression')
  const update = parseExpr(ctx)
  // Contextual `init` terminator. The expression parser leaves `init` on
  // the token stream because identifiers don't extend an expression past
  // a complete operand.
  const initTok = peek(ctx)
  if (initTok.kind !== 'ident' || initTok.value !== 'init') {
    throw new ParseError(`delay decl: expected 'init' after update expression, got ${formatTok(initTok)}`, initTok)
  }
  ctx.i++
  const init = parseExpr(ctx)
  return { op: 'delayDecl', name, update, init } as unknown as ExprNode
}

/** `param name: smoothed|trigger [= default]`.
 *  Surface kind `smoothed` maps to IR `type: 'param'`; `trigger` is identity.
 *  ('smoothed' is the surface word because 'param' is reserved as the
 *  declaration keyword.) */
function parseParamDecl(ctx: Ctx): ExprNode {
  consume(ctx, 'param', 'param keyword')
  const name = consume(ctx, 'ident', 'param name').value as string
  consume(ctx, ':', 'param `:` before kind')
  const kindTok = consume(ctx, 'ident', 'param kind (smoothed|trigger)')
  const kindRaw = kindTok.value as string
  let irKind: 'param' | 'trigger'
  if (kindRaw === 'smoothed') irKind = 'param'
  else if (kindRaw === 'trigger') irKind = 'trigger'
  else throw new ParseError(`param kind must be 'smoothed' or 'trigger', got '${kindRaw}'`, kindTok)
  const out: Record<string, unknown> = { op: 'paramDecl', name, type: irKind }
  if (eat(ctx, '=')) {
    if (irKind === 'trigger') {
      throw new ParseError(`trigger params cannot have a default value`, peek(ctx))
    }
    const valueExpr = parseExpr(ctx)
    if (typeof valueExpr !== 'number') {
      throw new ParseError(`param default must be a number literal`, peek(ctx))
    }
    out.value = valueExpr
  }
  return out as ExprNode
}

// ─────────────────────────────────────────────────────────────
// Assigns
// ─────────────────────────────────────────────────────────────

/** `next name = expr` — register update.
 *  Delays carry their update inside delayDecl, so target.kind is always
 *  'reg' here. */
function parseNextUpdate(ctx: Ctx): ExprNode {
  consume(ctx, 'next', 'next keyword')
  const name = consume(ctx, 'ident', 'next target name').value as string
  consume(ctx, '=', 'next `=` before expression')
  const expr = parseExpr(ctx)
  return {
    op: 'nextUpdate',
    target: { kind: 'reg', name },
    expr,
  } as unknown as ExprNode
}

/** `dac.out = expr` — boundary-leaf wire (per A4). */
function parseDacOutAssign(ctx: Ctx): ExprNode {
  // The leading 'dac' ident is at peek; consume and verify the dotted form.
  const dacTok = consume(ctx, 'ident', 'dac')
  if (dacTok.value !== 'dac') {
    throw new ParseError(`expected 'dac' for boundary-leaf wire, got ${formatTok(dacTok)}`, dacTok)
  }
  consume(ctx, '.', 'dac `.`')
  const portTok = consume(ctx, 'ident', 'dac port name')
  if (portTok.value !== 'out') {
    throw new ParseError(`dac has only one output port: 'out'. Got '${portTok.value}'`, portTok)
  }
  consume(ctx, '=', 'dac.out `=` before expression')
  const expr = parseExpr(ctx)
  return { op: 'outputAssign', name: 'dac.out', expr } as unknown as ExprNode
}

/** `name = ...` — either an instanceDecl (RHS is `Capitalized(...)`) or an
 *  outputAssign (RHS is anything else). */
function parseAssignOrInstance(ctx: Ctx): BodyItem {
  const nameTok = consume(ctx, 'ident', 'statement target name')
  const name = nameTok.value as string
  consume(ctx, '=', `\`=\` after '${name}'`)

  // Lookahead: is the RHS a capitalized identifier followed by `(` or `<`?
  // If so, parse as instanceDecl. Otherwise parse as outputAssign expr.
  const t = peek(ctx)
  const t2 = peek(ctx, 1)
  if (t.kind === 'ident' && typeof t.value === 'string' && isCapitalized(t.value)
      && (t2.kind === '(' || t2.kind === '<')) {
    return { kind: 'decl', node: parseInstanceRhs(ctx, name) }
  }

  const expr = parseExpr(ctx)
  return { kind: 'assign', node: { op: 'outputAssign', name, expr } as unknown as ExprNode }
}

/** Parse `ProgType[<typeArgs>](port: expr, ...)` — the RHS of an instance
 *  declaration. `name` is the instance binding's name. */
function parseInstanceRhs(ctx: Ctx, name: string): ExprNode {
  const typeTok = consume(ctx, 'ident', 'program type name')
  const programName = typeTok.value as string

  let typeArgs: Record<string, number> | undefined
  if (eat(ctx, '<')) {
    typeArgs = parseTypeArgs(ctx, programName)
  }

  consume(ctx, '(', `\`(\` after program type '${programName}'`)
  const inputs = parseInstanceInputs(ctx)
  consume(ctx, ')', `closing \`)\` of '${programName}' inputs`)

  const out: Record<string, unknown> = { op: 'instanceDecl', name, program: programName }
  if (typeArgs !== undefined && Object.keys(typeArgs).length > 0) out.type_args = typeArgs
  if (Object.keys(inputs).length > 0) out.inputs = inputs
  return out as ExprNode
}

/** Parse `<key=value, key=value>`. The opening `<` is already consumed. */
function parseTypeArgs(ctx: Ctx, owner: string): Record<string, number> {
  const args: Record<string, number> = {}
  if (peek(ctx).kind === '>') {
    ctx.i++
    return args
  }
  for (;;) {
    const k = consume(ctx, 'ident', `${owner}: type-arg name`).value as string
    consume(ctx, '=', `${owner}: \`=\` after type-arg name '${k}'`)
    const vTok = consume(ctx, 'num', `${owner}: type-arg value (number literal)`)
    if (!Number.isInteger(vTok.value)) {
      throw new ParseError(`${owner}: type-arg '${k}' must be an integer`, vTok)
    }
    if (k in args) {
      throw new ParseError(`${owner}: duplicate type-arg '${k}'`, vTok)
    }
    args[k] = vTok.value as number
    if (eat(ctx, '>')) return args
    consume(ctx, ',', `${owner}: \`,\` between type-args`)
  }
}

/** Parse `(port: expr, port: expr)` — keyword arg form. The `(` is already
 *  consumed; this stops at the matching `)` (left for the caller). */
function parseInstanceInputs(ctx: Ctx): Record<string, ExprNode> {
  const inputs: Record<string, ExprNode> = {}
  if (peek(ctx).kind === ')') return inputs
  for (;;) {
    const portTok = consume(ctx, 'ident', 'instance input port name')
    const port = portTok.value as string
    if (port in inputs) {
      throw new ParseError(`duplicate instance input '${port}'`, portTok)
    }
    consume(ctx, ':', `\`:\` after input port '${port}'`)
    inputs[port] = parseExpr(ctx)
    if (peek(ctx).kind === ')') return inputs
    consume(ctx, ',', `\`,\` between instance inputs`)
  }
}

// ─────────────────────────────────────────────────────────────
// Expression delegation
// ─────────────────────────────────────────────────────────────

/** Parse one expression at the current position, advancing the context. */
function parseExpr(ctx: Ctx): ExprNode {
  const { node, nextIdx } = parseExprFromTokens(ctx.toks, ctx.i)
  ctx.i = nextIdx
  return node
}
