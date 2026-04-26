/**
 * expressions.ts — surface-syntax expression parser (Phase B2, Stage 1).
 *
 * Consumes tokens produced by `lexer.ts`, produces `ExprNode` JSON identical
 * to what hand-written stdlib files contain. Bare identifiers that aren't
 * lexically bound by an enclosing combinator/let become a placeholder
 * `{op:'nameRef', name}` op for the elaborator (B6) to resolve to inputs,
 * registers, type-params, etc. based on the surrounding declaration's scope.
 *
 * Coverage (per design/surface-syntax.md Stage 1):
 *  - Literals: numbers, booleans, array literals `[a, b, c]`
 *  - Infix: arithmetic / comparison / logical / bitwise / shifts
 *  - Unary: -, !, ~
 *  - Function calls: `f(a, b)` — emits `call(nameRef('f'), [args])` for
 *    elaborator resolution. Combinators with lambdas are special-cased.
 *  - Dotted port refs: `inst.port` — emits `nestedOut(ref, output)`
 *  - Indexing: `a[i]` — emits `index(args)` builtin call
 *  - `let { x: e1, y: e2 } in body` — emits `let(bind, in)`; body parsed
 *    with x and y in the binder scope.
 *  - Combinator lambdas: `fold`, `scan`, `generate`, `iterate`, `chain`,
 *    `map2`, `zipWith` — each with the canonical IR shape, lambda binders
 *    pushed onto the scope while parsing the body.
 *  - Sample-rate / sample-index sentinels: `sample_rate()` and
 *    `sample_index()` parse as nullary calls and emit the matching ops.
 *
 * Out of scope (later sub-phases): match (B5), program/instance call sites
 * (handled by the elaborator since they require capitalization analysis +
 * type registry lookup).
 */

import { tokenize, type Tok, type TokKind } from './lexer.js'

// ─────────────────────────────────────────────────────────────
// ExprNode local type — kept loose so we don't import from compiler/expr.ts
// ─────────────────────────────────────────────────────────────

export type ExprNode =
  | number
  | boolean
  | ExprNode[]
  | { op: string; [k: string]: unknown }

// ─────────────────────────────────────────────────────────────
// Parse error
// ─────────────────────────────────────────────────────────────

export class ParseError extends Error {
  constructor(message: string, public tok: Tok) {
    super(`${tok.line}:${tok.col}: ${message}`)
  }
}

// ─────────────────────────────────────────────────────────────
// Parser context
// ─────────────────────────────────────────────────────────────

interface Ctx {
  toks: Tok[]
  i: number
  /** Names that are lexically bound by an enclosing let or combinator
   *  binder. A bare identifier matching any frame in this stack emits
   *  `binding(name)`; everything else emits `nameRef(name)`. */
  binders: Set<string>
}

function peek(ctx: Ctx, offset = 0): Tok {
  return ctx.toks[Math.min(ctx.i + offset, ctx.toks.length - 1)]
}

function consume(ctx: Ctx, kind: TokKind, what?: string): Tok {
  const t = ctx.toks[ctx.i]
  if (t.kind !== kind) {
    throw new ParseError(`expected ${what ?? kind}, got ${formatTokForError(t)}`, t)
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

function formatTokForError(t: Tok): string {
  if (t.kind === 'eof') return 'end of input'
  if (t.value !== undefined) return `${t.kind}(${JSON.stringify(t.value)})`
  return `'${t.kind}'`
}

// ─────────────────────────────────────────────────────────────
// Public entry points
// ─────────────────────────────────────────────────────────────

/** Parse a complete expression from source text. Throws ParseError if the
 *  input is malformed or has trailing tokens past the expression. */
export function parseExpr(src: string): ExprNode {
  const toks = tokenize(src)
  const ctx: Ctx = { toks, i: 0, binders: new Set() }
  const node = parseTopExpr(ctx)
  const trailing = ctx.toks[ctx.i]
  if (trailing.kind !== 'eof') {
    throw new ParseError(`unexpected trailing input: ${formatTokForError(trailing)}`, trailing)
  }
  return node
}

/** Parse one expression from a token stream starting at `ctx.i`. Used by
 *  upper-layer parsers (statements, declarations) that share a token stream. */
export function parseExprFromTokens(toks: Tok[], startIdx: number, binders?: Set<string>): { node: ExprNode; nextIdx: number } {
  const ctx: Ctx = { toks, i: startIdx, binders: binders ? new Set(binders) : new Set() }
  const node = parseTopExpr(ctx)
  return { node, nextIdx: ctx.i }
}

// ─────────────────────────────────────────────────────────────
// Precedence climbing — top-level expression
// ─────────────────────────────────────────────────────────────

function parseTopExpr(ctx: Ctx): ExprNode {
  return parseLogicalOr(ctx)
}

function binary(op: string, lhs: ExprNode, rhs: ExprNode): ExprNode {
  return { op, args: [lhs, rhs] }
}

function unary(op: string, operand: ExprNode): ExprNode {
  return { op, args: [operand] }
}

function parseLogicalOr(ctx: Ctx): ExprNode {
  let lhs = parseLogicalAnd(ctx)
  while (peek(ctx).kind === '||') {
    ctx.i++
    const rhs = parseLogicalAnd(ctx)
    lhs = binary('or', lhs, rhs)
  }
  return lhs
}

function parseLogicalAnd(ctx: Ctx): ExprNode {
  let lhs = parseBitwiseOr(ctx)
  while (peek(ctx).kind === '&&') {
    ctx.i++
    const rhs = parseBitwiseOr(ctx)
    lhs = binary('and', lhs, rhs)
  }
  return lhs
}

function parseBitwiseOr(ctx: Ctx): ExprNode {
  let lhs = parseBitwiseXor(ctx)
  while (peek(ctx).kind === '|') {
    ctx.i++
    const rhs = parseBitwiseXor(ctx)
    lhs = binary('bitOr', lhs, rhs)
  }
  return lhs
}

function parseBitwiseXor(ctx: Ctx): ExprNode {
  let lhs = parseBitwiseAnd(ctx)
  while (peek(ctx).kind === '^') {
    ctx.i++
    const rhs = parseBitwiseAnd(ctx)
    lhs = binary('bitXor', lhs, rhs)
  }
  return lhs
}

function parseBitwiseAnd(ctx: Ctx): ExprNode {
  let lhs = parseEquality(ctx)
  while (peek(ctx).kind === '&') {
    ctx.i++
    const rhs = parseEquality(ctx)
    lhs = binary('bitAnd', lhs, rhs)
  }
  return lhs
}

const EQUALITY_OPS: Partial<Record<TokKind, string>> = { '==': 'eq', '!=': 'neq' }
function parseEquality(ctx: Ctx): ExprNode {
  let lhs = parseRelational(ctx)
  for (;;) {
    const op = EQUALITY_OPS[peek(ctx).kind]
    if (!op) return lhs
    ctx.i++
    const rhs = parseRelational(ctx)
    lhs = binary(op, lhs, rhs)
  }
}

const RELATIONAL_OPS: Partial<Record<TokKind, string>> = { '<': 'lt', '<=': 'lte', '>': 'gt', '>=': 'gte' }
function parseRelational(ctx: Ctx): ExprNode {
  let lhs = parseShift(ctx)
  for (;;) {
    const op = RELATIONAL_OPS[peek(ctx).kind]
    if (!op) return lhs
    ctx.i++
    const rhs = parseShift(ctx)
    lhs = binary(op, lhs, rhs)
  }
}

const SHIFT_OPS: Partial<Record<TokKind, string>> = { '<<': 'lshift', '>>': 'rshift' }
function parseShift(ctx: Ctx): ExprNode {
  let lhs = parseAdditive(ctx)
  for (;;) {
    const op = SHIFT_OPS[peek(ctx).kind]
    if (!op) return lhs
    ctx.i++
    const rhs = parseAdditive(ctx)
    lhs = binary(op, lhs, rhs)
  }
}

const ADDITIVE_OPS: Partial<Record<TokKind, string>> = { '+': 'add', '-': 'sub' }
function parseAdditive(ctx: Ctx): ExprNode {
  let lhs = parseMultiplicative(ctx)
  for (;;) {
    const op = ADDITIVE_OPS[peek(ctx).kind]
    if (!op) return lhs
    ctx.i++
    const rhs = parseMultiplicative(ctx)
    lhs = binary(op, lhs, rhs)
  }
}

const MULTIPLICATIVE_OPS: Partial<Record<TokKind, string>> = { '*': 'mul', '/': 'div', '%': 'mod' }
function parseMultiplicative(ctx: Ctx): ExprNode {
  let lhs = parseUnary(ctx)
  for (;;) {
    const op = MULTIPLICATIVE_OPS[peek(ctx).kind]
    if (!op) return lhs
    ctx.i++
    const rhs = parseUnary(ctx)
    lhs = binary(op, lhs, rhs)
  }
}

const UNARY_OPS: Partial<Record<TokKind, string>> = { '-': 'neg', '!': 'not', '~': 'bitNot' }
function parseUnary(ctx: Ctx): ExprNode {
  const op = UNARY_OPS[peek(ctx).kind]
  if (op) {
    ctx.i++
    const operand = parseUnary(ctx)
    // Constant-fold `-<number-literal>` into a negative number. Matches the
    // canonical JSON form (`-0.5`, not `{op:'neg', args:[0.5]}`) and makes
    // array literals like `[1, -0.5, 0.25]` agree with stdlib JSON.
    if (op === 'neg' && typeof operand === 'number') return -operand
    return unary(op, operand)
  }
  return parsePostfix(ctx)
}

// ─────────────────────────────────────────────────────────────
// Postfix: dot-access, index, call
// ─────────────────────────────────────────────────────────────

function parsePostfix(ctx: Ctx): ExprNode {
  let node = parsePrimary(ctx)
  for (;;) {
    const t = peek(ctx)
    if (t.kind === '.') {
      ctx.i++
      const field = consume(ctx, 'ident', 'field name after `.`')
      // For node = nameRef(name), this is `instance.port`. The elaborator
      // converts to `nestedOut(ref: instance, output: port)`. For other
      // node shapes (e.g., array.length — not supported), emit the same
      // nestedOut form and let the elaborator decide.
      if (isNameRef(node)) {
        node = { op: 'nestedOut', ref: (node as { name: string }).name, output: field.value as string }
      } else {
        node = { op: 'fieldAccess', expr: node, field: field.value as string }
      }
    } else if (t.kind === '[') {
      ctx.i++
      const idx = parseTopExpr(ctx)
      consume(ctx, ']', 'closing `]`')
      node = { op: 'index', args: [node, idx] }
    } else if (t.kind === '(') {
      ctx.i++
      // Function-call form. If the callee is a known combinator name, emit
      // its structured op directly; otherwise emit a generic call for the
      // elaborator to resolve into a builtin op or user function call.
      if (isNameRef(node)) {
        const name = (node as { name: string }).name
        const combinator = parseCombinatorCall(ctx, name)
        if (combinator !== null) {
          node = combinator
        } else {
          const args = parseCallArgs(ctx)
          consume(ctx, ')', 'closing `)`')
          node = { op: 'call', callee: node, args }
        }
      } else {
        const args = parseCallArgs(ctx)
        consume(ctx, ')', 'closing `)`')
        node = { op: 'call', callee: node, args }
      }
    } else {
      return node
    }
  }
}

function parseCallArgs(ctx: Ctx): ExprNode[] {
  const args: ExprNode[] = []
  if (peek(ctx).kind === ')') return args
  args.push(parseTopExpr(ctx))
  while (eat(ctx, ',')) {
    args.push(parseTopExpr(ctx))
  }
  return args
}

function isNameRef(node: ExprNode): boolean {
  return typeof node === 'object' && node !== null && !Array.isArray(node)
    && (node as { op?: string }).op === 'nameRef'
}

// ─────────────────────────────────────────────────────────────
// Combinators: parse `(binder, ...) => body` style invocations
// ─────────────────────────────────────────────────────────────

/** Try to parse a known combinator call. Caller has consumed the callee
 *  identifier and the opening `(`. On match, consumes through the closing
 *  `)` and returns the structured node. On no-match (unknown name or shape),
 *  rewinds zero tokens and returns null — the caller falls back to generic
 *  call parsing. */
function parseCombinatorCall(ctx: Ctx, name: string): ExprNode | null {
  switch (name) {
    case 'fold':    return parseFoldOrScan(ctx, 'fold')
    case 'scan':    return parseFoldOrScan(ctx, 'scan')
    case 'generate': return parseGenerate(ctx)
    case 'iterate':  return parseIterateOrChain(ctx, 'iterate')
    case 'chain':    return parseIterateOrChain(ctx, 'chain')
    case 'map2':     return parseMap2(ctx)
    case 'zipWith':  return parseZipWith(ctx)
    default: return null
  }
}

/** fold(over, init, (acc, elem) => body) */
function parseFoldOrScan(ctx: Ctx, op: 'fold' | 'scan'): ExprNode {
  const over = parseTopExpr(ctx)
  consume(ctx, ',', `${op}: comma after over`)
  const init = parseTopExpr(ctx)
  consume(ctx, ',', `${op}: comma after init`)
  const lambda = parseLambdaArgs(ctx, 2, op)
  const [accVar, elemVar] = lambda.binders
  const body = parseLambdaBody(ctx, lambda.binders)
  consume(ctx, ')', `${op}: closing \`)\``)
  return { op, over, init, acc_var: accVar, elem_var: elemVar, body }
}

/** generate(count, (i) => body) */
function parseGenerate(ctx: Ctx): ExprNode {
  const count = parseTopExpr(ctx)
  consume(ctx, ',', 'generate: comma after count')
  const lambda = parseLambdaArgs(ctx, 1, 'generate')
  const [varName] = lambda.binders
  const body = parseLambdaBody(ctx, lambda.binders)
  consume(ctx, ')', 'generate: closing `)`')
  return { op: 'generate', count, var: varName, body }
}

/** iterate(count, init, (x) => body) — and chain with the same shape. */
function parseIterateOrChain(ctx: Ctx, op: 'iterate' | 'chain'): ExprNode {
  const count = parseTopExpr(ctx)
  consume(ctx, ',', `${op}: comma after count`)
  const init = parseTopExpr(ctx)
  consume(ctx, ',', `${op}: comma after init`)
  const lambda = parseLambdaArgs(ctx, 1, op)
  const [varName] = lambda.binders
  const body = parseLambdaBody(ctx, lambda.binders)
  consume(ctx, ')', `${op}: closing \`)\``)
  return { op, count, var: varName, init, body }
}

/** map2(over, (e) => body) */
function parseMap2(ctx: Ctx): ExprNode {
  const over = parseTopExpr(ctx)
  consume(ctx, ',', 'map2: comma after over')
  const lambda = parseLambdaArgs(ctx, 1, 'map2')
  const [elemVar] = lambda.binders
  const body = parseLambdaBody(ctx, lambda.binders)
  consume(ctx, ')', 'map2: closing `)`')
  return { op: 'map2', over, elem_var: elemVar, body }
}

/** zipWith(a, b, (x, y) => body) */
function parseZipWith(ctx: Ctx): ExprNode {
  const a = parseTopExpr(ctx)
  consume(ctx, ',', 'zipWith: comma after a')
  const b = parseTopExpr(ctx)
  consume(ctx, ',', 'zipWith: comma after b')
  const lambda = parseLambdaArgs(ctx, 2, 'zipWith')
  const [xVar, yVar] = lambda.binders
  const body = parseLambdaBody(ctx, lambda.binders)
  consume(ctx, ')', 'zipWith: closing `)`')
  return { op: 'zipWith', a, b, x_var: xVar, y_var: yVar, body }
}

/** Parse `(binder1, binder2, ...) =>` and return the binder names.
 *  Validates the expected arity (0+ allows any). */
function parseLambdaArgs(ctx: Ctx, expectedArity: number, ownerOp: string): { binders: string[] } {
  const open = consume(ctx, '(', `${ownerOp}: opening \`(\` for lambda`)
  const binders: string[] = []
  if (peek(ctx).kind !== ')') {
    binders.push(consume(ctx, 'ident', `${ownerOp}: binder name`).value as string)
    while (eat(ctx, ',')) {
      binders.push(consume(ctx, 'ident', `${ownerOp}: binder name`).value as string)
    }
  }
  consume(ctx, ')', `${ownerOp}: closing \`)\` of lambda args`)
  if (binders.length !== expectedArity) {
    throw new ParseError(
      `${ownerOp}: lambda expects ${expectedArity} binder(s), got ${binders.length}`,
      open,
    )
  }
  consume(ctx, '=>', `${ownerOp}: \`=>\` after lambda binders`)
  return { binders }
}

/** Parse a lambda body with the given binders pushed onto the parser's
 *  scope. Restores scope on return. */
function parseLambdaBody(ctx: Ctx, binders: string[]): ExprNode {
  const added: string[] = []
  for (const b of binders) {
    if (!ctx.binders.has(b)) {
      ctx.binders.add(b)
      added.push(b)
    }
  }
  try {
    return parseTopExpr(ctx)
  } finally {
    for (const b of added) ctx.binders.delete(b)
  }
}

// ─────────────────────────────────────────────────────────────
// Primary expressions
// ─────────────────────────────────────────────────────────────

function parsePrimary(ctx: Ctx): ExprNode {
  const t = peek(ctx)

  if (t.kind === 'num') {
    ctx.i++
    return t.value as number
  }

  if (t.kind === 'true')  { ctx.i++; return true }
  if (t.kind === 'false') { ctx.i++; return false }

  if (t.kind === '(') {
    ctx.i++
    const inner = parseTopExpr(ctx)
    consume(ctx, ')', 'closing `)`')
    return inner
  }

  if (t.kind === '[') {
    ctx.i++
    return parseArrayLiteral(ctx)
  }

  if (t.kind === 'let') {
    ctx.i++
    return parseLet(ctx)
  }

  if (t.kind === 'ident') {
    ctx.i++
    const name = t.value as string
    if (ctx.binders.has(name)) {
      return { op: 'binding', name }
    }
    return { op: 'nameRef', name }
  }

  // Sentinels are reserved tokens but appear in postfix-call form. The
  // parser currently lexes 'sample_rate' as an ident, so it's covered by
  // the ident branch above. Same for sample_index.

  throw new ParseError(`unexpected token in expression: ${formatTokForError(t)}`, t)
}

function parseArrayLiteral(ctx: Ctx): ExprNode {
  // The opening `[` has already been consumed.
  const items: ExprNode[] = []
  if (peek(ctx).kind !== ']') {
    items.push(parseTopExpr(ctx))
    while (eat(ctx, ',')) {
      // Tolerate trailing comma: `[1, 2, 3,]`
      if (peek(ctx).kind === ']') break
      items.push(parseTopExpr(ctx))
    }
  }
  consume(ctx, ']', 'closing `]` of array literal')
  return items
}

/** Parse `let { x: e1; y: e2 } in body` (the surface-syntax doc's form,
 *  using `:` to bind and `;` or `,` as separator) and emit
 *  `{op:'let', bind: {x: e1, y: e2}, in: body}`. The let-bindings are
 *  visible inside `body` as `{op:'binding', name}` placeholders. */
function parseLet(ctx: Ctx): ExprNode {
  consume(ctx, '{', 'let: opening `{`')
  const bind: Record<string, ExprNode> = {}
  const order: string[] = []
  while (peek(ctx).kind !== '}') {
    const nameTok = consume(ctx, 'ident', 'let binding name')
    const name = nameTok.value as string
    if (name in bind) {
      throw new ParseError(`let: duplicate binding name '${name}'`, nameTok)
    }
    consume(ctx, ':', 'let binding `:`')
    bind[name] = parseTopExpr(ctx)
    order.push(name)
    // Separators: `;` or `,` between bindings, optional before `}`
    if (eat(ctx, ';') || eat(ctx, ',')) {
      continue
    }
    break
  }
  consume(ctx, '}', 'let: closing `}`')
  consume(ctx, 'in', 'let: `in`')
  const added: string[] = []
  for (const n of order) {
    if (!ctx.binders.has(n)) {
      ctx.binders.add(n)
      added.push(n)
    }
  }
  try {
    const body = parseTopExpr(ctx)
    return { op: 'let', bind, in: body }
  } finally {
    for (const b of added) ctx.binders.delete(b)
  }
}
