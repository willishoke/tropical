/**
 * declarations.ts — program-declaration grammar (Phase B4, Stage 3).
 *
 * Parses a top-level program declaration:
 *
 *   program Name<TypeParams>(InputPorts) -> (OutputPorts) { body }
 *
 * Produces a `ProgramNode`-shaped value matching the existing
 * `tropical_program_2` schema:
 *
 *   {
 *     op: 'program',
 *     name: string,
 *     type_params?: { N: { type: 'int', default?: number }, ... },
 *     ports?: {
 *       inputs?:  Array<string | { name, type?, default?, bounds? }>,
 *       outputs?: Array<string | { name, type?, bounds? }>,
 *     },
 *     body: BlockNode,
 *   }
 *
 * Nested program decls inside a body are wrapped as
 * `{ op: 'programDecl', name: <inner program name>, program: ProgramNode }`.
 *
 * Surface coverage:
 *  - Type params: `<N: int = 8, M: int>` (optional default)
 *  - Port type: bare scalar identifiers (e.g. `signal`, `float`, `freq`),
 *    array form `Element[Shape...]` where each shape dim is a number
 *    literal or an identifier (resolved as `typeParam` at parse time
 *    since the parser tracks declared type-param names from the header)
 *  - Bounds: `in [lo, hi]` after a port type
 *  - Input ports: `name: type [= default] [in [lo, hi]]`; outputs omit
 *    the default
 *  - Bare-name ports (just `name`) emit a string entry
 *  - Nested `program` decls in body, recursive
 *
 * Out of scope (deferred to B5): ADTs and `match` (`type_defs` field).
 */

import { tokenize, type Tok, type TokKind } from './lexer.js'
import { parseExprFromTokens, type ExprNode, ParseError } from './expressions.js'
import { parseBodyFromTokens, type BlockNode, type BodyOptions } from './statements.js'

// ─────────────────────────────────────────────────────────────
// ProgramNode shape — kept loose to avoid cycles with compiler/program.ts
// ─────────────────────────────────────────────────────────────

export type ShapeDim = number | { op: 'typeParam'; name: string }

export type PortTypeDecl = string | { kind: 'array'; element: string; shape: ShapeDim[] }

export interface ProgramPortSpec {
  name: string
  type?: PortTypeDecl
  default?: ExprNode
  bounds?: [number | null, number | null]
}

export type ProgramPort = string | ProgramPortSpec

export interface ProgramPorts {
  inputs?: ProgramPort[]
  outputs?: ProgramPort[]
}

export interface ProgramNode {
  op: 'program'
  name: string
  type_params?: Record<string, { type: 'int'; default?: number }>
  ports?: ProgramPorts
  body: BlockNode
}

// ─────────────────────────────────────────────────────────────
// Parser context
// ─────────────────────────────────────────────────────────────

interface Ctx {
  toks: Tok[]
  i: number
  /** Type-param names in scope at the current point. Populated when a
   *  program header declares `<N: int, ...>`. Used by the port-type
   *  parser to recognize array shapes like `float[N]` as
   *  `{op:'typeParam',name:'N'}` rather than a bare name. */
  typeParams: Set<string>
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

// ─────────────────────────────────────────────────────────────
// Public entry points
// ─────────────────────────────────────────────────────────────

/** Parse a top-level program declaration from source text. */
export function parseProgram(src: string): ProgramNode {
  const toks = tokenize(src)
  const ctx: Ctx = { toks, i: 0, typeParams: new Set() }
  const node = parseProgramFromCtx(ctx)
  const trailing = ctx.toks[ctx.i]
  if (trailing.kind !== 'eof') {
    throw new ParseError(`unexpected trailing input after program: ${formatTok(trailing)}`, trailing)
  }
  return node
}

/** Parse a program declaration starting at the given token index. Used by
 *  the body parser via the BodyOptions.programDeclParser callback for
 *  nested program decls. */
export function parseProgramFromTokens(
  toks: Tok[], startIdx: number,
): { node: ProgramNode; nextIdx: number } {
  const ctx: Ctx = { toks, i: startIdx, typeParams: new Set() }
  const node = parseProgramFromCtx(ctx)
  return { node, nextIdx: ctx.i }
}

/** Body-parser hook: parse a nested `program` decl and wrap it as a
 *  `programDecl` body item. */
function parseNestedProgramDecl(
  toks: Tok[], startIdx: number,
): { node: ExprNode; nextIdx: number } {
  const { node: inner, nextIdx } = parseProgramFromTokens(toks, startIdx)
  return {
    node: { op: 'programDecl', name: inner.name, program: inner } as unknown as ExprNode,
    nextIdx,
  }
}

const NESTED_PROGRAM_OPTS: BodyOptions = { programDeclParser: parseNestedProgramDecl }

// ─────────────────────────────────────────────────────────────
// Program-declaration parser
// ─────────────────────────────────────────────────────────────

function parseProgramFromCtx(ctx: Ctx): ProgramNode {
  consume(ctx, 'program', 'program keyword')
  const nameTok = consume(ctx, 'ident', 'program name')
  const name = nameTok.value as string

  // Type params (optional)
  let typeParams: Record<string, { type: 'int'; default?: number }> | undefined
  if (peek(ctx).kind === '<') {
    typeParams = parseTypeParams(ctx)
    for (const tp of Object.keys(typeParams)) ctx.typeParams.add(tp)
  }

  // Input ports
  consume(ctx, '(', `\`(\` after program name '${name}'`)
  const inputs = parsePortList(ctx, /*allowDefault=*/ true)
  consume(ctx, ')', `closing \`)\` of inputs for '${name}'`)

  // Outputs (optional — defaults to [] if omitted)
  let outputs: ProgramPort[] | undefined
  if (eat(ctx, '->')) {
    consume(ctx, '(', `\`(\` after \`->\` for '${name}'`)
    outputs = parsePortList(ctx, /*allowDefault=*/ false)
    consume(ctx, ')', `closing \`)\` of outputs for '${name}'`)
  }

  // Body
  consume(ctx, '{', `\`{\` opening body of '${name}'`)
  const { block, nextIdx } = parseBodyFromTokens(ctx.toks, ctx.i, NESTED_PROGRAM_OPTS)
  ctx.i = nextIdx
  consume(ctx, '}', `\`}\` closing body of '${name}'`)

  // Pop type params from scope (each program declaration introduces its own)
  if (typeParams) {
    for (const tp of Object.keys(typeParams)) ctx.typeParams.delete(tp)
  }

  const node: ProgramNode = { op: 'program', name, body: block }
  if (typeParams && Object.keys(typeParams).length > 0) node.type_params = typeParams
  const ports: ProgramPorts = {}
  if (inputs.length > 0)  ports.inputs  = inputs
  if (outputs && outputs.length > 0) ports.outputs = outputs
  if (ports.inputs || ports.outputs) node.ports = ports
  return node
}

// ─────────────────────────────────────────────────────────────
// Type params: <N: int [= default], M: int>
// ─────────────────────────────────────────────────────────────

function parseTypeParams(ctx: Ctx): Record<string, { type: 'int'; default?: number }> {
  consume(ctx, '<', 'opening `<` of type params')
  const out: Record<string, { type: 'int'; default?: number }> = {}
  if (eat(ctx, '>')) return out
  for (;;) {
    const nameTok = consume(ctx, 'ident', 'type-param name')
    const name = nameTok.value as string
    if (name in out) {
      throw new ParseError(`duplicate type-param '${name}'`, nameTok)
    }
    consume(ctx, ':', `\`:\` after type-param '${name}'`)
    const typeTok = consume(ctx, 'ident', `type-param '${name}' type`)
    const typeName = typeTok.value as string
    if (typeName !== 'int') {
      throw new ParseError(`type-param '${name}' type must be 'int', got '${typeName}'`, typeTok)
    }
    const entry: { type: 'int'; default?: number } = { type: 'int' }
    if (eat(ctx, '=')) {
      const defTok = consume(ctx, 'num', `default for type-param '${name}'`)
      if (!Number.isInteger(defTok.value)) {
        throw new ParseError(`type-param '${name}' default must be an integer`, defTok)
      }
      entry.default = defTok.value as number
    }
    out[name] = entry
    if (eat(ctx, '>')) return out
    consume(ctx, ',', '`,` between type params')
  }
}

// ─────────────────────────────────────────────────────────────
// Port lists and port specs
// ─────────────────────────────────────────────────────────────

function parsePortList(ctx: Ctx, allowDefault: boolean): ProgramPort[] {
  const out: ProgramPort[] = []
  if (peek(ctx).kind === ')') return out
  for (;;) {
    out.push(parsePortSpec(ctx, allowDefault))
    if (peek(ctx).kind === ')') return out
    consume(ctx, ',', '`,` between port specs')
  }
}

/** Parse one port spec. Forms (input):
 *    name
 *    name: type
 *    name: type = default
 *    name: type in [lo, hi]
 *    name: type = default in [lo, hi]
 *  Outputs accept the same forms minus `= default`.
 *  When the spec has only a name (no type, default, or bounds), emits the
 *  bare-string form to match stdlib JSON convention. */
function parsePortSpec(ctx: Ctx, allowDefault: boolean): ProgramPort {
  const nameTok = consume(ctx, 'ident', 'port name')
  const name = nameTok.value as string

  if (peek(ctx).kind !== ':') {
    return name  // bare-string form
  }
  ctx.i++  // consume `:`

  const type = parsePortType(ctx)
  let defaultExpr: ExprNode | undefined
  let bounds: [number | null, number | null] | undefined

  // Optional `= default` (inputs only)
  if (peek(ctx).kind === '=') {
    if (!allowDefault) {
      throw new ParseError(`output ports cannot have a default value`, peek(ctx))
    }
    ctx.i++
    defaultExpr = parseExprAt(ctx)
  }

  // Optional `in [lo, hi]`
  if (peek(ctx).kind === 'in') {
    ctx.i++
    bounds = parseBounds(ctx)
  }

  const spec: ProgramPortSpec = { name, type }
  if (defaultExpr !== undefined) spec.default = defaultExpr
  if (bounds !== undefined) spec.bounds = bounds
  return spec
}

/** Parse a port type:
 *    Identifier                — bare scalar (e.g. `signal`, `float`, `freq`)
 *    Identifier[Dim, ...]      — array with shape dims (numeric or typeParam)
 *  The opening identifier is the element-type name. Anything else throws. */
function parsePortType(ctx: Ctx): PortTypeDecl {
  const elemTok = consume(ctx, 'ident', 'port type name')
  const element = elemTok.value as string
  if (peek(ctx).kind !== '[') return element

  ctx.i++  // consume `[`
  const shape: ShapeDim[] = []
  if (peek(ctx).kind !== ']') {
    shape.push(parseShapeDim(ctx))
    while (eat(ctx, ',')) {
      shape.push(parseShapeDim(ctx))
    }
  }
  consume(ctx, ']', `closing \`]\` of array type`)
  if (shape.length === 0) {
    throw new ParseError(`array type must have at least one shape dim`, elemTok)
  }
  return { kind: 'array', element, shape }
}

function parseShapeDim(ctx: Ctx): ShapeDim {
  const t = peek(ctx)
  if (t.kind === 'num') {
    ctx.i++
    if (!Number.isInteger(t.value) || (t.value as number) < 0) {
      throw new ParseError(`array shape dim must be a non-negative integer`, t)
    }
    return t.value as number
  }
  if (t.kind === 'ident') {
    ctx.i++
    const name = t.value as string
    if (!ctx.typeParams.has(name)) {
      throw new ParseError(
        `array shape dim '${name}' is not a declared type-param of the enclosing program`, t,
      )
    }
    return { op: 'typeParam', name }
  }
  throw new ParseError(`expected number or type-param name in array shape, got ${formatTok(t)}`, t)
}

/** Parse `[lo, hi]` after `in`. Each side may be `null` (sentinel) to
 *  indicate "no bound on this side", or a number literal (signed). */
function parseBounds(ctx: Ctx): [number | null, number | null] {
  consume(ctx, '[', '`[` opening bounds')
  const lo = parseBound(ctx)
  consume(ctx, ',', '`,` between bound lo/hi')
  const hi = parseBound(ctx)
  consume(ctx, ']', '`]` closing bounds')
  return [lo, hi]
}

function parseBound(ctx: Ctx): number | null {
  const t = peek(ctx)
  if (t.kind === 'ident' && t.value === 'null') {
    ctx.i++
    return null
  }
  // Allow `-1.0` etc. via expression parsing + literal extraction.
  const expr = parseExprAt(ctx)
  if (typeof expr !== 'number') {
    throw new ParseError(`bound must be a number literal or 'null'`, t)
  }
  return expr
}

// ─────────────────────────────────────────────────────────────
// Expression delegation
// ─────────────────────────────────────────────────────────────

function parseExprAt(ctx: Ctx): ExprNode {
  const { node, nextIdx } = parseExprFromTokens(ctx.toks, ctx.i)
  ctx.i = nextIdx
  return node
}
