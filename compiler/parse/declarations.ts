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

import { tokenize, type Tok } from './lexer.js'
import { parseExprFromTokens } from './expressions.js'
import { parseBodyFromTokens, type BodyOptions } from './statements.js'
import { commaList, consume, eat, formatTok, isContextualKw, peek, ParseError, type Cursor } from './shared.js'
import type {
  ExprNode, ProgramNode, ProgramPort, ProgramPortSpec, ProgramPorts,
  PortTypeDecl, ShapeDim, ScalarKind,
  TypeDef, StructTypeDef, StructField, SumTypeDef, SumVariant, AliasTypeDef,
  ProgramDeclNode,
} from './nodes.js'
import { nameRef } from './nodes.js'

// Re-export the node types so existing public-API consumers (tests etc.)
// keep their imports stable.
export type {
  ProgramNode, ProgramPort, ProgramPortSpec, ProgramPorts,
  PortTypeDecl, ShapeDim, ScalarKind,
  TypeDef, StructTypeDef, StructField, SumTypeDef, SumVariant, AliasTypeDef,
} from './nodes.js'

// ─────────────────────────────────────────────────────────────
// Parser context
// ─────────────────────────────────────────────────────────────

/** Parser context. The parser does NO scope analysis: every reference is
 *  emitted as a NameRefNode and the elaborator resolves them. */
interface Ctx extends Cursor {}

// ─────────────────────────────────────────────────────────────
// Public entry points
// ─────────────────────────────────────────────────────────────

/** Parse a top-level program declaration from source text. */
export function parseProgram(src: string): ProgramNode {
  const toks = tokenize(src)
  const ctx: Ctx = { toks, i: 0 }
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
  const ctx: Ctx = { toks, i: startIdx }
  const node = parseProgramFromCtx(ctx)
  return { node, nextIdx: ctx.i }
}

/** Body-parser hook: parse a nested `program` decl and wrap it as a
 *  `programDecl` body item. */
function parseNestedProgramDecl(
  toks: Tok[], startIdx: number,
): { node: ProgramDeclNode; nextIdx: number } {
  const { node: inner, nextIdx } = parseProgramFromTokens(toks, startIdx)
  const node: ProgramDeclNode = { op: 'programDecl', name: inner.name, program: inner }
  return { node, nextIdx }
}

/** Body-parser hook: dispatch `struct`/`enum`/`type` to the right ADT
 *  parser. The body parser collects the result into a separate `typeDefs`
 *  array (returned alongside the BlockNode), which we then route into
 *  the program's `ports.type_defs`. */
function parseBodyTypeDef(
  toks: Tok[], startIdx: number,
): { typeDef: TypeDef; nextIdx: number } {
  const ctx: Ctx = { toks, i: startIdx }
  const t = peek(ctx)
  let typeDef: TypeDef
  if (t.kind === 'struct') typeDef = parseStructDecl(ctx)
  else if (t.kind === 'enum') typeDef = parseEnumDecl(ctx)
  else if (t.kind === 'type') typeDef = parseAliasDecl(ctx)
  else throw new ParseError(`expected struct/enum/type, got ${formatTok(t)}`, t)
  return { typeDef, nextIdx: ctx.i }
}

const BODY_OPTS: BodyOptions = {
  programDeclParser: parseNestedProgramDecl,
  typeDefHandler: parseBodyTypeDef,
}

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

  // Optional contextual `breaks_cycles` flag — informs the legacy
  // flattener's cycle detector. Recognized as an ident token because the
  // lexer doesn't promote it to a reserved word.
  let breaksCycles = false
  if (isContextualKw(peek(ctx), 'breaks_cycles')) {
    ctx.i++
    breaksCycles = true
  }

  // Body
  consume(ctx, '{', `\`{\` opening body of '${name}'`)
  const { block, typeDefs, nextIdx } = parseBodyFromTokens(ctx.toks, ctx.i, BODY_OPTS)
  ctx.i = nextIdx
  consume(ctx, '}', `\`}\` closing body of '${name}'`)

  const node: ProgramNode = { op: 'program', name, body: block }
  if (typeParams && Object.keys(typeParams).length > 0) node.type_params = typeParams
  const ports: ProgramPorts = {}
  if (inputs.length > 0)  ports.inputs  = inputs
  if (outputs && outputs.length > 0) ports.outputs = outputs
  if (typeDefs.length > 0) ports.type_defs = typeDefs
  if (ports.inputs || ports.outputs || ports.type_defs) node.ports = ports
  if (breaksCycles) node.breaks_cycles = true
  return node
}

// ─────────────────────────────────────────────────────────────
// ADT decls — struct / enum / type alias (Phase B5)
// ─────────────────────────────────────────────────────────────

const SCALAR_KINDS: ReadonlySet<string> = new Set(['float', 'int', 'bool'])

function parseScalarKind(ctx: Ctx, what: string): ScalarKind {
  const t = consume(ctx, 'ident', what)
  const k = t.value as string
  if (!SCALAR_KINDS.has(k)) {
    throw new ParseError(`${what}: expected float/int/bool, got '${k}'`, t)
  }
  return k as ScalarKind
}

/** struct Name { field: scalarType, ... } */
function parseStructDecl(ctx: Ctx): StructTypeDef {
  consume(ctx, 'struct', 'struct keyword')
  const name = consume(ctx, 'ident', 'struct name').value as string
  consume(ctx, '{', `\`{\` after struct '${name}'`)
  const fields = parseFieldList(ctx, `struct '${name}'`)
  consume(ctx, '}', `\`}\` closing struct '${name}'`)
  return { kind: 'struct', name, fields }
}

/** Comma-separated `name: scalarType` list inside `{...}`. Used by both
 *  struct fields and sum-variant payloads. Duplicate detection is inline
 *  so the error position points at the duplicate's token, not the
 *  closing brace. */
function parseFieldList(ctx: Ctx, where: string): StructField[] {
  const seen = new Set<string>()
  return commaList(ctx, '}', () => {
    const nameTok = consume(ctx, 'ident', `${where}: field name`)
    const fieldName = nameTok.value as string
    if (seen.has(fieldName)) {
      throw new ParseError(`${where}: duplicate field '${fieldName}'`, nameTok)
    }
    seen.add(fieldName)
    consume(ctx, ':', `${where}: \`:\` after field name`)
    const scalar_type = parseScalarKind(ctx, `${where}: field type`)
    return { name: fieldName, scalar_type }
  })
}

/** enum Name { Variant, Variant(field: type, ...), ... } */
function parseEnumDecl(ctx: Ctx): SumTypeDef {
  consume(ctx, 'enum', 'enum keyword')
  const name = consume(ctx, 'ident', 'enum name').value as string
  consume(ctx, '{', `\`{\` after enum '${name}'`)
  const seen = new Set<string>()
  const variants = commaList(ctx, '}', () => {
    const v = parseSumVariant(ctx, name)
    if (seen.has(v.name)) {
      // The variant token has already been consumed; rewind one so the
      // error points at the variant name. (commaList doesn't expose the
      // token, but we can re-derive: it's the previous `ident` we just
      // consumed two-or-more tokens ago. Approximate with peek for now.)
      throw new ParseError(`enum '${name}': duplicate variant '${v.name}'`, peek(ctx))
    }
    seen.add(v.name)
    return v
  })
  consume(ctx, '}', `\`}\` closing enum '${name}'`)
  return { kind: 'sum', name, variants }
}

function parseSumVariant(ctx: Ctx, enumName: string): SumVariant {
  const nameTok = consume(ctx, 'ident', `enum '${enumName}': variant name`)
  const variantName = nameTok.value as string
  if (peek(ctx).kind !== '(') {
    return { name: variantName, payload: [] }
  }
  ctx.i++  // consume `(`
  const seenFields = new Set<string>()
  const payload = commaList(ctx, ')', () => {
    const pnameTok = consume(ctx, 'ident', `variant '${variantName}' field name`)
    const pname = pnameTok.value as string
    if (seenFields.has(pname)) {
      throw new ParseError(`variant '${variantName}': duplicate field '${pname}'`, pnameTok)
    }
    seenFields.add(pname)
    consume(ctx, ':', `variant '${variantName}' \`:\` after field name`)
    const scalar_type = parseScalarKind(ctx, `variant '${variantName}' field type`)
    return { name: pname, scalar_type }
  })
  consume(ctx, ')', `closing \`)\` of variant '${variantName}' payload`)
  return { name: variantName, payload }
}

/** type AliasName = baseScalar in [lo, hi] */
function parseAliasDecl(ctx: Ctx): AliasTypeDef {
  consume(ctx, 'type', 'type keyword')
  const name = consume(ctx, 'ident', 'alias name').value as string
  consume(ctx, '=', `\`=\` after alias '${name}'`)
  const baseTok = consume(ctx, 'ident', `base type for alias '${name}'`)
  consume(ctx, 'in', `\`in\` after base type for alias '${name}'`)
  const bounds = parseBounds(ctx)
  return { kind: 'alias', name, base: nameRef(baseTok.value as string), bounds }
}

// ─────────────────────────────────────────────────────────────
// Type params: <N: int [= default], M: int>
// ─────────────────────────────────────────────────────────────

function parseTypeParams(ctx: Ctx): Record<string, { type: 'int'; default?: number }> {
  consume(ctx, '<', 'opening `<` of type params')
  const out: Record<string, { type: 'int'; default?: number }> = {}
  commaList(ctx, '>', () => {
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
  })
  consume(ctx, '>', 'closing `>` of type params')
  return out
}

// ─────────────────────────────────────────────────────────────
// Port lists and port specs
// ─────────────────────────────────────────────────────────────

function parsePortList(ctx: Ctx, allowDefault: boolean): ProgramPort[] {
  return commaList(ctx, ')', () => parsePortSpec(ctx, allowDefault))
}

/** Parse one port spec. Forms (input):
 *    name
 *    name = default                   (no explicit type)
 *    name: type
 *    name: type = default
 *    name: type in [lo, hi]
 *    name: type = default in [lo, hi]
 *  Outputs accept the same forms minus `= default`.
 *  When the spec has only a name (no type, default, or bounds), emits the
 *  bare-string form to match stdlib JSON convention. The untyped-with-
 *  default form preserves legacy stdlib programs (e.g. BitCrusher,
 *  LadderFilter) whose port specs declare a default but no type. */
function parsePortSpec(ctx: Ctx, allowDefault: boolean): ProgramPort {
  const nameTok = consume(ctx, 'ident', 'port name')
  const name = nameTok.value as string

  let type: PortTypeDecl | undefined
  let defaultExpr: ExprNode | undefined
  let bounds: [number | null, number | null] | undefined

  if (peek(ctx).kind === ':') {
    ctx.i++  // consume `:`
    type = parsePortType(ctx)
  } else if (peek(ctx).kind !== '=') {
    return name  // bare-string form
  }

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

  const spec: ProgramPortSpec = { name }
  if (type !== undefined) spec.type = type
  if (defaultExpr !== undefined) spec.default = defaultExpr
  if (bounds !== undefined) spec.bounds = bounds
  return spec
}

/** Parse a port type:
 *    Identifier                — bare scalar (e.g. `signal`, `float`, `freq`)
 *    Identifier[Dim, ...]      — array with shape dims (numeric or NameRefNode)
 *  Both the element and shape-dim identifiers become NameRefNodes; the
 *  elaborator resolves them against scalar kinds + program type-params
 *  + type aliases. */
function parsePortType(ctx: Ctx): PortTypeDecl {
  const elemTok = consume(ctx, 'ident', 'port type name')
  const element = nameRef(elemTok.value as string)
  if (peek(ctx).kind !== '[') return element

  ctx.i++  // consume `[`
  const shape = commaList(ctx, ']', () => parseShapeDim(ctx))
  consume(ctx, ']', `closing \`]\` of array type`)
  if (shape.length === 0) {
    throw new ParseError(`array type must have at least one shape dim`, elemTok)
  }
  return { kind: 'array', element, shape }
}

/** Parse a single array-shape dim. Numeric literal or identifier — the
 *  identifier becomes a NameRefNode the elaborator resolves against the
 *  enclosing program's type-params. The parser does NO scope analysis;
 *  every name is a NameRef awaiting elaboration. */
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
    return nameRef(t.value as string)
  }
  throw new ParseError(`expected number or identifier in array shape, got ${formatTok(t)}`, t)
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

/** Parse a single bound: `null` sentinel, or a signed numeric literal.
 *  Direct lexer handling — no need to detour through the expression
 *  parser (which would only constant-fold `neg(<num>)` into a negative
 *  number anyway). The grammar is just `'-'? num | 'null'`. */
function parseBound(ctx: Ctx): number | null {
  const t = peek(ctx)
  if (isContextualKw(t, 'null')) {
    ctx.i++
    return null
  }
  let sign = 1
  if (eat(ctx, '-')) sign = -1
  const numTok = peek(ctx)
  if (numTok.kind !== 'num') {
    throw new ParseError(`bound must be a number literal or 'null', got ${formatTok(t)}`, t)
  }
  ctx.i++
  return sign * (numTok.value as number)
}

// ─────────────────────────────────────────────────────────────
// Expression delegation
// ─────────────────────────────────────────────────────────────

function parseExprAt(ctx: Ctx): ExprNode {
  const { node, nextIdx } = parseExprFromTokens(ctx.toks, ctx.i)
  ctx.i = nextIdx
  return node
}
