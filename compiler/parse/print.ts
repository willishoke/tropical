/**
 * print.ts — pretty-printer: ParsedProgram → literate program text.
 *
 * Inverse of the parser. The contract is round-trip equivalence:
 *
 *     parseProgram(printProgram(parseProgram(text))) === parseProgram(text)
 *
 * The printer walks the parsed tree and emits canonically-formatted
 * tropical source. Output choices:
 *
 *  - Program declaration wrapped in a single ```tropical fenced block
 *    (markdown literate-program shell).
 *  - Body items on their own lines, 2-space indented.
 *  - No `;` separators between statements (newlines suffice; the parser
 *    accepts both).
 *  - Binary expressions emit minimal parens — only when the operand's
 *    precedence is lower than the surrounding operator's.
 *  - Number literals use shortest round-trip representation
 *    (default JS `String(n)` covers this for finite numbers).
 *  - No prose preservation. Format-preserving rewrites are a follow-up;
 *    MVP emits a fresh document.
 */

import type {
  ParsedExpr,
  ParsedExprOp,
  Program,
  Block,
  BodyDecl,
  BodyAssign,
  RegDecl,
  DelayDecl,
  ParamDecl,
  InstanceDecl,
  ProgramDecl,
  OutputAssign,
  NextUpdate,
  ProgramPort,
  ProgramPortSpec,
  PortTypeDecl,
  ShapeDim,
  TypeDef,
  StructTypeDef,
  StructField,
  SumTypeDef,
  SumVariant,
  AliasTypeDef,
  Tag,
  Match,
  MatchArmEntry,
  Let,
  Fold,
  Scan,
  Generate,
  Iterate,
  Chain,
  Map2,
  ZipWith,
  Index,
  Call,
  NestedOut,
  Binding,
  NameRef,
  BinaryOp,
  UnaryOp,
  BinaryOpTag,
  TypeArgEntry,
  InstanceInputEntry,
  TagPayloadEntry,
} from './nodes.js'

// ─────────────────────────────────────────────────────────────
// Public entry points
// ─────────────────────────────────────────────────────────────

/** Print a parsed program as a literate program document (markdown with a
 *  single fenced `tropical` block). Round-trip via the parser is
 *  structurally identity. */
export function printProgram(prog: Program): string {
  return ['```tropical', printProgramDecl(prog, 0), '```', ''].join('\n')
}

/** Print a single expression — useful for tests and debugging. */
export function printExpr(expr: ParsedExpr): string {
  return printExprAt(expr, PREC_LOWEST)
}

/** Print a parsed program *without* the markdown wrapper. Used by tools
 *  that already have a literate-program shell.
 *
 *  Output order inside the body: type defs (struct/enum/type aliases)
 *  first, then body decls in source order, then body assigns. This
 *  matches the parser's expectations and reads top-down. */
export function printProgramDecl(prog: Program, indent: number): string {
  const ind = '  '.repeat(indent)
  const header = printProgramHeader(prog)
  const lines: string[] = []
  for (const td of prog.ports?.type_defs ?? []) {
    lines.push(printTypeDef(td, indent + 1))
  }
  for (const decl of prog.body.decls ?? []) {
    if (decl.op === 'programDecl') {
      lines.push(printProgramDeclItem(decl, indent + 1))
    } else {
      lines.push(printBodyDecl(decl, indent + 1))
    }
  }
  for (const assign of prog.body.assigns ?? []) {
    lines.push(printBodyAssign(assign, indent + 1))
  }
  return lines.length === 0
    ? `${ind}${header} { }`
    : `${ind}${header} {\n${lines.join('\n')}\n${ind}}`
}

// ─────────────────────────────────────────────────────────────
// Program header
// ─────────────────────────────────────────────────────────────

function printProgramHeader(prog: Program): string {
  let out = `program ${prog.name}`
  if (prog.type_params && Object.keys(prog.type_params).length > 0) {
    const parts: string[] = []
    for (const [name, info] of Object.entries(prog.type_params)) {
      let s = `${name}: ${info.type}`
      if (info.default !== undefined) s += ` = ${info.default}`
      parts.push(s)
    }
    out += `<${parts.join(', ')}>`
  }
  const inputs = prog.ports?.inputs ?? []
  out += `(${inputs.map(p => printPortSpec(p, /*allowDefault=*/ true)).join(', ')})`
  const outputs = prog.ports?.outputs ?? []
  if (outputs.length > 0) {
    out += ` -> (${outputs.map(p => printPortSpec(p, /*allowDefault=*/ false)).join(', ')})`
  }
  if (prog.breaks_cycles === true) out += ' breaks_cycles'
  return out
}

function printPortSpec(spec: ProgramPort, allowDefault: boolean): string {
  if (typeof spec === 'string') return spec
  let out = spec.name
  if (spec.type !== undefined) out += `: ${printPortType(spec.type)}`
  if (allowDefault && spec.default !== undefined) {
    out += ` = ${printExprAt(spec.default, PREC_LOWEST)}`
  }
  return out
}

function printPortType(pt: PortTypeDecl): string {
  if (isNameRef(pt)) return pt.name
  return `${pt.element.name}[${pt.shape.map(printShapeDim).join(', ')}]`
}

function printShapeDim(d: ShapeDim): string {
  if (typeof d === 'number') return String(d)
  return d.name
}

// ─────────────────────────────────────────────────────────────
// Body items
// ─────────────────────────────────────────────────────────────

function printBodyDecl(decl: BodyDecl, indent: number): string {
  const ind = '  '.repeat(indent)
  switch (decl.op) {
    case 'regDecl':       return ind + printRegDecl(decl)
    case 'delayDecl':     return ind + printDelayDecl(decl)
    case 'paramDecl':     return ind + printParamDecl(decl)
    case 'instanceDecl':  return ind + printInstanceDecl(decl)
    case 'programDecl':   return printProgramDeclItem(decl, indent)
  }
}

function printRegDecl(d: RegDecl): string {
  let out = `reg ${d.name}`
  if (d.type !== undefined) out += `: ${d.type.name}`
  out += ` = ${printExprAt(d.init, PREC_LOWEST)}`
  return out
}

function printDelayDecl(d: DelayDecl): string {
  const typeAnn = d.type !== undefined ? `: ${d.type.name}` : ''
  return `delay ${d.name}${typeAnn} = ${printExprAt(d.update, PREC_LOWEST)} init ${printExprAt(d.init, PREC_LOWEST)}`
}

function printParamDecl(d: ParamDecl): string {
  let out = `param ${d.name}: smoothed`
  if (d.value !== undefined) out += ` = ${d.value}`
  return out
}

function printInstanceDecl(d: InstanceDecl): string {
  let out = `${d.name} = ${d.program.name}`
  if (d.type_args && d.type_args.length > 0) {
    out += `<${d.type_args.map(printTypeArg).join(', ')}>`
  }
  out += `(${(d.inputs ?? []).map(printInstanceInput).join(', ')})`
  return out
}

function printTypeArg(e: TypeArgEntry): string {
  return `${e.param.name}=${e.value}`
}

function printInstanceInput(e: InstanceInputEntry): string {
  return `${e.port.name}: ${printExprAt(e.value, PREC_LOWEST)}`
}

function printProgramDeclItem(d: ProgramDecl, indent: number): string {
  return printProgramDecl(d.program, indent)
}

function printBodyAssign(a: BodyAssign, indent: number): string {
  const ind = '  '.repeat(indent)
  if (a.op === 'outputAssign') {
    return `${ind}${a.name} = ${printExprAt(a.expr, PREC_LOWEST)}`
  }
  // nextUpdate
  return `${ind}next ${a.target.name} = ${printExprAt(a.expr, PREC_LOWEST)}`
}

// ─────────────────────────────────────────────────────────────
// Expressions — precedence-aware
// ─────────────────────────────────────────────────────────────

// Higher number = tighter binding. Levels match the parser's INFIX_LEVELS.
const PREC_LOWEST    = 0
const PREC_OR        = 1   // ||
const PREC_AND       = 2   // &&
const PREC_BIT_OR    = 3   // |
const PREC_BIT_XOR   = 4   // ^
const PREC_BIT_AND   = 5   // &
const PREC_EQUALITY  = 6   // == !=
const PREC_RELATION  = 7   // < <= > >=
const PREC_SHIFT     = 8   // << >>
const PREC_ADDITIVE  = 9   // + -
const PREC_MULT      = 10  // * / %
const PREC_UNARY     = 11  // -, !, ~ (prefix)
const PREC_POSTFIX   = 12  // .access [index] (call)
const PREC_ATOM      = 13  // numbers, identifiers, parenthesized

const BINARY_PREC: Record<BinaryOpTag, number> = {
  or: PREC_OR,
  and: PREC_AND,
  bitOr: PREC_BIT_OR,
  bitXor: PREC_BIT_XOR,
  bitAnd: PREC_BIT_AND,
  eq: PREC_EQUALITY, neq: PREC_EQUALITY,
  lt: PREC_RELATION, lte: PREC_RELATION, gt: PREC_RELATION, gte: PREC_RELATION,
  lshift: PREC_SHIFT, rshift: PREC_SHIFT,
  add: PREC_ADDITIVE, sub: PREC_ADDITIVE,
  mul: PREC_MULT, div: PREC_MULT, mod: PREC_MULT,
}

const BINARY_SYM: Record<BinaryOpTag, string> = {
  or: '||', and: '&&',
  bitOr: '|', bitXor: '^', bitAnd: '&',
  eq: '==', neq: '!=',
  lt: '<', lte: '<=', gt: '>', gte: '>=',
  lshift: '<<', rshift: '>>',
  add: '+', sub: '-',
  mul: '*', div: '/', mod: '%',
}

function printExprAt(node: ParsedExpr, contextPrec: number): string {
  if (typeof node === 'number')  return formatNumber(node)
  if (typeof node === 'boolean') return node ? 'true' : 'false'
  if (Array.isArray(node)) {
    return `[${node.map(n => printExprAt(n, PREC_LOWEST)).join(', ')}]`
  }
  return printOpNode(node, contextPrec)
}

function formatNumber(n: number): string {
  // Prefer compact form. JS `String(n)` already does shortest-round-trip
  // for finite numbers in practice (e.g. 0.1 → "0.1", -0.5 → "-0.5").
  return Number.isFinite(n) ? String(n) : 'NaN'
}

function printOpNode(node: ParsedExprOp, contextPrec: number): string {
  switch (node.op) {
    case 'nameRef':   return (node as NameRef).name
    case 'binding':   return (node as Binding).name
    case 'nestedOut': return printNestedOut(node as NestedOut)
    case 'index':     return printIndex(node as Index)
    case 'call':      return printCall(node as Call)
    case 'tag':       return printTag(node as Tag)
    case 'match':     return printMatch(node as Match)
    case 'let':       return parens(printLet(node as Let), contextPrec, PREC_LOWEST)
    case 'fold':      return printFold(node as Fold)
    case 'scan':      return printScan(node as Scan)
    case 'generate':  return printGenerate(node as Generate)
    case 'iterate':   return printIterate(node as Iterate)
    case 'chain':     return printChain(node as Chain)
    case 'map2':      return printMap2(node as Map2)
    case 'zipWith':   return printZipWith(node as ZipWith)
    default:
      if (node.op in BINARY_PREC) return printBinary(node as BinaryOp, contextPrec)
      // Unary
      return printUnary(node as UnaryOp, contextPrec)
  }
}

function parens(s: string, contextPrec: number, ownPrec: number): string {
  return ownPrec < contextPrec ? `(${s})` : s
}

function printBinary(node: BinaryOp, contextPrec: number): string {
  const prec = BINARY_PREC[node.op]
  const sym = BINARY_SYM[node.op]
  // Left-associative: left side accepts equal-prec without parens; right
  // side requires strictly greater (i.e., must escalate to one tighter).
  const lhs = printExprAt(node.args[0], prec)
  const rhs = printExprAt(node.args[1], prec + 1)
  return parens(`${lhs} ${sym} ${rhs}`, contextPrec, prec)
}

const UNARY_SYM: Record<string, string> = { neg: '-', not: '!', bitNot: '~' }

function printUnary(node: UnaryOp, contextPrec: number): string {
  const sym = UNARY_SYM[node.op]
  if (sym === undefined) {
    // Defensive — every parsed UnaryOp should be in UNARY_SYM.
    throw new Error(`printer: unknown unary op '${node.op}'`)
  }
  // Unary binds tighter than any binary; emit without parens unless the
  // surrounding context demands them.
  return parens(`${sym}${printExprAt(node.args[0], PREC_UNARY)}`, contextPrec, PREC_UNARY)
}

function printNestedOut(node: NestedOut): string {
  const portName = isNameRef(node.output) ? node.output.name : String(node.output)
  return `${node.ref.name}.${portName}`
}

function printIndex(node: Index): string {
  return `${printExprAt(node.args[0], PREC_POSTFIX)}[${printExprAt(node.args[1], PREC_LOWEST)}]`
}

function printCall(node: Call): string {
  // The callee is always a NameRef from the parser (only ident-call form
  // is supported in expressions). Defensive cast for the unlikely future.
  const callee = isNameRef(node.callee) ? node.callee.name
    : printExprAt(node.callee, PREC_POSTFIX)
  const args = node.args.map(a => printExprAt(a, PREC_LOWEST))
  return `${callee}(${args.join(', ')})`
}

function printTag(node: Tag): string {
  if (!node.payload || node.payload.length === 0) {
    return `${node.variant.name} { }`
  }
  return `${node.variant.name} { ${node.payload.map(printTagPayloadEntry).join(', ')} }`
}

function printTagPayloadEntry(e: TagPayloadEntry): string {
  return `${e.field.name}: ${printExprAt(e.value, PREC_LOWEST)}`
}

function printMatch(node: Match): string {
  const arms = node.arms.map(printMatchArm).join(', ')
  return `match ${printExprAt(node.scrutinee, PREC_LOWEST)} { ${arms} }`
}

function printMatchArm(arm: MatchArmEntry): string {
  const v = arm.variant.name
  const body = printExprAt(arm.body, PREC_LOWEST)
  if (arm.binds.length === 0) return `${v} => ${body}`
  // The parser carries the user-source `field: bind` pairing in
  // `binds`, so round-trip preserves the field labels exactly.
  const pattern = arm.binds.map(b => `${b.field.name}: ${b.bind}`).join(', ')
  return `${v} { ${pattern} } => ${body}`
}

function printLet(node: Let): string {
  const entries = Object.entries(node.bind).map(([k, v]) =>
    `${k}: ${printExprAt(v, PREC_LOWEST)}`
  )
  return `let { ${entries.join('; ')} } in ${printExprAt(node.in, PREC_LOWEST)}`
}

function printFold(node: Fold): string {
  return `fold(${printExprAt(node.over, PREC_LOWEST)}, ${printExprAt(node.init, PREC_LOWEST)}, (${node.acc_var}, ${node.elem_var}) => ${printExprAt(node.body, PREC_LOWEST)})`
}

function printScan(node: Scan): string {
  return `scan(${printExprAt(node.over, PREC_LOWEST)}, ${printExprAt(node.init, PREC_LOWEST)}, (${node.acc_var}, ${node.elem_var}) => ${printExprAt(node.body, PREC_LOWEST)})`
}

function printGenerate(node: Generate): string {
  return `generate(${printExprAt(node.count, PREC_LOWEST)}, (${node.var}) => ${printExprAt(node.body, PREC_LOWEST)})`
}

function printIterate(node: Iterate): string {
  return `iterate(${printExprAt(node.count, PREC_LOWEST)}, ${printExprAt(node.init, PREC_LOWEST)}, (${node.var}) => ${printExprAt(node.body, PREC_LOWEST)})`
}

function printChain(node: Chain): string {
  return `chain(${printExprAt(node.count, PREC_LOWEST)}, ${printExprAt(node.init, PREC_LOWEST)}, (${node.var}) => ${printExprAt(node.body, PREC_LOWEST)})`
}

function printMap2(node: Map2): string {
  return `map2(${printExprAt(node.over, PREC_LOWEST)}, (${node.elem_var}) => ${printExprAt(node.body, PREC_LOWEST)})`
}

function printZipWith(node: ZipWith): string {
  return `zipWith(${printExprAt(node.a, PREC_LOWEST)}, ${printExprAt(node.b, PREC_LOWEST)}, (${node.x_var}, ${node.y_var}) => ${printExprAt(node.body, PREC_LOWEST)})`
}

// ─────────────────────────────────────────────────────────────
// Type defs (printed at program level when present)
// ─────────────────────────────────────────────────────────────

export function printTypeDef(td: TypeDef, indent: number): string {
  const ind = '  '.repeat(indent)
  if (td.kind === 'struct') return `${ind}${printStructTypeDef(td)}`
  if (td.kind === 'sum')    return `${ind}${printSumTypeDef(td)}`
  return `${ind}${printAliasTypeDef(td)}`
}

function printStructTypeDef(td: StructTypeDef): string {
  if (td.fields.length === 0) return `struct ${td.name} { }`
  return `struct ${td.name} { ${td.fields.map(printStructField).join(', ')} }`
}

function printStructField(f: StructField): string {
  return `${f.name}: ${f.scalar_type}`
}

function printSumTypeDef(td: SumTypeDef): string {
  return `enum ${td.name} { ${td.variants.map(printSumVariant).join(', ')} }`
}

function printSumVariant(v: SumVariant): string {
  if (v.payload.length === 0) return v.name
  return `${v.name}(${v.payload.map(printStructField).join(', ')})`
}

function printAliasTypeDef(td: AliasTypeDef): string {
  return `type ${td.name} = ${td.base.name}`
}

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────

function isNameRef(v: unknown): v is NameRef {
  return typeof v === 'object' && v !== null && !Array.isArray(v)
    && (v as { op?: unknown }).op === 'nameRef'
}
