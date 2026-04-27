/**
 * compiler/ir/elaborator.ts — parsed tree → resolved graph.
 *
 * The elaborator's job is substitution of free variables (NameRefNodes)
 * by their decl objects. It runs in a single top-down pass over the
 * parsed program: each declaration is constructed once when its
 * parsed-counterpart is encountered, registered in the appropriate
 * scope, and re-used by reference at every site that names it.
 *
 * Reference identity falls out of this discipline: every RegRef.decl
 * for a given register is `===` the same RegDecl object. The output is
 * a graph (it admits cycles via delays + feedback) where every reference
 * is a TypeScript reference, not a string lookup.
 *
 * This module is one function (`elaborate`) plus its helpers. There are
 * no factories, no classes, no smart-constructors. Each Decl object is
 * built by the elaborator at its single construction site; the resolved
 * IR's introduction rules are the literal object literals in this file.
 */

import type {
  ParsedExprNode,
  ExprOpNode as ParsedExprOpNode,
  NameRefNode as ParsedNameRefNode,
  ProgramNode as ParsedProgramNode,
  BlockNode as ParsedBlockNode,
  BodyDecl as ParsedBodyDecl,
  BodyAssign as ParsedBodyAssign,
  RegDeclNode as ParsedRegDecl,
  DelayDeclNode as ParsedDelayDecl,
  ParamDeclNode as ParsedParamDecl,
  InstanceDeclNode as ParsedInstanceDecl,
  ProgramDeclNode as ParsedProgramDecl,
  OutputAssignNode as ParsedOutputAssign,
  NextUpdateNode as ParsedNextUpdate,
  ProgramPort as ParsedProgramPort,
  PortTypeDecl as ParsedPortType,
  ShapeDim as ParsedShapeDim,
  TypeDef as ParsedTypeDef,
  StructTypeDef as ParsedStructTypeDef,
  SumTypeDef as ParsedSumTypeDef,
  AliasTypeDef as ParsedAliasTypeDef,
  StructField as ParsedStructField,
  CallNode as ParsedCallNode,
  TagNode as ParsedTag,
  MatchNode as ParsedMatch,
  LetNode as ParsedLetNode,
  FoldNode as ParsedFold,
  ScanNode as ParsedScan,
  GenerateNode as ParsedGenerate,
  IterateNode as ParsedIterate,
  ChainNode as ParsedChain,
  Map2Node as ParsedMap2,
  ZipWithNode as ParsedZipWith,
  IndexNode as ParsedIndex,
  NestedOutNode as ParsedNestedOut,
  BindingNode as ParsedBindingNode,
  BinaryOpNode as ParsedBinary,
  UnaryOpNode as ParsedUnary,
} from '../parse/nodes.js'
import type {
  ResolvedProgram, ResolvedBlock, ResolvedProgramPorts,
  ResolvedExpr, ResolvedExprOpNode,
  InputDecl, OutputDecl, TypeParamDecl,
  RegDecl, DelayDecl, ParamDecl, InstanceDecl, ProgramDecl, BodyDecl,
  BodyAssign, OutputAssign, NextUpdate,
  TypeDef, StructTypeDef, SumTypeDef, SumVariant, AliasTypeDef, StructField,
  PortType, ShapeDim, ScalarKind,
  BinderDecl,
  InputRef, RegRef, DelayRef, ParamRef, TypeParamRef, BindingRef,
  NestedOut,
  TagExpr, MatchExpr, MatchArm,
  LetExpr,
  FoldExpr, ScanExpr, GenerateExpr, IterateExpr, ChainExpr, Map2Expr, ZipWithExpr,
  ClampNode, SelectNode, IndexNode,
  BinaryOpNode, UnaryOpNode, UnaryOpTag,
  SampleRateNode, SampleIndexNode,
} from './nodes.js'
import { ElaborationError } from './nodes.js'

const SCALAR_KINDS: ReadonlySet<string> = new Set(['float', 'int', 'bool'])
const SCALAR_ALIASES: ReadonlySet<string> = new Set([
  // bare scalars
  'float', 'int', 'bool',
  // common builtin port-type aliases that pass through to ScalarKind
  // (these stay as ScalarKind, not AliasTypeDef, since they have no
  // bounds metadata — they're the user-facing names for raw types)
  'signal', 'freq', 'unipolar', 'bipolar',
])

/** Builtin port-type aliases that map to a ScalarKind. */
const BUILTIN_TYPE_TO_SCALAR: Record<string, ScalarKind> = {
  float: 'float', int: 'int', bool: 'bool',
  signal: 'float', freq: 'float', unipolar: 'float', bipolar: 'float',
}

/** Builtin nullary calls: `sample_rate()`, `sample_index()`. */
const NULLARY_CALLS: ReadonlySet<string> = new Set(['sample_rate', 'sample_index'])

/** Builtin unary function calls — surface name → resolved op tag. */
const UNARY_CALLS: Record<string, UnaryOpTag> = {
  sqrt: 'sqrt', abs: 'abs', neg: 'neg',
  floor: 'floor', ceil: 'ceil', round: 'round',
  not: 'not', bit_not: 'bitNot',
  to_int: 'toInt', to_bool: 'toBool', to_float: 'toFloat',
  float_exponent: 'floatExponent',
}

// ─────────────────────────────────────────────────────────────
// Scope
// ─────────────────────────────────────────────────────────────

interface Scope {
  inputs: Map<string, InputDecl>
  outputs: Map<string, OutputDecl>
  typeParams: Map<string, TypeParamDecl>
  regs: Map<string, RegDecl>
  delays: Map<string, DelayDecl>
  params: Map<string, ParamDecl>
  instances: Map<string, InstanceDecl>
  /** Sub-program decls visible in this scope (nested programDecl). */
  programs: Map<string, ResolvedProgram>
  /** Type defs (struct/sum/alias) by name. */
  typeDefs: Map<string, TypeDef>
  /** Variant name → its parent SumTypeDef + variant decl. Variants are
   *  unique across all sum types in a single program; the parser doesn't
   *  enforce that, so we check on registration here. */
  variantOf: Map<string, SumVariant>
  /** Active anonymous binders (let/combinator/match-arm). */
  binders: Map<string, BinderDecl>
  /** Parent scope — for nested programs to read outer type-defs and
   *  external program types. (Decls themselves don't leak — only
   *  type-defs and program registrations.) */
  parent?: Scope
}

function emptyScope(parent?: Scope): Scope {
  return {
    inputs: new Map(),
    outputs: new Map(),
    typeParams: new Map(),
    regs: new Map(),
    delays: new Map(),
    params: new Map(),
    instances: new Map(),
    programs: new Map(),
    typeDefs: new Map(),
    variantOf: new Map(),
    binders: new Map(),
    parent,
  }
}

/** Look up a name across scope categories in a defined order. Used when
 *  resolving a NameRefNode in expression position — the position has a
 *  fixed semantic intent (a value-producing reference), and we try each
 *  applicable scope. */
function lookupValueRef(scope: Scope, name: string): ResolvedExprOpNode | null {
  // Local binders (innermost-first via the Scope's own state)
  const binder = scope.binders.get(name)
  if (binder) {
    const ref: BindingRef = { op: 'bindingRef', decl: binder }
    return ref
  }
  const reg = scope.regs.get(name)
  if (reg) {
    const ref: RegRef = { op: 'regRef', decl: reg }
    return ref
  }
  const delay = scope.delays.get(name)
  if (delay) {
    const ref: DelayRef = { op: 'delayRef', decl: delay }
    return ref
  }
  const param = scope.params.get(name)
  if (param) {
    const ref: ParamRef = { op: 'paramRef', decl: param }
    return ref
  }
  const input = scope.inputs.get(name)
  if (input) {
    const ref: InputRef = { op: 'inputRef', decl: input }
    return ref
  }
  const tp = scope.typeParams.get(name)
  if (tp) {
    const ref: TypeParamRef = { op: 'typeParamRef', decl: tp }
    return ref
  }
  return null
}

/** Look up a sub-program by name (instances reference these by NameRef). */
function lookupProgram(scope: Scope, name: string): ResolvedProgram | null {
  let s: Scope | undefined = scope
  while (s) {
    const p = s.programs.get(name)
    if (p) return p
    s = s.parent
  }
  return null
}

/** Resolve a port-type's element name (must be a scalar kind or alias). */
function resolveElement(scope: Scope, ref: ParsedNameRefNode): ScalarKind | AliasTypeDef {
  const builtin = BUILTIN_TYPE_TO_SCALAR[ref.name]
  if (builtin) return builtin
  let s: Scope | undefined = scope
  while (s) {
    const td = s.typeDefs.get(ref.name)
    if (td !== undefined) {
      if (td.op !== 'aliasTypeDef') {
        throw new ElaborationError(
          `port type '${ref.name}' must be a scalar kind or alias; got ${td.op}`,
        )
      }
      return td
    }
    s = s.parent
  }
  throw new ElaborationError(`unknown type name '${ref.name}'`)
}

// ─────────────────────────────────────────────────────────────
// Public entry
// ─────────────────────────────────────────────────────────────

/** Resolve a parsed program to a graph IR. The returned object carries
 *  declared inputs/outputs/type-params/type-defs as Decl objects, and
 *  every reference inside the body is a direct edge to one of those
 *  decls. */
export function elaborate(prog: ParsedProgramNode): ResolvedProgram {
  return elaborateProgram(prog, undefined)
}

function elaborateProgram(prog: ParsedProgramNode, parent: Scope | undefined): ResolvedProgram {
  const scope = emptyScope(parent)

  // 1. Type-defs from `ports.type_defs` first — the elaborator's port-type
  //    + decl walks need them in scope.
  const typeDefs: TypeDef[] = []
  for (const td of prog.ports?.type_defs ?? []) {
    const resolved = resolveTypeDef(td, scope)
    registerTypeDef(scope, resolved)
    typeDefs.push(resolved)
  }

  // 2. Type-params (`<N: int = 4>`).
  const typeParams: TypeParamDecl[] = []
  if (prog.type_params) {
    for (const [name, info] of Object.entries(prog.type_params)) {
      const decl: TypeParamDecl = { op: 'typeParamDecl', name }
      if (info.default !== undefined) decl.default = info.default
      scope.typeParams.set(name, decl)
      typeParams.push(decl)
    }
  }

  // 3. Input ports + output ports. Inputs may have `default` exprs that
  //    can reference type-params in shape position — type-params are now
  //    in scope from step 2.
  const inputs: InputDecl[] = []
  for (const portSpec of prog.ports?.inputs ?? []) {
    const decl = resolveInputPort(portSpec, scope)
    if (scope.inputs.has(decl.name)) {
      throw new ElaborationError(`duplicate input port '${decl.name}'`)
    }
    scope.inputs.set(decl.name, decl)
    inputs.push(decl)
  }
  const outputs: OutputDecl[] = []
  for (const portSpec of prog.ports?.outputs ?? []) {
    const decl = resolveOutputPort(portSpec, scope)
    if (scope.outputs.has(decl.name)) {
      throw new ElaborationError(`duplicate output port '${decl.name}'`)
    }
    scope.outputs.set(decl.name, decl)
    outputs.push(decl)
  }

  // 4. Register body decls (reg/delay/param/instance/programDecl) first,
  //    so expressions in those decls and in body assigns can reference
  //    one another regardless of source order. Register builds decl
  //    shells with placeholder expressions; pairing is recorded so the
  //    second pass can fill them in.
  const pairing = new Map<ParsedBodyDecl, BodyDecl>()
  const decls = registerBodyDecls(prog.body, scope, pairing)

  // 5. Resolve expressions inside body decls (init/update/instance inputs).
  for (const [parsed, resolved] of pairing) {
    resolveDeclExpressions(parsed, resolved, scope)
  }

  // 6. Resolve body assigns.
  const assigns: BodyAssign[] = []
  for (const a of prog.body.assigns ?? []) {
    assigns.push(resolveAssign(a, scope))
  }

  const block: ResolvedBlock = { op: 'block', decls, assigns }
  const ports: ResolvedProgramPorts = { inputs, outputs, typeDefs }
  const resolved: ResolvedProgram = {
    op: 'program',
    name: prog.name,
    typeParams,
    ports,
    body: block,
  }

  // Make this program visible to its containing scope (for sibling
  // nested programs) — caller registers the wrapping ProgramDecl.
  return resolved
}

// ─────────────────────────────────────────────────────────────
// Type defs
// ─────────────────────────────────────────────────────────────

function resolveTypeDef(td: ParsedTypeDef, scope: Scope): TypeDef {
  if (td.kind === 'struct') return resolveStructTypeDef(td, scope)
  if (td.kind === 'sum')    return resolveSumTypeDef(td, scope)
  if (td.kind === 'alias')  return resolveAliasTypeDef(td, scope)
  // Defensive: discriminator should be exhaustive.
  throw new ElaborationError(`unknown type-def kind`)
}

function resolveStructTypeDef(td: ParsedStructTypeDef, scope: Scope): StructTypeDef {
  const fields = td.fields.map(f => resolveStructField(f, scope))
  return { op: 'structTypeDef', name: td.name, fields }
}

function resolveStructField(f: ParsedStructField, scope: Scope): StructField {
  // ParsedStructField has `scalar_type: 'float'|'int'|'bool'` (a literal).
  // The resolved field's `type` can also be an AliasTypeDef, but since
  // the parser only allows scalar literals here, all parsed fields land
  // as ScalarKind.
  return { op: 'structField', name: f.name, type: f.scalar_type }
}

function resolveSumTypeDef(td: ParsedSumTypeDef, scope: Scope): SumTypeDef {
  // Build the sum decl shell first so each variant can hold its
  // back-pointer.
  const sum: SumTypeDef = { op: 'sumTypeDef', name: td.name, variants: [] }
  for (const v of td.variants) {
    const variant: SumVariant = {
      op: 'sumVariant',
      name: v.name,
      payload: v.payload.map(f => resolveStructField(f, scope)),
      parent: sum,
    }
    sum.variants.push(variant)
  }
  return sum
}

function resolveAliasTypeDef(td: ParsedAliasTypeDef, scope: Scope): AliasTypeDef {
  if (!SCALAR_KINDS.has(td.base.name)) {
    throw new ElaborationError(
      `alias '${td.name}' base must be a scalar kind (float/int/bool); got '${td.base.name}'`,
    )
  }
  return {
    op: 'aliasTypeDef',
    name: td.name,
    base: td.base.name as ScalarKind,
    bounds: td.bounds,
  }
}

function registerTypeDef(scope: Scope, td: TypeDef): void {
  if (scope.typeDefs.has(td.name)) {
    throw new ElaborationError(`duplicate type def '${td.name}'`)
  }
  scope.typeDefs.set(td.name, td)
  if (td.op === 'sumTypeDef') {
    for (const v of td.variants) {
      if (scope.variantOf.has(v.name)) {
        throw new ElaborationError(
          `variant '${v.name}' is declared in multiple sum types — variant names must be unique`,
        )
      }
      scope.variantOf.set(v.name, v)
    }
  }
}

// ─────────────────────────────────────────────────────────────
// Port specs
// ─────────────────────────────────────────────────────────────

function resolveInputPort(spec: ParsedProgramPort, scope: Scope): InputDecl {
  if (typeof spec === 'string') {
    return { op: 'inputDecl', name: spec }
  }
  const decl: InputDecl = { op: 'inputDecl', name: spec.name }
  if (spec.type !== undefined) decl.type = resolvePortType(spec.type, scope)
  if (spec.default !== undefined) decl.default = resolveExpr(spec.default, scope)
  if (spec.bounds !== undefined) decl.bounds = spec.bounds
  return decl
}

function resolveOutputPort(spec: ParsedProgramPort, scope: Scope): OutputDecl {
  if (typeof spec === 'string') {
    return { op: 'outputDecl', name: spec }
  }
  const decl: OutputDecl = { op: 'outputDecl', name: spec.name }
  if (spec.type !== undefined) decl.type = resolvePortType(spec.type, scope)
  if (spec.bounds !== undefined) decl.bounds = spec.bounds
  return decl
}

function resolvePortType(pt: ParsedPortType, scope: Scope): PortType {
  if (isParsedNameRef(pt)) {
    const builtin = BUILTIN_TYPE_TO_SCALAR[pt.name]
    if (builtin) return { kind: 'scalar', scalar: builtin }
    const td = lookupTypeDef(scope, pt.name)
    if (td && td.op === 'aliasTypeDef') return { kind: 'alias', alias: td }
    throw new ElaborationError(`unknown port type '${pt.name}'`)
  }
  // Array form
  const element = resolveElement(scope, pt.element)
  const shape: ShapeDim[] = pt.shape.map(d => resolveShapeDim(d, scope))
  return { kind: 'array', element, shape }
}

function resolveShapeDim(d: ParsedShapeDim, scope: Scope): ShapeDim {
  if (typeof d === 'number') return d
  // d is a NameRefNode in shape position — must resolve to a TypeParamDecl.
  let s: Scope | undefined = scope
  while (s) {
    const tp = s.typeParams.get(d.name)
    if (tp) return tp
    s = s.parent
  }
  throw new ElaborationError(
    `array shape dim '${d.name}' is not a declared type-param of any enclosing program`,
  )
}

function lookupTypeDef(scope: Scope, name: string): TypeDef | null {
  let s: Scope | undefined = scope
  while (s) {
    const td = s.typeDefs.get(name)
    if (td !== undefined) return td
    s = s.parent
  }
  return null
}

// ─────────────────────────────────────────────────────────────
// Body decls — register first, then resolve expressions
// ─────────────────────────────────────────────────────────────

function registerBodyDecls(
  body: ParsedBlockNode,
  scope: Scope,
  pairing: Map<ParsedBodyDecl, BodyDecl>,
): BodyDecl[] {
  const out: BodyDecl[] = []
  // Programs first: nested sub-programs need to be resolved before any
  // sibling instance decls reference them.
  for (const d of body.decls ?? []) {
    if (isParsedProgramDecl(d)) {
      const inner = elaborateProgram(d.program, scope)
      const decl: ProgramDecl = { op: 'programDecl', name: d.name, program: inner }
      if (scope.programs.has(d.name)) {
        throw new ElaborationError(`duplicate nested program '${d.name}'`)
      }
      scope.programs.set(d.name, inner)
      out.push(decl)
      // No second-pass work for programDecl — the inner program was
      // fully elaborated above.
    }
  }

  // Then the rest, in source order. We construct decl shells (with
  // expressions left as placeholders) and register them in scope, so
  // forward refs work. Expressions are resolved in a second pass via
  // the pairing map.
  for (const d of body.decls ?? []) {
    if (isParsedProgramDecl(d)) continue  // already handled
    const decl = registerOneDecl(d, scope)
    pairing.set(d, decl)
    out.push(decl)
  }
  return out
}

function registerOneDecl(d: ParsedBodyDecl, scope: Scope): BodyDecl {
  if (d.op === 'regDecl')   return registerRegDecl(d, scope)
  if (d.op === 'delayDecl') return registerDelayDecl(d, scope)
  if (d.op === 'paramDecl') return registerParamDecl(d, scope)
  if (d.op === 'instanceDecl') return registerInstanceDecl(d, scope)
  // programDecl handled in pre-pass
  throw new ElaborationError(`unexpected body decl: ${(d as { op: string }).op}`)
}

function registerRegDecl(d: ParsedRegDecl, scope: Scope): RegDecl {
  if (scope.regs.has(d.name)) {
    throw new ElaborationError(`duplicate reg '${d.name}'`)
  }
  // type field: NameRef resolved to ScalarKind | AliasTypeDef
  let type: RegDecl['type']
  if (d.type) {
    const builtin = BUILTIN_TYPE_TO_SCALAR[d.type.name]
    if (builtin) type = builtin
    else {
      const td = lookupTypeDef(scope, d.type.name)
      if (td && td.op === 'aliasTypeDef') type = td
      else throw new ElaborationError(
        `reg '${d.name}': unknown type '${d.type.name}'`,
      )
    }
  }
  // init resolved later (second pass) — placeholder for now
  const decl: RegDecl = { op: 'regDecl', name: d.name, init: 0, ...(type ? { type } : {}) }
  scope.regs.set(d.name, decl)
  return decl
}

function registerDelayDecl(d: ParsedDelayDecl, scope: Scope): DelayDecl {
  if (scope.delays.has(d.name)) {
    throw new ElaborationError(`duplicate delay '${d.name}'`)
  }
  const decl: DelayDecl = { op: 'delayDecl', name: d.name, update: 0, init: 0 }
  scope.delays.set(d.name, decl)
  return decl
}

function registerParamDecl(d: ParsedParamDecl, scope: Scope): ParamDecl {
  if (scope.params.has(d.name)) {
    throw new ElaborationError(`duplicate param '${d.name}'`)
  }
  const decl: ParamDecl = { op: 'paramDecl', name: d.name, kind: d.type }
  if (d.value !== undefined) decl.value = d.value
  scope.params.set(d.name, decl)
  return decl
}

function registerInstanceDecl(d: ParsedInstanceDecl, scope: Scope): InstanceDecl {
  if (scope.instances.has(d.name)) {
    throw new ElaborationError(`duplicate instance '${d.name}'`)
  }
  // Resolve program type — only nested programs supported in B6.
  const targetProgram = lookupProgram(scope, d.program.name)
  if (!targetProgram) {
    throw new ElaborationError(
      `instance '${d.name}': program type '${d.program.name}' is not a nested program in scope. ` +
      `External program types (stdlib) require the loader to register them — defer to a later phase.`,
    )
  }
  const decl: InstanceDecl = {
    op: 'instanceDecl',
    name: d.name,
    type: targetProgram,
    typeArgs: [],
    inputs: [],
  }
  scope.instances.set(d.name, decl)
  return decl
}

/** Second-pass resolver: fill in the expression-shaped fields of an
 *  already-registered decl. The decl shell was created with placeholder
 *  values; this fills them in. Mutation of `resolved` is intentional —
 *  it's the same object held by reference in scope's maps. */
function resolveDeclExpressions(
  parsed: ParsedBodyDecl,
  resolved: BodyDecl,
  scope: Scope,
): void {
  if (parsed.op === 'regDecl' && resolved.op === 'regDecl') {
    resolved.init = resolveExpr(parsed.init, scope)
    return
  }
  if (parsed.op === 'delayDecl' && resolved.op === 'delayDecl') {
    resolved.update = resolveExpr(parsed.update, scope)
    resolved.init = resolveExpr(parsed.init, scope)
    return
  }
  if (parsed.op === 'instanceDecl' && resolved.op === 'instanceDecl') {
    resolveInstanceArgs(parsed, resolved, scope)
    return
  }
  if (parsed.op === 'paramDecl') return  // no expressions on paramDecl
  if (parsed.op === 'programDecl') return  // handled in pre-pass
  throw new ElaborationError(
    `internal: paired ${parsed.op} with ${resolved.op}`,
  )
}

function resolveInstanceArgs(
  parsed: ParsedInstanceDecl,
  resolved: InstanceDecl,
  scope: Scope,
): void {
  const targetProgram = resolved.type
  // Type args: resolve param NameRef → the target's TypeParamDecl.
  for (const entry of parsed.type_args ?? []) {
    const paramDecl = targetProgram.typeParams.find(p => p.name === entry.param.name)
    if (!paramDecl) {
      const expected = targetProgram.typeParams.map(p => p.name).join(', ') || '(none)'
      throw new ElaborationError(
        `instance '${resolved.name}': type-arg '${entry.param.name}' is not a declared type-param of '${targetProgram.name}' (have: ${expected})`,
      )
    }
    if (resolved.typeArgs.some(a => a.param === paramDecl)) {
      throw new ElaborationError(
        `instance '${resolved.name}': duplicate type-arg '${entry.param.name}'`,
      )
    }
    resolved.typeArgs.push({ param: paramDecl, value: entry.value })
  }
  // Inputs: resolve port NameRef → the target's InputDecl, value-expr resolved.
  for (const entry of parsed.inputs ?? []) {
    const portDecl = targetProgram.ports.inputs.find(p => p.name === entry.port.name)
    if (!portDecl) {
      const expected = targetProgram.ports.inputs.map(p => p.name).join(', ') || '(none)'
      throw new ElaborationError(
        `instance '${resolved.name}': input '${entry.port.name}' is not a declared port of '${targetProgram.name}' (have: ${expected})`,
      )
    }
    if (resolved.inputs.some(i => i.port === portDecl)) {
      throw new ElaborationError(
        `instance '${resolved.name}': duplicate input '${entry.port.name}'`,
      )
    }
    const value = resolveExpr(entry.value, scope)
    resolved.inputs.push({ port: portDecl, value })
  }
}

// ─────────────────────────────────────────────────────────────
// Body assigns
// ─────────────────────────────────────────────────────────────

function resolveAssign(a: ParsedBodyAssign, scope: Scope): BodyAssign {
  if (a.op === 'outputAssign') return resolveOutputAssign(a, scope)
  return resolveNextUpdate(a, scope)
}

function resolveOutputAssign(a: ParsedOutputAssign, scope: Scope): OutputAssign {
  let target: OutputAssign['target']
  if (a.name === 'dac.out') {
    target = { kind: 'dac' }
  } else {
    const out = scope.outputs.get(a.name)
    if (!out) {
      throw new ElaborationError(
        `outputAssign references unknown output port '${a.name}'`,
      )
    }
    target = out
  }
  return { op: 'outputAssign', target, expr: resolveExpr(a.expr, scope) }
}

function resolveNextUpdate(a: ParsedNextUpdate, scope: Scope): NextUpdate {
  const name = a.target.name
  const reg = scope.regs.get(name)
  if (reg) {
    return { op: 'nextUpdate', target: reg, expr: resolveExpr(a.expr, scope) }
  }
  const delay = scope.delays.get(name)
  if (delay) {
    return { op: 'nextUpdate', target: delay, expr: resolveExpr(a.expr, scope) }
  }
  throw new ElaborationError(
    `nextUpdate target '${name}' is not a declared reg or delay`,
  )
}

// ─────────────────────────────────────────────────────────────
// Expressions
// ─────────────────────────────────────────────────────────────

function resolveExpr(node: ParsedExprNode, scope: Scope): ResolvedExpr {
  if (typeof node === 'number' || typeof node === 'boolean') return node
  if (Array.isArray(node)) return node.map(n => resolveExpr(n, scope))
  return resolveOpNode(node, scope)
}

function resolveOpNode(node: ParsedExprOpNode, scope: Scope): ResolvedExprOpNode {
  // Discriminated-union switch on `op`. TypeScript narrows each branch
  // to its specific parsed-node type.
  switch (node.op) {
    case 'nameRef':   return resolveNameRef(node, scope)
    case 'binding':   return resolveParsedBinding(node, scope)
    case 'nestedOut': return resolveNestedOut(node, scope)
    case 'index':     return resolveIndex(node, scope)
    case 'call':      return resolveCall(node, scope)
    case 'tag':       return resolveTag(node, scope)
    case 'match':     return resolveMatch(node, scope)
    case 'let':       return resolveLet(node, scope)
    case 'fold':      return resolveFold(node, scope)
    case 'scan':      return resolveScan(node, scope)
    case 'generate':  return resolveGenerate(node, scope)
    case 'iterate':   return resolveIterate(node, scope)
    case 'chain':     return resolveChain(node, scope)
    case 'map2':      return resolveMap2(node, scope)
    case 'zipWith':   return resolveZipWith(node, scope)
    default:
      // Remaining branches are binary or unary ops by the discriminator.
      // BinaryOpNode and UnaryOpNode share the `args`-tuple shape; the
      // op tag selects between them.
      if (UNARY_OP_TAGS.has(node.op)) {
        return resolveUnary(node as ParsedUnary, scope)
      }
      return resolveBinary(node as ParsedBinary, scope)
  }
}

const UNARY_OP_TAGS: ReadonlySet<string> = new Set(['neg', 'not', 'bitNot'])

function resolveBinary(node: ParsedBinary, scope: Scope): BinaryOpNode {
  return {
    op: node.op,
    args: [resolveExpr(node.args[0], scope), resolveExpr(node.args[1], scope)],
  }
}

function resolveUnary(node: ParsedUnary, scope: Scope): UnaryOpNode {
  return {
    op: node.op as UnaryOpTag,
    args: [resolveExpr(node.args[0], scope)],
  }
}

function resolveNameRef(ref: ParsedNameRefNode, scope: Scope): ResolvedExprOpNode {
  const resolved = lookupValueRef(scope, ref.name)
  if (resolved) return resolved
  throw new ElaborationError(`unknown name '${ref.name}'`)
}

function resolveParsedBinding(node: ParsedBindingNode, scope: Scope): BindingRef {
  // The parser already determined this is bound; the elaborator confirms
  // the binder is in scope and links the ref to the decl.
  const binder = scope.binders.get(node.name)
  if (!binder) {
    throw new ElaborationError(
      `binding '${node.name}' is not in scope (parser said it was bound — likely a parser bug)`,
    )
  }
  return { op: 'bindingRef', decl: binder }
}

function resolveNestedOut(node: ParsedNestedOut, scope: Scope): NestedOut {
  const inst = scope.instances.get(node.ref.name)
  if (!inst) {
    throw new ElaborationError(
      `instance '${node.ref.name}' is not declared in this scope`,
    )
  }
  // node.output is NameRefNode | number; the parser preserves whichever form
  // the user wrote.
  const targetProgram = inst.type
  let output: OutputDecl | undefined
  if (typeof node.output === 'number') {
    output = targetProgram.ports.outputs[node.output]
  } else {
    output = targetProgram.ports.outputs.find(p => p.name === node.output.name)
  }
  if (!output) {
    const portList = targetProgram.ports.outputs.map(p => p.name).join(', ')
    const requested = typeof node.output === 'number' ? `index ${node.output}` : `'${node.output.name}'`
    throw new ElaborationError(
      `instance '${node.ref.name}': program '${targetProgram.name}' has no output ${requested} (have: ${portList})`,
    )
  }
  return { op: 'nestedOut', instance: inst, output }
}

function resolveIndex(node: ParsedIndex, scope: Scope): IndexNode {
  return {
    op: 'index',
    args: [resolveExpr(node.args[0], scope), resolveExpr(node.args[1], scope)],
  }
}

function resolveCall(node: ParsedCallNode, scope: Scope): ResolvedExprOpNode {
  // Generic call always has a NameRef callee from the parser (it's the
  // `f(args)` form where f was an ident). The elaborator either rewrites
  // to a builtin op, or rejects.
  if (!isParsedNameRef(node.callee)) {
    throw new ElaborationError(
      `unsupported call form: callee must be an identifier (no first-class function values yet)`,
    )
  }
  const fname = node.callee.name

  // Nullary sentinel calls
  if (NULLARY_CALLS.has(fname)) {
    if (node.args.length !== 0) {
      throw new ElaborationError(`'${fname}()' takes no arguments`)
    }
    if (fname === 'sample_rate') {
      const n: SampleRateNode = { op: 'sampleRate' }
      return n
    }
    const n: SampleIndexNode = { op: 'sampleIndex' }
    return n
  }

  // Unary builtins
  const unaryTag = UNARY_CALLS[fname]
  if (unaryTag) {
    if (node.args.length !== 1) {
      throw new ElaborationError(`'${fname}' takes 1 argument; got ${node.args.length}`)
    }
    const u: UnaryOpNode = { op: unaryTag, args: [resolveExpr(node.args[0], scope)] }
    return u
  }

  // Ternary builtins
  if (fname === 'clamp') {
    if (node.args.length !== 3) {
      throw new ElaborationError(`'clamp' takes 3 arguments (value, lo, hi); got ${node.args.length}`)
    }
    const c: ClampNode = {
      op: 'clamp',
      args: [
        resolveExpr(node.args[0], scope),
        resolveExpr(node.args[1], scope),
        resolveExpr(node.args[2], scope),
      ],
    }
    return c
  }
  if (fname === 'select') {
    if (node.args.length !== 3) {
      throw new ElaborationError(`'select' takes 3 arguments (cond, then, else); got ${node.args.length}`)
    }
    const s: SelectNode = {
      op: 'select',
      args: [
        resolveExpr(node.args[0], scope),
        resolveExpr(node.args[1], scope),
        resolveExpr(node.args[2], scope),
      ],
    }
    return s
  }

  throw new ElaborationError(
    `unknown function '${fname}'. The resolved IR has no escape hatch for unknown calls — ` +
    `add the builtin to the elaborator's registry, or use an instance declaration if it's a program type.`,
  )
}

function resolveTag(node: ParsedTag, scope: Scope): TagExpr {
  // Look up the variant in scope.variantOf (built when sum types were registered).
  const variantName = node.variant.name
  let variant: SumVariant | undefined
  let s: Scope | undefined = scope
  while (s) {
    const v = s.variantOf.get(variantName)
    if (v) { variant = v; break }
    s = s.parent
  }
  if (!variant) {
    throw new ElaborationError(`tag construction: unknown variant '${variantName}'`)
  }
  // Validate payload: every variant.payload field must be supplied;
  // no extras.
  const payload: TagExpr['payload'] = []
  const supplied = new Map<string, ResolvedExpr>()
  for (const entry of node.payload ?? []) {
    supplied.set(entry.field.name, resolveExpr(entry.value, scope))
  }
  for (const field of variant.payload) {
    const value = supplied.get(field.name)
    if (value === undefined) {
      throw new ElaborationError(
        `tag '${variantName}': missing payload field '${field.name}'`,
      )
    }
    payload.push({ field, value })
    supplied.delete(field.name)
  }
  if (supplied.size > 0) {
    const extras = [...supplied.keys()].join(', ')
    throw new ElaborationError(
      `tag '${variantName}': unknown payload field(s): ${extras}`,
    )
  }
  return { op: 'tag', variant, payload }
}

function resolveMatch(node: ParsedMatch, scope: Scope): MatchExpr {
  if (node.arms.length === 0) {
    throw new ElaborationError(`match expression has no arms`)
  }
  // Determine the sum type from the first arm; check all arms agree.
  const firstName = node.arms[0].variant.name
  let firstVariant: SumVariant | undefined
  let s: Scope | undefined = scope
  while (s) {
    firstVariant = s.variantOf.get(firstName)
    if (firstVariant) break
    s = s.parent
  }
  if (!firstVariant) {
    throw new ElaborationError(
      `match: unknown variant '${firstName}' in first arm`,
    )
  }
  const sumType = firstVariant.parent

  const seen = new Set<SumVariant>()
  const arms: MatchArm[] = []
  for (const a of node.arms) {
    const variant = sumType.variants.find(v => v.name === a.variant.name)
    if (!variant) {
      throw new ElaborationError(
        `match: variant '${a.variant.name}' is not a member of sum type '${sumType.name}'`,
      )
    }
    if (seen.has(variant)) {
      throw new ElaborationError(`match: duplicate arm for variant '${variant.name}'`)
    }
    seen.add(variant)

    // Build binders matching variant.payload; the parsed `bind` is
    // string | string[] | undefined. Empty payload arms must have no
    // binders; non-empty arms must bind every payload field.
    const bindNames = a.bind === undefined ? []
      : (Array.isArray(a.bind) ? a.bind : [a.bind])
    if (bindNames.length !== variant.payload.length) {
      throw new ElaborationError(
        `match arm '${variant.name}': expected ${variant.payload.length} binder(s) (one per payload field), got ${bindNames.length}`,
      )
    }
    const binders: BinderDecl[] = bindNames.map(name => ({ op: 'binderDecl', name }))
    // Push binders into scope, resolve body, pop.
    const body = withBinders(scope, binders, () => resolveExpr(a.body, scope))
    arms.push({ variant, binders, body })
  }

  // Exhaustiveness: every variant of sumType must have an arm.
  for (const v of sumType.variants) {
    if (!seen.has(v)) {
      throw new ElaborationError(
        `match on '${sumType.name}' is non-exhaustive: missing variant '${v.name}'`,
      )
    }
  }

  return {
    op: 'match',
    type: sumType,
    scrutinee: resolveExpr(node.scrutinee, scope),
    arms,
  }
}

function resolveLet(node: ParsedLetNode, scope: Scope): LetExpr {
  // Each binding's value evaluates in the OUTER scope (no let* — bindings
  // can't see siblings). Then all binders enter scope for the body.
  const binders: LetExpr['binders'] = []
  for (const [name, valueExpr] of Object.entries(node.bind)) {
    const binder: BinderDecl = { op: 'binderDecl', name }
    const value = resolveExpr(valueExpr, scope)
    binders.push({ binder, value })
  }
  const decls = binders.map(b => b.binder)
  const inResolved = withBinders(scope, decls, () => resolveExpr(node.in, scope))
  return { op: 'let', binders, in: inResolved }
}

function resolveFold(node: ParsedFold, scope: Scope): FoldExpr {
  const acc: BinderDecl = { op: 'binderDecl', name: node.acc_var }
  const elem: BinderDecl = { op: 'binderDecl', name: node.elem_var }
  const body = withBinders(scope, [acc, elem], () => resolveExpr(node.body, scope))
  return {
    op: 'fold',
    over: resolveExpr(node.over, scope),
    init: resolveExpr(node.init, scope),
    acc, elem, body,
  }
}

function resolveScan(node: ParsedScan, scope: Scope): ScanExpr {
  const acc: BinderDecl = { op: 'binderDecl', name: node.acc_var }
  const elem: BinderDecl = { op: 'binderDecl', name: node.elem_var }
  const body = withBinders(scope, [acc, elem], () => resolveExpr(node.body, scope))
  return {
    op: 'scan',
    over: resolveExpr(node.over, scope),
    init: resolveExpr(node.init, scope),
    acc, elem, body,
  }
}

function resolveGenerate(node: ParsedGenerate, scope: Scope): GenerateExpr {
  const iter: BinderDecl = { op: 'binderDecl', name: node.var }
  const body = withBinders(scope, [iter], () => resolveExpr(node.body, scope))
  return { op: 'generate', count: resolveExpr(node.count, scope), iter, body }
}

function resolveIterate(node: ParsedIterate, scope: Scope): IterateExpr {
  const iter: BinderDecl = { op: 'binderDecl', name: node.var }
  const body = withBinders(scope, [iter], () => resolveExpr(node.body, scope))
  return {
    op: 'iterate',
    count: resolveExpr(node.count, scope),
    init: resolveExpr(node.init, scope),
    iter, body,
  }
}

function resolveChain(node: ParsedChain, scope: Scope): ChainExpr {
  const iter: BinderDecl = { op: 'binderDecl', name: node.var }
  const body = withBinders(scope, [iter], () => resolveExpr(node.body, scope))
  return {
    op: 'chain',
    count: resolveExpr(node.count, scope),
    init: resolveExpr(node.init, scope),
    iter, body,
  }
}

function resolveMap2(node: ParsedMap2, scope: Scope): Map2Expr {
  const elem: BinderDecl = { op: 'binderDecl', name: node.elem_var }
  const body = withBinders(scope, [elem], () => resolveExpr(node.body, scope))
  return { op: 'map2', over: resolveExpr(node.over, scope), elem, body }
}

function resolveZipWith(node: ParsedZipWith, scope: Scope): ZipWithExpr {
  const x: BinderDecl = { op: 'binderDecl', name: node.x_var }
  const y: BinderDecl = { op: 'binderDecl', name: node.y_var }
  const body = withBinders(scope, [x, y], () => resolveExpr(node.body, scope))
  return {
    op: 'zipWith',
    a: resolveExpr(node.a, scope),
    b: resolveExpr(node.b, scope),
    x, y, body,
  }
}

// ─────────────────────────────────────────────────────────────
// Binder scope management
// ─────────────────────────────────────────────────────────────

function withBinders<T>(scope: Scope, binders: BinderDecl[], body: () => T): T {
  const prior: Array<{ name: string; was: BinderDecl | undefined }> = []
  for (const b of binders) {
    prior.push({ name: b.name, was: scope.binders.get(b.name) })
    scope.binders.set(b.name, b)
  }
  try {
    return body()
  } finally {
    for (const { name, was } of prior.reverse()) {
      if (was) scope.binders.set(name, was)
      else scope.binders.delete(name)
    }
  }
}

// ─────────────────────────────────────────────────────────────
// Type predicates over parsed nodes
// ─────────────────────────────────────────────────────────────

function isParsedNameRef(v: unknown): v is ParsedNameRefNode {
  return typeof v === 'object' && v !== null && !Array.isArray(v)
    && (v as { op?: unknown }).op === 'nameRef'
}

function isParsedProgramDecl(d: ParsedBodyDecl): d is ParsedProgramDecl {
  return d.op === 'programDecl'
}
