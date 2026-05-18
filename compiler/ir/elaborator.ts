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
  ParsedExpr,
  ParsedExprOp as ParsedExprOp,
  NameRef as ParsedNameRef,
  Program as ParsedProgram,
  Block as ParsedBlock,
  BodyDecl as ParsedBodyDecl,
  BodyAssign as ParsedBodyAssign,
  RegDecl as ParsedRegDecl,
  DelayDecl as ParsedDelayDecl,
  ParamDecl as ParsedParamDecl,
  InstanceDecl as ParsedInstanceDecl,
  ProgramDecl as ParsedProgramDecl,
  OutputAssign as ParsedOutputAssign,
  NextUpdate as ParsedNextUpdate,
  ProgramPort as ParsedProgramPort,
  PortTypeDecl as ParsedPortType,
  ShapeDim as ParsedShapeDim,
  TypeDef as ParsedTypeDef,
  StructTypeDef as ParsedStructTypeDef,
  SumTypeDef as ParsedSumTypeDef,
  AliasTypeDef as ParsedAliasTypeDef,
  StructField as ParsedStructField,
  Call as ParsedCall,
  Tag as ParsedTag,
  Match as ParsedMatch,
  Let as ParsedLet,
  Fold as ParsedFold,
  Scan as ParsedScan,
  Generate as ParsedGenerate,
  Iterate as ParsedIterate,
  Chain as ParsedChain,
  Map2 as ParsedMap2,
  ZipWith as ParsedZipWith,
  Index as ParsedIndex,
  NestedOut as ParsedNestedOut,
  Binding as ParsedBinding,
  BinaryOp as ParsedBinary,
  UnaryOp as ParsedUnary,
} from '../parse/nodes.js'
import type {
  ResolvedProgram, ResolvedBlock, ResolvedProgramPorts,
  ResolvedExpr, ResolvedExprOp,
  InputDecl, OutputDecl, TypeParamDecl,
  RegDecl, ParamDecl, InstanceDecl, ProgramDecl, BodyDecl,
  BodyAssign, OutputAssign,
  TypeDef, StructTypeDef, SumTypeDef, SumVariant, AliasTypeDef, StructField,
  PortType, ShapeDim, ScalarKind,
  BinderDecl,
  InputRef, RegRef, ParamRef, TypeParamRef, BindingRef,
  NestedOut,
  Tag, Match, MatchArm,
  Let,
  Fold, Scan, Generate, Iterate, Chain, Map2, ZipWith,
  Clamp, Select, Index,
  Zeros, ArraySet,
  BinaryOp, BinaryOpTag, UnaryOp, UnaryOpTag,
  SampleRate, SampleIndex,
} from './nodes.js'
import { ElaborationError } from './nodes.js'
import { findInstanceCycles } from './lowering/cycle_break.js'
import { CycleViolation, type CycleDiagnostic } from './elaboration_diagnostics.js'

const SCALAR_KINDS: ReadonlySet<string> = new Set(['float', 'int', 'bool'])
const SCALAR_ALIASES: ReadonlySet<string> = new Set([
  // bare scalars
  'float', 'int', 'bool',
  // common builtin port-type aliases that pass through to ScalarKind.
  // they are user-facing names for the raw scalar types — no separate
  // metadata, just the alias-to-base mapping.
  'signal', 'freq', 'unipolar', 'bipolar',
])

/** Builtin port-type aliases that map to a ScalarKind. */
const BUILTIN_TYPE_TO_SCALAR: Record<string, ScalarKind> = {
  float: 'float', int: 'int', bool: 'bool',
  signal: 'float', freq: 'float', unipolar: 'float', bipolar: 'float',
  phase:  'float',
}

/** Builtin nullary calls. Both snake_case and camelCase forms are
 *  recognized — stdlib (.trop) uses camelCase, older fixtures use
 *  snake_case. The elaborator is name-agnostic; the resolved IR carries
 *  the canonical (camelCase) op tag regardless of the surface form. */
const NULLARY_CALLS: ReadonlySet<string> = new Set([
  'sample_rate', 'sample_index',
  'sampleRate', 'sampleIndex',
])

/** Builtin unary function calls — surface name → resolved op tag. Both
 *  case conventions are accepted (see NULLARY_CALLS comment). */
const UNARY_CALLS: Record<string, UnaryOpTag> = {
  sqrt: 'sqrt', abs: 'abs', neg: 'neg',
  floor: 'floor', ceil: 'ceil', round: 'round',
  not: 'not',
  bit_not: 'bitNot', bitNot: 'bitNot',
  to_int: 'toInt', toInt: 'toInt',
  to_bool: 'toBool', toBool: 'toBool',
  to_float: 'toFloat', toFloat: 'toFloat',
  float_exponent: 'floatExponent', floatExponent: 'floatExponent',
}

/** Builtin binary function calls — surface name → resolved BinaryOp tag.
 *  These have call syntax (`ldexp(x, n)`) but the same shape as infix ops.
 *
 *  Note: `pow(x, y)` is intentionally absent. The stdlib `Pow` program
 *  (defined as `Exp(y * Log(x))`) is the canonical pow. The primitive
 *  op tag was a leak that bypassed the projection scheme; removed in
 *  the `kill-pow-primitive` PR. Anyone wanting power-of-two should use
 *  `ldexp(1, n)` (single fmul via IEEE-754 exponent injection); anyone
 *  wanting fractional exponents should instantiate `Pow`. */
const BINARY_CALLS: Record<string, BinaryOpTag> = {
  floor_div: 'floorDiv', floorDiv: 'floorDiv',
  ldexp: 'ldexp',
}

// ─────────────────────────────────────────────────────────────
// Scope
// ─────────────────────────────────────────────────────────────

interface Scope {
  inputs: Map<string, InputDecl>
  outputs: Map<string, OutputDecl>
  typeParams: Map<string, TypeParamDecl>
  /** Unified state-bearing decls. After Phase 0a, regs and former-delays
   *  share this single map; surface `delay` desugars at elaboration into
   *  a RegDecl with `update` populated. */
  regs: Map<string, RegDecl>
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
  /** Optional external program-type resolver. When set, instance decls
   *  naming a program not in any enclosing scope's `programs` map fall
   *  through to this. Inherited by nested program scopes. */
  resolveExternalProgram?: ExternalProgramResolver
}

function emptyScope(parent?: Scope): Scope {
  const scope: Scope = {
    inputs: new Map(),
    outputs: new Map(),
    typeParams: new Map(),
    regs: new Map(),
    params: new Map(),
    instances: new Map(),
    programs: new Map(),
    typeDefs: new Map(),
    variantOf: new Map(),
    binders: new Map(),
    parent,
  }
  // Nested programs inherit the resolver from their parent scope.
  if (parent?.resolveExternalProgram) {
    scope.resolveExternalProgram = parent.resolveExternalProgram
  }
  return scope
}

/** Look up a name across scope categories in a defined order. Used when
 *  resolving a NameRef in expression position — the position has a
 *  fixed semantic intent (a value-producing reference), and we try each
 *  applicable scope. */
function lookupValueRef(scope: Scope, name: string): ResolvedExprOp | null {
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
function resolveElement(scope: Scope, ref: ParsedNameRef): ScalarKind | AliasTypeDef {
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

/** Optional callback for resolving program-type names that aren't nested
 *  inside the program being elaborated. Mirrors `session.typeResolver`
 *  from `compiler/stdlib_loader.ts` — the elaborator stays a pure
 *  function but lets its caller stage cross-program lookups (e.g.
 *  elaborate stdlib in dependency order, feeding earlier results into
 *  later elaborations). Returns `undefined` for unknown names. */
export type ExternalProgramResolver = (name: string) => ResolvedProgram | undefined

/** Resolve a parsed program to a graph IR. The returned object carries
 *  declared inputs/outputs/type-params/type-defs as Decl objects, and
 *  every reference inside the body is a direct edge to one of those
 *  decls.
 *
 *  Optional `resolveExternalProgram`: when an `InstanceDecl` names a
 *  program type that isn't nested in this program's body, the elaborator
 *  consults the resolver. This is how stdlib elaboration works — sibling
 *  programs are elaborated first, then fed in via the resolver.
 *
 *  Post-Phase 4b: inter-instance cycles that don't pass through an
 *  explicit user register throw `CycleViolation`. */
export function elaborate(
  prog: ParsedProgram,
  resolveExternalProgram?: ExternalProgramResolver,
): ResolvedProgram {
  return elaborateProgram(prog, undefined, resolveExternalProgram)
}

/** Test-only escape hatch: elaborate without the strict-cycle-policy
 *  check. Used by trace_cycles.test.ts and friends to construct
 *  cyclic ResolvedPrograms for downstream pipeline testing (the
 *  cycle-break helper itself, decl-identity clone behavior under
 *  cycles, etc.). Production code paths must use `elaborate`. */
export function _elaborateForCyclicTest(
  prog: ParsedProgram,
  resolveExternalProgram?: ExternalProgramResolver,
): ResolvedProgram {
  return elaborateProgram(prog, undefined, resolveExternalProgram, { skipCycleCheck: true })
}

interface ElaborateInternalOpts {
  skipCycleCheck?: boolean
}

function elaborateProgram(
  prog: ParsedProgram,
  parent: Scope | undefined,
  resolveExternalProgram?: ExternalProgramResolver,
  internalOpts: ElaborateInternalOpts = {},
): ResolvedProgram {
  const scope = emptyScope(parent)
  scope.resolveExternalProgram = resolveExternalProgram

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
  const decls = registerBodyDecls(prog.body, scope, pairing, internalOpts)

  // 5. Resolve expressions inside body decls (init/update/instance inputs).
  for (const [parsed, resolved] of pairing) {
    resolveDeclExpressions(parsed, resolved, scope)
  }

  // 6. Resolve body assigns. NextUpdate assigns fold into their target
  //    reg's `update` field directly (A-canonical); resolveAssign
  //    returns null for those, which we filter out so the assigns array
  //    holds only structural assigns (OutputAssign today).
  const assigns: BodyAssign[] = []
  for (const a of prog.body.assigns ?? []) {
    const resolved = resolveAssign(a, scope)
    if (resolved !== null) assigns.push(resolved)
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

  // Phase 4b: strict cycle policy. Cycles in source code that don't
  // pass through an explicit user register throw `CycleViolation`.
  // The error message is port-detailed (Tier 2): names the cycle
  // members and the explicit `delay` statement the user could add to
  // break it. Tests that construct cyclic IR for downstream pipeline
  // exercise (the cycle-break helper, clone identity under cycles)
  // can opt out via the `_elaborateForCyclicTest` entry point.
  if (!internalOpts.skipCycleCheck) throwOnCycles(resolved)

  // Make this program visible to its containing scope (for sibling
  // nested programs) — caller registers the wrapping ProgramDecl.
  return resolved
}

function throwOnCycles(prog: ResolvedProgram): void {
  const cycles = findInstanceCycles(prog)
  if (cycles.length === 0) return
  // The cycle-break helper mutates the input, so we never call it
  // here. Build the suggested-fix snippet manually from the SCC.
  const diagnostics: CycleDiagnostic[] = cycles.map(scc => {
    const sortedScc = [...scc]
    const target = sortedScc[0]
    const suggestedFix =
      `Suggested fix: insert a 'delay' statement on one of '${target.name}'’s ` +
      `output ports to break the cycle explicitly. ` +
      `Example: 'delay ${target.name}_out_delayed = ${target.name}.<port> init 0' ` +
      `and route cycle members from ${target.name}_out_delayed instead.`
    return {
      kind: 'cycle',
      scc: sortedScc,
      programName: prog.name,
      suggestedFix,
    }
  })
  throw new CycleViolation(diagnostics)
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
  return decl
}

function resolveOutputPort(spec: ParsedProgramPort, scope: Scope): OutputDecl {
  if (typeof spec === 'string') {
    return { op: 'outputDecl', name: spec }
  }
  const decl: OutputDecl = { op: 'outputDecl', name: spec.name }
  if (spec.type !== undefined) decl.type = resolvePortType(spec.type, scope)
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
  // d is a NameRef in shape position — must resolve to a TypeParamDecl.
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
  body: ParsedBlock,
  scope: Scope,
  pairing: Map<ParsedBodyDecl, BodyDecl>,
  internalOpts: ElaborateInternalOpts = {},
): BodyDecl[] {
  const out: BodyDecl[] = []
  // Programs first: nested sub-programs need to be resolved before any
  // sibling instance decls reference them.
  for (const d of body.decls ?? []) {
    if (isParsedProgramDecl(d)) {
      const inner = elaborateProgram(d.program, scope, undefined, internalOpts)
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

/** Parsed `delay name = u init v` is surface sugar for a RegDecl with
 *  `update: u` and `init: v`. Both reg-class names (parsed regDecl and
 *  parsed delayDecl) land in the same unified scope.regs map. The
 *  `init` and `update` fields are populated during the second-pass
 *  expression resolution (resolveDeclExpressions). */
function registerDelayDecl(d: ParsedDelayDecl, scope: Scope): RegDecl {
  if (scope.regs.has(d.name)) {
    throw new ElaborationError(`duplicate reg/delay '${d.name}'`)
  }
  // init resolved later; update placeholder marks "expects update from
  // resolveDeclExpressions" but the placeholder is overwritten before
  // any consumer sees it. We avoid using `undefined` here so a stray
  // next-update can still detect the conflict (delay-form already
  // commits to having an update on the decl).
  const decl: RegDecl = { op: 'regDecl', name: d.name, init: 0, update: 0 }
  scope.regs.set(d.name, decl)
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
  // Resolve program type. First try nested programs visible in scope
  // (and any enclosing scope), then fall through to the external
  // resolver if one was provided.
  let targetProgram = lookupProgram(scope, d.program.name)
  if (!targetProgram) {
    const resolver = findResolver(scope)
    if (resolver) {
      const external = resolver(d.program.name)
      if (external) targetProgram = external
    }
  }
  if (!targetProgram) {
    throw new ElaborationError(
      `instance '${d.name}': program type '${d.program.name}' is not a nested program in scope ` +
      `and no external resolver provided it. Pass an ExternalProgramResolver to elaborate() ` +
      `to resolve cross-program references (e.g. stdlib types).`,
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

/** Walk up the scope chain to find the outermost (root) resolver. The
 *  resolver propagates from elaborate()'s caller to every enclosed scope
 *  via `emptyScope`, so the root scope's value is the canonical one. */
function findResolver(scope: Scope): ExternalProgramResolver | undefined {
  let s: Scope | undefined = scope
  while (s) {
    if (s.resolveExternalProgram) return s.resolveExternalProgram
    s = s.parent
  }
  return undefined
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
    // resolved.update intentionally not set here — a later NextUpdate
    // assign (resolveNextUpdate) folds onto the decl if present.
    return
  }
  if (parsed.op === 'delayDecl' && resolved.op === 'regDecl') {
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

/** Returns either an OutputAssign body-assign, or null if the parsed
 *  assign was a `next x = e` that's been folded into the target reg's
 *  `update` field as an A-canonical side effect. The caller filters
 *  nulls out of the assigns array. */
function resolveAssign(a: ParsedBodyAssign, scope: Scope): BodyAssign | null {
  if (a.op === 'outputAssign') return resolveOutputAssign(a, scope)
  foldNextUpdateIntoDecl(a, scope)
  return null
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

/** A-canonical normalization: `next x = e` folds into the target reg's
 *  `update` field directly. The resolved IR doesn't carry NextUpdate
 *  as a body-assign — every reg's full specification lives on its decl. */
function foldNextUpdateIntoDecl(a: ParsedNextUpdate, scope: Scope): void {
  const name = a.target.name
  const reg = scope.regs.get(name)
  if (!reg) {
    throw new ElaborationError(
      `next-update target '${name}' is not a declared reg or delay`,
    )
  }
  if (reg.update !== undefined) {
    throw new ElaborationError(
      `duplicate update for reg '${name}' (already set by ` +
      `decl-side update or earlier next-update)`,
    )
  }
  reg.update = resolveExpr(a.expr, scope)
}

// ─────────────────────────────────────────────────────────────
// Expressions
// ─────────────────────────────────────────────────────────────

function resolveExpr(node: ParsedExpr, scope: Scope): ResolvedExpr {
  if (typeof node === 'number' || typeof node === 'boolean') return node
  if (Array.isArray(node)) return node.map(n => resolveExpr(n, scope))
  return resolveOpNode(node, scope)
}

function resolveOpNode(node: ParsedExprOp, scope: Scope): ResolvedExprOp {
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
      // BinaryOp and UnaryOp share the `args`-tuple shape; the
      // op tag selects between them.
      if (UNARY_OP_TAGS.has(node.op)) {
        return resolveUnary(node as ParsedUnary, scope)
      }
      return resolveBinary(node as ParsedBinary, scope)
  }
}

const UNARY_OP_TAGS: ReadonlySet<string> = new Set(['neg', 'not', 'bitNot'])

function resolveBinary(node: ParsedBinary, scope: Scope): BinaryOp {
  return {
    op: node.op,
    args: [resolveExpr(node.args[0], scope), resolveExpr(node.args[1], scope)],
  }
}

function resolveUnary(node: ParsedUnary, scope: Scope): UnaryOp {
  return {
    op: node.op as UnaryOpTag,
    args: [resolveExpr(node.args[0], scope)],
  }
}

function resolveNameRef(ref: ParsedNameRef, scope: Scope): ResolvedExprOp {
  const resolved = lookupValueRef(scope, ref.name)
  if (resolved) return resolved
  throw new ElaborationError(`unknown name '${ref.name}'`)
}

function resolveParsedBinding(node: ParsedBinding, scope: Scope): BindingRef {
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
  // node.output is NameRef | number; the parser preserves whichever form
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

function resolveIndex(node: ParsedIndex, scope: Scope): Index {
  return {
    op: 'index',
    args: [resolveExpr(node.args[0], scope), resolveExpr(node.args[1], scope)],
  }
}

function resolveCall(node: ParsedCall, scope: Scope): ResolvedExprOp {
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
    if (fname === 'sample_rate' || fname === 'sampleRate') {
      const n: SampleRate = { op: 'sampleRate' }
      return n
    }
    const n: SampleIndex = { op: 'sampleIndex' }
    return n
  }

  // Unary builtins
  const unaryTag = UNARY_CALLS[fname]
  if (unaryTag) {
    if (node.args.length !== 1) {
      throw new ElaborationError(`'${fname}' takes 1 argument; got ${node.args.length}`)
    }
    const u: UnaryOp = { op: unaryTag, args: [resolveExpr(node.args[0], scope)] }
    return u
  }

  // Binary builtins (call syntax, same shape as infix BinaryOp)
  const binaryTag = BINARY_CALLS[fname]
  if (binaryTag) {
    if (node.args.length !== 2) {
      throw new ElaborationError(`'${fname}' takes 2 arguments; got ${node.args.length}`)
    }
    const b: BinaryOp = {
      op: binaryTag,
      args: [resolveExpr(node.args[0], scope), resolveExpr(node.args[1], scope)],
    }
    return b
  }

  // Array ops — produce array-shaped values; lowered to scalar primitives
  // by array_lower (Phase C6). The elaborator just constructs the node.
  if (fname === 'zeros') {
    if (node.args.length !== 1) {
      throw new ElaborationError(`'zeros' takes 1 argument (count); got ${node.args.length}`)
    }
    const z: Zeros = { op: 'zeros', count: resolveExpr(node.args[0], scope) }
    return z
  }
  if (fname === 'arraySet' || fname === 'array_set') {
    if (node.args.length !== 3) {
      throw new ElaborationError(`'${fname}' takes 3 arguments (arr, idx, value); got ${node.args.length}`)
    }
    const a: ArraySet = {
      op: 'arraySet',
      args: [
        resolveExpr(node.args[0], scope),
        resolveExpr(node.args[1], scope),
        resolveExpr(node.args[2], scope),
      ],
    }
    return a
  }

  // Ternary builtins
  if (fname === 'clamp') {
    if (node.args.length !== 3) {
      throw new ElaborationError(`'clamp' takes 3 arguments (value, lo, hi); got ${node.args.length}`)
    }
    const c: Clamp = {
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
    const s: Select = {
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

function resolveTag(node: ParsedTag, scope: Scope): Tag {
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
  const payload: Tag['payload'] = []
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

function resolveMatch(node: ParsedMatch, scope: Scope): Match {
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

    // Build binders matching variant.payload by field name. Empty
    // payload arms must have no binders; non-empty arms must bind every
    // payload field exactly once. Binders are emitted in variant.payload
    // declaration order (the order the IR consumer expects).
    if (a.binds.length !== variant.payload.length) {
      throw new ElaborationError(
        `match arm '${variant.name}': expected ${variant.payload.length} binder(s) (one per payload field), got ${a.binds.length}`,
      )
    }
    const bindByField = new Map<string, string>()
    for (const b of a.binds) {
      if (bindByField.has(b.field.name)) {
        throw new ElaborationError(
          `match arm '${variant.name}': duplicate pattern field '${b.field.name}'`,
        )
      }
      bindByField.set(b.field.name, b.bind)
    }
    const binders: BinderDecl[] = variant.payload.map(field => {
      const bindName = bindByField.get(field.name)
      if (bindName === undefined) {
        throw new ElaborationError(
          `match arm '${variant.name}': missing pattern binding for payload field '${field.name}'`,
        )
      }
      bindByField.delete(field.name)
      return { op: 'binderDecl', name: bindName }
    })
    if (bindByField.size > 0) {
      const extras = [...bindByField.keys()].join(', ')
      throw new ElaborationError(
        `match arm '${variant.name}': unknown pattern field(s): ${extras}`,
      )
    }
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

function resolveLet(node: ParsedLet, scope: Scope): Let {
  // Sequential `let*` semantics: each binder's value is resolved in a
  // scope that already contains the prior binders. Surface stdlib relies
  // on this — programs like Tanh write `let { c: clamp(...); c2: c * c }`
  // expecting `c2` to see `c`. This matches lower.ts's lowerLet (sibling
  // pass uses the same fix; cross-pass behavior must agree).
  //
  // Binders enter scope one at a time as we walk the entries; the body
  // sees all of them. We snapshot prior bindings so we can restore them
  // (mirroring withBinders) — this matters when the parent scope already
  // has a binder with the same name (shadowing).
  const binders: Let['binders'] = []
  const prior: Array<{ name: string; was: BinderDecl | undefined }> = []
  try {
    for (const [name, valueExpr] of Object.entries(node.bind)) {
      const binder: BinderDecl = { op: 'binderDecl', name }
      const value = resolveExpr(valueExpr, scope)
      binders.push({ binder, value })
      // After resolving the value, push this binder so subsequent
      // entries (and ultimately the body) can see it.
      prior.push({ name, was: scope.binders.get(name) })
      scope.binders.set(name, binder)
    }
    const inResolved = resolveExpr(node.in, scope)
    return { op: 'let', binders, in: inResolved }
  } finally {
    for (const { name, was } of prior.reverse()) {
      if (was) scope.binders.set(name, was)
      else scope.binders.delete(name)
    }
  }
}

function resolveFold(node: ParsedFold, scope: Scope): Fold {
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

function resolveScan(node: ParsedScan, scope: Scope): Scan {
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

function resolveGenerate(node: ParsedGenerate, scope: Scope): Generate {
  const iter: BinderDecl = { op: 'binderDecl', name: node.var }
  const body = withBinders(scope, [iter], () => resolveExpr(node.body, scope))
  return { op: 'generate', count: resolveExpr(node.count, scope), iter, body }
}

function resolveIterate(node: ParsedIterate, scope: Scope): Iterate {
  const iter: BinderDecl = { op: 'binderDecl', name: node.var }
  const body = withBinders(scope, [iter], () => resolveExpr(node.body, scope))
  return {
    op: 'iterate',
    count: resolveExpr(node.count, scope),
    init: resolveExpr(node.init, scope),
    iter, body,
  }
}

function resolveChain(node: ParsedChain, scope: Scope): Chain {
  const iter: BinderDecl = { op: 'binderDecl', name: node.var }
  const body = withBinders(scope, [iter], () => resolveExpr(node.body, scope))
  return {
    op: 'chain',
    count: resolveExpr(node.count, scope),
    init: resolveExpr(node.init, scope),
    iter, body,
  }
}

function resolveMap2(node: ParsedMap2, scope: Scope): Map2 {
  const elem: BinderDecl = { op: 'binderDecl', name: node.elem_var }
  const body = withBinders(scope, [elem], () => resolveExpr(node.body, scope))
  return { op: 'map2', over: resolveExpr(node.over, scope), elem, body }
}

function resolveZipWith(node: ParsedZipWith, scope: Scope): ZipWith {
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

function isParsedNameRef(v: unknown): v is ParsedNameRef {
  return typeof v === 'object' && v !== null && !Array.isArray(v)
    && (v as { op?: unknown }).op === 'nameRef'
}

function isParsedProgramDecl(d: ParsedBodyDecl): d is ParsedProgramDecl {
  return d.op === 'programDecl'
}
