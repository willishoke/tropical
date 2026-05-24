/**
 * compiler/ir/nodes.ts — resolved-phase IR types.
 *
 * The elaborator (`compiler/ir/elaborator.ts`) consumes a parsed tree
 * (`compiler/parse/nodes.ts`) and produces values of the types defined
 * here. Where the parsed tree had NameRef placeholders, the resolved
 * tree has direct decl references — every reference is a graph edge.
 *
 * Categorical shape
 * -----------------
 * Decls (introduction sites): InputDecl, OutputDecl, RegDecl, ParamDecl,
 *   TypeParamDecl, InstanceDecl, ProgramDecl, BinderDecl, plus the
 *   sum-type members (SumTypeDef, SumVariant, StructTypeDef,
 *   StructField, AliasTypeDef). Each carries an identity string `name`.
 *
 * Refs (uses): InputRef, RegRef, ParamRef, TypeParamRef, BindingRef.
 *   Each holds an `idx` — a branded integer identifying the referent.
 *   Position-indexed refs (RegRef, InputRef, ParamRef, OutputRef-in-
 *   assigns, TypeParamRef, NestedOut) index into the enclosing
 *   program's typed decl tables. BindingRef carries a unique-per-
 *   program `BinderIdx` minted by the elaborator at binder-creation
 *   time — same ID means same binder, different ID means different
 *   binder, regardless of name (shadowing is structurally impossible).
 *
 * Bridges between term-and-type levels: NestedOut ties an instance ref
 * to a specific output port of its program type; ResolvedTagNode and
 * ResolvedMatchNode tie expressions to sum-type variants.
 *
 * Graph property: the resolved tree admits cycles. A reg's `update`
 * may transitively reference its own register; an instance's input may
 * reference a value that depends on the same instance via feedback.
 *
 * Strings: `name` on every Decl is an identity string (the user's chosen
 * label). It is NOT a reference. There are no other strings in the
 * resolved IR.
 *
 * No `ResolvedCallNode`: builtin function calls are resolved to their
 * structured op; unknown calls are an elaboration error. User-defined
 * functions are not a tropical feature today; if they become one, they
 * earn their own resolved shape with a real function-decl reference.
 */

// ─────────────────────────────────────────────────────────────
// Identity strings — primitives that aren't references
// ─────────────────────────────────────────────────────────────

/** Permitted scalar element types. The resolved IR uses an enum literal,
 *  not a `NameRef` — these are language primitives, not decls. */
export type ScalarKind = 'float' | 'int' | 'bool'

// ─────────────────────────────────────────────────────────────
// Type-defs (struct / sum / alias)
// ─────────────────────────────────────────────────────────────

/** A field of a struct or a payload-field of a sum variant. The `type`
 *  is either a primitive scalar kind or — for alias types — a reference
 *  to an AliasTypeDef. */
export interface StructField {
  op: 'structField'
  name: string
  type: ScalarKind | AliasTypeDef
}

export interface StructTypeDef {
  op: 'structTypeDef'
  name: string
  fields: StructField[]
}

/** A single variant of a sum type. Carries a back-pointer to its parent
 *  SumTypeDef so consumers can navigate variant → type without a
 *  registry lookup. The cycle (decl ↔ variant) is fine; graphs allow it. */
export interface SumVariant {
  op: 'sumVariant'
  name: string
  payload: StructField[]
  parent: SumTypeDef
}

export interface SumTypeDef {
  op: 'sumTypeDef'
  name: string
  variants: SumVariant[]
}

export interface AliasTypeDef {
  op: 'aliasTypeDef'
  name: string
  base: ScalarKind
}

export type TypeDef = StructTypeDef | SumTypeDef | AliasTypeDef

// ─────────────────────────────────────────────────────────────
// Port types — the shape of a value flowing on a port
// ─────────────────────────────────────────────────────────────

/** A compile-time array shape dim: literal integer or a TypeParamDecl. */
export type ShapeDim = number | TypeParamDecl

/** Resolved port type: a primitive scalar kind, an alias decl, or an
 *  array of an element type with a shape. */
export type PortType =
  | { kind: 'scalar'; scalar: ScalarKind }
  | { kind: 'alias'; alias: AliasTypeDef }
  | { kind: 'array'; element: ScalarKind | AliasTypeDef; shape: ShapeDim[] }

// ─────────────────────────────────────────────────────────────
// Program-header decls (inputs, outputs, type-params)
// ─────────────────────────────────────────────────────────────

export interface InputDecl {
  op: 'inputDecl'
  name: string
  type?: PortType
  default?: ResolvedExpr
}

export interface OutputDecl {
  op: 'outputDecl'
  name: string
  type?: PortType
}

export interface TypeParamDecl {
  op: 'typeParamDecl'
  name: string
  default?: number
}

// ─────────────────────────────────────────────────────────────
// Body decls — names introduced in a program body
// ─────────────────────────────────────────────────────────────

/** Provenance tag set by `inlineInstances:liftClonedBody` when a reg
 *  was lifted from a sub-instance. The value is the originating
 *  session-level instance name. Used by post-strata passes that need
 *  to identify decls by their lineage without parsing the renamed
 *  `${instance}_${innerName}` prefix string. */
export type LiftedFrom = string

/** The single state-bearing IR primitive. Carries an init expression
 *  (evaluated at sample 0) and an optional update expression (evaluated
 *  at each subsequent sample to produce the next value). When `update`
 *  is undefined, the reg holds its current value.
 *
 *  A-canonical shape: the elaborator folds parsed `delay name = u init v`
 *  into `RegDecl { init: v, update: u }`, and folds parsed
 *  `next x = e` body-assigns into the corresponding RegDecl's `update`
 *  field. The resolved IR therefore has one canonical shape per concept:
 *  every reg's full specification lives on the decl itself. */
export interface RegDecl {
  op: 'regDecl'
  name: string
  init: ResolvedExpr
  update?: ResolvedExpr
  type?: ScalarKind | AliasTypeDef
  /** Optional provenance tag — set when the decl was lifted from a
   *  sub-instance during `inlineInstances`. */
  _liftedFrom?: LiftedFrom
}

export interface ParamDecl {
  op: 'paramDecl'
  name: string
  value?: number
}

export interface InstanceDecl {
  op: 'instanceDecl'
  name: string
  /** Pointer to the instance's program type. Retained during Phase 4a
   *  of issue #156 alongside the new `typeKey` field for dual-read;
   *  Phase 4b drops this pointer and the `typeKey` becomes the sole
   *  identity. The Bubble fix from PR #158 (Phase 3, topological
   *  registry build) ensures the pointer is set to the canonical
   *  strata-processed program at construction time. */
  type: ResolvedProgram
  /** Lookup key for the instance's program type in the enclosing
   *  program's `programRegistry`. During Phase 4a this is the parallel
   *  rep of `.type`; the cross-check
   *  `tests/equiv/registry_vs_pointer.test.ts` asserts they always
   *  agree. Phase 4b drops `.type` and reads exclusively through this
   *  key. Value convention: the target program's `name` (matches the
   *  existing `session.resolvedRegistry` keying). */
  typeKey: ProgramKey
  /** Type-arg bindings, by position in the target's `typeParams[]`. */
  typeArgs: Array<{ param: TypeParamIdx; value: number }>
  /** Input-wire bindings, by position in the target's `ports.inputs[]`. */
  inputs: Array<{ port: InputIdx; value: ResolvedExpr }>
}

/** Branded string key for a `ResolvedProgram` in a `programRegistry`.
 *  Convention: equals the target program's `name`. */
declare const __program_key_brand: unique symbol
export type ProgramKey = string & { readonly [__program_key_brand]: 'ProgramKey' }
export const programKey = (s: string): ProgramKey => s as ProgramKey

/** A nested `program` declaration introduces a program type into the
 *  outer's body scope. The `program` field is the resolved nested program
 *  itself; instance-decl references use its InputDecls/OutputDecls etc. */
export interface ProgramDecl {
  op: 'programDecl'
  name: string
  program: ResolvedProgram
}

export type BodyDecl =
  | RegDecl
  | ParamDecl
  | InstanceDecl
  | ProgramDecl

// ─────────────────────────────────────────────────────────────
// Body assigns — wires pinning a value to a port
// ─────────────────────────────────────────────────────────────

export interface OutputAssign {
  op: 'outputAssign'
  /** Either an OutputIdx into this program's `ports.outputs[]`, or the
   *  special `'dac'` boundary leaf for top-level patches that wire to
   *  the DAC. Migrated from `OutputDecl` pointer alongside the rest of
   *  the global-ref de Bruijn levels migration. */
  target: OutputIdx | { kind: 'dac' }
  expr: ResolvedExpr
}

/** NextUpdate exists only as a transitional surface-IR concept; the
 *  elaborator folds it into `RegDecl.update` and the resolved IR no
 *  longer carries it as a body-assign. Kept in the type union for
 *  legacy callers that construct it directly; future cleanup may
 *  remove it from BodyAssign entirely. */
export interface NextUpdate {
  op: 'nextUpdate'
  target: RegDecl
  expr: ResolvedExpr
}

export type BodyAssign = OutputAssign

// ─────────────────────────────────────────────────────────────
// Binders — anonymous names introduced by let / combinators / match arms
// ─────────────────────────────────────────────────────────────

/** A single anonymous binder. The parent node (Let, Fold, etc.,
 *  or MatchArm) determines the binder's role.
 *
 *  `idx` is a unique-per-program integer ID minted by the elaborator;
 *  BindingRefs use this idx (not pointer identity) to refer back to a
 *  binder. `name` is an identity string — the user's chosen label,
 *  retained for diagnostics; it is NOT a reference and is not used for
 *  lookup post-elaboration.
 *
 *  IDs are stable across rewrites that preserve the binder's structural
 *  role; passes that lift expressions across program boundaries
 *  (`inline_instances` lifting sub-program decls into a parent) shift
 *  binder IDs by an offset, parallel to how reg/param/instance indices
 *  are shifted. */
export interface BinderDecl {
  op: 'binderDecl'
  name: string
  idx: BinderIdx
}

// ─────────────────────────────────────────────────────────────
// De Bruijn levels — branded indices into program-level tables
// ─────────────────────────────────────────────────────────────

/** Branded position-indices. Position in the table IS the decl's
 *  identity. Stable across rewrites (an index doesn't shift just
 *  because a sibling decl was reshaped).
 *
 *  - RegIdx / ParamIdx / InstanceIdx index this program's
 *    `regs[]` / `params[]` / `instances[]` tables.
 *  - InputIdx / OutputIdx index `ports.inputs[]` / `ports.outputs[]`.
 *    Where they appear on a cross-program reference (InstanceDecl's
 *    typeArgs.param / inputs.port, NestedOut.output) they're indices
 *    into the TARGET program's tables — resolution requires the
 *    target program in hand.
 *  - TypeParamIdx indexes `typeParams[]`.
 *  - BinderIdx is a unique-per-program ID minted by the elaborator
 *    for each `let` / combinator / match-arm binder. Unlike the other
 *    indices, BinderIdx is NOT a position into a program-level
 *    table — binders live nested inside combinator-body subexpressions,
 *    so a flat table would have to walk the whole IR to populate.
 *    The ID is what `BindingRef.idx` carries; the BinderDecl itself
 *    is reachable via the combinator that introduced it. Cross-program
 *    lifting (`inline_instances`) shifts BinderIdx by an offset
 *    parallel to RegIdx/ParamIdx/InstanceIdx shifting; substitution
 *    passes (`array_lower`, `sum_lower`) key their substitution maps
 *    by BinderIdx. */
declare const __ir_idx_brand: unique symbol
export type IrIdx<B extends string> = number & { readonly [__ir_idx_brand]: B }

export type RegIdx       = IrIdx<'RegIdx'>
export type InputIdx     = IrIdx<'InputIdx'>
export type OutputIdx    = IrIdx<'OutputIdx'>
export type ParamIdx     = IrIdx<'ParamIdx'>
export type InstanceIdx  = IrIdx<'InstanceIdx'>
export type TypeParamIdx = IrIdx<'TypeParamIdx'>
export type BinderIdx    = IrIdx<'BinderIdx'>

/** Brand-applying constructors. Use these everywhere indices are
 *  built; never `as RegIdx` casts at call sites. */
export const regIdx       = (n: number): RegIdx       => n as RegIdx
export const inputIdx     = (n: number): InputIdx     => n as InputIdx
export const outputIdx    = (n: number): OutputIdx    => n as OutputIdx
export const paramIdx     = (n: number): ParamIdx     => n as ParamIdx
export const instanceIdx  = (n: number): InstanceIdx  => n as InstanceIdx
export const typeParamIdx = (n: number): TypeParamIdx => n as TypeParamIdx
export const binderIdx    = (n: number): BinderIdx    => n as BinderIdx

// ─────────────────────────────────────────────────────────────
// Refs — uses of decls by de Bruijn level
// ─────────────────────────────────────────────────────────────

export interface InputRef     { op: 'inputRef';     idx: InputIdx }
export interface RegRef       { op: 'regRef';       idx: RegIdx }
export interface ParamRef     { op: 'paramRef';     idx: ParamIdx }
export interface TypeParamRef { op: 'typeParamRef'; idx: TypeParamIdx }
export interface BindingRef   { op: 'bindingRef';   idx: BinderIdx }

/** Dotted port reference, indexed: parent's instance position +
 *  the output port position inside the instance's program type. To
 *  resolve: instance via `prog.instances[instance]`, then read
 *  `instance.type.ports.outputs[output]`. */
export interface NestedOut {
  op: 'nestedOut'
  instance: InstanceIdx
  output:   OutputIdx
}

// ─────────────────────────────────────────────────────────────
// Sentinel leaves — semantic primitives, no decl
// ─────────────────────────────────────────────────────────────

export interface SampleRate  { op: 'sampleRate' }
export interface SampleIndex { op: 'sampleIndex' }

// ─────────────────────────────────────────────────────────────
// Builtin op shapes — same as parsed but with resolved children
// ─────────────────────────────────────────────────────────────

export type BinaryOpTag =
  | 'add' | 'sub' | 'mul' | 'div' | 'mod'
  | 'lt' | 'lte' | 'gt' | 'gte' | 'eq' | 'neq'
  | 'and' | 'or'
  | 'bitAnd' | 'bitOr' | 'bitXor' | 'lshift' | 'rshift'
  // Numeric builtins surfaced as `f(a, b)` in source; same arity/shape as
  // the infix ops above, so they share BinaryOp rather than earning
  // their own resolved-IR node types.
  | 'floorDiv' | 'ldexp'

export interface BinaryOp {
  op: BinaryOpTag
  args: [ResolvedExpr, ResolvedExpr]
}

export type UnaryOpTag =
  | 'neg' | 'not' | 'bitNot'
  | 'sqrt' | 'abs' | 'floor' | 'ceil' | 'round'
  | 'floatExponent' | 'toInt' | 'toBool' | 'toFloat'

export interface UnaryOp {
  op: UnaryOpTag
  args: [ResolvedExpr]
}

/** `clamp(value, lo, hi)` — explicit user-written bound enforcement. */
export interface Clamp {
  op: 'clamp'
  args: [ResolvedExpr, ResolvedExpr, ResolvedExpr]
}

/** `select(cond, then, else)` — value-level if. */
export interface Select {
  op: 'select'
  args: [ResolvedExpr, ResolvedExpr, ResolvedExpr]
}

/** `index(arr, i)` — array element access. */
export interface Index {
  op: 'index'
  args: [ResolvedExpr, ResolvedExpr]
}

/** `zeros(count)` — array constructor producing `count` zero elements.
 *  An array op: array_lower (C6) lowers it to scalar primitives. */
export interface Zeros {
  op: 'zeros'
  count: ResolvedExpr
}

/** `arraySet(arr, idx, value)` — non-mutating "set the i-th element".
 *  An array op: array_lower (C6) lowers it. */
export interface ArraySet {
  op: 'arraySet'
  args: [ResolvedExpr, ResolvedExpr, ResolvedExpr]
}

// ─────────────────────────────────────────────────────────────
// Combinators — each one carries its binder declarations directly
// ─────────────────────────────────────────────────────────────

export interface Fold {
  op: 'fold'
  over: ResolvedExpr
  init: ResolvedExpr
  acc: BinderDecl
  elem: BinderDecl
  body: ResolvedExpr
}

export interface Scan {
  op: 'scan'
  over: ResolvedExpr
  init: ResolvedExpr
  acc: BinderDecl
  elem: BinderDecl
  body: ResolvedExpr
}

export interface Generate {
  op: 'generate'
  count: ResolvedExpr
  iter: BinderDecl
  body: ResolvedExpr
}

export interface Iterate {
  op: 'iterate'
  count: ResolvedExpr
  init: ResolvedExpr
  iter: BinderDecl
  body: ResolvedExpr
}

export interface Chain {
  op: 'chain'
  count: ResolvedExpr
  init: ResolvedExpr
  iter: BinderDecl
  body: ResolvedExpr
}

export interface Map2 {
  op: 'map2'
  over: ResolvedExpr
  elem: BinderDecl
  body: ResolvedExpr
}

export interface ZipWith {
  op: 'zipWith'
  a: ResolvedExpr
  b: ResolvedExpr
  x: BinderDecl
  y: BinderDecl
  body: ResolvedExpr
}

// ─────────────────────────────────────────────────────────────
// Let — multiple binder/value pairs, body sees them all
// ─────────────────────────────────────────────────────────────

export interface Let {
  op: 'let'
  /** Each entry introduces one binder. The `value` is evaluated in the
   *  enclosing scope (no let* semantics inside this single Let — bindings
   *  don't see siblings). Order is preserved for stable output. */
  binders: Array<{ binder: BinderDecl; value: ResolvedExpr }>
  in: ResolvedExpr
}

// ─────────────────────────────────────────────────────────────
// ADT expressions — tag construction + match elimination
// ─────────────────────────────────────────────────────────────

/** A sum-type variant constructor. `variant` carries a back-pointer to
 *  its `parent` SumTypeDef, so the type name is derivable without a
 *  registry lookup. */
export interface Tag {
  op: 'tag'
  variant: SumVariant
  /** Each entry is a payload field (StructField from variant.payload)
   *  paired with the value-expression bound to it. The elaborator
   *  validates that every variant.payload field has a matching entry. */
  payload: Array<{ field: StructField; value: ResolvedExpr }>
}

/** A single match arm. `binders` is an ordered list of binder decls
 *  (one per payload field, matching variant.payload order). The arm
 *  body sees these binders in scope. */
export interface MatchArm {
  variant: SumVariant
  binders: BinderDecl[]
  body: ResolvedExpr
}

/** Match expression: `type` is the sum type the elaborator inferred
 *  from the arms; `arms` is the ordered list. The elaborator validates
 *  exhaustiveness (every variant has an arm) and absence of duplicates. */
export interface Match {
  op: 'match'
  type: SumTypeDef
  scrutinee: ResolvedExpr
  arms: MatchArm[]
}

// ─────────────────────────────────────────────────────────────
// ResolvedExpr — the expression universe at the resolved phase
// ─────────────────────────────────────────────────────────────

/** Value-producing expressions in the resolved phase. */
export type ResolvedExpr =
  | number | boolean | ResolvedExpr[]
  | ResolvedExprOp

export type ResolvedExprOp =
  // Operators
  | BinaryOp | UnaryOp
  | Clamp | Select | Index
  // Array ops (lowered by array_lower in C6)
  | Zeros | ArraySet
  // References (graph edges)
  | InputRef | RegRef | ParamRef | TypeParamRef | BindingRef
  | NestedOut
  // Sentinels
  | SampleRate | SampleIndex
  // Combinators
  | Fold | Scan
  | Generate | Iterate | Chain
  | Map2 | ZipWith
  // Let
  | Let
  // ADT expressions
  | Tag | Match

// ─────────────────────────────────────────────────────────────
// Block + Program
// ─────────────────────────────────────────────────────────────

export interface ResolvedBlock {
  op: 'block'
  decls: BodyDecl[]
  assigns: BodyAssign[]
}

export interface ResolvedProgramPorts {
  inputs: InputDecl[]
  outputs: OutputDecl[]
  typeDefs: TypeDef[]
}

export interface ResolvedProgram {
  op: 'program'
  name: string
  typeParams: TypeParamDecl[]
  ports: ResolvedProgramPorts
  body: ResolvedBlock
  /** Typed decl tables, projected from `body.decls` by kind. Position
   *  in each array IS the decl's identity for the de Bruijn levels
   *  migration — the same `RegDecl` object appears at `body.decls[i]`
   *  and at `regs[k]` for some k. Populated by `withDeclTables`
   *  (`compiler/ir/decl_tables.ts`); every ResolvedProgram constructor
   *  must route through that helper to keep the tables in sync with
   *  `body.decls`. `InputDecl` and `OutputDecl` positional identity
   *  already lives in `ports.inputs` / `ports.outputs` — those arrays
   *  ARE their tables; no top-level duplicate. */
  regs:      RegDecl[]
  params:    ParamDecl[]
  instances: InstanceDecl[]
  /** Next-fresh BinderIdx for this program. The elaborator increments
   *  this for every binder it creates (let entries, combinator binders,
   *  match-arm payload binders). Cross-program lifting
   *  (`inline_instances`) shifts inner binder IDs by an offset and
   *  bumps the outer's `binderCount` by the inner's. */
  binderCount: number
  /** Registry of program types this program (transitively) references
   *  through its `instances`. Keyed by `ProgramKey` (the target
   *  program's `name`). Phase 4a of issue #156: dual-read with
   *  `instance.type`; the cross-check test asserts
   *  `instance.type === programRegistry.get(instance.typeKey)`.
   *  Phase 4b drops `instance.type` and this becomes the sole resolver. */
  programRegistry: ReadonlyMap<ProgramKey, ResolvedProgram>
}

// ─────────────────────────────────────────────────────────────
// Elaboration error
// ─────────────────────────────────────────────────────────────

/** Thrown by the elaborator when it encounters an unresolvable name,
 *  exhaustiveness violation, or other semantic error. */
export class ElaborationError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'ElaborationError'
  }
}
