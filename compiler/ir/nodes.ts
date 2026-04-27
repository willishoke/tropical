/**
 * compiler/ir/nodes.ts — resolved-phase IR types.
 *
 * The elaborator (`compiler/ir/elaborator.ts`) consumes a parsed tree
 * (`compiler/parse/nodes.ts`) and produces values of the types defined
 * here. Where the parsed tree had NameRefNode placeholders, the resolved
 * tree has direct decl references — every reference is a graph edge.
 *
 * Categorical shape
 * -----------------
 * Decls (introduction sites): InputDecl, OutputDecl, RegDecl, DelayDecl,
 *   ParamDecl, TypeParamDecl, InstanceDecl, ProgramDecl, BinderDecl, plus
 *   the sum-type members (SumTypeDef, SumVariant, StructTypeDef,
 *   StructField, AliasTypeDef). Each carries an identity string `name`.
 *
 * Refs (uses): InputRef, RegRef, DelayRef, ParamRef, TypeParamRef,
 *   BindingRef. Each holds `decl: <its decl type>`. Refs hold the decl
 *   by reference identity (===) — two `RegRef.decl` for the same
 *   register are the same object.
 *
 * Bridges between term-and-type levels: NestedOut ties an instance ref
 * to a specific output port of its program type; ResolvedTagNode and
 * ResolvedMatchNode tie expressions to sum-type variants.
 *
 * Graph property: the resolved tree admits cycles. A delay's `update`
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
 *  not a `NameRefNode` — these are language primitives, not decls. */
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
  bounds: [number | null, number | null]
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
  bounds?: [number | null, number | null]
}

export interface OutputDecl {
  op: 'outputDecl'
  name: string
  type?: PortType
  bounds?: [number | null, number | null]
}

export interface TypeParamDecl {
  op: 'typeParamDecl'
  name: string
  default?: number
}

// ─────────────────────────────────────────────────────────────
// Body decls — names introduced in a program body
// ─────────────────────────────────────────────────────────────

export interface RegDecl {
  op: 'regDecl'
  name: string
  init: ResolvedExpr
  type?: ScalarKind | AliasTypeDef
}

export interface DelayDecl {
  op: 'delayDecl'
  name: string
  update: ResolvedExpr
  init: ResolvedExpr
}

export interface ParamDecl {
  op: 'paramDecl'
  name: string
  kind: 'param' | 'trigger'   // surface 'smoothed' → IR 'param'
  value?: number
}

export interface InstanceDecl {
  op: 'instanceDecl'
  name: string
  type: ResolvedProgram
  /** Type-arg pairs: each holds a reference to one of the target's
   *  declared type params plus the integer value supplied at the
   *  instance site. */
  typeArgs: Array<{ param: TypeParamDecl; value: number }>
  /** Input wires: each holds a reference to one of the target's declared
   *  input ports plus the value-expression wired into it. The elaborator
   *  validates that every required input is supplied (defaults handle
   *  missing entries). */
  inputs: Array<{ port: InputDecl; value: ResolvedExpr }>
}

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
  | DelayDecl
  | ParamDecl
  | InstanceDecl
  | ProgramDecl

// ─────────────────────────────────────────────────────────────
// Body assigns — wires pinning a value to a port
// ─────────────────────────────────────────────────────────────

export interface OutputAssign {
  op: 'outputAssign'
  /** Either an OutputDecl from this program's outputs, or the special
   *  `'dac'` boundary leaf for top-level patches that wire to the DAC. */
  target: OutputDecl | { kind: 'dac' }
  expr: ResolvedExpr
}

export interface NextUpdate {
  op: 'nextUpdate'
  target: RegDecl | DelayDecl
  expr: ResolvedExpr
}

export type BodyAssign = OutputAssign | NextUpdate

// ─────────────────────────────────────────────────────────────
// Binders — anonymous names introduced by let / combinators / match arms
// ─────────────────────────────────────────────────────────────

/** A single anonymous binder. The parent node (LetExpr, FoldExpr, etc.,
 *  or MatchArm) determines the binder's role. The `name` is an identity
 *  string — the user's chosen label, not a reference. */
export interface BinderDecl {
  op: 'binderDecl'
  name: string
}

// ─────────────────────────────────────────────────────────────
// Refs — uses of decl objects
// ─────────────────────────────────────────────────────────────

export interface InputRef    { op: 'inputRef';    decl: InputDecl }
export interface RegRef      { op: 'regRef';      decl: RegDecl }
export interface DelayRef    { op: 'delayRef';    decl: DelayDecl }
export interface ParamRef    { op: 'paramRef';    decl: ParamDecl }
export interface TypeParamRef { op: 'typeParamRef'; decl: TypeParamDecl }
export interface BindingRef  { op: 'bindingRef';  decl: BinderDecl }

/** Dotted port reference resolved: a specific instance + a specific
 *  output port of that instance's program type. Both held by reference. */
export interface NestedOut {
  op: 'nestedOut'
  instance: InstanceDecl
  output: OutputDecl
}

// ─────────────────────────────────────────────────────────────
// Sentinel leaves — semantic primitives, no decl
// ─────────────────────────────────────────────────────────────

export interface SampleRateNode  { op: 'sampleRate' }
export interface SampleIndexNode { op: 'sampleIndex' }

// ─────────────────────────────────────────────────────────────
// Builtin op shapes — same as parsed but with resolved children
// ─────────────────────────────────────────────────────────────

export type BinaryOpTag =
  | 'add' | 'sub' | 'mul' | 'div' | 'mod'
  | 'lt' | 'lte' | 'gt' | 'gte' | 'eq' | 'neq'
  | 'and' | 'or'
  | 'bitAnd' | 'bitOr' | 'bitXor' | 'lshift' | 'rshift'

export interface BinaryOpNode {
  op: BinaryOpTag
  args: [ResolvedExpr, ResolvedExpr]
}

export type UnaryOpTag =
  | 'neg' | 'not' | 'bitNot'
  | 'sqrt' | 'abs' | 'floor' | 'ceil' | 'round'
  | 'floatExponent' | 'toInt' | 'toBool' | 'toFloat'

export interface UnaryOpNode {
  op: UnaryOpTag
  args: [ResolvedExpr]
}

/** `clamp(value, lo, hi)` — preserves bounds. */
export interface ClampNode {
  op: 'clamp'
  args: [ResolvedExpr, ResolvedExpr, ResolvedExpr]
}

/** `select(cond, then, else)` — value-level if. */
export interface SelectNode {
  op: 'select'
  args: [ResolvedExpr, ResolvedExpr, ResolvedExpr]
}

/** `index(arr, i)` — array element access. */
export interface IndexNode {
  op: 'index'
  args: [ResolvedExpr, ResolvedExpr]
}

// ─────────────────────────────────────────────────────────────
// Combinators — each one carries its binder declarations directly
// ─────────────────────────────────────────────────────────────

export interface FoldExpr {
  op: 'fold'
  over: ResolvedExpr
  init: ResolvedExpr
  acc: BinderDecl
  elem: BinderDecl
  body: ResolvedExpr
}

export interface ScanExpr {
  op: 'scan'
  over: ResolvedExpr
  init: ResolvedExpr
  acc: BinderDecl
  elem: BinderDecl
  body: ResolvedExpr
}

export interface GenerateExpr {
  op: 'generate'
  count: ResolvedExpr
  iter: BinderDecl
  body: ResolvedExpr
}

export interface IterateExpr {
  op: 'iterate'
  count: ResolvedExpr
  init: ResolvedExpr
  iter: BinderDecl
  body: ResolvedExpr
}

export interface ChainExpr {
  op: 'chain'
  count: ResolvedExpr
  init: ResolvedExpr
  iter: BinderDecl
  body: ResolvedExpr
}

export interface Map2Expr {
  op: 'map2'
  over: ResolvedExpr
  elem: BinderDecl
  body: ResolvedExpr
}

export interface ZipWithExpr {
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

export interface LetExpr {
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
export interface TagExpr {
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
export interface MatchExpr {
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
  | ResolvedExprOpNode

export type ResolvedExprOpNode =
  // Operators
  | BinaryOpNode | UnaryOpNode
  | ClampNode | SelectNode | IndexNode
  // References (graph edges)
  | InputRef | RegRef | DelayRef | ParamRef | TypeParamRef | BindingRef
  | NestedOut
  // Sentinels
  | SampleRateNode | SampleIndexNode
  // Combinators
  | FoldExpr | ScanExpr
  | GenerateExpr | IterateExpr | ChainExpr
  | Map2Expr | ZipWithExpr
  // Let
  | LetExpr
  // ADT expressions
  | TagExpr | MatchExpr

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
