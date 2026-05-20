/**
 * nodes.ts — strict discriminated-union node types for the .trop parser.
 *
 * Two kinds of strings live in the parsed tree:
 *
 *  1. Identity strings — names the user gave to declarations (RegDecl.name,
 *     InstanceDecl.name, Program.name, OutputAssign.name, paramDecl.name,
 *     etc.) and anonymous binder labels (Binding.name, Let.bind keys,
 *     MatchArm.bind). These are the user's chosen labels for things that have
 *     no other name; they never resolve to a different entity.
 *
 *  2. Reference strings — none. Every place where the user's source mentions
 *     something declared elsewhere is wrapped in NameRef. The elaborator
 *     resolves NameRefNodes to graph edges (direct references to decl objects)
 *     in a uniform pass. The parser does no scope analysis; it simply records
 *     "this is a name awaiting resolution at this position."
 *
 * Concretely: instance refs (`osc.out`), program-type names in instance
 * declarations (`SinOsc(...)`), variant names in tag/match, scalar-kind names
 * in port-type declarations and reg type annotations, type-param refs in
 * array shapes, alias base types — all become `NameRef`. The position
 * the NameRef appears in tells the elaborator which scope to resolve
 * against.
 *
 * Three categorically-distinct value universes:
 *
 *   ParsedExpr  — value-producing computations (literals, infix/unary,
 *                     calls, dotted refs, indexing, let, combinator bodies,
 *                     match, tag, parser-internal placeholders).
 *   BodyDecl        — declarations introducing names into program scope
 *                     (regDecl, delayDecl, paramDecl, instanceDecl,
 *                     programDecl).
 *   BodyAssign      — wires pinning a value to a port (outputAssign,
 *                     nextUpdate).
 *
 *   TypeDef         — type-level declarations (struct/sum/alias). Lives in
 *                     `ports.type_defs`, not in body.decls/assigns.
 *
 * `Block` carries homogeneously-typed arrays. `Tag` and `Match`
 * carry no `type` field; the elaborator fills that in from the sum-type
 * registry. The forthcoming elaborator (B6) defines a separate
 * `ResolvedExprNode` union (in compiler/ir/) that replaces every NameRef
 * with a direct decl reference. The elaborator's signature is
 * `ParsedExpr -> ResolvedExprNode`.
 *
 * `Expr` is exported as an alias for `ParsedExpr` so callers within
 * this directory can use the short name.
 *
 * Note on shadowing: the parser does not preserve let/combinator/match
 * binder shadowing — `let { x: 1 } in let { x: 2 } in x` produces two
 * `Binding { name: 'x' }` references that both refer to "any binder
 * named x" in the surrounding scope. The elaborator must disambiguate
 * if shadowing semantics matter for downstream stages.
 */

// ─────────────────────────────────────────────────────────────
// Expr — value-producing universe (parsed phase)
// ─────────────────────────────────────────────────────────────

/** Top-level parser-phase expression union: literals, arrays, and op-tagged
 *  objects emitted by the surface parser. */
export type ParsedExpr = number | boolean | ParsedExpr[] | ParsedExprOp

/** Convenience alias for use inside the parser, where there's only one
 *  phase. Cross-phase code should prefer the phase-explicit name. */
export type Expr = ParsedExpr

/** All op-tagged expression nodes the parser can emit. The `op` tag is the
 *  discriminator; downstream switch statements narrow exhaustively. */
export type ParsedExprOp =
  | BinaryOp
  | UnaryOp
  | Call
  | NameRef
  | Binding
  | NestedOut
  | Index
  | Let
  | Fold | Scan
  | Generate | Iterate | Chain
  | Map2 | ZipWith
  | Tag | Match

// ── Binary ops ────────────────────────────────────────────────

export type BinaryOpTag =
  | 'add' | 'sub' | 'mul' | 'div' | 'mod'
  | 'lt' | 'lte' | 'gt' | 'gte' | 'eq' | 'neq'
  | 'and' | 'or'
  | 'bitAnd' | 'bitOr' | 'bitXor' | 'lshift' | 'rshift'

export interface BinaryOp {
  op: BinaryOpTag
  args: [Expr, Expr]
}

// ── Unary ops ─────────────────────────────────────────────────

export type UnaryOpTag = 'neg' | 'not' | 'bitNot'

export interface UnaryOp {
  op: UnaryOpTag
  args: [Expr]
}

// ── Calls and references ──────────────────────────────────────

/** Generic function call. The elaborator resolves the callee — built-in
 *  ops with function-call surface (sqrt, clamp, etc.) get rewritten to
 *  their structured op; user functions stay as `call`. */
export interface Call {
  op: 'call'
  callee: Expr
  args: Expr[]
}

/** Unresolved name-reference placeholder. Every place where a parsed-tree
 *  node mentions another node by name (instance refs, program-type names
 *  in instance decls, variant names, scalar-kind names in port types,
 *  type-param refs in array shapes, ...) wraps that name in a NameRef.
 *  The elaborator resolves NameRefNodes to direct decl references.
 *  Position determines scope. */
export interface NameRef {
  op: 'nameRef'
  name: string
}

/** Convenience constructor — exists to give NameRef introduction a
 *  vocabulary, not to enforce anything (TypeScript object literals would
 *  work too). Use at every site that emits a NameRef so a future
 *  refactor (e.g. carrying source position) only needs to change here. */
export const nameRef = (name: string): NameRef => ({ op: 'nameRef', name })

/** Lexically-bound name: introduced by a `let`, combinator binder, or
 *  match-arm pattern. Body parsers track binders in scope and emit this
 *  for matching identifiers. */
export interface Binding {
  op: 'binding'
  name: string
}

/** Dotted port reference: `inst.port`. Both `ref` (the instance) and
 *  `output` (the port name on the referenced program type) are unresolved
 *  at parse time and wrapped in NameRef. The elaborator resolves
 *  `ref` against in-scope instances and `output` against the resolved
 *  program type's declared output ports. */
export interface NestedOut {
  op: 'nestedOut'
  ref: NameRef
  output: NameRef
}

/** Indexing: `arr[i]`. Args are [array, index]. */
export interface Index {
  op: 'index'
  args: [Expr, Expr]
}

// ── Bindings ──────────────────────────────────────────────────

/** `let { x: e1, y: e2 } in body` — body sees x and y as `binding(name)`. */
export interface Let {
  op: 'let'
  bind: Record<string, Expr>
  in: Expr
}

// ── Combinators ───────────────────────────────────────────────

/** `fold(over, init, (acc, elem) => body)` — left fold to scalar. */
export interface Fold {
  op: 'fold'
  over: Expr
  init: Expr
  acc_var: string
  elem_var: string
  body: Expr
}

/** `scan(over, init, (acc, elem) => body)` — like fold but keeps
 *  intermediates. Same shape. */
export interface Scan {
  op: 'scan'
  over: Expr
  init: Expr
  acc_var: string
  elem_var: string
  body: Expr
}

/** `generate(count, (i) => body)` — produce an array of body[i=0..N-1].
 *  `count` is an Expr (number literal or typeParam ref); the
 *  elaborator + array-lowering specialize it. */
export interface Generate {
  op: 'generate'
  count: Expr
  var: string
  body: Expr
}

/** `iterate(count, init, (x) => body)` — [init, f(init), f(f(init)), ...]. */
export interface Iterate {
  op: 'iterate'
  count: Expr
  var: string
  init: Expr
  body: Expr
}

/** `chain(count, init, (x) => body)` — apply body count times, threading. */
export interface Chain {
  op: 'chain'
  count: Expr
  var: string
  init: Expr
  body: Expr
}

/** `map2(over, (e) => body)` — single-binder map. */
export interface Map2 {
  op: 'map2'
  over: Expr
  elem_var: string
  body: Expr
}

/** `zipWith(a, b, (x, y) => body)` — two-array pointwise combine. */
export interface ZipWith {
  op: 'zipWith'
  a: Expr
  b: Expr
  x_var: string
  y_var: string
  body: Expr
}

// ── ADT expressions (parsed phase — no `type` field) ──────────

/** A single payload-field assignment in tag construction:
 *  `{ field: expr, field: expr }`. The field name is a NameRef
 *  awaiting resolution against the variant's declared payload fields. */
export interface TagPayloadEntry {
  field: NameRef
  value: Expr
}

/** `Variant { field: expr, ... }` — sum-type constructor.
 *  `variant` is unresolved at parse time and wrapped in NameRef;
 *  the elaborator resolves it against the sum-type registry (variant
 *  names uniquely identify a sum type). The sum-type name is filled in
 *  there too — the parsed Tag has no `type` field. */
export interface Tag {
  op: 'tag'
  variant: NameRef
  payload?: TagPayloadEntry[]
}

/** A single arm of a `match`: `Variant [{ field: name, ... }] => body`.
 *  `variant` is a NameRef resolved against the sum type's variants.
 *  `binds` is the ordered list of (payload-field-name, local-bind-name)
 *  pairings the user wrote in the pattern — empty when the variant has
 *  no payload. The field is wrapped in a NameRef awaiting resolution
 *  against the variant's declared payload; the bind name is a plain
 *  string (binders are anonymous — no decl exists). */
export interface MatchArmEntry {
  variant: NameRef
  binds: Array<{ field: NameRef; bind: string }>
  body: Expr
}

/** `match scrutinee { Variant => body, V { f: x } => body, ... }`.
 *  Arms are an ordered array (arm order is meaningful); the parser
 *  rejects duplicate variants at parse time. No `type` field at the
 *  parsed phase. */
export interface Match {
  op: 'match'
  scrutinee: Expr
  arms: MatchArmEntry[]
}

// ─────────────────────────────────────────────────────────────
// BodyDecl — declarations introducing names into program scope
// ─────────────────────────────────────────────────────────────

export type BodyDecl =
  | RegDecl
  | DelayDecl
  | ParamDecl
  | InstanceDecl
  | ProgramDecl

/** `reg name [: type] = init` — persistent state register.
 *  `type` is a NameRef (e.g., `float`, `signal`, or a user alias) the
 *  elaborator resolves against scalar kinds + the program's type aliases. */
export interface RegDecl {
  op: 'regDecl'
  name: string
  init: Expr
  type?: NameRef
}

/** `delay name[: type] = update_expr init init_value` — synthetic one-
 *  sample delay register. `update` is the next-tick value; `init` is the
 *  starting value. `type`, when present, is a sum-type name; the sum-
 *  decomposition pre-pass (in `compiler/session.ts`) consults it to
 *  expand sum-typed delays into N+1 scalar delay slots. */
export interface DelayDecl {
  op: 'delayDecl'
  name: string
  update: Expr
  init: Expr
  type?: NameRef
}

/** `param name: smoothed = default`. */
export interface ParamDecl {
  op: 'paramDecl'
  name: string
  value?: number
}

/** `<param=value, ...>` entry in an instance's type-args list. The param
 *  is a NameRef the elaborator resolves against the target program
 *  type's declared `type_params`. */
export interface TypeArgEntry {
  param: NameRef
  value: number
}

/** `(port: expr, ...)` entry in an instance's input keyword args. The
 *  port is a NameRef resolved against the target program type's
 *  declared input ports. */
export interface InstanceInputEntry {
  port: NameRef
  value: Expr
}

/** `name = ProgType<typeArgs>(port: expr, port: expr)` — instance of a
 *  registered program type. `program` is a NameRef resolved against
 *  the program type registry. */
export interface InstanceDecl {
  op: 'instanceDecl'
  name: string
  program: NameRef
  type_args?: TypeArgEntry[]
  inputs?: InstanceInputEntry[]
}

/** `program SubName(...) -> (...) { ... }` inside an outer body —
 *  introduces a nested program type into the outer's scope. */
export interface ProgramDecl {
  op: 'programDecl'
  name: string
  program: Program
}

// ─────────────────────────────────────────────────────────────
// BodyAssign — wires pinning a value to a port
// ─────────────────────────────────────────────────────────────

export type BodyAssign =
  | OutputAssign
  | NextUpdate

/** `port = expr` — wire `expr` to a declared output port (name) or to
 *  the DAC boundary leaf (name='dac.out'). */
export interface OutputAssign {
  op: 'outputAssign'
  name: string
  expr: Expr
}

/** `next regName = expr` — register update. `target.kind` is currently
 *  always `'reg'` from the surface; the `'delay'` branch in the IR is
 *  reserved for delays carrying their update separately (today they
 *  carry it inside DelayDecl). */
export interface NextUpdate {
  op: 'nextUpdate'
  target: { kind: 'reg' | 'delay'; name: string }
  expr: Expr
}

// ─────────────────────────────────────────────────────────────
// Block + Program-level types
// ─────────────────────────────────────────────────────────────

/** A program body: ordered decls + assigns. Type defs (struct/enum/type)
 *  do not live here — they're routed to `ports.type_defs` at parse time. */
export interface Block {
  op: 'block'
  decls: BodyDecl[]
  assigns: BodyAssign[]
}

/** Compile-time array-shape dimension: integer literal or NameRef.
 *  The NameRef is resolved by the elaborator against the enclosing
 *  program's declared type-params. */
export type ShapeDim = number | NameRef

/** Port type: bare scalar name, or array with element + shape. The
 *  element name is a NameRef (`float`/`int`/`bool` or a user alias)
 *  resolved against scalar kinds + program type aliases. */
export type PortTypeDecl =
  | NameRef
  | { kind: 'array'; element: NameRef; shape: ShapeDim[] }

export interface ProgramPortSpec {
  name: string
  type?: PortTypeDecl
  default?: Expr
  /** Source-level `in [lo, hi]` bound annotation. Lowered to explicit
   *  `clamp` ops by `lowerBoundsToClamps` (compiler/parse/lower_bounds.ts)
   *  before the IR is exposed to the elaborator. Either side may be
   *  `null` for an open bound (e.g. `freq` is `[0, null]`). The field
   *  is parser-internal — code outside the parser never observes it. */
  bounds?: [number | null, number | null]
}

/** A port entry: bare-name short form, or full spec. */
export type ProgramPort = string | ProgramPortSpec

// ── TypeDefs ──────────────────────────────────────────────────

export type ScalarKind = 'float' | 'int' | 'bool'

export interface StructField {
  name: string
  scalar_type: ScalarKind
}

export interface StructTypeDef {
  kind: 'struct'
  name: string
  fields: StructField[]
}

export interface SumVariant {
  name: string
  payload: StructField[]
}

export interface SumTypeDef {
  kind: 'sum'
  name: string
  variants: SumVariant[]
}

export interface AliasTypeDef {
  kind: 'alias'
  name: string
  base: NameRef
}

export type TypeDef = StructTypeDef | SumTypeDef | AliasTypeDef

// ── ProgramPorts + Program ────────────────────────────────

export interface ProgramPorts {
  inputs?: ProgramPort[]
  outputs?: ProgramPort[]
  type_defs?: TypeDef[]
}

/** A program declaration: header + body. The unit produced by parsing
 *  a top-level `program ...` declaration in `.trop`. The optional
 *  `breaks_cycles` flag is a hint to the legacy flattener's cycle
 *  detector; in `.trop` source it appears as a contextual keyword
 *  between the output list and the body brace (`program X(...) -> (...)
 *  breaks_cycles { ... }`). */
export interface Program {
  op: 'program'
  name: string
  type_params?: Record<string, { type: 'int'; default?: number }>
  ports?: ProgramPorts
  body: Block
  breaks_cycles?: boolean
}
