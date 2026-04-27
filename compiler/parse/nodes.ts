/**
 * nodes.ts — strict discriminated-union node types for the .trop parser.
 *
 * Three universes, kept categorically distinct:
 *
 *   ExprNode    — value-producing computations (literals, infix/unary,
 *                 calls, dotted refs, indexing, let, combinator bodies,
 *                 match, tag, parser-internal placeholders).
 *   BodyDecl    — declarations that introduce names into program scope
 *                 (regDecl, delayDecl, paramDecl, instanceDecl,
 *                 programDecl).
 *   BodyAssign  — wires that pin a value to a port (outputAssign,
 *                 nextUpdate).
 *
 *   TypeDef     — type-level declarations (struct/sum/alias). Lives in
 *                 `ports.type_defs`, not in body.decls/assigns.
 *
 * BlockNode therefore carries homogeneously-typed arrays: a body's decls
 * are all `BodyDecl`, its assigns all `BodyAssign`. A `regDecl` is not a
 * legal `ExprNode` — the type system enforces that a declaration cannot
 * appear in expression position, and vice versa.
 *
 * Pre-slottification only: the parser emits name-bearing variants
 * (`{op:'reg', name}`, `{op:'input', name}`, `{op:'nestedOut', ref}`).
 * The id-bearing variants used post-slottification (in
 * `compiler/session.ts:slottifyExpr`) live in `compiler/expr.ts`.
 */

// ─────────────────────────────────────────────────────────────
// ExprNode — value-producing universe
// ─────────────────────────────────────────────────────────────

/** Top-level expression union: literals, arrays, and op-tagged objects. */
export type ExprNode = number | boolean | ExprNode[] | ExprOpNode

/** All op-tagged expression nodes the parser can emit. The `op` tag is the
 *  discriminator; downstream switch statements narrow exhaustively. */
export type ExprOpNode =
  | BinaryOpNode
  | UnaryOpNode
  | CallNode
  | NameRefNode
  | BindingNode
  | NestedOutNode
  | FieldAccessNode
  | IndexNode
  | LetNode
  | FoldNode | ScanNode
  | GenerateNode | IterateNode | ChainNode
  | Map2Node | ZipWithNode
  | TagNode | MatchNode

// ── Binary ops ────────────────────────────────────────────────

export type BinaryOpTag =
  | 'add' | 'sub' | 'mul' | 'div' | 'mod'
  | 'lt' | 'lte' | 'gt' | 'gte' | 'eq' | 'neq'
  | 'and' | 'or'
  | 'bitAnd' | 'bitOr' | 'bitXor' | 'lshift' | 'rshift'

export interface BinaryOpNode {
  op: BinaryOpTag
  args: [ExprNode, ExprNode]
}

// ── Unary ops ─────────────────────────────────────────────────

export type UnaryOpTag = 'neg' | 'not' | 'bitNot'

export interface UnaryOpNode {
  op: UnaryOpTag
  args: [ExprNode]
}

// ── Calls and references ──────────────────────────────────────

/** Generic function call. The elaborator resolves the callee — built-in
 *  ops with function-call surface (sqrt, clamp, etc.) get rewritten to
 *  their structured op; user functions stay as `call`. */
export interface CallNode {
  op: 'call'
  callee: ExprNode
  args: ExprNode[]
}

/** Parser-internal placeholder for unresolved bare identifiers. The
 *  elaborator (B6) converts to `input`/`reg`/`typeParam`/etc. based on
 *  the surrounding declaration's scope. Emitted only by the parser. */
export interface NameRefNode {
  op: 'nameRef'
  name: string
}

/** Lexically-bound name: introduced by a `let`, combinator binder, or
 *  match-arm pattern. Body parsers track binders in scope and emit this
 *  for matching identifiers. */
export interface BindingNode {
  op: 'binding'
  name: string
}

/** Pre-slottification dotted port reference: `inst.port`. */
export interface NestedOutNode {
  op: 'nestedOut'
  ref: string
  output: string | number
}

/** Field access on a non-instance expression. Emitted as a fallback for
 *  forms the parser doesn't recognize as instance refs (`expr.field`).
 *  Currently has no consumer; reserved for future struct-field access. */
export interface FieldAccessNode {
  op: 'fieldAccess'
  expr: ExprNode
  field: string
}

/** Indexing: `arr[i]`. Args are [array, index]. */
export interface IndexNode {
  op: 'index'
  args: [ExprNode, ExprNode]
}

// ── Bindings ──────────────────────────────────────────────────

/** `let { x: e1, y: e2 } in body` — body sees x and y as `binding(name)`. */
export interface LetNode {
  op: 'let'
  bind: Record<string, ExprNode>
  in: ExprNode
}

// ── Combinators ───────────────────────────────────────────────

/** `fold(over, init, (acc, elem) => body)` — left fold to scalar. */
export interface FoldNode {
  op: 'fold'
  over: ExprNode
  init: ExprNode
  acc_var: string
  elem_var: string
  body: ExprNode
}

/** `scan(over, init, (acc, elem) => body)` — like fold but keeps
 *  intermediates. Same shape. */
export interface ScanNode {
  op: 'scan'
  over: ExprNode
  init: ExprNode
  acc_var: string
  elem_var: string
  body: ExprNode
}

/** `generate(count, (i) => body)` — produce an array of body[i=0..N-1].
 *  `count` is an ExprNode (number literal or typeParam ref); the
 *  elaborator + array-lowering specialize it. */
export interface GenerateNode {
  op: 'generate'
  count: ExprNode
  var: string
  body: ExprNode
}

/** `iterate(count, init, (x) => body)` — [init, f(init), f(f(init)), ...]. */
export interface IterateNode {
  op: 'iterate'
  count: ExprNode
  var: string
  init: ExprNode
  body: ExprNode
}

/** `chain(count, init, (x) => body)` — apply body count times, threading. */
export interface ChainNode {
  op: 'chain'
  count: ExprNode
  var: string
  init: ExprNode
  body: ExprNode
}

/** `map2(over, (e) => body)` — single-binder map. */
export interface Map2Node {
  op: 'map2'
  over: ExprNode
  elem_var: string
  body: ExprNode
}

/** `zipWith(a, b, (x, y) => body)` — two-array pointwise combine. */
export interface ZipWithNode {
  op: 'zipWith'
  a: ExprNode
  b: ExprNode
  x_var: string
  y_var: string
  body: ExprNode
}

// ── ADT expressions ───────────────────────────────────────────

/** `Variant { field: expr, ... }` — sum-type constructor.
 *  `type` is the sum-type name; the parser emits `''` and the elaborator
 *  fills it from variant-name lookup. */
export interface TagNode {
  op: 'tag'
  type: string
  variant: string
  payload?: Record<string, ExprNode>
}

/** A single arm of a `match`. `bind` is the local name(s) for payload
 *  fields (string for one binder, string[] for multiple); omitted when
 *  the variant has no payload. */
export interface MatchArm {
  bind?: string | string[]
  body: ExprNode
}

/** `match scrutinee { Variant => body, V { f: x } => body, ... }`.
 *  `type` is empty when emitted by the parser; elaborator fills. */
export interface MatchNode {
  op: 'match'
  type: string
  scrutinee: ExprNode
  arms: Record<string, MatchArm>
}

// ─────────────────────────────────────────────────────────────
// BodyDecl — declarations introducing names into program scope
// ─────────────────────────────────────────────────────────────

export type BodyDecl =
  | RegDeclNode
  | DelayDeclNode
  | ParamDeclNode
  | InstanceDeclNode
  | ProgramDeclNode

/** `reg name [: type] = init` — persistent state register. */
export interface RegDeclNode {
  op: 'regDecl'
  name: string
  init: ExprNode
  type?: string
}

/** `delay name = update_expr init init_value` — synthetic one-sample
 *  delay register. `update` is the next-tick value; `init` is the
 *  starting value. */
export interface DelayDeclNode {
  op: 'delayDecl'
  name: string
  update: ExprNode
  init: ExprNode
}

/** `param name: smoothed = default` or `param name: trigger`.
 *  The `type` field uses the IR vocabulary: surface `smoothed` →
 *  IR `'param'`. */
export interface ParamDeclNode {
  op: 'paramDecl'
  name: string
  type: 'param' | 'trigger'
  value?: number
}

/** `name = ProgType<typeArgs>(port: expr, port: expr)` — instance of a
 *  registered program type. */
export interface InstanceDeclNode {
  op: 'instanceDecl'
  name: string
  program: string
  type_args?: Record<string, number>
  inputs?: Record<string, ExprNode>
}

/** `program SubName(...) -> (...) { ... }` inside an outer body —
 *  introduces a nested program type into the outer's scope. */
export interface ProgramDeclNode {
  op: 'programDecl'
  name: string
  program: ProgramNode
}

// ─────────────────────────────────────────────────────────────
// BodyAssign — wires pinning a value to a port
// ─────────────────────────────────────────────────────────────

export type BodyAssign =
  | OutputAssignNode
  | NextUpdateNode

/** `port = expr` — wire `expr` to a declared output port (name) or to
 *  the DAC boundary leaf (name='dac.out'). */
export interface OutputAssignNode {
  op: 'outputAssign'
  name: string
  expr: ExprNode
}

/** `next regName = expr` — register update. `target.kind` is currently
 *  always `'reg'` from the surface; the `'delay'` branch in the IR is
 *  reserved for delays carrying their update separately (today they
 *  carry it inside DelayDeclNode). */
export interface NextUpdateNode {
  op: 'nextUpdate'
  target: { kind: 'reg' | 'delay'; name: string }
  expr: ExprNode
}

// ─────────────────────────────────────────────────────────────
// BlockNode + Program-level types
// ─────────────────────────────────────────────────────────────

/** A program body: ordered decls + assigns. Type defs (struct/enum/type)
 *  do not live here — they're routed to `ports.type_defs` at parse time. */
export interface BlockNode {
  op: 'block'
  decls: BodyDecl[]
  assigns: BodyAssign[]
}

/** Compile-time array-shape dimension: integer literal or type-param ref. */
export type ShapeDim = number | { op: 'typeParam'; name: string }

/** Port type: bare scalar name, or array with element + shape. */
export type PortTypeDecl = string | { kind: 'array'; element: string; shape: ShapeDim[] }

export interface ProgramPortSpec {
  name: string
  type?: PortTypeDecl
  default?: ExprNode
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
  base: string
  bounds: [number | null, number | null]
}

export type TypeDef = StructTypeDef | SumTypeDef | AliasTypeDef

// ── ProgramPorts + ProgramNode ────────────────────────────────

export interface ProgramPorts {
  inputs?: ProgramPort[]
  outputs?: ProgramPort[]
  type_defs?: TypeDef[]
}

/** A program declaration: header + body. The unit produced by parsing
 *  a top-level `program ...` declaration in `.trop`. */
export interface ProgramNode {
  op: 'program'
  name: string
  type_params?: Record<string, { type: 'int'; default?: number }>
  ports?: ProgramPorts
  body: BlockNode
}
