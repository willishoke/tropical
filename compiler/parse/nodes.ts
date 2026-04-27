/**
 * nodes.ts — strict discriminated-union node types for the .trop parser.
 *
 * Two kinds of strings live in the parsed tree:
 *
 *  1. Identity strings — names the user gave to declarations (RegDecl.name,
 *     InstanceDecl.name, ProgramNode.name, OutputAssign.name, paramDecl.name,
 *     etc.) and anonymous binder labels (BindingNode.name, LetNode.bind keys,
 *     MatchArm.bind). These are the user's chosen labels for things that have
 *     no other name; they never resolve to a different entity.
 *
 *  2. Reference strings — none. Every place where the user's source mentions
 *     something declared elsewhere is wrapped in NameRefNode. The elaborator
 *     resolves NameRefNodes to graph edges (direct references to decl objects)
 *     in a uniform pass. The parser does no scope analysis; it simply records
 *     "this is a name awaiting resolution at this position."
 *
 * Concretely: instance refs (`osc.out`), program-type names in instance
 * declarations (`SinOsc(...)`), variant names in tag/match, scalar-kind names
 * in port-type declarations and reg type annotations, type-param refs in
 * array shapes, alias base types — all become `NameRefNode`. The position
 * the NameRefNode appears in tells the elaborator which scope to resolve
 * against.
 *
 * Three categorically-distinct value universes:
 *
 *   ParsedExprNode  — value-producing computations (literals, infix/unary,
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
 * `BlockNode` carries homogeneously-typed arrays. `TagNode` and `MatchNode`
 * carry no `type` field; the elaborator fills that in from the sum-type
 * registry. The forthcoming elaborator (B6) defines a separate
 * `ResolvedExprNode` union (in compiler/ir/) that replaces every NameRefNode
 * with a direct decl reference. The elaborator's signature is
 * `ParsedExprNode -> ResolvedExprNode`.
 *
 * `ExprNode` is exported as an alias for `ParsedExprNode` so callers within
 * this directory can use the short name.
 *
 * Note on shadowing: the parser does not preserve let/combinator/match
 * binder shadowing — `let { x: 1 } in let { x: 2 } in x` produces two
 * `BindingNode { name: 'x' }` references that both refer to "any binder
 * named x" in the surrounding scope. The elaborator must disambiguate
 * if shadowing semantics matter for downstream stages.
 */

// ─────────────────────────────────────────────────────────────
// ExprNode — value-producing universe (parsed phase)
// ─────────────────────────────────────────────────────────────

/** Top-level parser-phase expression union: literals, arrays, and op-tagged
 *  objects emitted by the surface parser. */
export type ParsedExprNode = number | boolean | ParsedExprNode[] | ExprOpNode

/** Convenience alias for use inside the parser, where there's only one
 *  phase. Cross-phase code should prefer the phase-explicit name. */
export type ExprNode = ParsedExprNode

/** All op-tagged expression nodes the parser can emit. The `op` tag is the
 *  discriminator; downstream switch statements narrow exhaustively. */
export type ExprOpNode =
  | BinaryOpNode
  | UnaryOpNode
  | CallNode
  | NameRefNode
  | BindingNode
  | NestedOutNode
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

/** Unresolved name-reference placeholder. Every place where a parsed-tree
 *  node mentions another node by name (instance refs, program-type names
 *  in instance decls, variant names, scalar-kind names in port types,
 *  type-param refs in array shapes, ...) wraps that name in a NameRefNode.
 *  The elaborator resolves NameRefNodes to direct decl references.
 *  Position determines scope. */
export interface NameRefNode {
  op: 'nameRef'
  name: string
}

/** Convenience constructor — exists to give NameRefNode introduction a
 *  vocabulary, not to enforce anything (TypeScript object literals would
 *  work too). Use at every site that emits a NameRef so a future
 *  refactor (e.g. carrying source position) only needs to change here. */
export const nameRef = (name: string): NameRefNode => ({ op: 'nameRef', name })

/** Lexically-bound name: introduced by a `let`, combinator binder, or
 *  match-arm pattern. Body parsers track binders in scope and emit this
 *  for matching identifiers. */
export interface BindingNode {
  op: 'binding'
  name: string
}

/** Dotted port reference: `inst.port`. Both `ref` (the instance) and
 *  `output` (the port name on the referenced program type) are unresolved
 *  at parse time and wrapped in NameRefNode. The elaborator resolves
 *  `ref` against in-scope instances and `output` against the resolved
 *  program type's declared output ports. */
export interface NestedOutNode {
  op: 'nestedOut'
  ref: NameRefNode
  output: NameRefNode
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

// ── ADT expressions (parsed phase — no `type` field) ──────────

/** A single payload-field assignment in tag construction:
 *  `{ field: expr, field: expr }`. The field name is a NameRefNode
 *  awaiting resolution against the variant's declared payload fields. */
export interface TagPayloadEntry {
  field: NameRefNode
  value: ExprNode
}

/** `Variant { field: expr, ... }` — sum-type constructor.
 *  `variant` is unresolved at parse time and wrapped in NameRefNode;
 *  the elaborator resolves it against the sum-type registry (variant
 *  names uniquely identify a sum type). The sum-type name is filled in
 *  there too — the parsed TagNode has no `type` field. */
export interface TagNode {
  op: 'tag'
  variant: NameRefNode
  payload?: TagPayloadEntry[]
}

/** A single arm of a `match`: `Variant [{ field: name, ... }] => body`.
 *  `variant` is a NameRefNode resolved against the sum type's variants.
 *  `bind` is the local name(s) for payload fields (string for one
 *  binder, string[] for multiple); omitted when the variant has no
 *  payload. (`bind` is anonymous — no decl exists, so it stays a
 *  string.) */
export interface MatchArmEntry {
  variant: NameRefNode
  bind?: string | string[]
  body: ExprNode
}

/** `match scrutinee { Variant => body, V { f: x } => body, ... }`.
 *  Arms are an ordered array (arm order is meaningful); the parser
 *  rejects duplicate variants at parse time. No `type` field at the
 *  parsed phase. */
export interface MatchNode {
  op: 'match'
  scrutinee: ExprNode
  arms: MatchArmEntry[]
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

/** `reg name [: type] = init` — persistent state register.
 *  `type` is a NameRefNode (e.g., `float`, `signal`, or a user alias) the
 *  elaborator resolves against scalar kinds + the program's type aliases. */
export interface RegDeclNode {
  op: 'regDecl'
  name: string
  init: ExprNode
  type?: NameRefNode
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

/** `<param=value, ...>` entry in an instance's type-args list. The param
 *  is a NameRefNode the elaborator resolves against the target program
 *  type's declared `type_params`. */
export interface TypeArgEntry {
  param: NameRefNode
  value: number
}

/** `(port: expr, ...)` entry in an instance's input keyword args. The
 *  port is a NameRefNode resolved against the target program type's
 *  declared input ports. */
export interface InstanceInputEntry {
  port: NameRefNode
  value: ExprNode
}

/** `name = ProgType<typeArgs>(port: expr, port: expr)` — instance of a
 *  registered program type. `program` is a NameRefNode resolved against
 *  the program type registry. */
export interface InstanceDeclNode {
  op: 'instanceDecl'
  name: string
  program: NameRefNode
  type_args?: TypeArgEntry[]
  inputs?: InstanceInputEntry[]
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

/** Compile-time array-shape dimension: integer literal or NameRefNode.
 *  The NameRefNode is resolved by the elaborator against the enclosing
 *  program's declared type-params. */
export type ShapeDim = number | NameRefNode

/** Port type: bare scalar name, or array with element + shape. The
 *  element name is a NameRefNode (`float`/`int`/`bool` or a user alias)
 *  resolved against scalar kinds + program type aliases. */
export type PortTypeDecl =
  | NameRefNode
  | { kind: 'array'; element: NameRefNode; shape: ShapeDim[] }

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
  base: NameRefNode
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
