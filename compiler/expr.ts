/**
 * SignalExpr — symbolic expression wrapper. Port of tropical/expr.py.
 *
 * TypeScript has no operator overloading, so all operations are named
 * free functions: add(a, b), mul(a, b), sin(x), etc.
 *
 * SignalExpr is a pure wrapper around an ExprNode (JSON-serializable tree).
 * No C handles — all expression evaluation happens via FlatRuntime's plan JSON.
 */

import { broadcastShapes, type ScalarKind } from './term.js'

// ---------- ExprNode (JSON-serializable expression tree) ----------

/** An expression node — bare scalar, inline array, or a named op object.
 *  The op variant is currently a bag of fields; see ExprOpNodeStrict below
 *  for the closed parametric-arity discriminated union being phased in.
 *  Once walkers are migrated to use ExprOpNodeStrict, this type will be
 *  replaced by `number | boolean | ExprNode[] | ExprOpNodeStrict`. */
export type ExprNode =
  | number
  | boolean
  | ExprNode[]
  | { op: string; [key: string]: unknown }

// ─────────────────────────────────────────────────────────────
// Closed parametric-arity discriminated union (Phase 1)
//
// `Op<N, Tag>` is a tagged structure parameterized by arity N and a union
// of allowed op tags. ~45 fixed-arity-args ops factor into instantiations
// of this single family. Named-children ops (tag, match, let, ...) get
// bespoke interfaces because their structure is genuinely irreducible.
// Leaf ops form a small union. Decl ops are top-level only.
//
// The discriminated union enables:
//   - Type-narrowed field access (no `args as ExprNode[]` casts).
//   - Exhaustive switch checks via `assertNever` (compile error on missing
//     cases when a new op is added).
//   - One shared `mapChildren` utility that knows the per-op child shape.
//
// Currently exported alongside the bag-of-fields ExprNode so existing code
// continues to compile. Walkers migrate to use ExprOpNodeStrict
// incrementally; the broad ExprNode is replaced once migration completes.
// ─────────────────────────────────────────────────────────────

/** Build a tuple type of length N filled with T. Used for Op<N, Tag> args. */
export type Tuple<T, N extends number, R extends T[] = []> =
  R['length'] extends N ? R : Tuple<T, N, [...R, T]>

/** A tagged op with N positional ExprNode children at `args`. When N is a
 *  literal number (1, 2, 3, ...), `args` is a fixed-length tuple. When N is
 *  the type `number`, `args` is a variadic ExprNode[]. */
export interface Op<N extends number, Tag extends string> {
  op: Tag
  args: Tuple<ExprNode, N>
}

// ── Per-arity tag unions ─────────────────────────────────────────────────

/** Binary arithmetic ops. */
export type ArithBinTag = 'add' | 'sub' | 'mul' | 'div' | 'mod' | 'floorDiv' | 'ldexp' | 'pow'

/** Binary comparison ops. */
export type CompareBinTag = 'lt' | 'lte' | 'gt' | 'gte' | 'eq' | 'neq'

/** Binary bitwise ops. */
export type BitBinTag = 'bitAnd' | 'bitOr' | 'bitXor' | 'lshift' | 'rshift'

/** Binary logical ops. */
export type LogicalBinTag = 'and' | 'or'

/** All binary (arity-2) op tags. */
export type BinaryTag = ArithBinTag | CompareBinTag | BitBinTag | LogicalBinTag

/** A binary op node — args is exactly two ExprNode children. */
export type BinaryNode = Op<2, BinaryTag>

/** Unary op tags (arity 1). */
export type UnaryTag =
  | 'neg' | 'abs' | 'sqrt' | 'floor' | 'ceil' | 'round'
  | 'floatExponent' | 'not' | 'bitNot'
  | 'toInt' | 'toBool' | 'toFloat'

/** A unary op node — args is exactly one ExprNode child. */
export type UnaryNode = Op<1, UnaryTag>

/** Ternary op tags (arity 3). */
export type TernaryTag = 'select' | 'clamp' | 'arraySet'

/** A ternary op node — args is exactly three ExprNode children. */
export type TernaryNode = Op<3, TernaryTag>

/** Inline array-pack op: `{op: 'array', items: ExprNode[]}`. Variadic but
 *  uses an `items` field, not `args` — bespoke shape, lives in named-children
 *  category. */
export interface ArrayNode { op: 'array'; items: ExprNode[] }

// ── Op<N> with extra non-child metadata fields ──────────────────────────

/** Reshape: traversal is Op<1>; carries a static shape annotation. */
export interface ReshapeNode extends Op<1, 'reshape'> { shape: number[] }

/** Transpose: traversal is Op<1>; no extra fields. */
export interface TransposeNode extends Op<1, 'transpose'> {}

/** Slice: traversal is Op<1>; carries axis and range. */
export interface SliceNode extends Op<1, 'slice'> {
  axis: number
  start: number
  end: number
}

/** Reduce: traversal is Op<1>; carries axis and reduction op. */
export interface ReduceNode extends Op<1, 'reduce'> {
  axis: number
  reduce_op: 'add' | 'mul' | 'min' | 'max'
}

/** broadcast_to: traversal is Op<1>; carries target shape. */
export interface BroadcastToNode extends Op<1, 'broadcastTo'> { shape: number[] }

/** index: traversal is Op<2> (array, index). */
export interface IndexNode extends Op<2, 'index'> {}

/** matmul: traversal is Op<2>; carries shape and element type. */
export interface MatmulNode extends Op<2, 'matmul'> {
  shape_a: [number, number]
  shape_b: [number, number]
  element_type?: ScalarKind
}

/** map: traversal is Op<1> args[0] PLUS callee child. Bespoke shape. */
export interface MapNode extends Op<1, 'map'> { callee: ExprNode }

// ── Construction ops (no positional args; data lives in shape/values) ───

/** zeros({shape}): allocates a zero-filled array. No children. */
export interface ZerosNode { op: 'zeros'; shape: number[] }

/** ones({shape}): allocates a one-filled array. No children. */
export interface OnesNode { op: 'ones'; shape: number[] }

/** fill({shape, value}): broadcasts a single value to the target shape. */
export interface FillNode { op: 'fill'; shape: number[]; value: ExprNode }

/** array_literal({shape, values}): an inline array of ExprNode values. */
export interface ArrayLiteralNode { op: 'arrayLiteral'; shape: number[]; values: ExprNode[] }

/** matrix({rows}): a static 2D number-only matrix. No ExprNode children. */
export interface MatrixNode { op: 'matrix'; rows: number[][] }

// ── Named-children ops (children at named fields, not positional args) ──

/** Match arm at the strict-IR layer (body is an ExprNode, not ExprCoercible).
 *  The user-facing builder counterpart is {@link MatchArm} below, which accepts
 *  ExprCoercible bodies for ergonomics. */
export interface MatchArmStrict {
  bind?: string | string[]
  body: ExprNode
}

/** Coproduct injection: `tag<T, V>{payload?}`. Children live in payload values. */
export interface TagNode {
  op: 'tag'
  type: string
  variant: string
  payload?: Record<string, ExprNode>
}

/** Coproduct elimination: `match<T>(scrutinee) { variant: arm, ... }`. */
export interface MatchNode {
  op: 'match'
  type: string
  scrutinee: ExprNode
  arms: Record<string, MatchArmStrict>
}

/** let { name = expr; ... } in body */
export interface LetNode {
  op: 'let'
  bind: Record<string, ExprNode>
  in: ExprNode
}

/** function literal — used as the callee of `call` and `map`. */
export interface FunctionNode {
  op: 'function'
  param_count: number
  body: ExprNode
}

/** Call a function with positional arguments. */
export interface CallNode {
  op: 'call'
  callee: ExprNode
  args: ExprNode[]
}

/** Session-level delay node — `args[0]` is the value to delay one sample. */
export interface DelayNode {
  op: 'delay'
  args: [ExprNode]
  init?: number
  id?: string
}

/** Gateable subgraph wrapper emitted by the flattener. */
export interface SourceTagNode {
  op: 'sourceTag'
  source_instance: string
  gate_expr: ExprNode
  expr: ExprNode
  on_skip?: ExprNode
}

// Compile-time combinators (lowered before flatten). All have a `body`
// child; some have additional ExprNode children (init, over, a, b).

export interface GenerateNode { op: 'generate'; count: number; var: string; body: ExprNode }
export interface IterateNode  { op: 'iterate'; count: number; var: string; init: ExprNode; body: ExprNode }
export interface FoldNode     { op: 'fold'; over: ExprNode; init: ExprNode; acc: string; elem: string; body: ExprNode }
export interface ScanNode     { op: 'scan'; over: ExprNode; init: ExprNode; acc: string; elem: string; body: ExprNode }
export interface Map2Node     { op: 'map2'; over: ExprNode; elem: string; body: ExprNode }
export interface ZipWithNode  { op: 'zipWith'; a: ExprNode; b: ExprNode; x: string; y: string; body: ExprNode }
export interface ChainNode    { op: 'chain'; count: number; var: string; init: ExprNode; body: ExprNode }
export interface StrConcatNode { op: 'strConcat'; parts: ExprNode[] }
export interface GenerateDeclsNode { op: 'generateDecls'; count: number; var: string; decls: ExprNode[] }

/** All named-children ops in a single union for convenience. */
export type NamedChildrenNode =
  | TagNode | MatchNode | LetNode | FunctionNode | CallNode
  | DelayNode | SourceTagNode
  | GenerateNode | IterateNode | FoldNode | ScanNode
  | Map2Node | ZipWithNode | ChainNode | StrConcatNode | GenerateDeclsNode

// ── Leaf ops (no children) ──────────────────────────────────────────────

/** Pre-slottify input ref: `{op:'input', name}`. Post-slottify: `{op:'input', id}`. */
export interface InputNode { op: 'input'; id?: number; name?: string }

/** Pre-slottify register ref: `{op:'reg', name}`. Post-slottify: `{op:'reg', id}`. */
export interface RegRefNode { op: 'reg'; id?: number; name?: string }

/** Pre-slottify delay reference: `{op:'delayRef', id: 'name'}`. */
export interface DelayRefNode { op: 'delayRef'; id: string }

/** Post-slottify delay value read: `{op:'delayValue', node_id: N}`. */
export interface DelayValueNode { op: 'delayValue'; node_id: number }

/** Pre-slottify nested-output ref: `{op:'nestedOut', ref, output}`. */
export interface NestedOutNode { op: 'nestedOut'; ref: string; output: string | number }

/** Post-slottify nested-output ref. */
export interface NestedOutputNode { op: 'nestedOutput'; node_id: number; output_id: number }

/** Combinator-introduced binding placeholder; resolved at lowering time. */
export interface BindingNode { op: 'binding'; name: string }

/** Generic-program type-parameter placeholder; resolved at specialization. */
export interface TypeParamNode { op: 'typeParam'; name: string }

/** Sample-rate constant. */
export interface SampleRateNode { op: 'sampleRate' }

/** Current sample-index counter. */
export interface SampleIndexNode { op: 'sampleIndex' }

/** Smoothed control parameter handle (FFI). */
export interface SmoothedParamNode { op: 'smoothedParam'; _ptr: true; _handle: unknown }

/** One-shot trigger control parameter handle (FFI). */
export interface TriggerParamNode { op: 'triggerParam'; _ptr: true; _handle: unknown }

/** Typed scalar literal emitted by specialize.ts after type-arg substitution. */
export interface ConstNode {
  op: 'const'
  val: number | boolean
  type?: ScalarKind
}

/** All leaf ops in a single union. */
export type LeafNode =
  | InputNode | RegRefNode | DelayRefNode | DelayValueNode
  | NestedOutNode | NestedOutputNode | BindingNode | TypeParamNode
  | SampleRateNode | SampleIndexNode
  | SmoothedParamNode | TriggerParamNode | ConstNode

// ── Decl ops (top-level only — appear at decl/assign positions) ─────────

/** Wiring ref to an instance output. May project a sum-type variant field. */
export interface RefNode {
  op: 'ref'
  instance: string
  output: string | number
  project?: { variant: string; field: string }
}

/** Top-level instance declaration. */
export interface InstanceDeclNode {
  op: 'instanceDecl'
  name: string
  program: string
  inputs?: Record<string, ExprNode>
  type_args?: Record<string, number | ExprNode>
  gateable?: boolean
  gate_input?: ExprNode
}

/** Register declaration with optional initializer and type annotation. */
export interface RegDeclNode {
  op: 'regDecl'
  name: string
  init?: ExprNode
  type?: string
}

/** Delay declaration — sum-typed delays decompose into multiple scalar slots. */
export interface DelayDeclNode {
  op: 'delayDecl'
  name: string
  init?: ExprNode | number
  update?: ExprNode
  type?: string
}

/** Inline subprogram declaration. */
export interface ProgramDeclNode {
  op: 'programDecl'
  name: string
  program?: ExprNode
}

/** Output port assignment in a program body's `assigns` list. */
export interface OutputAssignNode {
  op: 'outputAssign'
  name: string
  expr?: ExprNode
}

/** Next-state assignment for a register or delay register. */
export interface NextUpdateNode {
  op: 'nextUpdate'
  target: { kind: 'reg' | 'delay'; name: string }
  expr?: ExprNode
}

/** Block: a list of decls + assigns inside a program body. */
export interface ProgramBlockNode {
  op: 'block'
  decls?: ExprNode[]
  assigns?: ExprNode[]
  value?: ExprNode | null
}

/** Top-level program node. The body is always a ProgramBlockNode. */
export interface ProgramOpNode {
  op: 'program'
  name: string
  type_params?: Record<string, { type: 'int'; default?: number }>
  sample_rate?: number
  breaks_cycles?: boolean
  ports?: unknown  // ProgramPorts; defined in program.ts
  body: ProgramBlockNode
}

/** All top-level decl/assign/structural ops. */
export type DeclNode =
  | RefNode
  | InstanceDeclNode | RegDeclNode | DelayDeclNode | ProgramDeclNode
  | OutputAssignNode | NextUpdateNode
  | ProgramBlockNode | ProgramOpNode

// ── The closed parametric-arity discriminated union ─────────────────────

/** Closed discriminated union of every op kind. Replaces the bag-of-fields
 *  `{op: string; [k]: unknown}` once walkers are migrated. Adding a new op
 *  forces touching this union and every exhaustive `mapChildren` switch. */
export type ExprOpNodeStrict =
  // Op<N> family — most ops factor through here.
  | UnaryNode | BinaryNode | TernaryNode
  // Inline array (variadic but uses `items`).
  | ArrayNode
  // Op<N> + extras — same-shape traversal, extra metadata fields.
  | ReshapeNode | TransposeNode | SliceNode | ReduceNode
  | BroadcastToNode | IndexNode | MatmulNode | MapNode
  // Construction ops with shape/values.
  | ZerosNode | OnesNode | FillNode | ArrayLiteralNode | MatrixNode
  // Named-children ops — bespoke per-op interfaces.
  | NamedChildrenNode
  // Leaves.
  | LeafNode
  // Top-level decls.
  | DeclNode

// ---------- SignalExpr ----------

export class SignalExpr {
  /** JSON-serializable expression tree. */
  _node: ExprNode
  /** Static shape, if known at module-definition time. undefined = scalar or unknown. */
  readonly shape: number[] | undefined

  private constructor(node: ExprNode, shape?: number[]) {
    this._node = node
    this.shape = shape
  }

  static fromNode(node: ExprNode, shape?: number[]): SignalExpr {
    return new SignalExpr(node, shape)
  }

  /** @deprecated Alias for fromNode — old code used fromHandle(handle, node). */
  static fromHandle(_handle: unknown, node: ExprNode): SignalExpr {
    return new SignalExpr(node)
  }

  /** Index into an array expression: expr[idx] */
  at(idx: ExprCoercible): SignalExpr {
    const i = coerce(idx)
    return new SignalExpr({ op: 'index', args: [this._node, i._node] })
  }

  /** Reshape this array to a new shape. */
  reshape(newShape: number[]): SignalExpr {
    return new SignalExpr({ op: 'reshape', args: [this._node], shape: newShape })
  }

  /** Transpose a 2D array. */
  transpose(): SignalExpr {
    return new SignalExpr({ op: 'transpose', args: [this._node] })
  }

  /** Slice along an axis: [start, end). */
  slice(axis: number, start: number, end: number): SignalExpr {
    return new SignalExpr({ op: 'slice', args: [this._node], axis, start, end })
  }

  /** Reduce along an axis with an associative op ('add', 'mul', 'min', 'max'). */
  reduce(axis: number, reduceOp: string): SignalExpr {
    return new SignalExpr({ op: 'reduce', args: [this._node], axis, reduce_op: reduceOp })
  }

  /** Sum all elements (reduce over axis 0 with 'add'). */
  sum(axis = 0): SignalExpr {
    return this.reduce(axis, 'add')
  }
}

// ---------- Coercion ----------

export type ExprCoercible = SignalExpr | number | boolean | ExprCoercible[]

/** Convert a scalar, boolean, array, or SignalExpr to a SignalExpr. */
export function coerce(value: ExprCoercible): SignalExpr {
  if (value instanceof SignalExpr) return value
  if (typeof value === 'boolean') return SignalExpr.fromNode(value)
  if (typeof value === 'number') return SignalExpr.fromNode(value)
  if (Array.isArray(value)) return arrayPack(value)
  throw new TypeError(`Cannot coerce ${typeof value} to SignalExpr`)
}

// ---------- Internal helpers ----------

function propagateBinaryShape(l: SignalExpr, r: SignalExpr): number[] | undefined {
  if (l.shape !== undefined && r.shape !== undefined) {
    return broadcastShapes(l.shape, r.shape) ?? undefined
  }
  if (l.shape !== undefined) return l.shape
  if (r.shape !== undefined) return r.shape
  return undefined
}

function binary(opName: string, lhs: ExprCoercible, rhs: ExprCoercible): SignalExpr {
  const l = coerce(lhs)
  const r = coerce(rhs)
  return SignalExpr.fromNode({ op: opName, args: [l._node, r._node] }, propagateBinaryShape(l, r))
}

function unary(opName: string, operand: ExprCoercible): SignalExpr {
  const o = coerce(operand)
  return SignalExpr.fromNode({ op: opName, args: [o._node] }, o.shape)
}

// ---------- Arithmetic ----------

export const add      = (lhs: ExprCoercible, rhs: ExprCoercible) => binary('add',       lhs, rhs)
export const sub      = (lhs: ExprCoercible, rhs: ExprCoercible) => binary('sub',       lhs, rhs)
export const mul      = (lhs: ExprCoercible, rhs: ExprCoercible) => binary('mul',       lhs, rhs)
export const div      = (lhs: ExprCoercible, rhs: ExprCoercible) => binary('div',       lhs, rhs)
export const floorDiv = (lhs: ExprCoercible, rhs: ExprCoercible) => binary('floorDiv', lhs, rhs)
export const mod      = (lhs: ExprCoercible, rhs: ExprCoercible) => binary('mod',       lhs, rhs)
export const matmul = (
  lhs: ExprCoercible,
  rhs: ExprCoercible,
  shape_a: [number, number],
  shape_b: [number, number],
  element_type: ScalarKind = 'float',
): SignalExpr => {
  if (shape_a[1] !== shape_b[0])
    throw new Error(`matmul: inner dimensions must match (${shape_a[1]} ≠ ${shape_b[0]})`)
  const l = coerce(lhs)
  const r = coerce(rhs)
  const [M] = shape_a
  const [, N] = shape_b
  return SignalExpr.fromNode(
    { op: 'matmul', args: [l._node, r._node], shape_a, shape_b, element_type },
    [M, N],
  )
}

// ---------- Comparison ----------

export const lt  = (lhs: ExprCoercible, rhs: ExprCoercible) => binary('lt',  lhs, rhs)
export const lte = (lhs: ExprCoercible, rhs: ExprCoercible) => binary('lte', lhs, rhs)
export const gt  = (lhs: ExprCoercible, rhs: ExprCoercible) => binary('gt',  lhs, rhs)
export const gte = (lhs: ExprCoercible, rhs: ExprCoercible) => binary('gte', lhs, rhs)
export const eq  = (lhs: ExprCoercible, rhs: ExprCoercible) => binary('eq',  lhs, rhs)
export const neq = (lhs: ExprCoercible, rhs: ExprCoercible) => binary('neq', lhs, rhs)

// ---------- Bitwise ----------

export const bitAnd  = (lhs: ExprCoercible, rhs: ExprCoercible) => binary('bitAnd', lhs, rhs)
export const bitOr   = (lhs: ExprCoercible, rhs: ExprCoercible) => binary('bitOr',  lhs, rhs)
export const bitXor  = (lhs: ExprCoercible, rhs: ExprCoercible) => binary('bitXor', lhs, rhs)
export const lshift  = (lhs: ExprCoercible, rhs: ExprCoercible) => binary('lshift',  lhs, rhs)
export const rshift  = (lhs: ExprCoercible, rhs: ExprCoercible) => binary('rshift',  lhs, rhs)
export const bitNot  = (operand: ExprCoercible) => unary('bitNot', operand)

// ---------- Unary / math ----------

export const neg        = (operand: ExprCoercible) => unary('neg', operand)
export const abs_       = (operand: ExprCoercible) => unary('abs', operand)
export const floatExponent = (operand: ExprCoercible) => unary('floatExponent', operand)
export const ldexp      = (lhs: ExprCoercible, rhs: ExprCoercible) => binary('ldexp', lhs, rhs)
export const logicalNot = (operand: ExprCoercible) => unary('not', operand)

// Scalar-type cast ops. Truncate-toward-zero (FPToSI) for to_int — not floor.
export const toInt   = (operand: ExprCoercible) => unary('toInt',   operand)
export const toBool  = (operand: ExprCoercible) => unary('toBool',  operand)
export const toFloat = (operand: ExprCoercible) => unary('toFloat', operand)
export const logicalAnd = (lhs: ExprCoercible, rhs: ExprCoercible) => binary('and', lhs, rhs)
export const logicalOr  = (lhs: ExprCoercible, rhs: ExprCoercible) => binary('or',  lhs, rhs)

export function clamp(value: ExprCoercible, lo: ExprCoercible, hi: ExprCoercible): SignalExpr {
  const v = coerce(value)
  const l = coerce(lo)
  const h = coerce(hi)
  const shape = propagateBinaryShape(v, l) ?? v.shape ?? l.shape ?? h.shape
  return SignalExpr.fromNode({ op: 'clamp', args: [v._node, l._node, h._node] }, shape)
}

export function select(cond: ExprCoercible, thenVal: ExprCoercible, elseVal: ExprCoercible): SignalExpr {
  const c = coerce(cond)
  const t = coerce(thenVal)
  const e = coerce(elseVal)
  const shape = propagateBinaryShape(t, e) ?? t.shape ?? e.shape ?? c.shape
  return SignalExpr.fromNode({ op: 'select', args: [c._node, t._node, e._node] }, shape)
}

// ---------- Array operations ----------

export function arrayPack(values: ExprCoercible[]): SignalExpr {
  const items = values.map(coerce)
  return SignalExpr.fromNode(items.map(e => e._node), [items.length])
}

export function arraySet(arrExpr: ExprCoercible, idx: ExprCoercible, val: ExprCoercible): SignalExpr {
  const a = coerce(arrExpr)
  const i = coerce(idx)
  const v = coerce(val)
  return SignalExpr.fromNode({ op: 'arraySet', args: [a._node, i._node, v._node] }, a.shape)
}

/** Build a matrix literal expression from a row-major 2D array of numbers. */
export function matrix(rows: number[][]): SignalExpr {
  return SignalExpr.fromNode({ op: 'matrix', rows })
}

// ---------- First-class array operations (static shapes) ----------

/**
 * Typed array literal with explicit shape.
 * Values are provided in row-major order and must match product(shape).
 */
export function arrayLiteral(shape: number[], values: ExprCoercible[]): SignalExpr {
  const items = values.map(v => coerce(v)._node)
  return SignalExpr.fromNode({ op: 'arrayLiteral', shape, values: items }, shape)
}

/** Create an array filled with zeros. */
export function zeros(shape: number[]): SignalExpr {
  return SignalExpr.fromNode({ op: 'zeros', shape }, shape)
}

/** Create an array filled with ones. */
export function ones(shape: number[]): SignalExpr {
  return SignalExpr.fromNode({ op: 'ones', shape }, shape)
}

/** Create an array filled with a constant value. */
export function fill(shape: number[], value: ExprCoercible): SignalExpr {
  return SignalExpr.fromNode({ op: 'fill', shape, value: coerce(value)._node }, shape)
}

/** Reshape an array to a new shape (total elements must match). */
export function reshape(arr: ExprCoercible, newShape: number[]): SignalExpr {
  return SignalExpr.fromNode({ op: 'reshape', args: [coerce(arr)._node], shape: newShape })
}

/** Transpose a 2D array (swap axes). */
export function transpose(arr: ExprCoercible): SignalExpr {
  return unary('transpose', arr)
}

/**
 * Slice an array along a dimension.
 * Returns elements [start, end) along the given axis.
 */
export function slice(arr: ExprCoercible, axis: number, start: number, end: number): SignalExpr {
  return SignalExpr.fromNode({ op: 'slice', args: [coerce(arr)._node], axis, start, end })
}

/**
 * Reduce an array along an axis using an associative operation.
 * reduceOp is one of: 'add', 'mul', 'min', 'max'.
 */
export function reduce(arr: ExprCoercible, axis: number, reduceOp: string): SignalExpr {
  return SignalExpr.fromNode({ op: 'reduce', args: [coerce(arr)._node], axis, reduce_op: reduceOp })
}

/** Explicitly broadcast an array to a target shape. */
export function broadcastTo(arr: ExprCoercible, shape: number[]): SignalExpr {
  return SignalExpr.fromNode({ op: 'broadcastTo', args: [coerce(arr)._node], shape }, shape)
}

/** Map a function over array elements: map(fn, arr) applies fn to each element. */
export function mapArray(fn: (elem: SignalExpr) => SignalExpr, arr: ExprCoercible): SignalExpr {
  // Build as: map(function(1, body), arr) — function takes 1 param (the element)
  const param = inputExpr(0)
  const body = fn(param)
  return SignalExpr.fromNode({
    op: 'map',
    callee: { op: 'function', param_count: 1, body: body._node },
    args: [coerce(arr)._node],
  })
}

// ---------- Function expressions ----------

export function exprFunction(paramCount: number, body: SignalExpr): SignalExpr {
  return SignalExpr.fromNode({ op: 'function', param_count: paramCount, body: body._node })
}

export function exprCall(fn: SignalExpr, args: ExprCoercible[]): SignalExpr {
  const coerced = args.map(coerce)
  return SignalExpr.fromNode({ op: 'call', callee: fn._node, args: coerced.map(e => e._node) })
}

// ---------- Sum-type wiring expression builders ----------

/**
 * Construct a variant value of a sum type (coproduct injection).
 * `payload` maps payload-field names to their ExprNode values; pass undefined
 * for nullary variants.
 */
export function tag(
  typeName: string,
  variant: string,
  payload?: Record<string, ExprCoercible>,
): SignalExpr {
  const node: { op: string; type: string; variant: string; payload?: Record<string, ExprNode> } = {
    op: 'tag', type: typeName, variant,
  }
  if (payload !== undefined) {
    const coerced: Record<string, ExprNode> = {}
    for (const [k, v] of Object.entries(payload)) coerced[k] = coerce(v)._node
    node.payload = coerced
  }
  return SignalExpr.fromNode(node)
}

/**
 * Match on a sum-typed scrutinee, dispatching to per-variant arms (coproduct
 * elimination via the universal property). Each arm specifies an optional
 * `bind` (string or string[]) naming the locally-available payload values,
 * plus a `body` expression that produces the arm's result. All arm bodies
 * must produce the same type.
 */
export interface MatchArm {
  bind?: string | string[]
  body: ExprCoercible
}

export function match(
  typeName: string,
  scrutinee: ExprCoercible,
  arms: Record<string, MatchArm>,
): SignalExpr {
  const armsNode: Record<string, { bind?: string | string[]; body: ExprNode }> = {}
  for (const [variant, arm] of Object.entries(arms)) {
    const armNode: { bind?: string | string[]; body: ExprNode } = { body: coerce(arm.body)._node }
    if (arm.bind !== undefined) armNode.bind = arm.bind
    armsNode[variant] = armNode
  }
  return SignalExpr.fromNode({
    op: 'match',
    type: typeName,
    scrutinee: coerce(scrutinee)._node,
    arms: armsNode,
  })
}

// ---------- Leaf node constructors ----------

export function sampleRate(): SignalExpr {
  return SignalExpr.fromNode({ op: 'sampleRate' })
}

export function sampleIndex(): SignalExpr {
  return SignalExpr.fromNode({ op: 'sampleIndex' })
}

export function inputExpr(inputId: number): SignalExpr {
  return SignalExpr.fromNode({ op: 'input', id: inputId })
}

export function registerExpr(regId: number): SignalExpr {
  return SignalExpr.fromNode({ op: 'reg', id: regId })
}

export function refExpr(instanceName: string, outputId: number): SignalExpr {
  return SignalExpr.fromNode({ op: 'ref', instance: instanceName, output: outputId })
}

export function nestedOutputExpr(nodeId: number, outputId: number): SignalExpr {
  return SignalExpr.fromNode({ op: 'nestedOutput', node_id: nodeId, output_id: outputId })
}

export function delayValueExpr(nodeId: number): SignalExpr {
  return SignalExpr.fromNode({ op: 'delayValue', node_id: nodeId })
}

/** Create a smoothed-param expression node for use in wiring expressions. */
export function paramExpr(paramHandle: unknown): SignalExpr {
  return SignalExpr.fromNode({ op: 'smoothedParam', _ptr: true, _handle: paramHandle })
}

/** Create a trigger-param expression node for use in wiring expressions. */
export function triggerParamExpr(paramHandle: unknown): SignalExpr {
  return SignalExpr.fromNode({ op: 'triggerParam', _ptr: true, _handle: paramHandle })
}

// ─────────────────────────────────────────────────────────────
// Expression validation
// ─────────────────────────────────────────────────────────────

const BINARY_OPS = new Set([
  'add', 'sub', 'mul', 'div', 'floorDiv', 'mod',
  'lt', 'lte', 'gt', 'gte', 'eq', 'neq',
  'bitAnd', 'bitOr', 'bitXor', 'lshift', 'rshift',
  'and', 'or',
  'ldexp',
])

const UNARY_OPS = new Set([
  'neg', 'abs', 'not', 'bitNot',
  'sqrt', 'floor', 'ceil', 'round',
  'floatExponent',
  'toInt', 'toBool', 'toFloat',
])

const TERNARY_OPS = new Set(['clamp', 'select'])

const LEAF_OPS = new Set([
  'input', 'reg', 'sampleRate', 'sampleIndex',
  'smoothedParam', 'triggerParam',
  'delayValue', 'delayRef',
  'nestedOutput', 'nestedOut',
  'binding',
])

/**
 * Validate an ExprNode tree structure. Throws on the first error with a path
 * describing where in the tree the problem is.
 *
 * This is a structural check — it verifies that ops have the right fields
 * and arg counts, not that module/output names resolve correctly.
 */
export function validateExpr(node: ExprNode, path = 'expr'): void {
  if (typeof node === 'number' || typeof node === 'boolean') return
  if (Array.isArray(node)) {
    for (let i = 0; i < node.length; i++) validateExpr(node[i], `${path}[${i}]`)
    return
  }
  if (typeof node !== 'object' || node === null) {
    throw new Error(`${path}: expected number, boolean, array, or {op: ...}, got ${typeof node}`)
  }

  const obj = node as Record<string, unknown>
  if (typeof obj.op !== 'string') {
    throw new Error(`${path}: missing or non-string 'op' field (got ${JSON.stringify(obj).slice(0, 100)})`)
  }
  const op = obj.op

  // Binary ops: require args array of length 2
  if (BINARY_OPS.has(op)) {
    if (!Array.isArray(obj.args)) {
      throw new Error(`${path}: '${op}' requires 'args' array, got ${obj.args === undefined ? 'undefined' : typeof obj.args}. Use {op: "${op}", args: [left, right]}`)
    }
    if ((obj.args as unknown[]).length !== 2) {
      throw new Error(`${path}: '${op}' requires exactly 2 args, got ${(obj.args as unknown[]).length}`)
    }
    validateExpr((obj.args as ExprNode[])[0], `${path}.args[0]`)
    validateExpr((obj.args as ExprNode[])[1], `${path}.args[1]`)
    return
  }

  // Unary ops: require args array of length 1
  if (UNARY_OPS.has(op)) {
    if (!Array.isArray(obj.args)) {
      throw new Error(`${path}: '${op}' requires 'args' array, got ${obj.args === undefined ? 'undefined' : typeof obj.args}. Use {op: "${op}", args: [x]}`)
    }
    if ((obj.args as unknown[]).length !== 1) {
      throw new Error(`${path}: '${op}' requires exactly 1 arg, got ${(obj.args as unknown[]).length}`)
    }
    validateExpr((obj.args as ExprNode[])[0], `${path}.args[0]`)
    return
  }

  // Ternary ops: require args array of length 3
  if (TERNARY_OPS.has(op)) {
    if (!Array.isArray(obj.args)) {
      throw new Error(`${path}: '${op}' requires 'args' array. Use {op: "${op}", args: [a, b, c]}`)
    }
    if ((obj.args as unknown[]).length !== 3) {
      throw new Error(`${path}: '${op}' requires exactly 3 args, got ${(obj.args as unknown[]).length}`)
    }
    for (let i = 0; i < 3; i++) validateExpr((obj.args as ExprNode[])[i], `${path}.args[${i}]`)
    return
  }

  // index / array_set: args array
  if (op === 'index') {
    if (!Array.isArray(obj.args) || (obj.args as unknown[]).length !== 2) {
      throw new Error(`${path}: 'index' requires args: [array, index]`)
    }
    validateExpr((obj.args as ExprNode[])[0], `${path}.args[0]`)
    validateExpr((obj.args as ExprNode[])[1], `${path}.args[1]`)
    return
  }
  if (op === 'arraySet') {
    if (!Array.isArray(obj.args) || (obj.args as unknown[]).length !== 3) {
      throw new Error(`${path}: 'arraySet' requires args: [array, index, value]`)
    }
    for (let i = 0; i < 3; i++) validateExpr((obj.args as ExprNode[])[i], `${path}.args[${i}]`)
    return
  }

  // ref: requires instance and output
  if (op === 'ref') {
    if (typeof obj.instance !== 'string') {
      throw new Error(`${path}: 'ref' requires 'instance' (string), got ${typeof obj.instance}. Use {op: "ref", instance: "name", output: "port"}`)
    }
    if (obj.output === undefined) {
      throw new Error(`${path}: 'ref' requires 'output'. Use {op: "ref", instance: "${obj.instance}", output: "port_name"}`)
    }
    return
  }

  // array / array_pack: recurse into items/args
  if (op === 'array' && Array.isArray(obj.items)) {
    for (let i = 0; i < (obj.items as unknown[]).length; i++) {
      validateExpr((obj.items as ExprNode[])[i], `${path}.items[${i}]`)
    }
    return
  }
  if (op === 'arrayPack' && Array.isArray(obj.args)) {
    for (let i = 0; i < (obj.args as unknown[]).length; i++) {
      validateExpr((obj.args as ExprNode[])[i], `${path}.args[${i}]`)
    }
    return
  }

  // matmul: args array of length 2
  if (op === 'matmul') {
    if (!Array.isArray(obj.args) || (obj.args as unknown[]).length !== 2) {
      throw new Error(`${path}: 'matmul' requires args: [a, b]`)
    }
    validateExpr((obj.args as ExprNode[])[0], `${path}.args[0]`)
    validateExpr((obj.args as ExprNode[])[1], `${path}.args[1]`)
    return
  }

  // call: callee + args
  if (op === 'call') {
    if (obj.callee !== undefined) validateExpr(obj.callee as ExprNode, `${path}.callee`)
    if (Array.isArray(obj.args)) {
      for (let i = 0; i < (obj.args as unknown[]).length; i++) {
        validateExpr((obj.args as ExprNode[])[i], `${path}.args[${i}]`)
      }
    }
    return
  }

  // Leaf ops: no recursion needed
  if (LEAF_OPS.has(op)) return

  // Combinators with nested expr fields
  if (op === 'let') {
    if (typeof obj.bind === 'object' && obj.bind !== null) {
      for (const [k, v] of Object.entries(obj.bind as Record<string, unknown>)) {
        validateExpr(v as ExprNode, `${path}.bind.${k}`)
      }
    }
    if (obj.in !== undefined) validateExpr(obj.in as ExprNode, `${path}.in`)
    return
  }
  if (op === 'generate' || op === 'chain' || op === 'iterate') {
    if (obj.init !== undefined) validateExpr(obj.init as ExprNode, `${path}.init`)
    if (obj.body !== undefined) validateExpr(obj.body as ExprNode, `${path}.body`)
    return
  }
  if (op === 'fold' || op === 'scan') {
    if (obj.arr !== undefined) validateExpr(obj.arr as ExprNode, `${path}.arr`)
    if (obj.init !== undefined) validateExpr(obj.init as ExprNode, `${path}.init`)
    if (obj.body !== undefined) validateExpr(obj.body as ExprNode, `${path}.body`)
    return
  }
  if (op === 'map2') {
    if (obj.arr !== undefined) validateExpr(obj.arr as ExprNode, `${path}.arr`)
    if (obj.body !== undefined) validateExpr(obj.body as ExprNode, `${path}.body`)
    return
  }
  if (op === 'zipWith') {
    if (obj.a !== undefined) validateExpr(obj.a as ExprNode, `${path}.a`)
    if (obj.b !== undefined) validateExpr(obj.b as ExprNode, `${path}.b`)
    if (obj.body !== undefined) validateExpr(obj.body as ExprNode, `${path}.body`)
    return
  }

  // ── Sum-type wiring expressions ───────────────────────────────────────────
  // tag (coproduct injection): {op, type, variant, payload?: Record<field, ExprNode>}
  if (op === 'tag') {
    if (typeof obj.type !== 'string')
      throw new Error(`${path}: 'tag' requires type: string (sum type name)`)
    if (typeof obj.variant !== 'string')
      throw new Error(`${path}: 'tag' requires variant: string`)
    if (obj.payload !== undefined) {
      if (typeof obj.payload !== 'object' || obj.payload === null || Array.isArray(obj.payload))
        throw new Error(`${path}: 'tag' payload must be an object {fieldName: ExprNode}`)
      for (const [k, v] of Object.entries(obj.payload as Record<string, unknown>))
        validateExpr(v as ExprNode, `${path}.payload.${k}`)
    }
    return
  }

  // match (coproduct elimination): {op, type, scrutinee, arms: Record<variantName, MatchArm>}
  // MatchArm = {bind?: string | string[], body: ExprNode}
  if (op === 'match') {
    if (typeof obj.type !== 'string')
      throw new Error(`${path}: 'match' requires type: string (sum type name)`)
    if (obj.scrutinee === undefined)
      throw new Error(`${path}: 'match' requires scrutinee: ExprNode`)
    validateExpr(obj.scrutinee as ExprNode, `${path}.scrutinee`)
    if (typeof obj.arms !== 'object' || obj.arms === null || Array.isArray(obj.arms))
      throw new Error(`${path}: 'match' arms must be an object {variantName: {bind?, body}}`)
    const arms = obj.arms as Record<string, unknown>
    if (Object.keys(arms).length === 0)
      throw new Error(`${path}: 'match' requires at least one arm`)
    for (const [variantName, arm] of Object.entries(arms)) {
      if (typeof arm !== 'object' || arm === null || Array.isArray(arm))
        throw new Error(`${path}.arms.${variantName}: arm must be an object {bind?, body}`)
      const a = arm as Record<string, unknown>
      if (a.bind !== undefined) {
        if (typeof a.bind !== 'string' && !Array.isArray(a.bind))
          throw new Error(`${path}.arms.${variantName}.bind: must be string or string[], got ${typeof a.bind}`)
        if (Array.isArray(a.bind))
          for (let i = 0; i < a.bind.length; i++)
            if (typeof a.bind[i] !== 'string')
              throw new Error(`${path}.arms.${variantName}.bind[${i}]: must be a string`)
      }
      if (a.body === undefined)
        throw new Error(`${path}.arms.${variantName}: missing required 'body' field`)
      validateExpr(a.body as ExprNode, `${path}.arms.${variantName}.body`)
    }
    return
  }

  // delay: args[0] is the expression to delay; init is a number; id is an optional string name
  if (op === 'delay') {
    if (!Array.isArray(obj.args) || (obj.args as unknown[]).length !== 1)
      throw new Error(`${path}: 'delay' requires args: [expr] — the expression whose value will be read next sample`)
    validateExpr((obj.args as ExprNode[])[0], `${path}.args[0]`)
    if (obj.init !== undefined && typeof obj.init !== 'number')
      throw new Error(`${path}: 'delay' init must be a number, got ${typeof obj.init}`)
    if (obj.id !== undefined && typeof obj.id !== 'string')
      throw new Error(`${path}: 'delay' id must be a string, got ${typeof obj.id}`)
    return
  }

  // ── Array construction ops ────────────────────────────────────────────────

  if (op === 'zeros' || op === 'ones') {
    if (!Array.isArray(obj.shape))
      throw new Error(`${path}: '${op}' requires shape: number[]`)
    return
  }
  if (op === 'fill') {
    if (!Array.isArray(obj.shape))
      throw new Error(`${path}: 'fill' requires shape: number[]`)
    if (obj.value === undefined)
      throw new Error(`${path}: 'fill' requires value: ExprNode`)
    validateExpr(obj.value as ExprNode, `${path}.value`)
    return
  }
  if (op === 'arrayLiteral') {
    if (!Array.isArray(obj.values))
      throw new Error(`${path}: 'arrayLiteral' requires values: ExprNode[]`)
    for (let i = 0; i < (obj.values as unknown[]).length; i++)
      validateExpr((obj.values as ExprNode[])[i], `${path}.values[${i}]`)
    return
  }

  // ── Array manipulation ops ────────────────────────────────────────────────

  if (op === 'reshape' || op === 'transpose') {
    if (!Array.isArray(obj.args) || (obj.args as unknown[]).length < 1)
      throw new Error(`${path}: '${op}' requires args: [arr]`)
    validateExpr((obj.args as ExprNode[])[0], `${path}.args[0]`)
    return
  }
  if (op === 'slice') {
    if (!Array.isArray(obj.args) || (obj.args as unknown[]).length < 1)
      throw new Error(`${path}: 'slice' requires args: [arr]`)
    validateExpr((obj.args as ExprNode[])[0], `${path}.args[0]`)
    if (typeof obj.start !== 'number') throw new Error(`${path}: 'slice' requires start: number`)
    if (typeof obj.end   !== 'number') throw new Error(`${path}: 'slice' requires end: number`)
    return
  }
  if (op === 'reduce') {
    if (!Array.isArray(obj.args) || (obj.args as unknown[]).length < 1)
      throw new Error(`${path}: 'reduce' requires args: [arr]`)
    validateExpr((obj.args as ExprNode[])[0], `${path}.args[0]`)
    if (typeof obj.reduce_op !== 'string')
      throw new Error(`${path}: 'reduce' requires reduce_op: string`)
    return
  }
  if (op === 'broadcastTo') {
    if (!Array.isArray(obj.args) || (obj.args as unknown[]).length < 1)
      throw new Error(`${path}: 'broadcastTo' requires args: [arr]`)
    validateExpr((obj.args as ExprNode[])[0], `${path}.args[0]`)
    return
  }
  if (op === 'map') {
    if (!Array.isArray(obj.args) || (obj.args as unknown[]).length < 1)
      throw new Error(`${path}: 'map' requires args: [arr]`)
    validateExpr((obj.args as ExprNode[])[0], `${path}.args[0]`)
    if (obj.callee !== undefined) validateExpr(obj.callee as ExprNode, `${path}.callee`)
    return
  }

  // matrix, function — structural pass-through
  if (op === 'matrix' || op === 'function') return

  // ── Unified IR: program / block / decls / assigns ────────────────────────
  // See `tropical_program_2` (design/surface-syntax.md). These nodes let a
  // program be represented as a single ExprNode tree whose body is a block
  // of declarations and assignments.

  if (op === 'program') {
    if (typeof obj.name !== 'string')
      throw new Error(`${path}: 'program' requires name: string`)
    if (obj.body !== undefined) validateExpr(obj.body as ExprNode, `${path}.body`)
    return
  }

  if (op === 'block') {
    if (obj.decls !== undefined) {
      if (!Array.isArray(obj.decls))
        throw new Error(`${path}: 'block' decls must be an array`)
      for (let i = 0; i < (obj.decls as unknown[]).length; i++)
        validateExpr((obj.decls as ExprNode[])[i], `${path}.decls[${i}]`)
    }
    if (obj.assigns !== undefined) {
      if (!Array.isArray(obj.assigns))
        throw new Error(`${path}: 'block' assigns must be an array`)
      for (let i = 0; i < (obj.assigns as unknown[]).length; i++)
        validateExpr((obj.assigns as ExprNode[])[i], `${path}.assigns[${i}]`)
    }
    if (obj.value !== undefined && obj.value !== null)
      validateExpr(obj.value as ExprNode, `${path}.value`)
    return
  }

  if (op === 'regDecl') {
    if (typeof obj.name !== 'string')
      throw new Error(`${path}: 'regDecl' requires name: string`)
    if (obj.init !== undefined) validateExpr(obj.init as ExprNode, `${path}.init`)
    return
  }

  if (op === 'delayDecl') {
    if (typeof obj.name !== 'string')
      throw new Error(`${path}: 'delayDecl' requires name: string`)
    if (obj.update !== undefined) validateExpr(obj.update as ExprNode, `${path}.update`)
    if (obj.init !== undefined) validateExpr(obj.init as ExprNode, `${path}.init`)
    // Optional `type` field — when present and naming a registered sum type,
    // the delay holds a bundle of scalar slots (one per (variant, field) pair
    // plus a discriminator). Init must then be a constant `tag` expression.
    // The structural check here only verifies the field's shape; sum-name
    // resolution and constant-fold validation happen at loadProgramDef time.
    if (obj.type !== undefined && typeof obj.type !== 'string')
      throw new Error(`${path}: 'delayDecl' type must be a string (registered sum/struct/scalar name)`)
    return
  }

  if (op === 'instanceDecl') {
    if (typeof obj.name !== 'string')
      throw new Error(`${path}: 'instanceDecl' requires name: string`)
    if (typeof obj.program !== 'string')
      throw new Error(`${path}: 'instanceDecl' requires program: string`)
    if (obj.inputs !== undefined && typeof obj.inputs === 'object' && obj.inputs !== null) {
      for (const [k, v] of Object.entries(obj.inputs as Record<string, unknown>))
        validateExpr(v as ExprNode, `${path}.inputs.${k}`)
    }
    if (obj.gateable !== undefined && typeof obj.gateable !== 'boolean')
      throw new Error(`${path}: 'instanceDecl' gateable must be boolean`)
    if (obj.gate_input !== undefined)
      validateExpr(obj.gate_input as ExprNode, `${path}.gate_input`)
    return
  }

  if (op === 'programDecl') {
    if (typeof obj.name !== 'string')
      throw new Error(`${path}: 'programDecl' requires name: string`)
    if (obj.program !== undefined) validateExpr(obj.program as ExprNode, `${path}.program`)
    return
  }

  if (op === 'outputAssign') {
    if (typeof obj.name !== 'string')
      throw new Error(`${path}: 'outputAssign' requires name: string`)
    if (obj.expr !== undefined) validateExpr(obj.expr as ExprNode, `${path}.expr`)
    return
  }

  if (op === 'nextUpdate') {
    if (typeof obj.target !== 'object' || obj.target === null)
      throw new Error(`${path}: 'nextUpdate' requires target: {kind, name}`)
    const tgt = obj.target as Record<string, unknown>
    if (tgt.kind !== 'reg' && tgt.kind !== 'delay')
      throw new Error(`${path}: 'nextUpdate' target.kind must be 'reg' or 'delay', got ${String(tgt.kind)}`)
    if (typeof tgt.name !== 'string')
      throw new Error(`${path}: 'nextUpdate' target.name must be a string`)
    if (obj.expr !== undefined) validateExpr(obj.expr as ExprNode, `${path}.expr`)
    return
  }

  // Unknown op
  throw new Error(`${path}: unknown op '${op}'`)
}
