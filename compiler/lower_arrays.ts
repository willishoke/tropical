/**
 * lower_arrays.ts — Lower first-class array operations to scalar/ArrayPack primitives.
 *
 * This pass transforms high-level array ops (zeros, ones, fill, array_literal,
 * reshape, transpose, slice, reduce, broadcast_to, map) into combinations of
 * ArrayPack, Index, and scalar arithmetic that the C++ FlatRuntime already supports.
 *
 * All shapes are static (known at compile time), so every expansion is fully unrolled.
 */

import type { ExprNode } from './expr.js'
import { shapeSize, shapeStrides } from './term.js'

// Minimal local BlockNode — mirrors program.ts's BlockNode without creating a circular import.
interface BlockNode {
  op: 'block'
  decls?: ExprNode[]
  assigns?: ExprNode[]
  value?: ExprNode | null
}

// ─────────────────────────────────────────────────────────────
// Layout table for array slot allocation
// ─────────────────────────────────────────────────────────────

export interface ArrayLayout {
  slotId: string
  shape: number[]
  strides: number[]
  totalScalars: number
}

/**
 * Lower all array operations in an expression tree.
 * Recursively walks the tree and expands array ops to primitives.
 *
 * memo: optional WeakMap for identity-based memoization. Pass a fresh WeakMap
 * at the call site to prevent O(2^N) traversal of shared DAG nodes.
 */
export function lowerArrayOps(node: ExprNode, memo?: WeakMap<object, ExprNode>): ExprNode {
  if (typeof node === 'number' || typeof node === 'boolean') return node
  if (Array.isArray(node)) return node.map(n => lowerArrayOps(n, memo))
  if (typeof node !== 'object' || node === null) return node

  if (memo) {
    const cached = memo.get(node as object)
    if (cached !== undefined) return cached
  }

  const obj = node as { op: string; [k: string]: unknown }
  let result: ExprNode

  switch (obj.op) {
    case 'array_literal':
      result = lowerArrayLiteral(obj, memo)
      break

    case 'zeros':
      result = lowerZeros(obj)
      break

    case 'ones':
      result = lowerOnes(obj)
      break

    case 'fill':
      result = lowerFill(obj, memo)
      break

    case 'reshape':
      result = lowerReshape(obj, memo)
      break

    case 'transpose':
      result = lowerTranspose(obj, memo)
      break

    case 'slice':
      result = lowerSlice(obj, memo)
      break

    case 'reduce':
      result = lowerReduce(obj, memo)
      break

    case 'broadcast_to':
      result = lowerBroadcastTo(obj, memo)
      break

    case 'map':
      result = lowerMap(obj, memo)
      break

    case 'matmul':
      result = lowerMatmul(obj, memo)
      break

    // ── Compile-time combinators ──

    case 'let':
      result = lowerLet(obj, memo)
      break

    case 'generate':
      result = lowerGenerate(obj, memo)
      break

    case 'iterate':
      result = lowerIterate(obj, memo)
      break

    case 'fold':
      result = lowerFold(obj, memo)
      break

    case 'scan':
      result = lowerScan(obj, memo)
      break

    case 'map2':
      result = lowerMap2(obj, memo)
      break

    case 'zip_with':
      result = lowerZipWith(obj, memo)
      break

    case 'chain':
      result = lowerChain(obj, memo)
      break

    // ── Sum-type wiring expressions ──
    // `tag` and `match` carry their sub-expressions inside non-op objects
    // (payload as Record<field, ExprNode>; arms as Record<variant, {bind, body}>),
    // which lowerChildren's generic walk doesn't traverse. Recurse explicitly
    // so any array ops or combinators nested inside them are lowered. The
    // tag/match node itself stays in place — actual lowering to scalar
    // bundle ops happens later, in flatten.ts where slot allocation lives.

    case 'tag':
      result = lowerTag(obj, memo)
      break

    case 'match':
      result = lowerMatch(obj, memo)
      break

    default:
      result = lowerChildren(obj, memo)
      break
  }

  // Only cache object results (not arrays/primitives) since memo is WeakMap
  if (memo && typeof result === 'object' && result !== null && !Array.isArray(result)) {
    memo.set(node as object, result)
  }

  return result
}

// ─────────────────────────────────────────────────────────────
// Child traversal (for non-array ops that may contain array ops)
// ─────────────────────────────────────────────────────────────

function lowerChildren(obj: Record<string, unknown>, memo?: WeakMap<object, ExprNode>): ExprNode {
  // Return the original node if no children actually change — this preserves DAG
  // identity so shared subexpressions remain shared through the lowering pass.
  const result: Record<string, unknown> = {}
  let changed = false
  for (const [k, v] of Object.entries(obj)) {
    if (Array.isArray(v)) {
      const arr = v as ExprNode[]
      const newArr = arr.map(n => lowerArrayOps(n, memo))
      const anyChanged = newArr.some((n, i) => n !== arr[i])
      result[k] = anyChanged ? newArr : arr
      if (anyChanged) changed = true
    } else if (typeof v === 'object' && v !== null && 'op' in v) {
      const newV = lowerArrayOps(v as ExprNode, memo)
      result[k] = newV
      if (newV !== v) changed = true
    } else {
      result[k] = v
    }
  }
  return changed ? result as ExprNode : obj as ExprNode
}

// ─────────────────────────────────────────────────────────────
// Individual lowerings
// ─────────────────────────────────────────────────────────────

/**
 * array_literal({shape, values}) → ArrayPack of lowered values
 */
function lowerArrayLiteral(obj: Record<string, unknown>, memo?: WeakMap<object, ExprNode>): ExprNode {
  const values = (obj.values as ExprNode[]).map(n => lowerArrayOps(n, memo))
  // Emit as inline array (ArrayPack)
  return values
}

/**
 * zeros({shape}) → ArrayPack of 0s
 */
function lowerZeros(obj: Record<string, unknown>): ExprNode {
  const shape = obj.shape as number[]
  const size = shapeSize(shape)
  return new Array(size).fill(0) as ExprNode
}

/**
 * ones({shape}) → ArrayPack of 1s
 */
function lowerOnes(obj: Record<string, unknown>): ExprNode {
  const shape = obj.shape as number[]
  const size = shapeSize(shape)
  return new Array(size).fill(1) as ExprNode
}

/**
 * fill({shape, value}) → ArrayPack of repeated value
 */
function lowerFill(obj: Record<string, unknown>, memo?: WeakMap<object, ExprNode>): ExprNode {
  const shape = obj.shape as number[]
  const size = shapeSize(shape)
  const value = lowerArrayOps(obj.value as ExprNode, memo)
  return new Array(size).fill(value) as ExprNode
}

/**
 * reshape({args: [arr], shape}) → lower arr, identity (flat data doesn't change)
 * Reshape is a no-op on the flat representation since we use row-major layout.
 * The data is the same — only the logical shape changes.
 */
function lowerReshape(obj: Record<string, unknown>, memo?: WeakMap<object, ExprNode>): ExprNode {
  const args = obj.args as ExprNode[]
  return lowerArrayOps(args[0], memo)
}

/**
 * transpose({args: [arr]}) → reindex a 2D array
 * For a [rows, cols] array, transpose produces [cols, rows] by reindexing.
 */
function lowerTranspose(obj: Record<string, unknown>, memo?: WeakMap<object, ExprNode>): ExprNode {
  const args = obj.args as ExprNode[]
  const arr = lowerArrayOps(args[0], memo)

  // If the inner is an inline array literal, we can statically reindex.
  // Otherwise, emit index operations.
  // For now: emit element-by-element Index ops.
  // The shape must be carried by the caller; we assume 2D here.
  // Since shapes are erased at this level, transpose just passes through.
  // Proper shape tracking happens at the type level; at the expression level
  // we emit Index-based reindexing when the shape is statically known.

  // Fallback: pass through — the C++ layer or a later pass resolves this
  return { op: 'transpose', args: [arr] }
}

/**
 * slice({args: [arr], axis, start, end}) → extract a contiguous sub-array
 * For 1D: arr[start..end] → ArrayPack of Index operations
 */
function lowerSlice(obj: Record<string, unknown>, memo?: WeakMap<object, ExprNode>): ExprNode {
  const args = obj.args as ExprNode[]
  const arr = lowerArrayOps(args[0], memo)
  const start = obj.start as number
  const end = obj.end as number
  const count = end - start

  // Emit: [Index(arr, start), Index(arr, start+1), ..., Index(arr, end-1)]
  const elements: ExprNode[] = []
  for (let i = 0; i < count; i++) {
    elements.push({ op: 'index', args: [arr, start + i] })
  }
  return elements
}

/**
 * reduce({args: [arr], axis, reduce_op}) → tree reduction
 * For 1D: reduce to a scalar via balanced binary tree.
 */
function lowerReduce(obj: Record<string, unknown>, memo?: WeakMap<object, ExprNode>): ExprNode {
  const args = obj.args as ExprNode[]
  const arr = lowerArrayOps(args[0], memo)
  const reduceOp = obj.reduce_op as string

  // If arr is an inline array, reduce statically
  if (Array.isArray(arr)) {
    return treeReduce(arr as ExprNode[], reduceOp)
  }

  // Otherwise, we can't statically reduce without knowing the size.
  // Pass through for the runtime or a later pass.
  return { op: 'reduce', args: [arr], axis: obj.axis, reduce_op: reduceOp }
}

/** Derive the ring ops (mul, add) from the element scalar type. */
function ringOpsForType(elementType: string): { mul_op: string; add_op: string } {
  switch (elementType) {
    case 'bool': return { mul_op: 'and', add_op: 'or' }
    case 'int':
    case 'float':
    default:     return { mul_op: 'mul', add_op: 'add' }
  }
}

/**
 * matmul({args: [A, B], shape_a: [M,K], shape_b: [K,N], element_type?})
 * → flat [M*N] array of scalar trees: C[i,j] = Σ_k A[i*K+k] * B[k*N+j]
 *
 * Ring ops are derived from element_type (default 'float'):
 *   'float' | 'int' → mul/add
 *   'bool'          → and/or  (boolean semiring: reachability, graph composition)
 *
 * New scalar kinds (e.g. tropical floats) extend ringOpsForType when added.
 */
function lowerMatmul(obj: Record<string, unknown>, memo?: WeakMap<object, ExprNode>): ExprNode {
  const args = obj.args as ExprNode[]
  const A = lowerArrayOps(args[0], memo)
  const B = lowerArrayOps(args[1], memo)
  const [M, K] = obj.shape_a as [number, number]
  const [, N] = obj.shape_b as [number, number]
  const { mul_op, add_op } = ringOpsForType((obj.element_type as string | undefined) ?? 'float')

  const elements: ExprNode[] = []
  for (let i = 0; i < M; i++) {
    for (let j = 0; j < N; j++) {
      const terms: ExprNode[] = []
      for (let k = 0; k < K; k++) {
        const aElem: ExprNode = { op: 'index', args: [A, i * K + k] }
        const bElem: ExprNode = { op: 'index', args: [B, k * N + j] }
        terms.push({ op: mul_op, args: [aElem, bElem] })
      }
      elements.push(treeReduce(terms, add_op))
    }
  }
  return elements
}

/** Build a balanced binary tree reduction. */
function treeReduce(elements: ExprNode[], op: string): ExprNode {
  if (elements.length === 0) {
    // Identity element for each op
    switch (op) {
      case 'add': return 0
      case 'mul': return 1
      case 'min': return Infinity
      case 'max': return -Infinity
      default: return 0
    }
  }
  if (elements.length === 1) return elements[0]

  const mid = Math.floor(elements.length / 2)
  const left = treeReduce(elements.slice(0, mid), op)
  const right = treeReduce(elements.slice(mid), op)
  return { op, args: [left, right] }
}

/**
 * broadcast_to({args: [arr], shape}) → replicate elements
 * Broadcasts a smaller array to a target shape by repeating elements.
 */
function lowerBroadcastTo(obj: Record<string, unknown>, memo?: WeakMap<object, ExprNode>): ExprNode {
  const args = obj.args as ExprNode[]
  const arr = lowerArrayOps(args[0], memo)
  const targetShape = obj.shape as number[]
  const targetSize = shapeSize(targetShape)

  // If arr is a scalar, just fill
  if (typeof arr === 'number') {
    return new Array(targetSize).fill(arr) as ExprNode
  }

  // If arr is an inline array and target is 1D, replicate
  if (Array.isArray(arr) && targetShape.length === 1) {
    if (arr.length === 1) {
      // broadcast [x] to [N]
      return new Array(targetSize).fill(arr[0]) as ExprNode
    }
    if (arr.length === targetSize) {
      return arr // already correct size
    }
  }

  // General case: pass through for runtime
  return { op: 'broadcast_to', args: [arr], shape: targetShape }
}

/**
 * map({callee, args: [arr]}) → apply function to each element
 * Unrolls into ArrayPack of call(fn, element) for each element.
 */
function lowerMap(obj: Record<string, unknown>, memo?: WeakMap<object, ExprNode>): ExprNode {
  const callee = obj.callee as ExprNode
  const args = obj.args as ExprNode[]
  const arr = lowerArrayOps(args[0], memo)

  // If arr is an inline array, unroll the map
  if (Array.isArray(arr)) {
    return arr.map(elem => ({
      op: 'call',
      callee,
      args: [elem],
    })) as ExprNode
  }

  // Otherwise, keep as map (runtime resolves)
  return { op: 'map', callee, args: [arr] }
}

// ─────────────────────────────────────────────────────────────
// Compile-time combinator expansion
// ─────────────────────────────────────────────────────────────

/**
 * Shadowed-bindings descriptor per binder op. `bodyFields` names fields whose
 * subtree is inside the binder's scope (so shadowed variables must be removed
 * from the substitution map before recursing into them). `nonBodyFields` are
 * fields outside the scope (count, init, the bind-value map of `let`, etc.)
 * that must see the outer bindings unchanged.
 *
 * Kept adjacent to the binder lowerings so it's easy to keep in sync.
 */
interface BinderInfo {
  /** Keys on the binder node that name shadowed variables (strings). */
  shadowingKeys: string[]
  /** Fields whose children are inside the binder body. */
  bodyFields: string[]
}

const BINDER_OPS: Record<string, BinderInfo> = {
  let:            { shadowingKeys: [], bodyFields: ['in'] },  // 'bind' handled specially
  generate:       { shadowingKeys: ['var'],            bodyFields: ['body'] },
  iterate:        { shadowingKeys: ['var'],            bodyFields: ['body'] },
  fold:           { shadowingKeys: ['acc', 'elem'],    bodyFields: ['body'] },
  scan:           { shadowingKeys: ['acc', 'elem'],    bodyFields: ['body'] },
  map2:           { shadowingKeys: ['elem'],           bodyFields: ['body'] },
  zip_with:       { shadowingKeys: ['x', 'y'],         bodyFields: ['body'] },
  chain:          { shadowingKeys: ['var'],            bodyFields: ['body'] },
  generate_decls: { shadowingKeys: ['var'],            bodyFields: ['decls'] },
}

/** Build a substitution map with the given names removed. Returns the input
 *  map unchanged when none of the names are bound, for pointer-identity
 *  short-circuiting in the common case. */
function shieldBindings(
  bindings: Map<string, ExprNode>,
  shadowed: readonly string[],
): Map<string, ExprNode> {
  let copy: Map<string, ExprNode> | null = null
  for (const name of shadowed) {
    if (bindings.has(name)) {
      if (copy === null) copy = new Map(bindings)
      copy.delete(name)
    }
  }
  return copy ?? bindings
}

/**
 * Substitute all `{ op: 'binding', name }` nodes in a tree with values from `bindings`.
 * Each call site should use a fresh memo to maintain correct DAG sharing per iteration.
 *
 * Scope-aware: when encountering a binder node (see `BINDER_OPS`), the
 * shadowed variables are removed from the substitution map before recursing
 * into the binder's body fields. Non-body fields still see the outer bindings.
 * For `let`, the `bind` values see the outer scope; only `in` is shielded from
 * any name bound in `bind`.
 */
function substituteBindings(
  node: ExprNode,
  bindings: Map<string, ExprNode>,
  memo?: WeakMap<object, ExprNode>,
): ExprNode {
  if (typeof node === 'number' || typeof node === 'boolean') return node
  if (Array.isArray(node)) return node.map(n => substituteBindings(n, bindings, memo))
  if (typeof node !== 'object' || node === null) return node

  if (memo) {
    const cached = memo.get(node as object)
    if (cached !== undefined) return cached
  }

  const obj = node as { op: string; [k: string]: unknown }
  if (obj.op === 'binding') {
    const val = bindings.get(obj.name as string)
    if (val !== undefined) return val
    // Unresolved binding — leave as-is (may be resolved by an outer combinator).
    // expandDeclGenerators rejects residuals after expansion; other lowerings
    // produce them transiently and resolve during subsequent passes.
    return node
  }

  // ── match: per-arm bindings ──────────────────────────────────────────────
  // Each arm of a match introduces its own (possibly empty) set of bindings
  // visible only inside that arm's body. The scrutinee is evaluated in the
  // outer scope. Per-arm bind shape: undefined | string | string[].
  if (obj.op === 'match') {
    const armsObj = obj.arms as Record<string, { bind?: string | string[]; body: ExprNode }>
    let changed = false
    const newScrutinee = substituteBindings(obj.scrutinee as ExprNode, bindings, memo)
    if (newScrutinee !== obj.scrutinee) changed = true
    const newArms: Record<string, { bind?: string | string[]; body: ExprNode }> = {}
    for (const [variantName, arm] of Object.entries(armsObj)) {
      const armBindNames = arm.bind === undefined
        ? []
        : (typeof arm.bind === 'string' ? [arm.bind] : arm.bind)
      const shielded = shieldBindings(bindings, armBindNames)
      const newBody = substituteBindings(arm.body, shielded, memo)
      if (newBody !== arm.body) changed = true
      newArms[variantName] = arm.bind === undefined
        ? { body: newBody }
        : { bind: arm.bind, body: newBody }
    }
    const out: ExprNode = changed
      ? { ...obj, scrutinee: newScrutinee, arms: newArms } as ExprNode
      : node
    if (memo) memo.set(node as object, out)
    return out
  }

  // Determine per-field substitution maps for binder nodes.
  const binder = BINDER_OPS[obj.op]
  const letBindNames: string[] = obj.op === 'let' && obj.bind && typeof obj.bind === 'object' && !Array.isArray(obj.bind)
    ? Object.keys(obj.bind as Record<string, unknown>)
    : []
  const bodyShielded: Map<string, ExprNode> = binder
    ? shieldBindings(bindings, [...binder.shadowingKeys.flatMap(k => {
        const v = obj[k]; return typeof v === 'string' ? [v] : []
      }), ...letBindNames])
    : bindings

  let changed = false
  const result: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(obj)) {
    const inBody = binder?.bodyFields.includes(k) ?? false
    const activeBindings = inBody ? bodyShielded : bindings
    if (Array.isArray(v)) {
      const arr = v as ExprNode[]
      const newArr = arr.map(n => substituteBindings(n, activeBindings, memo))
      if (newArr.some((n, i) => n !== arr[i])) changed = true
      result[k] = newArr
    } else if (typeof v === 'object' && v !== null && 'op' in v) {
      const newV = substituteBindings(v as ExprNode, activeBindings, memo)
      if (newV !== v) changed = true
      result[k] = newV
    } else if (typeof v === 'object' && v !== null && !('op' in v)) {
      // Record<string, ExprNode> fields (e.g. 'bind' in let nodes — these
      // evaluate in the outer scope and are NOT shielded).
      const rec = v as Record<string, ExprNode>
      const newRec: Record<string, ExprNode> = {}
      let recChanged = false
      for (const [rk, rv] of Object.entries(rec)) {
        const newRv = substituteBindings(rv, bindings, memo)
        if (newRv !== rv) recChanged = true
        newRec[rk] = newRv
      }
      if (recChanged) changed = true
      result[k] = recChanged ? newRec : rec
    } else {
      result[k] = v
    }
  }
  const out = changed ? result as ExprNode : node
  if (memo) memo.set(node as object, out)
  return out
}

/** let — sequential let* bindings, then substitute into body. */
function lowerLet(obj: Record<string, unknown>, memo?: WeakMap<object, ExprNode>): ExprNode {
  const bind = obj.bind as Record<string, ExprNode>
  const body = obj.in as ExprNode

  // Evaluate bindings sequentially — each can reference earlier ones
  const resolved = new Map<string, ExprNode>()
  for (const [name, expr] of Object.entries(bind)) {
    let lowered = lowerArrayOps(expr, memo)
    if (resolved.size > 0) {
      lowered = substituteBindings(lowered, resolved, new WeakMap())
    }
    resolved.set(name, lowered)
  }

  const substituted = substituteBindings(body, resolved, new WeakMap())
  return lowerArrayOps(substituted, memo)
}

/** generate(count, var, body) → [body[var=0], body[var=1], ...] */
function lowerGenerate(obj: Record<string, unknown>, memo?: WeakMap<object, ExprNode>): ExprNode {
  const count = obj.count as number
  const varName = obj.var as string
  const body = obj.body as ExprNode

  const elements: ExprNode[] = []
  for (let i = 0; i < count; i++) {
    const bindings = new Map<string, ExprNode>([[varName, i]])
    const substituted = substituteBindings(body, bindings, new WeakMap())
    elements.push(lowerArrayOps(substituted, memo))
  }
  return elements
}

/** iterate(count, init, var, body) → [init, body[var=init], body[var=body[var=init]], ...] */
function lowerIterate(obj: Record<string, unknown>, memo?: WeakMap<object, ExprNode>): ExprNode {
  const count = obj.count as number
  const varName = obj.var as string
  const init = lowerArrayOps(obj.init as ExprNode, memo)
  const body = obj.body as ExprNode

  const elements: ExprNode[] = []
  let current = init
  for (let i = 0; i < count; i++) {
    elements.push(current)
    const bindings = new Map<string, ExprNode>([[varName, current]])
    current = lowerArrayOps(substituteBindings(body, bindings, new WeakMap()), memo)
  }
  return elements
}

/** fold(over, init, acc_var, elem_var, body) → unrolled left fold to scalar. */
function lowerFold(obj: Record<string, unknown>, memo?: WeakMap<object, ExprNode>): ExprNode {
  const over = lowerArrayOps(obj.over as ExprNode, memo)
  const accVar = obj.acc_var as string
  const elemVar = obj.elem_var as string
  const body = obj.body as ExprNode

  if (!Array.isArray(over)) {
    // Cannot statically fold — pass through
    return lowerChildren(obj, memo)
  }

  let acc = lowerArrayOps(obj.init as ExprNode, memo)
  for (const elem of over as ExprNode[]) {
    const bindings = new Map<string, ExprNode>([[accVar, acc], [elemVar, elem]])
    acc = lowerArrayOps(substituteBindings(body, bindings, new WeakMap()), memo)
  }
  return acc
}

/** scan(over, init, acc_var, elem_var, body) → array of fold intermediates. */
function lowerScan(obj: Record<string, unknown>, memo?: WeakMap<object, ExprNode>): ExprNode {
  const over = lowerArrayOps(obj.over as ExprNode, memo)
  const accVar = obj.acc_var as string
  const elemVar = obj.elem_var as string
  const body = obj.body as ExprNode

  if (!Array.isArray(over)) {
    return lowerChildren(obj, memo)
  }

  const results: ExprNode[] = []
  let acc = lowerArrayOps(obj.init as ExprNode, memo)
  for (const elem of over as ExprNode[]) {
    const bindings = new Map<string, ExprNode>([[accVar, acc], [elemVar, elem]])
    acc = lowerArrayOps(substituteBindings(body, bindings, new WeakMap()), memo)
    results.push(acc)
  }
  return results
}

/** map2(over, elem_var, body) → [body[elem=e0], body[elem=e1], ...] */
function lowerMap2(obj: Record<string, unknown>, memo?: WeakMap<object, ExprNode>): ExprNode {
  const over = lowerArrayOps(obj.over as ExprNode, memo)
  const elemVar = obj.elem_var as string
  const body = obj.body as ExprNode

  if (!Array.isArray(over)) {
    return lowerChildren(obj, memo)
  }

  return (over as ExprNode[]).map(elem => {
    const bindings = new Map<string, ExprNode>([[elemVar, elem]])
    return lowerArrayOps(substituteBindings(body, bindings, new WeakMap()), memo)
  })
}

/** zip_with(a, b, x_var, y_var, body) → pointwise combination. */
function lowerZipWith(obj: Record<string, unknown>, memo?: WeakMap<object, ExprNode>): ExprNode {
  const a = lowerArrayOps(obj.a as ExprNode, memo)
  const b = lowerArrayOps(obj.b as ExprNode, memo)
  const xVar = obj.x_var as string
  const yVar = obj.y_var as string
  const body = obj.body as ExprNode

  if (!Array.isArray(a) || !Array.isArray(b)) {
    return lowerChildren(obj, memo)
  }

  const arrA = a as ExprNode[]
  const arrB = b as ExprNode[]
  const len = Math.min(arrA.length, arrB.length)
  const results: ExprNode[] = []
  for (let i = 0; i < len; i++) {
    const bindings = new Map<string, ExprNode>([[xVar, arrA[i]], [yVar, arrB[i]]])
    results.push(lowerArrayOps(substituteBindings(body, bindings, new WeakMap()), memo))
  }
  return results
}

/** chain(count, init, var, body) → apply body count times, threading result. */
function lowerChain(obj: Record<string, unknown>, memo?: WeakMap<object, ExprNode>): ExprNode {
  const count = obj.count as number
  const varName = obj.var as string
  const body = obj.body as ExprNode
  let current = lowerArrayOps(obj.init as ExprNode, memo)

  for (let i = 0; i < count; i++) {
    const bindings = new Map<string, ExprNode>([[varName, current]])
    current = lowerArrayOps(substituteBindings(body, bindings, new WeakMap()), memo)
  }
  return current
}

/**
 * tag — recurse into payload-field expressions so any array ops or combinators
 * nested inside them are lowered. The tag node itself stays in place; lowering
 * to scalar bundle writes happens in flatten.ts (Phase 3).
 */
function lowerTag(obj: Record<string, unknown>, memo?: WeakMap<object, ExprNode>): ExprNode {
  const payload = obj.payload as Record<string, ExprNode> | undefined
  if (payload === undefined) return obj as ExprNode
  let changed = false
  const newPayload: Record<string, ExprNode> = {}
  for (const [k, v] of Object.entries(payload)) {
    const newV = lowerArrayOps(v, memo)
    if (newV !== v) changed = true
    newPayload[k] = newV
  }
  return changed ? ({ ...obj, payload: newPayload } as ExprNode) : (obj as ExprNode)
}

/**
 * match — recurse into the scrutinee and each arm body so any array ops or
 * combinators nested inside them are lowered. The match node itself stays in
 * place; lowering to scalar bundle dispatch happens in flatten.ts (Phase 3).
 *
 * Arms with a `bind` field introduce a local payload binding. Lowering the
 * arm's body must happen with that binding in scope, but at this stage the
 * binding's value (a bundle slot read) doesn't exist yet — we just recurse
 * structurally. The `binding` node is left intact for substituteBindings to
 * resolve later.
 */
function lowerMatch(obj: Record<string, unknown>, memo?: WeakMap<object, ExprNode>): ExprNode {
  const arms = obj.arms as Record<string, { bind?: string | string[]; body: ExprNode }>
  let changed = false
  const newScrutinee = lowerArrayOps(obj.scrutinee as ExprNode, memo)
  if (newScrutinee !== obj.scrutinee) changed = true
  const newArms: Record<string, { bind?: string | string[]; body: ExprNode }> = {}
  for (const [variant, arm] of Object.entries(arms)) {
    const newBody = lowerArrayOps(arm.body, memo)
    if (newBody !== arm.body) changed = true
    newArms[variant] = arm.bind === undefined
      ? { body: newBody }
      : { bind: arm.bind, body: newBody }
  }
  return changed ? ({ ...obj, scrutinee: newScrutinee, arms: newArms } as ExprNode) : (obj as ExprNode)
}

// ─────────────────────────────────────────────────────────────
// Decl-level generate combinator
// ─────────────────────────────────────────────────────────────

/**
 * Evaluate a str_concat-bearing expression to a string after binding substitution.
 * Handles: string literals, integer literals, str_concat, and constant arithmetic (add/sub/mul).
 * Called only during generate_decls expansion, where binding variables are already
 * substituted to concrete integers — so arithmetic nodes contain only literals.
 */
function evaluateStrExpr(node: ExprNode): string {
  if (typeof node === 'number') return String(node)
  if (typeof node === 'string') return node
  if (typeof node !== 'object' || node === null || Array.isArray(node))
    throw new Error(`generate_decls: cannot use array or boolean as string value: ${JSON.stringify(node)}`)

  const obj = node as Record<string, unknown>

  if (obj.op === 'str_concat') {
    const parts = obj.parts as ExprNode[]
    return parts.map(evaluateStrExpr).join('')
  }
  if (obj.op === 'add') {
    const args = obj.args as ExprNode[]
    const l = Number(evaluateStrExpr(args[0])), r = Number(evaluateStrExpr(args[1]))
    if (isNaN(l) || isNaN(r))
      throw new Error(`generate_decls: 'add' args must be numbers in string expression`)
    return String(l + r)
  }
  if (obj.op === 'sub') {
    const args = obj.args as ExprNode[]
    return String(Number(evaluateStrExpr(args[0])) - Number(evaluateStrExpr(args[1])))
  }
  if (obj.op === 'mul') {
    const args = obj.args as ExprNode[]
    return String(Number(evaluateStrExpr(args[0])) * Number(evaluateStrExpr(args[1])))
  }

  throw new Error(`generate_decls: cannot evaluate string expression with op '${obj.op}': ${JSON.stringify(node)}`)
}

/**
 * Recursively walk an expanded decl and replace any str_concat nodes with their
 * evaluated string values. This resolves both the `name` field of instance_decl
 * and `instance` fields inside `ref` nodes that used str_concat templates.
 */
function resolveStrConcats(node: unknown): unknown {
  if (typeof node !== 'object' || node === null) return node
  if (Array.isArray(node)) return (node as unknown[]).map(resolveStrConcats)
  const obj = node as Record<string, unknown>
  if (obj.op === 'str_concat') return evaluateStrExpr(node as ExprNode)
  const result: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(obj)) {
    result[k] = resolveStrConcats(v)
  }
  return result
}

/** Decls that carry a unique `name` field — used for collision detection. */
const NAMED_DECL_OPS = new Set(['instance_decl', 'reg_decl', 'delay_decl', 'program_decl'])

/**
 * Walk a tree and throw if any `{op:'binding'}` node's name is NOT bound by
 * an enclosing binder in the tree itself. Names shielded by inner binders
 * (let, generate, iterate, ...) are legitimate — they'll be resolved by
 * the normal array-lowering pass later.
 *
 * This catches the typo case (`{binding j}` in a generator bound to `i`
 * with no inner binder rebinding `j`) while letting inner-combinator
 * bindings through.
 */
function assertNoResidualBindings(
  node: unknown,
  pathHint: string,
  scope: ReadonlySet<string> = new Set(),
): void {
  if (typeof node !== 'object' || node === null) return
  if (Array.isArray(node)) {
    for (const item of node) assertNoResidualBindings(item, pathHint, scope)
    return
  }
  const obj = node as Record<string, unknown>

  if (obj.op === 'binding') {
    const name = String(obj.name)
    if (!scope.has(name)) {
      throw new Error(
        `generate_decls: unresolved binding '${name}' in expanded decl ${pathHint}. ` +
        `Check the generator's 'var' matches the binding name and that no inner binders accidentally shadow it.`,
      )
    }
    return
  }

  // For binder ops, extend the scope with the names they bind before
  // recursing into their body fields. Non-body fields see the outer scope.
  const binder = BINDER_OPS[obj.op as string]
  let bodyScope: ReadonlySet<string> = scope
  if (binder) {
    const shadowed: string[] = []
    for (const key of binder.shadowingKeys) {
      const v = obj[key]
      if (typeof v === 'string') shadowed.push(v)
    }
    if (obj.op === 'let' && obj.bind && typeof obj.bind === 'object' && !Array.isArray(obj.bind)) {
      for (const k of Object.keys(obj.bind as Record<string, unknown>)) shadowed.push(k)
    }
    if (shadowed.length > 0) {
      const merged = new Set(scope)
      for (const n of shadowed) merged.add(n)
      bodyScope = merged
    }
  }

  for (const [k, v] of Object.entries(obj)) {
    const inBody = binder?.bodyFields.includes(k) ?? false
    assertNoResidualBindings(v, pathHint, inBody ? bodyScope : scope)
  }
}

/**
 * Expand any `generate_decls` entries in a block's decl list to concrete instance_decl / reg_decl / delay_decl entries.
 *
 * generate_decls schema:
 * ```json
 * {
 *   "op": "generate_decls",
 *   "count": 10,
 *   "var": "i",
 *   "decls": [
 *     {
 *       "op": "instance_decl",
 *       "name": { "op": "str_concat", "parts": ["VCO", { "op": "binding", "name": "i" }] },
 *       "program": "SinOsc",
 *       "inputs": { "freq": { "op": "mul", "args": [{ "op": "binding", "name": "i" }, 80] } }
 *     }
 *   ]
 * }
 * ```
 *
 * Semantics:
 * - Nested `generate_decls` in a template expand recursively (outer loop first,
 *   then inner).
 * - `substituteBindings` is scope-aware — an inner `let` / `generate` / etc.
 *   rebinding `var` shields the outer substitution.
 * - Names produced by expanded `instance_decl` / `reg_decl` / `delay_decl` /
 *   `program_decl` entries must be unique across the whole block (including
 *   pre-existing sibling decls); collisions throw.
 * - Any residual `{op:'binding'}` node after substitution + str_concat
 *   resolution is rejected (likely a typo between `var` and a binding name).
 *
 * Returns the original BlockNode unchanged if no generate_decls entries are present.
 */
export function expandDeclGenerators(block: BlockNode): BlockNode {
  const decls = block.decls
  if (!decls || decls.length === 0) return block

  let changed = false
  const expanded: ExprNode[] = []
  const seenNames = new Set<string>()

  /** Push a decl, checking it for name collisions with earlier decls in the block. */
  const pushChecked = (decl: ExprNode, pathHint: string): void => {
    if (typeof decl === 'object' && decl !== null && !Array.isArray(decl)) {
      const d = decl as Record<string, unknown>
      if (typeof d.op === 'string' && NAMED_DECL_OPS.has(d.op) && typeof d.name === 'string') {
        if (seenNames.has(d.name)) {
          throw new Error(
            `generate_decls: duplicate decl name '${d.name}' produced by ${pathHint}. ` +
            `Two entries in the block (generator output or explicit sibling) resolved to the same name.`,
          )
        }
        seenNames.add(d.name)
      }
    }
    expanded.push(decl)
  }

  for (const rawDecl of decls) {
    if (typeof rawDecl !== 'object' || rawDecl === null || Array.isArray(rawDecl)) {
      expanded.push(rawDecl)
      continue
    }
    const d = rawDecl as Record<string, unknown>
    if (d.op !== 'generate_decls') {
      pushChecked(rawDecl, 'explicit decl')
      continue
    }

    changed = true
    const count = d.count as number
    const varName = d.var as string
    const templates = d.decls as ExprNode[]

    for (let i = 0; i < count; i++) {
      const bindings = new Map<string, ExprNode>([[varName, i]])
      for (let t = 0; t < templates.length; t++) {
        const template = templates[t]
        const substituted = substituteBindings(template, bindings, new WeakMap())
        const hint = `generate_decls(var='${varName}') iteration ${varName}=${i}, template[${t}]`

        // Nested generate_decls: recurse before str_concat resolution or
        // residual-binding checks. The inner expansion resolves its own
        // bindings and str_concats; running them at this level would
        // misinterpret inner-scoped `{binding innerVar}` nodes.
        if (typeof substituted === 'object' && substituted !== null && !Array.isArray(substituted)
            && (substituted as Record<string, unknown>).op === 'generate_decls') {
          const inner = expandDeclGenerators({ op: 'block', decls: [substituted as ExprNode] })
          for (const innerDecl of inner.decls ?? []) pushChecked(innerDecl, hint)
          continue
        }

        // Non-generator decl: resolve str_concats to strings and reject
        // any residual binding that isn't shielded by an inner binder.
        const resolved = resolveStrConcats(substituted)
        assertNoResidualBindings(resolved, hint)

        const resolvedObj = resolved as Record<string, unknown>
        // Named decls must have their `name` field reduced to a plain string.
        if (typeof resolvedObj.op === 'string' && NAMED_DECL_OPS.has(resolvedObj.op)
            && typeof resolvedObj.name !== 'string') {
          throw new Error(
            `generate_decls: '${resolvedObj.op}' name did not resolve to a string in ${hint}: ${JSON.stringify(resolvedObj.name)}`,
          )
        }
        pushChecked(resolvedObj as unknown as ExprNode, hint)
      }
    }
  }

  return changed ? { ...block, decls: expanded } : block
}
