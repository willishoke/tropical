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

/**
 * matmul({args: [A, B], shape_a: [M,K], shape_b: [K,N]}) → flat [M*N] array of scalar trees
 * C[i,j] = Σ_k A[i*K+k] * B[k*N+j]  (row-major)
 */
function lowerMatmul(obj: Record<string, unknown>, memo?: WeakMap<object, ExprNode>): ExprNode {
  const args = obj.args as ExprNode[]
  const A = lowerArrayOps(args[0], memo)
  const B = lowerArrayOps(args[1], memo)
  const [M, K] = obj.shape_a as [number, number]
  const [, N] = obj.shape_b as [number, number]

  const elements: ExprNode[] = []
  for (let i = 0; i < M; i++) {
    for (let j = 0; j < N; j++) {
      const terms: ExprNode[] = []
      for (let k = 0; k < K; k++) {
        const aElem: ExprNode = { op: 'index', args: [A, i * K + k] }
        const bElem: ExprNode = { op: 'index', args: [B, k * N + j] }
        terms.push({ op: 'mul', args: [aElem, bElem] })
      }
      elements.push(treeReduce(terms, 'add'))
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
