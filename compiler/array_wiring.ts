/**
 * array_wiring.ts — Shape validation and auto-broadcasting for inter-module array connections.
 *
 * When connecting module outputs to inputs, this validates that array shapes are
 * compatible and inserts broadcast_to nodes when needed.
 */

import {
  type PortType,
  type ScalarKind,
  Float,
  portTypeEqual,
  portTypeToString,
  broadcastShapes,
} from './term.js'
import type { ExprNode } from './expr.js'

// Widening lattice: bool → int → float. A source kind widens to a dest kind
// when RANK[src] <= RANK[dst]. Narrowing must be explicit via to_int/to_bool.
const KIND_RANK: Record<ScalarKind, number> = { bool: 0, int: 1, float: 2 }

function widens(src: ScalarKind, dst: ScalarKind): boolean {
  return KIND_RANK[src] <= KIND_RANK[dst]
}

function narrowingHint(dst: ScalarKind): string {
  return dst === 'int' ? 'to_int()' : dst === 'bool' ? 'to_bool()' : 'to_float()'
}

export interface ConnectionCheck {
  compatible: boolean
  /** If compatible and broadcasting is needed, the broadcast_to wrapper expression. */
  broadcastExpr?: ExprNode
  /** Human-readable error if incompatible. */
  error?: string
  /** The resolved output shape after broadcasting. */
  resultShape?: number[]
}

/**
 * Check if a source port type is compatible with a destination port type,
 * accounting for array shape broadcasting.
 *
 * Returns a ConnectionCheck with:
 * - compatible: true if the connection is valid
 * - broadcastExpr: if shapes differ but are broadcast-compatible,
 *   wraps the ref expression in a broadcast_to
 * - error: human-readable explanation if incompatible
 */
export function checkArrayConnection(
  srcTypeIn: PortType | undefined,
  dstTypeIn: PortType | undefined,
  refExpr: ExprNode,
): ConnectionCheck {
  const srcType = srcTypeIn ?? Float
  const dstType = dstTypeIn ?? Float

  // If port types are structurally equal, pass through
  if (portTypeEqual(srcType, dstType)) {
    return { compatible: true }
  }

  // Scalar → scalar: widening (bool→int→float) is implicit, narrowing is explicit.
  if (srcType.tag === 'scalar' && dstType.tag === 'scalar') {
    if (srcType.scalar === dstType.scalar) return { compatible: true }
    if (widens(srcType.scalar, dstType.scalar)) return { compatible: true }
    return {
      compatible: false,
      error: `Lossy conversion: cannot narrow ${portTypeToString(srcType)} to ${portTypeToString(dstType)} — wrap source in ${narrowingHint(dstType.scalar)} to narrow explicitly`,
    }
  }

  // Scalar → array: broadcast scalar to array shape
  if (srcType.tag === 'scalar' && dstType.tag === 'array') {
    return {
      compatible: true,
      broadcastExpr: { op: 'broadcastTo', args: [refExpr], shape: dstType.shape },
      resultShape: dstType.shape,
    }
  }

  // Array → scalar: not auto-compatible (user must reduce explicitly)
  if (srcType.tag === 'array' && dstType.tag === 'scalar') {
    return {
      compatible: false,
      error: `Cannot connect ${portTypeToString(srcType)} to ${portTypeToString(dstType)} — reduce or index the array first`,
    }
  }

  // Array → array: check element-kind and shape compatibility.
  if (srcType.tag === 'array' && dstType.tag === 'array') {
    // Element kind must widen (bool→int→float). We only track scalar element kinds.
    if (srcType.element.tag === 'scalar' && dstType.element.tag === 'scalar') {
      const sk = srcType.element.scalar
      const dk = dstType.element.scalar
      if (sk !== dk && !widens(sk, dk)) {
        return {
          compatible: false,
          error: `Lossy conversion: cannot narrow ${portTypeToString(srcType)} to ${portTypeToString(dstType)} — wrap source in ${narrowingHint(dk)} to narrow explicitly`,
        }
      }
    } else if (!portTypeEqual(srcType.element, dstType.element)) {
      return {
        compatible: false,
        error: `Element type mismatch: source is ${portTypeToString(srcType)} but destination expects ${portTypeToString(dstType)}`,
      }
    }

    const resultShape = broadcastShapes(srcType.shape, dstType.shape)
    if (resultShape === null) {
      return {
        compatible: false,
        error: `Shape mismatch: source is ${portTypeToString(srcType)} but destination expects ${portTypeToString(dstType)} (shapes not broadcast-compatible)`,
      }
    }

    // If source shape already matches destination, no broadcast needed
    if (arraysEqual(srcType.shape, dstType.shape)) {
      return { compatible: true, resultShape }
    }

    // Source needs broadcasting to match destination
    return {
      compatible: true,
      broadcastExpr: { op: 'broadcastTo', args: [refExpr], shape: dstType.shape },
      resultShape,
    }
  }

  // Non-array, non-scalar types: fall back to structural equality
  if (!portTypeEqual(srcType, dstType)) {
    return {
      compatible: false,
      error: `Type mismatch: source is ${portTypeToString(srcType)} but destination expects ${portTypeToString(dstType)}`,
    }
  }

  return { compatible: true }
}

function arraysEqual(a: number[], b: number[]): boolean {
  if (a.length !== b.length) return false
  return a.every((v, i) => v === b[i])
}
