/**
 * array_wiring.ts — Shape validation and auto-broadcasting for inter-module array connections.
 *
 * When connecting module outputs to inputs, this validates that array shapes are
 * compatible and inserts broadcast_to nodes when needed.
 */

import { portTypeFromString } from './compiler.js'
import {
  type PortType,
  portTypeEqual,
  portTypeToString,
  broadcastShapes,
} from './term.js'
import type { ExprNode } from './expr.js'

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
  srcTypeStr: string | undefined,
  dstTypeStr: string | undefined,
  refExpr: ExprNode,
): ConnectionCheck {
  const srcType = portTypeFromString(srcTypeStr)
  const dstType = portTypeFromString(dstTypeStr)

  // If both are undefined or identical strings, allow (backwards compat)
  if (srcTypeStr === dstTypeStr) {
    return { compatible: true }
  }

  // If port types are structurally equal, pass through
  if (portTypeEqual(srcType, dstType)) {
    return { compatible: true }
  }

  // Scalar → scalar: existing behavior (type check was string-based)
  if (srcType.tag === 'scalar' && dstType.tag === 'scalar') {
    // Allow float↔float, but not float→bool etc.
    if (srcType.scalar !== dstType.scalar) {
      return {
        compatible: false,
        error: `Type mismatch: source is ${portTypeToString(srcType)} but destination expects ${portTypeToString(dstType)}`,
      }
    }
    return { compatible: true }
  }

  // Scalar → array: broadcast scalar to array shape
  if (srcType.tag === 'scalar' && dstType.tag === 'array') {
    return {
      compatible: true,
      broadcastExpr: { op: 'broadcast_to', args: [refExpr], shape: dstType.shape },
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

  // Array → array: check shape compatibility
  if (srcType.tag === 'array' && dstType.tag === 'array') {
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
      broadcastExpr: { op: 'broadcast_to', args: [refExpr], shape: dstType.shape },
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
