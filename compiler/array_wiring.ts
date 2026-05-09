/**
 * array_wiring.ts — Shape validation and auto-broadcasting for inter-module array connections.
 *
 * When connecting module outputs to inputs, this validates that array shapes are
 * compatible and inserts broadcast_to nodes when needed.
 */

import { broadcastShapes } from './term.js'
import type { PortType, ScalarKind } from './ir/nodes.js'
import { Float, portTypeEqual, portTypeToString } from './ir/port_type.js'
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
  if (srcType.kind === 'scalar' && dstType.kind === 'scalar') {
    if (srcType.scalar === dstType.scalar) return { compatible: true }
    if (widens(srcType.scalar, dstType.scalar)) return { compatible: true }
    return {
      compatible: false,
      error: `Lossy conversion: cannot narrow ${portTypeToString(srcType)} to ${portTypeToString(dstType)} — wrap source in ${narrowingHint(dstType.scalar)} to narrow explicitly`,
    }
  }

  // Post-strata shapes are concrete integers; type-params are gone after specialize.
  const concreteShape = (s: PortType extends infer P ? P extends { kind: 'array'; shape: infer S } ? S : never : never): number[] => {
    return (s as Array<number | { name: string }>).map(d => {
      if (typeof d !== 'number') throw new Error(`array shape carries unresolved type-param '${d.name}'`)
      return d
    })
  }
  const elemScalar = (e: ScalarKind | { base: ScalarKind }): ScalarKind =>
    typeof e === 'string' ? e : e.base

  // Scalar → array: broadcast scalar to array shape
  if (srcType.kind === 'scalar' && dstType.kind === 'array') {
    const dstShape = concreteShape(dstType.shape)
    return {
      compatible: true,
      broadcastExpr: { op: 'broadcastTo', args: [refExpr], shape: dstShape },
      resultShape: dstShape,
    }
  }

  // Array → scalar: not auto-compatible (user must reduce explicitly)
  if (srcType.kind === 'array' && dstType.kind === 'scalar') {
    return {
      compatible: false,
      error: `Cannot connect ${portTypeToString(srcType)} to ${portTypeToString(dstType)} — reduce or index the array first`,
    }
  }

  // Array → array: check element-kind and shape compatibility.
  if (srcType.kind === 'array' && dstType.kind === 'array') {
    const sk = elemScalar(srcType.element)
    const dk = elemScalar(dstType.element)
    if (sk !== dk && !widens(sk, dk)) {
      return {
        compatible: false,
        error: `Lossy conversion: cannot narrow ${portTypeToString(srcType)} to ${portTypeToString(dstType)} — wrap source in ${narrowingHint(dk)} to narrow explicitly`,
      }
    }

    const srcShape = concreteShape(srcType.shape)
    const dstShape = concreteShape(dstType.shape)
    const resultShape = broadcastShapes(srcShape, dstShape)
    if (resultShape === null) {
      return {
        compatible: false,
        error: `Shape mismatch: source is ${portTypeToString(srcType)} but destination expects ${portTypeToString(dstType)} (shapes not broadcast-compatible)`,
      }
    }

    // If source shape already matches destination, no broadcast needed
    if (arraysEqual(srcShape, dstShape)) {
      return { compatible: true, resultShape }
    }

    // Source needs broadcasting to match destination
    return {
      compatible: true,
      broadcastExpr: { op: 'broadcastTo', args: [refExpr], shape: dstShape },
      resultShape,
    }
  }

  // Alias / mixed cases: fall back to structural equality
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
