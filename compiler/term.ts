/**
 * term.ts — Free monoidal category term language for tropical.
 *
 * Objects are PortTypes, morphisms are modules, composition is patching,
 * tensor is parallel execution, trace is feedback with typed state.
 *
 * This is the categorical IR — constructed by the compiler from a mutable
 * patch graph, optimized by rewrite passes, then flattened to an execution
 * plan that the C++ runtime loads and JIT-compiles.
 */

import type { ExprNode } from './patch'

// ─────────────────────────────────────────────────────────────
// Port types (objects of the category)
// ─────────────────────────────────────────────────────────────

export type ScalarKind = 'float' | 'int' | 'bool'

export type PortType =
  | { tag: 'scalar'; scalar: ScalarKind }
  | { tag: 'array'; element: PortType; shape: number[] }
  | { tag: 'product'; factors: PortType[] }
  | { tag: 'coproduct'; summands: PortType[] }
  | { tag: 'unit' }

// Constructors
export const ScalarType = (s: ScalarKind): PortType => ({ tag: 'scalar', scalar: s })
export const Float: PortType = ScalarType('float')
export const Int: PortType = ScalarType('int')
export const Bool: PortType = ScalarType('bool')
export const Unit: PortType = { tag: 'unit' }
export const CoproductType = (summands: PortType[]): PortType => ({ tag: 'coproduct', summands })

/**
 * Construct an array type: element type with a static shape.
 * Shape is a list of dimension sizes, e.g. [4] for a vector, [4,4] for a matrix.
 */
export const ArrayType = (element: PortType, shape: number[]): PortType => {
  if (shape.length === 0) return element // shape [] degenerates to scalar
  return { tag: 'array', element, shape }
}

// ─────────────────────────────────────────────────────────────
// Shape algebra (numpy-style, static shapes)
// ─────────────────────────────────────────────────────────────

/**
 * Broadcast two shapes following numpy rules:
 * 1. Shapes are right-aligned
 * 2. For each dimension pair: equal sizes pass through, size-1 stretches, mismatch is an error
 * Returns the broadcasted shape, or null if incompatible.
 */
export function broadcastShapes(a: number[], b: number[]): number[] | null {
  const rank = Math.max(a.length, b.length)
  const result: number[] = new Array(rank)
  for (let i = 0; i < rank; i++) {
    const da = i < a.length ? a[a.length - 1 - i] : 1
    const db = i < b.length ? b[b.length - 1 - i] : 1
    if (da === db) {
      result[rank - 1 - i] = da
    } else if (da === 1) {
      result[rank - 1 - i] = db
    } else if (db === 1) {
      result[rank - 1 - i] = da
    } else {
      return null // incompatible
    }
  }
  return result
}

/** Compute row-major strides for a shape. */
export function shapeStrides(shape: number[]): number[] {
  const strides = new Array(shape.length)
  let stride = 1
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = stride
    stride *= shape[i]
  }
  return strides
}

/** Total number of elements in a shape. */
export function shapeSize(shape: number[]): number {
  let size = 1
  for (const d of shape) size *= d
  return size
}

/** Convert multi-dimensional indices to a flat index using row-major strides. */
export function flattenIndex(indices: number[], strides: number[]): number {
  let idx = 0
  for (let i = 0; i < indices.length; i++) idx += indices[i] * strides[i]
  return idx
}

/**
 * Count the total number of scalars in a PortType.
 * Scalars = 1, arrays = product(shape) * scalarCount(element), etc.
 */
export function scalarCount(t: PortType): number {
  switch (t.tag) {
    case 'scalar': return 1
    case 'unit': return 0
    case 'array': return shapeSize(t.shape) * scalarCount(t.element)
    case 'product': return t.factors.reduce((s, f) => s + scalarCount(f), 0)
    case 'coproduct': {
      const maxSummand = t.summands.reduce((max, s) => Math.max(max, scalarCount(s)), 0)
      return 1 + maxSummand  // tag + padded payload
    }
  }
}

/**
 * Build a product type, flattening nested products and eliminating units.
 * product([A]) = A, product([]) = Unit, product([A, Unit, B]) = product([A, B])
 */
export function product(factors: PortType[]): PortType {
  // Flatten nested products and filter units
  const flat: PortType[] = []
  for (const f of factors) {
    if (f.tag === 'unit') continue
    if (f.tag === 'product') flat.push(...f.factors)
    else flat.push(f)
  }
  if (flat.length === 0) return Unit
  if (flat.length === 1) return flat[0]
  return { tag: 'product', factors: flat }
}

// ─────────────────────────────────────────────────────────────
// Port type equality
// ─────────────────────────────────────────────────────────────

export function portTypeEqual(a: PortType, b: PortType): boolean {
  if (a.tag !== b.tag) return false
  switch (a.tag) {
    case 'scalar':
      return a.scalar === (b as typeof a).scalar
    case 'array': {
      const ba = b as typeof a
      if (a.shape.length !== ba.shape.length) return false
      if (!a.shape.every((d, i) => d === ba.shape[i])) return false
      return portTypeEqual(a.element, ba.element)
    }
    case 'product': {
      const bp = b as typeof a
      if (a.factors.length !== bp.factors.length) return false
      return a.factors.every((f, i) => portTypeEqual(f, bp.factors[i]))
    }
    case 'coproduct': {
      const bc = b as typeof a
      if (a.summands.length !== bc.summands.length) return false
      return a.summands.every((s, i) => portTypeEqual(s, bc.summands[i]))
    }
    case 'unit':
      return true
  }
}

/**
 * Human-readable string for a port type.
 */
export function portTypeToString(t: PortType): string {
  switch (t.tag) {
    case 'scalar': return t.scalar
    case 'array': return `${portTypeToString(t.element)}[${t.shape.join(',')}]`
    case 'product':
      return t.factors.map(portTypeToString).join(' ⊗ ')
    case 'coproduct':
      return t.summands.map(portTypeToString).join(' ⊕ ')
    case 'unit': return 'I'
  }
}

/**
 * Check if two array shapes are broadcast-compatible.
 * Returns the result shape, or null if incompatible.
 */
export function shapesCompatible(a: PortType, b: PortType): number[] | null {
  if (a.tag === 'array' && b.tag === 'array') {
    return broadcastShapes(a.shape, b.shape)
  }
  if (a.tag === 'array' && b.tag === 'scalar') return a.shape  // scalar broadcasts to any array
  if (a.tag === 'scalar' && b.tag === 'array') return b.shape
  if (a.tag === 'scalar' && b.tag === 'scalar') return []      // both scalar
  return null
}

// ─────────────────────────────────────────────────────────────
// Terms (morphisms of the free monoidal category)
// ─────────────────────────────────────────────────────────────

export type StateInit = number | boolean | number[] | number[][]

/** The body of a morphism — what it actually computes. */
export type MorphismBody =
  | { tag: 'expr'; inputNames: string[]; outputExprs: Record<string, ExprNode> }
  | { tag: 'primitive'; kind: PrimitiveKind }

export type PrimitiveKind =
  | 'sample_rate'
  | 'sample_index'

/**
 * Term in the free traced symmetric monoidal category.
 *
 *   id        : A → A                       (identity / wire)
 *   morphism  : A → B                       (a named module or expression)
 *   compose   : (A → B) × (B → C) → (A → C)  (sequential)
 *   tensor    : (A → B) × (C → D) → (A⊗C → B⊗D)  (parallel)
 *   trace     : (A⊗S → B⊗S) → (A → B)      (feedback with typed state)
 */
export type Term =
  | { tag: 'id'; portType: PortType }
  | { tag: 'morphism'; name: string; dom: PortType; cod: PortType; body: MorphismBody }
  | { tag: 'compose'; first: Term; second: Term }
  | { tag: 'tensor'; left: Term; right: Term }
  | { tag: 'trace'; stateType: PortType; init: StateInit; body: Term }

// Term constructors

export const id = (portType: PortType): Term => ({ tag: 'id', portType })

export const morphism = (
  name: string,
  dom: PortType,
  cod: PortType,
  body: MorphismBody,
): Term => ({ tag: 'morphism', name, dom, cod, body })

export function compose(first: Term, second: Term): Term {
  return { tag: 'compose', first, second }
}

export function tensor(left: Term, right: Term): Term {
  return { tag: 'tensor', left, right }
}

export function trace(stateType: PortType, init: StateInit, body: Term): Term {
  return { tag: 'trace', stateType, init, body }
}

/**
 * Compose a sequence of terms left-to-right: composeAll([f, g, h]) = f ; g ; h
 */
export function composeAll(terms: Term[]): Term {
  if (terms.length === 0) throw new Error('composeAll: empty list')
  return terms.reduce(compose)
}

/**
 * Tensor a list of terms: tensorAll([f, g, h]) = f ⊗ g ⊗ h
 * Returns id(Unit) for empty list.
 */
export function tensorAll(terms: Term[]): Term {
  if (terms.length === 0) return id(Unit)
  return terms.reduce(tensor)
}
