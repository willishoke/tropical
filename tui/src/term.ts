/**
 * term.ts — Free monoidal category term language for egress.
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
  | { tag: 'struct'; name: string }
  | { tag: 'sum'; name: string }
  | { tag: 'function'; params: PortType[]; returns: PortType }
  | { tag: 'product'; factors: PortType[] }
  | { tag: 'unit' }

// Constructors
export const ScalarType = (s: ScalarKind): PortType => ({ tag: 'scalar', scalar: s })
export const Float: PortType = ScalarType('float')
export const Int: PortType = ScalarType('int')
export const Bool: PortType = ScalarType('bool')
export const Unit: PortType = { tag: 'unit' }
export const StructType = (name: string): PortType => ({ tag: 'struct', name })
export const SumType = (name: string): PortType => ({ tag: 'sum', name })
export const FunctionType = (params: PortType[], returns: PortType): PortType =>
  ({ tag: 'function', params, returns })

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
    case 'struct':
    case 'sum':
      return a.name === (b as typeof a).name
    case 'function': {
      const bf = b as typeof a
      if (a.params.length !== bf.params.length) return false
      return a.params.every((p, i) => portTypeEqual(p, bf.params[i])) &&
             portTypeEqual(a.returns, bf.returns)
    }
    case 'product': {
      const bp = b as typeof a
      if (a.factors.length !== bp.factors.length) return false
      return a.factors.every((f, i) => portTypeEqual(f, bp.factors[i]))
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
    case 'struct': return t.name
    case 'sum': return t.name
    case 'function':
      return `(${t.params.map(portTypeToString).join(', ')}) → ${portTypeToString(t.returns)}`
    case 'product':
      return t.factors.map(portTypeToString).join(' ⊗ ')
    case 'unit': return 'I'
  }
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
