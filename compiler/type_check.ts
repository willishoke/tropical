/**
 * type_check.ts — Type inference and validation for categorical terms.
 *
 * Every term has a domain (input type) and codomain (output type).
 * Composition requires cod(first) = dom(second).
 * Trace requires body : A⊗S → B⊗S, inferring S from the body's type.
 */

import {
  type PortType,
  type Term,
  Unit,
  product,
  portTypeEqual,
  portTypeToString,
  broadcastShapes,
} from './term'

// ─────────────────────────────────────────────────────────────
// Type inference
// ─────────────────────────────────────────────────────────────

export interface MorphismType {
  dom: PortType
  cod: PortType
}

export class TypeError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'TypeError'
  }
}

/**
 * Infer the domain and codomain of a term.
 * Throws TypeError if the term is ill-typed.
 */
export function inferType(term: Term): MorphismType {
  switch (term.tag) {
    case 'id':
      return { dom: term.portType, cod: term.portType }

    case 'morphism':
      return { dom: term.dom, cod: term.cod }

    case 'compose': {
      const first = inferType(term.first)
      const second = inferType(term.second)
      if (!portTypeEqual(first.cod, second.dom)) {
        throw new TypeError(
          `Composition type mismatch: first term has codomain ${portTypeToString(first.cod)} ` +
          `but second term has domain ${portTypeToString(second.dom)}`
        )
      }
      return { dom: first.dom, cod: second.cod }
    }

    case 'tensor': {
      const left = inferType(term.left)
      const right = inferType(term.right)
      return {
        dom: product([left.dom, right.dom]),
        cod: product([left.cod, right.cod]),
      }
    }

    case 'trace': {
      const bodyType = inferType(term.body)
      // body must be A⊗S → B⊗S
      // Extract S from the body's domain and codomain
      const domFactors = splitTraceType(bodyType.dom, term.stateType)
      const codFactors = splitTraceType(bodyType.cod, term.stateType)

      if (domFactors === null) {
        throw new TypeError(
          `Trace: body domain ${portTypeToString(bodyType.dom)} does not contain ` +
          `state type ${portTypeToString(term.stateType)}`
        )
      }
      if (codFactors === null) {
        throw new TypeError(
          `Trace: body codomain ${portTypeToString(bodyType.cod)} does not contain ` +
          `state type ${portTypeToString(term.stateType)}`
        )
      }

      return { dom: domFactors.rest, cod: codFactors.rest }
    }
  }
}

/**
 * Given a type that should be of the form A⊗S, split out S and return the rest (A).
 * S is expected to be the last factor in a product.
 * Returns null if S is not found.
 */
function splitTraceType(
  t: PortType,
  stateType: PortType,
): { rest: PortType } | null {
  // If the whole thing is the state type, the "rest" is Unit
  if (portTypeEqual(t, stateType)) {
    return { rest: Unit }
  }

  // If it's a product, check if the trailing factors match the state type
  if (t.tag === 'product') {
    const factors = t.factors

    if (stateType.tag === 'product') {
      // Compound state: match last N factors against state's N factors
      const stateFactors = stateType.factors
      const n = stateFactors.length
      if (factors.length <= n) return null
      const tail = factors.slice(factors.length - n)
      if (tail.every((f, i) => portTypeEqual(f, stateFactors[i]))) {
        return { rest: product(factors.slice(0, factors.length - n)) }
      }
      return null
    }

    // Scalar state: match last single factor
    if (factors.length < 2) return null
    const last = factors[factors.length - 1]
    if (portTypeEqual(last, stateType)) {
      return { rest: product(factors.slice(0, -1)) }
    }

    return null
  }

  return null
}

/**
 * Type-check a term, returning the inferred type.
 * Convenience wrapper that catches and rethrows with context.
 */
export function typeCheck(term: Term, context?: string): MorphismType {
  try {
    return inferType(term)
  } catch (e) {
    if (e instanceof TypeError && context) {
      throw new TypeError(`${context}: ${e.message}`)
    }
    throw e
  }
}
