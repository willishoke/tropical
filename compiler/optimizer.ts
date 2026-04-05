/**
 * optimizer.ts — Categorical term rewriting passes.
 *
 * Each pass is a Term → Term function that preserves semantics:
 *   same domain, same codomain, same behavior.
 *
 * Current passes (structural — don't inspect morphism bodies):
 * 1. Identity elimination: compose(id, f) → f, tensor(f, id(Unit)) → f
 * 2. Compose flattening: right-associate for uniform pipeline structure
 * 3. Tensor flattening: right-associate nested tensors
 *
 * Future passes (require body inspection):
 * - Trace fusion: merge adjacent traces into single trace with product state
 * - Dead wire elimination: remove unused tensor factors
 * - Constant folding: all-literal morphism bodies → constants
 * - Wiring fusion: merge consecutive wiring morphisms
 */

import {
  type Term,
  compose, tensor, trace, id,
} from './term'

// ─────────────────────────────────────────────────────────────
// Main optimizer
// ─────────────────────────────────────────────────────────────

/** Run all optimization passes on a term. Iterates to a fixed point. */
export function optimize(term: Term): Term {
  let prev = term
  for (let i = 0; i < 10; i++) {
    let t = prev
    t = eliminateIdentity(t)
    t = flattenCompose(t)
    t = flattenTensor(t)
    if (termEqual(t, prev)) return t
    prev = t
  }
  return prev
}

// ─────────────────────────────────────────────────────────────
// Pass 1: Identity elimination
// ─────────────────────────────────────────────────────────────

/**
 * Remove identity morphisms from compositions and tensors.
 *
 * Laws applied:
 *   compose(id(A), f) = f           (left identity)
 *   compose(f, id(A)) = f           (right identity)
 *   tensor(f, id(I))  = f           (right unit)
 *   tensor(id(I), f)  = f           (left unit)
 */
export function eliminateIdentity(term: Term): Term {
  switch (term.tag) {
    case 'id':
    case 'morphism':
      return term

    case 'compose': {
      const first = eliminateIdentity(term.first)
      const second = eliminateIdentity(term.second)
      if (first.tag === 'id') return second
      if (second.tag === 'id') return first
      return compose(first, second)
    }

    case 'tensor': {
      const left = eliminateIdentity(term.left)
      const right = eliminateIdentity(term.right)
      if (right.tag === 'id' && right.portType.tag === 'unit') return left
      if (left.tag === 'id' && left.portType.tag === 'unit') return right
      return tensor(left, right)
    }

    case 'trace':
      return trace(term.stateType, term.init, eliminateIdentity(term.body))
  }
}

// ─────────────────────────────────────────────────────────────
// Pass 2: Compose flattening (right-association)
// ─────────────────────────────────────────────────────────────

/** Collect all terms in a composition chain into a flat list. */
function collectCompose(term: Term): Term[] {
  if (term.tag !== 'compose') return [term]
  return [...collectCompose(term.first), ...collectCompose(term.second)]
}

/** Right-fold a list of terms into a right-associated compose chain. */
function rightFoldCompose(terms: Term[]): Term {
  if (terms.length === 1) return terms[0]
  return compose(terms[0], rightFoldCompose(terms.slice(1)))
}

/**
 * Right-associate all compositions.
 *
 *   compose(compose(f, g), h)  →  compose(f, compose(g, h))
 *
 * Produces a uniform right-leaning spine, natural for left-to-right
 * pipeline execution.
 */
export function flattenCompose(term: Term): Term {
  switch (term.tag) {
    case 'id':
    case 'morphism':
      return term

    case 'compose': {
      const first = flattenCompose(term.first)
      const second = flattenCompose(term.second)
      const all = [...collectCompose(first), ...collectCompose(second)]
      return all.length === 1 ? all[0] : rightFoldCompose(all)
    }

    case 'tensor':
      return tensor(flattenCompose(term.left), flattenCompose(term.right))

    case 'trace':
      return trace(term.stateType, term.init, flattenCompose(term.body))
  }
}

// ─────────────────────────────────────────────────────────────
// Pass 3: Tensor flattening (right-association)
// ─────────────────────────────────────────────────────────────

/** Collect all terms in a tensor chain into a flat list. */
function collectTensor(term: Term): Term[] {
  if (term.tag !== 'tensor') return [term]
  return [...collectTensor(term.left), ...collectTensor(term.right)]
}

/** Right-fold a list of terms into a right-associated tensor chain. */
function rightFoldTensor(terms: Term[]): Term {
  if (terms.length === 1) return terms[0]
  return tensor(terms[0], rightFoldTensor(terms.slice(1)))
}

/**
 * Right-associate all tensor products.
 *
 *   tensor(tensor(f, g), h)  →  tensor(f, tensor(g, h))
 */
export function flattenTensor(term: Term): Term {
  switch (term.tag) {
    case 'id':
    case 'morphism':
      return term

    case 'compose':
      return compose(flattenTensor(term.first), flattenTensor(term.second))

    case 'tensor': {
      const left = flattenTensor(term.left)
      const right = flattenTensor(term.right)
      const all = [...collectTensor(left), ...collectTensor(right)]
      return all.length === 1 ? all[0] : rightFoldTensor(all)
    }

    case 'trace':
      return trace(term.stateType, term.init, flattenTensor(term.body))
  }
}

// ─────────────────────────────────────────────────────────────
// Term utilities
// ─────────────────────────────────────────────────────────────

/** Count the total number of nodes in a term tree. */
export function termSize(term: Term): number {
  switch (term.tag) {
    case 'id':
    case 'morphism':
      return 1
    case 'compose':
      return 1 + termSize(term.first) + termSize(term.second)
    case 'tensor':
      return 1 + termSize(term.left) + termSize(term.right)
    case 'trace':
      return 1 + termSize(term.body)
  }
}

/** Count the depth of the deepest compose chain (left-spine length). */
export function composeDepth(term: Term): number {
  if (term.tag !== 'compose') return 0
  return 1 + composeDepth(term.first)
}

/** Structural equality of terms (ignores morphism bodies). */
export function termEqual(a: Term, b: Term): boolean {
  if (a.tag !== b.tag) return false
  switch (a.tag) {
    case 'id':
      return a.portType === (b as typeof a).portType ||
        JSON.stringify(a.portType) === JSON.stringify((b as typeof a).portType)
    case 'morphism': {
      const bm = b as typeof a
      return a.name === bm.name &&
        JSON.stringify(a.dom) === JSON.stringify(bm.dom) &&
        JSON.stringify(a.cod) === JSON.stringify(bm.cod)
    }
    case 'compose': {
      const bc = b as typeof a
      return termEqual(a.first, bc.first) && termEqual(a.second, bc.second)
    }
    case 'tensor': {
      const bt = b as typeof a
      return termEqual(a.left, bt.left) && termEqual(a.right, bt.right)
    }
    case 'trace': {
      const btr = b as typeof a
      return JSON.stringify(a.stateType) === JSON.stringify(btr.stateType) &&
        termEqual(a.body, btr.body)
    }
  }
}
