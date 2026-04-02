/**
 * optimizer.test.ts — Tests for categorical term rewriting passes.
 */

import { describe, test, expect } from 'bun:test'
import * as fc from 'fast-check'
import {
  optimize,
  eliminateIdentity,
  flattenCompose,
  flattenTensor,
  termSize,
  composeDepth,
  termEqual,
} from './optimizer'
import {
  type PortType, type Term,
  Float, Int, Bool, Unit,
  id, morphism, compose, tensor, trace,
  product, portTypeEqual,
} from './term'
import { inferType } from './type_check'

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────

const body = { tag: 'expr' as const, inputNames: [] as string[], outputExprs: {} }
const f = morphism('f', Float, Int, body)
const g = morphism('g', Int, Bool, body)
const h = morphism('h', Bool, Float, body)
const p = morphism('p', Float, Float, body)
const q = morphism('q', Float, Float, body)

// ─────────────────────────────────────────────────────────────
// termEqual
// ─────────────────────────────────────────────────────────────

describe('termEqual', () => {
  test('same morphism', () => {
    expect(termEqual(f, f)).toBe(true)
  })

  test('different morphisms', () => {
    expect(termEqual(f, g)).toBe(false)
  })

  test('id vs morphism', () => {
    expect(termEqual(id(Float), f)).toBe(false)
  })

  test('compose equality', () => {
    expect(termEqual(compose(f, g), compose(f, g))).toBe(true)
    expect(termEqual(compose(f, g), compose(g, h))).toBe(false)
  })

  test('tensor equality', () => {
    expect(termEqual(tensor(f, g), tensor(f, g))).toBe(true)
    expect(termEqual(tensor(f, g), tensor(g, f))).toBe(false)
  })
})

// ─────────────────────────────────────────────────────────────
// termSize
// ─────────────────────────────────────────────────────────────

describe('termSize', () => {
  test('leaf nodes', () => {
    expect(termSize(f)).toBe(1)
    expect(termSize(id(Float))).toBe(1)
  })

  test('compose', () => {
    expect(termSize(compose(f, g))).toBe(3)
  })

  test('nested', () => {
    expect(termSize(compose(compose(f, g), h))).toBe(5)
  })
})

// ─────────────────────────────────────────────────────────────
// Identity elimination
// ─────────────────────────────────────────────────────────────

describe('eliminateIdentity', () => {
  test('left identity: compose(id, f) → f', () => {
    const term = compose(id(Float), f)
    const opt = eliminateIdentity(term)
    expect(termEqual(opt, f)).toBe(true)
  })

  test('right identity: compose(f, id) → f', () => {
    const term = compose(f, id(Int))
    const opt = eliminateIdentity(term)
    expect(termEqual(opt, f)).toBe(true)
  })

  test('both identities: compose(id, id) → id', () => {
    const term = compose(id(Float), id(Float))
    const opt = eliminateIdentity(term)
    expect(opt.tag).toBe('id')
  })

  test('right unit: tensor(f, id(Unit)) → f', () => {
    const term = tensor(f, id(Unit))
    const opt = eliminateIdentity(term)
    expect(termEqual(opt, f)).toBe(true)
  })

  test('left unit: tensor(id(Unit), f) → f', () => {
    const term = tensor(id(Unit), f)
    const opt = eliminateIdentity(term)
    expect(termEqual(opt, f)).toBe(true)
  })

  test('non-Unit id in tensor preserved', () => {
    const term = tensor(f, id(Float))
    const opt = eliminateIdentity(term)
    // Should NOT eliminate — id(Float) is not id(Unit)
    expect(opt.tag).toBe('tensor')
  })

  test('nested identity elimination', () => {
    // compose(id, compose(f, id)) → f
    const term = compose(id(Float), compose(f, id(Int)))
    const opt = eliminateIdentity(term)
    expect(termEqual(opt, f)).toBe(true)
  })

  test('identity inside trace body', () => {
    const inner = compose(id(Float), p)
    const term = trace(Float, 0, compose(inner, morphism('fb', Float, product([Float, Float]), body)))
    // The trace body should have the id eliminated
    const opt = eliminateIdentity(term)
    expect(opt.tag).toBe('trace')
    if (opt.tag === 'trace') {
      // The inner compose(id, p) should become just p
      expect(opt.body.tag).toBe('compose')
    }
  })

  test('no change when no identities', () => {
    const term = compose(f, g)
    const opt = eliminateIdentity(term)
    expect(termEqual(opt, term)).toBe(true)
  })

  test('reduces term size', () => {
    const term = compose(id(Float), compose(f, id(Int)))
    expect(termSize(term)).toBe(5) // compose, id, compose, f, id
    const opt = eliminateIdentity(term)
    expect(termSize(opt)).toBe(1) // just f
  })
})

// ─────────────────────────────────────────────────────────────
// Compose flattening
// ─────────────────────────────────────────────────────────────

describe('flattenCompose', () => {
  test('left-associated → right-associated', () => {
    // compose(compose(f, g), h) → compose(f, compose(g, h))
    const term = compose(compose(f, g), h)
    const opt = flattenCompose(term)
    // Should be right-associated: compose(f, compose(g, h))
    expect(opt.tag).toBe('compose')
    if (opt.tag === 'compose') {
      expect(termEqual(opt.first, f)).toBe(true)
      expect(opt.second.tag).toBe('compose')
      if (opt.second.tag === 'compose') {
        expect(termEqual(opt.second.first, g)).toBe(true)
        expect(termEqual(opt.second.second, h)).toBe(true)
      }
    }
  })

  test('already right-associated — no change', () => {
    const term = compose(f, compose(g, h))
    const opt = flattenCompose(term)
    expect(termEqual(opt, term)).toBe(true)
  })

  test('deeply nested left-association', () => {
    // ((f ; g) ; h) ; p = f ; (g ; (h ; p))
    const fgh = compose(compose(f, g), h) // f:F→I, g:I→B, h:B→F
    const term = compose(fgh, p) // p:F→F
    const opt = flattenCompose(term)
    // Should be: compose(f, compose(g, compose(h, p)))
    expect(composeDepth(term)).toBe(3) // left-leaning
    expect(composeDepth(opt)).toBe(1)  // right-leaning (first is f, not compose)
  })

  test('single term unchanged', () => {
    expect(termEqual(flattenCompose(f), f)).toBe(true)
  })

  test('flattens inside tensor', () => {
    const inner = compose(compose(p, p), p) // left-associated
    const term = tensor(inner, f)
    const opt = flattenCompose(term)
    expect(opt.tag).toBe('tensor')
    if (opt.tag === 'tensor') {
      // Left side should be right-associated (depth 1: root compose, then leaf)
      expect(composeDepth(opt.left)).toBe(1)
    }
  })

  test('flattens inside trace body', () => {
    const innerBody = morphism('fb', product([Float, Float]), product([Float, Float]), body)
    const leftAssoc = compose(compose(morphism('a', product([Float, Float]), product([Float, Float]), body), innerBody), innerBody)
    const term = trace(Float, 0, leftAssoc)
    const opt = flattenCompose(term)
    expect(opt.tag).toBe('trace')
    if (opt.tag === 'trace') {
      expect(composeDepth(opt.body)).toBe(1)
    }
  })
})

// ─────────────────────────────────────────────────────────────
// Tensor flattening
// ─────────────────────────────────────────────────────────────

describe('flattenTensor', () => {
  test('left-associated → right-associated', () => {
    const term = tensor(tensor(f, g), h)
    const opt = flattenTensor(term)
    expect(opt.tag).toBe('tensor')
    if (opt.tag === 'tensor') {
      expect(termEqual(opt.left, f)).toBe(true)
      expect(opt.right.tag).toBe('tensor')
    }
  })

  test('already right-associated — no change', () => {
    const term = tensor(f, tensor(g, h))
    const opt = flattenTensor(term)
    expect(termEqual(opt, term)).toBe(true)
  })

  test('single term unchanged', () => {
    expect(termEqual(flattenTensor(f), f)).toBe(true)
  })
})

// ─────────────────────────────────────────────────────────────
// Full optimizer
// ─────────────────────────────────────────────────────────────

describe('optimize', () => {
  test('combines identity elimination and flattening', () => {
    // compose(id, compose(compose(f, id), g))
    const term = compose(id(Float), compose(compose(f, id(Int)), g))
    const opt = optimize(term)
    // Should reduce to compose(f, g), right-associated
    expect(termEqual(opt, compose(f, g))).toBe(true)
  })

  test('idempotent: optimize(optimize(t)) = optimize(t)', () => {
    const term = compose(id(Float), compose(compose(f, id(Int)), g))
    const once = optimize(term)
    const twice = optimize(once)
    expect(termEqual(once, twice)).toBe(true)
  })

  test('empty tensor units eliminated', () => {
    const term = tensor(tensor(f, id(Unit)), id(Unit))
    const opt = optimize(term)
    expect(termEqual(opt, f)).toBe(true)
  })
})

// ─────────────────────────────────────────────────────────────
// Property-based: type preservation
// ─────────────────────────────────────────────────────────────

/** Generate a random scalar PortType. */
const arbScalar: fc.Arbitrary<PortType> = fc.constantFrom(Float, Int, Bool)

/** Generate a morphism between two types. */
function arbMorphism(dom: PortType, cod: PortType): fc.Arbitrary<Term> {
  return fc.string({ minLength: 1, maxLength: 6 }).map(name =>
    morphism(name, dom, cod, body)
  )
}

/** Generate a well-typed term with random domain and codomain. */
function arbTerm(maxDepth: number): fc.Arbitrary<{ term: Term; dom: PortType; cod: PortType }> {
  return fc.tuple(arbScalar, arbScalar).chain(([dom, cod]) =>
    arbTermTyped(dom, cod, maxDepth).map(term => ({ term, dom, cod }))
  )
}

function arbTermTyped(dom: PortType, cod: PortType, depth: number): fc.Arbitrary<Term> {
  if (depth <= 0) return arbMorphism(dom, cod)
  return fc.oneof(
    { weight: 2, arbitrary: arbMorphism(dom, cod) },
    // identity (only when types match)
    ...(portTypeEqual(dom, cod) ? [{ weight: 1, arbitrary: fc.constant(id(dom)) }] : []),
    // compose with random midpoint
    {
      weight: 2,
      arbitrary: arbScalar.chain(mid =>
        fc.tuple(
          arbTermTyped(dom, mid, depth - 1),
          arbTermTyped(mid, cod, depth - 1),
        ).map(([a, b]) => compose(a, b))
      ),
    },
  )
}

describe('type preservation (property-based)', () => {
  test('eliminateIdentity preserves types', () => {
    fc.assert(
      fc.property(arbTerm(2), ({ term, dom, cod }) => {
        const opt = eliminateIdentity(term)
        const t = inferType(opt)
        return portTypeEqual(t.dom, dom) && portTypeEqual(t.cod, cod)
      }),
      { numRuns: 300 },
    )
  })

  test('flattenCompose preserves types', () => {
    fc.assert(
      fc.property(arbTerm(2), ({ term, dom, cod }) => {
        const opt = flattenCompose(term)
        const t = inferType(opt)
        return portTypeEqual(t.dom, dom) && portTypeEqual(t.cod, cod)
      }),
      { numRuns: 300 },
    )
  })

  test('optimize preserves types', () => {
    fc.assert(
      fc.property(arbTerm(2), ({ term, dom, cod }) => {
        const opt = optimize(term)
        const t = inferType(opt)
        return portTypeEqual(t.dom, dom) && portTypeEqual(t.cod, cod)
      }),
      { numRuns: 300 },
    )
  })

  test('optimize never increases term size', () => {
    fc.assert(
      fc.property(arbTerm(2), ({ term }) => {
        const opt = optimize(term)
        return termSize(opt) <= termSize(term)
      }),
      { numRuns: 300 },
    )
  })

  test('optimize is idempotent', () => {
    fc.assert(
      fc.property(arbTerm(2), ({ term }) => {
        const once = optimize(term)
        const twice = optimize(once)
        return termEqual(once, twice)
      }),
      { numRuns: 300 },
    )
  })
})
