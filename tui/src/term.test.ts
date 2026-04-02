/**
 * term.test.ts — Property-based and unit tests for the categorical term system.
 *
 * Tests the Term ADT, type inference, and categorical laws using fast-check.
 */

import { describe, test, expect } from 'bun:test'
import * as fc from 'fast-check'
import {
  type PortType,
  type Term,
  Float, Int, Bool, Unit,
  ScalarType, StructType, SumType, FunctionType,
  product,
  portTypeEqual,
  portTypeToString,
  id, morphism, compose, tensor, trace,
  composeAll, tensorAll,
} from './term'
import { inferType, typeCheck, TypeError } from './type_check'
import { MorphismRegistry, type MorphismDef } from './morphism_registry'

// ─────────────────────────────────────────────────────────────
// Arbitrary generators
// ─────────────────────────────────────────────────────────────

/** Generate a random scalar PortType. */
const arbScalar: fc.Arbitrary<PortType> = fc.constantFrom(Float, Int, Bool)

/** Generate a random PortType (limited depth to avoid explosion). */
const arbPortType: fc.Arbitrary<PortType> = fc.letrec<{ type: PortType }>(tie => ({
  type: fc.oneof(
    { weight: 5, arbitrary: arbScalar },
    { weight: 1, arbitrary: fc.constant(Unit) },
    { weight: 2, arbitrary: fc.string({ minLength: 1, maxLength: 8 }).map(StructType) },
    {
      weight: 1,
      arbitrary: fc.tuple(
        fc.array(tie('type'), { minLength: 1, maxLength: 3 }),
        tie('type'),
      ).map(([params, ret]) => FunctionType(params, ret)),
    },
    {
      weight: 2,
      arbitrary: fc.array(tie('type'), { minLength: 2, maxLength: 4 }).map(product),
    },
  ),
})).type

/** Generate a well-typed morphism term with given domain and codomain. */
function arbMorphism(dom: PortType, cod: PortType): fc.Arbitrary<Term> {
  return fc.string({ minLength: 1, maxLength: 8 }).map(name =>
    morphism(name, dom, cod, { tag: 'expr', inputNames: [], outputExprs: {} })
  )
}

/**
 * Generate a random well-typed term given a fixed domain and codomain.
 * This avoids the filter() trap — we never generate terms that might not compose.
 */
function arbTermTyped(dom: PortType, cod: PortType, maxDepth: number): fc.Arbitrary<Term> {
  if (maxDepth <= 0) {
    return arbMorphism(dom, cod)
  }

  return fc.oneof(
    // Base: morphism
    { weight: 3, arbitrary: arbMorphism(dom, cod) },

    // Compose: pick a random midpoint type, generate f: dom→mid and g: mid→cod
    {
      weight: 2,
      arbitrary: arbPortType.chain(mid =>
        fc.tuple(
          arbTermTyped(dom, mid, maxDepth - 1),
          arbTermTyped(mid, cod, maxDepth - 1),
        ).map(([f, g]) => compose(f, g))
      ),
    },
  )
}

/**
 * Generate a random well-typed term with random domain and codomain.
 */
function arbTerm(maxDepth: number): fc.Arbitrary<{ term: Term; dom: PortType; cod: PortType }> {
  return fc.tuple(arbPortType, arbPortType).chain(([dom, cod]) =>
    arbTermTyped(dom, cod, maxDepth).map(term => ({ term, dom, cod }))
  )
}

// ─────────────────────────────────────────────────────────────
// Unit tests: PortType
// ─────────────────────────────────────────────────────────────

describe('PortType', () => {
  test('scalar equality', () => {
    expect(portTypeEqual(Float, Float)).toBe(true)
    expect(portTypeEqual(Float, Int)).toBe(false)
    expect(portTypeEqual(Bool, Bool)).toBe(true)
  })

  test('unit equality', () => {
    expect(portTypeEqual(Unit, Unit)).toBe(true)
    expect(portTypeEqual(Unit, Float)).toBe(false)
  })

  test('struct equality by name', () => {
    expect(portTypeEqual(StructType('Foo'), StructType('Foo'))).toBe(true)
    expect(portTypeEqual(StructType('Foo'), StructType('Bar'))).toBe(false)
  })

  test('function type equality', () => {
    const f1 = FunctionType([Float, Int], Bool)
    const f2 = FunctionType([Float, Int], Bool)
    const f3 = FunctionType([Float], Bool)
    expect(portTypeEqual(f1, f2)).toBe(true)
    expect(portTypeEqual(f1, f3)).toBe(false)
  })

  test('product flattening', () => {
    // product([A, product([B, C])]) should flatten to product([A, B, C])
    const inner = product([Int, Bool])
    const outer = product([Float, inner])
    expect(outer.tag).toBe('product')
    if (outer.tag === 'product') {
      expect(outer.factors.length).toBe(3)
    }
  })

  test('product unit elimination', () => {
    expect(portTypeEqual(product([Float, Unit]), Float)).toBe(true)
    expect(portTypeEqual(product([Unit, Unit]), Unit)).toBe(true)
    expect(portTypeEqual(product([]), Unit)).toBe(true)
  })

  test('product singleton unwrap', () => {
    expect(portTypeEqual(product([Float]), Float)).toBe(true)
  })

  test('portTypeToString', () => {
    expect(portTypeToString(Float)).toBe('float')
    expect(portTypeToString(Unit)).toBe('I')
    expect(portTypeToString(product([Float, Int]))).toBe('float ⊗ int')
    expect(portTypeToString(FunctionType([Float], Int))).toBe('(float) → int')
  })
})

// ─────────────────────────────────────────────────────────────
// Unit tests: Type inference
// ─────────────────────────────────────────────────────────────

describe('inferType', () => {
  const body = { tag: 'expr' as const, inputNames: [], outputExprs: {} }

  test('identity', () => {
    const t = inferType(id(Float))
    expect(portTypeEqual(t.dom, Float)).toBe(true)
    expect(portTypeEqual(t.cod, Float)).toBe(true)
  })

  test('morphism', () => {
    const t = inferType(morphism('f', Float, Int, body))
    expect(portTypeEqual(t.dom, Float)).toBe(true)
    expect(portTypeEqual(t.cod, Int)).toBe(true)
  })

  test('valid composition', () => {
    const f = morphism('f', Float, Int, body)
    const g = morphism('g', Int, Bool, body)
    const t = inferType(compose(f, g))
    expect(portTypeEqual(t.dom, Float)).toBe(true)
    expect(portTypeEqual(t.cod, Bool)).toBe(true)
  })

  test('invalid composition throws', () => {
    const f = morphism('f', Float, Int, body)
    const g = morphism('g', Bool, Float, body)
    expect(() => inferType(compose(f, g))).toThrow(TypeError)
  })

  test('tensor', () => {
    const f = morphism('f', Float, Int, body)
    const g = morphism('g', Bool, Float, body)
    const t = inferType(tensor(f, g))
    expect(portTypeEqual(t.dom, product([Float, Bool]))).toBe(true)
    expect(portTypeEqual(t.cod, product([Int, Float]))).toBe(true)
  })

  test('trace with product state', () => {
    // body : (Float ⊗ Int) → (Bool ⊗ Int), state = Int
    // trace should produce Float → Bool
    const dom = product([Float, Int])
    const cod = product([Bool, Int])
    const f = morphism('f', dom, cod, body)
    const t = inferType(trace(Int, 0, f))
    expect(portTypeEqual(t.dom, Float)).toBe(true)
    expect(portTypeEqual(t.cod, Bool)).toBe(true)
  })

  test('trace where entire type is state', () => {
    // body : Float → Float, state = Float
    // trace should produce Unit → Unit
    const f = morphism('f', Float, Float, body)
    const t = inferType(trace(Float, 0.0, f))
    expect(portTypeEqual(t.dom, Unit)).toBe(true)
    expect(portTypeEqual(t.cod, Unit)).toBe(true)
  })

  test('trace with wrong state type throws', () => {
    const f = morphism('f', Float, Int, body)
    expect(() => inferType(trace(Bool, false, f))).toThrow(TypeError)
  })
})

// ─────────────────────────────────────────────────────────────
// Unit tests: Morphism registry
// ─────────────────────────────────────────────────────────────

describe('MorphismRegistry', () => {
  test('register and find', () => {
    const reg = new MorphismRegistry()
    const def: MorphismDef = {
      name: 'round',
      fromType: Float,
      toType: Int,
      body: { op: 'round', a: { op: 'input', id: 0 } },
    }
    reg.register(def)
    expect(reg.findMorphisms(Float, Int)).toHaveLength(1)
    expect(reg.findMorphisms(Int, Float)).toHaveLength(0)
  })

  test('canonical morphism', () => {
    const reg = new MorphismRegistry()
    reg.register({ name: 'round', fromType: Float, toType: Int, body: 0 })
    reg.register({ name: 'truncate', fromType: Float, toType: Int, body: 0 })
    expect(reg.findCanonical(Float, Int)).toBeUndefined()

    reg.setCanonical('round')
    expect(reg.findCanonical(Float, Int)?.name).toBe('round')
  })

  test('duplicate name throws', () => {
    const reg = new MorphismRegistry()
    reg.register({ name: 'round', fromType: Float, toType: Int, body: 0 })
    expect(() =>
      reg.register({ name: 'round', fromType: Int, toType: Float, body: 0 })
    ).toThrow()
  })

  test('set canonical for unknown name throws', () => {
    const reg = new MorphismRegistry()
    expect(() => reg.setCanonical('nonexistent')).toThrow()
  })
})

// ─────────────────────────────────────────────────────────────
// Property-based tests: Categorical laws
// ─────────────────────────────────────────────────────────────

describe('categorical laws (property-based)', () => {
  const body = { tag: 'expr' as const, inputNames: [], outputExprs: {} }

  test('identity law: compose(id, f) has same type as f', () => {
    fc.assert(
      fc.property(arbTerm(1), ({ term, dom, cod }) => {
        const withId = compose(id(dom), term)
        const t = inferType(withId)
        return portTypeEqual(t.dom, dom) && portTypeEqual(t.cod, cod)
      }),
      { numRuns: 200 },
    )
  })

  test('identity law: compose(f, id) has same type as f', () => {
    fc.assert(
      fc.property(arbTerm(1), ({ term, dom, cod }) => {
        const withId = compose(term, id(cod))
        const t = inferType(withId)
        return portTypeEqual(t.dom, dom) && portTypeEqual(t.cod, cod)
      }),
      { numRuns: 200 },
    )
  })

  test('associativity: compose(compose(f,g),h) same type as compose(f,compose(g,h))', () => {
    // Use a fixed intermediate type to ensure composability
    fc.assert(
      fc.property(
        arbPortType, arbPortType, arbPortType, arbPortType,
        (a, b, c, d) => {
          const f = morphism('f', a, b, body)
          const g = morphism('g', b, c, body)
          const h = morphism('h', c, d, body)
          const left = inferType(compose(compose(f, g), h))
          const right = inferType(compose(f, compose(g, h)))
          return portTypeEqual(left.dom, right.dom) && portTypeEqual(left.cod, right.cod)
        },
      ),
      { numRuns: 200 },
    )
  })

  test('tensor is bifunctorial: types compose correctly', () => {
    fc.assert(
      fc.property(arbTerm(0), arbTerm(0), ({ term: f, dom: a, cod: b }, { term: g, dom: c, cod: d }) => {
        const t = inferType(tensor(f, g))
        return portTypeEqual(t.dom, product([a, c])) && portTypeEqual(t.cod, product([b, d]))
      }),
      { numRuns: 200 },
    )
  })

  test('interchange law: tensor(compose(f1,g1), compose(f2,g2)) same type as compose(tensor(f1,f2), tensor(g1,g2))', () => {
    fc.assert(
      fc.property(
        arbPortType, arbPortType, arbPortType, arbPortType, arbPortType, arbPortType,
        (a, b, c, d, e, f_type) => {
          const f1 = morphism('f1', a, b, body)
          const g1 = morphism('g1', b, c, body)
          const f2 = morphism('f2', d, e, body)
          const g2 = morphism('g2', e, f_type, body)

          const left = inferType(tensor(compose(f1, g1), compose(f2, g2)))
          const right = inferType(compose(tensor(f1, f2), tensor(g1, g2)))

          return portTypeEqual(left.dom, right.dom) && portTypeEqual(left.cod, right.cod)
        },
      ),
      { numRuns: 200 },
    )
  })

  test('unit law: tensor(f, id(Unit)) has same type as f', () => {
    fc.assert(
      fc.property(arbTerm(1), ({ term, dom, cod }) => {
        const t = inferType(tensor(term, id(Unit)))
        // product([dom, Unit]) should simplify to dom
        return portTypeEqual(t.dom, dom) && portTypeEqual(t.cod, cod)
      }),
      { numRuns: 200 },
    )
  })

  test('random well-typed terms always type-check', () => {
    fc.assert(
      fc.property(arbTerm(2), ({ term, dom, cod }) => {
        const t = inferType(term)
        return portTypeEqual(t.dom, dom) && portTypeEqual(t.cod, cod)
      }),
      { numRuns: 500 },
    )
  })
})
