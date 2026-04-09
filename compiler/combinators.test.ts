/**
 * combinators.test.ts — Tests for compile-time combinator expansion.
 */

import { describe, test, expect } from 'bun:test'
import { lowerArrayOps } from './lower_arrays'
import type { ExprNode } from './expr'

// Shorthand for binding reference
const b = (name: string): ExprNode => ({ op: 'binding', name })

describe('binding passthrough', () => {
  test('unresolved binding left as-is', () => {
    const node: ExprNode = { op: 'binding', name: 'x' }
    expect(lowerArrayOps(node)).toEqual(node)
  })
})

describe('let', () => {
  test('substitutes single binding', () => {
    const node: ExprNode = {
      op: 'let',
      bind: { x: 5 },
      in: { op: 'add', args: [b('x'), 1] },
    }
    expect(lowerArrayOps(node)).toEqual({ op: 'add', args: [5, 1] })
  })

  test('sequential let* semantics — later bindings see earlier ones', () => {
    const node: ExprNode = {
      op: 'let',
      bind: {
        a: 2,
        b: { op: 'add', args: [b('a'), 3] },
      },
      in: { op: 'mul', args: [b('a'), b('b')] },
    }
    const result = lowerArrayOps(node)
    // a = 2, b = add(2, 3) = {op: 'add', args: [2, 3]}
    // result = mul(2, add(2, 3))
    expect(result).toEqual({ op: 'mul', args: [2, { op: 'add', args: [2, 3] }] })
  })

  test('nested let', () => {
    const node: ExprNode = {
      op: 'let',
      bind: { x: 10 },
      in: {
        op: 'let',
        bind: { y: { op: 'add', args: [b('x'), 1] } },
        in: b('y'),
      },
    }
    // x=10, then inner let: y = add(10, 1), result = add(10, 1)
    expect(lowerArrayOps(node)).toEqual({ op: 'add', args: [10, 1] })
  })
})

describe('generate', () => {
  test('generate with index variable', () => {
    const node: ExprNode = {
      op: 'generate', count: 4, var: 'i',
      body: b('i'),
    }
    expect(lowerArrayOps(node)).toEqual([0, 1, 2, 3])
  })

  test('generate with expression body', () => {
    const node: ExprNode = {
      op: 'generate', count: 3, var: 'i',
      body: { op: 'mul', args: [b('i'), 2] },
    }
    expect(lowerArrayOps(node)).toEqual([
      { op: 'mul', args: [0, 2] },
      { op: 'mul', args: [1, 2] },
      { op: 'mul', args: [2, 2] },
    ])
  })

  test('repeat (generate with unused var)', () => {
    const node: ExprNode = {
      op: 'generate', count: 3, var: '_',
      body: 5.0,
    }
    expect(lowerArrayOps(node)).toEqual([5.0, 5.0, 5.0])
  })

  test('generate with count 0', () => {
    const node: ExprNode = {
      op: 'generate', count: 0, var: 'i',
      body: b('i'),
    }
    expect(lowerArrayOps(node)).toEqual([])
  })
})

describe('iterate', () => {
  test('iterate produces [init, f(init), f(f(init)), ...]', () => {
    const node: ExprNode = {
      op: 'iterate', count: 4, var: 'x',
      init: 1,
      body: { op: 'mul', args: [b('x'), 2] },
    }
    const result = lowerArrayOps(node) as ExprNode[]
    expect(result).toHaveLength(4)
    expect(result[0]).toBe(1)
    // result[1] = mul(1, 2)
    expect(result[1]).toEqual({ op: 'mul', args: [1, 2] })
    // result[2] = mul(mul(1, 2), 2)
    expect(result[2]).toEqual({ op: 'mul', args: [{ op: 'mul', args: [1, 2] }, 2] })
  })
})

describe('fold', () => {
  test('fold over inline array', () => {
    const node: ExprNode = {
      op: 'fold',
      over: [1, 2, 3],
      init: 0,
      acc_var: 'acc', elem_var: 'x',
      body: { op: 'add', args: [b('acc'), b('x')] },
    }
    const result = lowerArrayOps(node)
    // Unrolled: add(add(add(0, 1), 2), 3)
    expect(result).toEqual({
      op: 'add', args: [
        { op: 'add', args: [
          { op: 'add', args: [0, 1] },
          2,
        ] },
        3,
      ],
    })
  })

  test('fold over empty array returns init', () => {
    const node: ExprNode = {
      op: 'fold',
      over: [],
      init: 42,
      acc_var: 'acc', elem_var: 'x',
      body: { op: 'add', args: [b('acc'), b('x')] },
    }
    expect(lowerArrayOps(node)).toBe(42)
  })

  test('fold over generated array', () => {
    const node: ExprNode = {
      op: 'fold',
      over: { op: 'generate', count: 3, var: 'i', body: b('i') },
      init: 0,
      acc_var: 'sum', elem_var: 'x',
      body: { op: 'add', args: [b('sum'), b('x')] },
    }
    const result = lowerArrayOps(node)
    // generate produces [0, 1, 2], then fold: add(add(add(0, 0), 1), 2)
    expect(result).toEqual({
      op: 'add', args: [
        { op: 'add', args: [
          { op: 'add', args: [0, 0] },
          1,
        ] },
        2,
      ],
    })
  })
})

describe('scan', () => {
  test('scan keeps intermediates', () => {
    const node: ExprNode = {
      op: 'scan',
      over: [1, 2, 3],
      init: 0,
      acc_var: 'acc', elem_var: 'x',
      body: { op: 'add', args: [b('acc'), b('x')] },
    }
    const result = lowerArrayOps(node) as ExprNode[]
    expect(result).toHaveLength(3)
    // [add(0,1), add(add(0,1),2), add(add(add(0,1),2),3)]
    expect(result[0]).toEqual({ op: 'add', args: [0, 1] })
    expect(result[1]).toEqual({ op: 'add', args: [{ op: 'add', args: [0, 1] }, 2] })
    expect(result[2]).toEqual({
      op: 'add', args: [
        { op: 'add', args: [{ op: 'add', args: [0, 1] }, 2] },
        3,
      ],
    })
  })
})

describe('map2', () => {
  test('map2 transforms each element', () => {
    const node: ExprNode = {
      op: 'map2',
      over: [1, 2, 3],
      elem_var: 'x',
      body: { op: 'mul', args: [b('x'), 2] },
    }
    expect(lowerArrayOps(node)).toEqual([
      { op: 'mul', args: [1, 2] },
      { op: 'mul', args: [2, 2] },
      { op: 'mul', args: [3, 2] },
    ])
  })

  test('map2 over generated array', () => {
    const node: ExprNode = {
      op: 'map2',
      over: { op: 'generate', count: 2, var: 'i', body: b('i') },
      elem_var: 'x',
      body: { op: 'add', args: [b('x'), 10] },
    }
    expect(lowerArrayOps(node)).toEqual([
      { op: 'add', args: [0, 10] },
      { op: 'add', args: [1, 10] },
    ])
  })
})

describe('zip_with', () => {
  test('zip_with combines two arrays', () => {
    const node: ExprNode = {
      op: 'zip_with',
      a: [1, 2, 3],
      b: [10, 20, 30],
      x_var: 'x', y_var: 'y',
      body: { op: 'add', args: [b('x'), b('y')] },
    }
    expect(lowerArrayOps(node)).toEqual([
      { op: 'add', args: [1, 10] },
      { op: 'add', args: [2, 20] },
      { op: 'add', args: [3, 30] },
    ])
  })

  test('zip_with truncates to shorter array', () => {
    const node: ExprNode = {
      op: 'zip_with',
      a: [1, 2],
      b: [10, 20, 30],
      x_var: 'x', y_var: 'y',
      body: { op: 'mul', args: [b('x'), b('y')] },
    }
    expect(lowerArrayOps(node)).toEqual([
      { op: 'mul', args: [1, 10] },
      { op: 'mul', args: [2, 20] },
    ])
  })
})

describe('chain', () => {
  test('chain applies body N times', () => {
    const node: ExprNode = {
      op: 'chain', count: 3, var: 'x',
      init: 1,
      body: { op: 'mul', args: [b('x'), 0.5] },
    }
    const result = lowerArrayOps(node)
    // chain: mul(mul(mul(1, 0.5), 0.5), 0.5)
    expect(result).toEqual({
      op: 'mul', args: [
        { op: 'mul', args: [
          { op: 'mul', args: [1, 0.5] },
          0.5,
        ] },
        0.5,
      ],
    })
  })

  test('chain with count 0 returns init', () => {
    const node: ExprNode = {
      op: 'chain', count: 0, var: 'x',
      init: 42,
      body: { op: 'mul', args: [b('x'), 2] },
    }
    expect(lowerArrayOps(node)).toBe(42)
  })

  test('chain with count 1', () => {
    const node: ExprNode = {
      op: 'chain', count: 1, var: 'x',
      init: 10,
      body: { op: 'add', args: [b('x'), 1] },
    }
    expect(lowerArrayOps(node)).toEqual({ op: 'add', args: [10, 1] })
  })
})

describe('combinator composition', () => {
  test('fold over map2 result', () => {
    // sum of squares: fold(map2([1,2,3], x, mul(x,x)), 0, +)
    const node: ExprNode = {
      op: 'fold',
      over: {
        op: 'map2', over: [1, 2, 3], elem_var: 'x',
        body: { op: 'mul', args: [b('x'), b('x')] },
      },
      init: 0,
      acc_var: 'acc', elem_var: 'sq',
      body: { op: 'add', args: [b('acc'), b('sq')] },
    }
    const result = lowerArrayOps(node)
    // map2 produces [mul(1,1), mul(2,2), mul(3,3)]
    // fold produces add(add(add(0, mul(1,1)), mul(2,2)), mul(3,3))
    expect(result).toEqual({
      op: 'add', args: [
        { op: 'add', args: [
          { op: 'add', args: [0, { op: 'mul', args: [1, 1] }] },
          { op: 'mul', args: [2, 2] },
        ] },
        { op: 'mul', args: [3, 3] },
      ],
    })
  })

  test('let with generate', () => {
    const node: ExprNode = {
      op: 'let',
      bind: {
        coeffs: { op: 'generate', count: 3, var: 'i', body: { op: 'mul', args: [b('i'), 0.1] } },
      },
      in: {
        op: 'fold',
        over: b('coeffs'),
        init: 0,
        acc_var: 'sum', elem_var: 'c',
        body: { op: 'add', args: [b('sum'), b('c')] },
      },
    }
    const result = lowerArrayOps(node)
    // coeffs = [mul(0,0.1), mul(1,0.1), mul(2,0.1)]
    // fold: add(add(add(0, mul(0,0.1)), mul(1,0.1)), mul(2,0.1))
    expect(result).toEqual({
      op: 'add', args: [
        { op: 'add', args: [
          { op: 'add', args: [0, { op: 'mul', args: [0, 0.1] }] },
          { op: 'mul', args: [1, 0.1] },
        ] },
        { op: 'mul', args: [2, 0.1] },
      ],
    })
  })
})
