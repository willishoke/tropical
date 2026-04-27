/**
 * expressions.test.ts — coverage for the .trop expression parser (Phase B2).
 */

import { describe, test, expect } from 'bun:test'
import { parseExpr, ParseError, type ExprNode } from './expressions.js'

describe('expressions — literals', () => {
  test('integer literal is a bare number', () => {
    expect(parseExpr('42')).toBe(42)
  })

  test('float literal', () => {
    expect(parseExpr('3.14')).toBeCloseTo(3.14)
  })

  test('boolean literals', () => {
    expect(parseExpr('true')).toBe(true)
    expect(parseExpr('false')).toBe(false)
  })

  test('array literal', () => {
    expect(parseExpr('[1, 2, 3]')).toEqual([1, 2, 3])
  })

  test('empty array literal', () => {
    expect(parseExpr('[]')).toEqual([])
  })

  test('nested array literal', () => {
    expect(parseExpr('[[1, 2], [3, 4]]')).toEqual([[1, 2], [3, 4]])
  })

  test('trailing comma allowed in array literal', () => {
    expect(parseExpr('[1, 2, 3,]')).toEqual([1, 2, 3])
  })
})

describe('expressions — bare identifiers emit nameRef placeholder', () => {
  test('single identifier', () => {
    expect(parseExpr('foo')).toEqual({ op: 'nameRef', name: 'foo' })
  })

  test('underscore-prefixed identifier', () => {
    expect(parseExpr('_x')).toEqual({ op: 'nameRef', name: '_x' })
  })
})

describe('expressions — infix arithmetic', () => {
  test('addition', () => {
    expect(parseExpr('a + b')).toEqual({
      op: 'add',
      args: [{ op: 'nameRef', name: 'a' }, { op: 'nameRef', name: 'b' }],
    })
  })

  test('subtraction left-associative', () => {
    // a - b - c → (a - b) - c
    expect(parseExpr('a - b - c')).toEqual({
      op: 'sub',
      args: [
        { op: 'sub', args: [{ op: 'nameRef', name: 'a' }, { op: 'nameRef', name: 'b' }] },
        { op: 'nameRef', name: 'c' },
      ],
    })
  })

  test('multiplication binds tighter than addition', () => {
    // a + b * c → a + (b * c)
    expect(parseExpr('a + b * c')).toEqual({
      op: 'add',
      args: [
        { op: 'nameRef', name: 'a' },
        { op: 'mul', args: [{ op: 'nameRef', name: 'b' }, { op: 'nameRef', name: 'c' }] },
      ],
    })
  })

  test('parenthesized expression overrides precedence', () => {
    expect(parseExpr('(a + b) * c')).toEqual({
      op: 'mul',
      args: [
        { op: 'add', args: [{ op: 'nameRef', name: 'a' }, { op: 'nameRef', name: 'b' }] },
        { op: 'nameRef', name: 'c' },
      ],
    })
  })

  test('all arithmetic ops map to canonical names', () => {
    const cases: Array<[string, string]> = [
      ['a + b', 'add'], ['a - b', 'sub'], ['a * b', 'mul'],
      ['a / b', 'div'], ['a % b', 'mod'],
    ]
    for (const [src, op] of cases) {
      const parsed = parseExpr(src) as { op: string }
      expect(parsed.op).toBe(op)
    }
  })

  test('mod', () => {
    expect(parseExpr('5 % 3')).toEqual({ op: 'mod', args: [5, 3] })
  })
})

describe('expressions — comparison and logical', () => {
  test('comparison op names', () => {
    const cases: Array<[string, string]> = [
      ['a < b',  'lt'], ['a <= b', 'lte'],
      ['a > b',  'gt'], ['a >= b', 'gte'],
      ['a == b', 'eq'], ['a != b', 'neq'],
    ]
    for (const [src, op] of cases) {
      const parsed = parseExpr(src) as { op: string }
      expect(parsed.op).toBe(op)
    }
  })

  test('logical and / or use canonical op names', () => {
    expect((parseExpr('a && b') as { op: string }).op).toBe('and')
    expect((parseExpr('a || b') as { op: string }).op).toBe('or')
  })

  test('comparison binds tighter than logical', () => {
    // a < b && c < d → (a<b) && (c<d)
    const parsed = parseExpr('a < b && c < d') as { op: string; args: ExprNode[] }
    expect(parsed.op).toBe('and')
    expect((parsed.args[0] as { op: string }).op).toBe('lt')
    expect((parsed.args[1] as { op: string }).op).toBe('lt')
  })

  test('|| is lower precedence than &&', () => {
    // a || b && c → a || (b && c)
    const parsed = parseExpr('a || b && c') as { op: string; args: ExprNode[] }
    expect(parsed.op).toBe('or')
    expect((parsed.args[1] as { op: string }).op).toBe('and')
  })
})

describe('expressions — bitwise', () => {
  test('bit ops', () => {
    expect((parseExpr('a & b')  as { op: string }).op).toBe('bitAnd')
    expect((parseExpr('a | b')  as { op: string }).op).toBe('bitOr')
    expect((parseExpr('a ^ b')  as { op: string }).op).toBe('bitXor')
    expect((parseExpr('a << b') as { op: string }).op).toBe('lshift')
    expect((parseExpr('a >> b') as { op: string }).op).toBe('rshift')
  })

  test('additive binds tighter than shifts (C precedence)', () => {
    // a + b << c → (a + b) << c
    const parsed = parseExpr('a + b << c') as { op: string; args: ExprNode[] }
    expect(parsed.op).toBe('lshift')
    expect((parsed.args[0] as { op: string }).op).toBe('add')
  })

  test('equality binds tighter than bitwise & (C precedence)', () => {
    // a & b == c → a & (b == c)
    const parsed = parseExpr('a & b == c') as { op: string; args: ExprNode[] }
    expect(parsed.op).toBe('bitAnd')
    expect((parsed.args[1] as { op: string }).op).toBe('eq')
  })
})

describe('expressions — unary', () => {
  test('unary minus', () => {
    expect(parseExpr('-x')).toEqual({ op: 'neg', args: [{ op: 'nameRef', name: 'x' }] })
  })

  test('logical not', () => {
    expect(parseExpr('!flag')).toEqual({ op: 'not', args: [{ op: 'nameRef', name: 'flag' }] })
  })

  test('bitwise not', () => {
    expect(parseExpr('~bits')).toEqual({ op: 'bitNot', args: [{ op: 'nameRef', name: 'bits' }] })
  })

  test('unary binds tighter than binary', () => {
    // -a * b → (-a) * b
    expect(parseExpr('-a * b')).toEqual({
      op: 'mul',
      args: [
        { op: 'neg', args: [{ op: 'nameRef', name: 'a' }] },
        { op: 'nameRef', name: 'b' },
      ],
    })
  })

  test('chained unaries', () => {
    expect(parseExpr('--x')).toEqual({
      op: 'neg',
      args: [{ op: 'neg', args: [{ op: 'nameRef', name: 'x' }] }],
    })
  })
})

describe('expressions — postfix: dot, index, call', () => {
  test('dotted port reference emits nestedOut', () => {
    expect(parseExpr('osc.sin')).toEqual({ op: 'nestedOut', ref: 'osc', output: 'sin' })
  })

  test('indexing emits index op', () => {
    expect(parseExpr('a[i]')).toEqual({
      op: 'index',
      args: [{ op: 'nameRef', name: 'a' }, { op: 'nameRef', name: 'i' }],
    })
  })

  test('numeric index', () => {
    expect(parseExpr('a[3]')).toEqual({
      op: 'index',
      args: [{ op: 'nameRef', name: 'a' }, 3],
    })
  })

  test('generic function call emits call(nameRef, args) for elaborator', () => {
    expect(parseExpr('sqrt(x)')).toEqual({
      op: 'call',
      callee: { op: 'nameRef', name: 'sqrt' },
      args: [{ op: 'nameRef', name: 'x' }],
    })
  })

  test('multi-arg call', () => {
    expect(parseExpr('clamp(x, 0, 1)')).toEqual({
      op: 'call',
      callee: { op: 'nameRef', name: 'clamp' },
      args: [{ op: 'nameRef', name: 'x' }, 0, 1],
    })
  })

  test('chained dots and indexes', () => {
    // osc.out[0]
    expect(parseExpr('osc.out[0]')).toEqual({
      op: 'index',
      args: [{ op: 'nestedOut', ref: 'osc', output: 'out' }, 0],
    })
  })

  test('dot access on a non-identifier LHS is rejected', () => {
    // `(a + b).field` — the dotted form is reserved for `instance.port`.
    // Allowing it on arbitrary expressions would emit an IR shape no
    // downstream stage consumes; the parser refuses at parse time.
    expect(() => parseExpr('(a + b).field')).toThrow(/dot access requires an identifier/)
    expect(() => parseExpr('arr[0].field')).toThrow(/dot access requires an identifier/)
  })
})

describe('expressions — let binding', () => {
  test('single let binding', () => {
    expect(parseExpr('let { x: 1 } in x + x')).toEqual({
      op: 'let',
      bind: { x: 1 },
      in: {
        op: 'add',
        args: [{ op: 'binding', name: 'x' }, { op: 'binding', name: 'x' }],
      },
    })
  })

  test('multiple let bindings with semicolon separator', () => {
    expect(parseExpr('let { x: 1; y: 2 } in x + y')).toEqual({
      op: 'let',
      bind: { x: 1, y: 2 },
      in: {
        op: 'add',
        args: [{ op: 'binding', name: 'x' }, { op: 'binding', name: 'y' }],
      },
    })
  })

  test('let bindings with comma separator', () => {
    expect(parseExpr('let { x: 1, y: 2 } in x + y')).toEqual({
      op: 'let',
      bind: { x: 1, y: 2 },
      in: {
        op: 'add',
        args: [{ op: 'binding', name: 'x' }, { op: 'binding', name: 'y' }],
      },
    })
  })

  test('duplicate let binding name rejected', () => {
    expect(() => parseExpr('let { x: 1; x: 2 } in x')).toThrow(ParseError)
  })

  test('let binding scope ends after `in body`', () => {
    // After `in body`, x is not in scope; if it were used after the let
    // (not possible in expression position, but inside outer let body):
    // outer x stays unrelated. Cover with a nested let.
    const parsed = parseExpr('let { x: 1 } in let { y: x } in y') as {
      op: string; bind: Record<string, ExprNode>; in: { op: string; bind: Record<string, ExprNode>; in: ExprNode }
    }
    expect(parsed.op).toBe('let')
    // Outer x is bound when parsing inner let's `y: x` — should be a binding ref
    expect(parsed.in.bind.y).toEqual({ op: 'binding', name: 'x' })
    // Inner body refers to inner y
    expect(parsed.in.in).toEqual({ op: 'binding', name: 'y' })
  })
})

describe('expressions — combinators', () => {
  test('fold with two binders', () => {
    expect(parseExpr('fold(arr, 0, (acc, e) => acc + e)')).toEqual({
      op: 'fold',
      over: { op: 'nameRef', name: 'arr' },
      init: 0,
      acc_var: 'acc',
      elem_var: 'e',
      body: {
        op: 'add',
        args: [{ op: 'binding', name: 'acc' }, { op: 'binding', name: 'e' }],
      },
    })
  })

  test('scan has same shape as fold', () => {
    const parsed = parseExpr('scan(a, 0, (a, e) => a + e)') as { op: string }
    expect(parsed.op).toBe('scan')
  })

  test('generate with one binder', () => {
    expect(parseExpr('generate(8, (i) => i * i)')).toEqual({
      op: 'generate',
      count: 8,
      var: 'i',
      body: {
        op: 'mul',
        args: [{ op: 'binding', name: 'i' }, { op: 'binding', name: 'i' }],
      },
    })
  })

  test('iterate with init', () => {
    expect(parseExpr('iterate(4, 1, (x) => x * 2)')).toEqual({
      op: 'iterate',
      count: 4,
      var: 'x',
      init: 1,
      body: {
        op: 'mul',
        args: [{ op: 'binding', name: 'x' }, 2],
      },
    })
  })

  test('chain', () => {
    expect(parseExpr('chain(3, 0, (x) => x + 1)')).toEqual({
      op: 'chain',
      count: 3,
      var: 'x',
      init: 0,
      body: {
        op: 'add',
        args: [{ op: 'binding', name: 'x' }, 1],
      },
    })
  })

  test('map2', () => {
    expect(parseExpr('map2(arr, (e) => e * 2)')).toEqual({
      op: 'map2',
      over: { op: 'nameRef', name: 'arr' },
      elem_var: 'e',
      body: {
        op: 'mul',
        args: [{ op: 'binding', name: 'e' }, 2],
      },
    })
  })

  test('zipWith with two binders', () => {
    expect(parseExpr('zipWith(a, b, (x, y) => x + y)')).toEqual({
      op: 'zipWith',
      a: { op: 'nameRef', name: 'a' },
      b: { op: 'nameRef', name: 'b' },
      x_var: 'x',
      y_var: 'y',
      body: {
        op: 'add',
        args: [{ op: 'binding', name: 'x' }, { op: 'binding', name: 'y' }],
      },
    })
  })

  test('combinator binders shadow outer scope', () => {
    // Outer binder x via let; inner generate's i binder; verify both work.
    const parsed = parseExpr('let { x: 5 } in generate(3, (i) => x + i)') as {
      in: { body: { args: ExprNode[] } }
    }
    const body = parsed.in.body
    expect(body.args[0]).toEqual({ op: 'binding', name: 'x' })
    expect(body.args[1]).toEqual({ op: 'binding', name: 'i' })
  })

  test('combinator binders do not leak outside body', () => {
    // After the fold, `acc` should be unbound (a nameRef again).
    // Wrap with `let { y: <fold> } in y + acc` to test post-fold scope.
    const parsed = parseExpr('let { y: fold(a, 0, (acc, e) => acc + e) } in y + acc') as {
      in: { args: ExprNode[] }
    }
    expect(parsed.in.args[1]).toEqual({ op: 'nameRef', name: 'acc' })
  })

  test('shadowing is structurally invisible at parser level', () => {
    // `let { x: 1 } in let { x: 2 } in x` — the parser does not
    // disambiguate the two `x` binders. Both inner and outer x produce
    // {op:'binding', name:'x'} indistinguishably; the elaborator (or
    // any consumer that cares) is responsible for resolving by depth.
    // This test pins the documented behavior so a future change that
    // introduces de-Bruijn indices is intentional, not silent.
    const parsed = parseExpr('let { x: 1 } in let { x: 2 } in x') as {
      in: { in: ExprNode }
    }
    expect(parsed.in.in).toEqual({ op: 'binding', name: 'x' })
  })

  test('combinator binder reusing outer name', () => {
    // `let { e: 9 } in fold(arr, 0, (e, x) => e + x)` — fold's `e`
    // binder reuses the outer let's name. Inside the body, `e` refers
    // to the fold binder, not the let. Outside the fold, `e` is the
    // let binder. (Same shadowing-invisibility caveat as above.)
    const parsed = parseExpr(
      'let { e: 9 } in fold(arr, 0, (e, x) => e + x)',
    ) as { in: { body: { args: ExprNode[] } } }
    // Inside the fold body, `e` is a binding (refers to whichever
    // enclosing binder is named e — which is the fold's, in nesting
    // order, but the parser doesn't encode that).
    expect(parsed.in.body.args[0]).toEqual({ op: 'binding', name: 'e' })
  })

  test('match arm binder shadowing a let', () => {
    const parsed = parseExpr(`
      let { x: 1 } in
      match v {
        Some { value: x } => x,
        None => x
      }
    `) as { in: { arms: Record<string, { body: ExprNode }> } }
    expect(parsed.in.arms.Some.body).toEqual({ op: 'binding', name: 'x' })
    // Outer x is still in scope inside the None arm, so it's a binding too.
    expect(parsed.in.arms.None.body).toEqual({ op: 'binding', name: 'x' })
  })

  test('lambda arity mismatch rejected', () => {
    expect(() => parseExpr('fold(a, 0, (only) => only)')).toThrow(/lambda expects 2/)
  })
})

describe('expressions — error cases', () => {
  test('unbalanced parens', () => {
    expect(() => parseExpr('(a + b')).toThrow(ParseError)
  })

  test('unbalanced array literal', () => {
    expect(() => parseExpr('[1, 2')).toThrow(ParseError)
  })

  test('trailing input rejected', () => {
    expect(() => parseExpr('a + b extra')).toThrow(/unexpected trailing/)
  })

  test('empty input rejected', () => {
    expect(() => parseExpr('')).toThrow(ParseError)
  })

  test('binary op missing rhs', () => {
    expect(() => parseExpr('a +')).toThrow(ParseError)
  })

  test('error positions reflect token line/col', () => {
    let err: ParseError | undefined
    // Parse-time error: missing rhs of `+` at line 2, col 3
    try { parseExpr('a +\n  )') } catch (e) { err = e as ParseError }
    expect(err).toBeInstanceOf(ParseError)
    expect(err?.message).toMatch(/2:3/)
  })
})

describe('expressions — realistic samples', () => {
  test('audio expression: x.in = 3 * y.out + 1 (rhs only)', () => {
    expect(parseExpr('3 * y.out + 1')).toEqual({
      op: 'add',
      args: [
        {
          op: 'mul',
          args: [3, { op: 'nestedOut', ref: 'y', output: 'out' }],
        },
        1,
      ],
    })
  })

  test('clamp + mul', () => {
    // -1 constant-folds to a bare negative literal; matches stdlib JSON.
    expect(parseExpr('clamp(x * gain, -1, 1)')).toEqual({
      op: 'call',
      callee: { op: 'nameRef', name: 'clamp' },
      args: [
        { op: 'mul', args: [{ op: 'nameRef', name: 'x' }, { op: 'nameRef', name: 'gain' }] },
        -1,
        1,
      ],
    })
  })

  test('Sin polynomial fold pattern', () => {
    const src = 'fold([1, -0.16667, 0.00833], 0, (acc, c) => c + x2 * acc)'
    const parsed = parseExpr(src) as { op: string; over: ExprNode[]; init: number; body: { op: string } }
    expect(parsed.op).toBe('fold')
    expect(parsed.over).toEqual([1, -0.16667, 0.00833])
    expect(parsed.init).toBe(0)
    expect(parsed.body.op).toBe('add')
  })
})
