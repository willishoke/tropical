/**
 * walk.test.ts — verify mapChildren handles every category of ExprOpNodeStrict.
 *
 * The internal `assertNever` arm guarantees compile-time exhaustiveness; these
 * runtime tests verify the structural traversal: which fields are children,
 * preservation of reference identity when no child changes, and correctness
 * across each op category.
 */

import { describe, expect, test } from 'bun:test'
import { mapChildren, mapValues, mapArms } from './walk.js'
import type {
  ExprNode,
  ExprOpNodeStrict,
  AddNode, UnaryNode, TernaryNode, VariadicNode,
  ReshapeNode, MatmulNode, MapNode,
  FillNode, ArrayLiteralNode, MatrixNode,
  TagNode, MatchNode, MatchArmStrict, LetNode, FunctionNode, CallNode,
  DelayNode, SourceTagNode,
  GenerateNode, IterateNode, FoldNode, ScanNode,
  Map2Node, ZipWithNode, ChainNode, StrConcatNode, GenerateDeclsNode,
  InstanceDeclNode, RegDeclNode, DelayDeclNode, ProgramDeclNode,
  OutputAssignNode, NextUpdateNode, ProgramBlockNode,
} from './expr.js'

const inc = (n: ExprNode): ExprNode =>
  typeof n === 'number' ? n + 1 : n

const id = (n: ExprNode): ExprNode => n

// ─────────────────────────────────────────────────────────────
// Op<N> family — ~45 ops handled by one branch
// ─────────────────────────────────────────────────────────────

describe('mapChildren — Op<N> family', () => {
  test('binary op recurses into both args', () => {
    const n: AddNode = { op: 'add', args: [1, 2] }
    const result = mapChildren(n, inc) as AddNode
    expect(result.args).toEqual([2, 3])
  })

  test('unary op recurses into single arg', () => {
    const n: UnaryNode = { op: 'neg', args: [5] }
    const result = mapChildren(n, inc) as UnaryNode
    expect(result.args).toEqual([6])
  })

  test('ternary op recurses into all three args', () => {
    const n: TernaryNode = { op: 'select', args: [1, 2, 3] }
    const result = mapChildren(n, inc) as TernaryNode
    expect(result.args).toEqual([2, 3, 4])
  })

  test('variadic op handles arbitrary length', () => {
    const n: VariadicNode = { op: 'array', args: [1, 2, 3, 4, 5] }
    const result = mapChildren(n, inc) as VariadicNode
    expect(result.args).toEqual([2, 3, 4, 5, 6])
  })

  test('reference identity preserved when no children change', () => {
    const n: AddNode = { op: 'add', args: [1, 2] }
    expect(mapChildren(n, id)).toBe(n)
  })

  test('all binary tag variants traverse identically', () => {
    const variants = ['add', 'sub', 'mul', 'lt', 'eq', 'and', 'bit_and', 'lshift'] as const
    for (const op of variants) {
      const n = { op, args: [1, 2] } as AddNode
      const result = mapChildren(n, inc) as AddNode
      expect(result.args).toEqual([2, 3])
    }
  })
})

// ─────────────────────────────────────────────────────────────
// Op<N> + extras — same args traversal, extras pass through
// ─────────────────────────────────────────────────────────────

describe('mapChildren — Op<N> with extras', () => {
  test('reshape preserves shape, recurses into args', () => {
    const n: ReshapeNode = { op: 'reshape', args: [5], shape: [2, 3] }
    const result = mapChildren(n, inc) as ReshapeNode
    expect(result.args).toEqual([6])
    expect(result.shape).toEqual([2, 3])
  })

  test('matmul preserves shape_a, shape_b, element_type', () => {
    const n: MatmulNode = {
      op: 'matmul', args: [1, 2],
      shape_a: [2, 3], shape_b: [3, 4], element_type: 'float',
    }
    const result = mapChildren(n, inc) as MatmulNode
    expect(result.args).toEqual([2, 3])
    expect(result.shape_a).toEqual([2, 3])
    expect(result.shape_b).toEqual([3, 4])
    expect(result.element_type).toBe('float')
  })

  test('map recurses into both callee and args[0]', () => {
    const n: MapNode = {
      op: 'map',
      callee: 10,
      args: [20],
    }
    const result = mapChildren(n, inc) as MapNode
    expect(result.callee).toBe(11)
    expect(result.args).toEqual([21])
  })
})

// ─────────────────────────────────────────────────────────────
// Construction ops
// ─────────────────────────────────────────────────────────────

describe('mapChildren — construction ops', () => {
  test('zeros/ones/matrix have no children — return identity', () => {
    const z: ExprOpNodeStrict = { op: 'zeros', shape: [3] }
    const o: ExprOpNodeStrict = { op: 'ones', shape: [3] }
    const m: ExprOpNodeStrict = { op: 'matrix', rows: [[1, 2], [3, 4]] }
    expect(mapChildren(z, inc)).toBe(z)
    expect(mapChildren(o, inc)).toBe(o)
    expect(mapChildren(m, inc)).toBe(m)
  })

  test('fill recurses into value', () => {
    const n: FillNode = { op: 'fill', shape: [2], value: 7 }
    const result = mapChildren(n, inc) as FillNode
    expect(result.value).toBe(8)
    expect(result.shape).toEqual([2])
  })

  test('array_literal recurses into all values', () => {
    const n: ArrayLiteralNode = { op: 'array_literal', shape: [3], values: [1, 2, 3] }
    const result = mapChildren(n, inc) as ArrayLiteralNode
    expect(result.values).toEqual([2, 3, 4])
  })
})

// ─────────────────────────────────────────────────────────────
// Named-children ops
// ─────────────────────────────────────────────────────────────

describe('mapChildren — named-children ops', () => {
  test('tag with no payload returns identity', () => {
    const n: TagNode = { op: 'tag', type: 'Env', variant: 'Idle' }
    expect(mapChildren(n, inc)).toBe(n)
  })

  test('tag with payload recurses into payload values', () => {
    const n: TagNode = {
      op: 'tag', type: 'Env', variant: 'Decaying',
      payload: { level: 1, target: 5 },
    }
    const result = mapChildren(n, inc) as TagNode
    expect(result.payload).toEqual({ level: 2, target: 6 })
  })

  test('match recurses into scrutinee and each arm body', () => {
    const arm: MatchArmStrict = { bind: 'level', body: 10 }
    const n: MatchNode = {
      op: 'match', type: 'Env',
      scrutinee: 0,
      arms: { Idle: { body: 1 }, Decaying: arm },
    }
    const result = mapChildren(n, inc) as MatchNode
    expect(result.scrutinee).toBe(1)
    expect(result.arms.Idle.body).toBe(2)
    expect(result.arms.Decaying.bind).toBe('level')
    expect(result.arms.Decaying.body).toBe(11)
  })

  test('let recurses into bind values and body', () => {
    const n: LetNode = {
      op: 'let',
      bind: { x: 1, y: 2 },
      in: 10,
    }
    const result = mapChildren(n, inc) as LetNode
    expect(result.bind).toEqual({ x: 2, y: 3 })
    expect(result.in).toBe(11)
  })

  test('function recurses into body', () => {
    const n: FunctionNode = { op: 'function', param_count: 2, body: 5 }
    const result = mapChildren(n, inc) as FunctionNode
    expect(result.body).toBe(6)
    expect(result.param_count).toBe(2)
  })

  test('call recurses into callee and args', () => {
    const n: CallNode = { op: 'call', callee: 1, args: [2, 3] }
    const result = mapChildren(n, inc) as CallNode
    expect(result.callee).toBe(2)
    expect(result.args).toEqual([3, 4])
  })

  test('delay recurses into args[0]', () => {
    const n: DelayNode = { op: 'delay', args: [5], init: 0 }
    const result = mapChildren(n, inc) as DelayNode
    expect(result.args).toEqual([6])
    expect(result.init).toBe(0)
  })

  test('source_tag recurses into gate/expr/on_skip', () => {
    const n: SourceTagNode = {
      op: 'source_tag', source_instance: 'a',
      gate_expr: 1, expr: 2, on_skip: 3,
    }
    const result = mapChildren(n, inc) as SourceTagNode
    expect(result.gate_expr).toBe(2)
    expect(result.expr).toBe(3)
    expect(result.on_skip).toBe(4)
  })
})

// ─────────────────────────────────────────────────────────────
// Combinators
// ─────────────────────────────────────────────────────────────

describe('mapChildren — combinators', () => {
  test('generate recurses into body only (var/count are non-children)', () => {
    const n: GenerateNode = { op: 'generate', count: 3, var: 'i', body: 5 }
    const result = mapChildren(n, inc) as GenerateNode
    expect(result.body).toBe(6)
    expect(result.var).toBe('i')
    expect(result.count).toBe(3)
  })

  test('iterate recurses into init and body', () => {
    const n: IterateNode = { op: 'iterate', count: 3, var: 'x', init: 0, body: 1 }
    const result = mapChildren(n, inc) as IterateNode
    expect(result.init).toBe(1)
    expect(result.body).toBe(2)
  })

  test('fold/scan recurse into over/init/body', () => {
    const fold: FoldNode = {
      op: 'fold', over: 1, init: 2, acc: 'a', elem: 'e', body: 3,
    }
    const fr = mapChildren(fold, inc) as FoldNode
    expect(fr.over).toBe(2)
    expect(fr.init).toBe(3)
    expect(fr.body).toBe(4)

    const scan: ScanNode = {
      op: 'scan', over: 10, init: 20, acc: 'a', elem: 'e', body: 30,
    }
    const sr = mapChildren(scan, inc) as ScanNode
    expect(sr.over).toBe(11)
    expect(sr.init).toBe(21)
    expect(sr.body).toBe(31)
  })

  test('map2/zip_with recurse into their child fields', () => {
    const m2: Map2Node = { op: 'map2', over: 1, elem: 'e', body: 2 }
    const m2r = mapChildren(m2, inc) as Map2Node
    expect(m2r.over).toBe(2)
    expect(m2r.body).toBe(3)

    const zw: ZipWithNode = { op: 'zip_with', a: 1, b: 2, x: 'x', y: 'y', body: 3 }
    const zwr = mapChildren(zw, inc) as ZipWithNode
    expect(zwr.a).toBe(2)
    expect(zwr.b).toBe(3)
    expect(zwr.body).toBe(4)
  })

  test('chain recurses into init and body', () => {
    const n: ChainNode = { op: 'chain', count: 3, var: 'x', init: 0, body: 1 }
    const result = mapChildren(n, inc) as ChainNode
    expect(result.init).toBe(1)
    expect(result.body).toBe(2)
  })

  test('str_concat recurses into all parts', () => {
    const n: StrConcatNode = { op: 'str_concat', parts: [1, 2, 3] }
    const result = mapChildren(n, inc) as StrConcatNode
    expect(result.parts).toEqual([2, 3, 4])
  })

  test('generate_decls recurses into all decls', () => {
    const n: GenerateDeclsNode = {
      op: 'generate_decls', count: 2, var: 'i', decls: [1, 2],
    }
    const result = mapChildren(n, inc) as GenerateDeclsNode
    expect(result.decls).toEqual([2, 3])
  })
})

// ─────────────────────────────────────────────────────────────
// Decl ops
// ─────────────────────────────────────────────────────────────

describe('mapChildren — decl ops', () => {
  test('ref carries no children', () => {
    const n: ExprOpNodeStrict = { op: 'ref', instance: 'a', output: 0 }
    expect(mapChildren(n, inc)).toBe(n)
  })

  test('instance_decl recurses into inputs and gate_input', () => {
    const n: InstanceDeclNode = {
      op: 'instance_decl', name: 'a', program: 'X',
      inputs: { freq: 100, gain: 0.5 },
      gate_input: 1,
    }
    const result = mapChildren(n, inc) as InstanceDeclNode
    expect(result.inputs).toEqual({ freq: 101, gain: 1.5 })
    expect(result.gate_input).toBe(2)
  })

  test('reg_decl recurses into init when present', () => {
    const n: RegDeclNode = { op: 'reg_decl', name: 'r', init: 5 }
    const result = mapChildren(n, inc) as RegDeclNode
    expect(result.init).toBe(6)

    const empty: RegDeclNode = { op: 'reg_decl', name: 'r' }
    expect(mapChildren(empty, inc)).toBe(empty)
  })

  test('delay_decl recurses into update; init is recursed only when ExprNode', () => {
    // numeric init — no recursion
    const numInit: DelayDeclNode = { op: 'delay_decl', name: 'd', init: 0, update: 5 }
    const r1 = mapChildren(numInit, inc) as DelayDeclNode
    expect(r1.init).toBe(0)
    expect(r1.update).toBe(6)

    // tag-init (ExprNode) — recursed
    const tagInit: DelayDeclNode = {
      op: 'delay_decl', name: 'd',
      init: { op: 'tag', type: 'T', variant: 'V' },
      update: 5,
    }
    const r2 = mapChildren(tagInit, inc) as DelayDeclNode
    expect(r2.update).toBe(6)
    // init was an ExprNode and got passed through f (which doesn't change non-numbers)
    expect(r2.init).toEqual({ op: 'tag', type: 'T', variant: 'V' })
  })

  test('output_assign / next_update recurse into expr', () => {
    const oa: OutputAssignNode = { op: 'output_assign', name: 'out', expr: 5 }
    const oar = mapChildren(oa, inc) as OutputAssignNode
    expect(oar.expr).toBe(6)

    const nu: NextUpdateNode = {
      op: 'next_update', target: { kind: 'reg', name: 'r' }, expr: 10,
    }
    const nur = mapChildren(nu, inc) as NextUpdateNode
    expect(nur.expr).toBe(11)
  })

  test('block recurses into decls and assigns', () => {
    const n: ProgramBlockNode = {
      op: 'block', decls: [1, 2], assigns: [10, 20],
    }
    const result = mapChildren(n, inc) as ProgramBlockNode
    expect(result.decls).toEqual([2, 3])
    expect(result.assigns).toEqual([11, 21])
  })

  test('program_decl recurses into program when present', () => {
    const n: ProgramDeclNode = { op: 'program_decl', name: 'P', program: 5 }
    const result = mapChildren(n, inc) as ProgramDeclNode
    expect(result.program).toBe(6)
  })
})

// ─────────────────────────────────────────────────────────────
// Leaves
// ─────────────────────────────────────────────────────────────

describe('mapChildren — leaves', () => {
  test('every leaf op returns identity', () => {
    const leaves: ExprOpNodeStrict[] = [
      { op: 'input', id: 0 },
      { op: 'reg', id: 0 },
      { op: 'delay_ref', id: 'state' },
      { op: 'delay_value', node_id: 0 },
      { op: 'nested_out', ref: 'a', output: 0 },
      { op: 'nested_output', node_id: 0, output_id: 0 },
      { op: 'binding', name: 'x' },
      { op: 'type_param', name: 'N' },
      { op: 'sample_rate' },
      { op: 'sample_index' },
      { op: 'smoothed_param', _ptr: true, _handle: 'h' },
      { op: 'trigger_param', _ptr: true, _handle: 'h' },
      { op: 'const', val: 5 },
    ]
    for (const leaf of leaves) {
      expect(mapChildren(leaf, inc)).toBe(leaf)
    }
  })
})

// ─────────────────────────────────────────────────────────────
// Helper utilities
// ─────────────────────────────────────────────────────────────

describe('helper utilities', () => {
  test('mapValues preserves identity when no value changes', () => {
    const r = { a: 1, b: 2 }
    expect(mapValues(r, id)).toBe(r)
  })

  test('mapValues returns new record when any value changes', () => {
    const r = { a: 1, b: 2 }
    const result = mapValues(r, (v: number) => v + 10)
    expect(result).toEqual({ a: 11, b: 12 })
    expect(result).not.toBe(r)
  })

  test('mapArms returns identity when no arm body changes', () => {
    const a = { Idle: { body: 0 }, Decaying: { bind: 'l', body: 1 } as MatchArmStrict }
    expect(mapArms(a, id)).toBe(a)
  })

  test('mapArms preserves bind when body changes', () => {
    const a = { Decaying: { bind: 'l', body: 5 } as MatchArmStrict }
    const result = mapArms(a, inc)
    expect(result.Decaying.bind).toBe('l')
    expect(result.Decaying.body).toBe(6)
  })
})

// ─────────────────────────────────────────────────────────────
// End-to-end recursion: mapChildren composes with itself for deep walks
// ─────────────────────────────────────────────────────────────

describe('mapChildren composes for deep recursion', () => {
  test('a recursive walker built on mapChildren applies f everywhere', () => {
    const deepInc = (n: ExprNode): ExprNode => {
      if (typeof n === 'number') return n + 1
      if (typeof n !== 'object' || n === null) return n
      if (Array.isArray(n)) return n.map(deepInc)
      return mapChildren(n as ExprOpNodeStrict, deepInc)
    }

    // (1 + 2) * (3 + 4) → (2 + 3) * (4 + 5)
    const tree: ExprNode = {
      op: 'mul',
      args: [
        { op: 'add', args: [1, 2] },
        { op: 'add', args: [3, 4] },
      ],
    }
    const result = deepInc(tree) as ExprOpNodeStrict
    expect(result).toEqual({
      op: 'mul',
      args: [
        { op: 'add', args: [2, 3] },
        { op: 'add', args: [4, 5] },
      ],
    })
  })
})
