/**
 * lower_arrays.test.ts — Tests for array operation lowering.
 */

import { describe, test, expect } from 'bun:test'
import { lowerArrayOps, expandDeclGenerators } from './lower_arrays'
import type { ExprNode } from './expr'

describe('lowerArrayOps', () => {
  test('passes through scalars', () => {
    expect(lowerArrayOps(42)).toBe(42)
    expect(lowerArrayOps(true)).toBe(true)
  })

  test('passes through non-array ops', () => {
    const node: ExprNode = { op: 'add', args: [1, 2] }
    expect(lowerArrayOps(node)).toEqual({ op: 'add', args: [1, 2] })
  })

  test('lowers zeros to array of 0s', () => {
    const node: ExprNode = { op: 'zeros', shape: [4] }
    expect(lowerArrayOps(node)).toEqual([0, 0, 0, 0])
  })

  test('lowers ones to array of 1s', () => {
    const node: ExprNode = { op: 'ones', shape: [2, 2] }
    expect(lowerArrayOps(node)).toEqual([1, 1, 1, 1])
  })

  test('lowers fill to repeated value', () => {
    const node: ExprNode = { op: 'fill', shape: [3], value: 0.5 }
    expect(lowerArrayOps(node)).toEqual([0.5, 0.5, 0.5])
  })

  test('lowers array_literal to inline array', () => {
    const node: ExprNode = { op: 'array_literal', shape: [2, 2], values: [1, 2, 3, 4] }
    expect(lowerArrayOps(node)).toEqual([1, 2, 3, 4])
  })

  test('lowers reshape to identity (flat data unchanged)', () => {
    const node: ExprNode = { op: 'reshape', args: [[1, 2, 3, 4, 5, 6]], shape: [3, 2] }
    expect(lowerArrayOps(node)).toEqual([1, 2, 3, 4, 5, 6])
  })

  test('lowers slice to Index ops', () => {
    const node: ExprNode = { op: 'slice', args: [[10, 20, 30, 40, 50]], axis: 0, start: 1, end: 4 }
    const result = lowerArrayOps(node) as ExprNode[]
    expect(Array.isArray(result)).toBe(true)
    expect(result.length).toBe(3)
    // Each element should be an index op
    for (let i = 0; i < 3; i++) {
      const r = result[i] as Record<string, unknown>
      expect(r.op).toBe('index')
    }
  })

  test('lowers reduce on inline array to tree reduction', () => {
    const node: ExprNode = { op: 'reduce', args: [[1, 2, 3, 4]], axis: 0, reduce_op: 'add' }
    const result = lowerArrayOps(node)
    // Should be a tree of add ops, not an array
    expect(Array.isArray(result)).toBe(false)
    const r = result as Record<string, unknown>
    expect(r.op).toBe('add')
  })

  test('lowers reduce of single element to the element', () => {
    const node: ExprNode = { op: 'reduce', args: [[42]], axis: 0, reduce_op: 'add' }
    expect(lowerArrayOps(node)).toBe(42)
  })

  test('lowers broadcast_to scalar', () => {
    const node: ExprNode = { op: 'broadcast_to', args: [5], shape: [4] }
    expect(lowerArrayOps(node)).toEqual([5, 5, 5, 5])
  })

  test('lowers broadcast_to [1] to [N]', () => {
    const node: ExprNode = { op: 'broadcast_to', args: [[7]], shape: [3] }
    expect(lowerArrayOps(node)).toEqual([7, 7, 7])
  })

  test('lowers map on inline array', () => {
    const callee: ExprNode = {
      op: 'function', param_count: 1,
      body: { op: 'mul', args: [{ op: 'input', id: 0 }, 2] },
    }
    const node: ExprNode = { op: 'map', callee, args: [[1, 2, 3]] }
    const result = lowerArrayOps(node) as ExprNode[]
    expect(Array.isArray(result)).toBe(true)
    expect(result.length).toBe(3)
    // Each element should be a call
    for (const r of result) {
      expect((r as Record<string, unknown>).op).toBe('call')
    }
  })

  test('recursively lowers nested array ops', () => {
    // add(zeros([2]), ones([2])) should lower zeros and ones inside
    const node: ExprNode = {
      op: 'add',
      args: [
        { op: 'zeros', shape: [2] },
        { op: 'ones', shape: [2] },
      ],
    }
    const result = lowerArrayOps(node) as Record<string, unknown>
    expect(result.op).toBe('add')
    const args = result.args as ExprNode[]
    expect(args[0]).toEqual([0, 0])
    expect(args[1]).toEqual([1, 1])
  })
})

describe('lowerMatmul', () => {
  test('2x2 @ 2x2 produces 4 elements', () => {
    const node: ExprNode = {
      op: 'matmul',
      args: [[1, 2, 3, 4], [5, 6, 7, 8]],
      shape_a: [2, 2], shape_b: [2, 2],
    }
    const result = lowerArrayOps(node) as ExprNode[]
    expect(Array.isArray(result)).toBe(true)
    expect(result.length).toBe(4)
  })

  test('1x3 @ 3x1 produces a single-element array (dot product)', () => {
    const node: ExprNode = {
      op: 'matmul',
      args: [[1, 2, 3], [4, 5, 6]],
      shape_a: [1, 3], shape_b: [3, 1],
    }
    const result = lowerArrayOps(node) as ExprNode[]
    expect(Array.isArray(result)).toBe(true)
    expect(result.length).toBe(1)
  })

  test('2x3 @ 3x2 produces 4 elements', () => {
    const node: ExprNode = {
      op: 'matmul',
      args: [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
      shape_a: [2, 3], shape_b: [3, 2],
    }
    const result = lowerArrayOps(node) as ExprNode[]
    expect(Array.isArray(result)).toBe(true)
    expect(result.length).toBe(4)
  })

  test('no matmul nodes remain after lowering', () => {
    const node: ExprNode = {
      op: 'matmul',
      args: [[1, 2, 3, 4], [5, 6, 7, 8]],
      shape_a: [2, 2], shape_b: [2, 2],
    }
    const result = lowerArrayOps(node)
    expect(JSON.stringify(result)).not.toContain('"matmul"')
  })

  test('output elements are scalar op trees (index/mul/add, no matmul)', () => {
    const node: ExprNode = {
      op: 'matmul',
      args: [[1, 2, 3, 4], [5, 6, 7, 8]],
      shape_a: [2, 2], shape_b: [2, 2],
    }
    const result = lowerArrayOps(node)
    const json = JSON.stringify(result)
    expect(json).not.toContain('"matmul"')
    expect(json).toContain('"mul"')
    expect(json).toContain('"add"')
    expect(json).toContain('"index"') // index into inline arrays, folded by optimizer
  })

  test('identity @ vector leaves vector elements structurally correct', () => {
    // I2 = [[1,0],[0,1]], v = [a, b] — result should be [a, b]
    // With symbolic inputs, the index ops stay; check shape only
    const ref: ExprNode = { op: 'ref', instance: 'M', output: 'v' }
    const I2 = [1, 0, 0, 1]
    const node: ExprNode = {
      op: 'matmul',
      args: [I2, ref],
      shape_a: [2, 2], shape_b: [2, 1],
    }
    const result = lowerArrayOps(node) as ExprNode[]
    expect(Array.isArray(result)).toBe(true)
    expect(result.length).toBe(2) // 2x1 output
  })

  test('bool element_type uses logical and/or instead of mul/add', () => {
    const node: ExprNode = {
      op: 'matmul',
      args: [[1, 0, 0, 1], [1, 1, 0, 1]],
      shape_a: [2, 2], shape_b: [2, 2],
      element_type: 'bool',
    }
    const result = lowerArrayOps(node)
    const json = JSON.stringify(result)
    expect(json).toContain('"and"')
    expect(json).toContain('"or"')
    expect(json).not.toContain('"mul"')
    expect(json).not.toContain('"add"')
  })

  test('int element_type uses mul/add', () => {
    const node: ExprNode = {
      op: 'matmul',
      args: [[1, 2, 3, 4], [5, 6, 7, 8]],
      shape_a: [2, 2], shape_b: [2, 2],
      element_type: 'int',
    }
    const result = lowerArrayOps(node)
    const json = JSON.stringify(result)
    expect(json).toContain('"mul"')
    expect(json).toContain('"add"')
    expect(json).not.toContain('"and"')
  })
})

describe('expandDeclGenerators', () => {
  test('passes through block with no generate_decls', () => {
    const block = {
      op: 'block' as const,
      decls: [{ op: 'instance_decl', name: 'Osc', program: 'SinOsc', inputs: { freq: 440 } } as ExprNode],
      assigns: [],
    }
    expect(expandDeclGenerators(block)).toBe(block)
  })

  test('expands generate_decls with string name prefix and binding var', () => {
    const block = {
      op: 'block' as const,
      decls: [{
        op: 'generate_decls',
        count: 3,
        var: 'i',
        decls: [{
          op: 'instance_decl',
          name: { op: 'str_concat', parts: ['Osc', { op: 'binding', name: 'i' }] },
          program: 'SinOsc',
          inputs: { freq: { op: 'mul', args: [{ op: 'binding', name: 'i' }, 100] } },
        }],
      } as ExprNode],
      assigns: [],
    }
    const result = expandDeclGenerators(block)
    expect(result.decls).toHaveLength(3)
    const decls = result.decls as Record<string, unknown>[]
    expect(decls[0].name).toBe('Osc0')
    expect(decls[1].name).toBe('Osc1')
    expect(decls[2].name).toBe('Osc2')
    expect((decls[0].inputs as Record<string, unknown>).freq).toEqual({ op: 'mul', args: [0, 100] })
    expect((decls[1].inputs as Record<string, unknown>).freq).toEqual({ op: 'mul', args: [1, 100] })
    expect((decls[2].inputs as Record<string, unknown>).freq).toEqual({ op: 'mul', args: [2, 100] })
  })

  test('1-indexed names via add in str_concat', () => {
    const block = {
      op: 'block' as const,
      decls: [{
        op: 'generate_decls',
        count: 3,
        var: 'i',
        decls: [{
          op: 'instance_decl',
          name: { op: 'str_concat', parts: ['VCO', { op: 'add', args: [{ op: 'binding', name: 'i' }, 1] }] },
          program: 'SinOsc',
          inputs: {},
        }],
      } as ExprNode],
      assigns: [],
    }
    const result = expandDeclGenerators(block)
    const decls = result.decls as Record<string, unknown>[]
    expect(decls.map(d => d.name)).toEqual(['VCO1', 'VCO2', 'VCO3'])
  })

  test('count 0 produces no decls', () => {
    const block = {
      op: 'block' as const,
      decls: [{
        op: 'generate_decls',
        count: 0,
        var: 'i',
        decls: [{ op: 'instance_decl', name: 'X', program: 'SinOsc', inputs: {} }],
      } as ExprNode],
      assigns: [],
    }
    const result = expandDeclGenerators(block)
    expect(result.decls).toHaveLength(0)
  })

  test('multiple template decls per generate_decls iteration', () => {
    const block = {
      op: 'block' as const,
      decls: [{
        op: 'generate_decls',
        count: 2,
        var: 'i',
        decls: [
          {
            op: 'instance_decl',
            name: { op: 'str_concat', parts: ['Osc', { op: 'binding', name: 'i' }] },
            program: 'SinOsc',
            inputs: {},
          },
          {
            op: 'instance_decl',
            name: { op: 'str_concat', parts: ['Env', { op: 'binding', name: 'i' }] },
            program: 'EnvExpDecay',
            inputs: {},
          },
        ],
      } as ExprNode],
      assigns: [],
    }
    const result = expandDeclGenerators(block)
    const names = (result.decls as Record<string, unknown>[]).map(d => d.name)
    expect(names).toEqual(['Osc0', 'Env0', 'Osc1', 'Env1'])
  })

  test('preserves non-generate_decls entries alongside expanded ones', () => {
    const existing = { op: 'instance_decl', name: 'Static', program: 'Clock', inputs: {} } as ExprNode
    const block = {
      op: 'block' as const,
      decls: [
        existing,
        {
          op: 'generate_decls',
          count: 2,
          var: 'i',
          decls: [{ op: 'instance_decl', name: { op: 'str_concat', parts: ['G', { op: 'binding', name: 'i' }] }, program: 'SinOsc', inputs: {} }],
        } as ExprNode,
      ],
      assigns: [],
    }
    const result = expandDeclGenerators(block)
    expect(result.decls).toHaveLength(3)
    expect((result.decls as Record<string, unknown>[])[0]).toBe(existing)
    expect((result.decls as Record<string, unknown>[])[1].name).toBe('G0')
    expect((result.decls as Record<string, unknown>[])[2].name).toBe('G1')
  })

  test('throws on unevaluable name expression', () => {
    const block = {
      op: 'block' as const,
      decls: [{
        op: 'generate_decls',
        count: 1,
        var: 'i',
        decls: [{
          op: 'instance_decl',
          name: { op: 'sin', args: [{ op: 'binding', name: 'i' }] },
          program: 'SinOsc',
          inputs: {},
        }],
      } as ExprNode],
      assigns: [],
    }
    expect(() => expandDeclGenerators(block)).toThrow(/name did not resolve to a string/)
  })

  // ── Regression tests for issue #100 ─────────────────────────────────

  test('rejects name collision between explicit decl and generator output', () => {
    // Bob's reproduction: an explicit Osc0 followed by a generator that also
    // produces Osc0 silently clobbered the explicit one. Must throw.
    const block = {
      op: 'block' as const,
      decls: [
        { op: 'instance_decl', name: 'Osc0', program: 'SinOsc', inputs: { x: 440 } } as ExprNode,
        {
          op: 'generate_decls',
          count: 3,
          var: 'i',
          decls: [{
            op: 'instance_decl',
            name: { op: 'str_concat', parts: ['Osc', { op: 'binding', name: 'i' }] },
            program: 'SinOsc',
            inputs: { x: { op: 'mul', args: [{ op: 'binding', name: 'i' }, 100] } },
          }],
        } as ExprNode,
      ],
      assigns: [],
    }
    expect(() => expandDeclGenerators(block)).toThrow(/duplicate decl name 'Osc0'/)
  })

  test('rejects name collision between two templates in one iteration', () => {
    // Two templates that both resolve to the same name on the same iteration.
    // Without dedup, the second template silently overwrites the first.
    const block = {
      op: 'block' as const,
      decls: [{
        op: 'generate_decls',
        count: 2,
        var: 'i',
        decls: [
          {
            op: 'instance_decl',
            name: { op: 'str_concat', parts: ['X', { op: 'binding', name: 'i' }] },
            program: 'SinOsc',
            inputs: {},
          },
          {
            op: 'instance_decl',
            name: { op: 'str_concat', parts: ['X', { op: 'binding', name: 'i' }] },
            program: 'Clock',
            inputs: {},
          },
        ],
      } as ExprNode],
      assigns: [],
    }
    expect(() => expandDeclGenerators(block)).toThrow(/duplicate decl name 'X0'/)
  })

  test('inner generate shielded from outer generate_decls substitution', () => {
    // Outer generate_decls(var='i') with a template containing an expression-
    // level generate(var='i'). The inner body uses {binding i} which must
    // refer to the INNER var, not the outer's concrete value.
    //
    // Without scope-aware substitution: outer i=0 substitutes into inner body
    // before the inner runs, yielding freq=[1,1,1] instead of freq=[1,2,3].
    const block = {
      op: 'block' as const,
      decls: [{
        op: 'generate_decls',
        count: 1,
        var: 'i',
        decls: [{
          op: 'instance_decl',
          name: 'v0',
          program: 'SinOsc',
          inputs: {
            freq: {
              op: 'generate',
              count: 3,
              var: 'i',
              body: { op: 'add', args: [{ op: 'binding', name: 'i' }, 1] },
            },
          },
        }],
      } as ExprNode],
      assigns: [],
    }
    const result = expandDeclGenerators(block)
    const v0 = (result.decls![0] as Record<string, unknown>)
    const inputs = v0.inputs as Record<string, ExprNode>
    // The inner generate should still be intact, with its bindings untouched.
    const freqGen = inputs.freq as Record<string, unknown>
    expect(freqGen.op).toBe('generate')
    expect(freqGen.var).toBe('i')
    // The body binding ref must still reference 'i' (the inner's var).
    const body = freqGen.body as Record<string, unknown>
    const bindingRef = (body.args as ExprNode[])[0] as Record<string, unknown>
    expect(bindingRef.op).toBe('binding')
    expect(bindingRef.name).toBe('i')
  })

  test('inner let with same var name shielded from outer substitution', () => {
    // A template containing `let { i: 99 } in (binding i)` should emit the
    // same let intact — outer i=0 must NOT substitute into `in`.
    const block = {
      op: 'block' as const,
      decls: [{
        op: 'generate_decls',
        count: 1,
        var: 'i',
        decls: [{
          op: 'instance_decl',
          name: 'v0',
          program: 'SinOsc',
          inputs: {
            freq: {
              op: 'let',
              bind: { i: 99 },
              in: { op: 'binding', name: 'i' },
            },
          },
        }],
      } as ExprNode],
      assigns: [],
    }
    const result = expandDeclGenerators(block)
    const v0 = (result.decls![0] as Record<string, unknown>)
    const inputs = v0.inputs as Record<string, ExprNode>
    const letNode = inputs.freq as Record<string, unknown>
    expect(letNode.op).toBe('let')
    // The `in` body should still be a raw binding ref to 'i' — if outer
    // substitution leaked in, it would be `0` here.
    const inBody = letNode.in as Record<string, unknown>
    expect(inBody.op).toBe('binding')
    expect(inBody.name).toBe('i')
  })

  test('rejects residual binding node from typo', () => {
    // Generator binds 'i' but the template refers to 'j' by mistake. This
    // was previously slipping through substitution + validateExpr and
    // surfacing as a late crash at flatten/emit with no source context.
    const block = {
      op: 'block' as const,
      decls: [{
        op: 'generate_decls',
        count: 1,
        var: 'i',
        decls: [{
          op: 'instance_decl',
          name: 'v0',
          program: 'SinOsc',
          inputs: {
            freq: { op: 'mul', args: [{ op: 'binding', name: 'j' }, 100] },
          },
        }],
      } as ExprNode],
      assigns: [],
    }
    expect(() => expandDeclGenerators(block)).toThrow(/unresolved binding 'j'/)
  })

  test('nested generate_decls expands recursively (trees × coconuts)', () => {
    // Outer count=2, inner count=3 → 6 total instances with names
    // tree0_coco0, tree0_coco1, tree0_coco2, tree1_coco0, ...
    const block = {
      op: 'block' as const,
      decls: [{
        op: 'generate_decls',
        count: 2,
        var: 't',
        decls: [{
          op: 'generate_decls',
          count: 3,
          var: 'c',
          decls: [{
            op: 'instance_decl',
            name: { op: 'str_concat', parts: [
              'tree', { op: 'binding', name: 't' },
              '_coco', { op: 'binding', name: 'c' },
            ]},
            program: 'Coconut',
            inputs: {},
          }],
        }],
      } as ExprNode],
      assigns: [],
    }
    const result = expandDeclGenerators(block)
    const names = result.decls!.map(d => (d as Record<string, unknown>).name)
    expect(names).toEqual([
      'tree0_coco0', 'tree0_coco1', 'tree0_coco2',
      'tree1_coco0', 'tree1_coco1', 'tree1_coco2',
    ])
  })

  test('nested generate_decls: inner binding name shielded from outer var', () => {
    // Outer var='i', inner var='i' (same name). Inner body uses {binding i}.
    // Must refer to INNER's i, not outer's concrete value.
    const block = {
      op: 'block' as const,
      decls: [{
        op: 'generate_decls',
        count: 2,
        var: 'i',
        decls: [{
          op: 'generate_decls',
          count: 3,
          var: 'i',  // deliberately shadows outer
          decls: [{
            op: 'instance_decl',
            name: { op: 'str_concat', parts: [
              'v', { op: 'binding', name: 'i' },
            ]},
            program: 'SinOsc',
            inputs: {},
          }],
        }],
      } as ExprNode],
      assigns: [],
    }
    // Inner shadows outer, so inner's i=0..2 wins for name resolution.
    // 2 outer × 3 inner = 6 decls, but names collide (v0, v1, v2, v0, v1, v2)
    // → should throw on collision. If shielding were broken we'd still get
    // collisions, but for the wrong reason (outer leaking into inner's var
    // field). Either way, collision throws.
    expect(() => expandDeclGenerators(block)).toThrow(/duplicate decl name 'v0'/)
  })

  test('gateable with str_concat inside ref.instance inside gate_input', () => {
    // The scenario Bob flagged as un-tested but working: a generated
    // gateable instance whose gate_input references another generated
    // instance by a str_concat-computed name.
    const block = {
      op: 'block' as const,
      decls: [{
        op: 'generate_decls',
        count: 2,
        var: 'i',
        decls: [
          {
            op: 'instance_decl',
            name: { op: 'str_concat', parts: ['g', { op: 'binding', name: 'i' }] },
            program: 'SinOsc',
            inputs: {},
          },
          {
            op: 'instance_decl',
            name: { op: 'str_concat', parts: ['v', { op: 'binding', name: 'i' }] },
            program: 'Coconut',
            inputs: {},
            gateable: true,
            gate_input: {
              op: 'gt',
              args: [
                { op: 'ref',
                  instance: { op: 'str_concat', parts: ['g', { op: 'binding', name: 'i' }] },
                  output: 'out' },
                0.5,
              ],
            },
          },
        ],
      } as ExprNode],
      assigns: [],
    }
    const result = expandDeclGenerators(block)
    // Expect 4 decls: g0, v0, g1, v1.
    expect(result.decls).toHaveLength(4)
    const v0 = result.decls![1] as Record<string, unknown>
    expect(v0.name).toBe('v0')
    expect(v0.gateable).toBe(true)
    // gate_input's ref.instance must be a resolved string, not a str_concat object.
    const gateInput = v0.gate_input as Record<string, unknown>
    const refNode = (gateInput.args as ExprNode[])[0] as Record<string, unknown>
    expect(refNode.op).toBe('ref')
    expect(refNode.instance).toBe('g0')
  })
})
