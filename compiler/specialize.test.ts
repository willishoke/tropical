import { describe, test, expect } from 'bun:test'
import {
  specializeProgramNode,
  specializationCacheKey,
  resolveTypeArgs,
} from './specialize.js'
import type { ExprNode } from './expr.js'
import type { ProgramNode } from './program.js'

type RegDecl = { op: 'reg_decl'; name: string; init: unknown; type?: unknown }
type InstanceDecl = { op: 'instance_decl'; name: string; program: string; inputs?: Record<string, ExprNode>; type_args?: Record<string, number | ExprNode> }
type OutputAssign = { op: 'output_assign'; name: string; expr: ExprNode }
type NextUpdate = { op: 'next_update'; target: { kind: 'reg' | 'delay'; name: string }; expr: ExprNode }

function findDecl<T extends { op: string; name: string }>(prog: ProgramNode, op: T['op'], name: string): T {
  const decl = (prog.body.decls ?? []).find(d =>
    typeof d === 'object' && d !== null && !Array.isArray(d) &&
    (d as { op?: string }).op === op && (d as { name?: string }).name === name,
  )
  if (!decl) throw new Error(`no ${op} named '${name}'`)
  return decl as T
}

function findAssign<T extends { op: string }>(prog: ProgramNode, op: T['op'], matcher: (a: T) => boolean): T {
  const a = (prog.body.assigns ?? []).find(x =>
    typeof x === 'object' && x !== null && !Array.isArray(x) &&
    (x as { op?: string }).op === op && matcher(x as T),
  )
  if (!a) throw new Error(`no ${op} matching predicate`)
  return a as T
}

function makeGenericDelay(): ProgramNode {
  return {
    op: 'program',
    name: 'Delay',
    type_params: { N: { type: 'int', default: 44100 } },
    breaks_cycles: true,
    ports: {
      inputs: [{ name: 'x', default: 0 }],
      outputs: ['y'],
    },
    body: {
      op: 'block',
      decls: [
        { op: 'reg_decl', name: 'buf', init: { zeros: { type_param: 'N' } } } as unknown as ExprNode,
      ],
      assigns: [
        {
          op: 'output_assign',
          name: 'y',
          expr: {
            op: 'index',
            args: [
              { op: 'reg', name: 'buf' },
              { op: 'mod', args: [{ op: 'sample_index' }, { op: 'type_param', name: 'N' }] },
            ],
          },
        } as unknown as ExprNode,
        {
          op: 'next_update',
          target: { kind: 'reg', name: 'buf' },
          expr: {
            op: 'array_set',
            args: [
              { op: 'reg', name: 'buf' },
              { op: 'mod', args: [{ op: 'sample_index' }, { op: 'type_param', name: 'N' }] },
              { op: 'input', name: 'x' },
            ],
          },
        } as unknown as ExprNode,
      ],
    },
  }
}

describe('resolveTypeArgs', () => {
  const params = { N: { type: 'int' as const, default: 44100 } }

  test('passes numeric literals through', () => {
    expect(resolveTypeArgs({ N: 8 }, undefined, params, 'test')).toEqual({ N: 8 })
  })

  test('applies declared default when arg missing', () => {
    expect(resolveTypeArgs({}, undefined, params, 'test')).toEqual({ N: 44100 })
    expect(resolveTypeArgs(undefined, undefined, params, 'test')).toEqual({ N: 44100 })
  })

  test('resolves type_param refs against outer frame', () => {
    const arg = { op: 'type_param' as const, name: 'M' }
    expect(resolveTypeArgs({ N: arg }, { M: 16 }, params, 'test')).toEqual({ N: 16 })
  })

  test('throws on missing required param (no default)', () => {
    const required = { M: { type: 'int' as const } }
    expect(() => resolveTypeArgs({}, undefined, required, 'test')).toThrow(/missing required type_arg 'M'/)
  })

  test('throws on unknown arg key', () => {
    expect(() => resolveTypeArgs({ Z: 1 }, undefined, params, 'test')).toThrow(/unknown type_arg 'Z'/)
  })

  test('throws on non-integer value', () => {
    expect(() => resolveTypeArgs({ N: 3.14 }, undefined, params, 'test')).toThrow(/must be an integer/)
  })

  test('throws on unresolved type_param ref', () => {
    const arg = { op: 'type_param' as const, name: 'M' }
    expect(() => resolveTypeArgs({ N: arg }, undefined, params, 'test')).toThrow(/unresolved type_param 'M'/)
    expect(() => resolveTypeArgs({ N: arg }, { K: 5 }, params, 'test')).toThrow(/unresolved type_param 'M'/)
  })
})

describe('specializationCacheKey', () => {
  test('is stable under key ordering', () => {
    const a = specializationCacheKey('Delay', { N: 8, M: 4 })
    const b = specializationCacheKey('Delay', { M: 4, N: 8 })
    expect(a).toBe(b)
  })

  test('distinguishes different arg values', () => {
    const a = specializationCacheKey('Delay', { N: 8 })
    const b = specializationCacheKey('Delay', { N: 16 })
    expect(a).not.toBe(b)
  })
})

describe('specializeProgramNode', () => {
  test('substitutes type_param in output and next_update exprs', () => {
    const prog = makeGenericDelay()
    const spec = specializeProgramNode(prog, { N: 8 })

    // N is declared `type: 'int'`, so substitutions emit typed-const nodes.
    const yAssign = findAssign<OutputAssign>(spec, 'output_assign', a => a.name === 'y')
    const yExpr = yAssign.expr as { args: [unknown, { args: [unknown, unknown] }] }
    expect(yExpr.args[1].args[1]).toEqual({ op: 'const', val: 8, type: 'int' } as ExprNode)

    const bufUpdate = findAssign<NextUpdate>(spec, 'next_update', a => a.target.name === 'buf')
    const bufExpr = bufUpdate.expr as { args: [unknown, { args: [unknown, unknown] }, unknown] }
    expect(bufExpr.args[1].args[1]).toEqual({ op: 'const', val: 8, type: 'int' } as ExprNode)
  })

  test('substitutes { zeros: { type_param } } to { zeros: N }', () => {
    const prog = makeGenericDelay()
    const spec = specializeProgramNode(prog, { N: 512 })
    const buf = findDecl<RegDecl>(spec, 'reg_decl', 'buf')
    expect(buf.init).toEqual({ zeros: 512 })
  })

  test('leaves { zeros: N } literal untouched', () => {
    const prog: ProgramNode = {
      op: 'program',
      name: 'Fixed',
      ports: { outputs: ['y'] },
      body: {
        op: 'block',
        decls: [{ op: 'reg_decl', name: 'buf', init: { zeros: 100 } } as unknown as ExprNode],
        assigns: [{ op: 'output_assign', name: 'y', expr: 0 } as unknown as ExprNode],
      },
    }
    const spec = specializeProgramNode(prog, {})
    const buf = findDecl<RegDecl>(spec, 'reg_decl', 'buf')
    expect(buf.init).toEqual({ zeros: 100 })
  })

  test('is a deep clone — mutating output does not affect input', () => {
    const prog = makeGenericDelay()
    const spec = specializeProgramNode(prog, { N: 8 })
    const yAssign = findAssign<OutputAssign>(spec, 'output_assign', a => a.name === 'y')
    ;(yAssign.expr as { op: string }).op = 'mutated'
    const origAssign = findAssign<OutputAssign>(prog, 'output_assign', a => a.name === 'y')
    expect((origAssign.expr as { op: string }).op).toBe('index')
  })

  test('substitutes in input port defaults', () => {
    const prog: ProgramNode = {
      op: 'program',
      name: 'X',
      type_params: { N: { type: 'int' } },
      ports: {
        inputs: [{ name: 'x', default: { op: 'type_param', name: 'N' } }],
        outputs: ['y'],
      },
      body: {
        op: 'block',
        assigns: [{ op: 'output_assign', name: 'y', expr: { op: 'input', name: 'x' } } as unknown as ExprNode],
      },
    }
    const spec = specializeProgramNode(prog, { N: 42 })
    const xSpec = spec.ports!.inputs![0] as { name: string; default: ExprNode }
    expect(xSpec.default).toEqual({ op: 'const', val: 42, type: 'int' } as ExprNode)
  })

  test('substitutes in instance inputs and type_args', () => {
    const prog: ProgramNode = {
      op: 'program',
      name: 'Outer',
      type_params: { N: { type: 'int' } },
      ports: { inputs: [], outputs: ['y'] },
      body: {
        op: 'block',
        decls: [
          {
            op: 'instance_decl',
            name: 'd',
            program: 'Delay',
            inputs: { x: { op: 'type_param', name: 'N' } },
            type_args: { N: { op: 'type_param', name: 'N' } },
          } as unknown as ExprNode,
        ],
        assigns: [
          { op: 'output_assign', name: 'y', expr: { op: 'nested_out', ref: 'd', output: 'y' } } as unknown as ExprNode,
        ],
      },
    }
    const spec = specializeProgramNode(prog, { N: 256 })
    const d = findDecl<InstanceDecl>(spec, 'instance_decl', 'd')
    expect(d.inputs!.x).toEqual({ op: 'const', val: 256, type: 'int' } as ExprNode)
    expect(d.type_args).toEqual({ N: 256 })
  })

  test('throws on type_param ref with undeclared name', () => {
    const prog: ProgramNode = {
      op: 'program',
      name: 'X',
      type_params: { N: { type: 'int' } },
      ports: { outputs: ['y'] },
      body: {
        op: 'block',
        assigns: [{ op: 'output_assign', name: 'y', expr: { op: 'type_param', name: 'Z' } } as unknown as ExprNode],
      },
    }
    expect(() => specializeProgramNode(prog, { N: 8 })).toThrow(/undeclared type_param 'Z'/)
  })

  test('substitutes type_param inside combinator body', () => {
    const prog: ProgramNode = {
      op: 'program',
      name: 'X',
      type_params: { N: { type: 'int' } },
      ports: { outputs: ['y'] },
      body: {
        op: 'block',
        assigns: [
          {
            op: 'output_assign',
            name: 'y',
            expr: {
              op: 'generate',
              n: { op: 'type_param', name: 'N' },
              i: 'i',
              body: { op: 'binding', name: 'i' },
            },
          } as unknown as ExprNode,
        ],
      },
    }
    const spec = specializeProgramNode(prog, { N: 4 })
    const yAssign = findAssign<OutputAssign>(spec, 'output_assign', a => a.name === 'y')
    const y = yAssign.expr as { n: ExprNode }
    expect(y.n).toEqual({ op: 'const', val: 4, type: 'int' } as ExprNode)
  })

  test('substitutes type_param refs in input port type shapes', () => {
    const prog: ProgramNode = {
      op: 'program',
      name: 'ArrIn',
      type_params: { N: { type: 'int' } },
      ports: {
        inputs: [
          { name: 'values', type: { kind: 'array', element: 'float', shape: [{ op: 'type_param', name: 'N' }] } },
        ],
        outputs: ['y'],
      },
      body: {
        op: 'block',
        assigns: [{ op: 'output_assign', name: 'y', expr: 0 } as unknown as ExprNode],
      },
    }
    const spec = specializeProgramNode(prog, { N: 8 })
    const input = spec.ports!.inputs![0] as { name: string; type: unknown }
    expect(input.type).toEqual({ kind: 'array', element: 'float', shape: [8] })
  })

  test('substitutes type_param refs in output port type shapes', () => {
    const prog: ProgramNode = {
      op: 'program',
      name: 'ArrOut',
      type_params: { N: { type: 'int' } },
      ports: {
        outputs: [
          { name: 'out', type: { kind: 'array', element: 'float', shape: [{ op: 'type_param', name: 'N' }, 2] } },
        ],
      },
      body: {
        op: 'block',
        assigns: [{ op: 'output_assign', name: 'out', expr: 0 } as unknown as ExprNode],
      },
    }
    const spec = specializeProgramNode(prog, { N: 3 })
    const out = spec.ports!.outputs![0] as { name: string; type: unknown }
    expect(out.type).toEqual({ kind: 'array', element: 'float', shape: [3, 2] })
  })

  test('rejects a port type shape referencing an undeclared type_param', () => {
    const prog: ProgramNode = {
      op: 'program',
      name: 'BadArr',
      type_params: { N: { type: 'int' } },
      ports: {
        inputs: [
          { name: 'values', type: { kind: 'array', element: 'float', shape: [{ op: 'type_param', name: 'M' }] } },
        ],
        outputs: ['y'],
      },
      body: {
        op: 'block',
        assigns: [{ op: 'output_assign', name: 'y', expr: 0 } as unknown as ExprNode],
      },
    }
    expect(() => specializeProgramNode(prog, { N: 4 })).toThrow(/undeclared type_param 'M'/)
  })

  // ── Typed-const substitution ─────────────────────────────────
  //
  // type_params with declared `type: 'int'` (or 'bool') must emit a typed
  // const ExprNode `{op:'const', val:N, type:'int'}`, not a bare JS number.
  // The bare-number path forces emit_numeric to type it as float.

  test("int type_param ref inside ExprNode position substitutes to a typed-int const node", () => {
    const prog: ProgramNode = {
      op: 'program',
      name: 'X',
      type_params: { N: { type: 'int', default: 8 } },
      ports: { outputs: ['y'] },
      body: {
        op: 'block',
        decls: [{ op: 'reg_decl', name: 'step', init: 0, type: 'int' } as unknown as ExprNode],
        assigns: [
          { op: 'output_assign', name: 'y', expr: { op: 'reg', name: 'step' } } as unknown as ExprNode,
          {
            op: 'next_update',
            target: { kind: 'reg', name: 'step' },
            expr: {
              op: 'mod',
              args: [
                { op: 'add', args: [{ op: 'reg', name: 'step' }, 1] },
                { op: 'type_param', name: 'N' },
              ],
            },
          } as unknown as ExprNode,
        ],
      },
    }
    const spec = specializeProgramNode(prog, { N: 4 })
    const stepUpdate = findAssign<NextUpdate>(spec, 'next_update', a => a.target.name === 'step')
    const stepExpr = stepUpdate.expr as { args: [unknown, unknown] }
    expect(stepExpr.args[1]).toEqual({ op: 'const', val: 4, type: 'int' } as ExprNode)
  })
})
