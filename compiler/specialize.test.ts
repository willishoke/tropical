import { describe, test, expect } from 'bun:test'
import {
  specializeProgramJSON,
  specializationCacheKey,
  resolveTypeArgs,
} from './specialize.js'
import type { ProgramJSON } from './program.js'

function makeGenericDelay(): ProgramJSON {
  return {
    schema: 'tropical_program_1',
    name: 'Delay',
    type_params: { N: { type: 'int', default: 44100 } },
    inputs: ['x'],
    outputs: ['y'],
    regs: { buf: { zeros: { type_param: 'N' } } },
    input_defaults: { x: 0 },
    breaks_cycles: true,
    process: {
      outputs: {
        y: {
          op: 'index',
          args: [
            { op: 'reg', name: 'buf' },
            { op: 'mod', args: [{ op: 'sample_index' }, { op: 'type_param', name: 'N' }] },
          ],
        },
      },
      next_regs: {
        buf: {
          op: 'array_set',
          args: [
            { op: 'reg', name: 'buf' },
            { op: 'mod', args: [{ op: 'sample_index' }, { op: 'type_param', name: 'N' }] },
            { op: 'input', name: 'x' },
          ],
        },
      },
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

describe('specializeProgramJSON', () => {
  test('substitutes type_param in process outputs and next_regs', () => {
    const prog = makeGenericDelay()
    const spec = specializeProgramJSON(prog, { N: 8 })

    const yExpr = spec.process!.outputs.y as any
    expect(yExpr.args[1].args[1]).toBe(8)

    const bufExpr = spec.process!.next_regs!.buf as any
    expect(bufExpr.args[1].args[1]).toBe(8)
  })

  test('substitutes { zeros: { type_param } } to { zeros: N }', () => {
    const prog = makeGenericDelay()
    const spec = specializeProgramJSON(prog, { N: 512 })
    expect(spec.regs!.buf).toEqual({ zeros: 512 })
  })

  test('leaves { zeros: N } literal untouched', () => {
    const prog: ProgramJSON = {
      schema: 'tropical_program_1',
      name: 'Fixed',
      regs: { buf: { zeros: 100 } as any },
      process: { outputs: { y: 0 } },
      outputs: ['y'],
    }
    const spec = specializeProgramJSON(prog, {})
    expect(spec.regs!.buf).toEqual({ zeros: 100 })
  })

  test('is a deep clone — mutating output does not affect input', () => {
    const prog = makeGenericDelay()
    const spec = specializeProgramJSON(prog, { N: 8 })
    ;(spec.process!.outputs.y as any).op = 'mutated'
    const yExpr = prog.process!.outputs.y as any
    expect(yExpr.op).toBe('index')
  })

  test('substitutes in input_defaults', () => {
    const prog: ProgramJSON = {
      schema: 'tropical_program_1',
      name: 'X',
      type_params: { N: { type: 'int' } },
      inputs: ['x'],
      outputs: ['y'],
      input_defaults: { x: { op: 'type_param', name: 'N' } },
      process: { outputs: { y: { op: 'input', name: 'x' } } },
    }
    const spec = specializeProgramJSON(prog, { N: 42 })
    expect(spec.input_defaults!.x).toBe(42)
  })

  test('substitutes in instance inputs and type_args', () => {
    const prog: ProgramJSON = {
      schema: 'tropical_program_1',
      name: 'Outer',
      type_params: { N: { type: 'int' } },
      inputs: [],
      outputs: ['y'],
      instances: {
        d: {
          program: 'Delay',
          inputs: { x: { op: 'type_param', name: 'N' } },
          type_args: { N: { op: 'type_param', name: 'N' } },
        },
      },
      process: { outputs: { y: { op: 'nested_out', ref: 'd', output: 'y' } } },
    }
    const spec = specializeProgramJSON(prog, { N: 256 })
    expect(spec.instances!.d.inputs!.x).toBe(256)
    expect((spec.instances!.d as any).type_args).toEqual({ N: 256 })
  })

  test('throws on type_param ref with undeclared name', () => {
    const prog: ProgramJSON = {
      schema: 'tropical_program_1',
      name: 'X',
      type_params: { N: { type: 'int' } },
      outputs: ['y'],
      process: { outputs: { y: { op: 'type_param', name: 'Z' } } },
    }
    expect(() => specializeProgramJSON(prog, { N: 8 })).toThrow(/undeclared type_param 'Z'/)
  })

  test('substitutes type_param inside combinator body', () => {
    const prog: ProgramJSON = {
      schema: 'tropical_program_1',
      name: 'X',
      type_params: { N: { type: 'int' } },
      outputs: ['y'],
      process: {
        outputs: {
          y: {
            op: 'generate',
            n: { op: 'type_param', name: 'N' } as any,
            i: 'i',
            body: { op: 'binding', name: 'i' },
          } as any,
        },
      },
    }
    const spec = specializeProgramJSON(prog, { N: 4 })
    const y = spec.process!.outputs.y as any
    expect(y.n).toBe(4)
  })

  test('substitutes type_param refs in input port type shapes', () => {
    const prog: ProgramJSON = {
      schema: 'tropical_program_1',
      name: 'ArrIn',
      type_params: { N: { type: 'int' } },
      inputs: [
        { name: 'values', type: { kind: 'array', element: 'float', shape: [{ op: 'type_param', name: 'N' }] } },
      ],
      outputs: ['y'],
      process: { outputs: { y: 0 } },
    }
    const spec = specializeProgramJSON(prog, { N: 8 })
    const input = spec.inputs![0] as { name: string; type: any }
    expect(input.type).toEqual({ kind: 'array', element: 'float', shape: [8] })
  })

  test('substitutes type_param refs in output port type shapes', () => {
    const prog: ProgramJSON = {
      schema: 'tropical_program_1',
      name: 'ArrOut',
      type_params: { N: { type: 'int' } },
      outputs: [
        { name: 'out', type: { kind: 'array', element: 'float', shape: [{ op: 'type_param', name: 'N' }, 2] } },
      ],
      process: { outputs: { out: 0 } },
    }
    const spec = specializeProgramJSON(prog, { N: 3 })
    const out = spec.outputs![0] as { name: string; type: any }
    expect(out.type).toEqual({ kind: 'array', element: 'float', shape: [3, 2] })
  })

  test('rejects a port type shape referencing an undeclared type_param', () => {
    const prog: ProgramJSON = {
      schema: 'tropical_program_1',
      name: 'BadArr',
      type_params: { N: { type: 'int' } },
      inputs: [
        { name: 'values', type: { kind: 'array', element: 'float', shape: [{ op: 'type_param', name: 'M' }] } },
      ],
      outputs: ['y'],
      process: { outputs: { y: 0 } },
    }
    expect(() => specializeProgramJSON(prog, { N: 4 })).toThrow(/undeclared type_param 'M'/)
  })

  // ── Typed-const substitution (Phase 1) ──────────────────────
  //
  // type_params with declared `type: 'int'` (or 'bool') must emit a typed
  // const ExprNode `{op:'const', val:N, type:'int'}`, not a bare JS number.
  // The bare-number path forces emit_numeric to type it as float.

  test("int type_param ref inside ExprNode position substitutes to a typed-int const node", () => {
    const prog: ProgramJSON = {
      schema: 'tropical_program_1',
      name: 'X',
      type_params: { N: { type: 'int', default: 8 } },
      regs: { step: { init: 0, type: 'int' } as any },
      outputs: ['y'],
      process: {
        outputs: { y: { op: 'reg', name: 'step' } },
        next_regs: {
          step: {
            op: 'mod',
            args: [
              { op: 'add', args: [{ op: 'reg', name: 'step' }, 1] },
              { op: 'type_param', name: 'N' },
            ],
          },
        },
      },
    }
    const spec = specializeProgramJSON(prog, { N: 4 })
    const stepExpr = spec.process!.next_regs!.step as any
    // mod(args[0], args[1]) where args[1] was `{op:'type_param',name:'N'}`.
    expect(stepExpr.args[1]).toEqual({ op: 'const', val: 4, type: 'int' })
  })
})
