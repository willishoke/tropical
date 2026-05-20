import { describe, test, expect } from 'bun:test'
import type { ExprNode } from '../expr.js'
import type {
  ResolvedExpr, InputRef, ParamDecl, RegDecl, OutputAssign,
} from './nodes.js'
import { freeRefs, liftWireToProgram } from './wire_program.js'
import {
  instanceName, portName, portRef, rawName,
  type InstanceName,
} from './branded_names.js'
import { strataPipeline } from './strata.js'

// ─── Helper: pretty key-set for assertions ───────────────────────────────

const keysOf = (refs: ReadonlySet<{ instance: InstanceName; port: { toString(): string } }>) =>
  Array.from(refs).map(r => `${rawName(r.instance)}:${r.port}`).sort()

// ─── freeRefs ────────────────────────────────────────────────────────────

describe('freeRefs — leaves', () => {
  test('returns empty set for bare number', () => {
    expect(keysOf(freeRefs(42))).toEqual([])
  })

  test('returns empty set for bare boolean', () => {
    expect(keysOf(freeRefs(true))).toEqual([])
  })

  test('returns empty set for inline array of literals', () => {
    expect(keysOf(freeRefs([1, 2, 3]))).toEqual([])
  })

  test('returns empty set for sampleRate / sampleIndex', () => {
    expect(keysOf(freeRefs({ op: 'sampleRate' }))).toEqual([])
    expect(keysOf(freeRefs({ op: 'sampleIndex' }))).toEqual([])
  })

  test('returns empty set for param / trigger refs (not free refs)', () => {
    expect(keysOf(freeRefs({ op: 'param', name: 'cutoff' }))).toEqual([])
    expect(keysOf(freeRefs({ op: 'trigger', name: 'fire' }))).toEqual([])
  })
})

describe('freeRefs — refs', () => {
  test('collects a single ref', () => {
    const e: ExprNode = { op: 'ref', instance: 'a', output: 'out' }
    expect(keysOf(freeRefs(e))).toEqual(['a:out'])
  })

  test('deduplicates identical refs', () => {
    const e: ExprNode = {
      op: 'add',
      args: [
        { op: 'ref', instance: 'a', output: 'out' },
        { op: 'ref', instance: 'a', output: 'out' },
      ],
    }
    expect(keysOf(freeRefs(e))).toEqual(['a:out'])
  })

  test('collects multiple distinct refs', () => {
    const e: ExprNode = {
      op: 'add',
      args: [
        { op: 'ref', instance: 'a', output: 'x' },
        { op: 'ref', instance: 'b', output: 'y' },
      ],
    }
    expect(keysOf(freeRefs(e))).toEqual(['a:x', 'b:y'])
  })

  test('descends through nested binary ops', () => {
    // (a.x * b.y) + c.z
    const e: ExprNode = {
      op: 'add',
      args: [
        { op: 'mul', args: [
          { op: 'ref', instance: 'a', output: 'x' },
          { op: 'ref', instance: 'b', output: 'y' },
        ] },
        { op: 'ref', instance: 'c', output: 'z' },
      ],
    }
    expect(keysOf(freeRefs(e))).toEqual(['a:x', 'b:y', 'c:z'])
  })

  test('descends through unary / ternary ops', () => {
    const e: ExprNode = {
      op: 'select',
      args: [
        { op: 'gt', args: [
          { op: 'ref', instance: 'env', output: 'out' },
          0.5,
        ] },
        { op: 'neg', args: [{ op: 'ref', instance: 'osc', output: 'out' }] },
        { op: 'ref', instance: 'noise', output: 'out' },
      ],
    }
    expect(keysOf(freeRefs(e))).toEqual(['env:out', 'noise:out', 'osc:out'])
  })

  test('descends into inline array literals (`{op: array, items}`)', () => {
    const e: ExprNode = {
      op: 'array',
      items: [
        { op: 'ref', instance: 'a', output: 'x' },
        { op: 'ref', instance: 'b', output: 'y' },
      ],
    }
    expect(keysOf(freeRefs(e))).toEqual(['a:x', 'b:y'])
  })

  test('descends into bare-array (literal form) entries', () => {
    const e: ExprNode = [
      { op: 'ref', instance: 'a', output: 'x' },
      { op: 'ref', instance: 'b', output: 'y' },
    ]
    expect(keysOf(freeRefs(e))).toEqual(['a:x', 'b:y'])
  })

  test('descends into delay() args', () => {
    const e: ExprNode = {
      op: 'delay',
      args: [{ op: 'ref', instance: 'b', output: 'y' }],
      init: 0,
    }
    expect(keysOf(freeRefs(e))).toEqual(['b:y'])
  })

  test('descends into index(arr, i)', () => {
    const e: ExprNode = {
      op: 'index',
      args: [
        { op: 'ref', instance: 'clock', output: 'ratios_out' },
        2,
      ],
    }
    expect(keysOf(freeRefs(e))).toEqual(['clock:ratios_out'])
  })

  test('throws on numeric output (wire form must be string)', () => {
    const e: ExprNode = { op: 'ref', instance: 'a', output: 0 }
    expect(() => freeRefs(e)).toThrow(/output must be a string port name/)
  })
})

// ─── liftWireToProgram — structural ──────────────────────────────────────

describe('liftWireToProgram — structure', () => {
  test('literal-only wire produces a zero-input program', () => {
    const e: ExprNode = 42
    const refs = freeRefs(e)
    const prog = liftWireToProgram(e, refs, instanceName('__wire_0'))
    expect(prog.op).toBe('program')
    expect(prog.name).toBe('__wire_0')
    expect(prog.typeParams).toEqual([])
    expect(prog.ports.inputs).toEqual([])
    expect(prog.ports.outputs.length).toBe(1)
    expect(prog.ports.outputs[0].name).toBe('out')
  })

  test('single-ref wire produces one input port', () => {
    const e: ExprNode = { op: 'ref', instance: 'a', output: 'out' }
    const prog = liftWireToProgram(e, freeRefs(e), instanceName('__wire_1'))
    expect(prog.ports.inputs.length).toBe(1)
    expect(prog.ports.inputs[0].name).toBe('a__out')
  })

  test('two-ref wire produces two inputs in sorted order', () => {
    const e: ExprNode = {
      op: 'add',
      args: [
        { op: 'ref', instance: 'z', output: 'foo' },
        { op: 'ref', instance: 'a', output: 'bar' },
      ],
    }
    const prog = liftWireToProgram(e, freeRefs(e), instanceName('__wire_2'))
    expect(prog.ports.inputs.map(d => d.name)).toEqual(['a__bar', 'z__foo'])
  })

  test('nested instance names (dotted) get safe input names', () => {
    const e: ExprNode = { op: 'ref', instance: 'voice1.env', output: 'out' }
    const prog = liftWireToProgram(e, freeRefs(e), instanceName('__wire_3'))
    expect(prog.ports.inputs[0].name).toBe('voice1_env__out')
  })

  test('output port is named "out"', () => {
    const e: ExprNode = 0
    const prog = liftWireToProgram(e, freeRefs(e), instanceName('w'))
    expect(prog.ports.outputs[0].name).toBe('out')
  })

  test('body has exactly one outputAssign', () => {
    const e: ExprNode = { op: 'ref', instance: 'a', output: 'out' }
    const prog = liftWireToProgram(e, freeRefs(e), instanceName('w'))
    expect(prog.body.assigns.length).toBe(1)
    const assign = prog.body.assigns[0] as OutputAssign
    expect(assign.op).toBe('outputAssign')
    expect(assign.target).toBe(prog.ports.outputs[0])
  })

  test('ref translation: target is the inputRef pointing at the matching input decl', () => {
    const e: ExprNode = { op: 'ref', instance: 'a', output: 'out' }
    const prog = liftWireToProgram(e, freeRefs(e), instanceName('w'))
    const assign = prog.body.assigns[0] as OutputAssign
    const expr = assign.expr as InputRef
    expect(expr.op).toBe('inputRef')
    expect(expr.decl).toBe(prog.ports.inputs[0])
  })

  test('binary op translates with both args translated', () => {
    const e: ExprNode = {
      op: 'add',
      args: [
        { op: 'ref', instance: 'a', output: 'out' },
        2.5,
      ],
    }
    const prog = liftWireToProgram(e, freeRefs(e), instanceName('w'))
    const assign = prog.body.assigns[0] as OutputAssign
    const expr = assign.expr as { op: 'add'; args: [ResolvedExpr, ResolvedExpr] }
    expect(expr.op).toBe('add')
    expect((expr.args[0] as InputRef).op).toBe('inputRef')
    expect(expr.args[1]).toBe(2.5)
  })

  test('delay() produces a synthetic update-bearing RegDecl in the body', () => {
    // Post-Phase-0a: session `delay()` lowers to a RegDecl with update
    // populated. Reads of the value are RegRefs.
    const e: ExprNode = {
      op: 'delay',
      args: [{ op: 'ref', instance: 'b', output: 'y' }],
      init: 0.5,
    }
    const prog = liftWireToProgram(e, freeRefs(e), instanceName('w'))
    const regs = prog.body.decls.filter(d => d.op === 'regDecl') as RegDecl[]
    expect(regs.length).toBe(1)
    expect(regs[0].name).toBe('__sd0')
    expect(regs[0].init).toBe(0.5)
    expect(regs[0].update).toBeDefined()
    const assign = prog.body.assigns[0] as OutputAssign
    expect((assign.expr as { op: string }).op).toBe('regRef')
  })

  test('param ref produces a private ParamDecl', () => {
    const e: ExprNode = { op: 'param', name: 'cutoff' }
    const prog = liftWireToProgram(e, freeRefs(e), instanceName('w'))
    const params = prog.body.decls.filter(d => d.op === 'paramDecl') as ParamDecl[]
    expect(params.length).toBe(1)
    expect(params[0].name).toBe('cutoff')
  })

  test('legacy {op:trigger} ref aliases to a ParamDecl', () => {
    const e: ExprNode = { op: 'trigger', name: 'fire' }
    const prog = liftWireToProgram(e, freeRefs(e), instanceName('w'))
    const params = prog.body.decls.filter(d => d.op === 'paramDecl') as ParamDecl[]
    expect(params.length).toBe(1)
    expect(params[0].name).toBe('fire')
  })

  test('repeated param ref reuses one ParamDecl', () => {
    const e: ExprNode = {
      op: 'add',
      args: [
        { op: 'param', name: 'k' },
        { op: 'param', name: 'k' },
      ],
    }
    const prog = liftWireToProgram(e, freeRefs(e), instanceName('w'))
    const params = prog.body.decls.filter(d => d.op === 'paramDecl') as ParamDecl[]
    expect(params.length).toBe(1)
  })

  test('inline array literal translates to bare array of ResolvedExprs', () => {
    const e: ExprNode = { op: 'array', items: [1, 2, 3] }
    const prog = liftWireToProgram(e, freeRefs(e), instanceName('w'))
    const assign = prog.body.assigns[0] as OutputAssign
    expect(assign.expr).toEqual([1, 2, 3])
  })

  test('bare-array literal translates to bare ResolvedExpr array', () => {
    const e: ExprNode = [1, 2, 3]
    const prog = liftWireToProgram(e, freeRefs(e), instanceName('w'))
    const assign = prog.body.assigns[0] as OutputAssign
    expect(assign.expr).toEqual([1, 2, 3])
  })
})

// ─── liftWireToProgram — error paths ─────────────────────────────────────

describe('liftWireToProgram — errors', () => {
  test('rejects unknown op', () => {
    const e: ExprNode = { op: 'fold' as any, over: 0, init: 0, acc: 'a', elem: 'e', body: 0 }
    expect(() => liftWireToProgram(e, freeRefs(e), instanceName('w'))).toThrow(
      /unhandled wire-form op 'fold'/,
    )
  })

  test('rejects ref absent from freeRefSet (caller error)', () => {
    const e: ExprNode = { op: 'ref', instance: 'a', output: 'out' }
    const emptySet = new Set<ReturnType<typeof portRef>>()
    expect(() => liftWireToProgram(e, emptySet, instanceName('w'))).toThrow(
      /not in freeRefSet/,
    )
  })

  test('param and legacy {op:trigger} with same name share one ParamDecl', () => {
    const e: ExprNode = {
      op: 'add',
      args: [
        { op: 'param', name: 'shared' },
        { op: 'trigger', name: 'shared' },
      ],
    }
    const prog = liftWireToProgram(e, freeRefs(e), instanceName('w'))
    const params = prog.body.decls.filter(d => d.op === 'paramDecl') as ParamDecl[]
    expect(params.length).toBe(1)
    expect(params[0].name).toBe('shared')
  })
})

// ─── Round-trip with strata pipeline ─────────────────────────────────────

describe('liftWireToProgram — strata round-trip', () => {
  test('lifted program passes through strataPipeline cleanly', () => {
    const e: ExprNode = {
      op: 'add',
      args: [
        { op: 'mul', args: [
          { op: 'ref', instance: 'a', output: 'out' },
          2,
        ] },
        { op: 'ref', instance: 'b', output: 'y' },
      ],
    }
    const prog = liftWireToProgram(e, freeRefs(e), instanceName('__wire_round_trip'))
    // strataPipeline should be a no-op or near-no-op on a wire program:
    // no instances to inline, no sum types, no array combinators.
    const after = strataPipeline(prog)
    expect(after.op).toBe('program')
    expect(after.name).toBe('__wire_round_trip')
    expect(after.ports.inputs.length).toBe(2)
    expect(after.ports.outputs.length).toBe(1)
    expect(after.ports.outputs[0].name).toBe('out')
  })

  test('lifted program with delay passes through strata cleanly', () => {
    // Post-Phase-0a: delay desugars to RegDecl-with-update.
    const e: ExprNode = {
      op: 'delay',
      args: [{ op: 'ref', instance: 'b', output: 'y' }],
      init: 0,
    }
    const prog = liftWireToProgram(e, freeRefs(e), instanceName('__wire_delay'))
    const after = strataPipeline(prog)
    const regs = after.body.decls.filter(
      d => d.op === 'regDecl' && d.update !== undefined,
    )
    expect(regs.length).toBe(1)
  })
})
