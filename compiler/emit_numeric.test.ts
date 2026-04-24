/**
 * emit_numeric.test.ts — Tests for instruction emission (pipeline stage 7).
 *
 * Exercises emitNumericProgram(): ExprNode trees → FlatProgram instruction stream.
 * All tests are O4 (exact structural match) on instruction tags, operand kinds,
 * types, loop_count, strides, and slot indices.
 */

import { describe, test, expect } from 'bun:test'
import { emitNumericProgram, type FlatProgram, type NInstr, type NOperand } from './emit_numeric'
import type { ExprNode } from './expr'

/** Find the first instruction with the given tag. */
function findInstr(prog: FlatProgram, tag: string): NInstr | undefined {
  return prog.instructions.find(i => i.tag === tag)
}

/** Find all instructions with the given tag. */
function findAll(prog: FlatProgram, tag: string): NInstr[] {
  return prog.instructions.filter(i => i.tag === tag)
}

/** Get the output-copy instruction (the last one whose dst === output_targets[i]). */
function outputInstr(prog: FlatProgram, i = 0): NInstr {
  const dst = prog.output_targets[i]
  return prog.instructions.findLast(instr => instr.dst === dst)!
}

describe('emitNumericProgram', () => {

  // ── Terminals ──────────────────────────────────────────────

  test('number and boolean literals produce const operands', () => {
    const numProg = emitNumericProgram([42], [])
    const numOut = outputInstr(numProg)
    expect(numOut.tag).toBe('Add')  // output copy: Add(val, 0)
    expect(numOut.args[0]).toEqual({ kind: 'const', val: 42, scalar_type: 'float' })

    const boolProg = emitNumericProgram([true], [])
    const boolOut = outputInstr(boolProg)
    expect(boolOut.args[0]).toEqual({ kind: 'const', val: 1, scalar_type: 'bool' })
  })

  test('leaf node terminals: input, state_reg, sample_rate, sample_index, param', () => {
    const cases: [ExprNode, Partial<NOperand>][] = [
      [{ op: 'input', id: 0 },                              { kind: 'input', slot: 0, scalar_type: 'float' }],
      [{ op: 'reg', id: 0 },                                { kind: 'state_reg', slot: 0, scalar_type: 'float' }],
      [{ op: 'sample_rate' },                                { kind: 'rate', scalar_type: 'float' }],
      [{ op: 'sample_index' },                               { kind: 'tick', scalar_type: 'int' }],
      [{ op: 'smoothed_param', _ptr: true, _handle: 12345 }, { kind: 'param', ptr: '12345', scalar_type: 'float' }],
    ]
    for (const [expr, expected] of cases) {
      const prog = emitNumericProgram([expr], [])
      const out = outputInstr(prog)
      expect(out.args[0]).toMatchObject(expected)
    }
  })

  // ── Scalar ops ─────────────────────────────────────────────

  test('scalar binary: add(3, 4)', () => {
    const prog = emitNumericProgram([{ op: 'add', args: [3, 4] }], [])
    const add = findInstr(prog, 'Add')!
    expect(add).toBeDefined()
    expect(add.loop_count).toBe(1)
    expect(add.strides).toEqual([])
    expect(add.args[0]).toMatchObject({ kind: 'const', val: 3 })
    expect(add.args[1]).toMatchObject({ kind: 'const', val: 4 })
    expect(add.result_type).toBe('float')
    expect(prog.register_count).toBeGreaterThanOrEqual(2)
  })

  test('scalar unary: sqrt(4.0)', () => {
    const prog = emitNumericProgram([{ op: 'sqrt', args: [4.0] }], [])
    const sq = findInstr(prog, 'Sqrt')!
    expect(sq).toBeDefined()
    expect(sq.loop_count).toBe(1)
    expect(sq.args[0]).toMatchObject({ kind: 'const', val: 4.0 })
    expect(sq.result_type).toBe('float')
  })

  test('scalar ternary: select and clamp', () => {
    const selProg = emitNumericProgram([{ op: 'select', args: [true, 1, 2] }], [])
    const sel = findInstr(selProg, 'Select')!
    expect(sel).toBeDefined()
    expect(sel.args).toHaveLength(3)
    expect(sel.loop_count).toBe(1)
    expect(sel.args[0]).toMatchObject({ kind: 'const', val: 1, scalar_type: 'bool' })  // true → 1
    expect(sel.args[1]).toMatchObject({ kind: 'const', val: 1, scalar_type: 'float' })
    expect(sel.args[2]).toMatchObject({ kind: 'const', val: 2, scalar_type: 'float' })

    const clProg = emitNumericProgram([{ op: 'clamp', args: [5, 0, 10] }], [])
    const cl = findInstr(clProg, 'Clamp')!
    expect(cl).toBeDefined()
    expect(cl.args).toHaveLength(3)
    expect(cl.loop_count).toBe(1)
  })

  // ── Type inference ─────────────────────────────────────────

  test('type categories: bitwise→int, comparison→bool, sqrt→float', () => {
    const bitProg = emitNumericProgram([{ op: 'bit_and', args: [1, 2] }], [])
    expect(findInstr(bitProg, 'BitAnd')!.result_type).toBe('int')

    const cmpProg = emitNumericProgram([{ op: 'lt', args: [1, 2] }], [])
    expect(findInstr(cmpProg, 'Less')!.result_type).toBe('bool')

    const sqProg = emitNumericProgram([{ op: 'sqrt', args: [1] }], [])
    expect(findInstr(sqProg, 'Sqrt')!.result_type).toBe('float')
  })

  test('type promotion in arithmetic', () => {
    // float (input) + int (sample_index) → float
    const prog1 = emitNumericProgram(
      [{ op: 'add', args: [{ op: 'input', id: 0 }, { op: 'sample_index' }] }],
      [],
    )
    const add1 = findInstr(prog1, 'Add')!
    expect(add1.result_type).toBe('float')

    // int (reg 0) + bool (reg 1) → int
    const prog2 = emitNumericProgram(
      [{ op: 'add', args: [{ op: 'reg', id: 0 }, { op: 'reg', id: 1 }] }],
      [],
      [0, false],        // stateInit
      ['int', 'bool'],   // stateRegTypes
    )
    const add2 = findInstr(prog2, 'Add')!
    expect(add2.result_type).toBe('int')
  })

  // ── Array ops ──────────────────────────────────────────────

  test('inline array → Pack instruction', () => {
    // JS array literal
    const prog1 = emitNumericProgram([[1, 2, 3] as ExprNode], [])
    const pack1 = findInstr(prog1, 'Pack')!
    expect(pack1).toBeDefined()
    expect(pack1.args).toHaveLength(3)
    expect(prog1.array_slot_sizes).toContain(3)

    // {op:'array', items:[...]} JSON format
    const prog2 = emitNumericProgram([{ op: 'array', items: [4, 5] }], [])
    const pack2 = findInstr(prog2, 'Pack')!
    expect(pack2).toBeDefined()
    expect(pack2.args).toHaveLength(2)
    expect(prog2.array_slot_sizes).toContain(2)
  })

  test('elementwise binary with stride patterns', () => {
    // array + scalar → strides [1, 0]
    const prog1 = emitNumericProgram([{ op: 'add', args: [[1, 2, 3], 10] }], [])
    const add1 = findAll(prog1, 'Add').find(i => i.loop_count > 1)!
    expect(add1.loop_count).toBe(3)
    expect(add1.strides).toEqual([1, 0])

    // scalar + array → strides [0, 1]
    const prog2 = emitNumericProgram([{ op: 'add', args: [10, [1, 2, 3]] }], [])
    const add2 = findAll(prog2, 'Add').find(i => i.loop_count > 1)!
    expect(add2.loop_count).toBe(3)
    expect(add2.strides).toEqual([0, 1])

    // array + array → strides [1, 1]
    const prog3 = emitNumericProgram([{ op: 'add', args: [[1, 2, 3], [4, 5, 6]] }], [])
    const add3 = findAll(prog3, 'Add').find(i => i.loop_count > 1)!
    expect(add3.loop_count).toBe(3)
    expect(add3.strides).toEqual([1, 1])
  })

  test('size-1 array unboxing prevents JIT segfault', () => {
    // add([42], 1) — the [42] must be unboxed via Index[0] before scalar Add
    const prog = emitNumericProgram([{ op: 'add', args: [[42], 1] }], [])

    // Should have: Pack(size=1), Index (unbox), Add (scalar), Add (output copy)
    const pack = findInstr(prog, 'Pack')!
    expect(pack).toBeDefined()
    expect(prog.array_slot_sizes).toContain(1)

    const indices = findAll(prog, 'Index')
    expect(indices.length).toBeGreaterThanOrEqual(1)
    // The unbox Index reads from the array slot at const index 0
    const unbox = indices[0]
    expect(unbox.args[1]).toMatchObject({ kind: 'const', val: 0 })

    // The final Add that does the arithmetic should be scalar (loop_count=1)
    const adds = findAll(prog, 'Add')
    const scalarAdd = adds.find(a => a.loop_count === 1 && a.args.some(
      op => op.kind === 'const' && op.val === 1,
    ))
    expect(scalarAdd).toBeDefined()
    expect(scalarAdd!.loop_count).toBe(1)
  })

  test('index and array_set', () => {
    // index([10, 20, 30], 1)
    const idxProg = emitNumericProgram([{ op: 'index', args: [[10, 20, 30], 1] }], [])
    const pack = findInstr(idxProg, 'Pack')!
    expect(pack).toBeDefined()
    const idx = findInstr(idxProg, 'Index')!
    expect(idx).toBeDefined()
    expect(idx.loop_count).toBe(1)  // result is scalar

    // array_set([10, 20, 30], 1, 99)
    const setProg = emitNumericProgram([{ op: 'array_set', args: [[10, 20, 30], 1, 99] }], [])
    const setEl = findInstr(setProg, 'SetElement')!
    expect(setEl).toBeDefined()
    expect(setEl.loop_count).toBe(1)
  })

  // ── Output & register targets ──────────────────────────────

  test('output_targets: scalar output gets copy, array output gets Index[0]', () => {
    // Scalar output → last instr is Add(val, 0) copy
    const scalarProg = emitNumericProgram([42], [])
    const scalarOut = outputInstr(scalarProg)
    expect(scalarOut.tag).toBe('Add')
    expect(scalarOut.args[1]).toMatchObject({ kind: 'const', val: 0 })

    // Array output → last instr is Index(array_reg, 0)
    const arrProg = emitNumericProgram([[1, 2] as ExprNode], [])
    const arrOut = outputInstr(arrProg)
    expect(arrOut.tag).toBe('Index')
    expect(arrOut.args[0]).toMatchObject({ kind: 'array_reg' })
    expect(arrOut.args[1]).toMatchObject({ kind: 'const', val: 0 })
  })

  test('register_targets: null→-1, scalar update, array→-1', () => {
    const prog = emitNumericProgram(
      [42],
      [null, { op: 'add', args: [1, 2] }, [1, 2, 3]],
    )
    expect(prog.register_targets[0]).toBe(-1)          // null → no update
    expect(prog.register_targets[1]).toBeGreaterThan(-1) // scalar → valid temp
    expect(prog.register_targets[2]).toBe(-1)           // array → unsupported
  })

  // ── State & memoization ────────────────────────────────────

  test('typed state registers and array register init', () => {
    // Int register reads with correct scalar_type
    const intProg = emitNumericProgram(
      [{ op: 'reg', id: 0 }],
      [],
      [0],
      ['int'],
    )
    const intOut = outputInstr(intProg)
    expect(intOut.args[0]).toMatchObject({ kind: 'state_reg', slot: 0, scalar_type: 'int' })

    // Array register init: reg 1 backed by [1,2,3] → array_reg, not state_reg
    const arrProg = emitNumericProgram(
      [{ op: 'reg', id: 1 }],
      [],
      [0, [1, 2, 3]],
    )
    // The reg should compile as an array — output path extracts element 0
    const arrOut = outputInstr(arrProg)
    expect(arrOut.tag).toBe('Index')
    expect(arrOut.args[0]).toMatchObject({ kind: 'array_reg' })
    expect(arrProg.array_slot_sizes).toContain(3)
  })

  test('CSE: shared subexpression compiles once', () => {
    const shared = { op: 'add', args: [1, 2] } as ExprNode
    const prog = emitNumericProgram([{ op: 'mul', args: [shared, shared] }], [])

    // Only 1 Add instruction — the shared node is compiled once via WeakMap memo
    const adds = findAll(prog, 'Add').filter(i =>
      // Exclude the output-copy Add(val, 0)
      !(i.args.length === 2 && i.args[1]?.kind === 'const' && (i.args[1] as { val: number }).val === 0
        && i.args[0]?.kind === 'reg'),
    )
    expect(adds).toHaveLength(1)

    // Mul's two args reference the same reg slot
    const mul = findInstr(prog, 'Mul')!
    expect(mul).toBeDefined()
    expect(mul.args[0]).toEqual(mul.args[1])
  })

  // ── Bidirectional type inference (Phase 2) ──────────────────

  describe('literal resolution from register target context', () => {
    // Expected types flow from declared register slot types down through
    // register update expressions to numeric literals at the leaves.
    // Without this, `mod(int_reg, 8)` emits a float literal → float result,
    // then silently bitcasts into the int register on writeback (miscompile).

    test('literal in mod(int_reg, N) with int state register resolves to int', () => {
      const prog = emitNumericProgram(
        [],
        [{ op: 'mod', args: [
          { op: 'add', args: [{ op: 'reg', id: 0 }, 1] },
          8,
        ] }],
        [0],        // stateInit: int 0
        ['int'],    // stateRegTypes: int
      )
      const modI = findInstr(prog, 'Mod')!
      expect(modI).toBeDefined()
      expect(modI.result_type).toBe('int')
      // Literal 8 at position args[1] of Mod should carry int type, not float.
      expect(modI.args[1]).toMatchObject({ kind: 'const', val: 8, scalar_type: 'int' })

      // And the inner Add literal (1) should also resolve int from context.
      const addIs = findAll(prog, 'Add')
      const innerAdd = addIs.find(a =>
        a.args.length === 2
        && a.args[0]?.kind === 'state_reg'
        && a.args[1]?.kind === 'const'
        && (a.args[1] as { val: number }).val === 1,
      )
      expect(innerAdd).toBeDefined()
      expect(innerAdd!.result_type).toBe('int')
      expect(innerAdd!.args[1]).toMatchObject({ kind: 'const', val: 1, scalar_type: 'int' })
    })

    test('literal in comparison with float input stays float', () => {
      // gt(float_input, 0.5): the literal 0.5 must stay float — context peek
      // should read the first arg's type and pass it to the second.
      const prog = emitNumericProgram(
        [{ op: 'gt', args: [{ op: 'input', id: 0 }, 0.5] }],
        [],
      )
      const gt = findInstr(prog, 'Greater')!
      expect(gt).toBeDefined()
      expect(gt.args[1]).toMatchObject({ kind: 'const', val: 0.5, scalar_type: 'float' })
    })

    test('fractional literal in int context throws (lossy narrowing)', () => {
      // Writing 0.5 directly into an int register target is not representable;
      // the compiler must refuse rather than silently truncate.
      expect(() => {
        emitNumericProgram(
          [],
          [0.5 as ExprNode],
          [0],
          ['int'],
        )
      }).toThrow(/Lossy|narrow|int/)
    })
  })

  describe('typed-const operand (Phase 1)', () => {
    // A const ExprNode may carry an explicit `type` field (emitted by
    // specialize.ts for int/bool type_params). The emitter must honor it.

    test('typed-int const node produces int scalar_type operand', () => {
      const prog = emitNumericProgram(
        [{ op: 'add', args: [
          { op: 'reg', id: 0 },
          { op: 'const', val: 8, type: 'int' } as ExprNode,
        ] }],
        [],
        [0],
        ['int'],
      )
      const add = findInstr(prog, 'Add')!
      expect(add.args[1]).toMatchObject({ kind: 'const', val: 8, scalar_type: 'int' })
      expect(add.result_type).toBe('int')
    })

    test('typed-bool const node produces bool scalar_type operand', () => {
      const prog = emitNumericProgram(
        [{ op: 'select', args: [
          { op: 'const', val: 1, type: 'bool' } as ExprNode,
          10,
          20,
        ] }],
        [],
      )
      const sel = findInstr(prog, 'Select')!
      expect(sel.args[0]).toMatchObject({ kind: 'const', val: 1, scalar_type: 'bool' })
    })
  })

  describe('cast ops (Phase 3)', () => {
    // to_int / to_bool / to_float force their result type regardless of arg type.

    test('to_int on a float expression yields int result_type', () => {
      const prog = emitNumericProgram(
        [{ op: 'to_int', args: [{ op: 'add', args: [1.5, 2.5] }] }],
        [],
      )
      const cast = findInstr(prog, 'ToInt')!
      expect(cast.result_type).toBe('int')
    })

    test('to_float on an int register yields float result_type', () => {
      const prog = emitNumericProgram(
        [{ op: 'to_float', args: [{ op: 'reg', id: 0 }] }],
        [],
        [0],
        ['int'],
      )
      const cast = findInstr(prog, 'ToFloat')!
      expect(cast.result_type).toBe('float')
    })

    test('to_bool on an int constant yields bool result_type', () => {
      const prog = emitNumericProgram(
        [{ op: 'to_bool', args: [{ op: 'const', val: 3, type: 'int' } as ExprNode] }],
        [],
      )
      const cast = findInstr(prog, 'ToBool')!
      expect(cast.result_type).toBe('bool')
    })
  })

  describe('source_tag group_id tagging (Phase 6)', () => {
    test('tagged inner ops carry group_id; outer Select is ungated', () => {
      const tagged: ExprNode = {
        op: 'source_tag',
        source_instance: 'voice_0',
        gate_expr: true,
        expr: { op: 'add', args: [{ op: 'input', id: 0 }, 1] },
      }
      const prog = emitNumericProgram([tagged], [])
      const add = findInstr(prog, 'Add')!
      expect(add).toBeTruthy()
      expect(add.group_id).toMatch(/^voice_0#/)
      // The merge Select that sits outside the group must NOT carry group_id.
      const selects = findAll(prog, 'Select')
      expect(selects.length).toBe(1)
      expect(selects[0].group_id).toBeUndefined()
    })

    test('groups table records one entry per source_tag with a gate operand', () => {
      const tagged: ExprNode = {
        op: 'source_tag',
        source_instance: 'voice_0',
        gate_expr: true,
        expr: { op: 'add', args: [{ op: 'input', id: 0 }, 1] },
      }
      const prog = emitNumericProgram([tagged], [])
      expect(prog.groups).toBeTruthy()
      expect(prog.groups!.length).toBe(1)
      expect(prog.groups![0].id).toMatch(/^voice_0#/)
      expect(prog.groups![0].gate_operand).toBeTruthy()
    })

    test('on_skip: reg(N) compiles as a state_reg read on the merge path', () => {
      // Register-update wrapping: hold-on-skip semantic via on_skip = reg(0).
      const tagged: ExprNode = {
        op: 'source_tag',
        source_instance: 'voice_0',
        gate_expr: false,
        expr: 0,
        on_skip: { op: 'reg', id: 0 },
      }
      const prog = emitNumericProgram([tagged], [], [0], ['float'])
      const sel = findInstr(prog, 'Select')!
      // Third Select arg should be a state_reg read of slot 0.
      expect(sel.args[2]).toMatchObject({ kind: 'state_reg', slot: 0 })
    })

    test('absent groups field when no source_tag appears', () => {
      const prog = emitNumericProgram([{ op: 'add', args: [1, 2] }], [])
      expect(prog.groups).toBeUndefined()
    })
  })
})
