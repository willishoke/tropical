/**
 * emit_wasm.test.ts — unit tests for the plan_4 → WASM emitter.
 *
 * Each test constructs a minimal FlatPlan by hand, emits WASM, instantiates,
 * runs `process` for some samples, reads the output buffer and checks values.
 */

import { describe, test, expect } from 'bun:test'
import type { FlatPlan } from './flatten'
import type { NInstr, NOperand, ScalarType } from './emit_numeric'
import { emitWasm } from './emit_wasm'

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────

function plan(fields: {
  instructions: NInstr[]
  register_count: number
  array_slot_count?: number
  array_slot_sizes?: number[]
  output_targets: number[]
  outputs?: number[]
  register_targets?: number[]
  register_names?: string[]
  register_types?: ScalarType[]
  state_init?: (number | boolean)[]
}): FlatPlan {
  return {
    schema: 'tropical_plan_4',
    config: { sample_rate: 44100 },
    state_init: fields.state_init ?? [],
    register_names: fields.register_names ?? [],
    register_types: fields.register_types ?? [],
    array_slot_names: [],
    outputs: fields.outputs ?? fields.output_targets.map((_, i) => i),
    instructions: fields.instructions,
    register_count: fields.register_count,
    array_slot_count: fields.array_slot_count ?? 0,
    array_slot_sizes: fields.array_slot_sizes ?? [],
    output_targets: fields.output_targets,
    register_targets: fields.register_targets ?? [],
  }
}

const constF = (v: number): NOperand => ({ kind: 'const', val: v, scalar_type: 'float' })
const constI = (v: number): NOperand => ({ kind: 'const', val: v, scalar_type: 'int' })
const constB = (v: boolean): NOperand => ({ kind: 'const', val: v ? 1 : 0, scalar_type: 'bool' })
const reg = (slot: number, t: ScalarType = 'float'): NOperand => ({ kind: 'reg', slot, scalar_type: t })
const stateReg = (slot: number, t: ScalarType = 'float'): NOperand => ({ kind: 'state_reg', slot, scalar_type: t })
const arrReg = (slot: number): NOperand => ({ kind: 'array_reg', slot })
const tick: NOperand = { kind: 'tick', scalar_type: 'int' }

function mkInstr(tag: string, dst: number, args: NOperand[], resultType: ScalarType = 'float', loopCount = 1, strides: number[] = []): NInstr {
  return { tag, dst, args, loop_count: loopCount, strides, result_type: resultType }
}

async function runPlan(p: FlatPlan, blockSize = 8, blocks = 1, opts: { inputCount?: number; maxBlockSize?: number } = {}): Promise<Float64Array> {
  const { bytes, layout } = emitWasm(p, { maxBlockSize: blockSize * blocks, ...opts })
  const mod = await WebAssembly.compile(bytes)
  const instance = await WebAssembly.instantiate(mod, {})
  const memory = instance.exports.memory as WebAssembly.Memory
  const process_ = instance.exports.process as (blen: number, sidx: bigint) => void

  const total = blockSize * blocks
  const out = new Float64Array(total)
  for (let b = 0; b < blocks; b++) {
    process_(blockSize, BigInt(b * blockSize))
    const view = new Float64Array(memory.buffer, layout.outputOffset, blockSize)
    out.set(view, b * blockSize)
  }
  return out
}

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

describe('emit_wasm — trivial', () => {
  test('constant-add produces constant output scaled by 1/20', async () => {
    const p = plan({
      instructions: [mkInstr('Add', 0, [constF(10), constF(10)], 'float')],
      register_count: 1,
      output_targets: [0],
    })
    const out = await runPlan(p, 4, 1)
    for (let i = 0; i < out.length; i++) expect(out[i]).toBeCloseTo(1.0, 12) // (10+10)/20
  })
})

describe('emit_wasm — arithmetic float', () => {
  const cases: [string, number, number, number][] = [
    ['Add', 0.25, 0.125, 0.25 + 0.125],
    ['Sub', 3.5, 1.25, 3.5 - 1.25],
    ['Mul', 2, 0.5, 2 * 0.5],
    ['Div', 1, 4, 1 / 4],
    ['FloorDiv', 5.5, 2, Math.floor(5.5 / 2)],
    ['Mod', 5.5, 2, 5.5 - Math.floor(5.5 / 2) * 2],
  ]
  for (const [tag, a, b, expected] of cases) {
    test(`${tag}(${a}, ${b}) = ${expected}`, async () => {
      const p = plan({
        instructions: [mkInstr(tag, 0, [constF(a), constF(b)], 'float')],
        register_count: 1,
        output_targets: [0],
      })
      const out = await runPlan(p, 2, 1)
      expect(out[0]! * 20).toBeCloseTo(expected, 12)
    })
  }

  test('Div by zero returns 0', async () => {
    const p = plan({
      instructions: [mkInstr('Div', 0, [constF(7), constF(0)], 'float')],
      register_count: 1,
      output_targets: [0],
    })
    const out = await runPlan(p, 2, 1)
    expect(out[0]).toBe(0)
  })

  test('Mod by zero returns 0', async () => {
    const p = plan({
      instructions: [mkInstr('Mod', 0, [constF(7), constF(0)], 'float')],
      register_count: 1,
      output_targets: [0],
    })
    const out = await runPlan(p, 2, 1)
    expect(out[0]).toBe(0)
  })
})

describe('emit_wasm — arithmetic int', () => {
  // int results get stored as i64 in temps[]. Output mix reads them as f64
  // which will reinterpret the bits. To verify int arithmetic we route through
  // ToFloat before the mix.
  test('int Add then ToFloat', async () => {
    const p = plan({
      instructions: [
        mkInstr('Add', 0, [constI(3), constI(4)], 'int'),
        mkInstr('ToFloat', 1, [reg(0, 'int')], 'float'),
      ],
      register_count: 2,
      output_targets: [1],
    })
    const out = await runPlan(p, 2, 1)
    expect(out[0]! * 20).toBeCloseTo(7, 12)
  })
})

describe('emit_wasm — unary float', () => {
  test('Sqrt(16) = 4', async () => {
    const p = plan({
      instructions: [mkInstr('Sqrt', 0, [constF(16)], 'float')],
      register_count: 1, output_targets: [0],
    })
    const out = await runPlan(p, 2)
    expect(out[0]! * 20).toBeCloseTo(4, 12)
  })
  test('Abs(-7.5) = 7.5', async () => {
    const p = plan({
      instructions: [mkInstr('Abs', 0, [constF(-7.5)], 'float')],
      register_count: 1, output_targets: [0],
    })
    const out = await runPlan(p, 2)
    expect(out[0]! * 20).toBeCloseTo(7.5, 12)
  })
  test('Neg(3) = -3', async () => {
    const p = plan({
      instructions: [mkInstr('Neg', 0, [constF(3)], 'float')],
      register_count: 1, output_targets: [0],
    })
    const out = await runPlan(p, 2)
    expect(out[0]! * 20).toBeCloseTo(-3, 12)
  })
  test('Floor(3.7) = 3', async () => {
    const p = plan({
      instructions: [mkInstr('Floor', 0, [constF(3.7)], 'float')],
      register_count: 1, output_targets: [0],
    })
    const out = await runPlan(p, 2)
    expect(out[0]! * 20).toBeCloseTo(3, 12)
  })
  test('Ceil(3.1) = 4', async () => {
    const p = plan({
      instructions: [mkInstr('Ceil', 0, [constF(3.1)], 'float')],
      register_count: 1, output_targets: [0],
    })
    const out = await runPlan(p, 2)
    expect(out[0]! * 20).toBeCloseTo(4, 12)
  })
  test('Round(0.5) = 0 (banker rounding / WASM nearest)', async () => {
    const p = plan({
      instructions: [mkInstr('Round', 0, [constF(0.5)], 'float')],
      register_count: 1, output_targets: [0],
    })
    const out = await runPlan(p, 2)
    expect(out[0]).toBe(0) // round-to-nearest-even
  })
})

describe('emit_wasm — Ldexp / FloatExponent', () => {
  test('Ldexp(3, 4) = 3 * 16 = 48', async () => {
    const p = plan({
      instructions: [mkInstr('Ldexp', 0, [constF(3), constF(4)], 'float')],
      register_count: 1, output_targets: [0],
    })
    const out = await runPlan(p, 2)
    expect(out[0]! * 20).toBeCloseTo(48, 12)
  })
  test('FloatExponent(1024) = 10', async () => {
    const p = plan({
      instructions: [mkInstr('FloatExponent', 0, [constF(1024)], 'float')],
      register_count: 1, output_targets: [0],
    })
    const out = await runPlan(p, 2)
    expect(out[0]! * 20).toBeCloseTo(10, 12)
  })
})

describe('emit_wasm — comparisons + select', () => {
  test('Select(true, 2, 3) = 2', async () => {
    const p = plan({
      instructions: [
        mkInstr('Less', 0, [constF(1), constF(2)], 'bool'),
        mkInstr('Select', 1, [reg(0, 'bool'), constF(2), constF(3)], 'float'),
      ],
      register_count: 2, output_targets: [1],
    })
    const out = await runPlan(p, 2)
    expect(out[0]! * 20).toBeCloseTo(2, 12)
  })
  test('Select(false, 2, 3) = 3', async () => {
    const p = plan({
      instructions: [
        mkInstr('Less', 0, [constF(2), constF(1)], 'bool'),
        mkInstr('Select', 1, [reg(0, 'bool'), constF(2), constF(3)], 'float'),
      ],
      register_count: 2, output_targets: [1],
    })
    const out = await runPlan(p, 2)
    expect(out[0]! * 20).toBeCloseTo(3, 12)
  })
})

describe('emit_wasm — clamp', () => {
  test('Clamp float', async () => {
    const cases: [number, number][] = [[0.5, 0.5], [5, 1], [-5, -1]]
    for (const [input, expected] of cases) {
      const p = plan({
        instructions: [mkInstr('Clamp', 0, [constF(input), constF(-1), constF(1)], 'float')],
        register_count: 1, output_targets: [0],
      })
      const out = await runPlan(p, 2)
      expect(out[0]! * 20).toBeCloseTo(expected, 12)
    }
  })
})

describe('emit_wasm — bitwise', () => {
  test('BitAnd', async () => {
    const p = plan({
      instructions: [
        mkInstr('BitAnd', 0, [constI(0b1100), constI(0b1010)], 'int'),
        mkInstr('ToFloat', 1, [reg(0, 'int')], 'float'),
      ],
      register_count: 2, output_targets: [1],
    })
    const out = await runPlan(p, 2)
    expect(out[0]! * 20).toBeCloseTo(0b1000, 12)
  })
  test('LShift', async () => {
    const p = plan({
      instructions: [
        mkInstr('LShift', 0, [constI(1), constI(5)], 'int'),
        mkInstr('ToFloat', 1, [reg(0, 'int')], 'float'),
      ],
      register_count: 2, output_targets: [1],
    })
    const out = await runPlan(p, 2)
    expect(out[0]! * 20).toBeCloseTo(32, 12)
  })
  test('BitNot', async () => {
    // ~5 = -6 under two's complement
    const p = plan({
      instructions: [
        mkInstr('BitNot', 0, [constI(5)], 'int'),
        mkInstr('ToFloat', 1, [reg(0, 'int')], 'float'),
      ],
      register_count: 2, output_targets: [1],
    })
    const out = await runPlan(p, 2)
    expect(out[0]! * 20).toBeCloseTo(-6, 12)
  })
})

describe('emit_wasm — state register (counter)', () => {
  test('counter increments on each sample', async () => {
    // registers[0] (float) starts at 0, each sample: new = reg + 1, write to temps[1], writeback
    const p = plan({
      instructions: [
        mkInstr('Add', 0, [stateReg(0, 'float'), constF(1)], 'float'),
        mkInstr('ToFloat', 1, [reg(0, 'float')], 'float'),
      ],
      register_count: 2,
      output_targets: [1],
      register_targets: [0],  // registers[0] ← temps[0]
      register_names: ['counter'],
      register_types: ['float'],
      state_init: [0],
    })
    const out = await runPlan(p, 8)
    for (let i = 0; i < 8; i++) expect(out[i]! * 20).toBeCloseTo(i + 1, 12)
  })
})

describe('emit_wasm — tick operand', () => {
  test('sample index (tick) → output', async () => {
    // temps[0] = tick (as int), then temps[1] = ToFloat(temps[0])
    const p = plan({
      instructions: [
        mkInstr('Add', 0, [tick, constI(0)], 'int'),
        mkInstr('ToFloat', 1, [reg(0, 'int')], 'float'),
      ],
      register_count: 2, output_targets: [1],
    })
    const out = await runPlan(p, 5)
    for (let i = 0; i < 5; i++) expect(out[i]! * 20).toBeCloseTo(i, 12)
  })
})

describe('emit_wasm — Pack + Index', () => {
  test('pack 4 constants then index 2', async () => {
    // arr[0] = [10, 20, 30, 40]; temps[0] = arr[0][2]; output = temps[0]
    const p = plan({
      instructions: [
        mkInstr('Pack', 0, [constF(10), constF(20), constF(30), constF(40)], 'float'),
        mkInstr('Index', 0, [arrReg(0), constI(2)], 'float'),
      ],
      register_count: 1,
      array_slot_count: 1,
      array_slot_sizes: [4],
      output_targets: [0],
    })
    const out = await runPlan(p, 2)
    expect(out[0]! * 20).toBeCloseTo(30, 12)
  })
  test('Index out-of-range returns 0', async () => {
    const p = plan({
      instructions: [
        mkInstr('Pack', 0, [constF(10), constF(20)], 'float'),
        mkInstr('Index', 0, [arrReg(0), constI(5)], 'float'),
      ],
      register_count: 1, array_slot_count: 1, array_slot_sizes: [2],
      output_targets: [0],
    })
    const out = await runPlan(p, 2)
    expect(out[0]).toBe(0)
  })
})

describe('emit_wasm — elementwise', () => {
  test('elementwise Mul across arrays', async () => {
    // arr[0] = [1,2,3,4]; arr[1] = [10,10,10,10]; arr[2] = arr[0] * arr[1]; index 2
    const p = plan({
      instructions: [
        mkInstr('Pack', 0, [constF(1), constF(2), constF(3), constF(4)], 'float'),
        mkInstr('Pack', 1, [constF(10), constF(10), constF(10), constF(10)], 'float'),
        mkInstr('Mul', 2, [arrReg(0), arrReg(1)], 'float', 4, [1, 1]),
        mkInstr('Index', 0, [arrReg(2), constI(2)], 'float'),
      ],
      register_count: 1,
      array_slot_count: 3,
      array_slot_sizes: [4, 4, 4],
      output_targets: [0],
    })
    const out = await runPlan(p, 2)
    expect(out[0]! * 20).toBeCloseTo(30, 12)
  })
})
