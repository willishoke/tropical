/**
 * interpret.test.ts — Tests for the ExprNode tree interpreter.
 *
 * Tests evalExpr on individual ops (O4 exact match), then tests
 * interpretSamples on small programs with register state.
 */

import { describe, test, expect } from 'bun:test'
import { evalExpr, interpretSamples, type InterpretEnv } from './interpret'
import type { ExprNode } from './expr'
import type { FlatExpressions } from './flatten'

// ─── helpers ─────────────────────────────────────────────────

function env(overrides: Partial<InterpretEnv> = {}): InterpretEnv {
  return {
    sampleRate: 44100,
    sampleIndex: 0,
    registers: [],
    inputs: [],
    params: new Map(),
    ...overrides,
  }
}

// ─── literals ────────────────────────────────────────────────

describe('evalExpr literals', () => {
  test('number', () => {
    expect(evalExpr(42, env())).toBe(42)
    expect(evalExpr(0, env())).toBe(0)
    expect(evalExpr(-3.14, env())).toBe(-3.14)
  })

  test('boolean', () => {
    expect(evalExpr(true, env())).toBe(true)
    expect(evalExpr(false, env())).toBe(false)
  })

  test('inline array', () => {
    expect(evalExpr([1, 2, 3], env())).toEqual([1, 2, 3])
  })

  test('nested array expressions', () => {
    const node: ExprNode = [{ op: 'add', args: [1, 2] }, 10]
    expect(evalExpr(node, env())).toEqual([3, 10])
  })
})

// ─── terminals ───────────────────────────────────────────────

describe('evalExpr terminals', () => {
  test('input', () => {
    expect(evalExpr({ op: 'input', id: 0 }, env({ inputs: [440] }))).toBe(440)
    expect(evalExpr({ op: 'input', id: 1 }, env({ inputs: [440, 880] }))).toBe(880)
    expect(evalExpr({ op: 'input', id: 5 }, env())).toBe(0) // missing → 0
  })

  test('reg', () => {
    expect(evalExpr({ op: 'reg', id: 0 }, env({ registers: [0.5] }))).toBe(0.5)
    expect(evalExpr({ op: 'reg', id: 1 }, env({ registers: [0, [1, 2, 3]] }))).toEqual([1, 2, 3])
  })

  test('sample_rate', () => {
    expect(evalExpr({ op: 'sample_rate' }, env())).toBe(44100)
    expect(evalExpr({ op: 'sample_rate' }, env({ sampleRate: 48000 }))).toBe(48000)
  })

  test('sample_index', () => {
    expect(evalExpr({ op: 'sample_index' }, env({ sampleIndex: 100 }))).toBe(100)
  })

  test('smoothed_param', () => {
    const params = new Map([[12345, 0.75]])
    expect(evalExpr({ op: 'smoothed_param', _ptr: true, _handle: 12345 }, env({ params }))).toBe(0.75)
  })

  test('trigger_param', () => {
    const params = new Map([[99, 1]])
    expect(evalExpr({ op: 'trigger_param', _ptr: true, _handle: 99 }, env({ params }))).toBe(1)
  })

  test('missing param returns 0', () => {
    expect(evalExpr({ op: 'smoothed_param', _ptr: true, _handle: 999 }, env())).toBe(0)
  })
})

// ─── binary arithmetic ──────────────────────────────────────

describe('evalExpr binary arithmetic', () => {
  test('add', () => {
    expect(evalExpr({ op: 'add', args: [3, 4] }, env())).toBe(7)
  })

  test('sub', () => {
    expect(evalExpr({ op: 'sub', args: [10, 3] }, env())).toBe(7)
  })

  test('mul', () => {
    expect(evalExpr({ op: 'mul', args: [3, 4] }, env())).toBe(12)
  })

  test('div', () => {
    expect(evalExpr({ op: 'div', args: [10, 4] }, env())).toBe(2.5)
    expect(evalExpr({ op: 'div', args: [1, 0] }, env())).toBe(0) // div by zero
  })

  test('mod', () => {
    expect(evalExpr({ op: 'mod', args: [7, 3] }, env())).toBe(1)
    expect(evalExpr({ op: 'mod', args: [1, 0] }, env())).toBe(0) // mod by zero
  })

  test('floor_div', () => {
    expect(evalExpr({ op: 'floor_div', args: [7, 2] }, env())).toBe(3)
    expect(evalExpr({ op: 'floorDiv', args: [7, 2] }, env())).toBe(3)
  })
})

// ─── binary comparison ──────────────────────────────────────

describe('evalExpr comparison', () => {
  test('lt / lte / gt / gte', () => {
    expect(evalExpr({ op: 'lt', args: [1, 2] }, env())).toBe(true)
    expect(evalExpr({ op: 'lt', args: [2, 2] }, env())).toBe(false)
    expect(evalExpr({ op: 'lte', args: [2, 2] }, env())).toBe(true)
    expect(evalExpr({ op: 'gt', args: [3, 2] }, env())).toBe(true)
    expect(evalExpr({ op: 'gte', args: [2, 2] }, env())).toBe(true)
  })

  test('eq / neq', () => {
    expect(evalExpr({ op: 'eq', args: [5, 5] }, env())).toBe(true)
    expect(evalExpr({ op: 'eq', args: [5, 6] }, env())).toBe(false)
    expect(evalExpr({ op: 'neq', args: [5, 6] }, env())).toBe(true)
  })
})

// ─── binary bitwise ─────────────────────────────────────────

describe('evalExpr bitwise', () => {
  test('bit_and / bit_or / bit_xor', () => {
    expect(evalExpr({ op: 'bit_and', args: [0xFF, 0x0F] }, env())).toBe(0x0F)
    expect(evalExpr({ op: 'bit_or', args: [0xF0, 0x0F] }, env())).toBe(0xFF)
    expect(evalExpr({ op: 'bit_xor', args: [0xFF, 0x0F] }, env())).toBe(0xF0)
  })

  test('lshift / rshift', () => {
    expect(evalExpr({ op: 'lshift', args: [1, 4] }, env())).toBe(16)
    expect(evalExpr({ op: 'rshift', args: [16, 4] }, env())).toBe(1)
  })

  test('camelCase aliases', () => {
    expect(evalExpr({ op: 'bitAnd', args: [0xFF, 0x0F] }, env())).toBe(0x0F)
    expect(evalExpr({ op: 'bitOr', args: [0xF0, 0x0F] }, env())).toBe(0xFF)
    expect(evalExpr({ op: 'bitXor', args: [0xFF, 0x0F] }, env())).toBe(0xF0)
  })
})

// ─── binary logical ─────────────────────────────────────────

describe('evalExpr logical', () => {
  test('and / or', () => {
    expect(evalExpr({ op: 'and', args: [true, true] }, env())).toBe(true)
    expect(evalExpr({ op: 'and', args: [true, false] }, env())).toBe(false)
    expect(evalExpr({ op: 'or', args: [false, true] }, env())).toBe(true)
    expect(evalExpr({ op: 'or', args: [false, false] }, env())).toBe(false)
  })

  test('numeric truthiness', () => {
    expect(evalExpr({ op: 'and', args: [1, 1] }, env())).toBe(true)
    expect(evalExpr({ op: 'and', args: [1, 0] }, env())).toBe(false)
  })
})

// ─── unary math ─────────────────────────────────────────────

describe('evalExpr unary', () => {
  test('neg / abs', () => {
    expect(evalExpr({ op: 'neg', args: [5] }, env())).toBe(-5)
    expect(evalExpr({ op: 'abs', args: [-3] }, env())).toBe(3)
  })

  test('sqrt', () => {
    expect(evalExpr({ op: 'sqrt', args: [9] }, env())).toBeCloseTo(3, 10)
  })

  test('rounding', () => {
    expect(evalExpr({ op: 'floor', args: [2.7] }, env())).toBe(2)
    expect(evalExpr({ op: 'ceil', args: [2.1] }, env())).toBe(3)
    expect(evalExpr({ op: 'round', args: [2.5] }, env())).toBe(3)
  })

  test('not / bit_not', () => {
    expect(evalExpr({ op: 'not', args: [true] }, env())).toBe(false)
    expect(evalExpr({ op: 'not', args: [0] }, env())).toBe(true)
    expect(evalExpr({ op: 'bit_not', args: [0] }, env())).toBe(-1) // ~0 === -1
  })
})

// ─── ternary ────────────────────────────────────────────────

describe('evalExpr ternary', () => {
  test('select: true branch', () => {
    expect(evalExpr({ op: 'select', args: [true, 10, 20] }, env())).toBe(10)
  })

  test('select: false branch', () => {
    expect(evalExpr({ op: 'select', args: [false, 10, 20] }, env())).toBe(20)
  })

  test('select: numeric condition', () => {
    expect(evalExpr({ op: 'select', args: [1, 10, 20] }, env())).toBe(10)
    expect(evalExpr({ op: 'select', args: [0, 10, 20] }, env())).toBe(20)
  })

  test('clamp', () => {
    expect(evalExpr({ op: 'clamp', args: [5, 0, 10] }, env())).toBe(5)
    expect(evalExpr({ op: 'clamp', args: [-1, 0, 10] }, env())).toBe(0)
    expect(evalExpr({ op: 'clamp', args: [15, 0, 10] }, env())).toBe(10)
  })
})

// ─── array ops ──────────────────────────────────────────────

describe('evalExpr array ops', () => {
  test('array construction', () => {
    expect(evalExpr({ op: 'array', items: [1, 2, 3] }, env())).toEqual([1, 2, 3])
  })

  test('array with computed items', () => {
    expect(evalExpr({ op: 'array', items: [{ op: 'add', args: [1, 2] }, 10] }, env())).toEqual([3, 10])
  })

  test('index', () => {
    expect(evalExpr({ op: 'index', args: [[10, 20, 30], 1] }, env())).toBe(20)
  })

  test('index out of bounds returns 0', () => {
    expect(evalExpr({ op: 'index', args: [[10, 20], 5] }, env())).toBe(0)
  })

  test('array_set', () => {
    expect(evalExpr({ op: 'array_set', args: [[10, 20, 30], 1, 99] }, env())).toEqual([10, 99, 30])
  })

  test('broadcast_to scalar', () => {
    expect(evalExpr({ op: 'broadcast_to', args: [5], shape: [3] }, env())).toEqual([5, 5, 5])
  })

  test('broadcast_to array', () => {
    expect(evalExpr({ op: 'broadcast_to', args: [[1, 2]], shape: [4] }, env())).toEqual([1, 2, 1, 2])
  })

  test('matrix', () => {
    expect(evalExpr({ op: 'matrix', rows: [[1, 2], [3, 4]] }, env())).toEqual([1, 2, 3, 4])
  })
})

// ─── elementwise broadcasting ───────────────────────────────

describe('evalExpr elementwise', () => {
  test('array + scalar', () => {
    expect(evalExpr({ op: 'add', args: [[1, 2, 3], 10] }, env())).toEqual([11, 12, 13])
  })

  test('scalar + array', () => {
    expect(evalExpr({ op: 'add', args: [10, [1, 2, 3]] }, env())).toEqual([11, 12, 13])
  })

  test('array + array', () => {
    expect(evalExpr({ op: 'add', args: [[1, 2, 3], [10, 20, 30]] }, env())).toEqual([11, 22, 33])
  })

  test('unary on array', () => {
    expect(evalExpr({ op: 'neg', args: [[1, -2, 3]] }, env())).toEqual([-1, 2, -3])
  })
})

// ─── unsupported op ─────────────────────────────────────────

describe('evalExpr error handling', () => {
  test('throws on unsupported op', () => {
    expect(() => evalExpr({ op: 'nonexistent', args: [] }, env())).toThrow('unsupported op')
  })
})

// ─── interpretSamples ───────────────────────────────────────

describe('interpretSamples', () => {
  test('constant output', () => {
    const flat: FlatExpressions = {
      outputExprs: [5],
      registerExprs: [],
      stateInit: [],
      registerTypes: [],
      registerNames: [],
      outputIndices: [0],
      sampleRate: 44100,
    }
    const buf = interpretSamples(flat, 4)
    // 5 / 20.0 = 0.25
    for (let i = 0; i < 4; i++) {
      expect(buf[i]).toBeCloseTo(0.25, 10)
    }
  })

  test('two outputs mixed', () => {
    const flat: FlatExpressions = {
      outputExprs: [3, 7],
      registerExprs: [],
      stateInit: [],
      registerTypes: [],
      registerNames: [],
      outputIndices: [0, 1],
      sampleRate: 44100,
    }
    const buf = interpretSamples(flat, 4)
    // (3 + 7) / 20.0 = 0.5
    for (let i = 0; i < 4; i++) {
      expect(buf[i]).toBeCloseTo(0.5, 10)
    }
  })

  test('phase accumulator (sample-delay semantics)', () => {
    // Register 0: phase, init = 0
    // Register update: mod(add(reg(0), div(440, sample_rate)), 1.0)
    // Output 0: mul(sub(mul(reg(0), 2), 1), 10)
    //   maps phase [0,1) → [-10, 10)
    // Audio = output / 20.0 → [-0.5, 0.5)
    const flat: FlatExpressions = {
      outputExprs: [
        { op: 'mul', args: [
          { op: 'sub', args: [
            { op: 'mul', args: [{ op: 'reg', id: 0 }, 2] },
            1,
          ]},
          10,
        ]},
      ],
      registerExprs: [
        { op: 'mod', args: [
          { op: 'add', args: [
            { op: 'reg', id: 0 },
            { op: 'div', args: [440, { op: 'sample_rate' }] },
          ]},
          1.0,
        ]},
      ],
      stateInit: [0],
      registerTypes: ['float'],
      registerNames: ['phase'],
      outputIndices: [0],
      sampleRate: 44100,
    }

    const buf = interpretSamples(flat, 4)

    // Sample 0: phase=0, output = (0*2 - 1)*10 = -10, audio = -10/20 = -0.5
    expect(buf[0]).toBeCloseTo(-0.5, 10)

    // Sample 1: phase = 440/44100, output = (phase*2 - 1)*10, audio = that/20
    const phase1 = 440 / 44100
    const expected1 = (phase1 * 2 - 1) * 10 / 20
    expect(buf[1]).toBeCloseTo(expected1, 10)

    // Samples should be monotonically increasing (before wrap)
    expect(buf[1]).toBeGreaterThan(buf[0])
    expect(buf[2]).toBeGreaterThan(buf[1])
    expect(buf[3]).toBeGreaterThan(buf[2])
  })

  test('integer counter with modular wrap', () => {
    // Register 0: counter, init = 0
    // Register update: mod(add(reg(0), 1), 8)
    // Output: reg(0)
    const flat: FlatExpressions = {
      outputExprs: [{ op: 'reg', id: 0 }],
      registerExprs: [
        { op: 'mod', args: [{ op: 'add', args: [{ op: 'reg', id: 0 }, 1] }, 8] },
      ],
      stateInit: [0],
      registerTypes: ['float'],
      registerNames: ['counter'],
      outputIndices: [0],
      sampleRate: 44100,
    }

    const buf = interpretSamples(flat, 10)
    const expected = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1]
    for (let i = 0; i < 10; i++) {
      expect(buf[i]).toBeCloseTo(expected[i] / 20.0, 10)
    }
  })

  test('select gate: high when counter > 4', () => {
    // Register 0: phase, init = 0, update = add(reg(0), 1)
    // Output: select(gt(reg(0), 4), 1, 0)
    const flat: FlatExpressions = {
      outputExprs: [
        { op: 'select', args: [
          { op: 'gt', args: [{ op: 'reg', id: 0 }, 4] },
          1, 0,
        ]},
      ],
      registerExprs: [
        { op: 'add', args: [{ op: 'reg', id: 0 }, 1] },
      ],
      stateInit: [0],
      registerTypes: ['float'],
      registerNames: ['phase'],
      outputIndices: [0],
      sampleRate: 44100,
    }

    const buf = interpretSamples(flat, 8)
    // phase: 0,1,2,3,4,5,6,7 → gate: 0,0,0,0,0,1,1,1
    const expected = [0, 0, 0, 0, 0, 1, 1, 1]
    for (let i = 0; i < 8; i++) {
      expect(buf[i]).toBeCloseTo(expected[i] / 20.0, 10)
    }
  })

  test('sample_index increments', () => {
    const flat: FlatExpressions = {
      outputExprs: [{ op: 'sample_index' }],
      registerExprs: [],
      stateInit: [],
      registerTypes: [],
      registerNames: [],
      outputIndices: [0],
      sampleRate: 44100,
    }

    const buf = interpretSamples(flat, 4)
    expect(buf[0]).toBeCloseTo(0 / 20.0, 10)
    expect(buf[1]).toBeCloseTo(1 / 20.0, 10)
    expect(buf[2]).toBeCloseTo(2 / 20.0, 10)
    expect(buf[3]).toBeCloseTo(3 / 20.0, 10)
  })

  test('source_tag is pass-through (Phase 2)', () => {
    // source_tag wraps an inner expression without altering its value.
    // Gate semantics are added in Phase 5; for now the tag is inert metadata.
    const tagged: ExprNode = {
      op: 'source_tag',
      source_instance: 'voice_0',
      gate_expr: true,
      expr: { op: 'add', args: [{ op: 'input', id: 0 }, 10] },
    }
    expect(evalExpr(tagged, env({ inputs: [5] }))).toBe(15)
    expect(evalExpr(tagged, env({ inputs: [-3] }))).toBe(7)
  })

  test('array register state preserved across samples', () => {
    // Register 0: array [1, 2, 3], update = identity (reg(0))
    // Output: index(reg(0), 1)  → should always be 2
    const flat: FlatExpressions = {
      outputExprs: [{ op: 'index', args: [{ op: 'reg', id: 0 }, 1] }],
      registerExprs: [{ op: 'reg', id: 0 }],
      stateInit: [[1, 2, 3]],
      registerTypes: ['float'],
      registerNames: ['arr'],
      outputIndices: [0],
      sampleRate: 44100,
    }

    const buf = interpretSamples(flat, 4)
    for (let i = 0; i < 4; i++) {
      expect(buf[i]).toBeCloseTo(2 / 20.0, 10)
    }
  })
})
