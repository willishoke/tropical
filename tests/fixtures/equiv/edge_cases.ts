/**
 * Edge-case fixtures for tests/equiv/jit_vs_interp_stdlib.test.ts (Phase D P0.1).
 *
 * Each fixture is a hand-built program that stresses an axis the stdlib
 * corpus doesn't cover well: division by zero, sqrt of negatives, denormal
 * propagation, conditional branches with NaN inputs, etc. The fixtures
 * are programs (not patches) so they can be loaded as types and instantiated.
 *
 * Documents expected behavior at the boundary of the C++ FTZ/DAZ flag
 * (flush-to-zero / denormals-are-zero on x86). Where the JIT and pure-TS
 * interpreter diverge by design (e.g. `0/0` is NaN in interpret, 0 in JIT
 * because of div-guard), the fixture's `tolerance` is loosened or the
 * fixture is excluded from strict comparison.
 */

import type { ProgramNode } from '../../program.js'

export interface EdgeFixture {
  /** Display name for the test. */
  name: string
  /** Program node to load. */
  program: ProgramNode
  /** Inputs for the test instance — keyed by `${input_name}`. */
  inputs?: Record<string, import('../../expr.js').ExprNode>
  /** Output port name to read for the assertion (default: first output). */
  output?: string
  /** Tolerance for `toBeCloseTo` digit-of-precision check. Default 8. */
  tolerance?: number
  /** If set, sample[i] must be `Number.isFinite(...)` — catches NaN
   *  propagation regressions even when both sides agree on a NaN. */
  expectAllFinite?: boolean
  /** If true, skip strict numeric agreement and only check finiteness. */
  finitenessOnly?: boolean
}

/** div(x, 0) — both interpreter and JIT guard divide-by-zero, returning 0
 *  rather than +Infinity/NaN. This fixture pins that contract. */
const divByZero: EdgeFixture = {
  name: 'div_by_zero',
  program: {
    op: 'program',
    name: 'DivByZero',
    ports: { inputs: [{ name: 'x', default: 1 }], outputs: ['out'] },
    body: { op: 'block',
      assigns: [{ op: 'outputAssign', name: 'out',
        expr: { op: 'div', args: [{ op: 'input', name: 'x' }, 0] } }],
    },
  },
  expectAllFinite: true,
  tolerance: 10,
}

/** sqrt of a guaranteed-negative value. JS Math.sqrt(-1) is NaN; the JIT
 *  also produces NaN. Propagation is consistent on both sides. We assert
 *  finiteness fails for at least one sample (the negative branch) — a
 *  pure equivalence check would need NaN-aware comparison. */
const sqrtNegative: EdgeFixture = {
  name: 'sqrt_negative',
  program: {
    op: 'program',
    name: 'SqrtNegative',
    ports: { inputs: [{ name: 'x', default: -1 }], outputs: ['out'] },
    body: { op: 'block',
      assigns: [{ op: 'outputAssign', name: 'out',
        expr: { op: 'sqrt', args: [{ op: 'input', name: 'x' }] } }],
    },
  },
  finitenessOnly: true,  // both sides produce NaN; agreement on NaN is
                         // not testable via `toBeCloseTo`.
}

/** 1e-310 * 1e-310 produces a denormal on x86 without FTZ; with FTZ it
 *  flushes to zero. The interpreter (pure JS) never flushes; the JIT may
 *  or may not depending on the C++ build. The product is also below the
 *  smallest-positive normal (~2.2e-308), so this hits the denormal range
 *  before flushing. */
const denormalMul: EdgeFixture = {
  name: 'denormal_multiplication',
  program: {
    op: 'program',
    name: 'DenormalMul',
    ports: { inputs: [], outputs: ['out'] },
    body: { op: 'block',
      assigns: [{ op: 'outputAssign', name: 'out',
        expr: { op: 'mul', args: [1e-310, 1e-310] } }],
    },
  },
  expectAllFinite: true,
  finitenessOnly: true,  // tolerance for denormal divergence is system-dependent.
}

/** A reg that accumulates a small per-sample increment. Over many
 *  samples the running sum stays in healthy float range. Pin floor/abs
 *  on a sub-normal input to confirm those pass through cleanly. */
const stableAccumulator: EdgeFixture = {
  name: 'stable_accumulator',
  program: {
    op: 'program',
    name: 'StableAccum',
    ports: { inputs: [{ name: 'inc', default: 1e-10 }], outputs: ['out'] },
    body: { op: 'block',
      decls: [{ op: 'regDecl', name: 's', init: 0 }],
      assigns: [
        { op: 'outputAssign', name: 'out', expr: { op: 'reg', name: 's' } },
        { op: 'nextUpdate', target: { kind: 'reg', name: 's' },
          expr: { op: 'add', args: [{ op: 'reg', name: 's' }, { op: 'input', name: 'inc' }] } },
      ],
    },
  },
  expectAllFinite: true,
  tolerance: 12,
}

/** Conditional select where one branch produces a special value. JIT
 *  evaluates both branches eagerly (no short-circuit). */
const selectWithSpecials: EdgeFixture = {
  name: 'select_with_special_values',
  program: {
    op: 'program',
    name: 'SelectSpecial',
    ports: { inputs: [{ name: 'gate', default: 1 }], outputs: ['out'] },
    body: { op: 'block',
      assigns: [{ op: 'outputAssign', name: 'out',
        // gate ? 0.5 : (1.0 / 0.0)
        expr: { op: 'select', args: [
          { op: 'input', name: 'gate' },
          0.5,
          { op: 'div', args: [1.0, 0.0] },
        ] },
      }],
    },
  },
  // With gate=1, output should be 0.5 sample-for-sample.
  expectAllFinite: true,
  tolerance: 12,
}

/** Wholesale array-reg writeback. Produces an inline-array expression
 *  (here `generate(N, i => i + sampleIndex())`) and assigns it to a
 *  reg whose init is `zeros(N)` — i.e. `next arr = <expr>` rather than
 *  the in-place `next arr = arraySet(arr, idx, x)` pattern.
 *
 *  Regression test for a JIT bug where the writeback path emitted no
 *  copy from the expression's fresh array slot into the reg's
 *  persistent storage slot, leaving the reg permanently zeros (only
 *  arraySet, used by Delay, worked because it writes in-place to the
 *  persistent slot). The interpreter handled the case correctly all
 *  along, so the bug surfaced as a JIT/interp divergence.
 *
 *  After 4 samples (buffer 256, sampleIndex per buffer = 0,256,512,...):
 *  arr[2] on each subsequent sample reads the previous sample's value
 *  of (sampleIndex_of_previous_sample + 2). With the bug, the JIT
 *  always reads 0 from the persistent slot; the interpreter reads the
 *  scan output. */
const arrayRegWholesaleWriteback: EdgeFixture = {
  name: 'array_reg_wholesale_writeback',
  program: {
    op: 'program',
    name: 'ArrayRegWholesaleWriteback',
    ports: { inputs: [], outputs: ['out'] },
    body: { op: 'block',
      decls: [
        { op: 'regDecl', name: 'arr', init: { zeros: 4 } as any },
      ],
      assigns: [
        { op: 'outputAssign', name: 'out',
          expr: { op: 'index', args: [{ op: 'reg', name: 'arr' }, 3] } },
        { op: 'nextUpdate', target: { kind: 'reg', name: 'arr' },
          expr: { op: 'generate', count: 4, var: 'i',
            body: { op: 'add', args: [
              { op: 'sampleIndex' },
              { op: 'binding', name: 'i' },
            ] } } as any },
      ],
    },
  },
  expectAllFinite: true,
  tolerance: 12,
}

/** Bool-expected propagation through arithmetic. compileBinary's
 *  `secondExpected = l.scalarType` for comparisons used to forward the
 *  left arg's `bool` type to the right arg, and then non-comparison /
 *  non-bitwise binaries forwarded that `bool` straight to their own
 *  args. So in `eq(bool_value, x + 0.5)`, the `0.5` literal would try
 *  to narrow to bool and throw — even though arithmetic on float/int
 *  args can never produce bool.
 *
 *  Reproduces in cross_fm_evolved via `Mul(Greater(_,_), LessEq(_,_))`
 *  inside NoiseLFSR's tick computation, which becomes a Select cond
 *  after inlining and propagates bool through the comparison's RHS.
 *
 *  This fixture isolates the pattern: an `eq` whose left arg returns
 *  bool, whose right arg is an arithmetic expression with a non-0/1
 *  float literal. */
const boolPropagationThroughArith: EdgeFixture = {
  name: 'bool_propagation_through_arith',
  program: {
    op: 'program',
    name: 'BoolPropagation',
    ports: { inputs: [{ name: 'x', default: 0.3 }], outputs: ['out'] },
    body: { op: 'block',
      assigns: [{ op: 'outputAssign', name: 'out',
        // eq( gt(x, 0), x * 0.5 + 0.25 )
        expr: {
          op: 'eq',
          args: [
            { op: 'gt', args: [{ op: 'input', name: 'x' }, 0] },
            { op: 'add', args: [
              { op: 'mul', args: [{ op: 'input', name: 'x' }, 0.5] },
              0.25,
            ] },
          ],
        },
      }],
    },
  },
  expectAllFinite: true,
  tolerance: 12,
}

// ─────────────────────────────────────────────────────────────
// Phase B — wholesale-array writebacks beyond `generate`
// (TDD plan: ~/.claude/plans/we-re-doing-a-tdd-eager-waffle.md)
//
// Same shape as `arrayRegWholesaleWriteback`: `next arr = <expr>` where
// `<expr>` is built from a different combinator each fixture. Each one
// exercises an independent unrolling path in `array_lower.ts` followed
// by the same JIT writeback fix from 66ae9f9.
// ─────────────────────────────────────────────────────────────

/** (D) `next arr = select(cond, arr1, arr2)` — array-typed Select.
 *  cond is `sampleIndex < 4`; arr1=[1,2,3,4], arr2=[10,20,30,40].
 *  Sample 0..3 reads arr1; sample 4+ reads arr2. Output is index 2,
 *  so sample 0 → 3, sample 4 → 30 (post-/20: 0.15, 1.5).
 *
 *  Note: select on whole arrays may not be supported by the lowering
 *  pipeline today. If `applyFlatPlan` throws, that's a pinned
 *  type-error (per the plan); record here and downgrade to a
 *  direct-error fixture if needed. */
const arraySelectWriteback: EdgeFixture = {
  name: 'array_reg_select_writeback',
  program: {
    op: 'program',
    name: 'ArrayRegSelectWriteback',
    ports: { inputs: [], outputs: ['out'] },
    body: { op: 'block',
      decls: [
        { op: 'regDecl', name: 'arr', init: [0, 0, 0, 0] as any },
      ],
      assigns: [
        { op: 'outputAssign', name: 'out',
          expr: { op: 'index', args: [{ op: 'reg', name: 'arr' }, 2] } },
        { op: 'nextUpdate', target: { kind: 'reg', name: 'arr' },
          expr: { op: 'select', args: [
            { op: 'lt', args: [{ op: 'sampleIndex' }, 4] },
            [1, 2, 3, 4],
            [10, 20, 30, 40],
          ]} as any },
      ],
    },
  },
  expectAllFinite: true,
  tolerance: 12,
}

/** (D) `next arr = zipWith(arr_a, arr_b, (x, y) => x + y)` — exercises
 *  array_lower's zipWith unroll in nextUpdate position. The plan calls
 *  this `map2`, but tropical's `map2` is single-array; the right
 *  primitive for two arrays is `zipWith`. */
const arrayZipWithWriteback: EdgeFixture = {
  name: 'array_reg_zipwith_writeback',
  program: {
    op: 'program',
    name: 'ArrayRegZipWithWriteback',
    ports: { inputs: [], outputs: ['out'] },
    body: { op: 'block',
      decls: [
        { op: 'regDecl', name: 'arr', init: [0, 0, 0, 0] as any },
      ],
      assigns: [
        { op: 'outputAssign', name: 'out',
          expr: { op: 'index', args: [{ op: 'reg', name: 'arr' }, 2] } },
        { op: 'nextUpdate', target: { kind: 'reg', name: 'arr' },
          expr: { op: 'zipWith',
            a: [1, 2, 3, 4],
            b: [10, 20, 30, 40],
            x_var: 'x', y_var: 'y',
            body: { op: 'add', args: [
              { op: 'binding', name: 'x' },
              { op: 'binding', name: 'y' },
            ]},
          } as any },
      ],
    },
  },
  expectAllFinite: true,
  tolerance: 12,
}

/** (D) `next arr = scan(over, init, f)` — scan emits intermediate
 *  accumulators. With over=[1,2,3,4], init=0, f=acc+elem: result is
 *  [1, 3, 6, 10] (running sums). Reads index 3 → 10 (post-/20: 0.5). */
const arrayScanWriteback: EdgeFixture = {
  name: 'array_reg_scan_writeback',
  program: {
    op: 'program',
    name: 'ArrayRegScanWriteback',
    ports: { inputs: [], outputs: ['out'] },
    body: { op: 'block',
      decls: [
        { op: 'regDecl', name: 'arr', init: [0, 0, 0, 0] as any },
      ],
      assigns: [
        { op: 'outputAssign', name: 'out',
          expr: { op: 'index', args: [{ op: 'reg', name: 'arr' }, 3] } },
        { op: 'nextUpdate', target: { kind: 'reg', name: 'arr' },
          expr: { op: 'scan',
            over: [1, 2, 3, 4],
            init: 0,
            acc_var: 'acc', elem_var: 'x',
            body: { op: 'add', args: [
              { op: 'binding', name: 'acc' },
              { op: 'binding', name: 'x' },
            ]},
          } as any },
      ],
    },
  },
  expectAllFinite: true,
  tolerance: 12,
}

/** (D) `next arr = let { tmp = generate(N, ...) } in tmp` — the
 *  let-binding indirection between the producer combinator and the
 *  writeback exercises the let-elimination path. */
const arrayLetGenerateWriteback: EdgeFixture = {
  name: 'array_reg_let_generate_writeback',
  program: {
    op: 'program',
    name: 'ArrayRegLetGenerateWriteback',
    ports: { inputs: [], outputs: ['out'] },
    body: { op: 'block',
      decls: [
        { op: 'regDecl', name: 'arr', init: [0, 0, 0, 0] as any },
      ],
      assigns: [
        { op: 'outputAssign', name: 'out',
          expr: { op: 'index', args: [{ op: 'reg', name: 'arr' }, 2] } },
        { op: 'nextUpdate', target: { kind: 'reg', name: 'arr' },
          expr: { op: 'let',
            bind: {
              tmp: { op: 'generate', count: 4, var: 'i',
                body: { op: 'add', args: [
                  { op: 'sampleIndex' },
                  { op: 'binding', name: 'i' },
                ]},
              },
            },
            in: { op: 'binding', name: 'tmp' },
          } as any },
      ],
    },
  },
  expectAllFinite: true,
  tolerance: 12,
}

// ─────────────────────────────────────────────────────────────
// Phase C — known expected-type sites (TDD plan §Phase C)
// Targeted regressions; the universal property "expected= is a hint
// that doesn't change ⟦e⟧" is not class-tested here.
// ─────────────────────────────────────────────────────────────

/** (R) `clamp(bool_expr, 0, 1)` — bool result type would push down
 *  into the lo / hi literals. compileTernary handles both Select
 *  and Clamp; 52bd3a8 stripped bool for Select arms but the
 *  fix-comment didn't mention Clamp. The exact analog of that fix's
 *  repro for Select. Should compile without throwing
 *  "literal cannot narrow to bool". */
const clampBoolArms: EdgeFixture = {
  name: 'clamp_bool_arms',
  program: {
    op: 'program',
    name: 'ClampBoolArms',
    ports: { inputs: [{ name: 'x', default: 0.5 }], outputs: ['out'] },
    body: { op: 'block',
      assigns: [{ op: 'outputAssign', name: 'out',
        // clamp( gt(x, 0), 0, 1 ) — a bool input clamped to [0,1].
        // 0 and 1 are int literals; if the bool expectation propagates
        // to them they'd narrow to bool (legal) — but extending to a
        // hi=2 case forces the issue. Use [0, 2] to make the bug
        // visible: 2 cannot narrow to bool.
        expr: { op: 'clamp', args: [
          { op: 'gt', args: [{ op: 'input', name: 'x' }, 0] },
          0, 2,
        ]},
      }],
    },
  },
  expectAllFinite: true,
  tolerance: 12,
}

/** (R) `select(cond, gt(a,b), x*0.5)` — mixed bool/float arms. The
 *  cond's `expected='bool'` could leak into the float arm where 0.5
 *  literal would try to narrow to bool. Asserts the propagation
 *  strips bool before reaching the float arm. */
const selectMixedArms: EdgeFixture = {
  name: 'select_mixed_arms',
  program: {
    op: 'program',
    name: 'SelectMixedArms',
    ports: { inputs: [{ name: 'x', default: 0.3 }], outputs: ['out'] },
    body: { op: 'block',
      assigns: [{ op: 'outputAssign', name: 'out',
        // select( gt(x, 0), gt(x, 0.5), x * 0.5 + 0.25 )
        expr: { op: 'select', args: [
          { op: 'gt', args: [{ op: 'input', name: 'x' }, 0] },
          { op: 'gt', args: [{ op: 'input', name: 'x' }, 0.5] },
          { op: 'add', args: [
            { op: 'mul', args: [{ op: 'input', name: 'x' }, 0.5] },
            0.25,
          ]},
        ]},
      }],
    },
  },
  expectAllFinite: true,
  tolerance: 12,
}

/** (R) `eq(neq(a,b), neq(c,d))` — chained bool comparisons. Here
 *  expected='bool' *should* propagate (both args are themselves
 *  bools). Failure mode: over-aggressive bool-stripping pushes the
 *  inner neq's args to compile as float and miss bool-specific
 *  codegen. Pin that the chained bool→bool path still works. */
const eqOfNeqs: EdgeFixture = {
  name: 'eq_of_neqs',
  program: {
    op: 'program',
    name: 'EqOfNeqs',
    ports: { inputs: [{ name: 'x', default: 0.3 }], outputs: ['out'] },
    body: { op: 'block',
      assigns: [{ op: 'outputAssign', name: 'out',
        // eq( neq(x, 0), neq(x, 0.5) )
        expr: { op: 'eq', args: [
          { op: 'neq', args: [{ op: 'input', name: 'x' }, 0] },
          { op: 'neq', args: [{ op: 'input', name: 'x' }, 0.5] },
        ]},
      }],
    },
  },
  expectAllFinite: true,
  tolerance: 12,
}

/** (R) `expected='int'` through `div`. Symmetric to the bool leak:
 *  if an int hint propagates into a div whose result needs to be
 *  float, the result silently truncates. Here `add(int_reg, div(x, 2))`
 *  pushes int into div; div's float-typed result must NOT narrow to
 *  int. Pin the float result. */
const intHintThroughDiv: EdgeFixture = {
  name: 'int_hint_through_div',
  program: {
    op: 'program',
    name: 'IntHintThroughDiv',
    ports: { inputs: [{ name: 'x', default: 1 }], outputs: ['out'] },
    body: { op: 'block',
      decls: [
        // int-typed reg holding a constant int. Forces the add's left
        // arg expected to be int; the right (div) inherits the hint.
        { op: 'regDecl', name: 'k', init: 0, type: 'int' as any },
      ],
      assigns: [
        { op: 'nextUpdate', target: { kind: 'reg', name: 'k' },
          expr: { op: 'reg', name: 'k' } },
        { op: 'outputAssign', name: 'out',
          expr: { op: 'add', args: [
            { op: 'reg', name: 'k' },
            { op: 'div', args: [{ op: 'input', name: 'x' }, 2] },
          ]},
        },
      ],
    },
  },
  expectAllFinite: true,
  tolerance: 12,
}

// ─────────────────────────────────────────────────────────────
// Phase D — mutual register update (TDD plan §Phase D)
//
// Two-pass writeback isolation: each next-state must be a function of
// the *current-state* of all regs, not of intermediate post-update
// values. The 66ae9f9 fix-comment claims this invariant for array regs;
// these fixtures exercise it directly. Sample-by-sample exact pinning
// — failure mode is off-by-one-sample, not numeric drift.
// ─────────────────────────────────────────────────────────────

/** (D) Scalar mutual update — `next a = b + 1; next b = a + 1`. Both
 *  init 0. Read-before-write isolation: at sample 1 both read the
 *  previous (init=0) value of the other, so both become 1. At sample 2
 *  both read each other's value 1, both become 2. Sequence: a=b=t.
 *
 *  Bug shape: if the JIT writes `a = b + 1` first, then evaluates
 *  `b = a + 1` it sees the just-updated `a`. Pinned exactly to catch
 *  the off-by-one-sample drift this would produce.
 *
 *  Output is `a + b`. Sample 0: 0 + 0 = 0. Sample 1: 1 + 1 = 2.
 *  Sample 2: 2 + 2 = 4. After /20 mix scaling: 0, 0.1, 0.2, 0.3. */
const scalarMutualReg: EdgeFixture = {
  name: 'scalar_mutual_reg',
  program: {
    op: 'program',
    name: 'ScalarMutualReg',
    ports: { inputs: [], outputs: ['out'] },
    body: { op: 'block',
      decls: [
        { op: 'regDecl', name: 'a', init: 0 },
        { op: 'regDecl', name: 'b', init: 0 },
      ],
      assigns: [
        { op: 'outputAssign', name: 'out',
          expr: { op: 'add', args: [
            { op: 'reg', name: 'a' }, { op: 'reg', name: 'b' },
          ]}},
        { op: 'nextUpdate', target: { kind: 'reg', name: 'a' },
          expr: { op: 'add', args: [{ op: 'reg', name: 'b' }, 1] } },
        { op: 'nextUpdate', target: { kind: 'reg', name: 'b' },
          expr: { op: 'add', args: [{ op: 'reg', name: 'a' }, 1] } },
      ],
    },
  },
  expectAllFinite: true,
  tolerance: 12,
}

/** (D) Array mutual update — same temporal-isolation property at
 *  array type. `next arr1 = generate(N, i => arr2[i] + 1)`,
 *  `next arr2 = generate(N, i => arr1[i] + 1)`. Both init zeros(2).
 *
 *  Distinct from Phase B: B tests *writeback correctness* (does the
 *  array land in the persistent slot); D tests *temporal isolation*
 *  (does the writeback happen *after* all reads complete).
 *
 *  Output is arr1[0]. Same recurrence as the scalar version:
 *  arr1[0] = sample-index. After /20 mix: 0, 0.05, 0.1, ... */
const arrayMutualReg: EdgeFixture = {
  name: 'array_mutual_reg',
  program: {
    op: 'program',
    name: 'ArrayMutualReg',
    ports: { inputs: [], outputs: ['out'] },
    body: { op: 'block',
      decls: [
        { op: 'regDecl', name: 'arr1', init: [0, 0] as any },
        { op: 'regDecl', name: 'arr2', init: [0, 0] as any },
      ],
      assigns: [
        { op: 'outputAssign', name: 'out',
          expr: { op: 'index', args: [{ op: 'reg', name: 'arr1' }, 0] } },
        { op: 'nextUpdate', target: { kind: 'reg', name: 'arr1' },
          expr: { op: 'generate', count: 2, var: 'i',
            body: { op: 'add', args: [
              { op: 'index', args: [{ op: 'reg', name: 'arr2' }, { op: 'binding', name: 'i' }] },
              1,
            ]},
          } as any },
        { op: 'nextUpdate', target: { kind: 'reg', name: 'arr2' },
          expr: { op: 'generate', count: 2, var: 'i',
            body: { op: 'add', args: [
              { op: 'index', args: [{ op: 'reg', name: 'arr1' }, { op: 'binding', name: 'i' }] },
              1,
            ]},
          } as any },
      ],
    },
  },
  expectAllFinite: true,
  tolerance: 12,
}

// (D) Sum-typed delay carrying an array field — Phase B Test 11 (deferred).
// The plan calls for a variant whose payload is `float[N]` (e.g. a Box4
// holding a 4-element payload), with the wholesale writeback covered by
// the same fix as `arrayRegWholesaleWriteback`. Today this isn't
// expressible in the IR: `StructField` (compiler/ir/nodes.ts:54) is
// `{name, type: ScalarKind}` with no shape field, so array-typed payload
// fields can't even be represented. Both surface paths reject them
// before they reach the elaborator: the Zod schema (compiler/schema.ts:
// StructFieldSchema / VariantPayloadFieldSchema) declares only
// `name + scalar_type` and Zod's default strip mode silently drops
// extras; the .trop grammar (compiler/parse/declarations.ts) only
// accepts a scalar kind. Implementing the full path touches:
//   - nodes.ts (extend StructField + BinderDecl with optional shape)
//   - schema.ts + parse/declarations.ts (accept shape on the wire)
//   - elaborator.ts (resolveStructField passes shape through)
//   - sum_lower.ts (multi-slot allocation per (variant, field, idx);
//     extractSlotFromSumExpr returns scalar per index;
//     bindingsForArm substitutes an inline-array of slot DelayRefs)
//   - array_lower.ts (resolve `index` over the synthetic array binding
//     post-sumLower, since the binder's `body` may use the payload via
//     `index(binder, i)`)
// Tracked as a follow-up; not in this PR's scope.

export const EDGE_FIXTURES: EdgeFixture[] = [
  divByZero,
  sqrtNegative,
  denormalMul,
  stableAccumulator,
  selectWithSpecials,
  arrayRegWholesaleWriteback,
  boolPropagationThroughArith,
  // Phase B — wholesale-array writebacks
  arraySelectWriteback,
  arrayZipWithWriteback,
  arrayScanWriteback,
  arrayLetGenerateWriteback,
  // Phase C — known expected-type sites
  clampBoolArms,
  selectMixedArms,
  eqOfNeqs,
  intHintThroughDiv,
  // Phase D — mutual register update (read-before-write isolation)
  scalarMutualReg,
  arrayMutualReg,
]
