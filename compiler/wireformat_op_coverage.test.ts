/**
 * wireformat_op_coverage.test.ts — Phase F per-op expected-value table.
 *
 * Replaces the "JIT==interp AND not all zero" proxy with a hand-written
 * spec the codebase can be checked against. Each entry declares
 * (op, args, expected) — a tiny program is built around the op, run
 * for 8 samples through each backend (JIT, interpreter, WASM), and
 * each backend's output is asserted to equal the table's expected
 * value to 10 digits.
 *
 * The class this guards: a new `WireFormatOp` member added without
 * matching three-backend wiring (the `pow` class). Adding a new op
 * to the table forces the question "what's the right denotation?" at
 * test-write time, before either backend can silently emit a wrong
 * constant.
 *
 * Scope: expression-position ops (binary / unary / ternary / array
 * element) — *not* program-shape, decl, or wiring leaves whose
 * "denotation" requires a graph context.
 *
 * Requires libtropical.dylib (build with `make build` first).
 */

import { describe, test, expect } from 'bun:test'
import { makeSession, loadJSON, type ProgramFile } from './session.js'
import { loadStdlib, loadProgramAsType, type ProgramNode } from './program.js'
import { applyFlatPlan } from './apply_plan.js'
import { interpretSession } from './interpret_resolved.js'
import { compileSession } from './ir/compile_session.js'
import type { ExprNode } from './expr.js'
import type { FlatPlan } from './flat_plan.js'
import { emitWasm } from './emit_wasm.js'

const N_SAMPLES = 8

// ─────────────────────────────────────────────────────────────
// Per-op test entry
// ─────────────────────────────────────────────────────────────

interface OpEntry {
  /** Op name (member of WireFormatOp). */
  op: string
  /** ExprNode for the op, evaluated at every sample. May reference
   *  `sampleIndex` for sample-varying outputs. */
  expr: ExprNode
  /** Expected pre-/20-scaling output, one per sample. The audio mix
   *  divides by 20 in both backends; the test multiplies the read-back
   *  by 20 before comparing to this value. */
  expected: number[] | ((s: number) => number)
  /** Optional: skip a backend that doesn't yet support this op. */
  skip?: { wasm?: true }
}

const same = (v: number) => Array.from({ length: N_SAMPLES }, () => v)

// Build a constant 8-sample expected vector from a function of sample
// index `s`.
const each = (f: (s: number) => number) => Array.from({ length: N_SAMPLES }, (_, s) => f(s))

// Helpers to wrap non-float results into the audio mix.
//
// The audio output buffer is f64; the WASM emit's mix path
// unconditionally loads the temp slot as f64, so a slot holding an
// int/bool (encoded as i64) reinterprets the bits and reads as a
// denormal/NaN. The JIT and interp do the bool/int → float coercion
// at the mix boundary; WASM doesn't (gap tracked separately). Wrap
// non-float-typed outputs in `toFloat` here so the test exercises the
// op's *denotation* rather than the WASM mix's missing coercion.
const F  = (e: ExprNode): ExprNode => ({ op: 'toFloat', args: [e] })

const TABLE: OpEntry[] = [
  // ── Arithmetic binary ──
  { op: 'add',      expr: { op: 'add',      args: [3,    4] },           expected: same(7) },
  { op: 'sub',      expr: { op: 'sub',      args: [10,   3] },           expected: same(7) },
  { op: 'mul',      expr: { op: 'mul',      args: [3,    4] },           expected: same(12) },
  { op: 'div',      expr: { op: 'div',      args: [12,   4] },           expected: same(3) },
  { op: 'mod',      expr: { op: 'mod',      args: [10,   3] },           expected: same(1) },
  { op: 'floorDiv', expr: { op: 'floorDiv', args: [10,   3] },           expected: same(3) },
  { op: 'ldexp',    expr: { op: 'ldexp',    args: [3.0,  2] },           expected: same(12) },

  // ── Comparison (bool out → toFloat to land in audio mix) ──
  { op: 'lt',  expr: F({ op: 'lt',  args: [3, 4] }), expected: same(1) },
  { op: 'lte', expr: F({ op: 'lte', args: [4, 4] }), expected: same(1) },
  { op: 'gt',  expr: F({ op: 'gt',  args: [4, 3] }), expected: same(1) },
  { op: 'gte', expr: F({ op: 'gte', args: [4, 4] }), expected: same(1) },
  { op: 'eq',  expr: F({ op: 'eq',  args: [4, 4] }), expected: same(1) },
  { op: 'neq', expr: F({ op: 'neq', args: [3, 4] }), expected: same(1) },

  // ── Bitwise (int args, int out → toFloat for audio) ──
  { op: 'bitAnd', expr: F({ op: 'bitAnd', args: [{ op: 'toInt', args: [12] }, { op: 'toInt', args: [10] }]}), expected: same(8) },
  { op: 'bitOr',  expr: F({ op: 'bitOr',  args: [{ op: 'toInt', args: [12] }, { op: 'toInt', args: [10] }]}), expected: same(14) },
  { op: 'bitXor', expr: F({ op: 'bitXor', args: [{ op: 'toInt', args: [12] }, { op: 'toInt', args: [10] }]}), expected: same(6) },
  { op: 'lshift', expr: F({ op: 'lshift', args: [{ op: 'toInt', args: [3] }, { op: 'toInt', args: [2] }]}),   expected: same(12) },
  { op: 'rshift', expr: F({ op: 'rshift', args: [{ op: 'toInt', args: [12] }, { op: 'toInt', args: [2] }]}),  expected: same(3) },

  // ── Logical (bool args; scaled to 0/1) ──
  { op: 'and', expr: F({ op: 'and', args: [
      { op: 'gt', args: [4, 3] }, { op: 'gt', args: [5, 3] }
    ]}), expected: same(1) },
  { op: 'or',  expr: F({ op: 'or', args: [
      { op: 'gt', args: [4, 3] }, { op: 'lt', args: [5, 3] }
    ]}), expected: same(1) },

  // ── Unary ──
  { op: 'neg',           expr: { op: 'neg',           args: [3] },                 expected: same(-3) },
  { op: 'abs',           expr: { op: 'abs',           args: [-3] },                expected: same(3) },
  { op: 'sqrt',          expr: { op: 'sqrt',          args: [9] },                 expected: same(3) },
  { op: 'floor',         expr: { op: 'floor',         args: [3.7] },               expected: same(3) },
  { op: 'ceil',          expr: { op: 'ceil',          args: [3.2] },               expected: same(4) },
  { op: 'round',         expr: { op: 'round',         args: [3.5] },               expected: same(4) }, // banker's: 3.5 → 4
  { op: 'floatExponent', expr: { op: 'floatExponent', args: [4.0] },               expected: same(2) }, // 2^2 = 4 → exponent 2
  { op: 'not',           expr: F({ op: 'not',           args: [{ op: 'gt', args: [3, 4] }] }), expected: same(1) },
  { op: 'bitNot',        expr: F({ op: 'bitNot',        args: [{ op: 'toInt', args: [0] }]}),  expected: same(-1) },

  // ── Conversions ──
  { op: 'toInt',   expr: F({ op: 'toInt',   args: [3.7] }),                  expected: same(3) },
  { op: 'toBool',  expr: F({ op: 'toBool',  args: [3.7] }),                  expected: same(1) },
  { op: 'toFloat', expr: { op: 'toFloat', args: [{ op: 'toInt', args: [3] }] }, expected: same(3) },

  // ── Ternary ──
  { op: 'select', expr: { op: 'select', args: [
      { op: 'gt', args: [4, 3] }, 7, 11,
    ]}, expected: same(7) },
  { op: 'clamp', expr: { op: 'clamp', args: [12, 0, 10] }, expected: same(10) },

  // ── Array element ops ──
  // arraySet on an inline array, then index it.
  { op: 'arraySet+index', expr: { op: 'index', args: [
      { op: 'arraySet', args: [[1, 2, 3, 4], 2, 99] },
      2,
    ]}, expected: same(99) },
  { op: 'index',          expr: { op: 'index', args: [[10, 20, 30, 40], 2] }, expected: same(30) },

  // ── Sample-index leaf, to anchor that the framework drives s correctly ──
  { op: 'sampleIndex', expr: F({ op: 'sampleIndex' }), expected: each(s => s) },
]

// ─────────────────────────────────────────────────────────────
// Backend runners
// ─────────────────────────────────────────────────────────────

function buildProgram(name: string, expr: ExprNode): ProgramNode {
  return {
    op: 'program', name,
    ports: { inputs: [], outputs: ['out'] },
    body: { op: 'block', assigns: [
      { op: 'outputAssign', name: 'out', expr },
    ]},
  }
}

function setupSession(prog: ProgramNode, instanceName = 'inst'): ReturnType<typeof makeSession> {
  const session = makeSession(N_SAMPLES)
  loadStdlib(session)
  const type = loadProgramAsType(prog, session)!
  session.typeRegistry.set(prog.name, type)
  const inst = type.instantiateAs(instanceName)
  session.instanceRegistry.set(instanceName, inst)
  session.graphOutputs.push({ instance: instanceName, output: inst.outputNames[0] })
  return session
}

function runJit(prog: ProgramNode): Float64Array {
  const session = setupSession(prog)
  applyFlatPlan(session, session.runtime)
  session.graph.primeJit()
  session.graph.process()
  const out = new Float64Array(session.graph.outputBuffer)
  session.graph.dispose()
  return out
}

function runInterp(prog: ProgramNode): Float64Array {
  const session = setupSession(prog)
  const out = interpretSession(session, N_SAMPLES)
  session.graph.dispose()
  return out
}

async function runWasm(prog: ProgramNode): Promise<Float64Array> {
  const session = setupSession(prog)
  let plan: FlatPlan
  try {
    plan = compileSession(session)
  } finally {
    session.graph.dispose()
  }
  const { bytes, layout } = emitWasm(plan, { maxBlockSize: N_SAMPLES })
  const mod = await WebAssembly.compile(bytes)
  const instance = await WebAssembly.instantiate(mod, {})
  const memory = instance.exports.memory as WebAssembly.Memory
  const process_ = instance.exports.process as (blen: number, sidx: bigint) => void
  // Initialize register state to whatever the plan declares (usually
  // empty for these stateless ops).
  const dv = new DataView(memory.buffer)
  for (let i = 0; i < plan.state_init.length; i++) {
    const v = plan.state_init[i]
    const t = plan.register_types[i] ?? 'float'
    const off = layout.registersOffset + i * 8
    if (Array.isArray(v)) continue
    if (typeof v === 'boolean') dv.setBigInt64(off, v ? 1n : 0n, true)
    else if (t === 'int') dv.setBigInt64(off, BigInt(Math.trunc(v as number)), true)
    else if (t === 'bool') dv.setBigInt64(off, (v as number) !== 0 ? 1n : 0n, true)
    else dv.setFloat64(off, v as number, true)
  }
  process_(N_SAMPLES, 0n)
  return new Float64Array(memory.buffer, layout.outputOffset, N_SAMPLES).slice()
}

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

describe('Phase F — WireFormatOp × backend coverage table', () => {
  for (const entry of TABLE) {
    test(entry.op, async () => {
      const expected = Array.isArray(entry.expected)
        ? entry.expected
        : Array.from({ length: N_SAMPLES }, (_, s) => (entry.expected as (s: number) => number)(s))

      const prog = buildProgram(`OpCov_${entry.op.replace(/\W/g, '_')}`, entry.expr)

      // The audio mix divides by 20; multiply read-back by 20 before
      // checking against the table's pre-mix expected value. (Both
      // backends apply the same /20 — that's spec, not a workaround.)
      const SCALE = 20

      const jit = runJit(prog)
      for (let s = 0; s < N_SAMPLES; s++) {
        expect(jit[s] * SCALE).toBeCloseTo(expected[s], 10)
      }

      const interp = runInterp(prog)
      for (let s = 0; s < N_SAMPLES; s++) {
        expect(interp[s] * SCALE).toBeCloseTo(expected[s], 10)
      }

      if (!entry.skip?.wasm) {
        const wasm = await runWasm(prog)
        for (let s = 0; s < N_SAMPLES; s++) {
          expect(wasm[s] * SCALE).toBeCloseTo(expected[s], 10)
        }
      }
    })
  }
})
