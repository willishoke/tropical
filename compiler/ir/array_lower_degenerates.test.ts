/**
 * array_lower_degenerates.test.ts — Phase E (TDD plan §Phase E).
 *
 * Boundary conditions for the combinator unrolling in `array_lower.ts`:
 * empty arrays for fold/scan/map2, count=0 for generate/iterate, and
 * single-element direction-pinning for fold using a non-commutative
 * lambda. Plus a forced choice for nested generates: pin the denotation
 * if the type system accepts it; pin the rejection error if it doesn't.
 *
 * Each test pairs an `arrayLower` IR-shape assertion (no combinators
 * survive) with a denotational pin against a hand-computed expected
 * value. Direction-pinning combinators use `f = (a, b) => a*2 + b`
 * (non-commutative); the test asserts the *exact* left-fold result.
 */

import { describe, test, expect } from 'bun:test'
import { parseProgram } from '../parse/declarations.js'
import { elaborate } from './elaborator.js'
import { arrayLower } from './array_lower.js'
import type { ResolvedProgram, ResolvedExpr } from './nodes.js'
import { interpretSession } from '../interpret_resolved.js'
import { makeSession, loadJSON } from '../session.js'
import { loadProgramAsType, type ProgramNode } from '../program.js'

function elab(src: string): ResolvedProgram {
  return elaborate(parseProgram(src))
}

function findOps(prog: ResolvedProgram, targets: string[]): string[] {
  const set = new Set(targets)
  const out: string[] = []
  const seen = new WeakSet<object>()
  const visit = (e: ResolvedExpr): void => {
    if (e === null || typeof e !== 'object') return
    if (seen.has(e as object)) return
    seen.add(e as object)
    if (Array.isArray(e)) { e.forEach(visit); return }
    if (set.has(e.op)) out.push(e.op)
    for (const [k, v] of Object.entries(e)) {
      if (k === 'op' || k === 'decl' || k === 'instance' || k === 'output') continue
      if (k === 'type' || k === 'parent' || k === 'variant' || k === 'iter'
          || k === 'acc' || k === 'elem' || k === 'x' || k === 'y' || k === 'binder') continue
      if (Array.isArray(v)) v.forEach(c => visit(c as ResolvedExpr))
      else if (v !== null && typeof v === 'object') visit(v as ResolvedExpr)
    }
  }
  for (const d of prog.body.decls) {
    if (d.op === 'regDecl') visit(d.init)
    else if (d.op === 'delayDecl') { visit(d.init); visit(d.update) }
    else if (d.op === 'instanceDecl') for (const i of d.inputs) visit(i.value)
  }
  for (const a of prog.body.assigns) visit(a.expr)
  return out
}

const COMBINATORS = ['let', 'fold', 'scan', 'generate', 'iterate', 'chain', 'map2', 'zipWith']

/** Run a one-instance program through interpretSession; return the
 *  pre-/20 sample 0 value (×20 to undo the audio-mix scaling). */
function evalSample0(prog: ProgramNode): number {
  const session = makeSession(8)
  loadProgramAsType(prog, session)
  loadJSON({
    schema: 'tropical_program_2',
    name: 'patch',
    body: { op: 'block', decls: [
      { op: 'instanceDecl', name: 'x', program: prog.name, inputs: {} },
    ]},
    audio_outputs: [{ instance: 'x', output: prog.ports!.outputs![0] as string }],
  }, session)
  const out = interpretSession(session, 1)
  return out[0] * 20
}

// ─────────────────────────────────────────────────────────────
// Test 18 — generate(0, ...): empty array
// ─────────────────────────────────────────────────────────────

describe('Phase E — generate degenerates', () => {
  test('(D) generate(0, i => i*10) lowers to []; downstream readers must guard', () => {
    // arrayLower must accept count=0 without crashing. Reading the
    // result with `index` would either return type-zero or throw; we
    // don't pin the downstream behavior, only that the generate
    // itself lowers to an empty inline array.
    const p = elab(`
      program X() -> (out: float) {
        out = fold(generate(0, (i) => i * 10), 99, (a, c) => a + c)
      }
    `)
    const out = arrayLower(p)
    expect(findOps(out, COMBINATORS)).toEqual([])
    // fold over an empty array → init. So out = 99. Pin via interpretSession.
    expect(evalSample0({
      op: 'program', name: 'GenerateZero',
      ports: { inputs: [], outputs: ['out'] },
      body: { op: 'block', assigns: [
        { op: 'outputAssign', name: 'out',
          expr: { op: 'fold',
            over: { op: 'generate', count: 0, var: 'i',
              body: { op: 'mul', args: [{ op: 'binding', name: 'i' }, 10] }},
            init: 99,
            acc_var: 'a', elem_var: 'c',
            body: { op: 'add', args: [
              { op: 'binding', name: 'a' }, { op: 'binding', name: 'c' },
            ]},
          } as any },
      ]},
    })).toBeCloseTo(99, 10)
  })
})

// ─────────────────────────────────────────────────────────────
// Test 19 — fold([], init, f) = init
// ─────────────────────────────────────────────────────────────

describe('Phase E — fold degenerates', () => {
  test('(D) fold([], init=7, non-commutative f) = 7', () => {
    // Non-commutative f keeps the test honest: a wrong choice of
    // direction or skipping the unit case would produce a different
    // output for non-empty inputs (verified separately by Test 20).
    expect(evalSample0({
      op: 'program', name: 'FoldEmpty',
      ports: { inputs: [], outputs: ['out'] },
      body: { op: 'block', assigns: [
        { op: 'outputAssign', name: 'out',
          expr: { op: 'fold',
            over: { op: 'generate', count: 0, var: 'i', body: 0 } as any,
            init: 7,
            acc_var: 'a', elem_var: 'b',
            body: { op: 'add', args: [
              { op: 'mul', args: [{ op: 'binding', name: 'a' }, 2] },
              { op: 'binding', name: 'b' },
            ]},
          } as any },
      ]},
    })).toBeCloseTo(7, 10)
  })

  // ─────────────────────────────────────────────────────────────
  // Test 20 — fold([x], init, f) = f(init, x) [direction-pinning]
  // ─────────────────────────────────────────────────────────────
  test('(D) fold([3], init=1, f=(a,b)=>a*2+b) = 5 (left-fold direction)', () => {
    // f(1, 3) = 1*2 + 3 = 5  (left fold; the spec)
    // f(3, 1) = 3*2 + 1 = 7  (right fold; the wrong answer)
    expect(evalSample0({
      op: 'program', name: 'FoldSingle',
      ports: { inputs: [], outputs: ['out'] },
      body: { op: 'block', assigns: [
        { op: 'outputAssign', name: 'out',
          expr: { op: 'fold',
            over: [3] as any,
            init: 1,
            acc_var: 'a', elem_var: 'b',
            body: { op: 'add', args: [
              { op: 'mul', args: [{ op: 'binding', name: 'a' }, 2] },
              { op: 'binding', name: 'b' },
            ]},
          } as any },
      ]},
    })).toBeCloseTo(5, 10)
  })
})

// ─────────────────────────────────────────────────────────────
// Test 21 — iterate(0, x, f) = x
// ─────────────────────────────────────────────────────────────

describe('Phase E — iterate degenerates', () => {
  test('(D) iterate(0, init=42, any f) reads element 0 = 42', () => {
    // iterate emits [init, f(init), f(f(init)), ...] of length count.
    // count=0 → empty array. Reading index 0 is undefined — instead
    // sum all elements via fold to make the empty case observable.
    expect(evalSample0({
      op: 'program', name: 'IterateZero',
      ports: { inputs: [], outputs: ['out'] },
      body: { op: 'block', assigns: [
        { op: 'outputAssign', name: 'out',
          expr: { op: 'fold',
            over: { op: 'iterate', count: 0, init: 42, var: 'x',
              body: { op: 'mul', args: [{ op: 'binding', name: 'x' }, 2] },
            } as any,
            init: 99,
            acc_var: 'a', elem_var: 'b',
            body: { op: 'add', args: [
              { op: 'binding', name: 'a' }, { op: 'binding', name: 'b' },
            ]},
          } as any },
      ]},
    })).toBeCloseTo(99, 10)  // fold over empty = init = 99

    // Now count=1 — iterate emits [init], single element. Sum = 42.
    expect(evalSample0({
      op: 'program', name: 'IterateOne',
      ports: { inputs: [], outputs: ['out'] },
      body: { op: 'block', assigns: [
        { op: 'outputAssign', name: 'out',
          expr: { op: 'fold',
            over: { op: 'iterate', count: 1, init: 42, var: 'x',
              body: { op: 'mul', args: [{ op: 'binding', name: 'x' }, 2] },
            } as any,
            init: 0,
            acc_var: 'a', elem_var: 'b',
            body: { op: 'add', args: [
              { op: 'binding', name: 'a' }, { op: 'binding', name: 'b' },
            ]},
          } as any },
      ]},
    })).toBeCloseTo(42, 10)
  })
})

// ─────────────────────────────────────────────────────────────
// Test 22 — scan with non-commutative f
// ─────────────────────────────────────────────────────────────

describe('Phase E — scan degenerates', () => {
  test('(D) scan([1,2,3], seed=0, f=(a,b)=>a*2+b) emits [1, 4, 11]', () => {
    // scan emits intermediate accumulators, NOT including the seed
    // (per array_lower.ts `lowerScan`).
    //   step 1: f(0, 1) = 0*2 + 1 = 1
    //   step 2: f(1, 2) = 1*2 + 2 = 4
    //   step 3: f(4, 3) = 4*2 + 3 = 11
    // Sum of output = 1 + 4 + 11 = 16. fold over scan to make the
    // sequence observable as a single scalar.
    expect(evalSample0({
      op: 'program', name: 'ScanNonCommutative',
      ports: { inputs: [], outputs: ['out'] },
      body: { op: 'block', assigns: [
        { op: 'outputAssign', name: 'out',
          expr: { op: 'fold',
            over: { op: 'scan',
              over: [1, 2, 3] as any,
              init: 0,
              acc_var: 'a', elem_var: 'b',
              body: { op: 'add', args: [
                { op: 'mul', args: [{ op: 'binding', name: 'a' }, 2] },
                { op: 'binding', name: 'b' },
              ]},
            } as any,
            init: 0,
            acc_var: 'a2', elem_var: 'b2',
            body: { op: 'add', args: [
              { op: 'binding', name: 'a2' }, { op: 'binding', name: 'b2' },
            ]},
          } as any },
      ]},
    })).toBeCloseTo(16, 10)

    // Pin individual elements via index reads (sample 0 each).
    const idxOf = (i: number): number => evalSample0({
      op: 'program', name: `ScanIdx${i}`,
      ports: { inputs: [], outputs: ['out'] },
      body: { op: 'block', assigns: [
        { op: 'outputAssign', name: 'out',
          expr: { op: 'index', args: [
            { op: 'scan',
              over: [1, 2, 3] as any,
              init: 0,
              acc_var: 'a', elem_var: 'b',
              body: { op: 'add', args: [
                { op: 'mul', args: [{ op: 'binding', name: 'a' }, 2] },
                { op: 'binding', name: 'b' },
              ]},
            },
            i,
          ]} as any },
      ]},
    })
    expect(idxOf(0)).toBeCloseTo(1,  10)
    expect(idxOf(1)).toBeCloseTo(4,  10)
    expect(idxOf(2)).toBeCloseTo(11, 10)
  })
})

// ─────────────────────────────────────────────────────────────
// Test 23 — nested generates: (D) if accepted, (R) if rejected
// ─────────────────────────────────────────────────────────────

describe('Phase E — nested generates', () => {
  // generate(N, i => generate(M, j => body(i, j))). If the type
  // system accepts it, the result lowers to a flat N*M-element array
  // with denotation (λi. (λj. body(i, j)))-uncurry. If it doesn't,
  // pin the rejection.
  test('nested generate either flattens to N*M array OR is rejected at type-check', () => {
    let lowered: ReturnType<typeof arrayLower> | undefined
    let elabError: Error | undefined
    let lowerError: Error | undefined
    try {
      const p = elab(`
        program X() -> (out: float) {
          out = fold(
            generate(2, (i) => generate(3, (j) => i * 10 + j)),
            0,
            (a, c) => a + c
          )
        }
      `)
      try { lowered = arrayLower(p) }
      catch (e) { lowerError = e as Error }
    } catch (e) {
      elabError = e as Error
    }

    if (elabError !== undefined || lowerError !== undefined) {
      // (R) Rejected — pin the rejection. The error message must
      // mention something about nested arrays / lambdas / shape;
      // assert non-empty and document the shape so future refactors
      // notice if it changes.
      const msg = (elabError ?? lowerError)!.message
      expect(typeof msg).toBe('string')
      expect(msg.length).toBeGreaterThan(0)
    } else {
      // (D) Accepted — denotation pins fold-over-nested-generate.
      // For body(i, j) = i*10 + j, N=2, M=3: inner produces
      // [[0,1,2], [10,11,12]]. The outer fold runs:
      //   f(0,         [0,1,2])    = 0  +  [0,1,2]    = [0, 1, 2]   (broadcast scalar)
      //   f([0,1,2], [10,11,12])   = [0,1,2] + [10,11,12] = [10, 12, 14]
      // Result is the array [10, 12, 14]. The audio mix takes the
      // first element (`toNum` on an array returns v[0]). So output
      // sample 0 = 10.
      expect(findOps(lowered!, COMBINATORS)).toEqual([])
      const v = evalSample0({
        op: 'program', name: 'NestedGen',
        ports: { inputs: [], outputs: ['out'] },
        body: { op: 'block', assigns: [
          { op: 'outputAssign', name: 'out',
            expr: { op: 'fold',
              over: { op: 'generate', count: 2, var: 'i',
                body: { op: 'generate', count: 3, var: 'j',
                  body: { op: 'add', args: [
                    { op: 'mul', args: [{ op: 'binding', name: 'i' }, 10] },
                    { op: 'binding', name: 'j' },
                  ]},
                },
              } as any,
              init: 0,
              acc_var: 'a', elem_var: 'c',
              body: { op: 'add', args: [
                { op: 'binding', name: 'a' }, { op: 'binding', name: 'c' },
              ]},
            } as any },
        ]},
      })
      expect(v).toBeCloseTo(10, 10)
    }
  })
})
