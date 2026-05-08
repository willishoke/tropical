/**
 * cse_collisions.test.ts — Phase I (TDD plan §Phase I).
 *
 * Structural CSE in `emit_resolved.ts:297` keys ref-bearing nodes
 * (regRef / delayRef / paramRef / inputRef / nestedOut) on op + decl
 * identity. Two expressions that share decl identity collapse to one
 * temp slot; two with different decl identities don't.
 *
 * The categorical property: CSE preserves denotation. Two expressions
 * with the same denotation may be merged; two with different
 * denotations must NOT be merged.
 *
 * These tests probe two cases where the wrong choice would be
 * observable:
 *   1. Two delayDecls with the same `update` but DIFFERENT `init` —
 *      different denotations from sample 0; must not collapse.
 *   2. Two instances of the same program type with same wiring shape
 *      but different inputs — separate state ⇒ separate denotations;
 *      must not collapse.
 *
 * The plan revised Test 29 to pin the *observable* divergence (different
 * inputs ⇒ different outputs) rather than slot count, since the
 * slot-count check would pass even if non-collapse happened for the
 * wrong reason.
 */

import { describe, test, expect } from 'bun:test'
import { makeSession, loadJSON, type ExprNode } from '../session.js'
import { loadStdlib, loadProgramAsType, type ProgramNode } from '../program.js'
import { interpretSession } from '../interpret_resolved.js'
import { compileSession } from './compile_session.js'

const ACCUM: ProgramNode = {
  op: 'program',
  name: 'AccumX',
  ports: { inputs: [{ name: 'x', default: 0 }], outputs: ['out'] },
  body: { op: 'block',
    decls: [{ op: 'regDecl', name: 'acc', init: 0 }],
    assigns: [
      { op: 'outputAssign', name: 'out', expr: { op: 'reg', name: 'acc' } },
      { op: 'nextUpdate', target: { kind: 'reg', name: 'acc' },
        expr: { op: 'add', args: [{ op: 'reg', name: 'acc' }, { op: 'input', name: 'x' }] } },
    ],
  },
}

describe('Phase I — CSE near-collisions', () => {
  // ──────────────────────────────────────────────────────────
  // Test 28 — delayDecls with identical update, distinct init
  // ──────────────────────────────────────────────────────────
  test('(D) delayDecls with same update, distinct init — sample-0 diff is observable', () => {
    // Build a program with two delay decls: da (init 5) and db (init 11).
    // Both have identical update expression `0`. Output is da - db.
    // At sample 0, output = 5 - 11 = -6 (pre-/20). Post-/20: -0.3.
    // If CSE incorrectly merged these on identical-update, both delay
    // reads would resolve to the same slot and the diff would be 0.
    const Diff: ProgramNode = {
      op: 'program', name: 'DelayInitDiff',
      ports: { inputs: [], outputs: ['out'] },
      body: { op: 'block',
        decls: [
          { op: 'delayDecl', name: 'da', init: 5,  update: 0 },
          { op: 'delayDecl', name: 'db', init: 11, update: 0 },
        ],
        assigns: [{ op: 'outputAssign', name: 'out',
          expr: { op: 'sub', args: [
            { op: 'delayRef', id: 'da' },
            { op: 'delayRef', id: 'db' },
          ]},
        }],
      },
    }
    const session = makeSession(8)
    loadStdlib(session)
    loadProgramAsType(Diff, session)
    loadJSON({
      schema: 'tropical_program_2',
      name: 'patch',
      body: { op: 'block', decls: [
        { op: 'instanceDecl', name: 'd', program: 'DelayInitDiff', inputs: {} },
      ]},
      audio_outputs: [{ instance: 'd', output: 'out' }],
    }, session)

    // IR-shape: post-strata FlatProgram has two distinct delayValue slots.
    const plan = compileSession(session)
    const stateInit = plan.state_init
    // We expect at least 2 distinct delay slot inits among 5 / 11.
    const inits = stateInit.filter(v => typeof v === 'number')
    expect(inits.includes(5)).toBe(true)
    expect(inits.includes(11)).toBe(true)

    // Denotation: at sample 0 the output reads init values ⇒ diff = -6.
    const out = interpretSession(session, 1)
    expect(out[0] * 20).toBeCloseTo(-6, 10)
  })

  // ──────────────────────────────────────────────────────────
  // Test 29 — different instances same shape ⇒ separate state
  // ──────────────────────────────────────────────────────────
  test('(D) different instances same program / same wiring shape, different inputs ⇒ different outputs', () => {
    // Two AccumX instances wired with different constant `x` inputs.
    // Their state regs accumulate independently:
    //   sample 0: a1.out = 0, a2.out = 0
    //   sample 1: a1.out = 0.3, a2.out = 0.7
    //   sample 2: a1.out = 0.6, a2.out = 1.4
    // The audio mix is (a1 + a2)/20 = sample-by-sample sum scaled.
    // If CSE incorrectly merged the two instances on same-shape, they
    // would share state — both would read the same value.
    const session = makeSession(8)
    loadStdlib(session)
    loadProgramAsType(ACCUM, session)
    loadJSON({
      schema: 'tropical_program_2',
      name: 'patch',
      body: { op: 'block', decls: [
        { op: 'instanceDecl', name: 'a1', program: 'AccumX', inputs: { x: 0.3 } },
        { op: 'instanceDecl', name: 'a2', program: 'AccumX', inputs: { x: 0.7 } },
      ]},
      audio_outputs: [
        { instance: 'a1', output: 'out' },
        { instance: 'a2', output: 'out' },
      ],
    }, session)

    // Run via interp (no live JIT param-handle dependency).
    const out = interpretSession(session, 4)
    // Sample 0: a1=0, a2=0 → mix=0
    // Sample 1: a1=0.3, a2=0.7 → mix=1.0 → /20 = 0.05
    // Sample 2: a1=0.6, a2=1.4 → mix=2.0 → /20 = 0.1
    // Sample 3: a1=0.9, a2=2.1 → mix=3.0 → /20 = 0.15
    expect(out[0] * 20).toBeCloseTo(0, 10)
    expect(out[1] * 20).toBeCloseTo(1.0, 10)
    expect(out[2] * 20).toBeCloseTo(2.0, 10)
    expect(out[3] * 20).toBeCloseTo(3.0, 10)
    // If CSE collapsed the two instances, both would read the same
    // accumulator. With x=0.3 (the first instance's input chosen),
    // sample 1 mix would be 0.6, not 1.0. Pin sample 1 to the
    // separate-state value.
    expect(out[1] * 20).not.toBeCloseTo(0.6, 10)
  })
})
