/**
 * large_program.test.ts — Phase J (TDD plan §Phase J).
 *
 * Resource-gate test: the compiler must not internal-error on
 * realistic generated-code shapes. NO denotational content; this
 * exists purely to catch passes that recurse proportional to
 * expression depth without iteration / TCO, which would
 * stack-overflow in V8 around ~5000 frames. We test at depth 256
 * (select chain) and 1024 (fold) to leave margin while still firing
 * realistic generated-code shapes.
 *
 * Failure mode is a thrown stack overflow, not a wrong output. We
 * assert compilation completes and one sample of output is finite.
 * No deeper claim.
 */

import { describe, test, expect } from 'bun:test'
import { makeSession, loadJSON, type ExprNode } from '../session.js'
import { loadStdlib, loadProgramAsType, type ProgramNode } from '../program.js'
import { interpretSession } from '../interpret_resolved.js'

describe('Phase J — IR-size / stack blowup (resource gate)', () => {
  test('(S) 256-deep select chain + 1024-element fold: compiles and runs one sample', () => {
    // Build a deep nested select chain: select(c, 0, select(c, 1, select(c, 2, ...)))
    // 256 deep. With c = false (lt(2, 1)), the final value is the
    // deepest else-branch literal (255).
    let selectExpr: ExprNode = 255
    for (let i = 254; i >= 0; i--) {
      selectExpr = { op: 'select', args: [
        { op: 'lt', args: [2, 1] }, // always false
        i,
        selectExpr,
      ]}
    }

    // Also build a 1024-element fold over a generated array. Sum of
    // 0..1023 = 523776. The fold is a bottom-up accumulator tree
    // emitted by array_lower; depth ~1024.
    const foldExpr: ExprNode = {
      op: 'fold',
      over: { op: 'generate', count: 1024, var: 'i',
        body: { op: 'binding', name: 'i' } } as unknown as ExprNode,
      init: 0,
      acc_var: 'a', elem_var: 'b',
      body: { op: 'add', args: [
        { op: 'binding', name: 'a' },
        { op: 'binding', name: 'b' },
      ]},
    } as unknown as ExprNode

    // Combine into a single output: select_chain + fold.
    const Big: ProgramNode = {
      op: 'program', name: 'BigProgram',
      ports: { inputs: [], outputs: ['out'] },
      body: { op: 'block', assigns: [
        { op: 'outputAssign', name: 'out',
          expr: { op: 'add', args: [selectExpr, foldExpr] } },
      ]},
    }

    const session = makeSession(8)
    loadStdlib(session)
    loadProgramAsType(Big, session)
    loadJSON({
      schema: 'tropical_program_2',
      name: 'patch',
      body: { op: 'block', decls: [
        { op: 'instanceDecl', name: 'b', program: 'BigProgram', inputs: {} },
      ]},
      audio_outputs: [{ instance: 'b', output: 'out' }],
    }, session)

    // Resource gate: compile + one-sample evaluation completes
    // without throwing. The expected pre-/20 value is 255 + 523776 =
    // 524031, but we only assert finiteness here per the plan.
    const out = interpretSession(session, 1)
    expect(Number.isFinite(out[0])).toBe(true)
  }, /* timeout */ 30_000)
})
