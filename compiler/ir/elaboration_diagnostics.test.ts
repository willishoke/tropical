/**
 * elaboration_diagnostics.test.ts — Phase 4b (strict cycle policy).
 *
 * Pinned behavior:
 *   1. A program with no instance-level cycles elaborates cleanly.
 *   2. A cyclic program throws `CycleViolation` with port-detailed
 *      Tier-2 error messages naming the cycle members and suggested
 *      `delay` insertion.
 *   3. A user-explicit `delay` between would-be-cycle members
 *      elaborates successfully (the user broke the cycle).
 *   4. The stdlib (all 33 programs) elaborates cleanly — no
 *      inter-instance cycles in any stdlib program.
 *   5. The CycleViolation carries the SCCs as structured data.
 */

import { describe, test, expect } from 'bun:test'
import { parseProgram } from '../parse/declarations.js'
import { elaborate } from './elaborator.js'
import { CycleViolation } from './elaboration_diagnostics.js'
import { loadStdlib } from '../program.js'
import { makeSession } from '../session.js'
import type { ResolvedProgram } from './nodes.js'

function elab(src: string): ResolvedProgram {
  return elaborate(parseProgram(src))
}

describe('Phase 4b — elaborator strict cycle policy', () => {
  test('acyclic program: elaborates cleanly', () => {
    expect(() => elab('program X(a: float) -> (out: float) { out = a + 1 }')).not.toThrow()
  })

  test('two-instance cycle: throws CycleViolation', () => {
    expect(() => elab(`
      program Top() -> (out: float) {
        program Inner(in_: float) -> (out_: float) { out_ = in_ + 1 }
        a = Inner(in_: b.out_)
        b = Inner(in_: a.out_)
        out = a.out_
      }
    `)).toThrow(CycleViolation)
  })

  test('user-broken cycle (delay in between): elaborates cleanly', () => {
    expect(() => elab(`
      program Top() -> (out: float) {
        program Inner(in_: float) -> (out_: float) { out_ = in_ + 1 }
        delay z = a.out_ init 0
        a = Inner(in_: z)
        out = a.out_
      }
    `)).not.toThrow()
  })

  test('three-instance cycle: throws with all members named', () => {
    try {
      elab(`
        program Top() -> (out: float) {
          program Inner(in_: float) -> (out_: float) { out_ = in_ + 1 }
          a = Inner(in_: c.out_)
          b = Inner(in_: a.out_)
          c = Inner(in_: b.out_)
          out = a.out_
        }
      `)
      throw new Error('expected CycleViolation')
    } catch (e) {
      expect(e).toBeInstanceOf(CycleViolation)
      const v = e as CycleViolation
      expect(v.diagnostics.length).toBe(1)
      const memberNames = new Set(v.diagnostics[0].scc.map(i => i.name))
      expect(memberNames).toEqual(new Set(['a', 'b', 'c']))
      expect(v.message).toContain("cycle in program 'Top'")
      expect(v.message).toContain('Suggested fix')
    }
  })

  test('CycleViolation carries Tier-2 port-detail in suggested fix', () => {
    try {
      elab(`
        program Top() -> (out: float) {
          program Inner(in_: float) -> (out_: float) { out_ = in_ + 1 }
          a = Inner(in_: b.out_)
          b = Inner(in_: a.out_)
          out = a.out_
        }
      `)
      throw new Error('expected CycleViolation')
    } catch (e) {
      expect(e).toBeInstanceOf(CycleViolation)
      const v = e as CycleViolation
      expect(v.diagnostics[0].suggestedFix).toMatch(/delay (a|b)_out_delayed = (a|b)\./)
    }
  })

  test('stdlib: every program elaborates cleanly', () => {
    const session = makeSession(8)
    expect(() => loadStdlib(session)).not.toThrow()
  })
})
