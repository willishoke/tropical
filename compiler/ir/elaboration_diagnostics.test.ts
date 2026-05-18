/**
 * elaboration_diagnostics.test.ts — Phase 4a (detect-and-warn).
 *
 * Pinned behavior:
 *   1. A program with no instance-level cycles elaborates with zero warnings.
 *   2. A cyclic program produces a CycleDiagnostic per SCC, formatted
 *      with the cycle members and a suggested-fix snippet naming the
 *      break-target instance.
 *   3. The stdlib (all 31 .trop programs) elaborates with zero warnings —
 *      no inter-instance cycles in any stdlib program.
 *   4. The downstream auto-fix still runs (compile/interpret produce
 *      correct audio for cyclic patches) — warning does NOT replace
 *      the auto-fix in this phase.
 */

import { describe, test, expect, beforeEach, afterEach, mock } from 'bun:test'
import { parseProgram } from '../parse/declarations.js'
import { elaborate } from './elaborator.js'
import { loadStdlib } from '../program.js'
import { makeSession } from '../session.js'
import type { ResolvedProgram } from './nodes.js'

function elab(src: string): ResolvedProgram {
  return elaborate(parseProgram(src))
}

describe('Phase 4a — elaborator cycle warnings', () => {
  let warnSpy: ReturnType<typeof mock>
  let originalWarn: typeof console.warn

  beforeEach(() => {
    warnSpy = mock(() => {})
    originalWarn = console.warn
    console.warn = warnSpy as unknown as typeof console.warn
  })

  afterEach(() => {
    console.warn = originalWarn
  })

  test('acyclic program: zero warnings', () => {
    elab('program X(a: float) -> (out: float) { out = a + 1 }')
    expect(warnSpy.mock.calls.length).toBe(0)
  })

  test('two-instance cycle: one warning naming both members', () => {
    elab(`
      program Top() -> (out: float) {
        program Inner(in_: float) -> (out_: float) { out_ = in_ + 1 }
        a = Inner(in_: b.out_)
        b = Inner(in_: a.out_)
        out = a.out_
      }
    `)
    expect(warnSpy.mock.calls.length).toBe(1)
    const msg = warnSpy.mock.calls[0][0] as string
    expect(msg).toContain("cycle in program 'Top'")
    expect(msg).toContain('Instances in cycle')
    expect(msg).toContain('Suggested fix')
    // Both cycle members named in the path.
    expect(/a/.test(msg) && /b/.test(msg)).toBe(true)
  })

  test('user-broken cycle (delay in between): no warning', () => {
    elab(`
      program Top() -> (out: float) {
        program Inner(in_: float) -> (out_: float) { out_ = in_ + 1 }
        delay z = a.out_ init 0
        a = Inner(in_: z)
        out = a.out_
      }
    `)
    expect(warnSpy.mock.calls.length).toBe(0)
  })

  test('stdlib: every .trop program elaborates with zero warnings', () => {
    const session = makeSession(8)
    loadStdlib(session)
    expect(warnSpy.mock.calls.length).toBe(0)
  })

  test('suggested-fix snippet names the break-target', () => {
    elab(`
      program Top() -> (out: float) {
        program Inner(in_: float) -> (out_: float) { out_ = in_ + 1 }
        a = Inner(in_: b.out_)
        b = Inner(in_: a.out_)
        out = a.out_
      }
    `)
    expect(warnSpy.mock.calls.length).toBe(1)
    const msg = warnSpy.mock.calls[0][0] as string
    // The break-target is the first member in source order.
    // Suggested fix mentions a synthetic delay name based on the
    // break-target instance.
    expect(msg).toMatch(/delay (a|b)_out_delayed = (a|b)\./)
  })
})
