/**
 * compiler/ir/acyclic.ts — acyclicity check at the strata-pipeline
 * boundary.
 *
 * Thin wrapper around `findInstanceCycles` from
 * `compiler/ir/lowering/cycle_break.ts`. The detection algorithm lives
 * with the cycle-break helper; this module supplies the
 * strataPipeline-specific assertion + error class.
 *
 * Called from inside `strataPipeline` to enforce the contract that
 * its input is acyclic. Cycle-breaking is the responsibility of the
 * realization layer above the compiler — the elaborator throws on
 * source-level cycles and the session materializer extracts session-
 * level `delay()` ops. Any cycle reaching `strataPipeline` is a
 * caller bug, surfaced immediately rather than producing a malformed
 * plan downstream.
 */

import type { ResolvedProgram, InstanceDecl } from './nodes.js'
import { findInstanceCycles } from './lowering/cycle_break.js'

export { findInstanceCycles }

/** Thrown when a `ResolvedProgram` reaches `strataPipeline` carrying
 *  a non-trivial cycle in its inter-instance graph. Carries the
 *  detected SCCs as structured data so callers (tests, error
 *  formatters) can render the violation precisely. */
export class AcyclicityViolation extends Error {
  readonly sccs: ReadonlyArray<ReadonlyArray<InstanceDecl>>
  constructor(sccs: ReadonlyArray<ReadonlyArray<InstanceDecl>>) {
    const names = sccs.map(scc => scc.map(i => i.name).join(' → ')).join('; ')
    super(`strataPipeline: input contains an unbroken inter-instance cycle: ${names}`)
    this.name = 'AcyclicityViolation'
    this.sccs = sccs
  }
}

/** Throws `AcyclicityViolation` if `prog` carries any non-trivial
 *  cycle in its inter-instance graph. */
export function assertAcyclic(prog: ResolvedProgram): void {
  const cycles = findInstanceCycles(prog)
  if (cycles.length > 0) throw new AcyclicityViolation(cycles)
}
