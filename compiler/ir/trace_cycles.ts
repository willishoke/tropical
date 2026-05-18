/**
 * trace_cycles.ts — Phase C4 cycle-break shim.
 *
 * Post-Phase 1: the algorithm lives in `compiler/ir/lowering/cycle_break.ts`.
 * This file is a backwards-compatibility shim that calls the helper
 * and returns just the lowered program (matching the legacy
 * `traceCycles` signature consumed by `strataPipeline` today).
 *
 * Phase 3 retires the call from `strataPipeline` (cycle-breaking moves
 * out of the compiler to the standard realization's elaboration
 * layer); Phase 5 deletes this shim entirely. Until then, the export
 * exists so existing callers (and unit tests pinning denotation
 * through the legacy entry point) continue to work without rewrite.
 */

import type { ResolvedProgram } from './nodes.js'
import { breakInstanceCycles } from './lowering/cycle_break.js'

export function traceCycles(prog: ResolvedProgram): ResolvedProgram {
  return breakInstanceCycles(prog).lowered
}
