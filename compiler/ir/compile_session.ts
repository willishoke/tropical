/**
 * compile_session.ts — JIT-side session emit boundary.
 *
 * Three-phase compile:
 *   1. `liftWiresToInstances` — wire pre-process. Wires whose expressions
 *      contain array literals are extracted into anonymous `__wire_${i}`
 *      instances at session pre-compile time. The lifted programs go
 *      through the full strata pipeline so combinators lower correctly.
 *      Session-level `delay()` ops are *not* lifted here — they're
 *      handled by step 2.
 *   2. `extractSessionDelays` — hoists every top-level `delay()`-wrapped
 *      wire into a fresh module slot. The wire is rewritten to a
 *      `sessionSlot` read; the source expression is recorded in
 *      `session.delaySlotRegistry` for `compileSessionSlotted` to emit
 *      as a `WriteSlot` in the scheduler's `state_evolution` phase.
 *      This is the structural mechanism that keeps the MCP-built IR
 *      acyclic — every wire becomes a slot-to-slot copy with one
 *      sample of latency.
 *   3. `compileSessionSlotted` — produces the `tropical_plan_5` FlatPlan
 *      via the per-instance compile path.
 */

import type { SessionState } from '../session.js'
import type { FlatPlan, CompilationMode } from '../flat_plan'
import { liftWiresToInstances } from './lift_wires.js'
import { extractSessionDelays } from './lowering/extract_session_delays.js'
import { assertSessionAcyclic } from './lowering/session_cycle_check.js'

export interface CompileSessionOptions {
  /** Engine realization strategy. Defaults to `'fused'`. */
  compilation_mode?: CompilationMode
}

export function compileSession(
  session: SessionState,
  options: CompileSessionOptions = {},
): FlatPlan {
  // Pre-compile: hoist array-literal wires to anonymous programs.
  liftWiresToInstances(session)

  // Pre-compile: hoist unit-delay wires to module slots.
  extractSessionDelays(session)

  // Defensive invariant: after Phase 2+4 the instance dep graph is
  // acyclic by construction for every MCP-built session. This check
  // catches programmatic sessions that bypass `setWireExpr` and
  // produce true cycles.
  assertSessionAcyclic(session)

  // Lazy import to avoid a circular dependency (compile_session_slotted's
  // helpers import session.js types).
  const { compileSessionSlotted } = require('./compile_session_slotted.js') as
    typeof import('./compile_session_slotted.js')
  return compileSessionSlotted(session, options)
}

export type { Instance } from '../program_types.js'
