/**
 * compile_session.ts — JIT-side session emit boundary.
 *
 * Two-phase compile:
 *   1. `liftWiresToInstances` — wire pre-process. Wires whose expressions
 *      contain forms `translateNode` doesn't handle (array literals,
 *      session-level `delay()`) are extracted into anonymous
 *      `__wire_${i}` instances at session pre-compile time. The lifted
 *      programs go through the full strata pipeline so combinators
 *      lower correctly. After this pass, every wire is a simple
 *      `translateNode`-compatible form.
 *   2. `compileSessionSlotted` — produces the `tropical_plan_5` FlatPlan
 *      via the per-instance compile path.
 */

import type { SessionState } from '../session.js'
import type { FlatPlan } from '../flat_plan'
import { liftWiresToInstances } from './lift_wires.js'

export function compileSession(session: SessionState): FlatPlan {
  // Pre-compile: hoist complex wires to anonymous programs.
  liftWiresToInstances(session)

  // Lazy import to avoid a circular dependency (compile_session_slotted's
  // helpers import session.js types).
  const { compileSessionSlotted } = require('./compile_session_slotted.js') as
    typeof import('./compile_session_slotted.js')
  return compileSessionSlotted(session)
}

export type { Instance } from '../program_types.js'
