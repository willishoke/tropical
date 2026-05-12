/**
 * compile_session.ts — JIT-side session emit boundary.
 *
 * Delegates to `compileSessionSlotted`, the per-instance compile path
 * that produces `tropical_plan_5`. After PR-C (active-set runtime),
 * this is the only path; the legacy single-kernel materialize-then-
 * inline path is retired.
 */

import type { SessionState } from '../session.js'
import type { FlatPlan } from '../flat_plan'

export function compileSession(session: SessionState): FlatPlan {
  // Lazy import to avoid a circular dependency (compile_session_slotted's
  // helpers import session.js types).
  const { compileSessionSlotted } = require('./compile_session_slotted.js') as
    typeof import('./compile_session_slotted.js')
  return compileSessionSlotted(session)
}

export type { Instance } from '../program_types.js'
