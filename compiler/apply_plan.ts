/**
 * apply_plan.ts — Apply the compilation pipeline to a live session.
 *
 * Flattens the session's program graph and pushes to a FlatRuntime.
 * All expression trees are inlined in TS and sent as a single flat plan.
 *
 * Flow: SessionState → flattenSession() → tropical_plan_2 JSON → runtime.loadPlan()
 */

import type { SessionState } from './session'
import { flattenSession } from './flatten'
import type { Runtime } from './runtime/runtime'

/**
 * Flatten the session's program graph and push to a FlatRuntime.
 *
 * Call this after any mutation to inputExprNodes or graphOutputs.
 * All expression trees are inlined in TS and sent as a single flat plan —
 * no Graph/Module boundaries, just one kernel.
 */
export function applyFlatPlan(session: SessionState, runtime: Runtime): void {
  const plan = flattenSession(session)
  const json = JSON.stringify(plan)
  runtime.loadPlan(json)
}

export function applySessionWiring(session: SessionState): void {
  applyFlatPlan(session, session.runtime)
}
