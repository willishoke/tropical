/**
 * apply_plan.ts — Apply the compilation pipeline to a live session.
 *
 * Flattens the session's patch and pushes to a FlatRuntime.
 * All expression trees are inlined in TS and sent as a single flat plan.
 *
 * Flow: SessionState → flattenPatch() → egress_plan_2 JSON → runtime.loadPlan()
 */

import type { SessionState } from './patch'
import { flattenPatch } from './flatten'
import type { Runtime } from './runtime/runtime'

/**
 * Flatten the session's patch and push to a FlatRuntime.
 *
 * Call this after any mutation to inputExprNodes or graphOutputs.
 * All expression trees are inlined in TS and sent as a single flat plan —
 * no Graph/Module boundaries, just one kernel.
 */
export function applyFlatPlan(session: SessionState, runtime: Runtime): void {
  const plan = flattenPatch(session)
  const json = JSON.stringify(plan)
  runtime.loadPlan(json)
}
