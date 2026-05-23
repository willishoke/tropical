/**
 * apply_plan.ts — Apply the compilation pipeline to a live session.
 *
 * Flow: SessionState → compileSession() → tropical_plan_5 JSON →
 *       runtime.loadPlan()
 *
 * The branded internal `FlatPlan` collapses to a plain-JSON
 * `WireFlatPlan` via `toWirePlan(plan)` at the serialization
 * boundary. Brands erase at runtime; only `RegTarget` (sum type)
 * and the `DstSlot` tag on each `NInstr.dst` get explicit
 * structural conversion.
 */

import type { SessionState } from './session'
import { compileSession, type CompileSessionOptions } from './ir/compile_session'
import { toWirePlan } from './flat_plan'
import type { Runtime } from './runtime/runtime'

/**
 * Compile the session's program graph through the resolved-IR
 * pipeline and push to a FlatRuntime. Call this after any mutation
 * to `inputExprNodes` or `graphOutputs`.
 *
 * `options.compilation_mode` selects the engine realization
 * strategy (default `'fused'`; pass `'microkernel'` to dispatch
 * via the per-sample N+3 function path).
 */
export function applyFlatPlan(
  session: SessionState,
  runtime: Runtime,
  options: CompileSessionOptions = {},
): void {
  const plan = compileSession(session, options)
  const json = JSON.stringify(toWirePlan(plan))
  runtime.loadPlan(json)
}

export function applySessionWiring(
  session: SessionState,
  options: CompileSessionOptions = {},
): void {
  applyFlatPlan(session, session.runtime, options)
}
