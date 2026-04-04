/**
 * apply_plan.ts — Apply the compilation pipeline to a live session.
 *
 * This is the Phase 5 bridge: mutation tools change SessionState, then call
 * applySessionWiring() to push the new state to the C++ runtime in one shot.
 *
 * Flow: SessionState → CompilerInput → CompiledPatch → ExecutionPlan → JSON → graph.loadPlan()
 */

import type { SessionState } from './patch'
import { compilerInputFromSession, compilePatch } from './compiler'
import { generatePlan, planToJSON } from './plan'

export interface WiringTiming {
  ts_compile_ms: number
  cpp_total_ms: number
  wall_ms: number
  rebuilds: Array<{
    module_count: number
    input_programs_ms: number
    fused_jit_ms: number
    total_ms: number
  }>
}

/**
 * Recompile the session's wiring and outputs and push to the C++ graph.
 *
 * Call this after any mutation to inputExprNodes or graphOutputs.
 * Module additions/removals still go through the existing C API path —
 * this only handles wiring (input expressions) and output routing.
 *
 * The function:
 * 1. Snapshots SessionState into a pure CompilerInput
 * 2. Compiles to a CompiledPatch (topological sort, level grouping)
 * 3. Generates an ExecutionPlan (flat schedule)
 * 4. Clears existing wiring and loads the plan into the C++ runtime in one batch
 *
 * Returns WiringTiming when collectTiming is true.
 */
export function applySessionWiring(session: SessionState, collectTiming?: false): void
export function applySessionWiring(session: SessionState, collectTiming: true): WiringTiming
export function applySessionWiring(session: SessionState, collectTiming = false): WiringTiming | void {
  const t0 = performance.now()

  const input = compilerInputFromSession(session)
  const compiled = compilePatch(input)
  const plan = generatePlan(compiled, { buffer_length: session.graph.bufferLength })
  const json = planToJSON(plan)

  const t1 = performance.now()

  // clearWiring is handled inside loadPlan (clear_wiring_deferred + batch).
  session.graph.loadPlan(json)
  const t2 = performance.now()

  if (collectTiming) {
    return {
      ts_compile_ms: t1 - t0,
      cpp_total_ms:  t2 - t1,
      wall_ms:       t2 - t0,
      rebuilds:      session.graph.buildTimingEntries(),
    }
  }
}
