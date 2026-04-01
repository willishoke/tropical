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
 * 4. Clears existing wiring on the graph
 * 5. Loads the plan JSON into the C++ runtime
 */
export function applySessionWiring(session: SessionState): void {
  const input = compilerInputFromSession(session)
  const compiled = compilePatch(input)
  const plan = generatePlan(compiled, {
    buffer_length: session.graph.bufferLength,
  })
  const json = planToJSON(plan)

  session.graph.clearWiring()
  session.graph.loadPlan(json)
}
