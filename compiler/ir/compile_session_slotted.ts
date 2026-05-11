/**
 * compile_session_slotted.ts — slot-mode session compilation (M4).
 *
 * This is the new compile path that produces FlatPlans with slot
 * allocation metadata. It runs alongside the legacy `compileSession`
 * and is gated behind the `TROPICAL_SLOT_MODE` env var (or the
 * `slotMode` opt to compileSession).
 *
 * **What this milestone does:** runs the legacy compile pipeline
 * (materialize → strata → compileResolved) to produce the audio-correct
 * instruction stream, then attaches slot allocation metadata derived
 * from the session's slot registries. The instructions themselves
 * remain in legacy form (`input` / `reg` / `state_reg` / `param`
 * operands); the engine work in M5–M6 starts honoring the slot
 * metadata, and M8 migrates the instructions to use `slot` operands
 * directly.
 *
 * Why this incremental approach: a full operand-remapping rewrite is
 * the bulk of the slot-mode work and wants its own milestone. M4
 * proves the schema/plumbing/dispatch flow works end-to-end without
 * gating itself on the larger emit-rewrite, which lets M5–M7 land
 * engine support and equivalence tests against a stable plan shape.
 *
 * Audio behavior under M4: identical to the legacy path. The slot
 * fields are inert metadata — the kernel doesn't read them yet.
 * Equivalence with the legacy path is a regression test, not an
 * audio-correctness milestone.
 */

import type { SessionState } from '../session.js'
import type { FlatPlan } from '../flat_plan.js'
import { compileSessionLegacy } from './compile_session.js'

/**
 * Compile the current session in slot mode. Produces a FlatPlan whose
 * instructions are functionally identical to the legacy path but which
 * carries slot allocation metadata for the new engine path to consume.
 */
export function compileSessionSlotted(session: SessionState): FlatPlan {
  // Audio-correct base plan from the legacy pipeline. We reuse this
  // wholesale until M8 migrates the instruction encoding. Direct
  // call to compileSessionLegacy avoids a re-entry through the
  // env-var dispatch in compileSession.
  const legacy = compileSessionLegacy(session)

  // Emit slot metadata from the session registries populated in M3.
  // The slot space is the union of output slots and param slots — both
  // were assigned consecutive indices via allocateOutputSlots /
  // allocateParamSlot and share session.slotCount.
  const slotCount = session.slotCount
  const slotNames    = new Array<string>(slotCount).fill('')
  const slotDefaults = new Array<number>(slotCount).fill(0)

  for (const [name, idx] of session.outputSlotRegistry) {
    slotNames[idx] = name
  }
  for (const [name, idx] of session.paramSlotRegistry) {
    // Param slots: name = the param's bare name (no instance prefix);
    // distinguish from output slots in serialized form by prefixing
    // "param:" so consumers can route writes/reads correctly.
    slotNames[idx] = `param:${name}`
    // Default value for a param slot = the registered Param's value
    // (smoothed) or 0 for triggers.
    const param = session.paramRegistry.get(name)
    if (param !== undefined) slotDefaults[idx] = param.value
  }

  return {
    ...legacy,
    slot_count:    slotCount,
    slot_names:    slotNames,
    slot_defaults: slotDefaults,
  }
}

/** Read-time check for the slot-mode flag. Cheap; cached not necessary
 *  since this is called once per compile. */
export function slotModeEnabled(session: SessionState, opt?: boolean): boolean {
  if (opt !== undefined) return opt
  // Test-friendly env-var override. `false` / `0` / unset → off; any
  // other value → on. Bun reads process.env consistently.
  const env = process.env.TROPICAL_SLOT_MODE
  if (env === undefined || env === '' || env === '0' || env === 'false') return false
  return true
}
