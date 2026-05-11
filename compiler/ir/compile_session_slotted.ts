/**
 * compile_session_slotted.ts — slot-mode session compilation (M4–M8).
 *
 * As of M8, this is the default path invoked by `compileSession`. It
 * runs the legacy pipeline (materialize → strata → compileResolved) to
 * produce the audio-correct instruction stream, then attaches slot
 * allocation metadata derived from the session's slot registries.
 *
 * **What ships in M4–M8:**
 *   - slot_count / slot_names / slot_defaults populated in every plan
 *   - control-plane writes via tropical_runtime_set_slot work end-to-end
 *   - JIT honors `slot` operands and `WriteSlot` instructions when
 *     emitted (engine ready; emit_resolved doesn't yet emit them)
 *   - Hot-swap preserves slot values by name
 *
 * **What's deferred to a future milestone:**
 *   The instruction stream is still legacy: `input` / `reg` /
 *   `state_reg` / `param` operands. The full rewrite that emits `slot`
 *   operands instead — replacing `param` operands with slot reads,
 *   adding `WriteSlot` after each instance's outputs, etc. — is a
 *   substantial per-instance compile rewrite that warrants its own
 *   focused effort. The engine and equivalence harness are ready
 *   for it (M5–M7); only the TS emit path remains.
 *
 * Audio behavior today: byte-identical to the legacy path. The slot
 * fields are honored for control-plane writes but the kernel doesn't
 * yet read from slots in production patches.
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

/** @deprecated As of M8, slot mode is the default for `compileSession`.
 *  This function previously read TROPICAL_SLOT_MODE; it now always
 *  returns true. Kept as a tiny shim in case external callers
 *  reference it. Will be removed in a follow-up cleanup. */
export function slotModeEnabled(_session?: SessionState, _opt?: boolean): boolean {
  return true
}
