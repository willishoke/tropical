/**
 * flat_plan.ts — `tropical_plan_4` JSON schema.
 *
 * The FlatPlan type is the contract between the TS compiler layer and
 * the C++ engine. Every emit boundary (`compile_resolved.ts`,
 * `compile_session.ts`, the WASM emitter) produces this shape; the
 * runtime's `loadPlan` consumes it.
 *
 * Hoisted here in Phase D D3 so consumers don't have to import from
 * `flatten.ts`, which is being retired.
 */

import type { NInstr, GroupInfo, ScalarType } from './ir/emit_resolved'

export interface FlatPlan {
  schema: 'tropical_plan_4'
  config: { sampleRate: number }
  state_init: (number | boolean)[]
  register_names: string[]
  register_types: ScalarType[]
  array_slot_names: string[]
  outputs: number[]
  /** Compiled instruction stream from `emitNumericProgram`. */
  instructions:    NInstr[]
  register_count:  number
  array_slot_count: number
  array_slot_sizes: number[]
  output_targets:  number[]
  register_targets: number[]
  /** Gateable-subgraph metadata, if any source_tag wrappers were emitted. */
  groups?:         GroupInfo[]

  // ─── Slot model (additive) ─────────────────────────────────────────────
  // Inter-module shared slot array. Module outputs and control-plane
  // params land here; downstream module input expressions read from
  // `slot` operands referencing these indices. All three fields are
  // optional so legacy plans (no slot model) continue to validate.
  /** Total slots in the shared inter-module array. */
  slot_count?:    number
  /** "${instance}.${port}" for output slots; "${name}" for param slots.
   *  Length === slot_count. */
  slot_names?:    string[]
  /** Initial value per slot (params start at declared default; output
   *  slots typically start at 0). Length === slot_count. */
  slot_defaults?: number[]
}
