/**
 * compiler.ts — Main-thread plan-to-WASM compilation.
 *
 * Takes a pre-flattened `tropical_plan_4` JSON (produced at build time by
 * the native-side compiler pipeline) and emits a WASM module ready for
 * the AudioWorklet to instantiate.
 *
 * The full TS compiler pipeline (program → flatten → emit) stays on the
 * native side because its transitive imports pull in koffi via the native
 * runtime. For a demo site with curated patches, running the pipeline
 * offline at build time is actually the cleaner design: smaller browser
 * bundle, no cold-start compile, deterministic.
 */

import type { FlatPlan } from '../../compiler/flatten.js'
import { emitWasm } from '../../compiler/emit_wasm.js'
import type { LoadedPlan } from '../worklet/runtime.js'

/**
 * Compile a pre-flattened FlatPlan to a LoadedPlan for the worklet.
 *
 * Note: we post the raw WASM bytes (not a WebAssembly.Module) because
 * Chrome silently drops AudioWorklet port messages that contain
 * WebAssembly.Module — compilation happens inside the worklet.
 */
export async function compilePlan(plan: FlatPlan, maxBlockSize = 2048): Promise<LoadedPlan> {
  const { bytes, layout, paramPtrs } = emitWasm(plan, { maxBlockSize })
  return {
    bytes,
    layout,
    paramPtrs,
    stateInit: plan.state_init,
    registerTypes: plan.register_types,
    registerNames: plan.register_names,
    arraySlotNames: plan.array_slot_names,
  }
}

/** Parse a `tropical_plan_4` JSON string (e.g. fetched from /patches/foo.plan.json) and compile it. */
export async function compilePlanJson(json: string, maxBlockSize = 2048): Promise<LoadedPlan> {
  return compilePlan(JSON.parse(json) as FlatPlan, maxBlockSize)
}
