/**
 * manifest.ts — the runtime's consumption contract.
 *
 * A `KernelManifest` is everything the runtime needs to drive a Lean-emitted
 * wasm32 kernel: scratch/array/slot sizing, slot defaults, and sample rate. It is
 * a *subset* of `tropical_plan_5` — the runtime never sees the instruction
 * stream or any other compiler internal. This type IS the boundary between
 * tropical (the producer, via `diffcli compile-wasm`) and any consumer (this
 * demo today; an extracted `@tropical/runtime-wasm` package tomorrow).
 */
export type KernelManifest = {
  /** Kernel sample rate (Hz); the kernel reads it for time-based sources. */
  sampleRate: number
  /** Number of i64 SSA scratch cells required by the emitted kernel. The ABI
   * also reserves an equally sized zeroed register region, unused by current
   * production kernels. */
  registerCount: number
  /** Element count of each array-typed register slot. */
  arraySlotSizes: number[]
  /** Number of inter-module slots (instance output wires + param inputs). */
  slotCount: number
  /** Seed value for each slot. Params default here too. */
  slotDefaults: number[]
}
