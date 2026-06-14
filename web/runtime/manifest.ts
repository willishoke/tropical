/**
 * manifest.ts — the runtime's consumption contract.
 *
 * A `KernelManifest` is everything the runtime needs to drive a Lean-emitted
 * wasm32 kernel: region sizing, register/slot init, and the sample rate. It is
 * a *subset* of `tropical_plan_5` — the runtime never sees the instruction
 * stream or any other compiler internal. This type IS the boundary between
 * tropical (the producer, via `diffcli compile-wasm`) and any consumer (this
 * demo today; an extracted `@tropical/runtime-wasm` package tomorrow).
 */
export type RegisterType = 'float' | 'int' | 'bool'

export type KernelManifest = {
  /** Kernel sample rate (Hz); the kernel reads it for time-based sources. */
  sampleRate: number
  /** Number of i64-backed register cells (state + per-sample scratch sizing). */
  registerCount: number
  /** Type of each register, parallel to `stateInit`. */
  registerTypes: RegisterType[]
  /** Initial register values (float / int / bool), by index. */
  stateInit: (number | boolean)[]
  /** Element count of each array-typed register slot. */
  arraySlotSizes: number[]
  /** Number of inter-module slots (instance output wires + param inputs). */
  slotCount: number
  /** Seed value for each slot. Params default here too. */
  slotDefaults: number[]
}
