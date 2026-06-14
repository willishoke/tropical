/**
 * layout.ts — byte layout of the kernel's linear memory.
 *
 * The Lean-emitted kernel takes its working regions as pointer arguments
 * (inputs, registers, arrays, array_sizes, temps, slots, output). The host owns
 * one imported linear memory and places each region above the module's
 * `__heap_base` (its shadow stack + data). Pure arithmetic over a
 * `KernelManifest` — no plan internals, the native engine's analog is
 * `FlatRuntime`'s backing-store allocation.
 */
import type { KernelManifest } from './manifest.js'

export type KernelLayout = {
  inputs: number
  registers: number
  temps: number
  arrays: number          // base of the array backing store (== first slot)
  arrayOffsets: number[]  // absolute byte offset per array slot
  arraySizes: number      // i64[] holding each slot's element count
  slots: number
  output: number
  endByte: number         // first free byte after every region
}

/** Zeroed inputs region. The demo synths take no external audio input; a
 *  generous slack covers any `inputs[k]` the kernel might read. */
const INPUT_SLACK = 64

const align8 = (n: number): number => (n + 7) & ~7

export function computeLayout(
  manifest: KernelManifest,
  maxBlockSize: number,
  heapBase: number,
): KernelLayout {
  let p = align8(heapBase)
  const inputs = p;     p = align8(p + INPUT_SLACK * 8)
  const registers = p;  p = align8(p + manifest.registerCount * 8)
  const temps = p;      p = align8(p + manifest.registerCount * 8)
  const arrays = p
  const arrayOffsets: number[] = []
  for (const sz of manifest.arraySlotSizes) { arrayOffsets.push(p); p = align8(p + sz * 8) }
  const arraySizes = p; p = align8(p + manifest.arraySlotSizes.length * 8)
  const slots = p;      p = align8(p + manifest.slotCount * 8)
  const output = p;     p = align8(p + maxBlockSize * 8)
  return { inputs, registers, temps, arrays, arrayOffsets, arraySizes, slots, output, endByte: p }
}
