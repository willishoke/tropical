/**
 * layout.ts — byte layout of the kernel's linear memory.
 *
 * The port takes one pointer per input and one for the output
 * (design/typed-kernel.md); everything else lives inside the function. The
 * host owns one imported linear memory and places each input and the output
 * above the module's `__heap_base` (its shadow stack + data). Pure arithmetic
 * over a `KernelManifest`; the native engine's analog is `FlatRuntime`'s
 * argv table.
 */
import {
  kernelFrames, kernelOutputChannelCount, kernelParams, type KernelManifest,
} from './manifest.js'

export type KernelLayout = {
  frames: number
  tau: number             // i64[frames] — the Q32.32 clock of each frame
  params: number[]        // one f64 cell per param input, in port order
  output: number          // f64[frames * channels], frame-major
  endByte: number         // first free byte after every region
}

const align8 = (n: number): number => (n + 7) & ~7

export function computeLayout(
  manifest: KernelManifest,
  heapBase: number,
): KernelLayout {
  const frames = kernelFrames(manifest)
  let p = align8(heapBase)
  const tau = p;        p = align8(p + frames * 8)
  const params: number[] = []
  for (const _ of kernelParams(manifest)) { params.push(p); p += 8 }
  p = align8(p)
  const output = p
  p = align8(p + frames * kernelOutputChannelCount(manifest) * 8)
  return { frames, tau, params, output, endByte: p }
}
