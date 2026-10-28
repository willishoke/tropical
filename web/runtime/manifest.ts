/**
 * manifest.ts — the runtime's consumption contract: `tropical_kernel_1`.
 *
 * A kernel is one typed pure function (design/typed-kernel.md): a record of
 * named, static-shaped inputs — `tau` (the Q32.32 clock, shape `[frames]`)
 * and one scalar `f64` per param — and one output of shape
 * `[frames, channels]`, frame-major. This type IS the boundary between
 * tropical (the producer, via `diffcli compile-kernel`) and any consumer
 * (this player today; an extracted `@tropical/runtime-wasm` package
 * tomorrow). The manifest also nests the compiler's plan for its own gates;
 * the runtime never reads it.
 */
export type KernelElem = 'f64' | 'i64' | 'i64q32'
export type KernelRole = 'param' | 'signal' | 'table'

export type KernelPort = {
  name: string
  elem: KernelElem
  shape: number[]
  role?: KernelRole
  /** Seed value a host writes before the first call (params). */
  default?: number
  /** Output channel source names (outputs of shape [frames, C]). */
  channels?: string[]
}

export type KernelManifest = {
  schema: 'tropical_kernel_1'
  /** The rate the kernel was compiled for (Hz); a constant of the artifact. */
  rate: number
  /** `tau` first, then the params in port order. */
  inputs: KernelPort[]
  /** Exactly one output: `out`, `f64`, shape `[frames, channels]`. */
  outputs: KernelPort[]
}

export const MAX_KERNEL_OUTPUT_CHANNELS = 64

/** The static frame count per call, from the `tau` input's shape. */
export function kernelFrames(manifest: KernelManifest): number {
  const tau = manifest.inputs[0]
  if (!tau || tau.name !== 'tau' || tau.elem !== 'i64q32' || tau.shape.length !== 1)
    throw new RangeError('KernelManifest: the first input must be tau:i64q32[frames]')
  const frames = tau.shape[0]!
  if (!Number.isSafeInteger(frames) || frames < 1)
    throw new RangeError('KernelManifest: frames must be a positive integer')
  return frames
}

/** The param inputs (everything after `tau`), in port order. */
export function kernelParams(manifest: KernelManifest): KernelPort[] {
  return manifest.inputs.slice(1)
}

/** Resolve and validate the fixed output width once, off the audio callback. */
export function kernelOutputChannelCount(manifest: KernelManifest): number {
  const out = manifest.outputs[0]
  if (!out || manifest.outputs.length !== 1 || out.elem !== 'f64' || out.shape.length !== 2)
    throw new RangeError('KernelManifest: exactly one f64[frames, channels] output is required')
  if (out.shape[0] !== kernelFrames(manifest))
    throw new RangeError('KernelManifest: the output frame count must equal tau\'s')
  const count = out.shape[1]!
  if (!Number.isSafeInteger(count) || count < 1
      || count > MAX_KERNEL_OUTPUT_CHANNELS)
    throw new RangeError(
      `KernelManifest: output channels must be an integer in [1, ${MAX_KERNEL_OUTPUT_CHANNELS}]`)
  return count
}
