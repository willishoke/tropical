/**
 * curve.ts — one cycle of a kernel output channel, evaluated by the kernel
 * itself (`WasmKernel.renderAt`) for the player's display. Nothing here
 * computes a sample: every point is `f(τ)` at a chosen clock value.
 */
import { CLOCK_FRAC_BITS, type WasmKernel } from '../runtime/index.js'

/** What a patch's manifest entry declares to display: output `channel` over
 *  one cycle of `rate / freq` frames, `freq` fixed or the current value of
 *  the param `freqParam` (named without the `param:` prefix). */
export type CycleDisplay = { channel: number; freq?: number; freqParam?: string }

/** Renderer → host: `points` samples of one cycle starting at kernel frame
 *  `frame`. */
export type Curve = { type: 'curve'; frame: number; freq: number; data: Float64Array }

/** The cycle's frequency now: fixed, or read from the kernel's own param
 *  cell (the value it renders with). `null` when there is no positive one. */
export function displayFreq(k: WasmKernel, d: CycleDisplay): number | null {
  let f = d.freq
  if (d.freqParam !== undefined) {
    const i = k.paramIndex(`param:${d.freqParam}`)
    if (i < 0) return null
    f = k.getParam(i)
  }
  return f !== undefined && f > 0 && Number.isFinite(f) ? f : null
}

/** `points` values of output `channel` across one cycle of `rate / freq`
 *  frames from kernel frame `frame`: point `j` is the kernel at τ =
 *  `frame + j · cycle / points` frames, rounded to the clock's 2⁻¹⁶-frame
 *  grid. */
export function cycleCurve(k: WasmKernel, rate: number, frame: number, channel: number,
    freq: number, points: number): Float64Array {
  const one = 2 ** CLOCK_FRAC_BITS
  const step = (rate / freq) * one / points
  const tau0 = BigInt(frame) << BigInt(CLOCK_FRAC_BITS)
  const taus = BigInt64Array.from({ length: points }, (_, j) => tau0 + BigInt(Math.round(j * step)))
  const all = k.renderAt(taus)
  const C = k.outputChannelCount
  const out = new Float64Array(points)
  for (let j = 0; j < points; j++) out[j] = all[j * C + channel]!
  return out
}
