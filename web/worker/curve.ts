/**
 * curve.ts — one cycle of a kernel output channel, evaluated by the kernel
 * itself (`WasmKernel.renderAt`) for the player's display. Nothing here
 * computes a sample: every point is `f(τ)` at a chosen clock value.
 */
import { CLOCK_FRAC_BITS, type WasmKernel } from '../runtime/index.js'

/** What a patch's manifest entry declares to display: output `channel` (an
 *  audio channel or a tap) over one cycle of a clock of frequency `freq`,
 *  fixed or the current value of the param `freqParam` (named without the
 *  `param:` prefix). `label` captions it. */
export type CycleDisplay = { channel: number; freq?: number; freqParam?: string; label?: string }

/** Renderer → host: curve `index` of display list `generation` — `data`
 *  holds `points` samples of cycle `cycle` of the clock, starting at clock
 *  word `tau0` (kernel τ, Q48.16, as a decimal string). */
export type Curve = {
  type: 'curve'; generation: number; index: number; cycle: number; tau0: string; freq: number
  data: Float64Array
}

/** The most points one curve may ask for. */
export const MAX_CURVE_POINTS = 8192

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

/** The phasor increment at `freq`: `trunc(freq · 2³² / rate)`, computed as
 *  the stdlib clock computes it (`EmitArrow.phasorPhaseSig`), so the clock's
 *  phase at clock word τ is `⌊inc · τ / 2¹⁶⌋ mod 2³²`. */
export function phasorInc(freq: number, rate: number): bigint {
  return BigInt(Math.trunc((freq * 2 ** 32) / rate))
}

/** The clock words of `points` evenly spaced phases of the cycle containing
 *  kernel frame `frame`, for a clock of increment `inc` with phase 0 at τ = 0
 *  (a `ClockPhasor` with offset 0 on τ). Point `j` is the first word at or
 *  after phase `cycle + j / points`: `⌈(cycle·points + j) · 2⁴⁸ / (inc ·
 *  points)⌉`, exact in BigInt. Its phase is then that phase plus less than
 *  one word's worth, `inc / 2¹⁶` phase units, so point 0 is the cycle's phase
 *  0 on the Q48.16 grid. */
export function cycleTaus(inc: bigint, frame: number, points: number): { cycle: bigint; taus: BigInt64Array } {
  const F = BigInt(CLOCK_FRAC_BITS)
  const cycle = (inc * (BigInt(frame) << F)) >> 48n
  const N = BigInt(points), den = inc * N
  const taus = BigInt64Array.from({ length: points }, (_, j) => {
    const num = (cycle * N + BigInt(j)) << 48n
    return (num + den - 1n) / den
  })
  return { cycle, taus }
}

/** Output `channel` at `taus`: the kernel's own values, one per τ. */
export function channelAt(k: WasmKernel, taus: BigInt64Array, channel: number): Float64Array {
  const all = k.renderAt(taus)
  const C = k.outputChannelCount
  const out = new Float64Array(taus.length)
  for (let j = 0; j < taus.length; j++) out[j] = all[j * C + channel]!
  return out
}

/** One cycle of output `channel`: the clock cycle (frequency `freq` at the
 *  kernel's `rate`) that contains kernel frame `frame`, from its phase 0, at
 *  `points` points. `null` when the clock does not advance. */
export function cycleCurve(k: WasmKernel, rate: number, frame: number, channel: number,
    freq: number, points: number): { cycle: bigint; taus: BigInt64Array; data: Float64Array } | null {
  if (!Number.isInteger(points) || points < 2 || points > MAX_CURVE_POINTS)
    throw new RangeError(`curve points must be an integer in [2, ${MAX_CURVE_POINTS}]`)
  if (!Number.isInteger(channel) || channel < 0 || channel >= k.outputChannelCount)
    throw new RangeError(`no kernel output channel ${channel}`)
  const inc = phasorInc(freq, rate)
  if (inc <= 0n) return null
  const { cycle, taus } = cycleTaus(inc, frame, points)
  return { cycle, taus, data: channelAt(k, taus, channel) }
}
