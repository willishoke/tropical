/**
 * params.ts — live param writes under the manifest's disciplines
 * (design/host-param-dispatch.md, the normative contract).
 *
 * A host never picks a write verb: each param input of `tropical_kernel_1`
 * carries its discipline, and a write dispatches on it. This writer
 * implements the two disciplines a param port declares on itself:
 *
 *   - `raw` (or no discipline): write the value.
 *   - `anchor`: a frequency whose phase is `inc·C + φ`, `C` the clock the
 *     phasor reads (the frame clock, or the master clock the discipline's
 *     `clock` names). Changing `inc` alone jumps the phase by `Δinc·C`; the
 *     write bumps the `#phase` companion by exactly that jump, in the
 *     kernel's own integer arithmetic, so the waveform stays continuous.
 *
 * Anything else is refused by name. An integer param (an `i32`/`i64` word with
 * `frac_bits`) is written raw only: `anchor` re-anchors in float arithmetic,
 * and the compiler never attaches it to an integer port; a manifest that did
 * is refused here rather than written wrong, and so is a value outside an
 * integer param's format (its word would saturate). An `f32` param (a
 * binary32 artifact) is likewise written raw only, and a value past the
 * binary32 range is refused rather than written as an infinity. A glide's discipline is declared on a base
 * name that has no port of its own (its ports are the `#v0/#v1/#t0` companions),
 * so a host reading only the kernel type cannot see it; the player does not
 * guess one.
 */
import type { KernelManifest, KernelPort } from './manifest.js'
import { isWordPort, kernelParams, paramRangeError } from './manifest.js'
import type { WasmKernel } from './kernel.js'

type PortDiscipline = { discipline: string; companions?: string[]; clock?: string }

const TWO_POW_32 = 4294967296
const MASK32 = 0xffffffffn

/** The kernel's f64 → i64 conversion (`fptosi.sat`): truncate, saturate. */
function truncI64(x: number): bigint {
  if (Number.isNaN(x)) return 0n
  if (x >= 9223372036854775808) return (1n << 63n) - 1n
  if (x < -9223372036854775808) return -(1n << 63n)
  return BigInt(Math.trunc(x))
}

/** The signed i64 representative of `x` mod 2⁶⁴. */
function wrapI64(x: bigint): bigint {
  return BigInt.asIntN(64, x)
}

/** The phasor increment the kernel computes: `trunc(f·2³²/rate)`. */
function phaseIncrement(freq: number, rate: number): bigint {
  return truncI64((freq * TWO_POW_32) / rate)
}

/** `⌊scale·clk/2³²⌋` mod 2⁶⁴, the kernel's `scaleClockQ` (BigInt `>>` is
 *  arithmetic and `&` two's-complement, as the kernel's). */
function scaleClockQ(scale: bigint, clk: bigint): bigint {
  const hi = clk >> 32n
  const lo = clk & MASK32
  const pLoLo = (scale * (lo & 0xffffn)) >> 16n
  const pLo = (scale * (lo >> 16n) + pLoLo) >> 16n
  return wrapI64(scale * hi + pLo)
}

/** The phasor's phase on ℤ/2³² before its offset: `⌊inc·clk/2¹⁶⌋ mod 2³²`. */
function phaseQ32(inc: bigint, clk: bigint): bigint {
  return (inc * (clk >> 16n) + ((inc * (clk & 0xffffn)) >> 16n)) & MASK32
}

export class ParamWriter {
  private readonly index = new Map<string, number>()
  private readonly disciplines = new Map<string, PortDiscipline>()
  private readonly words = new Set<string>()
  private readonly binary32 = new Set<string>()
  private readonly ports: KernelPort[]

  constructor(private readonly manifest: KernelManifest) {
    this.ports = kernelParams(manifest)
    this.ports.forEach((port, j) => {
      this.index.set(port.name, j)
      if (isWordPort(port)) this.words.add(port.name)
      if (port.elem === 'f32') this.binary32.add(port.name)
      const d = (port as { discipline?: PortDiscipline }).discipline
      if (d) this.disciplines.set(port.name, d)
    })
  }

  /** The disciplines this writer can apply to `name` (a `param:` slot name),
   *  or the reason it cannot. */
  check(name: string): string | null {
    if (!this.index.has(name)) return `no param input '${name}' in this kernel`
    if (name.includes('#'))
      return `param '${name}' is a discipline companion; it is written through its base param, not directly`
    const d = this.disciplines.get(name)?.discipline ?? 'raw'
    if (d !== 'raw' && this.words.has(name))
      return `param '${name}' is an integer word; discipline '${d}' is not defined on it (integer params are written raw)`
    if (d !== 'raw' && this.binary32.has(name))
      return `param '${name}' is f32; discipline '${d}' is not defined on it (a binary32 artifact's params are written raw)`
    const companion = this.disciplines.get(name)?.companions?.[0]
    if (d === 'anchor' && companion !== undefined && this.words.has(`param:${companion}`))
      return `param '${name}': its anchor companion is an integer word; anchor is not defined on it`
    if (d !== 'raw' && d !== 'anchor')
      return `param '${name}' has discipline '${d}'; the player writes raw and anchor params only`
    return null
  }

  /** Write `value` to `name` as of frame `now` (the first frame the write is
   *  audible at: the kernel's next block). */
  write(kernel: WasmKernel, name: string, value: number, now: number): void {
    const problem = this.check(name)
    if (problem) throw new RangeError(problem)
    const j = this.index.get(name)!
    const range = paramRangeError(this.ports[j]!, value)
    if (range) throw new RangeError(`param '${name}': ${range}`)
    const d = this.disciplines.get(name)
    if (d?.discipline === 'anchor') {
      const companion = d.companions?.[0]
      const k = companion === undefined ? undefined : this.index.get(`param:${companion}`)
      if (k !== undefined) {
        // off₁ = off₀ + Φ(inc₀, C) − Φ(inc₁, C) on ℤ/2³², exactly; φ stays a
        // multiple of 2⁻³² in [0, 1).
        const rate = this.manifest.rate
        const clk = this.clockAt(kernel, d.clock, now)
        const inc0 = phaseIncrement(kernel.getParam(j), rate)
        const inc1 = phaseIncrement(value, rate)
        const phase = truncI64(kernel.getParam(k) * TWO_POW_32)
        const bumped = (phase + phaseQ32(inc0, clk) - phaseQ32(inc1, clk)) & MASK32
        kernel.setParam(k, Number(bumped) / TWO_POW_32)
      }
      // A missing companion degrades to raw (host-param-dispatch.md, `anchor`).
    }
    kernel.setParam(j, value)
  }

  /** The Q48.16 clock an anchored phasor reads at frame `now`: the frame
   *  clock, or the master clock `trunc(tau_base·rate·2¹⁶) +
   *  scaleClockQ(trunc(velocity·2³²), now·2¹⁶)` when `clock` names the
   *  velocity param (its velocity discipline names the origin). */
  private clockAt(kernel: WasmKernel, clock: string | undefined, now: number): bigint {
    const frame = wrapI64(BigInt(now) << 16n)
    if (clock === undefined) return frame
    const velocityName = `param:${clock}`
    const vi = this.index.get(velocityName)
    const origin = this.disciplines.get(velocityName)?.companions?.[0]
    const ti = origin === undefined ? undefined : this.index.get(`param:${origin}`)
    if (vi === undefined || ti === undefined)
      throw new RangeError(`anchor clock '${clock}' has no velocity/origin param in this kernel`)
    const originQ = truncI64(kernel.getParam(ti) * this.manifest.rate * 65536)
    const velocityQ = truncI64(kernel.getParam(vi) * TWO_POW_32)
    return wrapI64(originQ + scaleClockQ(velocityQ, frame))
  }
}
