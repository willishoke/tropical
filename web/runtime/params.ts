/**
 * params.ts — live param writes under the manifest's disciplines
 * (design/host-param-dispatch.md, the normative contract).
 *
 * A host never picks a write verb: each param input of `tropical_kernel_1`
 * carries its discipline, and a write dispatches on it. This writer
 * implements the two disciplines a param port declares on itself:
 *
 *   - `raw` (or no discipline): write the value.
 *   - `anchor`: a frequency whose phase is `inc·τ + φ`. Changing `inc` alone
 *     jumps the phase by `Δinc·τ`; the write bumps the `#phase` companion by
 *     exactly that jump, in the phasor's own quantized increments, so the
 *     waveform stays continuous at any τ.
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

type PortDiscipline = { discipline: string; companions?: string[] }

const TWO_POW_32 = 4294967296
const TWO_POW_32N = 4294967296n

/** The phasor increment the kernel computes: `trunc(f·2³²/rate)`. */
function phaseIncrement(freq: number, rate: number): bigint {
  return BigInt(Math.trunc((freq * TWO_POW_32) / rate))
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
        // φ is a multiple of 2⁻³² in [0, 1): keep it on that grid, exactly.
        const inc0 = phaseIncrement(kernel.getParam(j), this.manifest.rate)
        const inc1 = phaseIncrement(value, this.manifest.rate)
        const phase = BigInt(Math.round(kernel.getParam(k) * TWO_POW_32))
        const bumped = (((phase + (inc0 - inc1) * BigInt(now)) % TWO_POW_32N) + TWO_POW_32N) % TWO_POW_32N
        kernel.setParam(k, Number(bumped) / TWO_POW_32)
      }
      // A missing companion degrades to raw (host-param-dispatch.md, `anchor`).
    }
    kernel.setParam(j, value)
  }
}
