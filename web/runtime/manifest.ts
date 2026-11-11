/**
 * manifest.ts — the runtime's consumption contract: `tropical_kernel_1`.
 *
 * A kernel is one typed pure function (design/typed-kernel.md): a record of
 * named, static-shaped inputs — `tau` (the Q48.16 clock, shape `[frames]`)
 * and one scalar per param — and one output of shape `[frames, channels]`,
 * frame-major. A param is a float (`f64`, or `f32` in a binary32 artifact),
 * or an integer word (`i32`, `i64`) with `frac_bits`, its fixed-point format:
 * word `w` means `w · 2^-frac_bits`. The output is a float at the same format,
 * or `i32` with `frac_bits` when the program's output is an `int32`. The
 * format the artifact realizes `float` in is `realization.float` (absent:
 * `f64`); every float port is declared at it. This type IS the boundary between
 * tropical (the producer, via `diffcli compile-kernel`) and any consumer
 * (this player today; an extracted `@tropical/runtime-wasm` package
 * tomorrow). The manifest also nests the compiler's plan for its own gates;
 * the runtime never reads it.
 */
export type KernelElem = 'f64' | 'f32' | 'i64' | 'i64q16' | 'i32'

/** The IEEE formats an artifact may realize the `float` kind in. */
export type FloatFormat = 'f64' | 'f32'

/** The clock rail's fraction bits: τ is a signed i64 with this many fraction
 *  bits (Q48.16), whole sample `n` is `n << CLOCK_FRAC_BITS`. Mirrors
 *  `lean/Tropical/Ir/ClockFormat.lean`; the manifest names it as τ's elem. */
export const CLOCK_FRAC_BITS = 16
export const TAU_ELEM = 'i64q16'
export type KernelRole = 'param' | 'signal' | 'table'

export type KernelPort = {
  name: string
  elem: KernelElem
  shape: number[]
  role?: KernelRole
  /** Seed value a host writes before the first call (params). */
  default?: number
  /** How a host writes this param (`host-param-dispatch.md`): `raw`,
   *  `anchor` (with its `#phase` companion), … Absent means `raw`. */
  discipline?: { discipline: string; glide_dur_sec?: number; companions?: string[] }
  /** Output channel source names (outputs of shape [frames, C]). */
  channels?: string[]
  /** An integer port's fraction bits: word `w` means `w · 2^-frac_bits`. A
   *  param's `default` is then the word itself. */
  frac_bits?: number
}

export type KernelManifest = {
  schema: 'tropical_kernel_1'
  /** The rate the kernel was compiled for (Hz); a constant of the artifact. */
  rate: number
  /** `tau` first, then the params in port order. */
  inputs: KernelPort[]
  /** Exactly one output: `out`, a float or an `i32` word, shape
   *  `[frames, channels]`. */
  outputs: KernelPort[]
  /** How the artifact was realized. `float` is written only when it is not
   *  `f64` (a binary32 artifact); this host reads `f64` and `f32` and
   *  refuses any other. */
  realization?: { mode?: string; register_count?: number; float?: string }
  /** The compiler's plan, nested for its gates. The one field this host
   *  reads is `param_disciplines`, to refuse at load a binary32 artifact
   *  that pairs `f32` with a non-raw write discipline. */
  plan?: { param_disciplines?: Array<{ name: string; discipline: string; companions?: string[] }> }
}

export const MAX_KERNEL_OUTPUT_CHANNELS = 64

/** The format the artifact realizes `float` in: `realization.float`, `f64`
 *  when absent. Any other value is refused by name, and so is a float port
 *  declared at a different format (a host that read it at its declared
 *  element would bind a cell the kernel reads at another width). */
export function kernelFloatFormat(manifest: KernelManifest): FloatFormat {
  const declared = manifest.realization?.float
  if (declared !== undefined && declared !== 'f64' && declared !== 'f32')
    throw new RangeError(`KernelManifest: unknown float format '${String(declared)}'; this host reads f64 and f32 artifacts`)
  const fmt: FloatFormat = declared ?? 'f64'
  for (const port of [...manifest.inputs.slice(1), ...manifest.outputs])
    if ((port.elem === 'f64' || port.elem === 'f32') && port.elem !== fmt)
      throw new RangeError(`KernelManifest: port '${port.name}' is ${port.elem} in an artifact whose float format is ${fmt}`)
  if (fmt === 'f32') {
    // A binary32 artifact's params are written raw (the compiler refuses
    // anything else): glide, anchor and velocity re-anchor on sample counts
    // and phases binary32 does not hold exactly. Refused once, here; a
    // glided knob has no port of its own, so the plan's table is read too.
    const named = [
      ...(manifest.plan?.param_disciplines ?? []),
      ...manifest.inputs.slice(1).flatMap((p) => p.discipline ? [{ name: p.name, discipline: p.discipline.discipline }] : []),
    ]
    for (const d of named)
      if (d.discipline !== 'raw')
        throw new RangeError(`KernelManifest: param '${d.name}' has the '${d.discipline}' discipline in a binary32 artifact; a binary32 artifact's params are written raw`)
  }
  return fmt
}

/** The output's word scale: `2^-frac_bits` for an `i32` output (the host
 *  widens each word to the real value it means), or `null` for a float
 *  output. */
export function kernelOutputWordScale(manifest: KernelManifest): number | null {
  const out = manifest.outputs[0]
  return out?.elem === 'i32' ? 2 ** -(out.frac_bits ?? 0) : null
}

/** Bytes per output element: 8 for `f64`, 4 for `f32` and `i32`. */
export function kernelOutputElemBytes(manifest: KernelManifest): number {
  return manifest.outputs[0]?.elem === 'f64' ? 8 : 4
}

/** The static frame count per call, from the `tau` input's shape. */
export function kernelFrames(manifest: KernelManifest): number {
  const tau = manifest.inputs[0]
  if (!tau || tau.name !== 'tau' || tau.elem !== TAU_ELEM || tau.shape.length !== 1)
    throw new RangeError(`KernelManifest: the first input must be tau:${TAU_ELEM}[frames]`)
  const frames = tau.shape[0]!
  if (!Number.isSafeInteger(frames) || frames < 1)
    throw new RangeError('KernelManifest: frames must be a positive integer')
  return frames
}

/** The param inputs (everything after `tau`), in port order. Each is a
 *  scalar `f64`, `f32`, `i32` or `i64`; anything else is refused here, before
 *  a host lays out memory for it. */
export function kernelParams(manifest: KernelManifest): KernelPort[] {
  kernelFloatFormat(manifest)
  const params = manifest.inputs.slice(1)
  for (const port of params) {
    if (port.elem !== 'f64' && port.elem !== 'f32' && port.elem !== 'i32' && port.elem !== 'i64')
      throw new RangeError(`KernelManifest: param '${port.name}' is ${port.elem}; this host binds f64, f32, i32 and i64 params`)
    if (port.shape.length !== 0)
      throw new RangeError(`KernelManifest: param '${port.name}' is not a scalar`)
    if (port.elem === 'f32' && port.default !== undefined && Math.fround(port.default) !== port.default)
      // The default is the binary32 the artifact was compiled with, written
      // exactly; one that is not a binary32 would be rounded here.
      throw new RangeError(`KernelManifest: f32 param '${port.name}' default ${port.default} is not a binary32 value`)
    if (isWordPort(port) && (port.frac_bits ?? 0) > (port.elem === 'i32' ? 32 : 64))
      throw new RangeError(`KernelManifest: param '${port.name}' has more fraction bits than its word`)
    if (isWordPort(port) && port.default !== undefined) {
      const d = port.default
      // An integer port's default is its word. A JSON number reaches this
      // host as a double, which carries every i32 word but an i64 word only
      // up to 2^53: past that the parsed value may not be the word written
      // (INT64_MAX parses as 2^63), so it is refused, not rounded or wrapped.
      // The native host reads the word exactly; the two never disagree.
      if (port.elem === 'i64' ? !Number.isSafeInteger(d)
          : !(Number.isInteger(d) && d >= -2147483648 && d <= 2147483647))
        throw new RangeError(port.elem === 'i64'
          ? `KernelManifest: param '${port.name}' is i64 with default word ${d}, past ±(2^53 − 1): a JSON number does not carry it exactly in this host`
          : `KernelManifest: param '${port.name}' default ${d} is not an i32 word`)
    }
  }
  return params
}

/** The integer format's name and real range, as the compiler spells them:
 *  `int32 Q16.16` and `[-32768, 32768)`. */
function describeWord(port: KernelPort): string {
  const w = port.elem === 'i32' ? 32 : 64, n = port.frac_bits ?? 0
  return `${port.elem === 'i32' ? 'int32' : 'int'} Q${w - n}.${n}`
}
function wordRangeLabel(port: KernelPort): string {
  const k = (port.elem === 'i32' ? 32 : 64) - 1 - (port.frac_bits ?? 0)
  const end = k >= 0 ? (1n << BigInt(k)).toString() : '0.5'
  return `[-${end}, ${end})`
}

/** Why value `x` does not fit an integer port, or `null` when it does: its
 *  word `x · 2^frac_bits` must lie in the word's range (it is then truncated
 *  toward zero). Past the range the word would saturate, so a write is
 *  refused instead, as the compiler's `set_param` and the native host refuse
 *  it. */
export function wordRangeError(port: KernelPort, x: number): string | null {
  if (!isWordPort(port)) return null
  const scaled = x * 2 ** (port.frac_bits ?? 0)
  const hi = 2 ** ((port.elem === 'i32' ? 32 : 64) - 1)
  if (scaled >= -hi && scaled < hi) return null
  return `value ${x} is outside the range of ${describeWord(port)}, ${wordRangeLabel(port)}`
}

/** The largest finite binary32, `(2 − 2^-23) · 2^127`. */
const BINARY32_MAX = 3.4028234663852886e38

/** Why value `x` does not fit an `f32` port, or `null` when it does: every
 *  finite value rounds to the nearest binary32, except one past the largest
 *  finite binary32, which would become an infinity and is refused, as the
 *  native host refuses it. */
export function binary32RangeError(port: KernelPort, x: number): string | null {
  if (port.elem !== 'f32' || !Number.isFinite(x) || Number.isFinite(Math.fround(x))) return null
  return `value ${x} is outside the range of binary32, |x| <= ${BINARY32_MAX}`
}

/** Why value `x` cannot be written to a param port, or `null`. */
export function paramRangeError(port: KernelPort, x: number): string | null {
  return wordRangeError(port, x) ?? binary32RangeError(port, x)
}

/** Is this port an integer word? */
export function isWordPort(port: KernelPort): boolean {
  return port.elem === 'i32' || port.elem === 'i64'
}

/** The word a host writes for value `x` at an integer port: `x · 2^frac_bits`
 *  truncated toward zero and saturated to the word, NaN → 0 — the kernel's
 *  own float → integer conversion (Lean `Kernel.encodeWord`, C++
 *  `tropical_port::encode_word`). The scaling by a power of two is exact. */
export function encodeWord(port: KernelPort, x: number): number | bigint {
  if (Number.isNaN(x)) return port.elem === 'i64' ? 0n : 0
  const t = Math.trunc(x * 2 ** (port.frac_bits ?? 0))
  if (port.elem === 'i64') {
    if (t >= 2 ** 63) return (1n << 63n) - 1n
    if (t <= -(2 ** 63)) return -(1n << 63n)
    return BigInt(t)
  }
  return t >= 2147483647 ? 2147483647 : t <= -2147483648 ? -2147483648 : t
}

/** The real value word `w` of an integer port means. */
export function wordValue(port: KernelPort, w: number | bigint): number {
  return Number(w) * 2 ** -(port.frac_bits ?? 0)
}

/** Resolve and validate the fixed output width once, off the audio callback. */
export function kernelOutputChannelCount(manifest: KernelManifest): number {
  kernelFloatFormat(manifest)
  const out = manifest.outputs[0]
  if (!out || manifest.outputs.length !== 1
      || (out.elem !== 'f64' && out.elem !== 'f32' && out.elem !== 'i32')
      || out.shape.length !== 2)
    throw new RangeError('KernelManifest: exactly one f64, f32 or i32 [frames, channels] output is required')
  if (out.shape[0] !== kernelFrames(manifest))
    throw new RangeError('KernelManifest: the output frame count must equal tau\'s')
  const count = out.shape[1]!
  if (!Number.isSafeInteger(count) || count < 1
      || count > MAX_KERNEL_OUTPUT_CHANNELS)
    throw new RangeError(
      `KernelManifest: output channels must be an integer in [1, ${MAX_KERNEL_OUTPUT_CHANNELS}]`)
  return count
}
