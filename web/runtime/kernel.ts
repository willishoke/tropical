/**
 * kernel.ts — WasmKernel: instantiate + drive a Lean-emitted wasm32 kernel.
 *
 * The Web Audio analog of `engine/runtime/FlatRuntime` (native): it owns one
 * imported linear memory, holds the kernel's input record (tau and the
 * params) and its output, calls the port once per static-shaped block, and
 * applies an anti-click fade. It is a *player* — no live recompile, no
 * hot-swap (those live on the native instrument). Depends only on
 * `KernelManifest`; no compiler internals.
 */
import {
  CLOCK_FRAC_BITS, encodeWord, isWordPort, kernelFrames, kernelOutputChannelCount,
  kernelOutputWordScale, kernelParams, wordValue, type KernelManifest, type KernelPort,
} from './manifest.js'
import { computeLayout, type KernelLayout } from './layout.js'

const WASM_PAGE = 65536
/** The anti-click fade's length in frames (smoothstep, `beginFadeIn`/`Out`). */
export const FADE_FRAMES = 2048

/** `C` — the device-boundary output bound, applied in `process` (the worklet
 *  feed) and never in `render` (the equivalence surface). Must equal
 *  `kDeviceOutputBound` in engine/dac/TropicalDAC.hpp, which carries the
 *  rationale and the measurement. */
const DEVICE_OUTPUT_BOUND = 256.0

// @llvm.round = round half away from zero — the one math op with no wasm
// instruction (f64.nearest is ties-to-even), so the kernel imports it. Must
// match the native kernel's rounding bit-for-bit.
function roundTiesAway(x: number): number {
  const f = Math.floor(x), d = x - f
  return d < 0.5 ? f : d > 0.5 ? f + 1 : (x < 0 ? f : f + 1)
}

/** The minimum page count the module declares for its imported memory (0 if
 *  it imports none). The toolchain links the shadow stack first and sizes it
 *  to the kernel's own storage, so a large kernel can declare more than a
 *  fixed allocation would give it; `WebAssembly.Module.imports` does not
 *  expose limits, so read them from the import section. */
export function importedMemoryMinPages(wasmBytes: BufferSource): number {
  const b = ArrayBuffer.isView(wasmBytes)
    ? new Uint8Array(wasmBytes.buffer, wasmBytes.byteOffset, wasmBytes.byteLength)
    : new Uint8Array(wasmBytes)
  let p = 8 // magic + version
  const u32 = (): number => {
    let r = 0, shift = 0, byte: number
    do { byte = b[p++]!; r |= (byte & 0x7f) << shift; shift += 7 } while (byte & 0x80)
    return r >>> 0
  }
  const limitsMin = (): number => {
    const flags = b[p++]!
    const min = u32()
    if (flags & 1) u32()
    return min
  }
  while (p < b.length) {
    const id = b[p++]!
    const size = u32()
    const end = p + size
    if (id === 2) { // the import section
      for (let n = u32(); n > 0; n--) {
        for (let name = 0; name < 2; name++) { // module and field names
          const len = u32() // read first: `p += u32()` would add to the old p
          p += len
        }
        const kind = b[p++]!
        if (kind === 2) return limitsMin()
        if (kind === 0) u32()                   // func: type index
        else if (kind === 1) { p++; limitsMin() } // table: reftype + limits
        else if (kind === 3) p += 2             // global: valtype + mutability
      }
    }
    p = end
  }
  return 0
}

/** The port: one i32 offset per input (tau, then the params) and the output. */
type KernelFn = (...pointers: number[]) => void

export class WasmKernel {
  /** The frame index of the next block; a number is exact to 2⁵³ frames. */
  private frameIdx = 0
  /** The fade is a function of the frame, not a countdown, so a host that
   *  renders ahead can re-render any frame and get the same gain: a smoothstep
   *  from `fadeFrom` at `fadeAt` to `fadeTo` at `fadeAt + FADE_FRAMES`. */
  private fadeAt = 0
  private fadeFrom = 1
  private fadeTo = 1
  private readonly argv: number[]
  /** Views of tau (as u32 word pairs) and the output, made on first use and
   *  again only if the memory's buffer changes, so a block allocates nothing. */
  private tauWords?: Uint32Array
  private outView?: Float64Array
  /** An `i32` output's words, widened into `outView` (then a host-owned
   *  scratch) as `word · outScale` after each call. */
  private outWords?: Int32Array

  private constructor(
    private readonly memory: WebAssembly.Memory,
    private readonly kernel: KernelFn,
    private readonly layout: KernelLayout,
    readonly outputChannelCount: number,
    private readonly params: KernelPort[],
    private readonly outScale: number | null,
  ) {
    this.argv = [layout.tau, ...layout.params, layout.output]
  }

  /** The static frame count per call: the kernel's shape. */
  get frames(): number { return this.layout.frames }

  static async instantiate(
    wasmBytes: BufferSource,
    manifest: KernelManifest,
  ): Promise<WasmKernel> {
    const frames = kernelFrames(manifest)
    const outputChannelCount = kernelOutputChannelCount(manifest)
    // Region bytes are small and known; the unknown is the module's
    // __heap_base (its shadow stack + data segment). Allocate 1 MiB of slack
    // above the regions, or the module's declared minimum if a large kernel
    // frame needs more, then grow once if heapBase lands higher.
    const regionBytes =
      (frames + kernelParams(manifest).length + frames * outputChannelCount + 8) * 8
    const initialPages = Math.max(
      importedMemoryMinPages(wasmBytes),
      Math.ceil(((1 << 20) + regionBytes) / WASM_PAGE))
    const memory = new WebAssembly.Memory({ initial: initialPages })

    const module = await WebAssembly.compile(wasmBytes)
    const instance = await WebAssembly.instantiate(module, {
      env: { memory, round: roundTiesAway },
    })
    const kernel = instance.exports.tropical_kernel as KernelFn
    const heapBase = (instance.exports.__heap_base as WebAssembly.Global).value as number

    const layout = computeLayout(manifest, heapBase)
    const needPages = Math.ceil(layout.endByte / WASM_PAGE)
    if (needPages > initialPages) memory.grow(needPages - initialPages)

    const k = new WasmKernel(memory, kernel, layout, outputChannelCount,
      kernelParams(manifest), kernelOutputWordScale(manifest))
    k.initParams()
    return k
  }

  /** Seed every param from its default (an integer port's default is its
   *  word). */
  private initParams(): void {
    this.params.forEach((port, j) => {
      if (isWordPort(port)) this.writeWord(j, port, port.default ?? 0)
      else new Float64Array(this.memory.buffer)[this.layout.params[j]! >> 3] = port.default ?? 0
    })
  }

  private writeWord(index: number, port: KernelPort, w: number | bigint): void {
    const cell = this.layout.params[index]!
    if (port.elem === 'i64') new BigInt64Array(this.memory.buffer)[cell >> 3] = BigInt(w)
    else new Int32Array(this.memory.buffer)[cell >> 2] = Number(w)
  }

  /** Fade in from silence, starting at frame `at` (default: the next block). */
  beginFadeIn(at: number = this.frameIdx): void {
    this.fadeFrom = 0; this.fadeTo = 1; this.fadeAt = at
  }
  /** Fade out from the gain at frame `at` (default: the next block) and stay
   *  silent. */
  beginFadeOut(at: number = this.frameIdx): void {
    this.fadeFrom = this.gainAt(at); this.fadeTo = 0; this.fadeAt = at
  }

  /** The fade gain at `frame`. */
  gainAt(frame: number): number {
    if (frame >= this.fadeAt + FADE_FRAMES) return this.fadeTo
    if (frame <= this.fadeAt) return this.fadeFrom
    const t = (frame - this.fadeAt) / FADE_FRAMES
    return this.fadeFrom + (this.fadeTo - this.fadeFrom) * t * t * (3 - 2 * t)
  }

  /** The first frame from which `process` is silent for good (a finished
   *  fade-out), or Infinity. */
  get mutedFrom(): number {
    return this.fadeTo === 0 ? this.fadeAt + FADE_FRAMES : Infinity
  }

  /** Write a param input (port order after `tau`). This is how a host with
   *  live control drives a knob; `ParamWriter` applies the disciplines. An
   *  integer port takes the real value and stores its word (`encodeWord`). */
  setParam(index: number, value: number): void {
    const cell = this.layout.params[index]
    const port = this.params[index]
    if (cell === undefined || port === undefined) throw new RangeError(`no param input ${index}`)
    if (isWordPort(port)) this.writeWord(index, port, encodeWord(port, value))
    else new Float64Array(this.memory.buffer)[cell >> 3] = value
  }

  /** Read a param input's current value (port order after `tau`): for an
   *  integer port, the real value its word means. */
  getParam(index: number): number {
    const cell = this.layout.params[index]
    const port = this.params[index]
    if (cell === undefined || port === undefined) throw new RangeError(`no param input ${index}`)
    if (port.elem === 'i64') return wordValue(port, new BigInt64Array(this.memory.buffer)[cell >> 3]!)
    if (port.elem === 'i32') return wordValue(port, new Int32Array(this.memory.buffer)[cell >> 2]!)
    return new Float64Array(this.memory.buffer)[cell >> 3]!
  }

  /** Run the kernel once over its static block, with `tau` the clock of each
   *  frame from `frameIdx`, and return the frame-major output view. Each τ is
   *  written as its two little-endian u32 words of `idx << CLOCK_FRAC_BITS`:
   *  the low word holds the index's low bits above a zero fraction, the high
   *  word the rest (a u32 store wraps mod 2³², which is the i64's own wrap at
   *  the rail's horizon). The same 64 bits a `BigInt64Array` stores, without a
   *  BigInt per frame on the audio thread. */
  private renderBlock(frameIdx: number): Float64Array {
    const L = this.layout
    if (this.tauWords?.buffer !== this.memory.buffer) {
      this.tauWords = new Uint32Array(this.memory.buffer, L.tau, 2 * L.frames)
      if (this.outScale === null) {
        this.outView = new Float64Array(
          this.memory.buffer, L.output, L.frames * this.outputChannelCount)
      } else {
        this.outWords = new Int32Array(
          this.memory.buffer, L.output, L.frames * this.outputChannelCount)
        this.outView ??= new Float64Array(L.frames * this.outputChannelCount)
      }
    }
    const words = this.tauWords!
    const lowSpan = 2 ** (32 - CLOCK_FRAC_BITS)
    const one = 2 ** CLOCK_FRAC_BITS
    for (let s = 0; s < L.frames; s++) {
      const idx = frameIdx + s
      words[2 * s] = (idx % lowSpan) * one
      words[2 * s + 1] = Math.floor(idx / lowSpan) // a u32 store wraps mod 2³²
    }
    this.kernel(...this.argv)
    if (this.outScale !== null) {
      // An i32 output: widen each word to the real value it means.
      const words = this.outWords!, out = this.outView!, scale = this.outScale
      for (let k = 0; k < words.length; k++) out[k] = words[k]! * scale
    }
    return this.outView!
  }

  /** Render `n` frames and return a frame-major, interleaved copy of
   *  `n * outputChannelCount` f64 samples. The kernel's shape is static, so
   *  `n` frames are `ceil(n / frames)` calls, the last one's tail discarded:
   *  a closed form has no cost beyond the arithmetic. This is the kernel's
   *  *native* output — what the WASM≡JIT equivalence gate compares against
   *  `render-bytes`. */
  render(n: number): Float64Array {
    if (!Number.isSafeInteger(n) || n < 0)
      throw new RangeError('render frame count must be a non-negative integer')
    const out = new Float64Array(n * this.outputChannelCount)
    this.renderInto(out, n)
    return out
  }

  /** `render` into a caller-owned buffer of at least `n * outputChannelCount`
   *  f64s, allocating nothing (an audio-thread or per-frame caller). */
  renderInto(out: Float64Array, n: number): void {
    if (!Number.isSafeInteger(n) || n < 0)
      throw new RangeError('render frame count must be a non-negative integer')
    const C = this.outputChannelCount
    if (out.length < n * C)
      throw new RangeError('renderInto buffer is shorter than n * channels')
    for (let done = 0; done < n; done += this.layout.frames) {
      const block = this.renderBlock(this.frameIdx + done)
      const take = Math.min(this.layout.frames, n - done)
      out.set(block.subarray(0, take * C), done * C)
    }
    this.frameIdx += n
  }

  /** Move the clock: the next `render`/`process` starts at frame `frameIdx`.
   *  The kernel is a pure function of τ, so any frame costs the same as the
   *  next one — this is how a second consumer (a visual) evaluates the kernel
   *  at its own τ instead of receiving samples. */
  seek(frameIdx: number): void {
    if (!Number.isSafeInteger(frameIdx) || frameIdx < 0)
      throw new RangeError('seek frame must be a non-negative integer')
    this.frameIdx = frameIdx
  }

  /** The frame index the next block starts at. */
  get frame(): number { return this.frameIdx }

  /** Render `n` frames into planar host outputs (f32 audio) with the anti-click
   *  fade and the device-boundary clamp. Mono kernels are upmixed to every host
   *  channel; wider kernels map each compact channel independently.
   *
   *  The clamp lives HERE and not in `render` on purpose: `render` returns the
   *  kernel's own f64 output — the value of `f(τ)` — and that is the surface the
   *  WASM≡JIT equivalence gate compares against `diffcli render-bytes`. Bounding
   *  it there would make the artifact lie about the function it encodes. This
   *  method is the worklet feed: the point where a value becomes sound, and so
   *  the point where a sound-safety bound belongs. Exact mirror of the native
   *  side, where `FlatRuntime.outputBuffer` is honest and `TropicalDAC`'s
   *  callback clamps — see `kDeviceOutputBound` in engine/dac/TropicalDAC.hpp
   *  for `C`, its measurement, and the NaN/±0 argument for spelling it as
   *  ordered-compare + select rather than Math.min/Math.max.
   *
   *  It runs on the audio thread, so it reads each block's output view in
   *  place rather than going through `render`'s copy, and allocates nothing. */
  process(outputs: readonly Float32Array[], n: number): void {
    if (outputs.length === 0)
      throw new RangeError('process requires at least one host output channel')
    if (this.outputChannelCount > 1
        && outputs.length < this.outputChannelCount)
      throw new RangeError(
        `kernel requires ${this.outputChannelCount} host output channels; got ${outputs.length}`)
    for (const output of outputs)
      if (output.length < n)
        throw new RangeError('host output channel is shorter than the frame count')

    const C = this.outputChannelCount
    if (C > 1)
      for (let channel = C; channel < outputs.length; channel++)
        outputs[channel]!.fill(0, 0, n)

    // A block's tail past `n` is discarded and recomputed by the next call,
    // as in `render`: the closed form has no state to keep in step.
    for (let frame = 0; frame < n;) {
      const block = this.renderBlock(this.frameIdx)
      const take = Math.min(this.layout.frames, n - frame)
      const steady = this.frameIdx >= this.fadeAt + FADE_FRAMES ? this.fadeTo : -1
      for (let s = 0; s < take; s++, frame++) {
        const gain = steady >= 0 ? steady : this.gainAt(this.frameIdx + s)

        if (C === 1) {
          const v = block[s]! * gain
          const lo = v > -DEVICE_OUTPUT_BOUND ? v : -DEVICE_OUTPUT_BOUND
          const sample = lo < DEVICE_OUTPUT_BOUND ? lo : DEVICE_OUTPUT_BOUND
          for (const output of outputs) output[frame] = sample
          continue
        }
        for (let channel = 0; channel < C; channel++) {
          const v = block[s * C + channel]! * gain
          const lo = v > -DEVICE_OUTPUT_BOUND ? v : -DEVICE_OUTPUT_BOUND
          outputs[channel]![frame] =
            lo < DEVICE_OUTPUT_BOUND ? lo : DEVICE_OUTPUT_BOUND
        }
      }
      this.frameIdx += take
    }
  }
}
