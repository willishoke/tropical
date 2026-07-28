/**
 * kernel.ts — WasmKernel: instantiate + drive a Lean-emitted wasm32 kernel.
 *
 * The Web Audio analog of `engine/runtime/FlatRuntime` (native): it owns one
 * imported linear memory, initializes compatibility/register and slot regions, calls
 * the 11-argument kernel per audio block, and applies an anti-click fade. It is
 * a *player* — no live recompile, no state-transfer hot-swap (those live on the
 * native instrument). Depends only on `KernelManifest`; no compiler internals.
 */
import type { KernelManifest } from './manifest.js'
import { computeLayout, type KernelLayout } from './layout.js'

const WASM_PAGE = 65536
const FADE_SAMPLES = 2048

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

type KernelFn = (
  inputs: number, registers: number, arrays: number, arraySizes: number,
  temps: number, sampleRate: number, startIdx: bigint, paramPtrs: number,
  output: number, bufferLen: bigint, slots: number,
) => void

export class WasmKernel {
  private startIdx = 0n
  private fadeInRem = 0
  private fadeOutRem = 0

  private constructor(
    private readonly memory: WebAssembly.Memory,
    private readonly kernel: KernelFn,
    private readonly layout: KernelLayout,
    private readonly manifest: KernelManifest,
    readonly maxBlockSize: number,
  ) {}

  static async instantiate(
    wasmBytes: BufferSource,
    manifest: KernelManifest,
    maxBlockSize = 128,
  ): Promise<WasmKernel> {
    // Region bytes are small and known; the unknown is the module's
    // __heap_base (its shadow stack + data segment). Allocate 1 MiB of slack
    // above the regions, then grow once if heapBase lands higher.
    const arrayBytes = manifest.arraySlotSizes.reduce((a, s) => a + s, 0) * 8
    const regionBytes =
      (64 + 2 * manifest.registerCount + manifest.slotCount + maxBlockSize +
       manifest.arraySlotSizes.length) * 8 + arrayBytes
    const initialPages = Math.ceil(((1 << 20) + regionBytes) / WASM_PAGE)
    const memory = new WebAssembly.Memory({ initial: initialPages })

    const module = await WebAssembly.compile(wasmBytes)
    const instance = await WebAssembly.instantiate(module, {
      env: { memory, round: roundTiesAway },
    })
    const kernel = instance.exports.tropical_kernel as KernelFn
    const heapBase = (instance.exports.__heap_base as WebAssembly.Global).value as number

    const layout = computeLayout(manifest, maxBlockSize, heapBase)
    const needPages = Math.ceil(layout.endByte / WASM_PAGE)
    if (needPages > initialPages) memory.grow(needPages - initialPages)

    const k = new WasmKernel(memory, kernel, layout, manifest, maxBlockSize)
    k.initState()
    return k
  }

  /** Initialize the compatibility register region (empty for production plan
   *  5), array sizes, and slots from their defaults. */
  private initState(): void {
    const dv = new DataView(this.memory.buffer)
    const f64 = new Float64Array(this.memory.buffer)
    const m = this.manifest
    for (let i = 0; i < m.stateInit.length; i++) {
      const v = m.stateInit[i]
      if (Array.isArray(v)) continue
      const t = m.registerTypes[i] ?? 'float'
      const off = this.layout.registers + i * 8
      if (typeof v === 'boolean') dv.setBigInt64(off, v ? 1n : 0n, true)
      else if (t === 'int') dv.setBigInt64(off, BigInt(Math.trunc(v)), true)
      else if (t === 'bool') dv.setBigInt64(off, v !== 0 ? 1n : 0n, true)
      else dv.setFloat64(off, v as number, true)
    }
    m.arraySlotSizes.forEach((sz, i) =>
      dv.setBigInt64(this.layout.arraySizes + i * 8, BigInt(sz), true))
    for (let i = 0; i < m.slotCount; i++)
      f64[(this.layout.slots >> 3) + i] = m.slotDefaults[i] ?? 0
  }

  beginFadeIn(): void { this.fadeInRem = FADE_SAMPLES; this.fadeOutRem = 0 }
  beginFadeOut(): void { this.fadeOutRem = FADE_SAMPLES; this.fadeInRem = 0 }

  /** Write a slot directly. Params are slots in the session lowering, so this
   *  is how a host with live control drives them. (Unused by the demo.) */
  setSlot(index: number, value: number): void {
    new Float64Array(this.memory.buffer)[(this.layout.slots >> 3) + index] = value
  }

  /** Run the kernel for `n` (≤ maxBlockSize) samples; return a view of the
   *  f64 output region. This is the kernel's *native* output — what the
   *  WASM≡JIT equivalence gate compares against `render-bytes`. The view is
   *  valid until the next `render`/`process`. */
  render(n: number): Float64Array {
    const L = this.layout
    this.kernel(L.inputs, L.registers, L.arrays, L.arraySizes, L.temps,
                this.manifest.sampleRate, this.startIdx, /*param_ptrs*/ 0,
                L.output, BigInt(n), L.slots)
    this.startIdx += BigInt(n)
    return new Float64Array(this.memory.buffer, L.output, n)
  }

  /** Render `n` samples into `out` (f32 audio) with the anti-click fade and the
   *  device-boundary clamp.
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
   *  ordered-compare + select rather than Math.min/Math.max. */
  process(out: Float32Array, n: number): void {
    const f64 = this.render(n)
    for (let i = 0; i < n; i++) {
      let v = f64[i]!
      if (this.fadeInRem > 0) {
        const t = 1 - this.fadeInRem / FADE_SAMPLES
        v *= t * t * (3 - 2 * t); this.fadeInRem--
      } else if (this.fadeOutRem > 0) {
        const t = this.fadeOutRem / FADE_SAMPLES
        v *= t * t * (3 - 2 * t); this.fadeOutRem--
      }
      const lo = v > -DEVICE_OUTPUT_BOUND ? v : -DEVICE_OUTPUT_BOUND
      out[i] = lo < DEVICE_OUTPUT_BOUND ? lo : DEVICE_OUTPUT_BOUND
    }
  }
}
