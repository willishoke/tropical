/**
 * runtime.ts — WASM hot-swap runtime for the AudioWorklet.
 *
 * Web counterpart to engine/runtime/FlatRuntime.cpp. Holds two
 * WebAssembly.Instance slots + active index, supports click-free
 * hot-swap with state transfer by register/array name, applies a
 * 2048-sample Hermite smoothstep fade envelope, and snapshots
 * trigger params from the SharedArrayBuffer into WASM memory
 * once per audio block.
 *
 * This class lives on the audio thread (AudioWorkletGlobalScope).
 * Plans are compiled to WASM on the main thread; only WebAssembly.Module
 * objects cross the MessagePort into the worklet.
 */

import type { WasmLayout } from '../../compiler/wasm_memory_layout.js'

export type LoadedPlan = {
  /** Raw WASM bytes. Compiled inside the worklet to avoid a Chrome quirk
   *  where port.postMessage silently drops messages containing a
   *  WebAssembly.Module to an AudioWorklet. */
  bytes: Uint8Array
  layout: WasmLayout
  paramPtrs: string[]
  stateInit: (number | boolean)[]
  registerTypes: ('float' | 'int' | 'bool')[]
  registerNames: string[]
  arraySlotNames: string[]
}

type Slot = {
  instance: WebAssembly.Instance
  memory: WebAssembly.Memory
  processFn: (blen: number, sidx: bigint) => void
  layout: WasmLayout
  paramPtrs: string[]
  registerNames: string[]
  arraySlotNames: string[]
  registerTypes: ('float' | 'int' | 'bool')[]
}

const FADE_SAMPLES = 2048

export class WasmRuntime {
  private slots: [Slot | null, Slot | null] = [null, null]
  private activeIdx = 0
  private sampleIndex = 0n
  private fadeInRem = 0
  private fadeOutRem = 0

  /** SharedArrayBuffer view for control params; layout: two f64 per param — [value, frame_value]. */
  constructor(
    private paramsShared: Float64Array,
    /** Max params this runtime expects. */
    private maxParams: number,
  ) {}

  /** Swap in a freshly compiled plan. Transfers matching state by name. */
  async loadPlan(plan: LoadedPlan): Promise<void> {
    const { instance } = await WebAssembly.instantiate(plan.bytes, {})
    const memory = instance.exports.memory as WebAssembly.Memory
    const processFn = instance.exports.process as (blen: number, sidx: bigint) => void

    // Initialize state_init into registers. Array slots are zeroed (matching
    // the native parser at engine/runtime/NumericProgramParser.hpp:130-139).
    initRegisters(memory, plan.layout.registersOffset, plan.stateInit, plan.registerTypes)

    const newSlot: Slot = {
      instance, memory, processFn,
      layout: plan.layout,
      paramPtrs: plan.paramPtrs,
      registerNames: plan.registerNames,
      arraySlotNames: plan.arraySlotNames,
      registerTypes: plan.registerTypes,
    }

    // If there's an outgoing slot, transfer matching state.
    const outgoing = this.slots[this.activeIdx]
    if (outgoing) {
      transferRegisters(outgoing, newSlot)
      transferArrays(outgoing, newSlot)
    }

    const inactive = 1 - this.activeIdx
    this.slots[inactive] = newSlot
    this.activeIdx = inactive
    this.fadeInRem = FADE_SAMPLES
    // If there was a previous slot, the fade-in naturally handles the swap without pop.
  }

  /** Render `blockSize` samples into `outBuf` (mono). Returns number of samples written (== blockSize on success, 0 if no plan loaded). */
  process(outBuf: Float32Array, blockSize: number): number {
    const slot = this.slots[this.activeIdx]
    if (!slot) return 0

    // Snapshot params from SharedArrayBuffer into WASM memory.
    this.snapshotParams(slot)

    // Run kernel; it writes to memory[outputOffset..+blockSize*8] as f64.
    slot.processFn(blockSize, this.sampleIndex)
    this.sampleIndex += BigInt(blockSize)

    // Copy f64 output to f32 output buffer with fade envelope applied.
    const outF64 = new Float64Array(slot.memory.buffer, slot.layout.outputOffset, blockSize)
    for (let i = 0; i < blockSize; i++) {
      let v = outF64[i]!
      if (this.fadeInRem > 0) {
        const t = 1 - this.fadeInRem / FADE_SAMPLES
        const fade = t * t * (3 - 2 * t) // smoothstep
        v *= fade
        this.fadeInRem--
      } else if (this.fadeOutRem > 0) {
        const t = this.fadeOutRem / FADE_SAMPLES
        const fade = t * t * (3 - 2 * t)
        v *= fade
        this.fadeOutRem--
      }
      outBuf[i] = v
    }
    return blockSize
  }

  beginFadeIn(): void { this.fadeInRem = FADE_SAMPLES; this.fadeOutRem = 0 }
  beginFadeOut(): void { this.fadeOutRem = FADE_SAMPLES; this.fadeInRem = 0 }

  private snapshotParams(slot: Slot): void {
    // slot.paramPtrs are the canonical order used by emit_wasm.
    // Each ptr is a decimal-stringified SAB index (picked by WebParam on main thread).
    // SAB layout: params[i*2] = value, params[i*2 + 1] = frame_value
    const memView = new Float64Array(slot.memory.buffer)
    const tableBase = slot.layout.paramTableOffset / 8
    const frameBase = slot.layout.paramFrameOffset / 8
    for (let i = 0; i < slot.paramPtrs.length; i++) {
      const sabIdx = parseInt(slot.paramPtrs[i]!, 10)
      if (!Number.isFinite(sabIdx) || sabIdx < 0 || sabIdx * 2 >= this.paramsShared.length) continue
      memView[tableBase + i] = this.paramsShared[sabIdx * 2]!
      memView[frameBase + i] = this.paramsShared[sabIdx * 2 + 1]!
    }
    void this.maxParams
  }
}

// ─────────────────────────────────────────────────────────────
// State initialization & transfer helpers
// ─────────────────────────────────────────────────────────────

function initRegisters(
  memory: WebAssembly.Memory,
  regOffset: number,
  stateInit: (number | boolean)[],
  regTypes: ('float' | 'int' | 'bool')[],
): void {
  const dv = new DataView(memory.buffer)
  for (let i = 0; i < stateInit.length; i++) {
    const v = stateInit[i]
    if (Array.isArray(v)) continue // array-typed slot — zero-initialized by fresh WASM memory
    const t = regTypes[i] ?? 'float'
    const off = regOffset + i * 8
    if (typeof v === 'boolean') {
      dv.setBigInt64(off, v ? 1n : 0n, true)
    } else if (t === 'int') {
      dv.setBigInt64(off, BigInt(Math.trunc(v as number)), true)
    } else if (t === 'bool') {
      dv.setBigInt64(off, (v as number) !== 0 ? 1n : 0n, true)
    } else {
      dv.setFloat64(off, v as number, true)
    }
  }
}

function transferRegisters(from: Slot, to: Slot): void {
  // Copy registers by name match; type mismatch → skip.
  const dvFrom = new DataView(from.memory.buffer)
  const dvTo = new DataView(to.memory.buffer)
  for (let toIdx = 0; toIdx < to.registerNames.length; toIdx++) {
    const name = to.registerNames[toIdx]
    const fromIdx = from.registerNames.indexOf(name!)
    if (fromIdx < 0) continue
    if (from.registerTypes[fromIdx] !== to.registerTypes[toIdx]) continue
    const val = dvFrom.getBigInt64(from.layout.registersOffset + fromIdx * 8, true)
    dvTo.setBigInt64(to.layout.registersOffset + toIdx * 8, val, true)
  }
}

function transferArrays(from: Slot, to: Slot): void {
  const bufFrom = new Float64Array(from.memory.buffer)
  const bufTo = new Float64Array(to.memory.buffer)
  for (let toSlot = 0; toSlot < to.arraySlotNames.length; toSlot++) {
    const name = to.arraySlotNames[toSlot]
    const fromSlot = from.arraySlotNames.indexOf(name!)
    if (fromSlot < 0) continue
    const fromSize = from.layout.arraySizes[fromSlot]!
    const toSize = to.layout.arraySizes[toSlot]!
    if (fromSize !== toSize) continue
    const fromStart = from.layout.arrayOffsets[fromSlot]! / 8
    const toStart = to.layout.arrayOffsets[toSlot]! / 8
    bufTo.set(bufFrom.subarray(fromStart, fromStart + fromSize), toStart)
  }
}
