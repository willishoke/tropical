/**
 * processor.ts — tropical AudioWorkletProcessor.
 *
 * Runs in AudioWorkletGlobalScope (audio thread). Receives:
 *   - { type: 'load', plan: LoadedPlan } — hot-swap to a new WASM kernel
 *   - { type: 'fadeIn' } / { type: 'fadeOut' } — envelope control
 *   - { type: 'init', paramsSab: SharedArrayBuffer } — initial handshake
 *
 * `process()` is called by the host every 128 samples; delegates to
 * WasmRuntime for the actual render.
 */

import { WasmRuntime, type LoadedPlan } from './runtime.js'

type WorkletMsg =
  | { type: 'init'; paramsSab: SharedArrayBuffer; maxParams: number }
  | { type: 'load'; plan: LoadedPlan }
  | { type: 'fadeIn' }
  | { type: 'fadeOut' }

// AudioWorkletProcessor is a global in the worklet scope.
declare const AudioWorkletProcessor: {
  new (options?: AudioWorkletNodeOptions): AudioWorkletProcessor
}
declare function registerProcessor(name: string, processorCtor: unknown): void

interface AudioWorkletProcessor {
  readonly port: MessagePort
  process(inputs: Float32Array[][], outputs: Float32Array[][], parameters: Record<string, Float32Array>): boolean
}

class TropicalProcessor extends AudioWorkletProcessor {
  private runtime: WasmRuntime | null = null
  private pendingLoad: LoadedPlan | null = null

  constructor() {
    super()
    this.port.onmessage = (e: MessageEvent<WorkletMsg>) => this.onMessage(e.data)
  }

  private onMessage(msg: WorkletMsg): void {
    if (msg.type === 'init') {
      const view = new Float64Array(msg.paramsSab)
      this.runtime = new WasmRuntime(view, msg.maxParams)
    } else if (msg.type === 'load') {
      // Defer until next process() tick so the swap happens on the audio thread.
      this.pendingLoad = msg.plan
    } else if (msg.type === 'fadeIn' && this.runtime) {
      this.runtime.beginFadeIn()
    } else if (msg.type === 'fadeOut' && this.runtime) {
      this.runtime.beginFadeOut()
    }
  }

  process(_inputs: Float32Array[][], outputs: Float32Array[][], _parameters: Record<string, Float32Array>): boolean {
    if (!this.runtime) {
      // No runtime yet — emit silence.
      return true
    }
    if (this.pendingLoad) {
      const plan = this.pendingLoad
      this.pendingLoad = null
      // Fire-and-forget — WebAssembly.instantiate from a Module is synchronous
      // enough in practice, but it's typed async. We handle it here by
      // kicking off the promise and letting this block render silence.
      this.runtime.loadPlan(plan).catch((err) => {
        this.port.postMessage({ type: 'error', error: String(err) })
      })
      // First block after swap: emit silence to avoid using the not-yet-ready instance.
      const out0 = outputs[0]?.[0]
      if (out0) out0.fill(0)
      return true
    }

    const output = outputs[0]
    if (!output || output.length === 0) return true
    const mono = output[0]!
    const n = mono.length

    const rendered = this.runtime.process(mono, n)
    if (rendered > 0) {
      // Duplicate mono → all channels
      for (let ch = 1; ch < output.length; ch++) {
        output[ch]!.set(mono)
      }
    } else {
      mono.fill(0)
      for (let ch = 1; ch < output.length; ch++) output[ch]!.fill(0)
    }
    return true
  }
}

registerProcessor('tropical-processor', TropicalProcessor)
