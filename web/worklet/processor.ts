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
  | { type: 'init'; paramsSab: SharedArrayBuffer | ArrayBuffer; maxParams: number }
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
  private blockCount = 0
  private loadCompleted = false

  constructor() {
    super()
    this.port.onmessage = (e: MessageEvent<WorkletMsg>) => this.onMessage(e.data)
    this.diag('ctor — processor instantiated')
  }

  private diag(text: string, data?: unknown): void {
    this.port.postMessage({ type: 'diag', text, data })
  }

  private onMessage(msg: WorkletMsg): void {
    if (msg.type === 'init') {
      const view = new Float64Array(msg.paramsSab)
      this.runtime = new WasmRuntime(view, msg.maxParams)
      this.diag(`init — runtime created, maxParams=${msg.maxParams}, sab.byteLength=${msg.paramsSab.byteLength}`)
    } else if (msg.type === 'load') {
      this.pendingLoad = msg.plan
      this.diag(`load message queued — layout.outputOffset=${msg.plan.layout.outputOffset}, maxBlockSize=${msg.plan.layout.maxBlockSize}, pageCount=${msg.plan.layout.pageCount}, registers=${msg.plan.layout.registerCount}`)
    } else if (msg.type === 'fadeIn' && this.runtime) {
      this.runtime.beginFadeIn()
      this.diag('beginFadeIn')
    } else if (msg.type === 'fadeOut' && this.runtime) {
      this.runtime.beginFadeOut()
      this.diag('beginFadeOut')
    }
  }

  process(_inputs: Float32Array[][], outputs: Float32Array[][], _parameters: Record<string, Float32Array>): boolean {
    if (!this.runtime) return true

    if (this.pendingLoad) {
      const plan = this.pendingLoad
      this.pendingLoad = null
      this.loadCompleted = false
      this.runtime.loadPlan(plan)
        .then(() => { this.loadCompleted = true; this.diag('loadPlan resolved') })
        .catch((err) => { this.port.postMessage({ type: 'error', error: String(err) }) })
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
      for (let ch = 1; ch < output.length; ch++) {
        output[ch]!.set(mono)
      }
    } else {
      mono.fill(0)
      for (let ch = 1; ch < output.length; ch++) output[ch]!.fill(0)
    }

    // Emit block peak every ~1 s at 48k (~375 blocks) so we can see what's
    // actually reaching the output channel in DevTools console.
    this.blockCount++
    if (this.blockCount === 1 || this.blockCount === 20 || this.blockCount === 100 || this.blockCount % 400 === 0) {
      let peak = 0
      for (let i = 0; i < n; i++) peak = Math.max(peak, Math.abs(mono[i]!))
      this.diag(`block=${this.blockCount} rendered=${rendered} peak=${peak.toExponential(3)} loadDone=${this.loadCompleted}`)
    }
    return true
  }
}

registerProcessor('tropical-processor', TropicalProcessor)
