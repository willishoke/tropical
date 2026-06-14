/**
 * processor.ts — tropical AudioWorkletProcessor (player).
 *
 * Runs in AudioWorkletGlobalScope (audio thread). Receives:
 *   - { type: 'load', wasm, manifest } — instantiate a new kernel
 *   - { type: 'fadeIn' } / { type: 'fadeOut' } — envelope control
 *
 * Instantiates a WasmKernel (the extractable runtime package) and renders the
 * browser's 128-sample blocks. No SAB, no live recompile, no state-transfer
 * hot-swap — the demo plays precompiled patches. We post raw wasm *bytes* (not
 * a WebAssembly.Module) because Chrome silently drops worklet messages
 * containing a Module; the worklet compiles them itself.
 */
import { WasmKernel, type KernelManifest } from '../runtime/index.js'

type WorkletMsg =
  | { type: 'load'; wasm: ArrayBuffer; manifest: KernelManifest }
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
  private kernel: WasmKernel | null = null
  private pending: { wasm: ArrayBuffer; manifest: KernelManifest } | null = null

  constructor() {
    super()
    this.port.onmessage = (e: MessageEvent<WorkletMsg>) => {
      try { this.onMessage(e.data) }
      catch (err) {
        this.port.postMessage({ type: 'error', error: `onMessage threw: ${(err as Error).message ?? String(err)}` })
      }
    }
  }

  private onMessage(msg: WorkletMsg): void {
    if (msg.type === 'load') this.pending = { wasm: msg.wasm, manifest: msg.manifest }
    else if (msg.type === 'fadeIn') this.kernel?.beginFadeIn()
    else if (msg.type === 'fadeOut') this.kernel?.beginFadeOut()
  }

  process(_inputs: Float32Array[][], outputs: Float32Array[][], _parameters: Record<string, Float32Array>): boolean {
    // Instantiate a queued patch (async; the kernel arrives a few blocks later).
    if (this.pending) {
      const { wasm, manifest } = this.pending
      this.pending = null
      this.kernel = null
      WasmKernel.instantiate(new Uint8Array(wasm), manifest, 128)
        .then((k) => { this.kernel = k; k.beginFadeIn() })
        .catch((err) => this.port.postMessage({ type: 'error', error: String(err) }))
      const o = outputs[0]?.[0]; if (o) o.fill(0)
      return true
    }

    const output = outputs[0]
    if (!output || output.length === 0) return true
    const mono = output[0]!
    if (!this.kernel) { mono.fill(0); return true }

    this.kernel.process(mono, mono.length)
    for (let ch = 1; ch < output.length; ch++) output[ch]!.set(mono)
    return true
  }
}

registerProcessor('tropical-processor', TropicalProcessor)
