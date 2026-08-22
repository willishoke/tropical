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
import {
  WasmKernel, kernelOutputChannelCount, type KernelManifest,
} from '../runtime/index.js'

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

  private clear(output: Float32Array[]): void {
    for (const channel of output) channel.fill(0)
  }

  process(_inputs: Float32Array[][], outputs: Float32Array[][], _parameters: Record<string, Float32Array>): boolean {
    const output = outputs[0]
    if (!output || output.length === 0) return true

    // Instantiate a queued patch (async; the kernel arrives a few blocks later).
    if (this.pending) {
      const { wasm, manifest } = this.pending
      this.pending = null
      this.kernel = null
      let channelCount: number
      try { channelCount = kernelOutputChannelCount(manifest) }
      catch (err) {
        this.port.postMessage({ type: 'error', error: String(err) })
        this.clear(output)
        return true
      }
      if (channelCount > 1 && channelCount > output.length) {
        this.port.postMessage({
          type: 'error',
          error: `patch requires ${channelCount} output channels; node has ${output.length}`,
        })
        this.clear(output)
        return true
      }
      WasmKernel.instantiate(new Uint8Array(wasm), manifest, 128)
        .then((k) => { this.kernel = k; k.beginFadeIn() })
        .catch((err) => this.port.postMessage({ type: 'error', error: String(err) }))
      this.clear(output)
      return true
    }

    if (!this.kernel) { this.clear(output); return true }

    this.kernel.process(output, output[0]!.length)
    return true
  }
}

registerProcessor('tropical-processor', TropicalProcessor)
