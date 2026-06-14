/**
 * context.ts — AudioContext + AudioWorkletNode bootstrap for the browser demo.
 *
 *   - Create/resume AudioContext (must be triggered by a user gesture)
 *   - Load the worklet bundle and register 'tropical-processor'
 *   - Construct the worklet node, wire it to `destination`
 *   - Provide a high-level API to load a precompiled patch (.wasm + manifest)
 *     and trigger fade in/out.
 *
 * No SharedArrayBuffer / params: the demo is a player of precompiled patches,
 * so it needs no COOP/COEP and deploys as plain static files.
 */
import type { KernelManifest } from '../runtime/index.js'

export type TropicalHost = {
  context: AudioContext
  node: AudioWorkletNode
  /** Load a precompiled patch into the worklet. */
  loadPatch(wasm: ArrayBuffer, manifest: KernelManifest): void
  fadeIn(): void
  fadeOut(): void
  dispose(): Promise<void>
}

export type BootstrapOptions = {
  /** URL to the compiled worklet bundle (ESM). */
  workletUrl: string
  /** Number of output channels (mono upmixed to stereo by default). */
  outputChannels?: number
}

export async function startHost(opts: BootstrapOptions): Promise<TropicalHost> {
  const ctx = new AudioContext()
  if (ctx.state === 'suspended') await ctx.resume()

  await ctx.audioWorklet.addModule(opts.workletUrl)

  const node = new AudioWorkletNode(ctx, 'tropical-processor', {
    numberOfInputs: 0,
    numberOfOutputs: 1,
    outputChannelCount: [opts.outputChannels ?? 2],
  })

  node.port.onmessage = (e) => {
    const d = e.data
    if (d?.type === 'error') console.error('[tropical-worklet]', d.error)
    else if (d?.type === 'diag') console.log('[tropical-worklet]', d.text, d.data ?? '')
  }

  node.connect(ctx.destination)

  return {
    context: ctx,
    node,
    loadPatch(wasm, manifest) {
      node.port.postMessage({ type: 'load', wasm, manifest })
    },
    fadeIn() { node.port.postMessage({ type: 'fadeIn' }) },
    fadeOut() { node.port.postMessage({ type: 'fadeOut' }) },
    async dispose() {
      node.disconnect()
      await ctx.close()
    },
  }
}
