/**
 * context.ts — AudioContext + AudioWorkletNode bootstrap for the browser demo.
 *
 * Responsibilities:
 *   - Create/resume AudioContext (must be triggered by user gesture)
 *   - Load the worklet bundle and register 'tropical-processor'
 *   - Construct the worklet node, wire it to `destination`
 *   - Share a SharedArrayBuffer for params with the worklet
 *   - Provide a high-level API to load a compiled plan (WASM Module + layout)
 *     and trigger fade in/out.
 */

import type { LoadedPlan } from '../worklet/runtime.js'
import { ParamBank } from './params.js'

export type TropicalHost = {
  context: AudioContext
  node: AudioWorkletNode
  bank: ParamBank
  /** Load a plan into the worklet; plan contents will be transferred. */
  loadPlan(plan: LoadedPlan): void
  fadeIn(): void
  fadeOut(): void
  dispose(): Promise<void>
}

export type BootstrapOptions = {
  /** URL to the compiled worklet bundle (ESM). */
  workletUrl: string
  /** Max control params; sizes the SharedArrayBuffer. */
  maxParams?: number
  /** Number of output channels (mono upmixed to stereo by default). */
  outputChannels?: number
}

export async function startHost(opts: BootstrapOptions): Promise<TropicalHost> {
  const ctx = new AudioContext()
  if (ctx.state === 'suspended') await ctx.resume()

  await ctx.audioWorklet.addModule(opts.workletUrl)

  const maxParams = opts.maxParams ?? 256
  const bank = new ParamBank(maxParams)

  const node = new AudioWorkletNode(ctx, 'tropical-processor', {
    numberOfInputs: 0,
    numberOfOutputs: 1,
    outputChannelCount: [opts.outputChannels ?? 2],
  })

  // Send initial handshake. The worklet builds its WasmRuntime on receipt.
  node.port.postMessage({ type: 'init', paramsSab: bank.shared, maxParams })

  node.port.onmessage = (e) => {
    const d = e.data
    if (d?.type === 'error') {
      // eslint-disable-next-line no-console
      console.error('[tropical-worklet]', d.error)
    } else if (d?.type === 'diag') {
      // eslint-disable-next-line no-console
      console.log('[tropical-worklet]', d.text, d.data ?? '')
    }
  }

  node.connect(ctx.destination)

  return {
    context: ctx,
    node,
    bank,
    loadPlan(plan) {
      node.port.postMessage({ type: 'load', plan })
    },
    fadeIn() { node.port.postMessage({ type: 'fadeIn' }) },
    fadeOut() { node.port.postMessage({ type: 'fadeOut' }) },
    async dispose() {
      node.disconnect()
      await ctx.close()
    },
  }
}
