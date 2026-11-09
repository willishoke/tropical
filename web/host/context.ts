/**
 * context.ts — AudioContext + render-ahead bootstrap for the browser player.
 *
 *   - Create/resume AudioContext (must be triggered by a user gesture), with a
 *     `playback` latency hint: a player trades a little output latency for a
 *     device buffer that rides out scheduling jitter
 *   - Load the worklet bundle and register 'tropical-processor'
 *   - Start the render-ahead Worker and hand it and the worklet the two ends
 *     of a MessageChannel, so chunks and playhead reports bypass this thread
 *   - Provide a high-level API to load a precompiled patch (.wasm + manifest)
 *     and trigger fade in/out.
 *
 * No SharedArrayBuffer: the demo is a player of precompiled patches, so it
 * needs no COOP/COEP and deploys as plain static files. Knob writes and chunks
 * are messages; the renderer applies each write under its discipline.
 */
import {
  kernelOutputChannelCount, FADE_FRAMES, type KernelManifest,
} from '../runtime/index.js'
import { RENDER_AHEAD_DEFAULTS, type RenderAheadOptions } from '../worker/render_ahead.js'

export type TropicalHost = {
  context: AudioContext
  node: AudioWorkletNode
  worker: Worker
  /** Load a precompiled patch; it is spliced in after the current one fades
   *  out. */
  loadPatch(wasm: ArrayBuffer, manifest: KernelManifest): void
  fadeIn(): void
  /** Resolves once the fade-out has reached the speakers. */
  fadeOut(): Promise<void>
  /** Write a live param (a `param:` slot name) under its discipline. */
  setParam(name: string, value: number): void
  dispose(): Promise<void>
}

export type BootstrapOptions = {
  /** URL to the compiled worklet bundle (ESM). */
  workletUrl: string
  /** URL to the compiled render-ahead Worker bundle (ESM). */
  rendererUrl: string
  /** Host output width. Mono kernels upmix; wider kernels map independently. */
  outputChannels?: number
  /** The AudioContext latency hint; `playback` unless overridden. */
  latencyHint?: AudioContextLatencyCategory | number
  /** Render-ahead tuning (`worker/render_ahead.ts`). */
  renderAhead?: Partial<RenderAheadOptions>
  /** Underrun concealment tuning (`worklet/processor.ts`). */
  concealment?: { reserveFrames?: number; resumeFrames?: number; reportQuanta?: number; capacityFrames?: number }
}

export async function startHost(opts: BootstrapOptions): Promise<TropicalHost> {
  const outputChannels = opts.outputChannels ?? 2
  if (!Number.isSafeInteger(outputChannels) || outputChannels < 1)
    throw new RangeError('outputChannels must be a positive integer')
  const ctx = new AudioContext({ latencyHint: opts.latencyHint ?? 'playback' })
  if (ctx.state === 'suspended') await ctx.resume()

  await ctx.audioWorklet.addModule(opts.workletUrl)

  const node = new AudioWorkletNode(ctx, 'tropical-processor', {
    numberOfInputs: 0,
    numberOfOutputs: 1,
    outputChannelCount: [outputChannels],
  })

  const log = (e: MessageEvent, from: string) => {
    const d = e.data
    if (d?.type === 'error') console.error(`[tropical-${from}]`, d.error)
    else if (d?.type === 'diag') console.log(`[tropical-${from}]`, d.text, d.data ?? '')
  }
  node.port.onmessage = (e) => log(e, 'worklet')

  const worker = new Worker(opts.rendererUrl, { type: 'module' })
  worker.onmessage = (e) => log(e, 'renderer')
  worker.onerror = (e) => console.error('[tropical-renderer]', e.message)
  // The audio thread runs a device buffer of quanta back to back, so the
  // margin must cover one buffer or every splice lands late (and crossfades).
  const ahead = {
    ...RENDER_AHEAD_DEFAULTS,
    marginMs: Math.max(RENDER_AHEAD_DEFAULTS.marginMs, Math.ceil((ctx.baseLatency ?? 0) * 1000) + 5),
    ...opts.renderAhead,
  }
  const channel = new MessageChannel()
  worker.postMessage(
    { type: 'init', rate: ctx.sampleRate, port: channel.port1, options: ahead },
    [channel.port1])
  // The ring must hold the lookahead and the margin; twice that, to spare.
  const capacityFrames = Math.max(1 << 15,
    2 * Math.ceil(((ahead.lookaheadMs + ahead.marginMs) * ctx.sampleRate) / 1000) + ahead.chunkFrames)
  node.port.postMessage({
    type: 'init', port: channel.port2, channels: outputChannels,
    options: { capacityFrames, ...opts.concealment },
  }, [channel.port2])

  node.connect(ctx.destination)

  // A fade-out lands at the next splice, takes the fade, then crosses the
  // device's own latency.
  const fadeOutMs = () =>
    ahead.marginMs + (FADE_FRAMES / ctx.sampleRate) * 1000
    + ((ctx.baseLatency ?? 0) + (ctx.outputLatency ?? 0)) * 1000 + 20

  return {
    context: ctx,
    node,
    worker,
    loadPatch(wasm, manifest) {
      const required = kernelOutputChannelCount(manifest)
      if (required > 1 && required > outputChannels)
        throw new RangeError(
          `patch requires ${required} output channels; host has ${outputChannels}`)
      worker.postMessage({ type: 'load', wasm, manifest })
    },
    fadeIn() { worker.postMessage({ type: 'fadeIn' }) },
    fadeOut() {
      worker.postMessage({ type: 'fadeOut' })
      return new Promise((resolve) => setTimeout(resolve, fadeOutMs()))
    },
    setParam(name, value) { worker.postMessage({ type: 'param', name, value }) },
    async dispose() {
      // Whatever is still queued, fade it locally rather than cut it.
      node.port.postMessage({ type: 'mute' })
      await new Promise((resolve) => setTimeout(resolve, 20))
      worker.terminate()
      node.disconnect()
      await ctx.close()
    },
  }
}
