/**
 * runtime.ts — Thin wrapper around tropical_runtime_t (FlatRuntime).
 *
 * Replaces Graph for audio processing: receives flat plan JSON,
 * JIT-compiles to a single kernel, and runs it per sample.
 */

import * as b from './bindings.js'
import { DAC } from './audio.js'

const _registry = new FinalizationRegistry((handle: unknown) => {
  b.tropical_runtime_free(handle)
})

export class Runtime {
  _h: unknown

  constructor(bufferLength = 512) {
    this._h = b.check(b.tropical_runtime_new(bufferLength), 'runtime_new')
    _registry.register(this, this._h, this)
  }

  dispose(): void {
    _registry.unregister(this)
    b.tropical_runtime_free(this._h)
    this._h = null
  }

  /** Load a flat plan (tropical_plan_2) JSON string. Hot-swaps during live audio. */
  loadPlan(planJson: string): boolean {
    const ok = b.tropical_runtime_load_plan(this._h, planJson, planJson.length) as boolean
    if (!ok) {
      const cErr = b.tropical_last_error()
      throw new Error(`runtime loadPlan failed: ${cErr}`)
    }
    return ok
  }

  /** Process one buffer of audio. Fills outputBuffer. */
  process(): void {
    b.tropical_runtime_process(this._h)
  }

  /** Pointer to the output buffer. Valid until next process(). */
  get outputBuffer(): Float64Array {
    const ptr = b.tropical_runtime_output_buffer(this._h)
    return b.decodeDoubleBuffer(ptr, this.bufferLength)
  }

  get bufferLength(): number {
    return b.tropical_runtime_get_buffer_length(this._h) as number
  }

  beginFadeIn(): void {
    b.tropical_runtime_begin_fade_in(this._h)
  }

  beginFadeOut(): void {
    b.tropical_runtime_begin_fade_out(this._h)
  }

  isFadeOutComplete(): boolean {
    return b.tropical_runtime_is_fade_out_complete(this._h) as boolean
  }

  /** Create a DAC that reads audio from this runtime. */
  createDAC(sampleRate = 44100, channels = 2): DAC {
    return DAC.fromRuntime(this._h, sampleRate, channels)
  }
}
