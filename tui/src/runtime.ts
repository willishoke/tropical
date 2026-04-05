/**
 * runtime.ts — Thin wrapper around egress_runtime_t (FlatRuntime).
 *
 * Replaces Graph for audio processing: receives flat plan JSON,
 * JIT-compiles to a single kernel, and runs it per sample.
 */

import * as b from './bindings.js'

const _registry = new FinalizationRegistry((handle: unknown) => {
  b.egress_runtime_free(handle)
})

export class Runtime {
  _h: unknown

  constructor(bufferLength = 512) {
    this._h = b.check(b.egress_runtime_new(bufferLength), 'runtime_new')
    _registry.register(this, this._h, this)
  }

  dispose(): void {
    _registry.unregister(this)
    b.egress_runtime_free(this._h)
    this._h = null
  }

  /** Load a flat plan (egress_plan_2) JSON string. Hot-swaps during live audio. */
  loadPlan(planJson: string): boolean {
    const ok = b.egress_runtime_load_plan(this._h, planJson, planJson.length) as boolean
    if (!ok) {
      const cErr = b.egress_last_error()
      throw new Error(`runtime loadPlan failed: ${cErr}`)
    }
    return ok
  }

  /** Process one buffer of audio. Fills outputBuffer. */
  process(): void {
    b.egress_runtime_process(this._h)
  }

  /** Pointer to the output buffer. Valid until next process(). */
  get outputBuffer(): Float64Array {
    const ptr = b.egress_runtime_output_buffer(this._h)
    return b.decodeDoubleBuffer(ptr, this.bufferLength)
  }

  get bufferLength(): number {
    return b.egress_runtime_get_buffer_length(this._h) as number
  }

  beginFadeIn(): void {
    b.egress_runtime_begin_fade_in(this._h)
  }

  beginFadeOut(): void {
    b.egress_runtime_begin_fade_out(this._h)
  }

  isFadeOutComplete(): boolean {
    return b.egress_runtime_is_fade_out_complete(this._h) as boolean
  }
}
