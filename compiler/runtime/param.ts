/**
 * Param — control-rate parameter. Port of tropical/param.py.
 *
 * Wiring references these by name (`{op:'param', name}`); the materializer
 * (`compiler/ir/materialize_session.ts`) looks up the FFI handle off the
 * session's paramRegistry at compile time.
 */

import * as b from './bindings.js'

const _registry = new FinalizationRegistry((handle: unknown) => {
  b.tropical_param_free(handle)
})

export class Param {
  readonly _h: unknown

  /**
   * @param initValue  Initial value (no startup artifact — smoothing starts here).
   * @param timeConst  One-pole lowpass time constant in seconds (default 5 ms).
   *                   0.0 = no smoothing.
   */
  constructor(initValue: number, timeConst = 0.005) {
    this._h = b.check(b.tropical_param_new(initValue, timeConst), 'param_new')
    _registry.register(this, this._h, this)
  }

  get value(): number {
    return b.tropical_param_get(this._h) as number
  }

  set value(v: number) {
    b.tropical_param_set(this._h, v)
  }

  dispose(): void {
    _registry.unregister(this)
    b.tropical_param_free(this._h)
  }
}

