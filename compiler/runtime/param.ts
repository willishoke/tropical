/**
 * Param and Trigger — control-rate parameters. Port of tropical/param.py.
 */

import * as b from './bindings.js'
import { SignalExpr } from '../expr.js'

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

  /** Return a SmoothedParam SignalExpr node for use in wiring expressions. */
  asExpr(): SignalExpr {
    return SignalExpr.fromNode({ op: 'smoothedParam', name: '(unnamed)', _ptr: true, _handle: this._h })
  }

  dispose(): void {
    _registry.unregister(this)
    b.tropical_param_free(this._h)
  }
}

export class Trigger {
  readonly _h: unknown

  constructor() {
    this._h = b.check(b.tropical_param_new_trigger(), 'param_new_trigger')
    _registry.register(this, this._h, this)
  }

  /** Arm the trigger (atomic store of 1.0). Safe from any thread. */
  fire(): void {
    b.tropical_param_set(this._h, 1.0)
  }

  get value(): number {
    return b.tropical_param_get(this._h) as number
  }

  /** Return a TriggerParam SignalExpr node for use in wiring expressions. */
  asExpr(): SignalExpr {
    return SignalExpr.fromNode({ op: 'triggerParam', name: '(unnamed)', _ptr: true, _handle: this._h })
  }

  dispose(): void {
    _registry.unregister(this)
    b.tropical_param_free(this._h)
  }
}
