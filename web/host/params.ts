/**
 * params.ts — Browser-side smoothed-param and trigger.
 *
 * Web counterpart to compiler/runtime/param.ts. Instead of a native
 * ControlParam allocated on the C++ heap, each param claims a slot
 * in a SharedArrayBuffer (two f64 slots per param: value, frame_value).
 *
 * The slot index doubles as the `_handle` carried through the expression
 * tree. emit_numeric.ts stringifies it to `param.ptr`, and the worklet
 * runtime reads that string back as the SAB index when snapshotting.
 */

import { SignalExpr } from '../../compiler/expr.js'

export class ParamBank {
  /** SAB view, f64. Each param owns `[value, frame_value]`. */
  readonly view: Float64Array
  readonly shared: SharedArrayBuffer | ArrayBuffer
  /** Max params this bank supports. */
  readonly capacity: number
  private nextSlot = 0

  constructor(capacity = 256) {
    this.capacity = capacity
    // SharedArrayBuffer requires COOP/COEP (crossOriginIsolated). Fall back
    // to a plain ArrayBuffer when unavailable — the worklet will still
    // receive a snapshot via postMessage at init time; live param updates
    // won't cross the thread boundary but static params will work.
    const ok =
      typeof SharedArrayBuffer !== 'undefined' &&
      (typeof (globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated === 'undefined' ||
       (globalThis as { crossOriginIsolated: boolean }).crossOriginIsolated)
    this.shared = ok ? new SharedArrayBuffer(capacity * 2 * 8) : new ArrayBuffer(capacity * 2 * 8)
    this.view = new Float64Array(this.shared)
  }

  allocSlot(): number {
    if (this.nextSlot >= this.capacity) {
      throw new Error(`ParamBank: out of slots (capacity ${this.capacity})`)
    }
    return this.nextSlot++
  }
}

export class WebParam {
  readonly _h: number
  constructor(private bank: ParamBank, initValue = 0, public readonly timeConst = 0.005) {
    this._h = bank.allocSlot()
    this.value = initValue
  }

  get value(): number { return this.bank.view[this._h * 2]! }
  set value(v: number) { this.bank.view[this._h * 2] = v }

  asExpr(): SignalExpr {
    return SignalExpr.fromNode({ op: 'smoothed_param', name: '(unnamed)', _ptr: true, _handle: this._h })
  }
}

export class WebTrigger {
  readonly _h: number
  constructor(private bank: ParamBank) {
    this._h = bank.allocSlot()
  }

  fire(value = 1.0): void {
    // Writing to frame_value fires the trigger; kernels read it per-block.
    this.bank.view[this._h * 2 + 1] = value
  }

  get value(): number { return this.bank.view[this._h * 2 + 1]! }

  asExpr(): SignalExpr {
    return SignalExpr.fromNode({ op: 'trigger_param', name: '(unnamed)', _ptr: true, _handle: this._h })
  }
}
