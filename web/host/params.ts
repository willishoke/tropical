/**
 * params.ts — Browser-side smoothed-param and trigger.
 *
 * Web counterpart to the native param model. Instead of a native
 * ControlParam allocated on the C++ heap, each param claims a slot
 * in a SharedArrayBuffer (two f64 slots per param: value, frame_value).
 *
 * The slot index doubles as the param handle stringified to `param.ptr`
 * in the plan; the worklet runtime reads that string back as the SAB
 * index when snapshotting.
 */

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
