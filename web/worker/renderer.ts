/**
 * renderer.ts — the render-ahead Worker: owns the kernel and keeps the
 * worklet's queue `lookaheadMs` deep (`render_ahead.ts` has the logic).
 *
 * From the main thread:
 *   - { type: 'init', rate, port, options } — `port` is this end of a
 *     MessageChannel whose other end the host gave the worklet; chunks go out
 *     over it and playhead reports come back, so the main thread is not in
 *     the audio path
 *   - { type: 'load', wasm, manifest } — instantiate, splice in, fade in
 *   - { type: 'param', name, value } — a knob write, at the next splice
 *   - { type: 'fadeIn' } / { type: 'fadeOut' }
 *   - { type: 'display', display, points } — draw one cycle of a channel
 *     (`curve.ts`); `display: null` stops. The renderer answers
 *     { type: 'curve', frame, freq, data } at most ~30 times a second while
 *     the playhead moves, and after a knob write is spliced: each point is the
 *     kernel itself at a chosen τ (`WasmKernel.renderAt`), under the knob
 *     values it is rendering with, from the played frame.
 *
 * Chunks are transferred, not copied, and nothing is shared, so the player
 * still needs no SharedArrayBuffer and no COOP/COEP.
 */
import { WasmKernel, type KernelManifest } from '../runtime/index.js'
import { RenderAhead, type Chunk, type RenderAheadOptions } from './render_ahead.js'
import { cycleCurve, displayFreq, type CycleDisplay, type Curve } from './curve.js'

type RendererMsg =
  | { type: 'init'; rate: number; port: MessagePort; options?: Partial<RenderAheadOptions> }
  | { type: 'load'; wasm: ArrayBuffer; manifest: KernelManifest }
  | { type: 'param'; name: string; value: number }
  | { type: 'fadeIn' }
  | { type: 'fadeOut' }
  | { type: 'display'; display: CycleDisplay | null; points: number }

declare const self: {
  onmessage: ((e: MessageEvent<RendererMsg>) => void) | null
  postMessage(msg: unknown, transfer?: Transferable[]): void
}

const error = (err: unknown) =>
  self.postMessage({ type: 'error', error: (err as Error)?.message ?? String(err) })

let ra: RenderAhead | null = null

// The curve display: evaluated here, where the kernel lives, and posted to the
// page. Its rate is the kernel's own (the manifest's), which sets the cycle.
let display: { d: CycleDisplay; points: number } | null = null
const kernelRate = new WeakMap<WasmKernel, number>()
let lastCurveAt = -Infinity
let curveDirty = false
const CURVE_INTERVAL_MS = 33
function postCurve(force = false): void {
  if (!display || !ra) return
  const now = performance.now()
  if (!force && now - lastCurveAt < CURVE_INTERVAL_MS) return
  const src = ra.displaySource()
  const rate = src && kernelRate.get(src.kernel)
  if (!src || !rate) return
  const freq = displayFreq(src.kernel, display.d)
  if (freq === null || display.d.channel >= src.kernel.outputChannelCount) return
  lastCurveAt = now
  curveDirty = false
  const data = cycleCurve(src.kernel, rate, src.frame, display.d.channel, freq, display.points)
  const msg: Curve = { type: 'curve', frame: src.frame, freq, data }
  self.postMessage(msg, [data.buffer])
}

// Render one chunk per task and yield, so a knob that arrives mid-refill
// splices before more of the doomed future is rendered. A MessageChannel
// yields without setTimeout's nested-timer clamp.
const tick = new MessageChannel()
let scheduled = false
tick.port1.onmessage = () => {
  scheduled = false
  try {
    if (ra?.step()) wake()
    // A knob write is spliced inside `step`; redraw from the new values.
    if (curveDirty) postCurve(true)
  }
  catch (err) { error(err) }
}
function wake(): void {
  if (scheduled) return
  scheduled = true
  tick.port2.postMessage(null)
}

self.onmessage = (e) => {
  const msg = e.data
  if (msg.type === 'init') {
    const port = msg.port
    ra = new RenderAhead(
      msg.rate,
      (chunk: Chunk) => port.postMessage(chunk, chunk.data.map((d) => d.buffer)),
      () => performance.now(),
      msg.options,
      error,
    )
    port.onmessage = (p: MessageEvent<{ type: 'playhead'; frame: number }>) => {
      if (p.data?.type === 'playhead') {
        ra!.playhead(p.data.frame)
        wake()
        try { postCurve() } catch (err) { error(err) }
      }
    }
    return
  }
  if (!ra) { error(`renderer got '${msg.type}' before init`); return }
  const r = ra
  if (msg.type === 'load') {
    const token = r.beginLoad()
    WasmKernel.instantiate(new Uint8Array(msg.wasm), msg.manifest)
      .then((k) => { kernelRate.set(k, msg.manifest.rate); r.adopt(token, k, msg.manifest); curveDirty = true; wake() })
      .catch((err) => { r.cancelLoad(token); error(err) })
  }
  else if (msg.type === 'param') { r.setParam(msg.name, msg.value); curveDirty = true }
  else if (msg.type === 'display') { display = msg.display ? { d: msg.display, points: msg.points } : null; curveDirty = true }
  else if (msg.type === 'fadeIn') r.fadeIn()
  else if (msg.type === 'fadeOut') r.fadeOut()
  wake()
}
