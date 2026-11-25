/**
 * renderer.ts — the render-ahead Worker: owns the kernel and keeps the
 * worklet's queue `lookaheadMs` deep (`render_ahead.ts` has the logic).
 *
 * From the main thread:
 *   - { type: 'init', rate, port, options } — `port` is this end of a
 *     MessageChannel whose other end the host gave the worklet; chunks go out
 *     over it and playhead reports come back, so the main thread is not in
 *     the audio path
 *   - { type: 'load', wasm, manifest, audio? } — instantiate, splice in, fade
 *     in; `audio` names the kernel channels that are sound (default: all),
 *     the rest (taps) are never sent to the worklet
 *   - { type: 'param', name, value } — a knob write, at the next splice
 *   - { type: 'fadeIn' } / { type: 'fadeOut' }
 *   - { type: 'display', displays, points } — draw one cycle of each listed
 *     channel (`curve.ts`); `[]` stops. The renderer answers
 *     { type: 'curve', index, cycle, tau0, freq, data } per display, at most
 *     ~30 rounds a second: each point is the kernel itself at a chosen τ
 *     (`WasmKernel.renderAt`) under the newest spliced knob values, over the
 *     clock cycle that contains the playhead estimate, from its phase 0.
 *
 * Chunks are transferred, not copied, and nothing is shared, so the player
 * still needs no SharedArrayBuffer and no COOP/COEP.
 */
import { WasmKernel, type KernelManifest } from '../runtime/index.js'
import { RenderAhead, type Chunk, type RenderAheadOptions } from './render_ahead.js'
import { cycleCurve, displayFreq, MAX_CURVE_POINTS, type CycleDisplay, type Curve } from './curve.js'

type RendererMsg =
  | { type: 'init'; rate: number; port: MessagePort; options?: Partial<RenderAheadOptions> }
  | { type: 'load'; wasm: ArrayBuffer; manifest: KernelManifest; audio?: number[] }
  | { type: 'param'; name: string; value: number }
  | { type: 'fadeIn' }
  | { type: 'fadeOut' }
  | { type: 'display'; displays: CycleDisplay[]; points: number }

declare const self: {
  onmessage: ((e: MessageEvent<RendererMsg>) => void) | null
  postMessage(msg: unknown, transfer?: Transferable[]): void
}

const error = (err: unknown) =>
  self.postMessage({ type: 'error', error: (err as Error)?.message ?? String(err) })

let ra: RenderAhead | null = null

// The curve display: evaluated here, where the kernel lives, and posted to the
// page. Its rate is the kernel's own (the manifest's), which sets the cycle.
// A curve costs `points` kernel frames; at most one round of curves is drawn
// per CURVE_INTERVAL_MS, a knob change is coalesced into the next round, and
// no round is drawn while the renderer is behind its lookahead.
let displays: CycleDisplay[] = []
let points = 512
const kernelRate = new WeakMap<WasmKernel, number>()
let lastCurveAt = -Infinity
let curveDirty = false
let curveTimer: ReturnType<typeof setTimeout> | null = null
const CURVE_INTERVAL_MS = 33
function maybeCurves(): void {
  if (displays.length === 0 || !ra || ra.behind) return
  const wait = lastCurveAt + CURVE_INTERVAL_MS - performance.now()
  if (wait > 0) {
    // Coalesce: one more round when the interval is up, if anything changed.
    if (curveDirty && !curveTimer)
      curveTimer = setTimeout(() => { curveTimer = null; maybeCurves() }, wait)
    return
  }
  const src = ra.displaySource()
  const rate = src && kernelRate.get(src.kernel)
  if (!src || !rate) return
  lastCurveAt = performance.now()
  curveDirty = false
  displays.forEach((d, index) => {
    const freq = displayFreq(src.kernel, d)
    if (freq === null || d.channel >= src.kernel.outputChannelCount) return
    const c = cycleCurve(src.kernel, rate, src.frame, d.channel, freq, points)
    if (!c) return
    const msg: Curve = { type: 'curve', index, cycle: Number(c.cycle), tau0: c.taus[0]!.toString(), freq, data: c.data }
    self.postMessage(msg, [c.data.buffer])
  })
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
    if (curveDirty) maybeCurves()
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
        try { maybeCurves() } catch (err) { error(err) }
      }
    }
    return
  }
  if (!ra) { error(`renderer got '${msg.type}' before init`); return }
  const r = ra
  if (msg.type === 'load') {
    const token = r.beginLoad()
    WasmKernel.instantiate(new Uint8Array(msg.wasm), msg.manifest)
      .then((k) => {
        kernelRate.set(k, msg.manifest.rate)
        r.adopt(token, k, msg.manifest, msg.audio)
        curveDirty = true
        wake()
      })
      .catch((err) => { r.cancelLoad(token); error(err) })
  }
  else if (msg.type === 'param') { r.setParam(msg.name, msg.value); curveDirty = true }
  else if (msg.type === 'display') {
    if (!Number.isInteger(msg.points) || msg.points < 2 || msg.points > MAX_CURVE_POINTS)
      error(`display points must be an integer in [2, ${MAX_CURVE_POINTS}]`)
    else { displays = msg.displays; points = msg.points; curveDirty = true }
  }
  else if (msg.type === 'fadeIn') r.fadeIn()
  else if (msg.type === 'fadeOut') r.fadeOut()
  wake()
}
