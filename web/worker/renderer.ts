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
 *
 * Chunks are transferred, not copied, and nothing is shared, so the player
 * still needs no SharedArrayBuffer and no COOP/COEP.
 */
import { WasmKernel, type KernelManifest } from '../runtime/index.js'
import { RenderAhead, type Chunk, type RenderAheadOptions } from './render_ahead.js'

type RendererMsg =
  | { type: 'init'; rate: number; port: MessagePort; options?: Partial<RenderAheadOptions> }
  | { type: 'load'; wasm: ArrayBuffer; manifest: KernelManifest }
  | { type: 'param'; name: string; value: number }
  | { type: 'fadeIn' }
  | { type: 'fadeOut' }

declare const self: {
  onmessage: ((e: MessageEvent<RendererMsg>) => void) | null
  postMessage(msg: unknown): void
}

const error = (err: unknown) =>
  self.postMessage({ type: 'error', error: (err as Error)?.message ?? String(err) })

let ra: RenderAhead | null = null

// Render one chunk per task and yield, so a knob that arrives mid-refill
// splices before more of the doomed future is rendered. A MessageChannel
// yields without setTimeout's nested-timer clamp.
const tick = new MessageChannel()
let scheduled = false
tick.port1.onmessage = () => {
  scheduled = false
  try { if (ra?.step()) wake() }
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
      if (p.data?.type === 'playhead') { ra!.playhead(p.data.frame); wake() }
    }
    return
  }
  if (!ra) { error(`renderer got '${msg.type}' before init`); return }
  const r = ra
  if (msg.type === 'load') {
    const token = r.beginLoad()
    WasmKernel.instantiate(new Uint8Array(msg.wasm), msg.manifest)
      .then((k) => { r.adopt(token, k, msg.manifest); wake() })
      .catch((err) => { r.cancelLoad(token); error(err) })
  }
  else if (msg.type === 'param') r.setParam(msg.name, msg.value)
  else if (msg.type === 'fadeIn') r.fadeIn()
  else if (msg.type === 'fadeOut') r.fadeOut()
  wake()
}
