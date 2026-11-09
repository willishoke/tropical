/**
 * render_ahead.ts — the renderer's logic, free of any Worker API so a test can
 * drive it in virtual time (`renderer.ts` is the Worker shell around it).
 *
 * The kernel is a pure function of the clock, so it can be evaluated anywhere
 * on the timeline: the renderer keeps `lookaheadMs` of output queued ahead of
 * the worklet's playhead, in chunks tagged with their start frame and an
 * epoch. A control (a knob, a fade, a patch swap) never touches the queued
 * past: it lands at a *splice* frame `marginMs` past the playhead, the epoch
 * bumps, and the future is re-rendered from there. The worklet overwrites the
 * frames each chunk of the new epoch covers; until the renderer gets further,
 * the older epoch's frames past it stay queued as a fallback, valid audio
 * under the previous knob values. Knob latency is the margin; stall
 * tolerance is the lookahead.
 *
 * Frames are the worklet's frames (its playhead counts from node creation).
 * A kernel's own clock starts at its `origin`, the frame its patch was spliced
 * in at, as a freshly loaded kernel starts at τ = 0 today.
 */
import {
  ParamWriter, FADE_FRAMES, type KernelManifest, type WasmKernel,
} from '../runtime/index.js'

export type RenderAheadOptions = {
  /** How far ahead of the playhead to keep rendered: the stall tolerance. */
  lookaheadMs: number
  /** How far past the playhead a control lands: the knob latency. */
  marginMs: number
  /** Frames per chunk sent to the worklet. */
  chunkFrames: number
  /** While knobs move, each move throws away everything rendered past its
   *  splice. Until `controlHoldMs` after the last knob write, render only as
   *  far ahead as keeps that re-rendering within `controlBudget` of a core at
   *  a pointer's event rate (the kernel's cost is measured as it renders), so
   *  a heavy patch does not re-render the whole lookahead per drag step. */
  controlBudget: number
  controlHoldMs: number
}

export const RENDER_AHEAD_DEFAULTS: RenderAheadOptions = {
  lookaheadMs: 250,
  marginMs: 25,
  chunkFrames: 512,
  controlBudget: 0.5,
  controlHoldMs: 250,
}

/** Worker → worklet: `data` holds the kernel's compact channels, planar f32,
 *  faded and clamped (`WasmKernel.process`), covering `[start, start + n)`. */
export type Chunk = { type: 'chunk'; epoch: number; start: number; data: Float32Array[] }

/** A dragged slider posts about once per frame of the display. */
const KNOB_EVENTS_PER_S = 60

type Segment = {
  kernel: WasmKernel
  writer: ParamWriter
  /** The frame at which this kernel's τ is 0. */
  origin: number
  /** The first frame this segment plays (the previous one plays before it). */
  from: number
}

export class RenderAhead {
  readonly options: RenderAheadOptions
  private readonly rate: number
  private segments: Segment[] = []
  /** Everything before this frame has been sent in the current epoch. */
  private renderedUntil = 0
  /** Splices never move backward, so frames before the last one are final. */
  private lastSplice = 0
  private epoch = 0
  private reported = 0
  private reportedAt = 0
  private clockKnown = false
  private lastControlAt = -Infinity
  /** Render time over rendered time, smoothed. */
  private costShare = 0

  // Controls received since the last step, applied together at one splice.
  private writes = new Map<string, number>()
  private fade: 'in' | 'out' | null = null
  private incoming: { kernel: WasmKernel; manifest: KernelManifest } | null = null
  /** The newest load; an older load that finishes late is dropped. */
  private loadToken = 0
  /** Writes posted while a patch loads, for that patch at its τ = 0. */
  private held = new Map<string, number>()

  constructor(
    rate: number,
    private readonly send: (chunk: Chunk) => void,
    private readonly now: () => number,
    options: Partial<RenderAheadOptions> = {},
    private readonly onError: (message: string) => void = () => {},
  ) {
    this.rate = rate
    this.options = { ...RENDER_AHEAD_DEFAULTS, ...options }
  }

  private frames(ms: number): number { return Math.round((ms * this.rate) / 1000) }

  /** The worklet reports the next frame it will play. */
  playhead(frame: number): void {
    this.reported = frame
    this.reportedAt = this.now()
    this.clockKnown = true
  }

  /** The playhead now: the last report, advanced by the time since it. */
  private estimate(): number {
    if (!this.clockKnown) return 0
    return this.reported + Math.max(0, this.frames(this.now() - this.reportedAt))
  }

  /** A patch load started; returns the token `adopt` must present. Writes
   *  posted before it were for the old patch. */
  beginLoad(): number {
    this.writes.clear()
    this.held.clear()
    this.incoming = null
    return ++this.loadToken
  }

  /** The loaded kernel, spliced in after the current one fades out. */
  adopt(token: number, kernel: WasmKernel, manifest: KernelManifest): void {
    if (token !== this.loadToken) return
    this.incoming = { kernel, manifest }
  }

  /** A load that failed: the current patch plays on. */
  cancelLoad(token: number): void {
    if (token !== this.loadToken) return
    this.adoptedToken = token
    this.held.clear()
  }

  /** A load has begun and its kernel has not been spliced in yet. */
  private pendingLoad(): boolean { return this.loadToken > this.adoptedToken }
  private adoptedToken = 0

  setParam(name: string, value: number): void {
    if (this.segments.length === 0 || this.pendingLoad()) this.held.set(name, value)
    else this.writes.set(name, value)
  }

  /** The incoming patch fades in by itself; a fade-in asked for meanwhile is
   *  that one. */
  fadeIn(): void { if (!this.pendingLoad()) this.fade = 'in' }
  fadeOut(): void { this.fade = 'out' }

  /** Apply pending controls at one splice and render at most one chunk.
   *  Returns whether there is more to render right now. */
  step(): boolean {
    if (!this.clockKnown) return false
    const est = this.estimate()
    const margin = this.frames(this.options.marginMs)

    // The worklet is already past everything sent (an underrun): resume at the
    // playhead, not where rendering stopped. Nothing queued needs replacing.
    if (this.segments.length > 0 && this.renderedUntil <= est)
      this.renderedUntil = this.lastSplice = Math.max(est + margin, this.lastSplice)

    if (this.incoming || (this.segments.length > 0 && (this.writes.size > 0 || this.fade)))
      this.splice(est, margin)
    if (this.segments.length === 0) { this.fade = null; return false }

    const o = this.options
    const recent = this.now() - this.lastControlAt < o.controlHoldMs
    const ahead = this.frames(recent
      ? Math.min(o.lookaheadMs, (1000 * o.controlBudget) / (KNOB_EVENTS_PER_S * Math.max(this.costShare, 1e-9)))
      : o.lookaheadMs)
    const horizon = est + Math.max(ahead, margin + this.options.chunkFrames)
    if (this.renderedUntil >= horizon) return false
    this.renderChunk()
    return this.renderedUntil < horizon
  }

  private splice(est: number, margin: number): void {
    // As early as the margin allows, never before an earlier splice (frames
    // before it are final), and never past what is already queued: a write
    // takes effect at the boundary between old and new frames, so the anchor
    // discipline's continuity point must be that boundary.
    const S = this.segments.length === 0
      ? Math.max(est + margin, this.lastSplice)
      : Math.min(Math.max(est + margin, this.lastSplice), this.renderedUntil)
    this.lastSplice = S
    this.renderedUntil = S
    this.epoch++
    if (this.writes.size > 0) this.lastControlAt = this.now()

    // Segments that end at or before S are never rendered again.
    while (this.segments.length > 1 && this.segments[1]!.from <= S) this.segments.shift()

    if (this.incoming) {
      const { kernel, manifest } = this.incoming
      this.incoming = null
      this.adoptedToken = this.loadToken
      const writer = new ParamWriter(manifest)
      for (const [name, value] of this.held) this.write(writer, kernel, name, value, 0)
      this.held.clear()
      // A segment queued to start after S has not played; the new patch
      // replaces it.
      this.segments = this.segments.filter((s) => s.from <= S)
      const cur = this.segments[this.segments.length - 1]
      let origin = S
      if (cur && cur.kernel.mutedFrom > S - cur.origin) {
        cur.kernel.beginFadeOut(S - cur.origin)
        origin = S + FADE_FRAMES
      }
      kernel.beginFadeIn(0)
      this.segments.push({ kernel, writer, origin, from: origin })
      if (this.fade === 'in') this.fade = null // the new patch fades in anyway
    }

    const cur = this.segments[this.segments.length - 1]
    if (cur) {
      const at = Math.max(0, S - cur.origin)
      for (const [name, value] of this.writes) this.write(cur.writer, cur.kernel, name, value, at)
      if (this.fade === 'in') cur.kernel.beginFadeIn(at)
      else if (this.fade === 'out')
        for (const s of this.segments) s.kernel.beginFadeOut(Math.max(0, S - s.origin))
    }
    this.writes.clear()
    this.fade = null
  }

  private write(writer: ParamWriter, kernel: WasmKernel, name: string, value: number, at: number): void {
    try { writer.write(kernel, name, value, at) }
    catch (err) { this.onError(String(err)) }
  }

  /** Render `[renderedUntil, …)` up to a chunk, never across a segment
   *  boundary (each chunk has one kernel's channel count). */
  private renderChunk(): void {
    const a = this.renderedUntil
    let i = this.segments.length - 1
    while (i > 0 && this.segments[i]!.from > a) i--
    const seg = this.segments[i]!
    const next = this.segments[i + 1]
    const n = Math.min(this.options.chunkFrames, next ? next.from - a : Infinity)
    const k = seg.kernel
    const data = Array.from({ length: k.outputChannelCount }, () => new Float32Array(n))
    // A finished fade-out is silence; skip the kernel.
    if (a - seg.origin < k.mutedFrom) {
      const t0 = this.now()
      k.seek(a - seg.origin)
      k.process(data, n)
      const share = (this.now() - t0) / ((1000 * n) / this.rate)
      this.costShare += 0.1 * (share - this.costShare)
    }
    this.renderedUntil = a + n
    this.send({ type: 'chunk', epoch: this.epoch, start: a, data })
  }
}
