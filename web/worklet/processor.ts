/**
 * processor.ts — tropical AudioWorkletProcessor (player).
 *
 * Runs in AudioWorkletGlobalScope (audio thread) and renders nothing: the
 * render-ahead Worker (`worker/renderer.ts`) evaluates the kernel ahead of
 * time and sends chunks tagged with their start frame and epoch; this
 * processor writes them into a ring indexed by frame and copies out one
 * render quantum at a time. A stall on the renderer is absorbed by the ring
 * instead of landing on the audio thread.
 *
 * Receives on `port` (from the host):
 *   - { type: 'init', port, channels, options } — the worklet's end of the
 *     MessageChannel to the renderer, and the host output width
 *   - { type: 'mute' } — a short local fade before the host closes the context
 * Receives on the renderer port:
 *   - { type: 'chunk', epoch, start, data } — frames from `start`. A chunk of
 *     a newer epoch (a knob, fade, or patch splice) overwrites the frames it
 *     covers; frames past it keep the older epoch's audio until the renderer
 *     gets there, so a stall mid-refill plays stale-but-valid frames rather
 *     than nothing. An older epoch's chunk is dropped.
 * Sends to the renderer: { type: 'playhead', frame } every few quanta — the
 * next frame it will play, so the renderer knows where "now" is.
 *
 * Where the frames change epoch other than where the renderer spliced them,
 * they do not meet continuously: a write that lands on frames already
 * playing crossfades from what was queued, and any other such seam (the
 * newest epoch running into an older one, or the renderer resuming past an
 * older one after a stall) dips: fades out to the seam and back in.
 *
 * Underrun concealment: if the ring runs dry it never steps to zero. It keeps
 * a reserve of real frames, fades out over them when the ring would dip into
 * it, and on recovery fades in at the current playhead (the wall-clock τ),
 * not where playback stalled — the kernel is a function of τ, so the
 * renderer simply resumes there.
 */

type ProcessorOptions = {
  /** Frames kept back to fade out over; also the length of every
   *  concealment fade, dip, and crossfade. */
  reserveFrames?: number
  /** Contiguous frames needed before playback (re)starts. */
  resumeFrames?: number
  /** Quanta between playhead reports. */
  reportQuanta?: number
  /** Ring length; must exceed the renderer's lookahead plus its margin. */
  capacityFrames?: number
}

type Chunk = { type: 'chunk'; epoch: number; start: number; data: Float32Array[] }
type HostMsg =
  | { type: 'init'; port: MessagePort; channels?: number; options?: ProcessorOptions }
  | { type: 'mute' }

// AudioWorkletProcessor is a global in the worklet scope.
declare const AudioWorkletProcessor: {
  new (options?: AudioWorkletNodeOptions): AudioWorkletProcessor
}
declare function registerProcessor(name: string, processorCtor: unknown): void

interface AudioWorkletProcessor {
  readonly port: MessagePort
  process(inputs: Float32Array[][], outputs: Float32Array[][], parameters: Record<string, Float32Array>): boolean
}

/** Frames `[start, end)` of the ring hold epoch `epoch`. Runs are sorted and
 *  disjoint; a frame no run covers holds nothing. */
type Run = { start: number; end: number; epoch: number }

const smooth = (t: number): number => t * t * (3 - 2 * t)

class TropicalProcessor extends AudioWorkletProcessor {
  private renderer: MessagePort | null = null
  private reserve = 128
  private resume = 1024
  private reportQuanta = 4
  private capacity = 1 << 15
  /** Host-channel planes; mono chunks are written to every plane. */
  private ring: Float32Array[] = []
  private runs: Run[] = []
  private epoch = 0
  /** The splice frame of each recent epoch: the one seam the renderer made
   *  continuous (from the epoch before it). */
  private splices = new Map<number, number>()

  /** The next frame to play: the audio clock, counted from node creation. */
  private frame = 0
  private quanta = 0

  /** idle: nothing has played yet; concealed: in an underrun. */
  private state: 'idle' | 'playing' | 'concealed' = 'idle'
  /** Frames went by unplayed while not playing: the next start joins
   *  mid-stream and fades in. */
  private joinedLate = false
  private gain = 1
  private rampFrom = 1
  private rampTo = 1
  private rampPos = 0
  private rampLen = 0
  private muted = false
  /** A seam into an older epoch at `dipAt`; the fade down starts at
   *  `dipFrom`. -1 when none. */
  private dipAt = -1
  private dipFrom = -1

  /** At a write onto playing frames, the frames it replaced, crossfaded out. */
  private xfade: Float32Array[] = []
  private xfadePos = 0
  private xfadeLen = 0

  // Counters for diagnostics and the tests.
  underruns = 0
  concealedQuanta = 0
  lateSplices = 0
  dips = 0

  constructor() {
    super()
    this.port.onmessage = (e: MessageEvent<HostMsg>) => {
      const msg = e.data
      if (msg.type === 'init') {
        const o = msg.options ?? {}
        this.reserve = o.reserveFrames ?? this.reserve
        this.resume = o.resumeFrames ?? this.resume
        this.reportQuanta = o.reportQuanta ?? this.reportQuanta
        this.capacity = o.capacityFrames ?? this.capacity
        this.planes(msg.channels ?? 2)
        this.renderer = msg.port
        this.renderer.onmessage = (c: MessageEvent<Chunk>) => this.receive(c.data)
        this.renderer.postMessage({ type: 'playhead', frame: this.frame })
      }
      else if (msg.type === 'mute') { this.muted = true; this.ramp(0, this.reserve) }
    }
  }

  private planes(n: number): void {
    this.ring = Array.from({ length: n }, () => new Float32Array(this.capacity))
    this.xfade = Array.from({ length: n }, () => new Float32Array(this.reserve))
  }

  private receive(chunk: Chunk): void {
    if (chunk.type !== 'chunk' || chunk.epoch < this.epoch) return
    if (chunk.epoch > this.epoch) {
      this.splices.set(chunk.epoch, chunk.start)
      this.splices.delete(chunk.epoch - 16)
    }
    this.epoch = chunk.epoch
    const n = chunk.data[0]?.length ?? 0
    const P = this.frame
    const a = Math.max(chunk.start, P)
    const b = Math.min(chunk.start + n, P + this.capacity)
    if (a >= b) return
    // Landing on the frame about to play, over older frames, is a splice that
    // came late: the renderer made the new frames continuous with the old at
    // the splice frame, which has already played. Crossfade from what was
    // queued instead.
    if (chunk.start < P && this.state === 'playing') {
      const r = this.runAt(P)
      if (r && r.epoch < chunk.epoch) { this.lateSplices++; this.snapshot(P) }
    }
    if (this.ring.length === 0) this.planes(Math.max(2, chunk.data.length))
    const C = chunk.data.length
    for (let h = 0; h < this.ring.length; h++) {
      const plane = this.ring[h]!
      const src = C === 1 ? chunk.data[0]! : h < C ? chunk.data[h]! : null
      for (let f = a; f < b; f++) plane[f % this.capacity] = src ? src[f - chunk.start]! : 0
    }
    this.cover(a, b, chunk.epoch)
  }

  /** Mark `[a, b)` as holding `epoch`, over whatever it held. */
  private cover(a: number, b: number, epoch: number): void {
    const out: Run[] = []
    for (const r of this.runs) {
      if (r.end <= a || r.start >= b) { out.push(r); continue }
      if (r.start < a) out.push({ start: r.start, end: a, epoch: r.epoch })
      if (r.end > b) out.push({ start: b, end: r.end, epoch: r.epoch })
    }
    out.push({ start: a, end: b, epoch })
    out.sort((x, y) => x.start - y.start)
    this.runs = []
    for (const r of out) {
      const last = this.runs[this.runs.length - 1]
      if (last && last.end === r.start && last.epoch === r.epoch) last.end = r.end
      else this.runs.push(r)
    }
  }

  private runAt(f: number): Run | undefined {
    for (const r of this.runs) if (r.start <= f && f < r.end) return r
    return undefined
  }

  /** Copy up to a reserve of queued frames from `f` into the crossfade. */
  private snapshot(f: number): void {
    let len = 0
    const end = this.contiguousEnd(f)
    while (len < this.reserve && f + len < end) {
      for (let h = 0; h < this.ring.length; h++)
        this.xfade[h]![len] = this.ring[h]![(f + len) % this.capacity]!
      len++
    }
    this.xfadePos = 0
    this.xfadeLen = len
  }

  /** The end of the gap-free frames from `f` (`f` itself if none). */
  private contiguousEnd(f: number): number {
    let end = f
    for (const r of this.runs) {
      if (r.end <= end) continue
      if (r.start > end) break
      end = r.end
    }
    return end
  }

  private ramp(to: number, len: number): void {
    this.rampFrom = this.gain
    this.rampTo = to
    this.rampPos = 0
    this.rampLen = Math.max(1, len)
  }

  process(_inputs: Float32Array[][], outputs: Float32Array[][], _parameters: Record<string, Float32Array>): boolean {
    const output = outputs[0]
    if (!output || output.length === 0) return true
    const Q = output[0]!.length
    const P = this.frame

    // Drop what has gone by.
    while (this.runs.length > 0 && this.runs[0]!.end <= P) this.runs.shift()
    if (this.runs[0] && this.runs[0].start < P) {
      if (this.state !== 'playing') this.joinedLate = true
      this.runs[0].start = P
    }
    const head = this.runs[0]
    const headAt = head ? head.start - P : Infinity // 0 when the head is at P
    const avail = head ? this.contiguousEnd(head.start) - head.start : 0

    if (this.state === 'playing') {
      // Playing this quantum would leave less than the reserve: fade out over
      // the real frames that remain rather than step to zero later.
      if (headAt > 0 || avail < Q + this.reserve) {
        this.state = 'concealed'
        this.underruns++
        this.dipAt = this.dipFrom = -1
        this.ramp(0, Math.min(this.reserve, headAt > 0 ? 0 : avail))
        this.port.postMessage({ type: 'diag', text: 'underrun', data: { frame: P, queued: avail } })
      } else if (!this.muted) this.findSeam(P, Q)
    } else if (headAt < Q && avail >= Math.max(this.resume, Q - headAt + this.reserve) && !this.muted) {
      // A first start that is on time begins exactly where the renderer put
      // the patch (it fades itself in); anything else joins mid-stream.
      if (this.state === 'idle' && !this.joinedLate) { this.gain = this.rampTo = 1; this.rampLen = 0 }
      else { this.gain = 0; this.ramp(1, this.reserve) }
      this.state = 'playing'
      this.joinedLate = false
    }

    const reading = this.state === 'playing' || this.rampPos < this.rampLen
    let silent = true
    let ri = 0
    for (let s = 0; s < Q; s++) {
      const f = P + s
      if (f === this.dipFrom) this.ramp(0, this.dipAt - f)
      if (f === this.dipAt) { this.ramp(1, this.reserve); this.dipAt = this.dipFrom = -1 }
      if (this.rampPos < this.rampLen) {
        this.rampPos++
        this.gain = this.rampFrom + (this.rampTo - this.rampFrom) * smooth(this.rampPos / this.rampLen)
      }
      while (ri < this.runs.length && this.runs[ri]!.end <= f) ri++
      const have = reading && ri < this.runs.length && this.runs[ri]!.start <= f
      const xf = this.xfadePos < this.xfadeLen
      const w = xf ? smooth((this.xfadePos + 1) / (this.xfadeLen + 1)) : 1
      const at = f % this.capacity
      for (let h = 0; h < output.length; h++) {
        let v = have && h < this.ring.length ? this.ring[h]![at]! : 0
        if (xf && h < this.xfade.length) v = this.xfade[h]![this.xfadePos]! + (v - this.xfade[h]![this.xfadePos]!) * w
        v *= this.gain
        output[h]![s] = v
        if (v !== 0) silent = false
      }
      if (xf) this.xfadePos++
    }
    if (this.state === 'concealed' && silent) this.concealedQuanta++

    this.frame = P + Q
    if (++this.quanta % this.reportQuanta === 0)
      this.renderer?.postMessage({ type: 'playhead', frame: this.frame })
    return true
  }

  /** Schedule a dip at the first seam this quantum or the next reserve
   *  reaches — two epochs meeting anywhere but a splice: newest frames
   *  running into an older epoch's, or the renderer resuming past them — or
   *  drop one that newer frames have since covered. */
  private findSeam(P: number, Q: number): void {
    let seam = -1
    for (let i = 1; i < this.runs.length; i++) {
      const prev = this.runs[i - 1]!, r = this.runs[i]!
      if (r.start >= P + Q + this.reserve) break
      const spliced = r.epoch === prev.epoch + 1 && this.splices.get(r.epoch) === r.start
      if (prev.end === r.start && r.start > P && !spliced) { seam = r.start; break }
    }
    if (seam === this.dipAt) return
    if (seam < 0) {
      // Covered after all: if the fade down began, come back up.
      if (this.dipFrom >= 0 && this.dipFrom < P) this.ramp(1, this.reserve)
      this.dipAt = this.dipFrom = -1
      return
    }
    this.dips++
    this.dipAt = seam
    this.dipFrom = Math.max(P, seam - this.reserve)
  }
}

registerProcessor('tropical-processor', TropicalProcessor)
