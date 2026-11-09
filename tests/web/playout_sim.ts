/**
 * playout_sim.ts — the browser player's audio clock in virtual time, with no
 * audio device and no browser: an `AudioWorkletProcessor` polyfill, a device
 * that plays one render quantum per quantum period out of a small FIFO, and
 * a renderer thread that can be stalled on a schedule.
 *
 * Two players run under the same clock and stall schedule:
 *   - `direct` — the processor as it was before render-ahead: the kernel
 *     renders in `process()`, so a renderer stall is an audio-thread stall.
 *   - `ahead`  — `web/worklet/processor.ts` fed by `RenderAhead`
 *     (`web/worker/render_ahead.ts`) on its own thread; the stall lands on
 *     the renderer and the audio thread only copies.
 *
 * The device model is the usual one: the audio thread renders quantum k once
 * the FIFO (`deviceFrames` deep) has room for it; the device plays one
 * quantum per period, or a quantum of zeros if the next one is not ready, and
 * then plays the late one after it (the stream shifts, as a browser's does).
 */
import {
  WasmKernel, ParamWriter, type KernelManifest,
} from '../../web/runtime/index'
import { RenderAhead, type Chunk, type RenderAheadOptions } from '../../web/worker/render_ahead'

// ── AudioWorkletGlobalScope polyfill ────────────────────────────────────────

/** A MessagePort stand-in whose deliveries the simulation schedules. */
export class SimPort {
  onmessage: ((e: { data: unknown }) => void) | null = null
  constructor(private readonly out: (msg: unknown) => void = () => {}) {}
  postMessage(msg: unknown): void { this.out(msg) }
  deliver(msg: unknown): void { this.onmessage?.({ data: msg }) }
}

let nextHostPort: SimPort = new SimPort()
const g = globalThis as Record<string, unknown>
g.AudioWorkletProcessor = class { port = nextHostPort }
let registered: (new () => SimProcessor) | null = null
g.registerProcessor = (_name: string, ctor: new () => SimProcessor) => { registered = ctor }
await import('../../web/worklet/processor')

type SimProcessor = {
  port: SimPort
  process(i: Float32Array[][], o: Float32Array[][], p: Record<string, Float32Array>): boolean
  underruns: number
  concealedQuanta: number
  lateSplices: number
  dips: number
}

function newAheadProcessor(hostPort: SimPort): SimProcessor {
  nextHostPort = hostPort
  return new registered!()
}

/** The processor before render-ahead (origin/main's `processor.ts`): a knob
 *  write lands at the next block, the kernel renders in place. */
class DirectProcessor {
  private writes = new Map<string, number>()
  private pending: { k: WasmKernel; m: KernelManifest; inQuanta: number } | null = null
  constructor(private kernel: WasmKernel | null, private writer: ParamWriter | null) {}
  onMessage(msg: SimControl): void {
    if (msg.type === 'param') this.writes.set(msg.name, msg.value)
    else if (msg.type === 'load') {
      // instantiate is async: silence until the new kernel arrives
      this.pending = { k: msg.kernel, m: msg.manifest, inQuanta: msg.instantiateQuanta }
      this.kernel = null
      this.writes.clear()
    }
  }
  process(output: Float32Array[]): void {
    if (this.pending && this.pending.inQuanta-- <= 0) {
      this.kernel = this.pending.k
      this.writer = new ParamWriter(this.pending.m)
      this.kernel.beginFadeIn()
      this.pending = null
    }
    if (!this.kernel) { for (const c of output) c.fill(0); return }
    for (const [name, value] of this.writes) this.writer!.write(this.kernel, name, value, this.kernel.frame)
    this.writes.clear()
    this.kernel.process(output, output[0]!.length)
  }
}

// ── the simulation ──────────────────────────────────────────────────────────

export type Stall = { at: number; ms: number }

export type SimControl =
  | { type: 'param'; name: string; value: number }
  | { type: 'load'; kernel: WasmKernel; manifest: KernelManifest; instantiateQuanta: number }

export type SimConfig = {
  variant: 'direct' | 'ahead'
  kernel: WasmKernel
  manifest: KernelManifest
  durationMs: number
  /** Stalls of the thread that renders the kernel. */
  stalls?: Stall[]
  /** Main-thread controls, posted at `at` ms. */
  controls?: { at: number; msg: SimControl }[]
  quantum?: number
  /** FIFO depth between the audio thread and the device. */
  deviceFrames?: number
  hostChannels?: number
  /** Delivery latency of every message. */
  msgLatencyMs?: number
  /** Delivery latency of renderer → worklet chunks, if different. */
  chunkLatencyMs?: number
  /** Virtual cost of rendering one frame, on whichever thread renders. */
  renderMsPerFrame?: number
  /** Patch instantiation time in the renderer. */
  instantiateMs?: number
  renderAhead?: Partial<RenderAheadOptions>
}

export type SimResult = {
  /** Channel 0 as the device played it, one sample per device frame. */
  out: Float32Array
  rate: number
  /** Device frame at which the kernel's τ = 0 played. */
  origin: number
  /** `ahead`: the frame each epoch's splice landed at. */
  splices: number[]
  /** `ahead`: frames the renderer rendered, re-renders included. */
  rendered: number
  deviceUnderruns: number
  processor?: SimProcessor
}

function stallEnd(stalls: Stall[], t: number): number {
  for (const s of stalls) if (t >= s.at && t < s.at + s.ms) return s.at + s.ms
  return t
}

export function simulate(cfg: SimConfig): SimResult {
  const Q = cfg.quantum ?? 128
  const rate = cfg.manifest.rate
  const B = cfg.deviceFrames ?? 256
  const Bq = Math.ceil(B / Q)
  const H = cfg.hostChannels ?? 2
  const L = cfg.msgLatencyMs ?? 1
  const Lc = cfg.chunkLatencyMs ?? L
  const stalls = [...(cfg.stalls ?? [])].sort((a, b) => a.at - b.at)
  const qMs = (Q / rate) * 1000
  const slots = Math.ceil(cfg.durationMs / qMs)
  const out = new Float32Array(slots * Q)
  const controls = [...(cfg.controls ?? [])].sort((a, b) => a.at - b.at)

  // ── players ──
  type Msg = { arrival: number; run: () => void }
  /** Insert in arrival order (FIFO among equal arrivals). */
  const enqueue = (q: Msg[], m: Msg) => {
    let i = q.length
    while (i > 0 && q[i - 1]!.arrival > m.arrival) i--
    q.splice(i, 0, m)
  }
  const toWorklet: Msg[] = []
  const toWorker: Msg[] = []
  let produce: (start: number, output: Float32Array[]) => void
  let processor: SimProcessor | undefined
  let origin = 0
  const splices: number[] = []
  let totalRendered = 0

  if (cfg.variant === 'direct') {
    const p = new DirectProcessor(cfg.kernel, new ParamWriter(cfg.manifest))
    cfg.kernel.seek(0)
    cfg.kernel.beginFadeIn()
    for (const c of controls) toWorklet.push({ arrival: c.at + L, run: () => p.onMessage(c.msg) })
    produce = (start, output) => {
      while (toWorklet.length > 0 && toWorklet[0]!.arrival <= start) toWorklet.shift()!.run()
      p.process(output)
    }
  } else {
    let workerClock = 0
    const workletSide = new SimPort((msg) => enqueue(toWorker, {
      arrival: audioClock + L, run: () => ra.playhead((msg as { frame: number }).frame),
    }))
    const hostPort = new SimPort()
    processor = newAheadProcessor(hostPort)
    let firstChunk = -1
    let epoch = 0
    const ra = new RenderAhead(rate, (chunk: Chunk) => {
      if (firstChunk < 0) firstChunk = chunk.start
      if (chunk.epoch !== epoch) { epoch = chunk.epoch; splices.push(chunk.start) }
      rendered += chunk.data[0]!.length
      totalRendered += chunk.data[0]!.length
      enqueue(toWorklet, { arrival: workerClock + Lc, run: () => workletSide.deliver(chunk) })
    }, () => workerClock, cfg.renderAhead, (e) => { throw new Error(e) })
    let rendered = 0
    let audioClock = 0
    hostPort.deliver({ type: 'init', port: workletSide })
    const token = ra.beginLoad()
    cfg.kernel.seek(0)
    toWorker.push({ arrival: L, run: () => ra.adopt(token, cfg.kernel, cfg.manifest) })
    for (const c of controls) {
      const m = c.msg
      if (m.type === 'param') toWorker.push({ arrival: c.at + L, run: () => ra.setParam(m.name, m.value) })
      else {
        let t = 0
        toWorker.push({ arrival: c.at + L, run: () => { t = ra.beginLoad() } })
        toWorker.push({ arrival: c.at + L + (cfg.instantiateMs ?? 10), run: () => ra.adopt(t, m.kernel, m.manifest) })
      }
    }
    toWorker.sort((a, b) => a.arrival - b.arrival)

    // A kernel's render takes virtual time on the renderer's clock, where
    // RenderAhead measures it.
    const msPerFrame = cfg.renderMsPerFrame ?? 0
    const cost = (k: WasmKernel) => {
      const process = k.process.bind(k)
      k.process = (o, n) => { process(o, n); workerClock += n * msPerFrame }
    }
    cost(cfg.kernel)
    for (const c of controls) if (c.msg.type === 'load') cost(c.msg.kernel)

    const runWorker = (until: number) => {
      for (;;) {
        workerClock = stallEnd(stalls, workerClock)
        if (workerClock > until) return
        while (toWorker.length > 0 && toWorker[0]!.arrival <= workerClock) toWorker.shift()!.run()
        const before = rendered
        const more = ra.step()
        if (rendered === before && !more) {
          const next = toWorker[0]?.arrival ?? Infinity
          if (next > until) { workerClock = Math.max(workerClock, until); return }
          workerClock = Math.max(workerClock, next)
        }
      }
    }
    produce = (start, output) => {
      audioClock = start
      runWorker(start)
      while (toWorklet.length > 0 && toWorklet[0]!.arrival <= start) toWorklet.shift()!.run()
      processor!.process([], [output], {})
      if (origin === 0 && firstChunk >= 0) origin = firstChunk
    }
  }

  // ── the device ──
  const audioStalls = cfg.variant === 'direct' ? stalls : []
  const ready: number[] = []              // ready time per produced quantum
  const data: Float32Array[] = []         // channel 0 per produced quantum
  const consumed: number[] = []           // play time per played quantum
  let threadFree = 0
  let played = 0
  let deviceUnderruns = 0
  const output = Array.from({ length: H }, () => new Float32Array(Q))
  let directOrigin = -1
  for (let j = 0; j < slots; j++) {
    const T = j * qMs + (B / rate) * 1000
    // the audio thread renders every quantum the FIFO has room for by T
    for (;;) {
      const k = ready.length
      if (k - Bq >= played) break
      const room = k - Bq < 0 ? 0 : consumed[k - Bq]!
      const start = stallEnd(audioStalls, Math.max(threadFree, room))
      if (start > T) break
      for (const c of output) c.fill(0)
      produce(start, output)
      // the direct processor renders on this thread; render-ahead only copies
      const done = start + (cfg.variant === 'direct' ? Q * (cfg.renderMsPerFrame ?? 0) : 0)
      ready.push(done)
      data.push(output[0]!.slice())
      threadFree = done
    }
    if (played < ready.length && ready[played]! <= T) {
      out.set(data[played]!, j * Q)
      if (directOrigin < 0) directOrigin = j * Q
      data[played] = undefined as unknown as Float32Array
      consumed.push(T)
      played++
    } else deviceUnderruns++
  }
  return {
    out, rate, deviceUnderruns, processor, splices, rendered: totalRendered,
    origin: cfg.variant === 'direct' ? directOrigin : origin,
  }
}

// ── measures ────────────────────────────────────────────────────────────────

/** Quanta that played as all zeros after the stream first sounded, and the
 *  number of separate silent runs (underrun events). */
export function zeroQuanta(out: Float32Array, Q = 128): { quanta: number; events: number } {
  let started = false, quanta = 0, events = 0, inRun = false
  for (let q = 0; q + Q <= out.length; q += Q) {
    let zero = true
    for (let s = q; s < q + Q; s++) if (out[s] !== 0) { zero = false; break }
    if (!zero) { started = true; inRun = false; continue }
    if (!started) continue
    quanta++
    if (!inRun) { events++; inRun = true }
  }
  // a trailing silent run that never recovered still counts
  return { quanta, events }
}

/** The largest sample-to-sample step from `from` on. */
export function maxStep(out: Float32Array, from = 1): number {
  let m = 0
  for (let i = Math.max(1, from); i < out.length; i++) m = Math.max(m, Math.abs(out[i]! - out[i - 1]!))
  return m
}

/** The first index from which `a` and `b` differ, or -1. */
export function firstDiff(a: Float32Array, b: Float32Array): number {
  const n = Math.min(a.length, b.length)
  for (let i = 0; i < n; i++) if (a[i] !== b[i]) return i
  return -1
}

/** A seeded schedule of stalls: gaps uniform in [gapMin, gapMax] ms, lengths
 *  uniform in [min, max] ms (mulberry32). */
export function stallSchedule(seed: number, durationMs: number, min = 5, max = 200, gapMin = 300, gapMax = 2000): Stall[] {
  let a = seed >>> 0
  const rand = () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
  const stalls: Stall[] = []
  for (let t = 500 + rand() * gapMax; t < durationMs - max; ) {
    const ms = min + rand() * (max - min)
    stalls.push({ at: t, ms })
    t += ms + gapMin + rand() * (gapMax - gapMin)
  }
  return stalls
}
