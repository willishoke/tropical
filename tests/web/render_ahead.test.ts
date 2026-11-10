/**
 * render_ahead.test.ts — the browser player's underrun resilience, measured
 * in virtual time against shipped kernels (`playout_sim.ts`; no audio device,
 * no browser). The same stall schedule hits the thread that renders the
 * kernel: the audio thread for the old direct-render processor, the
 * render-ahead Worker for the new one.
 *
 *  1. Purity: with no knob moving, frames rendered ahead are the frames the
 *     direct processor renders in place, sample for sample.
 *  2. Seeded 5–200 ms stalls: render-ahead plays no zero-filled quantum and
 *     takes no step beyond the patch's own; the direct processor underruns.
 *  3. A sustained overload past the lookahead: underrun concealment fades out
 *     and back in, never stepping, and resumes at the wall-clock τ.
 *  4. A knob splice is exactly an in-place write at the splice frame (the
 *     anchor discipline's continuity point), `marginMs` after the move; a
 *     splice that lands late crossfades; a drag through the stall schedule
 *     shows what `controlBudget` trades.
 *  5. A patch swap fades the old patch out and the new one in, gap-free.
 *  6. The per-chunk render cost of every shipped patch, against which the
 *     lookahead default is set.
 *
 * Needs web/dist/patches (`bun web/build_patches.ts`).
 */
import { describe, test, expect, afterAll, setDefaultTimeout } from 'bun:test'
import { readFileSync, existsSync } from 'fs'
import { join, resolve, dirname } from 'path'
import { fileURLToPath } from 'url'
import { WasmKernel, ParamWriter, type KernelManifest } from '../../web/runtime/index'
import { RENDER_AHEAD_DEFAULTS } from '../../web/worker/render_ahead'
import {
  simulate, zeroQuanta, maxStep, firstDiff, stallSchedule, type SimConfig,
} from './playout_sim'

setDefaultTimeout(120_000)

const __dirname = dirname(fileURLToPath(import.meta.url))
const distDir = resolve(__dirname, '../../web/dist/patches')
const DEVICE_FRAMES = 256
const STALL_MS = 20_000
const stalls = stallSchedule(1, STALL_MS)
const overload = [{ at: 5000, ms: 600 }]
/** Render cost as a share of real time, rounded up from the per-chunk timing
 *  below (fixed, so the simulation is deterministic). */
const COST: Record<string, number> = { 'tz-flanger': 0.006, 'spectral-filter-fm': 0.05 }

async function load(slug: string): Promise<{ k: WasmKernel; m: KernelManifest }> {
  const m = JSON.parse(readFileSync(join(distDir, `${slug}.manifest.json`), 'utf-8')) as KernelManifest
  const k = await WasmKernel.instantiate(new Uint8Array(readFileSync(join(distDir, `${slug}.wasm`))), m)
  return { k, m }
}

async function run(slug: string, cfg: Omit<SimConfig, 'kernel' | 'manifest'>) {
  const { k, m } = await load(slug)
  return simulate({
    deviceFrames: DEVICE_FRAMES, renderMsPerFrame: ((COST[slug] ?? 0) * 1000) / m.rate,
    ...cfg, kernel: k, manifest: m,
  })
}

const rows: string[] = []
const row = (...cells: (string | number)[]) => rows.push(`| ${cells.join(' | ')} |`)

describe('render-ahead playout (virtual time, no audio device)', () => {
  if (!existsSync(join(distDir, 'index.json'))) {
    test.skip('(skipped: run `bun web/build_patches.ts` first)', () => {})
    return
  }

  for (const slug of ['tz-flanger', 'spectral-filter-fm']) {
    test(`${slug}: stalls, overload, purity`, async () => {
      const directClean = await run(slug, { variant: 'direct', durationMs: STALL_MS })
      const aheadClean = await run(slug, { variant: 'ahead', durationMs: STALL_MS })
      const normal = maxStep(directClean.out)

      // 1. purity
      const o = aheadClean.origin
      expect(o).toBeGreaterThan(0)
      const n = aheadClean.out.length - o
      expect(firstDiff(aheadClean.out.subarray(o), directClean.out.subarray(0, n))).toBe(-1)
      expect(zeroQuanta(aheadClean.out).quanta).toBe(0)

      // 2. seeded stalls
      const d = await run(slug, { variant: 'direct', durationMs: STALL_MS, stalls })
      const a = await run(slug, { variant: 'ahead', durationMs: STALL_MS, stalls })
      const dz = zeroQuanta(d.out), az = zeroQuanta(a.out)
      expect(dz.quanta).toBeGreaterThan(0) // the schedule does hurt the old path
      expect(az.quanta).toBe(0)
      expect(a.deviceUnderruns).toBe(0)
      expect(maxStep(a.out)).toBe(normal)

      // 3. sustained overload: one underrun, concealed, resumed on the wall clock
      const dov = await run(slug, { variant: 'direct', durationMs: STALL_MS, stalls: overload })
      const aov = await run(slug, { variant: 'ahead', durationMs: STALL_MS, stalls: overload })
      const dovz = zeroQuanta(dov.out), aovz = zeroQuanta(aov.out)
      expect(aovz.events).toBe(1)
      expect(aov.processor!.underruns).toBe(1)
      expect(maxStep(aov.out)).toBeLessThanOrEqual(normal * 1.25)
      const tail = aheadClean.rate // the last second
      expect(firstDiff(aov.out.subarray(-tail), aheadClean.out.subarray(-tail))).toBe(-1)

      const ratio = (x: Float32Array) => (maxStep(x) / normal).toFixed(2)
      row(slug, `seeded stalls (${stalls.length}, 5–200 ms)`, 'direct', dz.quanta, dz.events, ratio(d.out))
      row(slug, `seeded stalls (${stalls.length}, 5–200 ms)`, 'render-ahead', az.quanta, az.events, ratio(a.out))
      row(slug, 'overload (600 ms stall)', 'direct', dovz.quanta, dovz.events, ratio(dov.out))
      row(slug, 'overload (600 ms stall)', 'render-ahead', aovz.quanta, aovz.events, ratio(aov.out))
    })
  }

  test('spectral-filter-fm: a knob splice is an in-place write at the splice frame', async () => {
    const slug = 'spectral-filter-fm', dur = 3000, at = 1000
    for (const [name, value] of [['param:f0.value', 220], ['param:level.value', 0.1]] as const) {
      const controls = [{ at, msg: { type: 'param' as const, name, value } }]
      const latency: Record<string, number> = {}
      for (const variant of ['direct', 'ahead'] as const) {
        const base = await run(slug, { variant, durationMs: dur })
        const moved = await run(slug, { variant, durationMs: dur, controls })
        // the device frame playing when the knob moved
        const heard = (at * base.rate) / 1000 - DEVICE_FRAMES
        latency[variant] = firstDiff(base.out, moved.out) - heard
        if (variant !== 'ahead') continue
        // the same kernel written in place at the splice frame plays the same
        // samples: the splice is no different from today's write at a block
        const { k, m } = await load(slug)
        const S = moved.splices[1]! - moved.origin
        const N = moved.out.length - moved.origin
        const ref = [new Float32Array(N), new Float32Array(N)]
        k.beginFadeIn(0)
        k.process(ref.map((c) => c.subarray(0, S)), S)
        new ParamWriter(m).write(k, name, value, k.frame)
        k.process(ref.map((c) => c.subarray(S)), N - S)
        expect(firstDiff(ref[0]!, moved.out.subarray(moved.origin))).toBe(-1)
      }
      const margin = Math.round((RENDER_AHEAD_DEFAULTS.marginMs * 44100) / 1000)
      expect(latency.ahead!).toBeGreaterThanOrEqual(margin)
      expect(latency.ahead!).toBeLessThan(margin + 4 * 128 + DEVICE_FRAMES)
      const ms = (f: number) => `${f} (${((f / 44100) * 1000).toFixed(1)} ms)`
      row(slug, `knob ${name.replace('param:', '')} latency, frames`, 'direct', ms(latency.direct!), '', '')
      row(slug, `knob ${name.replace('param:', '')} latency, frames`, 'render-ahead', ms(latency.ahead!), '', '')
    }
  })

  test('spectral-filter-fm: a knob drag whose splices land late crossfades', async () => {
    const slug = 'spectral-filter-fm', dur = 3000
    const drag = Array.from({ length: 40 }, (_, i) => ({
      at: 1000 + i * 16,
      msg: { type: 'param' as const, name: 'param:f0.value', value: 110 * 2 ** (i / 40) },
    }))
    const clean = await run(slug, { variant: 'direct', durationMs: dur })
    const d = await run(slug, { variant: 'direct', durationMs: dur, controls: drag })
    // chunks take longer to arrive than the margin: every splice is late
    const a = await run(slug, { variant: 'ahead', durationMs: dur, controls: drag, chunkLatencyMs: 40 })
    expect(a.processor!.lateSplices).toBeGreaterThan(0)
    expect(zeroQuanta(a.out).quanta).toBe(0)
    const bound = 1.25 * Math.max(maxStep(clean.out), maxStep(d.out))
    expect(maxStep(a.out)).toBeLessThanOrEqual(bound)
  })

  test('spectral-filter-fm: a knob dragged through the whole stall schedule', async () => {
    // The trade-off `controlBudget` makes: a drag on a heavy patch renders
    // less far ahead, so a long stall during it can still underrun
    // (concealed). Older epochs' frames past the newest cover the rest.
    const slug = 'spectral-filter-fm'
    const drag = Array.from({ length: Math.floor((STALL_MS - 1000) / 16) }, (_, i) => ({
      at: 500 + i * 16,
      msg: { type: 'param' as const, name: 'param:center.value', value: 900 + 600 * Math.sin(i / 30) },
    }))
    const clean = await run(slug, { variant: 'direct', durationMs: STALL_MS, controls: drag })
    const normal = maxStep(clean.out)
    const d = await run(slug, { variant: 'direct', durationMs: STALL_MS, controls: drag, stalls })
    const a = await run(slug, { variant: 'ahead', durationMs: STALL_MS, controls: drag, stalls })
    const dz = zeroQuanta(d.out), az = zeroQuanta(a.out)
    expect(az.quanta).toBeLessThan(dz.quanta)
    expect(a.processor!.dips).toBeGreaterThan(0) // it did play older epochs
    expect(maxStep(a.out)).toBeLessThanOrEqual(normal * 1.25)
    const work = (a.rendered / (a.out.length - a.origin)).toFixed(1)
    row(slug, 'seeded stalls + a 60 Hz knob drag throughout', 'direct', dz.quanta, dz.events, (maxStep(d.out) / normal).toFixed(2))
    row(slug, `seeded stalls + a 60 Hz knob drag throughout (renders ${work}× real time)`, 'render-ahead', az.quanta, az.events, (maxStep(a.out) / normal).toFixed(2))
  })

  test('patch swap: the old patch fades out, the new one fades in, no gap', async () => {
    const dur = 3000
    const swap = async (variant: 'direct' | 'ahead') => {
      const next = await load('spectral-filter-fm')
      return run('tz-flanger', {
        variant, durationMs: dur,
        controls: [{ at: 1500, msg: { type: 'load', kernel: next.k, manifest: next.m, instantiateQuanta: 4 } }],
      })
    }
    const normal = Math.max(
      maxStep((await run('tz-flanger', { variant: 'direct', durationMs: dur })).out),
      maxStep((await run('spectral-filter-fm', { variant: 'direct', durationMs: dur })).out))
    const d = await swap('direct'), a = await swap('ahead')
    expect(zeroQuanta(a.out).quanta).toBe(0)
    expect(maxStep(a.out)).toBeLessThanOrEqual(normal)
    const dz = zeroQuanta(d.out), az = zeroQuanta(a.out)
    row('tz-flanger → spectral-filter-fm', 'patch swap', 'direct', dz.quanta, dz.events, (maxStep(d.out) / normal).toFixed(2))
    row('tz-flanger → spectral-filter-fm', 'patch swap', 'render-ahead', az.quanta, az.events, (maxStep(a.out) / normal).toFixed(2))
  })

  test('per-chunk render cost of every shipped patch', async () => {
    const entries = JSON.parse(readFileSync(join(distDir, 'index.json'), 'utf-8')) as { slug: string }[]
    const chunk = RENDER_AHEAD_DEFAULTS.chunkFrames
    const lines = [`| patch | µs / ${chunk}-frame chunk | % of real time | refill ${RENDER_AHEAD_DEFAULTS.lookaheadMs} ms |`, '|---|---|---|---|']
    for (const { slug } of entries) {
      const { k, m } = await load(slug)
      const out = Array.from({ length: k.outputChannelCount }, () => new Float32Array(chunk))
      for (let i = 0; i < 50; i++) k.process(out, chunk)
      const reps = 400
      const t0 = performance.now()
      for (let i = 0; i < reps; i++) k.process(out, chunk)
      const ms = (performance.now() - t0) / reps
      const share = ms / ((chunk / m.rate) * 1000)
      lines.push(`| ${slug} | ${(ms * 1000).toFixed(0)} | ${(share * 100).toFixed(2)} | ${(share * RENDER_AHEAD_DEFAULTS.lookaheadMs).toFixed(1)} ms |`)
    }
    console.log(`\n${lines.join('\n')}\n`)
  })

  afterAll(() => {
    if (rows.length === 0) return
    console.log([
      '',
      `| patch | case | player | zero-filled quanta | underrun events | max step / normal |`,
      '|---|---|---|---|---|---|',
      ...rows,
      '',
    ].join('\n'))
  })
})
