/**
 * render_at.test.ts — `WasmKernel.renderAt`: the kernel evaluated at
 * caller-chosen Q48.16 clock values, and the player's curve display built on
 * it (`web/worker/curve.ts`).
 *
 *  1. One period of a spline oscillator across N fractional-frame τ values
 *     equals the f64 reference spline at the exact integer phase each τ
 *     gives (`design/spline.md`), for both bases, at τ near 0 and ten hours
 *     in, on a two-channel kernel (so a channel-stride bug cannot pass). The
 *     τ values carry fraction bits the reference reads, so a kernel that
 *     dropped them would fail.
 *  2. At whole frames, `renderAt` is `render` bit for bit.
 *  3. The display path: `cycleCurve` is `renderAt` on its τ grid bit for bit,
 *     starts at the cycle's phase 0, closes, follows a param write, and two
 *     redraws within one cycle at fixed knobs draw the same curve.
 */
import { describe, test, expect, afterAll } from 'bun:test'
import { spawnSync } from 'bun'
import { readFileSync, writeFileSync, existsSync, mkdtempSync, rmSync } from 'fs'
import { join, resolve, dirname } from 'path'
import { tmpdir } from 'os'
import { fileURLToPath } from 'url'
import { WasmKernel, tauOfFrame, type KernelManifest } from '../../web/runtime/index'
import { channelAt, cycleCurve, displayFreq, phasorInc } from '../../web/worker/curve'

const __dirname = dirname(fileURLToPath(import.meta.url))
const repoRoot = resolve(__dirname, '../..')
const diffcli = resolve(repoRoot, 'lean/.lake/build/bin/diffcli')
const distDir = resolve(repoRoot, 'web/dist/patches')
const cliEnv = { ...process.env, PATH: `${process.env.HOME}/.elan/bin:${process.env.PATH}` }
const scratch = mkdtempSync(join(tmpdir(), 'render-at-'))
afterAll(() => rmSync(scratch, { recursive: true, force: true }))

type Basis = 'CatmullRom' | 'BSpline'
const FREQ = 220
const WAVE12 = [0, 0.8, 0.5, 0.9, 0.2, 0.4, 0, -0.4, -0.2, -0.9, -0.5, -0.8]
const WAVE5 = [0.25, -0.5, 0.75, 1, -0.125]

const phaseOf = (instance: string) => ({ op: 'toInt', args: [
  { op: 'mul', args: [{ op: 'ref', instance, output: 'phase' }, 4294967296] }] })

/** One clock at `FREQ` reading two periodic splines: channel 0 a Catmull-Rom
 *  over `WAVE12` whose points are params `c0…`, channel 1 a B-spline over
 *  `WAVE5` at fixed points. */
function twoSplines() {
  return {
    schema: 'tropical_program_2', name: 'render_at_splines',
    body: { op: 'block', decls: [
      ...WAVE12.map((value, j) => ({ op: 'paramDecl', name: `c${j}`, value })),
      { op: 'instanceDecl', name: 'clock', program: 'ClockPhasor', inputs: { clk: { op: 'clock' }, freq: FREQ } },
      { op: 'instanceDecl', name: 'cr', program: `PeriodicCatmullRom${WAVE12.length}`, inputs: {
        phase: phaseOf('clock'), ...Object.fromEntries(WAVE12.map((_, j) => [`c${j}`, { op: 'param', name: `c${j}` }])) } },
      { op: 'instanceDecl', name: 'bs', program: `PeriodicBSpline${WAVE5.length}`, inputs: {
        phase: phaseOf('clock'), ...Object.fromEntries(WAVE5.map((v, j) => [`c${j}`, v])) } },
    ], assigns: [
      { op: 'outputAssign', name: 'dac.out', expr: { op: 'ref', instance: 'cr', output: 'out' } },
      { op: 'outputAssign', name: 'dac.out', channel: 1, expr: { op: 'ref', instance: 'bs', output: 'out' } },
    ] },
  }
}

async function compile(name: string, program: unknown): Promise<{ k: WasmKernel; m: KernelManifest }> {
  const src = join(scratch, `${name}.json`), wasm = join(scratch, `${name}.wasm`)
  writeFileSync(src, JSON.stringify(program))
  const w = spawnSync([diffcli, 'compile-wasm', src, `--out=${wasm}`, '--frames', '128'], { cwd: repoRoot, env: cliEnv })
  if (w.exitCode !== 0) throw new Error(`compile-wasm failed: ${w.stderr?.toString()}`)
  const c = spawnSync([diffcli, 'compile-kernel', src, '--frames', '128'], { cwd: repoRoot, env: cliEnv })
  if (c.exitCode !== 0) throw new Error(`compile-kernel failed: ${c.stderr?.toString()}`)
  const m = JSON.parse(c.stdout.toString()) as KernelManifest
  return { k: await WasmKernel.instantiate(new Uint8Array(readFileSync(wasm)), m), m }
}

// ── The reference ─────────────────────────────────────────────────────────

/** The phase on ℤ/2³² at Q48.16 clock `tau`: `⌊inc · τ / 2¹⁶⌋ mod 2³²`. */
const phaseAt = (inc: bigint, tau: bigint) => ((inc * tau) >> 16n) & 0xffffffffn

/** The cardinal weights of the window at `t` (Tropicaltest/Spline.lean). */
function weights(basis: Basis, t: number): number[] {
  const t2 = t * t, t3 = t2 * t
  if (basis === 'CatmullRom')
    return [(-t3 + 2 * t2 - t) / 2, (3 * t3 - 5 * t2 + 2) / 2, (-3 * t3 + 4 * t2 + t) / 2, (t3 - t2) / 2]
  const u = 1 - t
  return [(u * u * u) / 6, (3 * t3 - 6 * t2 + 4) / 6, (-3 * t3 + 3 * t2 + 3 * t + 1) / 6, t3 / 6]
}

/** The periodic spline at integer phase `p` (Q0.32): segment `⌊p·K / 2³²⌋`,
 *  local coordinate the fraction, window `c[i−1 … i+2]` mod K. */
function splineAt(basis: Basis, cs: number[], p: bigint): number {
  const K = cs.length, s = p * BigInt(K)
  const i = Number(s >> 32n), t = Number(s & 0xffffffffn) / 2 ** 32
  const w = weights(basis, t)
  let v = 0
  for (let j = 0; j < 4; j++) v += w[j]! * cs[(((i - 1 + j) % K) + K) % K]!
  return v
}

const CHANNELS: [Basis, number[]][] = [['CatmullRom', WAVE12], ['BSpline', WAVE5]]
const FAR = 44100 * 36000 // ten hours in

describe('WasmKernel.renderAt', () => {
  const N = 1000

  test(`one period at ${N} fractional τ ≡ the f64 reference spline, both channels`, async () => {
    const { k, m } = await compile('two', twoSplines())
    expect(k.outputChannelCount).toBe(2)
    const inc = phasorInc(FREQ, m.rate)
    const cycle = (1n << 48n) / inc // one cycle in clock words
    for (const start of [0, FAR]) {
      const tau0 = tauOfFrame(start)
      const taus = BigInt64Array.from({ length: N }, (_, j) => tau0 + (cycle * BigInt(j)) / BigInt(N))
      const got = k.renderAt(taus)
      expect(got.length).toBe(2 * N)
      CHANNELS.forEach(([basis, cs], c) => {
        let worst = 0, fractional = 0, fractionMatters = 0
        for (let j = 0; j < N; j++) {
          const want = splineAt(basis, cs, phaseAt(inc, taus[j]!))
          worst = Math.max(worst, Math.abs(got[2 * j + c]! - want))
          if ((taus[j]! & 0xffffn) !== 0n) {
            fractional++
            if (splineAt(basis, cs, phaseAt(inc, taus[j]! & ~0xffffn)) !== want) fractionMatters++
          }
        }
        expect(worst).toBeLessThanOrEqual(1e-12)
        expect(fractional).toBeGreaterThan(N * 0.9)
        expect(fractionMatters).toBeGreaterThan(N * 0.9)
      })
    }
  })

  async function expectWholeFramesAreRender(k: WasmKernel, start: number, M: number) {
    k.seek(start)
    const rendered = k.render(M)
    const at = k.renderAt(BigInt64Array.from({ length: M }, (_, n) => tauOfFrame(start + n)))
    expect(at.length).toBe(rendered.length)
    for (let i = 0; i < at.length; i++) expect(at[i]).toBe(rendered[i]!)
    // renderAt leaves the frame counter alone
    expect(k.frame).toBe(start + M)
  }

  test('at whole frames renderAt ≡ render bit for bit (τ = 0 and ten hours in)', async () => {
    const { k } = await compile('whole', twoSplines())
    await expectWholeFramesAreRender(k, 0, 1000)
    await expectWholeFramesAreRender(k, FAR + 17, 300)
  })

  const shipped = existsSync(join(distDir, 'spline-voice.wasm'))
  ;(shipped ? test : test.skip)('at whole frames renderAt ≡ render bit for bit (shipped spline-voice, 3 channels)', async () => {
    const m = JSON.parse(readFileSync(join(distDir, 'spline-voice.manifest.json'), 'utf-8')) as KernelManifest
    const k = await WasmKernel.instantiate(new Uint8Array(readFileSync(join(distDir, 'spline-voice.wasm'))), m)
    expect(k.outputChannelCount).toBe(3)
    await expectWholeFramesAreRender(k, 3000, 777)
  })

  test('tauOfFrame is exact or refuses', () => {
    expect(tauOfFrame(1.5)).toBe(98304n)
    expect(tauOfFrame(2 ** -16)).toBe(1n)
    expect(() => tauOfFrame(1 / 3)).toThrow(RangeError)
    expect(() => tauOfFrame(2 ** 40)).toThrow(RangeError)
  })
})

describe('cycleCurve (the player\'s curve display)', () => {
  const P = 512

  test('is renderAt on its τ grid bit for bit, on each channel, and the reference spline', async () => {
    const { k, m } = await compile('display', twoSplines())
    const inc = phasorInc(FREQ, m.rate)
    for (const frame of [123457, FAR + 5]) {
      CHANNELS.forEach(([basis, cs], c) => {
        const curve = cycleCurve(k, m.rate, frame, c, FREQ, P)!
        const again = k.renderAt(curve.taus)
        for (let j = 0; j < P; j++) {
          expect(curve.data[j]).toBe(again[2 * j + c]!)
          expect(Math.abs(curve.data[j]! - splineAt(basis, cs, phaseAt(inc, curve.taus[j]!)))).toBeLessThanOrEqual(1e-12)
        }
      })
    }
  })

  test('starts at phase 0 of the cycle containing the frame, and closes', async () => {
    const { k, m } = await compile('phase0', twoSplines())
    const inc = phasorInc(FREQ, m.rate)
    const oneWord = inc >> 16n // the phase one clock word advances, at most
    for (const frame of [0, 99, 123457, FAR + 5]) {
      const c = cycleCurve(k, m.rate, frame, 0, FREQ, P)!
      const T = tauOfFrame(frame)
      // the frame lies in the drawn cycle
      expect(c.taus[0]! <= T).toBe(true)
      expect(T < c.taus[0]! + ((1n << 48n) + inc - 1n) / inc).toBe(true)
      // point 0 is the cycle's phase 0 on the Q48.16 grid, point j phase j/P
      for (let j = 0; j < P; j += 37) {
        const want = (BigInt(j) << 32n) / BigInt(P)
        const got = phaseAt(inc, c.taus[j]!)
        expect(got >= want && got <= want + oneWord + 1n).toBe(true)
      }
      // the cycle closes: phase 1 of this cycle (phase 0 of the next) is
      // point 0's value, to within one clock word of phase
      const next = BigInt64Array.of((((c.cycle + 1n) << 48n) + inc - 1n) / inc)
      expect(next[0]! > c.taus[P - 1]!).toBe(true)
      const end = channelAt(k, next, 0)[0]!
      expect(Math.abs(end - c.data[0]!)).toBeLessThan(1e-6)
    }
  })

  test('two redraws within one cycle draw the same curve; a later cycle the same shape', async () => {
    const { k, m } = await compile('redraw', twoSplines())
    // frames 10 and 150 into the cycle containing frame 100 000 (≈ 200 frames long)
    const start = Number(cycleCurve(k, m.rate, 100_000, 0, FREQ, 2)!.taus[0]! >> 16n) + 1
    const a = cycleCurve(k, m.rate, start + 10, 0, FREQ, P)!
    const b = cycleCurve(k, m.rate, start + 150, 0, FREQ, P)!
    expect(b.cycle).toBe(a.cycle)
    for (let j = 0; j < P; j++) expect(b.data[j]).toBe(a.data[j]!)
    const later = cycleCurve(k, m.rate, 100_000 + 44100 * 60, 0, FREQ, P)!
    expect(later.cycle > a.cycle).toBe(true)
    for (let j = 0; j < P; j++) expect(Math.abs(later.data[j]! - a.data[j]!)).toBeLessThan(1e-6)
  })

  test('follows a param write (the curve is the kernel, not a host recomputation)', async () => {
    const { k, m } = await compile('follows', twoSplines())
    const inc = phasorInc(FREQ, m.rate)
    const before = cycleCurve(k, m.rate, 5000, 0, FREQ, P)!
    const i = k.paramIndex('param:c3')
    expect(i).toBeGreaterThanOrEqual(0)
    k.setParam(i, -0.7)
    const after = cycleCurve(k, m.rate, 5000, 0, FREQ, P)!
    const moved = WAVE12.map((v, j) => (j === 3 ? -0.7 : v))
    let changed = 0
    for (let j = 0; j < P; j++) {
      if (after.data[j] !== before.data[j]) changed++
      expect(Math.abs(after.data[j]! - splineAt('CatmullRom', moved, phaseAt(inc, after.taus[j]!)))).toBeLessThanOrEqual(1e-12)
    }
    expect(changed).toBeGreaterThan(P / 4)
  })

  test('the display frequency comes from the kernel\'s own param cell', async () => {
    const { k } = await compile('freq', twoSplines())
    expect(displayFreq(k, { channel: 0, freq: FREQ })).toBe(FREQ)
    expect(displayFreq(k, { channel: 0, freqParam: 'missing' })).toBeNull()
    expect(displayFreq(k, { channel: 0, freqParam: 'c1' })).toBe(0.8)
    k.setParam(k.paramIndex('param:c1'), 3.5)
    expect(displayFreq(k, { channel: 0, freqParam: 'c1' })).toBe(3.5)
    expect(() => cycleCurve(k, 44100, 0, 0, FREQ, 1)).toThrow(RangeError)
    expect(() => cycleCurve(k, 44100, 0, 2, FREQ, 16)).toThrow(RangeError)
  })
})
