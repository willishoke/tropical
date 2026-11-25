/**
 * render_at.test.ts — `WasmKernel.renderAt`: the kernel evaluated at
 * caller-chosen Q48.16 clock values, the surface the player's curve display
 * draws from.
 *
 *  1. One period of a spline oscillator across N fractional-frame τ values
 *     equals the f64 reference spline at the exact integer phase each τ
 *     gives (`design/spline.md`), for both bases, at τ near 0 and ten hours
 *     in. The τ values carry fraction bits the reference reads, so a kernel
 *     that dropped them would fail.
 *  2. At whole frames, `renderAt` is `render` bit for bit (`seek` + `render`
 *     over the same frames), on the test patch and on the shipped
 *     `spline-voice` artifact.
 */
import { describe, test, expect } from 'bun:test'
import { spawnSync } from 'bun'
import { readFileSync, writeFileSync, existsSync, mkdtempSync } from 'fs'
import { join, resolve, dirname } from 'path'
import { tmpdir } from 'os'
import { fileURLToPath } from 'url'
import { WasmKernel, tauOfFrame, type KernelManifest } from '../../web/runtime/index'
import { cycleCurve, displayFreq } from '../../web/worker/curve'

const __dirname = dirname(fileURLToPath(import.meta.url))
const repoRoot = resolve(__dirname, '../..')
const diffcli = resolve(repoRoot, 'lean/.lake/build/bin/diffcli')
const distDir = resolve(repoRoot, 'web/dist/patches')
const cliEnv = { ...process.env, PATH: `${process.env.HOME}/.elan/bin:${process.env.PATH}` }
const scratch = mkdtempSync(join(tmpdir(), 'render-at-'))

type Basis = 'CatmullRom' | 'BSpline'
const FREQ = 220
const WAVE12 = [0, 0.8, 0.5, 0.9, 0.2, 0.4, 0, -0.4, -0.2, -0.9, -0.5, -0.8]
const WAVE5 = [0.25, -0.5, 0.75, 1, -0.125]

/** A clock at `FREQ` reading a periodic spline straight to the output. */
function splineProgram(basis: Basis, cs: number[]) {
  return {
    schema: 'tropical_program_2', name: 'render_at_spline',
    body: { op: 'block', decls: [
      { op: 'instanceDecl', name: 'clock', program: 'ClockPhasor', inputs: { clk: { op: 'clock' }, freq: FREQ } },
      { op: 'instanceDecl', name: 'wave', program: `Periodic${basis}${cs.length}`, inputs: {
        phase: { op: 'toInt', args: [
          { op: 'mul', args: [{ op: 'ref', instance: 'clock', output: 'phase' }, 4294967296] }] },
        ...Object.fromEntries(cs.map((v, j) => [`c${j}`, v])) } },
    ], assigns: [{ op: 'outputAssign', name: 'dac.out', expr: { op: 'ref', instance: 'wave', output: 'out' } }] },
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

/** The phasor increment, as the kernel computes it: `trunc(f · 2³² / rate)`. */
const incOf = (rate: number) => BigInt(Math.trunc((FREQ * 2 ** 32) / rate))

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

describe('WasmKernel.renderAt', () => {
  const N = 1000
  const FAR = 44100 * 36000 // ten hours in

  for (const [basis, cs] of [['CatmullRom', WAVE12], ['BSpline', WAVE5]] as const) {
    test(`${basis}${cs.length}: one period at ${N} fractional τ ≡ the f64 reference spline`, async () => {
      const { k, m } = await compile(`${basis}${cs.length}`, splineProgram(basis, cs))
      expect(k.outputChannelCount).toBe(1)
      const inc = incOf(m.rate)
      // One cycle is 2³² / inc frames, 2⁴⁸ / inc clock words.
      const cycle = (1n << 48n) / inc
      for (const start of [0, FAR]) {
        const tau0 = tauOfFrame(start)
        const taus = BigInt64Array.from({ length: N }, (_, j) => tau0 + (cycle * BigInt(j)) / BigInt(N))
        const got = k.renderAt(taus)
        let worst = 0, fractional = 0, fractionMatters = 0
        for (let j = 0; j < N; j++) {
          const want = splineAt(basis, cs, phaseAt(inc, taus[j]!))
          worst = Math.max(worst, Math.abs(got[j]! - want))
          if ((taus[j]! & 0xffffn) !== 0n) {
            fractional++
            const whole = splineAt(basis, cs, phaseAt(inc, taus[j]! & ~0xffffn))
            if (whole !== want) fractionMatters++
          }
        }
        expect(worst).toBeLessThanOrEqual(1e-12)
        expect(fractional).toBeGreaterThan(N * 0.9)
        expect(fractionMatters).toBeGreaterThan(N * 0.9)
        // the curve is the shape, not a flat line
        expect(Math.max(...got) - Math.min(...got)).toBeGreaterThan(0.5)
      }
    })
  }

  async function expectWholeFramesAreRender(k: WasmKernel, start: number, M: number) {
    k.seek(start)
    const rendered = k.render(M)
    const at = k.renderAt(BigInt64Array.from({ length: M }, (_, n) => tauOfFrame(start + n)))
    expect(at.length).toBe(rendered.length)
    for (let i = 0; i < at.length; i++) expect(at[i]).toBe(rendered[i]!)
    // renderAt leaves the frame counter alone
    expect(k.frame).toBe(start + M)
  }

  test('at whole frames renderAt ≡ render bit for bit (test patch, τ = 0 and ten hours in)', async () => {
    const { k } = await compile('whole', splineProgram('CatmullRom', WAVE12))
    await expectWholeFramesAreRender(k, 0, 1000)
    await expectWholeFramesAreRender(k, FAR + 17, 300)
  })

  test('at whole frames renderAt ≡ render bit for bit (shipped spline-voice)', async () => {
    if (!existsSync(join(distDir, 'spline-voice.wasm'))) {
      console.warn('(skipped: run `bun web/build_patches.ts` first)')
      return
    }
    const m = JSON.parse(readFileSync(join(distDir, 'spline-voice.manifest.json'), 'utf-8')) as KernelManifest
    const k = await WasmKernel.instantiate(new Uint8Array(readFileSync(join(distDir, 'spline-voice.wasm'))), m)
    await expectWholeFramesAreRender(k, 3000, 777)
  })

  // The player's display path: `cycleCurve` is renderAt on the cycle's τ grid,
  // so its points are the reference spline too, and the cycle closes.
  test('cycleCurve: the display\'s points are the kernel\'s curve', async () => {
    const { k, m } = await compile('display', splineProgram('CatmullRom', WAVE12))
    expect(displayFreq(k, { channel: 0, freq: FREQ })).toBe(FREQ)
    expect(displayFreq(k, { channel: 0, freqParam: 'missing' })).toBeNull()
    const P = 512, start = 123457
    const curve = cycleCurve(k, m.rate, start, 0, FREQ, P)
    const inc = incOf(m.rate), step = (m.rate / FREQ) * 65536 / P
    for (let j = 0; j < P; j++) {
      const tau = tauOfFrame(start) + BigInt(Math.round(j * step))
      expect(Math.abs(curve[j]! - splineAt('CatmullRom', WAVE12, phaseAt(inc, tau)))).toBeLessThanOrEqual(1e-12)
    }
  })

  test('tauOfFrame is exact or refuses', () => {
    expect(tauOfFrame(1.5)).toBe(98304n)
    expect(tauOfFrame(2 ** -16)).toBe(1n)
    expect(() => tauOfFrame(1 / 3)).toThrow(RangeError)
    expect(() => tauOfFrame(2 ** 40)).toThrow(RangeError)
  })
})
