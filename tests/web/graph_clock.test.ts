/**
 * graph_clock.test.ts — closed-form clock properties of the playground
 * generators (`sine`, `saw`, `pulse`) and the signal-time `delay`, checked on
 * small self-contained PatchGraphs compiled by `diffcli compile-wasm` and run
 * through WasmKernel at arbitrary τ (`seek`).
 *
 *  1. A phasor whose Q32 increment divides 2^32·2^14 is bitwise periodic over
 *     2^14 frames, near frame 0 and at frame ≈ 2^40 (the BLEP waveform is a
 *     pure function of the exact phase and increment). At the negated freq
 *     the saw and the pulse are the same waveforms played backward, bit for
 *     bit, and the saw is also the forward saw negated.
 *  2. A delay time of 0 is the source bitwise (the tap clock equals τ), so
 *     dry − tap is exactly 0.
 *  3. The delay tap resolves 1/65536 frame at frame ≈ 2^40: a whole-frame
 *     time reads the source k frames back (or ahead, for a negative time) bit
 *     for bit, and one more 1/65536 frame moves the tap.
 *  4. An anchored re-tune (the `#phase` companion) leaves the source bitwise
 *     unchanged at the switch frame, where a naive write jumps it; a delayed
 *     copy follows the NEW pitch — it is the re-tuned source's own history.
 *     A knob wired straight into several freq inlets anchors them all with
 *     one companion.
 */
import { describe, test, expect, setDefaultTimeout } from 'bun:test'
import { spawnSync } from 'bun'
import { writeFileSync, readFileSync, mkdtempSync } from 'fs'
import { tmpdir } from 'os'
import { join, resolve, dirname } from 'path'
import { fileURLToPath } from 'url'
import { WasmKernel, kernelParams, type KernelManifest } from '../../web/runtime/index'

setDefaultTimeout(60_000)

const __dirname = dirname(fileURLToPath(import.meta.url))
const repoRoot = resolve(__dirname, '../..')
const diffcli = process.env.TROPICAL_DIFFCLI
  ? resolve(process.env.TROPICAL_DIFFCLI)
  : resolve(repoRoot, 'lean/.lake/build/bin/diffcli')
const dir = mkdtempSync(join(tmpdir(), 'graph-clock-'))

const Q32 = 4294967296
const MOD = 1n << 32n
const wrap = (x: bigint): bigint => ((x % MOD) + MOD) % MOD

function run(args: string[]): Buffer {
  const r = spawnSync([diffcli, ...args], { cwd: repoRoot })
  if (r.exitCode !== 0) throw new Error(`diffcli ${args[0]} failed: ${r.stderr.toString()}`)
  return r.stdout as Buffer
}

type Loaded = {
  kernel: WasmKernel
  rate: number
  C: number
  set: (name: string, value: number) => void
  at: (frame: number, n: number) => Float64Array
}

async function load(name: string, graph: unknown): Promise<Loaded> {
  const path = join(dir, `${name}.json`)
  writeFileSync(path, JSON.stringify(graph))
  const wasmPath = join(dir, `${name}.wasm`)
  run(['compile-wasm', path, `--out=${wasmPath}`])
  const manifest = JSON.parse(run(['compile-kernel', path]).toString()) as KernelManifest
  const kernel = await WasmKernel.instantiate(new Uint8Array(readFileSync(wasmPath)), manifest)
  const names = kernelParams(manifest).map((p) => p.name)
  return {
    kernel,
    rate: manifest.rate,
    C: kernel.outputChannelCount,
    set: (param, value) => {
      const i = names.indexOf(`param:${param}`)
      if (i < 0) throw new Error(`kernel has no input param:${param}`)
      kernel.setParam(i, value)
    },
    at: (frame, n) => { kernel.seek(frame); return kernel.render(n).slice() },
  }
}

/** The phasor's per-sample Q32 increment, as the kernel computes it. */
const incOf = (freq: number, rate: number): bigint => BigInt(Math.trunc((freq * Q32) / rate))

const channel = (x: Float64Array, C: number, ch: number): Float64Array =>
  Float64Array.from({ length: x.length / C }, (_, s) => x[s * C + ch]!)

describe('playground generators and delay: closed-form clock', () => {
  // channels: 0 out, 1 saw, 2 sin, 3 sq, 4 tap
  const voices = {
    nodes: [
      { id: 'saw', kind: 'saw', params: { freq: 330 } },
      { id: 'sin', kind: 'sine', params: { freq: 330 } },
      { id: 'sq', kind: 'pulse', params: { freq: 330, width: 0.3 } },
      { id: 't', kind: 'knob', params: { value: 0.001 } },
      { id: 'tap', kind: 'delay', in: { in: ['saw'], amount: ['t'] } },
    ],
    out: 'tap',
    taps: ['saw', 'sin', 'sq', 'tap'],
  }

  test('an integer-ratio increment is bitwise periodic over 2^14 frames, near frame 0 and at frame ≈ 2^40', async () => {
    const k = await load('periodic', voices)
    // increment N + ½ with N = 10·2^20: trunc gives N, and 2^14·N ≡ 0 (mod 2^32)
    const freq = ((10 * 2 ** 20 + 0.5) * k.rate) / Q32
    expect(incOf(freq, k.rate)).toBe(10n * 2n ** 20n)
    for (const v of ['saw', 'sin', 'sq']) k.set(`${v}.freq`, freq)
    const P = 1 << 14
    for (const start of [0, 777, 123_456_789, 2 ** 40 + 3]) {
      const a = k.at(start, P), b = k.at(start + P, P)
      let bad = 0
      for (let s = 0; s < P; s++)
        for (const ch of [1, 2, 3]) if (!Object.is(a[s * k.C + ch], b[s * k.C + ch])) bad++
      expect(bad).toBe(0)
    }
    // and an increment that does not divide is not periodic
    for (const v of ['saw', 'sin', 'sq']) k.set(`${v}.freq`, freq * 1.01)
    const a = k.at(1000, P), b = k.at(1000 + P, P)
    let moved = 0
    for (let s = 0; s < P; s++) if (!Object.is(a[s * k.C + 1], b[s * k.C + 1])) moved++
    expect(moved).toBeGreaterThan(P / 2)
  })

  test('a negative freq plays the saw and the pulse backward, bit for bit', async () => {
    // The phasor at −f is the phasor at f read backward: its Q32 increment is
    // −inc (trunc is symmetric), so p(−f, n) = p(f, −n). The BLEP corrects the
    // reversed wrap too, so saw(−f)[n] = saw(f)[−n] and pulse(−f)[n] =
    // pulse(f)[−n]. A period of P frames makes frame −n readable inside a
    // forward render.
    const k = await load('reverse', voices)
    const freq = ((10 * 2 ** 20 + 0.5) * k.rate) / Q32
    const P = 1 << 14
    for (const start of [0, 2 ** 40 + 3]) {
      for (const v of ['saw', 'sq']) k.set(`${v}.freq`, freq)
      const fwd = k.at(start, P)
      for (const v of ['saw', 'sq']) k.set(`${v}.freq`, -freq)
      const rev = k.at(start, P)
      let bad = 0, corrected = 0
      for (let s = 0; s < P; s++) {
        // frame start + s at −f against frame −(start + s) at +f, which is
        // fwd[r] for start + r ≡ −(start + s) (mod P)
        const r = (((-2 * start - s) % P) + P) % P
        for (const ch of [1, 3]) if (!Object.is(rev[s * k.C + ch], fwd[r * k.C + ch])) bad++
        // the saw is odd about the wrap, so the reversed saw is also −saw(f)
        if (!Object.is(rev[s * k.C + 1], -fwd[s * k.C + 1]) && !(rev[s * k.C + 1] === 0 && fwd[s * k.C + 1] === 0)) bad++
        // and the BLEP is live at −f: samples next to the wrap are not 2p − 1
        const p = Number(wrap(-incOf(freq, k.rate) * BigInt(start + s))) / Q32
        if (rev[s * k.C + 1] !== 2 * p - 1) corrected++
      }
      expect(bad).toBe(0)
      expect(corrected).toBeGreaterThan(0)
    }
  })

  test('a signal delay time of 0 is the source, bitwise: dry − tap is exactly 0', async () => {
    const k = await load('null', {
      nodes: [
        { id: 'saw', kind: 'saw', params: { freq: 330 } },
        { id: 'lfo', kind: 'sine', params: { freq: 3 } },
        { id: 'depth', kind: 'knob', params: { value: 0.0015 } },
        { id: 'd', kind: 'ring', in: { in: ['lfo', 'depth'] } },
        { id: 'tap', kind: 'delay', in: { in: ['saw'], amount: ['d'] } },
        { id: 'neg', kind: 'knob', params: { value: -1 } },
        { id: 'wt', kind: 'ring', in: { in: ['tap', 'neg'] } },
        { id: 'fl', kind: 'mix', in: { in: ['saw', 'wt'] } },
      ],
      out: 'fl',
      taps: ['saw', 'tap', 'fl'],
    })
    for (const depth of [0, 0.0015]) {
      k.set('depth.value', depth)
      for (const start of [0, 98_765, 2 ** 40 + 11]) {
        const x = k.at(start, 512)
        let tapDiff = 0, fl = 0, dry = 0
        for (let s = 0; s < 512; s++) {
          tapDiff = Math.max(tapDiff, Math.abs(x[s * k.C + 2]! - x[s * k.C + 1]!))
          fl = Math.max(fl, Math.abs(x[s * k.C + 3]!))
          dry = Math.max(dry, Math.abs(x[s * k.C + 1]!))
        }
        expect(dry).toBeGreaterThan(0.5)
        if (depth === 0) {
          expect(tapDiff).toBe(0)
          expect(fl).toBe(0)
        } else {
          expect(tapDiff).toBeGreaterThan(1e-3) // a real sweep is not a null
        }
      }
    }
  })

  test('the delay tap resolves 1/65536 frame at frame ≈ 2^40, either sign', async () => {
    const k = await load('resolution', voices)
    const unit = 1 / (k.rate * 65536) // one clock-rail step, in seconds
    const n = 512, lag = 37
    for (const start of [1_000_003, 2 ** 40 + 5]) {
      // whole frames back: the tap at τ is the source at τ − lag, bit for bit
      // (+½ step keeps the float time clear of the truncation boundary)
      k.set('t.value', (lag * 65536 + 0.5) * unit)
      const tap = channel(k.at(start, n), k.C, 4)
      const back = channel(k.at(start - lag, n), k.C, 1)
      for (let s = 0; s < n; s++) expect(Object.is(tap[s], back[s])).toBe(true)
      // a negative time reads ahead: the source at τ + lag
      k.set('t.value', -(lag * 65536 + 0.5) * unit)
      const ahead = channel(k.at(start, n), k.C, 4)
      const fwd = channel(k.at(start + lag, n), k.C, 1)
      for (let s = 0; s < n; s++) expect(Object.is(ahead[s], fwd[s])).toBe(true)
      // one more step (1/65536 frame) moves the tap, by a sub-sample amount
      k.set('t.value', (lag * 65536 + 1.5) * unit)
      const nudged = channel(k.at(start, n), k.C, 4)
      let changed = 0, maxDiff = 0
      for (let s = 0; s < n; s++) {
        if (!Object.is(nudged[s], tap[s])) changed++
        maxDiff = Math.max(maxDiff, Math.abs(nudged[s]! - tap[s]!))
      }
      expect(changed).toBeGreaterThan(n / 2)
      expect(maxDiff).toBeLessThan(1e-3)
    }
  })

  test('an anchored re-tune is seamless at the switch frame; a delayed copy follows the new pitch', async () => {
    const k = await load('anchor', voices)
    const lag = 37
    const f0 = 330, f1 = 330 * 1.05
    const inc0 = incOf(f0, k.rate), inc1 = incOf(f1, k.rate)
    for (const now of [12_345_678, 9_876_543_211]) {
      // the default state: f0, offset 0, the tap `lag` frames back
      for (const v of ['saw', 'sin', 'sq']) { k.set(`${v}.freq`, f0); k.set(`${v}.freq#phase`, 0) }
      k.set('t.value', (lag * 65536 + 0.5) / (k.rate * 65536))
      const before = k.at(now, 2)
      // anchor: bump the offset by (inc₀ − inc₁)·now so the phase at `now` holds
      const off = wrap((inc0 - inc1) * BigInt(now))
      for (const v of ['saw', 'sin', 'sq']) { k.set(`${v}.freq`, f1); k.set(`${v}.freq#phase`, Number(off) / Q32) }
      const after = k.at(now, 2)
      for (const ch of [1, 2, 3]) expect(Object.is(after[ch], before[ch])).toBe(true)
      // the pitch did change (the pulse is flat between its edges, so check
      // the saw and the sine)
      for (const ch of [1, 2]) expect(after[k.C + ch]).not.toBe(before[k.C + ch])
      // the delayed copy is the re-tuned source `lag` frames back — its
      // history at the new pitch — not the old tap
      const history = k.at(now - lag, 1)
      expect(Object.is(after[4], history[1])).toBe(true)
      expect(after[4]).not.toBe(before[4])
      // the naive write (no anchor) jumps the source
      for (const v of ['saw', 'sin', 'sq']) k.set(`${v}.freq#phase`, 0)
      const naive = k.at(now, 1)
      expect(Math.abs(naive[1]! - before[1]!)).toBeGreaterThan(1e-3)
    }
  })

  test('a knob wired straight into several freq inlets anchors them all with one #phase companion', async () => {
    // channels: 0 out, 1 saw, 2 sin, 3 sq, 4 src, 5 fm — every generator's
    // freq is the knob `k`, so the host's bump (inc₀ − inc₁)·now, a function
    // of the knob's value alone, is the same for each
    const k = await load('knob-anchor', {
      nodes: [
        { id: 'k', kind: 'knob', params: { value: 330 } },
        { id: 'zero', kind: 'knob', params: { value: 0 } },
        { id: 'saw', kind: 'saw', in: { freq: ['k'] } },
        { id: 'sin', kind: 'sine', in: { freq: ['k'] } },
        { id: 'sq', kind: 'pulse', params: { width: 0.3 }, in: { freq: ['k'] } },
        { id: 'src', kind: 'source', params: { morph: 0.4 }, in: { freq: ['k'] } },
        { id: 'fm', kind: 'fm', params: { depth: 0 }, in: { in: ['zero'], carrier: ['k'] } },
        { id: 'mx', kind: 'mix', in: { in: ['saw', 'sin', 'sq', 'src', 'fm'] } },
      ],
      out: 'mx',
      taps: ['saw', 'sin', 'sq', 'src', 'fm'],
    })
    const f0 = 330, f1 = 330 * 1.05
    const inc0 = incOf(f0, k.rate), inc1 = incOf(f1, k.rate)
    for (const now of [12_345_678, 9_876_543_211]) {
      k.set('k.value', f0); k.set('k.value#phase', 0)
      const before = k.at(now, 2)
      k.set('k.value', f1); k.set('k.value#phase', Number(wrap((inc0 - inc1) * BigInt(now))) / Q32)
      const after = k.at(now, 2)
      for (const ch of [1, 2, 3, 4, 5]) expect(Object.is(after[ch], before[ch])).toBe(true)
      for (const ch of [1, 2, 4, 5]) expect(after[k.C + ch]).not.toBe(before[k.C + ch])
      // the naive write jumps every one of them
      k.set('k.value#phase', 0)
      const naive = k.at(now, 1)
      for (const ch of [1, 2, 4, 5]) expect(Math.abs(naive[ch]! - before[ch]!)).toBeGreaterThan(1e-3)
    }
  })
})
