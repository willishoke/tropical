/**
 * graph_wasm_vs_jit.test.ts — playground PatchGraphs through the wasm verbs.
 *
 * `diffcli compile-wasm` / `compile-kernel` accept a PatchGraph
 * (`{"nodes":[…],"out":…}`, detected by shape) and compile it through
 * `Playground.compilePlan` and the typed stage-0 split, exactly as
 * `render-graph` (the JIT) does. Each case renders both and compares every
 * output channel (`out` plus the named taps) sample for sample.
 *
 * The bank case also compiles with `TROPICAL_STAGE0=0` — no coefficient
 * prologue, the whole plan in the sample loop — and checks it renders the
 * same values: the split is an optimization, not a semantic requirement.
 */
import { describe, test, expect, setDefaultTimeout } from 'bun:test'
import { spawnSync } from 'bun'
import { writeFileSync, readFileSync, mkdtempSync } from 'fs'
import { tmpdir } from 'os'
import { join, resolve, dirname } from 'path'
import { fileURLToPath } from 'url'
import { WasmKernel, type KernelManifest } from '../../web/runtime/index'

setDefaultTimeout(60_000)

const __dirname = dirname(fileURLToPath(import.meta.url))
const repoRoot = resolve(__dirname, '../..')
const diffcli = process.env.TROPICAL_DIFFCLI
  ? resolve(process.env.TROPICAL_DIFFCLI)
  : resolve(repoRoot, 'lean/.lake/build/bin/diffcli')
const dir = mkdtempSync(join(tmpdir(), 'graph-wasm-'))

function run(args: string[], env: Record<string, string> = {}): Buffer {
  const r = spawnSync([diffcli, ...args], { cwd: repoRoot, env: { ...process.env, ...env } })
  if (r.exitCode !== 0) throw new Error(`diffcli ${args[0]} failed: ${r.stderr.toString()}`)
  return r.stdout as Buffer
}

/** The JIT: every channel, frame-major. */
function renderGraph(path: string, n: number, channels: number): Float64Array {
  const b = run(['render-graph', path, '--all-channels', '--frames', '1', '--buffer', String(n)])
  expect(b.byteLength).toBe(n * channels * 8)
  return new Float64Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength))
}

async function renderWasm(path: string, n: number, env: Record<string, string> = {}) {
  const wasmPath = join(dir, `${Math.random().toString(36).slice(2)}.wasm`)
  run(['compile-wasm', path, `--out=${wasmPath}`], env)
  const manifest = JSON.parse(run(['compile-kernel', path], env).toString()) as KernelManifest
  const k = await WasmKernel.instantiate(new Uint8Array(readFileSync(wasmPath)), manifest)
  return { manifest, out: k.render(n), channels: k.outputChannelCount }
}

const cases: Record<string, unknown> = {
  // a swept flanger over a source, the LFO wired as its modulator, three taps
  sflange: {
    nodes: [
      { id: 'osc', kind: 'source', params: { freq: 220, morph: 0.3 } },
      { id: 'lfo', kind: 'source', params: { freq: 3, morph: 1 } },
      { id: 'fl', kind: 'sflange', params: { depth: 0.002 }, in: { in: ['osc'], mod: ['lfo'] } },
      { id: 'g', kind: 'knob', params: { value: -0.5 } },
      { id: 'inv', kind: 'ring', in: { in: ['osc', 'g'] } },
      { id: 'sum', kind: 'mix', in: { in: ['fl', 'inv'] } },
    ],
    out: 'sum',
    taps: ['osc', 'lfo', 'fl'],
  },
  // a modal bank (coefficients hoisted to the stage-0 prologue) into a filter
  bank: {
    nodes: [
      { id: 'res', kind: 'resonator', params: { freq: 220, decay: 4 } },
      { id: 'flt', kind: 'filter', params: { cutoff: 900, resonance: 0.6 }, in: { in: ['res'] } },
    ],
    out: 'flt',
    taps: ['res'],
  },
  // the band-limited sources and a delay whose time is a signal (an LFO ×
  // a knob, either sign), its saw re-read at τ − d(τ), dry − tap
  blep: {
    nodes: [
      { id: 'saw', kind: 'saw', params: { freq: 330 } },
      { id: 'sq', kind: 'pulse', params: { freq: 220, width: 0.3 } },
      { id: 'sin', kind: 'sine', params: { freq: 110 } },
      { id: 'lfo', kind: 'sine', params: { freq: 3 } },
      { id: 'depth', kind: 'knob', params: { value: 0.002 } },
      { id: 'd', kind: 'ring', in: { in: ['lfo', 'depth'] } },
      { id: 'tap', kind: 'delay', in: { in: ['saw'], amount: ['d'] } },
      { id: 'neg', kind: 'knob', params: { value: -1 } },
      { id: 'wt', kind: 'ring', in: { in: ['tap', 'neg'] } },
      { id: 'fl', kind: 'mix', in: { in: ['saw', 'wt'] } },
      { id: 'out', kind: 'mix', in: { in: ['fl', 'sq', 'sin'] } },
    ],
    out: 'out',
    taps: ['saw', 'sq', 'sin', 'tap', 'fl'],
  },
  // a pitch driven by a τ-invariant cone (knob × knob) and by a signal (an LFO
  // + a knob): any wire into a freq inlet
  wiredFreq: {
    nodes: [
      { id: 'a', kind: 'knob', params: { value: 110 } },
      { id: 'b', kind: 'knob', params: { value: 2 } },
      { id: 'f', kind: 'ring', in: { in: ['a', 'b'] } },
      { id: 'saw', kind: 'saw', in: { freq: ['f'] } },
      { id: 'vib', kind: 'sine', params: { freq: 5 } },
      { id: 'amt', kind: 'knob', params: { value: 3 } },
      { id: 'vd', kind: 'ring', in: { in: ['vib', 'amt'] } },
      { id: 'f2', kind: 'mix', in: { in: ['vd', 'a'] } },
      { id: 'sw', kind: 'sine', in: { freq: ['f2'] } },
      { id: 'out', kind: 'mix', in: { in: ['saw', 'sw'] } },
    ],
    out: 'out',
    taps: ['saw', 'sw'],
  },
}

describe('PatchGraph → compile-wasm ≡ render-graph (JIT)', () => {
  const N = 1024
  for (const [name, graph] of Object.entries(cases)) {
    test(name, async () => {
      const path = join(dir, `${name}.json`)
      writeFileSync(path, JSON.stringify(graph))
      const { manifest, out, channels } = await renderWasm(path, N)
      const taps = (graph as { taps: string[] }).taps
      expect(channels).toBe(1 + taps.length)
      expect(manifest.outputs[0]!.channels).toEqual(
        ['__root__.out', ...taps.map((t) => `__root__.tap:${t}`)])
      const jit = renderGraph(path, N, channels)
      let maxDiff = 0, energy = 0
      for (let i = 0; i < jit.length; i++) {
        maxDiff = Math.max(maxDiff, Math.abs(out[i]! - jit[i]!))
        energy += jit[i]! * jit[i]!
      }
      expect(energy).toBeGreaterThan(1e-3)
      expect(maxDiff).toBeLessThan(1e-12)
    })
  }

  test('bank: the unsplit kernel (TROPICAL_STAGE0=0, coeff? := none) renders the same values', async () => {
    const path = join(dir, 'bank.json')
    writeFileSync(path, JSON.stringify(cases.bank))
    const split = await renderWasm(path, N)
    const unsplit = await renderWasm(path, N, { TROPICAL_STAGE0: '0' })
    let maxDiff = 0
    for (let i = 0; i < split.out.length; i++)
      maxDiff = Math.max(maxDiff, Math.abs(split.out[i]! - unsplit.out[i]!))
    expect(maxDiff).toBeLessThan(1e-12)
  })

  // A v3 hierarchy document is a graph too (`Playground.isGraphDocument`):
  // the hierarchy elaborator flattens it, so it compiles and renders exactly as
  // the flat graph it expands to.
  test('a v3 hierarchy document: compile-wasm ≡ render-graph, and ≡ its flat expansion', async () => {
    const v3 = {
      version: 3,
      definitions: [{
        id: 'test.lowpass', version: 1, input: 'i', output: 'o',
        parameters: [{ name: 'cutoff', default: 900 }],
        nodes: [
          { id: 'i', kind: 'module_input' },
          { id: 'f', kind: 'filter', params: { resonance: 0.6 }, bindings: { cutoff: 'cutoff' }, in: { in: ['i'] } },
          { id: 'o', kind: 'module_output', in: { in: ['f'] } },
        ],
      }],
      scene: {
        nodes: [
          { id: 'res', kind: 'resonator', params: { freq: 220, decay: 4 } },
          { id: 'lp', kind: 'module', definition: 'test.lowpass', definition_version: 1,
            params: { cutoff: 1200 }, in: { in: ['res'] } },
        ],
        out: 'lp',
      },
      taps: ['res'],
    }
    const flat = {
      nodes: [
        { id: 'res', kind: 'resonator', params: { freq: 220, decay: 4 } },
        { id: 'f', kind: 'filter', params: { cutoff: 1200, resonance: 0.6 }, in: { in: ['res'] } },
      ],
      out: 'f',
      taps: ['res'],
    }
    const v3Path = join(dir, 'v3.json'), flatPath = join(dir, 'v3-flat.json')
    writeFileSync(v3Path, JSON.stringify(v3))
    writeFileSync(flatPath, JSON.stringify(flat))
    const { out, channels } = await renderWasm(v3Path, N)
    expect(channels).toBe(2)
    const jit = renderGraph(v3Path, N, channels)
    const expanded = renderGraph(flatPath, N, channels)
    let maxDiff = 0, energy = 0
    for (let i = 0; i < jit.length; i++) {
      maxDiff = Math.max(maxDiff, Math.abs(out[i]! - jit[i]!))
      energy += jit[i]! * jit[i]!
      expect(Object.is(jit[i], expanded[i])).toBe(true)
    }
    expect(energy).toBeGreaterThan(1e-3)
    expect(maxDiff).toBeLessThan(1e-12)
  })

  test('a τ-invariant cone into a freq inlet costs what a knob does (hoisted to stage 0) and renders the same samples', () => {
    const census = (graph: unknown) => {
      const path = join(dir, `census-${Math.random().toString(36).slice(2)}.json`)
      writeFileSync(path, JSON.stringify(graph))
      const r = spawnSync([diffcli, 'render-graph', path, '--stage-census', '--frames', '1', '--buffer', '8'], { cwd: repoRoot })
      const err = r.stderr.toString()
      const audio = /audio instrs=(\d+)/.exec(err), s1 = /s1=(\d+)/.exec(err)
      if (!audio || !s1) throw new Error(`no census: ${err}`)
      const samples = run(['render-graph', path, '--frames', '1', '--buffer', '2048'])
      return { audio: Number(audio[1]), s1: Number(s1[1]), samples }
    }
    const knob = census({ nodes: [
      { id: 'a', kind: 'knob', params: { value: 220 } },
      { id: 's', kind: 'saw', in: { freq: ['a'] } }], out: 's' })
    const ring = census({ nodes: [
      { id: 'a', kind: 'knob', params: { value: 110 } },
      { id: 'b', kind: 'knob', params: { value: 2 } },
      { id: 'f', kind: 'ring', in: { in: ['a', 'b'] } },
      { id: 's', kind: 'saw', in: { freq: ['f'] } }], out: 's' })
    const mix = census({ nodes: [
      { id: 'a', kind: 'knob', params: { value: 110 } },
      { id: 'b', kind: 'knob', params: { value: 110 } },
      { id: 'f', kind: 'mix', in: { in: ['a', 'b'] } },
      { id: 's', kind: 'saw', in: { freq: ['f'] } }], out: 's' })
    for (const cone of [ring, mix]) {
      expect(cone.s1).toBe(knob.s1)
      expect(cone.audio).toBe(knob.audio)
      expect(cone.samples.equals(knob.samples)).toBe(true)
    }
  })
})

/** Power spectrum of a Hann-windowed block (radix-2, n a power of two). */
function powerSpectrum(x: Float64Array): Float64Array {
  const n = x.length
  const re = new Float64Array(n), im = new Float64Array(n)
  for (let i = 0; i < n; i++) re[i] = x[i]! * (0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n))
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1
    for (; j & bit; bit >>= 1) j ^= bit
    j ^= bit
    if (i < j) { [re[i], re[j]] = [re[j]!, re[i]!]; [im[i], im[j]] = [im[j]!, im[i]!] }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len
    for (let i = 0; i < n; i += len)
      for (let k = 0; k < len / 2; k++) {
        const wr = Math.cos(ang * k), wi = Math.sin(ang * k)
        const ar = re[i + k + len / 2]!, ai = im[i + k + len / 2]!
        const tr = ar * wr - ai * wi, ti = ar * wi + ai * wr
        re[i + k + len / 2] = re[i + k]! - tr; im[i + k + len / 2] = im[i + k]! - ti
        re[i + k] = re[i + k]! + tr; im[i + k] = im[i + k]! + ti
      }
  }
  const p = new Float64Array(n / 2)
  for (let i = 0; i < n / 2; i++) p[i] = re[i]! * re[i]! + im[i]! * im[i]!
  return p
}

describe('band-limited sources', () => {
  test('polyBLEP saw and pulse fold far less energy below Nyquist than naive ones, at +f and −f', async () => {
    // A 2.5 kHz saw at 44.1 kHz: harmonics above 22.05 kHz alias back between
    // the true harmonics. Count the power more than 3 bins from any true
    // harmonic n·f (n·f < Nyquist), relative to the total. The naive saw is
    // `source` at morph 0; the naive pulse is computed here from the exact
    // phase. At −f the phasor runs backward and the BLEP must still apply.
    const f = 2500, n = 1 << 15, width = 0.3
    const aliasRatio = (x: Float64Array, rate: number) => {
      // remove the mean first: a pulse's DC would leak into the low bins
      const mean = x.reduce((a, b) => a + b, 0) / x.length
      const p = powerSpectrum(x.map((v) => v - mean))
      const binHz = rate / n
      let total = 0, alias = 0
      for (let b = 1; b < p.length; b++) {
        total += p[b]!
        const hz = b * binHz
        const k = Math.round(hz / f)
        if (k < 1 || Math.abs(hz - k * f) > 3 * binHz) alias += p[b]!
      }
      return alias / total
    }
    const dB = (r: number) => 10 * Math.log10(r)
    const measure = async (freq: number) => {
      const path = join(dir, `alias${freq}.json`)
      writeFileSync(path, JSON.stringify({ nodes: [
        { id: 'saw', kind: 'saw', params: { freq } },
        { id: 'sq', kind: 'pulse', params: { freq, width } },
        { id: 'naive', kind: 'source', params: { freq, morph: 0 } }],
        out: 'saw', taps: ['saw', 'sq', 'naive'] }))
      const { out, channels, manifest } = await renderWasm(path, n)
      const rate = manifest.rate
      const ch = (c: number) => Float64Array.from({ length: n }, (_, i) => out[i * channels + c]!)
      const inc = BigInt(Math.trunc((freq * 2 ** 32) / rate))
      const naivePulse = Float64Array.from({ length: n }, (_, i) => {
        const p = Number(BigInt.asUintN(32, inc * BigInt(i))) / 2 ** 32
        return p < width ? 1 : -1
      })
      // the ideal band-limited waveforms, by additive synthesis at the exact
      // phase: the saw 2p − 1 truncated below Nyquist, and the pulse as the
      // difference of two such saws a width apart
      const harmonics = Math.floor(rate / 2 / Math.abs(freq))
      const idealSaw = (p: number) => {
        let v = 0
        for (let k = 1; k <= harmonics; k++) v -= Math.sin(2 * Math.PI * k * p) / k
        return (2 / Math.PI) * v
      }
      const ideal = (c: number) => Float64Array.from({ length: n }, (_, i) => {
        const p = Number(BigInt.asUintN(32, inc * BigInt(i))) / 2 ** 32
        return c === 1 ? idealSaw(p) : idealSaw((p - width + 1) % 1) - idealSaw(p) + 2 * width - 1
      })
      // the error energy against the ideal, relative to the ideal's: aliasing
      // and a too-wide (dulled) correction both raise it
      const fidelity = (x: Float64Array, y: Float64Array) => {
        let e = 0, t = 0
        for (let i = 0; i < n; i++) { e += (x[i]! - y[i]!) ** 2; t += y[i]! ** 2 }
        return dB(e / t)
      }
      return {
        saw: dB(aliasRatio(ch(1), rate)), pulse: dB(aliasRatio(ch(2), rate)),
        naiveSaw: dB(aliasRatio(ch(3), rate)), naivePulse: dB(aliasRatio(naivePulse, rate)),
        sawErr: fidelity(ch(1), ideal(1)), pulseErr: fidelity(ch(2), ideal(2)),
        naiveSawErr: fidelity(ch(3), ideal(1)), naivePulseErr: fidelity(naivePulse, ideal(2)),
      }
    }
    const pos = await measure(f), neg = await measure(-f)
    const show = (m: typeof pos) =>
      `saw ${m.saw.toFixed(1)} (naive ${m.naiveSaw.toFixed(1)}) · pulse ${m.pulse.toFixed(1)} (naive ${m.naivePulse.toFixed(1)}) dB; ` +
      `error vs ideal: saw ${m.sawErr.toFixed(1)} (naive ${m.naiveSawErr.toFixed(1)}) · pulse ${m.pulseErr.toFixed(1)} (naive ${m.naivePulseErr.toFixed(1)}) dB`
    console.log(`  alias power / total at +f: ${show(pos)}\n  at −f: ${show(neg)}`)
    // Measured at the BLEP's width |inc|: alias 14.4 dB (saw) and 14.5 dB
    // (pulse) under naive, error vs ideal 7.8 dB and 7.5 dB under naive. A
    // narrower correction (0.8·|inc|) aliases too much for the first bound; a
    // wider one (1.25·|inc|) aliases less but dulls the harmonics, which the
    // second bound catches.
    for (const m of [pos, neg]) {
      expect(m.saw).toBeLessThan(m.naiveSaw - 13)
      expect(m.pulse).toBeLessThan(m.naivePulse - 13)
      expect(m.sawErr).toBeLessThan(m.naiveSawErr - 7)
      expect(m.pulseErr).toBeLessThan(m.naivePulseErr - 7)
    }
    // −f is the +f waveform reversed: the same spectrum, so the same alias power
    expect(Math.abs(neg.saw - pos.saw)).toBeLessThan(0.01)
    expect(Math.abs(neg.pulse - pos.pulse)).toBeLessThan(0.01)
  })

  test('saw, pulse, sine and a swept delay are pure functions of τ', async () => {
    const path = join(dir, 'blep.json')
    writeFileSync(path, JSON.stringify(cases.blep))
    const wasmPath = join(dir, 'pure.wasm')
    run(['compile-wasm', path, `--out=${wasmPath}`])
    const manifest = JSON.parse(run(['compile-kernel', path]).toString()) as KernelManifest
    const k = await WasmKernel.instantiate(new Uint8Array(readFileSync(wasmPath)), manifest)
    const F = 2 ** 33 + 12_345, N = 777
    k.seek(F)
    const a = k.render(N)
    k.seek(5) // render something else in between
    k.render(1000)
    k.seek(F - 100) // and arrive at F from a different block alignment
    const b = k.render(N + 100).subarray(100 * k.outputChannelCount)
    expect(b.length).toBe(a.length)
    for (let i = 0; i < a.length; i++) expect(Object.is(a[i], b[i])).toBe(true)
  })
})
