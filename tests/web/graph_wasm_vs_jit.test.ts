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
  // a pitch driven by a τ-invariant cone (knob × knob), and vibrato done the
  // exact way: the LFO into an fm node's clock warp, its carrier a knob (a
  // time-varying wire into a freq inlet is refused; graph_freq_inlets)
  wiredFreq: {
    nodes: [
      { id: 'a', kind: 'knob', params: { value: 110 } },
      { id: 'b', kind: 'knob', params: { value: 2 } },
      { id: 'f', kind: 'ring', in: { in: ['a', 'b'] } },
      { id: 'saw', kind: 'saw', in: { freq: ['f'] } },
      { id: 'vib', kind: 'sine', params: { freq: 5 } },
      { id: 'sw', kind: 'fm', params: { depth: 3 }, in: { in: ['vib'], carrier: ['a'] } },
      { id: 'out', kind: 'mix', in: { in: ['saw', 'sw'] } },
    ],
    out: 'out',
    taps: ['saw', 'sw'],
  },
  // every knob that became an inlet, each driven by a signal (an LFO around
  // its knob value), plus pm on a wired freq. The frequency and rate inlets
  // take only wires constant within a call, so theirs are knob arithmetic.
  signalControls: {
    nodes: [
      { id: 'l', kind: 'sine', params: { freq: 2 } },
      { id: 'osc', kind: 'saw', params: { freq: 220 } },
      { id: 'res', kind: 'resonator', params: { freq: 220, decay: 4 } },
      ...lfoAround('morph', 0.3, 0.2), ...knobCone('rate', 20), ...lfoAround('dly', 0.003, 0.001),
      ...lfoAround('dec', 0.6, 0.2), ...lfoAround('dep', 0.002, 0.001), ...knobCone('car', 440),
      ...lfoAround('idx', 20, 10), ...lfoAround('cut', 900, 300), ...lfoAround('q', 0.5, 0.3),
      ...lfoAround('ctr', 900, 300), ...lfoAround('swp', 1.2, 0.4), ...knobCone('phr', 0.5),
      ...lfoAround('pmix', 0.5, 0.3), ...knobCone('pf', 330),
      { id: 'src', kind: 'source', params: { freq: 220 }, in: { morph: ['morph'] } },
      { id: 'pl', kind: 'pluck', params: { freq: 110 }, in: { event_rate: ['rate'], morph: ['morph'] } },
      { id: 'cb', kind: 'comb', in: { in: ['osc'], delay: ['dly'], decay: ['dec'] } },
      { id: 'fl', kind: 'flange', in: { in: ['osc'], depth: ['dep'] } },
      { id: 'fm', kind: 'fm', in: { in: ['l'], carrier: ['car'], depth: ['idx'] } },
      { id: 'flt', kind: 'filter', in: { in: ['res'], cutoff: ['cut'], resonance: ['q'] } },
      { id: 'ph', kind: 'phaser', in: { in: ['res'], center: ['ctr'], sweep: ['swp'], rate: ['phr'], mix: ['pmix'] } },
      { id: 'pm', kind: 'source', params: { morph: 1 }, in: { freq: ['pf'], pm: ['l'] } },
      { id: 'out', kind: 'mix', in: { in: ['src', 'pl', 'cb', 'fl', 'fm', 'flt', 'ph', 'pm'] } },
    ],
    out: 'out',
    taps: ['src', 'pl', 'cb', 'fl', 'fm', 'flt', 'ph', 'pm'],
  },
}

/** `id` = knob `value` × knob 1: knob arithmetic, constant within a call. */
function knobCone(id: string, value: number): object[] {
  return [
    { id: `${id}_v`, kind: 'knob', params: { value } },
    { id: `${id}_1`, kind: 'knob', params: { value: 1 } },
    { id, kind: 'ring', in: { in: [`${id}_v`, `${id}_1`] } },
  ]
}

/** `id` = knob `center` + `span`·(the LFO `l`): a signal that sits on a knob value. */
function lfoAround(id: string, center: number, span: number): object[] {
  return [
    { id: `${id}_c`, kind: 'knob', params: { value: center } },
    { id: `${id}_s`, kind: 'knob', params: { value: span } },
    { id: `${id}_m`, kind: 'ring', in: { in: ['l', `${id}_s`] } },
    { id, kind: 'mix', in: { in: [`${id}_c`, `${id}_m`] } },
  ]
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
    // A knob wired straight in is anchored, by its own `#phase` companion,
    // exactly as the node's own freq knob is, and costs what that knob does.
    // A cone is not anchored, so it is compared with an unanchored knob: one
    // passed through a one-input mix, which lowers to the knob's slot read.
    const own = census({ nodes: [
      { id: 's', kind: 'saw', params: { freq: 220 } }], out: 's' })
    const straight = census({ nodes: [
      { id: 'a', kind: 'knob', params: { value: 220 } },
      { id: 's', kind: 'saw', in: { freq: ['a'] } }], out: 's' })
    expect(straight.s1).toBe(own.s1)
    expect(straight.audio).toBe(own.audio)
    expect(straight.samples.equals(own.samples)).toBe(true)
    const knob = census({ nodes: [
      { id: 'a', kind: 'knob', params: { value: 220 } },
      { id: 'm', kind: 'mix', in: { in: ['a'] } },
      { id: 's', kind: 'saw', in: { freq: ['m'] } }], out: 's' })
    expect(knob.samples.equals(straight.samples)).toBe(true)
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

  // Every knob that became an inlet: a knob wired in (the knob form, reading
  // the knob's slot) against a τ-invariant cone (knob × knob, the signal
  // form). The cone hoists, so the per-sample (s1) stage count is equal, the
  // audio kernel's instruction count is equal (except as noted), and the two
  // render the same samples bit for bit.
  const ports: [kind: string, port: string, value: number, input: 'signal' | 'modal' | 'none', params: Record<string, number>][] = [
    ['source', 'morph', 0.3, 'none', { freq: 220, morph: 0.3 }],
    ['pluck', 'freq', 110, 'none', { freq: 110, morph: 0.4, event_rate: 20 }],
    ['pluck', 'morph', 0.4, 'none', { freq: 110, morph: 0.4, event_rate: 20 }],
    ['pluck', 'event_rate', 20, 'none', { freq: 110, morph: 0.4, event_rate: 20 }],
    ['comb', 'delay', 0.003, 'signal', { delay: 0.003, decay: 0.6 }],
    ['comb', 'decay', 0.6, 'signal', { delay: 0.003, decay: 0.6 }],
    ['flange', 'depth', 0.002, 'signal', { depth: 0.002 }],
    ['fm', 'carrier', 440, 'signal', { carrier: 440, depth: 20 }],
    ['fm', 'depth', 20, 'signal', { carrier: 440, depth: 20 }],
    ['filter', 'cutoff', 900, 'modal', { cutoff: 900, resonance: 0.6 }],
    ['filter', 'resonance', 0.6, 'modal', { cutoff: 900, resonance: 0.6 }],
    // out of the knob range: both forms clamp to it (a cutoff of 0 or a
    // resonance below ≈ −0.02 has no pole pair)
    ['filter', 'cutoff', 0, 'modal', { cutoff: 900, resonance: 0.6 }],
    ['filter', 'cutoff', 30000, 'modal', { cutoff: 900, resonance: 0.6 }],
    ['filter', 'resonance', -0.5, 'modal', { cutoff: 900, resonance: 0.6 }],
    ['filter', 'resonance', 1.5, 'modal', { cutoff: 900, resonance: 0.6 }],
    ['phaser', 'center', 900, 'modal', { center: 900, sweep: 1.2, rate: 0.5, mix: 0.4 }],
    ['phaser', 'sweep', 1.2, 'modal', { center: 900, sweep: 1.2, rate: 0.5, mix: 0.4 }],
    ['phaser', 'rate', 0.5, 'modal', { center: 900, sweep: 1.2, rate: 0.5, mix: 0.4 }],
    ['phaser', 'mix', 0.4, 'modal', { center: 900, sweep: 1.2, rate: 0.5, mix: 0.4 }],
  ]
  const portGraph = (kind: string, port: string, input: string, params: Record<string, number>,
      drivers: object[], driver: string) => {
    const nodes: object[] = []
    const inObj: Record<string, string[]> = { [port]: [driver] }
    if (input === 'signal') { nodes.push({ id: 'src', kind: 'source', params: { freq: 220, morph: 0.2 } }); inObj.in = ['src'] }
    if (input === 'modal') { nodes.push({ id: 'res', kind: 'resonator', params: { freq: 220, decay: 4 } }); inObj.in = ['res'] }
    nodes.push(...drivers, { id: 'x', kind, params, in: inObj })
    return { nodes, out: 'x' }
  }
  const stageCensus = (graph: unknown) => {
    const path = join(dir, `census-${Math.random().toString(36).slice(2)}.json`)
    writeFileSync(path, JSON.stringify(graph))
    const r = spawnSync([diffcli, 'render-graph', path, '--stage-census', '--frames', '1', '--buffer', '8'], { cwd: repoRoot })
    const err = r.stderr.toString()
    const audio = /audio instrs=(\d+)/.exec(err), s1 = /s1=(\d+)/.exec(err)
    if (!audio || !s1) throw new Error(`no census: ${err}`)
    const samples = run(['render-graph', path, '--frames', '1', '--buffer', '2048'])
    return { audio: Number(audio[1]), s1: Number(s1[1]), samples }
  }
  // A knob wired straight into an anchored inlet (a freq) is anchored and a
  // cone is not, so there the cone is compared with an unanchored knob: one
  // passed through a one-input mix, which lowers to the knob's slot read.
  const anchoredPorts = new Set(['pluck.freq', 'fm.carrier'])
  for (const [kind, port, value, input, params] of ports) {
    test(`${kind}.${port} = ${value}: a τ-invariant cone costs what a knob does and renders the same samples`, () => {
      const knob = anchoredPorts.has(`${kind}.${port}`)
        ? stageCensus(portGraph(kind, port, input, params, [
          { id: 'a', kind: 'knob', params: { value } },
          { id: 'm', kind: 'mix', in: { in: ['a'] } }], 'm'))
        : stageCensus(portGraph(kind, port, input, params,
          [{ id: 'a', kind: 'knob', params: { value } }], 'a'))
      const cone = stageCensus(portGraph(kind, port, input, params, [
        { id: 'a', kind: 'knob', params: { value } },
        { id: 'b', kind: 'knob', params: { value: 1 } },
        { id: 'c', kind: 'ring', in: { in: ['a', 'b'] } }], 'c'))
      expect(cone.s1).toBe(knob.s1)
      // The phaser's center and sweep feed one routed coefficient region per
      // all-pass section, and a region re-evaluates a non-leaf τ-invariant
      // operand inside its body (a knob is a slot operand there, the cone one
      // Mul). That is s0 work left in the audio plan — one Mul per section.
      const regionOperands = kind === 'phaser' && (port === 'center' || port === 'sweep') ? 6 : 0
      expect(cone.audio).toBe(knob.audio + regionOperands)
      expect(cone.samples.equals(knob.samples)).toBe(true)
    })
  }

  test('a knob with param_aliases is still a knob: the knob form, at the same cost and samples', () => {
    // A knob inside a module carries its param under an aliased name; wired
    // into an inlet it must still read its slot (the knob form), not lower as
    // a signal. 9000 Hz is past the cutoff range, so the forms would differ.
    const graph = (aliases: Record<string, string> | undefined) => ({ nodes: [
      { id: 'res', kind: 'resonator', params: { freq: 220, decay: 4 } },
      { id: 'k', kind: 'knob', params: { value: 9000 }, ...(aliases ? { param_aliases: aliases } : {}) },
      { id: 'f', kind: 'filter', params: { resonance: 0.5 }, in: { in: ['res'], cutoff: ['k'] } }], out: 'f' })
    const plain = stageCensus(graph(undefined))
    const aliased = stageCensus(graph({ value: 'probe.cut' }))
    expect(aliased.s1).toBe(plain.s1)
    expect(aliased.audio).toBe(plain.audio)
    expect(aliased.samples.equals(plain.samples)).toBe(true)
  })

  test('sflange is its delay/ring/mix expansion: bit for bit, and cheaper than the hand patch', () => {
    // The served-node patch sflange expands to: d = depth·lfo, the input
    // delayed by d and by −d, mixed with the dry/wet knobs (½, ¼ — the
    // former fixed weights, now knobs).
    const common = [
      { id: 'osc', kind: 'saw', params: { freq: 220 } },
      { id: 'lfo', kind: 'sine', params: { freq: 3 } },
      { id: 'depth', kind: 'knob', params: { value: 0.002 } },
      { id: 'dry', kind: 'knob', params: { value: 0.5 } },
      { id: 'wet', kind: 'knob', params: { value: 0.25 } },
    ]
    const served = stageCensus({ nodes: [...common,
      { id: 'fl', kind: 'sflange', in: { in: ['osc'], mod: ['lfo'], depth: ['depth'], dry: ['dry'], wet: ['wet'] } }],
      out: 'fl' })
    const hand = stageCensus({ nodes: [...common,
      { id: 'd', kind: 'ring', in: { in: ['depth', 'lfo'] } },
      { id: 'm1', kind: 'knob', params: { value: -1 } },
      { id: 'nd', kind: 'ring', in: { in: ['d', 'm1'] } },
      { id: 'back', kind: 'delay', in: { in: ['osc'], amount: ['d'] } },
      { id: 'fwd', kind: 'delay', in: { in: ['osc'], amount: ['nd'] } },
      { id: 'wd', kind: 'ring', in: { in: ['dry', 'osc'] } },
      { id: 'wb', kind: 'ring', in: { in: ['wet', 'back'] } },
      { id: 'wf', kind: 'ring', in: { in: ['wet', 'fwd'] } },
      { id: 'fl', kind: 'mix', in: { in: ['wd', 'wb', 'wf'] } }],
      out: 'fl' })
    expect(served.samples.equals(hand.samples)).toBe(true)
    // The hand patch's ring by −1 is a live slot product, so its ahead tap
    // converts its own time (mul, ·SR, ·2¹⁶, toInt); sflange's −d is a
    // negation, which the delay warp folds into the shared conversion.
    console.log(`  sflange audio instrs ${served.audio} (s1 ${served.s1}) · hand patch ${hand.audio} (s1 ${hand.s1})`)
    expect(served.s1).toBeLessThan(hand.s1)
  })

  test('sflange with dry and wet omitted is the former fixed-weight flanger, bit for bit', () => {
    // dry and wet default to the weights sflange baked before they were knobs
    // (½ dry, ¼ each tap): an existing document that names neither sounds as
    // it did, as do the same weights spelled as params or as knob wires.
    const nodes = (fl: object) => ({ nodes: [
      { id: 'osc', kind: 'saw', params: { freq: 220 } },
      { id: 'lfo', kind: 'sine', params: { freq: 3 } },
      { id: 'half', kind: 'knob', params: { value: 0.5 } },
      { id: 'quarter', kind: 'knob', params: { value: 0.25 } },
      { id: 'fl', kind: 'sflange', ...fl }], out: 'fl' })
    const omitted = stageCensus(nodes({ params: { depth: 0.002 }, in: { in: ['osc'], mod: ['lfo'] } }))
    const params = stageCensus(nodes({ params: { depth: 0.002, dry: 0.5, wet: 0.25 }, in: { in: ['osc'], mod: ['lfo'] } }))
    const wired = stageCensus(nodes({ params: { depth: 0.002 }, in: { in: ['osc'], mod: ['lfo'], dry: ['half'], wet: ['quarter'] } }))
    expect(omitted.samples.equals(params.samples)).toBe(true)
    expect(omitted.samples.equals(wired.samples)).toBe(true)
    const x = new Float64Array(omitted.samples.buffer, omitted.samples.byteOffset, omitted.samples.byteLength / 8)
    expect(x.some((v) => Math.abs(v) > 0.1)).toBe(true)
  })

  test('pm with a signal freq: a constant-cone freq ≡ the knob form, bit for bit', () => {
    const graph = (drivers: object[], driver: string) => ({ nodes: [
      { id: 'lfo', kind: 'sine', params: { freq: 3 } },
      ...drivers,
      { id: 'x', kind: 'source', params: { morph: 0.5 }, in: { freq: [driver], pm: ['lfo'] } }], out: 'x' })
    const knob = stageCensus(graph([{ id: 'a', kind: 'knob', params: { value: 330 } }], 'a'))
    const cone = stageCensus(graph([
      { id: 'a', kind: 'knob', params: { value: 330 } },
      { id: 'b', kind: 'knob', params: { value: 1 } },
      { id: 'c', kind: 'ring', in: { in: ['a', 'b'] } }], 'c'))
    expect(cone.s1).toBe(knob.s1)
    expect(cone.audio).toBe(knob.audio)
    expect(cone.samples.equals(knob.samples)).toBe(true)
  })
})

describe('compile-kernel --rate', () => {
  test('a graph compiled at 48 kHz: manifest rate, the kernel runs at it, wasm ≡ JIT', async () => {
    const path = join(dir, 'rate.json')
    // A 480 Hz sine has a 100-sample period at 48 kHz (91.875 at 44.1 kHz);
    // the default string's modes are built for the compile rate.
    writeFileSync(path, JSON.stringify({ nodes: [
      { id: 's', kind: 'sine', params: { freq: 480 } },
      { id: 'str', kind: 'string' }], out: 's', taps: ['s', 'str'] }))
    const wasmPath = join(dir, 'rate.wasm')
    run(['compile-wasm', path, `--out=${wasmPath}`, '--rate', '48000'])
    const manifest = JSON.parse(run(['compile-kernel', path, '--rate', '48000']).toString()) as KernelManifest
    expect(manifest.rate).toBe(48000)
    const k = await WasmKernel.instantiate(new Uint8Array(readFileSync(wasmPath)), manifest)
    const N = 1024, C = k.outputChannelCount
    const out = k.render(N)
    let periodErr = 0
    for (let i = 0; i + 100 < N; i++) periodErr = Math.max(periodErr, Math.abs(out[(i + 100) * C + 1]! - out[i * C + 1]!))
    expect(periodErr).toBeLessThan(1e-6)
    const b = run(['render-graph', path, '--all-channels', '--rate', '48000', '--frames', '1', '--buffer', String(N)])
    const jit = new Float64Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength))
    expect(jit.length).toBe(out.length)
    let maxDiff = 0
    for (let i = 0; i < jit.length; i++) maxDiff = Math.max(maxDiff, Math.abs(out[i]! - jit[i]!))
    expect(maxDiff).toBeLessThan(1e-12)
    // and the default rate is not 48 kHz: the same graph's sine is not 100-periodic
    const at44 = renderGraph(path, N, C)
    let err44 = 0
    for (let i = 0; i + 100 < N; i++) err44 = Math.max(err44, Math.abs(at44[(i + 100) * C + 1]! - at44[i * C + 1]!))
    expect(err44).toBeGreaterThan(1e-2)
  })

  test('a tropical_program_2 patch compiled at 48 kHz plays its pitch at 48 kHz', async () => {
    // pure-sine-440 at 48 kHz and at the default 44.1 kHz: each kernel, run at
    // its own rate, plays 440 Hz. (A kernel that ignored --rate would play
    // 440·48000/44100 ≈ 479 Hz when run at 48 kHz.)
    const path = resolve(repoRoot, 'web/patches/pure-sine-440.json')
    const m48 = JSON.parse(run(['compile-kernel', path, '--rate', '48000']).toString()) as KernelManifest
    const m44 = JSON.parse(run(['compile-kernel', path]).toString()) as KernelManifest
    expect(m48.rate).toBe(48000)
    expect(m44.rate).toBe(44100)
    const w48 = join(dir, 'p48.wasm'), w44 = join(dir, 'p44.wasm')
    run(['compile-wasm', path, `--out=${w48}`, '--rate=48000'])
    run(['compile-wasm', path, `--out=${w44}`])
    const n = 9600
    const k48 = await WasmKernel.instantiate(new Uint8Array(readFileSync(w48)), m48)
    const k44 = await WasmKernel.instantiate(new Uint8Array(readFileSync(w44)), m44)
    const hz48 = zeroCrossingHz(k48.render(n), k48.outputChannelCount, 0, 48000)
    const hz44 = zeroCrossingHz(k44.render(n), k44.outputChannelCount, 0, 44100)
    console.log(`  pure-sine-440 measured ${hz48.toFixed(4)} Hz at 48 kHz, ${hz44.toFixed(4)} Hz at 44.1 kHz`)
    expect(Math.abs(hz48 - 440)).toBeLessThan(0.01)
    expect(Math.abs(hz44 - 440)).toBeLessThan(0.01)
  })

  test('--rate N and --rate=N are the same flag; a bad rate or an unknown flag is an error', () => {
    const path = resolve(repoRoot, 'web/patches/pure-sine-440.json')
    const spaced = run(['compile-kernel', path, '--rate', '48000']).toString()
    expect(run(['compile-kernel', path, '--rate=48000']).toString()).toBe(spaced)
    expect(JSON.parse(spaced).rate).toBe(48000)
    for (const bad of [['--rate=0'], ['--rate', '-1'], ['--rate=44.1k'], ['--rate'], ['--rates=48000'], ['--frobnicate']]) {
      const r = spawnSync([diffcli, 'compile-kernel', path, ...bad], { cwd: repoRoot })
      expect(r.exitCode).not.toBe(0)
      expect(r.stdout.length).toBe(0)
    }
  })
})

/** The pitch of channel `ch` from its upward zero crossings, each placed by
 *  linear interpolation: (crossings − 1) periods over the span between the
 *  first and the last. */
function zeroCrossingHz(x: Float64Array, channels: number, ch: number, rate: number): number {
  const ups: number[] = []
  for (let i = 1; i < x.length / channels; i++) {
    const a = x[(i - 1) * channels + ch]!, b = x[i * channels + ch]!
    if (a < 0 && b >= 0) ups.push(i - 1 + a / (a - b))
  }
  return ((ups.length - 1) * rate) / (ups[ups.length - 1]! - ups[0]!)
}

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
