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
})
