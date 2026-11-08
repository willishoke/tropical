/**
 * bench.ts — wasm size and real-time cost of the spectral family
 * (design/spectral-family.md, "Measured cost").
 *
 *   bun benchmarks/spectral/bench.ts            # from the repo root, after `make lean`
 *
 * Each case is compiled by `diffcli compile-wasm` / `compile-kernel`
 * (`--fixtures` registers the spectral fixture programs) and driven through
 * the shipped runtime (`web/runtime`, WasmKernel, 128-frame calls) on one
 * thread: one second of warm-up, then the best of three 10-second renders.
 * The fraction printed is wall time over audio time.
 *
 *   kill_4x200_lowpass  the kill criterion: 4 voices (55, 68.75, 82.5, 96.25 Hz),
 *                       each its own 200-partial saw family under a complex
 *                       SVF lowpass (Q 8) whose cutoff is 800·2^(2·sin(2π·220·τ)),
 *                       an audio-rate LFO; 48 kHz
 *   saw_4x200           the same four families with no envelope; 48 kHz
 *   old_saw_4x24        90242aa's four 24-partial saws as four family terms; 44.1 kHz
 *   old_saw_graph       90242aa's own graph for those saws (96 sine nodes,
 *                       `1/n` ring × knob, mix), cut out of
 *                       tests/fixtures/spectral/synced-scope-90242aa.json; 44.1 kHz
 *   old_scope_graph     that whole graph (saws + flangers + 14 taps); 44.1 kHz
 */
import { spawnSync } from 'bun'
import { readFileSync, writeFileSync, mkdtempSync } from 'fs'
import { resolve, dirname, join } from 'path'
import { tmpdir } from 'os'
import { fileURLToPath } from 'url'
import { WasmKernel, type KernelManifest } from '../../web/runtime/index'

const here = dirname(fileURLToPath(import.meta.url))
const repoRoot = resolve(here, '../..')
const diffcli = resolve(repoRoot, 'lean/.lake/build/bin/diffcli')
const tmp = mkdtempSync(join(tmpdir(), 'spectral-bench-'))

function run(args: string[]): string {
  const r = spawnSync([diffcli, ...args], { cwd: repoRoot })
  if (r.exitCode !== 0) throw new Error(`diffcli ${args[0]} failed: ${r.stderr?.toString()}`)
  return r.stdout.toString()
}

function oldGraph(sawsOnly: boolean): string {
  const g = JSON.parse(readFileSync(resolve(repoRoot, 'tests/fixtures/spectral/synced-scope-90242aa.json'), 'utf8'))
  const path = join(tmp, sawsOnly ? 'old_saw_graph.json' : 'old_scope_graph.json')
  if (sawsOnly) {
    const keep = g.nodes.filter((n: { id: string }) => /^(g\d+|v\d+(_\d+)?|w\d+_\d+|saw\d|saws)$/.test(n.id))
    writeFileSync(path, JSON.stringify({ nodes: keep, out: 'saws', taps: [] }))
  } else {
    writeFileSync(path, JSON.stringify(g))
  }
  return path
}

type Case = { name: string; path: string; rate: number; fixtures: boolean }
const cases: Case[] = [
  { name: 'kill_4x200_lowpass', path: resolve(here, 'kill_4x200_lowpass.json'), rate: 48000, fixtures: true },
  { name: 'saw_4x200', path: resolve(here, 'saw_4x200.json'), rate: 48000, fixtures: true },
  { name: 'old_saw_4x24', path: resolve(here, 'old_saw_4x24.json'), rate: 44100, fixtures: true },
  { name: 'old_saw_graph', path: oldGraph(true), rate: 44100, fixtures: false },
  { name: 'old_scope_graph', path: oldGraph(false), rate: 44100, fixtures: false },
]

const only = process.argv.slice(2)
for (const c of cases) {
  if (only.length && !only.includes(c.name)) continue
  const flags = [`--rate=${c.rate}`, ...(c.fixtures ? ['--fixtures'] : [])]
  const wasmPath = join(tmp, `${c.name}.wasm`)
  run(['compile-wasm', c.path, `--out=${wasmPath}`, ...flags])
  const manifest = JSON.parse(run(['compile-kernel', c.path, ...flags])) as KernelManifest
  const bytes = new Uint8Array(readFileSync(wasmPath))
  const k = await WasmKernel.instantiate(bytes, manifest)
  const seconds = 10
  const n = c.rate * seconds
  const out = new Float64Array(n * k.outputChannelCount)
  k.renderInto(out, c.rate) // warm-up
  let best = Infinity
  for (let rep = 0; rep < 3; rep++) {
    k.seek(c.rate)
    const t0 = performance.now()
    k.renderInto(out, n)
    best = Math.min(best, (performance.now() - t0) / 1000)
  }
  console.log(JSON.stringify({
    case: c.name, wasmBytes: bytes.length, rate: c.rate, channels: k.outputChannelCount,
    seconds, wall: +best.toFixed(4), realTimeFraction: +(best / seconds).toFixed(4),
  }))
}
