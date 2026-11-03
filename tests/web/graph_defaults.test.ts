/**
 * graph_defaults.test.ts — a knob a PatchGraph node omits from `params`
 * seeds its slot with the vocabulary's declared default (the value
 * `get_vocabulary` advertises), not 0. A bare `comb` once had decay 0 (its
 * taps silent, its delay inert) and a bare `source` a freq of 0.
 */
import { describe, test, expect, setDefaultTimeout } from 'bun:test'
import { spawnSync } from 'bun'
import { writeFileSync, mkdtempSync } from 'fs'
import { tmpdir } from 'os'
import { join, resolve, dirname } from 'path'
import { fileURLToPath } from 'url'
import { kernelParams, type KernelManifest } from '../../web/runtime/index'

setDefaultTimeout(60_000)

const __dirname = dirname(fileURLToPath(import.meta.url))
const repoRoot = resolve(__dirname, '../..')
const diffcli = process.env.TROPICAL_DIFFCLI
  ? resolve(process.env.TROPICAL_DIFFCLI)
  : resolve(repoRoot, 'lean/.lake/build/bin/diffcli')
const dir = mkdtempSync(join(tmpdir(), 'graph-defaults-'))

function run(args: string[]): Buffer {
  const r = spawnSync([diffcli, ...args], { cwd: repoRoot })
  if (r.exitCode !== 0) throw new Error(`diffcli ${args[0]} failed: ${r.stderr.toString()}`)
  return r.stdout as Buffer
}

function compile(name: string, graph: unknown): { path: string; defaults: Map<string, number> } {
  const path = join(dir, `${name}.json`)
  writeFileSync(path, JSON.stringify(graph))
  const manifest = JSON.parse(run(['compile-kernel', path]).toString()) as KernelManifest
  return { path, defaults: new Map(kernelParams(manifest).map((p) => [p.name, p.default])) }
}

function render(path: string, n: number): Float64Array {
  const b = run(['render-graph', path, '--all-channels', '--frames', '1', '--buffer', String(n)])
  return new Float64Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength))
}

describe('omitted knobs seed their declared default', () => {
  test('bare source, saw, pluck and comb: every omitted knob is its vocabulary default', () => {
    const { defaults } = compile('bare', {
      nodes: [
        { id: 'src', kind: 'source' },
        { id: 'saw', kind: 'saw' },
        { id: 'pl', kind: 'pluck' },
        { id: 'cb', kind: 'comb', in: { in: ['saw'] } },
        { id: 'out', kind: 'mix', in: { in: ['src', 'pl', 'cb'] } },
      ],
      out: 'out',
    })
    // anchored knobs (freq) are one slot; glide knobs seed both glide
    // endpoints (`#v0`, `#v1`)
    const anchored: Record<string, number> = { 'src.freq': 220, 'saw.freq': 220, 'pl.freq': 110 }
    const glided: Record<string, number> = {
      'src.morph': 0, 'pl.morph': 0, 'pl.event_rate': 2, 'cb.delay': 0.012, 'cb.decay': 0.7,
    }
    for (const [name, value] of Object.entries(anchored)) expect(defaults.get(`param:${name}`)).toBe(value)
    for (const [name, value] of Object.entries(glided))
      for (const end of ['#v0', '#v1']) expect(defaults.get(`param:${name}${end}`)).toBe(value)
  })

  test('a bare comb sounds its taps: it is not the dry input', () => {
    const N = 2048
    const dry = compile('dry', { nodes: [{ id: 'saw', kind: 'saw' }], out: 'saw' })
    const comb = compile('comb', {
      nodes: [{ id: 'saw', kind: 'saw' }, { id: 'cb', kind: 'comb', in: { in: ['saw'] } }],
      out: 'cb',
    })
    const a = render(dry.path, N), b = render(comb.path, N)
    let diff = 0
    for (let i = 0; i < N; i++) diff = Math.max(diff, Math.abs(a[i]! - b[i]!))
    expect(diff).toBeGreaterThan(0.1)
  })
})
