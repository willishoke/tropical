/**
 * runtime_params.test.ts — the player's live knob writes (`ParamWriter`,
 * web/runtime/params.ts) against design/host-param-dispatch.md.
 *
 *  1. An anchored frequency (a knob wired straight into a `harmonic`'s freq)
 *     re-pitched ten hours in stays continuous: the `#phase` companion takes
 *     the jump `Δinc·now` exactly, and the new pitch is the knob's. The same
 *     write done raw jumps, so the check discriminates.
 *  2. A raw write is the value.
 *  3. Companions and disciplines the player does not implement are refused by
 *     name, never written raw.
 */
import { describe, test, expect, setDefaultTimeout } from 'bun:test'
import { spawnSync } from 'bun'
import { writeFileSync, readFileSync, mkdtempSync } from 'fs'
import { tmpdir } from 'os'
import { join, resolve, dirname } from 'path'
import { fileURLToPath } from 'url'
import { WasmKernel, ParamWriter, kernelParams, type KernelManifest } from '../../web/runtime/index'

setDefaultTimeout(120_000)

const __dirname = dirname(fileURLToPath(import.meta.url))
const repoRoot = resolve(__dirname, '../..')
const diffcli = process.env.TROPICAL_DIFFCLI
  ? resolve(process.env.TROPICAL_DIFFCLI)
  : resolve(repoRoot, 'lean/.lake/build/bin/diffcli')
const dir = mkdtempSync(join(tmpdir(), 'runtime-params-'))

function run(args: string[]): Buffer {
  const r = spawnSync([diffcli, ...args], { cwd: repoRoot })
  if (r.exitCode !== 0) throw new Error(`diffcli ${args[0]} failed: ${r.stderr.toString()}`)
  return r.stdout as Buffer
}

async function load(graph: unknown): Promise<{ k: WasmKernel; manifest: KernelManifest }> {
  const path = join(dir, `${Math.random().toString(36).slice(2)}.json`)
  writeFileSync(path, JSON.stringify(graph))
  const wasmPath = path.replace(/\.json$/, '.wasm')
  run(['compile-wasm', path, `--out=${wasmPath}`])
  const manifest = JSON.parse(run(['compile-kernel', path]).toString()) as KernelManifest
  return { k: await WasmKernel.instantiate(new Uint8Array(readFileSync(wasmPath)), manifest), manifest }
}

function zeroCrossingHz(x: Float64Array, rate: number): number {
  const ups: number[] = []
  for (let i = 1; i < x.length; i++) {
    const a = x[i - 1]!, b = x[i]!
    if (a < 0 && b >= 0) ups.push(i - 1 + a / (a - b))
  }
  return ((ups.length - 1) * rate) / (ups[ups.length - 1]! - ups[0]!)
}

// one partial: a pure sine at f0, so continuity is a bound on the step
const sineFamily = {
  nodes: [
    { id: 'f0', kind: 'knob', params: { value: 220 } },
    { id: 'h', kind: 'harmonic', params: { partials_max: 1, partials: 1 }, in: { freq: ['f0'] } },
  ],
  out: 'h',
}

describe('ParamWriter', () => {
  test('anchor: a re-pitch ten hours in is continuous and plays the new pitch; raw jumps', async () => {
    const far = 10 * 3600 * 44100 + 12345
    const n = 4096
    const run1 = async (anchored: boolean) => {
      const { k, manifest } = await load(sineFamily)
      const writer = new ParamWriter(manifest)
      expect(writer.check('param:f0.value')).toBeNull()
      k.seek(far)
      const before = k.render(n).slice()
      if (anchored) writer.write(k, 'param:f0.value', 330, k.frame)
      else k.setParam(kernelParams(manifest).findIndex((p) => p.name === 'param:f0.value'), 330)
      const after = k.render(n).slice()
      return { before, after, rate: manifest.rate }
    }
    const a = await run1(true)
    const peak = a.before.reduce((m, v) => Math.max(m, Math.abs(v)), 0)
    // the largest step a sine of this peak takes at 330 Hz, with margin
    const bound = 1.05 * peak * 2 * Math.PI * 330 / a.rate
    const step = Math.abs(a.after[0]! - a.before[n - 1]!)
    const hz = zeroCrossingHz(a.after, a.rate)
    const r = await run1(false)
    const rawStep = Math.abs(r.after[0]! - r.before[n - 1]!)
    console.log(`  anchored step ${step.toExponential(3)} (bound ${bound.toExponential(3)}), raw step ${rawStep.toExponential(3)}, pitch after ${hz.toFixed(4)} Hz`)
    expect(step).toBeLessThan(bound)
    expect(rawStep).toBeGreaterThan(bound)
    expect(Math.abs(hz - 330)).toBeLessThan(0.01)
  })

  test('raw: the write is the value; companions and unimplemented disciplines are refused', async () => {
    const { k, manifest } = await load({ ...sineFamily, nodes: [...sineFamily.nodes,
      { id: 'g', kind: 'knob', params: { value: 0.5 } },
      { id: 'v', kind: 'ring', in: { in: ['h', 'g'] } }], out: 'v' })
    const writer = new ParamWriter(manifest)
    const names = kernelParams(manifest).map((p) => p.name)
    writer.write(k, 'param:g.value', 0.25, k.frame)
    expect(k.getParam(names.indexOf('param:g.value'))).toBe(0.25)
    expect(writer.check('param:f0.value#phase')).toContain('discipline companion')
    expect(writer.check('param:master.velocity')).toContain("discipline 'velocity'")
    expect(writer.check('param:nope.value')).toContain('no param input')
    expect(() => writer.write(k, 'param:master.velocity', 2, 0)).toThrow(/velocity/)
  })
})
