/**
 * web_plans_vs_jit.test.ts — verify every shipped artifact in web/dist/patches/
 * matches native JIT output, sample for sample.
 *
 * Tests exactly what the browser fetches: the precompiled <slug>.wasm + the
 * <slug>.manifest.json, driven through the runtime package (WasmKernel). The
 * native oracle re-compiles the source patch and renders it through the JIT
 * (`diffcli render-bytes` / `render-graph`, every channel). If this passes, any
 * browser bug is in the host/worklet glue, not codegen.
 */

import { describe, test, expect } from 'bun:test'
import { spawnSync } from 'bun'
import { readFileSync, existsSync, writeFileSync } from 'fs'
import { join, resolve, dirname } from 'path'
import { fileURLToPath } from 'url'
import { WasmKernel, type KernelManifest } from '../../web/runtime/index'
import { expectAudible } from './energy_floor'

const __dirname = dirname(fileURLToPath(import.meta.url))
const repoRoot = resolve(__dirname, '../..')
const distDir = resolve(repoRoot, 'web/dist/patches')
const patchesDir = resolve(repoRoot, 'web/patches')
const diffcli = resolve(repoRoot, 'lean/.lake/build/bin/diffcli')
const cliEnv = { ...process.env, PATH: `${process.env.HOME}/.elan/bin:${process.env.PATH}` }

type IndexEntry = { slug: string; title: string; wasmPath: string; manifestPath: string }

/** Is this source a playground graph — a flat PatchGraph (`nodes`) or a v3
 *  hierarchy document (`version: 3`) — rather than a `tropical_program_2`
 *  patch (which names its `schema`)? Mirrors `Tropical.Playground.isGraphDocument`
 *  (lean/Tropical/Playground/Hierarchy.lean), the one detector the Lean front
 *  doors share; keep the two in step. */
function isGraphDocument(doc: { schema?: unknown; nodes?: unknown; version?: unknown }): boolean {
  return doc.schema === undefined && (Array.isArray(doc.nodes) || doc.version === 3)
}

/** Native oracle: compile the source patch and render it through the JIT,
 *  every output channel, frame-major. A `tropical_program_2` patch renders
 *  through `render-bytes --all-channels`; a PatchGraph through
 *  `render-graph --all-channels`. */
function runNative(slug: string, samples: number): { data: Float64Array; channels: number } {
  const source = join(patchesDir, `${slug}.json`)
  const parsed = JSON.parse(readFileSync(source, 'utf-8'))
  const asSamples = (b: Buffer) => ({
    data: new Float64Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)),
    channels: b.byteLength / (samples * 8),
  })
  if (isGraphDocument(parsed)) {
    const r = spawnSync([diffcli, 'render-graph', source, '--all-channels', '--frames', '1', '--buffer', String(samples)], { cwd: repoRoot, env: cliEnv })
    if (r.exitCode !== 0) throw new Error(`render-graph ${slug} failed: ${r.stderr?.toString()}`)
    return asSamples(r.stdout)
  }
  const c = spawnSync([diffcli, 'compile', source, '--mode=fused'], { cwd: repoRoot, env: cliEnv })
  if (c.exitCode !== 0) throw new Error(`compile ${slug} failed: ${c.stderr?.toString()}`)
  const planTmp = `/tmp/wpvj-${slug}.plan.json`
  writeFileSync(planTmp, c.stdout.toString())
  const r = spawnSync([diffcli, 'render-bytes', planTmp, '--all-channels', '--frames', '1', '--buffer', String(samples)], { cwd: repoRoot, env: cliEnv })
  if (r.exitCode !== 0) throw new Error(`render-bytes ${slug} failed: ${r.stderr?.toString()}`)
  return asSamples(r.stdout)
}

describe('web/dist shipped artifacts vs native JIT', () => {
  const indexPath = join(distDir, 'index.json')
  if (!existsSync(indexPath)) {
    test.skip('(skipped: run `bun web/build_patches.ts` first)', () => {})
    return
  }

  const entries = JSON.parse(readFileSync(indexPath, 'utf-8')) as IndexEntry[]
  const N = 512

  for (const e of entries) {
    test(`${e.slug}`, async () => {
      const { data: nat, channels } = runNative(e.slug, N)
      expect(channels).toBeGreaterThan(0)
      expectAudible(e.slug, nat)

      const wasmBytes = new Uint8Array(readFileSync(join(distDir, `${e.slug}.wasm`)))
      const manifest = JSON.parse(readFileSync(join(distDir, `${e.slug}.manifest.json`), 'utf-8')) as KernelManifest
      const k = await WasmKernel.instantiate(wasmBytes, manifest)
      const wasm = k.render(N).slice()
      const C = k.outputChannelCount

      expect(wasm.length).toBe(N * C)
      expect(channels).toBe(C) // the oracle carries every channel the artifact does
      let maxDiff = 0
      for (let s = 0; s < N; s++)
        for (let c = 0; c < channels; c++)
          maxDiff = Math.max(maxDiff, Math.abs(wasm[s * C + c]! - nat[s * channels + c]!))
      expect(maxDiff).toBeLessThan(1e-12)
    }, 20000) // compile (JIT) + wasm instantiate + dual render; the heaviest
              // patches (deeply-nested clocked oscillators) exceed the 5s default
  }
})
