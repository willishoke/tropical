/**
 * web_plans_vs_jit.test.ts — verify every shipped artifact in web/dist/patches/
 * matches native JIT output, sample for sample.
 *
 * Tests exactly what the browser fetches: the precompiled <slug>.wasm + the
 * <slug>.manifest.json, driven through the runtime package (WasmKernel). The
 * native oracle re-compiles the source patch and renders it through the JIT
 * (`diffcli render-bytes`). If this passes, any browser bug is in the
 * host/worklet glue, not codegen.
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

/** Native oracle: compile the source patch and render it through the JIT. */
function runNative(slug: string, samples: number): Float64Array {
  const c = spawnSync([diffcli, 'compile', join(patchesDir, `${slug}.json`), '--mode=fused'], { cwd: repoRoot, env: cliEnv })
  if (c.exitCode !== 0) throw new Error(`compile ${slug} failed: ${c.stderr?.toString()}`)
  const planTmp = `/tmp/wpvj-${slug}.plan.json`
  writeFileSync(planTmp, c.stdout.toString())
  const r = spawnSync([diffcli, 'render-bytes', planTmp, '--frames', '1', '--buffer', String(samples)], { cwd: repoRoot, env: cliEnv })
  if (r.exitCode !== 0) throw new Error(`render-bytes ${slug} failed: ${r.stderr?.toString()}`)
  const b = r.stdout
  const plan = JSON.parse(c.stdout.toString())
  const outputChannelCount = plan.output_channel_count ??
    (plan.sinks ?? []).reduce(
      (count: number, sink: { target?: number }) =>
        Math.max(count, (sink.target ?? 0) + 1),
      1,
    )
  return new Float64Array(
    b.buffer.slice(
      b.byteOffset, b.byteOffset + samples * outputChannelCount * 8))
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
      const nat = runNative(e.slug, N)
      expectAudible(e.slug, nat)

      const wasmBytes = new Uint8Array(readFileSync(join(distDir, `${e.slug}.wasm`)))
      const manifest = JSON.parse(readFileSync(join(distDir, `${e.slug}.manifest.json`), 'utf-8')) as KernelManifest
      const k = await WasmKernel.instantiate(wasmBytes, manifest, N)
      const wasm = k.render(N).slice()

      expect(wasm.length).toBe(nat.length)
      let maxDiff = 0
      for (let i = 0; i < nat.length; i++)
        maxDiff = Math.max(maxDiff, Math.abs(wasm[i]! - nat[i]!))
      expect(maxDiff).toBeLessThan(1e-12)
    }, 20000) // compile (JIT) + wasm instantiate + dual render; the heaviest
              // patches (deeply-nested clocked oscillators) exceed the 5s default
  }
})
