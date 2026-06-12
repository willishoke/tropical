/**
 * web_plans_vs_jit.test.ts — verify every precompiled plan in web/dist/patches/
 * matches native JIT output, sample for sample.
 *
 * If this passes, the bug in the browser is in the runtime/worklet layer.
 * If this fails, the bug is in the build_patches step or emit_wasm.
 *
 * Both backends come off the SAME on-disk `tropical_plan_5` wire plan:
 *   - WASM side: emit_wasm (relocated to web/wasm/) on the parsed plan.
 *   - Native side: `diffcli render-bytes <plan.json>` (the Lean JIT) —
 *     the dist `.plan.json` IS the wire plan, fed straight to the CLI.
 * No TS-compiler import and no native FFI — fully decoupled post-Phase-8.
 */

import { describe, test, expect } from 'bun:test'
import { spawnSync } from 'bun'
import { readFileSync, readdirSync, existsSync } from 'fs'
import { join, resolve, dirname } from 'path'
import { fileURLToPath } from 'url'
import { type FlatPlan, type WireFlatPlan, parseWirePlan } from '../../web/wasm/flat_plan'
import { emitWasm } from '../../web/wasm/emit_wasm'

const __dirname = dirname(fileURLToPath(import.meta.url))
// tests/web/ is two levels under the repo root, like tests/equiv/ was.
const repoRoot = resolve(__dirname, '../..')
const distDir = resolve(repoRoot, 'web/dist/patches')
const diffcli = resolve(repoRoot, 'lean/.lake/build/bin/diffcli')
// diffcli reads stdlib/parsed/manifest.json relative to its cwd, so run
// from the repo root and put elan on PATH for the Lean toolchain.
const cliEnv = { ...process.env, PATH: `${process.env.HOME}/.elan/bin:${process.env.PATH}` }

function initWasmState(memory: WebAssembly.Memory, regOffset: number, stateInit: (number | boolean)[], regTypes: string[]): void {
  const dv = new DataView(memory.buffer)
  for (let i = 0; i < stateInit.length; i++) {
    const v = stateInit[i]
    if (Array.isArray(v)) continue
    const t = regTypes[i] ?? 'float'
    const off = regOffset + i * 8
    if (typeof v === 'boolean') dv.setBigInt64(off, v ? 1n : 0n, true)
    else if (t === 'int') dv.setBigInt64(off, BigInt(Math.trunc(v as number)), true)
    else if (t === 'bool') dv.setBigInt64(off, (v as number) !== 0 ? 1n : 0n, true)
    else dv.setFloat64(off, v as number, true)
  }
}

async function runWasm(plan: FlatPlan, samples: number): Promise<Float64Array> {
  const { bytes, layout } = emitWasm(plan, { maxBlockSize: samples })
  const mod = await WebAssembly.compile(bytes)
  const instance = await WebAssembly.instantiate(mod, {})
  const memory = instance.exports.memory as WebAssembly.Memory
  const processFn = instance.exports.process as (blen: number, sidx: bigint) => void
  initWasmState(memory, layout.registersOffset, plan.state_init, plan.register_types)
  processFn(samples, 0n)
  return new Float64Array(memory.buffer, layout.outputOffset, samples).slice()
}

/** Native side: render the on-disk wire plan through the Lean engine's
 *  JIT via `diffcli render-bytes`. The dist `.plan.json` file IS the
 *  wire plan, so its path is passed straight to the CLI. */
function runNative(planPath: string, samples: number): Float64Array {
  const r = spawnSync([diffcli, 'render-bytes', planPath, '--frames', '1', '--buffer', String(samples)], { cwd: repoRoot, env: cliEnv })
  if (r.exitCode !== 0) {
    throw new Error(`diffcli render-bytes failed (exit ${r.exitCode}): ${r.stderr?.toString()}`)
  }
  const b = r.stdout
  return new Float64Array(b.buffer.slice(b.byteOffset, b.byteOffset + samples * 8))
}

function bufStats(buf: Float64Array): { peak: number; rms: number; nonZero: number } {
  let peak = 0, sumSq = 0, nz = 0
  for (let i = 0; i < buf.length; i++) {
    const v = buf[i]!
    if (Math.abs(v) > 1e-12) nz++
    peak = Math.max(peak, Math.abs(v))
    sumSq += v * v
  }
  return { peak, rms: Math.sqrt(sumSq / buf.length), nonZero: nz }
}

describe('web/dist precompiled plans vs native JIT', () => {
  if (!existsSync(distDir)) {
    test.skip('(skipped: run `bun web/build_patches.ts` first)', () => {})
    return
  }

  const planFiles = readdirSync(distDir).filter((f) => f.endsWith('.plan.json'))
  if (planFiles.length === 0) {
    test.skip('(no plans found in web/dist/patches)', () => {})
    return
  }

  const N = 512

  for (const file of planFiles) {
    test(`${file}`, async () => {
      const path = join(distDir, file)
      const wire = JSON.parse(readFileSync(path, 'utf-8')) as WireFlatPlan
      const plan: FlatPlan = parseWirePlan(wire)

      const nat = runNative(path, N)
      const wasm = await runWasm(plan, N)

      const natStats = bufStats(nat)
      const wasmStats = bufStats(wasm)

      // Log native peak so silent plans (pre-existing flattener issues with
      // certain stdlib programs) are visible without failing the WASM↔JIT
      // match check.
      if (natStats.peak < 1e-9) {
        // eslint-disable-next-line no-console
        console.log(`  ${file}: both backends silent (native peak ${natStats.peak.toExponential(1)}) — skipping diff`)
      }

      let maxDiff = 0, firstDiffIdx = -1
      for (let i = 0; i < N; i++) {
        const d = Math.abs(wasm[i]! - nat[i]!)
        if (d > maxDiff) {
          maxDiff = d
          if (d > 1e-9 && firstDiffIdx < 0) firstDiffIdx = i
        }
      }
      if (maxDiff > 1e-9) {
        // Emit a diagnostic with context so we know what's going on.
        const i0 = Math.max(0, firstDiffIdx - 2)
        const i1 = Math.min(N, firstDiffIdx + 4)
        const tail = []
        for (let i = i0; i < i1; i++) {
          tail.push(`${i}: nat=${nat[i]!.toFixed(9)} wasm=${wasm[i]!.toFixed(9)} Δ=${(wasm[i]! - nat[i]!).toExponential(2)}`)
        }
        // eslint-disable-next-line no-console
        console.log(
          `\n${file}: maxDiff=${maxDiff.toExponential(3)}; peaks nat=${natStats.peak.toFixed(4)} wasm=${wasmStats.peak.toFixed(4)}\n  ` +
          tail.join('\n  '),
        )
      }
      expect(maxDiff).toBeLessThan(1e-9)
    })
  }
})
