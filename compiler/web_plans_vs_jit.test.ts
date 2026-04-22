/**
 * web_plans_vs_jit.test.ts — verify every precompiled plan in web/dist/patches/
 * matches native JIT output, sample for sample.
 *
 * If this passes, the bug in the browser is in the runtime/worklet layer.
 * If this fails, the bug is in the build_patches step or emit_wasm.
 */

import { describe, test, expect } from 'bun:test'
import { readFileSync, readdirSync, existsSync } from 'fs'
import { join, resolve, dirname } from 'path'
import { fileURLToPath } from 'url'
import { makeSession } from './session'
import { type FlatPlan } from './flatten'
import { emitWasm } from './emit_wasm'

const __dirname = dirname(fileURLToPath(import.meta.url))
const distDir = resolve(__dirname, '../web/dist/patches')

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

function runNative(plan: FlatPlan, samples: number): Float64Array {
  const session = makeSession(samples)
  try {
    session.runtime.loadPlan(JSON.stringify(plan))
    session.runtime.process()
    return new Float64Array(session.runtime.outputBuffer)
  } finally {
    session.runtime.dispose()
  }
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
      const plan = JSON.parse(readFileSync(join(distDir, file), 'utf-8')) as FlatPlan

      const nat = runNative(plan, N)
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
