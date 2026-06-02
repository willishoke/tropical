/**
 * bench_wasm_vs_jit.ts — head-to-head benchmark: WASM emit vs JIT.
 *
 * Measures two things per patch:
 *   1. Compile time: TS → FlatPlan, then WASM-emit vs JIT-load (cold).
 *   2. Runtime: ns/sample under both backends on the same FlatPlan.
 *
 * Usage:
 *   bun run tests/bench/wasm_vs_jit.ts <patch.json>... [--frames=N] [--block=N]
 *
 *   --frames  Number of audio frames to time (default 4096).
 *   --block   Samples per frame, for both backends (default 256).
 *
 * Wipes ~/.cache/tropical/kernels/ before each load so JIT compile is cold.
 *
 * The WASM module is built with `maxBlockSize = block`, instantiated once,
 * and `process(block, sampleIdx)` is called once per frame with the sample
 * counter advanced. The native runtime is loaded into a session with the
 * same buffer length and `process()` is called once per frame.
 */
import { readFileSync, rmSync } from 'node:fs'
import { resolve, basename, join } from 'node:path'
import { homedir } from 'node:os'
import { makeSession, loadJSON } from '../../compiler/session.js'
import { loadStdlib as loadBuiltins } from '../../compiler/program.js'
import { compileSession } from '../../compiler/ir/compile_session.js'
import { toWirePlan, type FlatPlan } from '../../compiler/flat_plan.js'
import { emitWasm } from '../../compiler/emit_wasm.js'
import * as b from '../../compiler/runtime/bindings.js'

const args = process.argv.slice(2)
const framesArg = args.find(a => a.startsWith('--frames='))
const blockArg = args.find(a => a.startsWith('--block='))
const benchFrames = framesArg ? parseInt(framesArg.split('=')[1], 10) : 4096
const BLOCK = blockArg ? parseInt(blockArg.split('=')[1], 10) : 256
const patches = args.filter(a => !a.startsWith('--'))
const SAMPLE_RATE = 44100

if (patches.length === 0) {
  console.error('Usage: bun run tests/bench/wasm_vs_jit.ts <patch.json>... [--frames=N] [--block=N]')
  process.exit(1)
}

const cacheDir = process.env.XDG_CACHE_HOME
  ? join(process.env.XDG_CACHE_HOME, 'tropical', 'kernels')
  : join(homedir(), '.cache', 'tropical', 'kernels')

function clearJitCache(): void {
  rmSync(cacheDir, { recursive: true, force: true })
}

function initWasmRegisters(memory: WebAssembly.Memory, regOffset: number, stateInit: (number | boolean | unknown)[], regTypes: string[]): void {
  const dv = new DataView(memory.buffer)
  for (let i = 0; i < stateInit.length; i++) {
    const v = stateInit[i]
    const t = regTypes[i] ?? 'float'
    const off = regOffset + i * 8
    if (Array.isArray(v)) {
      dv.setFloat64(off, 0, true)
    } else if (typeof v === 'boolean') {
      dv.setBigInt64(off, v ? 1n : 0n, true)
    } else if (t === 'int') {
      dv.setBigInt64(off, BigInt(Math.trunc(v as number)), true)
    } else if (t === 'bool') {
      dv.setBigInt64(off, (v as number) !== 0 ? 1n : 0n, true)
    } else {
      dv.setFloat64(off, v as number, true)
    }
  }
}

type Row = {
  patch: string
  instrs: number
  arrays: number
  // Compile times (ms)
  tsMs: number          // FlatPlan production (shared)
  wasmEmitMs: number    // FlatPlan → WASM bytes
  wasmInstMs: number    // WebAssembly.compile + instantiate
  jsonMs: number        // FlatPlan → JSON
  jitLoadMs: number     // Native loadPlan (cold)
  // Sizes
  planJsonKb: number
  wasmKb: number
  // Runtime (ns/sample)
  jitNsPerSample: number
  wasmNsPerSample: number
  // Realtime ratios
  jitRtPct: number
  wasmRtPct: number
}

const rows: Row[] = []
const errors: { patch: string; stage: string; msg: string }[] = []

for (const p of patches) {
  const patchPath = resolve(p)
  const patchName = basename(p, '.json')
  const json = JSON.parse(readFileSync(patchPath, 'utf-8'))

  // ── shared: produce FlatPlan via the full session pipeline ──
  const tsSession = makeSession(BLOCK)
  let plan: FlatPlan
  let tsMs: number
  try {
    loadBuiltins(tsSession)
    const t0 = performance.now()
    // Walk the parts of loadJSON that don't touch the JIT, to isolate TS-only
    // cost. We do this by calling compileSession after instantiating the
    // session graph from the JSON. The simplest way to materialize the graph
    // shape is to call loadJSON on a *separate* session that doesn't try to
    // load the resulting plan into the runtime — but loadJSON does both in
    // one call, so instead we do: full loadJSON on tsSession (which also
    // loads the plan into native), then re-run compileSession() to get a
    // clean TS-only number on the second pass.
    loadJSON(json, tsSession)
    tsMs = performance.now() - t0
    plan = compileSession(tsSession)
  } catch (e: any) {
    errors.push({ patch: patchName, stage: 'compileSession', msg: e.message?.split('\n')[0]?.slice(0, 200) ?? String(e) })
    tsSession.runtime.dispose()
    continue
  }

  // Re-measure TS-only cost on a fresh session to back out JIT load from tsMs.
  const tsOnlySession = makeSession(BLOCK)
  let tsOnlyMs: number
  try {
    loadBuiltins(tsOnlySession)
    // Materialize same instance graph by mutating session — easier to just
    // re-run compileSession against the already-populated tsSession.
    const t = performance.now()
    compileSession(tsSession)
    tsOnlyMs = performance.now() - t
  } finally {
    tsOnlySession.runtime.dispose()
  }

  // ── WASM emit + instantiate ──
  let wasmEmitMs = 0, wasmInstMs = 0, wasmKb = 0
  let memory: WebAssembly.Memory | null = null
  let processWasm: ((blen: number, sidx: bigint) => void) | null = null
  let wasmLayout: { registersOffset: number; outputOffset: number } | null = null
  try {
    const tE0 = performance.now()
    const { bytes, layout } = emitWasm(plan, { maxBlockSize: BLOCK })
    wasmEmitMs = performance.now() - tE0
    wasmKb = bytes.length / 1024
    wasmLayout = layout

    const tI0 = performance.now()
    const mod = await WebAssembly.compile(bytes)
    const instance = await WebAssembly.instantiate(mod, {})
    wasmInstMs = performance.now() - tI0
    memory = instance.exports.memory as WebAssembly.Memory
    processWasm = instance.exports.process as (blen: number, sidx: bigint) => void

    // Seed registers from state_init (scalar entries only).
    const scalarStateInit = plan.state_init.map((v) => (Array.isArray(v) ? 0 : v)) as (number | boolean)[]
    initWasmRegisters(memory, layout.registersOffset, scalarStateInit, plan.register_types)
  } catch (e: any) {
    errors.push({ patch: patchName, stage: 'wasm', msg: e.message?.split('\n')[0]?.slice(0, 200) ?? String(e) })
  }

  // ── JIT compile time (cold cache) ──
  clearJitCache()
  const jitSession = makeSession(BLOCK)
  let jitLoadMs = 0, planJsonKb = 0, jsonMs = 0
  try {
    loadBuiltins(jitSession)
    const tJs0 = performance.now()
    const planJson = JSON.stringify(toWirePlan(plan))
    jsonMs = performance.now() - tJs0
    planJsonKb = planJson.length / 1024

    const tL0 = performance.now()
    jitSession.runtime.loadPlan(planJson)
    jitLoadMs = performance.now() - tL0
  } catch (e: any) {
    errors.push({ patch: patchName, stage: 'jitLoad', msg: e.message?.split('\n')[0]?.slice(0, 200) ?? String(e) })
    jitSession.runtime.dispose()
    if (!processWasm) continue
  }

  // ── Runtime: JIT ns/sample ──
  const warmupFrames = 32
  let jitNsPerSample = 0
  if (jitSession.runtime._h) {
    for (let i = 0; i < warmupFrames; i++) b.tropical_runtime_process(jitSession.runtime._h)
    const t0 = performance.now()
    for (let i = 0; i < benchFrames; i++) b.tropical_runtime_process(jitSession.runtime._h)
    const ms = performance.now() - t0
    jitNsPerSample = (ms * 1e6) / (benchFrames * BLOCK)
  }

  // ── Runtime: WASM ns/sample ──
  let wasmNsPerSample = 0
  if (processWasm) {
    let sampleIdx = 0n
    for (let i = 0; i < warmupFrames; i++) {
      processWasm(BLOCK, sampleIdx)
      sampleIdx += BigInt(BLOCK)
    }
    const t0 = performance.now()
    for (let i = 0; i < benchFrames; i++) {
      processWasm(BLOCK, sampleIdx)
      sampleIdx += BigInt(BLOCK)
    }
    const ms = performance.now() - t0
    wasmNsPerSample = (ms * 1e6) / (benchFrames * BLOCK)
  }

  const samplePeriodNs = 1e9 / SAMPLE_RATE
  const instrCount =
      plan.instance_functions.reduce((n, i) => n + i.instructions.length, 0)

  rows.push({
    patch: patchName,
    instrs: instrCount,
    arrays: plan.array_slot_sizes?.length ?? 0,
    tsMs: tsOnlyMs,
    wasmEmitMs,
    wasmInstMs,
    jsonMs,
    jitLoadMs,
    planJsonKb,
    wasmKb,
    jitNsPerSample,
    wasmNsPerSample,
    jitRtPct: (jitNsPerSample / samplePeriodNs) * 100,
    wasmRtPct: (wasmNsPerSample / samplePeriodNs) * 100,
  })

  try { jitSession.runtime.dispose() } catch {}
  try { tsSession.runtime.dispose() } catch {}
}

// ── pretty-print ─────────────────────────────────────────────────
function pad(s: string, n: number, right = false): string {
  if (s.length >= n) return s
  return right ? s.padEnd(n) : s.padStart(n)
}

console.log()
console.log(`Settings: frames=${benchFrames}, block=${BLOCK}, samples=${benchFrames * BLOCK}, sr=${SAMPLE_RATE}Hz`)
console.log()
console.log('Compile time (ms)                              | Size (KiB) | Runtime (ns/sample)')
console.log(pad('patch', 22, true) + ' | ' +
  pad('instr', 6) + ' ' + pad('arr', 4) + ' | ' +
  pad('ts', 7) + ' ' + pad('emit', 7) + ' ' + pad('inst', 7) + ' | ' +
  pad('json', 7) + ' ' + pad('load', 7) + ' | ' +
  pad('plan', 6) + ' ' + pad('wasm', 6) + ' | ' +
  pad('jit', 9) + ' ' + pad('wasm', 9) + ' ' + pad('ratio', 6) + ' | ' +
  pad('jit%', 6) + ' ' + pad('wasm%', 6))
console.log('-'.repeat(140))
for (const r of rows) {
  const ratio = r.jitNsPerSample > 0 ? (r.wasmNsPerSample / r.jitNsPerSample).toFixed(2) + 'x' : 'n/a'
  console.log(
    pad(r.patch, 22, true) + ' | ' +
    pad(r.instrs.toString(), 6) + ' ' + pad(r.arrays.toString(), 4) + ' | ' +
    pad(r.tsMs.toFixed(1), 7) + ' ' + pad(r.wasmEmitMs.toFixed(1), 7) + ' ' + pad(r.wasmInstMs.toFixed(1), 7) + ' | ' +
    pad(r.jsonMs.toFixed(1), 7) + ' ' + pad(r.jitLoadMs.toFixed(1), 7) + ' | ' +
    pad(r.planJsonKb.toFixed(0), 6) + ' ' + pad(r.wasmKb.toFixed(0), 6) + ' | ' +
    pad(r.jitNsPerSample.toFixed(1), 9) + ' ' + pad(r.wasmNsPerSample.toFixed(1), 9) + ' ' + pad(ratio, 6) + ' | ' +
    pad(r.jitRtPct.toFixed(1) + '%', 6) + ' ' + pad(r.wasmRtPct.toFixed(1) + '%', 6))
}

if (errors.length > 0) {
  console.log()
  console.log('Errors:')
  for (const e of errors) console.log(`  ${e.patch} [${e.stage}]: ${e.msg}`)
}

// Quick aggregate
const valid = rows.filter(r => r.jitNsPerSample > 0 && r.wasmNsPerSample > 0)
if (valid.length > 0) {
  const gm = (xs: number[]) => Math.exp(xs.reduce((s, x) => s + Math.log(x), 0) / xs.length)
  const ratios = valid.map(r => r.wasmNsPerSample / r.jitNsPerSample)
  console.log()
  console.log(`Geometric mean WASM/JIT runtime ratio: ${gm(ratios).toFixed(2)}x (across ${valid.length} patches)`)
}
