/**
 * End-to-end compile + execution bench:
 *   TS pipeline → JSON.stringify → JIT loadPlan → runtime.process loop.
 *
 * Usage:  bun run tests/bench/jit_runtime.ts <patch.json>... [--keep-cache] [--frames=N]
 *
 * Wipes ~/.cache/tropical/kernels/ before running so every JIT load is a
 * true cold compile. Pass --keep-cache to leave the disk cache in place
 * (useful when testing cache subsystem behavior).
 *
 * The OrcJitEngine in-memory cache is process-local, so each invocation
 * of this script starts cold for free.
 *
 * Per-patch kernel timing: runs runtime.process() N times after load
 * (default 4096 frames @ 256 samples/frame ≈ 23s of audio at 44.1kHz)
 * and reports ns/sample. This is the realtime-cost number — it must
 * stay well below sample_period (~22.7μs @ 44.1k) to avoid xruns.
 *
 * Companion to compile.ts (which times only the TS pipeline).
 */
import { readFileSync, writeFileSync, rmSync } from 'node:fs'
import { resolve, basename, join } from 'node:path'
import { homedir } from 'node:os'
import { makeSession, loadJSON } from '../../compiler/session.js'
import { loadStdlib as loadBuiltins } from '../../compiler/program.js'
import { compileSession } from '../../compiler/ir/compile_session.js'
import { toWirePlan } from '../../compiler/flat_plan.js'
import * as b from '../../compiler/runtime/bindings.js'

const args = process.argv.slice(2)
const keepCache = args.includes('--keep-cache')
const framesArg = args.find(a => a.startsWith('--frames='))
const benchFrames = framesArg ? parseInt(framesArg.split('=')[1], 10) : 4096
const patches = args.filter(a => !a.startsWith('--'))
const FRAME_SIZE = 256
const SAMPLE_RATE = 44100

if (patches.length === 0) {
  console.error('Usage: bun run tests/bench/jit_runtime.ts <patch.json>... [--keep-cache]')
  process.exit(1)
}

if (!keepCache) {
  const cacheDir = process.env.XDG_CACHE_HOME
    ? join(process.env.XDG_CACHE_HOME, 'tropical', 'kernels')
    : join(homedir(), '.cache', 'tropical', 'kernels')
  rmSync(cacheDir, { recursive: true, force: true })
  console.log(`(cold) cleared ${cacheDir}`)
}

const cols = [
  'patch',
  'total_ms', 'ts_ms', 'json_kb', 'jit_ms',
  'instrs', 'arrays',
  'ns/sample', 'rt_ratio',
]
console.log(cols.join('\t'))

for (const p of patches) {
  const patchPath = resolve(p)
  const json = JSON.parse(readFileSync(patchPath, 'utf-8'))

  // Cold end-to-end: fresh session, then loadJSON (compileSession +
  // stringify + loadPlan internally). Disk cache is wiped at startup
  // and the OrcJitEngine in-memory cache is empty in a fresh process.
  const session = makeSession(256)
  loadBuiltins(session)

  let totalMs: number
  try {
    const tStart = performance.now()
    loadJSON(json, session)
    totalMs = performance.now() - tStart
  } catch (e: any) {
    console.log(`${basename(p)}\tERR\t${e.message.split('\n')[0].slice(0, 200)}`)
    continue
  }

  // Re-run only the TS half against the now-populated session so we can
  // back out the JIT cost. The disk cache will hit on this second pass,
  // so the loadPlan inside compileSession isn't called here — we run
  // compileSession directly.
  const t1 = performance.now()
  const plan = compileSession(session)
  const tsMs = performance.now() - t1

  const t2 = performance.now()
  const planJson = JSON.stringify(toWirePlan(plan))
  const stringifyMs = performance.now() - t2

  const jitMs = totalMs - tsMs - stringifyMs

  // ── kernel execution timing ────────────────────────────────────────
  // Warmup a few frames (instruction cache, branch predictor), then
  // measure ns/sample over benchFrames * FRAME_SIZE samples.
  const warmupFrames = 32
  for (let i = 0; i < warmupFrames; i++) b.tropical_runtime_process(session.runtime._h)

  const tProc0 = performance.now()
  for (let i = 0; i < benchFrames; i++) b.tropical_runtime_process(session.runtime._h)
  const procMs = performance.now() - tProc0
  const totalSamples = benchFrames * FRAME_SIZE
  const nsPerSample = (procMs * 1e6) / totalSamples
  const samplePeriodNs = 1e9 / SAMPLE_RATE
  const rtRatio = nsPerSample / samplePeriodNs

  const arrays = plan.array_slot_sizes?.length ?? 0
  console.log(
    [
      basename(p),
      totalMs.toFixed(1),
      `${tsMs.toFixed(1)}+${stringifyMs.toFixed(1)}`,
      (planJson.length / 1024).toFixed(0),
      jitMs.toFixed(1),
      plan.scheduler_function.preamble.length
        + plan.scheduler_function.postamble.length
        + plan.instance_functions.reduce((n, i) => n + i.instructions.length, 0),
      arrays,
      nsPerSample.toFixed(1),
      `${(rtRatio * 100).toFixed(1)}%`,
    ].join('\t'),
  )

  writeFileSync(`/tmp/${basename(p)}.plan4.json`, planJson)
}
