/**
 * microkernel_vs_fused.ts — head-to-head runtime + compile benchmark.
 *
 * For each patch, compile and run twice — once in `fused` mode (single
 * monolithic LLVM kernel, the legacy default), once in `microkernel`
 * mode (N+3 per-sample functions dispatched by FlatRuntime). Report
 * ns/sample, rt_ratio (% of 44.1kHz sample period), and TS+JIT
 * compile-latency deltas per mode per patch.
 *
 * Usage:  bun run tests/bench/microkernel_vs_fused.ts [--frames=N] [--keep-cache]
 *
 * Default patches probe the cost/benefit curve:
 *   - bubble_drip          (2 instances)   — microkernel overhead most
 *                                            visible as a percentage
 *   - cross_fm_4           (8 instances)   — "honest" synth voice case
 *   - 8x SinOsc            (polyphony)     — the use case microkernels
 *                                            exist for; cross-voice fusion
 *                                            never paid off anyway
 *   - compressor_harmonics (41 instances)  — worst case for dispatch
 *
 * Writes a structured result file to /tmp/microkernel_bench.json that
 * the Phase 8 report consumes.
 */
import { readFileSync, writeFileSync, rmSync } from 'node:fs'
import { resolve, basename, join } from 'node:path'
import { homedir } from 'node:os'
import {
  makeSession, loadJSON,
  resolveProgramType, instantiate, outputNames,
} from '../../compiler/session.js'
import { loadStdlib as loadBuiltins } from '../../compiler/program.js'
import { compileSession } from '../../compiler/ir/compile_session.js'
import { toWirePlan, type CompilationMode } from '../../compiler/flat_plan.js'
import { applyFlatPlan } from '../../compiler/apply_plan.js'
import { wireKey, portRef, instanceName, portName } from '../../compiler/ir/branded_names.js'
import * as b from '../../compiler/runtime/bindings.js'

const args = process.argv.slice(2)
const keepCache = args.includes('--keep-cache')
const framesArg = args.find(a => a.startsWith('--frames='))
const BENCH_FRAMES = framesArg ? parseInt(framesArg.split('=')[1], 10) : 4096
const FRAME_SIZE   = 256
const SAMPLE_RATE  = 44100
const WARMUP_FRAMES = 32

if (!keepCache) {
  const cacheDir = process.env.XDG_CACHE_HOME
    ? join(process.env.XDG_CACHE_HOME, 'tropical', 'kernels')
    : join(homedir(), '.cache', 'tropical', 'kernels')
  rmSync(cacheDir, { recursive: true, force: true })
  console.log(`(cold) cleared ${cacheDir}`)
}

interface CaseResult {
  case:           string
  instances:      number
  mode:           CompilationMode
  ts_ms:          number
  stringify_ms:   number
  jit_ms:         number
  ns_per_sample:  number
  rt_ratio:       number
}

const results: CaseResult[] = []

// ──────────────────────────────────────────────────────────────────────────
// Bench one already-built session against the given mode. Each call gets
// a fresh session — comparing modes within the same session would invoke
// hot-swap state-transfer which is a separate axis.
// ──────────────────────────────────────────────────────────────────────────

function benchSession(
  caseName: string,
  buildSession: () => ReturnType<typeof makeSession>,
  mode: CompilationMode,
): CaseResult {
  const session = buildSession()

  const t1 = performance.now()
  const plan = compileSession(session, { compilation_mode: mode })
  const tsMs = performance.now() - t1

  const t2 = performance.now()
  const planJson = JSON.stringify(toWirePlan(plan))
  const stringifyMs = performance.now() - t2

  const t3 = performance.now()
  session.runtime.loadPlan(planJson)
  const jitMs = performance.now() - t3

  // ── kernel execution timing ────────────────────────────────────────
  for (let i = 0; i < WARMUP_FRAMES; i++)
    b.tropical_runtime_process(session.runtime._h)

  const tProc0 = performance.now()
  for (let i = 0; i < BENCH_FRAMES; i++)
    b.tropical_runtime_process(session.runtime._h)
  const procMs = performance.now() - tProc0

  const totalSamples = BENCH_FRAMES * FRAME_SIZE
  const nsPerSample  = (procMs * 1e6) / totalSamples
  const samplePeriod = 1e9 / SAMPLE_RATE

  return {
    case:          caseName,
    instances:     plan.instance_functions.length,
    mode,
    ts_ms:         tsMs,
    stringify_ms:  stringifyMs,
    jit_ms:        jitMs,
    ns_per_sample: nsPerSample,
    rt_ratio:      nsPerSample / samplePeriod,
  }
}

// ── Patch-file cases ──────────────────────────────────────────────────
function patchSessionFactory(patchPath: string) {
  const json = JSON.parse(readFileSync(patchPath, 'utf-8'))
  return () => {
    const session = makeSession(FRAME_SIZE)
    loadBuiltins(session)
    loadJSON(json, session)
    return session
  }
}

// ── Polyphony case: N independent SinOsc voices, freq-spread ──────────
function polyphonySessionFactory(voiceCount: number) {
  return () => {
    const session = makeSession(FRAME_SIZE)
    loadBuiltins(session)
    const { type, typeArgs } = resolveProgramType(session, 'SinOsc', undefined, undefined)
    for (let i = 0; i < voiceCount; i++) {
      const name = `osc${i}`
      const inst = instantiate(type, name, { baseTypeName: 'SinOsc', typeArgs })
      session.instanceRegistry.set(name, inst)
      session.inputExprNodes.set(
        wireKey(portRef(instanceName(name), portName('freq'))),
        110 + 22 * i,
      )
      session.graphOutputs.push({ instance: name, output: outputNames(inst)[0] })
    }
    return session
  }
}

interface BenchCase {
  name:    string
  factory: () => ReturnType<typeof makeSession>
}

// Polyphony at multiple sizes shows how dispatch cost scales with
// instance count — the property that matters most for the per-voice
// microkernel roadmap.
const CASES: BenchCase[] = [
  { name: 'bubble_drip',         factory: patchSessionFactory(resolve('patches/bubble_drip.json')) },
  { name: 'cross_fm_4',          factory: patchSessionFactory(resolve('patches/cross_fm_4.json')) },
  { name: 'odd_harmonics',       factory: patchSessionFactory(resolve('patches/odd_harmonics.json')) },
  { name: 'polyphony_4x_SinOsc', factory: polyphonySessionFactory(4) },
  { name: 'polyphony_8x_SinOsc', factory: polyphonySessionFactory(8) },
  { name: 'polyphony_16x_SinOsc', factory: polyphonySessionFactory(16) },
  { name: 'polyphony_32x_SinOsc', factory: polyphonySessionFactory(32) },
]

const COLS = [
  'case', 'instances', 'mode',
  'ts_ms', 'stringify_ms', 'jit_ms',
  'ns/sample', 'rt_ratio',
]
console.log(COLS.join('\t'))

for (const c of CASES) {
  for (const mode of ['fused', 'microkernel'] as const) {
    try {
      const r = benchSession(c.name, c.factory, mode)
      results.push(r)
      console.log(
        [
          r.case,
          r.instances,
          r.mode,
          r.ts_ms.toFixed(1),
          r.stringify_ms.toFixed(1),
          r.jit_ms.toFixed(1),
          r.ns_per_sample.toFixed(1),
          `${(r.rt_ratio * 100).toFixed(2)}%`,
        ].join('\t'),
      )
    } catch (e: any) {
      console.log(`${c.name}\t-\t${mode}\tERR\t${e.message.split('\n')[0].slice(0, 200)}`)
    }
  }
}

// ── Compute mode deltas for the report ────────────────────────────────
const deltas: Array<{
  case:           string
  instances:      number
  fused_ns:       number
  microkernel_ns: number
  slowdown:       number     // microkernel / fused; >1 means microkernel is slower
  fused_jit_ms:   number
  microkernel_jit_ms: number
}> = []
for (const c of CASES) {
  const f = results.find(r => r.case === c.name && r.mode === 'fused')
  const m = results.find(r => r.case === c.name && r.mode === 'microkernel')
  if (f && m) {
    deltas.push({
      case:               c.name,
      instances:          f.instances,
      fused_ns:           f.ns_per_sample,
      microkernel_ns:     m.ns_per_sample,
      slowdown:           m.ns_per_sample / f.ns_per_sample,
      fused_jit_ms:       f.jit_ms,
      microkernel_jit_ms: m.jit_ms,
    })
  }
}

console.log('\nDeltas (microkernel / fused):')
for (const d of deltas) {
  console.log(
    `  ${d.case.padEnd(22)} ` +
    `${d.instances}×inst  ` +
    `ns: ${d.fused_ns.toFixed(1)} → ${d.microkernel_ns.toFixed(1)} ` +
    `(${d.slowdown.toFixed(2)}×)  ` +
    `jit: ${d.fused_jit_ms.toFixed(0)} → ${d.microkernel_jit_ms.toFixed(0)} ms`,
  )
}

// ── Persist structured result file for Phase 8 report ─────────────────
const resultFile = '/tmp/microkernel_bench.json'
writeFileSync(resultFile, JSON.stringify({
  config: {
    frame_size:    FRAME_SIZE,
    bench_frames:  BENCH_FRAMES,
    warmup_frames: WARMUP_FRAMES,
    sample_rate:   SAMPLE_RATE,
  },
  results,
  deltas,
  timestamp:       new Date().toISOString(),
}, null, 2))
console.log(`\nResults written to ${resultFile}`)
