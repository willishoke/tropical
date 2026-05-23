/**
 * depth_vs_flat.ts — focused bench: legacy inline-everything path
 * (`inlineNested:true`) vs M11 fractal slot path (`inlineNested:false`).
 *
 * Scope: TWO cases, not sixteen. The 14-case version from earlier was
 * statistically meaningless (one trial per case, point estimates, cache
 * state leaked between cases) and most of the cases told the same story
 * at different magnifications. The two retained cases bracket the
 * deletion decision:
 *
 *   - OnePole               minimal nested (2 children)
 *                           — is the slot path's overhead detectable
 *                             at the smallest scale?
 *
 *   - polyphony_8x_Phaser16 stress (8 voices × 17 nested kernels = 136
 *                           kernel boundaries)
 *                           — is the overhead tolerable at scale?
 *
 * Methodology:
 *
 *   For each (case, mode):
 *     - ONE cold-cache compile trial. The engine has a process-
 *       singleton in-memory cache that survives `rmSync` of the disk
 *       cache; a second compile within the same process hits that
 *       cache and returns in <1ms. Measuring multiple cold compiles
 *       in one process requires subprocess isolation, which isn't
 *       worth the complexity for a snapshot bench. We measure cold
 *       once and accept the single-shot number.
 *     - N=3 runtime trials of 1024 frames × 256 samples each, take
 *       min for ns/sample. N=3 is the empirically-derived
 *       recommendation from runtime_noise_meta.ts: min-of-3 lands
 *       within 0.76% of asymptotic min at p95 confidence on the
 *       same hardware class.
 *
 *   Interleaving: round-robin (case, mode) across runtime trials so a
 *   thermal slope or frequency-scaling transition doesn't systematically
 *   bias one mode.
 *
 * Wall clock: ~3s (compile + runtime + cache clears).
 *
 * Output: /tmp/depth_bench.json
 *
 * Usage: bun run tests/bench/depth_vs_flat.ts
 */
import { writeFileSync, rmSync } from 'node:fs'
import { join } from 'node:path'
import { homedir } from 'node:os'
import {
  makeSession,
  resolveProgramType, instantiate, outputNames, inputNames,
} from '../../compiler/session.js'
import type { ExprNode } from '../../compiler/expr.js'
import { loadStdlib } from '../../compiler/program.js'
import { compileSession } from '../../compiler/ir/compile_session.js'
import { toWirePlan } from '../../compiler/flat_plan.js'
import { wireKey, portRef, instanceName, portName } from '../../compiler/ir/branded_names.js'
import * as b from '../../compiler/runtime/bindings.js'

const FRAME_SIZE       = 256
const FRAMES_PER_TRIAL = 1024
const RUNTIME_TRIALS   = 3        // derived from runtime_noise_meta.ts
// Compile time is measured ONCE per (case, mode). The engine has a
// process-singleton in-memory cache that survives `rmSync` of the
// disk cache, so a second compile of the same plan hits the in-memory
// cache and returns in <1ms. Measuring K cold compiles in one process
// would need subprocess isolation; for snapshot purposes a single
// cold-cache measurement per axis is what we actually want anyway.
const WARMUP_FRAMES    = 32
const SAMPLE_RATE      = 44100

type DepthMode = 'flat' | 'nested'

function pulseEvery(n: number): ExprNode {
  return { op: 'lt', args: [{ op: 'mod', args: [{ op: 'sampleIndex' }, n] }, 1] }
}
const DEFAULT_INPUTS: Record<string, ExprNode> = {
  freq: 220, x: 0.5, y: 0.5, audio: 0.5, input: 0.5, cv: 0.5,
  cutoff: 1000, q: 0.5, drive: 1.0, mix: 0.5, a: 0.3, b: 0.7,
  coeff: 0.4, feedback: 0.4, lfo_speed: 0.2, decay: 0.99,
  rate: 5, g: 0.1, resonance: 0.5,
  trigger: pulseEvery(64), clock: pulseEvery(32),
}
const wk = (i: string, p: string) => wireKey(portRef(instanceName(i), portName(p)))

// ── Session builders ──────────────────────────────────────────────────
type SessionFactory = (mode: DepthMode) => ReturnType<typeof makeSession>

const onePoleFactory: SessionFactory = (mode) => {
  const session = makeSession(FRAME_SIZE, { inlineNested: mode === 'flat' })
  loadStdlib(session)
  const { type, typeArgs } = resolveProgramType(session, 'OnePole', undefined, undefined)
  const inst = instantiate(type, 'inst', { baseTypeName: 'OnePole', typeArgs })
  session.instanceRegistry.set('inst', inst)
  for (const pn of inputNames(inst)) {
    if (pn in DEFAULT_INPUTS) session.inputExprNodes.set(wk('inst', pn), DEFAULT_INPUTS[pn])
  }
  session.graphOutputs.push({ instance: 'inst', output: outputNames(inst)[0] })
  return session
}

const polyphony8xPhaser16Factory: SessionFactory = (mode) => {
  const session = makeSession(FRAME_SIZE, { inlineNested: mode === 'flat' })
  loadStdlib(session)
  const { type, typeArgs } = resolveProgramType(session, 'Phaser16', undefined, undefined)
  for (let i = 0; i < 8; i++) {
    const name = `v${i}`
    const inst = instantiate(type, name, { baseTypeName: 'Phaser16', typeArgs })
    session.instanceRegistry.set(name, inst)
    for (const pn of inputNames(inst)) {
      if (pn === 'freq') session.inputExprNodes.set(wk(name, pn), 110 + 22 * i)
      else if (pn in DEFAULT_INPUTS) session.inputExprNodes.set(wk(name, pn), DEFAULT_INPUTS[pn])
    }
    session.graphOutputs.push({ instance: name, output: outputNames(inst)[0] })
  }
  return session
}

interface BenchCase { name: string; factory: SessionFactory }
const CASES: BenchCase[] = [
  { name: 'OnePole',                 factory: onePoleFactory },
  { name: 'polyphony_8x_Phaser16',   factory: polyphony8xPhaser16Factory },
]

// ── Stats ─────────────────────────────────────────────────────────────
function median(xs: number[]): number {
  const sorted = [...xs].sort((a, b) => a - b)
  const n = sorted.length
  return n % 2 === 1 ? sorted[(n - 1) / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2
}

// ── Cache control ─────────────────────────────────────────────────────
const cacheDir = process.env.XDG_CACHE_HOME
  ? join(process.env.XDG_CACHE_HOME, 'tropical', 'kernels')
  : join(homedir(), '.cache', 'tropical', 'kernels')
function clearCache() { rmSync(cacheDir, { recursive: true, force: true }) }

// ── Compile measurement (one cold trial per axis) ─────────────────────
interface CompileMeasurement {
  ts_ms:      number
  jit_ms:     number
  instances:  number
  plan_bytes: number
}

function measureCompile(factory: SessionFactory, mode: DepthMode): CompileMeasurement {
  clearCache()
  const session = factory(mode)
  const t1 = performance.now()
  const plan = compileSession(session)
  const ts_ms = performance.now() - t1
  const planJson = JSON.stringify(toWirePlan(plan))
  const t2 = performance.now()
  session.runtime.loadPlan(planJson)
  const jit_ms = performance.now() - t2
  return {
    ts_ms,
    jit_ms,
    instances:  plan.instance_functions.length,
    plan_bytes: planJson.length,
  }
}

// ── Runtime measurement (single session, N trials, min) ───────────────
// Compiled-and-loaded session passed in; this fn only times the loop.
interface RuntimeMeasurement {
  ns_per_sample_trials: number[]
  ns_per_sample_min:    number
  ns_per_sample_median: number
  ns_per_sample_max:    number
}

function runtimeTrial(session: ReturnType<typeof makeSession>): number {
  const samplesPerTrial = FRAMES_PER_TRIAL * FRAME_SIZE
  const t0 = performance.now()
  for (let f = 0; f < FRAMES_PER_TRIAL; f++)
    b.tropical_runtime_process(session.runtime._h)
  const elapsed = performance.now() - t0
  return (elapsed * 1e6) / samplesPerTrial
}

// ── Build + warmup one session per (case, mode) for runtime trials ────
interface PreparedSession {
  caseName: string
  mode:     DepthMode
  session:  ReturnType<typeof makeSession>
  trials:   number[]
}

function prepare(factory: SessionFactory, mode: DepthMode, caseName: string): PreparedSession {
  const session = factory(mode)
  const plan = compileSession(session)
  const planJson = JSON.stringify(toWirePlan(plan))
  session.runtime.loadPlan(planJson)
  for (let i = 0; i < WARMUP_FRAMES; i++)
    b.tropical_runtime_process(session.runtime._h)
  return { caseName, mode, session, trials: [] }
}

// ──────────────────────────────────────────────────────────────────────
// Run
// ──────────────────────────────────────────────────────────────────────

console.log('=== Compile measurements (one cold-cache trial per axis) ===')
console.log('case                          mode    instances    ts_ms   jit_ms  plan_kb')

const compile: Record<string, Record<DepthMode, CompileMeasurement>> = {}
for (const c of CASES) {
  compile[c.name] = {} as Record<DepthMode, CompileMeasurement>
  for (const mode of ['flat', 'nested'] as const) {
    const m = measureCompile(c.factory, mode)
    compile[c.name][mode] = m
    console.log(
      `${c.name.padEnd(30)}${mode.padEnd(8)}${String(m.instances).padStart(5)}     ` +
      `${m.ts_ms.toFixed(1).padStart(6)}  ${m.jit_ms.toFixed(1).padStart(7)}  ${(m.plan_bytes / 1024).toFixed(1).padStart(7)}`,
    )
  }
}

// Runtime: prepare all sessions, then interleave trials.
console.log('\n=== Runtime measurements (N=3 trials, interleaved, min reported) ===')
const prepared: PreparedSession[] = []
for (const c of CASES) {
  for (const mode of ['flat', 'nested'] as const) {
    prepared.push(prepare(c.factory, mode, c.name))
  }
}

// Round-robin: trial_k for every (case, mode) before advancing to trial_k+1
for (let k = 0; k < RUNTIME_TRIALS; k++) {
  for (const p of prepared) {
    p.trials.push(runtimeTrial(p.session))
  }
}

console.log('case                          mode    trials (ns/sample)            min     median  max')
const runtime: Record<string, Record<DepthMode, RuntimeMeasurement>> = {}
for (const p of prepared) {
  const sorted = [...p.trials].sort((a, b) => a - b)
  const m: RuntimeMeasurement = {
    ns_per_sample_trials: p.trials,
    ns_per_sample_min:    sorted[0],
    ns_per_sample_median: median(p.trials),
    ns_per_sample_max:    sorted[sorted.length - 1],
  }
  if (!runtime[p.caseName]) runtime[p.caseName] = {} as Record<DepthMode, RuntimeMeasurement>
  runtime[p.caseName][p.mode] = m
  const trialStr = p.trials.map(t => t.toFixed(1).padStart(7)).join(' ')
  console.log(
    `${p.caseName.padEnd(30)}${p.mode.padEnd(8)}${trialStr}    ` +
    `${m.ns_per_sample_min.toFixed(1).padStart(6)}  ${m.ns_per_sample_median.toFixed(1).padStart(6)}  ${m.ns_per_sample_max.toFixed(1).padStart(6)}`,
  )
}

// ── Deltas (the bottom line) ──────────────────────────────────────────
console.log('\n=== Deltas (nested / flat; ratios on min for runtime, median for compile) ===')
console.log('case                          ns_min  jit_med  plan')
const deltas: any[] = []
const samplePeriod = 1e9 / SAMPLE_RATE
for (const c of CASES) {
  const f = { ...runtime[c.name].flat,   ...compile[c.name].flat   }
  const n = { ...runtime[c.name].nested, ...compile[c.name].nested }
  const d = {
    case:              c.name,
    instances:         compile[c.name].nested.instances,
    flat_ns_min:       f.ns_per_sample_min,
    nested_ns_min:     n.ns_per_sample_min,
    ns_ratio:          n.ns_per_sample_min / f.ns_per_sample_min,
    flat_jit_ms:       compile[c.name].flat.jit_ms,
    nested_jit_ms:     compile[c.name].nested.jit_ms,
    jit_ratio:         compile[c.name].nested.jit_ms / compile[c.name].flat.jit_ms,
    flat_plan_bytes:   compile[c.name].flat.plan_bytes,
    nested_plan_bytes: compile[c.name].nested.plan_bytes,
    plan_ratio:        compile[c.name].nested.plan_bytes / compile[c.name].flat.plan_bytes,
    flat_rt_pct:       f.ns_per_sample_min / samplePeriod,
    nested_rt_pct:     n.ns_per_sample_min / samplePeriod,
  }
  deltas.push(d)
  console.log(
    `${d.case.padEnd(30)}${d.ns_ratio.toFixed(3)}×  ${d.jit_ratio.toFixed(3)}×   ${d.plan_ratio.toFixed(3)}×`,
  )
}

// ── Persist ───────────────────────────────────────────────────────────
const resultFile = '/tmp/depth_bench.json'
writeFileSync(resultFile, JSON.stringify({
  config: {
    frame_size:       FRAME_SIZE,
    frames_per_trial: FRAMES_PER_TRIAL,
    runtime_trials:   RUNTIME_TRIALS,
    warmup_frames:    WARMUP_FRAMES,
    sample_rate:      SAMPLE_RATE,
    methodology_note: 'min-of-3 for runtime per runtime_noise_meta.ts; ' +
                      'single cold trial for compile (engine has process-singleton ' +
                      'in-memory cache; cannot measure multiple cold compiles in one process)',
  },
  compile,
  runtime,
  deltas,
  timestamp: new Date().toISOString(),
}, null, 2))
console.log(`\nResults written to ${resultFile}`)
