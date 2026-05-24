/**
 * runtime_noise_meta.ts — one-shot characterization of audio-thread
 * runtime noise.
 *
 * Picks one realistic stress case (polyphony_8x_Phaser16 in nested
 * mode — the case where small relative deltas drive the deletion
 * decision), compiles once, then runs N=100 trials of the audio
 * loop and reports the empirical distribution of ns/sample.
 *
 * The point isn't to bench Phaser16 — it's to figure out how many
 * trials a real bench needs to estimate the asymptotic min to a
 * given confidence. Bootstrap analysis below: for each candidate
 * trial count N ∈ {1, 3, 5, 10, 20, 30}, draw B=1000 independent
 * N-trial subsamples from the 100-trial population, take the min of
 * each, and report how close that min lands to the 100-trial min.
 *
 * Output: /tmp/runtime_noise_meta.json + console summary.
 *
 * Usage: bun run tests/bench/runtime_noise_meta.ts
 *
 * Wall clock: ~10s (100 × ~6s of audio per trial).
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
const FRAMES_PER_TRIAL = 1024            // ~6s of audio per trial at 44.1kHz
const TRIALS           = 100
const WARMUP_FRAMES    = 32
const SAMPLE_RATE      = 44100
const BOOTSTRAP_B      = 1000
const CANDIDATE_NS     = [1, 2, 3, 5, 10, 20, 30]

// ── Clear cache (cold start) ──────────────────────────────────────────
const cacheDir = process.env.XDG_CACHE_HOME
  ? join(process.env.XDG_CACHE_HOME, 'tropical', 'kernels')
  : join(homedir(), '.cache', 'tropical', 'kernels')
rmSync(cacheDir, { recursive: true, force: true })

// ── Build the session (polyphony_8x_Phaser16, nested mode) ────────────
const wk = (i: string, p: string) => wireKey(portRef(instanceName(i), portName(p)))
const session = makeSession(FRAME_SIZE, { inlineNested: false })
loadStdlib(session)
const { type, typeArgs } = resolveProgramType(session, 'Phaser16', undefined, undefined)
for (let i = 0; i < 8; i++) {
  const name = `v${i}`
  const inst = instantiate(type, name, { baseTypeName: 'Phaser16', typeArgs })
  session.instanceRegistry.set(name, inst)
  for (const pn of inputNames(inst)) {
    if (pn === 'freq')  session.inputExprNodes.set(wk(name, pn), 110 + 22 * i)
    if (pn === 'input') session.inputExprNodes.set(wk(name, pn), 0.5)
    if (pn === 'feedback') session.inputExprNodes.set(wk(name, pn), 0.4)
    if (pn === 'lfo_speed') session.inputExprNodes.set(wk(name, pn), 0.2)
  }
  session.graphOutputs.push({ instance: name, output: outputNames(inst)[0] })
}

// ── Compile once ──────────────────────────────────────────────────────
const plan = compileSession(session)
const planJson = JSON.stringify(toWirePlan(plan))
session.runtime.loadPlan(planJson)

// ── Warmup ────────────────────────────────────────────────────────────
for (let i = 0; i < WARMUP_FRAMES; i++)
  b.tropical_runtime_process(session.runtime._h)

// ── Run trials ────────────────────────────────────────────────────────
const samples: number[] = []   // ns/sample per trial
const samplesPerTrial = FRAMES_PER_TRIAL * FRAME_SIZE

console.log(`Running ${TRIALS} trials of ${FRAMES_PER_TRIAL} frames (${samplesPerTrial} samples each)...`)
for (let t = 0; t < TRIALS; t++) {
  const t0 = performance.now()
  for (let f = 0; f < FRAMES_PER_TRIAL; f++)
    b.tropical_runtime_process(session.runtime._h)
  const elapsed = performance.now() - t0
  const ns = (elapsed * 1e6) / samplesPerTrial
  samples.push(ns)
  if ((t + 1) % 20 === 0) {
    process.stdout.write(`  ${t + 1}/${TRIALS} done\n`)
  }
}

// ── Descriptive statistics ────────────────────────────────────────────
function stats(xs: number[]) {
  const sorted = [...xs].sort((a, b) => a - b)
  const n = sorted.length
  const sum = sorted.reduce((s, x) => s + x, 0)
  const mean = sum / n
  const variance = sorted.reduce((s, x) => s + (x - mean) * (x - mean), 0) / n
  const stddev = Math.sqrt(variance)
  const pick = (p: number) => sorted[Math.min(n - 1, Math.floor(p * n))]
  return {
    n,
    min:    sorted[0],
    p10:    pick(0.10),
    p50:    pick(0.50),  // median
    mean,
    p90:    pick(0.90),
    p99:    pick(0.99),
    max:    sorted[n - 1],
    stddev,
    cv:     stddev / mean,  // coefficient of variation
  }
}

const s = stats(samples)
const samplePeriod = 1e9 / SAMPLE_RATE

console.log('\nFull distribution (ns/sample):')
console.log(`  n:      ${s.n}`)
console.log(`  min:    ${s.min.toFixed(2)}`)
console.log(`  p10:    ${s.p10.toFixed(2)}`)
console.log(`  median: ${s.p50.toFixed(2)}`)
console.log(`  mean:   ${s.mean.toFixed(2)}`)
console.log(`  p90:    ${s.p90.toFixed(2)}`)
console.log(`  p99:    ${s.p99.toFixed(2)}`)
console.log(`  max:    ${s.max.toFixed(2)}`)
console.log(`  stddev: ${s.stddev.toFixed(2)}`)
console.log(`  CV:     ${(s.cv * 100).toFixed(2)}%`)
console.log(`  rt%:    ${(s.min / samplePeriod * 100).toFixed(2)}% of sample period (best trial)`)

// ── Bootstrap: how does min-of-N converge to min-of-100? ──────────────
// For each candidate N, draw B independent N-element subsamples (with
// replacement from the 100-trial population), take min of each, and
// report ratio to the population min. The distribution of
// (subsample_min / population_min) tells us how stable the min
// statistic is as a function of N.
//
// We report:
//   - median ratio: the typical convergence
//   - p95 ratio:    the "you almost always do at least this well"
//                   (95% of N-trial runs land within this ratio of true min)
//   - max ratio:    worst-case in B draws
function bootstrap(samples: number[], N: number, B: number): { median: number, p95: number, max: number } {
  const popMin = Math.min(...samples)
  const ratios: number[] = []
  for (let b = 0; b < B; b++) {
    let subMin = Infinity
    for (let i = 0; i < N; i++) {
      const idx = Math.floor(Math.random() * samples.length)
      if (samples[idx] < subMin) subMin = samples[idx]
    }
    ratios.push(subMin / popMin)
  }
  ratios.sort((a, b) => a - b)
  const pick = (p: number) => ratios[Math.min(ratios.length - 1, Math.floor(p * ratios.length))]
  return {
    median: pick(0.50),
    p95:    pick(0.95),
    max:    ratios[ratios.length - 1],
  }
}

console.log('\nBootstrap: min-of-N convergence to min-of-100')
console.log('  (ratio = N-trial min / 100-trial min; closer to 1.00 = tighter convergence)')
console.log('  N      median   p95     max')
const bootstraps: Record<number, ReturnType<typeof bootstrap>> = {}
for (const N of CANDIDATE_NS) {
  const r = bootstrap(samples, N, BOOTSTRAP_B)
  bootstraps[N] = r
  console.log(
    `  ${String(N).padStart(3)}    ` +
    `${r.median.toFixed(4)}  ${r.p95.toFixed(4)}  ${r.max.toFixed(4)}`,
  )
}

// Recommended N: smallest such that p95 ratio ≤ 1.01 (95% of runs land
// within 1% of true min) AND median ≤ 1.005 (typical run within 0.5%).
const recommendedN = CANDIDATE_NS.find(N =>
  bootstraps[N].p95 <= 1.01 && bootstraps[N].median <= 1.005,
)
console.log(`\nRecommended N for ±1% confidence at p95: ${recommendedN ?? '> 30 (noise floor too high)'}`)

// ── Persist ───────────────────────────────────────────────────────────
const resultFile = '/tmp/runtime_noise_meta.json'
writeFileSync(resultFile, JSON.stringify({
  config: {
    case: 'polyphony_8x_Phaser16',
    mode: 'nested',
    frame_size: FRAME_SIZE,
    frames_per_trial: FRAMES_PER_TRIAL,
    samples_per_trial: samplesPerTrial,
    trials: TRIALS,
    warmup_frames: WARMUP_FRAMES,
    bootstrap_b: BOOTSTRAP_B,
  },
  samples,
  stats: s,
  bootstraps,
  recommended_n: recommendedN,
  timestamp: new Date().toISOString(),
}, null, 2))
console.log(`\nResults written to ${resultFile}`)
