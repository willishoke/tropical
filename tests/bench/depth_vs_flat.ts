/**
 * depth_vs_flat.ts — head-to-head bench: legacy inline-everything path
 * (`inlineNested:true`, the flat per-instance kernel) vs the M11
 * fractal slot path (`inlineNested:false`, every sub-instance is its
 * own kernel boundary with per-port input/output slots).
 *
 * Mirror of `microkernel_vs_fused.ts`'s structure. For each case,
 * compile + run twice — once per mode — and report:
 *   - ts_ms        TS pipeline time (parse, strata, partition)
 *   - jit_ms       LLVM IR generation + native compile
 *   - ns_per_sample  audio-thread cost
 *   - rt_ratio     fraction of a 44.1kHz sample period (lower = more headroom)
 *
 * The slot path's overhead comes from:
 *   - Extra WriteSlot/ReadSlot at every kernel boundary (LLVM should
 *     fold the store-load round trip in fused mode, but there's still
 *     a register pressure cost)
 *   - More LLVM instructions in the IR (per-child wire blocks aren't
 *     inlined into the parent body)
 *   - More slot-array accesses (parent writes child input slot, child
 *     reads parent-allocated output slot)
 *
 * Counterbalanced (potentially) by:
 *   - Less duplicated codegen — sibling instances of the same type
 *     share an LLVM function in microkernel mode (no-op in fused mode)
 *   - Smaller per-kernel instruction streams may compile faster
 *
 * Usage: bun run tests/bench/depth_vs_flat.ts [--frames=N] [--keep-cache]
 *
 * Output: /tmp/depth_bench.json (consumed by the Phase 6 snapshot doc)
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

type DepthMode = 'flat' | 'nested'

interface CaseResult {
  case:           string
  instances:      number
  mode:           DepthMode
  ts_ms:          number
  stringify_ms:   number
  jit_ms:         number
  ns_per_sample:  number
  rt_ratio:       number
  plan_bytes:     number
}

const results: CaseResult[] = []

// ── Default port values for stdlib programs that need them ────────────
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

// ──────────────────────────────────────────────────────────────────────
// Session builders
// ──────────────────────────────────────────────────────────────────────

// Single stdlib instance with default inputs.
function singleInstanceFactory(typeName: string, typeArgs?: Record<string, number>) {
  return (mode: DepthMode) => {
    const session = makeSession(FRAME_SIZE, { inlineNested: mode === 'flat' })
    loadStdlib(session)
    const { type, typeArgs: resolved } = resolveProgramType(session, typeName, typeArgs, undefined)
    const inst = instantiate(type, 'inst', { baseTypeName: typeName, typeArgs: resolved })
    session.instanceRegistry.set('inst', inst)
    for (const pn of inputNames(inst)) {
      if (pn in DEFAULT_INPUTS) {
        session.inputExprNodes.set(wk('inst', pn), DEFAULT_INPUTS[pn])
      }
    }
    session.graphOutputs.push({ instance: 'inst', output: outputNames(inst)[0] })
    return session
  }
}

// Polyphony of stdlib instances with freq spread (when applicable).
function polyphonyFactory(typeName: string, voiceCount: number, typeArgs?: Record<string, number>) {
  return (mode: DepthMode) => {
    const session = makeSession(FRAME_SIZE, { inlineNested: mode === 'flat' })
    loadStdlib(session)
    const { type, typeArgs: resolved } = resolveProgramType(session, typeName, typeArgs, undefined)
    for (let i = 0; i < voiceCount; i++) {
      const name = `v${i}`
      const inst = instantiate(type, name, { baseTypeName: typeName, typeArgs: resolved })
      session.instanceRegistry.set(name, inst)
      for (const pn of inputNames(inst)) {
        if (pn === 'freq') {
          session.inputExprNodes.set(wk(name, pn), 110 + 22 * i)
        } else if (pn in DEFAULT_INPUTS) {
          session.inputExprNodes.set(wk(name, pn), DEFAULT_INPUTS[pn])
        }
      }
      session.graphOutputs.push({ instance: name, output: outputNames(inst)[0] })
    }
    return session
  }
}

interface BenchCase {
  name:    string
  factory: (mode: DepthMode) => ReturnType<typeof makeSession>
}

// Cases ordered roughly from simple to stress. Phaser16 polyphony is
// where the slot path's overhead (if any) is most visible — 8 voices ×
// 17 kernels = 136 instance boundaries in nested mode, vs 8 monolithic
// flat kernels.
const CASES: BenchCase[] = [
  // ── Single instance — leaf (no children); slot path is a no-op
  { name: 'Sin',                      factory: singleInstanceFactory('Sin') },
  // ── Single instance — depth-1 children (2-4 sub-instances)
  { name: 'OnePole',                  factory: singleInstanceFactory('OnePole') },
  { name: 'Pow',                      factory: singleInstanceFactory('Pow') },
  { name: 'SVF',                      factory: singleInstanceFactory('SVF') },
  { name: 'LadderFilter',             factory: singleInstanceFactory('LadderFilter') },
  // ── Single instance — depth-2 (the honest fractal case)
  { name: 'Phaser16',                 factory: singleInstanceFactory('Phaser16') },
  // ── Single instance — depth-3 (Bubble has 5 children, BubbleCloud has 8 Bubbles)
  { name: 'Bubble',                   factory: singleInstanceFactory('Bubble') },
  { name: 'BubbleCloud',              factory: singleInstanceFactory('BubbleCloud') },
  // ── Polyphony — N voices of a leaf type (slot path no-op, baseline)
  { name: 'polyphony_8x_Sin',         factory: polyphonyFactory('Sin', 8) },
  { name: 'polyphony_32x_Sin',        factory: polyphonyFactory('Sin', 32) },
  // ── Polyphony — N voices of a nested type (slot path scales here)
  { name: 'polyphony_8x_SinOsc',      factory: polyphonyFactory('SinOsc', 8) },
  { name: 'polyphony_8x_OnePole',     factory: polyphonyFactory('OnePole', 8) },
  { name: 'polyphony_8x_SVF',         factory: polyphonyFactory('SVF', 8) },
  { name: 'polyphony_8x_LadderFilter', factory: polyphonyFactory('LadderFilter', 8) },
  { name: 'polyphony_4x_Phaser16',    factory: polyphonyFactory('Phaser16', 4) },
  { name: 'polyphony_8x_Phaser16',    factory: polyphonyFactory('Phaser16', 8) },
]

// ──────────────────────────────────────────────────────────────────────
// Bench one case in one mode
// ──────────────────────────────────────────────────────────────────────

function benchOne(name: string, factory: BenchCase['factory'], mode: DepthMode): CaseResult {
  const session = factory(mode)

  const t1 = performance.now()
  const plan = compileSession(session)
  const tsMs = performance.now() - t1

  const t2 = performance.now()
  const planJson = JSON.stringify(toWirePlan(plan))
  const stringifyMs = performance.now() - t2

  const t3 = performance.now()
  session.runtime.loadPlan(planJson)
  const jitMs = performance.now() - t3

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
    case:          name,
    instances:     plan.instance_functions.length,
    mode,
    ts_ms:         tsMs,
    stringify_ms:  stringifyMs,
    jit_ms:        jitMs,
    ns_per_sample: nsPerSample,
    rt_ratio:      nsPerSample / samplePeriod,
    plan_bytes:    planJson.length,
  }
}

// ──────────────────────────────────────────────────────────────────────
// Run
// ──────────────────────────────────────────────────────────────────────

const COLS = [
  'case', 'instances', 'mode',
  'ts_ms', 'jit_ms', 'plan_kb',
  'ns/sample', 'rt_ratio',
]
console.log(COLS.join('\t'))

for (const c of CASES) {
  for (const mode of ['flat', 'nested'] as const) {
    try {
      const r = benchOne(c.name, c.factory, mode)
      results.push(r)
      console.log(
        [
          r.case,
          r.instances,
          r.mode,
          r.ts_ms.toFixed(1),
          r.jit_ms.toFixed(1),
          (r.plan_bytes / 1024).toFixed(1),
          r.ns_per_sample.toFixed(1),
          `${(r.rt_ratio * 100).toFixed(2)}%`,
        ].join('\t'),
      )
    } catch (e: any) {
      console.log(`${c.name}\t-\t${mode}\tERR\t${e.message.split('\n')[0].slice(0, 200)}`)
    }
  }
}

// ──────────────────────────────────────────────────────────────────────
// Mode deltas — the bottom line for the deletion decision
// ──────────────────────────────────────────────────────────────────────

interface Delta {
  case:           string
  instances:      number
  flat_ns:        number
  nested_ns:      number
  slowdown:       number   // nested / flat; >1 = nested slower
  flat_jit_ms:    number
  nested_jit_ms:  number
  jit_ratio:      number   // nested_jit / flat_jit
  plan_size_ratio: number  // nested_bytes / flat_bytes
}

const deltas: Delta[] = []
for (const c of CASES) {
  const f = results.find(r => r.case === c.name && r.mode === 'flat')
  const n = results.find(r => r.case === c.name && r.mode === 'nested')
  if (f && n) {
    deltas.push({
      case:            c.name,
      instances:       n.instances,  // nested has more (every sub-instance is its own)
      flat_ns:         f.ns_per_sample,
      nested_ns:       n.ns_per_sample,
      slowdown:        n.ns_per_sample / f.ns_per_sample,
      flat_jit_ms:     f.jit_ms,
      nested_jit_ms:   n.jit_ms,
      jit_ratio:       n.jit_ms / f.jit_ms,
      plan_size_ratio: n.plan_bytes / f.plan_bytes,
    })
  }
}

console.log('\nDeltas (nested / flat):')
console.log('  case                            flat_inst  nested_inst  ns_ratio  jit_ratio  plan_ratio')
for (const d of deltas) {
  const f = results.find(r => r.case === d.case && r.mode === 'flat')!
  console.log(
    `  ${d.case.padEnd(30)} ` +
    `${String(f.instances).padStart(5)}      ` +
    `${String(d.instances).padStart(5)}        ` +
    `${d.slowdown.toFixed(2)}×    ` +
    `${d.jit_ratio.toFixed(2)}×     ` +
    `${d.plan_size_ratio.toFixed(2)}×`,
  )
}

const resultFile = '/tmp/depth_bench.json'
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
