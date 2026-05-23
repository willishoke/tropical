/**
 * microkernel_deep_probe.ts — measure the cost of "all the way down"
 * un-inlining at the M11 sub-instance level.
 *
 * The strata pipeline currently inlines all sub-instance nesting before
 * partition, which means stdlib programs reach the engine with empty
 * `children` arrays on every InstanceFunction. The "deep" microkernel
 * mode (TROPICAL_KEEP_NESTED=1) emits one LLVM function per node in
 * the InstanceFunction tree — but if children are empty, deep == shallow.
 *
 * To probe the cost without unblocking the deferred ancestor-resolution
 * work referenced in compiler/ir/strata.ts, this script SYNTHESIZES a
 * non-empty children tree: it takes a polyphony plan (N top-level
 * SinOsc voices) and rewraps it as ONE top-level synthetic parent with
 * those N voices as its children. Wiring is unchanged; only the tree
 * shape differs.
 *
 * Then bench three configurations on the same synthesized plan:
 *   - fused                 (one monolithic kernel; parent + N children
 *                            all inlined into one function)
 *   - microkernel (shallow) (one fn for the parent, children inlined
 *                            into that fn via recursive emit)
 *   - microkernel (deep)    (one fn per node: 1 parent fn + N child
 *                            fns, dispatched in post-order)
 *
 * The deep-vs-shallow delta is the "go one level finer" cost — exactly
 * the question the user asked.
 *
 * Run:
 *   bun run tests/bench/microkernel_deep_probe.ts          (no deep)
 *   TROPICAL_KEEP_NESTED=1 bun run tests/bench/microkernel_deep_probe.ts
 */
import { rmSync } from 'node:fs'
import { join } from 'node:path'
import { homedir } from 'node:os'
import { makeSession, resolveProgramType, instantiate, outputNames } from '../../compiler/session.js'
import { loadStdlib as loadBuiltins } from '../../compiler/program.js'
import { compileSession } from '../../compiler/ir/compile_session.js'
import { toWirePlan, type WireFlatPlan, type CompilationMode, type FlatPlan } from '../../compiler/flat_plan.js'
import { wireKey, portRef, instanceName, portName } from '../../compiler/ir/branded_names.js'
import * as b from '../../compiler/runtime/bindings.js'

const FRAME_SIZE    = 256
const BENCH_FRAMES  = 4096
const WARMUP_FRAMES = 32
const SAMPLE_RATE   = 44100

// Wipe disk cache so JIT compile timings are honest.
const cacheDir = process.env.XDG_CACHE_HOME
  ? join(process.env.XDG_CACHE_HOME, 'tropical', 'kernels')
  : join(homedir(), '.cache', 'tropical', 'kernels')
rmSync(cacheDir, { recursive: true, force: true })

const deepMode = process.env.TROPICAL_MK_DEEP === '1'
console.log(`TROPICAL_MK_DEEP=${deepMode ? '1' : '(unset)'}`)
console.log(`(cold) cleared ${cacheDir}`)

// ── Build a polyphony session, compile normally, then rewrap as a tree ──

const VOICE_COUNT = 16

function buildPolyphonySession() {
  const session = makeSession(FRAME_SIZE)
  loadBuiltins(session)
  const { type, typeArgs } = resolveProgramType(session, 'SinOsc', undefined, undefined)
  for (let i = 0; i < VOICE_COUNT; i++) {
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

// Synthesize the nested-children plan: one synthetic parent whose body
// is empty, with the N voices as its children. The unified offsets
// (register_offset, state_reg_offset, array_slot_offset) are unchanged
// — the children are addressed in the same unified slot space, so
// rewrapping just changes the tree shape.
function rewrapAsTree(plan: FlatPlan): FlatPlan {
  return {
    ...plan,
    instance_functions: [{
      name:              'synthetic_root',
      instance_name:     'synthetic_root',
      instructions:      [],
      register_offset:   plan.instance_functions[0]?.register_offset ?? (0 as never),
      state_reg_offset:  plan.instance_functions[0]?.state_reg_offset ?? (0 as never),
      array_slot_offset: plan.instance_functions[0]?.array_slot_offset ?? (0 as never),
      register_count:    0,
      register_targets:  [],
      children:          plan.instance_functions,
    }],
  }
}

function nodeCount(plan: FlatPlan): number {
  function walk(arr: FlatPlan['instance_functions']): number {
    let n = 0
    for (const f of arr) { n += 1 + walk(f.children) }
    return n
  }
  return walk(plan.instance_functions)
}

// ── Per-mode timing ──

interface Result {
  label:          string
  ns_per_sample:  number
  jit_ms:         number
  fn_emitted:     number   // expected number of LLVM functions (informational)
}

function bench(label: string, mode: CompilationMode, planFn: () => FlatPlan): Result {
  const session = buildPolyphonySession()
  const plan = planFn()
  // Patch the compilation_mode without recompiling the session graph.
  const patched: FlatPlan = { ...plan, compilation_mode: mode }
  const wire: WireFlatPlan = toWirePlan(patched)
  const json = JSON.stringify(wire)

  const t0 = performance.now()
  session.runtime.loadPlan(json)
  const jit_ms = performance.now() - t0

  for (let i = 0; i < WARMUP_FRAMES; i++) b.tropical_runtime_process(session.runtime._h)
  const t1 = performance.now()
  for (let i = 0; i < BENCH_FRAMES; i++) b.tropical_runtime_process(session.runtime._h)
  const proc_ms = performance.now() - t1

  const ns_per_sample = (proc_ms * 1e6) / (BENCH_FRAMES * FRAME_SIZE)
  const fn_emitted = nodeCount(patched)
  return { label, ns_per_sample, jit_ms, fn_emitted }
}

// Build both plan shapes up front (same TS compile, different rewrap).
const session = buildPolyphonySession()
const flatPlan  = compileSession(session)             // 16 top-level instances
const treePlan  = rewrapAsTree(flatPlan)              //  1 parent + 16 children

console.log(`\nflat plan:  ${flatPlan.instance_functions.length} top-level, ${nodeCount(flatPlan)} total nodes`)
console.log(`tree plan:  ${treePlan.instance_functions.length} top-level, ${nodeCount(treePlan)} total nodes`)
console.log()

// ── Run ──
console.log('mode\t\t\t\tns/sample\tjit_ms\tfns')

const results: Result[] = [
  bench('fused (flat plan)',              'fused',       () => flatPlan),
  bench('microkernel shallow (flat)',     'microkernel', () => flatPlan),
  bench('fused (tree plan)',              'fused',       () => treePlan),
  bench('microkernel shallow (tree)',     'microkernel', () => treePlan),
]

if (deepMode) {
  results.push(bench('microkernel deep (tree)',  'microkernel', () => treePlan))
}

for (const r of results) {
  console.log(
    `${r.label.padEnd(30)}\t${r.ns_per_sample.toFixed(1)}\t\t${r.jit_ms.toFixed(0)}\t${r.fn_emitted}`,
  )
}

console.log()
const baseline   = results.find(r => r.label === 'microkernel shallow (flat)')
const deep       = results.find(r => r.label === 'microkernel deep (tree)')
const treeShallow = results.find(r => r.label === 'microkernel shallow (tree)')
if (baseline && deep) {
  console.log(`shallow (flat, ${VOICE_COUNT} top-level): ${baseline.ns_per_sample.toFixed(1)} ns/sample`)
  console.log(`deep    (tree, 1 parent + ${VOICE_COUNT} children): ${deep.ns_per_sample.toFixed(1)} ns/sample`)
  console.log(`going one level deeper: ${(deep.ns_per_sample / baseline.ns_per_sample).toFixed(2)}× cost`)
}
if (baseline && treeShallow) {
  console.log(`tree shape penalty (shallow alone): ${(treeShallow.ns_per_sample / baseline.ns_per_sample).toFixed(2)}×`)
}
