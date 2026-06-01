/**
 * Quick benchmark: a 13-module patch compilation, to find the TS-pipeline
 * bottleneck (per-module and whole-session). Uses real stdlib types.
 */
import { makeSession, SessionState, instantiate, outputNames } from '../../compiler/session.js'
import { loadStdlib as loadBuiltins } from '../../compiler/program.js'
import { compileSession } from '../../compiler/ir/compile_session'
import { toWirePlan, type InstanceFunction } from '../../compiler/flat_plan.js'

const session: SessionState = makeSession()
loadBuiltins(session)

// 13 real (non-generic) stdlib modules — an oscillator/filter/effect mix.
const modules: [string, string][] = [
  ['SinOsc', 'Osc1'],
  ['BlepSaw', 'Osc2'],
  ['VCA', 'Amp1'],
  ['OnePole', 'LP1'],
  ['SVF', 'SVF1'],
  ['CrossFade', 'XFade1'],
  ['SoftClip', 'Clip1'],
  ['AllpassDelay', 'AP1'],
  ['CombDelay', 'Comb1'],
  ['EnvExpDecay', 'Env1'],
  ['SampleHold', 'SH1'],
  ['Tanh', 'Sat1'],
  ['WhiteNoise', 'Noise1'],
]

for (const [typeName, name] of modules) {
  const type = session.typeRegistry.get(typeName)!
  session.instanceRegistry.set(name, instantiate(type, name))
  // Tap each module's first output into the DAC mix so the full session
  // compile produces a non-degenerate plan.
  session.graphOutputs.push({ instance: name, output: outputNames(type)[0] })
}

/** Total instructions across a kernel and its nested children (the root
 *  path nests instance bodies as children of one root function). */
function countInstrs(fn: InstanceFunction): number {
  return fn.preamble_instructions.length + fn.instructions.length
    + fn.children.reduce((n, c) => n + c.pre_input_instructions.length + countInstrs(c), 0)
}

// Test individual modules to find which is slow
for (const [typeName, instanceName] of modules) {
  const solo = makeSession()
  loadBuiltins(solo)
  const type = solo.typeRegistry.get(typeName)!
  solo.instanceRegistry.set(instanceName, instantiate(type, instanceName))
  solo.graphOutputs.push({ instance: instanceName, output: outputNames(type)[0] })
  const t = performance.now()
  try {
    compileSession(solo)
    console.log(`  ${instanceName} (${typeName}): ${(performance.now() - t).toFixed(1)}ms`)
  } catch (e: any) {
    console.log(`  ${instanceName} (${typeName}): ERROR ${e.message} (${(performance.now() - t).toFixed(1)}ms)`)
  }
}

console.log(`\nModules: ${session.instanceRegistry.size}`)
console.log('Starting full compileSession...')

const t0 = performance.now()
const plan = compileSession(session)
const t1 = performance.now()
const instrCount =
    plan.scheduler_function.preamble.length
  + plan.scheduler_function.postamble.length
  + plan.instance_functions.reduce((n, i) => n + countInstrs(i), 0)
console.log(`compileSession: ${(t1 - t0).toFixed(1)}ms`)
console.log(`  instance_functions: ${plan.instance_functions.length} (root nests instances as children)`)
console.log(`  instructions: ${instrCount} (${plan.scheduler_function.preamble.length} preamble + bodies + ${plan.scheduler_function.postamble.length} postamble)`)
console.log(`  registers: ${plan.register_names.length}`)
console.log(`  array_slots: ${plan.array_slot_sizes.length} (sizes: ${plan.array_slot_sizes.join(', ')})`)
console.log(`  outputs: ${plan.scheduler_function.output_targets.length}`)

const t2 = performance.now()
const json = JSON.stringify(toWirePlan(plan))
const t3 = performance.now()
console.log(`JSON.stringify: ${(t3 - t2).toFixed(1)}ms (${(json.length / 1024).toFixed(0)}KB)`)
