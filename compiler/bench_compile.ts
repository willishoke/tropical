/**
 * Quick benchmark: reproduce the 13-module patch compilation to find bottleneck.
 */
import { makeSession, SessionState, instantiate, outputNames } from './session.js'
import { loadStdlib as loadBuiltins } from './program.js'
import { compileSession } from './ir/compile_session'
import { toWirePlan } from './flat_plan.js'

const session: SessionState = makeSession()
loadBuiltins(session)

// Instantiate the same 13 modules from the user's patch
const modules: [string, string][] = [
  ['Clock', 'Clock1'],
  ['BassDrum', 'Kick1'],
  ['ADEnvelope', 'BassEnv'],
  ['VCO', 'BassVCO'],
  ['LadderFilter', 'BassFilter'],
  ['VCA', 'BassVCA'],
  ['Compressor', 'BassComp'],
  ['VCO', 'PadA'],
  ['VCO', 'PadB'],
  ['VCO', 'PadC'],
  ['VCO', 'PadD'],
  ['Reverb', 'PadReverb'],
  ['Compressor', 'PadComp'],
]

for (const [typeName, instanceName] of modules) {
  const type = session.typeRegistry.get(typeName)!
  const inst = instantiate(type, instanceName)
  session.instanceRegistry.set(instanceName, inst)
}

// Set the input that triggers wire()
session.inputExprNodes.set('Clock1:freq', 2.1667)

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
  + plan.instance_functions.reduce((n, i) => n + i.instructions.length, 0)
console.log(`compileSession: ${(t1 - t0).toFixed(1)}ms`)
console.log(`  instances: ${plan.instance_functions.length}`)
console.log(`  instructions: ${instrCount} (${plan.scheduler_function.preamble.length} preamble + bodies + ${plan.scheduler_function.postamble.length} postamble)`)
console.log(`  registers: ${plan.register_names.length}`)
console.log(`  array_slots: ${plan.array_slot_sizes.length} (sizes: ${plan.array_slot_sizes.join(', ')})`)
console.log(`  outputs: ${plan.scheduler_function.output_targets.length}`)

const t2 = performance.now()
const json = JSON.stringify(toWirePlan(plan))
const t3 = performance.now()
console.log(`JSON.stringify: ${(t3 - t2).toFixed(1)}ms (${(json.length / 1024).toFixed(0)}KB)`)
