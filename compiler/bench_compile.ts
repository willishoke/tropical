/**
 * Quick benchmark: reproduce the 13-module patch compilation to find bottleneck.
 */
import { makeSession, SessionState } from './patch.js'
import { loadStdlib as loadBuiltins } from './program.js'
import { flattenPatch } from './flatten.js'

const session: SessionState = makeSession()
loadBuiltins(session.typeRegistry)

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
  const inst = type.instantiateAs(instanceName)
  session.instanceRegistry.set(instanceName, inst)
}

// Set the input that triggers wire()
session.inputExprNodes.set('Clock1:freq', 2.1667)

// Test individual modules to find which is slow
for (const [typeName, instanceName] of modules) {
  const solo = makeSession()
  loadBuiltins(solo.typeRegistry)
  const type = solo.typeRegistry.get(typeName)!
  solo.instanceRegistry.set(instanceName, type.instantiateAs(instanceName))
  solo.graphOutputs.push({ module: instanceName, output: type._def.outputNames[0] })
  const t = performance.now()
  try {
    flattenPatch(solo)
    console.log(`  ${instanceName} (${typeName}): ${(performance.now() - t).toFixed(1)}ms`)
  } catch (e: any) {
    console.log(`  ${instanceName} (${typeName}): ERROR ${e.message} (${(performance.now() - t).toFixed(1)}ms)`)
  }
}

console.log(`\nModules: ${session.instanceRegistry.size}`)
console.log('Starting full flattenPatch...')

const t0 = performance.now()
const plan = flattenPatch(session)
const t1 = performance.now()
console.log(`flattenPatch: ${(t1 - t0).toFixed(1)}ms`)
console.log(`  instructions: ${plan.instructions.length}`)
console.log(`  registers: ${plan.register_targets.length}`)
console.log(`  array_slots: ${plan.array_slot_sizes.length} (sizes: ${plan.array_slot_sizes.join(', ')})`)
console.log(`  outputs: ${plan.output_targets.length}`)

const t2 = performance.now()
const json = JSON.stringify(plan)
const t3 = performance.now()
console.log(`JSON.stringify: ${(t3 - t2).toFixed(1)}ms (${(json.length / 1024).toFixed(0)}KB)`)
