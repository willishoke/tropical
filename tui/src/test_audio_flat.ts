#!/usr/bin/env bun
/**
 * test_audio_flat.ts — Quick smoke test: play audio through FlatRuntime.
 *
 * Usage: bun run tui/src/test_audio_flat.ts
 *
 * Plays a VCO → VCA patch for 3 seconds, then stops.
 */

import { makeSession, loadPatchFromJSON } from './patch'
import { loadBuiltins } from './module_library'
import { applyFlatPlan } from './apply_plan'
import { Runtime } from './runtime'

const session = makeSession(512)
loadBuiltins(session.typeRegistry)
loadPatchFromJSON({
  schema: 'egress_patch_1',
  modules: [
    { type: 'VCO', name: 'VCO1' },
    { type: 'VCA', name: 'VCA1' },
  ],
}, session)

session.inputExprNodes.set('VCO1:freq', 440)
session.inputExprNodes.set('VCA1:audio', { op: 'ref', module: 'VCO1', output: 'saw' })
session.inputExprNodes.set('VCA1:cv', 0.3)
session.graphOutputs.push({ module: 'VCA1', output: 'out' })

const rt = new Runtime(512)
applyFlatPlan(session, rt)

console.log('Starting DAC (FlatRuntime)...')
const dac = rt.createDAC()
dac.start()
console.log('Playing 440 Hz saw → VCA for 3 seconds...')

setTimeout(() => {
  console.log('Stopping...')
  dac.stop()
  dac.dispose()
  rt.dispose()
  session.graph.dispose()
  console.log('Done.')
  process.exit(0)
}, 3000)
