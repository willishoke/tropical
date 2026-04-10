#!/usr/bin/env bun
/**
 * test_audio_flat.ts — Quick smoke test: play audio through FlatRuntime.
 *
 * Usage: bun run tui/src/test_audio_flat.ts
 *
 * Plays a VCO → VCA patch for 3 seconds, then stops.
 */

import { makeSession, loadPatchFromJSON } from '../patch'
import { loadStdlib as loadBuiltins } from '../program'
import { applyFlatPlan } from '../apply_plan'

const session = makeSession(512)
loadBuiltins(session.typeRegistry)
loadPatchFromJSON({
  schema: 'tropical_patch_1',
  modules: [
    { type: 'VCO', name: 'VCO1' },
    { type: 'VCA', name: 'VCA1' },
  ],
  input_exprs: [
    { module: 'VCO1', input: 'freq', expr: 440 },
    { module: 'VCA1', input: 'audio', expr: { op: 'ref', module: 'VCO1', output: 'saw' } },
    { module: 'VCA1', input: 'cv', expr: 0.3 },
  ],
  outputs: [{ module: 'VCA1', output: 'out' }],
}, session)

console.log('Starting DAC (FlatRuntime)...')
const dac = session.runtime.createDAC()
dac.start()
console.log('Playing 440 Hz saw → VCA for 3 seconds...')

setTimeout(() => {
  console.log('Stopping...')
  dac.stop()
  dac.dispose()
  session.runtime.dispose()
  console.log('Done.')
  process.exit(0)
}, 3000)
