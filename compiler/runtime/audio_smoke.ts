#!/usr/bin/env bun
/**
 * test_audio_flat.ts — Quick smoke test: play audio through FlatRuntime.
 *
 * Usage: bun run tui/src/test_audio_flat.ts
 *
 * Plays a VCO → VCA patch for 3 seconds, then stops.
 */

import { makeSession, loadJSON } from '../patch'
import { loadStdlib as loadBuiltins } from '../program'

const session = makeSession(512)
loadBuiltins(session.typeRegistry)
loadJSON({
  schema: 'tropical_program_1',
  name: 'smoke_test',
  instances: {
    VCO1: { program: 'VCO', inputs: { freq: 440 } },
    VCA1: { program: 'VCA', inputs: {
      audio: { op: 'ref', module: 'VCO1', output: 'saw' },
      cv: 0.3,
    }},
  },
  audio_outputs: [{ instance: 'VCA1', output: 'out' }],
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
