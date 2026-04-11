/**
 * test_patch.ts — standalone patch smoke-test, no audio device required.
 *
 * Usage:  bun run src/test_patch.ts <patch.json> [n_frames]
 *
 * Loads a patch, calls runtime.process() n_frames times,
 * and reports pass/fail. Exits non-zero on any thrown exception.
 */

import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { makeSession, loadJSON } from '../compiler/session.js'
import { loadStdlib as loadBuiltins } from '../compiler/program.js'
import * as b from '../compiler/runtime/bindings.js'

const patchArg = process.argv[2]
const nFrames  = parseInt(process.argv[3] ?? '128', 10)

if (!patchArg) {
  console.error('Usage: bun run src/test_patch.ts <patch.json> [n_frames]')
  process.exit(1)
}

const patchPath = resolve(patchArg)
console.log(`Loading patch: ${patchPath}`)
console.log(`Frames:        ${nFrames}`)

const json = JSON.parse(readFileSync(patchPath, 'utf-8'))

const session = makeSession(256)
loadBuiltins(session.typeRegistry)
loadJSON(json, session)

const runtime = session.runtime

console.log('Plan loaded. Processing frames...')

// Process frames and collect peak absolute value
let peak = 0
for (let i = 0; i < nFrames; i++) {
  b.tropical_runtime_process(runtime._h)
  const buf = runtime.outputBuffer
  for (let s = 0; s < buf.length; s++) {
    const abs = Math.abs(buf[s])
    if (abs > peak) peak = abs
  }
}

console.log(`Processed ${nFrames} frames.`)
console.log(`Peak output level: ${peak.toFixed(6)}`)

if (peak > 0 && peak < 100) {
  console.log('PASS — non-zero, non-exploding output.')
  process.exit(0)
} else if (peak === 0) {
  console.log('WARN — zero output (may be expected for silent patches).')
  process.exit(0)
} else {
  console.log('FAIL — output out of range.')
  process.exit(1)
}
