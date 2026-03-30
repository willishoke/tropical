/**
 * test_patch.ts — standalone patch smoke-test, no audio device required.
 *
 * Usage:  tsx src/test_patch.ts <patch.json> [n_frames]
 *
 * Loads a patch, primes the JIT, calls graph.process() n_frames times,
 * and reports pass/fail.  Exits non-zero on any thrown exception.
 */

import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import * as b from './bindings.js'
import { Graph } from './graph.js'
import { makeSession, loadPatchFromJSON } from './patch.js'
import { loadBuiltins } from './module_library.js'
import type { PatchJSON } from './patch.js'

const patchArg = process.argv[2]
const nFrames  = parseInt(process.argv[3] ?? '128', 10)

if (!patchArg) {
  console.error('Usage: tsx src/test_patch.ts <patch.json> [n_frames]')
  process.exit(1)
}

const patchPath = resolve(patchArg)
console.log(`Loading patch: ${patchPath}`)
console.log(`Frames:        ${nFrames}`)

const json: PatchJSON = JSON.parse(readFileSync(patchPath, 'utf-8'))

const session = makeSession(256)
loadBuiltins(session.typeRegistry)
loadPatchFromJSON(json, session)

const graph: Graph = session.graph

// Prime JIT (compiles kernels without needing audio hardware)
b.egress_graph_prime_jit(graph._h)
console.log('JIT primed.')

// Process frames and collect peak absolute value
let peak = 0
for (let i = 0; i < nFrames; i++) {
  b.egress_graph_process(graph._h)
  const buf = graph.outputBuffer
  for (let s = 0; s < buf.length; s++) {
    const abs = Math.abs(buf[s])
    if (abs > peak) peak = abs
  }
}

console.log(`Peak output: ${peak}`)

if (peak === 0) {
  console.error('FAIL — output is silent (all samples are zero)')
  process.exit(1)
}

console.log(`PASS — processed ${nFrames} frames, peak=${peak.toFixed(6)}`)
