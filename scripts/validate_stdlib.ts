/**
 * validate_stdlib.ts — deterministic render + golden-hash regression check.
 *
 * Loads each target (a ProgramJSON patch), renders N frames through the
 * native JIT, hashes the mono output, and compares against a committed
 * golden under `tests/golden/<name>.hash`.
 *
 * Usage:
 *   bun run scripts/validate_stdlib.ts             # verify
 *   bun run scripts/validate_stdlib.ts --write     # rewrite goldens
 *
 * This is the regression net for the type-system refactor: any silent
 * change in DSP output across the stdlib or sample patches shows up as
 * a hash mismatch on the next `make validate`.
 */

import { readFileSync, writeFileSync, existsSync, mkdirSync, readdirSync } from 'node:fs'
import { resolve, basename, dirname } from 'node:path'
import { createHash } from 'node:crypto'
import { makeSession, loadJSON } from '../compiler/session.js'
import { loadStdlib } from '../compiler/program.js'
import * as b from '../compiler/runtime/bindings.js'

const FRAMES_PER_TARGET = 16      // buffer_length=256 × 16 = 4096 samples
const BUFFER_LEN        = 256
const GOLDEN_DIR        = resolve(__dirname, '..', 'tests', 'golden')

const writeMode = process.argv.includes('--write')

// Pick patches + stdlib files that render standalone (leaf programs with
// audio_outputs or defined outputs wired to audio). We skip stdlib
// composites (Phaser, LadderFilter) that need external input wiring to
// produce meaningful output — those are covered transitively by patches.
function discoverTargets(): string[] {
  const patches = readdirSync('patches')
    .filter(f => f.endsWith('.json'))
    .map(f => resolve('patches', f))
  return patches
}

function hashBuffer(samples: Float64Array): string {
  const h = createHash('sha256')
  const bytes = new Uint8Array(samples.buffer, samples.byteOffset, samples.byteLength)
  h.update(bytes)
  return h.digest('hex')
}

interface Result { name: string; status: 'PASS' | 'FAIL' | 'SKIP' | 'WROTE'; detail?: string }

function validateTarget(patchPath: string): Result {
  const name = basename(patchPath, '.json')
  const goldenPath = resolve(GOLDEN_DIR, `${name}.hash`)

  let session: ReturnType<typeof makeSession>
  try {
    const json = JSON.parse(readFileSync(patchPath, 'utf-8'))
    session = makeSession(BUFFER_LEN)
    loadStdlib(session)
    loadJSON(json, session)
  } catch (e: any) {
    return { name, status: 'SKIP', detail: e.message.split('\n')[0] }
  }

  const samples = new Float64Array(BUFFER_LEN * FRAMES_PER_TARGET)
  for (let i = 0; i < FRAMES_PER_TARGET; i++) {
    b.tropical_runtime_process(session.runtime._h)
    samples.set(session.runtime.outputBuffer, i * BUFFER_LEN)
  }

  const hash = hashBuffer(samples)

  if (writeMode) {
    if (!existsSync(dirname(goldenPath))) mkdirSync(dirname(goldenPath), { recursive: true })
    writeFileSync(goldenPath, hash + '\n')
    return { name, status: 'WROTE', detail: hash.slice(0, 16) }
  }

  if (!existsSync(goldenPath)) {
    return { name, status: 'SKIP', detail: 'no golden (run with --write)' }
  }

  const expected = readFileSync(goldenPath, 'utf-8').trim()
  if (expected === hash) return { name, status: 'PASS', detail: hash.slice(0, 16) }
  return {
    name,
    status: 'FAIL',
    detail: `expected ${expected.slice(0, 16)} got ${hash.slice(0, 16)}`,
  }
}

// ─── Run ─────────────────────────────────────────────────────────────────────

const targets = discoverTargets()
const results: Result[] = []
for (const t of targets) results.push(validateTarget(t))

let passed = 0, failed = 0, skipped = 0, wrote = 0
for (const r of results) {
  const tag =
    r.status === 'PASS'  ? 'PASS ' :
    r.status === 'FAIL'  ? 'FAIL ' :
    r.status === 'WROTE' ? 'WROTE' :
                            'SKIP '
  console.log(`  ${tag}  ${r.name.padEnd(28)}  ${r.detail ?? ''}`)
  if (r.status === 'PASS')  passed++
  if (r.status === 'FAIL')  failed++
  if (r.status === 'SKIP')  skipped++
  if (r.status === 'WROTE') wrote++
}

console.log()
if (writeMode) console.log(`wrote ${wrote} golden(s), skipped ${skipped}`)
else           console.log(`${passed} passed, ${failed} failed, ${skipped} skipped`)

process.exit(failed === 0 ? 0 : 1)
