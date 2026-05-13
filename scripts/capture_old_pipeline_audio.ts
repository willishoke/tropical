/**
 * capture_old_pipeline_audio.ts — capture the OLD pipeline's audio
 * output for every fixture in `tests/fixtures/flat_plan/`.
 *
 * Run from a worktree checked out at `origin/main` (pre-active-set):
 *
 *     git worktree add /tmp/tropical-oldpipe origin/main
 *     cd /tmp/tropical-oldpipe && git submodule update --init --recursive
 *     make build
 *     cp $THIS_FILE scripts/capture_old_pipeline_audio.ts
 *     bun run scripts/capture_old_pipeline_audio.ts
 *     cp tests/golden/migration/*.json $REPO/tests/golden/migration/
 *
 * The goldens (sha256 + first-32-sample diagnostic) are then read by
 * `tests/equiv/migration_audio.test.ts` in the new branch,
 * which compiles the same fixture inputs through the active-set
 * pipeline and asserts byte-equal audio.
 *
 * The script lives in the active-set branch as a record; running it
 * requires the OLD pipeline's binary, which is only present at the
 * pre-active-set commits.
 */

import { readFileSync, readdirSync, writeFileSync, existsSync, mkdirSync } from 'node:fs'
import { resolve, basename, dirname, join } from 'node:path'
import { createHash } from 'node:crypto'
import { makeSession, loadJSON } from '../compiler/session.js'
import { loadStdlib } from '../compiler/program.js'
import { applySessionWiring } from '../compiler/apply_plan.js'

const FRAMES = 16
const BUFFER_LEN = 256
const FIXTURE_DIR = resolve(__dirname, '..', 'tests', 'fixtures', 'flat_plan')
const GOLDEN_DIR  = resolve(__dirname, '..', 'tests', 'golden', 'migration')

if (!existsSync(GOLDEN_DIR)) mkdirSync(GOLDEN_DIR, { recursive: true })

const files = readdirSync(FIXTURE_DIR).filter(f => f.endsWith('.json')).sort()

interface Golden {
  fixture:       string
  sample_count:  number
  hash:          string
  first_samples: number[]
}

for (const file of files) {
  const name = basename(file, '.json')
  const fixture = JSON.parse(readFileSync(join(FIXTURE_DIR, file), 'utf-8')) as
    { input: { schema: string; [k: string]: unknown } }

  const session = makeSession(BUFFER_LEN)
  loadStdlib(session)
  loadJSON(fixture.input, session)
  applySessionWiring(session)
  session.graph.primeJit()

  const samples = new Float64Array(BUFFER_LEN * FRAMES)
  for (let f = 0; f < FRAMES; f++) {
    session.graph.process()
    samples.set(session.graph.outputBuffer, f * BUFFER_LEN)
  }
  session.graph.dispose()

  const h = createHash('sha256')
  h.update(new Uint8Array(samples.buffer, samples.byteOffset, samples.byteLength))
  const hash = h.digest('hex')

  const golden: Golden = {
    fixture: file,
    sample_count: samples.length,
    hash,
    first_samples: Array.from(samples.subarray(0, 32)),
  }
  writeFileSync(join(GOLDEN_DIR, `${name}.json`), JSON.stringify(golden, null, 2) + '\n')
  // eslint-disable-next-line no-console
  console.log(`  CAPTURED  ${name.padEnd(28)}  ${hash.slice(0, 16)}`)
}
