/**
 * migration_audio.test.ts — pre/post-active-set audio equivalence gate.
 *
 * For every fixture in `tests/fixtures/flat_plan/`, this test
 * compiles the fixture's `input` (a `tropical_program_2`) through the
 * NEW pipeline (compileSession → tropical_plan_5 → native JIT), runs
 * the same 16 frames × 256 samples that the OLD pipeline ran, and
 * compares the audio output byte-for-byte against the corresponding
 * golden in `tests/golden/migration/<name>.json`.
 *
 * The goldens were captured by running `scripts/capture_old_pipeline_audio.ts`
 * from a worktree checked out at `origin/main` (before the active-set
 * runtime landed). Each golden carries a sha256 of the full sample
 * buffer plus the first 32 raw samples as a diagnostic. The hash
 * comparison is the regression check; the raw-sample comparison
 * helps diagnose drift when the hash fails.
 *
 * Failures here mean the new pipeline diverged in audio output from
 * the old pipeline on a canonical-coverage input — the kind of
 * regression that the interpreter-equivalence tests can't catch on
 * their own because the interpreter went through this same migration
 * (gateable→alive rename, same session→root lowering).
 *
 * The capture script lives under `scripts/` in this branch as a
 * record of how the goldens were generated; rerun it from a clean
 * `git worktree add` at `origin/main` (pre-active-set) to refresh.
 *
 * NOTE (dac-as-sink): the goldens were re-baselined to the current
 * pipeline when the DAC mix moved from a hardcoded engine `÷20` to a
 * sink `gain` MULTIPLY (1/20). The only divergence from the original
 * origin/main capture is that div→mul reassociation — ≤1 ULP per sample
 * (max abs 7e-18, max rel 2e-16), zero logic change. The cross-backend /
 * cross-lowering oracles (root_vs_flat, wasm_vs_jit) remain byte-exact
 * green; this gate guards future drift from the re-baselined reference.
 */

import { describe, test, expect } from 'bun:test'
import { readFileSync, readdirSync, existsSync } from 'node:fs'
import { resolve, basename, dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { createHash } from 'node:crypto'

import { makeSession, loadJSON } from '../../compiler/session.js'
import { loadStdlib } from '../../compiler/program.js'
import { applySessionWiring } from '../../compiler/apply_plan.js'

const __filename = fileURLToPath(import.meta.url)
const __dirname  = dirname(__filename)

const FRAMES      = 16
const BUFFER_LEN  = 256
const FIXTURE_DIR = resolve(__dirname, '..', 'fixtures', 'flat_plan')
const GOLDEN_DIR  = resolve(__dirname, '..', 'golden', 'migration')

interface Golden {
  fixture:       string
  sample_count:  number
  hash:          string
  first_samples: number[]
}

const fixtures = readdirSync(FIXTURE_DIR)
  .filter(f => f.endsWith('.json'))
  .sort()

const KNOWN_UNSUPPORTED = new Set<string>([])

describe('migration: old (plan_4) pipeline ≡ new (plan_5) pipeline audio', () => {
  for (const file of fixtures) {
    const name = basename(file, '.json')
    const goldenPath = join(GOLDEN_DIR, `${name}.json`)
    const isUnsupported = KNOWN_UNSUPPORTED.has(file)
    const testFn = isUnsupported ? test.skip : test

    testFn(name, () => {
      expect(existsSync(goldenPath)).toBe(true)
      const golden = JSON.parse(readFileSync(goldenPath, 'utf-8')) as Golden

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

      expect(samples.length).toBe(golden.sample_count)

      const h = createHash('sha256')
      h.update(new Uint8Array(samples.buffer, samples.byteOffset, samples.byteLength))
      const hash = h.digest('hex')

      if (hash !== golden.hash) {
        // Surface the first sample of divergence to make the failure
        // diagnosable without re-running the capture script.
        const newFirst = Array.from(samples.subarray(0, 32))
        const divergeAt = newFirst.findIndex((v, i) => v !== golden.first_samples[i])
        const detail = divergeAt < 0
          ? `first 32 samples match; divergence beyond sample 31`
          : `divergence at sample ${divergeAt}: old=${golden.first_samples[divergeAt]} new=${newFirst[divergeAt]}`
        throw new Error(`audio hash mismatch for ${name}: ${detail}`)
      }
      expect(hash).toBe(golden.hash)
    })
  }
})
