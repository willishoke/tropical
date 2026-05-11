/**
 * Golden FlatPlan round-trip: every frozen fixture must reproduce
 * byte-for-byte when its `tropical_program_2` input is loaded into a
 * session and re-compiled through `compileSession`. Protects the
 * resolved-IR pipeline from silent regressions.
 *
 * Pins the `tropical_plan_4` shape directly. Audio equivalence vs. the
 * legacy `flattenSession` was gated by the now-retired
 * `compile_session_equiv.test.ts` during the D2 cutover; post-cutover
 * this file is the regression gate.
 */

import { describe, test, expect } from 'bun:test'
import { readFileSync, readdirSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

import { makeSession, loadJSON } from './session.js'
import { loadStdlib } from './program.js'
// Use compileSessionLegacy directly so the env-var dispatch in
// `compileSession` (which routes to slot-mode under TROPICAL_SLOT_MODE)
// can't perturb the snapshot. These goldens are explicitly pinned to
// the legacy plan shape; slot-mode plans get their own equivalence
// tests in M7.
import { compileSessionLegacy as compileSession } from './ir/compile_session'

const __dirname = dirname(fileURLToPath(import.meta.url))
const FIXTURE_DIR = join(__dirname, '__fixtures__/flat_plan')

const fixtures = readdirSync(FIXTURE_DIR)
  .filter(f => f.endsWith('.json'))
  .sort()

/** Goldens anchor what `compileSession` emits. */

describe('FlatPlan golden fixtures', () => {
  for (const file of fixtures) {
    test(file, () => {
      const { input, expected_plan } = JSON.parse(
        readFileSync(join(FIXTURE_DIR, file), 'utf-8'),
      ) as { input: { schema: string; [k: string]: unknown }; expected_plan: unknown }

      const session = makeSession()
      loadStdlib(session)
      loadJSON(input, session)
      const plan = compileSession(session)

      expect(JSON.stringify(plan)).toBe(JSON.stringify(expected_plan))
    })
  }
})
