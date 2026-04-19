/**
 * Phase A round-trip: every frozen FlatPlan fixture must reproduce byte-for-byte
 * when its v1 input is migrated to v2, loaded via the v2 dispatch, and re-flattened.
 *
 * This is the core verification that the unified IR (tropical_program_2) is a
 * lossless re-shape of tropical_program_1. When it passes, Phase B can flip
 * on-disk emitters to v2 without breaking any downstream consumer.
 */

import { describe, test, expect } from 'bun:test'
import { readFileSync, readdirSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

import { makeSession, loadJSON, v1ProgramJSONToV2Node, v2NodeToFile } from './session.js'
import { loadStdlib } from './program.js'
import { flattenSession } from './flatten.js'
import type { ProgramJSON } from './program.js'

const __dirname = dirname(fileURLToPath(import.meta.url))
const FIXTURE_DIR = join(__dirname, '__fixtures__/flat_plan')

const fixtures = readdirSync(FIXTURE_DIR)
  .filter(f => f.endsWith('.json'))
  .sort()

describe('unified IR round-trip (v1 → v2 → ProgramDef → FlatPlan)', () => {
  for (const file of fixtures) {
    test(file, () => {
      const { input, expected_plan } = JSON.parse(
        readFileSync(join(FIXTURE_DIR, file), 'utf-8'),
      ) as { input: ProgramJSON; expected_plan: unknown }

      const { node, topLevel } = v1ProgramJSONToV2Node(input)
      const v2File = v2NodeToFile(node, topLevel)

      const session = makeSession()
      loadStdlib(session)
      loadJSON(v2File as unknown as { schema: string; [k: string]: unknown }, session)
      const plan = flattenSession(session)

      expect(JSON.stringify(plan)).toBe(JSON.stringify(expected_plan))
    })
  }
})
