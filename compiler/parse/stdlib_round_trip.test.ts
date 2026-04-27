/**
 * stdlib_round_trip.test.ts — smoke test for the .trop stdlib loader.
 *
 * Loads every `stdlib/*.trop` file via the production `loadStdlib` path
 * (markdown extract → parseProgram → lowerProgram → loadStdlibFromMap)
 * and confirms the registry is populated with the expected program names.
 *
 * If any .trop file fails to parse or lower, this test fails — preventing
 * silent breakage of files not exercised by other tests in the suite.
 *
 * The original B8c version of this test compared JSON-loaded vs trop-loaded
 * registries. With the JSON files removed in B8e, that comparison has no
 * source-of-truth to test against; the trop pipeline IS the source of truth.
 */

import { readdirSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, test, expect } from 'bun:test'
import { loadStdlib } from '../program.js'
import type { ProgramType } from '../program_types.js'

const __dirname = dirname(fileURLToPath(import.meta.url))
const stdlibDir = join(__dirname, '../../stdlib')

describe('stdlib loader — every .trop file loads cleanly', () => {
  test('typeRegistry + genericTemplates cover every top-level program in stdlib/', () => {
    // Loading drives the full markdown → parse → lower → loadStdlibFromMap
    // chain for every .trop file. If any file fails to parse or lower,
    // loadStdlib throws.
    const session = {
      typeRegistry: new Map<string, ProgramType>(),
      instanceRegistry: new Map(),
      paramRegistry: new Map(),
      triggerRegistry: new Map(),
      specializationCache: new Map(),
      genericTemplates: new Map<string, unknown>(),
    }
    loadStdlib(session)

    const tropFiles = readdirSync(stdlibDir).filter(f => f.endsWith('.trop')).sort()
    const expected = tropFiles.map(f => f.replace(/\.trop$/, ''))
    const loaded = new Set([
      ...session.typeRegistry.keys(),
      ...session.genericTemplates.keys(),
    ])
    for (const name of expected) {
      expect(loaded.has(name)).toBe(true)
    }
  })
})
