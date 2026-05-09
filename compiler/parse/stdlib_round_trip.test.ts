/**
 * stdlib_round_trip.test.ts — smoke test for the .trop stdlib loader.
 *
 * Loads every `stdlib/*.trop` file via the production `loadStdlib` path
 * (markdown extract → parseProgram → elaborate → strataPipeline) and
 * confirms the registry is populated with the expected program names.
 *
 * If any .trop file fails to parse or elaborate, this test fails —
 * preventing silent breakage of files not exercised by other tests.
 */

import { readdirSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, test, expect } from 'bun:test'
import { loadStdlib } from '../program.js'
import type { Compiled } from '../program_types.js'

const __dirname = dirname(fileURLToPath(import.meta.url))
const stdlibDir = join(__dirname, '../../stdlib')

describe('stdlib loader — every .trop file loads cleanly', () => {
  test('typeRegistry + genericTemplatesResolved cover every top-level program in stdlib/', () => {
    const session = {
      typeRegistry: new Map<string, Compiled>(),
      instanceRegistry: new Map(),
      paramRegistry: new Map(),
      triggerRegistry: new Map(),
      specializationCache: new Map(),
      genericTemplatesResolved: new Map<string, unknown>(),
      resolvedRegistry: new Map<string, unknown>(),
    }
    loadStdlib(session as Parameters<typeof loadStdlib>[0])

    const tropFiles = readdirSync(stdlibDir).filter(f => f.endsWith('.trop')).sort()
    const expected = tropFiles.map(f => f.replace(/\.trop$/, ''))
    const loaded = new Set([
      ...session.typeRegistry.keys(),
      ...session.genericTemplatesResolved.keys(),
    ])
    for (const name of expected) {
      expect(loaded.has(name)).toBe(true)
    }
  })
})
