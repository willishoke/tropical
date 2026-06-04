/**
 * stdlib_round_trip.test.ts — smoke test for the stdlib loader.
 *
 * Loads every `stdlib/*.md` literate program document via the production
 * `loadStdlib` path (markdown extract → parseProgram → elaborate →
 * strataPipeline) and confirms the registry is populated with the expected
 * program names.
 *
 * If any program file fails to parse or elaborate, this test fails —
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

describe('stdlib loader — every literate program file loads cleanly', () => {
  test('session.programs covers every top-level program in stdlib/', () => {
    const session = {
      typeRegistry: new Map<string, Compiled>(),
      instanceRegistry: new Map(),
      paramRegistry: new Map(),
      specializationCache: new Map(),
      programs: new Map<string, unknown>(),
    }
    loadStdlib(session as Parameters<typeof loadStdlib>[0])

    const programFiles = readdirSync(stdlibDir)
      .filter(f => f.endsWith('.md') && f !== 'README.md')
      .sort()
    const expected = programFiles.map(f => f.replace(/\.md$/, ''))
    // Phase 5 (issue #156): session.programs is the unified registry
    // holding both concrete (post-strata) and generic (raw template)
    // entries. typeRegistry covers only concretes; the combined
    // expected set lives in programs.
    const loaded = new Set(session.programs.keys())
    for (const name of expected) {
      expect(loaded.has(name)).toBe(true)
    }
  })
})
