/**
 * schema_audit.test.ts — Phase D P0.3.
 *
 * Asserts the in-tree corpus stays free of deprecated `tropical_program_2`
 * fields. The schema still accepts the deprecated fields with a load-time
 * warning (see `compiler/schema.ts` and `compiler/program.ts`), but new
 * stdlib/patches should not introduce them. Removal target: after Phase E.
 *
 * Currently grandfathered patches still using `audio_outputs` are tracked
 * in `GRANDFATHERED_AUDIO_OUTPUTS`. Migrating those to body
 * `outputAssign(name="dac.out")` wires is gated on D5 (snake-case → camelCase
 * op normalization at the JSON ingest boundary). When a patch migrates,
 * remove its name from the list — the test will fail if the list and the
 * actual file state drift.
 */

import { describe, test, expect } from 'bun:test'
import { readdirSync, readFileSync } from 'node:fs'
import { join } from 'node:path'

const PATCHES_DIR = join(__dirname, '..', 'patches')
const STDLIB_DIR = join(__dirname, '..', 'stdlib')

/** Patches that still use `audio_outputs` instead of body `dac.out` wires.
 *  Migration is gated on D5 (snake_case op normalization at ingest). */
const GRANDFATHERED_AUDIO_OUTPUTS: ReadonlySet<string> = new Set([
  'acid_noise.json',
  'bubble_cloud.json',
  'bubble_drip.json',
  'cross_fm_4.json',
  'cross_fm_evolved.json',
  'odd_harmonics.json',
  'sequencer_demo.json',
])

/** Patches that still use top-level `params`. Empty today; ratchets new
 *  additions out. */
const GRANDFATHERED_TOPLEVEL_PARAMS: ReadonlySet<string> = new Set([])

function patchFiles(): string[] {
  return readdirSync(PATCHES_DIR).filter(f => f.endsWith('.json')).sort()
}

function stdlibFiles(): string[] {
  return readdirSync(STDLIB_DIR).filter(f => f.endsWith('.trop')).sort()
}

function loadJson(path: string): Record<string, unknown> {
  return JSON.parse(readFileSync(path, 'utf-8')) as Record<string, unknown>
}

describe('schema_audit — in-tree corpus avoids deprecated fields', () => {
  test('stdlib `.trop` files cannot carry top-level `audio_outputs`/`params`', () => {
    // .trop surface syntax has no concept of these top-level fields, so this
    // is structurally true. Asserted by content scan as a future-proofing
    // signal: if anyone hand-edits a .trop file to inline JSON-shaped
    // metadata, this catches it.
    const offenders: string[] = []
    for (const f of stdlibFiles()) {
      const content = readFileSync(join(STDLIB_DIR, f), 'utf-8')
      // Match a top-level "audio_outputs"/"params" key — only at the start
      // of a line (no leading non-quote chars), to avoid false positives
      // inside code-block bodies that mention the words in prose.
      if (/^\s*"(audio_outputs|params)"\s*:/m.test(content)) {
        offenders.push(f)
      }
    }
    expect(offenders).toEqual([])
  })

  test('patches/*.json: `audio_outputs` usage matches the grandfathered list exactly', () => {
    const using: string[] = []
    for (const f of patchFiles()) {
      const obj = loadJson(join(PATCHES_DIR, f))
      if (Array.isArray(obj.audio_outputs) && obj.audio_outputs.length > 0) {
        using.push(f)
      }
    }
    using.sort()
    const expected = [...GRANDFATHERED_AUDIO_OUTPUTS].sort()

    // Two-way assertion: the actual set must equal the grandfathered set.
    // - New offenders ⇒ test fails, forces migration before merge.
    // - Migrated patches ⇒ the list is stale, prompting list maintenance.
    expect(using).toEqual(expected)
  })

  test('patches/*.json: top-level `params` usage matches the grandfathered list exactly', () => {
    const using: string[] = []
    for (const f of patchFiles()) {
      const obj = loadJson(join(PATCHES_DIR, f))
      if (Array.isArray(obj.params) && obj.params.length > 0) {
        using.push(f)
      }
    }
    using.sort()
    const expected = [...GRANDFATHERED_TOPLEVEL_PARAMS].sort()
    expect(using).toEqual(expected)
  })
})
