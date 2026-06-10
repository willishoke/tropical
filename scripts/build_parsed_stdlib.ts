/**
 * build_parsed_stdlib.ts — the Phase 4 stdlib bridge.
 *
 * Parses every stdlib/*.md exactly the way `loadStdlib`
 * (compiler/program.ts) does — sorted readdir, extractMarkdown, one
 * tropical code block per file, parseProgram — and writes the resulting
 * ParsedProgram JSON to stdlib/parsed/<Name>.json, plus a manifest.json
 * listing program names in the registration order loadStdlib produces
 * (the dependency-respecting fixed-point elaboration order of
 * loadStdlibFromResolved: each pass elaborates the programs whose
 * sibling refs already resolved; insertion order into the resolved map
 * is the registration order).
 *
 * The generated files are COMMITTED, not gitignored: both the parse
 * order (sorted filenames) and the elaboration fixed point are
 * deterministic, the corpus only changes when a stdlib source changes,
 * and committed files let the Lean side consume the bridge at boot
 * without a bun dependency in its build. Regeneration is cheap
 * (`bun run scripts/build_parsed_stdlib.ts`, < 1s) and `make diff-raise`
 * regenerates before diffing, so a stale corpus cannot pass the gate
 * silently.
 *
 * Usage:  bun run scripts/build_parsed_stdlib.ts
 */

import { mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { extractMarkdown } from '../compiler/parse/markdown.js'
import { parseProgram } from '../compiler/parse/declarations.js'
import { elaborate, type ExternalProgramResolver } from '../compiler/ir/elaborator.js'
import type { ResolvedProgram } from '../compiler/ir/nodes.js'

const stdlibDir = join(import.meta.dir, '../stdlib')
const outDir = join(stdlibDir, 'parsed')

// 1. Parse, the loadStdlib way.
const programFiles = readdirSync(stdlibDir)
  .filter(f => f.endsWith('.md') && f !== 'README.md')
  .sort()

const parsedByName = new Map<string, ReturnType<typeof parseProgram>>()
for (const file of programFiles) {
  const text = readFileSync(join(stdlibDir, file), 'utf-8')
  const ext = extractMarkdown(text)
  if (ext.blocks.length !== 1) {
    throw new Error(`${file}: expected exactly 1 tropical code block, got ${ext.blocks.length}`)
  }
  const parsed = parseProgram(ext.blocks[0].source)
  parsedByName.set(parsed.name, parsed)
}

// 2. Derive the registration order: the same fixed-point elaboration
//    loop loadStdlibFromResolved runs (insertion order into the
//    resolved map is dependency-respecting by construction).
const localResolved = new Map<string, ResolvedProgram>()
const externalResolver: ExternalProgramResolver = name => localResolved.get(name)
const remaining = new Map(parsedByName)
let progress = true
while (progress && remaining.size > 0) {
  progress = false
  for (const [name, parsed] of remaining) {
    try {
      localResolved.set(name, elaborate(parsed, externalResolver))
      remaining.delete(name)
      progress = true
    } catch {
      // Sibling not yet elaborated; retry next pass.
    }
  }
}
if (remaining.size > 0) {
  const [name, parsed] = remaining.entries().next().value as [string, ReturnType<typeof parseProgram>]
  elaborate(parsed, externalResolver)   // re-throw with the real error
  throw new Error(`build_parsed_stdlib: failed to elaborate '${name}'`)
}

// 3. Write the bridge files.
mkdirSync(outDir, { recursive: true })
for (const [name, parsed] of parsedByName) {
  writeFileSync(join(outDir, `${name}.json`), JSON.stringify(parsed, null, 2) + '\n')
}
const manifest = { programs: [...localResolved.keys()] }
writeFileSync(join(outDir, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n')

console.log(`wrote ${parsedByName.size} parsed programs + manifest to stdlib/parsed/`)
