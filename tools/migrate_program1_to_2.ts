/**
 * Migrate tropical_program_1 JSON files to tropical_program_2 (unified IR).
 *
 * Idempotent: files already at v2 are skipped. Unknown schemas error. No
 * semantic changes — shape only. The Phase A round-trip test already
 * proves v1 → v2 → ProgramDef reproduces byte-identical FlatPlan output.
 *
 * Usage:
 *   bun tools/migrate_program1_to_2.ts stdlib patches
 *   bun tools/migrate_program1_to_2.ts --check stdlib        # dry-run
 */

import { readFileSync, writeFileSync, readdirSync, statSync } from 'node:fs'
import { join, dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { v1ProgramJSONToV2Node, v2NodeToFile } from '../compiler/session.js'
import { parseProgram, parseProgramV2 } from '../compiler/schema.js'

const __dirname = dirname(fileURLToPath(import.meta.url))
const REPO = join(__dirname, '..')

interface Stats { migrated: number; skipped: number; errors: number }

function migrateFile(path: string, check: boolean): 'migrated' | 'skipped' | 'error' {
  const raw = JSON.parse(readFileSync(path, 'utf-8')) as { schema?: string; [k: string]: unknown }
  if (raw.schema === 'tropical_program_2') {
    // Validate: parseProgramV2 will throw if the file is malformed
    parseProgramV2(raw)
    console.log(`  skip (v2):  ${path}`)
    return 'skipped'
  }
  if (raw.schema !== 'tropical_program_1') {
    console.error(`  error:      ${path}: unknown schema '${raw.schema}'`)
    return 'error'
  }
  // Validate v1 parses cleanly before we touch anything
  const v1 = parseProgram(raw)
  const { node, topLevel } = v1ProgramJSONToV2Node(v1)
  const v2File = v2NodeToFile(node, topLevel)

  // Round-trip validation: the v2 file must parse back as a valid v2 file
  parseProgramV2(v2File)

  if (check) {
    console.log(`  would-migrate: ${path}`)
  } else {
    writeFileSync(path, JSON.stringify(v2File, null, 2) + '\n')
    console.log(`  migrated:   ${path}`)
  }
  return 'migrated'
}

function walkJSON(dir: string): string[] {
  const out: string[] = []
  for (const entry of readdirSync(dir)) {
    const p = join(dir, entry)
    if (statSync(p).isDirectory()) out.push(...walkJSON(p))
    else if (entry.endsWith('.json')) out.push(p)
  }
  return out
}

function main(): void {
  const args = process.argv.slice(2)
  const check = args.includes('--check')
  const dirs = args.filter(a => a !== '--check').map(d => resolve(REPO, d))

  if (dirs.length === 0) {
    console.error('usage: bun tools/migrate_program1_to_2.ts [--check] <dir>...')
    process.exit(2)
  }

  const stats: Stats = { migrated: 0, skipped: 0, errors: 0 }

  for (const dir of dirs) {
    console.log(`\n${dir}:`)
    for (const file of walkJSON(dir)) {
      try {
        const result = migrateFile(file, check)
        stats[result === 'error' ? 'errors' : result]++
      } catch (e) {
        console.error(`  error:      ${file}: ${(e as Error).message.slice(0, 200)}`)
        stats.errors++
      }
    }
  }

  console.log(`\n${check ? 'dry-run' : 'done'}: ${stats.migrated} migrated, ${stats.skipped} skipped, ${stats.errors} errors`)
  if (stats.errors > 0) process.exit(1)
}

main()
