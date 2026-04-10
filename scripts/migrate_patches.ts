#!/usr/bin/env bun
/**
 * migrate_patches.ts — Convert tropical_patch_1 files to tropical_program_1.
 *
 * Usage: bun scripts/migrate_patches.ts
 */

import { readFileSync, writeFileSync } from 'fs'
import { join } from 'path'
import { convertPatchToProgram } from '../compiler/program.js'
import type { PatchJSON } from '../compiler/patch.js'

const PATCHES_DIR = join(import.meta.dir, '../patches')
const files = [
  '31tet_otonal_seq.json',
  'acid_noise.json',
  'compressor_harmonics.json',
  'cross_fm_4.json',
  'int_seq_test.json',
  'melancholy_house.json',
  'odd_harmonics.json',
]

for (const file of files) {
  const path = join(PATCHES_DIR, file)
  const raw = JSON.parse(readFileSync(path, 'utf-8'))
  if (raw.schema === 'tropical_program_1') {
    console.log(`  ${file}: already tropical_program_1, skipping`)
    continue
  }
  if (raw.schema !== 'tropical_patch_1') {
    console.log(`  ${file}: unknown schema '${raw.schema}', skipping`)
    continue
  }
  const prog = convertPatchToProgram(raw as PatchJSON)
  prog.name = file.replace('.json', '')
  writeFileSync(path, JSON.stringify(prog, null, 2) + '\n')
  console.log(`  ${file}: migrated to tropical_program_1`)
}

console.log('\nDone.')
