/**
 * diff_parse.ts — the Phase 7 surface-parser differential gate.
 *
 * For every `stdlib/*.md`, parse it with side B (default: the Lean
 * `diffcli parse-md`) and compare the resulting ParsedProgram JSON,
 * structurally, against the committed `stdlib/parsed/<Name>.json` —
 * which is the TS parser's frozen output (produced by
 * scripts/build_parsed_stdlib.ts). Filenames equal program names, so
 * `stdlib/Foo.md` ↔ `stdlib/parsed/Foo.json`.
 *
 * Comparison is structural (scripts/diff/structural.ts): key-order
 * insensitive, numbers by IEEE-754 bit identity.
 *
 * Usage:
 *   bun run scripts/diff/diff_parse.ts --b="./lean/.lake/build/bin/diffcli"
 */

import { execFileSync } from 'node:child_process'
import { readFileSync, readdirSync } from 'node:fs'
import { resolve, basename } from 'node:path'
import { structuralDiff, formatDiffs } from './structural.js'

const STDLIB = 'stdlib'
const PARSED = 'stdlib/parsed'
const FIXTURES = 'tests/fixtures/surface'
const TS_PARSE = 'bun run scripts/diff/parse_cmd.ts'

function parseArgs(argv: string[]): { b: string } {
  let b = './lean/.lake/build/bin/diffcli'
  for (const arg of argv) if (arg.startsWith('--b=')) b = arg.slice(4)
  return { b }
}

function runParseMd(cmd: string, file: string): unknown {
  const parts = cmd.split(' ').filter(s => s.length > 0)
  const out = execFileSync(parts[0], [...parts.slice(1), 'parse-md', file], {
    encoding: 'utf-8',
    maxBuffer: 64 * 1024 * 1024,
  })
  return JSON.parse(out)
}

function runTsParse(file: string): unknown {
  const parts = TS_PARSE.split(' ').filter(s => s.length > 0)
  const out = execFileSync(parts[0], [...parts.slice(1), file], {
    encoding: 'utf-8',
    maxBuffer: 64 * 1024 * 1024,
  })
  return JSON.parse(out)
}

function main(argv: string[]): number {
  const { b } = parseArgs(argv)
  const mdFiles = readdirSync(STDLIB)
    .filter(f => f.endsWith('.md') && f !== 'README.md')
    .sort()

  let failed = 0
  for (const f of mdFiles) {
    const name = basename(f, '.md')
    const expected = JSON.parse(readFileSync(resolve(PARSED, `${name}.json`), 'utf-8'))
    let actual: unknown
    try {
      actual = runParseMd(b, resolve(STDLIB, f))
    } catch (e) {
      console.log(`✗ ${name}: parse-md crashed: ${(e as Error).message.split('\n')[0]}`)
      failed++
      continue
    }
    if (actual && typeof actual === 'object' && 'error' in (actual as object)) {
      console.log(`✗ ${name}: ${(actual as { error: string }).error}`)
      failed++
      continue
    }
    const diffs = structuralDiff(actual, expected)
    if (diffs.length > 0) {
      console.log(`✗ ${name}: ${diffs.length} diff(s)`)
      console.log(formatDiffs(diffs.slice(0, 6)).split('\n').map(l => '    ' + l).join('\n'))
      failed++
    } else {
      console.log(`✓ ${name}`)
    }
  }

  // Fixtures the stdlib corpus doesn't exercise (iterate/chain/map2/zipWith,
  // number forms, nested programs, match/tag). Lean parse-md vs TS parseProgram.
  let fixtures: string[] = []
  try {
    fixtures = readdirSync(FIXTURES).filter(f => f.endsWith('.md')).sort()
  } catch { /* no fixtures dir */ }

  let fxFailed = 0
  for (const f of fixtures) {
    const name = basename(f, '.md')
    const path = resolve(FIXTURES, f)
    let leanOut: unknown, tsOut: unknown
    try {
      leanOut = runParseMd(b, path)
      tsOut = runTsParse(path)
    } catch (e) {
      console.log(`✗ [fixture] ${name}: crashed: ${(e as Error).message.split('\n')[0]}`)
      fxFailed++
      continue
    }
    const diffs = structuralDiff(leanOut, tsOut)
    if (diffs.length > 0) {
      console.log(`✗ [fixture] ${name}: ${diffs.length} diff(s)`)
      console.log(formatDiffs(diffs.slice(0, 6)).split('\n').map(l => '    ' + l).join('\n'))
      fxFailed++
    } else {
      console.log(`✓ [fixture] ${name}`)
    }
  }

  console.log(`\nstdlib: ${mdFiles.length - failed}/${mdFiles.length} · fixtures: ${fixtures.length - fxFailed}/${fixtures.length}`)
  return failed + fxFailed === 0 ? 0 : 1
}

process.exit(main(process.argv.slice(2)))
