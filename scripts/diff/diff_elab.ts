/**
 * diff_elab.ts — the Phase 4 stage 3 (elaborator) differential gate.
 *
 * Each side is a command implementing the elab_cmd contract
 * (`<cmd> elab-stdlib <Name>` / `<cmd> elab-file <parsed.json>` →
 * canonical `tropical_resolved_1` JSON — or `{error}` — on stdout).
 * Both sides default to the TS pipeline; `make diff-elab` pits the
 * Lean port against the TS oracle with
 * `--b="./lean/.lake/build/bin/diffcli"`.
 *
 * Three corpora:
 *   elab-stdlib: every program in stdlib/parsed/manifest.json — the
 *                full stdlib through the elaborate-only chain.
 *   elab-file:   tests/fixtures/raise/*.json that raise successfully,
 *                pre-raised in-process to ParsedProgram JSON in a temp
 *                dir (ADTs, generics, combinators, bounds lowering).
 *   elab-file:   tests/fixtures/elab/*.json — hand-built ParsedProgram
 *                fixtures, mostly error cases (CycleViolation message
 *                parity, unknown names, duplicate regs, non-exhaustive
 *                matches, ...). Error strings are compared byte-wise.
 *
 * Comparison is structural (scripts/diff/structural.ts): key-order
 * insensitive, numbers by IEEE-754 bit identity.
 *
 * Usage:
 *   bun run scripts/diff/diff_elab.ts                  # TS vs TS
 *   bun run scripts/diff/diff_elab.ts --b="<cmd>"      # TS vs <cmd>
 */

import { execFileSync } from 'node:child_process'
import { readFileSync, readdirSync, writeFileSync, mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { structuralDiff, formatDiffs } from './structural.js'
import { normalizeProgramFile } from '../../compiler/session.js'
import { raiseProgram } from '../../compiler/parse/raise.js'

const TS_ELAB = 'bun run scripts/diff/elab_cmd.ts'

interface Args { a: string; b: string }

function parseArgs(argv: string[]): Args {
  const args: Args = { a: TS_ELAB, b: TS_ELAB }
  for (const arg of argv) {
    if (arg.startsWith('--a=')) args.a = arg.slice(4)
    else if (arg.startsWith('--b=')) args.b = arg.slice(4)
  }
  return args
}

function runVia(cmd: string, verb: 'elab-stdlib' | 'elab-file', arg: string): unknown {
  const parts = cmd.split(' ').filter(s => s.length > 0)
  const out = execFileSync(parts[0], [...parts.slice(1), verb, arg], {
    encoding: 'utf-8',
    maxBuffer: 256 * 1024 * 1024,
  })
  return JSON.parse(out)
}

function jsonFiles(dir: string, exclude: string[] = []): string[] {
  return readdirSync(dir)
    .filter(f => f.endsWith('.json') && !exclude.includes(f))
    .sort()
    .map(f => resolve(dir, f))
}

/** Pre-raise a legacy tropical_program_2 fixture to ParsedProgram JSON.
 *  Returns null when the fixture doesn't raise (e.g. schema_error) —
 *  those are diff_raise territory, not elaborator inputs. */
function raiseToParsed(file: string): unknown | null {
  try {
    const { node } = normalizeProgramFile(
      JSON.parse(readFileSync(file, 'utf-8')) as { schema?: string; [k: string]: unknown },
    )
    return raiseProgram(node)
  } catch {
    return null
  }
}

function main(argv: string[]): number {
  const { a, b } = parseArgs(argv)
  let failed = 0
  let total = 0

  const compare = (verb: 'elab-stdlib' | 'elab-file', arg: string, label: string): void => {
    total++
    let outA: unknown, outB: unknown
    try {
      outA = runVia(a, verb, arg)
      outB = runVia(b, verb, arg)
    } catch (e) {
      console.log(`  ERROR  ${verb.padEnd(12)} ${label.padEnd(28)}  ${e instanceof Error ? e.message.split('\n')[0] : String(e)}`)
      failed++
      return
    }
    const diffs = structuralDiff(outA, outB)
    if (diffs.length === 0) {
      console.log(`  PASS   ${verb.padEnd(12)} ${label}`)
    } else {
      console.log(`  FAIL   ${verb.padEnd(12)} ${label.padEnd(28)}  ${diffs.length}+ divergences`)
      console.log(formatDiffs(diffs))
      failed++
    }
  }

  // 1. The full stdlib, in manifest order.
  const manifest = JSON.parse(readFileSync('stdlib/parsed/manifest.json', 'utf-8')) as {
    programs: string[]
  }
  for (const name of manifest.programs) compare('elab-stdlib', name, name)

  // 2. Elaborable raise fixtures, pre-raised to ParsedProgram JSON.
  const tmp = mkdtempSync(join(tmpdir(), 'tropical-diff-elab-'))
  for (const f of jsonFiles('tests/fixtures/raise')) {
    const name = f.split('/').pop() ?? f
    const parsed = raiseToParsed(f)
    if (parsed === null) {
      console.log(`  SKIP   ${'elab-file'.padEnd(12)} ${name.padEnd(28)}  (does not raise)`)
      continue
    }
    const tmpFile = join(tmp, name)
    writeFileSync(tmpFile, JSON.stringify(parsed) + '\n')
    compare('elab-file', tmpFile, name)
  }

  // 3. Hand-built ParsedProgram fixtures (mostly error-message parity).
  for (const f of jsonFiles('tests/fixtures/elab')) {
    compare('elab-file', f, f.split('/').pop() ?? f)
  }

  console.log()
  console.log(failed === 0 ? `all ${total} elab outputs agree` : `${failed}/${total} diverged`)
  return failed === 0 ? 0 : 1
}

process.exit(main(process.argv.slice(2)))
