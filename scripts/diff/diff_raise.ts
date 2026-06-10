/**
 * diff_raise.ts — the Phase 4 parsed-layer differential gate.
 *
 * Each side is a command implementing the raise_cmd contract
 * (`<cmd> raise <file.json>` / `<cmd> parsed-roundtrip <file.json>` →
 * JSON on stdout). Both sides default to the TS pipeline, so the
 * harness is green TS-vs-TS; `make diff-raise` pits the Lean port
 * against the TS oracle with `--b="./lean/.lake/build/bin/diffcli"`.
 *
 * Three corpora:
 *   raise:    patches/*.json              (session-shaped programs; these
 *             fail raise by design — the gate is the exact error string)
 *   raise:    tests/fixtures/raise/*.json (happy-path legacy programs
 *             exercising every raise op mapping + bounds lowering)
 *   roundtrip: stdlib/parsed/*.json       (ParsedProgram JSON; side B
 *             decodes to its typed AST and re-encodes — codec fidelity)
 *
 * Comparison is structural (scripts/diff/structural.ts): key-order
 * insensitive, numbers by IEEE-754 bit identity.
 *
 * Usage:
 *   bun run scripts/diff/diff_raise.ts                  # TS vs TS
 *   bun run scripts/diff/diff_raise.ts --b="<cmd>"      # TS vs <cmd>
 */

import { execFileSync } from 'node:child_process'
import { readFileSync, readdirSync } from 'node:fs'
import { resolve } from 'node:path'
import { structuralDiff, formatDiffs } from './structural.js'

const TS_RAISE = 'bun run scripts/diff/raise_cmd.ts'

interface Args { a: string; b: string }

function parseArgs(argv: string[]): Args {
  const args: Args = { a: TS_RAISE, b: TS_RAISE }
  for (const arg of argv) {
    if (arg.startsWith('--a=')) args.a = arg.slice(4)
    else if (arg.startsWith('--b=')) args.b = arg.slice(4)
  }
  return args
}

function runVia(cmd: string, verb: 'raise' | 'parsed-roundtrip', file: string): unknown {
  const parts = cmd.split(' ').filter(s => s.length > 0)
  const out = execFileSync(parts[0], [...parts.slice(1), verb, file], {
    encoding: 'utf-8',
    maxBuffer: 64 * 1024 * 1024,
  })
  return JSON.parse(out)
}

function jsonFiles(dir: string, exclude: string[] = []): string[] {
  return readdirSync(dir)
    .filter(f => f.endsWith('.json') && !exclude.includes(f))
    .sort()
    .map(f => resolve(dir, f))
}

function main(argv: string[]): number {
  const { a, b } = parseArgs(argv)
  let failed = 0
  let total = 0

  const compare = (verb: 'raise' | 'parsed-roundtrip', file: string): void => {
    const name = file.split('/').pop() ?? file
    total++
    let outA: unknown, outB: unknown
    try {
      outA = runVia(a, verb, file)
      outB = runVia(b, verb, file)
    } catch (e) {
      console.log(`  ERROR  ${verb.padEnd(17)} ${name.padEnd(28)}  ${e instanceof Error ? e.message.split('\n')[0] : String(e)}`)
      failed++
      return
    }
    const diffs = structuralDiff(outA, outB)
    if (diffs.length === 0) {
      console.log(`  PASS   ${verb.padEnd(17)} ${name}`)
    } else {
      console.log(`  FAIL   ${verb.padEnd(17)} ${name.padEnd(28)}  ${diffs.length}+ divergences`)
      console.log(formatDiffs(diffs))
      failed++
    }
  }

  for (const f of jsonFiles('patches')) compare('raise', f)
  for (const f of jsonFiles('tests/fixtures/raise')) compare('raise', f)
  for (const f of jsonFiles('stdlib/parsed', ['manifest.json'])) {
    compare('parsed-roundtrip', f)
  }

  // Sanity: the roundtrip corpus on side A must literally be the file.
  // (Guards against a side-A regression making the gate vacuous.)
  for (const f of jsonFiles('stdlib/parsed', ['manifest.json'])) {
    const echoed = runVia(a, 'parsed-roundtrip', f)
    if (structuralDiff(echoed, JSON.parse(readFileSync(f, 'utf-8'))).length > 0) {
      console.log(`  FAIL   side A does not echo ${f}`)
      failed++
    }
  }

  console.log()
  console.log(failed === 0 ? `all ${total} raise outputs agree` : `${failed}/${total} diverged`)
  return failed === 0 ? 0 : 1
}

process.exit(main(process.argv.slice(2)))
