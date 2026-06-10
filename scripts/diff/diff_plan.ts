/**
 * diff_plan.ts — compile every corpus patch through two pipelines and
 * compare the plans structurally.
 *
 * Each side is a command implementing the compile_patch contract
 * (`<cmd> <patch> [--mode=<m>]` → plan JSON on stdout). Both sides
 * default to the TS pipeline, so the harness is green TS-vs-TS from
 * day one; as the Lean port lands, side B becomes
 * `lake exe diffcli compile` (Phase 6) without touching this script.
 *
 * Comparison is structural (scripts/diff/structural.ts): key-order
 * insensitive, numbers by IEEE-754 bit identity. Plan JSON is never
 * compared as text.
 *
 * Usage:
 *   bun run scripts/diff/diff_plan.ts                       # TS vs TS, patches/*.json
 *   bun run scripts/diff/diff_plan.ts --b="<cmd>"           # TS vs <cmd>
 *   bun run scripts/diff/diff_plan.ts --mode=microkernel p1.json p2.json
 */

import { execFileSync } from 'node:child_process'
import { readdirSync } from 'node:fs'
import { resolve } from 'node:path'
import { structuralDiff, formatDiffs } from './structural.js'

const TS_COMPILE = 'bun run scripts/diff/compile_patch.ts'

interface Args { a: string; b: string; mode: string | null; patches: string[] }

function parseArgs(argv: string[]): Args {
  const args: Args = { a: TS_COMPILE, b: TS_COMPILE, mode: null, patches: [] }
  for (const arg of argv) {
    if (arg.startsWith('--a=')) args.a = arg.slice(4)
    else if (arg.startsWith('--b=')) args.b = arg.slice(4)
    else if (arg.startsWith('--mode=')) args.mode = arg.slice(7)
    else args.patches.push(arg)
  }
  if (args.patches.length === 0) {
    args.patches = readdirSync('patches')
      .filter(f => f.endsWith('.json'))
      .map(f => resolve('patches', f))
  }
  return args
}

export function compileVia(cmd: string, patch: string, mode: string | null): unknown {
  const parts = cmd.split(' ').filter(s => s.length > 0)
  const argv = [...parts.slice(1), patch, ...(mode === null ? [] : [`--mode=${mode}`])]
  const out = execFileSync(parts[0], argv, { encoding: 'utf-8', maxBuffer: 64 * 1024 * 1024 })
  return JSON.parse(out)
}

function main(): number {
  const { a, b, mode, patches } = parseArgs(process.argv.slice(2))
  let failed = 0

  for (const patch of patches) {
    const name = patch.split('/').pop() ?? patch
    let planA: unknown, planB: unknown
    try {
      planA = compileVia(a, patch, mode)
      planB = compileVia(b, patch, mode)
    } catch (e) {
      console.log(`  ERROR  ${name.padEnd(28)}  ${e instanceof Error ? e.message.split('\n')[0] : String(e)}`)
      failed++
      continue
    }
    const diffs = structuralDiff(planA, planB)
    if (diffs.length === 0) {
      console.log(`  PASS   ${name}`)
    } else {
      console.log(`  FAIL   ${name.padEnd(28)}  ${diffs.length}+ divergences`)
      console.log(formatDiffs(diffs))
      failed++
    }
  }

  console.log()
  console.log(failed === 0 ? `all ${patches.length} plans agree` : `${failed}/${patches.length} diverged`)
  return failed === 0 ? 0 : 1
}

if (import.meta.main) process.exit(main())
