/**
 * diff_emit.ts — the Phase 6 stage-6b (per-program emit) differential
 * gate.
 *
 * Side A is the TS pipeline (`emit_cmd.ts`: elaborate → full strata,
 * inline mode → compileResolved); side B the Lean port
 * (`--b="./lean/.lake/build/bin/diffcli"`, emit-stdlib / emit-file).
 * Compared output is the wire-format `PerInstancePlan` JSON — temp and
 * array-slot numbering included, so CSE-partition divergence is loud.
 *
 * Corpus = the diff-strata corpus at inline mode: every manifest
 * program (generics re-run all-args=8), the elaborable raise fixtures,
 * and the specialize error probes (byte-wise error parity). Fractal
 * emit paths (nested slot maps, per-child pre-input) are exercised at
 * stage 6c/6d through the partitioner, not here.
 *
 * Usage:
 *   bun run scripts/diff/diff_emit.ts                  # TS vs TS
 *   bun run scripts/diff/diff_emit.ts --b="<cmd>"
 */

import { execFileSync } from 'node:child_process'
import { readFileSync, readdirSync, writeFileSync, mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { structuralDiff, formatDiffs } from './structural.js'
import { normalizeProgramFile } from '../../compiler/session.js'
import { raiseProgram } from '../../compiler/parse/raise.js'

const TS_EMIT = 'bun run scripts/diff/emit_cmd.ts'
const PARSED_DIR = 'stdlib/parsed'

interface Args { a: string; b: string }

function parseArgs(argv: string[]): Args {
  const args: Args = { a: TS_EMIT, b: TS_EMIT }
  for (const arg of argv) {
    if (arg.startsWith('--a=')) args.a = arg.slice(4)
    else if (arg.startsWith('--b=')) args.b = arg.slice(4)
  }
  return args
}

type Verb = 'emit-stdlib' | 'emit-file'

function runVia(cmd: string, verb: Verb, arg: string, flags: string[]): unknown {
  const parts = cmd.split(' ').filter(s => s.length > 0)
  const out = execFileSync(parts[0], [...parts.slice(1), verb, arg, ...flags], {
    encoding: 'utf-8',
    maxBuffer: 256 * 1024 * 1024,
  })
  return JSON.parse(out)
}

function jsonFiles(dir: string): string[] {
  return readdirSync(dir)
    .filter(f => f.endsWith('.json'))
    .sort()
    .map(f => resolve(dir, f))
}

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

function typeParamNames(name: string): string[] {
  const parsed = JSON.parse(readFileSync(`${PARSED_DIR}/${name}.json`, 'utf-8')) as {
    type_params?: Record<string, unknown>
  }
  return Object.keys(parsed.type_params ?? {})
}

function main(argv: string[]): number {
  const { a, b } = parseArgs(argv)
  const tmp = mkdtempSync(join(tmpdir(), 'tropical-diff-emit-'))
  let failed = 0
  let total = 0

  const compare = (verb: Verb, arg: string, label: string,
                   typeArgs?: Record<string, number>): void => {
    total++
    const flags = typeArgs === undefined ? [] : [`--type-args=${JSON.stringify(typeArgs)}`]
    let outA: unknown, outB: unknown
    try {
      outA = runVia(a, verb, arg, flags)
      outB = runVia(b, verb, arg, flags)
    } catch (e) {
      console.log(`  ERROR  ${verb.padEnd(12)} ${label.padEnd(34)}  ${e instanceof Error ? e.message.split('\n')[0] : String(e)}`)
      failed++
      return
    }
    const diffs = structuralDiff(outA, outB)
    if (diffs.length === 0) {
      console.log(`  PASS   ${verb.padEnd(12)} ${label}`)
    } else {
      console.log(`  FAIL   ${verb.padEnd(12)} ${label.padEnd(34)}  ${diffs.length}+ divergences`)
      console.log(formatDiffs(diffs))
      failed++
    }
  }

  const manifest = JSON.parse(readFileSync(`${PARSED_DIR}/manifest.json`, 'utf-8')) as {
    programs: string[]
  }

  // 1. The full stdlib; generics get a second, fully-explicit run.
  for (const name of manifest.programs) {
    compare('emit-stdlib', name, name)
    const params = typeParamNames(name)
    if (params.length > 0) {
      const args = Object.fromEntries(params.map(p => [p, 8]))
      compare('emit-stdlib', name, `${name}<all=8>`, args)
    }
  }

  // 2. Elaborable raise fixtures.
  for (const f of jsonFiles('tests/fixtures/raise')) {
    const name = f.split('/').pop() ?? f
    const parsed = raiseToParsed(f)
    if (parsed === null) continue
    const tmpFile = join(tmp, name)
    writeFileSync(tmpFile, JSON.stringify(parsed) + '\n')
    compare('emit-file', tmpFile, name)
  }

  // 3. Error probes (byte-wise message parity through specialize).
  const generics = manifest.programs.filter(n => typeParamNames(n).length > 0)
  if (generics.length > 0) {
    const g = generics[0]
    compare('emit-stdlib', g, `${g}<extra-arg>`, { __not_a_param: 1 })
    const first = typeParamNames(g)[0]
    compare('emit-stdlib', g, `${g}<non-integer>`, { [first]: 2.5 })
  }

  console.log()
  console.log(failed === 0
    ? `all ${total} emit outputs agree`
    : `${failed}/${total} diverged`)
  return failed === 0 ? 0 : 1
}

process.exit(main(process.argv.slice(2)))
