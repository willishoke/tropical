/**
 * diff_strata.ts — the Phase 5 (strata pipeline) differential gate.
 *
 * Whole-strata comparison with a hybrid prefix: side A is the full TS
 * pipeline; side B runs passes 1..K (`--k=K`) via the side-B command
 * (the Lean port: `--b="./lean/.lake/build/bin/diffcli"`), ships the
 * intermediate through the `tropical_resolved_1` codec, and the TS
 * suffix (`strata_cmd.ts strata-suffix --from=K`) completes passes
 * K+1..5. Only the FINAL post-strata output is compared — divergence
 * at stage K localizes to pass K by construction, with no
 * per-intermediate diff semantics. At K=5 the suffix is skipped and
 * the comparison is pure Lean-vs-TS.
 *
 * Both sides default to the TS command (TS-vs-TS green from day one —
 * which also property-checks codec round-trip transparency at the
 * prefix boundary, the hybrid's premise).
 *
 * Corpus, per --mode=inline AND --mode=nested:
 *   strata-stdlib: every program in stdlib/parsed/manifest.json;
 *                  generics additionally specialized with every
 *                  type-param = 8 (the codec-gate convention).
 *   strata-file:   tests/fixtures/raise/*.json that raise (pre-raised
 *                  in-process, the diff_elab pattern).
 *   error probes:  extra type-arg + non-integer type-arg on the first
 *                  generic in the manifest (byte-wise error parity).
 *
 * Comparison is structural (scripts/diff/structural.ts): key-order
 * insensitive, numbers by IEEE-754 bit identity.
 *
 * Usage:
 *   bun run scripts/diff/diff_strata.ts                  # TS vs TS, K=0
 *   bun run scripts/diff/diff_strata.ts --b="<cmd>" --k=2
 */

import { execFileSync } from 'node:child_process'
import { readFileSync, readdirSync, writeFileSync, mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { structuralDiff, formatDiffs } from './structural.js'
import { normalizeProgramFile } from '../../compiler/session.js'
import { raiseProgram } from '../../compiler/parse/raise.js'

const TS_STRATA = 'bun run scripts/diff/strata_cmd.ts'
const PASS_COUNT = 5
const PARSED_DIR = 'stdlib/parsed'

interface Args { a: string; b: string; k: number }

function parseArgs(argv: string[]): Args {
  const args: Args = { a: TS_STRATA, b: TS_STRATA, k: 0 }
  for (const arg of argv) {
    if (arg.startsWith('--a=')) args.a = arg.slice(4)
    else if (arg.startsWith('--b=')) args.b = arg.slice(4)
    else if (arg.startsWith('--k=')) args.k = Number(arg.slice(4))
  }
  if (!Number.isInteger(args.k) || args.k < 0 || args.k > PASS_COUNT) {
    throw new Error(`--k must be 0..${PASS_COUNT}`)
  }
  return args
}

type Verb = 'strata-stdlib' | 'strata-file' | 'strata-suffix'

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

/** Type-param names of a parsed-bridge program (`type_params` is an
 *  object keyed by name), for the generics<8> corpus arm. */
function typeParamNames(name: string): string[] {
  const parsed = JSON.parse(readFileSync(`${PARSED_DIR}/${name}.json`, 'utf-8')) as {
    type_params?: Record<string, unknown>
  }
  return Object.keys(parsed.type_params ?? {})
}

function main(argv: string[]): number {
  const { a, b, k } = parseArgs(argv)
  const tmp = mkdtempSync(join(tmpdir(), 'tropical-diff-strata-'))
  let failed = 0
  let total = 0
  let prefixId = 0

  const compare = (verb: 'strata-stdlib' | 'strata-file', arg: string, label: string,
                   mode: 'inline' | 'nested', typeArgs?: Record<string, number>): void => {
    total++
    const flags = [`--mode=${mode}`]
    if (typeArgs !== undefined) flags.push(`--type-args=${JSON.stringify(typeArgs)}`)
    let outA: unknown, outB: unknown
    try {
      outA = runVia(a, verb, arg, flags)
      const prefix = runVia(b, verb, arg, [...flags, `--upto=${k}`])
      if (k < PASS_COUNT && !isErrorOutput(prefix)) {
        const prefixFile = join(tmp, `prefix-${prefixId++}.json`)
        writeFileSync(prefixFile, JSON.stringify(prefix) + '\n')
        outB = runVia(TS_STRATA, 'strata-suffix', prefixFile, [...flags, `--from=${k}`])
      } else {
        outB = prefix
      }
    } catch (e) {
      console.log(`  ERROR  ${verb.padEnd(14)} ${label.padEnd(34)}  ${e instanceof Error ? e.message.split('\n')[0] : String(e)}`)
      failed++
      return
    }
    const diffs = structuralDiff(outA, outB)
    if (diffs.length === 0) {
      console.log(`  PASS   ${verb.padEnd(14)} ${label}`)
    } else {
      console.log(`  FAIL   ${verb.padEnd(14)} ${label.padEnd(34)}  ${diffs.length}+ divergences`)
      console.log(formatDiffs(diffs))
      failed++
    }
  }

  const manifest = JSON.parse(readFileSync(`${PARSED_DIR}/manifest.json`, 'utf-8')) as {
    programs: string[]
  }

  // Pre-raise the elaborable raise fixtures once (mode-independent).
  const raised: Array<{ file: string; name: string }> = []
  for (const f of jsonFiles('tests/fixtures/raise')) {
    const name = f.split('/').pop() ?? f
    const parsed = raiseToParsed(f)
    if (parsed === null) continue
    const tmpFile = join(tmp, name)
    writeFileSync(tmpFile, JSON.stringify(parsed) + '\n')
    raised.push({ file: tmpFile, name })
  }

  const generics = manifest.programs.filter(n => typeParamNames(n).length > 0)

  for (const mode of ['inline', 'nested'] as const) {
    console.log(`mode: ${mode}`)

    // 1. The full stdlib; generics get a second, fully-explicit run.
    for (const name of manifest.programs) {
      compare('strata-stdlib', name, name, mode)
      const params = typeParamNames(name)
      if (params.length > 0) {
        const args = Object.fromEntries(params.map(p => [p, 8]))
        compare('strata-stdlib', name, `${name}<all=8>`, mode, args)
      }
    }

    // 2. Elaborable raise fixtures.
    for (const { file, name } of raised) compare('strata-file', file, name, mode)

    // 3. Error probes (byte-wise message parity through specialize).
    if (generics.length > 0) {
      const g = generics[0]
      compare('strata-stdlib', g, `${g}<extra-arg>`, mode, { __not_a_param: 1 })
      const first = typeParamNames(g)[0]
      compare('strata-stdlib', g, `${g}<non-integer>`, mode, { [first]: 2.5 })
    }
  }

  console.log()
  console.log(failed === 0
    ? `all ${total} strata outputs agree (k=${k})`
    : `${failed}/${total} diverged (k=${k})`)
  return failed === 0 ? 0 : 1
}

function isErrorOutput(o: unknown): boolean {
  return typeof o === 'object' && o !== null && 'error' in o
}

process.exit(main(process.argv.slice(2)))
