/**
 * strata_cmd.ts — the TS side of the strata differential gate
 * (Phase 5).
 *
 * Three verbs. The first two mirror `lean/.lake/build/bin/diffcli`
 * (elaborate exactly like elab_cmd.ts, then run strata passes 0..upto);
 * the third is the harness-only suffix that completes a hybrid prefix
 * shipped through the `tropical_resolved_1` codec:
 *
 *   strata-stdlib <Name>   [--upto=K] [--mode=M] [--type-args=J]
 *   strata-file <parsed.json>  (same flags; resolver = full stdlib chain)
 *   strata-suffix <resolved.json> --from=K [--upto=K2] [--mode=M] [--type-args=J]
 *
 * Pass numbering (K = passes completed; the strataPipeline order):
 *   1 specialize (incl. the entry assertAcyclic)   2 sumLower
 *   3 inlineInstances (skipped in --mode=nested)   4 arrayLower
 *   5 identityElim
 * Default --upto=5 (full pipeline), --mode=inline, --type-args={}.
 *
 * --type-args is a JSON object keyed by type-param NAME; names are
 * resolved against the root program's own TypeParamDecl identities
 * (the identity contract specializeProgram enforces). An unknown name
 * is forwarded as a fresh decl so the "not a declared type-param"
 * error stays reachable — error strings are comparable outputs.
 *
 * ElaborationError / CycleViolation / AcyclicityViolation / pass
 * errors print `{error: <message>}` and exit 0, the diff_elab
 * convention. File/JSON/flag problems are harness errors (exit 1).
 *
 * Usage:  bun run scripts/diff/strata_cmd.ts <verb> <arg> [flags]
 */

import { readFileSync, writeFileSync } from 'node:fs'
import { elaborate, type ExternalProgramResolver } from '../../compiler/ir/elaborator.js'
import type { ResolvedProgram, TypeParamDecl } from '../../compiler/ir/nodes.js'
import { encodeResolved, decodeResolved } from '../../compiler/ir/resolved_codec.js'
import { assertAcyclic } from '../../compiler/ir/acyclic.js'
import { specializeProgram } from '../../compiler/ir/specialize.js'
import { sumLower } from '../../compiler/ir/sum_lower.js'
import { inlineInstances } from '../../compiler/ir/inline_instances.js'
import { arrayLower } from '../../compiler/ir/array_lower.js'
import { identityElim } from '../../compiler/ir/identity_elim.js'
import type { Program as ParsedProgram } from '../../compiler/parse/nodes.js'

const PARSED_DIR = 'stdlib/parsed'
const PASS_COUNT = 5

interface Flags {
  from: number
  upto: number
  mode: 'inline' | 'nested'
  typeArgs: Record<string, number>
}

function parseFlags(args: string[]): Flags {
  const flags: Flags = { from: 0, upto: PASS_COUNT, mode: 'inline', typeArgs: {} }
  for (const arg of args) {
    if (arg.startsWith('--from=')) flags.from = Number(arg.slice(7))
    else if (arg.startsWith('--upto=')) flags.upto = Number(arg.slice(7))
    else if (arg.startsWith('--mode=')) {
      const m = arg.slice(7)
      if (m !== 'inline' && m !== 'nested') throw new HarnessError(`unknown --mode=${m}`)
      flags.mode = m
    } else if (arg.startsWith('--type-args=')) {
      flags.typeArgs = JSON.parse(arg.slice(12)) as Record<string, number>
    } else if (arg.startsWith('--')) {
      throw new HarnessError(`unknown flag ${arg}`)
    }
  }
  if (!Number.isInteger(flags.from) || flags.from < 0 || flags.from > PASS_COUNT) {
    throw new HarnessError(`--from must be 0..${PASS_COUNT}`)
  }
  if (!Number.isInteger(flags.upto) || flags.upto < flags.from || flags.upto > PASS_COUNT) {
    throw new HarnessError(`--upto must be ${flags.from}..${PASS_COUNT}`)
  }
  return flags
}

/** Harness problems (bad flags, missing files) — exit 1, never `{error}`. */
class HarnessError extends Error {}

/** Resolve by-name type args against the program's own decl identities.
 *  Unknown names mint a fresh decl: specializeProgram then produces its
 *  byte-exact "not a declared type-param" error as a comparable output. */
function argsByName(
  prog: ResolvedProgram,
  byName: Record<string, number>,
): Map<TypeParamDecl, number> {
  const m = new Map<TypeParamDecl, number>()
  for (const [name, value] of Object.entries(byName)) {
    const decl = prog.typeParams.find(tp => tp.name === name)
    m.set(decl ?? { op: 'typeParamDecl', name }, value)
  }
  return m
}

/** Run passes `from..upto` (exclusive of pass indices >= upto). The
 *  composition at (0, 5) is strataPipeline's, spelled pass-by-pass so
 *  the suffix can resume anywhere. */
function runStrataRange(
  prog: ResolvedProgram,
  flags: Flags,
): ResolvedProgram {
  let p = prog
  const want = (k: number) => flags.from < k && flags.upto >= k
  if (want(1)) {
    assertAcyclic(p)
    p = specializeProgram(p, argsByName(p, flags.typeArgs))
  }
  if (want(2)) p = sumLower(p)
  if (want(3)) p = flags.mode === 'inline' ? inlineInstances(p) : p
  if (want(4)) p = arrayLower(p)
  if (want(5)) p = identityElim(p)
  return p
}

function manifestNames(): string[] {
  const manifest = JSON.parse(readFileSync(`${PARSED_DIR}/manifest.json`, 'utf-8')) as {
    programs: string[]
  }
  return manifest.programs
}

/** Raw-elaborated stdlib chain in manifest order — elab_cmd.ts's. */
function elabChain(names: string[]): Map<string, ResolvedProgram> {
  const resolved = new Map<string, ResolvedProgram>()
  const resolver: ExternalProgramResolver = name => resolved.get(name)
  for (const name of names) {
    const parsed = JSON.parse(
      readFileSync(`${PARSED_DIR}/${name}.json`, 'utf-8'),
    ) as ParsedProgram
    resolved.set(name, elaborate(parsed, resolver))
  }
  return resolved
}

function main(argv: string[]): number {
  const [verb, arg, ...rest] = argv
  const usage = 'usage: strata_cmd.ts strata-stdlib <Name> | strata-file <parsed.json> | strata-suffix <resolved.json> [--from=K] [--upto=K] [--mode=inline|nested] [--type-args={"N":8}]'
  if ((verb !== 'strata-stdlib' && verb !== 'strata-file' && verb !== 'strata-suffix') || arg === undefined) {
    console.error(usage)
    return 1
  }

  let flags: Flags
  try {
    flags = parseFlags(rest)
  } catch (e) {
    console.error(e instanceof Error ? e.message : String(e))
    return 1
  }
  if (verb !== 'strata-suffix' && flags.from !== 0) {
    console.error(`${verb} does not take --from`)
    return 1
  }

  let out: unknown
  try {
    let prog: ResolvedProgram
    if (verb === 'strata-stdlib') {
      const names = manifestNames()
      const i = names.indexOf(arg)
      if (i < 0) {
        console.error(`strata-stdlib: '${arg}' is not in ${PARSED_DIR}/manifest.json`)
        return 1
      }
      prog = elabChain(names.slice(0, i + 1)).get(arg)!
    } else if (verb === 'strata-file') {
      const chain = elabChain(manifestNames())
      const resolver: ExternalProgramResolver = name => chain.get(name)
      const parsed = JSON.parse(readFileSync(arg, 'utf-8')) as ParsedProgram
      prog = elaborate(parsed, resolver)
    } else {
      prog = decodeResolved(JSON.parse(readFileSync(arg, 'utf-8')))
    }
    out = encodeResolved(runStrataRange(prog, flags))
  } catch (e) {
    if (e instanceof HarnessError || !(e instanceof Error)) throw e
    out = { error: e.message }
  }
  // Synchronous write to fd 1: post-strata outputs (unrolled arrays,
  // inlined instances) exceed the pipe buffer, and `process.exit`
  // after an async stdout write truncates them.
  writeFileSync(1, JSON.stringify(out) + '\n')
  return 0
}

process.exit(main(process.argv.slice(2)))
