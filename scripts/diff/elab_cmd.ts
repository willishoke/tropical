/**
 * elab_cmd.ts — the TS side of the elaborator differential gate
 * (Phase 4 stage 3).
 *
 * Two verbs, mirroring `lean/.lake/build/bin/diffcli`:
 *
 *   elab-stdlib <Name>      Elaborate the pre-parsed stdlib bridge
 *                           (stdlib/parsed/) in manifest order up to
 *                           and including <Name>, threading an
 *                           ExternalProgramResolver over the raw
 *                           elaborated results (elaborate-only — NOT
 *                           loadProgramAsType/strata). Print the
 *                           `encodeResolved` JSON of <Name>.
 *
 *   elab-file <parsed.json> Elaborate a self-contained ParsedProgram
 *                           JSON file. The resolver is the full stdlib
 *                           raw-elaborated chain, so fixtures may
 *                           instantiate stdlib types.
 *
 * An ElaborationError or CycleViolation prints `{error: <message>}`
 * and exits 0 — error strings are comparable outputs, the same
 * convention as diff_raise. Anything else (fs errors, codec bugs)
 * propagates as a harness error.
 *
 * Usage:  bun run scripts/diff/elab_cmd.ts <verb> <arg>
 */

import { readFileSync } from 'node:fs'
import { elaborate, type ExternalProgramResolver } from '../../compiler/ir/elaborator.js'
import { ElaborationError, type ResolvedProgram } from '../../compiler/ir/nodes.js'
import { CycleViolation } from '../../compiler/ir/elaboration_diagnostics.js'
import { encodeResolved } from '../../compiler/ir/resolved_codec.js'
import type { Program as ParsedProgram } from '../../compiler/parse/nodes.js'

const PARSED_DIR = 'stdlib/parsed'

function manifestNames(): string[] {
  const manifest = JSON.parse(readFileSync(`${PARSED_DIR}/manifest.json`, 'utf-8')) as {
    programs: string[]
  }
  return manifest.programs
}

/** Raw-elaborated stdlib chain in manifest order: each program is
 *  elaborated with a resolver over the previously elaborated results.
 *  Manifest order is dependency-respecting by construction
 *  (scripts/build_parsed_stdlib.ts). */
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
  const [verb, arg] = argv
  if ((verb !== 'elab-stdlib' && verb !== 'elab-file') || arg === undefined) {
    console.error('usage: elab_cmd.ts elab-stdlib <Name> | elab-file <parsed.json>')
    return 1
  }

  let out: unknown
  try {
    if (verb === 'elab-stdlib') {
      const names = manifestNames()
      const i = names.indexOf(arg)
      if (i < 0) {
        console.error(`elab-stdlib: '${arg}' is not in ${PARSED_DIR}/manifest.json`)
        return 1
      }
      const chain = elabChain(names.slice(0, i + 1))
      out = encodeResolved(chain.get(arg)!)
    } else {
      const chain = elabChain(manifestNames())
      const resolver: ExternalProgramResolver = name => chain.get(name)
      const parsed = JSON.parse(readFileSync(arg, 'utf-8')) as ParsedProgram
      out = encodeResolved(elaborate(parsed, resolver))
    }
  } catch (e) {
    if (e instanceof ElaborationError || e instanceof CycleViolation) {
      out = { error: e.message }
    } else {
      throw e
    }
  }
  process.stdout.write(JSON.stringify(out) + '\n')
  return 0
}

process.exit(main(process.argv.slice(2)))
