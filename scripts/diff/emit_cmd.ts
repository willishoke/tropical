/**
 * emit_cmd.ts — the TS side of the per-program emit differential gate
 * (Phase 6 stage 6b).
 *
 * Two verbs mirroring `diffcli emit-stdlib` / `emit-file`:
 *
 *   emit-stdlib <Name>        [--type-args=J]
 *   emit-file <parsed.json>   [--type-args=J]
 *
 * Each elaborates exactly like elab_cmd.ts, runs the FULL strata
 * pipeline (inline mode — the production per-program realization),
 * then `compileResolved(prog, {})` and prints the wire-format
 * `PerInstancePlan` JSON. The fractal emit paths (nested slot maps,
 * per-child pre-input) are not reachable from a standalone program
 * post-inline; they're gated at stage 6c/6d through the partitioner.
 *
 * Strata/emit errors print `{error: <message>}` and exit 0 (errors are
 * comparable outputs); file/flag problems are harness errors (exit 1).
 *
 * Usage:  bun run scripts/diff/emit_cmd.ts <verb> <arg> [--type-args={"N":8}]
 */

import { readFileSync, writeFileSync } from 'node:fs'
import { elaborate, type ExternalProgramResolver } from '../../compiler/ir/elaborator.js'
import type { ResolvedProgram, TypeParamDecl } from '../../compiler/ir/nodes.js'
import { assertAcyclic } from '../../compiler/ir/acyclic.js'
import { specializeProgram } from '../../compiler/ir/specialize.js'
import { sumLower } from '../../compiler/ir/sum_lower.js'
import { inlineInstances } from '../../compiler/ir/inline_instances.js'
import { arrayLower } from '../../compiler/ir/array_lower.js'
import { identityElim } from '../../compiler/ir/identity_elim.js'
import { compileResolved } from '../../compiler/ir/compile_resolved.js'
import type { PerInstancePlan } from '../../compiler/flat_plan.js'
import { toWireRegTarget } from '../../compiler/flat_plan.js'
import { toWireInstr } from '../../compiler/ir/emit_resolved.js'
import type { Program as ParsedProgram } from '../../compiler/parse/nodes.js'

const PARSED_DIR = 'stdlib/parsed'

/** Harness problems (bad flags, missing files) — exit 1, never `{error}`. */
class HarnessError extends Error {}

function parseTypeArgs(args: string[]): Record<string, number> {
  for (const arg of args) {
    if (arg.startsWith('--type-args=')) {
      return JSON.parse(arg.slice(12)) as Record<string, number>
    }
    if (arg.startsWith('--')) throw new HarnessError(`unknown flag ${arg}`)
  }
  return {}
}

/** strata_cmd.ts's by-name type-arg resolution (identity contract). */
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

/** The full pipeline at inline mode (strataPipeline's composition). */
function runStrata(prog: ResolvedProgram, typeArgs: Record<string, number>): ResolvedProgram {
  assertAcyclic(prog)
  let p = specializeProgram(prog, argsByName(prog, typeArgs))
  p = sumLower(p)
  p = inlineInstances(p)
  p = arrayLower(p)
  p = identityElim(p)
  return p
}

function manifestNames(): string[] {
  const manifest = JSON.parse(readFileSync(`${PARSED_DIR}/manifest.json`, 'utf-8')) as {
    programs: string[]
  }
  return manifest.programs
}

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

/** Wire encoding of a PerInstancePlan (no production wire format
 *  exists — this shape is the gate's comparable output, mirrored by
 *  `Tropical.Plan.PerInstancePlan.toWire`). */
function toWirePerInstancePlan(plan: PerInstancePlan): unknown {
  return {
    register_count: plan.register_count,
    array_slot_count: plan.array_slot_count,
    array_slot_sizes: plan.array_slot_sizes,
    instructions: plan.instructions.map(toWireInstr),
    per_child_pre_input: plan.per_child_pre_input.map(b => b.map(toWireInstr)),
    output_targets: plan.output_targets,
    register_targets: plan.register_targets.map(toWireRegTarget),
    state_init: plan.state_init,
    register_names: plan.register_names,
    register_types: plan.register_types,
    array_slot_names: plan.array_slot_names,
  }
}

function main(argv: string[]): number {
  const [verb, arg, ...rest] = argv
  const usage = 'usage: emit_cmd.ts emit-stdlib <Name> | emit-file <parsed.json> [--type-args={"N":8}]'
  if ((verb !== 'emit-stdlib' && verb !== 'emit-file') || arg === undefined) {
    console.error(usage)
    return 1
  }

  let typeArgs: Record<string, number>
  try {
    typeArgs = parseTypeArgs(rest)
  } catch (e) {
    console.error(e instanceof Error ? e.message : String(e))
    return 1
  }

  let out: unknown
  try {
    let prog: ResolvedProgram
    if (verb === 'emit-stdlib') {
      const names = manifestNames()
      const i = names.indexOf(arg)
      if (i < 0) {
        console.error(`emit-stdlib: '${arg}' is not in ${PARSED_DIR}/manifest.json`)
        return 1
      }
      prog = elabChain(names.slice(0, i + 1)).get(arg)!
    } else {
      const chain = elabChain(manifestNames())
      const resolver: ExternalProgramResolver = name => chain.get(name)
      const parsed = JSON.parse(readFileSync(arg, 'utf-8')) as ParsedProgram
      prog = elaborate(parsed, resolver)
    }
    out = toWirePerInstancePlan(compileResolved(runStrata(prog, typeArgs), {}))
  } catch (e) {
    if (e instanceof HarnessError || !(e instanceof Error)) throw e
    out = { error: e.message }
  }
  // Synchronous write to fd 1 (the strata_cmd.ts truncation lesson).
  writeFileSync(1, JSON.stringify(out) + '\n')
  return 0
}

process.exit(main(process.argv.slice(2)))
