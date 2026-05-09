/**
 * strata.ts — runtime IR pipeline orchestrator.
 *
 * Composes the strata passes in order:
 *
 *   specialize → sumLower → traceCycles → inlineInstances → arrayLower
 *
 * After arrayLower, the result is a post-strata `ResolvedProgram`
 * consumed by either `compileResolved` (the JIT path) or
 * `interpret_resolved.ts` (the independent oracle).
 */

import type { ResolvedProgram, TypeParamDecl } from './nodes.js'
import { type Compiled, makeCompiled } from '../program_types.js'
import { specializeProgram } from './specialize.js'
import { sumLower } from './sum_lower.js'
import { traceCycles } from './trace_cycles.js'
import { inlineInstances } from './inline_instances.js'
import { arrayLower } from './array_lower.js'

export function strataPipeline(
  prog: ResolvedProgram,
  typeArgs: ReadonlyMap<TypeParamDecl, number> = new Map(),
): ResolvedProgram {
  const specialized = specializeProgram(prog, typeArgs)
  const summed = sumLower(specialized)
  const cyclic = traceCycles(summed)
  const inlined = inlineInstances(cyclic)
  return arrayLower(inlined)
}

/** Run the full strata pipeline + wrap the post-strata `ResolvedProgram`
 *  in a `Compiled` record. Helpers in `program_types.ts` expose
 *  port/register/default metadata via free functions over the resolved
 *  IR — no slot-indexed flattening upfront. The optional `displayName`
 *  rebrands the wrapped name for the specialization cache (e.g.
 *  `Type<N=8>`) without mutating `prog.name`. */
export function programTypeFromResolved(
  prog: ResolvedProgram,
  typeArgs: ReadonlyMap<TypeParamDecl, number>,
  opts?: { displayName?: string },
): Compiled {
  return makeCompiled(strataPipeline(prog, typeArgs), opts)
}
