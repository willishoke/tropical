/**
 * strata.ts — runtime IR pipeline orchestrator.
 *
 * Composes the strata passes in order:
 *
 *   assertAcyclic → specialize → sumLower → inlineInstances
 *                 → arrayLower → identityElim
 *
 * Post-Phase 3, `strataPipeline` is a *checking* pipeline: it requires
 * its input to be already-acyclic. Cycle-breaking is the responsibility
 * of the caller (the "standard realization" — elaborator output and
 * session materialization). The two entry points
 * (`programTypeFromResolved` and `materializeSession`) run
 * `breakInstanceCycles` on their inputs before invoking
 * `strataPipeline`; the `assertAcyclic` at strata entry guarantees
 * any escape would surface immediately rather than producing a
 * malformed plan.
 *
 * `identityElim` (M11 Phase 2) is the categorical identity-law rewrite:
 * it eliminates `InstanceDecl`s whose program body is the identity
 * morphism. Today (pre-Phase-4) it rarely fires at the per-program
 * level — `inlineInstances` runs first and absorbs trivial sub-instances.
 * Once Phase 4 retires `inlineInstances` from the session-level
 * pipeline, `identityElim` catches the trivial wire-programs that would
 * otherwise survive as no-op kernels.
 *
 * After identityElim, the result is a post-strata `ResolvedProgram`
 * consumed by either `compileResolved` (the JIT path) or
 * `interpret_resolved.ts` (the independent oracle).
 */

import type { ResolvedProgram, TypeParamDecl } from './nodes.js'
import { type Compiled, makeCompiled } from '../program_types.js'
import { specializeProgram } from './specialize.js'
import { sumLower } from './sum_lower.js'
import { inlineInstances } from './inline_instances.js'
import { arrayLower } from './array_lower.js'
import { identityElim } from './identity_elim.js'
import { assertAcyclic } from './acyclic.js'
import { breakInstanceCycles } from './lowering/cycle_break.js'

export interface StrataOptions {
  /** Whether to flatten nested `InstanceDecl`s via `inlineInstances`.
   *  Default `true` (legacy / per-program-emit behavior). The fractal
   *  session path passes `false` so sub-instances survive as kernel
   *  boundaries — every `InstanceDecl` at every level becomes its own
   *  kernel in `partition_recursive`. */
  inlineNested?: boolean
}

export function strataPipeline(
  prog: ResolvedProgram,
  typeArgs: ReadonlyMap<TypeParamDecl, number> = new Map(),
  options: StrataOptions = {},
): ResolvedProgram {
  const { inlineNested = true } = options
  // Acyclicity is the strataPipeline contract — callers must run
  // `breakInstanceCycles` (or equivalent) before invoking this. The
  // standard realization's call sites (`programTypeFromResolved` and
  // `materializeSession`) do this for the user.
  assertAcyclic(prog)
  const specialized = specializeProgram(prog, typeArgs)
  const summed = sumLower(specialized)
  const inlined = inlineNested ? inlineInstances(summed) : summed
  const arrayed = arrayLower(inlined)
  return identityElim(arrayed)
}

/** Run the full strata pipeline + wrap the resulting `ResolvedProgram`
 *  in a `Compiled` record. Helpers in `program_types.ts` expose
 *  port/register/default metadata via free functions over the resolved
 *  IR.
 *
 *  Currently uses default inlining (`inlineNested: true`). The fractal
 *  architecture (M11) is fully infrastructured — `partition_recursive`,
 *  `NestedOut → slot read` in emit_resolved, recursive JIT emit, schema
 *  field for `children` — but ACTIVATION (flipping to `inlineNested:
 *  false`) is deferred: the cross-kernel input-wiring path (child
 *  refers to parent's `inputRef`) needs an ancestor-resolution step
 *  before partition. Once that lands, set `inlineNested: false` here
 *  and full fractal activates uniformly. */
export function programTypeFromResolved(
  prog: ResolvedProgram,
  typeArgs: ReadonlyMap<TypeParamDecl, number>,
  opts?: { displayName?: string },
): Compiled {
  // Post-Phase 3: the cycle-break helper runs before strataPipeline.
  // Single edit point that propagates to all callers; Phase 4b will
  // replace this with strict elaborator-level cycle detection.
  const acyclic = breakInstanceCycles(prog).lowered
  return makeCompiled(strataPipeline(acyclic, typeArgs), opts)
}
