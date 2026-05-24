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
 *  Default behavior uses `inlineNested: true` — the legacy flat-IR
 *  path. The fractal session path (M11 activation) passes
 *  `inlineNested: false` so sub-instance `InstanceDecl`s survive
 *  into `partition_recursive`, where they become real kernel
 *  boundaries instead of being splatted into their parent's body.
 *  The slot-based input-wiring redesign makes the `false` path
 *  correct end-to-end; until that lands, callers should leave
 *  `inlineNested` at its default. */
export function programTypeFromResolved(
  prog: ResolvedProgram,
  typeArgs: ReadonlyMap<TypeParamDecl, number>,
  opts?: { displayName?: string; inlineNested?: boolean },
): Compiled {
  // Post-Phase 4b: the elaborator throws on cyclic source code, and
  // lifted wire-programs are acyclic by construction (no
  // InstanceDecls). The strataPipeline's `assertAcyclic` at entry
  // confirms the contract; no separate cycle-break call is needed
  // here.
  const strataOptions: StrataOptions = opts?.inlineNested !== undefined
    ? { inlineNested: opts.inlineNested } : {}
  return makeCompiled(strataPipeline(prog, typeArgs, strataOptions), opts)
}
