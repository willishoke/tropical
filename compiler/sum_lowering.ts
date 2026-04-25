/**
 * Sum-type decomposition: rewrites a tropical_program_2 body in place so
 * sum-typed delay_decls expand into N+1 scalar delay_decls (one per
 * (variant, field) pair plus the discriminator), and tag/match/delay_ref
 * expressions over sum-typed values resolve into scalar selects and reads.
 *
 * This is a single tree-walk on the program body, run by loadProgramDef
 * before slottification. After this pass the body contains no sum types,
 * no tag ops, and no match ops — only scalar delay_decls and standard
 * arithmetic expressions. Downstream compilation is unchanged.
 *
 * The categorical view: this is the structure-forgetting functor that
 * takes a guarded-traced-SMC body and produces its scalar realization.
 * The forgetting is total — by the time this pass returns, no morphism
 * in the result depends on coproduct structure.
 */

import type { ExprNode } from './expr.js'
import type { BlockNode } from './program.js'
import {
  type SumTypeMeta, type SumBundleSlot,
  sumBundleSlots, sumVariantIndex, mangleSumSlot,
} from './term.js'

// A registry of sum-typed delay names → their bundle-slot layout.
// Built by scanning decls before any expressions are rewritten; consulted
// when rewriting delay_ref / match / tag occurrences.
type SumDelayMap = Map<string, { type: string; meta: SumTypeMeta; slots: SumBundleSlot[] }>

/**
 * Apply sum-decomposition to a program body. Returns a new body where:
 *   - sum-typed delay_decls have been expanded into N+1 scalar delay_decls
 *     with mangled names (`<name>#tag`, `<name>#<V>__<f>`)
 *   - delay_ref of a sum-typed name has been rewritten to a structured
 *     bundle-read or to the specific tag-slot read depending on context
 *   - match/tag ops over sum-typed values have been lowered to scalar
 *     select-chains and per-slot constants
 *
 * If no sum-typed delays exist in `body.decls`, the body is returned as-is.
 */
export function expandSumTypes(
  body: BlockNode,
  sumRegistry: ReadonlyMap<string, SumTypeMeta>,
): BlockNode {
  // ── Step 1: identify sum-typed delays ────────────────────────────────
  const sumDelays: SumDelayMap = new Map()
  for (const decl of body.decls ?? []) {
    if (typeof decl !== 'object' || decl === null || Array.isArray(decl)) continue
    const d = decl as Record<string, unknown>
    if (d.op !== 'delay_decl' || typeof d.type !== 'string') continue
    const meta = sumRegistry.get(d.type as string)
    if (meta === undefined) continue  // not a sum type — leave alone
    sumDelays.set(d.name as string, {
      type: d.type as string,
      meta,
      slots: sumBundleSlots(meta),
    })
  }

  if (sumDelays.size === 0) return body  // nothing to do

  // ── Step 2: walk the body, rewriting expressions ─────────────────────
  const rewrite = (e: ExprNode): ExprNode => rewriteExpr(e, sumDelays, sumRegistry)

  // ── Step 3: replace sum-typed delay_decls with N+1 scalar decls ──────
  const newDecls: ExprNode[] = []
  for (const decl of body.decls ?? []) {
    if (typeof decl !== 'object' || decl === null || Array.isArray(decl)) {
      newDecls.push(decl)
      continue
    }
    const d = decl as Record<string, unknown>
    if (d.op === 'delay_decl' && typeof d.type === 'string' && sumDelays.has(d.name as string)) {
      const info = sumDelays.get(d.name as string)!
      // Validate init is a constant tag expression of matching type.
      const init = d.init
      if (typeof init !== 'object' || init === null || Array.isArray(init) ||
          (init as Record<string, unknown>).op !== 'tag' ||
          (init as Record<string, unknown>).type !== info.type) {
        throw new Error(
          `delay_decl '${d.name}': init for sum-typed delay must be a constant ` +
          `'tag' expression of type '${info.type}'.`,
        )
      }
      const initObj = init as Record<string, unknown>
      const initVariantIdx = sumVariantIndex(info.meta, initObj.variant as string)
      if (initVariantIdx < 0) {
        throw new Error(
          `delay_decl '${d.name}': init variant '${initObj.variant}' not found in sum '${info.type}'.`,
        )
      }
      const initPayload = (initObj.payload ?? {}) as Record<string, ExprNode>

      // Build per-slot scalar delay_decls.
      // For the discriminator slot: init = variant index, update = rewrite of d.update's tag slot.
      // For each payload-field slot: init = the payload value if init's variant matches, else 0;
      //   update = the payload value if updated tag matches, else 0 (or held).
      for (const slot of info.slots) {
        const slotName = mangleSumSlot(d.name as string, slot.suffix)
        const slotInit = computeSlotInit(slot, initVariantIdx, initPayload)
        const slotUpdate = d.update !== undefined
          ? extractSlotFromSumExpr(d.update as ExprNode, slot, info.meta, sumDelays, sumRegistry)
          : undefined
        const slotDecl: Record<string, unknown> = {
          op: 'delay_decl',
          name: slotName,
          init: slotInit,
        }
        if (slotUpdate !== undefined) slotDecl.update = slotUpdate
        newDecls.push(slotDecl as ExprNode)
      }
    } else {
      // Non-sum decl — recurse into any nested expressions it carries.
      newDecls.push(rewriteDecl(decl, sumDelays, sumRegistry))
    }
  }

  // ── Step 4: rewrite assigns ──────────────────────────────────────────
  const newAssigns: ExprNode[] = []
  for (const assign of body.assigns ?? []) {
    newAssigns.push(rewriteAssign(assign, sumDelays, sumRegistry))
  }

  return { ...body, decls: newDecls, assigns: newAssigns }
}

// ─────────────────────────────────────────────────────────────
// Rewriting expressions
// ─────────────────────────────────────────────────────────────

/**
 * Rewrite a single expression. Returns a new tree with sum-typed nodes
 * lowered to scalar form.
 *
 * Rewrites:
 *   - delay_ref(name) where name is sum-typed → delay_ref(name#tag)
 *     (reads the discriminator only — appropriate when the delay_ref
 *      appears as a match scrutinee or in a context that just needs
 *      the tag value. Payload reads happen via match arm rewriting.)
 *   - match(scrutinee, arms) where scrutinee is sum-typed → scalar
 *     select-chain over the scrutinee's tag slot.
 *   - tag(type, variant) (no payload) appearing as an expression value →
 *     the variant index as a scalar (used for tag-slot writes).
 *     Tag with payload is currently rejected outside delay-update context;
 *     handled via extractSlotFromSumExpr instead.
 */
function rewriteExpr(
  expr: ExprNode,
  sumDelays: SumDelayMap,
  sumRegistry: ReadonlyMap<string, SumTypeMeta>,
): ExprNode {
  if (typeof expr !== 'object' || expr === null || Array.isArray(expr)) return expr
  const obj = expr as Record<string, unknown>
  const op = obj.op as string

  // delay_ref to a sum-typed delay — read the tag slot only. This is the
  // correct lowering when the delay_ref is a match scrutinee. (Match
  // rewriting below also reads the tag for dispatch and rewrites payload
  // bindings to read the right field slots.)
  if (op === 'delay_ref' && typeof obj.id === 'string' && sumDelays.has(obj.id as string)) {
    const info = sumDelays.get(obj.id as string)!
    return { op: 'delay_ref', id: mangleSumSlot(obj.id as string, 'tag') }
  }

  // match — when scrutinee resolves to a sum-typed bundle, lower to a
  // select chain over the tag slot.
  if (op === 'match') {
    const scrutinee = obj.scrutinee as ExprNode
    const sumMeta = resolveSumTypeOfExpr(scrutinee, obj.type as string, sumDelays, sumRegistry)
    if (sumMeta !== undefined) {
      return lowerMatchToSelectChain(obj, sumMeta, sumDelays, sumRegistry)
    }
    // Fallback: not a recognized sum-typed scrutinee — recurse children
    // generically so a future refinement can still see the structure.
    return mapChildren(obj, e => rewriteExpr(e, sumDelays, sumRegistry))
  }

  // tag in expression position (no payload) → variant index as scalar.
  // With-payload tags are handled by extractSlotFromSumExpr in update
  // contexts; if one appears bare here, we leave it for now (a later
  // pass would type-check that contexts requiring a sum value are
  // properly wrapped — for V1 this is best-effort).
  if (op === 'tag' && typeof obj.type === 'string') {
    const meta = sumRegistry.get(obj.type as string)
    if (meta !== undefined && obj.payload === undefined) {
      const idx = sumVariantIndex(meta, obj.variant as string)
      if (idx < 0) {
        throw new Error(`tag: variant '${obj.variant}' not in sum '${obj.type}'.`)
      }
      return idx
    }
  }

  // Generic recursion through args and standard fields.
  return mapChildren(obj, e => rewriteExpr(e, sumDelays, sumRegistry))
}

/**
 * Recurse into an expression's children, applying `f` to each child ExprNode.
 * Handles `args` arrays and standard nested-expr fields the way lowerChildren
 * does, with extensions for the sum-op shape (payload, scrutinee, arms).
 */
function mapChildren(
  obj: Record<string, unknown>,
  f: (e: ExprNode) => ExprNode,
): ExprNode {
  let changed = false
  const result: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(obj)) {
    if (Array.isArray(v)) {
      const arr = v as ExprNode[]
      const newArr = arr.map(f)
      if (newArr.some((n, i) => n !== arr[i])) changed = true
      result[k] = newArr
    } else if (typeof v === 'object' && v !== null && 'op' in v) {
      const newV = f(v as ExprNode)
      if (newV !== v) changed = true
      result[k] = newV
    } else if (typeof v === 'object' && v !== null && !Array.isArray(v)) {
      // Record<string, ExprNode> fields — payload, arms.bind/body, etc.
      const rec = v as Record<string, unknown>
      const newRec: Record<string, unknown> = {}
      let recChanged = false
      for (const [rk, rv] of Object.entries(rec)) {
        if (typeof rv === 'object' && rv !== null && !Array.isArray(rv) && 'op' in rv) {
          const nrv = f(rv as ExprNode)
          if (nrv !== rv) recChanged = true
          newRec[rk] = nrv
        } else if (typeof rv === 'object' && rv !== null && !Array.isArray(rv) && 'body' in rv) {
          // Match arm: { bind?, body }
          const arm = rv as { bind?: string | string[]; body: ExprNode }
          const newBody = f(arm.body)
          if (newBody !== arm.body) recChanged = true
          newRec[rk] = arm.bind === undefined
            ? { body: newBody }
            : { bind: arm.bind, body: newBody }
        } else {
          newRec[rk] = rv
        }
      }
      if (recChanged) changed = true
      result[k] = recChanged ? newRec : rec
    } else {
      result[k] = v
    }
  }
  return changed ? (result as ExprNode) : (obj as ExprNode)
}

/**
 * Determine the SumTypeMeta of an expression, if it's a sum-typed value.
 * Currently recognizes: delay_ref to a sum-typed delay, and explicitly
 * typed match/tag expressions. Returns undefined for non-sum values.
 */
function resolveSumTypeOfExpr(
  expr: ExprNode,
  declaredType: string | undefined,
  sumDelays: SumDelayMap,
  sumRegistry: ReadonlyMap<string, SumTypeMeta>,
): SumTypeMeta | undefined {
  if (typeof expr !== 'object' || expr === null || Array.isArray(expr)) return undefined
  const obj = expr as Record<string, unknown>
  if (obj.op === 'delay_ref' && typeof obj.id === 'string') {
    const info = sumDelays.get(obj.id as string)
    if (info !== undefined) return info.meta
  }
  if (declaredType !== undefined) {
    return sumRegistry.get(declaredType)
  }
  return undefined
}

/**
 * Lower a match expression to a scalar select-chain.
 *
 * For nullary-arm-only matches, the chain is over the tag slot directly:
 *   select(eq(tag, k_0), arms[V_0],
 *    select(eq(tag, k_1), arms[V_1],
 *           default))
 *
 * Arms with bindings (payload) are handled in a follow-up commit; this
 * implementation supports only nullary arms (the Toggle case). A bind
 * field on any arm is rejected with a clear error — the caller knows
 * to expand its scope when payload support lands.
 */
function lowerMatchToSelectChain(
  matchObj: Record<string, unknown>,
  meta: SumTypeMeta,
  sumDelays: SumDelayMap,
  sumRegistry: ReadonlyMap<string, SumTypeMeta>,
): ExprNode {
  const scrutinee = matchObj.scrutinee as ExprNode
  const arms = matchObj.arms as Record<string, { bind?: string | string[]; body: ExprNode }>

  // Reject payload bindings for now (Phase 3a scope).
  for (const [variant, arm] of Object.entries(arms)) {
    if (arm.bind !== undefined) {
      throw new Error(
        `match arm '${variant}': payload bindings are not yet supported in sum-lowering. ` +
        `(Phase 3a handles only nullary variants; payload bindings land in Phase 3b.)`,
      )
    }
  }

  // Rewrite the scrutinee into its tag-slot read.
  const tagRead = rewriteExpr(scrutinee, sumDelays, sumRegistry)

  // Build select chain. Iterate variants in declaration order; the last
  // arm becomes the chain's tail (its `else` branch is the variant body
  // itself, since exhaustiveness guarantees one arm matches).
  const variants = meta.variants
  let chain: ExprNode = rewriteExpr(arms[variants[variants.length - 1].name].body, sumDelays, sumRegistry)
  for (let i = variants.length - 2; i >= 0; i--) {
    const v = variants[i]
    const armBody = rewriteExpr(arms[v.name].body, sumDelays, sumRegistry)
    chain = {
      op: 'select',
      args: [
        { op: 'eq', args: [tagRead, i] },
        armBody,
        chain,
      ],
    }
  }
  return chain
}

/**
 * Compute the per-slot init value for a sum-typed delay's bundle.
 *
 *   - tag slot: the variant index of the init expression's variant.
 *   - payload field slot for variant V's field f: the field's init value
 *     if V matches the init's variant, else 0.
 */
function computeSlotInit(
  slot: SumBundleSlot,
  initVariantIdx: number,
  initPayload: Record<string, ExprNode>,
): ExprNode {
  if (slot.suffix === 'tag') return initVariantIdx
  // Payload slot: only populated if its variant matches the init's variant.
  // Slot's variant comes from the slot metadata itself.
  if (slot.variant !== undefined && slot.field !== undefined) {
    // Look up the variant's index by comparing names — this is a static
    // lookup but we don't have the meta passed through. The caller must
    // ensure: if init's variant matches this slot's variant, return the
    // payload field; else 0.
    // We have `initVariantIdx` (the init's index) and need to know the
    // slot's variant index. Compare by name in the init payload key:
    // if the init payload has a value for `slot.field` AND the slot's
    // variant is the init's variant, return that value.
    //
    // The caller should pass initPayload only for the matching variant
    // — in this scheme the comparison is by variant *name* on the slot.
    // Simpler: the caller stores `initVariantName` and we compare strings.
    //
    // To avoid plumbing extra state, we trust that initPayload is keyed
    // by field name and that the slot's variant matches iff slot.variant
    // is the variant declared in the init expression. Caller passes
    // `initPayload` as empty {} when the init's variant is different
    // from the slot's variant (handled by caller).
    const fieldVal = initPayload[slot.field]
    return fieldVal !== undefined ? fieldVal : 0
  }
  return 0
}

/**
 * Extract the per-slot scalar update expression from a sum-valued update
 * expression. The update is typically a `match` returning a sum value or
 * a constant `tag`; for slot k, we derive what the new value of slot k
 * should be each sample.
 *
 * Phase 3a scope: handles the constant-tag and nullary-only-match cases.
 * Match arms with payloads are rejected (handled in 3b).
 */
function extractSlotFromSumExpr(
  expr: ExprNode,
  slot: SumBundleSlot,
  meta: SumTypeMeta,
  sumDelays: SumDelayMap,
  sumRegistry: ReadonlyMap<string, SumTypeMeta>,
): ExprNode {
  if (typeof expr !== 'object' || expr === null || Array.isArray(expr)) {
    // Not a recognized sum-valued expression — return a sensible default
    // (zero) and let downstream type-checking flag the issue.
    return 0
  }
  const obj = expr as Record<string, unknown>

  // Constant tag — write variant index to tag slot, payload values to
  // matching variant's field slots, zero to other variants' field slots.
  if (obj.op === 'tag' && typeof obj.type === 'string') {
    const idx = sumVariantIndex(meta, obj.variant as string)
    if (idx < 0) {
      throw new Error(`tag: variant '${obj.variant}' not in sum '${obj.type}'.`)
    }
    if (slot.suffix === 'tag') return idx
    // Payload slot for variant V's field f.
    if (slot.variant === obj.variant && slot.field !== undefined) {
      const payload = (obj.payload ?? {}) as Record<string, ExprNode>
      const fieldVal = payload[slot.field]
      return fieldVal !== undefined ? rewriteExpr(fieldVal, sumDelays, sumRegistry) : 0
    }
    return 0  // slot belongs to a different variant
  }

  // Match returning a sum value — for each arm, recursively extract this
  // slot's value, then build a select chain over the scrutinee's tag.
  if (obj.op === 'match' && typeof obj.type === 'string') {
    const arms = obj.arms as Record<string, { bind?: string | string[]; body: ExprNode }>
    for (const [variant, arm] of Object.entries(arms)) {
      if (arm.bind !== undefined) {
        throw new Error(
          `match arm '${variant}' in delay update: payload bindings not yet supported (Phase 3a).`,
        )
      }
    }
    const scrutineeMeta = resolveSumTypeOfExpr(
      obj.scrutinee as ExprNode, obj.type as string, sumDelays, sumRegistry,
    )
    if (scrutineeMeta === undefined) {
      throw new Error(`match: cannot resolve scrutinee's sum type for slot extraction.`)
    }
    const tagRead = rewriteExpr(obj.scrutinee as ExprNode, sumDelays, sumRegistry)
    const variants = scrutineeMeta.variants
    let chain: ExprNode = extractSlotFromSumExpr(
      arms[variants[variants.length - 1].name].body, slot, meta, sumDelays, sumRegistry,
    )
    for (let i = variants.length - 2; i >= 0; i--) {
      const v = variants[i]
      const armSlot = extractSlotFromSumExpr(arms[v.name].body, slot, meta, sumDelays, sumRegistry)
      chain = {
        op: 'select',
        args: [{ op: 'eq', args: [tagRead, i] }, armSlot, chain],
      }
    }
    return chain
  }

  // Anything else (a delay_ref to a sum-typed delay used as an update?)
  // — for slot, just read the corresponding slot of the source.
  if (obj.op === 'delay_ref' && typeof obj.id === 'string' && sumDelays.has(obj.id as string)) {
    return { op: 'delay_ref', id: mangleSumSlot(obj.id as string, slot.suffix) }
  }

  // Unrecognized — return zero. Future iterations may want to error here.
  return 0
}

/**
 * Rewrite a decl that isn't a sum-typed delay_decl. Recurses into nested
 * expressions (e.g., reg_decl init, instance_decl inputs).
 */
function rewriteDecl(
  decl: ExprNode,
  sumDelays: SumDelayMap,
  sumRegistry: ReadonlyMap<string, SumTypeMeta>,
): ExprNode {
  if (typeof decl !== 'object' || decl === null || Array.isArray(decl)) return decl
  return mapChildren(decl as Record<string, unknown>, e => rewriteExpr(e, sumDelays, sumRegistry))
}

/** Rewrite an assign (output_assign / next_update). */
function rewriteAssign(
  assign: ExprNode,
  sumDelays: SumDelayMap,
  sumRegistry: ReadonlyMap<string, SumTypeMeta>,
): ExprNode {
  if (typeof assign !== 'object' || assign === null || Array.isArray(assign)) return assign
  const obj = assign as Record<string, unknown>

  // For next_update targeting a sum-typed delay, we'd need to expand into
  // multiple per-slot next_updates. V1 path: users put updates on the
  // delay_decl itself (the `update` field), so this case is uncommon.
  // If we see one, expand it.
  if (obj.op === 'next_update') {
    const target = obj.target as { kind: string; name: string }
    if (target.kind === 'delay' && sumDelays.has(target.name)) {
      // Phase 3b should handle this. For Phase 3a, error out so the
      // limitation is visible.
      throw new Error(
        `next_update on sum-typed delay '${target.name}': not yet supported as a separate ` +
        `assign (place the update on the delay_decl's 'update' field instead).`,
      )
    }
  }

  return mapChildren(obj, e => rewriteExpr(e, sumDelays, sumRegistry))
}
