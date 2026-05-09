/**
 * sum_lower.ts — Phase C4: sum-type decomposition on the resolved IR.
 *
 * Decomposes every sum-typed `DelayDecl` (one whose `init` is a
 * `Tag`) into N+1 scalar `DelayDecl`s — a discriminator slot
 * (int) plus one slot per (variant, field) pair across all variants —
 * and lowers every `Match`/`Tag` to scalar select-chains and
 * variant-index literals.
 *
 * After this pass the program contains no `tag` or `match` expressions
 * and no sum-typed delays. Decl identity is preserved end-to-end:
 *   - Each replacement scalar `DelayDecl` is a fresh decl object.
 *   - References (`DelayRef`, `BindingRef`) are rewritten by
 *     identity replacement, never by name string.
 *
 * Algorithm mirrors `compiler/sum_lowering.ts` (legacy reference) so
 * the pre-`emit_numeric` slot layout of `EnvExpDecay`/`TriggerRamp`
 * matches the legacy pipeline byte-for-byte.
 *
 * Constraints (matching legacy):
 *   - A sum-typed delay's `init` MUST be a `Tag` (constant
 *     variant constructor). Anything else is a structural error.
 *   - Match-arm payload bindings are only supported when the
 *     scrutinee is a `DelayRef` to a sum-typed delay. Other
 *     scrutinee shapes throw.
 */

import type {
  ResolvedProgram, ResolvedExpr, ResolvedExprOp,
  ResolvedBlock,
  BodyDecl, BodyAssign, OutputAssign, NextUpdate,
  DelayDecl, BinderDecl,
  SumTypeDef, SumVariant, StructField,
  Tag, Match, MatchArm,
  DelayRef,
} from './nodes.js'

// ─────────────────────────────────────────────────────────────
// Sum-delay table — built once before any rewriting starts
// ─────────────────────────────────────────────────────────────

/** A single bundle slot replacing one sum-typed `DelayDecl`. */
interface SlotEntry {
  /** The fresh scalar `DelayDecl` for this slot. */
  decl: DelayDecl
  /** Variant this payload slot belongs to; undefined for the tag slot. */
  variant?: SumVariant
  /** Payload field this slot represents; undefined for the tag slot. */
  field?: StructField
}

/** Per-original-DelayDecl decomposition. */
interface SumDelayInfo {
  original: DelayDecl
  sumType: SumTypeDef
  /** Slot order: [tag, ...per-variant per-field-in-payload-order]. */
  slots: SlotEntry[]
  /** Convenience lookup: the single tag slot. */
  tagSlot: SlotEntry
  /** Lookup keyed by `${variantName}__${fieldName}` for payload slots. */
  payloadByKey: Map<string, SlotEntry>
}

type SumDelayMap = Map<DelayDecl, SumDelayInfo>

// ─────────────────────────────────────────────────────────────
// Public entry
// ─────────────────────────────────────────────────────────────

export function sumLower(prog: ResolvedProgram): ResolvedProgram {
  const sumDelays = collectSumDelays(prog.body.decls)
  if (sumDelays.size === 0 && !bodyHasSumExpr(prog.body)) return prog

  // Pre-allocate fresh DelayDecl objects for every non-sum DelayDecl
  // too. Reason: a sum-lowered program's `body.decls` contains the
  // tag-slot decl in place of the original sum-typed delay. Non-sum
  // delays in the same body must also be replaced (and their
  // `DelayRef`s rewritten to point at the new objects), otherwise
  // expressions in `body.assigns` end up mixing fresh sum-slot decls
  // with the originals — and the slot-table built downstream by
  // `loadProgramDefFromResolved` rejects refs whose decl isn't in the
  // table by identity.
  //
  // Sum-typed delays have their replacement decls already allocated
  // by `collectSumDelays`; non-sum delays get a fresh shell here.
  const delayMap = new Map<DelayDecl, DelayDecl>()
  for (const decl of prog.body.decls) {
    if (decl.op !== 'delayDecl') continue
    if (sumDelays.has(decl)) {
      // Map the original to its tag-slot decl. Used by `DelayRef`
      // rewriting outside of match-arm scrutinee contexts (e.g. when
      // a sum-typed delay is read on its own — not exercised by the
      // current stdlib but legal to express).
      delayMap.set(decl, sumDelays.get(decl)!.tagSlot.decl)
    } else {
      const fresh: DelayDecl = { op: 'delayDecl', name: decl.name, update: 0, init: 0 }
      if (decl.update !== undefined) fresh.update = decl.update
      if (decl.init !== undefined) fresh.init = decl.init
      delayMap.set(decl, fresh)
    }
  }

  const ctx: Ctx = { sumDelays, delayMap }

  // Rewrite decls: replace each sum-typed DelayDecl with its per-slot
  // expansion (preserving source position); for everything else,
  // recursively rewrite contained expressions.
  const newDecls: BodyDecl[] = []
  for (const decl of prog.body.decls) {
    if (decl.op === 'delayDecl' && sumDelays.has(decl)) {
      const info = sumDelays.get(decl)!
      // The slots' decl objects were already created during collection.
      // Now fill in their init/update, which depend on the rewritten
      // versions of the original's init (a Tag) and update (a
      // sum-valued expression).
      fillSlotsForSumDelay(info, decl, ctx)
      for (const slot of info.slots) newDecls.push(slot.decl)
    } else {
      newDecls.push(rewriteDecl(decl, ctx))
    }
  }

  // Rewrite assigns: any next_update targeting a sum-typed delay must
  // expand into N+1 next_updates (one per slot). Other assigns get
  // their expr rewritten.
  const newAssigns: BodyAssign[] = []
  for (const assign of prog.body.assigns) {
    if (assign.op === 'nextUpdate' && assign.target.op === 'delayDecl' && sumDelays.has(assign.target)) {
      const info = sumDelays.get(assign.target)!
      for (const slot of info.slots) {
        const slotExpr = extractSlotFromSumExpr(assign.expr, info, slot, ctx)
        newAssigns.push({ op: 'nextUpdate', target: slot.decl, expr: slotExpr })
      }
    } else {
      newAssigns.push(rewriteAssign(assign, ctx))
    }
  }

  const newBody: ResolvedBlock = { op: 'block', decls: newDecls, assigns: newAssigns }
  return { ...prog, body: newBody }
}

// ─────────────────────────────────────────────────────────────
// Sum-delay collection
// ─────────────────────────────────────────────────────────────

/**
 * Scan body decls for sum-typed `DelayDecl`s and pre-allocate their
 * per-slot replacements. The decls are created here (with placeholder
 * init/update set to 0) so cross-decl references — e.g. another
 * delay's update reads our delay — can be rewritten to point at the
 * fresh slot decls before init/update are filled in.
 */
function collectSumDelays(decls: BodyDecl[]): SumDelayMap {
  const out: SumDelayMap = new Map()
  for (const decl of decls) {
    if (decl.op !== 'delayDecl') continue
    const sumType = sumTypeOfDelayInit(decl)
    if (!sumType) continue

    const slots: SlotEntry[] = []
    const tagDecl: DelayDecl = {
      op: 'delayDecl',
      name: mangle(decl.name, 'tag'),
      update: 0,
      init: 0,
    }
    const tagSlot: SlotEntry = { decl: tagDecl }
    slots.push(tagSlot)

    const payloadByKey = new Map<string, SlotEntry>()
    for (const variant of sumType.variants) {
      for (const field of variant.payload) {
        const slotDecl: DelayDecl = {
          op: 'delayDecl',
          name: mangle(decl.name, `${variant.name}__${field.name}`),
          update: 0,
          init: 0,
        }
        const slot: SlotEntry = { decl: slotDecl, variant, field }
        slots.push(slot)
        payloadByKey.set(slotKey(variant, field), slot)
      }
    }

    out.set(decl, { original: decl, sumType, slots, tagSlot, payloadByKey })
  }
  return out
}

function sumTypeOfDelayInit(decl: DelayDecl): SumTypeDef | undefined {
  const init = decl.init
  if (typeof init !== 'object' || init === null || Array.isArray(init)) return undefined
  if (init.op !== 'tag') return undefined
  return init.variant.parent
}

function slotKey(variant: SumVariant, field: StructField): string {
  return `${variant.name}__${field.name}`
}

function mangle(base: string, suffix: string): string {
  return `${base}#${suffix}`
}

// ─────────────────────────────────────────────────────────────
// Slot init/update assembly for a sum-typed DelayDecl
// ─────────────────────────────────────────────────────────────

/**
 * After collection has pre-allocated slot decls, populate their
 * `init` (from the original's `Tag`) and `update` (per-slot
 * extraction of the sum-valued update expression).
 */
function fillSlotsForSumDelay(
  info: SumDelayInfo,
  orig: DelayDecl,
  ctx: Ctx,
): void {
  const init = orig.init
  if (typeof init !== 'object' || init === null || Array.isArray(init) || init.op !== 'tag') {
    throw new Error(
      `sumLower: delay '${orig.name}': init must be a constant tag expression`,
    )
  }
  const initTag = init as Tag
  const initVariantIdx = info.sumType.variants.indexOf(initTag.variant)
  if (initVariantIdx < 0) {
    throw new Error(
      `sumLower: delay '${orig.name}': init variant '${initTag.variant.name}' not in '${info.sumType.name}'`,
    )
  }

  const initPayload = new Map<string, ResolvedExpr>()
  for (const entry of initTag.payload) initPayload.set(entry.field.name, entry.value)

  for (const slot of info.slots) {
    if (slot.variant === undefined) {
      // Tag slot.
      slot.decl.init = initVariantIdx
    } else if (slot.variant === initTag.variant && slot.field !== undefined) {
      const v = initPayload.get(slot.field.name)
      slot.decl.init = v !== undefined ? rewriteExpr(v, ctx) : 0
    } else {
      slot.decl.init = 0
    }
    slot.decl.update = extractSlotFromSumExpr(orig.update, info, slot, ctx)
  }
}

// ─────────────────────────────────────────────────────────────
// Rewriting context
// ─────────────────────────────────────────────────────────────

interface Ctx {
  sumDelays: SumDelayMap
  /** Map every original DelayDecl in the body to its replacement.
   *  Sum-typed → the tag-slot decl; non-sum → a freshly cloned decl
   *  whose `update`/`init` will be filled in by `rewriteDecl`. Used
   *  by `DelayRef` rewriting so refs in expressions point at the
   *  decl objects that actually appear in the rewritten body. */
  delayMap: Map<DelayDecl, DelayDecl>
  /** Active per-binder substitutions introduced by match arms.
   *  A `BindingRef` whose decl is a key here is rewritten to the
   *  mapped expression. */
  bindings?: Map<BinderDecl, ResolvedExpr>
}

function withBindings(
  ctx: Ctx,
  extra: Map<BinderDecl, ResolvedExpr>,
): Ctx {
  if (extra.size === 0) return ctx
  const merged = new Map(ctx.bindings ?? [])
  for (const [k, v] of extra) merged.set(k, v)
  return { ...ctx, bindings: merged }
}

// ─────────────────────────────────────────────────────────────
// Decl / assign recursion
// ─────────────────────────────────────────────────────────────

function rewriteDecl(decl: BodyDecl, ctx: Ctx): BodyDecl {
  switch (decl.op) {
    case 'regDecl':
      return { ...decl, init: rewriteExpr(decl.init, ctx) }
    case 'delayDecl': {
      // Non-sum delay: fill the pre-allocated fresh decl's update /
      // init with rewritten expressions. The fresh decl was put in
      // `delayMap` ahead of time so any DelayRef pointing at the
      // original decl is rewritten to point at this fresh one.
      const fresh = ctx.delayMap.get(decl)
      if (!fresh) {
        throw new Error(`sumLower: missing delayMap entry for non-sum delay '${decl.name}'`)
      }
      fresh.update = rewriteExpr(decl.update, ctx)
      fresh.init = rewriteExpr(decl.init, ctx)
      return fresh
    }
    case 'paramDecl':
    case 'programDecl':
      return decl
    case 'instanceDecl':
      return {
        ...decl,
        inputs: decl.inputs.map(i => ({ port: i.port, value: rewriteExpr(i.value, ctx) })),
      }
  }
}

function rewriteAssign(assign: BodyAssign, ctx: Ctx): BodyAssign {
  if (assign.op === 'outputAssign') {
    const out: OutputAssign = { op: 'outputAssign', target: assign.target, expr: rewriteExpr(assign.expr, ctx) }
    return out
  }
  // nextUpdate: redirect the target to its delay-map replacement when
  // the target is a delay (regs aren't cloned by sumLower).
  let target = assign.target
  if (target.op === 'delayDecl') {
    const replacement = ctx.delayMap.get(target)
    if (replacement) target = replacement
  }
  const out: NextUpdate = { op: 'nextUpdate', target, expr: rewriteExpr(assign.expr, ctx) }
  return out
}

// ─────────────────────────────────────────────────────────────
// Expression rewriting
// ─────────────────────────────────────────────────────────────

function rewriteExpr(expr: ResolvedExpr, ctx: Ctx): ResolvedExpr {
  if (typeof expr === 'number' || typeof expr === 'boolean') return expr
  if (Array.isArray(expr)) return expr.map(e => rewriteExpr(e, ctx))
  return rewriteOp(expr, ctx)
}

function rewriteOp(node: ResolvedExprOp, ctx: Ctx): ResolvedExpr {
  switch (node.op) {
    // ── Bindings: substitute when the binder is in the active map. ──
    case 'bindingRef': {
      const sub = ctx.bindings?.get(node.decl)
      return sub !== undefined ? sub : node
    }

    // ── DelayRef: rewrite to the cloned/replacement decl. For a
    //    sum-typed source delay this is the tag slot; for non-sum
    //    it's the cloned scalar decl. (Match rewriting handles
    //    payload reads via per-arm substitution before reaching
    //    this case.) ──
    case 'delayRef': {
      const replacement = ctx.delayMap.get(node.decl)
      if (replacement && replacement !== node.decl) {
        return { op: 'delayRef', decl: replacement }
      }
      return node
    }

    // ── Tag in expression position (no payload) → variant index. ──
    case 'tag': {
      if (node.payload.length === 0) {
        const idx = node.variant.parent.variants.indexOf(node.variant)
        if (idx < 0) {
          throw new Error(`sumLower: variant '${node.variant.name}' missing from parent type`)
        }
        return idx
      }
      throw new Error(
        `sumLower: bare tag with payload (variant '${node.variant.name}') in non-update context`,
      )
    }

    // ── Match: lower to a scalar select chain over the scrutinee's
    //    tag-slot read. Per-arm payload bindings rewrite to slot
    //    reads of the scrutinee. ──
    case 'match':
      return lowerMatchToSelectChain(node, ctx)

    // ── Pass-through references / leaves ──
    case 'inputRef':
    case 'regRef':
    case 'paramRef':
    case 'typeParamRef':
    case 'sampleRate':
    case 'sampleIndex':
    case 'nestedOut':
      return node

    // ── Operators with uniform `args` arity. ──
    case 'add': case 'sub': case 'mul': case 'div': case 'mod':
    case 'lt': case 'lte': case 'gt': case 'gte': case 'eq': case 'neq':
    case 'and': case 'or':
    case 'bitAnd': case 'bitOr': case 'bitXor': case 'lshift': case 'rshift':
    case 'floorDiv': case 'ldexp':
      return { op: node.op, args: [rewriteExpr(node.args[0], ctx), rewriteExpr(node.args[1], ctx)] }
    case 'neg': case 'not': case 'bitNot':
    case 'sqrt': case 'abs': case 'floor': case 'ceil': case 'round':
    case 'floatExponent': case 'toInt': case 'toBool': case 'toFloat':
      return { op: node.op, args: [rewriteExpr(node.args[0], ctx)] }
    case 'clamp':
      return { op: 'clamp', args: [rewriteExpr(node.args[0], ctx), rewriteExpr(node.args[1], ctx), rewriteExpr(node.args[2], ctx)] }
    case 'select':
      return { op: 'select', args: [rewriteExpr(node.args[0], ctx), rewriteExpr(node.args[1], ctx), rewriteExpr(node.args[2], ctx)] }
    case 'index':
      return { op: 'index', args: [rewriteExpr(node.args[0], ctx), rewriteExpr(node.args[1], ctx)] }
    case 'arraySet':
      return { op: 'arraySet', args: [rewriteExpr(node.args[0], ctx), rewriteExpr(node.args[1], ctx), rewriteExpr(node.args[2], ctx)] }
    case 'zeros':
      return { op: 'zeros', count: rewriteExpr(node.count, ctx) }

    // ── Combinators (binders pass through; their bodies get rewritten). ──
    case 'fold':
      return { op: 'fold', over: rewriteExpr(node.over, ctx), init: rewriteExpr(node.init, ctx),
               acc: node.acc, elem: node.elem, body: rewriteExpr(node.body, ctx) }
    case 'scan':
      return { op: 'scan', over: rewriteExpr(node.over, ctx), init: rewriteExpr(node.init, ctx),
               acc: node.acc, elem: node.elem, body: rewriteExpr(node.body, ctx) }
    case 'generate':
      return { op: 'generate', count: rewriteExpr(node.count, ctx),
               iter: node.iter, body: rewriteExpr(node.body, ctx) }
    case 'iterate':
      return { op: 'iterate', count: rewriteExpr(node.count, ctx), init: rewriteExpr(node.init, ctx),
               iter: node.iter, body: rewriteExpr(node.body, ctx) }
    case 'chain':
      return { op: 'chain', count: rewriteExpr(node.count, ctx), init: rewriteExpr(node.init, ctx),
               iter: node.iter, body: rewriteExpr(node.body, ctx) }
    case 'map2':
      return { op: 'map2', over: rewriteExpr(node.over, ctx),
               elem: node.elem, body: rewriteExpr(node.body, ctx) }
    case 'zipWith':
      return { op: 'zipWith', a: rewriteExpr(node.a, ctx), b: rewriteExpr(node.b, ctx),
               x: node.x, y: node.y, body: rewriteExpr(node.body, ctx) }
    case 'let':
      return {
        op: 'let',
        binders: node.binders.map(b => ({ binder: b.binder, value: rewriteExpr(b.value, ctx) })),
        in: rewriteExpr(node.in, ctx),
      }
  }
}

// ─────────────────────────────────────────────────────────────
// Match → select chain (scalar-valued match)
// ─────────────────────────────────────────────────────────────

function lowerMatchToSelectChain(m: Match, ctx: Ctx): ResolvedExpr {
  // The scrutinee must reduce to a sum-typed value. We need access to
  // the per-variant payload slots to rewrite payload bindings.
  // V1 (mirrors legacy): only DelayRef-to-sum-typed-delay scrutinees
  // support payload-bearing arms. Nullary-only matches accept any
  // scrutinee that lowers to the variant's tag integer.
  const tagRead = scrutineeTagRead(m, ctx)

  // Build select chain in legacy order: iterate variants in the sum
  // type's declaration order; the LAST variant is the chain tail (its
  // body is the else-branch of the deepest select, with no comparison
  // — exhaustiveness guarantees one arm matches).
  const variants = m.type.variants
  const armBy = new Map<SumVariant, MatchArm>()
  for (const arm of m.arms) armBy.set(arm.variant, arm)

  const lowerArmBody = (arm: MatchArm): ResolvedExpr => {
    if (arm.binders.length === 0) return rewriteExpr(arm.body, ctx)
    const subs = bindingsForArm(m.scrutinee, arm, ctx)
    const innerCtx = withBindings(ctx, subs)
    return rewriteExpr(arm.body, innerCtx)
  }

  const lastVariant = variants[variants.length - 1]
  const lastArm = armBy.get(lastVariant)
  if (!lastArm) {
    throw new Error(`sumLower: match on '${m.type.name}' missing arm for '${lastVariant.name}'`)
  }
  let chain: ResolvedExpr = lowerArmBody(lastArm)
  for (let i = variants.length - 2; i >= 0; i--) {
    const v = variants[i]
    const arm = armBy.get(v)
    if (!arm) {
      throw new Error(`sumLower: match on '${m.type.name}' missing arm for '${v.name}'`)
    }
    const armBody = lowerArmBody(arm)
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
 * Lower the scrutinee to its tag-slot read. For a sum-typed delay,
 * the tag-slot is `{ op: 'delayRef', decl: info.tagSlot.decl }`.
 * For a constant tag, the read is the variant index integer. Other
 * scrutinee shapes fall through to a generic recursive rewrite —
 * sufficient for nullary-only matches whose scrutinee is itself a
 * `match` returning a tag.
 */
function scrutineeTagRead(m: Match, ctx: Ctx): ResolvedExpr {
  return rewriteExpr(m.scrutinee, ctx)
}

/**
 * Build the per-binder substitution map for a payload-bearing arm.
 * Only supports `DelayRef`-to-sum-typed-delay scrutinees (matching
 * the legacy V1 limitation).
 */
function bindingsForArm(
  scrutinee: ResolvedExpr,
  arm: MatchArm,
  ctx: Ctx,
): Map<BinderDecl, ResolvedExpr> {
  const subs = new Map<BinderDecl, ResolvedExpr>()
  if (arm.binders.length === 0) return subs

  if (typeof scrutinee !== 'object' || scrutinee === null || Array.isArray(scrutinee)
      || scrutinee.op !== 'delayRef') {
    throw new Error(
      `sumLower: match arm '${arm.variant.name}' has payload bindings but scrutinee is not a delay_ref`,
    )
  }
  const dRef = scrutinee as DelayRef
  const info = ctx.sumDelays.get(dRef.decl)
  if (!info) {
    throw new Error(
      `sumLower: match arm '${arm.variant.name}' scrutinee references non-sum delay '${dRef.decl.name}'`,
    )
  }
  if (arm.binders.length !== arm.variant.payload.length) {
    throw new Error(
      `sumLower: match arm '${arm.variant.name}': binders/payload arity mismatch`,
    )
  }
  for (let i = 0; i < arm.binders.length; i++) {
    const field = arm.variant.payload[i]
    const slot = info.payloadByKey.get(slotKey(arm.variant, field))
    if (!slot) {
      throw new Error(
        `sumLower: match arm '${arm.variant.name}': missing slot for field '${field.name}'`,
      )
    }
    subs.set(arm.binders[i], { op: 'delayRef', decl: slot.decl })
  }
  return subs
}

// ─────────────────────────────────────────────────────────────
// Sum-valued expression → per-slot scalar extraction
// ─────────────────────────────────────────────────────────────

/**
 * Extract the scalar update for one slot of a sum-typed delay's
 * update expression. Mirrors `extractSlotFromSumExpr` in
 * `compiler/sum_lowering.ts`.
 *
 * Recognized shapes for `expr`:
 *   - `Tag` — constant constructor; tag-slot gets the variant
 *     index, payload-slot gets either the literal value or 0
 *     depending on whether the slot's variant matches the tag.
 *   - `Match` returning a sum value — distribute slot extraction
 *     over each arm; build a select-chain over the scrutinee's tag
 *     read.
 *   - `DelayRef` to a sum-typed delay — read the matching slot of
 *     the source delay.
 *   - `select(c, a, b)` where both branches are sum-valued —
 *     distribute: `select(c, extract(a), extract(b))`.
 *   - Otherwise return 0 (undefined behavior; caller's malformed
 *     update).
 */
function extractSlotFromSumExpr(
  expr: ResolvedExpr,
  info: SumDelayInfo,
  slot: SlotEntry,
  ctx: Ctx,
): ResolvedExpr {
  if (typeof expr !== 'object' || expr === null || Array.isArray(expr)) return 0

  switch (expr.op) {
    case 'tag': {
      const idx = info.sumType.variants.indexOf(expr.variant)
      if (idx < 0) {
        throw new Error(`sumLower: tag variant '${expr.variant.name}' not in '${info.sumType.name}'`)
      }
      if (slot.variant === undefined) return idx
      if (slot.variant === expr.variant && slot.field !== undefined) {
        const entry = expr.payload.find(p => p.field === slot.field)
        return entry !== undefined ? rewriteExpr(entry.value, ctx) : 0
      }
      return 0
    }

    case 'match': {
      const tagRead = rewriteExpr(expr.scrutinee, ctx)
      const armBy = new Map<SumVariant, MatchArm>()
      for (const arm of expr.arms) armBy.set(arm.variant, arm)

      const armSlot = (arm: MatchArm): ResolvedExpr => {
        if (arm.binders.length === 0) return extractSlotFromSumExpr(arm.body, info, slot, ctx)
        const subs = bindingsForArm(expr.scrutinee, arm, ctx)
        const innerCtx = withBindings(ctx, subs)
        return extractSlotFromSumExpr(arm.body, info, slot, innerCtx)
      }

      const variants = expr.type.variants
      const lastVariant = variants[variants.length - 1]
      const lastArm = armBy.get(lastVariant)
      if (!lastArm) {
        throw new Error(`sumLower: match on '${expr.type.name}' missing arm for '${lastVariant.name}'`)
      }
      let chain: ResolvedExpr = armSlot(lastArm)
      for (let i = variants.length - 2; i >= 0; i--) {
        const v = variants[i]
        const arm = armBy.get(v)
        if (!arm) {
          throw new Error(`sumLower: match on '${expr.type.name}' missing arm for '${v.name}'`)
        }
        chain = {
          op: 'select',
          args: [{ op: 'eq', args: [tagRead, i] }, armSlot(arm), chain],
        }
      }
      return chain
    }

    case 'delayRef': {
      const srcInfo = ctx.sumDelays.get(expr.decl)
      if (!srcInfo) {
        // Reading a scalar delay as a sum value is malformed.
        return 0
      }
      if (slot.variant === undefined) {
        return { op: 'delayRef', decl: srcInfo.tagSlot.decl }
      }
      const srcSlot = srcInfo.payloadByKey.get(slotKey(slot.variant, slot.field!))
      if (srcSlot) return { op: 'delayRef', decl: srcSlot.decl }
      return 0
    }

    case 'select': {
      const [cond, then, alt] = expr.args
      return {
        op: 'select',
        args: [
          rewriteExpr(cond, ctx),
          extractSlotFromSumExpr(then, info, slot, ctx),
          extractSlotFromSumExpr(alt, info, slot, ctx),
        ],
      }
    }

    default:
      return 0
  }
}

// ─────────────────────────────────────────────────────────────
// Body inspection helpers
// ─────────────────────────────────────────────────────────────

function bodyHasSumExpr(body: ResolvedBlock): boolean {
  for (const decl of body.decls) {
    if (declHasSumExpr(decl)) return true
  }
  for (const a of body.assigns) {
    if (exprHasSumExpr(a.expr)) return true
  }
  return false
}

function declHasSumExpr(decl: BodyDecl): boolean {
  switch (decl.op) {
    case 'regDecl':   return exprHasSumExpr(decl.init)
    case 'delayDecl': return exprHasSumExpr(decl.init) || exprHasSumExpr(decl.update)
    case 'paramDecl': return false
    case 'instanceDecl':
      return decl.inputs.some(i => exprHasSumExpr(i.value))
    case 'programDecl':
      return false
  }
}

function exprHasSumExpr(expr: ResolvedExpr): boolean {
  if (typeof expr !== 'object' || expr === null) return false
  if (Array.isArray(expr)) return expr.some(exprHasSumExpr)
  switch (expr.op) {
    case 'tag':
    case 'match':
      return true
    case 'fold': case 'scan':
      return exprHasSumExpr(expr.over) || exprHasSumExpr(expr.init) || exprHasSumExpr(expr.body)
    case 'generate':
      return exprHasSumExpr(expr.count) || exprHasSumExpr(expr.body)
    case 'iterate': case 'chain':
      return exprHasSumExpr(expr.count) || exprHasSumExpr(expr.init) || exprHasSumExpr(expr.body)
    case 'map2':
      return exprHasSumExpr(expr.over) || exprHasSumExpr(expr.body)
    case 'zipWith':
      return exprHasSumExpr(expr.a) || exprHasSumExpr(expr.b) || exprHasSumExpr(expr.body)
    case 'let':
      return expr.binders.some(b => exprHasSumExpr(b.value)) || exprHasSumExpr(expr.in)
    case 'zeros':
      return exprHasSumExpr(expr.count)
    case 'add': case 'sub': case 'mul': case 'div': case 'mod':
    case 'lt': case 'lte': case 'gt': case 'gte': case 'eq': case 'neq':
    case 'and': case 'or':
    case 'bitAnd': case 'bitOr': case 'bitXor': case 'lshift': case 'rshift':
    case 'floorDiv': case 'ldexp':
    case 'neg': case 'not': case 'bitNot':
    case 'sqrt': case 'abs': case 'floor': case 'ceil': case 'round':
    case 'floatExponent': case 'toInt': case 'toBool': case 'toFloat':
    case 'clamp': case 'select': case 'index': case 'arraySet':
      return expr.args.some(exprHasSumExpr)
    default:
      return false
  }
}
