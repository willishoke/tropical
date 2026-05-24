/**
 * sum_lower.ts — sum-type decomposition on the resolved IR.
 *
 * Decomposes every sum-typed `RegDecl` (one whose `init` is a `Tag`)
 * into N+1 scalar `RegDecl`s — a discriminator slot (int) plus one
 * slot per (variant, field) pair across all variants — and lowers
 * every `Match`/`Tag` to scalar select-chains and variant-index
 * literals.
 *
 * After this pass the program contains no `tag` or `match` expressions
 * and no sum-typed regs.
 *
 * ## Pure construction
 *
 * The pass is structured as three pure phases:
 *
 *   1. `buildSpecs(prog)` — pure data. Walks the input body's decls,
 *      identifies which are sum-typed, and assigns the new RegIdx
 *      (position in the rewritten body's regs[] table) for every
 *      decl that will exist after lowering. Sum regs get N+1
 *      consecutive indices (tag + per-variant per-field payloads);
 *      non-sum regs get one index each. Returns a `SpecMap`.
 *
 *   2. `buildNewDecls(prog, specs)` — construction. For each old
 *      decl, builds the fully-formed replacement(s) with `init` and
 *      `update` expressions set at construction time. Sum regs
 *      expand to a tag-slot decl plus payload-slot decls; non-sum
 *      regs become fresh decls with rewritten expressions; instance
 *      decls get their input wires rewritten; param/program decls
 *      pass through.
 *
 *   3. `rewriteAssigns(prog, specs)` — same pattern for body assigns.
 *
 *  No mutation. No pre-allocated shells that get back-patched. Refs
 *  resolve via `specs` lookup (idx → spec → new idx) — the knot-tying
 *  problem dissolves because indices are pure data assigned in
 *  phase 1, valid as targets for refs constructed in phase 2/3 even
 *  though the target decl objects haven't been built yet.
 *
 * ## Constraints (matching legacy):
 *
 *   - A sum-typed reg's `init` MUST be a `Tag` (constant variant
 *     constructor). Anything else is a structural error.
 *   - Match-arm payload bindings are only supported when the
 *     scrutinee is a `RegRef` to a sum-typed reg. Other scrutinee
 *     shapes throw.
 */

import type {
  ResolvedProgram, ResolvedExpr, ResolvedExprOp,
  ResolvedBlock,
  BodyDecl, BodyAssign, OutputAssign,
  RegDecl, BinderDecl, BinderIdx,
  SumTypeDef, SumVariant, StructField,
  Tag, Match, MatchArm,
  RegRef,
  RegIdx,
} from './nodes.js'
import { regIdx } from './nodes.js'
import { withDeclTables } from './decl_tables.js'

// ─────────────────────────────────────────────────────────────
// Specs — pure structural data describing each old reg's new shape
// ─────────────────────────────────────────────────────────────

/** A single bundle slot replacing one sum-typed `RegDecl`. Position
 *  in the rewritten body's regs[] is recorded; the decl object itself
 *  is constructed later, in `buildSumRegSlots`. */
interface SlotSpec {
  /** Index in the rewritten body's regs[] table. */
  idx: RegIdx
  /** Variant this payload slot belongs to; undefined for the tag slot. */
  variant?: SumVariant
  /** Payload field this slot represents; undefined for the tag slot. */
  field?: StructField
}

/** Per-original-RegDecl decomposition for a sum-typed reg. */
interface SumRegInfo {
  sumType: SumTypeDef
  /** Slot order: [tag, ...per-variant per-field-in-payload-order]. */
  slots: SlotSpec[]
  tagSlot: SlotSpec
  payloadByKey: Map<string, SlotSpec>
}

type RegSpec =
  | { kind: 'sum';    info: SumRegInfo }
  | { kind: 'nonSum'; idx: RegIdx }

type SpecMap = Map<RegDecl, RegSpec>

// ─────────────────────────────────────────────────────────────
// Public entry
// ─────────────────────────────────────────────────────────────

export function sumLower(prog: ResolvedProgram): ResolvedProgram {
  if (!progHasAnySumWork(prog)) return prog

  // Phase 1 — pure data: assign new indices, identify sum-typed regs.
  const specs = buildSpecs(prog.body.decls)

  // Phase 2 — construction: build new decls with init/update fully set.
  const ctx: Ctx = { inProg: prog, specs }
  const newDecls = buildNewDecls(prog.body.decls, ctx)

  // Phase 3 — rewrite assigns.
  const newAssigns = prog.body.assigns.map(a => rewriteAssign(a, ctx))

  const newBody: ResolvedBlock = { op: 'block', decls: newDecls, assigns: newAssigns }
  return withDeclTables({ ...prog, body: newBody })
}

// ─────────────────────────────────────────────────────────────
// Phase 1 — spec collection
// ─────────────────────────────────────────────────────────────

function buildSpecs(decls: readonly BodyDecl[]): SpecMap {
  const specs: SpecMap = new Map()
  let nextIdx = 0
  for (const decl of decls) {
    if (decl.op !== 'regDecl') continue
    const sumType = sumTypeOfRegInit(decl)
    if (sumType) {
      const slots: SlotSpec[] = []
      const tagSlot: SlotSpec = { idx: regIdx(nextIdx++) }
      slots.push(tagSlot)
      const payloadByKey = new Map<string, SlotSpec>()
      for (const variant of sumType.variants) {
        for (const field of variant.payload) {
          const slot: SlotSpec = { idx: regIdx(nextIdx++), variant, field }
          slots.push(slot)
          payloadByKey.set(slotKey(variant, field), slot)
        }
      }
      specs.set(decl, { kind: 'sum', info: { sumType, slots, tagSlot, payloadByKey } })
    } else {
      specs.set(decl, { kind: 'nonSum', idx: regIdx(nextIdx++) })
    }
  }
  return specs
}

function sumTypeOfRegInit(decl: RegDecl): SumTypeDef | undefined {
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
// Phase 2 — decl construction (pure)
// ─────────────────────────────────────────────────────────────

function buildNewDecls(decls: readonly BodyDecl[], ctx: Ctx): BodyDecl[] {
  const out: BodyDecl[] = []
  for (const decl of decls) {
    if (decl.op === 'regDecl') {
      const spec = ctx.specs.get(decl)
      if (!spec) throw new Error(`sumLower: no spec for reg '${decl.name}' (internal)`)
      if (spec.kind === 'sum') {
        out.push(...buildSumRegSlots(decl, spec.info, ctx))
      } else {
        out.push(buildNonSumReg(decl, ctx))
      }
    } else if (decl.op === 'instanceDecl') {
      out.push({
        ...decl,
        inputs: decl.inputs.map(i => ({ port: i.port, value: rewriteExpr(i.value, ctx) })),
      })
    } else {
      // paramDecl, programDecl — pass through unchanged.
      out.push(decl)
    }
  }
  return out
}

function buildSumRegSlots(orig: RegDecl, info: SumRegInfo, ctx: Ctx): RegDecl[] {
  const init = orig.init
  if (typeof init !== 'object' || init === null || Array.isArray(init) || init.op !== 'tag') {
    throw new Error(`sumLower: reg '${orig.name}': init must be a constant tag expression`)
  }
  const initTag = init as Tag
  const initVariantIdx = info.sumType.variants.indexOf(initTag.variant)
  if (initVariantIdx < 0) {
    throw new Error(
      `sumLower: reg '${orig.name}': init variant '${initTag.variant.name}' not in '${info.sumType.name}'`,
    )
  }
  const initPayload = new Map<string, ResolvedExpr>()
  for (const entry of initTag.payload) initPayload.set(entry.field.name, entry.value)

  const out: RegDecl[] = []
  for (const slot of info.slots) {
    let slotInit: ResolvedExpr
    if (slot.variant === undefined) {
      slotInit = initVariantIdx
    } else if (slot.variant === initTag.variant && slot.field !== undefined) {
      const v = initPayload.get(slot.field.name)
      slotInit = v !== undefined ? rewriteExpr(v, ctx) : 0
    } else {
      slotInit = 0
    }
    const slotName = slot.variant === undefined
      ? mangle(orig.name, 'tag')
      : mangle(orig.name, `${slot.variant.name}__${slot.field!.name}`)
    const decl: RegDecl = {
      op: 'regDecl',
      name: slotName,
      init: slotInit,
    }
    if (orig.update !== undefined) {
      decl.update = extractSlotFromSumExpr(orig.update, info, slot, ctx)
    }
    out.push(decl)
  }
  return out
}

function buildNonSumReg(orig: RegDecl, ctx: Ctx): RegDecl {
  const decl: RegDecl = {
    op: 'regDecl',
    name: orig.name,
    init: rewriteExpr(orig.init, ctx),
  }
  if (orig.update !== undefined) decl.update = rewriteExpr(orig.update, ctx)
  if (orig.type !== undefined) decl.type = orig.type
  if (orig._liftedFrom !== undefined) decl._liftedFrom = orig._liftedFrom
  return decl
}

// ─────────────────────────────────────────────────────────────
// Rewriting context
// ─────────────────────────────────────────────────────────────

interface Ctx {
  /** Input program; resolves old RegRef.idx → source RegDecl. */
  inProg: ResolvedProgram
  /** Per-old-reg spec: either { kind: 'sum', info } or { kind: 'nonSum', idx }.
   *  Looked up via `ctx.specs.get(srcDecl)` after the source decl is
   *  found in `inProg.regs[oldIdx]`. */
  specs: SpecMap
  /** Active per-binder substitutions introduced by match arms.
   *  A `BindingRef` whose `idx` is a key here is rewritten to the
   *  mapped expression. */
  bindings?: Map<BinderIdx, ResolvedExpr>
}

function withBindings(
  ctx: Ctx,
  extra: Map<BinderIdx, ResolvedExpr>,
): Ctx {
  if (extra.size === 0) return ctx
  const merged = new Map(ctx.bindings ?? [])
  for (const [k, v] of extra) merged.set(k, v)
  return { ...ctx, bindings: merged }
}

// ─────────────────────────────────────────────────────────────
// Phase 3 — assign rewriting (delegates to expr rewriter)
// ─────────────────────────────────────────────────────────────

function rewriteAssign(assign: BodyAssign, ctx: Ctx): BodyAssign {
  const out: OutputAssign = {
    op: 'outputAssign',
    target: assign.target,
    expr: rewriteExpr(assign.expr, ctx),
  }
  return out
}

// ─────────────────────────────────────────────────────────────
// Expression rewriting (pure; no mutation, no shells)
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
      const sub = ctx.bindings?.get(node.idx)
      return sub !== undefined ? sub : node
    }

    // ── RegRef: rewrite to the new RegIdx via the spec map. For a
    //    sum-typed source reg this is the tag slot; for non-sum it's
    //    the new fresh decl's idx. Match-arm payload reads are handled
    //    via per-arm binding substitution before reaching this case. ──
    case 'regRef': {
      const srcDecl = ctx.inProg.regs[node.idx]
      if (!srcDecl) throw new Error(`sumLower: regRef idx=${node.idx} has no source in input program`)
      const spec = ctx.specs.get(srcDecl)
      if (!spec) return node
      const newIdx = spec.kind === 'sum' ? spec.info.tagSlot.idx : spec.idx
      return { op: 'regRef', idx: newIdx }
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
// Match lowering
// ─────────────────────────────────────────────────────────────

function lowerMatchToSelectChain(m: Match, ctx: Ctx): ResolvedExpr {
  const tagRead = scrutineeTagRead(m, ctx)
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

function scrutineeTagRead(m: Match, ctx: Ctx): ResolvedExpr {
  return rewriteExpr(m.scrutinee, ctx)
}

function bindingsForArm(
  scrutinee: ResolvedExpr,
  arm: MatchArm,
  ctx: Ctx,
): Map<BinderIdx, ResolvedExpr> {
  const subs = new Map<BinderIdx, ResolvedExpr>()
  if (arm.binders.length === 0) return subs

  if (typeof scrutinee !== 'object' || scrutinee === null || Array.isArray(scrutinee)
      || scrutinee.op !== 'regRef') {
    throw new Error(
      `sumLower: match arm '${arm.variant.name}' has payload bindings but scrutinee is not a reg_ref`,
    )
  }
  const rRef = scrutinee as RegRef
  const srcDecl = ctx.inProg.regs[rRef.idx]
  if (!srcDecl) {
    throw new Error(`sumLower: match arm scrutinee regRef idx=${rRef.idx} has no source decl`)
  }
  const spec = ctx.specs.get(srcDecl)
  if (!spec || spec.kind !== 'sum') {
    throw new Error(
      `sumLower: match arm '${arm.variant.name}' scrutinee references non-sum reg '${srcDecl.name}'`,
    )
  }
  if (arm.binders.length !== arm.variant.payload.length) {
    throw new Error(
      `sumLower: match arm '${arm.variant.name}': binders/payload arity mismatch`,
    )
  }
  for (let i = 0; i < arm.binders.length; i++) {
    const field = arm.variant.payload[i]
    const slot = spec.info.payloadByKey.get(slotKey(arm.variant, field))
    if (!slot) {
      throw new Error(
        `sumLower: match arm '${arm.variant.name}': missing slot for field '${field.name}'`,
      )
    }
    subs.set(arm.binders[i].idx, { op: 'regRef', idx: slot.idx })
  }
  return subs
}

// ─────────────────────────────────────────────────────────────
// Sum-valued expression → per-slot scalar extraction
// ─────────────────────────────────────────────────────────────

/**
 * Extract the scalar update for one slot of a sum-typed reg's update
 * expression.
 *
 * Recognized shapes for `expr`:
 *   - `Tag` — constant constructor; tag-slot gets the variant index,
 *     payload-slot gets either the literal value or 0 depending on
 *     whether the slot's variant matches the tag.
 *   - `Match` returning a sum value — distribute slot extraction over
 *     each arm; build a select-chain over the scrutinee's tag read.
 *   - `RegRef` to a sum-typed reg — read the matching slot of the
 *     source reg.
 *   - `select(c, a, b)` where both branches are sum-valued —
 *     distribute: `select(c, extract(a), extract(b))`.
 *   - Otherwise return 0 (undefined behavior; caller's malformed update).
 */
function extractSlotFromSumExpr(
  expr: ResolvedExpr,
  info: SumRegInfo,
  slot: SlotSpec,
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

    case 'regRef': {
      const srcDecl = ctx.inProg.regs[expr.idx]
      if (!srcDecl) return 0
      const srcSpec = ctx.specs.get(srcDecl)
      if (!srcSpec || srcSpec.kind !== 'sum') {
        // Reading a scalar reg as a sum value is malformed.
        return 0
      }
      if (slot.variant === undefined) {
        return { op: 'regRef', idx: srcSpec.info.tagSlot.idx }
      }
      const srcSlot = srcSpec.info.payloadByKey.get(slotKey(slot.variant, slot.field!))
      if (srcSlot) return { op: 'regRef', idx: srcSlot.idx }
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
// Fast-path detection: skip if no work to do
// ─────────────────────────────────────────────────────────────

function progHasAnySumWork(prog: ResolvedProgram): boolean {
  for (const decl of prog.body.decls) {
    if (decl.op === 'regDecl' && sumTypeOfRegInit(decl) !== undefined) return true
  }
  return bodyHasSumExpr(prog.body)
}

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
    case 'regDecl':
      return exprHasSumExpr(decl.init)
        || (decl.update !== undefined && exprHasSumExpr(decl.update))
    case 'instanceDecl':
      return decl.inputs.some(i => exprHasSumExpr(i.value))
    case 'paramDecl':
    case 'programDecl':
      return false
  }
}

function exprHasSumExpr(e: ResolvedExpr): boolean {
  if (typeof e !== 'object' || e === null) return false
  if (Array.isArray(e)) return e.some(exprHasSumExpr)
  const op = e.op
  if (op === 'tag' || op === 'match') return true
  switch (op) {
    case 'inputRef': case 'regRef': case 'paramRef':
    case 'typeParamRef': case 'bindingRef':
    case 'sampleRate': case 'sampleIndex': case 'nestedOut':
      return false
    case 'add': case 'sub': case 'mul': case 'div': case 'mod':
    case 'lt': case 'lte': case 'gt': case 'gte': case 'eq': case 'neq':
    case 'and': case 'or':
    case 'bitAnd': case 'bitOr': case 'bitXor': case 'lshift': case 'rshift':
    case 'floorDiv': case 'ldexp':
    case 'neg': case 'not': case 'bitNot':
    case 'sqrt': case 'abs': case 'floor': case 'ceil': case 'round':
    case 'floatExponent': case 'toInt': case 'toBool': case 'toFloat':
    case 'clamp': case 'select': case 'index': case 'arraySet':
      return (e.args as ResolvedExpr[]).some(exprHasSumExpr)
    case 'zeros':
      return exprHasSumExpr(e.count)
    case 'fold': case 'scan':
      return exprHasSumExpr(e.over) || exprHasSumExpr(e.init) || exprHasSumExpr(e.body)
    case 'generate':
      return exprHasSumExpr(e.count) || exprHasSumExpr(e.body)
    case 'iterate': case 'chain':
      return exprHasSumExpr(e.count) || exprHasSumExpr(e.init) || exprHasSumExpr(e.body)
    case 'map2':
      return exprHasSumExpr(e.over) || exprHasSumExpr(e.body)
    case 'zipWith':
      return exprHasSumExpr(e.a) || exprHasSumExpr(e.b) || exprHasSumExpr(e.body)
    case 'let':
      return e.binders.some(b => exprHasSumExpr(b.value)) || exprHasSumExpr(e.in)
  }
}
