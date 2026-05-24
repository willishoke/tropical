/**
 * array_lower.ts — Phase C6: combinator unrolling and array-op lowering.
 *
 * After this pass:
 *   - No `let`, `fold`, `scan`, `generate`, `iterate`, `chain`, `map2`,
 *     `zipWith` ops remain anywhere in the program.
 *   - No `bindingRef` survives — every BinderDecl introduced by a
 *     combinator/let has been substituted away.
 *   - Every `zeros{count}` whose `count` is a numeric literal has been
 *     replaced by an inline array `[0, 0, ..., 0]`. (When `count` is an
 *     unresolved `TypeParamRef`, it survives — that case is unreachable
 *     after `specialize`.)
 *
 * Survivors that the post-arrayLower form admits and `loadProgramDefFromResolved`
 * already handles:
 *   - Inline arrays (`ResolvedExpr[]`) — element of arrayPack-style values.
 *   - `index(arr, i)` — left as-is (never constant-folded over inline
 *     literals; the legacy `lower_arrays.ts` does the same).
 *   - `arraySet(arr, i, v)` — left as-is. The legacy `Delay.trop`'s
 *     `next buf = arraySet(...)` survives lowering.
 *
 * Substitution discipline (the categorical win over the legacy):
 * the legacy `compiler/lower_arrays.ts` substitutes by name string,
 * which forced an elaborate "shielded scope" map for nested binders
 * with shadowing variable names. Here every `BindingRef.idx` is a
 * unique-per-program integer minted by the elaborator at binder-
 * creation time — substitution is by idx-keyed map
 * (`Map<BinderIdx, ResolvedExpr>`), so shadowing is structurally
 * impossible and we don't carry the legacy's shielding apparatus.
 *
 * Sharing: each combinator iteration uses a fresh `WeakMap` memo. A
 * memo is only valid for one substitution map (the bindings present at
 * that recursion); reusing one across iterations would conflate
 * different `acc`/`elem` values. Within an iteration, repeated
 * occurrences of the same input expression collapse to the same output
 * — that's what keeps Phaser16-style bodies from going exponential.
 */

import type {
  ResolvedProgram,
  ResolvedExpr, ResolvedExprOp,
  BodyDecl, BodyAssign, OutputAssign,
  BinderDecl, BinderIdx,
  Tag, Match, MatchArm,
} from './nodes.js'
import { cloneResolvedProgram } from './clone.js'

// ─────────────────────────────────────────────────────────────
// Public entry
// ─────────────────────────────────────────────────────────────

/**
 * arrayLower clones the input program (when work is needed) and mutates
 * the clone's decls in place. Why:
 *
 *   - Decl identity must be preserved across this pass. A `RegRef` /
 *     `DelayRef` in `body.assigns` (a NextUpdate.target) is matched
 *     against the decls in `body.decls` by `===` identity in
 *     `loadProgramDefFromResolved`'s slot table. Replacing decls with
 *     `{...decl, init, update}` (the natural functional approach)
 *     would orphan every existing ref and break the slot lookup.
 *
 *   - Mutating the *input* would surprise callers — strataPipeline can
 *     receive the same `ResolvedProgram` multiple times (e.g.
 *     `phase_c_equiv.test.ts` runs each fixture through the pipeline
 *     once, then re-runs partial paths for byte-equality). Side
 *     effects would corrupt subsequent runs.
 *
 *   - Cloning produces fresh decls; mutating those clones is safe.
 *     The `progNeedsLowering` precheck means we only clone (and incur
 *     allocation cost) when there's actual work to do — for trivial
 *     programs the input passes through unchanged by reference.
 */
export function arrayLower(prog: ResolvedProgram): ResolvedProgram {
  if (!progNeedsLowering(prog)) return prog

  const cloned = cloneResolvedProgram(prog)
  const memo = new WeakMap<ResolvedExprOp, ResolvedExpr>()
  const empty: SubstMap = EMPTY_SUBST

  // Lower input defaults in place on the cloned port decls. Defaults
  // are typically literals; combinators here are uncommon but legal.
  for (const inp of cloned.ports.inputs) {
    if (inp.default !== undefined) {
      const lowered = lowerExpr(inp.default, empty, memo)
      if (lowered !== inp.default) inp.default = lowered
    }
  }

  for (const decl of cloned.body.decls) {
    lowerDeclInPlace(decl, empty, memo)
  }

  // Rewrite assigns. Their `target` refs (the cloned RegDecl/DelayDecl
  // objects via the cloner's dedup table) carry through unchanged.
  cloned.body.assigns = cloned.body.assigns.map(a => lowerAssign(a, empty, memo))

  return cloned
}

// ─────────────────────────────────────────────────────────────
// Detection — fast path when there's nothing to lower
// ─────────────────────────────────────────────────────────────

function progNeedsLowering(prog: ResolvedProgram): boolean {
  for (const inp of prog.ports.inputs) {
    if (inp.default !== undefined && exprNeedsLowering(inp.default)) return true
  }
  for (const decl of prog.body.decls) {
    if (declNeedsLowering(decl)) return true
  }
  for (const assign of prog.body.assigns) {
    if (exprNeedsLowering(assign.expr)) return true
  }
  return false
}

function declNeedsLowering(decl: BodyDecl): boolean {
  switch (decl.op) {
    case 'regDecl':
      return exprNeedsLowering(decl.init)
        || (decl.update !== undefined && exprNeedsLowering(decl.update))
    case 'paramDecl':    return false
    case 'instanceDecl':
      // M11 fractal: also check sub-program for surviving combinators.
      // Sub-program bodies are lowered recursively when arrayLower fires.
      return decl.inputs.some(i => exprNeedsLowering(i.value))
        || progNeedsLowering(decl.type)
    case 'programDecl':  return false
  }
}

function exprNeedsLowering(expr: ResolvedExpr): boolean {
  if (typeof expr === 'number' || typeof expr === 'boolean') return false
  if (Array.isArray(expr)) return expr.some(exprNeedsLowering)
  return opNeedsLowering(expr)
}

function opNeedsLowering(node: ResolvedExprOp): boolean {
  switch (node.op) {
    case 'let': case 'fold': case 'scan': case 'generate':
    case 'iterate': case 'chain': case 'map2': case 'zipWith':
      return true
    case 'bindingRef':
      // A residual bindingRef indicates a combinator scope that hasn't
      // been entered yet; lowering will resolve or carry it. Treat as
      // "needs lowering" so the recursive walk runs and substitutes.
      return true
    case 'zeros':
      return true
    case 'inputRef': case 'regRef': case 'paramRef':
    case 'typeParamRef': case 'nestedOut':
    case 'sampleRate': case 'sampleIndex':
      return false
    case 'tag':
      return node.payload.some(p => exprNeedsLowering(p.value))
    case 'match':
      return exprNeedsLowering(node.scrutinee)
        || node.arms.some(a => exprNeedsLowering(a.body))
    case 'add': case 'sub': case 'mul': case 'div': case 'mod':
    case 'lt': case 'lte': case 'gt': case 'gte': case 'eq': case 'neq':
    case 'and': case 'or':
    case 'bitAnd': case 'bitOr': case 'bitXor': case 'lshift': case 'rshift':
    case 'floorDiv': case 'ldexp':
    case 'neg': case 'not': case 'bitNot':
    case 'sqrt': case 'abs': case 'floor': case 'ceil': case 'round':
    case 'floatExponent': case 'toInt': case 'toBool': case 'toFloat':
    case 'clamp': case 'select': case 'index': case 'arraySet':
      return node.args.some(exprNeedsLowering)
  }
}

// ─────────────────────────────────────────────────────────────
// Substitution map — keyed by BinderIdx (unique-per-program integer)
// ─────────────────────────────────────────────────────────────

type SubstMap = ReadonlyMap<BinderIdx, ResolvedExpr>
const EMPTY_SUBST: SubstMap = new Map()

/** Extend `subst` with one (idx → expr) pair. Returns a fresh map. */
function extend1(subst: SubstMap, idx: BinderIdx, expr: ResolvedExpr): SubstMap {
  const m = new Map(subst)
  m.set(idx, expr)
  return m
}

/** Extend `subst` with multiple pairs. Returns a fresh map. */
function extendN(subst: SubstMap, pairs: Array<[BinderIdx, ResolvedExpr]>): SubstMap {
  const m = new Map(subst)
  for (const [d, e] of pairs) m.set(d, e)
  return m
}

// ─────────────────────────────────────────────────────────────
// Decl / assign rewriting
// ─────────────────────────────────────────────────────────────

type Memo = WeakMap<ResolvedExprOp, ResolvedExpr>

/** Lower a body decl in place. Mutates the decl's expression-shaped
 *  fields when lowering produces a different expression; otherwise
 *  leaves them alone. Decl identity is preserved by reference. */
function lowerDeclInPlace(decl: BodyDecl, subst: SubstMap, memo: Memo): void {
  switch (decl.op) {
    case 'regDecl': {
      const init = lowerExpr(decl.init, subst, memo)
      if (init !== decl.init) decl.init = init
      if (decl.update !== undefined) {
        const update = lowerExpr(decl.update, subst, memo)
        if (update !== decl.update) decl.update = update
      }
      return
    }
    case 'paramDecl':
    case 'programDecl':
      return
    case 'instanceDecl': {
      // M11 fractal: InstanceDecls survive through strata. Lower this
      // instance's input expressions AND recursively lower the sub-
      // program's body (its own `let`/`fold`/combinators must be
      // lowered here — arrayLower only sees the top-level program of
      // each call, and the per-type strata that constructed this
      // sub-program lowered ITS top-level but not its grand-children).
      for (const i of decl.inputs) {
        const value = lowerExpr(i.value, subst, memo)
        if (value !== i.value) i.value = value
      }
      // Recurse into the sub-program's body. The clone-then-lower
      // discipline still holds because cloneResolvedProgram deep-clones
      // through InstanceDecl.type, so `decl.type` is a fresh tree we
      // can mutate.
      const subEmpty: SubstMap = EMPTY_SUBST
      for (const subDecl of decl.type.body.decls) {
        lowerDeclInPlace(subDecl, subEmpty, memo)
      }
      decl.type.body.assigns = decl.type.body.assigns.map(a => lowerAssign(a, subEmpty, memo))
      return
    }
  }
}

function lowerAssign(assign: BodyAssign, subst: SubstMap, memo: Memo): BodyAssign {
  // Post-Phase-0a: BodyAssign is OutputAssign-only.
  const expr = lowerExpr(assign.expr, subst, memo)
  if (expr === assign.expr) return assign
  const out: OutputAssign = { op: 'outputAssign', target: assign.target, expr }
  return out
}

// ─────────────────────────────────────────────────────────────
// Expression lowering
// ─────────────────────────────────────────────────────────────

function lowerExpr(expr: ResolvedExpr, subst: SubstMap, memo: Memo): ResolvedExpr {
  if (typeof expr === 'number' || typeof expr === 'boolean') return expr
  if (Array.isArray(expr)) {
    let changed = false
    const out: ResolvedExpr[] = expr.map(e => {
      const ne = lowerExpr(e, subst, memo)
      if (ne !== e) changed = true
      return ne
    })
    return changed ? out : expr
  }
  // Memoize — but only when subst is empty. With a non-empty subst the
  // map participates in the result and the memo (which is keyed by
  // input node only) would conflate calls under different bindings.
  // Practical impact: the top-level walk runs under empty subst and
  // sees deeply-shared sub-DAGs (e.g. Phaser16 chained ap_N bodies);
  // each combinator iteration runs under its own non-empty subst with
  // a fresh memo allocated by the caller below.
  if (subst.size === 0) {
    const cached = memo.get(expr)
    if (cached !== undefined) return cached
    const out = lowerOp(expr, subst, memo)
    memo.set(expr, out)
    return out
  }
  return lowerOp(expr, subst, memo)
}

function lowerOp(node: ResolvedExprOp, subst: SubstMap, memo: Memo): ResolvedExpr {
  switch (node.op) {
    // ── BindingRef: substitute when the binder is in the active map.
    //    Residuals (binder belongs to an outer combinator we haven't
    //    entered) survive — the caller's wrapping iteration will resolve
    //    them when it pushes its own bindings. ──
    case 'bindingRef': {
      const v = subst.get(node.idx)
      return v !== undefined ? v : node
    }

    // ── Combinators: unroll. Each iteration runs with a fresh subst
    //    and a fresh memo (the latter because the per-iteration subst
    //    differs and the existing memo would cache wrong values). ──
    case 'let':      return lowerLet(node, subst, memo)
    case 'fold':     return lowerFold(node, subst, memo)
    case 'scan':     return lowerScan(node, subst, memo)
    case 'generate': return lowerGenerate(node, subst, memo)
    case 'iterate':  return lowerIterate(node, subst, memo)
    case 'chain':    return lowerChain(node, subst, memo)
    case 'map2':     return lowerMap2(node, subst, memo)
    case 'zipWith':  return lowerZipWith(node, subst, memo)

    // ── zeros{count}: lower to inline `[0, ..., 0]` when count is a
    //    numeric literal. Otherwise leave it (e.g. count is a
    //    typeParamRef in an unspecialized program; arrayLower runs
    //    after specialize so this is a defensive case for callers
    //    that bypass the strata pipeline). ──
    case 'zeros': {
      const count = lowerExpr(node.count, subst, memo)
      if (typeof count === 'number') {
        return new Array(count).fill(0)
      }
      return count === node.count ? node : { op: 'zeros', count }
    }

    // ── References / leaves: pass through. ──
    case 'inputRef': case 'regRef': case 'paramRef':
    case 'typeParamRef': case 'nestedOut':
    case 'sampleRate': case 'sampleIndex':
      return node

    // ── Tag: recurse into payload values; leave structure intact. ──
    case 'tag': {
      let changed = false
      const payload = node.payload.map(p => {
        const value = lowerExpr(p.value, subst, memo)
        if (value !== p.value) changed = true
        return changed ? { field: p.field, value } : p
      })
      if (!changed) return node
      const fresh: Tag = { op: 'tag', variant: node.variant, payload }
      return fresh
    }

    // ── Match: recurse into scrutinee and arm bodies. Arm binders
    //    (payload bindings) stay — they're resolved by sum_lower
    //    when the scrutinee is a sum-typed DelayRef. arrayLower runs
    //    after sum_lower in the strata, so well-formed programs
    //    have no Match surviving here; defensive recursion. ──
    case 'match': {
      let changed = false
      const scrutinee = lowerExpr(node.scrutinee, subst, memo)
      if (scrutinee !== node.scrutinee) changed = true
      const arms: MatchArm[] = node.arms.map(arm => {
        const body = lowerExpr(arm.body, subst, memo)
        if (body !== arm.body) changed = true
        return body === arm.body ? arm : { variant: arm.variant, binders: arm.binders, body }
      })
      if (!changed) return node
      const fresh: Match = { op: 'match', type: node.type, scrutinee, arms }
      return fresh
    }

    // ── Operators: recurse into args, identity-preserve when nothing
    //    changed. ──
    case 'add': case 'sub': case 'mul': case 'div': case 'mod':
    case 'lt': case 'lte': case 'gt': case 'gte': case 'eq': case 'neq':
    case 'and': case 'or':
    case 'bitAnd': case 'bitOr': case 'bitXor': case 'lshift': case 'rshift':
    case 'floorDiv': case 'ldexp': {
      const a = lowerExpr(node.args[0], subst, memo)
      const b = lowerExpr(node.args[1], subst, memo)
      return (a === node.args[0] && b === node.args[1]) ? node : { op: node.op, args: [a, b] }
    }
    case 'neg': case 'not': case 'bitNot':
    case 'sqrt': case 'abs': case 'floor': case 'ceil': case 'round':
    case 'floatExponent': case 'toInt': case 'toBool': case 'toFloat': {
      const a = lowerExpr(node.args[0], subst, memo)
      return a === node.args[0] ? node : { op: node.op, args: [a] }
    }
    case 'clamp': case 'select': case 'arraySet': {
      const a = lowerExpr(node.args[0], subst, memo)
      const b = lowerExpr(node.args[1], subst, memo)
      const c = lowerExpr(node.args[2], subst, memo)
      return (a === node.args[0] && b === node.args[1] && c === node.args[2])
        ? node
        : { op: node.op, args: [a, b, c] }
    }
    case 'index': {
      const a = lowerExpr(node.args[0], subst, memo)
      const b = lowerExpr(node.args[1], subst, memo)
      return (a === node.args[0] && b === node.args[1]) ? node : { op: 'index', args: [a, b] }
    }
  }
}

// ─────────────────────────────────────────────────────────────
// Combinator unrolling
// ─────────────────────────────────────────────────────────────

/**
 * `let { b1: v1; b2: v2; ... } in body` — sequential let* binding.
 * Each value is evaluated in the current subst extended with the
 * already-bound entries; the body sees them all. Note that the
 * elaborator already enforces let* semantics: a later value's body
 * may contain `bindingRef` to an earlier binder, which arrives here
 * as a structural ref. We resolve those refs by extending subst
 * step-by-step.
 */
function lowerLet(
  node: { op: 'let'; binders: Array<{ binder: BinderDecl; value: ResolvedExpr }>; in: ResolvedExpr },
  subst: SubstMap,
  memo: Memo,
): ResolvedExpr {
  let cur: SubstMap = subst
  for (const entry of node.binders) {
    const value = lowerExpr(entry.value, cur, new WeakMap())
    cur = extend1(cur, entry.binder.idx, value)
  }
  return lowerExpr(node.in, cur, new WeakMap())
}

/** `fold(over, init, acc, elem, body)` — left fold to a scalar. The
 *  `over` argument must lower to an inline array (we statically
 *  enumerate its elements). */
function lowerFold(
  node: { op: 'fold'; over: ResolvedExpr; init: ResolvedExpr; acc: BinderDecl; elem: BinderDecl; body: ResolvedExpr },
  subst: SubstMap,
  memo: Memo,
): ResolvedExpr {
  const overLowered = lowerExpr(node.over, subst, memo)
  if (!Array.isArray(overLowered)) {
    throw new Error(`arrayLower: fold's 'over' did not lower to a static array`)
  }
  let acc = lowerExpr(node.init, subst, memo)
  for (const elem of overLowered) {
    const inner = extendN(subst, [[node.acc.idx, acc], [node.elem.idx, elem]])
    acc = lowerExpr(node.body, inner, new WeakMap())
  }
  return acc
}

/** `scan(over, init, acc, elem, body)` — like fold but emits the
 *  intermediate accumulators as an array. The first element is the
 *  result of the first body application (matching the legacy `lowerScan`,
 *  which pushes after each iteration — `init` itself is NOT included). */
function lowerScan(
  node: { op: 'scan'; over: ResolvedExpr; init: ResolvedExpr; acc: BinderDecl; elem: BinderDecl; body: ResolvedExpr },
  subst: SubstMap,
  memo: Memo,
): ResolvedExpr {
  const overLowered = lowerExpr(node.over, subst, memo)
  if (!Array.isArray(overLowered)) {
    throw new Error(`arrayLower: scan's 'over' did not lower to a static array`)
  }
  const out: ResolvedExpr[] = []
  let acc = lowerExpr(node.init, subst, memo)
  for (const elem of overLowered) {
    const inner = extendN(subst, [[node.acc.idx, acc], [node.elem.idx, elem]])
    acc = lowerExpr(node.body, inner, new WeakMap())
    out.push(acc)
  }
  return out
}

/** `generate(count, iter, body)` — emit `[body[iter=0], body[iter=1],
 *  ..., body[iter=count-1]]`. The `count` must lower to a numeric
 *  literal. */
function lowerGenerate(
  node: { op: 'generate'; count: ResolvedExpr; iter: BinderDecl; body: ResolvedExpr },
  subst: SubstMap,
  memo: Memo,
): ResolvedExpr {
  const n = lowerExpr(node.count, subst, memo)
  if (typeof n !== 'number') {
    throw new Error(`arrayLower: generate's 'count' did not lower to a numeric literal`)
  }
  const out: ResolvedExpr[] = []
  for (let i = 0; i < n; i++) {
    const inner = extend1(subst, node.iter.idx, i)
    out.push(lowerExpr(node.body, inner, new WeakMap()))
  }
  return out
}

/** `iterate(count, init, iter, body)` — emit `[init, f(init), f(f(init)),
 *  ..., f^(count-1)(init)]` (matching the legacy `lowerIterate`: the
 *  initial value is the first element, and each subsequent element is
 *  the body applied to the previous). */
function lowerIterate(
  node: { op: 'iterate'; count: ResolvedExpr; init: ResolvedExpr; iter: BinderDecl; body: ResolvedExpr },
  subst: SubstMap,
  memo: Memo,
): ResolvedExpr {
  const n = lowerExpr(node.count, subst, memo)
  if (typeof n !== 'number') {
    throw new Error(`arrayLower: iterate's 'count' did not lower to a numeric literal`)
  }
  const out: ResolvedExpr[] = []
  let cur = lowerExpr(node.init, subst, memo)
  for (let i = 0; i < n; i++) {
    out.push(cur)
    const inner = extend1(subst, node.iter.idx, cur)
    cur = lowerExpr(node.body, inner, new WeakMap())
  }
  return out
}

/** `chain(count, init, iter, body)` — apply `body` `count` times,
 *  threading the accumulator through. Result is the final accumulator
 *  (NOT an array). */
function lowerChain(
  node: { op: 'chain'; count: ResolvedExpr; init: ResolvedExpr; iter: BinderDecl; body: ResolvedExpr },
  subst: SubstMap,
  memo: Memo,
): ResolvedExpr {
  const n = lowerExpr(node.count, subst, memo)
  if (typeof n !== 'number') {
    throw new Error(`arrayLower: chain's 'count' did not lower to a numeric literal`)
  }
  let cur = lowerExpr(node.init, subst, memo)
  for (let i = 0; i < n; i++) {
    const inner = extend1(subst, node.iter.idx, cur)
    cur = lowerExpr(node.body, inner, new WeakMap())
  }
  return cur
}

/** `map2(over, elem, body)` — emit `[body[elem=e0], body[elem=e1], ...]`. */
function lowerMap2(
  node: { op: 'map2'; over: ResolvedExpr; elem: BinderDecl; body: ResolvedExpr },
  subst: SubstMap,
  memo: Memo,
): ResolvedExpr {
  const overLowered = lowerExpr(node.over, subst, memo)
  if (!Array.isArray(overLowered)) {
    throw new Error(`arrayLower: map2's 'over' did not lower to a static array`)
  }
  return overLowered.map(e => {
    const inner = extend1(subst, node.elem.idx, e)
    return lowerExpr(node.body, inner, new WeakMap())
  })
}

/** `zipWith(a, b, x, y, body)` — pointwise combination. The shorter
 *  array determines the output length (matching the legacy). */
function lowerZipWith(
  node: { op: 'zipWith'; a: ResolvedExpr; b: ResolvedExpr; x: BinderDecl; y: BinderDecl; body: ResolvedExpr },
  subst: SubstMap,
  memo: Memo,
): ResolvedExpr {
  const aLowered = lowerExpr(node.a, subst, memo)
  const bLowered = lowerExpr(node.b, subst, memo)
  if (!Array.isArray(aLowered) || !Array.isArray(bLowered)) {
    throw new Error(`arrayLower: zipWith's 'a' and 'b' did not both lower to static arrays`)
  }
  const n = Math.min(aLowered.length, bLowered.length)
  const out: ResolvedExpr[] = []
  for (let i = 0; i < n; i++) {
    const inner = extendN(subst, [[node.x.idx, aLowered[i]], [node.y.idx, bLowered[i]]])
    out.push(lowerExpr(node.body, inner, new WeakMap()))
  }
  return out
}
