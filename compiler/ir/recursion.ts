/**
 * recursion.ts — structural recursion over `ResolvedExpr`.
 *
 * One generic walker (`mapExpr`) owns the structural traversal across
 * every `ResolvedExprOp` variant. Each pass that walks the expression
 * graph supplies a small set of hooks describing only the cases it
 * cares about; everything else recurses structurally for free. The
 * payoff: passes like `specialize`, `inline_instances`,
 * `identity_elim`, and `array_lower` no longer each duplicate a
 * ~200-line op-by-op switch, and the "defensive clone before
 * mutating" pattern goes away because the result is constructed
 * functionally — no mutation, no shared-state risk.
 *
 * Hook contract:
 *   - `expr(e)`: return a replacement to short-circuit recursion at
 *     this node, or `NoRewrite` to recurse structurally.
 *   - `binder(b)`: transform a `BinderDecl` (combinator/let/arm
 *     binding-site decl). Called when the walker descends into a
 *     combinator node. If omitted, binders pass through unchanged.
 *   - `shapeDim(d)`: transform a `ShapeDim` (used by `mapPortType`,
 *     not by `mapExpr` directly — kept here for one-stop hook docs).
 *
 * Identity-on-no-change: when `expr` returns `NoRewrite` and a node
 * has no children that changed, the walker returns the same node
 * object (preserving DAG sharing with the source). Callers that
 * require fresh objects everywhere should not rely on this — but
 * for functional passes it's the natural and efficient behavior.
 *
 * Why a Symbol for `NoRewrite`: it's a unique runtime value distinct
 * from any valid `ResolvedExpr`, with no string-tag collision risk.
 * Ports cleanly to Haskell as a separate nullary data constructor
 * (`data RewriteResult = NoRewrite | Rewrite ResolvedExpr`).
 */

import type {
  ResolvedExpr, ResolvedExprOp,
  BinderDecl, ShapeDim, PortType,
  Tag, Match, MatchArm,
  Let, Fold, Scan, Generate, Iterate, Chain, Map2, ZipWith,
} from './nodes.js'

// ─────────────────────────────────────────────────────────────
// NoRewrite sentinel + hook signature
// ─────────────────────────────────────────────────────────────

export const NoRewrite: unique symbol = Symbol('NoRewrite')
export type NoRewrite = typeof NoRewrite

export type ExprRewrite = (e: ResolvedExpr) => ResolvedExpr | NoRewrite

export interface MapHooks {
  /** Called for every `ResolvedExpr` encountered, before structural
   *  recursion. Return a value to replace; return `NoRewrite` to
   *  recurse. */
  expr?: ExprRewrite
  /** Optional binder transformer. When the walker descends into a
   *  combinator, let, or match arm, this hook is called on each
   *  `BinderDecl`. Default: pass through unchanged. */
  binder?: (b: BinderDecl) => BinderDecl
}

// ─────────────────────────────────────────────────────────────
// mapExpr — the structural walker
// ─────────────────────────────────────────────────────────────

export function mapExpr(e: ResolvedExpr, hooks: MapHooks): ResolvedExpr {
  if (hooks.expr !== undefined) {
    const handled = hooks.expr(e)
    if (handled !== NoRewrite) return handled
  }
  if (typeof e === 'number' || typeof e === 'boolean') return e
  if (Array.isArray(e)) {
    let changed = false
    const out = e.map(x => {
      const r = mapExpr(x, hooks)
      if (r !== x) changed = true
      return r
    })
    return changed ? out : e
  }
  return mapOpNode(e, hooks)
}

function mapOpNode(node: ResolvedExprOp, hooks: MapHooks): ResolvedExprOp {
  const recur = (x: ResolvedExpr) => mapExpr(x, hooks)
  const mapBinder = (b: BinderDecl) => hooks.binder !== undefined ? hooks.binder(b) : b

  switch (node.op) {
    // Leaves — no children to walk.
    case 'inputRef': case 'regRef': case 'paramRef': case 'typeParamRef':
    case 'nestedOut': case 'bindingRef':
    case 'sampleRate': case 'sampleIndex':
      return node

    // ADT.
    case 'tag': {
      let changed = false
      const payload = node.payload.map(p => {
        const v = recur(p.value)
        if (v !== p.value) changed = true
        return v === p.value ? p : { field: p.field, value: v }
      })
      if (!changed) return node
      const fresh: Tag = { op: 'tag', variant: node.variant, payload }
      return fresh
    }
    case 'match': {
      const scrutinee = recur(node.scrutinee)
      let armsChanged = false
      const arms: MatchArm[] = node.arms.map(arm => {
        const body = recur(arm.body)
        const binders = hooks.binder !== undefined ? arm.binders.map(mapBinder) : arm.binders
        const bChanged = binders !== arm.binders || body !== arm.body
        if (bChanged) armsChanged = true
        return bChanged ? { variant: arm.variant, binders, body } : arm
      })
      if (scrutinee === node.scrutinee && !armsChanged) return node
      const fresh: Match = { op: 'match', type: node.type, scrutinee, arms }
      return fresh
    }

    // Let.
    case 'let': {
      let changed = false
      const binders = node.binders.map(entry => {
        const binder = mapBinder(entry.binder)
        const value = recur(entry.value)
        if (binder !== entry.binder || value !== entry.value) changed = true
        return (binder === entry.binder && value === entry.value) ? entry : { binder, value }
      })
      const inResult = recur(node.in)
      if (!changed && inResult === node.in) return node
      const fresh: Let = { op: 'let', binders, in: inResult }
      return fresh
    }

    // Combinators with one binder.
    case 'generate': {
      const count = recur(node.count)
      const iter = mapBinder(node.iter)
      const body = recur(node.body)
      if (count === node.count && iter === node.iter && body === node.body) return node
      const fresh: Generate = { op: 'generate', count, iter, body }
      return fresh
    }
    case 'iterate': {
      const count = recur(node.count)
      const init  = recur(node.init)
      const iter  = mapBinder(node.iter)
      const body  = recur(node.body)
      if (count === node.count && init === node.init && iter === node.iter && body === node.body) return node
      const fresh: Iterate = { op: 'iterate', count, init, iter, body }
      return fresh
    }
    case 'chain': {
      const count = recur(node.count)
      const init  = recur(node.init)
      const iter  = mapBinder(node.iter)
      const body  = recur(node.body)
      if (count === node.count && init === node.init && iter === node.iter && body === node.body) return node
      const fresh: Chain = { op: 'chain', count, init, iter, body }
      return fresh
    }
    case 'map2': {
      const over = recur(node.over)
      const elem = mapBinder(node.elem)
      const body = recur(node.body)
      if (over === node.over && elem === node.elem && body === node.body) return node
      const fresh: Map2 = { op: 'map2', over, elem, body }
      return fresh
    }

    // Combinators with two binders.
    case 'fold': {
      const over = recur(node.over)
      const init = recur(node.init)
      const acc  = mapBinder(node.acc)
      const elem = mapBinder(node.elem)
      const body = recur(node.body)
      if (over === node.over && init === node.init && acc === node.acc && elem === node.elem && body === node.body) return node
      const fresh: Fold = { op: 'fold', over, init, acc, elem, body }
      return fresh
    }
    case 'scan': {
      const over = recur(node.over)
      const init = recur(node.init)
      const acc  = mapBinder(node.acc)
      const elem = mapBinder(node.elem)
      const body = recur(node.body)
      if (over === node.over && init === node.init && acc === node.acc && elem === node.elem && body === node.body) return node
      const fresh: Scan = { op: 'scan', over, init, acc, elem, body }
      return fresh
    }
    case 'zipWith': {
      const a = recur(node.a)
      const b = recur(node.b)
      const x = mapBinder(node.x)
      const y = mapBinder(node.y)
      const body = recur(node.body)
      if (a === node.a && b === node.b && x === node.x && y === node.y && body === node.body) return node
      const fresh: ZipWith = { op: 'zipWith', a, b, x, y, body }
      return fresh
    }

    // Operators with uniform `args` arity.
    case 'add': case 'sub': case 'mul': case 'div': case 'mod':
    case 'lt': case 'lte': case 'gt': case 'gte': case 'eq': case 'neq':
    case 'and': case 'or':
    case 'bitAnd': case 'bitOr': case 'bitXor': case 'lshift': case 'rshift':
    case 'floorDiv': case 'ldexp': {
      const a = recur(node.args[0])
      const b = recur(node.args[1])
      if (a === node.args[0] && b === node.args[1]) return node
      return { op: node.op, args: [a, b] }
    }
    case 'neg': case 'not': case 'bitNot':
    case 'sqrt': case 'abs': case 'floor': case 'ceil': case 'round':
    case 'floatExponent': case 'toInt': case 'toBool': case 'toFloat': {
      const a = recur(node.args[0])
      if (a === node.args[0]) return node
      return { op: node.op, args: [a] }
    }
    case 'clamp': case 'select': case 'arraySet': {
      const a = recur(node.args[0])
      const b = recur(node.args[1])
      const c = recur(node.args[2])
      if (a === node.args[0] && b === node.args[1] && c === node.args[2]) return node
      return { op: node.op, args: [a, b, c] }
    }
    case 'index': {
      const a = recur(node.args[0])
      const b = recur(node.args[1])
      if (a === node.args[0] && b === node.args[1]) return node
      return { op: 'index', args: [a, b] }
    }
    case 'zeros': {
      const count = recur(node.count)
      if (count === node.count) return node
      return { op: 'zeros', count }
    }
  }
}

// ─────────────────────────────────────────────────────────────
// Port-type / shape-dim walker (small, used by specialize)
// ─────────────────────────────────────────────────────────────

export function mapPortType(
  pt: PortType,
  shapeDim: (d: ShapeDim) => ShapeDim,
): PortType {
  if (pt.kind !== 'array') return pt
  let changed = false
  const shape = pt.shape.map(d => {
    const r = shapeDim(d)
    if (r !== d) changed = true
    return r
  })
  if (!changed) return pt
  return { kind: 'array', element: pt.element, shape }
}
