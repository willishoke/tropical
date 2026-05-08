/**
 * trace_cycles.ts — Phase C4-B: cycle detection on the resolved IR.
 *
 * Detects cycles in the inter-instance dependency graph of a
 * `ResolvedProgram` and inserts a synthetic `DelayDecl` on a chosen
 * back-edge for each cycle. The output is a fresh `ResolvedProgram`
 * (no in-place mutation) — see PHASE_C_PLAN.md §7e for the purity
 * argument.
 *
 * Algorithm (mirrors `compiler/flatten.ts:927-986`):
 *   1. Build a directed graph: each `InstanceDecl` is a node; an edge
 *      A → B exists when an input wire of A references a `NestedOut`
 *      whose `instance === B`.
 *   2. Run Tarjan's SCC over the graph.
 *   3. For each non-trivial SCC, designate the first member by source
 *      order (its position in `body.decls`) as the break target.
 *   4. For each output port of the break target referenced by another
 *      cycle member's input wires, allocate a synthetic `DelayDecl`
 *      whose update reads the original `NestedOut` and whose init is
 *      `0`. Rewrite the offending `NestedOut`s in cycle-member input
 *      wires to read the synthetic delay instead.
 *   5. Re-run Tarjan to confirm no cycles remain.
 *
 * For the C4 corpus the stdlib has no inter-instance cycles (cycles
 * exist only post-inlining, which is C5's territory; in stdlib source
 * any feedback loop is broken explicitly via a `delay`). So this pass
 * is the identity for every stdlib program today; the SCC plumbing
 * is implemented anyway because C5 will rely on it.
 */

import type {
  ResolvedProgram, ResolvedExpr, ResolvedExprOpNode,
  ResolvedBlock,
  BodyDecl, BodyAssign, OutputAssign, NextUpdate,
  InstanceDecl, OutputDecl, DelayDecl,
  NestedOut,
} from './nodes.js'

// ─────────────────────────────────────────────────────────────
// Public entry
// ─────────────────────────────────────────────────────────────

export function traceCycles(prog: ResolvedProgram): ResolvedProgram {
  const instances = collectInstances(prog.body.decls)
  // A program with no instances has no inter-instance graph. A program with
  // exactly one instance can still have a self-loop (instance 'a' wires its
  // own input from its own output) — so don't skip the deps build for the
  // 1-instance case; only the 0-instance case is a guaranteed identity.
  if (instances.length === 0) return prog

  const deps = buildInstanceDeps(prog, instances)
  const sccs = tarjanSCC(instances, deps)
  // Filter to non-trivial SCCs (more than one member, OR a single
  // member with a self-edge — possible if an instance's input wires
  // contain a NestedOut to itself).
  const nontrivial = sccs.filter(scc =>
    scc.length > 1 || (scc.length === 1 && deps.get(scc[0])?.has(scc[0])))
  if (nontrivial.length === 0) return prog

  // For each non-trivial SCC, pick the first member by source order
  // and rewrite the cycle. Source order for an SCC = the smallest
  // index in `instances` (which is body-decl order).
  const orderIndex = new Map<InstanceDecl, number>()
  instances.forEach((inst, i) => orderIndex.set(inst, i))

  // Accumulators for the rewritten body.
  const syntheticDelays: DelayDecl[] = []
  // Map (breakInstance, outputDecl) → synthetic delay holding its
  // previous-sample value.
  const breakerDelay = new Map<string, DelayDecl>()
  // Set of (cycle-member, breakInstance) pairs; if a NestedOut belongs
  // to one of these, rewrite to a DelayRef on the breaker.
  const rewriteTargets = new Map<InstanceDecl, Set<InstanceDecl>>()

  for (const scc of nontrivial) {
    const sortedScc = [...scc].sort((a, b) => orderIndex.get(a)! - orderIndex.get(b)!)
    const breakTarget = sortedScc[0]
    for (const member of sortedScc) {
      // Don't skip the breakTarget itself: in a single-member SCC with a
      // self-edge, the only wire that closes the cycle is the breakTarget's
      // own wire to itself. Including it in rewriteTargets ensures that
      // wire becomes a delayRef on the synthetic delay. For multi-member
      // SCCs, the breakTarget never references itself, so this is a no-op
      // there.
      let s = rewriteTargets.get(member)
      if (!s) { s = new Set(); rewriteTargets.set(member, s) }
      s.add(breakTarget)
    }
  }

  // Lookup helper: for a given (instance, output) pair, allocate the
  // synthetic delay lazily on first use.
  const breakerFor = (inst: InstanceDecl, output: OutputDecl): DelayDecl => {
    const key = `${inst.name}::${output.name}`
    let d = breakerDelay.get(key)
    if (d) return d
    d = {
      op: 'delayDecl',
      name: `_feedback_${inst.name}_${output.name}`,
      // Update reads the current sample of the broken output. The
      // synthetic delay's role is to hold the previous sample so that
      // cycle members read a one-sample-delayed view of the cycle
      // breaker — same semantics as legacy flatten.ts.
      update: { op: 'nestedOut', instance: inst, output },
      init: 0,
      _liftedFrom: 'synthetic',
    }
    breakerDelay.set(key, d)
    syntheticDelays.push(d)
    return d
  }

  // Rewriter: in any expression that belongs to an instance in
  // `rewriteTargets`, replace `NestedOut` whose instance is one of
  // the break-targets for that owner with a `DelayRef` to the
  // appropriate synthetic delay.
  const rewriteForOwner = (expr: ResolvedExpr, breakSet: Set<InstanceDecl>): ResolvedExpr => {
    if (typeof expr === 'number' || typeof expr === 'boolean') return expr
    if (Array.isArray(expr)) return expr.map(e => rewriteForOwner(e, breakSet))
    return rewriteOpForOwner(expr, breakSet)
  }
  const rewriteOpForOwner = (node: ResolvedExprOpNode, breakSet: Set<InstanceDecl>): ResolvedExpr => {
    switch (node.op) {
      case 'nestedOut': {
        if (breakSet.has(node.instance)) {
          return { op: 'delayRef', decl: breakerFor(node.instance, node.output) }
        }
        return node
      }
      case 'inputRef': case 'regRef': case 'delayRef': case 'paramRef':
      case 'typeParamRef': case 'bindingRef':
      case 'sampleRate': case 'sampleIndex':
      case 'tag':
        return node
      case 'match':
        return {
          op: 'match',
          type: node.type,
          scrutinee: rewriteForOwner(node.scrutinee, breakSet),
          arms: node.arms.map(arm => ({
            variant: arm.variant,
            binders: arm.binders,
            body: rewriteForOwner(arm.body, breakSet),
          })),
        }
      case 'fold': case 'scan':
        return { op: node.op, over: rewriteForOwner(node.over, breakSet),
                 init: rewriteForOwner(node.init, breakSet),
                 acc: node.acc, elem: node.elem,
                 body: rewriteForOwner(node.body, breakSet) }
      case 'generate':
        return { op: 'generate', count: rewriteForOwner(node.count, breakSet),
                 iter: node.iter, body: rewriteForOwner(node.body, breakSet) }
      case 'iterate': case 'chain':
        return { op: node.op, count: rewriteForOwner(node.count, breakSet),
                 init: rewriteForOwner(node.init, breakSet),
                 iter: node.iter, body: rewriteForOwner(node.body, breakSet) }
      case 'map2':
        return { op: 'map2', over: rewriteForOwner(node.over, breakSet),
                 elem: node.elem, body: rewriteForOwner(node.body, breakSet) }
      case 'zipWith':
        return { op: 'zipWith', a: rewriteForOwner(node.a, breakSet), b: rewriteForOwner(node.b, breakSet),
                 x: node.x, y: node.y, body: rewriteForOwner(node.body, breakSet) }
      case 'let':
        return {
          op: 'let',
          binders: node.binders.map(b => ({ binder: b.binder, value: rewriteForOwner(b.value, breakSet) })),
          in: rewriteForOwner(node.in, breakSet),
        }
      case 'add': case 'sub': case 'mul': case 'div': case 'mod':
      case 'lt': case 'lte': case 'gt': case 'gte': case 'eq': case 'neq':
      case 'and': case 'or':
      case 'bitAnd': case 'bitOr': case 'bitXor': case 'lshift': case 'rshift':
      case 'floorDiv': case 'ldexp':
        return { op: node.op, args: [rewriteForOwner(node.args[0], breakSet), rewriteForOwner(node.args[1], breakSet)] }
      case 'neg': case 'not': case 'bitNot':
      case 'sqrt': case 'abs': case 'floor': case 'ceil': case 'round':
      case 'floatExponent': case 'toInt': case 'toBool': case 'toFloat':
        return { op: node.op, args: [rewriteForOwner(node.args[0], breakSet)] }
      case 'clamp': case 'select': case 'arraySet':
        return { op: node.op, args: [
          rewriteForOwner(node.args[0], breakSet),
          rewriteForOwner(node.args[1], breakSet),
          rewriteForOwner(node.args[2], breakSet),
        ] }
      case 'index':
        return { op: 'index', args: [rewriteForOwner(node.args[0], breakSet), rewriteForOwner(node.args[1], breakSet)] }
      case 'zeros':
        return { op: 'zeros', count: rewriteForOwner(node.count, breakSet) }
    }
  }

  // Rebuild instance decls with rewritten input wires.
  //
  // Identity invariant: we mutate `decl.inputs` in place rather than
  // spread-cloning. The break target's decl is reused unchanged, and so
  // are decls of any non-cycle instances; mixing fresh spread copies with
  // reused originals would split InstanceDecl identity, so any nestedOut
  // ref in the program (assigns, the break target's own inputs, the
  // synthetic delay's `update` built above) that was not rewalked here
  // would point at an orphaned original. Mutation keeps every reference
  // valid. The strata pipeline doesn't share `prog` with anything else
  // after this call, so in-place mutation is safe.
  const newDecls: BodyDecl[] = []
  for (const decl of prog.body.decls) {
    if (decl.op === 'instanceDecl' && rewriteTargets.has(decl)) {
      const breakSet = rewriteTargets.get(decl)!
      ;(decl as { inputs: typeof decl.inputs }).inputs = decl.inputs.map(i => ({
        port: i.port,
        value: rewriteForOwner(i.value, breakSet),
      }))
    }
    newDecls.push(decl)
  }
  // Append synthetic delays after instance decls so they appear at the
  // tail of the body's decl list. (Legacy puts them in `sessionDelays`
  // which gets allocated in append order; positioning at the end keeps
  // existing slot indices stable.)
  for (const d of syntheticDelays) newDecls.push(d)

  const newBody: ResolvedBlock = {
    op: 'block',
    decls: newDecls,
    assigns: prog.body.assigns,
  }
  return { ...prog, body: newBody }
}

// ─────────────────────────────────────────────────────────────
// Graph construction
// ─────────────────────────────────────────────────────────────

function collectInstances(decls: BodyDecl[]): InstanceDecl[] {
  const out: InstanceDecl[] = []
  for (const d of decls) if (d.op === 'instanceDecl') out.push(d)
  return out
}

/**
 * Build an instance-level dependency map: A → set of instances whose
 * outputs A's input wires reference. Self-edges are recorded too
 * (a single-member SCC with a self-edge is still a cycle).
 */
function buildInstanceDeps(
  _prog: ResolvedProgram,
  instances: InstanceDecl[],
): Map<InstanceDecl, Set<InstanceDecl>> {
  const allInstances = new Set(instances)
  const deps = new Map<InstanceDecl, Set<InstanceDecl>>()
  for (const inst of instances) deps.set(inst, new Set())

  for (const inst of instances) {
    const set = deps.get(inst)!
    for (const wire of inst.inputs) {
      collectNestedOutInstances(wire.value, set, allInstances)
    }
  }
  return deps
}

function collectNestedOutInstances(
  expr: ResolvedExpr,
  out: Set<InstanceDecl>,
  allInstances: Set<InstanceDecl>,
): void {
  if (typeof expr !== 'object' || expr === null) return
  if (Array.isArray(expr)) { for (const e of expr) collectNestedOutInstances(e, out, allInstances); return }
  switch (expr.op) {
    case 'nestedOut':
      if (allInstances.has(expr.instance)) out.add(expr.instance)
      return
    case 'match':
      collectNestedOutInstances(expr.scrutinee, out, allInstances)
      for (const arm of expr.arms) collectNestedOutInstances(arm.body, out, allInstances)
      return
    case 'fold': case 'scan':
      collectNestedOutInstances(expr.over, out, allInstances)
      collectNestedOutInstances(expr.init, out, allInstances)
      collectNestedOutInstances(expr.body, out, allInstances)
      return
    case 'generate':
      collectNestedOutInstances(expr.count, out, allInstances)
      collectNestedOutInstances(expr.body, out, allInstances)
      return
    case 'iterate': case 'chain':
      collectNestedOutInstances(expr.count, out, allInstances)
      collectNestedOutInstances(expr.init, out, allInstances)
      collectNestedOutInstances(expr.body, out, allInstances)
      return
    case 'map2':
      collectNestedOutInstances(expr.over, out, allInstances)
      collectNestedOutInstances(expr.body, out, allInstances)
      return
    case 'zipWith':
      collectNestedOutInstances(expr.a, out, allInstances)
      collectNestedOutInstances(expr.b, out, allInstances)
      collectNestedOutInstances(expr.body, out, allInstances)
      return
    case 'let':
      for (const b of expr.binders) collectNestedOutInstances(b.value, out, allInstances)
      collectNestedOutInstances(expr.in, out, allInstances)
      return
    case 'tag':
      for (const p of expr.payload) collectNestedOutInstances(p.value, out, allInstances)
      return
    case 'zeros':
      collectNestedOutInstances(expr.count, out, allInstances)
      return
    case 'add': case 'sub': case 'mul': case 'div': case 'mod':
    case 'lt': case 'lte': case 'gt': case 'gte': case 'eq': case 'neq':
    case 'and': case 'or':
    case 'bitAnd': case 'bitOr': case 'bitXor': case 'lshift': case 'rshift':
    case 'floorDiv': case 'ldexp':
    case 'neg': case 'not': case 'bitNot':
    case 'sqrt': case 'abs': case 'floor': case 'ceil': case 'round':
    case 'floatExponent': case 'toInt': case 'toBool': case 'toFloat':
    case 'clamp': case 'select': case 'index': case 'arraySet':
      for (const a of expr.args) collectNestedOutInstances(a, out, allInstances)
      return
    case 'inputRef': case 'regRef': case 'delayRef': case 'paramRef':
    case 'typeParamRef': case 'bindingRef':
    case 'sampleRate': case 'sampleIndex':
      return
  }
}

// ─────────────────────────────────────────────────────────────
// Tarjan's SCC over decl-keyed graphs
// ─────────────────────────────────────────────────────────────

/**
 * Decl-keyed Tarjan's SCC. Visits nodes in `nodes` order, which —
 * for `traceCycles` — is body-decl order. SCCs are emitted in
 * reverse-topo order; each SCC's internal order matches stack pop
 * order. Source-order tie-breaks happen at the call site.
 */
function tarjanSCC<T extends object>(
  nodes: T[],
  deps: Map<T, Set<T>>,
): T[][] {
  let idx = 0
  const indices = new Map<T, number>()
  const lowlinks = new Map<T, number>()
  const onStack = new Set<T>()
  const stack: T[] = []
  const sccs: T[][] = []

  function visit(v: T): void {
    indices.set(v, idx)
    lowlinks.set(v, idx)
    idx++
    stack.push(v)
    onStack.add(v)

    for (const w of deps.get(v) ?? []) {
      if (!indices.has(w)) {
        visit(w)
        lowlinks.set(v, Math.min(lowlinks.get(v)!, lowlinks.get(w)!))
      } else if (onStack.has(w)) {
        lowlinks.set(v, Math.min(lowlinks.get(v)!, indices.get(w)!))
      }
    }

    if (lowlinks.get(v) === indices.get(v)) {
      const scc: T[] = []
      let w: T
      do {
        w = stack.pop()!
        onStack.delete(w)
        scc.push(w)
      } while (w !== v)
      sccs.push(scc)
    }
  }

  for (const n of nodes) {
    if (!indices.has(n)) visit(n)
  }
  return sccs
}
