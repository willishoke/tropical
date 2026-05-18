/**
 * compiler/ir/lowering/cycle_break.ts — shared cycle-break helper.
 *
 * This module is the single home of the cycle-detection + cycle-break
 * algorithm for tropical's resolved IR. Two consumers:
 *
 *   - `compiler/ir/trace_cycles.ts` — Phase 0+ shim that calls
 *     `breakInstanceCycles` to produce the post-trace IR for the
 *     compiler's strata pipeline. (Phase 3 retires this in-pipeline
 *     use; cycle-breaking moves to the realization layer above the
 *     compiler.)
 *
 *   - `compiler/ir/acyclic.ts` — the strataPipeline-boundary
 *     acyclicity check, which consumes only `findInstanceCycles`
 *     (the detector) and `AcyclicityViolation` to assert the
 *     post-trace invariant.
 *
 * Algorithm:
 *   1. Build an instance-level dependency graph: instance A depends
 *      on instance B iff some `NestedOut` ref in A's input wires
 *      reads from B's outputs.
 *   2. Run Tarjan's SCC over the graph.
 *   3. Non-trivial SCCs are cycles: more than one member, OR a
 *      single member with a self-edge.
 *   4. For each cycle, pick the first member by source order
 *      (instance position in body.decls) as the break target.
 *   5. For each output of the break target referenced by cycle
 *      members, allocate a synthetic `RegDecl` whose update reads
 *      the current-sample value of that output. The synthetic reg
 *      is tagged `_liftedFrom: 'synthetic'`. Cycle members rewrite
 *      `NestedOut(breakTarget.out)` to `RegRef(syntheticReg)`,
 *      reading the previous-sample value and breaking the cycle
 *      via the unit-delay semantics of the reg.
 *
 * Pure: returns a fresh `ResolvedProgram` when rewrites occur, or
 * the input by identity when no cycles are present.
 */

import type {
  ResolvedProgram, ResolvedExpr, ResolvedExprOp,
  ResolvedBlock,
  BodyDecl,
  InstanceDecl, OutputDecl, RegDecl,
} from '../nodes.js'

// ─────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────

/** Find every non-trivial SCC in `prog`'s inter-instance dependency
 *  graph. A non-trivial SCC has more than one member OR a single
 *  member with a self-edge. Pure: no mutation, no side effects. */
export function findInstanceCycles(prog: ResolvedProgram): InstanceDecl[][] {
  const instances = collectInstances(prog.body.decls)
  if (instances.length === 0) return []
  const deps = buildInstanceDeps(instances)
  const sccs = tarjanSCC(instances, deps)
  return sccs.filter(scc =>
    scc.length > 1 || (scc.length === 1 && deps.get(scc[0])?.has(scc[0])),
  )
}

/** Provenance for one broken cycle. Returned alongside the lowered
 *  program so callers (debuggers, error formatters) can localize. */
export interface BrokenCycle {
  readonly scc: ReadonlyArray<InstanceDecl>
  readonly breakTarget: InstanceDecl
  /** Output ports of `breakTarget` that were promoted to synthetic
   *  regs (one per port referenced by other cycle members). */
  readonly breakPorts: ReadonlyArray<OutputDecl>
}

/** Result of `breakInstanceCycles`. */
export interface CycleBreakResult {
  readonly lowered: ResolvedProgram
  readonly syntheticRegs: ReadonlyArray<RegDecl>
  readonly cycles: ReadonlyArray<BrokenCycle>
}

/** Break every cycle in `prog`'s inter-instance graph by inserting
 *  synthetic regs (one-sample-delay registers). Returns the lowered
 *  acyclic program plus provenance metadata. When `prog` has no
 *  cycles, returns the input by identity (lowered === prog,
 *  syntheticRegs empty, cycles empty). */
export function breakInstanceCycles(prog: ResolvedProgram): CycleBreakResult {
  const instances = collectInstances(prog.body.decls)
  if (instances.length === 0) {
    return { lowered: prog, syntheticRegs: [], cycles: [] }
  }

  const deps = buildInstanceDeps(instances)
  const sccs = tarjanSCC(instances, deps)
  const nontrivial = sccs.filter(scc =>
    scc.length > 1 || (scc.length === 1 && deps.get(scc[0])?.has(scc[0])),
  )
  if (nontrivial.length === 0) {
    return { lowered: prog, syntheticRegs: [], cycles: [] }
  }

  // Per non-trivial SCC: pick the first member by source order
  // (smallest index in `instances`, which is body-decl order) as
  // the break target. Record the (cycle-member → break-target)
  // edges so the rewriter can rewrite NestedOuts pointing at the
  // break target.
  const orderIndex = new Map<InstanceDecl, number>()
  instances.forEach((inst, i) => orderIndex.set(inst, i))

  const syntheticRegs: RegDecl[] = []
  const breakerReg = new Map<string, RegDecl>()
  const rewriteTargets = new Map<InstanceDecl, Set<InstanceDecl>>()
  const cyclesProvenance: BrokenCycle[] = []
  const breakerPortsByCycle = new Map<InstanceDecl, OutputDecl[]>()

  for (const scc of nontrivial) {
    const sortedScc = [...scc].sort((a, b) => orderIndex.get(a)! - orderIndex.get(b)!)
    const breakTarget = sortedScc[0]
    breakerPortsByCycle.set(breakTarget, [])
    for (const member of sortedScc) {
      let s = rewriteTargets.get(member)
      if (!s) { s = new Set(); rewriteTargets.set(member, s) }
      s.add(breakTarget)
    }
    cyclesProvenance.push({
      scc: sortedScc,
      breakTarget,
      breakPorts: breakerPortsByCycle.get(breakTarget)!,
    })
  }

  // Allocate the synthetic reg for a given (breakTarget, output) on
  // first use; subsequent references resolve to the same reg.
  const breakerFor = (inst: InstanceDecl, output: OutputDecl): RegDecl => {
    const key = `${inst.name}::${output.name}`
    let d = breakerReg.get(key)
    if (d) return d
    d = {
      op: 'regDecl',
      name: `_feedback_${inst.name}_${output.name}`,
      // The synthetic reg's update reads the current sample of the
      // broken output; the reg holds that value to make it readable
      // one sample later by the cycle members. The `_feedback_` name
      // prefix distinguishes these from user-written regs (consumed
      // by trace_cycles.test.ts and any future analyses that want
      // to identify cycle-break regs). Post-Phase 4b strict policy
      // means production code paths never produce these (cycles in
      // source throw at elaborate-time); the helper survives for
      // direct invocation by tests and future realizations.
      update: { op: 'nestedOut', instance: inst, output },
      init: 0,
    }
    breakerReg.set(key, d)
    syntheticRegs.push(d)
    const ports = breakerPortsByCycle.get(inst)
    if (ports) ports.push(output)
    return d
  }

  // Rewriter: in any expression that belongs to an instance in
  // `rewriteTargets`, replace `NestedOut` whose instance is one of
  // the break-targets for that owner with a `RegRef` to the
  // appropriate synthetic reg.
  const rewriteForOwner = (expr: ResolvedExpr, breakSet: Set<InstanceDecl>): ResolvedExpr => {
    if (typeof expr === 'number' || typeof expr === 'boolean') return expr
    if (Array.isArray(expr)) return expr.map(e => rewriteForOwner(e, breakSet))
    return rewriteOpForOwner(expr, breakSet)
  }
  const rewriteOpForOwner = (node: ResolvedExprOp, breakSet: Set<InstanceDecl>): ResolvedExpr => {
    switch (node.op) {
      case 'nestedOut': {
        if (breakSet.has(node.instance)) {
          return { op: 'regRef', decl: breakerFor(node.instance, node.output) }
        }
        return node
      }
      case 'inputRef': case 'regRef': case 'paramRef':
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

  // Rebuild instance decls with rewritten input wires. We mutate
  // `decl.inputs` in place rather than spread-cloning to preserve
  // InstanceDecl identity — non-cycle instances stay the same object,
  // so any other expression in the program (assigns, other decls'
  // inits, synthetic regs' updates built above) referencing them
  // remains valid by ===.
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
  // Synthetic regs go at the tail of the body's decl list, preserving
  // existing slot-allocation order for non-synthetic decls.
  for (const d of syntheticRegs) newDecls.push(d)

  const newBody: ResolvedBlock = {
    op: 'block',
    decls: newDecls,
    assigns: prog.body.assigns,
  }
  return {
    lowered: { ...prog, body: newBody },
    syntheticRegs,
    cycles: cyclesProvenance,
  }
}

// ─────────────────────────────────────────────────────────────
// Internals — graph construction + SCC
// ─────────────────────────────────────────────────────────────

function collectInstances(decls: BodyDecl[]): InstanceDecl[] {
  const out: InstanceDecl[] = []
  for (const d of decls) if (d.op === 'instanceDecl') out.push(d)
  return out
}

function buildInstanceDeps(
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
  if (Array.isArray(expr)) {
    for (const e of expr) collectNestedOutInstances(e, out, allInstances)
    return
  }
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
    case 'inputRef': case 'regRef': case 'paramRef':
    case 'typeParamRef': case 'bindingRef':
    case 'sampleRate': case 'sampleIndex':
      return
  }
}

/** Tarjan's strongly connected components algorithm. */
function tarjanSCC<T>(
  nodes: T[],
  deps: ReadonlyMap<T, ReadonlySet<T>>,
): T[][] {
  let index = 0
  const indexOf = new Map<T, number>()
  const lowlink = new Map<T, number>()
  const onStack = new Set<T>()
  const stack: T[] = []
  const sccs: T[][] = []

  const strongConnect = (v: T): void => {
    indexOf.set(v, index)
    lowlink.set(v, index)
    index++
    stack.push(v)
    onStack.add(v)
    const successors = deps.get(v) ?? new Set<T>()
    for (const w of successors) {
      if (!indexOf.has(w)) {
        strongConnect(w)
        lowlink.set(v, Math.min(lowlink.get(v)!, lowlink.get(w)!))
      } else if (onStack.has(w)) {
        lowlink.set(v, Math.min(lowlink.get(v)!, indexOf.get(w)!))
      }
    }
    if (lowlink.get(v) === indexOf.get(v)) {
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

  for (const v of nodes) {
    if (!indexOf.has(v)) strongConnect(v)
  }
  return sccs
}
