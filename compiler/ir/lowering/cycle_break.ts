/**
 * compiler/ir/lowering/cycle_break.ts — shared cycle-break helper.
 *
 * This module is the single home of the cycle-detection + cycle-break
 * algorithm for tropical's resolved IR. Cycle-breaking is the
 * responsibility of the realization layer above the compiler (the
 * elaborator throws `CycleViolation` on source-level cycles, the
 * session materializer extracts session-level `delay()` ops); this
 * file exposes the shared algorithm for realization-side use and for
 * the strataPipeline-boundary acyclicity assertion in
 * `compiler/ir/acyclic.ts`, which consumes only `findInstanceCycles`
 * (the detector) and `AcyclicityViolation` to assert the invariant
 * at the compile entry.
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
  RegIdx, InstanceIdx, OutputIdx,
} from '../nodes.js'
import { regIdx } from '../nodes.js'
import { withDeclTables, getInstanceType } from '../decl_tables.js'

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

  // Allocate the synthetic reg for a given (breakTargetIdx, outputIdx)
  // on first use; subsequent references resolve to the same reg. Returns
  // the new RegIdx (position in the EVENTUAL post-break body, which is
  // existing regs followed by syntheticRegs).
  const existingRegCount = prog.regs.length
  const breakerIdxFor = (instI: InstanceIdx, outputI: OutputIdx): RegIdx => {
    const inst = prog.instances[instI]
    const outputDecl = getInstanceType(prog, inst).ports.outputs[outputI]
    const key = `${inst.name}::${outputDecl.name}`
    const cached = breakerReg.get(key)
    if (cached) {
      const cachedIdx = syntheticRegs.indexOf(cached)
      return regIdx(existingRegCount + cachedIdx)
    }
    const newRegIdx = regIdx(existingRegCount + syntheticRegs.length)
    const d: RegDecl = {
      op: 'regDecl',
      name: `_feedback_${inst.name}_${outputDecl.name}`,
      // The synthetic reg's update reads the current sample of the
      // broken output; the reg holds that value to make it readable
      // one sample later by the cycle members. The `_feedback_` name
      // prefix distinguishes these from user-written regs.
      update: { op: 'nestedOut', instance: instI, output: outputI },
      init: 0,
    }
    breakerReg.set(key, d)
    syntheticRegs.push(d)
    const ports = breakerPortsByCycle.get(inst)
    if (ports) ports.push(outputDecl)
    return newRegIdx
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
        const inst = prog.instances[node.instance]
        if (inst && breakSet.has(inst)) {
          return { op: 'regRef', idx: breakerIdxFor(node.instance, node.output) }
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
    lowered: withDeclTables({ ...prog, body: newBody }),
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
  // collectNestedOutInstances resolves nestedOut.instance (now InstanceIdx)
  // against the local instances array — these ARE the body's instances
  // in order, matching the InstanceIdx assignment.
  const instanceByIdx = instances
  for (const inst of instances) {
    const set = deps.get(inst)!
    for (const wire of inst.inputs) {
      collectNestedOutInstances(wire.value, set, allInstances, instanceByIdx)
    }
  }
  return deps
}

function collectNestedOutInstances(
  expr: ResolvedExpr,
  out: Set<InstanceDecl>,
  allInstances: Set<InstanceDecl>,
  instanceByIdx: readonly InstanceDecl[],
): void {
  if (typeof expr !== 'object' || expr === null) return
  if (Array.isArray(expr)) {
    for (const e of expr) collectNestedOutInstances(e, out, allInstances, instanceByIdx)
    return
  }
  switch (expr.op) {
    case 'nestedOut': {
      const inst = instanceByIdx[expr.instance]
      if (inst && allInstances.has(inst)) out.add(inst)
      return
    }
    case 'match':
      collectNestedOutInstances(expr.scrutinee, out, allInstances, instanceByIdx)
      for (const arm of expr.arms) collectNestedOutInstances(arm.body, out, allInstances, instanceByIdx)
      return
    case 'fold': case 'scan':
      collectNestedOutInstances(expr.over, out, allInstances, instanceByIdx)
      collectNestedOutInstances(expr.init, out, allInstances, instanceByIdx)
      collectNestedOutInstances(expr.body, out, allInstances, instanceByIdx)
      return
    case 'generate':
      collectNestedOutInstances(expr.count, out, allInstances, instanceByIdx)
      collectNestedOutInstances(expr.body, out, allInstances, instanceByIdx)
      return
    case 'iterate': case 'chain':
      collectNestedOutInstances(expr.count, out, allInstances, instanceByIdx)
      collectNestedOutInstances(expr.init, out, allInstances, instanceByIdx)
      collectNestedOutInstances(expr.body, out, allInstances, instanceByIdx)
      return
    case 'map2':
      collectNestedOutInstances(expr.over, out, allInstances, instanceByIdx)
      collectNestedOutInstances(expr.body, out, allInstances, instanceByIdx)
      return
    case 'zipWith':
      collectNestedOutInstances(expr.a, out, allInstances, instanceByIdx)
      collectNestedOutInstances(expr.b, out, allInstances, instanceByIdx)
      collectNestedOutInstances(expr.body, out, allInstances, instanceByIdx)
      return
    case 'let':
      for (const b of expr.binders) collectNestedOutInstances(b.value, out, allInstances, instanceByIdx)
      collectNestedOutInstances(expr.in, out, allInstances, instanceByIdx)
      return
    case 'tag':
      for (const p of expr.payload) collectNestedOutInstances(p.value, out, allInstances, instanceByIdx)
      return
    case 'zeros':
      collectNestedOutInstances(expr.count, out, allInstances, instanceByIdx)
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
      for (const a of expr.args) collectNestedOutInstances(a, out, allInstances, instanceByIdx)
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
