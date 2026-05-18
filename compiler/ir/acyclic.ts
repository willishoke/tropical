/**
 * compiler/ir/acyclic.ts — acyclicity check at the strata-pipeline boundary.
 *
 * Detects non-trivial strongly connected components in a `ResolvedProgram`'s
 * inter-instance dependency graph and exposes:
 *
 *   - `findInstanceCycles(prog)`: pure SCC detection over the program's
 *     inter-instance graph, returning the non-trivial SCCs (cycles).
 *     Used both by `assertAcyclic` and (in future phases) by the
 *     elaborator's strict-cycle-policy check.
 *
 *   - `assertAcyclic(prog)`: throws `AcyclicityViolation` if any
 *     non-trivial SCC exists. Currently called after `traceCycles`
 *     inside `strataPipeline`, where it is a tautology — the trace
 *     pass has already broken every cycle. After Phase 3 (when the
 *     trace pass moves out of the compiler) this assertion becomes
 *     load-bearing: any cycle reaching `strataPipeline` is a caller
 *     bug.
 *
 * The SCC algorithm here is intentionally a duplicate of the one in
 * `trace_cycles.ts`. Phase 1 will extract a single shared helper that
 * both `traceCycles` and `assertAcyclic` consume. This file deliberately
 * stays small and self-contained so the assertion can be added to
 * `strataPipeline` without coupling to the trace pass's internals.
 */

import type {
  ResolvedProgram, ResolvedExpr, ResolvedExprOp,
  BodyDecl, InstanceDecl,
} from './nodes.js'

// ─────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────

/** Thrown when a `ResolvedProgram` reaches `strataPipeline` carrying
 *  a non-trivial cycle in its inter-instance graph. Carries the
 *  detected SCCs as structured data so callers (tests, error
 *  formatters) can render the violation precisely. */
export class AcyclicityViolation extends Error {
  readonly sccs: ReadonlyArray<ReadonlyArray<InstanceDecl>>
  constructor(sccs: ReadonlyArray<ReadonlyArray<InstanceDecl>>) {
    const names = sccs.map(scc => scc.map(i => i.name).join(' → ')).join('; ')
    super(`strataPipeline: input contains an unbroken inter-instance cycle: ${names}`)
    this.name = 'AcyclicityViolation'
    this.sccs = sccs
  }
}

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

/** Throws `AcyclicityViolation` if `prog` carries any non-trivial
 *  cycle in its inter-instance graph. */
export function assertAcyclic(prog: ResolvedProgram): void {
  const cycles = findInstanceCycles(prog)
  if (cycles.length > 0) throw new AcyclicityViolation(cycles)
}

// ─────────────────────────────────────────────────────────────
// Internals — SCC detection
// ─────────────────────────────────────────────────────────────

function collectInstances(decls: BodyDecl[]): InstanceDecl[] {
  const out: InstanceDecl[] = []
  for (const d of decls) if (d.op === 'instanceDecl') out.push(d)
  return out
}

/** Inter-instance dependency map: A → set of instances whose outputs
 *  appear in A's input wires (via `NestedOut` refs). Self-edges
 *  recorded too. */
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
  walkOpChildren(expr, e => collectNestedOutInstances(e, out, allInstances))
  if (expr.op === 'nestedOut' && allInstances.has(expr.instance)) {
    out.add(expr.instance)
  }
}

function walkOpChildren(
  node: ResolvedExprOp,
  visit: (e: ResolvedExpr) => void,
): void {
  switch (node.op) {
    case 'nestedOut':
    case 'inputRef': case 'regRef': case 'paramRef':
    case 'typeParamRef': case 'bindingRef':
    case 'sampleRate': case 'sampleIndex':
      return
    case 'match':
      visit(node.scrutinee)
      for (const arm of node.arms) visit(arm.body)
      return
    case 'fold': case 'scan':
      visit(node.over); visit(node.init); visit(node.body); return
    case 'generate':
      visit(node.count); visit(node.body); return
    case 'iterate': case 'chain':
      visit(node.count); visit(node.init); visit(node.body); return
    case 'map2':
      visit(node.over); visit(node.body); return
    case 'zipWith':
      visit(node.a); visit(node.b); visit(node.body); return
    case 'let':
      for (const b of node.binders) visit(b.value)
      visit(node.in); return
    case 'tag':
      for (const p of node.payload) visit(p.value); return
    case 'zeros':
      visit(node.count); return
    case 'add': case 'sub': case 'mul': case 'div': case 'mod':
    case 'lt': case 'lte': case 'gt': case 'gte': case 'eq': case 'neq':
    case 'and': case 'or':
    case 'bitAnd': case 'bitOr': case 'bitXor': case 'lshift': case 'rshift':
    case 'floorDiv': case 'ldexp':
    case 'neg': case 'not': case 'bitNot':
    case 'sqrt': case 'abs': case 'floor': case 'ceil': case 'round':
    case 'floatExponent': case 'toInt': case 'toBool': case 'toFloat':
    case 'clamp': case 'select': case 'index': case 'arraySet':
      for (const a of node.args) visit(a); return
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
