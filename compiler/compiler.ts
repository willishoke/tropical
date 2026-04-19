/**
 * compiler.ts — Graph utilities for the compilation pipeline.
 *
 * Provides dependency graph construction, topological sorting (Kahn's with
 * level grouping), cycle detection (Tarjan's SCC), and port type conversion.
 * Used by flatten.ts to determine execution order.
 */

import type { ExprNode } from './session'
import {
  type PortType,
  Float, Int, Bool, Unit, StructType, ArrayType,
} from './term'

// ─────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────

export class CompilerError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'CompilerError'
  }
}

// ─────────────────────────────────────────────────────────────
// Port type conversion
// ─────────────────────────────────────────────────────────────

/**
 * Convert a string type annotation (from ProgramDef) to a PortType.
 *
 * Supports:
 *   'float', 'int', 'bool', 'unit'       — scalars
 *   'float[4]', 'float[4,4]'             — arrays with static shapes
 *   'array'                                — float[1] (legacy compat)
 *   'matrix'                               — float[1,1] (legacy compat)
 *   anything else                          — struct type
 */
export function portTypeFromString(s: string | undefined): PortType {
  if (s === undefined) return Float

  // Check for array syntax: type[d1,d2,...] e.g. float[8], int[4,4]
  const arrayMatch = s.match(/^(\w+)\[([^\]]+)\]$/)
  if (arrayMatch) {
    const elementType = portTypeFromString(arrayMatch[1])
    const shape = arrayMatch[2].split(',').map(d => {
      const n = parseInt(d.trim(), 10)
      if (isNaN(n) || n <= 0) throw new Error(`Invalid array dimension '${d.trim()}' in type '${s}'`)
      return n
    })
    return ArrayType(elementType, shape)
  }

  switch (s) {
    case 'float': return Float
    case 'int':   return Int
    case 'bool':  return Bool
    case 'unit':  return Unit
    case 'array': return ArrayType(Float, [1])  // legacy: bare 'array' — shape from init value
    case 'matrix': return ArrayType(Float, [1, 1])  // legacy: bare 'matrix'
    default:      return StructType(s)
  }
}

// ─────────────────────────────────────────────────────────────
// Instance info
// ─────────────────────────────────────────────────────────────

/** Structural type information for a program instance — pure data, no C handles. */
export interface InstanceInfo {
  name: string
  typeName: string
  inputNames: string[]
  outputNames: string[]
  registerNames: string[]
  inputTypes: PortType[]
  outputTypes: PortType[]
  registerTypes: PortType[]
}

/** The subset of ProgramDef the compiler reads. */
interface ProgramDefLike {
  typeName: string
  inputNames: string[]
  outputNames: string[]
  registerNames: string[]
  inputPortTypes: (PortType | undefined)[]
  outputPortTypes: (PortType | undefined)[]
  registerPortTypes: (PortType | undefined)[]
}

/** Extract InstanceInfo from a program definition. Undeclared ports default to Float. */
export function extractInstanceInfo(name: string, def: ProgramDefLike): InstanceInfo {
  const fillFloat = (t: PortType | undefined): PortType => t ?? Float
  return {
    name,
    typeName: def.typeName,
    inputNames: [...def.inputNames],
    outputNames: [...def.outputNames],
    registerNames: [...def.registerNames],
    inputTypes: def.inputPortTypes.map(fillFloat),
    outputTypes: def.outputPortTypes.map(fillFloat),
    registerTypes: def.registerPortTypes.map(fillFloat),
  }
}

// ─────────────────────────────────────────────────────────────
// Expression dependency extraction
// ─────────────────────────────────────────────────────────────

/** Extract instance names referenced via 'ref' ops in an ExprNode tree. */
export function exprDependencies(node: ExprNode): Set<string> {
  const deps = new Set<string>()
  walkExprRefs(node, deps)
  return deps
}

function walkExprRefs(node: ExprNode, deps: Set<string>): void {
  if (typeof node === 'number' || typeof node === 'boolean') return
  if (Array.isArray(node)) {
    for (const item of node) walkExprRefs(item, deps)
    return
  }
  const n = node as { op: string; [k: string]: unknown }
  if (n.op === 'ref') {
    deps.add(n.instance as string)
    return
  }
  // Walk standard args
  for (const arg of ((n.args as ExprNode[]) ?? [])) walkExprRefs(arg, deps)
  // Walk special nested expression forms
  if (n.op === 'construct_struct')
    for (const f of ((n.fields as ExprNode[]) ?? [])) walkExprRefs(f, deps)
  if (n.op === 'field_access')
    walkExprRefs(n.struct_expr as ExprNode, deps)
  if (n.op === 'construct_variant')
    for (const p of ((n.payload as ExprNode[]) ?? [])) walkExprRefs(p, deps)
  if (n.op === 'match_variant') {
    walkExprRefs(n.scrutinee as ExprNode, deps)
    for (const b of ((n.branches as ExprNode[]) ?? [])) walkExprRefs(b, deps)
  }
}

// ─────────────────────────────────────────────────────────────
// Reverse reachability — find all instances needed for a set of outputs
// ─────────────────────────────────────────────────────────────

/**
 * Walk backward from a set of root expressions to find all instances they
 * transitively depend on. Used by export_program to scope which instances
 * are included in an exported composite.
 */
export function reachableInstances(
  rootExprs: ExprNode[],
  inputExprNodes: Map<string, ExprNode>,
  allInstances: Set<string>,
): Set<string> {
  const reachable = new Set<string>()
  const queue: string[] = []

  // Seed from roots
  for (const expr of rootExprs) {
    for (const dep of exprDependencies(expr)) {
      if (allInstances.has(dep) && !reachable.has(dep)) {
        reachable.add(dep)
        queue.push(dep)
      }
    }
  }

  // BFS through wiring
  while (queue.length > 0) {
    const name = queue.pop()!
    for (const [key, expr] of inputExprNodes) {
      if (!key.startsWith(`${name}:`)) continue
      for (const dep of exprDependencies(expr)) {
        if (allInstances.has(dep) && !reachable.has(dep)) {
          reachable.add(dep)
          queue.push(dep)
        }
      }
    }
  }

  return reachable
}

// ─────────────────────────────────────────────────────────────
// Dependency graph
// ─────────────────────────────────────────────────────────────

/**
 * Build a dependency graph: instance → set of instances it depends on.
 * Dependencies come from 'ref' ops in inputExprNodes.
 */
export function buildDependencyGraph(
  instanceNames: Iterable<string>,
  inputExprNodes: Map<string, ExprNode>,
  cycleBreakers?: Set<string>,
): Map<string, Set<string>> {
  const nameSet = new Set(instanceNames)
  const graph = new Map<string, Set<string>>()
  for (const name of nameSet) graph.set(name, new Set())

  for (const [key, expr] of inputExprNodes) {
    const instanceName = key.split(':')[0]
    if (!nameSet.has(instanceName)) continue
    const refs = exprDependencies(expr)
    const deps = graph.get(instanceName)!
    for (const ref of refs) {
      if (nameSet.has(ref) && ref !== instanceName) {
        // Skip edges to cycle-breaking modules — their outputs are
        // previous-sample register reads, not combinational dependencies.
        if (cycleBreakers?.has(ref)) continue
        deps.add(ref)
      }
    }
  }

  return graph
}

// ─────────────────────────────────────────────────────────────
// Topological sort (Kahn's algorithm with level grouping)
// ─────────────────────────────────────────────────────────────

export interface TopologicalResult {
  /** Modules in topological execution order. */
  order: string[]
  /** Modules grouped by parallel execution level. */
  levels: string[][]
  /** True if all modules were sorted (false means cycles exist). */
  complete: boolean
}

export function topologicalSort(
  deps: Map<string, Set<string>>,
): TopologicalResult {
  const inDegree = new Map<string, number>()
  const consumers = new Map<string, Set<string>>()

  for (const name of deps.keys()) {
    inDegree.set(name, 0)
    consumers.set(name, new Set())
  }
  for (const [consumer, producers] of deps) {
    inDegree.set(consumer, producers.size)
    for (const producer of producers) {
      consumers.get(producer)?.add(consumer)
    }
  }

  const order: string[] = []
  const levels: string[][] = []
  let queue = [...inDegree.entries()]
    .filter(([_, d]) => d === 0)
    .map(([n]) => n)
    .sort()

  while (queue.length > 0) {
    levels.push([...queue])
    order.push(...queue)
    const next: string[] = []
    for (const node of queue) {
      for (const c of consumers.get(node) ?? []) {
        const d = inDegree.get(c)! - 1
        inDegree.set(c, d)
        if (d === 0) next.push(c)
      }
    }
    queue = next.sort()
  }

  return { order, levels, complete: order.length === deps.size }
}

// ─────────────────────────────────────────────────────────────
// Tarjan's SCC — cycle detection
// ─────────────────────────────────────────────────────────────

/** Find strongly connected components. Cycles are SCCs with more than one member. */
export function tarjanSCC(deps: Map<string, Set<string>>): string[][] {
  let idx = 0
  const indices = new Map<string, number>()
  const lowlinks = new Map<string, number>()
  const onStack = new Set<string>()
  const stack: string[] = []
  const sccs: string[][] = []

  function visit(v: string): void {
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
      const scc: string[] = []
      let w: string
      do {
        w = stack.pop()!
        onStack.delete(w)
        scc.push(w)
      } while (w !== v)
      sccs.push(scc)
    }
  }

  for (const v of deps.keys()) {
    if (!indices.has(v)) visit(v)
  }

  return sccs
}
