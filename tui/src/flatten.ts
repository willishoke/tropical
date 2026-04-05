/**
 * flatten.ts — Flatten a patch into a single egress_plan_2 kernel.
 *
 * Takes a SessionState and produces a flat plan JSON where all module
 * expression trees are inlined: input() nodes are substituted with wiring
 * expressions, ref() nodes are resolved by inlining the referenced module's
 * output expression. The result has zero inter-module boundaries — just one
 * flat set of output_exprs and register_exprs.
 */

import type { ExprNode } from './expr.js'
import type { SessionState } from './patch.js'
import type { ModuleInstance } from './module.js'
import {
  type CompilerInput, type ModuleInfo,
  compilerInputFromSession, extractModuleInfo,
  buildDependencyGraph, topologicalSort,
} from './compiler.js'

// ─────────────────────────────────────────────────────────────
// egress_plan_2 schema
// ─────────────────────────────────────────────────────────────

export interface FlatPlan {
  schema: 'egress_plan_2'
  config: { sample_rate: number }
  output_exprs: ExprNode[]
  register_exprs: ExprNode[]
  state_init: (number | boolean | number[])[]
  register_names: string[]
  outputs: number[]
}

// ─────────────────────────────────────────────────────────────
// Expression tree manipulation
// ─────────────────────────────────────────────────────────────

/** Deep-clone an ExprNode tree. */
function cloneExpr(node: ExprNode): ExprNode {
  if (typeof node === 'number' || typeof node === 'boolean') return node
  if (Array.isArray(node)) return node.map(cloneExpr)
  const obj = node as { op: string; [k: string]: unknown }
  const result: Record<string, unknown> = { op: obj.op }
  for (const [k, v] of Object.entries(obj)) {
    if (k === 'op') continue
    if (Array.isArray(v)) {
      result[k] = (v as ExprNode[]).map(cloneExpr)
    } else if (typeof v === 'object' && v !== null && 'op' in v) {
      result[k] = cloneExpr(v as ExprNode)
    } else {
      result[k] = v
    }
  }
  return result as ExprNode
}

/**
 * Substitute all input(id) nodes in an expression tree.
 * inputMap maps input slot ID → replacement expression.
 */
function substituteInputs(node: ExprNode, inputMap: Map<number, ExprNode>): ExprNode {
  if (typeof node === 'number' || typeof node === 'boolean') return node
  if (Array.isArray(node)) return node.map(n => substituteInputs(n, inputMap))

  const obj = node as { op: string; [k: string]: unknown }

  if (obj.op === 'input') {
    const replacement = inputMap.get(obj.id as number)
    if (replacement !== undefined) return cloneExpr(replacement)
    return node
  }

  const result: Record<string, unknown> = { op: obj.op }
  for (const [k, v] of Object.entries(obj)) {
    if (k === 'op') continue
    if (Array.isArray(v)) {
      result[k] = (v as ExprNode[]).map(n => substituteInputs(n, inputMap))
    } else if (typeof v === 'object' && v !== null && 'op' in v) {
      result[k] = substituteInputs(v as ExprNode, inputMap)
    } else {
      result[k] = v
    }
  }
  return result as ExprNode
}

/**
 * Rewrite register(id) nodes with an offset.
 * Register IDs in a module's local expression tree are 0-based;
 * in the flat plan they need to be offset by the module's register base.
 */
function offsetRegisters(node: ExprNode, offset: number): ExprNode {
  if (offset === 0) return node
  if (typeof node === 'number' || typeof node === 'boolean') return node
  if (Array.isArray(node)) return node.map(n => offsetRegisters(n, offset))

  const obj = node as { op: string; [k: string]: unknown }

  if (obj.op === 'reg') {
    return { op: 'reg', id: (obj.id as number) + offset }
  }

  const result: Record<string, unknown> = { op: obj.op }
  for (const [k, v] of Object.entries(obj)) {
    if (k === 'op') continue
    if (Array.isArray(v)) {
      result[k] = (v as ExprNode[]).map(n => offsetRegisters(n, offset))
    } else if (typeof v === 'object' && v !== null && 'op' in v) {
      result[k] = offsetRegisters(v as ExprNode, offset)
    } else {
      result[k] = v
    }
  }
  return result as ExprNode
}

/**
 * Resolve all ref(module, output) nodes by inlining the referenced module's
 * output expression. outputExprs maps "moduleName" → array of output ExprNodes
 * (already fully resolved for modules earlier in topological order).
 */
function resolveRefs(node: ExprNode, outputExprs: Map<string, ExprNode[]>): ExprNode {
  if (typeof node === 'number' || typeof node === 'boolean') return node
  if (Array.isArray(node)) return node.map(n => resolveRefs(n, outputExprs))

  const obj = node as { op: string; [k: string]: unknown }

  if (obj.op === 'ref') {
    const moduleName = obj.module as string
    const outputId = obj.output as number
    const moduleOutputs = outputExprs.get(moduleName)
    if (!moduleOutputs) throw new Error(`flatten: unresolved ref to unknown module '${moduleName}'`)
    if (outputId >= moduleOutputs.length) throw new Error(`flatten: ref output ${outputId} out of range for '${moduleName}'`)
    return cloneExpr(moduleOutputs[outputId])
  }

  const result: Record<string, unknown> = { op: obj.op }
  for (const [k, v] of Object.entries(obj)) {
    if (k === 'op') continue
    if (Array.isArray(v)) {
      result[k] = (v as ExprNode[]).map(n => resolveRefs(n, outputExprs))
    } else if (typeof v === 'object' && v !== null && 'op' in v) {
      result[k] = resolveRefs(v as ExprNode, outputExprs)
    } else {
      result[k] = v
    }
  }
  return result as ExprNode
}

// ─────────────────────────────────────────────────────────────
// Main flatten
// ─────────────────────────────────────────────────────────────

/**
 * Flatten a session's patch into a single egress_plan_2 plan.
 *
 * Process:
 * 1. Topologically sort modules
 * 2. For each module in order:
 *    a. Build input map: input(N) → wiring expression (with refs already resolved)
 *    b. Substitute inputs in the module's output and register expressions
 *    c. Offset register IDs to the module's base in the flat register array
 *    d. Resolve remaining refs (from wiring exprs that reference earlier modules)
 *    e. Record the resolved output expressions for use by later modules' refs
 * 3. Collect all outputs matching graphOutputs
 * 4. Emit egress_plan_2 JSON
 */
export function flattenPatch(session: SessionState): FlatPlan {
  const { instanceRegistry, inputExprNodes, graphOutputs } = session

  // Build module info map
  const moduleInfos = new Map<string, ModuleInfo>()
  const moduleInstances = new Map<string, ModuleInstance>()
  for (const [name, inst] of instanceRegistry) {
    moduleInfos.set(name, extractModuleInfo(name, inst._def))
    moduleInstances.set(name, inst)
  }

  // Topological sort
  const deps = buildDependencyGraph(moduleInfos.keys(), inputExprNodes)
  const { order, complete } = topologicalSort(deps)
  if (!complete) throw new Error('flatten: topological sort incomplete — cycles exist')

  // Flat register arrays
  const flatOutputExprs: ExprNode[] = []
  const flatRegisterExprs: ExprNode[] = []
  const flatStateInit: (number | boolean | number[])[] = []
  const flatRegisterNames: string[] = []

  // Track each module's register base offset and resolved output expressions
  const resolvedOutputs = new Map<string, ExprNode[]>()
  const moduleOutputBase = new Map<string, number>()

  // Process each module in topological order
  let registerBase = 0
  for (const name of order) {
    const inst = moduleInstances.get(name)!
    const def = inst._def

    moduleOutputBase.set(name, flatOutputExprs.length)

    // Build input substitution map: input(i) → wiring expression
    const inputMap = new Map<number, ExprNode>()
    for (let i = 0; i < def.inputNames.length; i++) {
      const key = `${name}:${def.inputNames[i]}`
      const wiringExpr = inputExprNodes.get(key)
      if (wiringExpr !== undefined) {
        // Resolve refs in the wiring expression (they reference earlier modules)
        inputMap.set(i, resolveRefs(wiringExpr, resolvedOutputs))
      } else {
        // Use default value or 0
        const defaultExpr = def.inputDefaults[i]
        inputMap.set(i, defaultExpr?._node ?? 0)
      }
    }

    // Process each output expression
    const moduleResolvedOutputs: ExprNode[] = []
    for (let i = 0; i < def.outputExprNodes.length; i++) {
      let expr = cloneExpr(def.outputExprNodes[i])
      expr = substituteInputs(expr, inputMap)
      expr = offsetRegisters(expr, registerBase)
      expr = resolveRefs(expr, resolvedOutputs)
      flatOutputExprs.push(expr)
      moduleResolvedOutputs.push(expr)
    }
    resolvedOutputs.set(name, moduleResolvedOutputs)

    // Process each register expression
    for (let i = 0; i < def.registerExprNodes.length; i++) {
      const regNode = def.registerExprNodes[i]
      if (regNode !== null) {
        let expr = cloneExpr(regNode)
        expr = substituteInputs(expr, inputMap)
        expr = offsetRegisters(expr, registerBase)
        expr = resolveRefs(expr, resolvedOutputs)
        flatRegisterExprs.push(expr)
      } else {
        // No update — register holds its value (identity: reg(registerBase + i))
        flatRegisterExprs.push({ op: 'reg', id: registerBase + i })
      }

      // Register name and init value
      flatRegisterNames.push(`${name}_${def.registerNames[i]}`)
      const initVal = def.registerInitValues[i]
      if (typeof initVal === 'number' || typeof initVal === 'boolean') {
        flatStateInit.push(initVal)
      } else if (Array.isArray(initVal)) {
        flatStateInit.push(initVal as number[])
      } else {
        flatStateInit.push(0)
      }
    }

    registerBase += def.registerNames.length
  }

  // Build output indices: map graphOutputs → flat output indices
  const outputIndices: number[] = []
  let outputBase = 0
  const moduleOutputStart = new Map<string, number>()

  // Recompute output starts
  for (const name of order) {
    moduleOutputStart.set(name, outputBase)
    const inst = moduleInstances.get(name)!
    outputBase += inst._def.outputNames.length
  }

  for (const { module, output } of graphOutputs) {
    const inst = moduleInstances.get(module)
    if (!inst) continue
    const outputIdx = inst._def.outputNames.indexOf(output)
    if (outputIdx === -1) continue
    const flatIdx = (moduleOutputStart.get(module) ?? 0) + outputIdx
    outputIndices.push(flatIdx)
  }

  return {
    schema: 'egress_plan_2',
    config: { sample_rate: 44100 },
    output_exprs: flatOutputExprs,
    register_exprs: flatRegisterExprs,
    state_init: flatStateInit,
    register_names: flatRegisterNames,
    outputs: outputIndices,
  }
}
