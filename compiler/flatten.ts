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
import type { ModuleInstance, NestedCallDef } from './module.js'
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
  if (typeof node !== 'object' || node === null) return node
  const obj = node as Record<string, unknown>
  const result: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(obj)) {
    if (Array.isArray(v)) {
      result[k] = (v as ExprNode[]).map(cloneExpr)
    } else if (typeof v === 'object' && v !== null && !('_ptr' in (v as Record<string, unknown>))) {
      result[k] = cloneExpr(v as ExprNode)
    } else {
      result[k] = v
    }
  }
  return result as ExprNode
}

/**
 * Inline call(function(body), args) patterns.
 * A call whose callee is a function literal is expanded by substituting
 * input(N) nodes in the function body with the corresponding argument.
 * This eliminates function/call nodes that the C++ plan_loader doesn't support.
 */
function inlineCalls(node: ExprNode): ExprNode {
  if (typeof node === 'number' || typeof node === 'boolean') return node
  if (Array.isArray(node)) return node.map(inlineCalls)
  if (typeof node !== 'object' || node === null) return node

  const obj = node as Record<string, unknown>

  // First, recursively inline in children
  const result: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(obj)) {
    if (Array.isArray(v)) {
      result[k] = (v as ExprNode[]).map(inlineCalls)
    } else if (typeof v === 'object' && v !== null && 'op' in v) {
      result[k] = inlineCalls(v as ExprNode)
    } else {
      result[k] = v
    }
  }

  // Now check if this is a call(function(...), args) that can be inlined
  if (result.op === 'call') {
    const callee = result.callee as ExprNode
    if (typeof callee === 'object' && !Array.isArray(callee) && (callee as Record<string, unknown>).op === 'function') {
      const fnNode = callee as Record<string, unknown>
      const body = fnNode.body as ExprNode
      const args = result.args as ExprNode[]
      // Substitute input(N) → args[N] in the function body
      const argMap = new Map<number, ExprNode>()
      for (let i = 0; i < args.length; i++) {
        argMap.set(i, args[i])
      }
      return substituteInputs(body, argMap)
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
 * Rewrite delay_value(nodeId) nodes as register reads.
 * Each delay state becomes a flat register at (delayBase + nodeId).
 */
function resolveDelayValues(node: ExprNode, delayBase: number): ExprNode {
  if (typeof node === 'number' || typeof node === 'boolean') return node
  if (Array.isArray(node)) return node.map(n => resolveDelayValues(n, delayBase))
  if (typeof node !== 'object' || node === null) return node

  const obj = node as { op: string; [k: string]: unknown }

  if (obj.op === 'delay_value') {
    return { op: 'reg', id: delayBase + (obj.node_id as number) }
  }

  const result: Record<string, unknown> = { op: obj.op }
  for (const [k, v] of Object.entries(obj)) {
    if (k === 'op') continue
    if (Array.isArray(v)) {
      result[k] = (v as ExprNode[]).map(n => resolveDelayValues(n, delayBase))
    } else if (typeof v === 'object' && v !== null && 'op' in v) {
      result[k] = resolveDelayValues(v as ExprNode, delayBase)
    } else {
      result[k] = v
    }
  }
  return result as ExprNode
}

/**
 * Compute the total register count for a nested call (including its own nested calls, recursively).
 * Returns: named registers + delay states + sum of all nested calls' totals.
 */
function nestedCallRegCount(nc: NestedCallDef): number {
  const d = nc.moduleDef
  let total = d.registerNames.length + d.delayUpdateNodes.length
  for (const sub of d.nestedCalls) {
    total += nestedCallRegCount(sub)
  }
  return total
}

/**
 * Resolve all nested_output(nodeId, outputId) nodes in an expression tree.
 *
 * For each nested call, the nested module's output expression is inlined with:
 *   - input(i) → call argument i (in the parent module's local expression space)
 *   - reg(j) → reg(nestedRegBase + j)   (offset into parent's local register space)
 *   - delay_value(k) → reg(nestedDelayBase + k)
 *
 * nestedRegStart is the local register offset where the first nested call's registers begin
 * (typically: parent's named registers + parent's delay states).
 */
function resolveNestedOutputs(
  node: ExprNode,
  nestedCalls: NestedCallDef[],
  nestedRegStart: number,
): ExprNode {
  if (nestedCalls.length === 0) return node
  // Pre-compute each nested call's local register base and resolved output expressions
  const resolvedNestedOutputs = new Map<number, ExprNode[]>()
  let regCursor = nestedRegStart

  for (let ncIdx = 0; ncIdx < nestedCalls.length; ncIdx++) {
    const nc = nestedCalls[ncIdx]
    const nd = nc.moduleDef
    const ncRegBase = regCursor
    const ncDelayBase = ncRegBase + nd.registerNames.length
    // Base for this nested call's own nested calls
    const ncNestedStart = ncDelayBase + nd.delayUpdateNodes.length

    const resolved: ExprNode[] = []
    for (let outId = 0; outId < nd.outputExprNodes.length; outId++) {
      let expr = cloneExpr(nd.outputExprNodes[outId])
      // First resolve any nested calls within the nested module itself
      expr = resolveNestedOutputs(expr, nd.nestedCalls, ncNestedStart)
      expr = inlineCalls(expr)
      expr = offsetRegisters(expr, ncRegBase)
      expr = resolveDelayValues(expr, ncDelayBase)
      // Substitute nested module's input(i) → call argument i
      // Call args may reference earlier nested_output nodes — resolve those first
      const argMap = new Map<number, ExprNode>()
      for (let i = 0; i < nc.callArgNodes.length; i++) {
        let arg = cloneExpr(nc.callArgNodes[i])
        // Resolve any nested_output refs in the call args (e.g., chained allpass stages)
        arg = substituteNestedOutputRefs(arg, resolvedNestedOutputs)
        argMap.set(i, arg)
      }
      expr = substituteInputs(expr, argMap)
      resolved.push(expr)
    }
    resolvedNestedOutputs.set(ncIdx, resolved)

    regCursor += nestedCallRegCount(nc)
  }

  return substituteNestedOutputRefs(node, resolvedNestedOutputs)
}

/** Replace nested_output(nodeId, outputId) nodes with pre-resolved expressions. */
function substituteNestedOutputRefs(
  node: ExprNode,
  resolved: Map<number, ExprNode[]>,
): ExprNode {
  if (typeof node === 'number' || typeof node === 'boolean') return node
  if (Array.isArray(node)) return node.map(n => substituteNestedOutputRefs(n, resolved))
  if (typeof node !== 'object' || node === null) return node

  const obj = node as { op: string; [k: string]: unknown }

  if (obj.op === 'nested_output') {
    const nodeId = obj.node_id as number
    const outputId = obj.output_id as number
    const outputs = resolved.get(nodeId)
    if (!outputs) throw new Error(`flatten: unresolved nested_output node_id=${nodeId}`)
    if (outputId >= outputs.length) throw new Error(`flatten: nested_output output_id=${outputId} out of range`)
    return cloneExpr(outputs[outputId])
  }

  const result: Record<string, unknown> = { op: obj.op }
  for (const [k, v] of Object.entries(obj)) {
    if (k === 'op') continue
    if (Array.isArray(v)) {
      result[k] = (v as ExprNode[]).map(n => substituteNestedOutputRefs(n, resolved))
    } else if (typeof v === 'object' && v !== null && 'op' in v) {
      result[k] = substituteNestedOutputRefs(v as ExprNode, resolved)
    } else {
      result[k] = v
    }
  }
  return result as ExprNode
}

/**
 * Collect register update expressions for all nested calls within a module.
 * Returns { exprs, names, inits } arrays to append to the flat register lists.
 * The expressions are in the parent module's local space (0-based parent regs)
 * and still need the parent's processExpr pipeline applied.
 *
 * nestedRegStart: local offset where nested registers begin.
 */
function collectNestedRegisterExprs(
  nestedCalls: NestedCallDef[],
  nestedRegStart: number,
  parentName: string,
  resolvedNestedOutputs: Map<number, ExprNode[]>,
): {
  exprs: ExprNode[]
  names: string[]
  inits: (number | boolean | number[])[]
} {
  const exprs: ExprNode[] = []
  const names: string[] = []
  const inits: (number | boolean | number[])[] = []
  let regCursor = nestedRegStart

  for (let ncIdx = 0; ncIdx < nestedCalls.length; ncIdx++) {
    const nc = nestedCalls[ncIdx]
    const nd = nc.moduleDef
    const ncRegBase = regCursor
    const ncDelayBase = ncRegBase + nd.registerNames.length
    const ncNestedStart = ncDelayBase + nd.delayUpdateNodes.length

    // Build input substitution map for this nested call
    const argMap = new Map<number, ExprNode>()
    for (let i = 0; i < nc.callArgNodes.length; i++) {
      let arg = cloneExpr(nc.callArgNodes[i])
      arg = substituteNestedOutputRefs(arg, resolvedNestedOutputs)
      argMap.set(i, arg)
    }

    // Process helper for nested module's expressions
    const processNestedExpr = (rawNode: ExprNode): ExprNode => {
      let expr = cloneExpr(rawNode)
      expr = resolveNestedOutputs(expr, nd.nestedCalls, ncNestedStart)
      expr = inlineCalls(expr)
      expr = offsetRegisters(expr, ncRegBase)
      expr = resolveDelayValues(expr, ncDelayBase)
      expr = substituteInputs(expr, argMap)
      return expr
    }

    // Named registers
    for (let i = 0; i < nd.registerExprNodes.length; i++) {
      const regNode = nd.registerExprNodes[i]
      if (regNode !== null) {
        exprs.push(processNestedExpr(regNode))
      } else {
        // Identity: hold current value
        exprs.push({ op: 'reg', id: ncRegBase + i })
      }
      names.push(`${parentName}_nested${ncIdx}_${nd.registerNames[i]}`)
      const initVal = nd.registerInitValues[i]
      if (typeof initVal === 'number' || typeof initVal === 'boolean') {
        inits.push(initVal)
      } else if (Array.isArray(initVal)) {
        inits.push(initVal as number[])
      } else {
        inits.push(0)
      }
    }

    // Delay states
    for (let i = 0; i < nd.delayUpdateNodes.length; i++) {
      const updateNode = nd.delayUpdateNodes[i]
      if (updateNode !== null && updateNode !== undefined) {
        exprs.push(processNestedExpr(updateNode as ExprNode))
      } else {
        exprs.push({ op: 'reg', id: ncDelayBase + i })
      }
      names.push(`${parentName}_nested${ncIdx}_delay_${i}`)
      inits.push(nd.delayInitValues[i] ?? 0)
    }

    // Recursively collect from sub-nested calls
    if (nd.nestedCalls.length > 0) {
      const sub = collectNestedRegisterExprs(nd.nestedCalls, ncNestedStart, `${parentName}_nested${ncIdx}`, resolvedNestedOutputs)
      for (const e of sub.exprs) exprs.push(e)
      for (const n of sub.names) names.push(n)
      for (const v of sub.inits) inits.push(v)
    }

    regCursor += nestedCallRegCount(nc)
  }

  return { exprs, names, inits }
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
 * output expression. outputExprs maps "moduleName" → array of output ExprNodes.
 * outputNames maps "moduleName" → array of output name strings (for name→index lookup).
 */
function resolveRefs(
  node: ExprNode,
  outputExprs: Map<string, ExprNode[]>,
  outputNames: Map<string, string[]>,
): ExprNode {
  if (typeof node === 'number' || typeof node === 'boolean') return node
  if (Array.isArray(node)) return node.map(n => resolveRefs(n, outputExprs, outputNames))
  if (typeof node !== 'object' || node === null) return node

  const obj = node as { op: string; [k: string]: unknown }

  if (obj.op === 'ref') {
    const moduleName = obj.module as string
    const moduleOutputs = outputExprs.get(moduleName)
    if (!moduleOutputs) throw new Error(`flatten: unresolved ref to unknown module '${moduleName}'`)

    let outputId: number
    if (typeof obj.output === 'number') {
      outputId = obj.output
    } else {
      // String output name — resolve to index
      const names = outputNames.get(moduleName)
      if (!names) throw new Error(`flatten: no output names for module '${moduleName}'`)
      outputId = names.indexOf(obj.output as string)
      if (outputId === -1) throw new Error(`flatten: unknown output '${obj.output}' on module '${moduleName}'`)
    }

    if (outputId >= moduleOutputs.length) throw new Error(`flatten: ref output ${outputId} out of range for '${moduleName}'`)
    return cloneExpr(moduleOutputs[outputId])
  }

  const result: Record<string, unknown> = { op: obj.op }
  for (const [k, v] of Object.entries(obj)) {
    if (k === 'op') continue
    if (Array.isArray(v)) {
      result[k] = (v as ExprNode[]).map(n => resolveRefs(n, outputExprs, outputNames))
    } else if (typeof v === 'object' && v !== null && 'op' in v) {
      result[k] = resolveRefs(v as ExprNode, outputExprs, outputNames)
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
  const resolvedOutputNames = new Map<string, string[]>()
  const moduleOutputBase = new Map<string, number>()

  // Process each module in topological order
  let registerBase = 0
  for (const name of order) {
    const inst = moduleInstances.get(name)!
    const def = inst._def

    moduleOutputBase.set(name, flatOutputExprs.length)

    // Delay states occupy registers immediately after the module's named registers
    const delayBase = registerBase + def.registerNames.length
    // Nested call registers start after named registers + delay states
    const nestedRegStart = def.registerNames.length + def.delayUpdateNodes.length

    // Build input substitution map: input(i) → wiring expression
    const inputMap = new Map<number, ExprNode>()
    for (let i = 0; i < def.inputNames.length; i++) {
      const key = `${name}:${def.inputNames[i]}`
      const wiringExpr = inputExprNodes.get(key)
      if (wiringExpr !== undefined) {
        // Resolve refs in the wiring expression (they reference earlier modules)
        inputMap.set(i, resolveRefs(wiringExpr, resolvedOutputs, resolvedOutputNames))
      } else {
        // Use default value or 0
        const defaultExpr = def.inputDefaults[i]
        const node = defaultExpr?._node ?? 0
        inputMap.set(i, inlineCalls(typeof node === 'object' ? cloneExpr(node) : node))
      }
    }

    // Helper: full expression pipeline for a local module expression
    const processExpr = (rawNode: ExprNode): ExprNode => {
      let expr = cloneExpr(rawNode)
      // Resolve nested module calls (ModuleType.call()) before other transforms
      expr = resolveNestedOutputs(expr, def.nestedCalls, nestedRegStart)
      expr = inlineCalls(expr)
      expr = offsetRegisters(expr, registerBase)
      expr = resolveDelayValues(expr, delayBase)
      expr = substituteInputs(expr, inputMap)
      expr = resolveRefs(expr, resolvedOutputs, resolvedOutputNames)
      return expr
    }

    // Process each output expression
    // Order matters: offset registers BEFORE substituting inputs/resolving refs,
    // because wiring expressions already have globally-correct register IDs.
    const moduleResolvedOutputs: ExprNode[] = []
    for (let i = 0; i < def.outputExprNodes.length; i++) {
      const expr = processExpr(def.outputExprNodes[i])
      flatOutputExprs.push(expr)
      moduleResolvedOutputs.push(expr)
    }
    resolvedOutputs.set(name, moduleResolvedOutputs)
    resolvedOutputNames.set(name, [...def.outputNames])

    // Process each named register expression
    for (let i = 0; i < def.registerExprNodes.length; i++) {
      const regNode = def.registerExprNodes[i]
      if (regNode !== null) {
        flatRegisterExprs.push(processExpr(regNode))
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

    // Process delay state registers (delay_value(N) reads from delayBase + N)
    for (let i = 0; i < def.delayUpdateNodes.length; i++) {
      const updateNode = def.delayUpdateNodes[i]
      if (updateNode !== null && updateNode !== undefined) {
        flatRegisterExprs.push(processExpr(updateNode as ExprNode))
      } else {
        // No update — hold current value
        flatRegisterExprs.push({ op: 'reg', id: delayBase + i })
      }
      flatRegisterNames.push(`${name}_delay_${i}`)
      flatStateInit.push(def.delayInitValues[i] ?? 0)
    }

    // Process nested module call registers (stateful ModuleType.call() inside defineModule)
    if (def.nestedCalls.length > 0) {
      // Build resolved nested outputs map for call-arg resolution
      const resolvedNestedOutputs = new Map<number, ExprNode[]>()
      let ncRegCursor = nestedRegStart
      for (let ncIdx = 0; ncIdx < def.nestedCalls.length; ncIdx++) {
        const nc = def.nestedCalls[ncIdx]
        const nd = nc.moduleDef
        const ncRegBase = ncRegCursor
        const ncDelayBase = ncRegBase + nd.registerNames.length
        const ncNestedStart = ncDelayBase + nd.delayUpdateNodes.length

        const resolved: ExprNode[] = []
        for (let outId = 0; outId < nd.outputExprNodes.length; outId++) {
          let expr = cloneExpr(nd.outputExprNodes[outId])
          expr = resolveNestedOutputs(expr, nd.nestedCalls, ncNestedStart)
          expr = inlineCalls(expr)
          expr = offsetRegisters(expr, ncRegBase)
          expr = resolveDelayValues(expr, ncDelayBase)
          const argMap = new Map<number, ExprNode>()
          for (let i = 0; i < nc.callArgNodes.length; i++) {
            let arg = cloneExpr(nc.callArgNodes[i])
            arg = substituteNestedOutputRefs(arg, resolvedNestedOutputs)
            argMap.set(i, arg)
          }
          expr = substituteInputs(expr, argMap)
          resolved.push(expr)
        }
        resolvedNestedOutputs.set(ncIdx, resolved)
        ncRegCursor += nestedCallRegCount(nc)
      }

      const nested = collectNestedRegisterExprs(def.nestedCalls, nestedRegStart, name, resolvedNestedOutputs)
      for (let i = 0; i < nested.exprs.length; i++) {
        // Apply the parent module's global transforms (offsetRegisters, substituteInputs, resolveRefs)
        let expr = nested.exprs[i]
        expr = offsetRegisters(expr, registerBase)
        expr = substituteInputs(expr, inputMap)
        expr = resolveRefs(expr, resolvedOutputs, resolvedOutputNames)
        flatRegisterExprs.push(expr)
        flatRegisterNames.push(nested.names[i])
        flatStateInit.push(nested.inits[i])
      }
    }

    // Total registers for this module: named + delays + all nested calls
    let totalNestedRegs = 0
    for (const nc of def.nestedCalls) totalNestedRegs += nestedCallRegCount(nc)
    registerBase += def.registerNames.length + def.delayUpdateNodes.length + totalNestedRegs
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
