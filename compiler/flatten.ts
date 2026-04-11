/**
 * flatten.ts — Flatten a session into a single tropical_plan_4 kernel.
 *
 * Takes a SessionState and produces a flat plan JSON where all instance
 * expression trees are inlined: input() nodes are substituted with wiring
 * expressions, ref() nodes are resolved by inlining the referenced instance's
 * output expression. The result has zero inter-instance boundaries — just one
 * flat instruction stream (tropical_plan_4).
 */

import type { ExprNode } from './expr.js'
import type { SessionState } from './session.js'
import type { ProgramInstance, NestedCall } from './program_types.js'
import {
  type CompilerInput, type InstanceInfo,
  compilerInputFromSession, extractInstanceInfo,
  buildDependencyGraph, topologicalSort,
} from './compiler.js'
import { lowerArrayOps } from './lower_arrays.js'
import { portTypeToString } from './term.js'
import { checkArrayConnection } from './array_wiring.js'
import { emitNumericProgram, type NInstr, type ScalarType } from './emit_numeric.js'

// ─────────────────────────────────────────────────────────────
// Wiring type validation
// ─────────────────────────────────────────────────────────────

export class FlattenError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'FlattenError'
  }
}

/**
 * Infer the output type string of an ExprNode, given the instance type context.
 * Returns undefined when the type cannot be statically determined.
 */
function inferExprOutputType(
  expr: ExprNode,
  instanceInfos: Map<string, InstanceInfo>,
): string | undefined {
  if (typeof expr === 'number') return 'float'
  if (typeof expr === 'boolean') return 'bool'
  if (Array.isArray(expr)) return `float[${(expr as unknown[]).length}]`
  if (typeof expr !== 'object' || expr === null) return undefined
  const obj = expr as Record<string, unknown>
  switch (obj.op as string) {
    case 'ref': {
      const modInfo = instanceInfos.get(obj.instance as string)
      if (!modInfo) return undefined
      const outName = obj.output as string | number
      const outIdx = typeof outName === 'number' ? outName : modInfo.outputNames.indexOf(outName)
      if (outIdx === -1 || outIdx >= modInfo.outputTypes.length) return undefined
      return portTypeToString(modInfo.outputTypes[outIdx])
    }
    case 'broadcast_to':
      return `float[${(obj.shape as number[]).join(',')}]`
    case 'zeros':
    case 'ones':
    case 'fill':
    case 'array_literal':
      return `float[${(obj.shape as number[]).join(',')}]`
    default:
      return undefined
  }
}

/**
 * Validate all wiring expressions against their destination port types.
 * Throws FlattenError on incompatible connections (e.g. array → scalar).
 * Inserts broadcast_to wrappers for compatible shape mismatches (e.g. scalar → array[4]).
 * Returns a new Map with any necessary broadcast wrappers applied.
 */
export function normalizeWiringTypes(
  instanceInfos: Map<string, InstanceInfo>,
  inputExprNodes: Map<string, ExprNode>,
): Map<string, ExprNode> {
  const result = new Map(inputExprNodes)

  for (const [key, expr] of inputExprNodes) {
    const colonIdx = key.indexOf(':')
    const instanceName = key.slice(0, colonIdx)
    const inputName = key.slice(colonIdx + 1)

    const modInfo = instanceInfos.get(instanceName)
    if (!modInfo) continue

    const inputIdx = modInfo.inputNames.indexOf(inputName)
    if (inputIdx === -1) continue

    const dstTypeStr = portTypeToString(modInfo.inputTypes[inputIdx])
    const srcTypeStr = inferExprOutputType(expr, instanceInfos)
    if (srcTypeStr === undefined) continue

    const check = checkArrayConnection(srcTypeStr, dstTypeStr, expr)
    if (!check.compatible) {
      throw new FlattenError(
        `Wiring type mismatch on '${instanceName}'.${inputName}: ${check.error}`
      )
    }
    if (check.broadcastExpr) {
      result.set(key, check.broadcastExpr)
    }
  }

  return result
}

// ─────────────────────────────────────────────────────────────
// tropical_plan_4 schema
// ─────────────────────────────────────────────────────────────

export interface FlatPlan {
  schema: 'tropical_plan_4'
  config: { sample_rate: number }
  state_init: (number | boolean)[]
  register_names: string[]
  register_types: ScalarType[]
  array_slot_names: string[]
  outputs: number[]
  // Compiled instruction stream (from emitNumericProgram)
  instructions:    NInstr[]
  register_count:  number
  array_slot_count: number
  array_slot_sizes: number[]
  output_targets:  number[]
  register_targets: number[]
}

// ─────────────────────────────────────────────────────────────
// Expression tree manipulation
// ─────────────────────────────────────────────────────────────

/** Deep-clone an ExprNode tree. */
function cloneExpr(node: ExprNode, memo?: WeakMap<object, ExprNode>): ExprNode {
  if (typeof node === 'number' || typeof node === 'boolean') return node
  if (Array.isArray(node)) return node.map(n => cloneExpr(n, memo))
  if (typeof node !== 'object' || node === null) return node
  if (memo) {
    const cached = memo.get(node as object)
    if (cached !== undefined) return cached
  }
  const obj = node as Record<string, unknown>
  const result: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(obj)) {
    if (Array.isArray(v)) {
      result[k] = (v as ExprNode[]).map(n => cloneExpr(n, memo))
    } else if (typeof v === 'object' && v !== null && !('_ptr' in (v as Record<string, unknown>))) {
      result[k] = cloneExpr(v as ExprNode, memo)
    } else {
      result[k] = v
    }
  }
  const cloned = result as ExprNode
  if (memo) memo.set(node as object, cloned)
  return cloned
}

/**
 * Inline call(function(body), args) patterns.
 * A call whose callee is a function literal is expanded by substituting
 * input(N) nodes in the function body with the corresponding argument.
 * This eliminates function/call nodes that the C++ plan_loader doesn't support.
 */
function inlineCalls(node: ExprNode, memo?: WeakMap<object, ExprNode>): ExprNode {
  if (typeof node === 'number' || typeof node === 'boolean') return node
  if (Array.isArray(node)) return node.map(n => inlineCalls(n, memo))
  if (typeof node !== 'object' || node === null) return node

  if (memo) {
    const cached = memo.get(node as object)
    if (cached !== undefined) return cached
  }

  const obj = node as Record<string, unknown>

  // First, recursively inline in children
  let changed = false
  const result: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(obj)) {
    if (Array.isArray(v)) {
      const arr = v as ExprNode[]
      const newArr = arr.map(n => inlineCalls(n, memo))
      if (newArr.some((n, i) => n !== arr[i])) changed = true
      result[k] = newArr
    } else if (typeof v === 'object' && v !== null && 'op' in v) {
      const newV = inlineCalls(v as ExprNode, memo)
      if (newV !== v) changed = true
      result[k] = newV
    } else if (typeof v === 'object' && v !== null && !Array.isArray(v)) {
      // Record<string, ExprNode> fields (e.g. 'bind' in let nodes)
      const rec = v as Record<string, ExprNode>
      const newRec: Record<string, ExprNode> = {}
      let recChanged = false
      for (const [rk, rv] of Object.entries(rec)) {
        const newRv = inlineCalls(rv, memo)
        if (newRv !== rv) recChanged = true
        newRec[rk] = newRv
      }
      if (recChanged) changed = true
      result[k] = recChanged ? newRec : rec
    } else {
      result[k] = v
    }
  }

  let out: ExprNode
  // Now check if this is a call(function(...), args) that can be inlined
  if ((changed ? result.op : obj.op) === 'call') {
    const src = changed ? result : obj
    const callee = src.callee as ExprNode
    if (typeof callee === 'object' && !Array.isArray(callee) && (callee as Record<string, unknown>).op === 'function') {
      const fnNode = callee as Record<string, unknown>
      const body = fnNode.body as ExprNode
      const args = src.args as ExprNode[]
      // Substitute input(N) → args[N] in the function body
      const argMap = new Map<number, ExprNode>()
      for (let i = 0; i < args.length; i++) {
        argMap.set(i, args[i])
      }
      out = substituteInputs(body, argMap)
    } else {
      out = changed ? result as ExprNode : node
    }
  } else {
    out = changed ? result as ExprNode : node
  }

  if (memo) memo.set(node as object, out)
  return out
}

/**
 * Substitute all input(id) nodes in an expression tree.
 * inputMap maps input slot ID → replacement expression.
 *
 * Identity memo (optional): avoids re-processing the same object when the expression
 * is a DAG with shared subexpressions. This preserves structural sharing and prevents
 * exponential blowup when the same argument is referenced many times (e.g. in tanh).
 * Replacements are returned directly (not cloned) so call-site sharing is preserved.
 */
function substituteInputs(node: ExprNode, inputMap: Map<number, ExprNode>, memo?: WeakMap<object, ExprNode>): ExprNode {
  if (typeof node === 'number' || typeof node === 'boolean') return node
  if (Array.isArray(node)) return node.map(n => substituteInputs(n, inputMap, memo))

  if (memo) {
    const cached = memo.get(node as object)
    if (cached !== undefined) return cached
  }

  const obj = node as { op: string; [k: string]: unknown }

  if (obj.op === 'input') {
    const replacement = inputMap.get(obj.id as number)
    // Return directly — all occurrences of input(N) share the same replacement object,
    // keeping the expression a compact DAG rather than duplicating the arg subtree.
    if (replacement !== undefined) return replacement
    return node
  }

  let changed = false
  const result: Record<string, unknown> = { op: obj.op }
  for (const [k, v] of Object.entries(obj)) {
    if (k === 'op') continue
    if (Array.isArray(v)) {
      const arr = v as ExprNode[]
      const newArr = arr.map(n => substituteInputs(n, inputMap, memo))
      if (newArr.some((n, i) => n !== arr[i])) changed = true
      result[k] = newArr
    } else if (typeof v === 'object' && v !== null && 'op' in v) {
      const newV = substituteInputs(v as ExprNode, inputMap, memo)
      if (newV !== v) changed = true
      result[k] = newV
    } else if (typeof v === 'object' && v !== null && !Array.isArray(v)) {
      const rec = v as Record<string, ExprNode>
      const newRec: Record<string, ExprNode> = {}
      let recChanged = false
      for (const [rk, rv] of Object.entries(rec)) {
        const newRv = substituteInputs(rv, inputMap, memo)
        if (newRv !== rv) recChanged = true
        newRec[rk] = newRv
      }
      if (recChanged) changed = true
      result[k] = recChanged ? newRec : rec
    } else {
      result[k] = v
    }
  }
  if (!changed) {
    if (memo) memo.set(node as object, node)
    return node
  }
  const out = result as ExprNode
  if (memo) memo.set(node as object, out)
  return out
}

/**
 * Rewrite delay_value(nodeId) nodes as register reads.
 * Each delay state becomes a flat register at (delayBase + nodeId).
 */
function resolveDelayValues(node: ExprNode, delayBase: number, memo?: WeakMap<object, ExprNode>): ExprNode {
  if (typeof node === 'number' || typeof node === 'boolean') return node
  if (Array.isArray(node)) return node.map(n => resolveDelayValues(n, delayBase, memo))
  if (typeof node !== 'object' || node === null) return node

  if (memo) {
    const cached = memo.get(node as object)
    if (cached !== undefined) return cached
  }

  const obj = node as { op: string; [k: string]: unknown }

  if (obj.op === 'delay_value') {
    const out = { op: 'reg', id: delayBase + (obj.node_id as number) }
    if (memo) memo.set(node as object, out)
    return out
  }

  let changed = false
  const result: Record<string, unknown> = { op: obj.op }
  for (const [k, v] of Object.entries(obj)) {
    if (k === 'op') continue
    if (Array.isArray(v)) {
      const arr = v as ExprNode[]
      const newArr = arr.map(n => resolveDelayValues(n, delayBase, memo))
      if (newArr.some((n, i) => n !== arr[i])) changed = true
      result[k] = newArr
    } else if (typeof v === 'object' && v !== null && 'op' in v) {
      const newV = resolveDelayValues(v as ExprNode, delayBase, memo)
      if (newV !== v) changed = true
      result[k] = newV
    } else if (typeof v === 'object' && v !== null && !Array.isArray(v)) {
      const rec = v as Record<string, ExprNode>
      const newRec: Record<string, ExprNode> = {}
      let recChanged = false
      for (const [rk, rv] of Object.entries(rec)) {
        const newRv = resolveDelayValues(rv, delayBase, memo)
        if (newRv !== rv) recChanged = true
        newRec[rk] = newRv
      }
      if (recChanged) changed = true
      result[k] = recChanged ? newRec : rec
    } else {
      result[k] = v
    }
  }
  if (!changed) {
    if (memo) memo.set(node as object, node)
    return node
  }
  const out = result as ExprNode
  if (memo) memo.set(node as object, out)
  return out
}

/**
 * Compute the total register count for a nested call (including its own nested calls, recursively).
 * Returns: named registers + delay states + sum of all nested calls' totals.
 */
function nestedCallRegCount(nc: NestedCall): number {
  const d = nc.programDef
  let total = d.registerNames.length + d.delayUpdateNodes.length
  for (const sub of d.nestedCalls) {
    total += nestedCallRegCount(sub)
  }
  return total
}

/**
 * Resolve all nested_output(nodeId, outputId) nodes in an expression tree.
 *
 * For each nested call, the nested program's output expression is inlined with:
 *   - input(i) → call argument i (in the parent program's local expression space)
 *   - reg(j) → reg(nestedRegBase + j)   (offset into parent's local register space)
 *   - delay_value(k) → reg(nestedDelayBase + k)
 *
 * nestedRegStart is the local register offset where the first nested call's registers begin
 * (typically: parent's named registers + parent's delay states).
 */
function resolveNestedOutputs(
  node: ExprNode,
  nestedCalls: NestedCall[],
  nestedRegStart: number,
): ExprNode {
  if (nestedCalls.length === 0) return node
  // Pre-compute each nested call's local register base and resolved output expressions
  const resolvedNestedOutputs = new Map<number, ExprNode[]>()
  let regCursor = nestedRegStart

  for (let ncIdx = 0; ncIdx < nestedCalls.length; ncIdx++) {
    const nc = nestedCalls[ncIdx]
    const nd = nc.programDef
    const ncRegBase = regCursor
    const ncDelayBase = ncRegBase + nd.registerNames.length
    // Base for this nested call's own nested calls
    const ncNestedStart = ncDelayBase + nd.delayUpdateNodes.length

    const resolved: ExprNode[] = []
    for (let outId = 0; outId < nd.outputExprNodes.length; outId++) {
      // Clone template preserving structural sharing — shared nodes (e.g. x2 in tanh)
      // clone once and are referenced multiple times rather than copied exponentially.
      const cloneMemo = new WeakMap<object, ExprNode>()
      let expr = cloneExpr(nd.outputExprNodes[outId], cloneMemo)
      // First resolve any nested calls within the nested program itself
      expr = resolveNestedOutputs(expr, nd.nestedCalls, ncNestedStart)
      // Each pass uses its own identity memo to preserve DAG sharing through the pipeline.
      const inlineMemo = new WeakMap<object, ExprNode>()
      expr = inlineCalls(expr, inlineMemo)
      const offsetMemo = new WeakMap<object, ExprNode>()
      expr = offsetRegisters(expr, ncRegBase, offsetMemo)
      const delayMemo = new WeakMap<object, ExprNode>()
      expr = resolveDelayValues(expr, ncDelayBase, delayMemo)
      // Substitute nested program's input(i) → call argument i
      // Call args may reference earlier nested_output nodes — resolve those first
      const argMap = new Map<number, ExprNode>()
      const argRefMemo = new WeakMap<object, ExprNode>()
      for (let i = 0; i < nc.callArgNodes.length; i++) {
        let arg = cloneExpr(nc.callArgNodes[i])
        // Resolve any nested_output refs in the call args (e.g., chained allpass stages)
        arg = substituteNestedOutputRefs(arg, resolvedNestedOutputs, argRefMemo)
        argMap.set(i, arg)
      }
      // Identity memo: if input(N) appears multiple times in the template body (via shared
      // nodes), all occurrences map to the same replacement object — no exponential fanout.
      const substMemo = new WeakMap<object, ExprNode>()
      expr = substituteInputs(expr, argMap, substMemo)
      resolved.push(expr)
    }
    resolvedNestedOutputs.set(ncIdx, resolved)

    regCursor += nestedCallRegCount(nc)
  }

  return substituteNestedOutputRefs(node, resolvedNestedOutputs)
}

/** Replace nested_output(nodeId, outputId) nodes with pre-resolved expressions.
 *
 * Returns the resolved expression directly (not cloned) so multiple references to
 * the same nested_output share a single physical object — keeping the result a compact
 * DAG. The identity memo avoids re-traversing shared non-terminal nodes.
 */
function substituteNestedOutputRefs(
  node: ExprNode,
  resolved: Map<number, ExprNode[]>,
  memo?: WeakMap<object, ExprNode>,
): ExprNode {
  if (typeof node === 'number' || typeof node === 'boolean') return node
  if (Array.isArray(node)) return node.map(n => substituteNestedOutputRefs(n, resolved, memo))
  if (typeof node !== 'object' || node === null) return node

  if (memo) {
    const cached = memo.get(node as object)
    if (cached !== undefined) return cached
  }

  const obj = node as { op: string; [k: string]: unknown }

  if (obj.op === 'nested_output') {
    const nodeId = obj.node_id as number
    const outputId = obj.output_id as number
    const outputs = resolved.get(nodeId)
    if (!outputs) throw new Error(`flatten: unresolved nested_output node_id=${nodeId}`)
    if (outputId >= outputs.length) throw new Error(`flatten: nested_output output_id=${outputId} out of range`)
    // Return directly — no clone. All references to this call-site share one object.
    return outputs[outputId]
  }

  let changed = false
  const result: Record<string, unknown> = { op: obj.op }
  for (const [k, v] of Object.entries(obj)) {
    if (k === 'op') continue
    if (Array.isArray(v)) {
      const arr = v as ExprNode[]
      const newArr = arr.map(n => substituteNestedOutputRefs(n, resolved, memo))
      if (newArr.some((n, i) => n !== arr[i])) changed = true
      result[k] = newArr
    } else if (typeof v === 'object' && v !== null && 'op' in v) {
      const newV = substituteNestedOutputRefs(v as ExprNode, resolved, memo)
      if (newV !== v) changed = true
      result[k] = newV
    } else if (typeof v === 'object' && v !== null && !Array.isArray(v)) {
      const rec = v as Record<string, ExprNode>
      const newRec: Record<string, ExprNode> = {}
      let recChanged = false
      for (const [rk, rv] of Object.entries(rec)) {
        const newRv = substituteNestedOutputRefs(rv, resolved, memo)
        if (newRv !== rv) recChanged = true
        newRec[rk] = newRv
      }
      if (recChanged) changed = true
      result[k] = recChanged ? newRec : rec
    } else {
      result[k] = v
    }
  }
  if (!changed) {
    if (memo) memo.set(node as object, node)
    return node
  }
  const out = result as ExprNode
  if (memo) memo.set(node as object, out)
  return out
}

/**
 * Collect register update expressions for all nested calls within a program instance.
 * Returns { exprs, names, inits } arrays to append to the flat register lists.
 * The expressions are in the parent program's local space (0-based parent regs)
 * and still need the parent's processExpr pipeline applied.
 *
 * nestedRegStart: local offset where nested registers begin.
 */
function collectNestedRegisterExprs(
  nestedCalls: NestedCall[],
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
    const nd = nc.programDef
    const ncRegBase = regCursor
    const ncDelayBase = ncRegBase + nd.registerNames.length
    const ncNestedStart = ncDelayBase + nd.delayUpdateNodes.length

    // Build input substitution map for this nested call
    const argMap = new Map<number, ExprNode>()
    const argRefMemo = new WeakMap<object, ExprNode>()
    for (let i = 0; i < nc.callArgNodes.length; i++) {
      let arg = cloneExpr(nc.callArgNodes[i])
      arg = substituteNestedOutputRefs(arg, resolvedNestedOutputs, argRefMemo)
      argMap.set(i, arg)
    }

    // Process helper for nested program's expressions
    const processNestedExpr = (rawNode: ExprNode): ExprNode => {
      const cloneMemo = new WeakMap<object, ExprNode>()
      let expr = cloneExpr(rawNode, cloneMemo)
      expr = resolveNestedOutputs(expr, nd.nestedCalls, ncNestedStart)
      const inlineMemo = new WeakMap<object, ExprNode>()
      expr = inlineCalls(expr, inlineMemo)
      const offsetMemo = new WeakMap<object, ExprNode>()
      expr = offsetRegisters(expr, ncRegBase, offsetMemo)
      const delayMemo = new WeakMap<object, ExprNode>()
      expr = resolveDelayValues(expr, ncDelayBase, delayMemo)
      const substMemo = new WeakMap<object, ExprNode>()
      expr = substituteInputs(expr, argMap, substMemo)
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
 * Register IDs in a program's local expression tree are 0-based;
 * in the flat plan they need to be offset by the instance's register base.
 */
function offsetRegisters(node: ExprNode, offset: number, memo?: WeakMap<object, ExprNode>): ExprNode {
  if (offset === 0) return node
  if (typeof node === 'number' || typeof node === 'boolean') return node
  if (Array.isArray(node)) return node.map(n => offsetRegisters(n, offset, memo))

  if (memo) {
    const cached = memo.get(node as object)
    if (cached !== undefined) return cached
  }

  const obj = node as { op: string; [k: string]: unknown }

  if (obj.op === 'reg') {
    const out = { op: 'reg', id: (obj.id as number) + offset }
    if (memo) memo.set(node as object, out)
    return out
  }

  let changed = false
  const result: Record<string, unknown> = { op: obj.op }
  for (const [k, v] of Object.entries(obj)) {
    if (k === 'op') continue
    if (Array.isArray(v)) {
      const arr = v as ExprNode[]
      const newArr = arr.map(n => offsetRegisters(n, offset, memo))
      if (newArr.some((n, i) => n !== arr[i])) changed = true
      result[k] = newArr
    } else if (typeof v === 'object' && v !== null && 'op' in v) {
      const newV = offsetRegisters(v as ExprNode, offset, memo)
      if (newV !== v) changed = true
      result[k] = newV
    } else if (typeof v === 'object' && v !== null && !Array.isArray(v)) {
      const rec = v as Record<string, ExprNode>
      const newRec: Record<string, ExprNode> = {}
      let recChanged = false
      for (const [rk, rv] of Object.entries(rec)) {
        const newRv = offsetRegisters(rv, offset, memo)
        if (newRv !== rv) recChanged = true
        newRec[rk] = newRv
      }
      if (recChanged) changed = true
      result[k] = recChanged ? newRec : rec
    } else {
      result[k] = v
    }
  }
  if (!changed) {
    if (memo) memo.set(node as object, node)
    return node
  }
  const out = result as ExprNode
  if (memo) memo.set(node as object, out)
  return out
}

/**
 * Resolve all ref(instance, output) nodes by inlining the referenced instance's
 * output expression. outputExprs maps "instanceName" → array of output ExprNodes.
 * outputNames maps "instanceName" → array of output name strings (for name→index lookup).
 */
function resolveRefs(
  node: ExprNode,
  outputExprs: Map<string, ExprNode[]>,
  outputNames: Map<string, string[]>,
  memo?: WeakMap<object, ExprNode>,
): ExprNode {
  if (typeof node === 'number' || typeof node === 'boolean') return node
  if (Array.isArray(node)) return node.map(n => resolveRefs(n, outputExprs, outputNames, memo))
  if (typeof node !== 'object' || node === null) return node

  if (memo) {
    const cached = memo.get(node as object)
    if (cached !== undefined) return cached
  }

  const obj = node as { op: string; [k: string]: unknown }

  if (obj.op === 'ref') {
    const instanceName = obj.instance as string
    const instanceOutputs = outputExprs.get(instanceName)
    if (!instanceOutputs) throw new Error(`flatten: unresolved ref to unknown instance '${instanceName}'`)

    let outputId: number
    if (typeof obj.output === 'number') {
      outputId = obj.output
    } else {
      // String output name — resolve to index
      const names = outputNames.get(instanceName)
      if (!names) throw new Error(`flatten: no output names for instance '${instanceName}'`)
      outputId = names.indexOf(obj.output as string)
      if (outputId === -1) throw new Error(`flatten: unknown output '${obj.output}' on instance '${instanceName}'`)
    }

    if (outputId >= instanceOutputs.length) throw new Error(`flatten: ref output ${outputId} out of range for '${instanceName}'`)
    // Return directly — the resolved output is already fully processed and immutable.
    // Sharing is safe; the emitter's identity-based CSE compiles each unique node once.
    return instanceOutputs[outputId]
  }

  let changed = false
  const result: Record<string, unknown> = { op: obj.op }
  for (const [k, v] of Object.entries(obj)) {
    if (k === 'op') continue
    if (Array.isArray(v)) {
      const arr = v as ExprNode[]
      const newArr = arr.map(n => resolveRefs(n, outputExprs, outputNames, memo))
      if (newArr.some((n, i) => n !== arr[i])) changed = true
      result[k] = newArr
    } else if (typeof v === 'object' && v !== null && 'op' in v) {
      const newV = resolveRefs(v as ExprNode, outputExprs, outputNames, memo)
      if (newV !== v) changed = true
      result[k] = newV
    } else if (typeof v === 'object' && v !== null && !Array.isArray(v)) {
      const rec = v as Record<string, ExprNode>
      const newRec: Record<string, ExprNode> = {}
      let recChanged = false
      for (const [rk, rv] of Object.entries(rec)) {
        const newRv = resolveRefs(rv, outputExprs, outputNames, memo)
        if (newRv !== rv) recChanged = true
        newRec[rk] = newRv
      }
      if (recChanged) changed = true
      result[k] = recChanged ? newRec : rec
    } else {
      result[k] = v
    }
  }
  if (!changed) {
    if (memo) memo.set(node as object, node)
    return node
  }
  const out = result as ExprNode
  if (memo) memo.set(node as object, out)
  return out
}

// ─────────────────────────────────────────────────────────────
// Main flatten
// ─────────────────────────────────────────────────────────────

/**
 * Flatten a session's patch into a single tropical_plan_4 plan.
 *
 * Process:
 * 1. Topologically sort instances
 * 2. For each instance in order:
 *    a. Build input map: input(N) → wiring expression (with refs already resolved)
 *    b. Substitute inputs in the instance's output and register expressions
 *    c. Offset register IDs to the instance's base in the flat register array
 *    d. Resolve remaining refs (from wiring exprs that reference earlier instances)
 *    e. Record the resolved output expressions for use by later instances' refs
 * 3. Collect all outputs matching graphOutputs
 * 4. Compile expression trees → FlatProgram via emitNumericProgram
 * 5. Emit tropical_plan_4 JSON
 */
export function flattenSession(session: SessionState): FlatPlan {
  const { instanceRegistry, graphOutputs } = session

  // Build instance info map
  const instanceInfos = new Map<string, InstanceInfo>()
  const instances = new Map<string, ProgramInstance>()
  for (const [name, inst] of instanceRegistry) {
    instanceInfos.set(name, extractInstanceInfo(name, inst._def))
    instances.set(name, inst)
  }

  // Validate and normalize wiring types (throws FlattenError on incompatible connections,
  // inserts broadcast_to wrappers for shape mismatches from patch-file or direct wiring)
  const inputExprNodes = normalizeWiringTypes(instanceInfos, session.inputExprNodes)

  // Identify cycle-breaking instances (outputs depend only on registers, not current inputs).
  // These can break feedback cycles: their outputs are previous-sample state reads.
  const cycleBreakers = new Set<string>()
  const cycleBreakerList: string[] = []
  for (const [name, inst] of instances) {
    if (inst._def.breaksCycles) {
      cycleBreakers.add(name)
      cycleBreakerList.push(name)
    }
  }

  // Topological sort — edges to cycle-breaking instances are excluded since
  // their outputs are previous-sample register reads, not combinational deps.
  const deps = buildDependencyGraph(instanceInfos.keys(), inputExprNodes, cycleBreakers)
  const { order, complete } = topologicalSort(deps)
  if (!complete) throw new Error('flatten: topological sort incomplete — cycles exist')

  // Split order: cycle-breaking instances first (output pre-computation),
  // then non-cycle-breaking instances in topological order.
  const nonCbOrder = order.filter(n => !cycleBreakers.has(n))

  // Flat register arrays
  const flatOutputExprs: ExprNode[] = []
  const flatRegisterExprs: ExprNode[] = []
  const flatStateInit: (number | boolean | number[])[] = []
  const flatRegisterNames: string[] = []
  const flatRegisterTypes: ScalarType[] = []

  // Track each instance's register base offset and resolved output expressions
  const resolvedOutputs = new Map<string, ExprNode[]>()
  const resolvedOutputNames = new Map<string, string[]>()
  const exprBase = new Map<string, number>()

  // ── Phase 1: Pre-compute cycle-breaking instance outputs ──
  // Their outputs are purely state-derived (register reads from previous sample),
  // so they can be resolved before any other instances. Register updates are deferred.
  const deferredCbUpdates: Array<{
    name: string
    registerBase: number
    delayBase: number
    nestedRegStart: number
    regStartIdx: number
  }> = []

  let registerBase = 0
  for (const name of cycleBreakerList) {
    const inst = instances.get(name)!
    const def = inst._def
    const myRegBase = registerBase

    exprBase.set(name, flatOutputExprs.length)
    const delayBase = registerBase + def.registerNames.length
    const nestedRegStart = def.registerNames.length + def.delayUpdateNodes.length

    // Process output expressions — no input substitution or ref resolution needed
    // (cycle-breaking instance outputs are purely register-derived)
    const cbOutputExprs: ExprNode[] = []
    for (const outputNode of def.outputExprNodes) {
      const cloneMemo = new WeakMap<object, ExprNode>()
      let expr = cloneExpr(outputNode, cloneMemo)
      expr = resolveNestedOutputs(expr, def.nestedCalls, nestedRegStart)
      const inlineMemo = new WeakMap<object, ExprNode>()
      expr = inlineCalls(expr, inlineMemo)
      const offsetMemo = new WeakMap<object, ExprNode>()
      expr = offsetRegisters(expr, registerBase, offsetMemo)
      const delayMemo = new WeakMap<object, ExprNode>()
      expr = resolveDelayValues(expr, delayBase, delayMemo)
      const lowerMemo = new WeakMap<object, ExprNode>()
      expr = lowerArrayOps(expr, lowerMemo)
      flatOutputExprs.push(expr)
      cbOutputExprs.push(expr)
    }
    resolvedOutputs.set(name, cbOutputExprs)
    resolvedOutputNames.set(name, [...def.outputNames])

    // Push register metadata with placeholder update expressions (overwritten in phase 3)
    const regStartIdx = flatRegisterExprs.length
    for (let i = 0; i < def.registerNames.length; i++) {
      flatRegisterExprs.push({ op: 'reg', id: registerBase + i })
      flatRegisterNames.push(`${name}_${def.registerNames[i]}`)
      const portType = def.registerPortTypes[i]
      flatRegisterTypes.push(
        portType === 'int' ? 'int' : portType === 'bool' ? 'bool' : 'float',
      )
      const initVal = def.registerInitValues[i]
      if (typeof initVal === 'number' || typeof initVal === 'boolean') {
        flatStateInit.push(initVal)
      } else if (Array.isArray(initVal)) {
        flatStateInit.push(initVal as number[])
      } else {
        flatStateInit.push(0)
      }
    }
    for (let i = 0; i < def.delayUpdateNodes.length; i++) {
      flatRegisterExprs.push({ op: 'reg', id: delayBase + i })
      flatRegisterNames.push(`${name}_delay_${i}`)
      flatRegisterTypes.push('float')
      flatStateInit.push(def.delayInitValues[i] ?? 0)
    }

    deferredCbUpdates.push({ name, registerBase: myRegBase, delayBase, nestedRegStart, regStartIdx })

    let totalNestedRegs = 0
    for (const nc of def.nestedCalls) totalNestedRegs += nestedCallRegCount(nc)
    registerBase += def.registerNames.length + def.delayUpdateNodes.length + totalNestedRegs
  }

  // ── Phase 2: Process non-cycle-breaking instances in topological order ──
  for (const name of nonCbOrder) {
    const inst = instances.get(name)!
    const def = inst._def

    exprBase.set(name, flatOutputExprs.length)

    // Delay states occupy registers immediately after the instance's named registers
    const delayBase = registerBase + def.registerNames.length
    // Nested call registers start after named registers + delay states
    const nestedRegStart = def.registerNames.length + def.delayUpdateNodes.length

    // Build input substitution map: input(i) → wiring expression
    const inputMap = new Map<number, ExprNode>()
    for (let i = 0; i < def.inputNames.length; i++) {
      const key = `${name}:${def.inputNames[i]}`
      const wiringExpr = inputExprNodes.get(key)
      if (wiringExpr !== undefined) {
        // Resolve refs in the wiring expression (they reference earlier instances)
        inputMap.set(i, resolveRefs(wiringExpr, resolvedOutputs, resolvedOutputNames))
      } else {
        // Use default value or 0
        const defaultExpr = def.inputDefaults[i]
        const node = defaultExpr?._node ?? 0
        inputMap.set(i, inlineCalls(typeof node === 'object' ? cloneExpr(node) : node))
      }
    }

    // Helper: full expression pipeline for a local module expression.
    // Each pass uses its own identity memo so shared subexpressions (DAG nodes) are
    // processed exactly once per pass — preventing exponential work across all stages.
    const processExpr = (rawNode: ExprNode): ExprNode => {
      const cloneMemo = new WeakMap<object, ExprNode>()
      let expr = cloneExpr(rawNode, cloneMemo)
      // Resolve nested program calls (ProgramType.call()) before other transforms
      expr = resolveNestedOutputs(expr, def.nestedCalls, nestedRegStart)
      const inlineMemo = new WeakMap<object, ExprNode>()
      expr = inlineCalls(expr, inlineMemo)
      const offsetMemo = new WeakMap<object, ExprNode>()
      expr = offsetRegisters(expr, registerBase, offsetMemo)
      const delayMemo = new WeakMap<object, ExprNode>()
      expr = resolveDelayValues(expr, delayBase, delayMemo)
      const substMemo = new WeakMap<object, ExprNode>()
      expr = substituteInputs(expr, inputMap, substMemo)
      const refsMemo = new WeakMap<object, ExprNode>()
      expr = resolveRefs(expr, resolvedOutputs, resolvedOutputNames, refsMemo)
      // Lower array ops (zeros, ones, fill, array_literal, etc.) to primitives.
      // lowerArrayOps returns the original node unchanged when no array ops exist,
      // preserving DAG identity throughout.
      const lowerMemo = new WeakMap<object, ExprNode>()
      expr = lowerArrayOps(expr, lowerMemo)
      return expr
    }

    // Process each output expression
    // Order matters: offset registers BEFORE substituting inputs/resolving refs,
    // because wiring expressions already have globally-correct register IDs.
    const instOutputExprs: ExprNode[] = []
    for (let i = 0; i < def.outputExprNodes.length; i++) {
      const expr = processExpr(def.outputExprNodes[i])
      flatOutputExprs.push(expr)
      instOutputExprs.push(expr)
    }
    resolvedOutputs.set(name, instOutputExprs)
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

      // Register name, init value, and type
      flatRegisterNames.push(`${name}_${def.registerNames[i]}`)
      const portType = def.registerPortTypes[i]
      flatRegisterTypes.push(
        portType === 'int' ? 'int' : portType === 'bool' ? 'bool' : 'float',
      )
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
      flatRegisterTypes.push('float')
      flatStateInit.push(def.delayInitValues[i] ?? 0)
    }

    // Process nested program call registers
    if (def.nestedCalls.length > 0) {
      // Build resolved nested outputs map for call-arg resolution
      const resolvedNestedOutputs = new Map<number, ExprNode[]>()
      let ncRegCursor = nestedRegStart
      for (let ncIdx = 0; ncIdx < def.nestedCalls.length; ncIdx++) {
        const nc = def.nestedCalls[ncIdx]
        const nd = nc.programDef
        const ncRegBase = ncRegCursor
        const ncDelayBase = ncRegBase + nd.registerNames.length
        const ncNestedStart = ncDelayBase + nd.delayUpdateNodes.length

        const resolved: ExprNode[] = []
        for (let outId = 0; outId < nd.outputExprNodes.length; outId++) {
          const cloneMemo = new WeakMap<object, ExprNode>()
          let expr = cloneExpr(nd.outputExprNodes[outId], cloneMemo)
          expr = resolveNestedOutputs(expr, nd.nestedCalls, ncNestedStart)
          const inlineMemo = new WeakMap<object, ExprNode>()
          expr = inlineCalls(expr, inlineMemo)
          const offsetMemo = new WeakMap<object, ExprNode>()
          expr = offsetRegisters(expr, ncRegBase, offsetMemo)
          const delayMemo = new WeakMap<object, ExprNode>()
          expr = resolveDelayValues(expr, ncDelayBase, delayMemo)
          const argMap = new Map<number, ExprNode>()
          const argRefMemo = new WeakMap<object, ExprNode>()
          for (let i = 0; i < nc.callArgNodes.length; i++) {
            let arg = cloneExpr(nc.callArgNodes[i])
            arg = substituteNestedOutputRefs(arg, resolvedNestedOutputs, argRefMemo)
            argMap.set(i, arg)
          }
          const substMemo = new WeakMap<object, ExprNode>()
          expr = substituteInputs(expr, argMap, substMemo)
          resolved.push(expr)
        }
        resolvedNestedOutputs.set(ncIdx, resolved)
        ncRegCursor += nestedCallRegCount(nc)
      }

      const nested = collectNestedRegisterExprs(def.nestedCalls, nestedRegStart, name, resolvedNestedOutputs)
      for (let i = 0; i < nested.exprs.length; i++) {
        // Apply the parent instance's global transforms (offsetRegisters, substituteInputs, resolveRefs)
        let expr = nested.exprs[i]
        const nestedOffsetMemo = new WeakMap<object, ExprNode>()
        expr = offsetRegisters(expr, registerBase, nestedOffsetMemo)
        const nestedSubstMemo = new WeakMap<object, ExprNode>()
        expr = substituteInputs(expr, inputMap, nestedSubstMemo)
        const nestedRefsMemo = new WeakMap<object, ExprNode>()
        expr = resolveRefs(expr, resolvedOutputs, resolvedOutputNames, nestedRefsMemo)
        flatRegisterExprs.push(expr)
        flatRegisterNames.push(nested.names[i])
        flatRegisterTypes.push('float')
        flatStateInit.push(nested.inits[i])
      }
    }

    // Total registers for this instance: named + delays + all nested calls
    let totalNestedRegs = 0
    for (const nc of def.nestedCalls) totalNestedRegs += nestedCallRegCount(nc)
    registerBase += def.registerNames.length + def.delayUpdateNodes.length + totalNestedRegs
  }

  // ── Phase 3: Resolve deferred register updates for cycle-breaking instances ──
  // Now that all non-cycle-breaking instances are resolved, we can compute
  // the register update expressions (which depend on other instances' outputs).
  for (const { name, registerBase: rBase, delayBase, nestedRegStart, regStartIdx } of deferredCbUpdates) {
    const inst = instances.get(name)!
    const def = inst._def

    // Build input substitution map (all refs are now resolvable)
    const inputMap = new Map<number, ExprNode>()
    for (let i = 0; i < def.inputNames.length; i++) {
      const key = `${name}:${def.inputNames[i]}`
      const wiringExpr = inputExprNodes.get(key)
      if (wiringExpr !== undefined) {
        inputMap.set(i, resolveRefs(wiringExpr, resolvedOutputs, resolvedOutputNames))
      } else {
        const defaultExpr = def.inputDefaults[i]
        const node = defaultExpr?._node ?? 0
        inputMap.set(i, inlineCalls(typeof node === 'object' ? cloneExpr(node) : node))
      }
    }

    const processExpr = (rawNode: ExprNode): ExprNode => {
      const cloneMemo = new WeakMap<object, ExprNode>()
      let expr = cloneExpr(rawNode, cloneMemo)
      expr = resolveNestedOutputs(expr, def.nestedCalls, nestedRegStart)
      const inlineMemo = new WeakMap<object, ExprNode>()
      expr = inlineCalls(expr, inlineMemo)
      const offsetMemo = new WeakMap<object, ExprNode>()
      expr = offsetRegisters(expr, rBase, offsetMemo)
      const delayMemo = new WeakMap<object, ExprNode>()
      expr = resolveDelayValues(expr, delayBase, delayMemo)
      const substMemo = new WeakMap<object, ExprNode>()
      expr = substituteInputs(expr, inputMap, substMemo)
      const refsMemo = new WeakMap<object, ExprNode>()
      expr = resolveRefs(expr, resolvedOutputs, resolvedOutputNames, refsMemo)
      const lowerMemo = new WeakMap<object, ExprNode>()
      expr = lowerArrayOps(expr, lowerMemo)
      return expr
    }

    // Overwrite register update placeholders
    let regIdx = regStartIdx
    for (let i = 0; i < def.registerExprNodes.length; i++) {
      const regNode = def.registerExprNodes[i]
      if (regNode !== null) {
        flatRegisterExprs[regIdx] = processExpr(regNode)
      }
      regIdx++
    }

    // Delay state updates
    for (let i = 0; i < def.delayUpdateNodes.length; i++) {
      const updateNode = def.delayUpdateNodes[i]
      if (updateNode !== null && updateNode !== undefined) {
        flatRegisterExprs[regIdx] = processExpr(updateNode as ExprNode)
      }
      regIdx++
    }

    // Nested call register updates
    if (def.nestedCalls.length > 0) {
      const resolvedNestedOutputs = new Map<number, ExprNode[]>()
      let ncRegCursor = nestedRegStart
      for (let ncIdx = 0; ncIdx < def.nestedCalls.length; ncIdx++) {
        const nc = def.nestedCalls[ncIdx]
        const nd = nc.programDef
        const ncRegBase = ncRegCursor
        const ncDelayBase = ncRegBase + nd.registerNames.length
        const ncNestedStart = ncDelayBase + nd.delayUpdateNodes.length

        const resolved: ExprNode[] = []
        for (let outId = 0; outId < nd.outputExprNodes.length; outId++) {
          const cloneMemo = new WeakMap<object, ExprNode>()
          let expr = cloneExpr(nd.outputExprNodes[outId], cloneMemo)
          expr = resolveNestedOutputs(expr, nd.nestedCalls, ncNestedStart)
          const inlineMemo = new WeakMap<object, ExprNode>()
          expr = inlineCalls(expr, inlineMemo)
          const offsetMemo = new WeakMap<object, ExprNode>()
          expr = offsetRegisters(expr, ncRegBase, offsetMemo)
          const delayMemo = new WeakMap<object, ExprNode>()
          expr = resolveDelayValues(expr, ncDelayBase, delayMemo)
          const argMap = new Map<number, ExprNode>()
          const argRefMemo = new WeakMap<object, ExprNode>()
          for (let i = 0; i < nc.callArgNodes.length; i++) {
            let arg = cloneExpr(nc.callArgNodes[i])
            arg = substituteNestedOutputRefs(arg, resolvedNestedOutputs, argRefMemo)
            argMap.set(i, arg)
          }
          const substMemo = new WeakMap<object, ExprNode>()
          expr = substituteInputs(expr, argMap, substMemo)
          resolved.push(expr)
        }
        resolvedNestedOutputs.set(ncIdx, resolved)
        ncRegCursor += nestedCallRegCount(nc)
      }

      const nested = collectNestedRegisterExprs(def.nestedCalls, nestedRegStart, name, resolvedNestedOutputs)
      for (let i = 0; i < nested.exprs.length; i++) {
        let expr = nested.exprs[i]
        const nestedOffsetMemo = new WeakMap<object, ExprNode>()
        expr = offsetRegisters(expr, rBase, nestedOffsetMemo)
        const nestedSubstMemo = new WeakMap<object, ExprNode>()
        expr = substituteInputs(expr, inputMap, nestedSubstMemo)
        const nestedRefsMemo = new WeakMap<object, ExprNode>()
        expr = resolveRefs(expr, resolvedOutputs, resolvedOutputNames, nestedRefsMemo)
        flatRegisterExprs[regIdx] = expr
        regIdx++
      }
    }
  }

  // Build output indices: map graphOutputs → flat output indices
  // Combined order: cycle-breaking instances first, then topological order
  const fullOrder = [...cycleBreakerList, ...nonCbOrder]
  const outputIndices: number[] = []
  let outputBase = 0
  const outputStart = new Map<string, number>()

  // Recompute output starts
  for (const name of fullOrder) {
    outputStart.set(name, outputBase)
    const inst = instances.get(name)!
    outputBase += inst._def.outputNames.length
  }

  for (const { instance, output } of graphOutputs) {
    const inst = instances.get(instance)
    if (!inst) continue
    const outputIdx = inst._def.outputNames.indexOf(output)
    if (outputIdx === -1) continue
    const flatIdx = (outputStart.get(instance) ?? 0) + outputIdx
    outputIndices.push(flatIdx)
  }

  // Compile expression trees → flat instruction stream
  const program = emitNumericProgram(flatOutputExprs, flatRegisterExprs, flatStateInit, flatRegisterTypes)

  // Compute array slot names for state transfer on hot-swap.
  // Array slots are allocated in the order array entries appear in flatStateInit,
  // matching the order of program.array_slot_sizes.
  const arraySlotNames: string[] = []
  for (let i = 0; i < flatStateInit.length; i++) {
    if (Array.isArray(flatStateInit[i])) {
      arraySlotNames.push(flatRegisterNames[i])
    }
  }

  return {
    schema: 'tropical_plan_4',
    config: { sample_rate: 44100 },
    state_init: flatStateInit as (number | boolean)[],
    register_names: flatRegisterNames,
    register_types: flatRegisterTypes,
    array_slot_names: arraySlotNames,
    outputs: outputIndices,
    instructions:     program.instructions,
    register_count:   program.register_count,
    array_slot_count: program.array_slot_count,
    array_slot_sizes: program.array_slot_sizes,
    output_targets:   program.output_targets,
    register_targets: program.register_targets,
  }
}

