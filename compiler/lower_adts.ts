/**
 * lower_adts.ts — Lower algebraic data type operations to Pack/Index/Select.
 *
 * This pass runs after flatten and lower_arrays, transforming high-level ADT
 * ops into primitives the emitter already supports:
 *
 *   construct("Point", {x: e1, y: e2})  →  Pack([e1, e2])
 *   project("Point", "x", expr)         →  Index(expr_slots, offset)
 *   inject("Temp", "Hot")               →  tag_value (scalar, no payload)
 *   inject("NoteEvent", "NoteOn", ...)   →  Pack([tag, payload...])
 *   match(type, scrutinee, branches)     →  nested Select(Equal(tag, N), ...)
 *   bound("name")                        →  Index(scrutinee_slots, offset)
 *
 * All shapes are static (known at compile time via the TypeRegistry).
 */

import type { ExprNode } from './expr.js'
import { TypeRegistry } from './type_registry.js'

/**
 * Lower all ADT operations in an expression tree.
 * Recursively walks the tree, expanding ADT ops to Pack/Index/Select.
 *
 * @param node The expression tree to lower.
 * @param registry The type registry for resolving type definitions.
 * @param memo Optional WeakMap for identity-based memoization.
 */
export function lowerAdtOps(
  node: ExprNode,
  registry: TypeRegistry,
  memo?: WeakMap<object, ExprNode>,
): ExprNode {
  if (typeof node === 'number' || typeof node === 'boolean') return node
  if (Array.isArray(node)) return node.map(n => lowerAdtOps(n, registry, memo))
  if (typeof node !== 'object' || node === null) return node

  if (memo) {
    const cached = memo.get(node as object)
    if (cached !== undefined) return cached
  }

  const obj = node as { op: string; [k: string]: unknown }
  let result: ExprNode

  switch (obj.op) {
    case 'construct':
      result = lowerConstruct(obj, registry, memo)
      break
    case 'project':
      result = lowerProject(obj, registry, memo)
      break
    case 'inject':
      result = lowerInject(obj, registry, memo)
      break
    case 'match':
      result = lowerMatch(obj, registry, memo)
      break
    default:
      result = lowerChildren(obj, registry, memo)
      break
  }

  if (memo && typeof result === 'object' && result !== null && !Array.isArray(result)) {
    memo.set(node as object, result)
  }

  return result
}

// ─────────────────────────────────────────────────────────────
// Child traversal
// ─────────────────────────────────────────────────────────────

function lowerChildren(
  obj: Record<string, unknown>,
  registry: TypeRegistry,
  memo?: WeakMap<object, ExprNode>,
): ExprNode {
  const result: Record<string, unknown> = {}
  let changed = false
  for (const [k, v] of Object.entries(obj)) {
    if (Array.isArray(v)) {
      const arr = v as ExprNode[]
      const newArr = arr.map(n => lowerAdtOps(n, registry, memo))
      const anyChanged = newArr.some((n, i) => n !== arr[i])
      result[k] = anyChanged ? newArr : arr
      if (anyChanged) changed = true
    } else if (typeof v === 'object' && v !== null && 'op' in v) {
      const newV = lowerAdtOps(v as ExprNode, registry, memo)
      result[k] = newV
      if (newV !== v) changed = true
    } else {
      result[k] = v
    }
  }
  return changed ? result as ExprNode : obj as ExprNode
}

// ─────────────────────────────────────────────────────────────
// construct → Pack
// ─────────────────────────────────────────────────────────────

/**
 * construct("Point", {x: e1, y: e2}) → [e1, e2]  (inline array = Pack)
 *
 * Fields are packed in declaration order from the type definition.
 */
function lowerConstruct(
  obj: Record<string, unknown>,
  registry: TypeRegistry,
  memo?: WeakMap<object, ExprNode>,
): ExprNode {
  const typeName = obj.type_name as string
  const fields = obj.fields as Record<string, ExprNode>
  const fieldNames = registry.fieldNames(typeName)

  const elements: ExprNode[] = fieldNames.map(name => {
    const fieldExpr = fields[name]
    if (fieldExpr === undefined) throw new Error(`construct '${typeName}': missing field '${name}'`)
    return lowerAdtOps(fieldExpr, registry, memo)
  })

  // Single-field product: just the scalar value, no Pack needed
  if (elements.length === 1) return elements[0]
  return elements  // inline array → Pack
}

// ─────────────────────────────────────────────────────────────
// project → Index
// ─────────────────────────────────────────────────────────────

/**
 * project("Point", "x", expr) → Index(lowered_expr, offset)
 */
function lowerProject(
  obj: Record<string, unknown>,
  registry: TypeRegistry,
  memo?: WeakMap<object, ExprNode>,
): ExprNode {
  const typeName = obj.type_name as string
  const field = obj.field as string
  const expr = lowerAdtOps(obj.expr as ExprNode, registry, memo)
  const offset = registry.fieldOffset(typeName, field)
  const totalSlots = registry.productSlotCount(typeName)

  // Single-field product: the expression IS the scalar value
  if (totalSlots === 1) return expr

  return { op: 'index', args: [expr, offset] }
}

// ─────────────────────────────────────────────────────────────
// inject → tag value or Pack([tag, payload...])
// ─────────────────────────────────────────────────────────────

/**
 * inject("Temp", "Hot")                     → 0.0 (tag only, slot count = 1)
 * inject("NoteEvent", "NoteOn", {pitch, vel}) → [0.0, pitch, vel]
 */
function lowerInject(
  obj: Record<string, unknown>,
  registry: TypeRegistry,
  memo?: WeakMap<object, ExprNode>,
): ExprNode {
  const typeName = obj.type_name as string
  const variantName = obj.variant as string
  const payloadExprs = obj.payload as Record<string, ExprNode> | undefined
  const tag = registry.variantTag(typeName, variantName)
  const totalSlots = registry.coproductSlotCount(typeName)

  if (totalSlots === 1) {
    // Nullary-only coproduct: just the tag value
    return tag
  }

  // Build: [tag, payload_field_0, payload_field_1, ..., padding_zeros...]
  const payloadFields = registry.variantPayloadFields(typeName, variantName)
  const elements: ExprNode[] = [tag]  // tag at slot 0

  for (const field of payloadFields) {
    const fieldExpr = payloadExprs?.[field.name]
    if (fieldExpr === undefined) throw new Error(`inject '${typeName}::${variantName}': missing payload field '${field.name}'`)
    elements.push(lowerAdtOps(fieldExpr, registry, memo))
  }

  // Pad with zeros to reach totalSlots (max payload across all variants)
  while (elements.length < totalSlots) {
    elements.push(0)
  }

  return elements  // inline array → Pack
}

// ─────────────────────────────────────────────────────────────
// match → nested Select(Equal(tag, N), branchBody, ...)
// ─────────────────────────────────────────────────────────────

/**
 * match("Temp", scrutinee, {Hot: {body: e1}, Cold: {body: e2}})
 *   → Select(Equal(tag, 0), lowered_e1, Select(Equal(tag, 1), lowered_e2, 0))
 *
 * For coproducts with payloads, bound("name") nodes in branch bodies are
 * substituted with Index(scrutinee_slots, offset) references.
 */
function lowerMatch(
  obj: Record<string, unknown>,
  registry: TypeRegistry,
  memo?: WeakMap<object, ExprNode>,
): ExprNode {
  const typeName = obj.type_name as string
  const scrutinee = lowerAdtOps(obj.scrutinee as ExprNode, registry, memo)
  const branches = obj.branches as Record<string, { bind?: string[]; body: ExprNode }>
  const variantNames = registry.variantNames(typeName)
  const totalSlots = registry.coproductSlotCount(typeName)

  // Extract tag from scrutinee
  const tagExpr: ExprNode = totalSlots === 1
    ? scrutinee  // tag-only coproduct, scrutinee IS the tag
    : { op: 'index', args: [scrutinee, 0] }

  // Build branch bodies (last to first for nested Select)
  // Select(Equal(tag, 0), body0, Select(Equal(tag, 1), body1, ... Select(Equal(tag, N-1), bodyN-1, 0)))
  const loweredBodies: ExprNode[] = []
  for (const variantName of variantNames) {
    const branch = branches[variantName]
    if (!branch) throw new Error(`match '${typeName}': missing branch for variant '${variantName}'`)

    // Substitute bound variables → Index into scrutinee
    let body = branch.body
    if (branch.bind && branch.bind.length > 0) {
      const payloadFields = registry.variantPayloadFields(typeName, variantName)
      // Build substitution map: bound name → Index(scrutinee, offset)
      // Tag is slot 0, payload fields start at slot 1
      const subs = new Map<string, ExprNode>()
      for (let i = 0; i < branch.bind.length; i++) {
        const bindName = branch.bind[i]
        const fieldIdx = payloadFields.findIndex(f => f.name === bindName)
        if (fieldIdx === -1) {
          // bind name maps positionally
          const offset = 1 + i  // tag at 0, then payload fields
          subs.set(bindName, { op: 'index', args: [scrutinee, offset] })
        } else {
          const offset = 1 + fieldIdx
          subs.set(bindName, { op: 'index', args: [scrutinee, offset] })
        }
      }
      body = substituteBound(body, subs)
    }

    loweredBodies.push(lowerAdtOps(body, registry, memo))
  }

  // Build nested Select from right to left
  let result: ExprNode = 0  // fallback (should never be reached in well-typed code)
  for (let i = variantNames.length - 1; i >= 0; i--) {
    const cond: ExprNode = { op: 'eq', args: [tagExpr, i] }
    result = { op: 'select', args: [cond, loweredBodies[i], result] }
  }

  return result
}

// ─────────────────────────────────────────────────────────────
// Bound variable substitution
// ─────────────────────────────────────────────────────────────

/**
 * Replace all bound("name") nodes with the corresponding expression
 * from the substitution map.
 */
function substituteBound(node: ExprNode, subs: Map<string, ExprNode>): ExprNode {
  if (typeof node === 'number' || typeof node === 'boolean') return node
  if (Array.isArray(node)) return node.map(n => substituteBound(n, subs))
  if (typeof node !== 'object' || node === null) return node

  const obj = node as { op: string; [k: string]: unknown }
  if (obj.op === 'bound') {
    const name = obj.name as string
    const sub = subs.get(name)
    if (!sub) throw new Error(`bound('${name}'): no substitution available`)
    return sub
  }

  // Walk children
  const result: Record<string, unknown> = {}
  let changed = false
  for (const [k, v] of Object.entries(obj)) {
    if (Array.isArray(v)) {
      const arr = v as ExprNode[]
      const newArr = arr.map(n => substituteBound(n, subs))
      const anyChanged = newArr.some((n, i) => n !== arr[i])
      result[k] = anyChanged ? newArr : arr
      if (anyChanged) changed = true
    } else if (typeof v === 'object' && v !== null && 'op' in v) {
      const newV = substituteBound(v as ExprNode, subs)
      result[k] = newV
      if (newV !== v) changed = true
    } else {
      result[k] = v
    }
  }
  return changed ? result as ExprNode : obj as ExprNode
}
