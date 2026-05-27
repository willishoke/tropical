/**
 * extract_session_delays.ts — pre-emit pass that hoists every
 * `delay()` op found anywhere in a wire expression into a session
 * module slot.
 *
 * Two ingest paths put `delay()` ops into `session.inputExprNodes`:
 *
 *   - MCP wire helpers (`setWireExpr` in `session.ts`) wrap every
 *     wire at its top level — the cross-mod auto-delay mechanism.
 *   - Hand-written JSON patches (loaded via `loadProgramAsSession`)
 *     can place `delay()` anywhere in a wire's arithmetic tree
 *     (e.g. `add(220, delay(mul(d2.y + d3.y, 2)))` for cross-FM).
 *
 * This pass walks every wire expression recursively. For each
 * `{op:'delay', args:[src], init?, id?}` it encounters:
 *
 *   1. Allocate a fresh module slot via `session.slotCount++`.
 *   2. Replace the delay node in-place (in the expression tree)
 *      with `{op: 'sessionSlot', index: slotIdx}`.
 *   3. Push a `DelaySlotEntry` to `session.delaySlotRegistry` so
 *      `compileSessionSlotted` emits a `WriteSlot` in the
 *      scheduler's `state_evolution` phase.
 *
 * The walk is outer-first so the WriteSlot for an outer
 * `delay(delay(x))` lands before the WriteSlot for the inner one.
 * The state_evolution phase runs in registration order; outer-first
 * means the outer slot's read of the inner sessionSlot picks up the
 * inner's *previous* sample value before the inner is rewritten —
 * the chain accumulates 1 sample per nested delay.
 *
 * After this pass:
 *
 *   - `session.inputExprNodes` contains no `delay()` ops anywhere.
 *     Every remaining op is a `translateNode`-compatible form
 *     (`ref`, `sessionSlot`, scalar arithmetic, …).
 *   - `session.delaySlotRegistry` carries one entry per extracted
 *     `delay()`.
 *   - The instance dep graph (built by `computeInstanceTopoOrder`
 *     from `inputExprNodes`) loses every edge that ran through a
 *     `delay()` — for fully-delayed cycles the graph becomes
 *     acyclic by construction.
 *
 * Idempotent: re-running finds no delays and is a no-op.
 */

import type { ExprNode, SessionState, DelaySlotEntry } from '../../session.js'
import { slotKey, instanceName as toInstanceName } from '../branded_names.js'

type DelayNode = {
  op:    'delay'
  args:  [ExprNode]
  init?: number
  id?:   string
  // ExprNode's object variant carries an open `[key: string]: unknown`
  // index signature; mirror it here so the type predicate below is
  // assignable to ExprNode.
  [key: string]: unknown
}

/** Determine whether a delay's source expression resolves to an array
 *  shape, and if so, its element count. Currently recognizes only
 *  `{op:'ref', instance, output}` against an array-typed output —
 *  the closed set of array-shaped expressions that survive past
 *  `liftWiresToInstances` (which lifts every wire containing an
 *  array literal). Other array-producing patterns require their own
 *  analysis pass and fall through to scalar handling. */
function inferArraySourceShape(
  expr: ExprNode,
  session: SessionState,
): { size: number } | undefined {
  if (typeof expr !== 'object' || expr === null || Array.isArray(expr)) return undefined
  const obj = expr as Record<string, unknown>
  if (obj.op !== 'ref') return undefined
  if (typeof obj.instance !== 'string' || typeof obj.output !== 'string') return undefined
  const sourceKey = slotKey(toInstanceName(obj.instance), obj.output)
  const meta = session.outputPortMeta.get(sourceKey)
  if (meta === undefined) return undefined
  if (meta.arraySlot === undefined || meta.arraySize === undefined) return undefined
  return { size: meta.arraySize }
}

/** Detect a session-level `delay()` op. */
function isDelay(expr: ExprNode): expr is DelayNode {
  if (typeof expr !== 'object' || expr === null || Array.isArray(expr)) return false
  const obj = expr as { op?: unknown; args?: unknown }
  return obj.op === 'delay'
      && Array.isArray(obj.args)
      && obj.args.length === 1
}

/** Walk `expr`, extract every `delay()` subtree into a fresh module
 *  slot, push a registry entry, and return the rewritten expression
 *  where every delay is replaced by a `sessionSlot` read.
 *
 *  The traversal is outer-first: when we hit a delay, we record its
 *  source (with the delay's nested delays *already rewritten* via the
 *  recursive call before recording) and push the entry. The push
 *  order is `[outer, …inner pushed during recursion]`, which is the
 *  correct execution order in `state_evolution` for accumulating
 *  latency on nested chains. */
function rewriteAndCollect(
  expr: ExprNode,
  session: SessionState,
  context: string,
): ExprNode {
  if (typeof expr !== 'object' || expr === null) return expr
  if (Array.isArray(expr)) {
    return expr.map((e, i) => rewriteAndCollect(e as ExprNode, session, `${context}[${i}]`))
  }

  if (isDelay(expr)) {
    const init = expr.init ?? 0
    const id   = expr.id   ?? `__autodelay:${context}#${session.delaySlotRegistry.length}`

    // Shape-polymorphic delay: discriminate on source shape. Scalar
    // sources allocate a module slot (the unified slot[] array,
    // float64-backed); array sources allocate an ioArraySlot (the
    // session's array-slot space). Both kinds get one entry in
    // `delaySlotRegistry`; `state_evolution` branches on `isArray`
    // to choose the WriteSlot vs. elementwise-Add emission path.
    //
    // Shape inference at this layer is intentionally limited to the
    // forms `extractSessionDelays` actually encounters: `{op:'ref'}`
    // to an array-typed output. Other array-producing forms (bare
    // literals, `{op:'array'/'arrayLiteral'}`) are caught upstream
    // by `liftWiresToInstances` and never reach here. Complex array-
    // valued expressions (arithmetic on array refs, etc.) would
    // need their own analysis pass; for now they fall through to the
    // scalar path and surface as a clear error later.
    const arrayShape = inferArraySourceShape(expr.args[0], session)
    if (arrayShape !== undefined) {
      const arraySlot = session.ioArraySlotCount
      session.ioArraySlotCount += 1
      session.ioArraySlotSizes.push(arrayShape.size)
      session.ioArraySlotNames.push(id)
      const entry: DelaySlotEntry = {
        slotName:   id,
        init,
        sourceExpr: null as unknown as ExprNode,
        scalarType: 'float',
        isArray:    true,
        arraySlot,
        arraySize:  arrayShape.size,
      }
      session.delaySlotRegistry.push(entry)
      entry.sourceExpr = rewriteAndCollect(expr.args[0], session, `${context}.delay`)
      return { op: 'sessionArraySlot', index: arraySlot, size: arrayShape.size }
    }

    const slotIdx = session.slotCount
    session.slotCount += 1
    // Push the outer entry first; the recursion into args[0] may
    // extract nested delays which append after. The state_evolution
    // emitter walks the registry in push order — outer-first writes
    // make the outer read the inner's previous-sample value, so the
    // latency stacks (one sample per nested delay).
    const entry: DelaySlotEntry = {
      slotIdx,
      slotName:   id,
      init,
      sourceExpr: null as unknown as ExprNode,  // patched below
      scalarType: 'float',
    }
    session.delaySlotRegistry.push(entry)
    entry.sourceExpr = rewriteAndCollect(expr.args[0], session, `${context}.delay`)

    return { op: 'sessionSlot', index: slotIdx }
  }

  // Generic recurse: rewrite every child ExprNode. Walk args, items
  // (array literals), and any other expression-valued fields.
  const obj = expr as Record<string, unknown>
  const out: Record<string, unknown> = { ...obj }
  for (const [k, v] of Object.entries(obj)) {
    if (k === 'op') continue
    if (Array.isArray(v)) {
      out[k] = v.map((e, i) =>
        typeof e === 'object' || typeof e === 'number' || typeof e === 'boolean'
          ? rewriteAndCollect(e as ExprNode, session, `${context}.${k}[${i}]`)
          : e
      )
    } else if (typeof v === 'object' && v !== null) {
      // Inline expressions (e.g. `match.scrutinee`); recurse.
      out[k] = rewriteAndCollect(v as ExprNode, session, `${context}.${k}`)
    }
  }
  return out as ExprNode
}

/** Walk every wire in `session.inputExprNodes`, extract every
 *  `delay()` op into a fresh module slot, rewrite the wire to a
 *  `sessionSlot` read, and record the source for later WriteSlot
 *  emission. Mutates session in place. */
export function extractSessionDelays(session: SessionState): void {
  for (const [key, expr] of session.inputExprNodes) {
    const rewritten = rewriteAndCollect(expr, session, key)
    if (rewritten !== expr) {
      session.inputExprNodes.set(key, rewritten)
    }
  }
}
