/**
 * extract_session_delays.ts — pre-emit pass that hoists session-level
 * `delay()`-wrapped wires into module slots.
 *
 * Every MCP-built wire is wrapped in a unit delay by `setWireExpr`
 * (see `session.ts`'s "Wire storage" section). This pass walks
 * `session.inputExprNodes`, allocates a fresh module slot per
 * top-level `{op:'delay', args:[src], init, id}` wire, rewrites the
 * wire to `{op:'sessionSlot', index: slotIdx}`, and records the
 * source expression in `session.delaySlotRegistry` for the JIT
 * compile step to emit `WriteSlot` instructions into the scheduler's
 * `state_evolution` phase.
 *
 * After this pass:
 *
 *   - `session.inputExprNodes` contains no top-level `delay()` ops.
 *     Every wire is a `translateNode`-compatible form (slot read +
 *     scalar arithmetic).
 *   - `session.delaySlotRegistry` carries one entry per extracted
 *     delay — slot index, slot name (for hot-swap state transfer),
 *     init value (for `slot_defaults`), un-delayed source expression
 *     (translated into a `WriteSlot` at compile time).
 *   - The instance dependency graph (built by
 *     `computeInstanceTopoOrder` from `inputExprNodes`) is acyclic
 *     by construction — `sessionSlot` ops carry no instance refs.
 *
 * Wires whose top-level is *not* a `delay()` (legacy patches loaded
 * via `loadJSON`, programmatic test setups bypassing MCP helpers,
 * etc.) are left untouched. They go through the existing
 * `translateNode` path directly. The defensive `assertSessionAcyclic`
 * in Phase 5 catches any such session whose un-extracted wires form
 * an inter-instance cycle.
 *
 * Idempotent: re-running finds no top-level delays and is a no-op.
 */

import type { ExprNode, SessionState, DelaySlotEntry } from '../../session.js'

/** Detect a top-level session delay wrapper. */
function isDelayWrap(expr: ExprNode): expr is {
  op: 'delay'; args: [ExprNode]; init?: number; id?: string
} {
  if (typeof expr !== 'object' || expr === null || Array.isArray(expr)) return false
  const obj = expr as { op?: unknown; args?: unknown }
  return obj.op === 'delay'
      && Array.isArray(obj.args)
      && obj.args.length === 1
}

/** Walk `session.inputExprNodes`. For each top-level `delay()`
 *  wire, allocate a module slot, rewrite the wire to a sessionSlot
 *  read, and record the source expression for later WriteSlot
 *  emission. Mutates session in place. */
export function extractSessionDelays(session: SessionState): void {
  for (const [key, expr] of session.inputExprNodes) {
    if (!isDelayWrap(expr)) continue
    const wrap = expr as { args: [ExprNode]; init?: number; id?: string }
    const src  = wrap.args[0]
    const init = wrap.init ?? 0
    const id   = wrap.id   ?? `__autodelay:${key}`

    const slotIdx  = session.slotCount
    session.slotCount += 1
    const slotName = id

    const entry: DelaySlotEntry = {
      slotIdx,
      slotName,
      init,
      sourceExpr: src,
      scalarType: 'float',
    }
    session.delaySlotRegistry.push(entry)

    session.inputExprNodes.set(key, { op: 'sessionSlot', index: slotIdx })
  }
}
