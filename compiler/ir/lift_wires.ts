/**
 * lift_wires.ts — pre-compile pass that lifts complex wire expressions
 * to anonymous program instances.
 *
 * Wires whose expression contains forms that `translateNode` doesn't
 * support (array literals, session-level `delay()`) are extracted
 * into anonymous programs at session pre-compile time. The lifted program goes through the full strata
 * pipeline — `arrayLower` handles array literals; `delay()` desugars
 * to a synthetic `RegDecl` with `update` populated, which the
 * standard slot allocator handles like any user-written reg — and
 * the existing per-instance compile path then treats the lifted
 * instance like any other. The lifted programs have no InstanceDecls
 * in their bodies, so they're acyclic by construction and satisfy
 * `strataPipeline`'s acyclic-input contract.
 *
 * After this pass:
 *   - `session.inputExprNodes` entries contain only "simple" wires that
 *     `translateNode` can handle (refs, params, triggers, sentinels,
 *     scalar arithmetic).
 *   - `session.instanceRegistry` has one extra `__wire_${i}` instance
 *     per lifted wire.
 *   - Each lifted instance's free refs are wired in `inputExprNodes`
 *     so the existing compile path resolves them via slot reads.
 *
 * Idempotent on already-lifted sessions: re-running finds no
 * complex wires and is a no-op.
 */

import type { SessionState, ExprNode } from '../session.js'
import { freeRefs, liftWireToProgram } from './wire_program.js'
import {
  instanceName, rawName, type PortRef, type WireKey,
  wireKey as toWireKey, portRef, portName as toPortName,
} from './branded_names.js'
import { programTypeFromResolved } from './strata.js'
import { instantiate } from '../program_types.js'
import { allocateOutputSlots } from '../session.js'

// ─── Detection: which wires need to be lifted? ────────────────────────────

/** Returns true if any subtree of `expr` contains a form that
 *  `translateNode` in `compile_session_slotted_helpers.ts` rejects.
 *  Only array literals trigger a lift: `delay()` is now handled by
 *  the session-level extraction pass (`extractSessionDelays`) which
 *  allocates a module slot and emits a `WriteSlot` instruction into
 *  the scheduler's state-evolution phase. Lifting delays into
 *  anonymous wire-program instances would recreate the dep-graph
 *  cycle the delay was meant to break — the session-level extraction
 *  keeps the dep graph acyclic by construction. */
export function needsWireLift(expr: ExprNode): boolean {
  if (Array.isArray(expr)) return true   // bare array literal
  if (typeof expr !== 'object' || expr === null) return false
  const obj = expr as Record<string, unknown>
  const op = obj.op
  if (op === 'array' || op === 'arrayLiteral') return true
  // Recurse into known child-shapes.
  if (Array.isArray(obj.args)) {
    for (const a of obj.args as ExprNode[]) if (needsWireLift(a)) return true
  }
  if (Array.isArray(obj.items)) {
    for (const a of obj.items as ExprNode[]) if (needsWireLift(a)) return true
  }
  return false
}

// ─── Lift one wire ────────────────────────────────────────────────────────

/** Returns the synthesized name for the lifted wire-instance. The
 *  counter is per-session, stored on `session.nameCounters` under a
 *  reserved key so the existing nextName mechanism doesn't collide. */
function nextWireName(session: SessionState): string {
  const key = '__wire'
  const n = (session.nameCounters.get(key) ?? 0) + 1
  session.nameCounters.set(key, n)
  return `__wire_${n}`
}

/** Lift a single wire's `ExprNode` to an anonymous program instance.
 *  Mutates `session` — registers the instance and wires its free refs
 *  back to their original sources. Returns the replacement `ExprNode`:
 *  `{op:'ref', instance: '__wire_N', output: 'out'}`. */
function liftOneWire(
  session: SessionState,
  expr: ExprNode,
  context: string,
): ExprNode {
  const refs = freeRefs(expr)
  const synthName = instanceName(nextWireName(session))

  // Build the synthetic ResolvedProgram. The wire-program lift
  // accepts free refs and a synthesized name; output port is 'out'.
  const lifted = liftWireToProgram(expr, refs, synthName)

  // Run strata to lower combinators (let, array literals via
  // arrayLower; session delay desugars to a synthetic RegDecl).
  const compiled = programTypeFromResolved(lifted, new Map(), {
    displayName:  rawName(synthName),
    inlineNested: session.inlineNested,
  })

  // Register the type so materializeSession (oracle path) can find it
  // by name, then instantiate and allocate output slots.
  session.typeRegistry.set(rawName(synthName), compiled)
  session.programs.set(rawName(synthName), lifted)
  const inst = instantiate(compiled, rawName(synthName), {
    baseTypeName: rawName(synthName),
  })
  session.instanceRegistry.set(rawName(synthName), inst)
  allocateOutputSlots(session, synthName, compiled)

  // Wire each free ref to its corresponding input on the lifted
  // instance. The input naming convention matches `liftWireToProgram`:
  // `${instance.replace('.','_')}__${port}`.
  const refList = Array.from(refs).sort((a, b) =>
    `${rawName(a.instance)}:${rawName(a.port)}`.localeCompare(
      `${rawName(b.instance)}:${rawName(b.port)}`,
    ),
  )
  for (const ref of refList) {
    const inputName = `${rawName(ref.instance).replace(/\./g, '_')}__${rawName(ref.port)}`
    const key = toWireKey(portRef(synthName, toPortName(inputName)))
    // Forward the original ref expression unchanged — this is now a
    // simple `{op:'ref',...}` wire that translateNode handles.
    session.inputExprNodes.set(key, {
      op: 'ref',
      instance: rawName(ref.instance),
      output: rawName(ref.port),
    })
  }

  void context
  return { op: 'ref', instance: rawName(synthName), output: 'out' }
}

// ─── Entry point ──────────────────────────────────────────────────────────

/** Walk `session.inputExprNodes`. For every wire whose expression
 *  contains a form `translateNode` doesn't handle, lift it to an
 *  anonymous `__wire_${i}` program instance. After this pass, every
 *  remaining wire is `translateNode`-compatible. */
export function liftWiresToInstances(session: SessionState): void {
  // Capture wires before mutation; we'll modify the map as we go but
  // never want to re-lift a wire we just inserted.
  const toLift: Array<{ key: WireKey; expr: ExprNode }> = []
  for (const [key, expr] of session.inputExprNodes) {
    if (needsWireLift(expr)) toLift.push({ key, expr })
  }

  for (const { key, expr } of toLift) {
    const replacement = liftOneWire(session, expr, key)
    session.inputExprNodes.set(key, replacement)
  }
}

// Re-exports for callers
export type { PortRef }
