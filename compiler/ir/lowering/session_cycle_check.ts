/**
 * session_cycle_check.ts — defensive acyclicity invariant for
 * `compileSession`.
 *
 * After `extractSessionDelays` runs, every MCP-built wire is a
 * `sessionSlot` read (no inter-instance dependency in the dep graph)
 * or a sub-expression that doesn't traverse another instance, so the
 * instance dep graph is acyclic by construction.
 *
 * This check guards against programmatic `SessionState` constructions
 * that bypass the MCP helpers — test code, legacy `loadJSON` paths,
 * future ingest paths — by re-running cycle detection on the
 * post-extraction `inputExprNodes` and throwing a
 * `SessionCycleViolation` if any non-trivial SCC remains. In practice
 * this never fires for MCP-built sessions; it's a tripwire for
 * everything else.
 */

import type { SessionState, ExprNode } from '../../session.js'
import { tarjanSCC } from '../../compiler.js'
import { parseWireKey } from '../branded_names.js'

export class SessionCycleViolation extends Error {
  constructor(public readonly cycles: ReadonlyArray<ReadonlyArray<string>>) {
    super(formatMessage(cycles))
    this.name = 'SessionCycleViolation'
  }
}

function formatMessage(cycles: ReadonlyArray<ReadonlyArray<string>>): string {
  const lines = cycles.map(c => `  - ${c.join(' → ')} → ${c[0]}`)
  return [
    'compileSession: session graph contains inter-instance cycles ' +
      'that don\'t pass through a session-level delay():',
    ...lines,
    '',
    'MCP wire helpers auto-wrap every wire in a unit delay, which ' +
      'breaks cycles at the session level. If you\'re building a ' +
      'SessionState programmatically (test fixture, legacy ingest), ' +
      'either route wires through `setWireExpr` from compiler/session.ts ' +
      'or explicitly wrap each back-edge in `{op:\'delay\', args:[...], init:0}`.',
  ].join('\n')
}

function collectInstanceRefs(expr: ExprNode | undefined, out: Set<string>): void {
  if (expr === undefined || expr === null) return
  if (typeof expr !== 'object') return
  if (Array.isArray(expr)) {
    for (const e of expr) collectInstanceRefs(e as ExprNode, out)
    return
  }
  const obj = expr as Record<string, unknown>
  if (obj.op === 'ref' && typeof obj.instance === 'string') {
    out.add(obj.instance)
    return
  }
  for (const v of Object.values(obj)) {
    if (typeof v === 'object' && v !== null) {
      collectInstanceRefs(v as ExprNode, out)
    }
  }
}

/** Throw `SessionCycleViolation` if `session`'s post-extraction
 *  inter-instance dep graph has any non-trivial SCC. Reuses the
 *  shared `tarjanSCC` algorithm from `compiler/compiler.ts`.
 *
 *  An SCC is non-trivial if it has more than one member, or a single
 *  member with a self-edge. */
export function assertSessionAcyclic(session: SessionState): void {
  const deps = new Map<string, Set<string>>()
  for (const name of session.instanceRegistry.keys()) {
    deps.set(name, new Set())
  }
  for (const [key, expr] of session.inputExprNodes) {
    let consumer
    try { consumer = parseWireKey(key).instance } catch { continue }
    const producers = deps.get(consumer)
    if (producers === undefined) continue
    collectInstanceRefs(expr, producers)
    // Self-edges are NOT filtered: a self-wire that bypassed
    // `setWireExpr` is a 1-sample latency contract violation we want
    // to flag, not silently allow. `computeInstanceTopoOrder` does
    // filter self-edges, but that's for codegen scheduling on the
    // post-extraction graph where self-wires have already been
    // rewritten to slot reads (no instance ref).
  }
  const sccs = tarjanSCC(deps)
  const nontrivial = sccs.filter(scc =>
    scc.length > 1 || (scc.length === 1 && deps.get(scc[0])?.has(scc[0])),
  )
  if (nontrivial.length > 0) {
    throw new SessionCycleViolation(nontrivial)
  }
}
