/**
 * compiler/ir/compile_session.ts — Phase D D2 session emit boundary.
 *
 * `compileSession(session)` materializes a session via
 * `materializeSessionForEmit` (see `materialize_session.ts`), then runs
 * the resolved IR through `compileResolved` to produce a
 * `tropical_plan_4` plan ready for the JIT.
 *
 * This module is the JIT-side bookend; the interpreter pulls
 * `materializeSessionToResolvedIR` directly from `materialize_session.ts`
 * without dragging in `FlatPlan` / `compileResolved`.
 */

import type { ParamDecl } from './nodes.js'
import type { SessionState } from '../session.js'
import type { FlatPlan } from '../flat_plan'
import { compileResolved } from './compile_resolved.js'
import { materializeSessionForEmit } from './materialize_session.js'

/** Default session compile path (M8: slot mode is the default).
 *
 *  Delegates to `compileSessionSlotted`, which produces a FlatPlan with
 *  slot allocation metadata populated from session.outputSlotRegistry /
 *  paramSlotRegistry. The instruction stream is byte-equal to the
 *  legacy path under the M4–M7 scope: slot fields are honored by the
 *  engine for control-plane writes (set_slot) but the JIT's instruction
 *  emission still uses legacy `param` / `input` operands, not `slot`
 *  operands.
 *
 *  Tests that pin the legacy plan shape exactly (golden fixtures)
 *  should call `compileSessionLegacy` directly.
 *
 *  Future state (post-M8): a real `compileSessionSlotted` rewrite that
 *  emits `slot` operands and `WriteSlot` instructions per the plan
 *  doc. The engine and equivalence harness are ready (M5–M7); the
 *  remaining work is per-instance compileResolved + operand remapping
 *  in compile_session_slotted.ts. */
export function compileSession(session: SessionState): FlatPlan {
  // Lazy import to avoid a circular import (compile_session_slotted
  // imports this file for compileSessionLegacy).
  const { compileSessionSlotted } = require('./compile_session_slotted.js') as
    typeof import('./compile_session_slotted.js')
  return compileSessionSlotted(session)
}

/** Legacy compile path — single-kernel, instances inlined, no slot metadata.
 *  Exported for tests that pin the legacy plan shape exactly (golden
 *  fixtures) and as the base path that compileSessionSlotted wraps. */
export function compileSessionLegacy(session: SessionState): FlatPlan {
  const { lowered, paramDecls } = materializeSessionForEmit(session)
  // Build paramHandles from the materializer's ParamDecls. Each ParamDecl
  // is keyed by name; the session's paramRegistry / triggerRegistry give
  // us the FFI handle for it. Decls without a live registry entry get
  // skipped — emit_resolved emits const 0 in that case.
  const paramHandles = new Map<ParamDecl, { ptr: string }>()
  for (const [name, decl] of paramDecls) {
    const reg = decl.kind === 'trigger' ? session.triggerRegistry : session.paramRegistry
    const live = reg.get(name)
    if (live !== undefined && (live as { _h?: unknown })._h !== undefined) {
      paramHandles.set(decl, { ptr: String((live as { _h: unknown })._h) })
    }
  }
  return compileResolved(lowered, { paramHandles })
}

export type { Instance } from '../program_types.js'
