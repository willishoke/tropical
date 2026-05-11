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

export function compileSession(session: SessionState): FlatPlan {
  // Slot-mode dispatch (M4). When TROPICAL_SLOT_MODE is set, route
  // through the slot-aware compile path — same instructions, plus
  // slot allocation metadata. The env-var flag avoids dragging slot
  // mode into every test until equivalence (M7) is verified.
  // Lazy import to avoid a circular import (compile_session_slotted
  // imports this file for the legacy fallback).
  const env = process.env.TROPICAL_SLOT_MODE
  if (env !== undefined && env !== '' && env !== '0' && env !== 'false') {
    // Dynamic require keeps this branch out of the legacy-only callgraph.
    const { compileSessionSlotted } = require('./compile_session_slotted.js') as
      typeof import('./compile_session_slotted.js')
    return compileSessionSlotted(session)
  }
  return compileSessionLegacy(session)
}

/** Legacy compile path — single-kernel, instances inlined, no slot metadata.
 *  Exported so the slot-mode path can call it as a base. */
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
