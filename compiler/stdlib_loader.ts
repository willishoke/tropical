/**
 * stdlib_loader.ts — pure (no-fs) stdlib registration.
 *
 * Accepts a map of raw stdlib JSON payloads keyed by program name and
 * wires them into a session's type registry with an on-demand resolver.
 * Shared between the disk-reading loader in program.ts and the browser
 * bundle (stdlib_bundled.ts).
 */

import type { SessionState } from './session.js'
import { normalizeProgramFile } from './session.js'
import { loadProgramAsType } from './program.js'
import type { ProgramType } from './program_types.js'

type StdlibTarget =
  | Map<string, ProgramType>
  | Pick<
      SessionState,
      | 'typeRegistry'
      | 'instanceRegistry'
      | 'paramRegistry'
      | 'triggerRegistry'
      | 'specializationCache'
      | 'genericTemplates'
    >

function toSession(target: StdlibTarget) {
  if (target instanceof Map) {
    return {
      typeRegistry: target,
      instanceRegistry: new Map(),
      paramRegistry: new Map(),
      triggerRegistry: new Map(),
      specializationCache: new Map(),
      genericTemplates: new Map(),
    } as Pick<
      SessionState,
      | 'typeRegistry'
      | 'instanceRegistry'
      | 'paramRegistry'
      | 'triggerRegistry'
      | 'specializationCache'
      | 'genericTemplates'
    > &
      Partial<Pick<SessionState, 'typeResolver'>>
  }
  return target as typeof target & Partial<Pick<SessionState, 'typeResolver'>>
}

/**
 * Register stdlib types from a pre-loaded map of raw JSON payloads.
 * Keys are program names; values are the parsed JSON (either schema version).
 *
 * Types are indexed first, then loaded on demand via a resolver installed on
 * `session.typeResolver` — dependencies resolve recursively regardless of
 * insertion order. Generic templates are registered without instantiation
 * (instantiation requires type_args at use sites).
 */
export function loadStdlibFromMap(
  target: StdlibTarget,
  rawByName: Map<string, unknown> | Record<string, unknown>,
): void {
  const session = toSession(target)
  const index: Map<string, unknown> =
    rawByName instanceof Map ? rawByName : new Map(Object.entries(rawByName))

  const loading = new Set<string>()
  session.typeResolver = (name: string): ProgramType | undefined => {
    const existing = session.typeRegistry.get(name)
    if (existing) return existing
    if (session.genericTemplates.has(name)) return undefined
    if (loading.has(name)) {
      throw new Error(`Circular stdlib dependency: ${[...loading, name].join(' → ')}`)
    }
    const raw = index.get(name)
    if (raw === undefined) return undefined
    loading.add(name)
    const { node } = normalizeProgramFile(raw as { schema?: string; [k: string]: unknown })
    const type = loadProgramAsType(node, session)
    loading.delete(name)
    return type
  }

  for (const name of index.keys()) {
    if (!session.typeRegistry.has(name) && !session.genericTemplates.has(name)) {
      session.typeResolver(name)
    }
  }
}
