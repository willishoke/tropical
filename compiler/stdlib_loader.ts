/**
 * stdlib_loader.ts — pure (no-fs) stdlib registration.
 *
 * Accepts a map of raw stdlib JSON payloads keyed by program name and
 * wires them into a session's type registry. Used by the browser bundle
 * (stdlib_bundled.ts) where filesystem access is unavailable; the
 * disk-reading entry (`loadStdlib` in `program.ts`) bypasses this and
 * parses .trop sources directly.
 *
 * Pipeline: raw v2 JSON → raise → ParsedProgram → elaborate →
 * ResolvedProgram → strataPipeline → loadProgramDefFromResolved.
 */

import type { SessionState } from './session.js'
import { normalizeProgramFile } from './session.js'
import type { Compiled } from './program_types.js'
import { elaborate, type ExternalProgramResolver } from './ir/elaborator.js'
import { programTypeFromResolved } from './ir/strata.js'
import type { ResolvedProgram } from './ir/nodes.js'
import { raiseProgram } from './parse/raise.js'
import { parseProgram as parseTropicalProgram } from './parse/declarations.js'

type StdlibTarget =
  | Map<string, Compiled>
  | Pick<
      SessionState,
      | 'typeRegistry'
      | 'instanceRegistry'
      | 'paramRegistry'
      | 'specializationCache'
      | 'genericTemplatesResolved'
      | 'resolvedRegistry'
    > & Partial<Pick<SessionState, 'typeResolver' | 'inlineNested'>>

function toSession(target: StdlibTarget) {
  if (target instanceof Map) {
    return {
      typeRegistry: target,
      instanceRegistry: new Map(),
      paramRegistry: new Map(),
      specializationCache: new Map(),
      genericTemplatesResolved: new Map<string, ResolvedProgram>(),
      resolvedRegistry: new Map<string, ResolvedProgram>(),
    } as Pick<
      SessionState,
      | 'typeRegistry'
      | 'instanceRegistry'
      | 'paramRegistry'
      | 'specializationCache'
      | 'genericTemplatesResolved'
      | 'resolvedRegistry'
    > &
      Partial<Pick<SessionState, 'typeResolver'>>
  }
  return target
}

/**
 * Register stdlib types from a pre-loaded map of raw v2 JSON payloads.
 * Keys are program names; values are tropical_program_2 JSON.
 *
 * Each payload is raised back to `ParsedProgram` (raise is the JSON →
 * parser-shape bridge), elaborated against a fixed-point sibling
 * resolver, then either stashed in `genericTemplatesResolved` (generic)
 * or compiled through the strata pipeline (concrete).
 *
 * Known limitation: `raiseProgram` drops match-arm payload field labels
 * (substitutes `_unknown` placeholders) because the legacy
 * tropical_program_2 schema doesn't carry them. Programs with sum-type
 * pattern matching (e.g. TriggerRamp, EnvExpDecay) therefore fail to
 * elaborate via this path. The disk-loading `loadStdlib` in `program.ts`
 * bypasses raise — it parses .trop sources directly — and is the
 * supported entry. The bundled-stdlib path here is left in place for
 * environments without filesystem access; under sum-using bundled
 * payloads it raises an explicit error rather than silently producing
 * a broken registry.
 */
export function loadStdlibFromMap(
  target: StdlibTarget,
  rawByName: Map<string, unknown> | Record<string, unknown>,
): void {
  const session = toSession(target)
  const index: Map<string, unknown> =
    rawByName instanceof Map ? rawByName : new Map(Object.entries(rawByName))

  const parsedByName = new Map<string, ReturnType<typeof raiseProgram>>()
  for (const [name, raw] of index) {
    const { node } = normalizeProgramFile(raw as { schema?: string; [k: string]: unknown })
    parsedByName.set(name, raiseProgram(node))
  }

  const localResolved = new Map<string, ResolvedProgram>()
  const externalResolver: ExternalProgramResolver = name => localResolved.get(name)

  const remaining = new Map(parsedByName)
  let progress = true
  while (progress && remaining.size > 0) {
    progress = false
    for (const [name, parsed] of remaining) {
      try {
        const resolved = elaborate(parsed, externalResolver)
        localResolved.set(name, resolved)
        remaining.delete(name)
        progress = true
      } catch {
        // Sibling not yet elaborated; retry next pass.
      }
    }
  }
  if (remaining.size > 0) {
    const [name, parsed] = remaining.entries().next().value as [string, ReturnType<typeof raiseProgram>]
    elaborate(parsed, externalResolver)
    throw new Error(`loadStdlibFromMap: failed to elaborate '${name}'`)
  }

  for (const [name, prog] of localResolved) {
    if (prog.typeParams.length > 0) {
      session.genericTemplatesResolved.set(name, prog)
      continue
    }
    if (session.typeRegistry.has(name)) continue
    const type = programTypeFromResolved(prog, new Map(), { inlineNested: session.inlineNested })
    session.typeRegistry.set(name, type)
    session.resolvedRegistry.set(name, prog)
  }

  if (!session.typeResolver) {
    session.typeResolver = (n: string): Compiled | undefined => {
      return session.typeRegistry.get(n)
    }
  }
}

/**
 * Register stdlib types from a pre-loaded map of raw .trop source strings.
 * Keys are program names; values are the source text (the inside of the
 * single `tropical` code block, post-markdown extraction).
 *
 * Used by the browser bundle (`compiler/stdlib_bundled.ts`) where
 * filesystem access is unavailable. Internally identical to the disk-side
 * `loadStdlib`: parse → elaborate (fixed-point sibling resolver) → strata.
 */
export function loadStdlibFromSources(
  target: StdlibTarget,
  sourcesByName: Map<string, string> | Record<string, string>,
): void {
  const session = toSession(target)
  const index: Map<string, string> =
    sourcesByName instanceof Map ? sourcesByName : new Map(Object.entries(sourcesByName))

  const parsedByName = new Map<string, ReturnType<typeof parseTropicalProgram>>()
  for (const [name, src] of index) {
    parsedByName.set(name, parseTropicalProgram(src))
  }

  const localResolved = new Map<string, ResolvedProgram>()
  const externalResolver: ExternalProgramResolver = name => localResolved.get(name)

  const remaining = new Map(parsedByName)
  let progress = true
  while (progress && remaining.size > 0) {
    progress = false
    for (const [name, parsed] of remaining) {
      try {
        const resolved = elaborate(parsed, externalResolver)
        localResolved.set(name, resolved)
        remaining.delete(name)
        progress = true
      } catch {
        // Sibling not yet elaborated; retry next pass.
      }
    }
  }
  if (remaining.size > 0) {
    const [name, parsed] = remaining.entries().next().value as [string, ReturnType<typeof parseTropicalProgram>]
    elaborate(parsed, externalResolver)
    throw new Error(`loadStdlibFromSources: failed to elaborate '${name}'`)
  }

  for (const [name, prog] of localResolved) {
    if (prog.typeParams.length > 0) {
      session.genericTemplatesResolved.set(name, prog)
      continue
    }
    if (session.typeRegistry.has(name)) continue
    const type = programTypeFromResolved(prog, new Map(), { inlineNested: session.inlineNested })
    session.typeRegistry.set(name, type)
    session.resolvedRegistry.set(name, prog)
  }

  if (!session.typeResolver) {
    session.typeResolver = (n: string): Compiled | undefined => {
      return session.typeRegistry.get(n)
    }
  }
}
