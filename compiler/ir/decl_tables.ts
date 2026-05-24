/**
 * decl_tables.ts — projection of body.decls into typed flat tables.
 *
 * Every `ResolvedProgram` carries typed flat arrays for each decl kind
 * (`regs`, `params`, `instances`) projected from the heterogeneous
 * `body.decls: BodyDecl[]`. Position in the projected array IS the
 * decl's identity — the de Bruijn levels migration (planned next
 * phase) uses these positions as the indices that replace the
 * pointer-based ref types.
 *
 * `InputDecl` and `OutputDecl` positional identity already lives in
 * `ports.inputs` / `ports.outputs`; no top-level duplicate.
 *
 * Decls in the tables are the SAME OBJECTS as decls in `body.decls`
 * (no clone). Pointer identity is preserved during the transition.
 *
 * Every constructor of a `ResolvedProgram` must route through
 * `withDeclTables` so the tables stay in sync with `body.decls`. The
 * type system enforces this via the required `regs` / `params` /
 * `instances` fields on the `ResolvedProgram` interface.
 */

import type {
  ResolvedProgram, ResolvedProgramPorts, ResolvedBlock,
  RegDecl, ParamDecl, InstanceDecl, BodyDecl,
  ProgramKey,
} from './nodes.js'

/** Replace every entry in `prog.programRegistry` whose key appears in
 *  `byName` with the canonical version. Used by the topological
 *  typeRegistry build (`compiler/program.ts`) to ensure that by the
 *  time a program is consumed (by `materialize_session`,
 *  `partition_recursive`, etc.), every reachable program in the
 *  registry is its strata-processed canonical form rather than the
 *  raw elaborated version still hanging around from the elaborator's
 *  resolver.
 *
 *  Returns the same program (by identity) if no relinking happens,
 *  so the strata fast-path of "no-op identity" still triggers
 *  downstream. */
export function relinkProgramRegistry(
  prog: ResolvedProgram,
  byName: ReadonlyMap<string, ResolvedProgram>,
): ResolvedProgram {
  const newReg = new Map<ProgramKey, ResolvedProgram>()
  let changed = false
  for (const [key, value] of prog.programRegistry) {
    const canonical = byName.get(key)
    if (canonical && canonical !== value) {
      newReg.set(key, canonical)
      changed = true
    } else {
      newReg.set(key, value)
    }
  }
  if (!changed) return prog
  return withDeclTables({ ...prog, programRegistry: newReg })
}

/** Project a flat `BodyDecl[]` into typed tables by kind. Order
 *  within each table matches body-decl order — the same order
 *  `buildSlotMaps` uses for slot allocation. */
export function buildDeclTables(decls: readonly BodyDecl[]): {
  regs:      RegDecl[]
  params:    ParamDecl[]
  instances: InstanceDecl[]
} {
  const regs:      RegDecl[]      = []
  const params:    ParamDecl[]    = []
  const instances: InstanceDecl[] = []
  for (const d of decls) {
    switch (d.op) {
      case 'regDecl':      regs.push(d);      break
      case 'paramDecl':    params.push(d);    break
      case 'instanceDecl': instances.push(d); break
      case 'programDecl':  /* type-decl only; no runtime identity */ break
    }
  }
  return { regs, params, instances }
}

/** Look up the `ResolvedProgram` for an `InstanceDecl` via its
 *  enclosing program's `programRegistry`. Post-Phase-4b this is the
 *  ONLY way to resolve an instance's type — the `.type` pointer is
 *  gone. Throws if the registry has no entry (a registry-build bug,
 *  not a user error). */
export function getInstanceType(
  enclosing: ResolvedProgram,
  inst: InstanceDecl,
): ResolvedProgram {
  const t = enclosing.programRegistry.get(inst.typeKey)
  if (t === undefined) {
    throw new Error(
      `getInstanceType: instance '${inst.name}' typeKey '${inst.typeKey}' ` +
      `not found in enclosing program '${enclosing.name}' registry ` +
      `(keys: ${[...enclosing.programRegistry.keys()].join(', ')}). ` +
      `This is a registry-build bug; check buildProgramRegistry call sites.`,
    )
  }
  return t
}

/** Validate that an explicitly-supplied `programRegistry` covers
 *  every `InstanceDecl`'s `typeKey`. Construction sites (elaborator,
 *  clone, materialize_session) build the registry as they build the
 *  program; this just confirms they didn't miss anything. */
export function validateProgramRegistry(
  instances: readonly InstanceDecl[],
  registry: ReadonlyMap<ProgramKey, ResolvedProgram>,
): void {
  for (const inst of instances) {
    if (!registry.has(inst.typeKey)) {
      throw new Error(
        `validateProgramRegistry: instance '${inst.name}' typeKey '${inst.typeKey}' ` +
        `is not in the supplied program registry ` +
        `(keys present: ${[...registry.keys()].join(', ') || '(empty)'}). ` +
        `Construction site must add the target program to the registry before mkProgram/withDeclTables.`,
      )
    }
  }
}

/** Construct a `ResolvedProgram` from its constituent parts, projecting
 *  the decl tables from `body.decls` in one step. THE only canonical
 *  way to build a `ResolvedProgram` — every constructor site (elaborator,
 *  clone, strata passes, materialize_session) routes through here so the
 *  tables can't drift from `body.decls`. */
export function mkProgram(args: {
  name: string
  typeParams: ResolvedProgram['typeParams']
  ports: ResolvedProgramPorts
  body: ResolvedBlock
  binderCount: number
  /** Optional. Defaults to an empty map; programs with no `InstanceDecl`s
   *  in `body.decls` get away with the default. Callers whose body
   *  contains instances MUST supply a registry covering every
   *  `instance.typeKey` — `validateProgramRegistry` throws otherwise. */
  programRegistry?: ReadonlyMap<ProgramKey, ResolvedProgram>
}): ResolvedProgram {
  const tables = buildDeclTables(args.body.decls)
  const registry = args.programRegistry ?? new Map<ProgramKey, ResolvedProgram>()
  validateProgramRegistry(tables.instances, registry)
  return {
    op: 'program',
    name:        args.name,
    typeParams:  args.typeParams,
    ports:       args.ports,
    body:        args.body,
    binderCount: args.binderCount,
    ...tables,
    programRegistry: registry,
  }
}

/** Lift an existing `ResolvedProgram`-shaped value into a fully-formed
 *  `ResolvedProgram` by (re)projecting the decl tables from
 *  `body.decls`. Useful for spread-update sites that change `body` and
 *  need the tables refreshed:
 *
 *      return withDeclTables({ ...prog, body: newBody })
 *
 *  Accepts a structurally-incomplete program (tables possibly stale or
 *  missing) and returns a structurally-complete one. The decl objects
 *  themselves are reused (no clone); only the tables are rebuilt. */
export function withDeclTables(
  prog: Omit<ResolvedProgram, 'regs' | 'params' | 'instances'> &
        Partial<Pick<ResolvedProgram, 'regs' | 'params' | 'instances'>>,
): ResolvedProgram {
  const tables = buildDeclTables(prog.body.decls)
  const registry = prog.programRegistry ?? new Map<ProgramKey, ResolvedProgram>()
  validateProgramRegistry(tables.instances, registry)
  return {
    op:          'program',
    name:        prog.name,
    typeParams:  prog.typeParams,
    ports:       prog.ports,
    body:        prog.body,
    binderCount: prog.binderCount,
    ...tables,
    programRegistry: registry,
  }
}
