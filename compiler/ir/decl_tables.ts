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
} from './nodes.js'

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
}): ResolvedProgram {
  return {
    op: 'program',
    name:       args.name,
    typeParams: args.typeParams,
    ports:      args.ports,
    body:       args.body,
    ...buildDeclTables(args.body.decls),
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
  return {
    op:         'program',
    name:       prog.name,
    typeParams: prog.typeParams,
    ports:      prog.ports,
    body:       prog.body,
    ...buildDeclTables(prog.body.decls),
  }
}
