/**
 * specialize.ts — Phase C3: type-param substitution on resolved IR.
 *
 * Functional rewrite (no clone). Produces a fresh `ResolvedProgram`
 * per (template, type-args) pair by walking the source with
 * `mapExpr` + small helpers, substituting:
 *   - `TypeParamRef.idx` (in expression position) → integer literal
 *   - `ShapeDim` that's a `TypeParamDecl` → integer
 *
 * The root program's `typeParams` list is emptied (no params remain
 * after substitution). Nested programs in `programRegistry` are
 * shared by reference — they're separate compilation units with
 * their own typeParams scopes; this caller isn't specializing them.
 * Sum/struct/alias type defs and port-type aliases pass through
 * shared (they carry no per-specialization data; sum_lower compares
 * variants by `===`).
 *
 * Purity: this function does not consult or modify any cache. The
 * cache lives in the loader (Phase C7) — the call site is responsible
 * for memoizing on the (template, args) pair.
 *
 * `InstanceDecl.typeArgs[i].value` is currently typed as `number`
 * (parser only admits integer literals), so no substitution is needed
 * there. The value passes through unchanged.
 */

import type {
  ResolvedProgram, ResolvedExpr,
  TypeParamDecl, ShapeDim, PortType,
  InputDecl, OutputDecl, BodyDecl, BodyAssign, OutputAssign,
  RegDecl, InstanceDecl,
  TypeParamIdx,
} from './nodes.js'
import { typeParamIdx } from './nodes.js'
import { mkProgram } from './decl_tables.js'
import { mapExpr, mapPortType, NoRewrite, type ExprRewrite } from './recursion.js'

export function specializeProgram(
  prog: ResolvedProgram,
  typeArgs: ReadonlyMap<TypeParamDecl, number>,
): ResolvedProgram {
  const subst = buildSubst(prog, typeArgs)
  // Short-circuit: a non-generic program with no args to substitute
  // is already "specialized." Return the input by identity. This keeps
  // the stratum a no-op for the common stdlib case where every program
  // is non-generic, which the strata orchestrator relies on.
  if (subst.size === 0) return prog

  // Idx-keyed mirror of subst for TypeParamRef lookup (refs carry
  // TypeParamIdx into the source program's typeParams[]).
  const byIdx = new Map<TypeParamIdx, number>()
  for (let i = 0; i < prog.typeParams.length; i++) {
    const v = subst.get(prog.typeParams[i])
    if (v !== undefined) byIdx.set(typeParamIdx(i), v)
  }

  const exprRewrite: ExprRewrite = e => {
    if (typeof e === 'object' && !Array.isArray(e) && 'op' in e && e.op === 'typeParamRef') {
      const v = byIdx.get(e.idx)
      if (v !== undefined) return v
    }
    return NoRewrite
  }
  const rewriteExpr = (e: ResolvedExpr) => mapExpr(e, { expr: exprRewrite })

  const shapeDim = (d: ShapeDim): ShapeDim => {
    if (typeof d === 'number') return d
    const v = subst.get(d)
    return v !== undefined ? v : d
  }

  const portType = (pt: PortType) => mapPortType(pt, shapeDim)

  const mapInputDecl = (i: InputDecl): InputDecl => {
    const fresh: InputDecl = { op: 'inputDecl', name: i.name }
    if (i.type    !== undefined) fresh.type    = portType(i.type)
    if (i.default !== undefined) fresh.default = rewriteExpr(i.default)
    return fresh
  }
  const mapOutputDecl = (o: OutputDecl): OutputDecl => {
    const fresh: OutputDecl = { op: 'outputDecl', name: o.name }
    if (o.type !== undefined) fresh.type = portType(o.type)
    return fresh
  }

  const mapDecl = (d: BodyDecl): BodyDecl => {
    switch (d.op) {
      case 'regDecl': {
        const fresh: RegDecl = {
          op: 'regDecl',
          name: d.name,
          init: rewriteExpr(d.init),
        }
        if (d.type !== undefined) fresh.type = d.type   // ScalarKind | AliasTypeDef (shared)
        if (d._liftedFrom !== undefined) fresh._liftedFrom = d._liftedFrom
        if (d.update !== undefined) fresh.update = rewriteExpr(d.update)
        return fresh
      }
      case 'paramDecl':
        // Session-scoped by name; preserved by identity (the materializer's
        // paramHandles table keys on this object).
        return d
      case 'instanceDecl': {
        const fresh: InstanceDecl = {
          op: 'instanceDecl',
          name: d.name,
          typeKey: d.typeKey,
          typeArgs: d.typeArgs.map(a => ({ param: a.param, value: a.value })),
          inputs:   d.inputs.map(i => ({ port: i.port, value: rewriteExpr(i.value) })),
        }
        return fresh
      }
      case 'programDecl':
        // Nested program decls aren't specialized (they have their own
        // typeParams scope). Pass through; their `program` field is a
        // separate ResolvedProgram value, shared by reference.
        return d
    }
  }

  const mapAssign = (a: BodyAssign): BodyAssign => {
    const fresh: OutputAssign = {
      op: 'outputAssign',
      target: a.target,
      expr: rewriteExpr(a.expr),
    }
    return fresh
  }

  return mkProgram({
    name: prog.name,
    typeParams: [],   // emptied: every reference to a typeParam has been substituted
    ports: {
      inputs:   prog.ports.inputs.map(mapInputDecl),
      outputs:  prog.ports.outputs.map(mapOutputDecl),
      typeDefs: prog.ports.typeDefs,
    },
    body: {
      op: 'block',
      decls:   prog.body.decls.map(mapDecl),
      assigns: prog.body.assigns.map(mapAssign),
    },
    binderCount: prog.binderCount,
    programRegistry: prog.programRegistry,   // sub-programs shared
  })
}

/**
 * Validate `typeArgs` against `prog.typeParams` and fill in defaults
 * for any param the caller didn't supply. Throws a clear error for:
 *   - extra args (a key in `typeArgs` that's not a declared type-param)
 *   - missing required args (a declared type-param with no `default`
 *     and no entry in `typeArgs`)
 *   - non-integer values
 */
function buildSubst(
  prog: ResolvedProgram,
  typeArgs: ReadonlyMap<TypeParamDecl, number>,
): ReadonlyMap<TypeParamDecl, number> {
  const declared = new Set(prog.typeParams)
  for (const param of typeArgs.keys()) {
    if (!declared.has(param)) {
      throw new Error(
        `specializeProgram('${prog.name}'): type-arg '${param.name}' is not a declared ` +
        `type-param (have: ${declaredNames(prog) || '(none)'})`,
      )
    }
  }
  const subst = new Map<TypeParamDecl, number>()
  for (const param of prog.typeParams) {
    if (typeArgs.has(param)) {
      const v = typeArgs.get(param) as number
      if (!Number.isInteger(v)) {
        throw new Error(
          `specializeProgram('${prog.name}'): type-arg '${param.name}' must be an integer, got ${v}`,
        )
      }
      subst.set(param, v)
    } else if (param.default !== undefined) {
      subst.set(param, param.default)
    } else {
      throw new Error(
        `specializeProgram('${prog.name}'): missing required type-arg '${param.name}' (no default)`,
      )
    }
  }
  return subst
}

function declaredNames(prog: ResolvedProgram): string {
  return prog.typeParams.map(p => p.name).join(', ')
}
