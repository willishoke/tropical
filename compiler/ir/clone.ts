/**
 * clone.ts — functional copy of a `ResolvedProgram`.
 *
 * Post-Phase-4b (issue #156): much of the original cloner's machinery
 * collapsed. Indexed refs (RegRef, ParamRef, InputRef, NestedOut,
 * BindingRef, TypeParamRef) carry integers, not decl pointers, so
 * the old "Map<old, new> per decl kind" dedup tables that preserved
 * pointer identity within a single program no longer earn their
 * keep — the index already IS the identity, and shifting (regOffset,
 * paramOffset, instanceOffset, binderOffset) handles cross-program
 * lifting. What remains:
 *
 *   - `nestedPrograms` memo: sub-programs are deep-cloned so passes
 *     like array_lower's M11 fractal mode can mutate them in place
 *     without affecting the source. Two instances of the same source
 *     sub-program share one cloned program via this memo.
 *   - `typeParams` dedup: a `TypeParamDecl` can appear both in
 *     `prog.typeParams[]` and as the dim of a `ShapeDim`. Both
 *     occurrences must clone to the same fresh decl so ShapeDim
 *     resolves consistently.
 *   - Three substitution maps used by specialize / inline_instances:
 *     `subst` / `substByDecl` (type-param substitution) and
 *     `inputSubst` (input → outer-expression).
 *
 * Sharing decisions for SumTypeDef / SumVariant / StructTypeDef /
 * StructField / AliasTypeDef are unchanged: SHARED across clones,
 * because downstream passes (sum_lower) compare variants by `===`.
 *
 * Used by: specialize (Phase C3), inline_instances (Phase C5),
 * identity_elim and array_lower (defensive — they mutate, the clone
 * insulates the source).
 */

import type {
  ResolvedProgram,
  ResolvedExpr, ResolvedExprOp,
  InputDecl, OutputDecl, TypeParamDecl,
  RegDecl, ParamDecl, InstanceDecl, ProgramDecl, BodyDecl,
  BodyAssign, OutputAssign,
  PortType, ShapeDim,
  BinderDecl,
  Tag, Match, MatchArm,
  Let,
  Fold, Scan, Generate, Iterate, Chain, Map2, ZipWith,
  InputIdx, TypeParamIdx,
  RegIdx as RegIdx_t, ParamIdx as ParamIdx_t, InstanceIdx as InstanceIdx_t,
  BinderIdx as BinderIdx_t,
  ProgramKey,
} from './nodes.js'
import { typeParamIdx, binderIdx } from './nodes.js'

// ─────────────────────────────────────────────────────────────
// Clone context — substitution maps and idx offsets
// ─────────────────────────────────────────────────────────────

interface CloneTable {
  /** Dedup for TypeParamDecl: a single source decl can be reached both
   *  via `prog.typeParams[]` and via a `ShapeDim` reference inside a
   *  port type. Both paths must yield the same cloned object so
   *  downstream consumers see consistent identity. */
  typeParams: Map<TypeParamDecl, TypeParamDecl>
  /** Sub-program memo. Two `InstanceDecl`s of the same source
   *  sub-program share one cloned program. */
  nestedPrograms: Map<ResolvedProgram, ResolvedProgram>
  /** Specialize: TypeParamRef whose idx is a key here is rewritten to
   *  the corresponding integer literal. Refs whose idx is NOT in subst
   *  belong to a nested program's own type-params (caller isn't
   *  specializing them) and pass through. Keyed by TypeParamIdx
   *  (position in the root program's typeParams). */
  subst?:        ReadonlyMap<TypeParamIdx, number>
  /** Mirror of `subst` keyed by decl object — used for ShapeDim
   *  substitution where the dim is `number | TypeParamDecl` (no idx). */
  substByDecl?:  ReadonlyMap<TypeParamDecl, number>
  /** The program whose `typeParams` should be emptied in the clone
   *  (the specialization root). Nested programs retain their own
   *  typeParams. */
  rootProgram?: ResolvedProgram
  /** Inline-substitution: an InputRef whose idx is a key here is
   *  replaced by the corresponding ResolvedExpr from the outer site.
   *  The substituted expression is already in the outer's namespace
   *  and passes through by reference (preserves DAG sharing). */
  inputSubst?: ReadonlyMap<InputIdx, ResolvedExpr>
  /** Idx offsets applied to surviving indexed refs in the cloned
   *  program. Used when the clone will be lifted into an outer
   *  program: refs inside the cloned body need to point at the
   *  lifted positions, not the cloned-local positions. */
  regOffset?:      number
  paramOffset?:    number
  instanceOffset?: number
  binderOffset?:   number
}

function emptyTable(): CloneTable {
  return {
    typeParams:     new Map(),
    nestedPrograms: new Map(),
  }
}

// ─────────────────────────────────────────────────────────────
// Public entry points
// ─────────────────────────────────────────────────────────────

export function cloneResolvedProgram(prog: ResolvedProgram): ResolvedProgram {
  return cloneProgram(prog, emptyTable())
}

/**
 * Clone-and-substitute. Used by Phase C3 (specialize): produces a fresh
 * `ResolvedProgram` with every `TypeParamRef` whose decl is a key in
 * `subst` replaced by the corresponding integer literal, and every
 * `ShapeDim` that's a `TypeParamDecl` likewise. The root program's
 * `typeParams` list is emptied; nested programs retain their own
 * type-params (this caller is only specializing the root).
 */
export function cloneWithSubst(
  prog: ResolvedProgram,
  subst: ReadonlyMap<TypeParamDecl, number>,
): ResolvedProgram {
  const t = emptyTable()
  const byIdx = new Map<TypeParamIdx, number>()
  for (let i = 0; i < prog.typeParams.length; i++) {
    const v = subst.get(prog.typeParams[i])
    if (v !== undefined) byIdx.set(typeParamIdx(i), v)
  }
  t.subst       = byIdx
  t.substByDecl = subst
  t.rootProgram = prog
  return cloneProgram(prog, t)
}

/**
 * Clone-and-substitute-inputs. Used by Phase C5 (inlineInstances):
 * produces a fresh `ResolvedProgram` with every `InputRef` whose idx
 * is a key in `inputSubst` replaced by the wired-in expression from
 * the outer site. The substituted expressions are in the outer's
 * namespace and pass through by reference — they are NOT re-cloned,
 * preserving DAG sharing and avoiding namespace confusion.
 */
export function cloneWithInputSubst(
  prog: ResolvedProgram,
  inputSubst: ReadonlyMap<InputIdx, ResolvedExpr>,
  shifts?: { regOffset?: number; paramOffset?: number; instanceOffset?: number; binderOffset?: number },
): ResolvedProgram {
  const t = emptyTable()
  t.inputSubst = inputSubst
  if (shifts?.regOffset      !== undefined) t.regOffset      = shifts.regOffset
  if (shifts?.paramOffset    !== undefined) t.paramOffset    = shifts.paramOffset
  if (shifts?.instanceOffset !== undefined) t.instanceOffset = shifts.instanceOffset
  if (shifts?.binderOffset   !== undefined) t.binderOffset   = shifts.binderOffset
  return cloneProgram(prog, t)
}

// ─────────────────────────────────────────────────────────────
// Program-level clone
// ─────────────────────────────────────────────────────────────

function cloneProgram(prog: ResolvedProgram, t: CloneTable): ResolvedProgram {
  const cached = t.nestedPrograms.get(prog)
  if (cached) return cached

  // Build a shell with placeholders so a self-referential nested
  // program (rare but admissible) can find this clone via the memo
  // during the recursive walk below. Fields are filled in immediately
  // after.
  const shell: ResolvedProgram = {
    op: 'program',
    name: prog.name,
    typeParams: [],
    ports: { inputs: [], outputs: [], typeDefs: prog.ports.typeDefs },
    body: { op: 'block', decls: [], assigns: [] },
    regs:      [],
    params:    [],
    instances: [],
    binderCount: prog.binderCount + (t.binderOffset ?? 0),
    // Populated below from the source's registry, with each
    // sub-program deep-cloned via the nestedPrograms memo.
    programRegistry: new Map(),
  }
  t.nestedPrograms.set(prog, shell)

  // Type-params: emptied on the specialization root (refs are
  // substituted by integer literals); preserved on nested programs.
  shell.typeParams = t.rootProgram === prog
    ? []
    : prog.typeParams.map(tp => cloneTypeParamDecl(tp, t))

  // Inputs/outputs: cloned as fresh objects (decls are mutable
  // shapes). Port types reference TypeParamDecl via ShapeDim, which
  // goes through the typeParams dedup.
  shell.ports = {
    inputs:  prog.ports.inputs.map(i => {
      const fresh: InputDecl = { op: 'inputDecl', name: i.name }
      if (i.type    !== undefined) fresh.type    = clonePortType(i.type, t)
      if (i.default !== undefined) fresh.default = cloneExpr(i.default, t)
      return fresh
    }),
    outputs: prog.ports.outputs.map(o => {
      const fresh: OutputDecl = { op: 'outputDecl', name: o.name }
      if (o.type !== undefined) fresh.type = clonePortType(o.type, t)
      return fresh
    }),
    typeDefs: prog.ports.typeDefs,   // shared
  }

  // Body decls: single functional pass. Indexed refs in expressions
  // carry their own idx (with optional offset); no need for the old
  // shell-then-fill cycle handling that pointer-based RegRefs required.
  shell.body.decls   = prog.body.decls.map(d => cloneBodyDecl(d, t))
  shell.body.assigns = prog.body.assigns.map(a => cloneAssign(a, t))

  // Rebuild typed tables from the cloned body.decls (matches the
  // original-order invariant that buildDeclTables enforces).
  const tablesFromDecls = projectDeclTables(shell.body.decls)
  shell.regs      = tablesFromDecls.regs
  shell.params    = tablesFromDecls.params
  shell.instances = tablesFromDecls.instances

  // Clone the source's registry into the shell's registry. Each
  // sub-program is cloned via the nestedPrograms memo; same source
  // sub-program yields the same cloned instance.
  const clonedReg = new Map<ProgramKey, ResolvedProgram>()
  for (const [key, srcSub] of prog.programRegistry) {
    clonedReg.set(key, cloneProgram(srcSub, t))
  }
  shell.programRegistry = clonedReg

  return shell
}

function projectDeclTables(decls: readonly BodyDecl[]): {
  regs: RegDecl[]; params: ParamDecl[]; instances: InstanceDecl[]
} {
  const regs:      RegDecl[]      = []
  const params:    ParamDecl[]    = []
  const instances: InstanceDecl[] = []
  for (const d of decls) {
    switch (d.op) {
      case 'regDecl':      regs.push(d);      break
      case 'paramDecl':    params.push(d);    break
      case 'instanceDecl': instances.push(d); break
      case 'programDecl':  /* type-decl only */ break
    }
  }
  return { regs, params, instances }
}

// ─────────────────────────────────────────────────────────────
// TypeParam — dedup via shared map (port-type ShapeDim cross-ref)
// ─────────────────────────────────────────────────────────────

function cloneTypeParamDecl(d: TypeParamDecl, t: CloneTable): TypeParamDecl {
  const cached = t.typeParams.get(d)
  if (cached) return cached
  const fresh: TypeParamDecl = { op: 'typeParamDecl', name: d.name }
  if (d.default !== undefined) fresh.default = d.default
  t.typeParams.set(d, fresh)
  return fresh
}

// ─────────────────────────────────────────────────────────────
// Body decls — single functional pass (no shell-then-fill)
// ─────────────────────────────────────────────────────────────

function cloneBodyDecl(d: BodyDecl, t: CloneTable): BodyDecl {
  switch (d.op) {
    case 'regDecl': {
      const fresh: RegDecl = {
        op: 'regDecl',
        name: d.name,
        init: cloneExpr(d.init, t),
      }
      if (d.type !== undefined) fresh.type = d.type   // ScalarKind | AliasTypeDef (shared)
      if (d._liftedFrom !== undefined) fresh._liftedFrom = d._liftedFrom
      if (d.update !== undefined) fresh.update = cloneExpr(d.update, t)
      return fresh
    }
    case 'paramDecl':
      // Session-scoped by name; the materializer's paramHandles table
      // keys on this decl's identity. Cloning would mint a fresh decl
      // the table doesn't know — preserve identity. (inline_instances
      // documents the same invariant for the lift path.)
      return d
    case 'instanceDecl': {
      const fresh: InstanceDecl = {
        op: 'instanceDecl',
        name: d.name,
        typeKey: d.typeKey,
        typeArgs: d.typeArgs.map(a => ({ param: a.param, value: a.value })),
        inputs:   d.inputs.map(i => ({ port: i.port, value: cloneExpr(i.value, t) })),
      }
      return fresh
    }
    case 'programDecl':
      return {
        op: 'programDecl',
        name: d.name,
        program: cloneProgram(d.program, t),
      }
  }
}

// ─────────────────────────────────────────────────────────────
// Assigns
// ─────────────────────────────────────────────────────────────

function cloneAssign(a: BodyAssign, t: CloneTable): BodyAssign {
  // OutputAssign-only. Target is OutputIdx or the `dac` sentinel —
  // cloning preserves output order, so the index passes through.
  const target: typeof a.target =
    typeof a.target === 'object' && 'kind' in a.target
      ? { kind: 'dac' }
      : a.target
  const fresh: OutputAssign = {
    op: 'outputAssign',
    target,
    expr: cloneExpr(a.expr, t),
  }
  return fresh
}

// ─────────────────────────────────────────────────────────────
// Port types and shape dims
// ─────────────────────────────────────────────────────────────

function clonePortType(pt: PortType, t: CloneTable): PortType {
  switch (pt.kind) {
    case 'scalar': return { kind: 'scalar', scalar: pt.scalar }
    case 'alias':  return { kind: 'alias', alias: pt.alias }   // shared
    case 'array':  return {
      kind: 'array',
      element: pt.element,    // ScalarKind | AliasTypeDef (shared)
      shape: pt.shape.map(d => cloneShapeDim(d, t)),
    }
  }
}

function cloneShapeDim(d: ShapeDim, t: CloneTable): ShapeDim {
  if (typeof d === 'number') return d
  if (t.substByDecl !== undefined) {
    const v = t.substByDecl.get(d)
    if (v !== undefined) return v
  }
  return cloneTypeParamDecl(d, t)
}

// ─────────────────────────────────────────────────────────────
// Binder — no dedup; idx + offset is the identity
// ─────────────────────────────────────────────────────────────

function cloneBinder(b: BinderDecl, t: CloneTable): BinderDecl {
  const shifted = (b.idx as number) + (t.binderOffset ?? 0)
  return { op: 'binderDecl', name: b.name, idx: binderIdx(shifted) }
}

// ─────────────────────────────────────────────────────────────
// Expressions
// ─────────────────────────────────────────────────────────────

function cloneExpr(e: ResolvedExpr, t: CloneTable): ResolvedExpr {
  if (typeof e === 'number' || typeof e === 'boolean') return e
  if (Array.isArray(e)) return e.map(x => cloneExpr(x, t))
  // Specialize-time substitution.
  if (t.subst !== undefined && e.op === 'typeParamRef') {
    const v = t.subst.get(e.idx)
    if (v !== undefined) return v
  }
  // Inline-time input substitution.
  if (t.inputSubst !== undefined && e.op === 'inputRef') {
    const v = t.inputSubst.get(e.idx)
    if (v !== undefined) return v
  }
  return cloneOpNode(e, t)
}

function cloneOpNode(node: ResolvedExprOp, t: CloneTable): ResolvedExprOp {
  switch (node.op) {
    // Indexed refs with optional offset shifting for lift scenarios.
    case 'inputRef':     return { op: 'inputRef',     idx: node.idx }
    case 'regRef':       return {
      op: 'regRef',
      idx: (t.regOffset !== undefined
        ? (node.idx as number) + t.regOffset
        : node.idx) as RegIdx_t,
    }
    case 'paramRef':     return {
      op: 'paramRef',
      idx: (t.paramOffset !== undefined
        ? (node.idx as number) + t.paramOffset
        : node.idx) as ParamIdx_t,
    }
    case 'typeParamRef': return { op: 'typeParamRef', idx: node.idx }
    case 'nestedOut':    return {
      op: 'nestedOut',
      instance: (t.instanceOffset !== undefined
        ? (node.instance as number) + t.instanceOffset
        : node.instance) as InstanceIdx_t,
      output: node.output,
    }
    case 'bindingRef':   return {
      op: 'bindingRef',
      idx: (t.binderOffset !== undefined
        ? (node.idx as number) + t.binderOffset
        : node.idx) as BinderIdx_t,
    }
    case 'sampleRate':  return { op: 'sampleRate' }
    case 'sampleIndex': return { op: 'sampleIndex' }

    // ADT — variant shared, payload/arms cloned.
    case 'tag': {
      const fresh: Tag = {
        op: 'tag',
        variant: node.variant,
        payload: node.payload.map(p => ({ field: p.field, value: cloneExpr(p.value, t) })),
      }
      return fresh
    }
    case 'match': {
      const arms: MatchArm[] = node.arms.map(arm => ({
        variant: arm.variant,
        binders: arm.binders.map(b => cloneBinder(b, t)),
        body: cloneExpr(arm.body, t),
      }))
      const fresh: Match = {
        op: 'match',
        type: node.type,
        scrutinee: cloneExpr(node.scrutinee, t),
        arms,
      }
      return fresh
    }

    // Combinators.
    case 'let': {
      const fresh: Let = {
        op: 'let',
        binders: node.binders.map(b => ({
          binder: cloneBinder(b.binder, t),
          value:  cloneExpr(b.value, t),
        })),
        in: cloneExpr(node.in, t),
      }
      return fresh
    }
    case 'fold': {
      const fresh: Fold = {
        op: 'fold',
        over: cloneExpr(node.over, t),
        init: cloneExpr(node.init, t),
        acc: cloneBinder(node.acc, t),
        elem: cloneBinder(node.elem, t),
        body: cloneExpr(node.body, t),
      }
      return fresh
    }
    case 'scan': {
      const fresh: Scan = {
        op: 'scan',
        over: cloneExpr(node.over, t),
        init: cloneExpr(node.init, t),
        acc: cloneBinder(node.acc, t),
        elem: cloneBinder(node.elem, t),
        body: cloneExpr(node.body, t),
      }
      return fresh
    }
    case 'generate': {
      const fresh: Generate = {
        op: 'generate',
        count: cloneExpr(node.count, t),
        iter: cloneBinder(node.iter, t),
        body: cloneExpr(node.body, t),
      }
      return fresh
    }
    case 'iterate': {
      const fresh: Iterate = {
        op: 'iterate',
        count: cloneExpr(node.count, t),
        init: cloneExpr(node.init, t),
        iter: cloneBinder(node.iter, t),
        body: cloneExpr(node.body, t),
      }
      return fresh
    }
    case 'chain': {
      const fresh: Chain = {
        op: 'chain',
        count: cloneExpr(node.count, t),
        init: cloneExpr(node.init, t),
        iter: cloneBinder(node.iter, t),
        body: cloneExpr(node.body, t),
      }
      return fresh
    }
    case 'map2': {
      const fresh: Map2 = {
        op: 'map2',
        over: cloneExpr(node.over, t),
        elem: cloneBinder(node.elem, t),
        body: cloneExpr(node.body, t),
      }
      return fresh
    }
    case 'zipWith': {
      const fresh: ZipWith = {
        op: 'zipWith',
        a: cloneExpr(node.a, t),
        b: cloneExpr(node.b, t),
        x: cloneBinder(node.x, t),
        y: cloneBinder(node.y, t),
        body: cloneExpr(node.body, t),
      }
      return fresh
    }

    // Operators.
    case 'add': case 'sub': case 'mul': case 'div': case 'mod':
    case 'lt': case 'lte': case 'gt': case 'gte': case 'eq': case 'neq':
    case 'and': case 'or':
    case 'bitAnd': case 'bitOr': case 'bitXor': case 'lshift': case 'rshift':
    case 'floorDiv': case 'ldexp': {
      return {
        op: node.op,
        args: [cloneExpr(node.args[0], t), cloneExpr(node.args[1], t)],
      }
    }
    case 'neg': case 'not': case 'bitNot':
    case 'sqrt': case 'abs': case 'floor': case 'ceil': case 'round':
    case 'floatExponent': case 'toInt': case 'toBool': case 'toFloat': {
      return { op: node.op, args: [cloneExpr(node.args[0], t)] }
    }
    case 'clamp': case 'select': case 'arraySet': {
      return {
        op: node.op,
        args: [
          cloneExpr(node.args[0], t),
          cloneExpr(node.args[1], t),
          cloneExpr(node.args[2], t),
        ],
      }
    }
    case 'index': {
      return {
        op: 'index',
        args: [cloneExpr(node.args[0], t), cloneExpr(node.args[1], t)],
      }
    }
    case 'zeros': {
      return { op: 'zeros', count: cloneExpr(node.count, t) }
    }
  }
}
