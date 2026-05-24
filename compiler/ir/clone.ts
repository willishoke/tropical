/**
 * clone.ts — graph cloner for `ResolvedProgram`.
 *
 * Produces a deep copy of a `ResolvedProgram` with reference identity
 * preserved within the clone: every decl is cloned exactly once, and
 * every `*Ref.decl` field in the cloned tree points at the cloned
 * decl object.
 *
 * Why a graph cloner (not `structuredClone`):
 * - The resolved IR is a graph, not a tree. A `RegDecl`'s `init` may
 *   transitively reference its own `RegDecl` (delays + feedback);
 *   `structuredClone` doesn't preserve that kind of cyclic identity
 *   for application-level objects.
 * - Phase C3 (specialize) and Phase C5 (inlineInstances) both need
 *   to produce fresh decls per call site. Centralizing the clone
 *   discipline here keeps both sites honest.
 *
 * Sharing decisions:
 * - `SumTypeDef`, `SumVariant`, `AliasTypeDef`, `StructTypeDef`,
 *   `StructField` — SHARED (`===` preserved). They carry no
 *   per-specialization data; cloning them would break variant
 *   identity in `Match.arms[i].variant` and `Tag.variant`,
 *   which downstream passes (sum_lower) compare by `===`.
 * - Decls with pointer identity (`InputDecl`, `OutputDecl`,
 *   `TypeParamDecl`, `RegDecl`, `ParamDecl`, `InstanceDecl`,
 *   `ProgramDecl`) — CLONED, with the `Map<old, new>` dedup table
 *   ensuring each appears at most once.
 * - `BinderDecl` — cloned per-occurrence with the optional
 *   `binderOffset` applied to the idx. No dedup table needed:
 *   `BinderDecl` identity is the integer `idx`, not the object,
 *   so two clones with the same (possibly shifted) idx are
 *   semantically identical.
 *
 * Construction discipline:
 * - For every decl, the new object is inserted into the dedup table
 *   BEFORE recursing into its children. Self-referential decls
 *   (a `RegDecl.init` containing a `RegRef` to the same decl) get
 *   the cloned decl identity from the table on the recursive visit.
 *
 * Used by: Phase C3 (specialize), Phase C5 (inlineInstances).
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
} from './nodes.js'
import { typeParamIdx, binderIdx } from './nodes.js'
import { buildDeclTables, buildProgramRegistry } from './decl_tables.js'

// ─────────────────────────────────────────────────────────────
// Dedup table — Map<old, new> covers every cloned decl kind
// ─────────────────────────────────────────────────────────────

interface CloneTable {
  inputs:     Map<InputDecl, InputDecl>
  outputs:    Map<OutputDecl, OutputDecl>
  typeParams: Map<TypeParamDecl, TypeParamDecl>
  /** Unified state-bearing decls — former DelayDecls now live here too. */
  regs:       Map<RegDecl, RegDecl>
  params:     Map<ParamDecl, ParamDecl>
  instances:  Map<InstanceDecl, InstanceDecl>
  programs:   Map<ProgramDecl, ProgramDecl>
  /** Nested ResolvedPrograms (held by `InstanceDecl.type` and
   *  `ProgramDecl.program`). Memoized so two instances of the same
   *  nested program share the cloned program object. */
  nestedPrograms: Map<ResolvedProgram, ResolvedProgram>
  /** Optional substitution map for Phase C3 specialize: TypeParamRef
   *  whose idx is a key here is rewritten to the corresponding integer.
   *  Refs whose idx is NOT in subst pass through — they belong to a
   *  nested program's own type-params (caller isn't specializing them).
   *  Keyed by TypeParamIdx (position in the root program's typeParams).
   *  ShapeDim substitution uses `substByDecl` because ShapeDim carries
   *  TypeParamDecl pointers, not indices. */
  subst?:        ReadonlyMap<TypeParamIdx, number>
  /** Mirror of `subst` keyed by decl object — used for ShapeDim
   *  substitution where the dim is `number | TypeParamDecl` (no idx).
   *  Built alongside `subst` in cloneWithSubst. */
  substByDecl?:  ReadonlyMap<TypeParamDecl, number>
  /** The program whose `typeParams` should be emptied in the clone
   *  (the specialization root). Nested programs retain their own
   *  `typeParams` since this caller isn't substituting them. */
  rootProgram?: ResolvedProgram
  /** Optional substitution map for Phase C5 inlineInstances: InputRef
   *  whose idx is a key here is replaced by the corresponding
   *  ResolvedExpr (from the wired-in expression at the outer site).
   *  The substituted expression is in the *outer* program's namespace
   *  and is not cloned again — it passes through by reference, which
   *  preserves DAG sharing across multiple uses. */
  inputSubst?: ReadonlyMap<InputIdx, ResolvedExpr>
  /** Idx offsets applied to surviving indexed refs in the cloned
   *  program. After cloning, the cloned program's decls will be lifted
   *  into an outer program at known offsets; refs inside the cloned
   *  body need to point at the lifted positions, not the cloned-local
   *  positions. Applied to RegRef, ParamRef, NestedOut.instance.
   *  Zero (or unset) means no shift — useful for standalone clones
   *  that aren't being lifted. */
  regOffset?:      number
  paramOffset?:    number
  instanceOffset?: number
  /** Idx offset applied to BinderDecl.idx and BindingRef.idx
   *  throughout the cloned subtree. Like regOffset/paramOffset, used
   *  when lifting a sub-program's binders into an outer program's
   *  binder namespace during inlineInstances. */
  binderOffset?:   number
}

function emptyTable(): CloneTable {
  return {
    inputs:     new Map(),
    outputs:    new Map(),
    typeParams: new Map(),
    regs:       new Map(),
    params:     new Map(),
    instances:  new Map(),
    programs:   new Map(),
    nestedPrograms: new Map(),
  }
}

// ─────────────────────────────────────────────────────────────
// Entry point
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
  // Build the index-keyed mirror of subst for ref-position substitution.
  // ShapeDim sites still use the decl-keyed map because ShapeDim's
  // TypeParamDecl variant carries a decl pointer, not an index.
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
 * produces a fresh `ResolvedProgram` with every `InputRef` whose decl
 * is a key in `inputSubst` replaced by the wired-in expression from
 * the outer site. The substituted expressions are in the outer's
 * namespace and pass through by reference — they are NOT re-cloned,
 * preserving DAG sharing and avoiding namespace confusion.
 *
 * The caller is expected to splice the cloned program's body into
 * the outer program; the cloned `ports.inputs` and `ports.outputs`
 * become orphaned post-splice (no longer referenced from anything).
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

function cloneProgram(prog: ResolvedProgram, t: CloneTable): ResolvedProgram {
  const cached = t.nestedPrograms.get(prog)
  if (cached) return cached

  // Build the program shell first so children can find it via the
  // memo on recursion (e.g., a nested program decl whose body refers
  // back through scope to itself, though current parser disallows
  // that — defensive against future shapes).
  // Shell satisfies ResolvedProgram with empty decl tables; the tables
  // are re-derived after fill-in completes (line near return below).
  // Clone uses a shell-then-fill pattern (mutation during construction)
  // to handle cross-decl references. Phase 0 just adds the table re-
  // derivation; the structural cleanup of the shell pattern itself is
  // a later phase.
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
    // Placeholder; rebuilt from cloned instances at end of cloneProgram.
    programRegistry: new Map(),
  }
  t.nestedPrograms.set(prog, shell)

  // Type-params first — port types and decl init exprs may reference them.
  // Specialization root: drop the typeParams (substitution will replace
  // every reference to them; their decls have no purpose post-clone).
  // Nested programs retain their typeParams unchanged — they're not what
  // this caller is specializing.
  if (t.rootProgram === prog) {
    shell.typeParams = []
  } else {
    shell.typeParams = prog.typeParams.map(tp => cloneTypeParamDecl(tp, t))
  }

  // Inputs and outputs.
  shell.ports = {
    inputs:  prog.ports.inputs.map(i => cloneInputDecl(i, t)),
    outputs: prog.ports.outputs.map(o => cloneOutputDecl(o, t)),
    typeDefs: prog.ports.typeDefs,   // shared
  }

  // Body decls — register all decl shells first so cross-references
  // resolve, then fill in the expression-shaped fields. Mirrors the
  // elaborator's two-pass discipline.
  const declShells: BodyDecl[] = []
  for (const d of prog.body.decls) {
    declShells.push(cloneBodyDeclShell(d, t))
  }
  // Resolve expression-shaped fields now that all decl shells are
  // registered in the table.
  for (let i = 0; i < prog.body.decls.length; i++) {
    fillBodyDecl(prog.body.decls[i], declShells[i], t)
  }
  shell.body.decls = declShells

  shell.body.assigns = prog.body.assigns.map(a => cloneAssign(a, t))

  // Re-derive decl tables now that body.decls is fully populated. The
  // tables on the shell were placeholder `[]` during the fill phase
  // (necessary so cross-decl references in the table cache could be
  // resolved against the shell's identity); now that fill is done the
  // tables are projected from the canonical body.decls.
  const tables = buildDeclTables(shell.body.decls)
  shell.regs      = tables.regs
  shell.params    = tables.params
  shell.instances = tables.instances
  // Rebuild programRegistry now that cloned instances exist. Each
  // cloned InstanceDecl's typeKey was copied verbatim from the source;
  // the cloned `.type` ResolvedProgram is the canonical version for
  // this clone tree (via t.nestedPrograms memo), so the registry
  // entries are consistent.
  shell.programRegistry = buildProgramRegistry(tables.instances)

  return shell
}

// ─────────────────────────────────────────────────────────────
// Decl cloning — shells inserted into table BEFORE recursing
// ─────────────────────────────────────────────────────────────

function cloneTypeParamDecl(d: TypeParamDecl, t: CloneTable): TypeParamDecl {
  const cached = t.typeParams.get(d)
  if (cached) return cached
  const fresh: TypeParamDecl = { op: 'typeParamDecl', name: d.name }
  if (d.default !== undefined) fresh.default = d.default
  t.typeParams.set(d, fresh)
  return fresh
}

function cloneInputDecl(d: InputDecl, t: CloneTable): InputDecl {
  const cached = t.inputs.get(d)
  if (cached) return cached
  const fresh: InputDecl = { op: 'inputDecl', name: d.name }
  t.inputs.set(d, fresh)
  if (d.type !== undefined)   fresh.type = clonePortType(d.type, t)
  if (d.default !== undefined) fresh.default = cloneExpr(d.default, t)
  return fresh
}

function cloneOutputDecl(d: OutputDecl, t: CloneTable): OutputDecl {
  const cached = t.outputs.get(d)
  if (cached) return cached
  const fresh: OutputDecl = { op: 'outputDecl', name: d.name }
  t.outputs.set(d, fresh)
  if (d.type !== undefined)   fresh.type = clonePortType(d.type, t)
  return fresh
}

/** Pre-register a body decl: returns a shell with placeholder
 *  expressions. The expression fields are filled in afterwards by
 *  `fillBodyDecl`. */
function cloneBodyDeclShell(d: BodyDecl, t: CloneTable): BodyDecl {
  switch (d.op) {
    case 'regDecl': {
      const fresh: RegDecl = { op: 'regDecl', name: d.name, init: 0 as ResolvedExpr }
      if (d.type !== undefined) fresh.type = d.type   // ScalarKind | AliasTypeDef (shared)
      if (d._liftedFrom !== undefined) fresh._liftedFrom = d._liftedFrom
      // update placeholder added only if original had one; the value is
      // overwritten in fillBodyDecl.
      if (d.update !== undefined) fresh.update = 0 as ResolvedExpr
      t.regs.set(d, fresh)
      return fresh
    }
    case 'paramDecl': {
      // ParamDecls are session-scoped by name (the materializer holds
      // the canonical decl in ctx.paramDecls and compile_session keys
      // FFI handles by that decl's identity). Cloning would mint a
      // fresh decl whose identity the materializer's table doesn't
      // know — paramHandles.get(decl) then misses, emit_resolved emits
      // const 0 instead of a `param` operand, and the parameter
      // silently never reaches the JIT. Preserve identity here.
      // inline_instances.ts already documents this invariant:
      // "ParamDecls and ProgramDecls are lifted as-is (no rename:
      // ParamDecls are session-scoped by name)."
      t.params.set(d, d)
      return d
    }
    case 'instanceDecl': {
      // Instance type-program is cloned via the nested-program memo
      // (so two instances of the same nested program share cloned type).
      // typeKey is preserved across clone — cloneProgram doesn't rename
      // the program, so the same key still resolves it in the (rebuilt)
      // registry.
      const fresh: InstanceDecl = {
        op: 'instanceDecl',
        name: d.name,
        type: cloneProgram(d.type, t),
        typeKey: d.typeKey,
        typeArgs: [],
        inputs: [],
      }
      t.instances.set(d, fresh)
      return fresh
    }
    case 'programDecl': {
      const fresh: ProgramDecl = {
        op: 'programDecl',
        name: d.name,
        program: cloneProgram(d.program, t),
      }
      t.programs.set(d, fresh)
      return fresh
    }
  }
}

function fillBodyDecl(orig: BodyDecl, fresh: BodyDecl, t: CloneTable): void {
  if (orig.op === 'regDecl' && fresh.op === 'regDecl') {
    fresh.init = cloneExpr(orig.init, t)
    if (orig.update !== undefined) {
      fresh.update = cloneExpr(orig.update, t)
    }
    return
  }
  if (orig.op === 'instanceDecl' && fresh.op === 'instanceDecl') {
    // typeArgs and inputs are indexed (positions into the target
    // program's typeParams[] / ports.inputs[]). Since cloneProgram
    // preserves the order of decls, the target's positions don't
    // shift — indices pass through unchanged.
    fresh.typeArgs = orig.typeArgs.map(a => ({
      param: a.param,
      value: a.value,
    }))
    fresh.inputs = orig.inputs.map(i => ({
      port:  i.port,
      value: cloneExpr(i.value, t),
    }))
    return
  }
  // paramDecl, programDecl — no expression-shaped fields to fill.
}

// ─────────────────────────────────────────────────────────────
// Assigns
// ─────────────────────────────────────────────────────────────

function cloneAssign(a: BodyAssign, t: CloneTable): BodyAssign {
  // Post-Phase-0a: BodyAssign is OutputAssign-only. NextUpdate folded
  // into RegDecl.update at elaboration time. Target is now OutputIdx
  // (index into prog.ports.outputs[]) or the dac sentinel — cloning
  // preserves output order, so the index passes through unchanged.
  const target: typeof a.target =
    typeof a.target === 'object' && 'kind' in a.target
      ? { kind: 'dac' }                 // sentinel — fresh object
      : a.target                         // OutputIdx — pass through
  const fresh: OutputAssign = {
    op: 'outputAssign',
    target,
    expr: cloneExpr(a.expr, t),
  }
  return fresh
}

function lookupRegDecl(d: RegDecl, t: CloneTable): RegDecl {
  const cloned = t.regs.get(d)
  if (!cloned) throw new Error(`clone: unregistered RegDecl '${d.name}'`)
  return cloned
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
// Expressions
// ─────────────────────────────────────────────────────────────

function cloneExpr(e: ResolvedExpr, t: CloneTable): ResolvedExpr {
  if (typeof e === 'number' || typeof e === 'boolean') return e
  if (Array.isArray(e)) return e.map(x => cloneExpr(x, t))
  // Specialize-time substitution: a TypeParamRef whose idx is a key
  // in the subst map collapses to the integer literal. The subst map
  // is keyed by TypeParamIdx (position in the ROOT program's
  // typeParams), populated by cloneWithSubst. Refs whose idx is NOT
  // in subst (i.e., the ref belongs to a nested program's own
  // type-params) pass through to cloneOpNode for the usual ref-clone
  // treatment. (Nested-program scope tracking is not yet supported
  // here — tropical's stdlib has no nested programDecls, so this
  // hasn't been an issue in practice; if it becomes one, push the
  // current-program onto a stack in CloneTable and disambiguate.)
  if (t.subst !== undefined && e.op === 'typeParamRef') {
    const v = t.subst.get(e.idx)
    if (v !== undefined) return v
  }
  // Inline-time substitution (Phase C5): an InputRef whose idx is a
  // key in inputSubst collapses to the wired-in expression from the
  // outer site. The substituted expression is already in the outer's
  // namespace; pass it through by reference (preserves DAG sharing
  // when the same input is used multiple times in the inner body).
  if (t.inputSubst !== undefined && typeof e === 'object' && e.op === 'inputRef') {
    const v = t.inputSubst.get(e.idx)
    if (v !== undefined) return v
  }
  return cloneOpNode(e, t)
}

/** Clone a BinderDecl, applying the optional binderOffset. No dedup
 *  table: BinderDecl identity is the (possibly shifted) idx, not the
 *  object — two clones of the same source binder produce semantically
 *  equivalent BinderDecl objects with the same idx, which is what
 *  matters for substitution lookups in array_lower/sum_lower. */
function cloneBinder(b: BinderDecl, t: CloneTable): BinderDecl {
  const shifted = (b.idx as number) + (t.binderOffset ?? 0)
  return { op: 'binderDecl', name: b.name, idx: binderIdx(shifted) }
}

function cloneOpNode(node: ResolvedExprOp, t: CloneTable): ResolvedExprOp {
  switch (node.op) {
    // Indexed refs pass through, with optional offset shifting applied
    // for inlineInstances' lift scenario. Cloning alone preserves decl
    // order, but a clone being lifted into another program needs its
    // indices remapped to the outer's namespace. The InputSubst case
    // doesn't shift inputRefs because they're substituted to outer
    // expressions before they could need remapping.
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
    // BindingRef: shift the idx by binderOffset (parallel to RegRef,
    // ParamRef, NestedOut.instance shifting). No object identity to
    // preserve; the same source idx + same offset always yields the
    // same target idx, so all refs to the same binder in the source
    // collapse to the same idx in the clone.
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
        variant: node.variant,    // shared
        payload: node.payload.map(p => ({ field: p.field, value: cloneExpr(p.value, t) })),
      }
      return fresh
    }
    case 'match': {
      const arms: MatchArm[] = node.arms.map(arm => ({
        variant: arm.variant,    // shared
        binders: arm.binders.map(b => cloneBinder(b, t)),
        body: cloneExpr(arm.body, t),
      }))
      const fresh: Match = {
        op: 'match',
        type: node.type,    // shared
        scrutinee: cloneExpr(node.scrutinee, t),
        arms,
      }
      return fresh
    }

    // Combinators — each carries its binder decls.
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

    // Operators — uniform `args` shape.
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
