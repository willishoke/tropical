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
 *   identity in `MatchExpr.arms[i].variant` and `TagExpr.variant`,
 *   which downstream passes (sum_lower) compare by `===`.
 * - All other decls (`InputDecl`, `OutputDecl`, `TypeParamDecl`,
 *   `RegDecl`, `DelayDecl`, `ParamDecl`, `InstanceDecl`,
 *   `ProgramDecl`, `BinderDecl`) — CLONED, with the `Map<old, new>`
 *   dedup table ensuring each appears at most once.
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
  ResolvedProgram, ResolvedBlock, ResolvedProgramPorts,
  ResolvedExpr, ResolvedExprOpNode,
  InputDecl, OutputDecl, TypeParamDecl,
  RegDecl, DelayDecl, ParamDecl, InstanceDecl, ProgramDecl, BodyDecl,
  BodyAssign, OutputAssign, NextUpdate,
  PortType, ShapeDim,
  BinderDecl,
  TagExpr, MatchExpr, MatchArm,
  LetExpr,
  FoldExpr, ScanExpr, GenerateExpr, IterateExpr, ChainExpr, Map2Expr, ZipWithExpr,
} from './nodes.js'

// ─────────────────────────────────────────────────────────────
// Dedup table — Map<old, new> covers every cloned decl kind
// ─────────────────────────────────────────────────────────────

interface CloneTable {
  inputs:     Map<InputDecl, InputDecl>
  outputs:    Map<OutputDecl, OutputDecl>
  typeParams: Map<TypeParamDecl, TypeParamDecl>
  regs:       Map<RegDecl, RegDecl>
  delays:     Map<DelayDecl, DelayDecl>
  params:     Map<ParamDecl, ParamDecl>
  instances:  Map<InstanceDecl, InstanceDecl>
  programs:   Map<ProgramDecl, ProgramDecl>
  binders:    Map<BinderDecl, BinderDecl>
  /** Nested ResolvedPrograms (held by `InstanceDecl.type` and
   *  `ProgramDecl.program`). Memoized so two instances of the same
   *  nested program share the cloned program object. */
  nestedPrograms: Map<ResolvedProgram, ResolvedProgram>
  /** Optional substitution map for Phase C3 specialize: TypeParamRef
   *  in expression position and TypeParamDecl in ShapeDim position
   *  whose decl is a key here are rewritten to the corresponding
   *  integer. Refs whose decl is NOT in subst are passed through —
   *  they belong to a nested program's own type-params that this
   *  caller is not specializing. */
  subst?: ReadonlyMap<TypeParamDecl, number>
  /** The program whose `typeParams` should be emptied in the clone
   *  (the specialization root). Nested programs retain their own
   *  `typeParams` since this caller isn't substituting them. */
  rootProgram?: ResolvedProgram
  /** Optional substitution map for Phase C5 inlineInstances: InputRef
   *  whose decl is a key here is replaced by the corresponding
   *  ResolvedExpr (from the wired-in expression at the outer site).
   *  The substituted expression is in the *outer* program's namespace
   *  and is not cloned again — it passes through by reference, which
   *  preserves DAG sharing across multiple uses. */
  inputSubst?: ReadonlyMap<InputDecl, ResolvedExpr>
}

function emptyTable(): CloneTable {
  return {
    inputs:     new Map(),
    outputs:    new Map(),
    typeParams: new Map(),
    regs:       new Map(),
    delays:     new Map(),
    params:     new Map(),
    instances:  new Map(),
    programs:   new Map(),
    binders:    new Map(),
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
  t.subst = subst
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
  inputSubst: ReadonlyMap<InputDecl, ResolvedExpr>,
): ResolvedProgram {
  const t = emptyTable()
  t.inputSubst = inputSubst
  return cloneProgram(prog, t)
}

function cloneProgram(prog: ResolvedProgram, t: CloneTable): ResolvedProgram {
  const cached = t.nestedPrograms.get(prog)
  if (cached) return cached

  // Build the program shell first so children can find it via the
  // memo on recursion (e.g., a nested program decl whose body refers
  // back through scope to itself, though current parser disallows
  // that — defensive against future shapes).
  const shell: ResolvedProgram = {
    op: 'program',
    name: prog.name,
    typeParams: [],
    ports: { inputs: [], outputs: [], typeDefs: prog.ports.typeDefs },
    body: { op: 'block', decls: [], assigns: [] },
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
      t.regs.set(d, fresh)
      return fresh
    }
    case 'delayDecl': {
      const fresh: DelayDecl = { op: 'delayDecl', name: d.name, update: 0, init: 0 }
      if (d._liftedFrom !== undefined) fresh._liftedFrom = d._liftedFrom
      t.delays.set(d, fresh)
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
      const fresh: InstanceDecl = {
        op: 'instanceDecl',
        name: d.name,
        type: cloneProgram(d.type, t),
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
    return
  }
  if (orig.op === 'delayDecl' && fresh.op === 'delayDecl') {
    fresh.update = cloneExpr(orig.update, t)
    fresh.init = cloneExpr(orig.init, t)
    return
  }
  if (orig.op === 'instanceDecl' && fresh.op === 'instanceDecl') {
    // typeArgs reference the cloned program's typeParams. Use the
    // cloned-program's typeParams via the dedup table.
    fresh.typeArgs = orig.typeArgs.map(a => ({
      param: cloneTypeParamDecl(a.param, t),
      value: a.value,
    }))
    fresh.inputs = orig.inputs.map(i => ({
      port:  cloneInputDecl(i.port, t),
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
  if (a.op === 'outputAssign') {
    const target: OutputDecl | { kind: 'dac' } =
      'op' in a.target
        ? cloneOutputDecl(a.target, t)
        : { kind: 'dac' }   // sentinel — fresh object, semantically a singleton
    const fresh: OutputAssign = {
      op: 'outputAssign',
      target,
      expr: cloneExpr(a.expr, t),
    }
    return fresh
  }
  // nextUpdate
  const target: RegDecl | DelayDecl = a.target.op === 'regDecl'
    ? lookupRegDecl(a.target, t)
    : lookupDelayDecl(a.target, t)
  const fresh: NextUpdate = {
    op: 'nextUpdate',
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

function lookupDelayDecl(d: DelayDecl, t: CloneTable): DelayDecl {
  const cloned = t.delays.get(d)
  if (!cloned) throw new Error(`clone: unregistered DelayDecl '${d.name}'`)
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
  if (t.subst !== undefined) {
    const v = t.subst.get(d)
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
  // Specialize-time substitution: a TypeParamRef whose decl is a key
  // in the subst map collapses to the integer literal. Refs whose decl
  // is NOT in subst (i.e., the decl belongs to a nested program's own
  // type-params) pass through to cloneOpNode, where they get the usual
  // ref-clone treatment.
  if (t.subst !== undefined && e.op === 'typeParamRef') {
    const v = t.subst.get(e.decl)
    if (v !== undefined) return v
  }
  // Inline-time substitution (Phase C5): an InputRef whose decl is a
  // key in inputSubst collapses to the wired-in expression from the
  // outer site. The substituted expression is already in the outer's
  // namespace; pass it through by reference (preserves DAG sharing
  // when the same input is used multiple times in the inner body).
  if (t.inputSubst !== undefined && typeof e === 'object' && e.op === 'inputRef') {
    const v = t.inputSubst.get(e.decl)
    if (v !== undefined) return v
  }
  return cloneOpNode(e, t)
}

function cloneBinder(b: BinderDecl, t: CloneTable): BinderDecl {
  const cached = t.binders.get(b)
  if (cached) return cached
  const fresh: BinderDecl = { op: 'binderDecl', name: b.name }
  t.binders.set(b, fresh)
  return fresh
}

function cloneOpNode(node: ResolvedExprOpNode, t: CloneTable): ResolvedExprOpNode {
  switch (node.op) {
    // Refs — point at the cloned decl via the dedup table.
    case 'inputRef':  return { op: 'inputRef',  decl: cloneInputDecl(node.decl, t) }
    case 'regRef':    return { op: 'regRef',    decl: lookupRegDecl(node.decl, t) }
    case 'delayRef':  return { op: 'delayRef',  decl: lookupDelayDecl(node.decl, t) }
    case 'paramRef': {
      const cloned = t.params.get(node.decl)
      if (!cloned) throw new Error(`clone: unregistered ParamDecl '${node.decl.name}'`)
      return { op: 'paramRef', decl: cloned }
    }
    case 'typeParamRef': return { op: 'typeParamRef', decl: cloneTypeParamDecl(node.decl, t) }
    case 'bindingRef':   return { op: 'bindingRef',   decl: cloneBinder(node.decl, t) }
    case 'nestedOut': {
      const inst = t.instances.get(node.instance)
      if (!inst) throw new Error(`clone: unregistered InstanceDecl '${node.instance.name}'`)
      // The output decl belongs to inst.type (the cloned nested program);
      // resolve it via cloneOutputDecl which routes through the table.
      return { op: 'nestedOut', instance: inst, output: cloneOutputDecl(node.output, t) }
    }
    case 'sampleRate':  return { op: 'sampleRate' }
    case 'sampleIndex': return { op: 'sampleIndex' }

    // ADT — variant shared, payload/arms cloned.
    case 'tag': {
      const fresh: TagExpr = {
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
      const fresh: MatchExpr = {
        op: 'match',
        type: node.type,    // shared
        scrutinee: cloneExpr(node.scrutinee, t),
        arms,
      }
      return fresh
    }

    // Combinators — each carries its binder decls.
    case 'let': {
      const fresh: LetExpr = {
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
      const fresh: FoldExpr = {
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
      const fresh: ScanExpr = {
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
      const fresh: GenerateExpr = {
        op: 'generate',
        count: cloneExpr(node.count, t),
        iter: cloneBinder(node.iter, t),
        body: cloneExpr(node.body, t),
      }
      return fresh
    }
    case 'iterate': {
      const fresh: IterateExpr = {
        op: 'iterate',
        count: cloneExpr(node.count, t),
        init: cloneExpr(node.init, t),
        iter: cloneBinder(node.iter, t),
        body: cloneExpr(node.body, t),
      }
      return fresh
    }
    case 'chain': {
      const fresh: ChainExpr = {
        op: 'chain',
        count: cloneExpr(node.count, t),
        init: cloneExpr(node.init, t),
        iter: cloneBinder(node.iter, t),
        body: cloneExpr(node.body, t),
      }
      return fresh
    }
    case 'map2': {
      const fresh: Map2Expr = {
        op: 'map2',
        over: cloneExpr(node.over, t),
        elem: cloneBinder(node.elem, t),
        body: cloneExpr(node.body, t),
      }
      return fresh
    }
    case 'zipWith': {
      const fresh: ZipWithExpr = {
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
