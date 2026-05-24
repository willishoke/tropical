/**
 * inline_instances.ts — Phase C5: splice each `InstanceDecl` into its parent.
 *
 * After this pass, `prog.body.decls` contains no `InstanceDecl` and
 * no expression in the program contains a `NestedOut` ref. The inner
 * program's body is fully spliced into the outer:
 *
 *   1. Each generic instance is specialized via `specializeProgram`
 *      using the integer values from `instanceDecl.typeArgs`.
 *   2. Sub-instances inside the (specialized) inner are inlined first
 *      (depth-first, bottom-up). After this, the inner has zero
 *      `InstanceDecl`s of its own.
 *   3. The inner is cloned with input substitution: every `InputRef`
 *      whose decl belongs to the inner's `ports.inputs` is replaced
 *      by the wired-in expression from `instanceDecl.inputs[port]`.
 *      The substituted expression passes through by reference,
 *      preserving DAG sharing.
 *   4. Cloned `RegDecl`s are lifted into the outer's `body.decls`,
 *      renamed `${instance.name}_${innerName}`. Post-Phase-0a the
 *      update expression travels on `decl.update` (no separate
 *      NextUpdate assigns), so lifting the decl carries its update
 *      with it — no per-assign rewrite needed. `ProgramDecl`s and
 *      `ParamDecl`s are lifted as-is (no rename: ParamDecls are
 *      session-scoped by name, ProgramDecls are passive type bindings).
 *   5. The cloned inner's `outputAssign` expressions are recorded in
 *      a substitution table keyed by the *template's* `OutputDecl`
 *      (matched by position to the cloned program's outputs). Every
 *      `NestedOut { instance, output }` reference in the outer's
 *      surviving expressions is replaced by the corresponding
 *      output expression.
 *
 * Decl ordering: instances are processed in the order they appear in
 * `body.decls`. Lifted decls are appended to `body.decls` in the order
 * (inner regs, inner params, inner programDecls) per instance —
 * matching the legacy walker's depth-first traversal in
 * `flatten.ts:collectNestedRegisterExprs`.
 *
 * Naming convention: `${instance.name}_${decl.name}` for lifted regs.
 * The legacy flat plan uses `${parentName}_nested${N}_...` at flatten
 * time; our convention is more readable and equivalent at the level
 * the runtime sees (the flatten step adds the parent instance prefix).
 * Slot identity, not slot name, is what the JIT consumes.
 *
 * Pure: no global state, no input mutation. Returns a fresh
 * `ResolvedProgram` when any inlining occurred; the input is returned
 * by reference when there were no instances to inline (cheap fast
 * path matching the rest of the strata pipeline).
 */

import type {
  ResolvedProgram, ResolvedBlock,
  ResolvedExpr, ResolvedExprOp,
  BodyDecl, BodyAssign, OutputAssign,
  InputDecl, OutputDecl, InstanceDecl,
  TypeParamDecl,
  Tag, Match, MatchArm,
  Let,
  Fold, Scan, Generate, Iterate, Chain, Map2, ZipWith,
  InputIdx, OutputIdx, InstanceIdx,
} from './nodes.js'
import { inputIdx, outputIdx, instanceIdx } from './nodes.js'
import { specializeProgram } from './specialize.js'
import { sumLower } from './sum_lower.js'
import { cloneResolvedProgram, cloneWithInputSubst } from './clone.js'
import { mkProgram } from './decl_tables.js'

export function inlineInstances(prog: ResolvedProgram): ResolvedProgram {
  // Fast path: no instances at this level means there's nothing to do.
  // Pass through by reference. (Sub-program decl bodies don't get
  // walked here — they're passive type bindings; the runtime never
  // evaluates them, so we don't pay clone cost on those.)
  if (!hasInstanceDecl(prog)) return prog

  // Clone the outer first so we have full ownership of decl identity.
  // After cloning, surviving reg/delay decls are fresh objects, and
  // their RegRef/DelayRef sites in body.assigns and other decls' init
  // expressions point at those fresh objects. We can then splice in
  // the inlined-inner bodies, mutating the cloned outer's body in
  // place without touching the input.
  const outer = cloneResolvedProgram(prog)

  // Process every instance in declaration order. We build:
  //   - liftedDecls: cloned reg/param/programDecls to append (post-
  //     Phase-0a regs carry their update expression on the decl, so
  //     no separate lifted-assigns table is needed)
  //   - nestedOutSubst: Map<template OutputDecl, resolved expr> for
  //     replacing NestedOut refs in the outer's surviving expressions
  //
  // Note: when the outer is cloned by cloneResolvedProgram, each
  // InstanceDecl's `type` field is *also* deep-cloned (via the
  // nestedPrograms memo). The cloned InstanceDecl's typeArgs use the
  // cloned program's typeParams; specializeProgram below handles
  // those correctly. The outer's NestedOut refs now key off the
  // cloned OutputDecls (since cloneResolvedProgram routes nested-out
  // through the dedup table to the cloned program's outputs).
  // Keyed by (instance, output): two instances of the same program
  // share the same OutputDecl objects (because cloneResolvedProgram
  // memoizes nested programs), so an OutputDecl alone can't
  // distinguish them. The InstanceDecl is the disambiguator.
  const liftedDecls: BodyDecl[] = []
  // Substitution key is InstanceIdx (position in outer.instances[]) so
  // substExpr can look up node.instance directly. Inner key is OutputIdx.
  const nestedOutSubst = new Map<InstanceIdx, Map<OutputIdx, ResolvedExpr>>()

  const survivingDecls: BodyDecl[] = []
  // Track instance position as we iterate body.decls (matches the
  // ordering used by elaborator + decl_tables to assign InstanceIdx).
  //
  // For lifting: every RegRef / ParamRef inside a cloned sub-program's
  // body uses idx values that are valid in the CLONED program. After
  // lifting into outer, those idx values need to point at positions in
  // the merged body. The merged body is `survivingDecls + liftedDecls`,
  // so per-kind position = outer's own count + count lifted from
  // earlier instances + position-within-this-lift. We pass offsets
  // (outer count + lifted-so-far) to cloneWithInputSubst, which adds
  // them during clone — the cloned program comes out with already-
  // shifted refs that point at the lifted positions.
  const outerRegCount   = outer.regs.length
  const outerParamCount = outer.params.length
  // Track binder ID allocation as we lift sub-program binders. Each
  // inlined inner contributes its binderCount; subsequent inlines
  // start at the running total. After the loop the running total goes
  // into the merged program's binderCount.
  let liftedBinderCount = 0
  let instCount = 0
  for (const decl of outer.body.decls) {
    if (decl.op !== 'instanceDecl') {
      survivingDecls.push(decl)
      continue
    }
    const regOffset    = outerRegCount    + liftedDecls.filter(d => d.op === 'regDecl').length
    const paramOffset  = outerParamCount  + liftedDecls.filter(d => d.op === 'paramDecl').length
    const binderOffset = outer.binderCount + liftedBinderCount
    liftedBinderCount += inlineOneInstance(
      decl,
      instanceIdx(instCount++),
      liftedDecls,
      nestedOutSubst,
      regOffset,
      paramOffset,
      binderOffset,
    )
  }

  // ── Substitute NestedOut refs in surviving outer expressions ──
  // Decl init/update fields and assigns may contain NestedOut refs
  // pointing at the (now-removed) instances. Walk them and replace.
  // RegDecl identity from the outer-clone is preserved (substDecl
  // mutates init/update on the cloned decl in place).
  //
  // The memo preserves DAG sharing across all expressions in the
  // program: a subexpression that appears on multiple paths is
  // walked exactly once. Without memoization, the substitution
  // explodes exponentially in programs where wired-in expressions
  // are referenced many times (e.g., a chain of allpass stages
  // all sharing an LFO input).
  // Substitute on BOTH surviving and lifted decls. Lifted
  // expressions came from `cloneWithInputSubst`, which only
  // substitutes inputs — not NestedOuts. A lifted reg's
  // init/update may contain `nestedOut(otherInstance.out)` (when
  // the inner program wired one of its inputs from a sibling
  // instance's output). The single substExpr pass at the end
  // resolves all of them in a topo-free walk.
  const memo = new WeakMap<object, ResolvedExpr>()
  const newDecls: BodyDecl[] = [
    ...survivingDecls.map(d => substDecl(d, nestedOutSubst, memo)),
    ...liftedDecls.map(d => substDecl(d, nestedOutSubst, memo)),
  ]
  const newAssigns: BodyAssign[] = outer.body.assigns.map(a => substAssign(a, nestedOutSubst, memo))

  const block: ResolvedBlock = { op: 'block', decls: newDecls, assigns: newAssigns }
  return mkProgram({
    name: outer.name,
    typeParams: outer.typeParams,
    ports: outer.ports,
    body: block,
    binderCount: outer.binderCount + liftedBinderCount,
  })
}

// ─────────────────────────────────────────────────────────────
// Per-instance inlining
// ─────────────────────────────────────────────────────────────

function inlineOneInstance(
  decl: InstanceDecl,
  instIdx: InstanceIdx,
  liftedDecls: BodyDecl[],
  nestedOutSubst: Map<InstanceIdx, Map<OutputIdx, ResolvedExpr>>,
  regOffset: number,
  paramOffset: number,
  binderOffset: number,
): number {
  // 1. Specialize the inner program. For non-generic instances
  //    (typeArgs.length === 0), this is a no-op identity — the
  //    instance's program is already concrete.
  const specialized = specializeInner(decl)

  // 2a. Lower sums in the specialized inner BEFORE recursing into
  //     deeper instances or lifting decls. The strata pipeline runs
  //     sumLower before inlineInstances on the *outer* program; the
  //     same ordering must hold per-instance, otherwise a sum-typed
  //     delay inside an inlined inner program (e.g. EnvExpDecay's
  //     `state` when used inside Bubble) leaks into the outer in its
  //     unlowered form, and the slot table built by
  //     `loadProgramDefFromResolved` rejects the residual sum-typed
  //     decl. sumLower is an identity on programs without sums.
  const summed = sumLower(specialized)

  // 2b. Recursively inline sub-instances inside the (specialized,
  //     sum-lowered) inner. Depth-first, bottom-up: by the time we
  //     splice the inner's body here, it has zero InstanceDecls of
  //     its own.
  const flattened = inlineInstances(summed)

  // 3. Build the input substitution map from the wired-in expressions.
  //    Each entry pairs an inner InputDecl with the outer's wired
  //    expression. Any inputs the user didn't wire fall back to the
  //    inner's declared default; missing required inputs are an
  //    elaboration-time error and shouldn't reach this pass.
  const inputSubst = buildInputSubst(decl, flattened)

  // 4. Clone the (specialized + sub-inlined) inner with input
  //    substitution AND idx shifting. The cloned program's RegRefs /
  //    ParamRefs get their idx shifted by (outer's existing count +
  //    previously-lifted count), so when the cloned decls are appended
  //    to the outer's body, the refs already point at the lifted
  //    positions. InputRefs are substituted to outer expressions
  //    before any shift could matter.
  const cloned = cloneWithInputSubst(flattened, inputSubst, { regOffset, paramOffset, binderOffset })

  // 5. Lift the cloned inner's body decls into the outer. Names are
  //    prefixed with the instance name to avoid collisions when
  //    multiple instances of the same program are inlined. Post-Phase-0a
  //    reg updates live on the decl, so lifting the decl carries its
  //    update with it — no separate assign-lifting needed.
  liftClonedBody(decl.name, cloned, liftedDecls)

  // 6. Record output expressions for NestedOut substitution.
  //    NestedOut.output is now OutputIdx (position into the target's
  //    ports.outputs[]); we map each position to its cloned expression.
  recordOutputs(decl, instIdx, cloned, nestedOutSubst)

  // Return the inner's binderCount so the caller can advance its
  // running total — subsequent inlines start their binderOffset
  // beyond this inner's allocations.
  return flattened.binderCount
}

/**
 * Specialize the inner program if generic. Builds a TypeParamDecl-keyed
 * substitution map from the instance's typeArgs (which carry decl
 * references directly).
 */
function specializeInner(decl: InstanceDecl): ResolvedProgram {
  if (decl.type.typeParams.length === 0 && decl.typeArgs.length === 0) {
    return decl.type
  }
  // typeArgs.param is now TypeParamIdx (position in target's typeParams[]).
  // Convert idx → decl by looking up in decl.type.typeParams.
  const subst = new Map<TypeParamDecl, number>()
  for (const a of decl.typeArgs) {
    const pd = decl.type.typeParams[a.param]
    if (pd === undefined) {
      throw new Error(
        `inlineInstances: instance '${decl.name}' typeArg idx=${a.param} out of range ` +
        `(target '${decl.type.name}' has ${decl.type.typeParams.length} typeParams)`,
      )
    }
    subst.set(pd, a.value)
  }
  return specializeProgram(decl.type, subst)
}

/**
 * Build the InputDecl-keyed substitution map for input references
 * inside the inner program. Inputs the user wired explicitly take
 * priority; otherwise the inner's declared default is used (as
 * carried on the InputDecl). After specialization, the inner's
 * InputDecls may differ from the original template's, so we index
 * by position to find each.
 */
function buildInputSubst(
  decl: InstanceDecl,
  inner: ResolvedProgram,
): ReadonlyMap<InputIdx, ResolvedExpr> {
  // decl.inputs[k].port is now InputIdx into decl.type.ports.inputs[].
  // The cloned `inner` may have different InputDecl objects but the
  // positions match (specialize preserves port order). The output
  // map is keyed by InputIdx — positions in `inner.ports.inputs[]`,
  // which is what InputRef.idx inside the cloned body references.
  const wiredByIdx = new Map<number, ResolvedExpr>()
  for (const w of decl.inputs) wiredByIdx.set(w.port, w.value)
  const subst = new Map<InputIdx, ResolvedExpr>()
  for (let i = 0; i < decl.type.ports.inputs.length; i++) {
    const innerPort = inner.ports.inputs[i]
    if (innerPort === undefined) {
      throw new Error(
        `inlineInstances: instance '${decl.name}' input arity mismatch ` +
        `(template: ${decl.type.ports.inputs.length}, specialized: ${inner.ports.inputs.length})`,
      )
    }
    const wired = wiredByIdx.get(i)
    if (wired !== undefined) {
      subst.set(inputIdx(i), wired)
      continue
    }
    if (innerPort.default !== undefined) {
      subst.set(inputIdx(i), innerPort.default)
      continue
    }
    // No wire, no default: the elaborator should have caught it.
    // Reaching here means the input is unused by the inner body, in
    // which case leaving it unsubstituted is harmless.
  }
  return subst
}

/**
 * Lift the cloned inner's body decls into the outer. RegDecls are
 * renamed `${instance.name}_${innerName}`; ParamDecls and ProgramDecls
 * are lifted as-is. Post-Phase-0a reg updates live on the decl, so
 * lifting the decl carries its update with it. outputAssign assigns
 * are NOT lifted — they're consumed by `recordOutputs` to build the
 * NestedOut substitution table.
 */
function liftClonedBody(
  instanceName: string,
  cloned: ResolvedProgram,
  liftedDecls: BodyDecl[],
): void {
  for (const d of cloned.body.decls) {
    switch (d.op) {
      case 'regDecl':
        d.name = `${instanceName}_${d.name}`
        // Stamp provenance with the current outer's name. Each lift
        // overwrites: the post-strata tag is the *outermost* (session-
        // level) instance the decl ultimately came from. Consumers
        // (e.g. applyGateableWraps) match against gateable session
        // instances by name.
        d._liftedFrom = instanceName
        liftedDecls.push(d)
        break
      case 'paramDecl':
        // Session-scoped: keep the original name. Params with the
        // same name across instances refer to the same session param.
        liftedDecls.push(d)
        break
      case 'programDecl':
        // Passive type binding (the inner already inlined its own
        // instances of this nested program). Keep as-is.
        liftedDecls.push(d)
        break
      case 'instanceDecl':
        // The recursive inlineInstances call should have removed
        // every InstanceDecl from `cloned`. Reaching here is a bug.
        throw new Error(
          `inlineInstances: post-recurse: cloned inner '${cloned.name}' still has ` +
          `instanceDecl '${d.name}' — depth-first invariant violated`,
        )
    }
  }

  // Post-Phase-0a: NextUpdate body-assigns are gone. Reg updates live
  // on the decl itself (set at elaboration), so lifting the decl
  // carries the update with it — no per-assign lifting needed here.
  // OutputAssigns are recorded separately by recordOutputs.
}

/**
 * Record the inner's output expressions in `nestedOutSubst`, keyed by
 * the *template's* OutputDecls (which are what the outer's NestedOut
 * refs point at). We match template → cloned by position; each cloned
 * outputAssign expression becomes the substitution value.
 *
 * The recorded expression is the cloned-and-input-substituted output
 * expression. It may still contain `NestedOut` refs for *outer*-scope
 * instances (e.g., a chained allpass's `x` substituted with the
 * previous instance's output). Those are handled by the outer's final
 * `substExpr` pass.
 */
function recordOutputs(
  decl: InstanceDecl,
  instIdx: InstanceIdx,
  cloned: ResolvedProgram,
  nestedOutSubst: Map<InstanceIdx, Map<OutputIdx, ResolvedExpr>>,
): void {
  // Build cloned-output-position → expression map from the cloned
  // assigns. After the migration, OutputAssign.target is an OutputIdx
  // (position into cloned.ports.outputs[]).
  const clonedOutExprByIdx = new Map<number, ResolvedExpr>()
  for (const a of cloned.body.assigns) {
    if (a.op !== 'outputAssign') continue
    if (typeof a.target === 'number') clonedOutExprByIdx.set(a.target, a.expr)
  }

  // For each output position of the template, bind the cloned
  // expression. The outer's NestedOut refs carry the template's
  // OutputIdx (same position, since specialize preserves output
  // order). Key the per-instance substitution table by OutputIdx so
  // the outer's substExpr pass can look it up directly.
  const perInstance = new Map<OutputIdx, ResolvedExpr>()
  nestedOutSubst.set(instIdx, perInstance)

  const templateOutputs = decl.type.ports.outputs
  const clonedOutputs   = cloned.ports.outputs
  for (let i = 0; i < templateOutputs.length; i++) {
    if (clonedOutputs[i] === undefined) {
      throw new Error(
        `inlineInstances: instance '${decl.name}' output arity mismatch ` +
        `(template: ${templateOutputs.length}, cloned: ${clonedOutputs.length})`,
      )
    }
    const expr = clonedOutExprByIdx.get(i)
    if (expr === undefined) {
      throw new Error(
        `inlineInstances: instance '${decl.name}': program '${cloned.name}' has no ` +
        `output_assign for output '${clonedOutputs[i].name}' (idx ${i})`,
      )
    }
    perInstance.set(outputIdx(i), expr)
  }
}

// ─────────────────────────────────────────────────────────────
// Probes
// ─────────────────────────────────────────────────────────────

function hasInstanceDecl(prog: ResolvedProgram): boolean {
  for (const d of prog.body.decls) if (d.op === 'instanceDecl') return true
  return false
}

// ─────────────────────────────────────────────────────────────
// NestedOut substitution — exhaustive expression walker
// ─────────────────────────────────────────────────────────────

/**
 * Substitute NestedOut refs in a decl's expression-shaped fields.
 * The decl object's identity is preserved (mutates init/update of
 * the cloned decl in place). Identity preservation matters because
 * RegRef/DelayRef sites elsewhere in the program point at this
 * decl object — replacing the decl would orphan those refs.
 *
 * Memoization (`memo`) preserves DAG sharing across the whole
 * program: a subexpression visited via two different paths gets
 * walked once, returning the same fresh result both times.
 */
function substDecl(
  d: BodyDecl,
  subst: Map<InstanceIdx, Map<OutputIdx, ResolvedExpr>>,
  memo: WeakMap<object, ResolvedExpr>,
): BodyDecl {
  switch (d.op) {
    case 'regDecl':
      d.init = substExpr(d.init, subst, memo)
      if (d.update !== undefined) {
        d.update = substExpr(d.update, subst, memo)
      }
      return d
    case 'paramDecl':
    case 'programDecl':
      return d
    case 'instanceDecl':
      throw new Error(`inlineInstances: substDecl on surviving InstanceDecl '${d.name}'`)
  }
}

/**
 * Substitute NestedOut refs in an assign's expression. We allocate
 * a fresh assign object (assigns are leaves — nothing else points at
 * them) but preserve the `target` decl/OutputDecl reference so identity
 * matches the outer's body decls. Post-Phase-0a BodyAssign is
 * OutputAssign-only.
 */
function substAssign(
  a: BodyAssign,
  subst: Map<InstanceIdx, Map<OutputIdx, ResolvedExpr>>,
  memo: WeakMap<object, ResolvedExpr>,
): BodyAssign {
  const fresh: OutputAssign = { op: 'outputAssign', target: a.target, expr: substExpr(a.expr, subst, memo) }
  return fresh
}

function substExpr(
  e: ResolvedExpr,
  subst: Map<InstanceIdx, Map<OutputIdx, ResolvedExpr>>,
  memo: WeakMap<object, ResolvedExpr>,
): ResolvedExpr {
  if (typeof e === 'number' || typeof e === 'boolean') return e
  if (Array.isArray(e)) {
    const cached = memo.get(e)
    if (cached !== undefined) return cached
    const out = e.map(x => substExpr(x, subst, memo))
    memo.set(e, out)
    return out
  }
  const cached = memo.get(e)
  if (cached !== undefined) return cached
  const out = substOpNode(e, subst, memo)
  memo.set(e, out)
  return out
}

function substOpNode(
  node: ResolvedExprOp,
  subst: Map<InstanceIdx, Map<OutputIdx, ResolvedExpr>>,
  memo: WeakMap<object, ResolvedExpr>,
): ResolvedExpr {
  const recur = (x: ResolvedExpr) => substExpr(x, subst, memo)

  switch (node.op) {
    case 'nestedOut': {
      const perInstance = subst.get(node.instance)
      if (perInstance === undefined) {
        throw new Error(
          `inlineInstances: nestedOut to instance idx=${node.instance} output idx=${node.output} ` +
          `— instance not inlined?`,
        )
      }
      const v = perInstance.get(node.output)
      if (v === undefined) {
        throw new Error(
          `inlineInstances: nestedOut to instance idx=${node.instance} output idx=${node.output} ` +
          `has no resolved expression for that output`,
        )
      }
      // Walk the substituted expression too: the recorded expression
      // may itself contain `NestedOut` refs to *outer*-scope
      // instances (chained allpass: ap_N's body has `nestedOut(ap_{N-1}.y)`
      // inside it). The memo guarantees each subexpression is walked
      // at most once, so even long chains run in linear time.
      return recur(v)
    }
    case 'inputRef':
    case 'regRef':
    case 'paramRef':
    case 'typeParamRef':
    case 'bindingRef':
      return node
    case 'sampleRate':
    case 'sampleIndex':
      return node

    // Uniform binary ops.
    case 'add': case 'sub': case 'mul': case 'div': case 'mod':
    case 'lt':  case 'lte': case 'gt':  case 'gte': case 'eq':  case 'neq':
    case 'and': case 'or':
    case 'bitAnd': case 'bitOr': case 'bitXor': case 'lshift': case 'rshift':
    case 'floorDiv': case 'ldexp':
      return { op: node.op, args: [recur(node.args[0]), recur(node.args[1])] }

    // Unary ops.
    case 'neg': case 'not': case 'bitNot':
    case 'sqrt': case 'abs': case 'floor': case 'ceil': case 'round':
    case 'floatExponent': case 'toInt': case 'toBool': case 'toFloat':
      return { op: node.op, args: [recur(node.args[0])] }

    // Ternary ops.
    case 'clamp':
      return { op: 'clamp', args: [recur(node.args[0]), recur(node.args[1]), recur(node.args[2])] }
    case 'select':
      return { op: 'select', args: [recur(node.args[0]), recur(node.args[1]), recur(node.args[2])] }
    case 'arraySet':
      return { op: 'arraySet', args: [recur(node.args[0]), recur(node.args[1]), recur(node.args[2])] }
    case 'index':
      return { op: 'index', args: [recur(node.args[0]), recur(node.args[1])] }
    case 'zeros':
      return { op: 'zeros', count: recur(node.count) }

    // Combinators — preserve binders by reference (substitution doesn't
    // touch BinderDecls), recurse into expression-shaped fields.
    case 'fold': {
      const fresh: Fold = {
        op: 'fold',
        over: recur(node.over),
        init: recur(node.init),
        acc:  node.acc,
        elem: node.elem,
        body: recur(node.body),
      }
      return fresh
    }
    case 'scan': {
      const fresh: Scan = {
        op: 'scan',
        over: recur(node.over),
        init: recur(node.init),
        acc:  node.acc,
        elem: node.elem,
        body: recur(node.body),
      }
      return fresh
    }
    case 'generate': {
      const fresh: Generate = {
        op: 'generate',
        count: recur(node.count),
        iter: node.iter,
        body: recur(node.body),
      }
      return fresh
    }
    case 'iterate': {
      const fresh: Iterate = {
        op: 'iterate',
        count: recur(node.count),
        init:  recur(node.init),
        iter:  node.iter,
        body:  recur(node.body),
      }
      return fresh
    }
    case 'chain': {
      const fresh: Chain = {
        op: 'chain',
        count: recur(node.count),
        init:  recur(node.init),
        iter:  node.iter,
        body:  recur(node.body),
      }
      return fresh
    }
    case 'map2': {
      const fresh: Map2 = {
        op: 'map2',
        over: recur(node.over),
        elem: node.elem,
        body: recur(node.body),
      }
      return fresh
    }
    case 'zipWith': {
      const fresh: ZipWith = {
        op: 'zipWith',
        a: recur(node.a),
        b: recur(node.b),
        x: node.x,
        y: node.y,
        body: recur(node.body),
      }
      return fresh
    }
    case 'let': {
      const fresh: Let = {
        op: 'let',
        binders: node.binders.map(b => ({ binder: b.binder, value: recur(b.value) })),
        in: recur(node.in),
      }
      return fresh
    }
    case 'tag': {
      const fresh: Tag = {
        op: 'tag',
        variant: node.variant,
        payload: node.payload.map(p => ({ field: p.field, value: recur(p.value) })),
      }
      return fresh
    }
    case 'match': {
      const arms: MatchArm[] = node.arms.map(arm => ({
        variant: arm.variant,
        binders: arm.binders,
        body:    recur(arm.body),
      }))
      const fresh: Match = {
        op: 'match',
        type: node.type,
        scrutinee: recur(node.scrutinee),
        arms,
      }
      return fresh
    }
  }
}
