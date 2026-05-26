/**
 * inline_instances.ts — splice each `InstanceDecl` into its parent.
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
 *   3. The inner is rewritten with input substitution: every
 *      `InputRef` whose decl belongs to the inner's `ports.inputs`
 *      is replaced by the wired-in expression from
 *      `instanceDecl.inputs[port]`. The substituted expression
 *      passes through by reference, preserving DAG sharing.
 *   4. Rewritten `RegDecl`s are lifted into the outer's `body.decls`,
 *      renamed `${instance.name}_${innerName}`. The update
 *      expression travels on `decl.update`, so lifting the decl
 *      carries its update with it — no per-assign rewrite needed.
 *      `ProgramDecl`s and `ParamDecl`s are lifted as-is (no rename:
 *      ParamDecls are session-scoped by name, ProgramDecls are
 *      passive type bindings).
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
  RegDecl, BinderDecl,
  TypeParamDecl,
  Tag, Match, MatchArm,
  Let,
  Fold, Scan, Generate, Iterate, Chain, Map2, ZipWith,
  InputIdx, OutputIdx, InstanceIdx,
  RegIdx, ParamIdx, BinderIdx,
} from './nodes.js'
import { inputIdx, outputIdx, instanceIdx, binderIdx } from './nodes.js'
import { specializeProgram } from './specialize.js'
import { sumLower } from './sum_lower.js'
import { mkProgram, getInstanceType } from './decl_tables.js'
import { mapExpr, NoRewrite, type ExprRewrite } from './recursion.js'

export function inlineInstances(prog: ResolvedProgram): ResolvedProgram {
  // Fast path: no instances at this level means there's nothing to do.
  // Pass through by reference. (Sub-program decl bodies don't get
  // walked here — they're passive type bindings; the runtime never
  // evaluates them, so we don't pay clone cost on those.)
  if (!hasInstanceDecl(prog)) return prog

  // Functional rewrite: walk prog's body in order, partition decls
  // into survivors (non-instance) and lifted (from inlined inners).
  // Surviving InstanceDecls are absorbed by `inlineOneInstance`,
  // which calls `inlineSubstProgram` to produce fresh inner decls
  // (with offsets pre-shifted) and pushes them into `liftedDecls`.
  // The original `prog` is never mutated; the result is built fresh
  // via `mkProgram` at the bottom.
  //
  // Per-kind offset rule: every RegRef / ParamRef inside a freshly
  // substituted sub-program uses idx values that are valid in that
  // (private) sub-program. After lifting into outer, those idx values
  // need to point at positions in the merged body
  // `survivingDecls ++ liftedDecls`. So per-kind position =
  // outer's own count + count lifted from earlier instances +
  // position-within-this-lift. We pass offsets (outer count +
  // lifted-so-far) to `inlineSubstProgram`, which adds them to every
  // surviving indexed ref during the rewrite.
  const liftedDecls: BodyDecl[] = []
  const nestedOutSubst = new Map<InstanceIdx, Map<OutputIdx, ResolvedExpr>>()
  const survivingDecls: BodyDecl[] = []
  const outerRegCount   = prog.regs.length
  const outerParamCount = prog.params.length
  let liftedBinderCount = 0
  let instCount = 0
  for (const decl of prog.body.decls) {
    if (decl.op !== 'instanceDecl') {
      survivingDecls.push(decl)
      continue
    }
    const regOffset    = outerRegCount    + liftedDecls.filter(d => d.op === 'regDecl').length
    const paramOffset  = outerParamCount  + liftedDecls.filter(d => d.op === 'paramDecl').length
    const binderOffset = prog.binderCount + liftedBinderCount
    liftedBinderCount += inlineOneInstance(
      decl,
      prog,
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
  const newAssigns: BodyAssign[] = prog.body.assigns.map(a => substAssign(a, nestedOutSubst, memo))

  const block: ResolvedBlock = { op: 'block', decls: newDecls, assigns: newAssigns }
  return mkProgram({
    name: prog.name,
    typeParams: prog.typeParams,
    ports: prog.ports,
    body: block,
    binderCount: prog.binderCount + liftedBinderCount,
    // Post-inline: every instance has been lifted away. Zero remaining
    // InstanceDecls means zero typeKey references means an empty
    // registry is sufficient (validateProgramRegistry will pass).
    programRegistry: new Map(),
  })
}

// ─────────────────────────────────────────────────────────────
// Per-instance inlining
// ─────────────────────────────────────────────────────────────

function inlineOneInstance(
  decl: InstanceDecl,
  enclosing: ResolvedProgram,
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
  const specialized = specializeInner(decl, enclosing)

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
  const inputSubst = buildInputSubst(decl, flattened, enclosing)

  // 4. Functional rewrite of the (specialized + sub-inlined) inner
  //    with input substitution AND idx shifting. The rewritten
  //    program's RegRefs / ParamRefs get their idx shifted by (outer's
  //    existing count + previously-lifted count), so when the lifted
  //    decls are appended to the outer's body, the refs already point
  //    at the lifted positions. InputRefs are substituted to outer
  //    expressions before any shift could matter.
  const cloned = inlineSubstProgram(flattened, inputSubst, { regOffset, paramOffset, binderOffset })

  // 5. Lift the cloned inner's body decls into the outer. Names are
  //    prefixed with the instance name to avoid collisions when
  //    multiple instances of the same program are inlined. Post-Phase-0a
  //    reg updates live on the decl, so lifting the decl carries its
  //    update with it — no separate assign-lifting needed.
  liftClonedBody(decl.name, cloned, liftedDecls)

  // 6. Record output expressions for NestedOut substitution.
  //    NestedOut.output is now OutputIdx (position into the target's
  //    ports.outputs[]); we map each position to its cloned expression.
  recordOutputs(decl, enclosing, instIdx, cloned, nestedOutSubst)

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
function specializeInner(decl: InstanceDecl, enclosing: ResolvedProgram): ResolvedProgram {
  const declType = getInstanceType(enclosing, decl)
  if (declType.typeParams.length === 0 && decl.typeArgs.length === 0) {
    return declType
  }
  // typeArgs.param is now TypeParamIdx (position in target's typeParams[]).
  // Convert idx → decl by looking up in declType.typeParams.
  const subst = new Map<TypeParamDecl, number>()
  for (const a of decl.typeArgs) {
    const pd = declType.typeParams[a.param]
    if (pd === undefined) {
      throw new Error(
        `inlineInstances: instance '${decl.name}' typeArg idx=${a.param} out of range ` +
        `(target '${declType.name}' has ${declType.typeParams.length} typeParams)`,
      )
    }
    subst.set(pd, a.value)
  }
  return specializeProgram(declType, subst)
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
  enclosing: ResolvedProgram,
): ReadonlyMap<InputIdx, ResolvedExpr> {
  // decl.inputs[k].port is InputIdx into the template's
  // ports.inputs[]. The cloned `inner` may have different InputDecl
  // objects but the positions match (specialize preserves port order).
  // The output map is keyed by InputIdx — positions in
  // `inner.ports.inputs[]`, which is what InputRef.idx inside the
  // cloned body references.
  const declType = getInstanceType(enclosing, decl)
  const wiredByIdx = new Map<number, ResolvedExpr>()
  for (const w of decl.inputs) wiredByIdx.set(w.port, w.value)
  const subst = new Map<InputIdx, ResolvedExpr>()
  for (let i = 0; i < declType.ports.inputs.length; i++) {
    const innerPort = inner.ports.inputs[i]
    if (innerPort === undefined) {
      throw new Error(
        `inlineInstances: instance '${decl.name}' input arity mismatch ` +
        `(template: ${declType.ports.inputs.length}, specialized: ${inner.ports.inputs.length})`,
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
  enclosing: ResolvedProgram,
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

  const templateOutputs = getInstanceType(enclosing, decl).ports.outputs
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
 * Substitute NestedOut refs in a decl's expression-shaped fields,
 * returning a fresh decl when the rewrite changed anything (or the
 * input by reference when unchanged). Indexed refs in expressions
 * (RegRef/ParamRef/InputRef) carry integers post-Phase-1, so
 * replacing the decl object doesn't orphan anything — positions
 * stay stable because the body.decls order is preserved.
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
    case 'regDecl': {
      const init   = substExpr(d.init, subst, memo)
      const update = d.update !== undefined ? substExpr(d.update, subst, memo) : undefined
      if (init === d.init && update === d.update) return d
      const fresh: RegDecl = { op: 'regDecl', name: d.name, init }
      if (d.type !== undefined) fresh.type = d.type
      if (d._liftedFrom !== undefined) fresh._liftedFrom = d._liftedFrom
      if (update !== undefined) fresh.update = update
      return fresh
    }
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

// ─────────────────────────────────────────────────────────────
// Functional input-substitution + offset shifting
// (replaces the former cloneWithInputSubst from clone.ts)
// ─────────────────────────────────────────────────────────────

/** Produce a fresh `ResolvedProgram` from `inner` with:
 *  - every `InputRef.idx` in `inputSubst` replaced by the outer-site
 *    expression (passed through by reference; the outer caller owns
 *    namespace consistency).
 *  - every surviving `RegRef`, `ParamRef`, `BindingRef`, and
 *    `BinderDecl.idx` shifted by the corresponding offset so the
 *    rewritten program's decls can be appended into the outer body
 *    at their final positions.
 *
 *  Sub-programs in `inner.programRegistry` are shared by reference —
 *  the inner is already strata-processed and its sub-programs are
 *  immutable values. */
function inlineSubstProgram(
  inner: ResolvedProgram,
  inputSubst: ReadonlyMap<InputIdx, ResolvedExpr>,
  shifts: { regOffset: number; paramOffset: number; binderOffset: number },
): ResolvedProgram {
  const { regOffset, paramOffset, binderOffset } = shifts

  const exprRewrite: ExprRewrite = e => {
    if (typeof e !== 'object' || Array.isArray(e)) return NoRewrite
    if (e.op === 'inputRef') {
      const v = inputSubst.get(e.idx)
      // Substituted outer expression passes through by reference;
      // mapExpr will not descend into it because we return a value
      // (not NoRewrite). Preserves DAG sharing of the outer wires.
      return v !== undefined ? v : NoRewrite
    }
    if (e.op === 'regRef') {
      return { op: 'regRef', idx: ((e.idx as number) + regOffset) as RegIdx }
    }
    if (e.op === 'paramRef') {
      return { op: 'paramRef', idx: ((e.idx as number) + paramOffset) as ParamIdx }
    }
    if (e.op === 'bindingRef') {
      return { op: 'bindingRef', idx: ((e.idx as number) + binderOffset) as BinderIdx }
    }
    return NoRewrite
  }

  const binderHook = (b: BinderDecl): BinderDecl => ({
    op: 'binderDecl',
    name: b.name,
    idx: binderIdx((b.idx as number) + binderOffset),
  })

  const rewriteExpr = (e: ResolvedExpr) => mapExpr(e, { expr: exprRewrite, binder: binderHook })

  const mapInputDecl = (i: InputDecl): InputDecl => {
    // Inputs become orphaned post-lift (the outer doesn't reference
    // them) but we still clone the structure for completeness. Default
    // expressions go through the rewrite so any RegRefs/etc. inside
    // them are shifted consistently.
    const fresh: InputDecl = { op: 'inputDecl', name: i.name }
    if (i.type    !== undefined) fresh.type    = i.type
    if (i.default !== undefined) fresh.default = rewriteExpr(i.default)
    return fresh
  }
  const mapOutputDecl = (o: OutputDecl): OutputDecl => {
    const fresh: OutputDecl = { op: 'outputDecl', name: o.name }
    if (o.type !== undefined) fresh.type = o.type
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
        if (d.type !== undefined) fresh.type = d.type
        if (d._liftedFrom !== undefined) fresh._liftedFrom = d._liftedFrom
        if (d.update !== undefined) fresh.update = rewriteExpr(d.update)
        return fresh
      }
      case 'paramDecl':
        return d   // session-scoped; preserve identity
      case 'instanceDecl': {
        // Post-recurse `flattened` should have no instances, but be
        // defensive: pass through with rewritten input wire exprs.
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
    name: inner.name,
    typeParams: inner.typeParams,
    ports: {
      inputs:   inner.ports.inputs.map(mapInputDecl),
      outputs:  inner.ports.outputs.map(mapOutputDecl),
      typeDefs: inner.ports.typeDefs,
    },
    body: {
      op: 'block',
      decls:   inner.body.decls.map(mapDecl),
      assigns: inner.body.assigns.map(mapAssign),
    },
    binderCount: inner.binderCount + binderOffset,
    programRegistry: inner.programRegistry,
  })
}
