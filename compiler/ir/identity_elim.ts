/**
 * identity_elim.ts — categorical identity-law rewrite as a strata pass.
 *
 * The identity law in the cartesian-with-trace category of signal-flow
 * graphs:
 *
 *   id_T ∘ f = f = f ∘ id_T
 *
 * Operational realization: an `InstanceDecl` whose body is the identity
 * morphism on its input — `out = inputRef(p)` for some input port `p` —
 * is a no-op kernel. Composing it with any consumer is equivalent to the
 * consumer alone. We rewrite by inlining: every `NestedOut(I, o)` ref to
 * the identity instance's output `o` becomes the expression that was
 * wired into `p` at the consumer site, and the `InstanceDecl` is dropped
 * from the body.
 *
 * Under the fractal architecture (M11 Phase 4+, `inlineInstances`
 * retired), trivial wires lifted by `liftWireToProgram` would survive
 * as identity `InstanceDecl`s — one kernel per trivial wire — bloating
 * slot space and adding pointless read/write instructions per sample.
 * `identityElim` recognizes and eliminates them, returning the wire to
 * a direct slot-read on the consumer side. The pass is the structural
 * analogue of compiler peephole optimizations, but it isn't a peephole:
 * it's a categorical equation made operational, exactly parallel to the
 * other strata passes (each one mechanizes some category-theoretic law
 * about the IR).
 *
 * Implementation discipline: we never mutate the input. When work needs
 * to be done, we deep-clone via `cloneResolvedProgram` and mutate the
 * clone freely. This keeps the rewrite local — surviving `InstanceDecl`
 * objects preserve their identity inside the clone, so cross-decl
 * `NestedOut` refs stay consistent without an explicit rename pass.
 *
 * Pure, idempotent, meaning-preserving. Runs at every level of the
 * strata pipeline because the pipeline itself recurses through nested
 * programs.
 */

import type {
  ResolvedProgram, ResolvedExpr, ResolvedExprOp,
  InstanceDecl, InputDecl, OutputDecl,
  InstanceIdx, OutputIdx,
  Fold, Scan, Generate, Iterate, Chain, Map2, ZipWith, Let, Tag, Match,
} from './nodes.js'
import { instanceIdx, outputIdx } from './nodes.js'
import { cloneResolvedProgram } from './clone.js'
import { withDeclTables } from './decl_tables.js'

// ─── Detection ─────────────────────────────────────────────────────────────

/** An identity `InstanceDecl` carries enough information to rewire its
 *  consumers. For each output port we record the resolved expression
 *  that would substitute for `NestedOut(I, output)`. */
interface IdentityDeclRewrite {
  readonly inst: InstanceDecl
  /** For each of the instance's outputs (by OutputIdx), the expression
   *  to substitute for `NestedOut(I, output)`. */
  readonly outputSub: ReadonlyMap<OutputIdx, ResolvedExpr>
}

/** Recognize whether an InstanceDecl's program body is the identity
 *  morphism. A program is identity when its body has no decls (no
 *  state, no nested instances), and every output is assigned exactly
 *  one `inputRef` (each output forwards a specific input). */
function detectIdentity(inst: InstanceDecl): IdentityDeclRewrite | null {
  const prog = inst.type

  // No state, no nested instances, no parameters in the body.
  if (prog.body.decls.length > 0) return null

  // Map from this instance's wired port position to the wired expression.
  // inst.inputs[k].port is InputIdx into prog.ports.inputs[].
  const wiredByPos = new Map<number, ResolvedExpr>()
  for (const i of inst.inputs) wiredByPos.set(i.port, i.value)

  // For each output position, find its outputAssign expression.
  // OutputAssign.target is OutputIdx or the dac sentinel.
  const outputSub = new Map<OutputIdx, ResolvedExpr>()
  for (let oi = 0; oi < prog.ports.outputs.length; oi++) {
    const assigns = prog.body.assigns.filter(
      a => a.op === 'outputAssign' && typeof a.target === 'number' && a.target === oi,
    )
    if (assigns.length !== 1) return null   // missing or duplicated

    const expr = (assigns[0] as { expr: ResolvedExpr }).expr
    // Identity case: expr is exactly `inputRef(idx)` for some input port.
    if (typeof expr !== 'object' || expr === null || Array.isArray(expr)) return null
    if (expr.op !== 'inputRef') return null

    const wired = wiredByPos.get(expr.idx)
    if (wired === undefined) {
      // The instance must have a wire for every required input; if not,
      // we don't have enough information to inline. Defensive — should
      // not happen in well-formed IR.
      return null
    }
    outputSub.set(outputIdx(oi), wired)
  }

  // Must have at least one output to be a meaningful identity.
  if (outputSub.size === 0) return null

  return { inst, outputSub }
}

// ─── Substitution ─────────────────────────────────────────────────────────

interface SubstCtx {
  /** Eliminated-instance lookup: nestedOut(i, o) where i is eliminated
   *  resolves to the wired-in expression for that port. */
  eliminatedSub: ReadonlyMap<InstanceIdx, ReadonlyMap<OutputIdx, ResolvedExpr>>
  /** Surviving-instance remap: oldIdx → newIdx after eliminated decls
   *  drop out of body.instances. */
  survivorRemap: ReadonlyMap<InstanceIdx, InstanceIdx>
}

function substExpr(
  e: ResolvedExpr,
  ctx: SubstCtx,
  memo: WeakMap<object, ResolvedExpr>,
): ResolvedExpr {
  if (typeof e === 'number' || typeof e === 'boolean') return e
  if (Array.isArray(e)) {
    const cached = memo.get(e)
    if (cached !== undefined) return cached
    const out = e.map(x => substExpr(x, ctx, memo))
    memo.set(e, out)
    return out
  }
  const cached = memo.get(e)
  if (cached !== undefined) return cached
  const out = substOpNode(e, ctx, memo)
  memo.set(e, out)
  return out
}

function substOpNode(
  node: ResolvedExprOp,
  ctx: SubstCtx,
  memo: WeakMap<object, ResolvedExpr>,
): ResolvedExpr {
  const recur = (x: ResolvedExpr) => substExpr(x, ctx, memo)

  switch (node.op) {
    case 'nestedOut': {
      const elimSub = ctx.eliminatedSub.get(node.instance)
      if (elimSub !== undefined) {
        const v = elimSub.get(node.output)
        if (v === undefined) return node
        // The replacement may itself contain nestedOut refs to other
        // identity instances; walk recursively so chains collapse.
        return recur(v)
      }
      // Surviving instance: remap idx to its new position. If the
      // remap is the identity (no shifts ahead of it), this is a no-op.
      const newIdx = ctx.survivorRemap.get(node.instance)
      if (newIdx !== undefined && newIdx !== node.instance) {
        return { op: 'nestedOut', instance: newIdx, output: node.output }
      }
      return node
    }
    case 'inputRef':
    case 'regRef':
    case 'paramRef':
    case 'typeParamRef':
    case 'bindingRef':
    case 'sampleRate':
    case 'sampleIndex':
      return node

    // Binary
    case 'add': case 'sub': case 'mul': case 'div': case 'mod':
    case 'lt':  case 'lte': case 'gt':  case 'gte': case 'eq':  case 'neq':
    case 'and': case 'or':
    case 'bitAnd': case 'bitOr': case 'bitXor': case 'lshift': case 'rshift':
    case 'floorDiv': case 'ldexp':
      return { op: node.op, args: [recur(node.args[0]), recur(node.args[1])] }

    // Unary
    case 'neg': case 'not': case 'bitNot':
    case 'sqrt': case 'abs': case 'floor': case 'ceil': case 'round':
    case 'floatExponent': case 'toInt': case 'toBool': case 'toFloat':
      return { op: node.op, args: [recur(node.args[0])] }

    // Ternary
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
      const fresh: Generate = { op: 'generate', count: recur(node.count), iter: node.iter, body: recur(node.body) }
      return fresh
    }
    case 'iterate': {
      const fresh: Iterate = { op: 'iterate', count: recur(node.count), init: recur(node.init), iter: node.iter, body: recur(node.body) }
      return fresh
    }
    case 'chain': {
      const fresh: Chain = { op: 'chain', count: recur(node.count), init: recur(node.init), iter: node.iter, body: recur(node.body) }
      return fresh
    }
    case 'map2': {
      const fresh: Map2 = { op: 'map2', over: recur(node.over), elem: node.elem, body: recur(node.body) }
      return fresh
    }
    case 'zipWith': {
      const fresh: ZipWith = { op: 'zipWith', a: recur(node.a), b: recur(node.b), x: node.x, y: node.y, body: recur(node.body) }
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
      const fresh: Match = {
        op: 'match',
        type: node.type,
        scrutinee: recur(node.scrutinee),
        arms: node.arms.map(arm => ({
          variant: arm.variant,
          binders: arm.binders,
          body: recur(arm.body),
        })),
      }
      return fresh
    }
  }
}

// ─── Pass entry point ─────────────────────────────────────────────────────

/** Eliminate identity InstanceDecls from a `ResolvedProgram`. Idempotent
 *  — applying twice produces the same result as applying once. Does not
 *  mutate the input program.
 *
 *  After dropping eliminated instances, the surviving instances'
 *  positions shift — every InstanceIdx ref to a survivor needs remapping
 *  to its new position. The substitution walk does both jobs in one
 *  pass: nestedOut(eliminatedIdx) → expr; nestedOut(survivorOldIdx) →
 *  nestedOut(survivorNewIdx). */
export function identityElim(prog: ResolvedProgram): ResolvedProgram {
  // Fast path: detect any identities on the input WITHOUT cloning. If
  // none, return the original program unchanged.
  const anyIdentity = prog.body.decls.some(
    d => d.op === 'instanceDecl' && detectIdentity(d) !== null,
  )
  if (!anyIdentity) return prog

  // Clone so we can mutate freely. Clone preserves decl positions so
  // InstanceIdx values transfer directly across the boundary.
  const cloned = cloneResolvedProgram(prog)

  // Build position-keyed identity table: oldInstanceIdx → outputSub map
  // OR (for survivors) → newInstanceIdx after eliminated decls drop out.
  const eliminatedSub = new Map<InstanceIdx, ReadonlyMap<OutputIdx, ResolvedExpr>>()
  const survivorRemap = new Map<InstanceIdx, InstanceIdx>()
  {
    let newPos = 0
    for (let oldPos = 0; oldPos < cloned.instances.length; oldPos++) {
      const inst = cloned.instances[oldPos]
      const id = detectIdentity(inst)
      if (id !== null) {
        eliminatedSub.set(instanceIdx(oldPos), id.outputSub)
      } else {
        survivorRemap.set(instanceIdx(oldPos), instanceIdx(newPos++))
      }
    }
  }
  if (eliminatedSub.size === 0) return cloned   // defensive

  const ctx: SubstCtx = { eliminatedSub, survivorRemap }
  const memo = new WeakMap<object, ResolvedExpr>()

  // Walk every expression once: rewrite nestedOut for both eliminated
  // (substitute) and survivor (remap idx) cases. Surviving instances'
  // input wires also get walked.
  for (const decl of cloned.body.decls) {
    if (decl.op === 'instanceDecl') {
      const oldIdx = cloned.instances.indexOf(decl) as number   // safe: decl is in cloned.instances
      if (eliminatedSub.has(instanceIdx(oldIdx))) continue   // dropping it
      for (const i of decl.inputs) {
        i.value = substExpr(i.value, ctx, memo)
      }
    } else if (decl.op === 'regDecl') {
      decl.init = substExpr(decl.init, ctx, memo)
      if (decl.update !== undefined) {
        decl.update = substExpr(decl.update, ctx, memo)
      }
    }
  }
  for (const a of cloned.body.assigns) {
    a.expr = substExpr(a.expr, ctx, memo)
  }

  // Drop eliminated InstanceDecls from the body.
  cloned.body.decls = cloned.body.decls.filter(d => {
    if (d.op !== 'instanceDecl') return true
    const oldIdx = cloned.instances.indexOf(d)
    return !eliminatedSub.has(instanceIdx(oldIdx))
  })
  // Re-project tables to match the post-filter body.
  return withDeclTables({
    ...cloned,
    body: { ...cloned.body },   // already mutated in place
  })
}
