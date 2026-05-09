/**
 * lower_bounds.ts — Desugar `in [lo, hi]` port-spec annotations into
 * explicit `clamp` ops on output assigns and input defaults.
 *
 * Bounds are a parse-time notation, not an IR feature. After this pass
 * runs, the parsed AST has:
 *  • `clamp(default_expr, lo, hi)` wherever an input port had bounds and
 *    a default value;
 *  • `clamp(rhs, lo, hi)` wherever an output port had bounds and the body
 *    has an `outputAssign` for it;
 *  • no `bounds` field on any port spec.
 *
 * Bounded inputs *without* defaults are documentation-only — there is no
 * syntactic position at which to insert a clamp without inspecting body
 * scopes (let-bindings can shadow input names). The bound is dropped
 * silently. Bounded outputs without explicit assigns are an error.
 *
 * Built-in port-type aliases (`signal`, `bipolar`, `unipolar`, `phase`,
 * `freq`) carry implicit bounds applied here. Explicit `in [lo, hi]`
 * overrides the alias bounds. User-declared aliases (`type X = float`)
 * carry no bounds in this iteration; they're pure name aliases.
 *
 * One-sided bounds:
 *  • `[lo, hi]`   → `clamp(expr, lo, hi)`
 *  • `[lo, null]` → `select(expr > lo, expr, lo)`     (max(expr, lo))
 *  • `[null, hi]` → `select(expr < hi, expr, hi)`     (min(expr, hi))
 *  • `[null, null]` → expr (no-op)
 */

import type {
  Expr, Program, ProgramPort, ProgramPortSpec, PortTypeDecl, NameRef,
  Call, BinaryOp,
} from './nodes.js'
import { nameRef } from './nodes.js'

/** Local copy of the elaborator's nameRef predicate — `parse/nodes.ts`
 *  doesn't export one and we don't want a layering dep on the elaborator
 *  from inside the parser. */
function isNameRef(v: unknown): v is NameRef {
  return typeof v === 'object' && v !== null && !Array.isArray(v)
    && (v as { op?: unknown }).op === 'nameRef'
}

/** Built-in port-type aliases with implicit bounds. */
const BUILTIN_PORT_BOUNDS: Record<string, [number | null, number | null]> = {
  signal:   [-1, 1],
  bipolar:  [-1, 1],
  unipolar: [0, 1],
  phase:    [0, 1],
  freq:     [0, null],
}

type Bounds = [number | null, number | null]

/** Effective bounds for a port spec: explicit `in [...]` if present,
 *  otherwise the built-in alias's bounds (if any), otherwise none. */
function effectiveBounds(spec: ProgramPortSpec): Bounds | null {
  if (spec.bounds !== undefined) return spec.bounds
  return aliasBounds(spec.type)
}

function aliasBounds(t: PortTypeDecl | undefined): Bounds | null {
  if (t === undefined || !isNameRef(t)) return null
  return BUILTIN_PORT_BOUNDS[t.name] ?? null
}

/** Wrap an expression in the bounds-enforcing op chain.
 *
 *  We emit the parser-level `call(nameRef('<op>'), [...])` shape rather
 *  than the elaborator-level `{op: '<op>', args: [...]}` shape so that
 *  parse → print → re-parse converges: the printer prints
 *  `clamp(X, lo, hi)` and the re-parse sees that as a generic call.
 *  The elaborator turns both `call(clamp, ...)` and `{op:'clamp',...}`
 *  into the same resolved op, so semantics are unchanged.
 *
 *  Idempotent: if `expr` is already a wrapper matching `bounds` exactly,
 *  returns it unchanged. */
function wrapWithBound(expr: Expr, bounds: Bounds): Expr {
  const [lo, hi] = bounds
  if (alreadyWrapped(expr, bounds)) return expr
  if (lo !== null && hi !== null) {
    const node: Call = { op: 'call', callee: nameRef('clamp'), args: [expr, lo, hi] }
    return node
  }
  if (lo !== null) {
    const cond: BinaryOp = { op: 'gt', args: [expr, lo] }
    const node: Call = { op: 'call', callee: nameRef('select'), args: [cond, expr, lo] }
    return node
  }
  if (hi !== null) {
    const cond: BinaryOp = { op: 'lt', args: [expr, hi] }
    const node: Call = { op: 'call', callee: nameRef('select'), args: [cond, expr, hi] }
    return node
  }
  return expr  // [null, null] — no enforcement to insert
}

/** True if `expr` already enforces the same `bounds`. Recognizes both
 *  the direct-op shape (`{op:'clamp', args:[X, lo, hi]}`) the elaborator
 *  produces and the parser's call shape (`call(nameRef('clamp'), ...)`).
 *  Used to make `lowerBoundsToClamps` idempotent so parse → print →
 *  parse round-trips converge. */
function alreadyWrapped(expr: Expr, bounds: Bounds): boolean {
  if (typeof expr !== 'object' || expr === null || Array.isArray(expr)) return false
  const e = expr as unknown as Record<string, unknown>
  const args = e.args as unknown[] | undefined
  if (!Array.isArray(args)) return false

  const [lo, hi] = bounds
  const callee = e.callee as { op?: unknown; name?: unknown } | undefined
  const isCallTo = (name: string) =>
    e.op === 'call' && callee?.op === 'nameRef' && callee.name === name

  // Two-sided clamp.
  if ((e.op === 'clamp' || isCallTo('clamp')) && args.length === 3) {
    return args[1] === lo && args[2] === hi
  }
  // One-sided select(gt(X, lo), X, lo) / select(lt(X, hi), X, hi).
  if ((e.op === 'select' || isCallTo('select')) && args.length === 3) {
    const cond = args[0] as { op?: unknown; args?: unknown[] } | undefined
    if (lo !== null && hi === null
        && cond?.op === 'gt' && Array.isArray(cond.args)
        && cond.args.length === 2 && cond.args[1] === lo
        && args[2] === lo) return true
    if (hi !== null && lo === null
        && cond?.op === 'lt' && Array.isArray(cond.args)
        && cond.args.length === 2 && cond.args[1] === hi
        && args[2] === hi) return true
  }
  return false
}

/** Run lowering on every program in the tree (recurse into nested
 *  `programDecl` body decls). Mutates the input — port specs lose their
 *  `bounds` field and matching body assigns get clamp wrappers. */
export function lowerBoundsToClamps(prog: Program): Program {
  // Recurse into nested programs first. Body decls of op `programDecl`
  // carry an inner `program: Program`. The outer pass operates on
  // `prog` afterward so its body sees lowered nested forms.
  for (const decl of prog.body.decls ?? []) {
    if (decl.op === 'programDecl' && decl.program) {
      lowerBoundsToClamps(decl.program)
    }
  }

  // Inputs: wrap `default` if a default exists; bounded inputs without
  // defaults drop their bounds silently (documentation-only).
  for (const portSpec of (prog.ports?.inputs ?? [])) {
    const spec = liftSpec(portSpec)
    if (spec === null) continue
    const bounds = effectiveBounds(spec)
    if (bounds && spec.default !== undefined) {
      spec.default = wrapWithBound(spec.default, bounds)
    }
    // Always strip the explicit bounds field so downstream code can't
    // observe it. (Built-in alias bounds are looked up by alias name —
    // not on the spec — and don't survive past this pass either.)
    delete spec.bounds
  }

  // Outputs: find the matching `outputAssign` and wrap its expr.
  const outputBoundsByName = new Map<string, Bounds>()
  for (const portSpec of (prog.ports?.outputs ?? [])) {
    const spec = liftSpec(portSpec)
    if (spec === null) continue
    const bounds = effectiveBounds(spec)
    if (bounds) outputBoundsByName.set(spec.name, bounds)
    delete spec.bounds
  }

  if (outputBoundsByName.size > 0) {
    for (const assign of (prog.body.assigns ?? [])) {
      if (assign.op !== 'outputAssign') continue
      const bounds = outputBoundsByName.get(assign.name)
      if (!bounds) continue
      assign.expr = wrapWithBound(assign.expr, bounds)
    }
  }

  return prog
}

/** Bare-string ports cannot carry bounds; lift to a spec object only when
 *  it already is one. (Stripping is a no-op for bare strings.) */
function liftSpec(p: ProgramPort): ProgramPortSpec | null {
  return typeof p === 'string' ? null : p
}
