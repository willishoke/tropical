/**
 * wire_program.ts — lift a wire ExprNode to a ResolvedProgram.
 *
 * A wire is a `ref(instance, output)` and a transformation expression
 * that turns one or more such refs into a value consumed by a downstream
 * instance. Today wires are handled by a parallel mini-compiler
 * (`translateNode` in `compile_session_slotted_helpers.ts`) that knows
 * only a subset of expression shapes. Under the fractal architecture
 * (M11 Phase 5), wires lift to anonymous programs and flow through the
 * same per-program path as user-authored types.
 *
 * The lift is precise: given an `ExprNode` `e` with free refs `r₁..rₖ`,
 * we synthesize a program shape-identical to one a user could have
 * written:
 *
 *   program __wire_${i}(r₁_out, ..., rₖ_out) -> (out) {
 *     out = <translated e>
 *   }
 *
 * Every site in the input expression that referenced `r_j` becomes a
 * read of the corresponding `InputDecl`. Params and triggers become
 * private `ParamDecl`s in the lifted program's body (the param-name
 * unification with the surrounding session happens at runtime via the
 * FFI handle, not at the IR level). Session-level `delay()` becomes a
 * private synthetic `DelayDecl` — once the session-level strata pipeline
 * runs `traceCycles` over the inter-instance graph, all inter-program
 * cycles are explicit, so wire-internal delay is uniform with delay
 * anywhere else.
 *
 * Both exports are pure, total, and stateless. No engine, no FFI, no
 * session. They are the foundational lift; Phase 4's partitioner and
 * Phase 5's materializer consume them.
 */

import type { ExprNode } from '../expr.js'
import type {
  ResolvedProgram, ResolvedExpr,
  InputDecl, OutputDecl, ParamDecl, DelayDecl,
  BodyDecl,
  ResolvedBlock,
} from './nodes.js'
import {
  type InstanceName, type PortName, type PortRef,
  instanceName as makeInstanceName,
  portName as makePortName,
  portRef as makePortRef,
  portRefKey, rawName,
} from './branded_names.js'

// ─── Op sets ────────────────────────────────────────────────────────────────
// Mirror the sets in `materialize_session.ts:translateExpr`. Keep these in
// sync if either changes; the two should agree on which ops a wire-form
// expression may carry.

const BINARY_OPS: ReadonlySet<string> = new Set([
  'add', 'sub', 'mul', 'div', 'mod',
  'lt', 'lte', 'gt', 'gte', 'eq', 'neq',
  'and', 'or',
  'bitAnd', 'bitOr', 'bitXor', 'lshift', 'rshift',
  'floorDiv', 'ldexp',
])

const UNARY_OPS: ReadonlySet<string> = new Set([
  'neg', 'not', 'bitNot',
  'sqrt', 'abs', 'floor', 'ceil', 'round',
  'floatExponent', 'toInt', 'toBool', 'toFloat',
])

const TERNARY_OPS: ReadonlySet<string> = new Set(['clamp', 'select', 'arraySet'])

// ─── Free-variable scan ────────────────────────────────────────────────────

/** Walk an `ExprNode` tree and collect every reference to an instance
 *  output. Returns deduplicated `PortRef`s; the iteration order matches
 *  first-encounter order in the tree. */
export function freeRefs(expr: ExprNode): ReadonlySet<PortRef> {
  const byKey = new Map<string, PortRef>()

  const walk = (e: ExprNode): void => {
    if (typeof e === 'number' || typeof e === 'boolean') return
    if (Array.isArray(e)) {
      for (const item of e) walk(item)
      return
    }
    if (typeof e !== 'object' || e === null) return

    const obj = e as Record<string, unknown>
    const op = obj.op

    if (op === 'ref') {
      const instStr = obj.instance
      const outVal = obj.output
      if (typeof instStr !== 'string') {
        throw new Error(`freeRefs: ref node missing string 'instance' field`)
      }
      if (typeof outVal !== 'string') {
        // Numeric output indices are post-elaboration; wire-form is string.
        throw new Error(
          `freeRefs: ref(${instStr}, ${String(outVal)}) — output must be a ` +
          `string port name in wire-form, got ${typeof outVal}`,
        )
      }
      const ref = makePortRef(makeInstanceName(instStr), makePortName(outVal))
      const key = portRefKey(ref)
      if (!byKey.has(key)) byKey.set(key, ref)
      return
    }

    // Generic structural recursion. Wire-form ops carry children at
    // `args` (positional) and/or `items` (inline array literals). Other
    // shapes (let, fold, match, etc.) don't appear in wire expressions
    // and aren't recursed into — `liftWireToProgram` will reject them
    // at translate time anyway.
    if (Array.isArray(obj.args)) {
      for (const a of obj.args as ExprNode[]) walk(a)
    }
    if (Array.isArray(obj.items)) {
      for (const a of obj.items as ExprNode[]) walk(a)
    }
  }

  walk(expr)
  return new Set(byKey.values())
}

// ─── Lift to ResolvedProgram ───────────────────────────────────────────────

interface TranslateContext {
  /** Map from canonical PortRef key to the InputDecl that represents it. */
  readonly refToInput: ReadonlyMap<string, InputDecl>
  /** Param/Trigger decls accumulated during translation. Keyed by param
   *  name. Mutated as the translator encounters new refs. */
  readonly paramDecls: Map<string, ParamDecl>
  /** Synthetic DelayDecls created from `delay()` expressions. */
  readonly syntheticDelays: DelayDecl[]
}

/** Lift a wire `ExprNode` to a `ResolvedProgram` with the given synthesized
 *  name. The synthesized program has:
 *
 *  - one `InputDecl` per `PortRef` in `freeRefSet` (deterministic order:
 *    sorted by canonical `instance:port` key)
 *  - one `OutputDecl` named `out`
 *  - one `outputAssign` binding `out` to the translated expression
 *  - inline `ParamDecl`s for any `param`/`trigger` refs in the expression
 *  - inline `DelayDecl`s for any `delay()` calls in the expression
 *
 *  Pure, total. Output is shape-identical to a user-authored single-decl
 *  program; the per-program strata pipeline accepts it without
 *  modification. */
export function liftWireToProgram(
  expr: ExprNode,
  freeRefSet: ReadonlySet<PortRef>,
  synthName: InstanceName,
): ResolvedProgram {
  // Sort refs by canonical key so input ordering is deterministic
  // across calls (the same wire produces the same program shape).
  const sortedRefs = Array.from(freeRefSet).sort((a, b) =>
    portRefKey(a).localeCompare(portRefKey(b)),
  )

  const inputDecls: InputDecl[] = []
  const refToInput = new Map<string, InputDecl>()
  for (const ref of sortedRefs) {
    // Name the input deterministically from the ref. Double-underscore
    // separator avoids collisions with user port names (which can't
    // contain `__` since dots aren't allowed in PortName and `__` is
    // a stable, distinctive infix).
    const inputName = `${rawName(ref.instance).replace(/\./g, '_')}__${rawName(ref.port)}`
    const decl: InputDecl = { op: 'inputDecl', name: inputName }
    inputDecls.push(decl)
    refToInput.set(portRefKey(ref), decl)
  }

  const outputDecl: OutputDecl = { op: 'outputDecl', name: 'out' }

  const ctx: TranslateContext = {
    refToInput,
    paramDecls: new Map(),
    syntheticDelays: [],
  }

  const translated = translateExpr(expr, ctx)

  const bodyDecls: BodyDecl[] = [
    ...ctx.paramDecls.values(),
    ...ctx.syntheticDelays,
  ]

  const body: ResolvedBlock = {
    op: 'block',
    decls: bodyDecls,
    assigns: [{ op: 'outputAssign', target: outputDecl, expr: translated }],
  }

  return {
    op: 'program',
    name: rawName(synthName),
    typeParams: [],
    ports: {
      inputs:   inputDecls,
      outputs:  [outputDecl],
      typeDefs: [],
    },
    body,
  }
}

// ─── Internal: ExprNode → ResolvedExpr ─────────────────────────────────────

function translateExpr(expr: ExprNode, ctx: TranslateContext): ResolvedExpr {
  if (typeof expr === 'number')  return expr
  if (typeof expr === 'boolean') return expr
  if (Array.isArray(expr)) {
    return expr.map(e => translateExpr(e, ctx)) as ResolvedExpr
  }
  if (typeof expr !== 'object' || expr === null) {
    throw new Error(`liftWireToProgram: invalid expr value: ${JSON.stringify(expr)}`)
  }

  const obj = expr as Record<string, unknown>
  const op = obj.op
  if (typeof op !== 'string') {
    throw new Error(`liftWireToProgram: expression missing op tag`)
  }

  // ── Refs to instance outputs → inputRefs on the lifted program ──
  if (op === 'ref') {
    const instStr = obj.instance as string
    const outStr = obj.output as string
    const ref = makePortRef(makeInstanceName(instStr), makePortName(outStr))
    const inputDecl = ctx.refToInput.get(portRefKey(ref))
    if (inputDecl === undefined) {
      // Should not happen: freeRefs collected this ref. Either the
      // caller passed a stale freeRefSet, or the expression mutated
      // between scanning and lifting.
      throw new Error(
        `liftWireToProgram: ref ${portRefKey(ref)} not in freeRefSet — ` +
        `pass the same set returned by freeRefs(expr)`,
      )
    }
    return { op: 'inputRef', decl: inputDecl }
  }

  // ── Param/trigger refs → inline ParamDecls in the lifted program ──
  if (op === 'param' || op === 'paramExpr') {
    return paramOrTriggerRef(obj.name as string, 'param', ctx)
  }
  if (op === 'trigger' || op === 'triggerParamExpr') {
    return paramOrTriggerRef(obj.name as string, 'trigger', ctx)
  }

  // ── Sentinels ──
  if (op === 'sampleRate')  return { op: 'sampleRate' }
  if (op === 'sampleIndex') return { op: 'sampleIndex' }

  // ── Binary / unary / ternary ──
  if (BINARY_OPS.has(op)) {
    const args = (obj.args as ExprNode[]).map(a => translateExpr(a, ctx))
    return { op, args: [args[0], args[1]] } as ResolvedExpr
  }
  if (UNARY_OPS.has(op)) {
    const args = (obj.args as ExprNode[]).map(a => translateExpr(a, ctx))
    return { op, args: [args[0]] } as ResolvedExpr
  }
  if (TERNARY_OPS.has(op)) {
    const args = (obj.args as ExprNode[]).map(a => translateExpr(a, ctx))
    return { op, args: [args[0], args[1], args[2]] } as ResolvedExpr
  }

  // ── Index ──
  if (op === 'index') {
    const args = (obj.args as ExprNode[]).map(a => translateExpr(a, ctx))
    return { op: 'index', args: [args[0], args[1]] }
  }

  // ── Inline array literal ──
  if (op === 'array') {
    const items = (obj.items as ExprNode[]).map(item => translateExpr(item, ctx))
    return items as ResolvedExpr
  }

  // ── Session-level delay → synthetic DelayDecl + delayRef ──
  if (op === 'delay') {
    const argsArr = obj.args as ExprNode[]
    if (!Array.isArray(argsArr) || argsArr.length !== 1) {
      throw new Error(`liftWireToProgram: delay requires args: [expr], got ${JSON.stringify(argsArr)}`)
    }
    const update = translateExpr(argsArr[0], ctx)
    const init = typeof obj.init === 'number' ? obj.init : 0
    const decl: DelayDecl = {
      op: 'delayDecl',
      name: `__sd${ctx.syntheticDelays.length}`,
      update,
      init,
    }
    ctx.syntheticDelays.push(decl)
    return { op: 'delayRef', decl }
  }

  throw new Error(`liftWireToProgram: unhandled wire-form op '${op}'`)
}

function paramOrTriggerRef(
  name: string,
  kind: 'param' | 'trigger',
  ctx: TranslateContext,
): ResolvedExpr {
  let decl = ctx.paramDecls.get(name)
  if (decl === undefined) {
    decl = { op: 'paramDecl', name, kind }
    ctx.paramDecls.set(name, decl)
  } else if (decl.kind !== kind) {
    throw new Error(
      `liftWireToProgram: param/trigger name collision on '${name}' ` +
      `(declared as '${decl.kind}', ref demands '${kind}')`,
    )
  }
  return { op: 'paramRef', decl }
}
