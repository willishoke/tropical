/**
 * wire_program.ts — lift a wire ExprNode to a ResolvedProgram.
 *
 * A wire is a `ref(instance, output)` and a transformation expression
 * that turns one or more such refs into a value consumed by a downstream
 * instance. The fractal-architecture path lifts each wire to an
 * anonymous program that flows through the same per-program path as
 * user-authored types; the legacy parallel mini-compiler
 * (`translateNode` in `compile_session_slotted_helpers.ts`) still
 * handles wires whose shape is a strict subset of the fractal lift.
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
 * session. They are the foundational lift; `partition_recursive` and
 * the session materializer consume them.
 */

import type { ExprNode } from '../expr.js'
import type {
  ResolvedProgram, ResolvedExpr,
  InputDecl, OutputDecl, ParamDecl, RegDecl,
  BodyDecl,
  ResolvedBlock,
  InputIdx,
  PortType,
} from './nodes.js'
import { inputIdx, outputIdx, paramIdx, regIdx } from './nodes.js'
import {
  type InstanceName, type PortName, type PortRef, type WireKey,
  instanceName as makeInstanceName,
  portName as makePortName,
  portRef as makePortRef,
  wireKey, rawName,
} from './branded_names.js'
import { mkProgram } from './decl_tables.js'

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

/** Infer the lifted program's output port type from the top-level
 *  expression shape. Currently handles only array literals (bare
 *  arrays and `{op:'array'/'arrayLiteral', items:[…]}` shapes), which
 *  is the closed set `needsWireLift` triggers on — keep this in sync.
 *  Returns undefined for scalar-shaped expressions; the caller
 *  defaults to scalar `float` in that case.
 *
 *  Element type defaults to `'float'`. The element scalars of an
 *  array wire expression (which may be `ref`s, params, or
 *  arithmetic) all read as `float` at the session-translate boundary.
 *  Refinement (`int`/`bool` elements) is left to a later pass; the
 *  IR shape doesn't yet need that distinction at the lift layer. */
function inferOutputPortType(expr: ExprNode): PortType | undefined {
  if (Array.isArray(expr)) {
    return { kind: 'array', element: 'float', shape: [expr.length] }
  }
  if (typeof expr === 'object' && expr !== null) {
    const obj = expr as Record<string, unknown>
    if ((obj.op === 'array' || obj.op === 'arrayLiteral') && Array.isArray(obj.items)) {
      return { kind: 'array', element: 'float', shape: [(obj.items as unknown[]).length] }
    }
  }
  return undefined
}

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
      const key = wireKey(ref)
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
  /** Map from canonical wire key to the InputIdx that represents it.
   *  Indices are the position of the InputDecl in the synthesized
   *  program's ports.inputs[] (assigned at decl creation time). */
  readonly refToInputIdx: ReadonlyMap<WireKey, InputIdx>
  /** Param/Trigger decls accumulated during translation. Keyed by param
   *  name. Mutated as the translator encounters new refs. ParamIdx is
   *  derived from iteration position at lookup time. */
  readonly paramDecls: Map<string, ParamDecl>
  /** Synthetic RegDecls (post-Phase-0a: `update` populated, semantically
   *  a one-sample delay) created from `delay()` expressions. */
  readonly syntheticRegs: RegDecl[]
}

/** Lift a wire `ExprNode` to a `ResolvedProgram` with the given synthesized
 *  name. The synthesized program has:
 *
 *  - one `InputDecl` per `PortRef` in `freeRefSet` (deterministic order:
 *    sorted by canonical `instance:port` key)
 *  - one `OutputDecl` named `out`
 *  - one `outputAssign` binding `out` to the translated expression
 *  - inline `ParamDecl`s for any `param`/`trigger` refs in the expression
 *  - inline `RegDecl`s (with `update` populated) for any `delay()` calls
 *    in the expression — post-Phase-0a the unified state primitive
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
    wireKey(a).localeCompare(wireKey(b)),
  )

  const inputDecls: InputDecl[] = []
  const refToInputIdx = new Map<WireKey, InputIdx>()
  for (const ref of sortedRefs) {
    // Name the input deterministically from the ref. Double-underscore
    // separator avoids collisions with user port names (which can't
    // contain `__` since dots aren't allowed in PortName and `__` is
    // a stable, distinctive infix).
    const inputName = `${rawName(ref.instance).replace(/\./g, '_')}__${rawName(ref.port)}`
    const decl: InputDecl = { op: 'inputDecl', name: inputName }
    const i = inputIdx(inputDecls.length)
    inputDecls.push(decl)
    refToInputIdx.set(wireKey(ref), i)
  }

  // Infer the output port type from the top-level expression shape.
  // Array-shaped wires (bare `[...]` literals, or `{op:'array'/
  // 'arrayLiteral', items:[...]}`) need an array-typed output so the
  // session-level allocator sees the producer as an array source and
  // the consumer's input alias logic can bind to it. Untyped (default
  // `undefined`) means scalar `float` — fine for scalar wire
  // expressions, wrong for array literals.
  const outputType = inferOutputPortType(expr)
  const outputDecl: OutputDecl = outputType === undefined
    ? { op: 'outputDecl', name: 'out' }
    : { op: 'outputDecl', name: 'out', type: outputType }

  const ctx: TranslateContext = {
    refToInputIdx,
    paramDecls: new Map(),
    syntheticRegs: [],
  }

  const translated = translateExpr(expr, ctx)

  const bodyDecls: BodyDecl[] = [
    ...ctx.paramDecls.values(),
    ...ctx.syntheticRegs,
  ]

  const body: ResolvedBlock = {
    op: 'block',
    decls: bodyDecls,
    // Lifted programs have exactly one output ('out') at position 0.
    assigns: [{ op: 'outputAssign', target: outputIdx(0), expr: translated }],
  }

  return mkProgram({
    name: rawName(synthName),
    typeParams: [],
    ports: {
      inputs:   inputDecls,
      outputs:  [outputDecl],
      typeDefs: [],
    },
    body,
    // Wire-lifted programs translate session ExprNodes (scalar
    // arithmetic only) to ResolvedExpr — no combinators, no
    // let-bindings, no binders.
    binderCount: 0,
    // No InstanceDecls in the lifted body (wire expressions are pure
    // arithmetic over ports + regs), so the registry is empty.
    programRegistry: new Map(),
  })
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
    const i = ctx.refToInputIdx.get(wireKey(ref))
    if (i === undefined) {
      // Should not happen: freeRefs collected this ref. Either the
      // caller passed a stale freeRefSet, or the expression mutated
      // between scanning and lifting.
      throw new Error(
        `liftWireToProgram: ref ${wireKey(ref)} not in freeRefSet — ` +
        `pass the same set returned by freeRefs(expr)`,
      )
    }
    return { op: 'inputRef', idx: i }
  }

  // ── Param refs → inline ParamDecls in the lifted program ──
  if (op === 'param' || op === 'paramExpr'
      || op === 'trigger' || op === 'triggerParamExpr') {
    return paramRefIntoCtx(obj.name as string, ctx)
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

  // ── Session-level delay → synthetic RegDecl-with-update + regRef ──
  if (op === 'delay') {
    const argsArr = obj.args as ExprNode[]
    if (!Array.isArray(argsArr) || argsArr.length !== 1) {
      throw new Error(`liftWireToProgram: delay requires args: [expr], got ${JSON.stringify(argsArr)}`)
    }
    const update = translateExpr(argsArr[0], ctx)
    const init = typeof obj.init === 'number' ? obj.init : 0
    // body.decls = [...paramDecls, ...syntheticRegs]; only RegDecls
    // contribute to body.regs[]. So RegIdx for a synthetic reg = its
    // position within ctx.syntheticRegs.
    const newIdx = regIdx(ctx.syntheticRegs.length)
    const decl: RegDecl = {
      op: 'regDecl',
      name: `__sd${ctx.syntheticRegs.length}`,
      update,
      init,
    }
    ctx.syntheticRegs.push(decl)
    return { op: 'regRef', idx: newIdx }
  }

  throw new Error(`liftWireToProgram: unhandled wire-form op '${op}'`)
}

function paramRefIntoCtx(
  name: string,
  ctx: TranslateContext,
): ResolvedExpr {
  let decl = ctx.paramDecls.get(name)
  if (decl === undefined) {
    decl = { op: 'paramDecl', name }
    ctx.paramDecls.set(name, decl)
  }
  // ParamIdx is the position in ctx.paramDecls iteration order.
  let pi = -1
  let i = 0
  for (const n of ctx.paramDecls.keys()) {
    if (n === name) { pi = i; break }
    i++
  }
  return { op: 'paramRef', idx: paramIdx(pi) }
}
