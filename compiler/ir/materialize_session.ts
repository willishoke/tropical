/**
 * compiler/ir/materialize_session.ts — Session → ResolvedProgram materialization.
 *
 * `materializeSessionToResolvedIR(session)` turns a session — a partially-typed
 * graph of `Instance`s plus session-keyed wiring `ExprNode`s plus
 * `dac.out` graph outputs — into a synthetic top-level `ResolvedProgram`,
 * then runs the strata pipeline. The result feeds either the JIT
 * (`compileSession`) or the pure-TS interpreter (`interpret_resolved`).
 *
 * Per PHASE_D_PLAN §5e, this is the highest-risk single piece of code in
 * Phase D: the session represents a partially-typed graph; the resolved IR
 * represents a fully-typed graph; the materializer is the coproduct
 * injection between the two universes.
 *
 * Invariants (per category-theorist review §5e):
 *  - Zero scope analysis. The translator does no name resolution other
 *    than `instanceRegistry.get(instanceName)` and
 *    `findOutput(instance, outputName)`. Every reference becomes a
 *    direct decl-identity ResolvedExpr ref.
 *  - Coproduct injection: instance types come pre-resolved (resolved
 *    registry / specialization cache); the materializer only stitches
 *    them into a top-level shell.
 *  - Decl creation is bounded: the materializer creates fresh
 *    `DelayDecl` objects for session-level `delay()` nodes and fresh
 *    `OutputDecl` objects for graph_outputs. Nothing else.
 */

import type {
  ResolvedProgram, ResolvedExpr, ResolvedExprOp, ResolvedBlock,
  ResolvedProgramPorts,
  InputDecl, OutputDecl, RegDecl, ParamDecl, DelayDecl, InstanceDecl,
  BodyDecl, BodyAssign, OutputAssign,
  TypeParamDecl,
} from './nodes.js'
import type { ExprNode } from '../expr.js'
import type { SessionState } from '../session.js'
import type { Instance } from '../program_types.js'
import { strataPipeline } from './strata.js'
import { specializeProgram } from './specialize.js'
import { cloneResolvedProgram } from './clone.js'

/**
 * Run the session-to-ResolvedProgram materialization pipeline end-to-end:
 * synthesize a top-level program, route gate expressions through strata
 * for nestedOut inlining, run the strata pipeline, post-strata wrap
 * gateable lifted decls. Returns the post-strata `ResolvedProgram` ready
 * for either `compileResolved` (→ tropical_plan_4 for the JIT) or the
 * pure-TS interpreter (D3 D3-b).
 *
 * Shared by `compileSession` and `interpretSession` so both backends see
 * the exact same IR — the property `jit_interp_equiv` rests on.
 */
export function materializeSessionToResolvedIR(session: SessionState): ResolvedProgram {
  return materializeSessionForEmit(session).lowered
}

/** Variant of `materializeSessionToResolvedIR` that also exposes the
 *  ParamDecl map so `compileSession` can look up FFI handles per decl. */
export function materializeSessionForEmit(session: SessionState): {
  lowered: ResolvedProgram
  paramDecls: Map<string, ParamDecl>
} {
  const ctx = makeContext(session)
  const synthetic = materializeSessionInner(session, ctx)
  const gateableInstances = ctx.gateableInstances

  // For each gateable instance, append a synthetic outputDecl + outputAssign
  // carrying its gate expression. This routes the gate expressions through
  // `strataPipeline` so any `nestedOut` refs to other instances get inlined
  // alongside the rest. Post-strata, we read the inlined gate back off the
  // synthetic output, then strip both the output decl and assign before
  // passing to `compileResolved` so the gate doesn't appear in the plan's
  // audio outputs.
  const gateOutputDecls = new Map<string, OutputDecl>()
  for (const [instName, gate] of gateableInstances) {
    const gateOutDecl: OutputDecl = { op: 'outputDecl', name: `__gate__${instName}` }
    synthetic.ports.outputs.push(gateOutDecl)
    synthetic.body.assigns.push({ op: 'outputAssign', target: gateOutDecl, expr: gate })
    gateOutputDecls.set(instName, gateOutDecl)
  }

  const lowered = strataPipeline(synthetic)

  // Read back the post-strata inlined gate expressions by name (strata
  // doesn't rename OutputDecls but it may rebuild the program shell,
  // breaking object identity — match on the `__gate__` prefix instead).
  const inlinedGates = new Map<string, ResolvedExpr>()
  const synthOutputNames = new Set<string>()
  for (const instName of gateableInstances.keys()) synthOutputNames.add(`__gate__${instName}`)
  for (const a of lowered.body.assigns) {
    if (a.op !== 'outputAssign') continue
    if (!('op' in a.target)) continue
    if (a.target.op !== 'outputDecl') continue
    if (!a.target.name.startsWith('__gate__')) continue
    const instName = a.target.name.slice('__gate__'.length)
    inlinedGates.set(instName, a.expr)
  }
  // Strip the synthetic outputs and assigns before lowering.
  if (synthOutputNames.size > 0) {
    lowered.ports.outputs = lowered.ports.outputs.filter(o => !synthOutputNames.has(o.name))
    lowered.body.assigns = lowered.body.assigns.filter(a => {
      if (a.op !== 'outputAssign') return true
      if (!('op' in a.target)) return true
      return !synthOutputNames.has(a.target.name)
    })
  }

  if (inlinedGates.size > 0) applyGateableWraps(lowered, inlinedGates)
  return { lowered, paramDecls: ctx.paramDecls }
}

/** Wrap every lifted reg/delay update and output expression whose
 *  origin is a gateable session instance with `select(gate, expr,
 *  fallback)`. Identifies origin via the `_liftedFrom` provenance tag
 *  that `inlineInstances:liftClonedBody` stamps onto each lifted decl
 *  — replaces the §2.3 D7 name-prefix anti-pattern.
 *
 *  Synthetic delays from `traceCycles` are tagged
 *  `_liftedFrom: 'synthetic'`; they don't belong to any gateable
 *  instance and are skipped. */
function applyGateableWraps(
  prog: ResolvedProgram,
  gateableInstances: ReadonlyMap<string, ResolvedExpr>,
): void {
  /** Look up the gate for a decl based on its `_liftedFrom` tag. Returns
   *  null when the decl isn't from a gateable instance (or not lifted at
   *  all — outer-program decls untagged). */
  const gateFor = (decl: { _liftedFrom?: string }): ResolvedExpr | null => {
    if (decl._liftedFrom === undefined) return null
    if (decl._liftedFrom === 'synthetic') return null
    const gate = gateableInstances.get(decl._liftedFrom)
    return gate === undefined ? null : gate
  }

  // Skip an expression that was already wrapped pre-strata (the
  // gateable instance's OWN decls had `select(gate, raw, fallback)`
  // applied by `wrapTypeOutputsPreStrata` and strata's input
  // substitution embedded the same `gate` object identity).
  const alreadyWrapped = (expr: ResolvedExpr, gate: ResolvedExpr): boolean => {
    return typeof expr === 'object' && expr !== null && !Array.isArray(expr)
      && expr.op === 'select' && expr.args[0] === gate
  }

  // Wrap nextUpdate assigns whose target reg/delay was lifted from a
  // gateable instance and isn't already wrapped (sub-instance decls
  // didn't exist pre-strata).
  for (const a of prog.body.assigns) {
    if (a.op !== 'nextUpdate') continue
    const gate = gateFor(a.target)
    if (gate === null) continue
    if (alreadyWrapped(a.expr, gate)) continue
    const fallback: ResolvedExpr = a.target.op === 'regDecl'
      ? { op: 'regRef', decl: a.target as RegDecl }
      : { op: 'delayRef', decl: a.target as DelayDecl }
    a.expr = { op: 'select', args: [gate, a.expr, fallback] }
  }

  // Wrap delay decls' decl.update field for delays without a parallel
  // nextUpdate. Same skip-if-already-wrapped logic.
  for (const d of prog.body.decls) {
    if (d.op !== 'delayDecl') continue
    const gate = gateFor(d)
    if (gate === null) continue
    const haveNextUpdate = prog.body.assigns.some(
      a => a.op === 'nextUpdate' && a.target === d,
    )
    if (haveNextUpdate) continue
    if (alreadyWrapped(d.update, gate)) continue
    const fallback: ResolvedExpr = { op: 'delayRef', decl: d }
    d.update = { op: 'select', args: [gate, d.update, fallback] }
  }

  // dac-target outputAssigns from gateable instances are ALREADY
  // wrapped: pre-strata, the gateable type's outputAssigns got
  // `select(__gate__, raw, 0)` applied. Strata's nestedOut substitution
  // carried that wrapped form into wherever a1.out was referenced —
  // including the synthetic dac.out outputAssign (which is just
  // `nestedOut(a1, out)` syntactically, replaced inline). So no
  // additional output wrapping is needed here. The wraps composed by
  // strata also flow into other gateable instances' gate expressions
  // (so `a2`'s gate `a1.out > 1.5` reads the wrapped a1.out, matching
  // legacy `flatten.ts:wrapOutput` semantics).
}

function makeContext(session: SessionState): MaterializeContext {
  return {
    instanceDecls:       new Map(),
    paramDecls:          new Map(),
    syntheticDelayDecls: [],
    exprMemo:            new WeakMap(),
    gateableInstances:   new Map(),
    session,
  }
}

/** Test-only export: expose the synthetic top-level builder so equiv
 *  diagnostics can inspect the IR before strata runs. */
export const _materializeSessionForTesting = materializeSession

// ─────────────────────────────────────────────────────────────────────────────
// Session → synthetic ResolvedProgram
// ─────────────────────────────────────────────────────────────────────────────

interface MaterializeContext {
  /** Fresh InstanceDecl per session instance name. Identity-keyed; shared
   *  across the wiring translation so refs map to the same object. */
  instanceDecls: Map<string, InstanceDecl>
  /** ParamDecl per param/trigger name. Created lazily as wiring expressions
   *  reference them; same identity reused across all references. */
  paramDecls: Map<string, ParamDecl>
  /** Synthetic DelayDecls for session-level `delay()` nodes extracted
   *  from wiring expressions. The translator creates one DelayDecl per
   *  `delay` node; subsequent translations of the same identity-shared
   *  ExprNode reuse the same decl via exprMemo. */
  syntheticDelayDecls: DelayDecl[]
  /** Identity memoization: shared session ExprNode objects produce shared
   *  ResolvedExpr objects so downstream CSE memo (in resolvedToSlotted +
   *  emit_numeric) treats them as identical. */
  exprMemo: WeakMap<object, ResolvedExpr>
  /** For each gateable session instance, the resolved gate expression to
   *  apply post-strata. Wrapping happens after `inlineInstances` has
   *  lifted all sub-instance regs/delays into the synthetic top-level,
   *  so every register in the gateable lineage gets wrapped. The legacy
   *  `flatten.ts` wraps at the flat-register level for the same reason. */
  gateableInstances: Map<string, ResolvedExpr>
  /** Direct lookup into the session for type-resolution + port lookup. */
  session: SessionState
}

function materializeSession(session: SessionState): ResolvedProgram {
  return materializeSessionInner(session, makeContext(session))
}

function materializeSessionInner(session: SessionState, ctx: MaterializeContext): ResolvedProgram {

  // 1. Build InstanceDecl per session instance, in iteration order.
  //    Each instance's `type` is resolved via the session's resolved
  //    registry / generic templates; `inputs` is filled in step 2.
  for (const [name, inst] of session.instanceRegistry) {
    const decl = buildInstanceDecl(name, inst, ctx)
    ctx.instanceDecls.set(name, decl)
  }

  // 2. Translate session.inputExprNodes → InstanceDecl.inputs entries.
  //    Key shape is `${instance}:${input}`.
  for (const [key, expr] of session.inputExprNodes) {
    const colon = key.indexOf(':')
    if (colon < 0) continue
    const instName = key.slice(0, colon)
    const inputName = key.slice(colon + 1)
    const instDecl = ctx.instanceDecls.get(instName)
    if (instDecl === undefined) continue  // stale wiring entry
    const port = instDecl.type.ports.inputs.find(p => p.name === inputName)
    if (port === undefined) {
      throw new Error(
        `compileSession: instance '${instName}' has no input port '${inputName}' on type '${instDecl.type.name}'.`,
      )
    }
    const value = translateExpr(expr, ctx)
    instDecl.inputs.push({ port, value })
  }

  // 3. Build OutputDecls + OutputAssigns from session.graphOutputs (dac.out
  //    wires). Each graph_output entry produces one OutputDecl in
  //    ports.outputs (named after `${instance}.${output}`) and one
  //    OutputAssign whose expr reads that instance's output.
  const outputDecls: OutputDecl[] = []
  const outputAssigns: OutputAssign[] = []
  for (const go of session.graphOutputs) {
    const instDecl = ctx.instanceDecls.get(go.instance)
    if (instDecl === undefined) {
      throw new Error(`compileSession: graph output references unknown instance '${go.instance}'.`)
    }
    const outDecl = instDecl.type.ports.outputs.find(p => p.name === go.output)
    if (outDecl === undefined) {
      throw new Error(
        `compileSession: instance '${go.instance}' has no output port '${go.output}' on type '${instDecl.type.name}'.`,
      )
    }
    const sessionOutput: OutputDecl = {
      op: 'outputDecl',
      name: `${go.instance}.${go.output}`,
    }
    if (outDecl.type !== undefined) sessionOutput.type = outDecl.type
    outputDecls.push(sessionOutput)
    const ref: ResolvedExprOp = { op: 'nestedOut', instance: instDecl, output: outDecl }
    outputAssigns.push({ op: 'outputAssign', target: sessionOutput, expr: ref })
  }

  // 4. Assemble body decls in source order: synthetic delays first
  //    (so their slots claim low indices, matching legacy convention
  //    where session delays come before instance regs), then instance
  //    decls, then params.
  const bodyDecls: BodyDecl[] = []
  for (const decl of ctx.syntheticDelayDecls)   bodyDecls.push(decl)
  for (const decl of ctx.instanceDecls.values()) bodyDecls.push(decl)
  for (const decl of ctx.paramDecls.values())    bodyDecls.push(decl)

  const bodyAssigns: BodyAssign[] = outputAssigns

  const block: ResolvedBlock = {
    op: 'block',
    decls:   bodyDecls,
    assigns: bodyAssigns,
  }

  const ports: ResolvedProgramPorts = {
    inputs:   [] as InputDecl[],
    outputs:  outputDecls,
    typeDefs: [],
  }

  const prog: ResolvedProgram = {
    op: 'program',
    name: '__session__',
    typeParams: [] as TypeParamDecl[],
    ports,
    body: block,
  }
  return prog
}

// ─────────────────────────────────────────────────────────────────────────────
// Build InstanceDecl
// ─────────────────────────────────────────────────────────────────────────────

function buildInstanceDecl(
  name: string,
  inst: Instance,
  ctx: MaterializeContext,
): InstanceDecl {
  const session = ctx.session
  const baseTypeName = inst.baseTypeName
  const rawTypeArgs = inst.typeArgs ?? {}

  // Try non-generic first (concrete type registered with this base name).
  const registered = session.resolvedRegistry.get(baseTypeName)
  let resolvedType: ResolvedProgram | undefined
  let typeArgsList: Array<{ param: TypeParamDecl; value: number }> = []

  if (registered !== undefined && registered.typeParams.length === 0) {
    resolvedType = registered
  } else {
    // Generic case: pull the template, build TypeParamDecl-keyed subst,
    // re-specialize. Mirrors session.ts:resolveProgramType.
    const template = session.genericTemplatesResolved.get(baseTypeName)
      ?? registered
    if (template === undefined) {
      throw new Error(
        `compileSession: instance '${name}' has type '${baseTypeName}' which is not registered as a resolved program.`,
      )
    }
    const subst = new Map<TypeParamDecl, number>()
    for (const [paramName, value] of Object.entries(rawTypeArgs)) {
      const decl = template.typeParams.find(p => p.name === paramName)
      if (!decl) {
        throw new Error(
          `compileSession: instance '${name}' carries unknown type-arg '${paramName}' for '${baseTypeName}'.`,
        )
      }
      subst.set(decl, value)
      typeArgsList.push({ param: decl, value })
    }
    // Fill in defaults for any unspecified param.
    for (const decl of template.typeParams) {
      if (subst.has(decl)) continue
      if (decl.default === undefined) {
        throw new Error(
          `compileSession: instance '${name}' missing required type-arg '${decl.name}' for '${baseTypeName}' (no default).`,
        )
      }
      subst.set(decl, decl.default)
      typeArgsList.push({ param: decl, value: decl.default })
    }
    resolvedType = specializeProgram(template, subst)
  }

  // After buildInstanceDecl, `resolvedType` is fully specialized (empty
  // typeParams). `inlineInstances` will then short-circuit on the
  // empty-typeParams + empty-typeArgs path. Carrying typeArgsList through
  // would re-trigger specializeProgram against an already-specialized
  // template, which throws "type-arg X is not a declared type-param".
  const decl: InstanceDecl = {
    op: 'instanceDecl',
    name,
    type: resolvedType,
    typeArgs: [],
    inputs: [],
  }

  // Gateable: two-phase wrap.
  //
  // PRE-STRATA: wrap the type's outputAssigns + own regDecl/delayDecl
  // updates with `select(__gate__, raw, fallback)`. This is necessary
  // for *outputs* because strata's nestedOut substitution captures the
  // gateable instance's output expression at inline time — if we wrap
  // post-strata, other instances' wiring/gates that reference this
  // gateable's output have already captured the *un*wrapped form.
  //
  // POST-STRATA: wrap any decls lifted from sub-instances of this
  // gateable instance (matched by name prefix `${name}_`). These
  // didn't exist pre-strata.
  //
  // The gate is plumbed in as a synthetic `__gate__` input on the
  // cloned type so cloning doesn't have to handle outer-scope refs in
  // an embedded gate expression.
  if (inst.gateable) {
    if (inst.gateInput === undefined) {
      throw new Error(
        `compileSession: instance '${name}' is gateable but has no gateInput.`,
      )
    }
    const gateExpr = translateExpr(inst.gateInput, ctx)
    ctx.gateableInstances.set(name, gateExpr)
    wrapTypeOutputsPreStrata(decl, gateExpr)
  }

  return decl
}

/** Pre-strata wrap of a gateable instance's TYPE outputs and own
 *  reg/delay updates via a synthetic `__gate__` input. */
function wrapTypeOutputsPreStrata(decl: InstanceDecl, gateExpr: ResolvedExpr): void {
  const cloned = cloneResolvedProgram(decl.type)
  const gateInputDecl: InputDecl = { op: 'inputDecl', name: '__gate__' }
  cloned.ports.inputs.push(gateInputDecl)

  const gateRef: ResolvedExpr = { op: 'inputRef', decl: gateInputDecl }

  for (const a of cloned.body.assigns) {
    if (a.op === 'outputAssign') {
      a.expr = { op: 'select', args: [gateRef, a.expr, 0] }
    } else if (a.op === 'nextUpdate') {
      const fallback: ResolvedExpr = a.target.op === 'regDecl'
        ? { op: 'regRef', decl: a.target as RegDecl }
        : { op: 'delayRef', decl: a.target as DelayDecl }
      a.expr = { op: 'select', args: [gateRef, a.expr, fallback] }
    }
  }
  for (const d of cloned.body.decls) {
    if (d.op !== 'delayDecl') continue
    const haveNextUpdate = cloned.body.assigns.some(
      a => a.op === 'nextUpdate' && a.target === d,
    )
    if (haveNextUpdate) continue
    const fallback: ResolvedExpr = { op: 'delayRef', decl: d }
    d.update = { op: 'select', args: [gateRef, d.update, fallback] }
  }

  decl.type = cloned
  decl.inputs.push({ port: gateInputDecl, value: gateExpr })
}

// ─────────────────────────────────────────────────────────────────────────────
// ExprNode → ResolvedExpr translation
// ─────────────────────────────────────────────────────────────────────────────

const BINARY_OPS = new Set([
  'add', 'sub', 'mul', 'div', 'mod',
  'lt', 'lte', 'gt', 'gte', 'eq', 'neq',
  'and', 'or',
  'bitAnd', 'bitOr', 'bitXor', 'lshift', 'rshift',
  'floorDiv', 'ldexp',
])

const UNARY_OPS = new Set([
  'neg', 'not', 'bitNot',
  'sqrt', 'abs', 'floor', 'ceil', 'round',
  'floatExponent', 'toInt', 'toBool', 'toFloat',
])

const TERNARY_OPS = new Set(['clamp', 'select', 'arraySet'])

function translateExpr(expr: ExprNode, ctx: MaterializeContext): ResolvedExpr {
  if (typeof expr === 'number')  return expr
  if (typeof expr === 'boolean') return expr
  if (Array.isArray(expr)) {
    const cached = ctx.exprMemo.get(expr)
    if (cached !== undefined) return cached
    const out = expr.map(e => translateExpr(e, ctx)) as ResolvedExpr
    ctx.exprMemo.set(expr, out)
    return out
  }
  if (typeof expr !== 'object' || expr === null) {
    throw new Error(`compileSession: invalid expr value: ${JSON.stringify(expr)}`)
  }

  const cached = ctx.exprMemo.get(expr)
  if (cached !== undefined) return cached

  const obj = expr as Record<string, unknown>
  const op = obj.op
  if (typeof op !== 'string') {
    throw new Error(`compileSession: expression missing op tag: ${JSON.stringify(expr).slice(0, 100)}`)
  }

  const out = translateOpNode(obj, op, ctx)
  ctx.exprMemo.set(expr, out)
  return out
}

function translateOpNode(
  obj: Record<string, unknown>,
  op: string,
  ctx: MaterializeContext,
): ResolvedExpr {
  // ── Reference ops ─────────────────────────────────────────────────
  if (op === 'ref') {
    const instName = obj.instance as string
    const outputName = obj.output as string
    const instDecl = ctx.instanceDecls.get(instName)
    if (instDecl === undefined) {
      throw new Error(`compileSession: ref to unknown instance '${instName}'.`)
    }
    const outDecl = instDecl.type.ports.outputs.find(p => p.name === outputName)
    if (outDecl === undefined) {
      throw new Error(
        `compileSession: ref to '${instName}.${outputName}' — '${outputName}' is not a port on type '${instDecl.type.name}'.`,
      )
    }
    return { op: 'nestedOut', instance: instDecl, output: outDecl }
  }

  if (op === 'param' || op === 'paramExpr') {
    return paramRef(obj.name as string, 'param', ctx)
  }

  if (op === 'trigger' || op === 'triggerParamExpr') {
    return paramRef(obj.name as string, 'trigger', ctx)
  }

  if (op === 'sampleRate')  return { op: 'sampleRate' }
  if (op === 'sampleIndex') return { op: 'sampleIndex' }

  // ── Pass-through binary / unary / ternary ─────────────────────────
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

  if (op === 'index') {
    const args = (obj.args as ExprNode[]).map(a => translateExpr(a, ctx))
    return { op: 'index', args: [args[0], args[1]] }
  }

  // Array literal: `{op:'array', items:[...]}` → bare array (ResolvedExpr[]).
  // The resolved IR represents arrays as ResolvedExpr[]; the parser-level
  // wrapper drops away.
  if (op === 'array') {
    const items = (obj.items as ExprNode[]).map(item => translateExpr(item, ctx))
    return items as ResolvedExpr
  }

  // Session-level `delay()`: extract into a synthesized DelayDecl whose
  // `update` is the inner expression; the original delay node becomes a
  // delayRef. Init defaults to 0 (legacy convention). The decl is
  // appended to ctx.syntheticDelayDecls so it lands in body.decls.
  // exprMemo reuse means a shared `delay()` ExprNode produces a single
  // DelayDecl, matching legacy CSE.
  if (op === 'delay') {
    const update = translateExpr((obj.args as ExprNode[])[0], ctx)
    const init = typeof obj.init === 'number' ? obj.init : 0
    const decl: DelayDecl = { op: 'delayDecl', name: `__sd${ctx.syntheticDelayDecls.length}`, update, init }
    ctx.syntheticDelayDecls.push(decl)
    return { op: 'delayRef', decl }
  }

  // TODO: gateable subgraph wiring (source_tag)
  // TODO: broadcastTo / matmul (less common in patches; defer)

  throw new Error(`compileSession: unhandled wiring op '${op}' (TODO: extend translator coverage).`)
}

function paramRef(name: string, kind: 'param' | 'trigger', ctx: MaterializeContext): ResolvedExpr {
  let decl = ctx.paramDecls.get(name)
  if (decl === undefined) {
    decl = { op: 'paramDecl', name, kind }
    ctx.paramDecls.set(name, decl)
  } else if (decl.kind !== kind) {
    throw new Error(
      `compileSession: param/trigger name collision on '${name}' (declared as '${decl.kind}', ref demands '${kind}').`,
    )
  }
  return { op: 'paramRef', decl }
}
