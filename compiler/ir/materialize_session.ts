/**
 * materialize_session.ts — Session → ResolvedProgram materialization.
 *
 * Lifts a session — a partially-typed graph of `Instance`s plus
 * session-keyed wiring `ExprNode`s plus `dac.out` graph outputs — into
 * a synthetic top-level `ResolvedProgram` and runs the strata
 * pipeline.
 *
 * The JIT path (`compile_session_slotted.ts`) DOES NOT use this
 * materialization any more — it compiles each instance standalone and
 * the scheduler drives per-instance dispatch. This file is retained
 * as the **interpreter oracle**: `interpret_resolved.ts` walks the
 * resolved IR produced here, and `tests/equiv/jit_vs_interp` cross-checks the
 * two evaluators sample-for-sample.
 *
 * ## Alive semantics in the oracle
 *
 * The JIT realizes `aliveInput` by skipping the instance's kernel
 * when alive evaluates to false; the instance's output slot retains
 * its last write. The interpreter (single-program oracle) realizes
 * the same I/O semantics structurally via `select(alive, raw,
 * fallback)` wraps on the gated instance's outputs and own
 * reg/delay updates:
 *
 *   - **Output wrap** (pre-strata): `out = select(alive, raw_out, 0)`
 *     ensures consumers downstream see a held value, not the live
 *     computation, when alive is false.
 *   - **Reg/delay wrap** (pre-strata for own decls; post-strata for
 *     decls lifted by `inlineInstances` from nested sub-instances):
 *     `next_value = select(alive, raw_next, current_value)` ensures
 *     state freezes while alive is false.
 *
 * Inter-instance refs inside the alive expression are translated as
 * normal (current-sample reads). The JIT side reads slot values from
 * the start of the sample loop (previous-sample writes), so there is
 * a one-sample-delay discrepancy when an alive expression references
 * another instance's current-sample output. For most patches —
 * param-driven alive, self-referencing envelope alive — the
 * discrepancy is structural (alive depends on register state, which
 * is one-sample-delayed in both backends anyway). Tests that exercise
 * cross-instance current-sample alive references should drive
 * through the JIT path only.
 *
 * ## Invariants
 *
 *  - Zero scope analysis. The translator does no name resolution other
 *    than `instanceRegistry.get(instanceName)` and
 *    `findOutput(instance, outputName)`. Every reference becomes a
 *    direct decl-identity ResolvedExpr ref.
 *  - Decl creation is bounded: the materializer creates fresh
 *    `RegDecl` objects (with `update` populated) for session-level
 *    `delay()` nodes and fresh `OutputDecl` objects for graph_outputs.
 */

import type {
  ResolvedProgram, ResolvedExpr, ResolvedExprOp, ResolvedBlock,
  ResolvedProgramPorts,
  InputDecl, OutputDecl, RegDecl, ParamDecl, InstanceDecl,
  BodyDecl, BodyAssign, OutputAssign,
  TypeParamDecl,
} from './nodes.js'
import type { ExprNode } from '../expr.js'
import type { SessionState } from '../session.js'
import type { Instance } from '../program_types.js'
import { strataPipeline } from './strata.js'
import { specializeProgram } from './specialize.js'
import { cloneResolvedProgram } from './clone.js'

/** Run the session-to-ResolvedProgram materialization end-to-end. */
export function materializeSessionToResolvedIR(session: SessionState): ResolvedProgram {
  return materializeSessionForEmit(session).lowered
}

/** Variant that also exposes the ParamDecl map (paramHandle stitching
 *  is no longer needed on the JIT side post-PR-A, but the interpreter
 *  threads param values through this map). */
export function materializeSessionForEmit(session: SessionState): {
  lowered: ResolvedProgram
  paramDecls: Map<string, ParamDecl>
} {
  const ctx = makeContext(session)
  const synthetic = materializeSessionInner(session, ctx)
  const aliveInstances = ctx.aliveInstances

  // For each instance with explicit alive wiring, append a synthetic
  // outputDecl + outputAssign carrying its alive expression. This
  // routes the alive expressions through `strataPipeline` so any
  // `nestedOut` refs to other instances get inlined alongside the
  // rest. Post-strata, we read the inlined alive back off the
  // synthetic output, then strip both the output decl and assign
  // before passing to `compileResolved` so the alive expression
  // doesn't appear in the plan's audio outputs.
  const aliveOutputDecls = new Map<string, OutputDecl>()
  for (const [instName, alive] of aliveInstances) {
    const decl: OutputDecl = { op: 'outputDecl', name: `__alive__${instName}` }
    synthetic.ports.outputs.push(decl)
    synthetic.body.assigns.push({ op: 'outputAssign', target: decl, expr: alive })
    aliveOutputDecls.set(instName, decl)
  }

  const lowered = strataPipeline(synthetic)

  // Read back the post-strata inlined alive expressions by name.
  const inlinedAlives = new Map<string, ResolvedExpr>()
  const synthOutputNames = new Set<string>()
  for (const instName of aliveInstances.keys()) synthOutputNames.add(`__alive__${instName}`)
  for (const a of lowered.body.assigns) {
    if (a.op !== 'outputAssign') continue
    if (!('op' in a.target)) continue
    if (a.target.op !== 'outputDecl') continue
    if (!a.target.name.startsWith('__alive__')) continue
    const instName = a.target.name.slice('__alive__'.length)
    inlinedAlives.set(instName, a.expr)
  }
  // Strip the synthetic outputs and assigns.
  if (synthOutputNames.size > 0) {
    lowered.ports.outputs = lowered.ports.outputs.filter(o => !synthOutputNames.has(o.name))
    lowered.body.assigns = lowered.body.assigns.filter(a => {
      if (a.op !== 'outputAssign') return true
      if (!('op' in a.target)) return true
      return !synthOutputNames.has(a.target.name)
    })
  }

  if (inlinedAlives.size > 0) applyAliveWraps(lowered, inlinedAlives)
  return { lowered, paramDecls: ctx.paramDecls }
}

/** Wrap every lifted reg update whose origin is an alive-eligible
 *  session instance with `select(alive, raw, fallback)`. Identifies
 *  origin via the `_liftedFrom` provenance tag stamped by
 *  `inlineInstances:liftClonedBody`.
 *
 *  Synthetic regs from `traceCycles` are tagged
 *  `_liftedFrom: 'synthetic'` and don't belong to any instance —
 *  skipped here.
 *
 *  Post-Phase-0a: reg updates live on `decl.update`; NextUpdate
 *  body-assigns are gone. The wrap is applied directly to the decl. */
function applyAliveWraps(
  prog: ResolvedProgram,
  aliveInstances: ReadonlyMap<string, ResolvedExpr>,
): void {
  const aliveFor = (decl: { _liftedFrom?: string }): ResolvedExpr | null => {
    if (decl._liftedFrom === undefined) return null
    if (decl._liftedFrom === 'synthetic') return null
    const a = aliveInstances.get(decl._liftedFrom)
    return a === undefined ? null : a
  }

  // Skip an expression that was already wrapped pre-strata (the
  // alive instance's OWN decls had `select(alive, raw, fallback)`
  // applied by `wrapTypeOutputsForAlive`, and strata's input
  // substitution embedded the same `alive` object identity).
  const alreadyWrapped = (expr: ResolvedExpr, alive: ResolvedExpr): boolean => {
    return typeof expr === 'object' && expr !== null && !Array.isArray(expr)
      && expr.op === 'select' && expr.args[0] === alive
  }

  for (const d of prog.body.decls) {
    if (d.op !== 'regDecl') continue
    if (d.update === undefined) continue
    const alive = aliveFor(d)
    if (alive === null) continue
    if (alreadyWrapped(d.update, alive)) continue
    const fallback: ResolvedExpr = { op: 'regRef', decl: d }
    d.update = { op: 'select', args: [alive, d.update, fallback] }
  }

  // dac-target outputAssigns from alive-eligible instances are
  // ALREADY wrapped: pre-strata, the instance's own outputAssigns got
  // `select(__alive__, raw, 0)`. Strata's nestedOut substitution
  // carried that wrapped form into wherever the output was
  // referenced.
}

function makeContext(session: SessionState): MaterializeContext {
  return {
    instanceDecls:       new Map(),
    paramDecls:          new Map(),
    syntheticRegDecls:   [],
    exprMemo:            new WeakMap(),
    aliveInstances:      new Map(),
    session,
  }
}

/** Test-only export. */
export const _materializeSessionForTesting = materializeSession

// ─────────────────────────────────────────────────────────────────────────────
// Session → synthetic ResolvedProgram
// ─────────────────────────────────────────────────────────────────────────────

interface MaterializeContext {
  instanceDecls: Map<string, InstanceDecl>
  paramDecls: Map<string, ParamDecl>
  /** Synthetic RegDecls (with update populated) generated from session-
   *  level `delay()` expressions in wires. Each becomes a stateful slot
   *  one sample late. Named `__sd${idx}` for uniqueness within the
   *  session-level synthetic program. */
  syntheticRegDecls: RegDecl[]
  exprMemo: WeakMap<object, ResolvedExpr>
  /** For each session instance with an explicit aliveInput, the
   *  resolved alive expression to apply post-strata. Wrapping happens
   *  after `inlineInstances` has lifted all sub-instance regs/delays
   *  into the synthetic top-level so every register in the alive
   *  lineage gets wrapped. */
  aliveInstances: Map<string, ResolvedExpr>
  session: SessionState
}

function materializeSession(session: SessionState): ResolvedProgram {
  return materializeSessionInner(session, makeContext(session))
}

function materializeSessionInner(session: SessionState, ctx: MaterializeContext): ResolvedProgram {
  for (const [name, inst] of session.instanceRegistry) {
    const decl = buildInstanceDecl(name, inst, ctx)
    ctx.instanceDecls.set(name, decl)
  }

  for (const [key, expr] of session.inputExprNodes) {
    const colon = key.indexOf(':')
    if (colon < 0) continue
    const instName = key.slice(0, colon)
    const inputName = key.slice(colon + 1)
    const instDecl = ctx.instanceDecls.get(instName)
    if (instDecl === undefined) continue
    const port = instDecl.type.ports.inputs.find(p => p.name === inputName)
    if (port === undefined) {
      throw new Error(
        `compileSession: instance '${instName}' has no input port '${inputName}' on type '${instDecl.type.name}'.`,
      )
    }
    const value = translateExpr(expr, ctx)
    instDecl.inputs.push({ port, value })
  }

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

  const bodyDecls: BodyDecl[] = []
  for (const decl of ctx.syntheticRegDecls)      bodyDecls.push(decl)
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

  return {
    op: 'program',
    name: '__session__',
    typeParams: [] as TypeParamDecl[],
    ports,
    body: block,
  }
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

  const registered = session.resolvedRegistry.get(baseTypeName)
  let resolvedType: ResolvedProgram | undefined
  let typeArgsList: Array<{ param: TypeParamDecl; value: number }> = []

  if (registered !== undefined && registered.typeParams.length === 0) {
    resolvedType = registered
  } else {
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

  const decl: InstanceDecl = {
    op: 'instanceDecl',
    name,
    type: resolvedType,
    typeArgs: [],
    inputs: [],
  }

  // Alive wrap: same two-phase mechanism as the legacy gateable wrap.
  //   - PRE-STRATA: wrap the type's outputAssigns + own RegDecl
  //     updates with `select(alive, raw, fallback)`. Necessary because
  //     strata's nestedOut substitution captures the alive instance's
  //     output expression at inline time — wrapping post-strata would
  //     miss it.
  //   - POST-STRATA: wrap any decls lifted from sub-instances
  //     (matched by `_liftedFrom`). These don't exist pre-strata.
  //
  // The alive value is plumbed in as a synthetic `__alive__` input on
  // the cloned type so cloning doesn't have to handle outer-scope
  // refs in an embedded alive expression.
  if (inst.aliveInput !== undefined) {
    const aliveExpr = translateExpr(inst.aliveInput, ctx)
    ctx.aliveInstances.set(name, aliveExpr)
    wrapTypeOutputsForAlive(decl, aliveExpr)
  }

  return decl
}

/** Pre-strata wrap of an alive-eligible instance's TYPE outputs and
 *  own reg/delay updates via a synthetic `__alive__` input. */
function wrapTypeOutputsForAlive(decl: InstanceDecl, aliveExpr: ResolvedExpr): void {
  const cloned = cloneResolvedProgram(decl.type)
  const aliveInputDecl: InputDecl = { op: 'inputDecl', name: '__alive__' }
  cloned.ports.inputs.push(aliveInputDecl)

  const aliveRef: ResolvedExpr = { op: 'inputRef', decl: aliveInputDecl }

  for (const a of cloned.body.assigns) {
    // Post-Phase-0a: BodyAssign is OutputAssign-only.
    a.expr = { op: 'select', args: [aliveRef, a.expr, 0] }
  }
  for (const d of cloned.body.decls) {
    if (d.op !== 'regDecl') continue
    if (d.update === undefined) continue
    const fallback: ResolvedExpr = { op: 'regRef', decl: d }
    d.update = { op: 'select', args: [aliveRef, d.update, fallback] }
  }

  decl.type = cloned
  decl.inputs.push({ port: aliveInputDecl, value: aliveExpr })
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

  if (op === 'array') {
    const items = (obj.items as ExprNode[]).map(item => translateExpr(item, ctx))
    return items as ResolvedExpr
  }

  if (op === 'delay') {
    const update = translateExpr((obj.args as ExprNode[])[0], ctx)
    const init = typeof obj.init === 'number' ? obj.init : 0
    // Post-Phase-0a: session `delay()` lowers to a synthetic RegDecl
    // with update populated. Reads of the value are RegRefs.
    const decl: RegDecl = { op: 'regDecl', name: `__sd${ctx.syntheticRegDecls.length}`, update, init }
    ctx.syntheticRegDecls.push(decl)
    return { op: 'regRef', decl }
  }

  throw new Error(`compileSession: unhandled wiring op '${op}'.`)
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
