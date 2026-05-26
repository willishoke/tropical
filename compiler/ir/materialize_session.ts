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
  InputIdx, OutputIdx, ParamIdx, InstanceIdx, RegIdx,
  ProgramKey,
} from './nodes.js'
import { inputIdx, outputIdx, paramIdx, instanceIdx, regIdx, programKey } from './nodes.js'
import type { ExprNode } from '../expr.js'
import type { SessionState } from '../session.js'
import type { Instance } from '../program_types.js'
import { strataPipeline } from './strata.js'
import { findInstanceCycles } from './lowering/cycle_break.js'
import { CycleViolation, type CycleDiagnostic } from './elaboration_diagnostics.js'
import { specializeProgram } from './specialize.js'
import { parseWireKey } from './branded_names.js'
import { mkProgram } from './decl_tables.js'

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

  // Strict cycle policy at the session boundary. Cycles in session
  // wiring that don't pass through an explicit `delay()` in the wire
  // throw `CycleViolation` here, with port-detailed error messages
  // referencing session-level instance names. Sessions can break
  // cycles explicitly via `delay(...)` in wire expressions, which
  // materializes to a synthetic RegDecl and breaks the inter-instance
  // dependency chain.
  const cycles = findInstanceCycles(synthetic)
  if (cycles.length > 0) {
    const diagnostics: CycleDiagnostic[] = cycles.map(scc => {
      const sortedScc = [...scc]
      const target = sortedScc[0]
      return {
        kind: 'cycle',
        scc: sortedScc,
        programName: '__session__',
        suggestedFix:
          `Suggested fix: wrap a wire from '${target.name}' in 'delay(...)' ` +
          `to break the cycle explicitly. Example: wire { instance: '${sortedScc[1]?.name ?? '<consumer>'}', ` +
          `input: '<port>', expr: { op: 'delay', args: [{ op: 'ref', instance: '${target.name}', output: '<port>' }] } }.`,
      }
    })
    throw new CycleViolation(diagnostics)
  }
  // Honor the session's inlineNested mode. The session lift's strata
  // run controls whether sub-instances inside the synthetic top-level
  // get inlined (inlineNested:true, default) or preserved as kernel
  // boundaries for the slot-based fractal path (inlineNested:false).
  const lowered = strataPipeline(synthetic, new Map(), { inlineNested: session.inlineNested })
  return { lowered, paramDecls: ctx.paramDecls }
}

function makeContext(session: SessionState): MaterializeContext {
  return {
    instanceDecls:       new Map(),
    paramDecls:          new Map(),
    programRegistry:     new Map(),
    syntheticRegDecls:   [],
    exprMemo:            new WeakMap(),
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
  /** Per-typeKey ResolvedProgram for each instance type referenced in
   *  the session. Populated by `buildInstanceDecl` and used by the
   *  port-lookup sites below to resolve instance type info without
   *  the dropped `.type` pointer. */
  programRegistry: Map<ProgramKey, ResolvedProgram>
  /** Synthetic RegDecls (with update populated) generated from session-
   *  level `delay()` expressions in wires. Each becomes a stateful slot
   *  one sample late. Named `__sd${idx}` for uniqueness within the
   *  session-level synthetic program. */
  syntheticRegDecls: RegDecl[]
  exprMemo: WeakMap<object, ResolvedExpr>
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
    let ref
    try { ref = parseWireKey(key) } catch { continue }
    const instName = ref.instance
    const inputName = ref.port
    const instDecl = ctx.instanceDecls.get(instName)
    if (instDecl === undefined) continue
    const instType = ctx.programRegistry.get(instDecl.typeKey)!
    const portI = instType.ports.inputs.findIndex(p => p.name === inputName)
    if (portI < 0) {
      throw new Error(
        `compileSession: instance '${instName}' has no input port '${inputName}' on type '${instType.name}'.`,
      )
    }
    const value = translateExpr(expr, ctx)
    instDecl.inputs.push({ port: inputIdx(portI), value })
  }

  // Compute InstanceIdx for each session instance: their position in
  // the eventual body.instances[], which is the iteration order of
  // ctx.instanceDecls (Maps preserve insertion order).
  const instanceIdxByName = new Map<string, InstanceIdx>()
  {
    let pos = 0
    for (const name of ctx.instanceDecls.keys()) {
      instanceIdxByName.set(name, instanceIdx(pos++))
    }
  }

  const outputDecls: OutputDecl[] = []
  const outputAssigns: OutputAssign[] = []
  for (const go of session.graphOutputs) {
    const instDecl = ctx.instanceDecls.get(go.instance)
    if (instDecl === undefined) {
      throw new Error(`compileSession: graph output references unknown instance '${go.instance}'.`)
    }
    const instType = ctx.programRegistry.get(instDecl.typeKey)!
    const outputI = instType.ports.outputs.findIndex(p => p.name === go.output)
    if (outputI < 0) {
      throw new Error(
        `compileSession: instance '${go.instance}' has no output port '${go.output}' on type '${instType.name}'.`,
      )
    }
    const outDecl = instType.ports.outputs[outputI]
    const sessionOutput: OutputDecl = {
      op: 'outputDecl',
      name: `${go.instance}.${go.output}`,
    }
    if (outDecl.type !== undefined) sessionOutput.type = outDecl.type
    const sessionOutI = outputDecls.length
    outputDecls.push(sessionOutput)
    const ref: ResolvedExprOp = {
      op: 'nestedOut',
      instance: instanceIdxByName.get(go.instance)!,
      output: outputIdx(outputI),
    }
    outputAssigns.push({ op: 'outputAssign', target: outputIdx(sessionOutI), expr: ref })
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

  return mkProgram({
    name: '__session__',
    typeParams: [] as TypeParamDecl[],
    ports,
    body: block,
    // Session-level wires are scalar expressions with no combinators
    // or let-bindings — no binders are introduced. The synthetic
    // program's binderCount is therefore 0.
    binderCount: 0,
    programRegistry: ctx.programRegistry,
  })
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

  const registered = session.programs.get(baseTypeName)
  let resolvedType: ResolvedProgram | undefined
  let typeArgsList: Array<{ param: TypeParamDecl; value: number }> = []

  if (registered !== undefined && registered.typeParams.length === 0) {
    // Non-generic: use the canonical post-strata form directly.
    resolvedType = registered
  } else {
    // Generic template (or missing); specialize with the supplied
    // type args. After Phase 5 unification, templates and concrete
    // programs both live in session.programs — distinguished by
    // typeParams length.
    const template = registered
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

  const tk = programKey(resolvedType.name)
  // Register the resolved type so port-lookup sites can resolve it
  // without a `.type` pointer. If two instances share the same type,
  // the registry holds one canonical entry (matches the existing
  // session.resolvedRegistry canonicalization).
  ctx.programRegistry.set(tk, resolvedType)
  const decl: InstanceDecl = {
    op: 'instanceDecl',
    name,
    typeKey: tk,
    typeArgs: [],
    inputs: [],
  }

  return decl
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
    const instType = ctx.programRegistry.get(instDecl.typeKey)!
    const outputI = instType.ports.outputs.findIndex(p => p.name === outputName)
    if (outputI < 0) {
      throw new Error(
        `compileSession: ref to '${instName}.${outputName}' — '${outputName}' is not a port on type '${instType.name}'.`,
      )
    }
    // InstanceIdx is the iteration position of instName in ctx.instanceDecls.
    // Maps preserve insertion order, so this matches the order the instance
    // decls land in body.decls (and thus body.instances) later.
    let instI = -1
    let i = 0
    for (const name of ctx.instanceDecls.keys()) {
      if (name === instName) { instI = i; break }
      i++
    }
    return { op: 'nestedOut', instance: instanceIdx(instI), output: outputIdx(outputI) }
  }

  if (op === 'param' || op === 'paramExpr'
      || op === 'trigger' || op === 'triggerParamExpr') {
    return paramRef(obj.name as string, ctx)
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
    // with update populated. Reads of the value are RegRefs. The
    // synthetic regs are placed at the FRONT of body.decls (before
    // instance decls and param decls — see materializeSessionInner),
    // so their RegIdx equals their position in ctx.syntheticRegDecls.
    const newIdx = regIdx(ctx.syntheticRegDecls.length)
    const decl: RegDecl = { op: 'regDecl', name: `__sd${ctx.syntheticRegDecls.length}`, update, init }
    ctx.syntheticRegDecls.push(decl)
    return { op: 'regRef', idx: newIdx }
  }

  throw new Error(`compileSession: unhandled wiring op '${op}'.`)
}

function paramRef(name: string, ctx: MaterializeContext): ResolvedExpr {
  let decl = ctx.paramDecls.get(name)
  if (decl === undefined) {
    decl = { op: 'paramDecl', name }
    ctx.paramDecls.set(name, decl)
  }
  // ParamIdx is the iteration position in ctx.paramDecls. Params land
  // in body.decls after syntheticRegDecls and instanceDecls, but ParamIdx
  // indexes into body.params[] (filtered by op), which preserves
  // insertion order of paramDecls.
  let pi = -1
  let i = 0
  for (const n of ctx.paramDecls.keys()) {
    if (n === name) { pi = i; break }
    i++
  }
  return { op: 'paramRef', idx: paramIdx(pi) }
}
