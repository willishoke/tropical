/**
 * session_to_parsed.ts — SPIKE: session → `ParsedProgram` serializer.
 *
 * This is exploratory, not a production path. It exists to answer one
 * architectural question: is `materialize_session.ts` a redundant,
 * hand-rolled re-implementation of the elaborator? (See the spike test
 * `tests/spike/session_as_parsed_program.test.ts`.)
 *
 * The claim under test: a session is morally a `tropical_program_2` —
 * a program whose body decls are its top-level instances wired together,
 * with per-wire unit delays expressed as `delay` decls and params as
 * `param` decls. If so, the session can serialize to a `ParsedProgram`
 * and feed the SAME `elaborate` front door the surface path uses, with
 * the instances' already-resolved types supplied through the elaborator's
 * existing `ExternalProgramResolver` hook (the LINK seam). Then
 * `materializeSessionToResolvedIR` + `extractSessionDelays`' bespoke
 * RegDecl reconstruction collapse into "emit NameRefs, call elaborate."
 *
 * Scope (Phase B slice): scalar wiring, scalar session delays, params,
 * non-generic or position-typeArg'd instances. Array wires / array
 * session delays / sum-typed wiring are out of slice and throw — the
 * spike is a differential against `materialize_session.ts`'s *scalar*
 * path, nothing more.
 *
 * Consumes the session AFTER `liftWiresToInstances` + `extractSessionDelays`
 * have run (the same pre-state `materializeSessionToResolvedIR` consumes),
 * so the two builders take identical input and the differential is fair.
 */

import type { ExprNode, SessionState } from '../session.js'
import type {
  Program as ParsedProgram,
  Block as ParsedBlock,
  BodyDecl as ParsedBodyDecl,
  ParsedExpr,
  DelayDecl as ParsedDelayDecl,
  ParamDecl as ParsedParamDecl,
  InstanceDecl as ParsedInstanceDecl,
  InstanceInputEntry,
} from '../parse/nodes.js'
import { nameRef } from '../parse/nodes.js'
import type { ResolvedProgram } from './nodes.js'
import { parseWireKey } from './branded_names.js'
import { computeInstanceTopoOrder } from './compile_session_slotted_helpers.js'

/** Builtin call ops that survive in scalar wires as `f(args)` — mirrors
 *  `raise.ts`'s BUILTIN_CALL_OPS so the elaborator rewrites them back to
 *  their structured ops. */
const BUILTIN_CALL_OPS: ReadonlySet<string> = new Set([
  'select', 'clamp', 'round', 'ldexp', 'floorDiv',
  'sqrt', 'abs', 'floatExponent', 'arraySet',
  'floor', 'ceil', 'toInt', 'toBool', 'toFloat',
])

const BUILTIN_NULLARY_OPS: ReadonlySet<string> = new Set([
  'sampleRate', 'sampleIndex',
])

const BINARY_OPS: ReadonlySet<string> = new Set([
  'add', 'sub', 'mul', 'div', 'mod',
  'lt', 'lte', 'gt', 'gte', 'eq', 'neq',
  'and', 'or',
  'bitAnd', 'bitOr', 'bitXor', 'lshift', 'rshift',
])

const UNARY_OPS: ReadonlySet<string> = new Set(['neg', 'not', 'bitNot'])

/** Build the type-name → ResolvedProgram resolver the elaborator's
 *  `ExternalProgramResolver` hook expects. This is the LINK supply: the
 *  session's instances are already post-strata `ResolvedProgram`s; we
 *  hand them to `elaborate` by name instead of re-elaborating from
 *  source. Keyed on `prog.name` so it matches the `typeKey =
 *  programKey(prog.name)` the elaborator mints in `registerInstanceDecl`. */
export function sessionTypeResolver(
  session: SessionState,
): (name: string) => ResolvedProgram | undefined {
  const byName = new Map<string, ResolvedProgram>()
  for (const inst of session.instanceRegistry.values()) {
    const prog = inst.compiled.prog
    if (!byName.has(prog.name)) byName.set(prog.name, prog)
  }
  return (name) => byName.get(name)
}

/** Serialize a (post-extraction) session into a `ParsedProgram` whose
 *  body is the session graph. Pure; throws on out-of-slice shapes. */
export function sessionToParsedProgram(session: SessionState): ParsedProgram {
  // index → slotName for `{op:'sessionSlot', index}` reads (the rewrite
  // `extractSessionDelays` left behind), so a wire reading a hoisted
  // delay resolves to the delay decl's name.
  const slotNameByIndex = new Map<number, string>()
  for (const entry of session.delaySlotRegistry) {
    if (entry.isArray) {
      throw new Error('session_to_parsed: array session delays are out of slice')
    }
    if (entry.slotIdx === undefined) {
      throw new Error('session_to_parsed: scalar delay entry missing slotIdx')
    }
    slotNameByIndex.set(entry.slotIdx, entry.slotName)
  }

  const decls: ParsedBodyDecl[] = []
  const paramNames = new Set<string>()

  // 1. Per-wire scalar delays → `delay` decls. `delay name = update init v`
  //    is surface sugar the elaborator folds into `RegDecl { init, update }`
  //    — exactly the synthetic reg `materializeSessionDelaySlots` builds by
  //    hand. Same `slotName`, so the two line up by name.
  for (const entry of session.delaySlotRegistry) {
    const update = wireExprToParsed(entry.sourceExpr, slotNameByIndex, paramNames)
    const decl: ParsedDelayDecl = {
      op: 'delayDecl',
      name: entry.slotName,
      update,
      init: entry.init,
    }
    decls.push(decl)
  }

  // 2. Instances → instance decls, in topological order (the order
  //    `materialize_session` uses; irrelevant to the elaborator, which
  //    registers all decls before resolving expressions, but kept so the
  //    serialized form mirrors the materializer's).
  const order = computeInstanceTopoOrder(session)
  const wiresByInstance = groupWiresByInstance(session)
  for (const name of order) {
    const inst = session.instanceRegistry.get(name)
    if (inst === undefined) continue
    // NB: no `type_args`. `inst.compiled.prog` is already the specialized,
    // monomorphized post-strata form — its name carries the specialization
    // (`Delay<N=8>`) and it declares no type-params. Re-emitting `<N=8>`
    // would ask the elaborator to specialize an already-specialized program
    // and throw. This mirrors `buildInstanceDecl`'s `typeArgs: []`: the
    // session resolves generics once at `resolveProgramType` time, and the
    // root only ever LINKs the already-reduced result.
    const decl: ParsedInstanceDecl = {
      op: 'instanceDecl',
      name,
      program: nameRef(inst.compiled.prog.name),
    }
    const wires = wiresByInstance.get(name)
    if (wires !== undefined && wires.length > 0) {
      const inputs: InstanceInputEntry[] = []
      for (const { port, expr } of wires) {
        inputs.push({
          port: nameRef(port),
          value: wireExprToParsed(expr, slotNameByIndex, paramNames),
        })
      }
      decl.inputs = inputs
    }
    decls.push(decl)
  }

  // 3. Params referenced anywhere in wiring → `param` decls. The
  //    elaborator resolves a bare `{op:'param', name}` (→ nameRef) against
  //    the body's param decls, mirroring the ParamDecls
  //    `materialize_session` synthesizes.
  for (const pname of [...paramNames].sort()) {
    const decl: ParsedParamDecl = { op: 'paramDecl', name: pname }
    decls.push(decl)
  }

  const body: ParsedBlock = { op: 'block', decls, assigns: [] }
  // No ports: the root carries no input/output ports. DAC outputs stay a
  // sidecar (`session.graphOutputs`), exactly as the root-program path
  // keeps them off the materialized root today.
  return { op: 'program', name: '__session__', body }
}

interface Wire { port: string; expr: ExprNode }

function groupWiresByInstance(session: SessionState): Map<string, Wire[]> {
  const out = new Map<string, Wire[]>()
  for (const [key, expr] of session.inputExprNodes) {
    let ref
    try { ref = parseWireKey(key) } catch { continue }
    const list = out.get(ref.instance) ?? []
    list.push({ port: ref.port, expr })
    out.set(ref.instance, list)
  }
  // Stable order within an instance — by port name. The elaborator
  // resolves inputs by name, so order is semantically irrelevant; sorting
  // just makes the serialized form deterministic.
  for (const list of out.values()) list.sort((a, b) => a.port.localeCompare(b.port))
  return out
}

/** Translate a session wire `ExprNode` to a `ParsedExpr`. Handles the
 *  closed set of ops scalar wires carry; throws on anything else so
 *  out-of-slice input fails loudly rather than mis-encoding. */
function wireExprToParsed(
  expr: ExprNode,
  slotNameByIndex: Map<number, string>,
  paramNames: Set<string>,
): ParsedExpr {
  if (typeof expr === 'number') return expr
  if (typeof expr === 'boolean') return expr
  if (Array.isArray(expr)) {
    return expr.map(e => wireExprToParsed(e as ExprNode, slotNameByIndex, paramNames))
  }
  if (typeof expr !== 'object' || expr === null) {
    throw new Error(`session_to_parsed: invalid wire value ${JSON.stringify(expr)}`)
  }
  const obj = expr as Record<string, unknown>
  const op = obj.op
  if (typeof op !== 'string') {
    throw new Error(`session_to_parsed: wire node missing op: ${JSON.stringify(obj)}`)
  }

  // Producer output read → dotted nested-out ref (resolved by the
  // elaborator against in-scope instances + the target's output ports).
  if (op === 'ref') {
    return {
      op: 'nestedOut',
      ref: nameRef(obj.instance as string),
      output: nameRef(obj.output as string),
    }
  }
  // Hoisted-delay read → ref to the delay decl by its slot name.
  if (op === 'sessionSlot') {
    const name = slotNameByIndex.get(obj.index as number)
    if (name === undefined) {
      throw new Error(`session_to_parsed: sessionSlot index ${String(obj.index)} has no delay decl`)
    }
    return nameRef(name)
  }
  // Param / legacy trigger → name ref (collected for paramDecl emission).
  if (op === 'param' || op === 'trigger') {
    const name = obj.name as string
    paramNames.add(name)
    return nameRef(name)
  }
  if (op === 'delay') {
    throw new Error('session_to_parsed: unextracted delay() in wire — run extractSessionDelays first')
  }
  if (op === 'sessionArraySlot') {
    throw new Error('session_to_parsed: array session slot is out of slice')
  }

  if (BUILTIN_NULLARY_OPS.has(op)) {
    return { op: 'call', callee: nameRef(op), args: [] }
  }
  if (BUILTIN_CALL_OPS.has(op)) {
    const args = (obj.args as ExprNode[]).map(a => wireExprToParsed(a, slotNameByIndex, paramNames))
    return { op: 'call', callee: nameRef(op), args }
  }
  if (BINARY_OPS.has(op)) {
    const args = obj.args as [ExprNode, ExprNode]
    return {
      op: op as never,
      args: [
        wireExprToParsed(args[0], slotNameByIndex, paramNames),
        wireExprToParsed(args[1], slotNameByIndex, paramNames),
      ],
    }
  }
  if (UNARY_OPS.has(op)) {
    const args = obj.args as [ExprNode]
    return { op: op as never, args: [wireExprToParsed(args[0], slotNameByIndex, paramNames)] }
  }

  throw new Error(`session_to_parsed: out-of-slice wire op '${op}'`)
}
