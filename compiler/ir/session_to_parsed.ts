/**
 * session_to_parsed.ts — session → `ParsedProgram` serializer.
 *
 * The session→root lowering. A session is morally a `tropical_program_2`:
 * a program whose body decls are its top-level instances wired together,
 * with per-wire unit delays expressed as `delay` decls and params as
 * `param` decls. `compile_session_slotted.ts:buildSessionRoot` serializes
 * the session here and feeds the result to the SAME `elaborate` front door
 * the surface path uses — the instances' already-resolved types are
 * supplied through the elaborator's `ExternalProgramResolver` hook
 * (`sessionTypeResolver`), so this LINKs the already-reduced instances
 * rather than re-elaborating them from source. The elaborator folds the
 * `delay` decls into root `RegDecl`s; its strict cycle policy rejects
 * undelayed inter-instance cycles. This replaced the hand-rolled
 * `materialize_session.ts` (a partial re-implementation of name resolution
 * + RegDecl reconstruction) — see `git log` and `session_to_parsed.test.ts`.
 *
 * Consumes the session AFTER `liftWiresToInstances` + `extractSessionDelays`
 * have run: array-literal wires are already lifted to anonymous producer
 * instances, and every `delay()` op is hoisted into
 * `session.delaySlotRegistry` (scalar → `slotIdx`, array → `arraySlot`)
 * with the wire rewritten to a `sessionSlot` / `sessionArraySlot` read.
 *
 * Handles the full session wire surface: scalar + array session delays,
 * params, generics (already monomorphized in `inst.compiled.prog`), and
 * lifted array-port wiring. Out-of-surface ops throw rather than
 * mis-encode (none arise from the MCP wire tools or `.json` patches today;
 * the equivalence gate is `tests/equiv/root_vs_flat.test.ts`).
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
  // delay resolves to the delay decl's name. Two namespaces:
  // `{op:'sessionSlot', index}` (scalar) keys on `slotIdx`;
  // `{op:'sessionArraySlot', index}` (array) keys on `arraySlot`.
  const slotNameByIndex = new Map<number, string>()
  const arraySlotNameByIndex = new Map<number, string>()
  for (const entry of session.delaySlotRegistry) {
    if (entry.isArray) {
      if (entry.arraySlot === undefined) {
        throw new Error(`session_to_parsed: array delay '${entry.slotName}' missing arraySlot`)
      }
      arraySlotNameByIndex.set(entry.arraySlot, entry.slotName)
    } else {
      if (entry.slotIdx === undefined) {
        throw new Error(`session_to_parsed: scalar delay '${entry.slotName}' missing slotIdx`)
      }
      slotNameByIndex.set(entry.slotIdx, entry.slotName)
    }
  }
  const slots: SlotNames = { scalar: slotNameByIndex, array: arraySlotNameByIndex }

  const decls: ParsedBodyDecl[] = []
  const paramNames = new Set<string>()

  // 1. Per-wire delays → `delay` decls. `delay name = update init v` is
  //    surface sugar the elaborator folds into `RegDecl { init, update }`.
  //    The `slotName` carries through as the reg name (the hot-swap state
  //    key). Array session delays get an `arraySize`-long literal-array
  //    init, so the elaborator types the reg as an array and the backend
  //    allocates a backing slot.
  for (const entry of session.delaySlotRegistry) {
    const update = wireExprToParsed(entry.sourceExpr, slots, paramNames)
    let init: ParsedExpr
    if (entry.isArray) {
      if (entry.arraySize === undefined) {
        throw new Error(`session_to_parsed: array delay '${entry.slotName}' missing arraySize`)
      }
      init = new Array<number>(entry.arraySize).fill(entry.init)
    } else {
      init = entry.init
    }
    const decl: ParsedDelayDecl = { op: 'delayDecl', name: entry.slotName, update, init }
    decls.push(decl)
  }

  // 2. Instances → instance decls, in topological order. Order is
  //    irrelevant to the elaborator (it registers all decls before
  //    resolving expressions), but topo order keeps the serialized form
  //    stable and matches the instance order the per-instance path uses.
  const order = computeInstanceTopoOrder(session)
  const wiresByInstance = groupWiresByInstance(session)
  for (const name of order) {
    const inst = session.instanceRegistry.get(name)
    if (inst === undefined) continue
    // NB: no `type_args`. `inst.compiled.prog` is already the specialized,
    // monomorphized post-strata form — its name carries the specialization
    // (`Delay<N=8>`) and it declares no type-params. Re-emitting `<N=8>`
    // would ask the elaborator to specialize an already-specialized program
    // and throw. The session resolves generics once at `resolveProgramType`
    // time; the root only ever LINKs the already-reduced result, so it
    // emits no `type_args`.
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
          value: wireExprToParsed(expr, slots, paramNames),
        })
      }
      decl.inputs = inputs
    }
    decls.push(decl)
  }

  // 3. Params referenced anywhere in wiring → `param` decls. The
  //    elaborator resolves a bare `{op:'param', name}` (→ nameRef) against
  //    the body's param decls. The session compiler later resolves each
  //    root `ParamDecl` to its control-plane slot (see `paramSlots` in
  //    compile_session_slotted.ts).
  for (const pname of [...paramNames].sort()) {
    const decl: ParsedParamDecl = { op: 'paramDecl', name: pname }
    decls.push(decl)
  }

  const body: ParsedBlock = { op: 'block', decls, assigns: [] }
  // No ports: the root carries no input/output ports. DAC outputs stay a
  // sidecar (`session.graphOutputs`), tapped by `emitDacStitch` in the
  // scheduler postamble — they never enter the root body.
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

/** index→slotName maps for the two hoisted-delay namespaces. */
interface SlotNames {
  scalar: Map<number, string>
  array:  Map<number, string>
}

/** Translate a session wire `ExprNode` to a `ParsedExpr`. Handles the
 *  closed set of ops session wires carry; throws on anything else so
 *  out-of-slice input fails loudly rather than mis-encoding. */
function wireExprToParsed(
  expr: ExprNode,
  slots: SlotNames,
  paramNames: Set<string>,
): ParsedExpr {
  if (typeof expr === 'number') return expr
  if (typeof expr === 'boolean') return expr
  if (Array.isArray(expr)) {
    return expr.map(e => wireExprToParsed(e as ExprNode, slots, paramNames))
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
  // Hoisted scalar-delay read → ref to the delay decl by its slot name.
  if (op === 'sessionSlot') {
    const name = slots.scalar.get(obj.index as number)
    if (name === undefined) {
      throw new Error(`session_to_parsed: sessionSlot index ${String(obj.index)} has no delay decl`)
    }
    return nameRef(name)
  }
  // Hoisted array-delay read → ref to the array delay decl by slot name.
  if (op === 'sessionArraySlot') {
    const name = slots.array.get(obj.index as number)
    if (name === undefined) {
      throw new Error(`session_to_parsed: sessionArraySlot index ${String(obj.index)} has no array delay decl`)
    }
    return nameRef(name)
  }
  // Array literal in op form (`{op:'array', items}`) → bare array literal.
  if (op === 'array' && Array.isArray(obj.items)) {
    return (obj.items as ExprNode[]).map(e => wireExprToParsed(e, slots, paramNames))
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

  if (BUILTIN_NULLARY_OPS.has(op)) {
    return { op: 'call', callee: nameRef(op), args: [] }
  }
  if (BUILTIN_CALL_OPS.has(op)) {
    const args = (obj.args as ExprNode[]).map(a => wireExprToParsed(a, slots, paramNames))
    return { op: 'call', callee: nameRef(op), args }
  }
  if (BINARY_OPS.has(op)) {
    const args = obj.args as [ExprNode, ExprNode]
    return {
      op: op as never,
      args: [
        wireExprToParsed(args[0], slots, paramNames),
        wireExprToParsed(args[1], slots, paramNames),
      ],
    }
  }
  if (UNARY_OPS.has(op)) {
    const args = obj.args as [ExprNode]
    return { op: op as never, args: [wireExprToParsed(args[0], slots, paramNames)] }
  }

  throw new Error(`session_to_parsed: out-of-slice wire op '${op}'`)
}
