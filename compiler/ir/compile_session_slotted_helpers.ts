/**
 * compile_session_slotted_helpers.ts — M9a operand remapping + merging.
 *
 * Helpers used by `compileSessionSlotted` to produce a real per-instance
 * compile-then-merge plan. For each session instance, `compileResolved`
 * is called on the instance's standalone `Compiled.prog`. The resulting
 * FlatPlan is then remapped:
 *
 *  - `input` operands (referring to the instance's external input ports)
 *    are rewritten to `slot` operands pointing at the wire's source
 *    output slot, or `const` operands for literal-wired inputs.
 *  - `reg` / `state_reg` / `array_reg` slot indices are offset into a
 *    unified register space across all instances.
 *  - `dst` temp indices are offset by the same.
 *  - `register_targets` entries get the same offset, plus append.
 *  - A `WriteSlot` instruction is appended for each output port to
 *    publish the computed value to the instance's output slot.
 *
 * After all instances are processed, a DAC stitching phase reads each
 * graphOutput's source slot into a fresh temp and exposes it through
 * `output_targets` + `outputs` so the kernel's existing mix-bus
 * mechanism sums them into the audio buffer.
 *
 * M9a scope: single-source ref-chain patches only. Throws on fan-in,
 * arbitrary input expressions, params/triggers in input expressions,
 * arrays/sums. Subsequent sub-milestones lift each limitation.
 */

import type { SessionState, ExprNode } from '../session.js'
import type { PerInstancePlan } from '../flat_plan.js'
import type {
  NOperand, NInstr, ScalarType,
} from './emit_resolved.js'
import { BINARY_TAG, UNARY_TAG, TERNARY_TAG } from './emit_resolved.js'
import { topologicalSort } from '../compiler.js'

/** Raised when a session uses a shape the per-instance compile path
 *  doesn't yet support: nested instance calls in input expressions,
 *  sums survived to the session layer, type-level paramDecls embedded
 *  in a program type's body, etc. After PR-C there is no auto-
 *  fallback; this surfaces directly to the caller.
 *
 *  Kept as a marker class so we can distinguish "shape we know we
 *  don't handle yet" from genuine compiler bugs. */
export class SlotShapeUnsupportedError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'SlotShapeUnsupportedError'
  }
}

/** Resolved binding for a module input port. Replaces the previously
 *  coupled `(expr | undefined, defaultVal | undefined)` parameter pair
 *  with a single discriminated union — the consumer doesn't have to
 *  reconstruct "is this wired or unwired" from two correlated fields.
 *
 *  - `wired`: the input has an ExprNode wired to it. Could be a literal,
 *    a ref, a param, or an arbitrary expression — `translateNode`
 *    handles all of these.
 *  - `literal`: the input has no wiring. Carries the resolved fallback
 *    value (port declared default, or 0 if no default exists). The
 *    resolver owns the "default → 0" fallback so consumers don't
 *    re-implement it. */
export type InputBinding =
  | { kind: 'wired';   expr:  ExprNode }
  | { kind: 'literal'; value: number | boolean }

/** Allocates fresh temps in the unified register space and collects
 *  NInstrs that need to run somewhere in the kernel — either before
 *  the consuming instance's body (for input expressions) or in the
 *  scheduler's pre/postamble (for alive expressions and DAC stitch).
 *  Decoupling the emitter from where the instructions land lets the
 *  same `translateNode` machinery serve all three callers. */
export interface PreambleEmitter {
  instrs: NInstr[]
  allocTemp(): number
}

/** Translate an arbitrary ExprNode into NInstrs emitted to the
 *  preamble, returning the operand that holds the result. M9c-supported
 *  ops: refs, params/triggers, literals, all entries in BINARY_TAG /
 *  UNARY_TAG / TERNARY_TAG. Anything else throws with a pointer to the
 *  appropriate follow-on milestone. */
function translateNode(
  expr: ExprNode,
  scalarType: ScalarType,
  session: SessionState,
  emitter: PreambleEmitter,
  context: string,  // for error messages: "${instance}.${port}"
): NOperand {
  // ── leaves ──
  if (typeof expr === 'number') {
    return { kind: 'const', val: expr, scalar_type: scalarType }
  }
  if (typeof expr === 'boolean') {
    return { kind: 'const', val: expr ? 1 : 0, scalar_type: scalarType }
  }
  if (Array.isArray(expr)) {
    throw new SlotShapeUnsupportedError(
      `compileSessionSlotted: array-shaped input expression at '${context}' ` +
      `not yet supported. Arrays land in M9d.`,
    )
  }
  if (typeof expr !== 'object' || expr === null) {
    throw new Error(
      `compileSessionSlotted: unrecognized input expression at '${context}': ${typeof expr}`,
    )
  }

  const obj = expr as Record<string, unknown>
  const op = obj.op

  // ── refs and params (leaves) ──
  if (op === 'ref' && typeof obj.instance === 'string' && typeof obj.output === 'string') {
    const key = `${obj.instance}.${obj.output}`
    const slotIdx = session.outputSlotRegistry.get(key)
    if (slotIdx === undefined) {
      throw new SlotShapeUnsupportedError(
        `compileSessionSlotted: wire src '${key}' at '${context}' has no allocated output slot.`,
      )
    }
    return { kind: 'slot', index: slotIdx, scalar_type: scalarType }
  }
  if ((op === 'param' || op === 'paramExpr') && typeof obj.name === 'string') {
    const slotIdx = session.paramSlotRegistry.get(obj.name)
    if (slotIdx === undefined) {
      throw new SlotShapeUnsupportedError(
        `compileSessionSlotted: param '${obj.name}' at '${context}' has no allocated slot.`,
      )
    }
    return { kind: 'slot', index: slotIdx, scalar_type: scalarType }
  }
  if ((op === 'trigger' || op === 'triggerParamExpr') && typeof obj.name === 'string') {
    const slotIdx = session.paramSlotRegistry.get(obj.name)
    if (slotIdx === undefined) {
      throw new SlotShapeUnsupportedError(
        `compileSessionSlotted: trigger '${obj.name}' at '${context}' has no allocated slot.`,
      )
    }
    return { kind: 'slot', index: slotIdx, scalar_type: scalarType }
  }

  // ── builtins ──
  if (op === 'sampleRate')  return { kind: 'rate',  scalar_type: scalarType }
  if (op === 'sampleIndex') return { kind: 'tick',  scalar_type: scalarType }

  // ── arithmetic / logical / comparison ──
  if (typeof op === 'string' && BINARY_TAG[op]) {
    const args = (obj.args as ExprNode[])
    if (args.length !== 2) {
      throw new Error(
        `compileSessionSlotted: binary op '${op}' at '${context}' needs 2 args, got ${args.length}.`,
      )
    }
    // Result type for comparisons is bool; for arithmetic, promote inputs.
    const tag = BINARY_TAG[op]
    const resultType: ScalarType = isComparisonTag(tag) ? 'bool'
      : isBitwiseTag(tag) ? 'int'
      : scalarType
    const argType: ScalarType = isComparisonTag(tag) ? 'float'
      : isBitwiseTag(tag) ? 'int'
      : isLogicalTag(tag) ? 'bool'
      : resultType
    const a = translateNode(args[0], argType, session, emitter, context)
    const b = translateNode(args[1], argType, session, emitter, context)
    const dst = emitter.allocTemp()
    emitter.instrs.push({
      tag, dst, args: [a, b], loop_count: 1, strides: [], result_type: resultType,
    })
    return { kind: 'reg', slot: dst, scalar_type: resultType }
  }
  if (typeof op === 'string' && UNARY_TAG[op]) {
    const args = (obj.args as ExprNode[])
    if (args.length !== 1) {
      throw new Error(
        `compileSessionSlotted: unary op '${op}' at '${context}' needs 1 arg, got ${args.length}.`,
      )
    }
    const tag = UNARY_TAG[op]
    const resultType: ScalarType =
      tag === 'ToInt' ? 'int' : tag === 'ToBool' ? 'bool' : tag === 'ToFloat' ? 'float'
      : tag === 'Not' ? 'bool'
      : tag === 'BitNot' ? 'int'
      : scalarType
    const a = translateNode(args[0], resultType, session, emitter, context)
    const dst = emitter.allocTemp()
    emitter.instrs.push({
      tag, dst, args: [a], loop_count: 1, strides: [], result_type: resultType,
    })
    return { kind: 'reg', slot: dst, scalar_type: resultType }
  }
  if (typeof op === 'string' && TERNARY_TAG[op]) {
    const args = (obj.args as ExprNode[])
    if (args.length !== 3) {
      throw new Error(
        `compileSessionSlotted: ternary op '${op}' at '${context}' needs 3 args, got ${args.length}.`,
      )
    }
    const tag = TERNARY_TAG[op]
    // select: args = [cond:bool, then:T, else:T]; clamp: args = [val, lo, hi] all same type
    const condType: ScalarType = tag === 'Select' ? 'bool' : scalarType
    const a = translateNode(args[0], condType, session, emitter, context)
    const b = translateNode(args[1], scalarType, session, emitter, context)
    const c = translateNode(args[2], scalarType, session, emitter, context)
    const dst = emitter.allocTemp()
    emitter.instrs.push({
      tag, dst, args: [a, b, c], loop_count: 1, strides: [], result_type: scalarType,
    })
    return { kind: 'reg', slot: dst, scalar_type: scalarType }
  }

  // ── unknown ──
  throw new SlotShapeUnsupportedError(
    `compileSessionSlotted: input expression at '${context}' uses op '${op}' which is not ` +
    `yet supported. Nested instance calls (e.g. Sin(x: ...)), array ops, and fan-in (combine) ` +
    `land in M9d.`,
  )
}

function isComparisonTag(tag: string): boolean {
  return tag === 'Less' || tag === 'LessEq' || tag === 'Greater' || tag === 'GreaterEq'
    || tag === 'Equal' || tag === 'NotEqual'
}
function isBitwiseTag(tag: string): boolean {
  return tag === 'BitAnd' || tag === 'BitOr' || tag === 'BitXor'
    || tag === 'LShift' || tag === 'RShift'
}
function isLogicalTag(tag: string): boolean {
  return tag === 'And' || tag === 'Or'
}

/** Compile a session's aliveInput ExprNode into NInstrs emitted to the
 *  scheduler preamble, returning the operand that feeds the WriteSlot
 *  into the instance's `__alive__` slot.
 *
 *  Inter-instance refs inside an alive expression are implicitly
 *  one-sample-delayed: at preamble time, no WriteSlot has fired this
 *  sample yet, so any `ref` operand reads the slot's previous-sample
 *  value. This is the documented `Topological ordering with alive
 *  dependencies` semantic from the runtime plan — it lets alive be
 *  driven by its own instance's output (envelope-driven self-sleep)
 *  without introducing a graph cycle. */
export function translateAliveExpr(
  expr: ExprNode,
  session: SessionState,
  emitter: PreambleEmitter,
  context: string,
): NOperand {
  return translateNode(expr, 'bool', session, emitter, context)
}

// ─────────────────────────────────────────────────────────────────────────────
// Topological order from session.inputExprNodes
// ─────────────────────────────────────────────────────────────────────────────

/** Walk an ExprNode and collect the instance names of every `{op:'ref'}`
 *  node it transitively references. Used to build per-instance
 *  dependency sets for topological sort. */
function collectInstanceRefs(expr: ExprNode | undefined, out: Set<string>): void {
  if (expr === undefined || expr === null) return
  if (typeof expr !== 'object') return
  if (Array.isArray(expr)) {
    for (const e of expr) collectInstanceRefs(e as ExprNode, out)
    return
  }
  const obj = expr as Record<string, unknown>
  if (obj.op === 'ref' && typeof obj.instance === 'string') {
    out.add(obj.instance)
    return
  }
  // Walk args / nested fields generically
  for (const v of Object.values(obj)) {
    if (typeof v === 'object' && v !== null) {
      collectInstanceRefs(v as ExprNode, out)
    }
  }
}

/** Build a topological order over session instances using
 *  inputExprNodes references. Instances with no wiring come first.
 *
 *  Cycles do not throw. The slot architecture handles them
 *  implicitly: every inter-instance ref reads from a slot, and the
 *  slot's value is either the current-sample WriteSlot (when the
 *  producer ran earlier this iteration) or the previous-sample
 *  WriteSlot (when the producer hasn't run yet, including back-edges
 *  in a cycle). This is the structural equivalent of `traceCycles`
 *  inserting a synthetic unit-delay on a back-edge — the slot itself
 *  IS the unit delay. We just pick a consistent order across cyclic
 *  instances so the topology is fully sequenced. */
export function computeInstanceTopoOrder(session: SessionState): string[] {
  const deps = new Map<string, Set<string>>()
  for (const name of session.instanceRegistry.keys()) {
    deps.set(name, new Set())
  }
  for (const [key, expr] of session.inputExprNodes) {
    const colon = key.indexOf(':')
    if (colon < 0) continue
    const consumer = key.slice(0, colon)
    const producers = deps.get(consumer)
    if (producers === undefined) continue
    collectInstanceRefs(expr, producers)
    // Self-references resolve through the slot's previous-sample
    // value naturally; drop them from the dep graph so Kahn's makes
    // progress.
    producers.delete(consumer)
  }
  const result = topologicalSort(deps)
  if (result.complete) return result.order

  // Cycle present: emit the topologically-sorted prefix, then append
  // remaining (cyclic) instances in insertion order. Within an SCC,
  // back-edge reads see the previous sample's slot value — the
  // category-theoretic unit-delay endomorphism realized through the
  // slot array.
  const seen = new Set(result.order)
  const tail: string[] = []
  for (const name of session.instanceRegistry.keys()) {
    if (!seen.has(name)) tail.push(name)
  }
  return [...result.order, ...tail]
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-instance plan remapping
// ─────────────────────────────────────────────────────────────────────────────

/** Context for remapping a single instance's FlatPlan into the unified
 *  session plan space. */
export interface RemapContext {
  /** Session instance name. Used for outputSlotRegistry lookups + errors. */
  instanceName: string
  /** Cumulative temp index offset before this instance. All `dst` and
   *  `reg` operand `slot` fields shift by this. */
  regOffset: number
  /** Cumulative state-register offset (regs + delays). All `state_reg`
   *  operand `slot` fields and `register_targets` entries shift by this. */
  stateRegOffset: number
  /** Cumulative array-slot offset. All `array_reg` operand `slot` fields
   *  shift by this. */
  arraySlotOffset: number
  /** Resolve an input port (by name) to its binding — either a wired
   *  ExprNode or a literal fallback. The resolver owns the
   *  "unwired → default → 0" fallback chain so the consumer just sees
   *  one of two cases. */
  inputBindingFor: (portName: string) => InputBinding
  /** Resolve an output port (by name) to its allocated output-slot
   *  index in the session's unified slot array. */
  outputSlotFor: (portName: string) => number
  /** Names of the instance's input ports, in port-declaration order.
   *  `input` operands index into this; we look up the port name then
   *  consult `inputBindingFor`. */
  inputPortNames: string[]
  /** Per-output-port name + scalar type, indexed parallel to the
   *  instance's program output port declarations. Used for emitting
   *  WriteSlot instructions at the end of the instance's body. */
  outputPortNames: string[]
  outputScalarTypes: ScalarType[]
}

/** Resolve an input port's binding to a concrete NOperand.
 *  - Literal binding → const operand
 *  - Wired binding → delegate to translateNode, which handles every
 *    ExprNode shape (refs, params, triggers, arithmetic, etc.)
 *    uniformly. Preamble emission happens transparently via the
 *    emitter when the wired expression is non-leaf. */
function inputBindingToOperand(
  binding: InputBinding,
  scalarType: ScalarType,
  session: SessionState,
  context: string,
  emitter: PreambleEmitter,
): NOperand {
  if (binding.kind === 'literal') {
    const v = typeof binding.value === 'boolean' ? (binding.value ? 1 : 0) : binding.value
    return { kind: 'const', val: v, scalar_type: scalarType }
  }
  return translateNode(binding.expr, scalarType, session, emitter, context)
}

/** Apply per-instance operand and slot-index remapping. Returns a fresh
 *  set of instructions ready to merge into the unified instruction
 *  stream:
 *  - `preamble`: instructions emitted to compute arbitrary input
 *    expressions (M9c). These go BEFORE the instance's body in the
 *    unified stream.
 *  - `body`: the instance's body with operands remapped.
 *  - `writeSlots`: WriteSlot instructions per output port, AFTER the body.
 *  Also returns `tempsConsumed` — the number of temps allocated for
 *  preamble computations, which the caller adds to its running offset. */
export function remapInstancePlan(
  plan: PerInstancePlan,
  ctx: RemapContext,
  session: SessionState,
): {
  preamble:      NInstr[]
  body:          NInstr[]
  writeSlots:    NInstr[]
  tempsConsumed: number
} {
  // Preamble emitter: allocates fresh temps in the unified register space.
  // Temps after the instance's own register_count get used here.
  let preambleNextTemp = ctx.regOffset + plan.register_count
  const preamble: NInstr[] = []
  const emitter: PreambleEmitter = {
    instrs: preamble,
    allocTemp: () => preambleNextTemp++,
  }

  const remapOperand = (op: NOperand): NOperand => {
    switch (op.kind) {
      case 'const': return op
      case 'rate':  return op
      case 'tick':  return op
      case 'slot':  return op
      case 'input': {
        const portName = ctx.inputPortNames[op.slot]
        if (portName === undefined) {
          throw new Error(
            `compileSessionSlotted: instance '${ctx.instanceName}' input operand slot=${op.slot} ` +
            `out of range (only ${ctx.inputPortNames.length} input ports).`,
          )
        }
        return inputBindingToOperand(
          ctx.inputBindingFor(portName),
          op.scalar_type,
          session,
          `${ctx.instanceName}.${portName}`,
          emitter,
        )
      }
      case 'reg':       return { ...op, slot: op.slot + ctx.regOffset }
      case 'state_reg': return { ...op, slot: op.slot + ctx.stateRegOffset }
      case 'array_reg': return { ...op, slot: op.slot + ctx.arraySlotOffset }
      case 'param':
        // M9b: under slot mode, session-level param/trigger refs in
        // input expressions are resolved by inputExprToOperand to
        // `slot` operands before reaching here. A legacy `param`
        // operand surviving means the per-instance plan's body
        // referenced a paramDecl at the type level (e.g., a stdlib
        // type that uses an inline {op:'param'} ExprNode internally).
        // Auto-fall-back to the legacy path — type-level params are
        // a follow-up scope item.
        throw new SlotShapeUnsupportedError(
          `compileSessionSlotted: legacy 'param' operand encountered ` +
          `in '${ctx.instanceName}'. Session-level params should resolve ` +
          `to slot operands before this point. This usually means the ` +
          `program type's body has an internal {op:'param'} ExprNode ` +
          `— this case is not yet handled.`,
        )
    }
  }

  // `dst` semantics depend on the instruction:
  //   - `Pack`, `SetElement`, and any elementwise op (loop_count > 1)
  //     write to an ARRAY slot (`arrays[dst]`). Shift by arraySlotOffset.
  //   - Everything else (scalar ops, Index, casts, WriteSlot) writes
  //     to a TEMP slot. Shift by regOffset.
  // (WriteSlot is emitted by remapInstancePlan AFTER the body and uses
  // an absolute slot index, so it doesn't go through this remap.)
  const dstShiftFor = (instr: NInstr): number => {
    if (instr.loop_count > 1) return ctx.arraySlotOffset
    if (instr.tag === 'Pack')       return ctx.arraySlotOffset
    if (instr.tag === 'SetElement') return ctx.arraySlotOffset
    return ctx.regOffset
  }

  const body: NInstr[] = plan.instructions.map(instr => ({
    ...instr,
    dst: instr.dst + dstShiftFor(instr),
    args: instr.args.map(remapOperand),
  }))

  // After the body, emit one WriteSlot per output port. The output
  // value lives in plan.output_targets[i] (a temp index, pre-offset);
  // we shift it into the unified register space and route to the
  // instance's output slot.
  const writeSlots: NInstr[] = []
  for (let i = 0; i < ctx.outputPortNames.length; i++) {
    const portName = ctx.outputPortNames[i]
    const slotIdx  = ctx.outputSlotFor(portName)
    const tempIdx  = plan.output_targets[i]
    if (tempIdx === undefined) {
      throw new Error(
        `compileSessionSlotted: instance '${ctx.instanceName}' missing output_targets[${i}] ` +
        `for port '${portName}'.`,
      )
    }
    const scalarType = ctx.outputScalarTypes[i]
    writeSlots.push({
      tag: 'WriteSlot',
      dst: slotIdx,
      args: [{ kind: 'reg', slot: tempIdx + ctx.regOffset, scalar_type: scalarType }],
      loop_count: 1,
      strides: [],
      result_type: scalarType,
    })
  }

  return {
    preamble,
    body,
    writeSlots,
    tempsConsumed: preambleNextTemp - (ctx.regOffset + plan.register_count),
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// DAC stitching: read graphOutput source slots into fresh temps,
// expose them through output_targets + outputs so the kernel's
// existing mix-bus mechanism sums them into the audio buffer.
// ─────────────────────────────────────────────────────────────────────────────

export interface DacStitchResult {
  /** Instructions that read each graphOutput's slot into a fresh temp. */
  instructions: NInstr[]
  /** Temp indices added (one per graphOutput). These get appended to
   *  the unified plan's output_targets. */
  outputTargets: number[]
  /** Indices into outputTargets — one per graphOutput, identity mapping. */
  outputs: number[]
  /** How many new temps were allocated. The caller adds this to the
   *  cumulative register count. */
  tempCount: number
}

/** Emit DAC stitching for the session's graphOutputs. Each entry pulls
 *  its source instance's output slot into a fresh temp; the kernel's
 *  mix mechanism handles the final sum-to-output-buffer. */
export function emitDacStitch(
  session: SessionState,
  regOffsetAtDacStart: number,
): DacStitchResult {
  const instructions: NInstr[] = []
  const outputTargets: number[] = []
  const outputs: number[] = []
  let nextTemp = regOffsetAtDacStart

  for (let i = 0; i < session.graphOutputs.length; i++) {
    const go = session.graphOutputs[i]
    const key = `${go.instance}.${go.output}`
    const slotIdx = session.outputSlotRegistry.get(key)
    if (slotIdx === undefined) {
      throw new SlotShapeUnsupportedError(
        `compileSessionSlotted: dac wire '${key}' has no allocated output slot. ` +
        `(Did add_instance populate outputSlotRegistry for the source instance?)`,
      )
    }
    // Read the slot into a fresh temp via an Add+const(0) op. Cheaper
    // than introducing a dedicated "ReadSlot" instruction — LLVM will
    // fold the +0 away.
    const tempIdx = nextTemp++
    instructions.push({
      tag: 'Add',
      dst: tempIdx,
      args: [
        { kind: 'slot',  index: slotIdx, scalar_type: 'float' },
        { kind: 'const', val: 0,         scalar_type: 'float' },
      ],
      loop_count: 1,
      strides: [],
      result_type: 'float',
    })
    outputTargets.push(tempIdx)
    outputs.push(i)
  }

  return {
    instructions,
    outputTargets,
    outputs,
    tempCount: outputTargets.length,
  }
}
