/**
 * compile_session_slotted_helpers.ts — per-instance compile-then-merge.
 *
 * For each session instance, `compileResolved` is called on the
 * instance's standalone `Compiled.prog`. The resulting
 * `PerInstancePlan` is then remapped into the unified plan_5 space:
 *
 *  - `input` operands are rewritten to `slot` operands pointing at
 *    the wire's source output slot, or `const` for literal inputs.
 *  - `reg` / `state_reg` / `array_reg` indices are shifted by per-
 *    instance offsets in their respective namespaces.
 *  - `dst` is shifted by the namespace appropriate to the
 *    instruction's writeback class — guaranteed at the type level
 *    via `DstSlot`'s discriminated tag.
 *  - `register_targets` entries are either a temp (shifted) or the
 *    `arrayManaged` sentinel (passed through structurally — no
 *    arithmetic possible).
 *  - A `WriteSlot` is appended per output port to publish the
 *    computed value into the instance's allocated output slot.
 *
 * After all instances are processed, the scheduler postamble's DAC
 * stitch reads each graphOutput's source slot into a fresh temp; the
 * kernel mix-bus sums them into the audio buffer.
 *
 * ## Branded-types discipline
 *
 * Offsets are typed as `TempOffset` / `StateRegOffset` /
 * `ArraySlotOffset` so mixing them across namespaces is a compile
 * error. The `dstShiftFor` rule that caused the Phaser segfault is
 * now expressed as a `switch` on the `DstSlot` tag; adding a new
 * dst-namespace kind triggers a non-exhaustive-match warning.
 */

import type { SessionState, ExprNode } from '../session.js'
import type { PerInstancePlan, RegTarget } from '../flat_plan.js'
import { TempTarget, ArrayManagedTarget } from '../flat_plan.js'
import type {
  NOperand, NInstr, ScalarType, DstSlot,
} from './emit_resolved.js'
import {
  BINARY_TAG, UNARY_TAG, TERNARY_TAG,
  instrScalar, instrArray, instrPack, instrSetElement, instrIndex,
  instrWriteSlot,
  opConst, opTemp, opSlot, opStateReg, opArray, opRate, opTick,
} from './emit_resolved.js'
import {
  type TempIdx, type StateRegIdx, type ArraySlotIdx, type ModuleSlotIdx,
  type TempOffset, type StateRegOffset, type ArraySlotOffset,
  tempIdx, stateRegIdx, arraySlotIdx, moduleSlotIdx,
  tempOffset, stateRegOffset, arraySlotOffset,
  shiftTemp, shiftStateReg, shiftArraySlot,
  rawIdx, rawOffset,
  ZERO_TEMP_OFFSET,
} from './slot_indices.js'
import { topologicalSort } from '../compiler.js'
import {
  parseWireKey, slotKey,
  instanceName as toInstanceName,
} from './branded_names.js'

/** Raised when a session uses a shape the per-instance compile path
 *  doesn't yet support. Marker class for diagnostics. */
export class SlotShapeUnsupportedError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'SlotShapeUnsupportedError'
  }
}

/** Resolved binding for a module input port. */
export type InputBinding =
  | { kind: 'wired';   expr:  ExprNode }
  | { kind: 'literal'; value: number | boolean }

/** Allocates fresh temps in the unified register space and collects
 *  NInstrs. Used by `translateNode` for input expressions, alive
 *  expressions, and DAC stitch — all of which share the same
 *  per-sample temp namespace. */
export interface PreambleEmitter {
  instrs: NInstr[]
  /** Allocate a fresh unified-space temp slot. */
  allocTemp(): TempIdx
}

/** Translate an arbitrary ExprNode into NInstrs emitted to the
 *  preamble, returning the operand that holds the result.
 *
 *  Exported so `extractSessionDelays`'s WriteSlot emission can reuse
 *  the same translator that drives input wires. */
export function translateNode(
  expr: ExprNode,
  scalarType: ScalarType,
  session: SessionState,
  emitter: PreambleEmitter,
  context: string,
): NOperand {
  // ── leaves ──
  if (typeof expr === 'number') {
    return opConst(expr, scalarType)
  }
  if (typeof expr === 'boolean') {
    return opConst(expr ? 1 : 0, scalarType)
  }
  if (Array.isArray(expr)) {
    throw new SlotShapeUnsupportedError(
      `compileSessionSlotted: array-shaped input expression at '${context}' ` +
      `not yet supported.`,
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
    const key = slotKey(toInstanceName(obj.instance), obj.output)
    const slotIdxRaw = session.outputSlotRegistry.get(key)
    if (slotIdxRaw === undefined) {
      throw new SlotShapeUnsupportedError(
        `compileSessionSlotted: wire src '${key}' at '${context}' has no allocated output slot.`,
      )
    }
    return opSlot(moduleSlotIdx(slotIdxRaw), scalarType)
  }
  if ((op === 'param' || op === 'paramExpr'
       || op === 'trigger' || op === 'triggerParamExpr')
      && typeof obj.name === 'string') {
    const slotIdxRaw = session.paramSlotRegistry.get(obj.name)
    if (slotIdxRaw === undefined) {
      throw new SlotShapeUnsupportedError(
        `compileSessionSlotted: param '${obj.name}' at '${context}' has no allocated slot.`,
      )
    }
    return opSlot(moduleSlotIdx(slotIdxRaw), scalarType)
  }

  // ── builtins ──
  if (op === 'sampleRate')  return { kind: 'rate', scalar_type: scalarType }
  if (op === 'sampleIndex') return { kind: 'tick', scalar_type: scalarType }

  // ── session-internal slot read (emitted by extractSessionDelays) ──
  // The wire was a `delay()` that got hoisted to a module slot in the
  // pre-emit pass. The slot is updated each sample by the scheduler's
  // state_evolution phase; reading it returns the previous sample's
  // source value — the unit-delay endomorphism.
  if (op === 'sessionSlot' && typeof obj.index === 'number') {
    return opSlot(moduleSlotIdx(obj.index), scalarType)
  }

  // ── arithmetic / logical / comparison ──
  if (typeof op === 'string' && BINARY_TAG[op]) {
    const args = (obj.args as ExprNode[])
    if (args.length !== 2) {
      throw new Error(
        `compileSessionSlotted: binary op '${op}' at '${context}' needs 2 args, got ${args.length}.`,
      )
    }
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
    emitter.instrs.push(instrScalar(tag, dst, [a, b], resultType))
    return opTemp(dst, resultType)
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
    emitter.instrs.push(instrScalar(tag, dst, [a], resultType))
    return opTemp(dst, resultType)
  }
  if (typeof op === 'string' && TERNARY_TAG[op]) {
    const args = (obj.args as ExprNode[])
    if (args.length !== 3) {
      throw new Error(
        `compileSessionSlotted: ternary op '${op}' at '${context}' needs 3 args, got ${args.length}.`,
      )
    }
    const tag = TERNARY_TAG[op]
    const condType: ScalarType = tag === 'Select' ? 'bool' : scalarType
    const a = translateNode(args[0], condType, session, emitter, context)
    const b = translateNode(args[1], scalarType, session, emitter, context)
    const c = translateNode(args[2], scalarType, session, emitter, context)
    const dst = emitter.allocTemp()
    emitter.instrs.push(instrScalar(tag, dst, [a, b, c], scalarType))
    return opTemp(dst, scalarType)
  }

  throw new SlotShapeUnsupportedError(
    `compileSessionSlotted: input expression at '${context}' uses op '${op}' which is not ` +
    `yet supported.`,
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

// ─────────────────────────────────────────────────────────────────────────────
// Topological order
// ─────────────────────────────────────────────────────────────────────────────

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
  for (const v of Object.values(obj)) {
    if (typeof v === 'object' && v !== null) {
      collectInstanceRefs(v as ExprNode, out)
    }
  }
}

/** Build a topological order over session instances. Cycles do not
 *  throw; back-edges resolve through slots' previous-sample values
 *  (the slot is the unit-delay endomorphism). */
export function computeInstanceTopoOrder(session: SessionState): string[] {
  const deps = new Map<string, Set<string>>()
  for (const name of session.instanceRegistry.keys()) {
    deps.set(name, new Set())
  }
  for (const [key, expr] of session.inputExprNodes) {
    let consumer
    try { consumer = parseWireKey(key).instance } catch { continue }
    const producers = deps.get(consumer)
    if (producers === undefined) continue
    collectInstanceRefs(expr, producers)
    producers.delete(consumer)
  }
  const result = topologicalSort(deps)
  if (result.complete) return result.order

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

/** Context for remapping a single instance's PerInstancePlan into
 *  the unified session plan space. */
export interface RemapContext {
  instanceName: string
  regOffset:       TempOffset
  stateRegOffset:  StateRegOffset
  arraySlotOffset: ArraySlotOffset
  inputBindingFor: (portName: string) => InputBinding
  outputSlotFor:   (portName: string) => ModuleSlotIdx
  inputPortNames:  string[]
  outputPortNames: string[]
  outputScalarTypes: ScalarType[]
}

function inputBindingToOperand(
  binding: InputBinding,
  scalarType: ScalarType,
  session: SessionState,
  context: string,
  emitter: PreambleEmitter,
): NOperand {
  if (binding.kind === 'literal') {
    const v = typeof binding.value === 'boolean' ? (binding.value ? 1 : 0) : binding.value
    return opConst(v, scalarType)
  }
  return translateNode(binding.expr, scalarType, session, emitter, context)
}

/** Shift a `DstSlot` by the appropriate per-instance offset. The
 *  pattern-match exhausts the `DstSlot` union; adding a new dst
 *  namespace forces this function to fail compilation, exactly the
 *  forcing function the original `dstShiftFor` lacked. */
function shiftDst(dst: DstSlot, ctx: RemapContext): DstSlot {
  switch (dst.kind) {
    case 'temp':       return { kind: 'temp',       slot:  shiftTemp(dst.slot, ctx.regOffset) }
    case 'array':      return { kind: 'array',      slot:  shiftArraySlot(dst.slot, ctx.arraySlotOffset) }
    // Pre-remap-only dst. Slot is session-absolute (an index into
    // session.ioArraySlot*); collapse to plain 'array' with the
    // same value — the slot is already an absolute index into the
    // FlatPlan's array_slot_sizes (session I/O slots occupy the
    // bottom of that namespace per compileSessionSlotted's accumulator
    // seeding). The categorical move: at this boundary, the
    // kernel-local vs session-level distinction has done its job
    // (telling shiftDst which arithmetic to apply); past here,
    // every array dst is just an absolute global index.
    case 'sessionArray': return { kind: 'array', slot: dst.slot }
    // Module slots (WriteSlot dst) are absolute already — the
    // session compiler emits them with the final slot index.
    case 'moduleSlot': return dst
  }
}

/** Apply per-instance operand and slot-index remapping. Returns
 *  fresh instructions ready to merge into the session plan. */
export function remapInstancePlan(
  plan: PerInstancePlan,
  ctx: RemapContext,
  session: SessionState,
): {
  preamble:             NInstr[]
  /** Per-child pre-input WriteSlot blocks, parallel to the body's
   *  sub-instance order. Each block runs in THIS instance's namespace
   *  immediately before the corresponding child's body. */
  perChildPreInput:     NInstr[][]
  body:                 NInstr[]
  writeSlots:           NInstr[]
  tempsConsumed:        number
} {
  // Preamble emitter: allocates fresh temps in the unified register
  // space, starting after the instance's own register_count block.
  let preambleNext = rawOffset(ctx.regOffset) + plan.register_count
  const preamble: NInstr[] = []
  const emitter: PreambleEmitter = {
    instrs: preamble,
    allocTemp: () => {
      const slot = tempIdx(preambleNext)
      preambleNext += 1
      return slot
    },
  }

  const remapOperand = (op: NOperand): NOperand => {
    switch (op.kind) {
      case 'const': return op
      case 'rate':  return op
      case 'tick':  return op
      case 'slot':  return op
      case 'input': {
        const portName = ctx.inputPortNames[rawIdx(op.slot)]
        if (portName === undefined) {
          throw new Error(
            `compileSessionSlotted: instance '${ctx.instanceName}' input operand slot=${rawIdx(op.slot)} ` +
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
      case 'reg':       return { ...op, slot: shiftTemp(op.slot, ctx.regOffset) }
      case 'state_reg': return { ...op, slot: shiftStateReg(op.slot, ctx.stateRegOffset) }
      case 'array_reg': return { ...op, slot: shiftArraySlot(op.slot, ctx.arraySlotOffset) }
      // Pre-remap-only operand: slot is session-absolute. Collapse to
      // plain `array_reg` with the same slot value; the kernel-local
      // shift doesn't apply because the slot was never in the
      // kernel-local namespace to begin with. See `shiftDst`'s
      // 'sessionArray' case for the symmetric move on writebacks.
      case 'session_array_reg': return { kind: 'array_reg', slot: op.slot }
      case 'param':
        // Session-level params resolve to `slot` operands via
        // `translateNode` before reaching here. A surviving `param`
        // means a type-level inline ExprNode used `{op:'param'}` —
        // not currently supported.
        throw new SlotShapeUnsupportedError(
          `compileSessionSlotted: legacy 'param' operand encountered ` +
          `in '${ctx.instanceName}'. Session-level params should resolve ` +
          `to slot operands before this point.`,
        )
    }
  }

  const body: NInstr[] = plan.instructions.map(instr => ({
    ...instr,
    dst:  shiftDst(instr.dst, ctx),
    args: instr.args.map(remapOperand),
  }))

  // Slot-based parent→child input wiring: each child's pre-input block
  // (parent's wires → WriteSlot to that child's input slot) gets the
  // same per-instance operand/dst remap as the main body. The blocks
  // reference the same temp/state/array spaces (Emitter allocates them
  // in one namespace), and WriteSlot dst's are already absolute
  // module-slot indices. Returned as a per-child array so the caller
  // can attach each block to its corresponding child InstanceFunction
  // — the engine then dispatches `block → child.body` interleaved.
  const perChildPreInput: NInstr[][] = plan.per_child_pre_input.map(block =>
    block.map(instr => ({
      ...instr,
      dst:  shiftDst(instr.dst, ctx),
      args: instr.args.map(remapOperand),
    })),
  )

  // Output writebacks per declared port. Two dispatches:
  //
  //   scalar/alias ports → one WriteSlot to a module slot
  //     (output_targets contributes 1 temp per scalar element).
  //   array ports        → N SetElements into the port's session-
  //     absolute array slot (output_targets contributes N temps,
  //     one per element, in row-major order).
  //
  // We're now in session-level code (post per-instance emit, outside
  // the kernel's local namespace) and constructing instructions in
  // absolute coordinates directly — so writes to the array slot use
  // `instrSetElement` (`array` dst kind, not the pre-remap
  // `sessionArray`). Same for the `opArray` operand. The kernel-local
  // ↔ session-absolute distinction only matters for instructions
  // built INSIDE `emit_resolved`'s Emitter, where it informs the
  // later remap shift; writeSlots constructed here never pass
  // through remap.
  //
  // `plan.output_targets` is in port-major-then-element-major order;
  // we walk it monotonically.
  const writeSlots: NInstr[] = []
  let targetIdx = 0
  for (let portI = 0; portI < ctx.outputPortNames.length; portI++) {
    const portName = ctx.outputPortNames[portI]
    const portKey  = slotKey(toInstanceName(ctx.instanceName), portName)
    const meta = session.outputPortMeta.get(portKey)
    if (meta === undefined) {
      throw new Error(
        `compileSessionSlotted: instance '${ctx.instanceName}' port '${portName}' ` +
        `missing outputPortMeta entry (allocateOutputSlots should have run).`,
      )
    }
    // Array port dispatch — meta.arraySlot is the session-absolute
    // array-slot index; meta.arraySize is the element count.
    if (meta.arraySlot !== undefined && meta.arraySize !== undefined) {
      const arrSlot = arraySlotIdx(meta.arraySlot)
      const arrOp   = opArray(arrSlot)
      for (let elemI = 0; elemI < meta.arraySize; elemI++) {
        const localTemp = plan.output_targets[targetIdx]
        if (localTemp === undefined) {
          throw new Error(
            `compileSessionSlotted: instance '${ctx.instanceName}' missing output_targets[${targetIdx}] ` +
            `for array port '${portName}' element ${elemI}.`,
          )
        }
        const absTemp = shiftTemp(localTemp, ctx.regOffset)
        writeSlots.push(instrSetElement(
          arrSlot,
          [arrOp, opConst(elemI, 'int'), opTemp(absTemp, 'float')],
        ))
        targetIdx++
      }
      continue
    }
    // Scalar/alias port dispatch.
    for (let scalarI = 0; scalarI < meta.scalarSlotNames.length; scalarI++) {
      const scalarSlotName = meta.scalarSlotNames[scalarI]
      const slotIdxRaw = session.outputSlotRegistry.get(scalarSlotName)
      if (slotIdxRaw === undefined) {
        throw new Error(
          `compileSessionSlotted: scalar slot '${scalarSlotName}' not in outputSlotRegistry.`,
        )
      }
      const slotIdx = moduleSlotIdx(slotIdxRaw)
      const localTemp = plan.output_targets[targetIdx]
      if (localTemp === undefined) {
        throw new Error(
          `compileSessionSlotted: instance '${ctx.instanceName}' missing output_targets[${targetIdx}] ` +
          `for scalar slot '${scalarSlotName}' (port '${portName}', element ${scalarI}).`,
        )
      }
      const scalarType = meta.scalarTypes[scalarI]
      const absTemp = shiftTemp(localTemp, ctx.regOffset)
      writeSlots.push(instrWriteSlot(slotIdx, opTemp(absTemp, scalarType), scalarType))
      targetIdx++
    }
  }
  if (targetIdx !== plan.output_targets.length) {
    throw new Error(
      `compileSessionSlotted: instance '${ctx.instanceName}' has ${plan.output_targets.length} ` +
      `output_targets but only ${targetIdx} were consumed by slot expansion. ` +
      `This indicates a port-shape / emit mismatch.`,
    )
  }

  return {
    preamble,
    perChildPreInput,
    body,
    writeSlots,
    tempsConsumed: preambleNext - (rawOffset(ctx.regOffset) + plan.register_count),
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// DAC stitching
// ─────────────────────────────────────────────────────────────────────────────

export interface DacStitchResult {
  instructions:  NInstr[]
  /** Temp indices added (one per graphOutput). Appended to the
   *  unified plan's `output_targets`. */
  outputTargets: TempIdx[]
  outputs:       number[]
  tempCount:     number
}

/** Emit DAC stitching: each graphOutput's source slot is read into
 *  a fresh temp; the kernel mix bus sums those temps into the audio
 *  output buffer. Runs in the scheduler postamble so it observes
 *  the current sample's WriteSlots (alive instances have written;
 *  asleep instances retain). */
export function emitDacStitch(
  session: SessionState,
  regOffsetAtDacStart: TempOffset,
): DacStitchResult {
  const instructions:  NInstr[]   = []
  const outputTargets: TempIdx[]  = []
  const outputs:       number[]   = []
  let nextTemp = rawOffset(regOffsetAtDacStart)

  for (let i = 0; i < session.graphOutputs.length; i++) {
    const go = session.graphOutputs[i]
    const key = slotKey(toInstanceName(go.instance), go.output)
    const slotIdxRaw = session.outputSlotRegistry.get(key)
    if (slotIdxRaw === undefined) {
      throw new SlotShapeUnsupportedError(
        `compileSessionSlotted: dac wire '${key}' has no allocated output slot.`,
      )
    }
    const dst = tempIdx(nextTemp); nextTemp += 1
    // Read slot → temp via `Add slot, 0` (LLVM folds the +0).
    instructions.push(instrScalar(
      'Add', dst,
      [opSlot(moduleSlotIdx(slotIdxRaw), 'float'), opConst(0, 'float')],
      'float',
    ))
    outputTargets.push(dst)
    outputs.push(i)
  }

  return {
    instructions,
    outputTargets,
    outputs,
    tempCount: outputTargets.length,
  }
}

// Re-export utilities the session compiler also needs.
export {
  tempOffset, stateRegOffset, arraySlotOffset,
  tempIdx, stateRegIdx, arraySlotIdx, moduleSlotIdx,
  shiftTemp, shiftStateReg, shiftArraySlot,
  rawOffset, rawIdx,
  ZERO_TEMP_OFFSET,
}
// Unused imports kept around for the re-export surface; quiet the
// linter.
void instrArray; void instrPack; void instrSetElement; void instrIndex
void opStateReg; void opArray
