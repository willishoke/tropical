/**
 * compiler/ir/emit_resolved.ts — `ResolvedExpr → FlatProgram` emitter.
 *
 * Walks a `ResolvedExpr` directly via a single emitter pass. Refs
 * become operand kinds via decl-identity slot lookups; the dispatch
 * is exhaustive over the closed `ResolvedExprOp` union.
 *
 * ## Branded internal IR (active-set rigor refactor)
 *
 * The emitter produces a `FlatProgram` whose every integer namespace
 * carries a brand (`TempIdx`, `StateRegIdx`, `ArraySlotIdx`,
 * `InputPortIdx`). `NInstr.dst` is a tagged `DstSlot` union rather
 * than a bare `number`, so the namespace (temp / array / module
 * slot) is explicit at construction time. The two production bugs
 * that prompted this refactor — `-1` sentinel arithmetic and
 * dst-namespace confusion — are unrepresentable in this shape.
 *
 * All instruction emission goes through typed constructors
 * (`instrScalar`, `instrArray`, `instrPack`, `instrSetElement`,
 * `instrIndex`, `instrWriteSlot`, `instrSmoothParam`) — direct
 * object-literal construction of `NInstr` is avoided. Constructors
 * enforce the dst's namespace matches the tag's writeback class.
 *
 * ## Wire format
 *
 * `WireNInstr` and `WireNOperand` are plain JSON-shaped types with
 * raw `number` indices and a flat `dst: number`. `toWireInstr(i)`
 * collapses the discriminated dst back to a number for the JSON
 * serialization boundary; the engine parses that shape.
 */

import type {
  ResolvedExpr, ResolvedExprOp, ResolvedProgram,
  RegDecl, InputDecl, InstanceDecl, OutputDecl, ParamDecl, PortType,
  InputIdx, OutputIdx, ParamIdx, InstanceIdx,
} from './nodes.js'
import { instanceIdx, inputIdx as inputIdxOf } from './nodes.js'
import { getInstanceType } from './decl_tables.js'
import {
  type TempIdx, type StateRegIdx, type ArraySlotIdx, type ModuleSlotIdx,
  type InputPortIdx,
  tempIdx, arraySlotIdx, stateRegIdx, inputPortIdx, moduleSlotIdx,
  rawIdx,
} from './slot_indices.js'
import type { RegTarget } from '../flat_plan.js'
import { TempTarget, ArrayManagedTarget } from '../flat_plan.js'

// ─────────────────────────────────────────────────────────────
// Public types — operand variants with branded slot indices
// ─────────────────────────────────────────────────────────────

export type ScalarType = 'float' | 'int' | 'bool'

export type NOperand =
  | { kind: 'const';             val: number;          scalar_type: ScalarType }
  | { kind: 'input';             slot: InputPortIdx;   scalar_type: ScalarType }
  | { kind: 'reg';               slot: TempIdx;        scalar_type: ScalarType }
  | { kind: 'array_reg';         slot: ArraySlotIdx }
  /** Session-level array slot operand. Pre-remap-only kind: carries a
   *  session-absolute array slot index (into `session.ioArraySlot*`).
   *  `remapInstancePlan` converts to `array_reg` (passthrough slot value)
   *  on its way to the FlatPlan, so the wire format never sees this
   *  kind. Construction sites: array-typed `InputRef` / `NestedOut` in
   *  per-instance compile, and array-typed source `ref` in session
   *  translateNode. The categorical move this encodes: keep the
   *  kernel-local vs session-level distinction as IR tag through
   *  remap (where the distinction is consumed), drop it at the
   *  FlatPlan boundary where the engine genuinely doesn't care. */
  | { kind: 'session_array_reg'; slot: ArraySlotIdx }
  | { kind: 'state_reg';         slot: StateRegIdx;    scalar_type: ScalarType }
  | { kind: 'param';             ptr:  string;         scalar_type: ScalarType }
  | { kind: 'rate';                                    scalar_type: ScalarType }
  | { kind: 'tick';                                    scalar_type: ScalarType }
  | { kind: 'slot';              index: ModuleSlotIdx; scalar_type: ScalarType }

/** Discriminated dst — the namespace of an instruction's writeback. */
export type DstSlot =
  | { kind: 'temp';         slot: TempIdx }
  | { kind: 'array';        slot: ArraySlotIdx }
  /** Pre-remap-only dst. Mirror of `session_array_reg`: writes to a
   *  session-absolute array slot (e.g., parent emitting `arraySet`
   *  against a child's input array slot, or an array-typed output
   *  port's element-store). `remapInstancePlan`'s `shiftDst` converts
   *  to `kind: 'array'` (passthrough slot value). Wire format never
   *  sees this kind — `dstSlotToWire` throws if it does. */
  | { kind: 'sessionArray'; slot: ArraySlotIdx }
  | { kind: 'moduleSlot';   index: ModuleSlotIdx }

export type NInstr = {
  tag:         string
  dst:         DstSlot
  args:        NOperand[]
  loop_count:  number
  strides:     number[]
  result_type: ScalarType
}

export interface FlatProgram {
  register_count:   number
  array_slot_count: number
  array_slot_sizes: number[]
  instructions:     NInstr[]
  /** Slot-based parent→child input wiring (the fractal path).
   *  Parallel to the caller's `nestedInstances` array:
   *  `per_child_pre_input[k]` holds the
   *  WriteSlot block for the k-th sub-instance — the parent runs that
   *  block in its own namespace immediately before invoking that
   *  child's kernel body. Per-child placement (vs hoisting into a
   *  single pre-children block) preserves sibling-to-sibling NestedOut
   *  dependencies, since child[j]'s body has already produced its
   *  outputs by the time child[k]'s pre-input wires evaluate (k > j).
   *  Empty array for leaf kernels and the legacy flat path. */
  per_child_pre_input: NInstr[][]
  /** Per-output-port temp index (local; the session compiler shifts). */
  output_targets:   TempIdx[]
  register_targets: RegTarget[]
}

// ─── Instruction constructors ───────────────────────────────────────────────
// Each constructor brands its dst with the right namespace. The type
// system enforces the brand at the call site.

export const instrScalar = (
  tag: string, dst: TempIdx, args: NOperand[], result_type: ScalarType,
): NInstr => ({
  tag, dst: { kind: 'temp', slot: dst }, args,
  loop_count: 1, strides: [], result_type,
})

export const instrArray = (
  tag: string, dst: ArraySlotIdx, args: NOperand[],
  loop_count: number, strides: number[], result_type: ScalarType,
): NInstr => ({
  tag, dst: { kind: 'array', slot: dst }, args,
  loop_count, strides, result_type,
})

/** Pre-remap-only: elementwise instruction writing to a session-
 *  absolute array slot. Mirrors `instrArray`; remap converts the dst's
 *  kind 'sessionArray' → 'array' (passthrough slot value). Used by
 *  parent→child array-input wiring (elementwise copy from a wire's
 *  array source into the child's input array slot) and array-typed
 *  output emission (copy from the body's computed array into the
 *  port's session-absolute output array slot). */
export const instrSessionArray = (
  tag: string, dst: ArraySlotIdx, args: NOperand[],
  loop_count: number, strides: number[], result_type: ScalarType,
): NInstr => ({
  tag, dst: { kind: 'sessionArray', slot: dst }, args,
  loop_count, strides, result_type,
})

export const instrPack = (dst: ArraySlotIdx, args: NOperand[]): NInstr => ({
  tag: 'Pack', dst: { kind: 'array', slot: dst }, args,
  loop_count: 1, strides: [], result_type: 'float',
})

export const instrSetElement = (
  dst: ArraySlotIdx, args: [NOperand, NOperand, NOperand],
): NInstr => ({
  tag: 'SetElement', dst: { kind: 'array', slot: dst }, args,
  loop_count: 1, strides: [], result_type: 'float',
})

/** Pre-remap-only: SetElement against a session-absolute array slot.
 *  remapInstancePlan converts `dst.kind: 'sessionArray'` → `'array'`
 *  (passthrough slot value) so the post-remap FlatPlan only ever
 *  contains 'array' dsts. */
export const instrSessionSetElement = (
  dst: ArraySlotIdx, args: [NOperand, NOperand, NOperand],
): NInstr => ({
  tag: 'SetElement', dst: { kind: 'sessionArray', slot: dst }, args,
  loop_count: 1, strides: [], result_type: 'float',
})

export const instrIndex = (
  dst: TempIdx, args: [NOperand, NOperand], result_type: ScalarType,
): NInstr => ({
  tag: 'Index', dst: { kind: 'temp', slot: dst }, args,
  loop_count: 1, strides: [], result_type,
})

export const instrWriteSlot = (
  dst: ModuleSlotIdx, value: NOperand, scalar_type: ScalarType = 'float',
): NInstr => ({
  tag: 'WriteSlot', dst: { kind: 'moduleSlot', index: dst }, args: [value],
  loop_count: 1, strides: [], result_type: scalar_type,
})

export const instrSmoothParam = (
  dst: TempIdx, paramPtr: string, stateRegSlot: StateRegIdx, coeff: number,
): NInstr => ({
  tag: 'SmoothParam', dst: { kind: 'temp', slot: dst },
  args: [
    { kind: 'param', ptr: paramPtr, scalar_type: 'float' },
    { kind: 'state_reg', slot: stateRegSlot, scalar_type: 'float' },
    { kind: 'const', val: coeff, scalar_type: 'float' },
  ],
  loop_count: 1, strides: [], result_type: 'float',
})

// ─── Operand constructors (typed) ───────────────────────────────────────────

export const opConst    = (val: number, scalar_type: ScalarType = 'float'): NOperand =>
  ({ kind: 'const', val, scalar_type })
export const opTemp     = (slot: TempIdx, scalar_type: ScalarType): NOperand =>
  ({ kind: 'reg', slot, scalar_type })
export const opArray    = (slot: ArraySlotIdx): NOperand =>
  ({ kind: 'array_reg', slot })
/** Pre-remap-only operand. See `session_array_reg` doc on NOperand for
 *  the lifecycle. Slot is a session-absolute index into
 *  `session.ioArraySlot*`. */
export const opSessionArray = (slot: ArraySlotIdx): NOperand =>
  ({ kind: 'session_array_reg', slot })
export const opStateReg = (slot: StateRegIdx, scalar_type: ScalarType): NOperand =>
  ({ kind: 'state_reg', slot, scalar_type })
export const opInput    = (slot: InputPortIdx, scalar_type: ScalarType): NOperand =>
  ({ kind: 'input', slot, scalar_type })
export const opSlot     = (index: ModuleSlotIdx, scalar_type: ScalarType): NOperand =>
  ({ kind: 'slot', index, scalar_type })
export const opParam    = (ptr: string, scalar_type: ScalarType = 'float'): NOperand =>
  ({ kind: 'param', ptr, scalar_type })
export const opRate: NOperand = { kind: 'rate', scalar_type: 'float' }
export const opTick: NOperand = { kind: 'tick', scalar_type: 'int' }

// ─── Wire format ────────────────────────────────────────────────────────────
// The C++ engine parses this shape from JSON. Brands erase at runtime
// so operand fields auto-flatten; only `dst` (the DstSlot
// discriminated union) needs structural conversion to a plain number.

export type WireNOperand =
  | { kind: 'const';     val: number; scalar_type: ScalarType }
  | { kind: 'input';     slot: number; scalar_type: ScalarType }
  | { kind: 'reg';       slot: number; scalar_type: ScalarType }
  | { kind: 'array_reg'; slot: number }
  | { kind: 'state_reg'; slot: number; scalar_type: ScalarType }
  | { kind: 'param';     ptr: string;  scalar_type: ScalarType }
  | { kind: 'rate';      scalar_type: ScalarType }
  | { kind: 'tick';      scalar_type: ScalarType }
  | { kind: 'slot';      index: number; scalar_type: ScalarType }

export type WireDstKind = 'temp' | 'array' | 'moduleSlot'

export interface WireNInstr {
  tag: string
  /** The slot index in the writeback namespace selected by `dst_kind`. */
  dst: number
  /** Discriminator for `dst` — preserves the in-memory `DstSlot` union's
   *  kind tag through serialization. Reconstructed by the engine into
   *  a typed `DstKind` so dispatch in `emit_instr` is direct (no
   *  reconstruction from `tag + loop_count` proxies, which silently
   *  misclassifies degenerate `loop_count==1` array writes). */
  dst_kind: WireDstKind
  args: WireNOperand[]
  loop_count: number
  strides: number[]
  result_type: ScalarType
}

/** Project the `DstSlot` kind tag to its wire-format string. Mirrors
 *  `dstSlotToWire`'s exhaustive switch so the two stay in lockstep. */
export const dstSlotKindToWire = (d: DstSlot): WireDstKind => {
  switch (d.kind) {
    case 'temp':         return 'temp'
    case 'array':        return 'array'
    case 'moduleSlot':   return 'moduleSlot'
    case 'sessionArray':
      throw new Error(
        `dstSlotKindToWire: 'sessionArray' dst leaked to wire format. ` +
        `remapInstancePlan must convert it to 'array' before serialization.`,
      )
  }
}

export const dstSlotToWire = (d: DstSlot): number => {
  switch (d.kind) {
    case 'temp':         return rawIdx(d.slot)
    case 'array':        return rawIdx(d.slot)
    case 'moduleSlot':   return rawIdx(d.index)
    case 'sessionArray':
      // Pre-remap kind reaching the wire format means a code path
      // skipped `remapInstancePlan`. The remap is the ONLY place that
      // collapses sessionArray → array. Loud failure here surfaces the
      // bug at the boundary rather than silently corrupting the
      // FlatPlan with a phantom slot index.
      throw new Error(
        `dstSlotToWire: 'sessionArray' dst leaked to wire format. ` +
        `remapInstancePlan must convert it to 'array' before serialization.`,
      )
  }
}

// ─── DstSlot accessors that assert namespace at the call site ──────────────
// Used by consumers (emit_wasm, etc.) that know which namespace an
// instruction's dst belongs to from the tag. Wrong namespace → throw,
// catching IR-construction bugs in the call site that built the instr.

export const dstAsTemp = (i: NInstr): TempIdx => {
  if (i.dst.kind !== 'temp') {
    throw new Error(`emit: expected temp dst for tag='${i.tag}', got ${i.dst.kind}`)
  }
  return i.dst.slot
}

export const dstAsArray = (i: NInstr): ArraySlotIdx => {
  if (i.dst.kind !== 'array') {
    // sessionArray reaching this accessor is a remap-omission bug.
    // Other kinds are tag/dst mismatches at the construction site.
    const hint = i.dst.kind === 'sessionArray'
      ? ` (sessionArray indicates remapInstancePlan was bypassed)`
      : ''
    throw new Error(`emit: expected array dst for tag='${i.tag}', got ${i.dst.kind}${hint}`)
  }
  return i.dst.slot
}

export const dstAsModuleSlot = (i: NInstr): ModuleSlotIdx => {
  if (i.dst.kind !== 'moduleSlot') {
    throw new Error(`emit: expected moduleSlot dst for tag='${i.tag}', got ${i.dst.kind}`)
  }
  return i.dst.index
}

export const toWireInstr = (i: NInstr): WireNInstr => ({
  tag:         i.tag,
  dst:         dstSlotToWire(i.dst),
  dst_kind:    dstSlotKindToWire(i.dst),
  // Branded primitives erase at runtime; the cast tells TS to drop
  // the brand without producing a value-level conversion.
  args:        i.args as unknown as WireNOperand[],
  loop_count:  i.loop_count,
  strides:     i.strides,
  result_type: i.result_type,
})

// ─────────────────────────────────────────────────────────────
// Slot tables — passed in by compile_resolved.ts / compile_session.ts
// ─────────────────────────────────────────────────────────────

export interface EmitSlots {
  /** Kept for back-compat; emit_resolved now uses InputRef.idx directly
   *  (which equals position in ports.inputs[], same as the value
   *  buildSlotMaps produces). The map can be empty without breaking
   *  the new code path. */
  inputs: Map<InputDecl, number>
  /** Same: emit_resolved now uses RegRef.idx directly. */
  regs:   Map<RegDecl, number>
  /** Total scalar-register count. */
  regCount: number
  /** FFI handle metadata per param. Keyed by ParamIdx now (replacing
   *  the prior ParamDecl pointer key). */
  paramHandles: Map<ParamIdx, { ptr: string }>
  /** Module slot indices for sub-instance outputs that this kernel's
   *  body references via `NestedOut`. Map shape:
   *  InstanceIdx → OutputIdx → moduleSlotIdx. When undefined
   *  (legacy / per-program path), NestedOut throws as before. */
  nestedOutputSlots?: Map<InstanceIdx, Map<OutputIdx, number>>
  /** Module slot indices for THIS program's own input ports. When set,
   *  `InputRef(idx)` lowers to a Slot read from `inputSlotOverride.get(idx)`
   *  instead of an `Input` operand. Used by the fractal path. */
  inputSlotOverride?: Map<InputIdx, number>
  /** Per-child module-slot map for sub-instance INPUTS. Map shape:
   *  InstanceIdx → InputIdx → moduleSlotIdx. Used by the fractal
   *  path when emitting a parent kernel. */
  nestedInputSlots?: Map<InstanceIdx, Map<InputIdx, number>>
  /** Session-level array-slot indices for THIS program's own array-
   *  typed input ports. When set, `InputRef(arr_port)` lowers to an
   *  `opArray` operand pointing at the recorded slot (the parent —
   *  session-level wiring or a containing kernel — writes elements
   *  into this slot via `arraySet` in `pre_input_instructions`).
   *  Indexed by InputIdx; size info accompanies each entry. */
  inputArraySlots?: Map<InputIdx, { slot: number; size: number }>
  /** Per-child session-level array-slot indices for sub-instance
   *  array-typed INPUTS. The parent kernel writes the child's input
   *  array via `arraySet` against the recorded slot in its
   *  per-child pre_input block. */
  nestedInputArraySlots?: Map<InstanceIdx, Map<InputIdx, { slot: number; size: number }>>
  /** Per-child session-level array-slot indices for sub-instance
   *  array-typed OUTPUTS. The parent reads the child's array output
   *  via `index` against the recorded slot. */
  nestedOutputArraySlots?: Map<InstanceIdx, Map<OutputIdx, { slot: number; size: number }>>
}

// ─────────────────────────────────────────────────────────────
// Op-tag mappings
// ─────────────────────────────────────────────────────────────

export const BINARY_TAG: Record<string, string> = {
  add: 'Add', sub: 'Sub', mul: 'Mul', div: 'Div', mod: 'Mod',
  floorDiv: 'FloorDiv',
  lt: 'Less', lte: 'LessEq', gt: 'Greater', gte: 'GreaterEq',
  eq: 'Equal', neq: 'NotEqual',
  bitAnd: 'BitAnd', bitOr: 'BitOr', bitXor: 'BitXor',
  lshift: 'LShift', rshift: 'RShift',
  and: 'And', or: 'Or',
  ldexp: 'Ldexp',
}

export const UNARY_TAG: Record<string, string> = {
  neg: 'Neg', abs: 'Abs', sqrt: 'Sqrt',
  floor: 'Floor', ceil: 'Ceil', round: 'Round',
  not: 'Not', bitNot: 'BitNot',
  floatExponent: 'FloatExponent',
  toInt: 'ToInt', toBool: 'ToBool', toFloat: 'ToFloat',
}

export const TERNARY_TAG: Record<string, string> = {
  select: 'Select', clamp: 'Clamp',
}

const CAST_RESULT: Record<string, ScalarType> = {
  ToInt: 'int', ToBool: 'bool', ToFloat: 'float',
}

const BITWISE_TAGS = new Set(['BitAnd', 'BitOr', 'BitXor', 'LShift', 'RShift', 'BitNot'])
const COMPARISON_TAGS = new Set(['Less', 'LessEq', 'Greater', 'GreaterEq', 'Equal', 'NotEqual', 'Not', 'And', 'Or'])
const TRANSCENDENTAL_TAGS = new Set(['Sqrt', 'Floor', 'Ceil', 'Round', 'Ldexp', 'FloatExponent'])

function promoteTypes(a: ScalarType, b: ScalarType): ScalarType {
  if (a === 'float' || b === 'float') return 'float'
  if (a === 'int' || b === 'int') return 'int'
  return 'bool'
}

function inferResultType(tag: string, argTypes: ScalarType[]): ScalarType {
  if (CAST_RESULT[tag]) return CAST_RESULT[tag]
  if (BITWISE_TAGS.has(tag)) return 'int'
  if (COMPARISON_TAGS.has(tag)) return 'bool'
  if (TRANSCENDENTAL_TAGS.has(tag)) return 'float'
  if (tag === 'Select') return promoteTypes(argTypes[1] ?? 'float', argTypes[2] ?? 'float')
  if (tag === 'Clamp') return argTypes[0] ?? 'float'
  if (argTypes.length === 0) return 'float'
  return argTypes.reduce(promoteTypes)
}

// ─────────────────────────────────────────────────────────────
// Internal compile result
// ─────────────────────────────────────────────────────────────

type ScalarResult = { isArray: false; op: NOperand; scalarType: ScalarType }
type ArrayResult  = { isArray: true;  op: NOperand; size: number; scalarType: ScalarType }
type CompileResult = ScalarResult | ArrayResult

// ─────────────────────────────────────────────────────────────
// Emitter
// ─────────────────────────────────────────────────────────────

class Emitter {
  private nextReg       = 0
  private nextArraySlot = 0
  private arraySizes:   number[] = []
  private instrs:       NInstr[] = []

  // Structural CSE — same shape as emit_numeric (issue #131).
  private hashTable = new Map<string, number>()
  private hashCache = new WeakMap<object, number>()
  private memo      = new Map<string, CompileResult>()

  // ResolvedExpr-array regs surface as `regRef` to a regDecl whose
  // init is an array. The map keys the regDecl's slot to its
  // backing array-slot and length.
  private arrayRegMap = new Map<number, { slot: ArraySlotIdx; size: number }>()

  private regTypes = new Map<number, ScalarType>()
  private stateRegTypes: ScalarType[]
  private inputPortTypes: ScalarType[]
  private slots: EmitSlots

  constructor(
    slots: EmitSlots,
    stateInit: (number | boolean | number[])[],
    stateRegTypes: ScalarType[],
    inputPortTypes: ScalarType[],
  ) {
    this.slots = slots
    this.stateRegTypes = stateRegTypes
    this.inputPortTypes = inputPortTypes
    for (let i = 0; i < stateInit.length; i++) {
      const init = stateInit[i]
      if (Array.isArray(init)) {
        const slot = this.allocArraySlot(init.length)
        this.arrayRegMap.set(i, { slot, size: init.length })
      }
    }
  }

  private allocReg(): TempIdx {
    const slot = tempIdx(this.nextReg)
    this.nextReg += 1
    return slot
  }

  private allocArraySlot(size: number): ArraySlotIdx {
    const slot = arraySlotIdx(this.nextArraySlot)
    this.nextArraySlot += 1
    this.arraySizes.push(size)
    return slot
  }

  private emit(instr: NInstr): void {
    this.instrs.push(instr)
  }

  // ── Terminal check ──────────────────────────────────────────
  private tryTerminal(node: ResolvedExpr, expected?: ScalarType): { op: NOperand; scalarType: ScalarType } | null {
    if (typeof node === 'number') {
      const t = this.resolveNumericLiteralType(node, expected)
      return { op: opConst(node, t), scalarType: t }
    }
    if (typeof node === 'boolean') return { op: opConst(node ? 1 : 0, 'bool'), scalarType: 'bool' }
    if (Array.isArray(node)) return null
    if (typeof node !== 'object' || node === null) return { op: opConst(0, 'float'), scalarType: 'float' }
    const obj = node as ResolvedExprOp
    switch (obj.op) {
      case 'inputRef': {
        // RegIdx/InputIdx/etc are the de Bruijn levels into the
        // program's typed decl tables; slots.inputs is keyed by the
        // same InputDecl objects, in the same order. So obj.idx IS
        // the slot number (slots.ts assigns slot[i] = position i).
        const slot = obj.idx as number
        // Array-typed input port: defer to compileNodeUncached, which
        // produces an `array_reg` operand pointing at the session-
        // level array slot the port is bound to. Mirrors how state
        // arrays (regRef + arrayRegMap.has(slot)) defer for non-
        // scalar lowering.
        if (this.slots.inputArraySlots !== undefined
            && this.slots.inputArraySlots.has(obj.idx)) {
          return null
        }
        const portT = this.inputPortTypes[slot] ?? 'float'
        // Fractal path: when the program is being compiled as a
        // sub-instance kernel, its inputs live in module slots
        // pre-written by the parent's WriteSlot in per_child_pre_input.
        // Lower `InputRef(d)` to a slot read instead of opInput.
        if (this.slots.inputSlotOverride !== undefined) {
          const overrideSlot = this.slots.inputSlotOverride.get(obj.idx)
          if (overrideSlot !== undefined) {
            return { op: opSlot(moduleSlotIdx(overrideSlot), portT), scalarType: portT }
          }
        }
        return { op: opInput(inputPortIdx(slot), portT), scalarType: portT }
      }
      case 'regRef': {
        const slot = obj.idx as number
        if (this.arrayRegMap.has(slot)) return null
        const regType = this.stateRegTypes[slot] ?? 'float'
        return { op: opStateReg(stateRegIdx(slot), regType), scalarType: regType }
      }
      case 'paramRef': {
        const handle = this.slots.paramHandles.get(obj.idx)
        if (handle === undefined) {
          // No live FFI handle — emit zero, matching the legacy fallback.
          return { op: opConst(0, 'float'), scalarType: 'float' }
        }
        return { op: opParam(handle.ptr, 'float'), scalarType: 'float' }
      }
      case 'sampleRate':  return { op: opRate, scalarType: 'float' }
      case 'sampleIndex': return { op: opTick, scalarType: 'int' }
      case 'nestedOut': {
        // Array-typed sub-instance output: defer to compileNodeUncached,
        // which produces an `array_reg` operand pointing at the session-
        // level array slot the child's output port is bound to.
        if (this.slots.nestedOutputArraySlots !== undefined) {
          const perInstArr = this.slots.nestedOutputArraySlots.get(obj.instance)
          if (perInstArr !== undefined && perInstArr.has(obj.output)) {
            return null
          }
        }
        // Fractal compile: a NestedOut references a sub-instance's
        // output, which lives in a module slot allocated by
        // partition_recursive. Read it as `slot[index]`. The scalar
        // type comes from the output port's declared type.
        if (this.slots.nestedOutputSlots === undefined) {
          throw new Error(
            `emit_resolved: NestedOut to instance idx=${obj.instance} output idx=${obj.output} ` +
            `requires nestedOutputSlots — pass them via EmitSlots when invoking emit ` +
            `on a fractal kernel.`,
          )
        }
        const perInst = this.slots.nestedOutputSlots.get(obj.instance)
        if (perInst === undefined) {
          throw new Error(
            `emit_resolved: NestedOut to instance idx=${obj.instance} — ` +
            `no slot map entry. partition_recursive should record every child ` +
            `instance before emitting the parent kernel.`,
          )
        }
        const slotRaw = perInst.get(obj.output)
        if (slotRaw === undefined) {
          throw new Error(
            `emit_resolved: NestedOut to instance idx=${obj.instance} output idx=${obj.output} — ` +
            `output port not in slot map.`,
          )
        }
        // Determine the scalar type from the output port's declared type.
        // Look up the output decl via the program in scope. Without prog
        // in scope here, default to float. partition_recursive can
        // thread precise typing through nestedOutputSlots in a followup
        // if non-float NestedOuts ever need it.
        const scalarType: ScalarType = 'float'
        return { op: opSlot(moduleSlotIdx(slotRaw), scalarType), scalarType }
      }
    }
    return null
  }

  private resolveNumericLiteralType(val: number, expected?: ScalarType): ScalarType {
    if (expected === 'int') {
      if (!Number.isInteger(val)) {
        throw new Error(`Lossy conversion: literal ${val} cannot narrow to int. Wrap the source in to_int() to narrow explicitly.`)
      }
      return 'int'
    }
    if (expected === 'bool') {
      if (val !== 0 && val !== 1) {
        throw new Error(`Lossy conversion: literal ${val} cannot narrow to bool. Wrap the source in to_bool() to narrow explicitly.`)
      }
      return 'bool'
    }
    return 'float'
  }

  // ── Structural CSE id ───────────────────────────────────────
  // Ref-bearing nodes are keyed by op + decl IDENTITY (resolved IR
  // has cycles via init/update fields). Other ops hash recursively.
  private declIds = new WeakMap<object, number>()
  private nextDeclId = 0
  private declIdOf(decl: object): number {
    let id = this.declIds.get(decl)
    if (id === undefined) {
      id = this.nextDeclId++
      this.declIds.set(decl, id)
    }
    return id
  }

  private structuralId(node: object): number {
    const cached = this.hashCache.get(node)
    if (cached !== undefined) return cached
    let key: string
    if (Array.isArray(node)) {
      key = `a:${node.map(c => this.structuralKey(c)).join(',')}`
    } else {
      const obj = node as Record<string, unknown>
      const op = String(obj.op)
      // Indexed refs: key on op + integer idx (stable across rewrites).
      if (op === 'regRef' || op === 'paramRef'
          || op === 'inputRef' || op === 'typeParamRef'
          || op === 'bindingRef') {
        key = `op:${op}|idx=${obj.idx as number}`
      } else if (op === 'nestedOut') {
        key = `op:${op}|inst=${obj.instance as number}|out=${obj.output as number}`
      } else {
        const parts: string[] = [`op:${op}`]
        const fieldNames = Object.keys(obj).filter(k => k !== 'op').sort()
        for (const k of fieldNames) parts.push(`${k}=${this.structuralKey(obj[k])}`)
        key = parts.join('|')
      }
    }
    let id = this.hashTable.get(key)
    if (id === undefined) {
      id = this.hashTable.size
      this.hashTable.set(key, id)
    }
    this.hashCache.set(node, id)
    return id
  }

  private structuralKey(v: unknown): string {
    if (v === null) return 'null'
    if (typeof v === 'number')  return `n:${v}`
    if (typeof v === 'boolean') return `b:${v}`
    if (typeof v === 'string')  return `s:${v}`
    if (typeof v === 'object')  return `i:${this.structuralId(v as object)}`
    return `u:${typeof v}`
  }

  // ── Compile a node to a CompileResult ──────────────────────
  compileNode(node: ResolvedExpr, expected?: ScalarType): CompileResult {
    const terminal = this.tryTerminal(node, expected)
    if (terminal !== null) return { isArray: false, op: terminal.op, scalarType: terminal.scalarType }

    const key = `${this.structuralId(node as object)}:${expected ?? ''}`
    const cached = this.memo.get(key)
    if (cached !== undefined) return cached

    const result = this.compileNodeUncached(node, expected)
    this.memo.set(key, result)
    return result
  }

  private compileNodeUncached(node: ResolvedExpr, expected?: ScalarType): CompileResult {
    if (Array.isArray(node)) return this.compilePack(node, expected)

    const obj = node as ResolvedExprOp

    // Array-typed regRef (filtered out by tryTerminal).
    if (obj.op === 'regRef') {
      const slot = obj.idx as number
      const arr = this.arrayRegMap.get(slot)
      if (arr) return { isArray: true, op: opArray(arr.slot), size: arr.size, scalarType: 'float' }
      throw new Error(`emit_resolved: regRef to non-array slot ${slot} reached compileNodeUncached unexpectedly`)
    }

    // Array-typed inputRef (filtered out by tryTerminal). The port is
    // bound to a session-level array slot recorded in
    // `slots.inputArraySlots`. We emit a `session_array_reg` operand —
    // the pre-remap kind that carries a session-absolute slot index.
    // `remapInstancePlan` converts this to `array_reg` (passthrough
    // slot value) so the post-remap FlatPlan / wire format only ever
    // contains `array_reg`. Same operand kind for both `index` reads
    // and `arraySet` writebacks (compileSetElement dispatches on the
    // operand kind to choose the right dst).
    if (obj.op === 'inputRef') {
      const info = this.slots.inputArraySlots?.get(obj.idx)
      if (info === undefined) {
        throw new Error(`emit_resolved: inputRef to non-array port idx=${obj.idx} reached compileNodeUncached unexpectedly`)
      }
      return {
        isArray: true,
        op: opSessionArray(arraySlotIdx(info.slot)),
        size: info.size,
        scalarType: 'float',
      }
    }

    // Array-typed NestedOut. Sub-instance has an array-typed output port
    // bound to a session-level array slot via `slots.nestedOutputArraySlots`.
    // The parent reads with `index(NestedOut(child, port), i)`; this returns
    // the operand for that read path. Same `session_array_reg` lifecycle
    // as the inputRef-array case above.
    if (obj.op === 'nestedOut') {
      const perInst = this.slots.nestedOutputArraySlots?.get(obj.instance)
      const info    = perInst?.get(obj.output)
      if (info !== undefined) {
        return {
          isArray: true,
          op: opSessionArray(arraySlotIdx(info.slot)),
          size: info.size,
          scalarType: 'float',
        }
      }
      // Fall through: this NestedOut isn't array-typed, so tryTerminal
      // should have produced a scalar slot read. Reaching here is a bug.
      throw new Error(`emit_resolved: nestedOut to non-array sub-instance output reached compileNodeUncached unexpectedly`)
    }

    const binTag = BINARY_TAG[obj.op]
    if (binTag) {
      const opNode = obj as Extract<ResolvedExprOp, { args: [ResolvedExpr, ResolvedExpr] }>
      return this.compileBinary(binTag, opNode.args, expected)
    }

    const uniTag = UNARY_TAG[obj.op]
    if (uniTag) {
      const opNode = obj as Extract<ResolvedExprOp, { args: [ResolvedExpr] }>
      return this.compileUnary(uniTag, opNode.args[0], expected)
    }

    if (obj.op === 'clamp')  return this.compileTernary('Clamp',  [obj.args[0], obj.args[1], obj.args[2]], expected)
    if (obj.op === 'select') return this.compileTernary('Select', [obj.args[0], obj.args[1], obj.args[2]], expected)
    if (obj.op === 'arraySet') return this.compileSetElement([obj.args[0], obj.args[1], obj.args[2]])

    if (obj.op === 'index') return this.compileIndex([obj.args[0], obj.args[1]])

    if (obj.op === 'zeros') {
      const c = this.compileNode(obj.count, 'int')
      const n = c.op.kind === 'const' && typeof c.op.val === 'number' ? c.op.val : 0
      const slot = this.allocArraySlot(n)
      this.emit(instrPack(slot, new Array(n).fill(opConst(0, 'float'))))
      return { isArray: true, op: opArray(slot), size: n, scalarType: 'float' }
    }

    switch (obj.op) {
      case 'fold': case 'scan': case 'generate': case 'iterate':
      case 'chain': case 'map2': case 'zipWith':
      case 'let':
      case 'tag': case 'match':
      case 'typeParamRef': case 'bindingRef':
        throw new Error(`emit_resolved: '${obj.op}' should have been lowered before emit`)
      // 'nestedOut' is handled in tryTerminal (fractal compile slot read);
      // it should never reach this exhaustiveness check.
    }

    const _exhaustive: never = obj as never
    void _exhaustive
    throw new Error(`emit_resolved: unhandled op (TypeScript exhaustiveness escape)`)
  }

  // ── Unbox a size-1 array to a scalar via Index[0]. ──
  private unboxArray(arr: ArrayResult): ScalarResult {
    const dst = this.allocReg()
    const rt = arr.scalarType
    this.regTypes.set(dst, rt)
    this.emit(instrIndex(dst, [arr.op, opConst(0, 'int')], rt))
    return { isArray: false, op: opTemp(dst, rt), scalarType: rt }
  }

  // ── Compile an inline JS array to a Pack instruction. ──
  private compilePack(elements: ResolvedExpr[], expected?: ScalarType): ArrayResult {
    const size = elements.length
    const slot = this.allocArraySlot(size)
    const args: NOperand[] = elements.map(e => {
      const r = this.compileNode(e, expected)
      return r.isArray ? opConst(0, 'float') : r.op
    })
    this.emit(instrPack(slot, args))
    return { isArray: true, op: opArray(slot), size, scalarType: 'float' }
  }

  // ── Compile a binary op. ──
  private compileBinary(tag: string, argNodes: [ResolvedExpr, ResolvedExpr], expected?: ScalarType): CompileResult {
    const propagated = expected === 'bool' ? undefined : expected
    const argExpected = BITWISE_TAGS.has(tag) ? 'int' as ScalarType
      : COMPARISON_TAGS.has(tag) ? undefined
      : propagated
    let l = this.compileNode(argNodes[0], argExpected)
    const secondExpected = COMPARISON_TAGS.has(tag)
      ? (l.isArray ? 'float' : (l.scalarType === 'bool' ? undefined : l.scalarType))
      : argExpected
    let r = this.compileNode(argNodes[1], secondExpected)
    if (l.isArray && l.size === 1) l = this.unboxArray(l)
    if (r.isArray && r.size === 1) r = this.unboxArray(r)

    const rt = inferResultType(tag, [l.scalarType, r.scalarType])

    if (!l.isArray && !r.isArray) {
      const dst = this.allocReg()
      this.regTypes.set(dst, rt)
      this.emit(instrScalar(tag, dst, [l.op, r.op], rt))
      return { isArray: false, op: opTemp(dst, rt), scalarType: rt }
    }

    const size = l.isArray ? l.size : (r as ArrayResult).size
    const slot = this.allocArraySlot(size)
    const strides = [l.isArray ? 1 : 0, r.isArray ? 1 : 0]
    this.emit(instrArray(tag, slot, [l.op, r.op], size, strides, rt))
    return { isArray: true, op: opArray(slot), size, scalarType: rt }
  }

  // ── Compile a unary op. ──
  private compileUnary(tag: string, argNode: ResolvedExpr, expected?: ScalarType): CompileResult {
    const argExpected = TRANSCENDENTAL_TAGS.has(tag) ? undefined
      : COMPARISON_TAGS.has(tag) ? undefined
      : tag === 'BitNot' ? 'int' as ScalarType
      : expected
    let a = this.compileNode(argNode, argExpected)
    if (a.isArray && a.size === 1) a = this.unboxArray(a)

    const rt = inferResultType(tag, [a.scalarType])

    if (!a.isArray) {
      const dst = this.allocReg()
      this.regTypes.set(dst, rt)
      this.emit(instrScalar(tag, dst, [a.op], rt))
      return { isArray: false, op: opTemp(dst, rt), scalarType: rt }
    }

    const slot = this.allocArraySlot(a.size)
    this.emit(instrArray(tag, slot, [a.op], a.size, [1], rt))
    return { isArray: true, op: opArray(slot), size: a.size, scalarType: rt }
  }

  // ── Compile a ternary op. ──
  private compileTernary(tag: string, argNodes: [ResolvedExpr, ResolvedExpr, ResolvedExpr], expected?: ScalarType): CompileResult {
    const condExpected: ScalarType | undefined = tag === 'Select' ? 'bool' : expected
    const armExpected = expected === 'bool' ? undefined : expected
    let a = this.compileNode(argNodes[0], condExpected)
    let b = this.compileNode(argNodes[1], armExpected)
    let c = this.compileNode(argNodes[2], armExpected)

    if (a.isArray && a.size === 1) a = this.unboxArray(a)
    if (b.isArray && b.size === 1) b = this.unboxArray(b)
    if (c.isArray && c.size === 1) c = this.unboxArray(c)

    const rt = inferResultType(tag, [a.scalarType, b.scalarType, c.scalarType])
    const anyArray = a.isArray || b.isArray || c.isArray
    if (!anyArray) {
      const dst = this.allocReg()
      this.regTypes.set(dst, rt)
      this.emit(instrScalar(tag, dst, [a.op, b.op, c.op], rt))
      return { isArray: false, op: opTemp(dst, rt), scalarType: rt }
    }

    const size = (a.isArray ? a.size : b.isArray ? b.size : (c as ArrayResult).size)
    const slot = this.allocArraySlot(size)
    const strides = [a.isArray ? 1 : 0, b.isArray ? 1 : 0, c.isArray ? 1 : 0]
    this.emit(instrArray(tag, slot, [a.op, b.op, c.op], size, strides, rt))
    return { isArray: true, op: opArray(slot), size, scalarType: rt }
  }

  // ── Index. ──
  private compileIndex(argNodes: [ResolvedExpr, ResolvedExpr]): ScalarResult {
    const arr = this.compileNode(argNodes[0])
    const idx = this.compileNode(argNodes[1], 'int')
    // The Index op requires its array operand to actually be an
    // array (an `array_reg` operand). A scalar here means we're
    // trying to index into something that won't behave like an
    // array at runtime — typically an array-typed input port
    // whose value can't be materialized as `array_reg` by the
    // per-instance path. Silently substituting a `const 0`
    // placeholder (the previous behavior) produced a kernel that
    // segfaulted the moment the JIT dereferenced
    // `arrays[args[0].slot]` with an uninitialized union field.
    // Fail loudly instead.
    if (!arr.isArray) {
      throw new Error(
        `emit_resolved: 'index' op has non-array operand. This usually means an ` +
        `array-typed input port (e.g. \`sequence: int[N]\`) is being indexed inside ` +
        `the program body — the per-instance compile path doesn't yet materialize ` +
        `array input operands.`,
      )
    }
    const dst = this.allocReg()
    const rt = arr.scalarType
    this.regTypes.set(dst, rt)
    const idxOp: NOperand = idx.isArray ? opConst(0, 'int') : idx.op
    this.emit(instrIndex(dst, [arr.op, idxOp], rt))
    return { isArray: false, op: opTemp(dst, rt), scalarType: rt }
  }

  // ── ArraySet. ──
  private compileSetElement(argNodes: [ResolvedExpr, ResolvedExpr, ResolvedExpr]): ArrayResult {
    const arr = this.compileNode(argNodes[0])
    const idx = this.compileNode(argNodes[1])
    const val = this.compileNode(argNodes[2])

    if (!arr.isArray) {
      const size = 1
      const slot = this.allocArraySlot(size)
      return { isArray: true, op: opArray(slot), size, scalarType: 'float' }
    }

    const arrOp: NOperand = arr.op
    const idxOp: NOperand = idx.isArray ? opConst(0, 'float') : idx.op
    const valOp: NOperand = val.isArray ? opConst(0, 'float') : val.op
    // The target array slot is the one the array operand already
    // points at. Two array-bearing operand kinds reach here:
    //   - `array_reg`         — kernel-local slot (gets shifted at remap)
    //   - `session_array_reg` — session-absolute slot (passthrough at remap)
    // Dispatch on kind so the dst carries the same namespace tag as
    // the source operand; remapInstancePlan handles the rest.
    switch (arr.op.kind) {
      case 'array_reg':
        this.emit(instrSetElement(arr.op.slot, [arrOp, idxOp, valOp]))
        break
      case 'session_array_reg':
        this.emit(instrSessionSetElement(arr.op.slot, [arrOp, idxOp, valOp]))
        break
      default:
        throw new Error(
          `emit_resolved: compileSetElement expected array_reg or session_array_reg operand, got ${arr.op.kind}`,
        )
    }
    return { isArray: true, op: arr.op, size: arr.size, scalarType: 'float' }
  }

  // ── Top-level emit driver ──
  emitProgram(
    outputExprs: ResolvedExpr[],
    registerExprs: (ResolvedExpr | null)[],
    outputPortScalarCounts: number[],
    nested: NestedContext,
  ): FlatProgram {
    const output_targets: TempIdx[] = []
    const register_targets: RegTarget[] = []

    // ── Fractal: emit per-child pre-input WriteSlots ──
    // For each child, evaluate each wired input expression in THIS
    // kernel's scope (where wire refs to our regs, params, and inputs
    // resolve naturally) and emit a WriteSlot into the child's pre-
    // allocated input slot. The child reads from the slot when its
    // body runs.
    //
    // Per-child segregation matters: the engine's emit_kernel_block
    // dispatches `per_child_pre_input[k] → children[k].body` in
    // sequence, then the parent's main body, then writebacks. By
    // placing each child's pre-input block immediately before its
    // dispatch (rather than hoisting them all into one parent-wide
    // pre-children block), wires for child[k] can read NestedOuts of
    // any sibling child[j] with j < k — child[j]'s body has already
    // run and written its output slot.
    //
    // Implementation note on CSE: temps live in the per-instance
    // namespace, which IS the same LLVM function across pre-input,
    // children, main, and writebacks. A subexpression computed in
    // per_child_pre_input[0] (and stored in some temp[X]) can be
    // CSE-reused by per_child_pre_input[1] without re-emitting the
    // compute — execution order guarantees temp[X] is live by then.
    // We slice each child's block off the running `this.instrs`
    // list, then truncate so the main body emission picks up at the
    // pre-child boundary.
    const per_child_pre_input: NInstr[][] = []
    const preChildBaseline = this.instrs.length
    {
      const scalarSlotMaps = this.slots.nestedInputSlots
      const arraySlotMaps  = this.slots.nestedInputArraySlots
      // nested.instances is body-decl order = prog.instances order, so
      // position k in this array IS the InstanceIdx for that child.
      // Empty list = legacy flat path, loop body never runs.
      for (let k = 0; k < nested.instances.length; k++) {
        const decl = nested.instances[k]
        const childStart = this.instrs.length
        const childInstanceIdx = instanceIdx(k)
        const childScalarMap = scalarSlotMaps?.get(childInstanceIdx)
        const childArrayMap  = arraySlotMaps?.get(childInstanceIdx)
        // Build a wired-by-port lookup so we can iterate ALL ports
        // (not just decl.inputs). Unwired ports need to receive
        // their declared port default — otherwise the slot retains
        // its allocation-time default of 0, which silently masks
        // the port's actual default (e.g., Bubble's attack_g: 0.05
        // becomes 0 if a parent program doesn't wire it explicitly,
        // killing env_smooth evolution).
        const wiredByPort = new Map<number, ResolvedExpr>()
        for (const inp of decl.inputs) wiredByPort.set(inp.port as number, inp.value)
        const ports = getInstanceType(nested.enclosing, decl).ports.inputs
        for (let i = 0; i < ports.length; i++) {
          const portDecl = ports[i]
          const wireExpr = wiredByPort.get(i)
            ?? portDecl.default
            ?? 0
          const portIdx = inputIdxOf(i)
          // Discriminate the port's allocation class. Scalar ports
          // land in `nestedInputSlots` (module-slot indices); array
          // ports land in `nestedInputArraySlots` (session-array slot
          // indices + sizes). A port appears in at most one map; if
          // it appears in neither, no parent-side wiring is needed.
          const scalarSlot = childScalarMap?.get(portIdx)
          const arrayInfo  = childArrayMap ?.get(portIdx)
          if (scalarSlot !== undefined) {
            const portT = inputDeclScalarType(portDecl)
            const r = this.compileNode(wireExpr, portT)
            // Scalar port — if the wire resolves to an array, project element 0.
            const valOp: NOperand = r.isArray
              ? (() => {
                  const dst = this.allocReg()
                  this.regTypes.set(dst, r.scalarType)
                  this.emit(instrIndex(dst, [r.op, opConst(0, 'int')], r.scalarType))
                  return opTemp(dst, r.scalarType)
                })()
              : r.op
            this.emit(instrWriteSlot(moduleSlotIdx(scalarSlot), valOp, portT))
            continue
          }
          if (arrayInfo !== undefined) {
            // Array port — compile the wire expression as an array,
            // then emit a single elementwise copy into the child's
            // session-absolute array slot. `instrSessionArray` carries
            // a `sessionArray` dst kind that `remapInstancePlan`
            // converts to `array` (passthrough slot value).
            //
            // The source operand may be `array_reg` (kernel-local —
            // e.g., a literal `[60,64,67,72]` packed in this kernel's
            // own array space) or `session_array_reg` (e.g., a NestedOut
            // to a sibling instance's array output). Either flows
            // through the engine's elementwise loop identically; remap
            // shifts kernel-local slot refs and passes session-absolute
            // ones through.
            //
            // Single Add-with-stride-0-on-rhs (size = arrayInfo.size,
            // strides = [1, 0]) is the same elementwise-copy idiom
            // used for array-reg writebacks in the register-update
            // pass below.
            const r = this.compileNode(wireExpr, 'float')
            if (!r.isArray) {
              throw new Error(
                `emit_resolved: array-typed child input port at idx=${i} of '${decl.name}' ` +
                `received scalar-shaped wire expression; expected array of size ${arrayInfo.size}`,
              )
            }
            if (r.size !== arrayInfo.size) {
              throw new Error(
                `emit_resolved: array-typed child input port at idx=${i} of '${decl.name}' ` +
                `has size ${arrayInfo.size}, wire expression evaluates to array of size ${r.size}`,
              )
            }
            this.emit(instrSessionArray(
              'Add',
              arraySlotIdx(arrayInfo.slot),
              [r.op, opConst(0, 'float')],
              arrayInfo.size,
              [1, 0],
              'float',
            ))
            continue
          }
          // Port has no allocated parent-side slot — nothing to emit.
        }
        const childEnd = this.instrs.length
        per_child_pre_input.push(this.instrs.slice(childStart, childEnd))
      }
    }
    // All pre-child blocks have been sliced out into per_child_pre_input;
    // truncate the running list so the main body emission below starts
    // back at the pre-child boundary. Temps allocated by the wire
    // evaluations stay reserved in the reg counter — they remain valid
    // operands in the unified per-instance namespace, just no longer
    // appear in `this.instrs` (their setter lives in the per-child
    // block, which the engine emits before the main body anyway).
    this.instrs = this.instrs.slice(0, preChildBaseline)

    // Output-targets contract: ONE entry per scalar slot of every
    // declared output port. Scalar/alias ports contribute 1
    // entry; array ports of total scalar count N contribute N entries
    // in row-major order matching `expandPortToSlots`'s naming.
    //
    // The number of targets per port is determined by the DECLARED port
    // shape (`outputPortScalarCounts`), not by the runtime shape of the
    // computed expression. Backward-compat case: when a scalar-declared
    // port receives an array-shaped expression, we project to element 0
    // (the historical behavior for under-specified types).
    if (outputPortScalarCounts.length !== outputExprs.length) {
      throw new Error(
        `emitProgram: outputPortScalarCounts.length (${outputPortScalarCounts.length}) ` +
        `must match outputExprs.length (${outputExprs.length})`,
      )
    }
    for (let portI = 0; portI < outputExprs.length; portI++) {
      const expr = outputExprs[portI]
      const declaredCount = outputPortScalarCounts[portI]
      const r = this.compileNode(expr, 'float')

      if (declaredCount === 1) {
        // Scalar port. If the expression is array-shaped, take [0]
        // (backward-compat); otherwise emit a single scalar copy.
        const dst = this.allocReg()
        if (r.isArray) {
          this.regTypes.set(dst, r.scalarType)
          this.emit(instrIndex(dst, [r.op, opConst(0, 'int')], r.scalarType))
        } else {
          this.regTypes.set(dst, r.scalarType)
          this.emit(instrScalar('Add', dst, [r.op, opConst(0, r.scalarType)], r.scalarType))
        }
        output_targets.push(dst)
      } else {
        // Array port. Expression must be array-shaped of matching size.
        if (!r.isArray) {
          throw new Error(
            `emitProgram: output port ${portI} declared as array of scalar count ` +
            `${declaredCount}, but expression is scalar`,
          )
        }
        if (r.size !== declaredCount) {
          throw new Error(
            `emitProgram: output port ${portI} declared as array of scalar count ` +
            `${declaredCount}, but expression has size ${r.size}`,
          )
        }
        for (let elemI = 0; elemI < declaredCount; elemI++) {
          const dst = this.allocReg()
          this.regTypes.set(dst, r.scalarType)
          this.emit(instrIndex(dst, [r.op, opConst(elemI, 'int')], r.scalarType))
          output_targets.push(dst)
        }
      }
    }

    // Two-pass register update — see history-rich block below for the
    // read-before-write isolation reasoning.
    //
    // Scalar regs: a temp target is recorded; the engine emits the
    // store after the sample body.
    //
    // Array regs: register_targets[i] = ArrayManagedTarget signals
    // "no scalar writeback needed". The pass 2 below emits an
    // explicit elementwise copy from the source slot to the
    // persistent array slot — that copy is what makes future
    // samples see the new value. The in-place case (arraySet on the
    // reg's own slot) skips the copy.
    //
    // The copy itself is `Add x, 0` with loop_count = size, stride
    // [1, 0]. The OrcJitEngine's elementwise-loop machinery handles
    // it; no engine changes needed.
    type ArrayCopy = { src: NOperand; dst: ArraySlotIdx; size: number }
    const arrayCopies: ArrayCopy[] = []
    for (let ri = 0; ri < registerExprs.length; ri++) {
      const expr = registerExprs[ri]
      if (expr === null) {
        register_targets.push(ArrayManagedTarget)
        continue
      }
      const regExpected = this.stateRegTypes[ri]
      const r = this.compileNode(expr, regExpected)
      if (r.isArray) {
        const arrInfo = this.arrayRegMap.get(ri)
        if (!arrInfo) {
          throw new Error(`emitProgram: array-result update on non-array reg ${ri}`)
        }
        if (r.op.kind !== 'array_reg') {
          throw new Error(`emitProgram: array result has non-array operand kind ${r.op.kind}`)
        }
        if (rawIdx(r.op.slot) !== rawIdx(arrInfo.slot)) {
          arrayCopies.push({ src: r.op, dst: arrInfo.slot, size: arrInfo.size })
        }
        register_targets.push(ArrayManagedTarget)
      } else {
        const dst = this.allocReg()
        this.regTypes.set(dst, r.scalarType)
        this.emit(instrScalar('Add', dst, [r.op, opConst(0, r.scalarType)], r.scalarType))
        register_targets.push(TempTarget(dst))
      }
    }
    for (const c of arrayCopies) {
      this.emit(instrArray('Add', c.dst, [c.src, opConst(0, 'float')], c.size, [1, 0], 'float'))
    }

    return {
      register_count:   this.nextReg,
      array_slot_count: this.nextArraySlot,
      array_slot_sizes: this.arraySizes,
      instructions:     this.instrs,
      per_child_pre_input,
      output_targets,
      register_targets,
    }
  }
}

function inputDeclScalarType(d: InputDecl): ScalarType {
  if (d.type === undefined) return 'float'
  if (d.type.kind === 'scalar') return d.type.scalar
  if (d.type.kind === 'alias')  return d.type.alias.base
  return 'float'
}

// ─────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────

export interface EmitResolvedInputs {
  outputExprs: ResolvedExpr[]
  /** One entry per output port: total scalar slot count derived from
   *  the port's declared shape (1 for scalar/alias; product of shape
   *  dims for arrays). Determines how many `output_targets` the emit
   *  produces per port. */
  outputPortScalarCounts: number[]
  registerExprs: (ResolvedExpr | null)[]
  stateInit: (number | boolean | number[])[]
  stateRegTypes: ScalarType[]
  inputPortTypes: ScalarType[]
  slots: EmitSlots
  /** Fractal context: sub-instance decls whose input wires need
   *  parent-side WriteSlot emission. Each entry's `inp.value` (wire
   *  expression) is compiled in the parent's scope; the result is
   *  written to the slot named in
   *  `slots.nestedInputSlots.get(decl).get(inp.port)`. The WriteSlot
   *  instructions land in `per_child_pre_input[k]`, parallel to
   *  `nestedInstances[k]`, so the engine can run each block
   *  immediately before recursing into its corresponding child. The
   *  list is empty when this kernel has no surviving sub-instances
   *  (flat path); the `enclosing` program is always present because
   *  it's the program whose kernel we're emitting — there's no
   *  "missing enclosing" case. Resolving each `instances[k].typeKey`
   *  reads `enclosing.programRegistry`. */
  nested: NestedContext
}

export interface NestedContext {
  instances: InstanceDecl[]
  enclosing: ResolvedProgram
}

export function emitResolvedProgram(input: EmitResolvedInputs): FlatProgram {
  const e = new Emitter(input.slots, input.stateInit, input.stateRegTypes, input.inputPortTypes)
  return e.emitProgram(input.outputExprs, input.registerExprs, input.outputPortScalarCounts, input.nested)
}
