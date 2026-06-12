/**
 * plan_types.ts — branded-IR types + pure serialization helpers for the
 * WASM backend.
 *
 * This is the web-survivor slice of the build-host emit_resolved module: the
 * `FlatProgram` / `NInstr` / `NOperand` / `DstSlot` shapes the WASM
 * emitter and the `tropical_plan_5` schema consume, plus the pure
 * wire-conversion helpers (`toWireInstr`, `dstSlotToWire`,
 * `dstSlotKindToWire`) and the namespace-asserting dst accessors
 * (`dstAsTemp`, `dstAsArray`, `dstAsModuleSlot`).
 *
 * Deliberately NOT relocated: the `Emitter` class and the
 * `emitResolvedProgram` entry point (which pull in the strata IR via
 * `./nodes.js` and `./decl_tables.js`). Those live on the build host;
 * the browser runtime only needs the post-strata plan shape, not the
 * emitter that produces it.
 *
 * ## Wire format
 *
 * `WireNInstr` and `WireNOperand` are plain JSON-shaped types with raw
 * `number` indices and a flat `dst: number`. `toWireInstr(i)` collapses
 * the discriminated dst back to a number for the JSON serialization
 * boundary; the engine parses that shape.
 */

import {
  type TempIdx, type StateRegIdx, type ArraySlotIdx, type ModuleSlotIdx,
  type InputPortIdx,
  rawIdx,
} from './slot_indices.js'

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
   *  kind. */
  | { kind: 'session_array_reg'; slot: ArraySlotIdx }
  | { kind: 'state_reg';         slot: StateRegIdx;    scalar_type: ScalarType }
  | { kind: 'param';             ptr:  string;         scalar_type: ScalarType }
  /** Read an input source — the dual of a sink, materialized at the runtime
   *  boundary. `index` keys into `FlatPlan.sources[]`; the engine switches
   *  on the source's `kind` (tick / rate / future ADC) to the appropriate
   *  kernel argument. v1: index 0 = tick, index 1 = rate. */
  | { kind: 'source';            index: number;        scalar_type: ScalarType }
  | { kind: 'slot';              index: ModuleSlotIdx; scalar_type: ScalarType }

/** Discriminated dst — the namespace of an instruction's writeback. */
export type DstSlot =
  | { kind: 'temp';         slot: TempIdx }
  | { kind: 'array';        slot: ArraySlotIdx }
  /** Pre-remap-only dst. Mirror of `session_array_reg`: writes to a
   *  session-absolute array slot. `remapInstancePlan`'s `shiftDst`
   *  converts to `kind: 'array'` (passthrough slot value). Wire format
   *  never sees this kind — `dstSlotToWire` throws if it does. */
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
   *  `per_child_pre_input[k]` holds the WriteSlot block for the k-th
   *  sub-instance — the parent runs that block in its own namespace
   *  immediately before invoking that child's kernel body. Empty array
   *  for leaf kernels and the legacy flat path. */
  per_child_pre_input: NInstr[][]
  /** Per-output-port temp index (local; the session compiler shifts). */
  output_targets:   TempIdx[]
  register_targets: RegTarget[]
}

// ─────────────────────────────────────────────────────────────
// RegTarget — the sum type FlatProgram.register_targets carries.
// Mirrors flat_plan.ts's RegTarget; defined here so plan_types.ts has
// no import cycle with flat_plan.ts. (flat_plan.ts re-declares the
// runnable-plan version with its own constructors.)
// ─────────────────────────────────────────────────────────────

export type RegTarget =
  | { kind: 'temp';         slot: TempIdx }
  | { kind: 'arrayManaged' }

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
  /** Legacy plan_4 sentinel — production sessions emit `source` instead.
   *  `parseOperand` upgrades these to `source` at parse time. */
  | { kind: 'rate';      scalar_type: ScalarType }
  | { kind: 'tick';      scalar_type: ScalarType }
  /** Read input source `sources[index]`. */
  | { kind: 'source';    index: number; scalar_type: ScalarType }
  | { kind: 'slot';      index: number; scalar_type: ScalarType }

export type WireDstKind = 'temp' | 'array' | 'moduleSlot'

export interface WireNInstr {
  tag: string
  /** The slot index in the writeback namespace selected by `dst_kind`. */
  dst: number
  /** Discriminator for `dst` — preserves the in-memory `DstSlot` union's
   *  kind tag through serialization. Reconstructed by the engine into
   *  a typed `DstKind` so dispatch in `emit_instr` is direct. */
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
