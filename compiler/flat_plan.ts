/**
 * flat_plan.ts — `tropical_plan_5` schema with discriminated/branded
 * internal representation.
 *
 * Two layers:
 *
 *   - **Internal**: `FlatPlan`, `PerInstancePlan`, `InstanceFunction`,
 *     `SchedulerFunction`. Uses branded `TempIdx` / `ArraySlotIdx` /
 *     etc. for every slot index, and a `RegTarget` sum type instead
 *     of `-1`-sentinelled `number`. The pipeline operates on this
 *     layer; the type system rejects cross-namespace arithmetic.
 *
 *   - **Wire**: `WireFlatPlan`. Plain numbers and arrays exactly as
 *     the C++ engine parses (`tropical_plan_5` JSON). Produced only
 *     at the JSON-stringify boundary via `toWirePlan(plan)`. The
 *     engine doesn't see the internal layer.
 *
 * Brands erase at runtime, so converting `FlatPlan → WireFlatPlan`
 * for branded primitives is a no-op cast. Only `register_targets`
 * (the `RegTarget` sum type) needs an actual structural transform.
 *
 * ## tropical_plan_5: per-instance kernel layout
 *
 * Each session instance compiles to its own `InstanceFunction` slice;
 * the `SchedulerFunction` orchestrates them per sample:
 *
 *     preamble (currently empty; reserved for future per-sample setup)
 *       for each instance: instance body + writebacks
 *     state_evolution (WriteSlot per extracted delay)
 *     postamble (DAC reads)
 *     output mix
 *
 * Hot-swap stays compatible: the entire `KernelState` swaps atomically;
 * state transfer matches register/array/slot names regardless of which
 * instance function they belong to.
 */

import type { NInstr, WireNInstr, WireNOperand, NOperand, ScalarType, DstSlot } from './ir/emit_resolved'
import {
  type TempIdx, type StateRegIdx, type ArraySlotIdx, type ModuleSlotIdx,
  type TempOffset, type StateRegOffset, type ArraySlotOffset,
  tempIdx, stateRegIdx, arraySlotIdx, moduleSlotIdx, inputPortIdx,
  tempOffset, stateRegOffset, arraySlotOffset,
  rawIdx, rawOffset,
} from './ir/slot_indices'
import { toWireInstr } from './ir/emit_resolved'

// ─── CompilationMode: how the engine should realize the plan ───────────────
// `fused` (default, legacy): one monolithic LLVM kernel function that
// inlines every instance body inside the outer sample loop.
// `microkernel`: N+1 LLVM functions in one module (preamble, per-instance
// kernels, state_evolution, postamble_mix); the C++ scheduler dispatches
// them via function pointers per sample. The field is part of the plan
// because the cache must be partitioned by mode — different return types
// from `compile_*` mean a fused-mode cache hit cannot satisfy a
// microkernel-mode query.

export type CompilationMode = 'fused' | 'microkernel'

/** Parse a wire-format mode string, defaulting to 'fused' for legacy
 *  plans that pre-date the field. Throws on unknown strings — we fail
 *  closed rather than silently picking a default the caller didn't
 *  intend. */
export function parseCompilationMode(s: string | undefined): CompilationMode {
  if (s === undefined || s === 'fused') return 'fused'
  if (s === 'microkernel') return 'microkernel'
  throw new Error(`flat_plan: unknown compilation_mode '${s}' (expected 'fused' | 'microkernel')`)
}

// ─── RegTarget: sum type replacing the `-1`-sentinel int field ──────────────
// Each state register either consumes a scalar temp (`temp`) or is
// managed entirely by inline array-write instructions earlier in the
// stream (`arrayManaged`). The legacy `register_targets: number[]`
// with `-1` for the latter is the literal anti-pattern that produced
// bug 1.

export type RegTarget =
  | { kind: 'temp';         slot: TempIdx }
  | { kind: 'arrayManaged' }

export const TempTarget = (slot: TempIdx): RegTarget => ({ kind: 'temp', slot })
export const ArrayManagedTarget: RegTarget = { kind: 'arrayManaged' }

/** Wire format: `-1` for arrayManaged, raw temp index otherwise. */
export const toWireRegTarget = (t: RegTarget): number =>
  t.kind === 'temp' ? rawIdx(t.slot) : -1

/** Parse the wire `number[]` back into the sum type. */
export const fromWireRegTarget = (n: number, mkTemp: (n: number) => TempIdx): RegTarget =>
  n < 0 ? ArrayManagedTarget : { kind: 'temp', slot: mkTemp(n) }

// ─── PerInstancePlan: output of compileResolved ─────────────────────────────
// Not a runnable plan; the smallest slice of work a single instance's
// kernel performs. The session compiler packs one per instance into
// the `instance_functions[]` of a `FlatPlan`.
//
// Indices here are LOCAL to the instance (no offset shift applied).
// The session compiler shifts them by the per-instance offsets when
// packing.

export interface PerInstancePlan {
  /** Total temp count this instance needs in its local space. */
  register_count:   number
  /** Local array slot count (shifted by `array_slot_offset` at pack time). */
  array_slot_count: number
  array_slot_sizes: number[]
  instructions:     NInstr[]
  /** Per-child WriteSlot instructions, parallel to the body's
   *  `InstanceDecl` order. `per_child_pre_input[k]` is the wire-
   *  computation + WriteSlot block that runs immediately BEFORE the
   *  k-th child's kernel body. Each block evaluates the child's input
   *  wires in THIS kernel's scope (where every ref resolves) and
   *  WriteSlots the result to the child's pre-allocated input slot.
   *  Interleaving with the child dispatch (vs running all blocks
   *  upfront) preserves sibling-to-sibling NestedOut dependencies:
   *  child[k]'s wire can read child[j].output for j < k because
   *  child[j] has already written its output slot by the time
   *  child[k]'s wires evaluate. Empty array for leaf kernels and
   *  for the legacy `inlineNested:true` path. */
  per_child_pre_input: NInstr[][]
  /** Per-output-port temp index (local). */
  output_targets:   TempIdx[]
  /** State register writeback: `register_targets[i]` is the local
   *  temp index whose value feeds state register `i`, or
   *  `ArrayManagedTarget` for array-typed regs that manage their own
   *  persistence via inline writes. */
  register_targets: RegTarget[]
  /** State register init values (length == register_targets count). */
  state_init:       (number | boolean)[]
  /** Name per state register slot — used for hot-swap state transfer. */
  register_names:   string[]
  /** Scalar type per state register slot. */
  register_types:   ScalarType[]
  /** Name per array slot. */
  array_slot_names: string[]
}

// ─── InstanceFunction: a per-instance slice inside a FlatPlan ───────────────
// Same shape as PerInstancePlan but indices are ABSOLUTE (post-shift)
// and the offsets that produced the shift are recorded.

export interface InstanceFunction {
  /** Mangled symbol name (informational). */
  name:              string
  /** Session instance name (e.g., `voice7`). May be a dotted path
   *  (`voice7.env`) for nested kernels in the fractal architecture. */
  instance_name:     string
  /** Instructions that run at the START of this kernel's block —
   *  before children dispatch, before main body. Holds the temp-
   *  compute instructions produced by translating session-wired
   *  input expressions (e.g., `trigger: pulseEvery(64)`) into
   *  temps that the child pre_input WriteSlots reference. Splitting
   *  this out of `instructions` matters: children get dispatched
   *  before `instructions` in the engine's emit_kernel_block, so any
   *  temp a child's pre_input references MUST exist before children
   *  start. Pre-Phase-3 (Bubble fix), this was concatenated into
   *  `instructions` and Bubble's per-child trigger writes wrote
   *  zeros. */
  preamble_instructions: NInstr[]
  /** Instructions with operands already shifted into unified space. */
  instructions:      NInstr[]
  /** Slot-based parent→child input wiring (the fractal path).
   *  Instructions the PARENT runs in its own namespace just before
   *  invoking this kernel:
   *  evaluate each wire expression, then `WriteSlot` the value to
   *  this kernel's pre-allocated input slot. The child reads from
   *  the slot at the start of its body. Per-child placement (vs
   *  hoisted to a single pre-children block on the parent) preserves
   *  sibling-to-sibling NestedOut dependencies — child[k]'s wire can
   *  read child[j].output for j < k because child[j]'s body has
   *  already run by the time child[k]'s pre-input wires evaluate.
   *  Empty for top-level kernels and for `inlineNested:true` plans. */
  pre_input_instructions: NInstr[]
  /** Cumulative offset into the unified temp space. */
  register_offset:   TempOffset
  /** Cumulative offset into the unified state-register space. */
  state_reg_offset:  StateRegOffset
  /** Cumulative offset into the unified array-slot space. */
  array_slot_offset: ArraySlotOffset
  /** Temps this instance consumes (local count). */
  register_count:    number
  /** Writebacks for this instance — entries are either an absolute
   *  unified temp index or `arrayManaged`. */
  register_targets:  RegTarget[]
  /** Nested kernels (the fractal architecture: one kernel per
   *  InstanceDecl at every nesting depth). Sub-InstanceDecls within
   *  this program's body become child kernels, emitted inside this
   *  kernel's body. Empty for leaf kernels. */
  children:          InstanceFunction[]
}

// ─── SchedulerFunction: the top-level per-sample driver ─────────────────────

export interface SchedulerFunction {
  /** Per-sample setup. Currently empty; reserved for future per-sample
   *  scheduler work. Runs before any instance bodies. */
  preamble:        NInstr[]
  /** Per-sample state evolution. Runs after instance dispatches and
   *  before the observation postamble. Holds `WriteSlot` instructions
   *  that update session-level delay slots from the current sample's
   *  source instance outputs — the MCP wire auto-delay convention is
   *  realized here. Reads in this phase see the current sample's
   *  instance writebacks; writes become visible to the NEXT sample's
   *  preamble and instance kernels (which read these slots at the
   *  start of their bodies and therefore see the previous postamble-
   *  era write = one sample of latency per wire). */
  state_evolution: NInstr[]
  /** Per-sample observation / teardown. Holds DAC mix-bus reads from
   *  each graphOutput slot — observes the current sample's WriteSlot
   *  values. */
  postamble:       NInstr[]
  /** Temps (in the unified register space) summed into the audio output. */
  output_targets:  TempIdx[]
  /** Indices into `output_targets` (identity mapping today). */
  outputs:         number[]
}

// ─── FlatPlan: the runnable plan ────────────────────────────────────────────

export interface FlatPlan {
  schema: 'tropical_plan_5'
  config: { sampleRate: number }

  /** Engine realization strategy. See `CompilationMode`. */
  compilation_mode: CompilationMode

  // ── Unified state (across all instance functions + scheduler) ────────
  state_init:       (number | boolean)[]
  register_names:   string[]
  register_types:   ScalarType[]
  array_slot_names: string[]
  register_count:   number
  array_slot_count: number
  array_slot_sizes: number[]

  // ── Multi-function layout ─────────────────────────────────────────────
  instance_functions: InstanceFunction[]
  scheduler_function: SchedulerFunction

  // ── Inter-module slot array ───────────────────────────────────────────
  slot_count:    number
  slot_names:    string[]
  slot_defaults: number[]
}

// ─── Wire format ────────────────────────────────────────────────────────────
// Plain JSON-friendly types matching what `engine/runtime/NumericProgramParser.hpp`
// parses. Branded numbers serialize as plain numbers automatically;
// `RegTarget` arrays get explicit conversion.

export interface WireInstanceFunction {
  name:              string
  instance_name:     string
  instructions:      WireNInstr[]
  /** Preamble — runs at the start of this kernel's block, before
   *  children dispatch. Used for translating session-wired input
   *  expressions into temps that child pre_input WriteSlots
   *  reference. Optional in the wire format (legacy plans had this
   *  bundled inside `instructions`; parsed as [] when missing). */
  preamble_instructions?: WireNInstr[]
  /** Slot-based parent→child input wiring (per-child). Parent's
   *  WriteSlot instructions that run just before this kernel's body.
   *  May be missing in legacy JSON (parsed as []). */
  pre_input_instructions?: WireNInstr[]
  register_offset:   number
  state_reg_offset:  number
  array_slot_offset: number
  register_count:    number
  register_targets:  number[]
  /** Nested kernels. May be missing in legacy JSON (parsed as []). */
  children?:         WireInstanceFunction[]
}

export interface WireSchedulerFunction {
  preamble:        WireNInstr[]
  /** Optional in the wire format for backward compatibility with
   *  hand-crafted JSON fixtures and previously-captured precompiled
   *  plans that predate the dedicated state-evolution phase. Parsed
   *  as `[]` when missing. */
  state_evolution?: WireNInstr[]
  postamble:       WireNInstr[]
  output_targets:  number[]
  outputs:         number[]
}

export interface WireFlatPlan {
  schema:           'tropical_plan_5'
  config:           { sampleRate: number }
  /** Optional on the wire: omitted means 'fused' (legacy default).
   *  When present, must be 'fused' | 'microkernel'. */
  compilation_mode?: CompilationMode
  state_init:       (number | boolean)[]
  register_names:   string[]
  register_types:   ScalarType[]
  array_slot_names: string[]
  register_count:   number
  array_slot_count: number
  array_slot_sizes: number[]
  instance_functions: WireInstanceFunction[]
  scheduler_function: WireSchedulerFunction
  slot_count:    number
  slot_names:    string[]
  slot_defaults: number[]
}

// ─── Parse wire format back into branded internal types ────────────────────
//
// Used by callers that read precompiled plans from disk and want to
// run them through the rich-typed pipeline (e.g. emit_wasm). The
// only namespace-decision rule in the entire pipeline lives here —
// once a plan has been parsed, the rich `DstSlot` carries the
// namespace forward and downstream consumers pattern-match on it
// instead of reapplying the rule.

const isArrayDst = (i: WireNInstr): boolean =>
  i.loop_count > 1 || i.tag === 'Pack' || i.tag === 'SetElement'

const isModuleSlotDst = (i: WireNInstr): boolean =>
  i.tag === 'WriteSlot'

const parseDstSlot = (i: WireNInstr): DstSlot => {
  if (isModuleSlotDst(i)) return { kind: 'moduleSlot', index: moduleSlotIdx(i.dst) }
  if (isArrayDst(i))      return { kind: 'array',      slot:  arraySlotIdx(i.dst) }
  return                         { kind: 'temp',       slot:  tempIdx(i.dst) }
}

const parseOperand = (o: WireNOperand): NOperand => {
  switch (o.kind) {
    case 'const':     return o
    case 'rate':      return o
    case 'tick':      return o
    case 'param':     return o
    case 'input':     return { kind: 'input',     slot:  inputPortIdx(o.slot),   scalar_type: o.scalar_type }
    case 'reg':       return { kind: 'reg',       slot:  tempIdx(o.slot),        scalar_type: o.scalar_type }
    case 'array_reg': return { kind: 'array_reg', slot:  arraySlotIdx(o.slot) }
    case 'state_reg': return { kind: 'state_reg', slot:  stateRegIdx(o.slot),    scalar_type: o.scalar_type }
    case 'slot':      return { kind: 'slot',      index: moduleSlotIdx(o.index), scalar_type: o.scalar_type }
  }
}

const parseInstr = (i: WireNInstr): NInstr => ({
  tag:         i.tag,
  dst:         parseDstSlot(i),
  args:        i.args.map(parseOperand),
  loop_count:  i.loop_count,
  strides:     i.strides,
  result_type: i.result_type,
})

const parseRegTargetFromWire = (n: number): RegTarget =>
  n < 0 ? ArrayManagedTarget : TempTarget(tempIdx(n))

/** Recursive parse of an InstanceFunction wire structure. */
const parseInstanceFn = (inst: WireInstanceFunction): InstanceFunction => ({
  name:              inst.name,
  instance_name:     inst.instance_name,
  instructions:      inst.instructions.map(parseInstr),
  preamble_instructions:  (inst.preamble_instructions  ?? []).map(parseInstr),
  pre_input_instructions: (inst.pre_input_instructions ?? []).map(parseInstr),
  register_offset:   tempOffset(inst.register_offset),
  state_reg_offset:  stateRegOffset(inst.state_reg_offset),
  array_slot_offset: arraySlotOffset(inst.array_slot_offset),
  register_count:    inst.register_count,
  register_targets:  inst.register_targets.map(parseRegTargetFromWire),
  children:          (inst.children ?? []).map(parseInstanceFn),
})

/** Recursive serialize of an InstanceFunction to wire structure. */
const toWireInstanceFn = (inst: InstanceFunction): WireInstanceFunction => ({
  name:              inst.name,
  instance_name:     inst.instance_name,
  instructions:      inst.instructions.map(toWireInstr),
  register_offset:   rawOffset(inst.register_offset),
  state_reg_offset:  rawOffset(inst.state_reg_offset),
  array_slot_offset: rawOffset(inst.array_slot_offset),
  register_count:    inst.register_count,
  register_targets:  inst.register_targets.map(toWireRegTarget),
  // Omit `preamble_instructions`, `pre_input_instructions`, and
  // `children` from the wire when empty so existing JSON consumers
  // (golden fixtures, hand-crafted plan_5 tests) see exactly the bytes
  // they expect today. Populated entries are emitted normally.
  ...(inst.preamble_instructions.length > 0
    ? { preamble_instructions: inst.preamble_instructions.map(toWireInstr) }
    : {}),
  ...(inst.pre_input_instructions.length > 0
    ? { pre_input_instructions: inst.pre_input_instructions.map(toWireInstr) }
    : {}),
  ...(inst.children.length > 0 ? { children: inst.children.map(toWireInstanceFn) } : {}),
})

/** Parse a wire-format plan (as read from disk JSON or constructed
 *  by hand in legacy tests) into the branded internal `FlatPlan`.
 *  Inverse of `toWirePlan`. */
export function parseWirePlan(wire: WireFlatPlan): FlatPlan {
  return {
    schema:           wire.schema,
    config:           wire.config,
    compilation_mode: parseCompilationMode(wire.compilation_mode),
    state_init:       wire.state_init,
    register_names:   wire.register_names,
    register_types:   wire.register_types,
    array_slot_names: wire.array_slot_names,
    register_count:   wire.register_count,
    array_slot_count: wire.array_slot_count,
    array_slot_sizes: wire.array_slot_sizes,
    slot_count:       wire.slot_count,
    slot_names:       wire.slot_names,
    slot_defaults:    wire.slot_defaults,
    instance_functions: wire.instance_functions.map(parseInstanceFn),
    scheduler_function: {
      preamble:        wire.scheduler_function.preamble.map(parseInstr),
      // Omitted from legacy wire-format plans → empty list (no
      // state-evolution work to perform; equivalent to the pre-Phase-3
      // pipeline).
      state_evolution: (wire.scheduler_function.state_evolution ?? []).map(parseInstr),
      postamble:       wire.scheduler_function.postamble.map(parseInstr),
      output_targets:  wire.scheduler_function.output_targets.map(tempIdx),
      outputs:         wire.scheduler_function.outputs,
    },
  }
}

/** Convert a typed FlatPlan to its wire-format equivalent. Brands
 *  erase at runtime so most fields pass through unchanged; only
 *  `RegTarget[]` and the discriminated `NInstr` need structural
 *  conversion. */
export function toWirePlan(plan: FlatPlan): WireFlatPlan {
  return {
    schema:           plan.schema,
    config:           plan.config,
    // Omit 'fused' from the wire format so golden JSON fixtures
    // (which predate this field) don't gain a spurious key. Only
    // 'microkernel' emits explicitly.
    ...(plan.compilation_mode === 'microkernel'
      ? { compilation_mode: 'microkernel' as const }
      : {}),
    state_init:       plan.state_init,
    register_names:   plan.register_names,
    register_types:   plan.register_types,
    array_slot_names: plan.array_slot_names,
    register_count:   plan.register_count,
    array_slot_count: plan.array_slot_count,
    array_slot_sizes: plan.array_slot_sizes,
    slot_count:       plan.slot_count,
    slot_names:       plan.slot_names,
    slot_defaults:    plan.slot_defaults,
    instance_functions: plan.instance_functions.map(toWireInstanceFn),
    scheduler_function: {
      preamble:       plan.scheduler_function.preamble.map(toWireInstr),
      // Omit empty state_evolution from the wire format so existing
      // golden JSON fixtures (which predate this field) don't gain
      // a spurious empty array. Populated state_evolution arrays
      // emit normally.
      ...(plan.scheduler_function.state_evolution.length > 0
        ? { state_evolution: plan.scheduler_function.state_evolution.map(toWireInstr) }
        : {}),
      postamble:      plan.scheduler_function.postamble.map(toWireInstr),
      output_targets: plan.scheduler_function.output_targets.map(rawIdx),
      outputs:        plan.scheduler_function.outputs,
    },
  }
}
