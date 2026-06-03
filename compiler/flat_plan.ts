/**
 * flat_plan.ts — `tropical_plan_5` schema with discriminated/branded
 * internal representation.
 *
 * Two layers:
 *
 *   - **Internal**: `FlatPlan`, `PerInstancePlan`, `InstanceFunction`,
 *     `SinkSpec`. Uses branded `TempIdx` / `ArraySlotIdx` /
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
 * ## tropical_plan_5: kernel layout
 *
 * The session lowers to a single root `InstanceFunction` whose `children`
 * are the session instances. Per sample the engine runs:
 *
 *     for each instance (topo order, nested): body + writebacks
 *       (session-level per-wire delays are root RegDecl writebacks here)
 *     for each sink: output[target] = gain · Σ slots[sink.inputs]
 *
 * There is no scheduler tier and no special output tail beyond the sink
 * op. (The legacy `output_targets`/`outputs` temp-mix is a top-level
 * carrier used only by sink-less plan_4 / single-kernel fixtures.)
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
// `microkernel`: N+3 LLVM functions in one module (preamble, per-instance
// kernels, state_evolution, postamble_mix); the C++ scheduler dispatches
// them via function pointers per sample. Top-level session instances
// each get their own function; nested children are inlined into their
// parent's function.
// `microkernel-deep`: like `microkernel`, but children are also emitted
// as their own LLVM functions instead of being inlined — one function
// per `InstanceFunction` at every nesting depth. Requires the plan to
// carry non-empty `children` arrays (i.e., the session was compiled
// with `inlineNested: false`).
// The field is part of the plan because the cache must be partitioned
// by mode — different return types from `compile_*` mean a fused-mode
// cache hit cannot satisfy a microkernel-mode query.

export type CompilationMode = 'fused' | 'microkernel' | 'microkernel-deep'

/** Parse a wire-format mode string, defaulting to 'fused' for legacy
 *  plans that pre-date the field. Throws on unknown strings — we fail
 *  closed rather than silently picking a default the caller didn't
 *  intend. */
export function parseCompilationMode(s: string | undefined): CompilationMode {
  if (s === undefined || s === 'fused') return 'fused'
  if (s === 'microkernel')      return 'microkernel'
  if (s === 'microkernel-deep') return 'microkernel-deep'
  throw new Error(`flat_plan: unknown compilation_mode '${s}' (expected 'fused' | 'microkernel' | 'microkernel-deep')`)
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

// ─── SinkSpec: a device-bound output sink (the DAC, neutrally named) ─────────
//
// The one effectful node — the exit from the pure signal graph to a device.
// A sink reads a set of output module slots, sums them, scales by its own
// `gain`, and writes the result to output channel/device `target`. Sinks are
// a FAMILY, not a singleton: multiple output devices/channels are normal
// (and dually, sources — Tick/Rate — are a family on the input side). v1
// emits exactly one sink (target 0 → the single audio output buffer); the
// representation is N-ready, the engine currently realizes target 0.
//
// Lives at the plan boundary, never inside the pure ResolvedProgram passes —
// `emitSinks` materializes it at the `compileSession` boundary from
// `session.graphOutputs`. This is the categorically-correct home for the
// effect: a morphism in the emit IR, absent from the trace-free PROP.

export interface SinkSpec {
  /** Output module-slot indices summed into this sink (read directly via
   *  the engine's slot load — no intermediate temps / DAC-stitch postamble). */
  inputs: ModuleSlotIdx[]
  /** The sink's own output scale (was the hardcoded engine `÷20`; now data).
   *  Default 1/20 preserves v1 audio exactly. */
  gain:   number
  /** Output device/channel index. 0 = the default audio output buffer. */
  target: number
}

export interface WireSinkSpec {
  inputs: number[]
  gain:   number
  target: number
}

// ─── SourceSpec: a runtime-bound input source (Tick/Rate, neutrally named) ───
//
// The dual of `SinkSpec` — the *entry* from the runtime to the pure signal
// graph. Where sinks WRITE device outputs at the end of each sample, sources
// PROVIDE values that the kernel reads (the sample index, the sample rate,
// future external inputs like ADC or MIDI clock). Sources are a FAMILY: the
// schema is N-source ready; v1 always emits the canonical pair in fixed order:
//
//     sources[0] = { kind: 'tick' }   // current sample index  (integer)
//     sources[1] = { kind: 'rate' }   // current sample rate   (float)
//
// IR refs (`SampleIndex`, `SampleRate`) lower to `{kind:'source', index}`
// operands at the plan boundary; the engine resolves each by switching on
// `program.sources[index].kind` to the appropriate kernel argument. The plan
// thus DECLARES what it consumes — symmetric with how it declares its outputs
// in `sinks[]` — while the engine keeps the existing efficient kernel args.

export type SourceKind = 'tick' | 'rate'

export interface SourceSpec {
  kind: SourceKind
}

export interface WireSourceSpec {
  kind: SourceKind
}

/** Canonical source ordering. Sessions always emit the pair in this order
 *  so emit-time and engine-time index agreement is mechanical. */
export const SOURCE_TICK_INDEX = 0
export const SOURCE_RATE_INDEX = 1
export const DEFAULT_SOURCES: SourceSpec[] = [
  { kind: 'tick' },
  { kind: 'rate' },
]

/** Predicate used by `toWirePlan` to elide the `sources` field when its
 *  value matches the canonical pair — keeps pre-sources wire goldens
 *  byte-stable when nothing real has changed. */
export const isDefaultSources = (s: readonly SourceSpec[]): boolean =>
  s.length === DEFAULT_SOURCES.length
  && s.every((sp, i) => sp.kind === DEFAULT_SOURCES[i]!.kind)

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
  /** Device-bound output sinks (the plan_5 output path). */
  sinks: SinkSpec[]
  /** Runtime-bound input sources. Dual of `sinks`. Canonical order:
   *  `[{kind:'tick'}, {kind:'rate'}]`; `{op:'source', index:i}` operands
   *  resolve via `sources[i].kind`. v1 always emits both. */
  sources: SourceSpec[]
  /** Legacy temp-mix carrier — top-level output temps summed ÷20 by the
   *  engine when `sinks` is empty (the plan_4 / single-kernel path).
   *  Sessions never set this; they use `sinks`. */
  output_targets?: TempIdx[]
  outputs?:        number[]

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
  /** Device-bound output sinks (the plan_5 output path). */
  sinks?:        WireSinkSpec[]
  /** Runtime-bound input sources. Optional in the wire for backcompat
   *  with plan_4 / pre-sources fixtures; parsed as `DEFAULT_SOURCES`
   *  (the canonical [tick, rate] pair) when missing. */
  sources?:      WireSourceSpec[]
  /** Legacy temp-mix carrier (top-level, plan_4-style) — engine sums
   *  these output temps ÷20 when `sinks` is absent. */
  output_targets?: number[]
  outputs?:        number[]
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

/** Parse the wire-format `dst_kind` tag into the in-memory `DstSlot`
 *  union. Falls back to the legacy proxy (`tag + loop_count`) for
 *  fixtures that pre-date the explicit `dst_kind` field — `Pack`/
 *  `SetElement` → array, `WriteSlot` → moduleSlot, `loop_count > 1`
 *  → array, otherwise temp. New plans always carry `dst_kind`;
 *  legacy fixtures (some test cases, older patches) hit the
 *  fallback. */
const parseDstSlot = (i: WireNInstr): DstSlot => {
  const kind = i.dst_kind ?? deriveLegacyDstKind(i)
  switch (kind) {
    case 'moduleSlot': return { kind: 'moduleSlot', index: moduleSlotIdx(i.dst) }
    case 'array':      return { kind: 'array',      slot:  arraySlotIdx(i.dst) }
    case 'temp':       return { kind: 'temp',       slot:  tempIdx(i.dst) }
  }
}

const deriveLegacyDstKind = (i: WireNInstr): 'temp' | 'array' | 'moduleSlot' => {
  if (i.tag === 'WriteSlot') return 'moduleSlot'
  if (i.tag === 'Pack' || i.tag === 'SetElement') return 'array'
  if (i.loop_count > 1) return 'array'
  return 'temp'
}

const parseOperand = (o: WireNOperand): NOperand => {
  switch (o.kind) {
    case 'const':     return o
    case 'param':     return o
    // Legacy plan_4 / pre-sources fixtures emit `rate`/`tick` directly;
    // upgrade them to indexed source operands so the internal type
    // is fully migrated. Engine-side parser does the dual upgrade.
    case 'rate':      return { kind: 'source', index: SOURCE_RATE_INDEX, scalar_type: o.scalar_type }
    case 'tick':      return { kind: 'source', index: SOURCE_TICK_INDEX, scalar_type: o.scalar_type }
    case 'source':    return { kind: 'source', index: o.index,            scalar_type: o.scalar_type }
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
    // Pre-sources plans get the canonical [tick, rate] pair so any `source`
    // operands that may arrive (if a future plan layer emits them) still
    // resolve. Plans that explicitly carry `sources` round-trip exactly.
    sources: wire.sources !== undefined
      ? wire.sources.map(s => ({ kind: s.kind }))
      : [...DEFAULT_SOURCES],
    sinks: (wire.sinks ?? []).map(s => ({
      inputs: s.inputs.map(moduleSlotIdx),
      gain:   s.gain,
      target: s.target,
    })),
    ...(wire.output_targets !== undefined
      ? { output_targets: wire.output_targets.map(tempIdx), outputs: wire.outputs ?? [] }
      : {}),
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
    // (which predate this field) don't gain a spurious key. Other
    // modes emit explicitly.
    ...(plan.compilation_mode === 'fused'
      ? {}
      : { compilation_mode: plan.compilation_mode }),
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
    // Omit empty sinks from the wire so legacy golden fixtures (which
    // predate the field) don't gain a spurious key. Populated sinks emit
    // normally.
    ...(plan.sinks.length > 0
      ? { sinks: plan.sinks.map(s => ({ inputs: s.inputs.map(rawIdx), gain: s.gain, target: s.target })) }
      : {}),
    // Sources: always emit on the production path (sessions construct
    // [tick, rate]). Omit when equal to the canonical default so existing
    // golden JSON byte goldens (pre-sources) round-trip cleanly.
    ...(isDefaultSources(plan.sources)
      ? {}
      : { sources: plan.sources.map(s => ({ kind: s.kind })) }),
    // Legacy temp-mix carrier (plan_4 / sink-less plans only).
    ...(plan.output_targets !== undefined
      ? { output_targets: plan.output_targets.map(rawIdx), outputs: plan.outputs ?? [] }
      : {}),
  }
}
