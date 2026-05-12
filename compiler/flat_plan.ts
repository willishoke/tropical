/**
 * flat_plan.ts — `tropical_plan_5` JSON schema.
 *
 * The FlatPlan type is the contract between the TS compiler layer and
 * the C++ engine. Every emit boundary (`compile_resolved.ts`,
 * `compile_session.ts`, the WASM emitter) produces this shape; the
 * runtime's `loadPlan` consumes it.
 *
 * ## tropical_plan_5: per-instance kernel layout
 *
 * The plan separates the per-sample compute into:
 *
 *   - **`instance_functions[]`**: one entry per session instance. Each
 *     entry carries that instance's own instruction stream, plus its
 *     absolute offsets into the unified register / state-reg /
 *     array-slot space (all instances share those flat arrays so the
 *     kernel signature stays fixed).
 *   - **`scheduler_function`**: the top-level driver. Its `preamble`
 *     evaluates the per-instance alive expressions, writing each
 *     resolved `bool` value into the instance's `__alive__` slot
 *     **before** any instance kernel runs. The C++ side wraps the
 *     instance calls in `if (slots[alive_slot_index] > 0.5) call ...`,
 *     so sleep-eligible instances skip their internal compute. For
 *     always-on instances (alive defaults to literal `1.0` per slot),
 *     the conditional folds away post-inlining and the optimized IR
 *     matches a unified kernel byte-for-byte.
 *
 * The top-level fields — `register_count`, `array_slot_*`, etc. — are
 * the **unified** sizes summed across all instances plus the
 * scheduler preamble. The per-instance offsets carve that flat space
 * into per-instance slices, so each instance function can read/write
 * registers using absolute indices baked into its instruction stream.
 *
 * Hot-swap remains compatible: the entire `KernelState` swap is atomic,
 * and state transfer matches register/array/slot names regardless of
 * which instance function they belong to.
 */

import type { NInstr, ScalarType } from './ir/emit_resolved'

/** Per-program compile result. `compileResolved` produces this for a
 *  single `ResolvedProgram`; the session-level compiler bundles one
 *  per instance into `instance_functions[]` of a `FlatPlan`. Not a
 *  runnable plan on its own (no schema, no slot map, no scheduler) —
 *  it's the smallest slice of work a per-instance kernel performs. */
export interface PerInstancePlan {
  /** Total temp count the instance kernel needs in its local space.
   *  The session compiler shifts these into the unified register
   *  space when packing into a `FlatPlan`. */
  register_count:   number
  /** Array slot count (also local; shifted by `array_slot_offset` at pack time). */
  array_slot_count: number
  array_slot_sizes: number[]
  instructions:     NInstr[]
  /** Per-output-port temp index (local). */
  output_targets:   number[]
  /** State register writeback: `register_targets[i]` is the local
   *  temp index whose value feeds state register `i`. */
  register_targets: number[]
  /** State register init values (length == register_targets count). */
  state_init:       (number | boolean)[]
  /** Name per state register slot — used for hot-swap state transfer. */
  register_names:   string[]
  /** Scalar type per state register slot. */
  register_types:   ScalarType[]
  /** Name per array slot. */
  array_slot_names: string[]
}

/** A single instance's compiled kernel slice. */
export interface InstanceFunction {
  /** Mangled symbol name in the JIT module (informational). */
  name:              string
  /** Session instance name (e.g., `voice7`). */
  instance_name:     string
  /** Instructions belonging to this instance. Operands referencing
   *  unified register / state-reg / array-slot spaces are already
   *  shifted by the corresponding offset; the kernel does not re-add. */
  instructions:      NInstr[]
  /** Cumulative offset into the unified temp/register space at which
   *  this instance's `register_count` block starts. The compiler shifts
   *  every `dst` and `reg`-operand `slot` by this much before emission. */
  register_offset:   number
  /** Cumulative offset into the unified state-register space. Shifts
   *  every `state_reg`-operand `slot` and `register_targets[i]`. */
  state_reg_offset:  number
  /** Cumulative offset into the unified array-slot space. Shifts every
   *  `array_reg`-operand `slot`. */
  array_slot_offset: number
  /** Number of temps this instance consumes (its `register_count` from
   *  the per-instance compileResolved). Used by the engine for sanity
   *  checks; the unified top-level `register_count` already covers it. */
  register_count:    number
  /** Delay writebacks: `register_targets[i]` is the temp index (in the
   *  unified register space) whose value feeds state register `i +
   *  state_reg_offset`. The engine emits these stores after the body. */
  register_targets:  number[]
  /** Slot index of this instance's `__alive__` bool slot. The scheduler
   *  wraps the call in `if (slots[alive_slot_index] > 0.5) call ...`.
   *  Every instance has one — defaults to literal 1.0 so the conditional
   *  folds at JIT-time for always-on instances. */
  alive_slot_index:  number
}

/** Scheduler top-level driver. The C++ runtime sequences each sample as:
 *
 *     preamble ; for each instance: (alive_i > 0.5) ? call instance_i ; postamble
 *
 *  Alive expressions land in `preamble` so the scheduler can read
 *  every instance's alive slot before dispatching. DAC mix reads land
 *  in `postamble` so they observe the current sample's WriteSlot
 *  values (alive instances have written; asleep instances retain). */
export interface SchedulerFunction {
  /** Instructions evaluated once per sample BEFORE any instance fires.
   *  Holds the alive-expression evaluation + WriteSlot to each
   *  instance's `__alive__` slot. Reads of other instances' output
   *  slots see the **previous sample's** values, by virtue of running
   *  before any WriteSlot fires this sample — alive expressions are
   *  thus implicitly one-sample-delayed in their inter-instance
   *  references. */
  preamble:        NInstr[]
  /** Instructions evaluated once per sample AFTER every instance has
   *  run. Holds the DAC mix-bus reads (each `graphOutput` slot into a
   *  fresh temp; the mix sums those temps into the audio buffer). */
  postamble:       NInstr[]
  /** Temps (in the unified register space) whose values are summed
   *  into the audio output buffer. Indexed by the entries in
   *  `outputs`. */
  output_targets:  number[]
  /** Indices into `output_targets` (identity mapping today; reserved
   *  for future per-channel mix routing). */
  outputs:         number[]
}

export interface FlatPlan {
  schema: 'tropical_plan_5'
  config: { sampleRate: number }

  // ── Unified state (across all instance functions + scheduler) ────────
  /** Initial value per state register slot. Length === register_targets
   *  total across all instance_functions plus any scheduler-owned regs
   *  (none today; reserved). */
  state_init:       (number | boolean)[]
  /** Name of each state register slot, in unified order. */
  register_names:   string[]
  /** Scalar type of each state register slot. */
  register_types:   ScalarType[]
  /** Name of each array slot, in unified order. */
  array_slot_names: string[]
  /** Total temp count across all instance functions plus scheduler
   *  preamble. Allocated once by `FlatRuntime::load_plan`. */
  register_count:   number
  /** Total array-slot count. */
  array_slot_count: number
  /** Element count per array slot. */
  array_slot_sizes: number[]

  // ── Multi-function layout ─────────────────────────────────────────────
  instance_functions: InstanceFunction[]
  scheduler_function: SchedulerFunction

  // ── Inter-module slot array (carried over from plan_4) ───────────────
  /** Total slots in the shared inter-module array. */
  slot_count:    number
  /** Name per slot. `"${instance}.${port}"` for output slots,
   *  `param:${name}` for param slots, `"${instance}.__alive__"` for
   *  alive slots. */
  slot_names:    string[]
  /** Initial value per slot. Alive slots default to `1.0`. */
  slot_defaults: number[]
}
