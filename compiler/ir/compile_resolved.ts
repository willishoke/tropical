/**
 * compile_resolved.ts — per-program emit boundary.
 *
 * Takes a post-strata `ResolvedProgram` (no instances, no
 * combinators, no instance refs) and produces a `PerInstancePlan` —
 * the per-instance slice of a `tropical_plan_5` `FlatPlan`. The
 * session-level compiler in `compile_session_slotted.ts` packs one
 * `PerInstancePlan` per session instance into `instance_functions[]`.
 *
 * This function does not produce a runnable plan on its own. The
 * shape it returns is intentionally smaller than `FlatPlan`: no
 * schema, no slot maps, no scheduler. The session compiler owns the
 * runnable-plan boundary.
 */

import type { ResolvedProgram, ResolvedExpr, OutputDecl, RegDecl, ParamDecl, PortType, InstanceDecl, InputDecl, ParamIdx, InputIdx, OutputIdx, InstanceIdx } from './nodes.js'
import type { PerInstancePlan } from '../flat_plan.js'
import { buildSlotMaps, type SlotMaps } from './slots.js'
import { emitResolvedProgram, type EmitSlots, type ScalarType } from './emit_resolved.js'

/** Param-handle bindings for FFI param/trigger decls embedded in a
 *  program type's body, plus optional nested-output and nested-input
 *  slot maps for the fractal compile path. */
export interface CompileResolvedContext {
  /** Keyed by ParamIdx. */
  paramHandles?:      Map<ParamIdx, { ptr: string }>
  /** Session param module-slot indices, keyed by ParamIdx. When set,
   *  `ParamRef` lowers to a slot read (the session slot-based param
   *  model) instead of an FFI handle. Threaded by the root-program
   *  session lowering for the root kernel. */
  paramSlots?:        Map<ParamIdx, number>
  /** Per-sub-instance, per-output-port module-slot map. Keyed by
   *  InstanceIdx (in this program) and OutputIdx (in the instance's
   *  type). Populated by `partition_recursive` for fractal compile;
   *  `NestedOut` refs lower to Slot reads via this map. */
  nestedOutputSlots?: Map<InstanceIdx, Map<OutputIdx, number>>
  /** Per-sub-instance, per-input-port module-slot map. Keyed by
   *  InstanceIdx and InputIdx. Populated by `partition_recursive` for
   *  slot-based input wiring. The parent's compile emits a
   *  `WriteSlot` into each named slot; entries land in
   *  `per_child_pre_input[k]` (parallel to the body's nested-instance
   *  order) so the engine runs each block immediately before recursing
   *  into its corresponding child. */
  nestedInputSlots?:  Map<InstanceIdx, Map<InputIdx, number>>
  /** Module-slot indices for THIS program's own input ports. Set when
   *  the program is being compiled as a sub-instance kernel — its
   *  `InputRef(idx)` lowers to a slot read from
   *  `inputSlotOverride.get(idx)` instead of the legacy `opInput`. */
  inputSlotOverride?: Map<InputIdx, number>
  /** Session-level array-slot indices for THIS program's own
   *  array-typed input ports. When set, `InputRef(arr_port)` lowers to
   *  a `session_array_reg` operand pointing at the recorded slot.
   *  Indexed by InputIdx; size accompanies each entry. */
  inputArraySlots?:        Map<InputIdx, { slot: number; size: number }>
  /** Per-sub-instance, per-input-port session-array-slot map. Keyed
   *  by InstanceIdx and InputIdx. Populated when a child has array-
   *  typed input ports. The parent's per_child_pre_input emission
   *  uses these to write an elementwise copy from the wire expression
   *  into the child's session-absolute input array slot. */
  nestedInputArraySlots?:  Map<InstanceIdx, Map<InputIdx, { slot: number; size: number }>>
  /** Per-sub-instance, per-output-port session-array-slot map. Keyed
   *  by InstanceIdx and OutputIdx. Populated when a child has array-
   *  typed output ports. The parent's `NestedOut(child, output_arr)`
   *  lowers to a `session_array_reg` operand pointing at the recorded
   *  slot. */
  nestedOutputArraySlots?: Map<InstanceIdx, Map<OutputIdx, { slot: number; size: number }>>
}

/** Compile a `ResolvedProgram` to a `PerInstancePlan`.
 *
 *  Accepts both flat (no `InstanceDecl`s in body) and fractal (sub-
 *  `InstanceDecl`s preserved) shapes. In the fractal case, the caller
 *  must pass `nestedOutputSlots` so that `NestedOut` refs resolve to
 *  slot reads; the sub-`InstanceDecl`s themselves are NOT compiled
 *  here — they're sibling kernels emitted by the partitioner. */
export function compileResolved(prog: ResolvedProgram, ctx: CompileResolvedContext = {}): PerInstancePlan {
  const slots = buildSlotMaps(prog)

  // Fractal: surviving InstanceDecls are sibling kernels (handled by
  // partition_recursive). The body's NestedOut refs resolve via
  // `ctx.nestedOutputSlots`. If the caller didn't provide a slot map
  // for a nested-bodied program, that's a contract error — let
  // emit_resolved's terminal check throw a descriptive message.

  // ── Output expressions ──
  // OutputAssign.target is now OutputIdx (or the dac sentinel). Map
  // output position → expression; iterate output ports in order to
  // produce the per-port expression array.
  const outputExprByIdx = new Map<number, ResolvedExpr>()
  for (const a of prog.body.assigns) {
    if (a.op !== 'outputAssign') continue
    if (typeof a.target === 'number') {
      outputExprByIdx.set(a.target, a.expr)
    }
  }
  const outputExprs: ResolvedExpr[] = prog.ports.outputs.map((out, i) => {
    const expr = outputExprByIdx.get(i)
    if (expr === undefined) {
      throw new Error(`compileResolved: program '${prog.name}' output '${out.name}' has no outputAssign.`)
    }
    return expr
  })

  // ── Register update expressions ──
  // Post-Phase-0a: every reg's update (if any) lives on `decl.update`.
  // NextUpdate body-assigns are gone (folded into the decl by the
  // elaborator). A reg with `update === undefined` holds its current
  // value (semantically a register); a reg with `update` set evaluates
  // the expression to produce the next sample's value (semantically a
  // delay or stateful accumulator).
  const registerExprs: (ResolvedExpr | null)[] = []
  const stateInit:     (number | boolean | number[])[] = []
  const registerNames: string[] = []
  const registerTypes: ScalarType[] = []

  for (const d of slots.regDecls) {
    registerNames.push(d.name)
    registerTypes.push(regScalarType(d))
    stateInit.push(regInit(d))
    registerExprs.push(d.update === undefined ? null : d.update)
  }

  const inputPortTypes: ScalarType[] = slots.inputDecls.map(d => {
    if (d.type === undefined) return 'float'
    if (d.type.kind === 'scalar') return d.type.scalar
    if (d.type.kind === 'alias')  return d.type.alias.base
    if (typeof d.type.element === 'string') return d.type.element
    return d.type.element.base
  })

  const emitSlots: EmitSlots = {
    inputs:                  slots.inputs,
    regs:                    slots.regs,
    regCount:                slots.regDecls.length,
    paramHandles:            ctx.paramHandles ?? new Map(),
    paramSlots:              ctx.paramSlots,
    nestedOutputSlots:       ctx.nestedOutputSlots,
    nestedInputSlots:        ctx.nestedInputSlots,
    inputSlotOverride:       ctx.inputSlotOverride,
    inputArraySlots:         ctx.inputArraySlots,
    nestedInputArraySlots:   ctx.nestedInputArraySlots,
    nestedOutputArraySlots:  ctx.nestedOutputArraySlots,
  }

  // Fractal: collect sub-instance decls so emit_resolved can emit
  // a `WriteSlot` for each of their wired inputs in
  // `per_child_pre_input[k]`. The order here is significant — it's
  // the dispatch order partition_recursive will use when packing
  // children into the InstanceFunction tree, so per_child_pre_input
  // and the children array stay parallel. Empty when the program has
  // no nested instances (flat path).
  const nestedInstances: InstanceDecl[] = []
  for (const d of prog.body.decls) {
    if (d.op === 'instanceDecl') nestedInstances.push(d)
  }

  // Per-port scalar slot counts derived from declared output shapes.
  // Scalar/alias = 1; array = product of shape dims. Drives the
  // output_targets expansion in emit_resolved.
  const outputPortScalarCounts = prog.ports.outputs.map(outputPortScalarCount)

  const program = emitResolvedProgram({
    outputExprs,
    outputPortScalarCounts,
    registerExprs,
    stateInit,
    stateRegTypes: registerTypes,
    inputPortTypes,
    slots: emitSlots,
    nested: { instances: nestedInstances, enclosing: prog },
  })

  const arraySlotNames: string[] = []
  for (let i = 0; i < stateInit.length; i++) {
    if (Array.isArray(stateInit[i])) arraySlotNames.push(registerNames[i])
  }

  return {
    register_count:   program.register_count,
    array_slot_count: program.array_slot_count,
    array_slot_sizes: program.array_slot_sizes,
    instructions:     program.instructions,
    per_child_pre_input: program.per_child_pre_input,
    output_targets:   program.output_targets,
    register_targets: program.register_targets,
    state_init:       stateInit as (number | boolean)[],
    register_names:   registerNames,
    register_types:   registerTypes,
    array_slot_names: arraySlotNames,
  }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

function regScalarType(d: RegDecl): ScalarType {
  if (d.type === undefined) return 'float'
  if (typeof d.type === 'string') return d.type
  return d.type.base
}

function regInit(d: RegDecl): number | boolean | number[] {
  const init = d.init
  if (typeof init === 'number') return init
  if (typeof init === 'boolean') return init
  if (Array.isArray(init)) return init as number[]
  if (typeof init === 'object' && init !== null && (init as { op?: string }).op === 'zeros') {
    const count = (init as { count: ResolvedExpr }).count
    if (typeof count !== 'number') {
      throw new Error('compileResolved: zeros count must be a literal integer')
    }
    return new Array(count).fill(0)
  }
  throw new Error('compileResolved: register init must lower to a literal value')
}

/** Total scalar slot count for an output port's declared shape. Scalar
 *  and alias ports count as 1; array ports as the product of their
 *  shape dimensions. Used to drive `output_targets` expansion. */
function outputPortScalarCount(decl: OutputDecl): number {
  const t: PortType | undefined = decl.type
  if (t === undefined) return 1
  if (t.kind === 'scalar') return 1
  if (t.kind === 'alias')  return 1
  // array
  let total = 1
  for (const dim of t.shape) {
    if (typeof dim !== 'number') {
      throw new Error(
        `compileResolved: output port '${decl.name}' has unresolved ` +
        `type-param dimension; ensure specialize ran first`,
      )
    }
    total *= dim
  }
  return total
}

void buildSlotMaps
export type { SlotMaps }
