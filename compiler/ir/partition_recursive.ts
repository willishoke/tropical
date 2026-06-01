/**
 * partition_recursive.ts — recursive partitioner for fractal compilation.
 *
 * Walks a `ResolvedProgram` whose body may contain nested `InstanceDecl`s
 * and produces a tree of `InstanceFunction`s — one per `InstanceDecl` at
 * every level of nesting. Each kernel:
 *
 *   - has its own state-reg / temp / array-slot offsets in the unified
 *     namespaces (no per-kernel isolation; just bookkeeping)
 *   - has its own output slots (full-path: `voice1.env.out` etc.)
 *   - emits WriteSlots for each scalar slot of every output port
 *
 * Children's inputs are substituted into their bodies at partition time
 * via `cloneWithInputSubst` — by the time `compileResolved` runs on a
 * child, no `InputRef` survives in the body. The substituted child
 * compiles as a self-contained kernel.
 *
 * Cross-kernel reads in the parent's body (`NestedOut(child, output)`)
 * resolve to slot reads via `compileResolved`'s `nestedOutputSlots`
 * context — populated here as we allocate child output slots before
 * compiling the parent.
 */

import type {
  ResolvedProgram, InstanceDecl, InputDecl, OutputDecl, ParamDecl,
  InputIdx, OutputIdx, InstanceIdx, ParamIdx,
} from './nodes.js'
import { inputIdx, outputIdx, instanceIdx } from './nodes.js'
import { getInstanceType } from './decl_tables.js'
import type { SessionState, ExprNode } from '../session.js'
import type {
  InstanceFunction, RegTarget,
} from '../flat_plan.js'
import { TempTarget, ArrayManagedTarget } from '../flat_plan.js'
import type { NInstr } from './emit_resolved.js'
import {
  tempIdx, moduleSlotIdx,
  tempOffset, stateRegOffset, arraySlotOffset,
} from './slot_indices.js'
import { type ScalarType } from './emit_resolved.js'
import { instanceName as toInstanceName, slotKey } from './branded_names.js'
import { compileResolved } from './compile_resolved.js'
import { makeCompiled, type Compiled } from '../program_types.js'
import {
  inputNames, outputNames, outputPortTypes, rawInputDefaults,
} from '../program_types.js'
import { allocateOutputSlots, allocateInputSlots } from '../session.js'
import {
  remapInstancePlan, type RemapContext, type InputBinding,
} from './compile_session_slotted_helpers.js'

// ─── Accumulator ──────────────────────────────────────────────────────────

/** Mutable state threaded through the recursion. The unified slot
 *  namespaces accumulate as kernels are emitted; the order matches
 *  depth-first walk of the kernel tree (children before parent). */
export interface PartitionAccumulators {
  nextRegRaw:        number
  nextStateRaw:      number
  nextArrayRaw:      number
  registerNames:     string[]
  registerTypes:     ScalarType[]
  stateInit:         (number | boolean)[]
  arraySlotSizes:    number[]
  arraySlotNames:    string[]
}

export function makeAccumulators(): PartitionAccumulators {
  return {
    nextRegRaw:        0,
    nextStateRaw:      0,
    nextArrayRaw:      0,
    registerNames:     [],
    registerTypes:     [],
    stateInit:         [],
    arraySlotSizes:    [],
    arraySlotNames:    [],
  }
}

// ─── Helpers ──────────────────────────────────────────────────────────────

const scalarOf = (t: ReturnType<typeof outputPortTypes>[number]): ScalarType => {
  if (t === undefined) return 'float'
  if (t.kind === 'scalar') return t.scalar
  if (t.kind === 'alias')  return t.alias.base as ScalarType
  return 'float'
}

/** Look up the (first scalar) slot index for an instance's output port.
 *  Returns undefined for array-typed ports (which have no scalar slot,
 *  only a session-array slot — see `lookupOutputArraySlot`). */
function lookupOutputSlot(
  session: SessionState,
  instancePath: string,
  portName: string,
): number | undefined {
  const portKey = slotKey(toInstanceName(instancePath), portName)
  const meta = session.outputPortMeta.get(portKey)
  if (meta === undefined || meta.scalarSlotNames.length === 0) return undefined
  return session.outputSlotRegistry.get(meta.scalarSlotNames[0])
}

/** Look up the (first scalar) slot index for an instance's INPUT port.
 *  Mirror of `lookupOutputSlot` against the input-slot registry.
 *  Returns undefined for array-typed ports. */
function lookupInputSlot(
  session: SessionState,
  instancePath: string,
  portName: string,
): number | undefined {
  const portKey = slotKey(toInstanceName(instancePath), portName)
  const meta = session.inputPortMeta.get(portKey)
  if (meta === undefined || meta.scalarSlotNames.length === 0) return undefined
  return session.inputSlotRegistry.get(meta.scalarSlotNames[0])
}

/** Look up the session-array slot index + size for an instance's
 *  array-typed output port. Sibling of `lookupOutputSlot` for the
 *  array allocation class. Returns undefined when the port either
 *  doesn't exist on this instance or isn't array-typed. */
function lookupOutputArraySlot(
  session: SessionState,
  instancePath: string,
  portName: string,
): { slot: number; size: number } | undefined {
  const portKey = slotKey(toInstanceName(instancePath), portName)
  const meta = session.outputPortMeta.get(portKey)
  if (meta === undefined || meta.arraySlot === undefined || meta.arraySize === undefined) return undefined
  return { slot: meta.arraySlot, size: meta.arraySize }
}

/** Look up the session-array slot index + size for an instance's
 *  array-typed INPUT port. Sibling of `lookupInputSlot` for the
 *  array allocation class. */
function lookupInputArraySlot(
  session: SessionState,
  instancePath: string,
  portName: string,
): { slot: number; size: number } | undefined {
  const portKey = slotKey(toInstanceName(instancePath), portName)
  const meta = session.inputPortMeta.get(portKey)
  if (meta === undefined || meta.arraySlot === undefined || meta.arraySize === undefined) return undefined
  return { slot: meta.arraySlot, size: meta.arraySize }
}

/** Sentinel `instancePath` for the synthetic session root (Option A).
 *  `partitionKernel` treats it as naming-transparent: the root's
 *  children and its own registers carry BARE names (no `__root__.`
 *  prefix), so child output-slot keys (`amp.out`) and per-instance
 *  register names land byte-for-byte where the flat per-instance path
 *  puts them. Without this the root would prefix every nested name and
 *  break `emitDacStitch`'s `graphOutputs` slot lookup and hot-swap
 *  state-transfer-by-name. */
export const ROOT_INSTANCE_PATH = '__root__'

/** Join an instance path with a child segment, transparently for the
 *  synthetic root (no prefix). For any real instance path this is the
 *  ordinary dotted join. */
function joinInstancePath(parent: string, child: string): string {
  return parent === ROOT_INSTANCE_PATH ? child : `${parent}.${child}`
}

// ─── Entry point ──────────────────────────────────────────────────────────

/** Result of compiling a single kernel: the InstanceFunction (with
 *  nested children populated) plus the slot map describing this
 *  kernel's outputs so the caller can wire NestedOut refs in the
 *  parent's body. */
export interface PartitionedKernel {
  fn: InstanceFunction
  /** Per output-port slot index of this kernel's outputs. Used by the
   *  parent's `nestedOutputSlots` map when compiling the parent. */
  outputSlots: Map<OutputDecl, number>
}

/** Partition a single kernel (recursively). Mutates `session` (slot
 *  allocations) and `acc` (offset accumulators).
 *
 *  `inputBindingFor` provides the binding for each input port of THIS
 *  kernel. For top-level session instances, it consults
 *  `session.inputExprNodes`. For nested kernels, all input ports are
 *  handled by slot-based parent→child wiring (the child's
 *  `InputRef(idx)` lowers to a Slot read via `inputSlotOverride`), so
 *  this function is unused (no `opInput` operands survive). Pass
 *  `() => undefined` for nested calls; the defaults-fallback handles
 *  unused declared inputs.
 */
export function partitionKernel(
  instancePath: string,
  prog: ResolvedProgram,
  compiled: Compiled,
  inputBindingFor: (portName: string) => InputBinding | undefined,
  defaults: Record<string, ExprNode>,
  paramHandles: Map<ParamIdx, { ptr: string }>,
  session: SessionState,
  acc: PartitionAccumulators,
  /** Slot-based input wiring (scalar/alias class): when set, THIS
   *  kernel is being compiled as a sub-instance. Its scalar
   *  `InputRef(idx)` operands lower to Slot reads from the slot
   *  indices recorded here, instead of the legacy `opInput`.
   *  Top-level callers pass `undefined`. */
  inputSlotOverride?: Map<InputIdx, number>,
  /** Slot-based input wiring (array class): sibling of
   *  `inputSlotOverride` for array-typed input ports. THIS kernel's
   *  array-typed `InputRef(idx)` lowers to a `session_array_reg`
   *  operand pointing at the recorded session-array slot. Scalar and
   *  array maps are disjoint by construction — each port appears in
   *  exactly one. */
  inputArraySlots?:  Map<InputIdx, { slot: number; size: number }>,
  /** Session param module-slot indices for THIS kernel, keyed by the
   *  kernel program's ParamIdx. When set, `ParamRef` lowers to a slot
   *  read (the slot-based session param model). NOT propagated to child
   *  recursions — `ParamIdx` is per-program, so a child resolves its own
   *  params via its own (typically empty) handle/slot maps. */
  paramSlots?:       Map<ParamIdx, number>,
): PartitionedKernel {
  // ── 1. Recurse into sub-InstanceDecls.
  //    For each child:
  //      • Allocate output slots (parent reads via NestedOut → Slot)
  //      • Allocate INPUT slots (parent writes via WriteSlot in
  //        per_child_pre_input[k]; child reads via Slot in its body)
  //      • Build per-child slot maps for both directions
  //      • Recurse into the child WITHOUT substituting its inputs —
  //        the child's InputRefs lower via `inputSlotOverride` instead
  //
  //    The slot-based path replaces the legacy `cloneWithInputSubst`
  //    substitution, which leaked scope (wire expressions could carry
  //    refs into the child's namespace where they couldn't resolve).
  //    Under slots, the wire is evaluated in the parent's scope where
  //    every ref resolves, and only the *value* crosses the boundary.
  const children: InstanceFunction[] = []
  // Maps are now keyed by indices: InstanceIdx for this program's
  // instances (position in body.decls instance order), InputIdx/OutputIdx
  // for the target program's port positions. compileResolved + emit
  // consume these in index form.
  //
  // Two parallel allocation classes per port direction:
  //   - scalar/alias ports → module slots (`nested*Slots`, values are
  //     plain slot indices). Parent emits `WriteSlot`/`opSlot`.
  //   - array ports → session-array slots (`nested*ArraySlots`, values
  //     are `{slot, size}` records). Parent emits elementwise copies
  //     via `instrSessionArray` / `opSessionArray`.
  // A port appears in exactly one class. The class is determined by
  // the port's IRPortType.kind at allocation time (see
  // `allocateOutputSlots` / `allocateInputSlots`).
  const nestedOutputSlots      = new Map<InstanceIdx, Map<OutputIdx, number>>()
  const nestedInputSlots       = new Map<InstanceIdx, Map<InputIdx,  number>>()
  const nestedOutputArraySlots = new Map<InstanceIdx, Map<OutputIdx, { slot: number; size: number }>>()
  const nestedInputArraySlots  = new Map<InstanceIdx, Map<InputIdx,  { slot: number; size: number }>>()

  let childInstIdx = 0
  for (const decl of prog.body.decls) {
    if (decl.op !== 'instanceDecl') continue
    const childPath = joinInstancePath(instancePath, decl.name)
    const thisInstIdx = instanceIdx(childInstIdx++)

    const declType = getInstanceType(prog, decl)
    const childCompiled = makeCompiled(declType, { displayName: declType.name })

    // Allocate both directions of slots for this child. The allocator
    // dispatches per-port on the port's IRPortType.kind: scalar/alias
    // → module slot in `outputSlotRegistry`/`inputSlotRegistry`; array
    // → session-array slot in `ioArraySlot*`. Either way the result
    // lands on the port's WirePortMeta.
    allocateOutputSlots(session, toInstanceName(childPath), childCompiled)
    allocateInputSlots (session, toInstanceName(childPath), childCompiled)

    // Build slot maps for parent's NestedOut → child output reads.
    // Each output port appears in EXACTLY ONE of the two maps,
    // determined by its allocation class.
    const childOutputMap      = new Map<OutputIdx, number>()
    const childOutputArrayMap = new Map<OutputIdx, { slot: number; size: number }>()
    for (let i = 0; i < declType.ports.outputs.length; i++) {
      const outDecl = declType.ports.outputs[i]
      const outIdx = outputIdx(i)
      const scalarSlot = lookupOutputSlot(session, childPath, outDecl.name)
      if (scalarSlot !== undefined) {
        childOutputMap.set(outIdx, scalarSlot)
        continue
      }
      const arrayInfo = lookupOutputArraySlot(session, childPath, outDecl.name)
      if (arrayInfo !== undefined) childOutputArrayMap.set(outIdx, arrayInfo)
    }
    nestedOutputSlots     .set(thisInstIdx, childOutputMap)
    nestedOutputArraySlots.set(thisInstIdx, childOutputArrayMap)

    // Build slot maps for parent's per_child_pre_input writes →
    // child input. Same per-port dispatch as outputs.
    const childInputMap      = new Map<InputIdx, number>()
    const childInputArrayMap = new Map<InputIdx, { slot: number; size: number }>()
    for (let i = 0; i < declType.ports.inputs.length; i++) {
      const inDecl = declType.ports.inputs[i]
      const inIdx = inputIdx(i)
      const scalarSlot = lookupInputSlot(session, childPath, inDecl.name)
      if (scalarSlot !== undefined) {
        childInputMap.set(inIdx, scalarSlot)
        continue
      }
      const arrayInfo = lookupInputArraySlot(session, childPath, inDecl.name)
      if (arrayInfo !== undefined) childInputArrayMap.set(inIdx, arrayInfo)
    }
    nestedInputSlots     .set(thisInstIdx, childInputMap)
    nestedInputArraySlots.set(thisInstIdx, childInputArrayMap)

    // Recursively compile the child. The child's own array-typed
    // input ports surface as `inputArraySlots` (a sibling of
    // `inputSlotOverride` for the array allocation class). Scalar
    // inputs still lower via `inputSlotOverride`; array inputs lower
    // to `session_array_reg` operands via `inputArraySlots`.
    const childResult = partitionKernel(
      childPath, declType, childCompiled,
      /* inputBindingFor*/ () => undefined,
      /* defaults       */ rawInputDefaults(childCompiled),
      paramHandles,
      session, acc,
      /* inputSlotOverride */ childInputMap,
      /* inputArraySlots   */ childInputArrayMap,
    )
    children.push(childResult.fn)
  }

  // ── 2. Compile this kernel's body via compileResolved. With the
  //    nested context populated, `compileResolved` will:
  //      • Emit per-child pre_input blocks: scalar inputs become
  //        WriteSlots into nestedInputSlots; array inputs become
  //        elementwise copies into nestedInputArraySlots.
  //      • Lower scalar NestedOut refs in the parent body via
  //        nestedOutputSlots; lower array NestedOut refs via
  //        nestedOutputArraySlots (to session_array_reg operands).
  //      • Lower scalar InputRef refs in THIS kernel's body via
  //        inputSlotOverride; lower array InputRef refs via
  //        inputArraySlots (to session_array_reg operands).
  const plan = compileResolved(prog, {
    paramHandles,
    paramSlots,
    nestedOutputSlots,
    nestedInputSlots,
    nestedOutputArraySlots,
    nestedInputArraySlots,
    inputSlotOverride,
    inputArraySlots,
  })

  // ── 3. Remap the plan into the unified slot/temp space.
  const inPortNames  = inputNames(compiled)
  const outPortNames = outputNames(compiled)
  const outPortTypes = outputPortTypes(compiled).map(scalarOf)

  const ctx: RemapContext = {
    instanceName: instancePath,
    regOffset:       tempOffset(acc.nextRegRaw),
    stateRegOffset:  stateRegOffset(acc.nextStateRaw),
    arraySlotOffset: arraySlotOffset(acc.nextArrayRaw),
    inputBindingFor: (portName: string) => {
      const binding = inputBindingFor(portName)
      if (binding !== undefined) return binding
      const d = defaults[portName]
      const value = (typeof d === 'number' || typeof d === 'boolean') ? d : 0
      return { kind: 'literal', value }
    },
    outputSlotFor: (portName: string) => {
      const slot = lookupOutputSlot(session, instancePath, portName)
      if (slot === undefined) {
        throw new Error(
          `partitionKernel: instance '${instancePath}' port '${portName}' has no allocated slot`,
        )
      }
      return moduleSlotIdx(slot)
    },
    inputPortNames:    inPortNames,
    outputPortNames:   outPortNames,
    outputScalarTypes: outPortTypes,
  }

  const { preamble, perChildPreInput, body, writeSlots, tempsConsumed } = remapInstancePlan(plan, ctx, session)
  // The preamble holds temp-computes for session-wired input translations
  // (e.g., a wire's `pulseEvery(64)` expression resolves to instructions
  // emitting into a temp). Those temps are referenced by the per-child
  // WriteSlots in perChildPreInput. Children dispatch BEFORE this kernel's
  // main `instructions`, so the preamble has to live in a separate field
  // that the engine emits before children. Bundling it into
  // `instructions` (as the legacy code did) puts the compute AFTER its
  // use — works for literal session inputs (no preamble emission), broken
  // for expression-shaped session inputs like Bubble's pulseEvery trigger.
  const instanceInstructions: NInstr[] = [...body, ...writeSlots]

  // Attach each per-child pre-input block to its corresponding child
  // InstanceFunction. The pre-input wires live in THIS kernel's
  // namespace (parent evaluates them) but are stored on the child so
  // the engine's `emit_kernel_block` runs `child.pre_input_instructions`
  // immediately before recursing into that child — preserving sibling-
  // to-sibling NestedOut dependencies.
  //
  // `children` and `perChildPreInput` are kept parallel by construction:
  // both are built from `prog.body.decls` in the same order (this
  // function above; compileResolved's nestedInstances collection).
  if (perChildPreInput.length !== children.length) {
    throw new Error(
      `partitionKernel: instance '${instancePath}': perChildPreInput length ` +
      `(${perChildPreInput.length}) does not match children length ` +
      `(${children.length}). emit_resolved + compileResolved must produce ` +
      `one block per nested InstanceDecl in body order.`,
    )
  }
  for (let i = 0; i < children.length; i++) {
    children[i] = { ...children[i], pre_input_instructions: perChildPreInput[i] }
  }

  // Shift per-instance register_targets into the unified temp space.
  const shiftedTargets: RegTarget[] = plan.register_targets.map(t => {
    if (t.kind === 'arrayManaged') return ArrayManagedTarget
    return TempTarget(tempIdx(t.slot + acc.nextRegRaw))
  })

  const fn: InstanceFunction = {
    name:              `instance_${instancePath.replace(/\./g, '_')}`,
    instance_name:     instancePath,
    preamble_instructions:  preamble,
    instructions:      instanceInstructions,
    // Top-level kernels run no parent-side pre-input wires (their
    // inputs come from session inputBindings translated into the
    // preamble). Sub-kernels get this populated by the parent's
    // partitionKernel call via the `for (let i = 0; ...)` loop above
    // — that loop overwrites `children[i]` with a copy carrying the
    // child's `pre_input_instructions`.
    pre_input_instructions: [],
    register_offset:   tempOffset(acc.nextRegRaw),
    state_reg_offset:  stateRegOffset(acc.nextStateRaw),
    array_slot_offset: arraySlotOffset(acc.nextArrayRaw),
    register_count:    plan.register_count + tempsConsumed,
    register_targets:  shiftedTargets,
    children,
  }

  // ── 4. Update accumulators (THIS kernel's contribution to the unified
  //    namespaces). Note: children's accumulator updates already happened
  //    via the recursive calls above, so we just add this kernel's own.
  for (const n of plan.register_names) acc.registerNames.push(joinInstancePath(instancePath, n))
  acc.registerTypes.push(...plan.register_types)
  for (const v of plan.state_init) acc.stateInit.push(v as number | boolean)
  acc.arraySlotSizes.push(...plan.array_slot_sizes)
  for (const n of plan.array_slot_names) acc.arraySlotNames.push(joinInstancePath(instancePath, n))

  acc.nextRegRaw   += plan.register_count + tempsConsumed
  acc.nextStateRaw += plan.state_init.length
  acc.nextArrayRaw += plan.array_slot_count

  // ── 5. Build the parent-visible output slot map (one entry per
  //    output port; for array ports this is the first scalar slot).
  const outputSlots = new Map<OutputDecl, number>()
  for (const outDecl of prog.ports.outputs) {
    const slot = lookupOutputSlot(session, instancePath, outDecl.name)
    if (slot !== undefined) outputSlots.set(outDecl, slot)
  }

  return { fn, outputSlots }
}
