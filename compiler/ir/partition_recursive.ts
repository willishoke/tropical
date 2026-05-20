/**
 * partition_recursive.ts — recursive partitioner for fractal compilation.
 *
 * Walks a `ResolvedProgram` whose body may contain nested `InstanceDecl`s
 * and produces a tree of `InstanceFunction`s — one per `InstanceDecl` at
 * every level of nesting. Each kernel:
 *
 *   - has its own alive slot (full-path name: `voice1.env.__alive__`)
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
 *
 * No engine-side change needed: `OrcJitEngine.cpp::emit_kernel_block`
 * already recursively emits child basic blocks inside the parent's
 * alive-conditional.
 */

import type {
  ResolvedProgram, ResolvedExpr, InstanceDecl, InputDecl, OutputDecl, ParamDecl,
} from './nodes.js'
import type { SessionState, ExprNode } from '../session.js'
import type {
  InstanceFunction, RegTarget,
} from '../flat_plan.js'
import { TempTarget, ArrayManagedTarget } from '../flat_plan.js'
import type { NInstr } from './emit_resolved.js'
import {
  type TempIdx, type ModuleSlotIdx,
  tempIdx, moduleSlotIdx,
  tempOffset, stateRegOffset, arraySlotOffset,
} from './slot_indices.js'
import { type ScalarType, instrWriteSlot, opConst } from './emit_resolved.js'
import { instanceName as toInstanceName, slotKey } from './branded_names.js'
import { compileResolved } from './compile_resolved.js'
import { cloneWithInputSubst } from './clone.js'
import { makeCompiled, type Compiled } from '../program_types.js'
import {
  inputNames, outputNames, outputPortTypes, rawInputDefaults,
} from '../program_types.js'
import { allocateOutputSlots } from '../session.js'
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
  /** Alive-slot preamble emissions, one per kernel. Filled as we walk
   *  the tree; appended to the scheduler preamble at the end. */
  alivePreambleOps:  Array<{ aliveSlot: ModuleSlotIdx; expr: ExprNode | undefined }>
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
    alivePreambleOps:  [],
  }
}

// ─── Helpers ──────────────────────────────────────────────────────────────

const scalarOf = (t: ReturnType<typeof outputPortTypes>[number]): ScalarType => {
  if (t === undefined) return 'float'
  if (t.kind === 'scalar') return t.scalar
  if (t.kind === 'alias')  return t.alias.base as ScalarType
  return 'float'
}

/** Look up the (first scalar) slot index for an instance's output port. */
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
 *  allocations) and `acc` (offset accumulators + alive preamble ops).
 *
 *  `inputBindingFor` provides the binding for each input port of THIS
 *  kernel. For top-level session instances, it consults
 *  `session.inputExprNodes`. For nested kernels, all input ports have
 *  been substituted out by `cloneWithInputSubst`, so this function is
 *  unused (no `opInput` operands survive). Pass `() => undefined` for
 *  nested calls; the defaults-fallback handles unused declared inputs.
 */
export function partitionKernel(
  instancePath: string,
  prog: ResolvedProgram,
  compiled: Compiled,
  aliveInput: ExprNode | undefined,
  inputBindingFor: (portName: string) => InputBinding | undefined,
  defaults: Record<string, ExprNode>,
  paramHandles: Map<ParamDecl, { ptr: string }>,
  session: SessionState,
  acc: PartitionAccumulators,
): PartitionedKernel {
  // ── 1. Recurse into sub-InstanceDecls. Substitute their inputs;
  //    allocate their output slots; build the nestedOutputSlots map for
  //    this kernel's NestedOut refs.
  const children: InstanceFunction[] = []
  const nestedOutputSlots = new Map<InstanceDecl, Map<OutputDecl, number>>()

  for (const decl of prog.body.decls) {
    if (decl.op !== 'instanceDecl') continue
    const childPath = `${instancePath}.${decl.name}`

    // Substitute the child's input refs with the parent's wired expressions.
    const inputSubst = new Map<InputDecl, ResolvedExpr>()
    for (const inp of decl.inputs) inputSubst.set(inp.port, inp.value)
    const substChildProg = cloneWithInputSubst(decl.type, inputSubst)
    const childCompiled = makeCompiled(substChildProg, { displayName: decl.type.name })

    // Allocate output slots (+ __alive__) for the child.
    allocateOutputSlots(session, toInstanceName(childPath), childCompiled)

    // Build slot map for parent's NestedOut → child output reads.
    const childSlotMap = new Map<OutputDecl, number>()
    for (const outDecl of decl.type.ports.outputs) {
      const slotIdx = lookupOutputSlot(session, childPath, outDecl.name)
      if (slotIdx !== undefined) childSlotMap.set(outDecl, slotIdx)
    }
    nestedOutputSlots.set(decl, childSlotMap)

    // Recursively compile the child. Children inherit their parent's
    // paramHandles. AliveInput is undefined for nested (default 1.0).
    // No external input bindings — all substituted into the body.
    const childResult = partitionKernel(
      childPath, substChildProg, childCompiled,
      /* aliveInput     */ undefined,
      /* inputBindingFor*/ () => undefined,
      /* defaults       */ rawInputDefaults(childCompiled),
      paramHandles,
      session, acc,
    )
    children.push(childResult.fn)
  }

  // ── 2. Compile this kernel's body via compileResolved.
  const plan = compileResolved(prog, {
    paramHandles,
    nestedOutputSlots,
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

  const { preamble, body, writeSlots, tempsConsumed } = remapInstancePlan(plan, ctx, session)
  const instanceInstructions: NInstr[] = [...preamble, ...body, ...writeSlots]

  const aliveSlotRaw = session.outputSlotRegistry.get(slotKey(toInstanceName(instancePath), '__alive__'))
  if (aliveSlotRaw === undefined) {
    throw new Error(
      `partitionKernel: '${instancePath}' has no __alive__ slot — allocateOutputSlots should have reserved it`,
    )
  }
  const aliveSlot = moduleSlotIdx(aliveSlotRaw)

  // Record this kernel's alive expression so the caller can build the
  // scheduler preamble (which writes 1.0 or the alive expr each sample,
  // letting GVN fold the default-alive case).
  acc.alivePreambleOps.push({ aliveSlot, expr: aliveInput })

  // Shift per-instance register_targets into the unified temp space.
  const shiftedTargets: RegTarget[] = plan.register_targets.map(t => {
    if (t.kind === 'arrayManaged') return ArrayManagedTarget
    return TempTarget(tempIdx(t.slot + acc.nextRegRaw))
  })

  const fn: InstanceFunction = {
    name:              `instance_${instancePath.replace(/\./g, '_')}`,
    instance_name:     instancePath,
    instructions:      instanceInstructions,
    register_offset:   tempOffset(acc.nextRegRaw),
    state_reg_offset:  stateRegOffset(acc.nextStateRaw),
    array_slot_offset: arraySlotOffset(acc.nextArrayRaw),
    register_count:    plan.register_count + tempsConsumed,
    register_targets:  shiftedTargets,
    alive_slot_index:  aliveSlot,
    children,
  }

  // ── 4. Update accumulators (THIS kernel's contribution to the unified
  //    namespaces). Note: children's accumulator updates already happened
  //    via the recursive calls above, so we just add this kernel's own.
  for (const n of plan.register_names) acc.registerNames.push(`${instancePath}.${n}`)
  acc.registerTypes.push(...plan.register_types)
  for (const v of plan.state_init) acc.stateInit.push(v as number | boolean)
  acc.arraySlotSizes.push(...plan.array_slot_sizes)
  for (const n of plan.array_slot_names) acc.arraySlotNames.push(`${instancePath}.${n}`)

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
