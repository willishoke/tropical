/**
 * compile_session_slotted.ts — session compilation for `tropical_plan_5`.
 *
 * Each session instance compiles standalone (via `compileResolved`) and
 * lands in its own `InstanceFunction` entry. The JIT emits a single
 * LLVM kernel that loops the audio buffer; per sample it runs each
 * instance's body + writebacks in topological order, then reads
 * graphOutput slots in the postamble:
 *
 *     for each sample:
 *       for i in 0..N: body_i; writebacks_i
 *       <state evolution: WriteSlot per extracted delay>
 *       <postamble: DAC stitch — read graphOutput slots into mix temps>
 *
 * ## Branded discipline
 *
 * The unified accumulators carry branded counts/offsets per
 * namespace (`TempIdx`, `StateRegIdx`, `ArraySlotIdx`,
 * `ModuleSlotIdx`). Cross-namespace arithmetic is a compile error —
 * the literal shape of the Phaser bug we hit during PR review.
 */

import { allocateOutputSlots, allocateInputSlots, type SessionState } from '../session.js'
import {
  instanceName as toInstanceName, portName as toPortName,
  portRef, wireKey, childInstance,
  type InstanceName,
} from './branded_names.js'
import type { FlatPlan, InstanceFunction, SchedulerFunction, CompilationMode } from '../flat_plan.js'
import type { ResolvedProgram, InputIdx } from './nodes.js'
import { inputIdx } from './nodes.js'
import { getInstanceType } from './decl_tables.js'
import type { NInstr } from './emit_resolved.js'
import { instrWriteSlot, instrArray, opArray, opConst } from './emit_resolved.js'
import { arraySlotIdx } from './slot_indices.js'
import {
  computeInstanceTopoOrder, emitDacStitch,
  type PreambleEmitter,
  translateNode,
} from './compile_session_slotted_helpers.js'
import {
  tempIdx, moduleSlotIdx, tempOffset,
  rawOffset,
} from './slot_indices.js'
import { partitionKernel, makeAccumulators } from './partition_recursive.js'
import { makeCompiled, type Compiled } from '../program_types.js'
import { slotKey } from './branded_names.js'

export interface CompileSessionSlottedOptions {
  /** Engine realization strategy. Defaults to `'fused'`. */
  compilation_mode?: CompilationMode
}

/** Compile the session into a `tropical_plan_5` `FlatPlan`. */
export function compileSessionSlotted(
  session: SessionState,
  options: CompileSessionSlottedOptions = {},
): FlatPlan {
  return compileSessionSlottedPerInstance(session, options.compilation_mode ?? 'fused')
}

/** Recursively allocate output slots for an instance and every nested
 *  instance in its program tree. Idempotent; partition_recursive
 *  re-issues the same calls during its own walk and finds them
 *  already populated.
 *
 *  Separated from input allocation so we can do all outputs first,
 *  then all inputs — `allocateInputSlots`'s alias logic for
 *  array-typed inputs needs to look up the producer's `outputPortMeta`
 *  and won't find it if the producer hasn't been processed yet. */
function preallocateOutputsRecursive(
  session: SessionState,
  instancePath: InstanceName,
  prog: ResolvedProgram,
  compiled: Compiled,
): void {
  allocateOutputSlots(session, instancePath, compiled)
  for (const decl of prog.body.decls) {
    if (decl.op !== 'instanceDecl') continue
    const childPath = childInstance(instancePath, decl.name)
    const declType = getInstanceType(prog, decl)
    const childCompiled = makeCompiled(declType, { displayName: declType.name })
    preallocateOutputsRecursive(session, childPath, declType, childCompiled)
  }
}

/** Recursively allocate input slots for an instance and every nested
 *  instance in its program tree. Runs AFTER
 *  `preallocateOutputsRecursive` for every instance, so the alias
 *  check in `allocateInputSlots` can see all producers' outputs. */
function preallocateInputsRecursive(
  session: SessionState,
  instancePath: InstanceName,
  prog: ResolvedProgram,
  compiled: Compiled,
): void {
  allocateInputSlots(session, instancePath, compiled)
  for (const decl of prog.body.decls) {
    if (decl.op !== 'instanceDecl') continue
    const childPath = childInstance(instancePath, decl.name)
    const declType = getInstanceType(prog, decl)
    const childCompiled = makeCompiled(declType, { displayName: declType.name })
    preallocateInputsRecursive(session, childPath, declType, childCompiled)
  }
}

/** Build the per-instance `inputArraySlots` map by reading
 *  `session.inputPortMeta`. The kernel's array-typed `InputRef(idx)`
 *  lowers to a `session_array_reg` operand pointing at the recorded
 *  slot (which may be a fresh allocation or an alias to a producer's
 *  array output, depending on `allocateInputSlots`'s alias check). */
function buildInputArraySlots(
  session: SessionState,
  instancePath: InstanceName,
  prog: ResolvedProgram,
): Map<InputIdx, { slot: number; size: number }> {
  const out = new Map<InputIdx, { slot: number; size: number }>()
  for (let i = 0; i < prog.ports.inputs.length; i++) {
    const portDecl = prog.ports.inputs[i]
    const meta = session.inputPortMeta.get(slotKey(instancePath, portDecl.name))
    if (meta === undefined) continue
    if (meta.arraySlot === undefined || meta.arraySize === undefined) continue
    out.set(inputIdx(i), { slot: meta.arraySlot, size: meta.arraySize })
  }
  return out
}

function compileSessionSlottedPerInstance(
  session: SessionState,
  compilationMode: CompilationMode,
): FlatPlan {
  // Two-phase slot pre-allocation — all outputs (recursive into
  // every nested instance) first, then all inputs (recursive). The
  // split is forced by `allocateInputSlots`'s alias logic: for an
  // array-typed input wired via `{op:'ref', instance, output}`, it
  // looks up the producer's `outputPortMeta` to decide whether to
  // bind the same array slot. That lookup is well-defined only if
  // the producer's outputs are already allocated.
  //
  // Both calls are idempotent; partition_recursive re-issues them
  // during its own walk and they short-circuit on the existing meta.
  for (const [name, inst] of session.instanceRegistry) {
    if (inst.compiled !== undefined) {
      preallocateOutputsRecursive(session, toInstanceName(name), inst.compiled.prog, inst.compiled)
    }
  }
  for (const [name, inst] of session.instanceRegistry) {
    if (inst.compiled !== undefined) {
      preallocateInputsRecursive(session, toInstanceName(name), inst.compiled.prog, inst.compiled)
    }
  }

  const order = computeInstanceTopoOrder(session)

  // Recursive partition: for each top-level session instance, walk its
  // ResolvedProgram tree and emit a kernel per InstanceDecl at every
  // level. Slot allocations and register/state/array offsets are
  // threaded through `acc`.
  const acc = makeAccumulators()
  // Seed the array-slot accumulator with the session-level I/O array
  // slots — these occupy global indices [0, ioArraySlotCount). Each
  // per-kernel partition will allocate its own local array slots
  // starting at the current `nextArrayRaw`, which means kernel-local
  // slots are guaranteed to land at globalIdx >= ioArraySlotCount.
  // The categorical reading: kernel-local and session-level array
  // slots share a single linear array buffer at runtime, with the
  // layout `[I/O slots ‖ kernel-local slots]`. The IR-level kind tag
  // (`session_array_reg` vs `array_reg`) survives just long enough
  // for remap to apply the right shift; post-remap, both kinds
  // collapse to `array_reg` with the absolute index already baked in.
  for (let i = 0; i < session.ioArraySlotCount; i++) {
    acc.arraySlotSizes.push(session.ioArraySlotSizes[i])
    acc.arraySlotNames.push(session.ioArraySlotNames[i])
  }
  acc.nextArrayRaw = session.ioArraySlotCount
  const instanceFunctions: InstanceFunction[] = []

  for (const instName of order) {
    const inst = session.instanceRegistry.get(instName)
    if (inst === undefined) continue
    const compiled = inst.compiled
    if (compiled === undefined) {
      throw new Error(
        `compileSessionSlotted: instance '${instName}' has no Compiled type.`,
      )
    }

    // Array-typed input ports surface to the kernel via
    // `inputArraySlots`; scalar inputs continue to flow through
    // `inputBindingFor` → `translateNode`. The two are disjoint by
    // port type — a given InputIdx appears in exactly one.
    const topInputArraySlots = buildInputArraySlots(session, toInstanceName(instName), compiled.prog)

    const { fn } = partitionKernel(
      /* instancePath   */ instName,
      /* prog           */ compiled.prog,
      /* compiled       */ compiled,
      /* inputBindingFor*/ (portNameStr) => {
        const expr = session.inputExprNodes.get(
          wireKey(portRef(toInstanceName(instName), toPortName(portNameStr))),
        )
        if (expr !== undefined) return { kind: 'wired', expr }
        return undefined
      },
      /* defaults       */ (() => {
        const out: Record<string, import('../session.js').ExprNode> = {}
        for (const d of compiled.prog.ports.inputs) {
          if (d.default !== undefined) {
            const init = d.default
            if (typeof init === 'number' || typeof init === 'boolean') {
              out[d.name] = init
            }
          }
        }
        return out
      })(),
      /* paramHandles      */ new Map(),
      session,
      acc,
      /* inputSlotOverride */ undefined,
      /* inputArraySlots   */ topInputArraySlots,
    )
    instanceFunctions.push(fn)
  }

  // DAC stitching reads each graphOutput slot AFTER all instances
  // dispatch — observes the current sample's WriteSlots.
  const schedulerPreamble: NInstr[] = []
  const dac = emitDacStitch(session, tempOffset(acc.nextRegRaw))
  const dacEndRaw = acc.nextRegRaw + dac.tempCount

  // ── State-evolution phase: one WriteSlot per extracted delay.
  //    Runs after instance kernels (which produce the current sample's
  //    output slot values) and before the postamble. Reads source
  //    instances' current-sample outputs and writes them to the delay
  //    slot; next sample, the wire's slot read returns this value —
  //    exactly one sample of latency per MCP wire.
  const stateEvolution: NInstr[] = []
  let stateEvolutionNextTempRaw = dacEndRaw
  const stateEvolutionEmitter: PreambleEmitter = {
    instrs: stateEvolution,
    allocTemp: () => {
      const slot = tempIdx(stateEvolutionNextTempRaw)
      stateEvolutionNextTempRaw += 1
      return slot
    },
  }
  for (const entry of session.delaySlotRegistry) {
    if (entry.isArray) {
      // Array delay: emit `delaySlot[i] = src[i]` for i in [0, size)
      // as a single elementwise `Add` with stride-0 const broadcast.
      // The engine dispatches on the instruction's `dst_kind` tag,
      // so a `kind:'array'` dst with `loop_count=1` (size-1 case)
      // takes the same elementwise path as larger N — no scalar
      // fallthrough, no proxy-on-loop_count surprises.
      //
      // Source is constrained to `{op:'ref', instance, output}` to
      // an array-typed output — the only shape extractSessionDelays
      // currently produces array entries for. Richer array source
      // expressions (arithmetic on refs, etc.) would route through
      // an array-capable translateNode in a separate follow-up.
      if (entry.arraySlot === undefined || entry.arraySize === undefined) {
        throw new Error(
          `compileSessionSlotted: array delay entry '${entry.slotName}' missing arraySlot/arraySize`,
        )
      }
      const sourceExpr = entry.sourceExpr
      if (typeof sourceExpr !== 'object' || sourceExpr === null || Array.isArray(sourceExpr)) {
        throw new Error(
          `compileSessionSlotted: array delay '${entry.slotName}' source must be a ref`,
        )
      }
      const obj = sourceExpr as Record<string, unknown>
      if (obj.op !== 'ref' || typeof obj.instance !== 'string' || typeof obj.output !== 'string') {
        throw new Error(
          `compileSessionSlotted: array delay '${entry.slotName}' source must be {op:'ref', instance, output}`,
        )
      }
      const producerKey = slotKey(toInstanceName(obj.instance), obj.output)
      const producerMeta = session.outputPortMeta.get(producerKey)
      if (producerMeta?.arraySlot === undefined) {
        throw new Error(
          `compileSessionSlotted: array delay '${entry.slotName}' source '${producerKey}' has no array output slot`,
        )
      }
      stateEvolution.push(instrArray(
        'Add',
        arraySlotIdx(entry.arraySlot),
        [opArray(arraySlotIdx(producerMeta.arraySlot)), opConst(0, 'float')],
        entry.arraySize,
        [1, 0],
        'float',
      ))
      continue
    }
    if (entry.slotIdx === undefined) {
      throw new Error(
        `compileSessionSlotted: scalar delay entry '${entry.slotName}' missing slotIdx`,
      )
    }
    const sourceOp = translateNode(
      entry.sourceExpr,
      entry.scalarType,
      session,
      stateEvolutionEmitter,
      entry.slotName,
    )
    stateEvolution.push(
      instrWriteSlot(moduleSlotIdx(entry.slotIdx), sourceOp, entry.scalarType),
    )
  }

  const schedulerFunction: SchedulerFunction = {
    preamble:        schedulerPreamble,
    state_evolution: stateEvolution,
    postamble:       dac.instructions,
    output_targets:  dac.outputTargets,
    outputs:         dac.outputs,
  }

  return {
    schema: 'tropical_plan_5',
    config: { sampleRate: 44100 },
    compilation_mode:  compilationMode,
    state_init:        acc.stateInit,
    register_names:    acc.registerNames,
    register_types:    acc.registerTypes,
    array_slot_names:  acc.arraySlotNames,
    array_slot_count:  acc.nextArrayRaw,
    array_slot_sizes:  acc.arraySlotSizes,
    register_count:    stateEvolutionNextTempRaw,
    instance_functions: instanceFunctions,
    scheduler_function: schedulerFunction,
    ...buildSlotMetadata(session),
  }
}

/** Compute slot allocation metadata from the session registries. */
function buildSlotMetadata(session: SessionState): {
  slot_count: number; slot_names: string[]; slot_defaults: number[]
} {
  const slotCount    = session.slotCount
  const slotNames    = new Array<string>(slotCount).fill('')
  const slotDefaults = new Array<number>(slotCount).fill(0)
  for (const [name, idx] of session.outputSlotRegistry) {
    slotNames[idx] = name
  }
  for (const [name, idx] of session.paramSlotRegistry) {
    slotNames[idx] = `param:${name}`
    const param = session.paramRegistry.get(name)
    if (param !== undefined) slotDefaults[idx] = param.value
  }
  for (const entry of session.delaySlotRegistry) {
    // Scalar delays occupy module slots in this registry; array
    // delays use the session ioArraySlot space and don't show up
    // here. The metadata for array delays lives on the FlatPlan's
    // `array_slot_names` / `array_slot_sizes` (seeded from
    // `session.ioArraySlot*` at compile time).
    if (entry.slotIdx === undefined) continue
    slotNames[entry.slotIdx]    = entry.slotName
    slotDefaults[entry.slotIdx] = entry.init
  }
  // Input slots for sub-instance ports (the fractal path). Prefixed
  // with `input:` so the namespace is distinct from output-slot names
  // (which carry `instance.port` without prefix) and from
  // `param:` / delay-slot names. Defaults are 0 (parent overwrites
  // every sample via WriteSlot before the child reads).
  for (const [name, idx] of session.inputSlotRegistry) {
    slotNames[idx] = `input:${name}`
  }
  return { slot_count: slotCount, slot_names: slotNames, slot_defaults: slotDefaults }
}


/** @deprecated Slot mode is the only mode. */
export function slotModeEnabled(_session?: SessionState, _opt?: boolean): boolean {
  return true
}

// Silence unused-import warnings for re-export-style symbols.
void rawOffset
