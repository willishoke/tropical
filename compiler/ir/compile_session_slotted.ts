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
import type { FlatPlan, InstanceFunction, CompilationMode } from '../flat_plan.js'
import { DEFAULT_SOURCES } from '../flat_plan.js'
import type { ResolvedProgram, InputIdx, ParamIdx } from './nodes.js'
import { inputIdx, paramIdx } from './nodes.js'
import { getInstanceType } from './decl_tables.js'
import type { NInstr } from './emit_resolved.js'
import { instrWriteSlot, instrArray, opArray, opConst } from './emit_resolved.js'
import { arraySlotIdx } from './slot_indices.js'
import {
  computeInstanceTopoOrder, emitSinks,
  type PreambleEmitter,
  translateNode,
} from './compile_session_slotted_helpers.js'
import {
  tempIdx, moduleSlotIdx, tempOffset,
  rawOffset,
} from './slot_indices.js'
import { partitionKernel, makeAccumulators, ROOT_INSTANCE_PATH } from './partition_recursive.js'
import { elaborate } from './elaborator.js'
import { sessionToParsedProgram, sessionTypeResolver } from './session_to_parsed.js'
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
  const mode = options.compilation_mode ?? 'fused'
  // The session is serialized to a synthetic root program and lowered
  // through the shared `partitionKernel` path. (The legacy per-instance
  // scheduler lowering was retired once the `root_vs_flat` oracle had
  // validated this path — see git history.)
  return compileSessionSlottedFromParsed(session, sessionToParsedProgram(session), mode)
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
/** Serialize a (post-extraction) session into a synthetic root
 *  `ResolvedProgram` by running it through the shared `elaborate` front
 *  door. The session's instances are already post-strata `ResolvedProgram`s;
 *  `sessionTypeResolver` hands them to the elaborator by name through its
 *  `ExternalProgramResolver` hook, so this LINKs the already-reduced
 *  instances rather than re-elaborating them from source. Per-wire unit
 *  delays (hoisted into `session.delaySlotRegistry` by
 *  `extractSessionDelays`) serialize to `delay` decls the elaborator folds
 *  into root `RegDecl`s; params serialize to `param` decls. */
function buildSessionRoot(session: SessionState): ResolvedProgram {
  return elaborate(sessionToParsedProgram(session), sessionTypeResolver(session))
}

/** The root lowering with a caller-supplied `ParsedProgram` (the
 *  Phase 3 seam, kept for the classic in-process path — the oracle
 *  engine's `compileSessionSlotted` feeds it `sessionToParsedProgram`).
 *  Elaborates against the session's instances (LINK, not
 *  re-elaboration) and hands the root to the resolved entry point. */
export function compileSessionSlottedFromParsed(
  session: SessionState,
  parsed: ReturnType<typeof sessionToParsedProgram>,
  compilationMode: CompilationMode,
): FlatPlan {
  return compileSessionSlottedFromResolved(
    session,
    elaborate(parsed, sessionTypeResolver(session)),
    compilationMode,
  )
}

/** The root lowering with a caller-supplied root `ResolvedProgram`
 *  (the Phase 4 stage-4a seam: the Lean engine elaborates the session
 *  root itself and ships its `tropical_resolved_1` encoding; the
 *  service decodes it and partitions against the adopted session
 *  state). NB: the decoded root's `programRegistry` holds decoded
 *  *copies* of the session-canonical programs — the stage-1 codec gate
 *  proved decoded copies compile identically through `partitionKernel`,
 *  so no relink to the session registries is performed. */
export function compileSessionSlottedFromResolved(
  session: SessionState,
  root: ResolvedProgram,
  compilationMode: CompilationMode,
): FlatPlan {
  // Two-phase slot pre-allocation — identical to the per-instance
  // path. partitionKernel re-issues these idempotently during its own
  // child walk.
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

  const acc = makeAccumulators()
  // Seed the array-slot accumulator with the session-level I/O array
  // slots (verbatim from the per-instance path) so kernel-local array
  // slots land at globalIdx >= ioArraySlotCount.
  for (let i = 0; i < session.ioArraySlotCount; i++) {
    acc.arraySlotSizes.push(session.ioArraySlotSizes[i])
    acc.arraySlotNames.push(session.ioArraySlotNames[i])
  }
  acc.nextArrayRaw = session.ioArraySlotCount

  // Lower the session's synthetic root once. The root is the session
  // serialized to a `ParsedProgram` and run through the SAME `elaborate`
  // front door the surface path uses, with the instances'
  // already-resolved types supplied via the elaborator's external-resolver
  // hook (LINK, not re-elaboration). The root is naming-transparent
  // (ROOT_INSTANCE_PATH), so child output slots / register names land under
  // bare paths exactly where the flat per-instance path puts them —
  // `emitDacStitch` and hot-swap state-transfer-by-name resolve unchanged.
  const rootCompiled = makeCompiled(root, { displayName: '__session__' })

  // Session params are slot-based (`param:name` module slots driven by
  // the control plane via setSlot, transferred by name on hot-swap).
  // The materializer turned each `{op:'param', name}` wire into a root
  // `ParamDecl`; map each root ParamIdx to its session param slot so
  // `compileResolved` lowers the ref to a slot read (mirroring the
  // per-instance `translateNode`), instead of a dead FFI handle.
  // `root.params[i]` corresponds to `paramIdx(i)` (decl-table order).
  const paramSlots = new Map<ParamIdx, number>()
  for (let i = 0; i < root.params.length; i++) {
    const slot = session.paramSlotRegistry.get(root.params[i].name)
    if (slot !== undefined) paramSlots.set(paramIdx(i), slot)
  }

  const { fn } = partitionKernel(
    /* instancePath    */ ROOT_INSTANCE_PATH,
    /* prog            */ root,
    /* compiled        */ rootCompiled,
    /* inputBindingFor */ () => undefined,
    /* defaults        */ {},
    /* paramHandles    */ new Map(),
    session,
    acc,
    /* inputSlotOverride */ undefined,
    /* inputArraySlots   */ undefined,
    /* paramSlots        */ paramSlots,
  )
  const instanceFunctions: InstanceFunction[] = [fn]

  // Outputs are device-bound sinks (read graphOutput slots directly after
  // the root kernel has run). No scheduler: outputs are device-bound
  // sinks, and the per-wire delays are root RegDecl writebacks inside the
  // root kernel (a trailing read-old/write-new batch), preserving
  // one-sample latency by construction.
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
    register_count:    acc.nextRegRaw,
    instance_functions: instanceFunctions,
    sinks:              emitSinks(session),
    sources:            [...DEFAULT_SOURCES],
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
void buildSessionRoot
