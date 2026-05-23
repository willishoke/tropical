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

import { allocateOutputSlots, type SessionState } from '../session.js'
import {
  instanceName as toInstanceName, portName as toPortName,
  portRef, wireKey,
} from './branded_names.js'
import type { FlatPlan, InstanceFunction, SchedulerFunction, CompilationMode } from '../flat_plan.js'
import type { NInstr } from './emit_resolved.js'
import { instrWriteSlot } from './emit_resolved.js'
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

function compileSessionSlottedPerInstance(
  session: SessionState,
  compilationMode: CompilationMode,
): FlatPlan {
  // Auto-allocate output slots for any instance whose owner hasn't
  // pre-allocated. `add_instance` / `loadProgramAsSession` already do
  // this eagerly; tests sometimes poke `instanceRegistry` directly.
  for (const [name, inst] of session.instanceRegistry) {
    if (inst.compiled !== undefined) allocateOutputSlots(session, toInstanceName(name), inst.compiled)
  }

  const order = computeInstanceTopoOrder(session)

  // Recursive partition: for each top-level session instance, walk its
  // ResolvedProgram tree and emit a kernel per InstanceDecl at every
  // level. Slot allocations and register/state/array offsets are
  // threaded through `acc`.
  const acc = makeAccumulators()
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
      /* paramHandles   */ new Map(),
      session,
      acc,
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
    slotNames[entry.slotIdx]    = entry.slotName
    slotDefaults[entry.slotIdx] = entry.init
  }
  return { slot_count: slotCount, slot_names: slotNames, slot_defaults: slotDefaults }
}


/** @deprecated Slot mode is the only mode. */
export function slotModeEnabled(_session?: SessionState, _opt?: boolean): boolean {
  return true
}

// Silence unused-import warnings for re-export-style symbols.
void rawOffset
