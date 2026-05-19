/**
 * compile_session_slotted.ts — session compilation for `tropical_plan_5`.
 *
 * Each session instance compiles standalone (via `compileResolved`) and
 * lands in its own `InstanceFunction` entry. The JIT emits a single
 * LLVM kernel that loops the audio buffer; per sample it evaluates
 * the alive expression of each instance (in the scheduler preamble),
 * conditionally runs the instance's body + writebacks, then reads
 * graphOutput slots in the postamble:
 *
 *     for each sample:
 *       <preamble: alive_i ← WriteSlot of evaluated expression>
 *       for i in 0..N: if (slots[alive_slot_i] > 0.5) { body_i; writebacks_i }
 *       <postamble: DAC stitch — read graphOutput slots into mix temps>
 *
 * Default-alive instances see a `WriteSlot const 1.0` in the
 * preamble. LLVM's GVN forwards the store-to-load through the same
 * kernel pass, folds the conditional to `if (true)`, and jump-
 * threading eliminates the branch — the body inlines
 * unconditionally, matching a unified kernel byte-for-byte.
 *
 * Asleep instances skip their internal compute; their last-published
 * output slot persists. `materialize_session.ts` (interpreter oracle)
 * realizes the same I/O semantics structurally via `select(alive,
 * raw, fallback)` wraps.
 *
 * ## Branded discipline
 *
 * The unified accumulators carry branded counts/offsets per
 * namespace (`TempIdx`, `StateRegIdx`, `ArraySlotIdx`,
 * `ModuleSlotIdx`). Cross-namespace arithmetic is a compile error —
 * the literal shape of the Phaser bug we hit during PR review.
 */

import { allocateOutputSlots, type SessionState } from '../session.js'
import type { FlatPlan, InstanceFunction, SchedulerFunction } from '../flat_plan.js'
import type { NInstr, NOperand } from './emit_resolved.js'
import { instrWriteSlot, opConst } from './emit_resolved.js'
import {
  computeInstanceTopoOrder, emitDacStitch,
  type PreambleEmitter,
  translateAliveExpr,
  translateNode,
} from './compile_session_slotted_helpers.js'
import {
  type ModuleSlotIdx,
  tempIdx, moduleSlotIdx, tempOffset,
  rawOffset,
} from './slot_indices.js'
import { partitionKernel, makeAccumulators } from './partition_recursive.js'

/** Compile the session into a `tropical_plan_5` `FlatPlan`. */
export function compileSessionSlotted(session: SessionState): FlatPlan {
  return compileSessionSlottedPerInstance(session)
}

function compileSessionSlottedPerInstance(session: SessionState): FlatPlan {
  // Auto-allocate output slots for any instance whose owner hasn't
  // pre-allocated. `add_instance` / `loadProgramAsSession` already do
  // this eagerly; tests sometimes poke `instanceRegistry` directly.
  for (const [name, inst] of session.instanceRegistry) {
    if (inst.compiled !== undefined) allocateOutputSlots(session, name, inst.compiled)
  }

  const order = computeInstanceTopoOrder(session)

  // Recursive partition: for each top-level session instance, walk its
  // ResolvedProgram tree and emit a kernel per InstanceDecl at every
  // level. Slot allocations, register/state/array offsets, alive
  // preamble operands are all threaded through `acc`.
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
      /* aliveInput     */ inst.aliveInput,
      /* inputBindingFor*/ (portName) => {
        const expr = session.inputExprNodes.get(`${instName}:${portName}`)
        if (expr !== undefined) return { kind: 'wired', expr }
        return undefined
      },
      /* defaults       */ (() => {
        // rawInputDefaults is imported from program_types.js — but we
        // need the Compiled here. Use lazy access via property.
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

  // ── Scheduler preamble: emit alive WriteSlots for every kernel in
  //    the tree. Writing every sample lets GVN forward the constant 1.0
  //    in the default case so the dispatch conditional folds to inline.
  //    Done in a second pass so nested kernels' alive emissions are
  //    included; acc.alivePreambleOps was populated during partition.
  const schedulerPreamble: NInstr[] = []
  let preambleNextTempRaw = acc.nextRegRaw
  const aliveEmitter: PreambleEmitter = {
    instrs: schedulerPreamble,
    allocTemp: () => {
      const slot = tempIdx(preambleNextTempRaw)
      preambleNextTempRaw += 1
      return slot
    },
  }

  for (const op of acc.alivePreambleOps) {
    const aliveOperand: NOperand = op.expr === undefined
      ? opConst(1, 'float')
      : translateAliveExpr(
          op.expr, session, aliveEmitter,
          `__alive__@${rawOffset(op.aliveSlot)}`,
        )
    schedulerPreamble.push(instrWriteSlot(op.aliveSlot, aliveOperand, 'float'))
  }

  // DAC stitching reads each graphOutput slot AFTER all instances
  // dispatch — observes the current sample's WriteSlots.
  const dac = emitDacStitch(session, tempOffset(preambleNextTempRaw))
  const dacEndRaw = preambleNextTempRaw + dac.tempCount

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
    // __alive__ slots default to 1.0 so an instance is alive until
    // explicitly silenced.
    if (name.endsWith('.__alive__')) slotDefaults[idx] = 1
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
