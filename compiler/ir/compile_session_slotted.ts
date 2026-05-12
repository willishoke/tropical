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
import type { FlatPlan, InstanceFunction, SchedulerFunction, RegTarget } from '../flat_plan.js'
import { TempTarget, ArrayManagedTarget } from '../flat_plan.js'
import type { NInstr, ScalarType, NOperand } from './emit_resolved.js'
import { instrWriteSlot, opConst } from './emit_resolved.js'
import { compileResolved } from './compile_resolved.js'
import {
  computeInstanceTopoOrder, remapInstancePlan, emitDacStitch,
  type RemapContext, type PreambleEmitter,
  translateAliveExpr,
} from './compile_session_slotted_helpers.js'
import {
  type TempIdx, type ModuleSlotIdx,
  tempIdx, moduleSlotIdx,
  tempOffset, stateRegOffset, arraySlotOffset,
  rawOffset,
} from './slot_indices.js'
import {
  inputNames, outputNames, outputPortTypes,
  rawInputDefaults,
} from '../program_types.js'

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

  // ── Unified accumulators across all instance functions + scheduler ───
  // Plain counts (number) for things that are just "how many in this
  // namespace"; the offsets we hand to RemapContext are branded so
  // cross-namespace arithmetic can't compile.
  const allRegisterNames:   string[]            = []
  const allRegisterTypes:   ScalarType[]        = []
  const allStateInit:       (number | boolean)[] = []
  const allArraySlotSizes:  number[]            = []
  const allArraySlotNames:  string[]            = []

  let nextRegRaw    = 0
  let nextStateRaw  = 0
  let nextArrayRaw  = 0

  const instanceFunctions: InstanceFunction[] = []
  const schedulerPreamble: NInstr[] = []

  // First pass: compile each instance, building instance_functions
  // entries and accumulating unified state.
  for (const instName of order) {
    const inst = session.instanceRegistry.get(instName)
    if (inst === undefined) continue
    const compiled = inst.compiled
    if (compiled === undefined) {
      throw new Error(
        `compileSessionSlotted: instance '${instName}' has no Compiled type.`,
      )
    }

    const plan = compileResolved(compiled.prog, { paramHandles: new Map() })

    const inPortNames  = inputNames(compiled)
    const outPortNames = outputNames(compiled)
    const outPortTypes = outputPortTypes(compiled).map(scalarOf)
    const defaults     = rawInputDefaults(compiled)

    const ctx: RemapContext = {
      instanceName: instName,
      regOffset:       tempOffset(nextRegRaw),
      stateRegOffset:  stateRegOffset(nextStateRaw),
      arraySlotOffset: arraySlotOffset(nextArrayRaw),
      inputBindingFor: (portName) => {
        const expr = session.inputExprNodes.get(`${instName}:${portName}`)
        if (expr !== undefined) return { kind: 'wired', expr }
        const d = defaults[portName]
        const value = (typeof d === 'number' || typeof d === 'boolean') ? d : 0
        return { kind: 'literal', value }
      },
      outputSlotFor: (portName) => moduleSlotIdx(requireOutputSlot(session, instName, portName)),
      inputPortNames:    inPortNames,
      outputPortNames:   outPortNames,
      outputScalarTypes: outPortTypes,
    }

    const { preamble, body, writeSlots, tempsConsumed } = remapInstancePlan(plan, ctx, session)
    const instanceInstructions: NInstr[] = [...preamble, ...body, ...writeSlots]

    const aliveSlot = moduleSlotIdx(
      requireOutputSlot(session, instName, '__alive__'),
    )

    // Shift per-instance register_targets into the unified temp
    // space. ArrayManagedTarget passes through structurally — no
    // arithmetic can corrupt it, exactly the property the `-1`
    // sentinel didn't provide.
    const shiftedTargets: RegTarget[] = plan.register_targets.map(t => {
      if (t.kind === 'arrayManaged') return ArrayManagedTarget
      return TempTarget(tempIdx(t.slot + nextRegRaw))
    })

    instanceFunctions.push({
      name:              `instance_${instName}`,
      instance_name:     instName,
      instructions:      instanceInstructions,
      register_offset:   tempOffset(nextRegRaw),
      state_reg_offset:  stateRegOffset(nextStateRaw),
      array_slot_offset: arraySlotOffset(nextArrayRaw),
      register_count:    plan.register_count + tempsConsumed,
      register_targets:  shiftedTargets,
      alive_slot_index:  aliveSlot,
    })

    // Accumulate unified state. These arrays mirror per-instance
    // contributions in instance order; their indices line up with
    // (state_reg_offset + i), (array_slot_offset + i), etc.
    for (const n of plan.register_names) allRegisterNames.push(`${instName}.${n}`)
    allRegisterTypes.push(...plan.register_types)
    for (const v of plan.state_init) allStateInit.push(v as number | boolean)
    allArraySlotSizes.push(...plan.array_slot_sizes)
    for (const n of plan.array_slot_names) allArraySlotNames.push(`${instName}.${n}`)

    nextRegRaw   += plan.register_count + tempsConsumed
    nextStateRaw += plan.state_init.length
    nextArrayRaw += plan.array_slot_count
  }

  // Second pass: emit alive WriteSlots into the scheduler preamble.
  // Writing every sample lets GVN forward the constant 1.0 in the
  // default case so the dispatch conditional folds to inline.
  let preambleNextTempRaw = nextRegRaw
  const aliveEmitter: PreambleEmitter = {
    instrs: schedulerPreamble,
    allocTemp: () => {
      const slot = tempIdx(preambleNextTempRaw)
      preambleNextTempRaw += 1
      return slot
    },
  }

  for (const instName of order) {
    const inst = session.instanceRegistry.get(instName)
    if (inst === undefined) continue
    const aliveSlotRaw = session.outputSlotRegistry.get(`${instName}.__alive__`)
    if (aliveSlotRaw === undefined) continue
    const aliveSlot = moduleSlotIdx(aliveSlotRaw)

    const aliveOperand: NOperand = inst.aliveInput === undefined
      ? opConst(1, 'float')
      : translateAliveExpr(
          inst.aliveInput, session, aliveEmitter,
          `${instName}.__alive__`,
        )
    schedulerPreamble.push(instrWriteSlot(aliveSlot, aliveOperand, 'float'))
  }

  // DAC stitching reads each graphOutput slot AFTER all instances
  // dispatch — observes the current sample's WriteSlots.
  const dac = emitDacStitch(session, tempOffset(preambleNextTempRaw))
  const dacEndRaw = preambleNextTempRaw + dac.tempCount

  const schedulerFunction: SchedulerFunction = {
    preamble:       schedulerPreamble,
    postamble:      dac.instructions,
    output_targets: dac.outputTargets,
    outputs:        dac.outputs,
  }

  return {
    schema: 'tropical_plan_5',
    config: { sampleRate: 44100 },
    state_init:        allStateInit,
    register_names:    allRegisterNames,
    register_types:    allRegisterTypes,
    array_slot_names:  allArraySlotNames,
    array_slot_count:  nextArrayRaw,
    array_slot_sizes:  allArraySlotSizes,
    register_count:    dacEndRaw,
    instance_functions: instanceFunctions,
    scheduler_function: schedulerFunction,
    ...buildSlotMetadata(session),
  }
}

/** Look up an instance's output slot, throwing with a clear message
 *  if it's missing. Centralised so error wording stays consistent. */
function requireOutputSlot(session: SessionState, instName: string, portName: string): number {
  const key = `${instName}.${portName}`
  const idx = session.outputSlotRegistry.get(key)
  if (idx === undefined) {
    throw new Error(
      `compileSessionSlotted: output slot for '${key}' not allocated.`,
    )
  }
  return idx
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
  return { slot_count: slotCount, slot_names: slotNames, slot_defaults: slotDefaults }
}

/** Reduce a PortType to a ScalarType for emission. */
function scalarOf(
  t: { kind?: string; scalar?: ScalarType; alias?: { base: ScalarType }; element?: ScalarType | { base: ScalarType } } | undefined,
): ScalarType {
  if (t === undefined) return 'float'
  if (t.kind === 'scalar' && t.scalar !== undefined) return t.scalar
  if (t.kind === 'alias' && t.alias !== undefined) return t.alias.base
  if (t.kind === 'array' && t.element !== undefined) {
    return typeof t.element === 'string' ? (t.element as ScalarType) : t.element.base
  }
  return 'float'
}

/** @deprecated Slot mode is the only mode. */
export function slotModeEnabled(_session?: SessionState, _opt?: boolean): boolean {
  return true
}

// Silence unused-import warnings for re-export-style symbols.
void rawOffset
