/**
 * compile_session_slotted.ts — session compilation for `tropical_plan_5`.
 *
 * Each session instance compiles standalone (via `compileResolved`) and
 * lands in its own `InstanceFunction` entry. The JIT emits one
 * `alwaysinline` LLVM function per instance plus a single scheduler
 * function that loops the audio buffer, evaluates each instance's
 * alive expression (in the scheduler preamble), and conditionally
 * dispatches to the instance functions:
 *
 *     for each sample:
 *       <preamble: alive_i ← evaluate aliveInput_i; WriteSlot alive_slot_i>
 *       for i in 0..N: if (alive_i > 0.5) call instance_i(...)
 *       <DAC stitch: read graphOutput slots into mix temps>
 *
 * For an instance with no aliveInput, the preamble writes the literal
 * `1.0` to its alive slot. LLVM's GVN forwards the store-to-load
 * through the same kernel pass, the conditional folds to `if (true)`,
 * and the inlined body runs unconditionally — byte-equal to a unified
 * kernel.
 *
 * For an instance with an aliveInput ExprNode, the preamble compiles
 * that expression (via the shared `translateNode` machinery used for
 * input expressions) and writes the bool result. Asleep instances
 * skip their internal compute; their last-published output slot
 * persists. See `materialize_session.ts` for the matching interpreter
 * semantics.
 */

import { allocateOutputSlots, type SessionState } from '../session.js'
import type { FlatPlan, InstanceFunction, SchedulerFunction } from '../flat_plan.js'
import type { NInstr, ScalarType } from './emit_resolved.js'
import { compileResolved } from './compile_resolved.js'
import {
  computeInstanceTopoOrder, remapInstancePlan, emitDacStitch,
  type RemapContext, type PreambleEmitter,
  translateAliveExpr,
} from './compile_session_slotted_helpers.js'
import {
  inputNames, outputNames, outputPortTypes,
  rawInputDefaults,
} from '../program_types.js'

/** Compile the session into a `tropical_plan_5` `FlatPlan`. There is no
 *  fallback path — every shape supported by the per-instance compile
 *  must compile here. Shapes that don't yet work (e.g., type-level
 *  params, nested instance calls in input expressions) throw clear
 *  errors pointing at the relevant follow-up scope. */
export function compileSessionSlotted(session: SessionState): FlatPlan {
  return compileSessionSlottedPerInstance(session)
}

function compileSessionSlottedPerInstance(session: SessionState): FlatPlan {
  // Auto-allocate output slots for any instance the caller hasn't
  // pre-allocated. `add_instance` and `loadProgramAsSession` already
  // do this eagerly, but tests sometimes poke `instanceRegistry`
  // directly. Auto-allocation makes the boundary forgiving without
  // weakening the rest of the invariants.
  for (const [name, inst] of session.instanceRegistry) {
    if (inst.compiled !== undefined) allocateOutputSlots(session, name, inst.compiled)
  }

  const order = computeInstanceTopoOrder(session)

  // ── Unified accumulators across all instance functions + scheduler ───
  const allRegisterNames:  string[] = []
  const allRegisterTypes:  ScalarType[] = []
  const allStateInit:      (number | boolean)[] = []
  const allArraySlotSizes: number[] = []
  const allArraySlotNames: string[] = []
  const allRegisterTargets: number[] = []

  let regOffset       = 0  // unified temp count (NInstr.dst + reg-operand slot)
  let stateRegOffset  = 0  // unified state-register count
  let arraySlotOffset = 0  // unified array-slot count

  const instanceFunctions: InstanceFunction[] = []

  // Scheduler preamble — emitted once per sample, before any instance
  // fires. Holds:
  //   1. Alive expression WriteSlots (one per instance; literal 1.0
  //      for instances without aliveInput).
  //   2. DAC stitching reads (each graphOutput slot → mix temp).
  //
  // For (1), we accumulate temps into the unified register space
  // *after* every instance's body — preamble lives at the top of the
  // sample loop but its temps come last in temp-index space, so they
  // can't collide with instance regs.
  const schedulerPreamble: NInstr[] = []

  // First pass: compile each instance, building the instance_functions
  // entries and accumulating the unified state.
  for (const instName of order) {
    const inst = session.instanceRegistry.get(instName)
    if (inst === undefined) continue
    const compiled = inst.compiled
    if (compiled === undefined) {
      throw new Error(
        `compileSessionSlotted: instance '${instName}' has no Compiled type. ` +
        `(Did add_instance fail silently?)`,
      )
    }

    // Compile this instance standalone. Param handles stay empty here
    // — session-level params resolve to slot operands at remap time;
    // type-level inline params would surface as a `param` operand and
    // remap throws.
    const plan = compileResolved(compiled.prog, { paramHandles: new Map() })

    const inPortNames  = inputNames(compiled)
    const outPortNames = outputNames(compiled)
    const outPortTypes = outputPortTypes(compiled).map(scalarOf)
    const defaults     = rawInputDefaults(compiled)

    const ctx: RemapContext = {
      instanceName: instName,
      regOffset,
      stateRegOffset,
      arraySlotOffset,
      inputBindingFor: (portName) => {
        const expr = session.inputExprNodes.get(`${instName}:${portName}`)
        if (expr !== undefined) return { kind: 'wired', expr }
        const d = defaults[portName]
        const value = (typeof d === 'number' || typeof d === 'boolean') ? d : 0
        return { kind: 'literal', value }
      },
      outputSlotFor: (portName) => {
        const key = `${instName}.${portName}`
        const idx = session.outputSlotRegistry.get(key)
        if (idx === undefined) {
          throw new Error(
            `compileSessionSlotted: output slot for '${key}' not allocated. ` +
            `(Did add_instance populate outputSlotRegistry?)`,
          )
        }
        return idx
      },
      inputPortNames:    inPortNames,
      outputPortNames:   outPortNames,
      outputScalarTypes: outPortTypes,
    }

    const { preamble, body, writeSlots, tempsConsumed } = remapInstancePlan(plan, ctx, session)

    // Per-instance instructions: input-preamble (computes wired inputs
    // into temps) → body → writeSlots. All three are skipped when the
    // instance is asleep.
    const instanceInstructions: NInstr[] = [...preamble, ...body, ...writeSlots]

    const aliveSlotIdx = session.outputSlotRegistry.get(`${instName}.__alive__`)
    if (aliveSlotIdx === undefined) {
      throw new Error(
        `compileSessionSlotted: instance '${instName}' has no __alive__ slot. ` +
        `(allocateOutputSlots should have allocated one.)`,
      )
    }

    // Per-instance register_targets entries that are negative
    // (`-1`) mean "no scalar writeback for this slot" — used for
    // array-typed state registers, which manage their persistence
    // via in-place SetElement / strided Add writes elsewhere in the
    // instruction stream. Leave the sentinel intact rather than
    // shifting it into a (bogus, scalar) temp index.
    const shiftTempIdx = (t: number): number => (t < 0 ? t : t + regOffset)

    instanceFunctions.push({
      name:              `instance_${instName}`,
      instance_name:     instName,
      instructions:      instanceInstructions,
      register_offset:   regOffset,
      state_reg_offset:  stateRegOffset,
      array_slot_offset: arraySlotOffset,
      register_count:    plan.register_count + tempsConsumed,
      register_targets:  plan.register_targets.map(shiftTempIdx),
      alive_slot_index:  aliveSlotIdx,
    })

    // Accumulate state contributions into the unified plan-level arrays.
    for (const n of plan.register_names) allRegisterNames.push(`${instName}.${n}`)
    allRegisterTypes.push(...plan.register_types)
    for (const v of plan.state_init) allStateInit.push(v as number | boolean)
    allArraySlotSizes.push(...plan.array_slot_sizes)
    for (const n of plan.array_slot_names) allArraySlotNames.push(`${instName}.${n}`)
    for (const t of plan.register_targets) {
      allRegisterTargets.push(shiftTempIdx(t))
    }

    regOffset       += plan.register_count + tempsConsumed
    stateRegOffset  += plan.state_init.length
    arraySlotOffset += plan.array_slot_count
  }

  // Second pass: emit alive WriteSlot instructions into the scheduler
  // preamble. Writing every sample (even for default-alive instances)
  // lets LLVM's GVN forward the constant 1.0 through the same kernel
  // pass — the load-then-cmp-then-br chain folds to a pure inline.
  //
  // For instances with aliveInput, translateAliveExpr compiles the
  // expression into preamble temps and returns the final operand to
  // WriteSlot.
  let preambleNextTemp = regOffset
  const aliveEmitter: PreambleEmitter = {
    instrs: schedulerPreamble,
    allocTemp: () => preambleNextTemp++,
  }

  for (const instName of order) {
    const inst = session.instanceRegistry.get(instName)
    if (inst === undefined) continue
    const aliveSlotIdx = session.outputSlotRegistry.get(`${instName}.__alive__`)
    if (aliveSlotIdx === undefined) continue

    let aliveOperand: import('./emit_resolved.js').NOperand
    if (inst.aliveInput === undefined) {
      // Default: always alive. Literal 1.0.
      aliveOperand = { kind: 'const', val: 1, scalar_type: 'float' }
    } else {
      aliveOperand = translateAliveExpr(
        inst.aliveInput,
        session,
        aliveEmitter,
        `${instName}.__alive__`,
      )
    }
    schedulerPreamble.push({
      tag:        'WriteSlot',
      dst:        aliveSlotIdx,
      args:       [aliveOperand],
      loop_count: 1,
      strides:    [],
      result_type: 'float',
    })
  }

  regOffset = preambleNextTemp

  // DAC stitching reads each graphOutput slot AFTER all instances
  // have dispatched, so the reads see the current sample's WriteSlot
  // values (alive instances have written this sample; asleep
  // instances retain the previous WriteSlot).
  const dac = emitDacStitch(session, regOffset)
  regOffset += dac.tempCount

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
    array_slot_count:  arraySlotOffset,
    array_slot_sizes:  allArraySlotSizes,
    register_count:    regOffset,
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
    // explicitly silenced. Other output slots start at 0.
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
