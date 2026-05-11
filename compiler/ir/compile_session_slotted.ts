/**
 * compile_session_slotted.ts — slot-mode session compilation (M4–M9).
 *
 * Two paths coexist during M9 development:
 *
 *   - **Legacy-wrap path** (default through M9a): runs `compileSessionLegacy`
 *     and attaches slot allocation metadata. Audio output is byte-identical
 *     to the pre-slot-model path. The slot fields are honored for
 *     control-plane writes (set_slot) but the instruction stream still
 *     uses legacy `param` / `input` / `reg` operands.
 *
 *   - **Per-instance path** (M9a opt-in via `TROPICAL_SLOT_OPS=1`): each
 *     session instance is compiled standalone via `compileResolved`, its
 *     operands are remapped into the unified register/slot space, and
 *     WriteSlot instructions publish each output to its allocated slot.
 *     A DAC stitching phase reads graphOutput source slots into temps
 *     for the kernel's existing mix-bus mechanism.
 *
 * The per-instance path replaces the legacy path's `materializeSession +
 * inlineInstances` flattening with explicit slot boundaries — the
 * architectural goal of the slot model. M9a covers single-source ref
 * chains; M9b–M9d incrementally lift the limitations (params, arbitrary
 * input expressions, fan-in, arrays, sums). M9e deletes the legacy path
 * once equivalence is verified across the full stdlib fixture set.
 */

import type { SessionState } from '../session.js'
import type { FlatPlan } from '../flat_plan.js'
import type { ScalarType } from './emit_resolved.js'
import { compileResolved } from './compile_resolved.js'
import { compileSessionLegacy } from './compile_session.js'
import {
  computeInstanceTopoOrder, remapInstancePlan, emitDacStitch,
  type RemapContext,
} from './compile_session_slotted_helpers.js'
import {
  inputNames, outputNames, inputPortTypes, outputPortTypes,
  rawInputDefaults,
} from '../program_types.js'

/** Compile the current session in slot mode. Dispatches between the
 *  legacy-wrap and the per-instance paths based on
 *  `TROPICAL_SLOT_OPS`. Both paths produce a FlatPlan with slot
 *  allocation metadata; the per-instance path additionally emits
 *  `slot` operands and `WriteSlot` instructions in the instruction
 *  stream. */
export function compileSessionSlotted(session: SessionState): FlatPlan {
  const useOps = useSlotOps()
  return useOps
    ? compileSessionSlottedPerInstance(session)
    : compileSessionSlottedMetadataOnly(session)
}

/** M4–M8 path: wrap the legacy plan with slot metadata. Untouched
 *  instruction stream. */
function compileSessionSlottedMetadataOnly(session: SessionState): FlatPlan {
  const legacy = compileSessionLegacy(session)
  return { ...legacy, ...buildSlotMetadata(session) }
}

/** M9a path: per-instance compileResolved + operand remapping + DAC stitch.
 *  Throws clear errors on shapes not yet supported (fan-in, arbitrary
 *  input expressions, params/triggers in input expressions). */
function compileSessionSlottedPerInstance(session: SessionState): FlatPlan {
  const order = computeInstanceTopoOrder(session)

  // Accumulators for the unified plan
  const allInstructions: import('./emit_resolved.js').NInstr[] = []
  const allRegisterNames: string[] = []
  const allRegisterTypes: ScalarType[] = []
  const allStateInit: (number | boolean)[] = []
  const allArraySlotSizes: number[] = []
  const allArraySlotNames: string[] = []
  const allRegisterTargets: number[] = []

  let regOffset = 0          // cumulative temp count (NInstr.dst + reg-operand slot)
  let stateRegOffset = 0     // cumulative state-register count (state_reg slots + register_targets index space)
  let arraySlotOffset = 0    // cumulative array-slot count (array_reg slots)

  for (const instName of order) {
    const inst = session.instanceRegistry.get(instName)
    if (inst === undefined) continue
    const compiled = inst.compiled
    if (compiled === undefined) {
      throw new Error(
        `compileSessionSlotted (M9a): instance '${instName}' has no Compiled type. ` +
        `(Did add_instance fail silently?)`,
      )
    }

    // Compile this instance standalone. M9a doesn't yet support param
    // handles in input expressions, so paramHandles stays empty — any
    // {op:'param'} ExprNode would surface as a 'param' operand and the
    // remap throws.
    const plan = compileResolved(compiled.prog, { paramHandles: new Map() })

    // Build remap context for this instance
    const inPortNames  = inputNames(compiled)
    const outPortNames = outputNames(compiled)
    const inPortTypes  = inputPortTypes(compiled).map(scalarOf)
    const outPortTypes = outputPortTypes(compiled).map(scalarOf)
    const defaults     = rawInputDefaults(compiled)
    const inputDefaults: Array<number | boolean | undefined> = inPortNames.map(name => {
      const d = defaults[name]
      if (typeof d === 'number' || typeof d === 'boolean') return d
      return undefined
    })

    const ctx: RemapContext = {
      instanceName: instName,
      regOffset,
      stateRegOffset,
      arraySlotOffset,
      inputExprFor: (portName) =>
        session.inputExprNodes.get(`${instName}:${portName}`),
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
      inputPortNames: inPortNames,
      inputDefaults,
      outputPortNames: outPortNames,
      outputScalarTypes: outPortTypes,
    }

    const { preamble, body, writeSlots, tempsConsumed } = remapInstancePlan(plan, ctx, session)
    // Order: preamble (compute input expressions) → body → writeSlots (publish outputs)
    allInstructions.push(...preamble, ...body, ...writeSlots)

    // Accumulate the per-instance state contributions into the unified plan
    for (const n of plan.register_names) allRegisterNames.push(`${instName}.${n}`)
    allRegisterTypes.push(...plan.register_types)
    for (const v of plan.state_init) allStateInit.push(v as number | boolean)
    allArraySlotSizes.push(...plan.array_slot_sizes)
    for (const n of plan.array_slot_names) allArraySlotNames.push(`${instName}.${n}`)
    for (const t of plan.register_targets) {
      // register_targets[i] is the temp index whose value feeds state-reg i;
      // shift the temp index by this instance's regOffset.
      allRegisterTargets.push(t + regOffset)
    }

    regOffset += plan.register_count + tempsConsumed
    stateRegOffset += plan.state_init.length
    arraySlotOffset += plan.array_slot_count
  }

  // DAC stitching: read each graphOutput source slot into a fresh temp
  // and expose it through output_targets + outputs.
  const dac = emitDacStitch(session, regOffset)
  allInstructions.push(...dac.instructions)
  regOffset += dac.tempCount

  return {
    schema: 'tropical_plan_4',
    config: { sampleRate: 44100 },
    state_init:       allStateInit,
    register_names:   allRegisterNames,
    register_types:   allRegisterTypes,
    array_slot_names: allArraySlotNames,
    outputs:          dac.outputs,
    instructions:     allInstructions,
    register_count:   regOffset,
    array_slot_count: arraySlotOffset,
    array_slot_sizes: allArraySlotSizes,
    output_targets:   dac.outputTargets,
    register_targets: allRegisterTargets,
    ...buildSlotMetadata(session),
  }
}

/** Read-time check for the M9 per-instance opt-in. */
function useSlotOps(): boolean {
  const env = process.env.TROPICAL_SLOT_OPS
  return env !== undefined && env !== '' && env !== '0' && env !== 'false'
}

/** Compute slot allocation metadata from the session registries. Shared
 *  between the metadata-only and per-instance paths. */
function buildSlotMetadata(session: SessionState): {
  slot_count: number; slot_names: string[]; slot_defaults: number[]
} {
  const slotCount    = session.slotCount
  const slotNames    = new Array<string>(slotCount).fill('')
  const slotDefaults = new Array<number>(slotCount).fill(0)
  for (const [name, idx] of session.outputSlotRegistry) slotNames[idx] = name
  for (const [name, idx] of session.paramSlotRegistry) {
    slotNames[idx] = `param:${name}`
    const param = session.paramRegistry.get(name)
    if (param !== undefined) slotDefaults[idx] = param.value
  }
  return { slot_count: slotCount, slot_names: slotNames, slot_defaults: slotDefaults }
}

/** Reduce a PortType to a ScalarType for emission. Mirrors the small
 *  shim in `compile_resolved.ts` for input port type extraction. */
function scalarOf(t: { kind?: string; scalar?: ScalarType; alias?: { base: ScalarType }; element?: ScalarType | { base: ScalarType } } | undefined): ScalarType {
  if (t === undefined) return 'float'
  if (t.kind === 'scalar' && t.scalar !== undefined) return t.scalar
  if (t.kind === 'alias' && t.alias !== undefined) return t.alias.base
  if (t.kind === 'array' && t.element !== undefined) {
    return typeof t.element === 'string' ? (t.element as ScalarType) : t.element.base
  }
  return 'float'
}

/** @deprecated As of M8, slot mode is the default for `compileSession`.
 *  Kept as a stub for any external callers. */
export function slotModeEnabled(_session?: SessionState, _opt?: boolean): boolean {
  return true
}
