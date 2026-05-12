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

import type { ResolvedProgram, ResolvedExpr, OutputDecl, RegDecl, DelayDecl, ParamDecl } from './nodes.js'
import type { PerInstancePlan } from '../flat_plan.js'
import { buildSlotMaps, type SlotMaps } from './slots.js'
import { emitResolvedProgram, type EmitSlots, type ScalarType } from './emit_resolved.js'

/** Param-handle bindings for FFI param/trigger decls embedded in a
 *  program type's body. Empty on the per-instance path (session-level
 *  params resolve through `slot` operands during input remapping;
 *  type-level params survive into emission and surface here). */
export interface CompileResolvedContext {
  paramHandles?: Map<ParamDecl, { ptr: string }>
}

/** Compile a post-strata `ResolvedProgram` to a `PerInstancePlan`. */
export function compileResolved(prog: ResolvedProgram, ctx: CompileResolvedContext = {}): PerInstancePlan {
  const slots = buildSlotMaps(prog)

  // Any surviving `instanceDecl` means `inlineInstances` didn't run.
  if (slots.instanceDecls.length > 0) {
    throw new Error(
      `compileResolved: program '${prog.name}' has ${slots.instanceDecls.length} surviving instanceDecl entries; ` +
      `compileResolved expects post-strata (post-inlineInstances) input.`,
    )
  }

  // ── Output expressions ──
  const outputExprByDecl = new Map<OutputDecl, ResolvedExpr>()
  for (const a of prog.body.assigns) {
    if (a.op !== 'outputAssign') continue
    if ('op' in a.target && a.target.op === 'outputDecl') {
      outputExprByDecl.set(a.target, a.expr)
    }
  }
  const outputExprs: ResolvedExpr[] = prog.ports.outputs.map(out => {
    const expr = outputExprByDecl.get(out)
    if (expr === undefined) {
      throw new Error(`compileResolved: program '${prog.name}' output '${out.name}' has no outputAssign.`)
    }
    return expr
  })

  // ── Register update expressions ──
  const regUpdateByDecl   = new Map<RegDecl, ResolvedExpr>()
  const delayUpdateByDecl = new Map<DelayDecl, ResolvedExpr>()
  for (const a of prog.body.assigns) {
    if (a.op !== 'nextUpdate') continue
    if (a.target.op === 'regDecl')   regUpdateByDecl.set(a.target, a.expr)
    if (a.target.op === 'delayDecl') delayUpdateByDecl.set(a.target, a.expr)
  }

  const registerExprs: (ResolvedExpr | null)[] = []
  const stateInit:     (number | boolean | number[])[] = []
  const registerNames: string[] = []
  const registerTypes: ScalarType[] = []

  for (const d of slots.regDecls) {
    registerNames.push(d.name)
    registerTypes.push(regScalarType(d))
    stateInit.push(regInit(d))
    const u = regUpdateByDecl.get(d)
    registerExprs.push(u === undefined ? null : u)
  }

  for (const d of slots.delayDecls) {
    registerNames.push(d.name)
    registerTypes.push(delayScalarType(d))
    stateInit.push(delayInit(d))
    const u = delayUpdateByDecl.get(d) ?? d.update
    registerExprs.push(u)
  }

  const inputPortTypes: ScalarType[] = slots.inputDecls.map(d => {
    if (d.type === undefined) return 'float'
    if (d.type.kind === 'scalar') return d.type.scalar
    if (d.type.kind === 'alias')  return d.type.alias.base
    if (typeof d.type.element === 'string') return d.type.element
    return d.type.element.base
  })

  const emitSlots: EmitSlots = {
    inputs:       slots.inputs,
    regs:         slots.regs,
    delays:       slots.delays,
    regCount:     slots.regDecls.length,
    paramHandles: ctx.paramHandles ?? new Map(),
  }

  const program = emitResolvedProgram({
    outputExprs,
    registerExprs,
    stateInit,
    stateRegTypes: registerTypes,
    inputPortTypes,
    slots: emitSlots,
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

function delayScalarType(_d: DelayDecl): ScalarType {
  return 'float'
}

function delayInit(d: DelayDecl): number {
  if (typeof d.init === 'number') return d.init
  if (typeof d.init === 'boolean') return d.init ? 1 : 0
  return 0
}

void buildSlotMaps
export type { SlotMaps }
