/**
 * program.ts — tropical_program_2 (ProgramNode) types, loading, saving, and stdlib.
 *
 * A program is an ExprNode of op `program`: a named set of ports wrapping a
 * body `block` that declares regs, delays, instances, and nested programs,
 * and carries output assignments plus per-tick reg/delay updates.
 */

import type { ExprNode } from './expr.js'
import { validateExpr } from './expr.js'
import type { TypeDefJSON, SessionState } from './session.js'
import { resolveProgramType, allocateParamSlot, allocateOutputSlots, reconstructWireDelays } from './session.js'
import { applyFlatPlan } from './apply_plan.js'
import { Param } from './runtime/param.js'
import {
  type Compiled,
  instantiate,
  inputNames, outputNames,
  inputPortType, outputPortType,
  inputIndex, outputIndex,
  rawInputDefaults,
} from './program_types.js'
import { exprDependencies, reachableInstances, buildDependencyGraph, topologicalSort } from './compiler.js'
import type { RawTypeArgs } from './specialize.js'
import { Float, portTypeEqual } from './ir/port_type.js'
import {
  wireKey, parseWireKey, portRef, rawName,
  instanceName as toInstanceName, portName as toPortName,
  type WireKey,
} from './ir/branded_names.js'
import { raiseProgram } from './parse/raise.js'
import { elaborate, type ExternalProgramResolver } from './ir/elaborator.js'
import { programTypeFromResolved } from './ir/strata.js'
import type { ResolvedProgram, PortType } from './ir/nodes.js'
import { relinkProgramRegistry } from './ir/decl_tables.js'

// ─────────────────────────────────────────────────────────────
// Program schema
// ─────────────────────────────────────────────────────────────

/** Compile-time shape dimension: a concrete int or a reference to an
 *  outer type parameter to be substituted during specialization. */
export type ShapeDim = number | { op: 'typeParam'; name: string }

/** Structured port/reg type declaration. Scalars and aliases are bare strings;
 *  arrays use the structured form so type_param refs can appear in shapes. */
export type PortTypeDecl = string | { kind: 'array'; element: string; shape: ShapeDim[] }

// ─────────────────────────────────────────────────────────────
// ProgramNode — typed view of the tropical_program_2 ExprNode shape
// ─────────────────────────────────────────────────────────────
//
// A program is an ExprNode of op `program`. The types below are a typed
// view of that JSON — the runtime value is a plain object; the TypeScript
// types only constrain the fields we care about.

/** Port declaration: scalar/alias name or `{name, type?, default?}`. */
export interface ProgramPortSpec {
  name: string
  type?: PortTypeDecl
  default?: ExprNode
}

export interface ProgramPorts {
  inputs?: Array<string | ProgramPortSpec>
  outputs?: Array<string | ProgramPortSpec>
  type_defs?: TypeDefJSON[]
}

/** Body block: ordered decls + assigns. `value` is unused (assigns-canonical). */
export interface BlockNode {
  op: 'block'
  decls?: ExprNode[]
  assigns?: ExprNode[]
  value?: ExprNode | null
}

/** Program — the unified IR. Same shape whether nested (inside `program_decl`)
 *  or at the top level of a `tropical_program_2` file. */
export interface ProgramNode {
  op: 'program'
  name: string
  type_params?: Record<string, { type: 'int'; default?: number }>
  sample_rate?: number
  breaks_cycles?: boolean
  ports?: ProgramPorts
  body: BlockNode
}

/** Top-level session metadata carried alongside a root program in a v2 file.
 *  Both fields are deprecated: post-A3 saves emit params via paramDecl body
 *  decls, post-A4 saves emit audio_outputs via outputAssign(name='dac.out')
 *  body assigns. The fields remain in the type for the deprecated read path
 *  used by pre-A4 patches. (Source-level `config` was removed in A5: sample
 *  rate and buffer length are runtime/host concerns, not source concerns.) */
export interface ProgramTopLevel {
  params?: Array<{ name: string; value?: number; time_const?: number; type?: 'param' | 'trigger' }>
  audio_outputs?: Array<
    | { instance: string; output: string | number }
    | { expr: ExprNode }
  >
}

/** On-disk `tropical_program_2` file shape: a program plus session metadata. */
export interface ProgramFile extends ProgramTopLevel {
  schema: 'tropical_program_2'
  name: string
  type_params?: ProgramNode['type_params']
  sample_rate?: number
  breaks_cycles?: boolean
  ports?: ProgramPorts
  body: BlockNode
}

// ─────────────────────────────────────────────────────────────
// Program loading
// ─────────────────────────────────────────────────────────────

/** Iterate instance_decl entries in a ProgramNode's body. */
export function* instanceDecls(prog: ProgramNode): Iterable<{
  name: string
  program: string
  inputs?: Record<string, ExprNode>
  type_args?: Record<string, number | ExprNode>
}> {
  for (const d of prog.body?.decls ?? []) {
    if (typeof d !== 'object' || d === null || Array.isArray(d)) continue
    const obj = d as Record<string, unknown>
    if (obj.op !== 'instanceDecl') continue
    yield {
      name: obj.name as string,
      program: obj.program as string,
      inputs: obj.inputs as Record<string, ExprNode> | undefined,
      type_args: obj.type_args as Record<string, number | ExprNode> | undefined,
    }
  }
}

/** Iterate program_decl entries in a ProgramNode's body. */
function* programDecls(prog: ProgramNode): Iterable<{ name: string; program: ProgramNode }> {
  for (const d of prog.body?.decls ?? []) {
    if (typeof d !== 'object' || d === null || Array.isArray(d)) continue
    const obj = d as Record<string, unknown>
    if (obj.op !== 'programDecl') continue
    yield { name: obj.name as string, program: obj.program as ProgramNode }
  }
}

/** Iterate paramDecl entries in a ProgramNode's body. Each entry declares a
 *  control-rate param that the program reads via paramExpr. Only the top-level
 *  program's paramDecls populate session.paramRegistry; nested-program
 *  paramDecls (inside `programDecl` entries) are not auto-hoisted. */
export function* paramDecls(prog: ProgramNode): Iterable<{
  name: string
  value?: number
  time_const?: number
  type: 'param' | 'trigger'
}> {
  for (const d of prog.body?.decls ?? []) {
    if (typeof d !== 'object' || d === null || Array.isArray(d)) continue
    const obj = d as Record<string, unknown>
    if (obj.op !== 'paramDecl') continue
    yield {
      name: obj.name as string,
      value: typeof obj.value === 'number' ? obj.value : undefined,
      time_const: typeof obj.time_const === 'number' ? obj.time_const : undefined,
      type: obj.type === 'trigger' ? 'trigger' : 'param',
    }
  }
}

/** Iterate `outputAssign` entries whose `name` is "dac.out" — the audio output
 *  boundary leaf. These are body-resident wires to the DAC; multiple entries
 *  accumulate (mix-bus). The expression must be a ref-shaped node; arbitrary
 *  expressions defer to a later phase. */
export function* dacWires(prog: ProgramNode): Iterable<{ expr: ExprNode }> {
  for (const a of prog.body?.assigns ?? []) {
    if (typeof a !== 'object' || a === null || Array.isArray(a)) continue
    const obj = a as Record<string, unknown>
    if (obj.op !== 'outputAssign' || obj.name !== 'dac.out') continue
    yield { expr: obj.expr as ExprNode }
  }
}

type ParamSpec = { name: string; value?: number; time_const?: number; type: 'param' | 'trigger' }

/** Merge param sources: body paramDecls canonical, topLevel.params is a
 *  deprecated fallback for pre-A3 patches. Body wins on name collisions; a
 *  one-time deprecation warning fires if topLevel.params is non-empty. */
function mergeParamSources(prog: ProgramNode, topLevel: ProgramTopLevel): ParamSpec[] {
  const out: ParamSpec[] = []
  const seen = new Set<string>()
  for (const p of paramDecls(prog)) {
    out.push(p)
    seen.add(p.name)
  }
  if (topLevel.params && topLevel.params.length > 0) {
    if (!_topLevelParamsWarned) {
      console.warn(
        'tropical: file-root `params` is deprecated; declare params via paramDecl entries in the program body. Pre-A3 patches load with a fallback.',
      )
      _topLevelParamsWarned = true
    }
    for (const p of topLevel.params) {
      if (seen.has(p.name)) continue
      out.push({
        name: p.name,
        value: p.value,
        time_const: p.time_const,
        type: p.type === 'trigger' ? 'trigger' : 'param',
      })
      seen.add(p.name)
    }
  }
  return out
}
let _topLevelParamsWarned = false

/** Populate session.paramRegistry from a ParamSpec list. Legacy trigger
 *  specs become plain Params with init=0 and time_const=0 (no smoothing) —
 *  the trigger primitive at the runtime layer was dead code; from the
 *  control thread's view, "fire" is just `param.value = 1.0`.
 *  Idempotent: skip names already present. */
function applyParamSpecs(session: SessionState, specs: ParamSpec[]): void {
  for (const p of specs) {
    if (!session.paramRegistry.has(p.name)) {
      const initValue = p.value ?? 0.0
      const timeConst = p.type === 'trigger' ? 0.0 : (p.time_const ?? 0.005)
      session.paramRegistry.set(p.name, new Param(initValue, timeConst))
    }
    allocateParamSlot(session, p.name)
  }
}

type GraphOutput = { instance: string; output: string }

/** Resolve a body dac-wire expression to {instance, output}. Today's
 *  session.graphOutputs stores only named instance outputs; the expression
 *  must be a ref-shaped node. */
function resolveDacWireExprToGraphOutput(
  expr: ExprNode,
  session: SessionState,
  context: string,
): GraphOutput {
  if (typeof expr !== 'object' || expr === null || Array.isArray(expr)) {
    throw new Error(`${context}: dac.out wire requires a ref-shaped expression (use {op:'ref',instance,output}); got literal/array.`)
  }
  const e = expr as { op?: string; instance?: unknown; output?: unknown }
  if (e.op !== 'ref') {
    throw new Error(`${context}: dac.out wire requires expr.op === 'ref'; got '${String(e.op)}'.`)
  }
  if (typeof e.instance !== 'string') {
    throw new Error(`${context}: dac.out wire ref.instance must be a string`)
  }
  const inst = session.instanceRegistry.get(e.instance)
  if (!inst) {
    throw new Error(`${context}: dac.out wire references unknown instance '${e.instance}'.`)
  }
  let outputName: string
  if (typeof e.output === 'number') {
    if (e.output < 0 || e.output >= outputNames(inst).length) {
      throw new Error(`${context}: dac.out wire output index ${e.output} out of range for '${e.instance}' (${outputNames(inst).length} outputs).`)
    }
    outputName = outputNames(inst)[e.output]
  } else if (typeof e.output === 'string') {
    if (!outputNames(inst).includes(e.output)) {
      throw new Error(`${context}: dac.out wire references unknown output '${e.output}' on '${e.instance}'. Valid: ${outputNames(inst).join(', ')}`)
    }
    outputName = e.output
  } else {
    throw new Error(`${context}: dac.out wire ref.output must be a number or string`)
  }
  return { instance: e.instance, output: outputName }
}

/** Collect graph outputs from canonical body wires (outputAssign with name="dac.out")
 *  and from the deprecated topLevel.audio_outputs fallback. Body wires take precedence
 *  in source order; legacy entries are appended after. Emits a one-time deprecation
 *  warning when topLevel.audio_outputs is non-empty. */
function collectGraphOutputs(
  prog: ProgramNode,
  topLevel: ProgramTopLevel,
  session: SessionState,
  context: string,
): GraphOutput[] {
  const out: GraphOutput[] = []
  for (const w of dacWires(prog)) {
    out.push(resolveDacWireExprToGraphOutput(w.expr, session, context))
  }
  if (topLevel.audio_outputs && topLevel.audio_outputs.length > 0) {
    if (!_topLevelAudioOutputsWarned) {
      console.warn(
        'tropical: file-root `audio_outputs` is deprecated; wire to the DAC via outputAssign with name="dac.out" in the program body.',
      )
      _topLevelAudioOutputsWarned = true
    }
    for (const o of topLevel.audio_outputs) {
      if ('expr' in o) {
        throw new Error(`${context}: file-root audio_outputs[].expr form not supported. Use {instance, output} or migrate to body dac.out wires.`)
      }
      const inst = session.instanceRegistry.get(o.instance)
      if (!inst) throw new Error(`${context}: audio_outputs references unknown instance '${o.instance}'.`)
      out.push({ instance: o.instance, output: String(o.output) })
    }
  }
  return out
}
let _topLevelAudioOutputsWarned = false

/**
 * Load a ProgramNode into a session, replacing all existing state.
 * `topLevel` carries session-scoped metadata (params, audio_outputs, config).
 */
export function loadProgramAsSession(
  prog: ProgramNode,
  topLevel: ProgramTopLevel,
  session: SessionState,
): void {
  // Clear session state
  session.dac = null
  session.instanceRegistry.clear()
  session.graphOutputs.length = 0
  session.paramRegistry.clear()
  session.inputExprNodes.clear()
  session.nameCounters.clear()
  session.typeAliasRegistry.clear()
  session.sumTypeRegistry.clear()
  session.structTypeRegistry.clear()
  // Slot model (M3): clear slot registries on session load too, otherwise
  // slot indices accumulate across loads and collide with the new instances.
  session.outputSlotRegistry.clear()
  session.paramSlotRegistry.clear()
  session.outputPortMeta.clear()
  session.inputExprs.clear()
  session.slotCount = 0
  // Lean-port finding #7: the input-slot and session-array registries
  // (and the delay registry) also accumulate across loads. A stale
  // `inputSlotRegistry` index past the fresh session's slotCount writes
  // holes into slot_names (JSON nulls the engine parser rejects), and
  // stale ioArraySlots inflate array_slot_count with phantom slots.
  session.inputSlotRegistry.clear()
  session.inputPortMeta.clear()
  session.ioArraySlotCount = 0
  session.ioArraySlotSizes.length = 0
  session.ioArraySlotNames.length = 0
  session.delaySlotRegistry = []
  // Note: typeRegistry, programs, and specializationCache are NOT
  // cleared — they hold the stdlib + any session-defined types that
  // the loaded program may instantiate.

  // Register type defs (aliases, sums, structs) from ports.type_defs before anything else
  for (const td of prog.ports?.type_defs ?? []) {
    if (td.kind === 'alias') {
      session.typeAliasRegistry.set(td.name, { base: td.base })
    } else if (td.kind === 'sum') {
      session.sumTypeRegistry.set(td.name, {
        name: td.name,
        variants: td.variants.map(v => ({
          name: v.name,
          payload: v.payload.map(f => ({ name: f.name, scalar: f.scalar_type })),
        })),
      })
    } else if (td.kind === 'struct') {
      session.structTypeRegistry.set(td.name, {
        fields: td.fields.map(f => ({ name: f.name, scalar: f.scalar_type })),
      })
    }
  }

  // Register inline program definitions (loadProgramAsType handles registration)
  for (const sub of programDecls(prog)) {
    loadProgramAsType({ ...sub.program, name: sub.name }, session)
  }

  // Create params and triggers before instances (instances may reference them).
  // Body paramDecls are canonical; topLevel.params is a deprecated fallback for
  // pre-A3 patches. When both are present, body wins (dedup by name).
  applyParamSpecs(session, mergeParamSources(prog, topLevel))

  // Instantiate programs
  for (const inst of instanceDecls(prog)) {
    const { type, typeArgs } = resolveProgramType(session, inst.program, inst.type_args as RawTypeArgs | undefined, undefined)
    const instance = instantiate(type, inst.name, { baseTypeName: inst.program, typeArgs })
    session.instanceRegistry.set(instance.name, instance)
    // Slot model: allocate output slots for the instance, parallel to
    // what MCP add_instance does. Required for the per-instance
    // compile path to find each output's slot index.
    allocateOutputSlots(session, toInstanceName(instance.name), type)

    // Populate wiring from instance inputs, in the program's declared
    // input-port order (not the JSON object's key order — clients and
    // relays may not preserve it; the canonical order is the port list).
    if (inst.inputs) {
      const declared = inputNames(instance)
      const orderOf = (k: string) => {
        const i = declared.indexOf(k)
        return i === -1 ? declared.length : i
      }
      const keys = Object.keys(inst.inputs).sort((a, b) => orderOf(a) - orderOf(b))
      for (const input of keys) {
        const expr = inst.inputs[input]
        validateExpr(expr, `${inst.name}.${input}`)
        session.inputExprNodes.set(
          wireKey(portRef(toInstanceName(inst.name), toPortName(input))),
          expr,
        )
      }
    }
  }

  // Apply input defaults from each instance's program definition
  for (const [name, inst] of session.instanceRegistry) {
    const defaults = rawInputDefaults(inst)
    for (const [inputName, value] of Object.entries(defaults)) {
      const key = wireKey(portRef(toInstanceName(name), toPortName(inputName)))
      if (!session.inputExprNodes.has(key)) {
        session.inputExprNodes.set(key, value)
      }
    }
  }

  // Set audio outputs
  // Set audio outputs — body dac.out wires canonical, topLevel.audio_outputs deprecated fallback
  for (const o of collectGraphOutputs(prog, topLevel, session, 'loadProgramAsSession')) {
    session.graphOutputs.push(o)
  }

  // Compile and load
  applyFlatPlan(session, session.runtime)
}

/**
 * Load a ProgramNode as a Compiled (registerable in typeRegistry).
 * Programs with inline `program_decl` entries get their subprograms registered first.
 *
 * The strata pipeline is the only path: ProgramNode JSON is raised to
 * `ParsedProgramNode`, elaborated against the session's type registry to a
 * `ResolvedProgram`, and (for non-generics) compiled through
 * `programTypeFromResolved`. Generic programs (with `type_params`) are
 * stored in `session.programs` as raw templates and materialize on
 * instantiation via `resolveProgramType`.
 */
export function loadProgramAsType(
  prog: ProgramNode,
  session: Pick<SessionState, 'typeRegistry' | 'instanceRegistry' | 'paramRegistry' | 'specializationCache' | 'programs'> & Partial<Pick<SessionState, 'typeAliasRegistry' | 'typeResolver' | 'sumTypeRegistry' | 'structTypeRegistry' | 'inlineNested'>>,
): Compiled | undefined {
  // Register type defs (aliases, sums, structs) from type_defs before processing subprograms
  for (const td of prog.ports?.type_defs ?? []) {
    if (td.kind === 'alias' && session.typeAliasRegistry) {
      session.typeAliasRegistry.set(td.name, { base: td.base })
    } else if (td.kind === 'sum' && session.sumTypeRegistry) {
      session.sumTypeRegistry.set(td.name, {
        name: td.name,
        variants: td.variants.map(v => ({
          name: v.name,
          payload: v.payload.map(f => ({ name: f.name, scalar: f.scalar_type })),
        })),
      })
    } else if (td.kind === 'struct' && session.structTypeRegistry) {
      session.structTypeRegistry.set(td.name, {
        fields: td.fields.map(f => ({ name: f.name, scalar: f.scalar_type })),
      })
    }
  }

  // Register inline subprograms first (each handles its own registration)
  for (const sub of programDecls(prog)) {
    loadProgramAsType({ ...sub.program, name: sub.name }, session)
  }

  // Raise to ParsedProgramNode + elaborate against the session-built
  // registry. The external resolver lets nested instanceDecls
  // reference previously registered sibling programs — generic
  // templates and concrete programs both live in `session.programs`.
  const parsed = raiseProgram(prog)
  const externalResolver: ExternalProgramResolver = name => {
    const cached = session.programs.get(name)
    if (cached) return cached
    // Trigger lazy load (e.g. stdlib resolver) which populates session.programs.
    if (session.typeResolver) {
      session.typeResolver(name)
      const late = session.programs.get(name)
      if (late) return late
    }
    return undefined
  }
  const resolved = elaborate(parsed, externalResolver)

  // Generic: stash the raw template; specialization happens lazily.
  if (resolved.typeParams.length > 0) {
    session.programs.set(prog.name, resolved)
    return undefined
  }

  const type = programTypeFromResolved(resolved, new Map(), { inlineNested: session.inlineNested })
  session.typeRegistry.set(prog.name, type)
  session.programs.set(prog.name, type.prog)   // canonical post-strata
  return type
}

/**
 * Merge a ProgramNode into an existing session (additive — no state clearing).
 */
export function mergeProgramIntoSession(
  prog: ProgramNode,
  topLevel: ProgramTopLevel,
  session: SessionState,
): void {
  // Fail fast on name collisions
  for (const inst of instanceDecls(prog)) {
    if (session.instanceRegistry.has(inst.name))
      throw new Error(`merge collision: instance '${inst.name}' already exists.`)
  }
  for (const p of mergeParamSources(prog, topLevel)) {
    if (session.paramRegistry.has(p.name))
      throw new Error(`merge collision: param '${p.name}' already exists.`)
  }

  // Register type defs (aliases, sums, structs) from type_defs (additive)
  for (const td of prog.ports?.type_defs ?? []) {
    if (td.kind === 'alias') {
      session.typeAliasRegistry.set(td.name, { base: td.base })
    } else if (td.kind === 'sum') {
      session.sumTypeRegistry.set(td.name, {
        name: td.name,
        variants: td.variants.map(v => ({
          name: v.name,
          payload: v.payload.map(f => ({ name: f.name, scalar: f.scalar_type })),
        })),
      })
    } else if (td.kind === 'struct') {
      session.structTypeRegistry.set(td.name, {
        fields: td.fields.map(f => ({ name: f.name, scalar: f.scalar_type })),
      })
    }
  }

  // Register inline program definitions (loadProgramAsType handles registration)
  for (const sub of programDecls(prog)) {
    loadProgramAsType({ ...sub.program, name: sub.name }, session)
  }

  // Create params and triggers — body paramDecls canonical, topLevel fallback
  applyParamSpecs(session, mergeParamSources(prog, topLevel))

  // Instantiate programs
  for (const inst of instanceDecls(prog)) {
    const { type, typeArgs } = resolveProgramType(session, inst.program, inst.type_args as RawTypeArgs | undefined, undefined)
    const instance = instantiate(type, inst.name, { baseTypeName: inst.program, typeArgs })
    session.instanceRegistry.set(instance.name, instance)
    // Slot model: allocate output slots for the instance, parallel to
    // what MCP add_instance does. Required for the per-instance
    // compile path to find each output's slot index.
    allocateOutputSlots(session, toInstanceName(instance.name), type)

    // Populate wiring from instance inputs, in the program's declared
    // input-port order (not the JSON object's key order — clients and
    // relays may not preserve it; the canonical order is the port list).
    if (inst.inputs) {
      const declared = inputNames(instance)
      const orderOf = (k: string) => {
        const i = declared.indexOf(k)
        return i === -1 ? declared.length : i
      }
      const keys = Object.keys(inst.inputs).sort((a, b) => orderOf(a) - orderOf(b))
      for (const input of keys) {
        const expr = inst.inputs[input]
        validateExpr(expr, `${inst.name}.${input}`)
        session.inputExprNodes.set(
          wireKey(portRef(toInstanceName(inst.name), toPortName(input))),
          expr,
        )
      }
    }
  }

  // Apply input defaults
  for (const [name, inst] of session.instanceRegistry) {
    const defaults = rawInputDefaults(inst)
    for (const [inputName, value] of Object.entries(defaults)) {
      const key = wireKey(portRef(toInstanceName(name), toPortName(inputName)))
      if (!session.inputExprNodes.has(key)) {
        session.inputExprNodes.set(key, value)
      }
    }
  }

  // Append audio outputs — body dac.out wires canonical, topLevel.audio_outputs deprecated fallback
  for (const o of collectGraphOutputs(prog, topLevel, session, 'mergeProgramIntoSession')) {
    session.graphOutputs.push(o)
  }

  // Recompile
  applyFlatPlan(session, session.runtime)
}

// ─────────────────────────────────────────────────────────────
// Stdlib loading
// ─────────────────────────────────────────────────────────────

import { readFileSync, readdirSync } from 'fs'
import { join, dirname } from 'path'
import { fileURLToPath } from 'url'
import { extractMarkdown } from './parse/markdown.js'
import { parseProgram as parseTropicalProgram } from './parse/declarations.js'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

type StdlibTarget =
  | Map<string, Compiled>
  | (Pick<SessionState, 'typeRegistry' | 'instanceRegistry' | 'paramRegistry' | 'specializationCache' | 'programs'>
    & Partial<Pick<SessionState, 'typeAliasRegistry' | 'typeResolver' | 'inlineNested'>>)

function asLoadSession(target: StdlibTarget): Pick<
  SessionState,
  'typeRegistry' | 'instanceRegistry' | 'paramRegistry'
  | 'specializationCache' | 'programs'
> & Partial<Pick<SessionState, 'typeAliasRegistry' | 'typeResolver' | 'inlineNested'>> {
  if (target instanceof Map) {
    return {
      typeRegistry: target,
      instanceRegistry: new Map(),
      paramRegistry: new Map(),
      specializationCache: new Map(),
      programs: new Map(),
    }
  }
  return target
}

/**
 * Load all stdlib program files into a type registry. Reads `../stdlib/*.md`
 * (literate program documents; `README.md` is the one non-program doc and
 * is skipped by name) from disk via the literate-program pipeline:
 *
 *     read → extractMarkdown → parseProgram → elaborate → ResolvedProgram →
 *     strataPipeline → loadProgramDefFromResolved
 *
 * Both generic templates and non-generic programs land in
 * `session.programs` (the unified registry); non-generics ALSO get
 * compiled and added to `typeRegistry`. Cross-program references via
 * the elaborator find their targets in `session.programs`.
 *
 * Browser builds use `loadStdlibFromMap` (in `stdlib_loader.ts`) directly
 * with a bundled JSON map (see compiler/stdlib_bundled.ts, generated by
 * web/bundle_stdlib.ts).
 */
export function loadStdlib(target: StdlibTarget): void {
  const stdlibDir = join(__dirname, '../stdlib')
  const programFiles = readdirSync(stdlibDir)
    .filter(f => f.endsWith('.md') && f !== 'README.md')
    .sort()

  const parsedByName = new Map<string, ReturnType<typeof parseTropicalProgram>>()
  for (const file of programFiles) {
    const path = join(stdlibDir, file)
    const text = readFileSync(path, 'utf-8')
    const ext = extractMarkdown(text)
    if (ext.blocks.length !== 1) {
      throw new Error(`${path}: expected exactly 1 tropical code block, got ${ext.blocks.length}`)
    }
    const parsed = parseTropicalProgram(ext.blocks[0].source)
    parsedByName.set(parsed.name, parsed)
  }
  loadStdlibFromResolved(target, parsedByName)
}

/**
 * Register stdlib types from a map of pre-parsed `ParsedProgram`s. Elaborates
 * each against a sibling-resolver, then either:
 *   - stashes the `ResolvedProgram` in `genericTemplatesResolved` (generics)
 *   - or compiles via `programTypeFromResolved` and pushes into
 *     `typeRegistry` (non-generics, also cached in `resolvedRegistry`).
 *
 * Elaboration uses fixed-point iteration: each pass elaborates the programs
 * whose external refs can be resolved by the registry built so far.
 */
function loadStdlibFromResolved(
  target: StdlibTarget,
  parsedByName: Map<string, ReturnType<typeof parseTropicalProgram>>,
): void {
  const session = asLoadSession(target)

  const localResolved = new Map<string, ResolvedProgram>()
  const externalResolver: ExternalProgramResolver = name => localResolved.get(name)

  const remaining = new Map(parsedByName)
  let progress = true
  while (progress && remaining.size > 0) {
    progress = false
    for (const [name, parsed] of remaining) {
      try {
        const resolved = elaborate(parsed, externalResolver)
        localResolved.set(name, resolved)
        remaining.delete(name)
        progress = true
      } catch {
        // Sibling not yet elaborated; try again next pass.
      }
    }
  }
  if (remaining.size > 0) {
    // Surface the first failure for diagnostics.
    const [name, parsed] = remaining.entries().next().value as [string, ReturnType<typeof parseTropicalProgram>]
    elaborate(parsed, externalResolver)   // re-throw with the real error
    throw new Error(`loadStdlibFromResolved: failed to elaborate '${name}'`)
  }

  // Walk in elaboration order (insertion order in localResolved is
  // dependency-respecting by construction). For each program, relink
  // its sub-instance type pointers to the canonical strata-processed
  // versions already in the registry-being-built, then strata-process
  // and register. This is the topological build: by the time a program
  // is consumed downstream, every sub-instance's `instanceDecl.type`
  // points at the canonical processed program — not at the raw
  // elaborated version returned by the elaborator's resolver during
  // the try-elaborate loop above.
  //
  // Without this relink, sub-instance programs would arrive at
  // consumers (compileResolved, partition_recursive)
  // with un-strata-processed body — which is
  // exactly the gap that Bubble/BubbleCloud have been hitting in
  // nested mode ("compileResolved: register init must lower to a
  // literal value").
  const processedByName = new Map<string, ResolvedProgram>()
  for (const [name, prog] of localResolved) {
    if (prog.typeParams.length > 0) {
      // Generic template: store raw (typeParams populated) in the
      // unified registry. Specialization happens at instantiation
      // time via resolveProgramType.
      session.programs.set(name, prog)
      continue
    }
    if (session.typeRegistry.has(name)) continue
    const relinked = relinkProgramRegistry(prog, processedByName)
    const type = programTypeFromResolved(relinked, new Map(), { inlineNested: session.inlineNested })
    session.typeRegistry.set(name, type)
    session.programs.set(name, type.prog)   // canonical post-strata
    processedByName.set(name, type.prog)
  }

  // typeResolver: callers fall back here when looking up a stdlib type
  // that wasn't pre-registered concretely (e.g. a generic template name
  // that resolveProgramType then re-checks against session.programs).
  if (!session.typeResolver) {
    session.typeResolver = (n: string): Compiled | undefined => {
      return session.typeRegistry.get(n)
    }
  }
}

// ─────────────────────────────────────────────────────────────
// Program saving
// ─────────────────────────────────────────────────────────────

/**
 * Serialize the current session to a v2 ProgramNode + top-level metadata.
 */
export function saveProgramFromSession(
  session: SessionState,
): { node: ProgramNode; topLevel: ProgramTopLevel } {
  const decls: ExprNode[] = []

  // paramDecls go first so they're declared before the instances that may
  // reference them via paramExpr.
  for (const [name, p] of session.paramRegistry) {
    decls.push({ op: 'paramDecl', name, value: p.value } as ExprNode)
  }

  for (const [name, inst] of session.instanceRegistry) {
    const entry: Record<string, unknown> = { op: 'instanceDecl', name, program: inst.baseTypeName }
    if (inst.typeArgs) entry.type_args = inst.typeArgs

    // Merge wiring for this instance
    const inputs: Record<string, ExprNode> = {}
    for (const portName of inputNames(inst)) {
      const key = wireKey(portRef(toInstanceName(name), toPortName(portName)))
      const expr = session.inputExprNodes.get(key)
      // Reverse the compile's delay-hoisting: a post-compile wire holds
      // `sessionSlot` reads whose source lives in delaySlotRegistry.
      // Emit the authored `delay(<source>)` form so the patch reloads.
      if (expr !== undefined) inputs[portName] = reconstructWireDelays(expr, session.delaySlotRegistry)
    }
    if (Object.keys(inputs).length > 0) entry.inputs = inputs
    decls.push(entry as ExprNode)
  }

  // dac.out wires emitted as body assigns (canonical post-A4)
  const assigns: ExprNode[] = []
  for (const o of session.graphOutputs) {
    const inst = session.instanceRegistry.get(o.instance)
    if (!inst) continue // session has a stale entry; skip silently rather than crash on save
    const outputIdx = outputNames(inst).indexOf(o.output)
    if (outputIdx < 0) continue
    assigns.push({
      op: 'outputAssign',
      name: 'dac.out',
      expr: { op: 'ref', instance: o.instance, output: outputIdx },
    } as unknown as ExprNode)
  }

  const node: ProgramNode = {
    op: 'program',
    name: 'patch',
    body: { op: 'block', decls, assigns: assigns.length ? assigns : undefined },
  }

  const topLevel: ProgramTopLevel = {}
  return { node, topLevel }
}

// ─────────────────────────────────────────────────────────────
// Export session as reusable composite program
// ─────────────────────────────────────────────────────────────

export interface ExportProgramOpts {
  /** Name for the new program type. */
  name: string
  /**
   * Declared inputs: map from new input name → "instance:port" in the session.
   * The current wiring expression for that port becomes the input_default.
   */
  inputs: Record<string, string>
  /**
   * Declared outputs: map from new output name → { instance, output }.
   */
  outputs: Record<string, { instance: string; output: string }>
}

/**
 * Crystallize part of a live session into a reusable composite ProgramNode.
 *
 * Walks backward from the declared outputs to find all reachable instances.
 * Rewrites wiring so that exposed ports become program inputs (with the
 * current wiring folded into each port's `default`), and instance refs
 * stay internal.
 *
 * Returns a ProgramNode that can be registered as a type and instantiated.
 */
export function exportSessionAsProgram(
  session: SessionState,
  opts: ExportProgramOpts,
): ProgramNode {
  const { name, inputs, outputs } = opts

  // Validate output mappings and build output ref expressions
  const outputKeys = Object.keys(outputs)
  const outputExprs: Record<string, ExprNode> = {}
  const rootExprs: ExprNode[] = []
  for (const [outName, ref] of Object.entries(outputs)) {
    const inst = session.instanceRegistry.get(ref.instance)
    if (!inst) throw new Error(`export: output '${outName}' references unknown instance '${ref.instance}'.`)
    if (!outputNames(inst).includes(ref.output))
      throw new Error(`export: instance '${ref.instance}' has no output '${ref.output}'. Available: ${outputNames(inst).join(', ')}`)
    const refExpr: ExprNode = { op: 'ref', instance: ref.instance, output: ref.output }
    outputExprs[outName] = refExpr
    rootExprs.push(refExpr)
  }

  // Validate input mappings. exposedKeys collects validated wire
  // keys (brand re-applied after parse — the input strings came from
  // user JSON, so they're untrusted until parseWireKey accepts them).
  const inputKeys = Object.keys(inputs)
  const exposedKeys = new Set<WireKey>()
  for (const [inputName, target] of Object.entries(inputs)) {
    let ref
    try {
      ref = parseWireKey(target)
    } catch {
      throw new Error(`export: input '${inputName}' target must be "instance:port", got '${target}'.`)
    }
    const instName = ref.instance
    const portNameStr = ref.port
    const inst = session.instanceRegistry.get(instName)
    if (!inst) throw new Error(`export: input '${inputName}' references unknown instance '${instName}'.`)
    if (!inputNames(inst).includes(portNameStr))
      throw new Error(`export: instance '${instName}' has no input '${portNameStr}'. Available: ${inputNames(inst).join(', ')}`)
    exposedKeys.add(wireKey(ref))
  }

  // Walk backward from outputs to find all needed instances
  const allInstances = new Set(session.instanceRegistry.keys())
  const reachable = reachableInstances(rootExprs, session.inputExprNodes, allInstances)

  // Also include instances reachable from exposed input wiring defaults,
  // since those expressions will become input_defaults and may reference
  // internal instances.
  for (const target of exposedKeys) {
    const currentExpr = session.inputExprNodes.get(target)
    if (currentExpr) {
      for (const dep of exprDependencies(currentExpr)) {
        if (allInstances.has(dep) && !reachable.has(dep)) {
          // Add dep and its transitive deps
          const extra = reachableInstances([currentExpr], session.inputExprNodes, allInstances)
          for (const e of extra) reachable.add(e)
        }
      }
    }
  }

  // Check for dangling references — inputs that reference instances outside
  // the reachable set and are not being exposed as program inputs
  for (const instName of reachable) {
    for (const [key, expr] of session.inputExprNodes) {
      if (!key.startsWith(`${instName}:`)) continue
      if (exposedKeys.has(key)) continue  // will be replaced with {op:"input"}
      for (const dep of exprDependencies(expr)) {
        if (!reachable.has(dep) && allInstances.has(dep)) {
          throw new Error(
            `export: instance '${instName}' wiring '${key}' references '${dep}' which is outside the exported subgraph. ` +
            `Either expose it as an input or include '${dep}' in the output dependency chain.`
          )
        }
      }
    }
  }

  // Rewrite session-level ref nodes to nested_out for internal instances
  function rewriteRefs(node: ExprNode): ExprNode {
    if (typeof node === 'number' || typeof node === 'boolean') return node
    if (Array.isArray(node)) return node.map(rewriteRefs)
    const obj = node as Record<string, unknown>
    if (obj.op === 'ref' && reachable.has(obj.instance as string)) {
      return { op: 'nestedOut', ref: obj.instance, output: obj.output } as unknown as ExprNode
    }
    // Recurse into args
    if ('args' in obj) {
      return { ...obj, args: (obj.args as ExprNode[]).map(rewriteRefs) } as unknown as ExprNode
    }
    return node
  }

  // Gather port type metadata from the source instances so exported
  // inputs/outputs round-trip through re-parse.
  const isDefaultPortType = (t: PortType | undefined): boolean =>
    t === undefined || portTypeEqual(t, Float)

  const portTypeToDecl = (t: PortType): PortTypeDecl => {
    switch (t.kind) {
      case 'scalar': return t.scalar
      case 'alias':  return t.alias.name
      case 'array': {
        const elem = typeof t.element === 'string' ? t.element : t.element.name
        // Post-strata shapes are concrete integers (specialize has run).
        const shape = t.shape.map(d => {
          if (typeof d !== 'number') {
            throw new Error(`export: array shape carries unresolved type-param '${d.name}'`)
          }
          return d
        })
        return { kind: 'array', element: elem, shape }
      }
    }
  }

  // Collect per-port defaults from current wiring of exposed ports.
  // Targets were already validated at the start of this function; re-
  // parse to recover the brand for the Map lookup.
  const inputDefaults: Record<string, ExprNode> = {}
  for (const [inputName, target] of Object.entries(inputs)) {
    const currentExpr = session.inputExprNodes.get(wireKey(parseWireKey(target)))
    if (currentExpr !== undefined) {
      inputDefaults[inputName] = rewriteRefs(currentExpr)
    }
  }

  const inputEntries: Array<string | ProgramPortSpec> = inputKeys.map(inputName => {
    const target = inputs[inputName]
    const ref = parseWireKey(target)
    const inst = session.instanceRegistry.get(ref.instance)!
    const idx = inputIndex(inst, ref.port)
    const pt = inputPortType(inst, idx)
    const dflt = inputDefaults[inputName]
    const entry: ProgramPortSpec = { name: inputName }
    if (!isDefaultPortType(pt)) entry.type = portTypeToDecl(pt!)
    if (dflt !== undefined) entry.default = dflt
    return entry.type === undefined && entry.default === undefined
      ? inputName
      : entry
  })

  const outputEntries: Array<string | ProgramPortSpec> = outputKeys.map(outName => {
    const ref = outputs[outName]
    const inst = session.instanceRegistry.get(ref.instance)!
    const idx = outputIndex(inst, ref.output)
    const pt = outputPortType(inst, idx)
    const entry: ProgramPortSpec = { name: outName }
    if (!isDefaultPortType(pt)) entry.type = portTypeToDecl(pt!)
    return entry.type === undefined ? outName : entry
  })

  // Topologically sort reachable instances so dependencies come first.
  // loadProgramDef and the flattener both process nested calls sequentially,
  // so a call arg referencing a later call would fail.
  const depGraph = buildDependencyGraph(reachable, session.inputExprNodes)
  const { order, complete } = topologicalSort(depGraph)
  if (!complete) {
    // Cycles exist — they need a breaks_cycles annotation to resolve.
    // For now, include all reachable instances in whatever order topo produced,
    // plus any that weren't reached (cycle members).
    for (const instName of reachable) {
      if (!order.includes(instName)) order.push(instName)
    }
  }

  const decls: ExprNode[] = []
  for (const instName of order) {
    const inst = session.instanceRegistry.get(instName)!
    const entry: Record<string, unknown> = {
      op: 'instanceDecl',
      name: instName,
      program: inst.baseTypeName,
    }
    if (inst.typeArgs) entry.type_args = inst.typeArgs

    // Copy wiring, rewriting exposed ports to {op:"input", name:...}
    // and ref→nested_out for sibling instances
    const instInputs: Record<string, ExprNode> = {}
    for (const portName of inputNames(inst)) {
      const key = wireKey(portRef(toInstanceName(instName), toPortName(portName)))
      if (exposedKeys.has(key)) {
        const inputName = Object.entries(inputs).find(([_, t]) => t === key)![0]
        instInputs[portName] = { op: 'input', name: inputName }
      } else {
        const expr = session.inputExprNodes.get(key)
        if (expr !== undefined) instInputs[portName] = rewriteRefs(expr)
      }
    }
    if (Object.keys(instInputs).length > 0) entry.inputs = instInputs

    decls.push(entry as ExprNode)
  }

  // Output assigns — reference internal instances via nested_out
  const assigns: ExprNode[] = []
  for (const [outName, ref] of Object.entries(outputs)) {
    assigns.push({
      op: 'outputAssign',
      name: outName,
      expr: { op: 'nestedOut', ref: ref.instance, output: ref.output },
    } as ExprNode)
  }

  return {
    op: 'program',
    name,
    ports: { inputs: inputEntries, outputs: outputEntries },
    body: { op: 'block', decls, assigns },
  }
}
