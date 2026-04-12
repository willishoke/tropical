/**
 * program.ts — ProgramJSON schema, loading, saving, and stdlib.
 *
 * A program has inputs, outputs, state, subprograms, and a process body.
 * - A program with `process` and no `instances` is a leaf.
 * - A program with `instances` and `audio_outputs` is a top-level graph.
 * - A program with `instances`, `inputs`, and computed `outputs` is a reusable composite.
 */

import type { ExprNode } from './expr.js'
import { validateExpr } from './expr.js'
import type { TypeDefJSON, SessionState } from './session.js'
import { loadProgramDef } from './session.js'
import { applyFlatPlan } from './apply_plan.js'
import { Param, Trigger } from './runtime/param.js'
import { ProgramType } from './program_types.js'

// ─────────────────────────────────────────────────────────────
// ProgramJSON schema
// ─────────────────────────────────────────────────────────────

export interface ProgramJSON {
  schema: 'tropical_program_1'
  name: string

  /** Inputs to this program. Empty or absent = top-level (no external inputs). */
  inputs?: Array<string | { name: string; type?: string; default?: ExprNode; bounds?: [number | null, number | null] }>
  /** Output declarations — names for leaf programs, expressions for composites. */
  outputs?: Array<string | { name: string; type?: string; bounds?: [number | null, number | null] }>

  /** Scalar/array state registers. */
  regs?: Record<string, number | boolean | number[] | number[][] | { init: number | boolean | number[] | number[][]; type: string }>
  /** Named delay nodes. */
  delays?: Record<string, { update: ExprNode; init?: number }>
  /** Sample rate override. */
  sample_rate?: number
  /** Default values for inputs. */
  input_defaults?: Record<string, ExprNode>

  /** Inline subprogram definitions (reusable within this program). */
  programs?: Record<string, ProgramJSON>
  /** Instantiated subprograms. */
  instances?: Record<string, {
    program: string
    inputs?: Record<string, ExprNode>
  }>

  /** Process body for leaf programs (direct computation). */
  process?: {
    outputs: Record<string, ExprNode>
    next_regs?: Record<string, ExprNode>
  }

  /** Audio output routing (top-level programs only). */
  audio_outputs?: Array<
    | { instance: string; output: string | number }
    | { expr: ExprNode }
  >
  /** Control parameters (top-level programs only). */
  params?: Array<{
    name: string
    value?: number
    time_const?: number
    type?: 'param' | 'trigger'
  }>
  /** Runtime configuration. */
  config?: { buffer_length?: number; sample_rate?: number }
  /** Inline ADT type definitions. */
  type_defs?: TypeDefJSON[]
  /** When true, outputs depend only on previous-sample state — allows feedback cycles. */
  breaks_cycles?: boolean
}

// ─────────────────────────────────────────────────────────────
// Program loading
// ─────────────────────────────────────────────────────────────

/**
 * Load a ProgramJSON into a session, replacing all existing state.
 */
export function loadProgramAsSession(
  prog: ProgramJSON,
  session: SessionState,
): void {
  // Clear session state
  session.dac = null
  session.instanceRegistry.clear()
  session.graphOutputs.length = 0
  session.paramRegistry.clear()
  session.triggerRegistry.clear()
  session.inputExprNodes.clear()
  session._nameCounters.clear()

  // Register inline program definitions
  if (prog.programs) {
    for (const [name, subProg] of Object.entries(prog.programs)) {
      const type = loadProgramAsType({ ...subProg, name }, session)
      session.typeRegistry.set(name, type)
    }
  }

  // Create params and triggers before instances (instances may reference them)
  for (const p of prog.params ?? []) {
    if (p.type === 'trigger') {
      session.triggerRegistry.set(p.name, new Trigger())
    } else {
      session.paramRegistry.set(p.name, new Param(p.value ?? 0.0, p.time_const ?? 0.005))
    }
  }

  // Instantiate programs
  for (const [name, inst] of Object.entries(prog.instances ?? {})) {
    const type = session.typeRegistry.get(inst.program)
    if (!type) throw new Error(`Unknown program type '${inst.program}'.`)
    const instance = type.instantiateAs(name)
    session.instanceRegistry.set(instance.name, instance)

    // Populate wiring from instance inputs
    if (inst.inputs) {
      for (const [input, expr] of Object.entries(inst.inputs)) {
        validateExpr(expr, `${name}.${input}`)
        session.inputExprNodes.set(`${name}:${input}`, expr)
      }
    }
  }

  // Apply input defaults from each instance's program definition
  for (const [name, inst] of session.instanceRegistry) {
    const defaults = inst._def.rawInputDefaults
    for (const [inputName, value] of Object.entries(defaults)) {
      const key = `${name}:${inputName}`
      if (!session.inputExprNodes.has(key)) {
        session.inputExprNodes.set(key, value)
      }
    }
  }

  // Set audio outputs
  for (const out of prog.audio_outputs ?? []) {
    if ('expr' in out) {
      throw new Error('Output expressions not supported in plan-based path. Use instance output refs instead.')
    }
    const inst = session.instanceRegistry.get(out.instance)
    if (!inst) throw new Error(`Output instance '${out.instance}' not found.`)
    session.graphOutputs.push({ instance: out.instance, output: String(out.output) })
  }

  // Compile and load
  applyFlatPlan(session, session.runtime)
}

/**
 * Load a leaf ProgramJSON as a ProgramType (registerable in typeRegistry).
 * Programs with inline `programs` get their subprograms registered first.
 */
export function loadProgramAsType(
  prog: ProgramJSON,
  session: Pick<SessionState, 'typeRegistry' | 'instanceRegistry' | 'paramRegistry' | 'triggerRegistry'>,
): ProgramType {
  // Register inline subprograms first
  if (prog.programs) {
    for (const [name, subProg] of Object.entries(prog.programs)) {
      const subType = loadProgramAsType({ ...subProg, name }, session)
      session.typeRegistry.set(name, subType)
    }
  }

  return loadProgramDef(prog, session)
}

/**
 * Merge a ProgramJSON into an existing session (additive — no state clearing).
 */
export function mergeProgramIntoSession(
  prog: ProgramJSON,
  session: SessionState,
): void {
  // Fail fast on name collisions
  for (const name of Object.keys(prog.instances ?? {})) {
    if (session.instanceRegistry.has(name))
      throw new Error(`merge collision: instance '${name}' already exists.`)
  }
  for (const p of prog.params ?? []) {
    if (session.paramRegistry.has(p.name) || session.triggerRegistry.has(p.name))
      throw new Error(`merge collision: param/trigger '${p.name}' already exists.`)
  }

  // Register inline program definitions
  if (prog.programs) {
    for (const [name, subProg] of Object.entries(prog.programs)) {
      const type = loadProgramAsType({ ...subProg, name }, session)
      session.typeRegistry.set(name, type)
    }
  }

  // Create params and triggers
  for (const p of prog.params ?? []) {
    if (p.type === 'trigger') {
      session.triggerRegistry.set(p.name, new Trigger())
    } else {
      session.paramRegistry.set(p.name, new Param(p.value ?? 0.0, p.time_const ?? 0.005))
    }
  }

  // Instantiate programs
  for (const [name, inst] of Object.entries(prog.instances ?? {})) {
    const type = session.typeRegistry.get(inst.program)
    if (!type) throw new Error(`Unknown program type '${inst.program}'.`)
    const instance = type.instantiateAs(name)
    session.instanceRegistry.set(instance.name, instance)

    // Populate wiring from instance inputs
    if (inst.inputs) {
      for (const [input, expr] of Object.entries(inst.inputs)) {
        validateExpr(expr, `${name}.${input}`)
        session.inputExprNodes.set(`${name}:${input}`, expr)
      }
    }
  }

  // Apply input defaults
  for (const [name, inst] of session.instanceRegistry) {
    const defaults = inst._def.rawInputDefaults
    for (const [inputName, value] of Object.entries(defaults)) {
      const key = `${name}:${inputName}`
      if (!session.inputExprNodes.has(key)) {
        session.inputExprNodes.set(key, value)
      }
    }
  }

  // Append audio outputs
  for (const out of prog.audio_outputs ?? []) {
    if ('expr' in out) {
      throw new Error('Output expressions not supported in plan-based path. Use instance output refs instead.')
    }
    const inst = session.instanceRegistry.get(out.instance)
    if (!inst) throw new Error(`Output instance '${out.instance}' not found.`)
    session.graphOutputs.push({ instance: out.instance, output: String(out.output) })
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

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

/**
 * Load all stdlib ProgramJSON files into a type registry.
 * Accepts either a full session or just a typeRegistry Map.
 */
export function loadStdlib(
  target: Map<string, ProgramType> | Pick<SessionState, 'typeRegistry' | 'instanceRegistry' | 'paramRegistry' | 'triggerRegistry'>,
): void {
  // If given a bare Map, wrap it in a minimal session-like object
  const session: Pick<SessionState, 'typeRegistry' | 'instanceRegistry' | 'paramRegistry' | 'triggerRegistry'> =
    target instanceof Map
      ? { typeRegistry: target, instanceRegistry: new Map(), paramRegistry: new Map(), triggerRegistry: new Map() }
      : target

  const stdlibDir = join(__dirname, '../stdlib')
  const files = readdirSync(stdlibDir).filter(f => f.endsWith('.json')).sort()
  for (const file of files) {
    const prog = JSON.parse(readFileSync(join(stdlibDir, file), 'utf-8')) as ProgramJSON
    const type = loadProgramAsType(prog, session)
    session.typeRegistry.set(prog.name, type)
  }
}

// ─────────────────────────────────────────────────────────────
// Program saving
// ─────────────────────────────────────────────────────────────

/**
 * Serialize the current session to a ProgramJSON.
 */
export function saveProgramFromSession(
  session: SessionState,
): ProgramJSON {
  const prog: ProgramJSON = { schema: 'tropical_program_1', name: 'patch' }

  // Instances
  if (session.instanceRegistry.size) {
    prog.instances = {}
    for (const [name, inst] of session.instanceRegistry) {
      prog.instances[name] = { program: inst.typeName }
    }

    // Merge wiring into instance inputs
    for (const [key, node] of session.inputExprNodes) {
      const [module, input] = key.split(':')
      const inst = prog.instances[module]
      if (inst) {
        if (!inst.inputs) inst.inputs = {}
        inst.inputs[input] = node
      }
    }
  }

  // Audio outputs
  if (session.graphOutputs.length) {
    prog.audio_outputs = session.graphOutputs.map(o => ({
      instance: o.instance, output: o.output,
    }))
  }

  // Params and triggers
  const params: NonNullable<ProgramJSON['params']> = []
  for (const [name, p] of session.paramRegistry) {
    params.push({ name, value: p.value, time_const: 0.005 })
  }
  for (const [name] of session.triggerRegistry) {
    params.push({ name, type: 'trigger' })
  }
  if (params.length) prog.params = params

  return prog
}
