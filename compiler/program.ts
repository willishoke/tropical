/**
 * program.ts — Unified program schema that subsumes both ModuleDefJSON and PatchJSON.
 *
 * A program has inputs, outputs, state, subprograms, and a process body.
 * - A program with `process` and no `instances` is a leaf (like ModuleDefJSON).
 * - A program with `instances` and `audio_outputs` is a top-level graph (like PatchJSON).
 * - A program with `instances`, `inputs`, and computed `outputs` is a reusable composite.
 */

import type { ExprNode } from './expr.js'
import type {
  ModuleDefJSON, PatchJSON, TypeDefJSON, SessionState, NestedModuleJSON,
} from './patch.js'
import { loadModuleFromJSON } from './patch.js'
import type { ValueCoercible, RegInit } from './module.js'
import { ModuleType } from './module.js'

// ─────────────────────────────────────────────────────────────
// ProgramJSON schema
// ─────────────────────────────────────────────────────────────

export interface ProgramJSON {
  schema: 'tropical_program_1'
  name: string

  /** Inputs to this program. Empty or absent = top-level (no external inputs). */
  inputs?: Array<string | { name: string; type?: string; default?: ExprNode }>
  /** Output declarations — names for leaf programs, expressions for composites. */
  outputs?: Array<string | { name: string; type?: string }>

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
// Conversion: PatchJSON → ProgramJSON
// ─────────────────────────────────────────────────────────────

export function convertPatchToProgram(patch: PatchJSON): ProgramJSON {
  const prog: ProgramJSON = {
    schema: 'tropical_program_1',
    name: 'patch',
  }

  // module_defs → programs
  if (patch.module_defs?.length) {
    prog.programs = {}
    for (const def of patch.module_defs) {
      prog.programs[def.name] = convertModuleDefToProgram(def)
    }
  }

  // modules → instances (with wiring merged in)
  if (patch.modules?.length) {
    prog.instances = {}
    for (const m of patch.modules) {
      const name = m.name ?? m.type
      prog.instances[name] = { program: m.type }
    }

    // Merge connections into instance inputs
    for (const conn of patch.connections ?? []) {
      const inst = prog.instances[conn.dst]
      if (inst) {
        if (!inst.inputs) inst.inputs = {}
        inst.inputs[String(conn.dst_input)] = {
          op: 'ref', module: conn.src, output: conn.src_output,
        } as ExprNode
      }
    }

    // Merge input_exprs into instance inputs
    for (const ie of patch.input_exprs ?? []) {
      const inst = prog.instances[ie.module]
      if (inst) {
        if (!inst.inputs) inst.inputs = {}
        inst.inputs[String(ie.input)] = ie.expr
      }
    }
  }

  // outputs → audio_outputs
  if (patch.outputs?.length) {
    prog.audio_outputs = patch.outputs.map(o => {
      if ('expr' in o) return { expr: o.expr }
      return { instance: o.module, output: o.output }
    })
  }

  if (patch.params?.length) prog.params = patch.params
  if (patch.config) prog.config = patch.config
  if (patch.type_defs?.length) prog.type_defs = patch.type_defs

  return prog
}

// ─────────────────────────────────────────────────────────────
// Conversion: ModuleDefJSON → ProgramJSON
// ─────────────────────────────────────────────────────────────

export function convertModuleDefToProgram(def: ModuleDefJSON): ProgramJSON {
  const prog: ProgramJSON = {
    schema: 'tropical_program_1',
    name: def.name,
    inputs: def.inputs,
    outputs: def.outputs,
    process: def.process,
  }

  if (def.regs)            prog.regs = def.regs
  if (def.delays)          prog.delays = def.delays
  if (def.sample_rate)     prog.sample_rate = def.sample_rate
  if (def.input_defaults)  prog.input_defaults = def.input_defaults

  // nested → instances
  if (def.nested) {
    prog.instances = {}
    for (const [alias, spec] of Object.entries(def.nested)) {
      prog.instances[alias] = {
        program: spec.type,
        inputs: spec.inputs,
      }
    }
  }

  return prog
}

// ─────────────────────────────────────────────────────────────
// Conversion: ProgramJSON → PatchJSON (backward compatibility)
// ─────────────────────────────────────────────────────────────

export function convertProgramToPatch(prog: ProgramJSON): PatchJSON {
  const patch: PatchJSON = {
    schema: 'tropical_patch_1',
    modules: [],
  }

  // programs → module_defs
  if (prog.programs) {
    patch.module_defs = []
    for (const [, subProg] of Object.entries(prog.programs)) {
      patch.module_defs.push(convertProgramToModuleDef(subProg))
    }
  }

  // instances → modules + input_exprs
  if (prog.instances) {
    const inputExprs: NonNullable<PatchJSON['input_exprs']> = []
    for (const [name, inst] of Object.entries(prog.instances)) {
      patch.modules.push({ type: inst.program, name })
      if (inst.inputs) {
        for (const [input, expr] of Object.entries(inst.inputs)) {
          inputExprs.push({ module: name, input, expr })
        }
      }
    }
    if (inputExprs.length) patch.input_exprs = inputExprs
  }

  // audio_outputs → outputs
  if (prog.audio_outputs?.length) {
    patch.outputs = prog.audio_outputs.map(o => {
      if ('expr' in o) return { expr: o.expr }
      return { module: o.instance, output: o.output }
    })
  }

  if (prog.params?.length) patch.params = prog.params
  if (prog.config) patch.config = prog.config
  if (prog.type_defs?.length) patch.type_defs = prog.type_defs

  return patch
}

function convertProgramToModuleDef(prog: ProgramJSON): ModuleDefJSON {
  const def: ModuleDefJSON = {
    name: prog.name,
    inputs: prog.inputs ?? [],
    outputs: prog.outputs ?? [],
    process: prog.process ?? { outputs: {} },
  }

  if (prog.regs)            def.regs = prog.regs
  if (prog.delays)          def.delays = prog.delays
  if (prog.sample_rate)     def.sample_rate = prog.sample_rate
  if (prog.input_defaults)  def.input_defaults = prog.input_defaults
  if (prog.breaks_cycles)   def.breaks_cycles = true

  // instances → nested
  if (prog.instances) {
    def.nested = {}
    for (const [alias, inst] of Object.entries(prog.instances)) {
      def.nested[alias] = {
        type: inst.program,
        inputs: inst.inputs ?? {},
      }
    }
  }

  return def
}

// ─────────────────────────────────────────────────────────────
// Program loading
// ─────────────────────────────────────────────────────────────

/**
 * Load a ProgramJSON into a session as a top-level program (like loadPatchFromJSON).
 * For programs with `instances` and `audio_outputs`, converts to PatchJSON and loads.
 * For leaf programs with `process`, registers as a module type and instantiates.
 */
export function loadProgramAsSession(
  prog: ProgramJSON,
  session: SessionState,
  loadPatch: (json: PatchJSON, session: SessionState) => void,
): void {
  const patch = convertProgramToPatch(prog)
  loadPatch(patch, session)
}

/**
 * Load a leaf ProgramJSON as a ModuleType (registerable in typeRegistry).
 * Converts to ModuleDefJSON and delegates to loadModuleFromJSON.
 * Programs with inline `programs` get their subprograms registered first.
 */
export function loadProgramAsType(
  prog: ProgramJSON,
  session: Pick<SessionState, 'typeRegistry' | 'instanceRegistry' | 'paramRegistry' | 'triggerRegistry'>,
): ModuleType {
  // Register inline subprograms first
  if (prog.programs) {
    for (const [name, subProg] of Object.entries(prog.programs)) {
      const subType = loadProgramAsType({ ...subProg, name }, session)
      session.typeRegistry.set(name, subType)
    }
  }

  const def = convertProgramToModuleDef(prog)
  return loadModuleFromJSON(def, session)
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
  target: Map<string, ModuleType> | Pick<SessionState, 'typeRegistry' | 'instanceRegistry' | 'paramRegistry' | 'triggerRegistry'>,
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
  savePatch: (session: SessionState) => PatchJSON,
): ProgramJSON {
  const patch = savePatch(session)
  return convertPatchToProgram(patch)
}
