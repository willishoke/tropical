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
import { exprDependencies, reachableInstances, buildDependencyGraph, topologicalSort } from './compiler.js'

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
    /** Compile-time type args for generic programs. Numeric literals, or
     *  `{op:"type_param",name}` to forward from the outer program's type_params. */
    type_args?: Record<string, number | ExprNode>
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
  /** Compile-time type parameters. Each instance of a program with type_params must supply
   *  a matching type_arg (or the declared default is used). */
  type_params?: Record<string, { type: 'int'; default?: number }>
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
  session.typeAliasRegistry.clear()

  // Register type aliases from type_defs before anything else
  for (const td of prog.type_defs ?? []) {
    if (td.kind === 'alias') {
      session.typeAliasRegistry.set(td.name, { base: td.base, bounds: td.bounds })
    }
  }

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
    const type = session.typeRegistry.get(inst.program) ?? session.typeResolver?.(inst.program)
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
  session: Pick<SessionState, 'typeRegistry' | 'instanceRegistry' | 'paramRegistry' | 'triggerRegistry'> & Partial<Pick<SessionState, 'typeAliasRegistry' | 'typeResolver'>>,
): ProgramType {
  // Register type aliases from type_defs before processing subprograms
  if (session.typeAliasRegistry) {
    for (const td of prog.type_defs ?? []) {
      if (td.kind === 'alias') {
        session.typeAliasRegistry.set(td.name, { base: td.base, bounds: td.bounds })
      }
    }
  }

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

  // Register type aliases from type_defs (additive)
  for (const td of prog.type_defs ?? []) {
    if (td.kind === 'alias') {
      session.typeAliasRegistry.set(td.name, { base: td.base, bounds: td.bounds })
    }
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
    const type = session.typeRegistry.get(inst.program) ?? session.typeResolver?.(inst.program)
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
 * Types are indexed first, then loaded on demand — dependencies resolve
 * recursively regardless of alphabetical file ordering.
 * Accepts either a full session or just a typeRegistry Map.
 */
export function loadStdlib(
  target: Map<string, ProgramType> | Pick<SessionState, 'typeRegistry' | 'instanceRegistry' | 'paramRegistry' | 'triggerRegistry'>,
): void {
  // If given a bare Map, wrap it in a minimal session-like object
  const session: Pick<SessionState, 'typeRegistry' | 'instanceRegistry' | 'paramRegistry' | 'triggerRegistry'> & Partial<Pick<SessionState, 'typeResolver'>> =
    target instanceof Map
      ? { typeRegistry: target, instanceRegistry: new Map(), paramRegistry: new Map(), triggerRegistry: new Map() }
      : target

  const stdlibDir = join(__dirname, '../stdlib')
  const files = readdirSync(stdlibDir).filter(f => f.endsWith('.json')).sort()

  // Index all stdlib files by program name
  const index = new Map<string, string>()
  for (const file of files) {
    const path = join(stdlibDir, file)
    const prog = JSON.parse(readFileSync(path, 'utf-8')) as ProgramJSON
    index.set(prog.name, path)
  }

  // Set up on-demand resolver — loads a stdlib type (and its deps) on first reference
  const loading = new Set<string>()
  session.typeResolver = (name: string): ProgramType | undefined => {
    const existing = session.typeRegistry.get(name)
    if (existing) return existing
    if (loading.has(name)) throw new Error(`Circular stdlib dependency: ${[...loading, name].join(' → ')}`)
    const path = index.get(name)
    if (!path) return undefined
    loading.add(name)
    const prog = JSON.parse(readFileSync(path, 'utf-8')) as ProgramJSON
    const type = loadProgramAsType(prog, session)
    session.typeRegistry.set(prog.name, type)
    loading.delete(name)
    return type
  }

  // Eagerly load all indexed types (resolver handles dependency order)
  for (const name of index.keys()) {
    if (!session.typeRegistry.has(name)) {
      session.typeResolver(name)
    }
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
 * Crystallize part of a live session into a reusable composite ProgramJSON.
 *
 * Walks backward from the declared outputs to find all reachable instances.
 * Rewrites wiring so that exposed ports become program inputs (with the
 * current wiring as input_defaults), and instance refs stay internal.
 *
 * Returns a ProgramJSON that can be registered as a type and instantiated.
 */
export function exportSessionAsProgram(
  session: SessionState,
  opts: ExportProgramOpts,
): ProgramJSON {
  const { name, inputs, outputs } = opts

  // Validate output mappings and build output ref expressions
  const outputNames = Object.keys(outputs)
  const outputExprs: Record<string, ExprNode> = {}
  const rootExprs: ExprNode[] = []
  for (const [outName, ref] of Object.entries(outputs)) {
    const inst = session.instanceRegistry.get(ref.instance)
    if (!inst) throw new Error(`export: output '${outName}' references unknown instance '${ref.instance}'.`)
    if (!inst.outputNames.includes(ref.output))
      throw new Error(`export: instance '${ref.instance}' has no output '${ref.output}'. Available: ${inst.outputNames.join(', ')}`)
    const refExpr: ExprNode = { op: 'ref', instance: ref.instance, output: ref.output }
    outputExprs[outName] = refExpr
    rootExprs.push(refExpr)
  }

  // Validate input mappings
  const inputNames = Object.keys(inputs)
  const exposedKeys = new Set<string>()   // "instance:port" keys being exposed as inputs
  for (const [inputName, target] of Object.entries(inputs)) {
    const [instName, portName] = target.split(':')
    if (!portName) throw new Error(`export: input '${inputName}' target must be "instance:port", got '${target}'.`)
    const inst = session.instanceRegistry.get(instName)
    if (!inst) throw new Error(`export: input '${inputName}' references unknown instance '${instName}'.`)
    if (!inst.inputNames.includes(portName))
      throw new Error(`export: instance '${instName}' has no input '${portName}'. Available: ${inst.inputNames.join(', ')}`)
    exposedKeys.add(target)
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
      return { op: 'nested_out', ref: obj.instance, output: obj.output } as unknown as ExprNode
    }
    // Recurse into args
    if ('args' in obj) {
      return { ...obj, args: (obj.args as ExprNode[]).map(rewriteRefs) } as unknown as ExprNode
    }
    return node
  }

  // Build the exported ProgramJSON
  const prog: ProgramJSON = {
    schema: 'tropical_program_1',
    name,
    inputs: inputNames,
    outputs: outputNames,
  }

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

  prog.instances = {}
  for (const instName of order) {
    const inst = session.instanceRegistry.get(instName)!
    const entry: { program: string; inputs?: Record<string, ExprNode> } = {
      program: inst.typeName,
    }

    // Copy wiring, rewriting exposed ports to {op:"input", name:...}
    // and ref→nested_out for sibling instances
    for (const portName of inst.inputNames) {
      const key = `${instName}:${portName}`
      if (exposedKeys.has(key)) {
        // Find which program input maps to this port
        const inputName = Object.entries(inputs).find(([_, t]) => t === key)![0]
        if (!entry.inputs) entry.inputs = {}
        entry.inputs[portName] = { op: 'input', name: inputName }
      } else {
        const expr = session.inputExprNodes.get(key)
        if (expr !== undefined) {
          if (!entry.inputs) entry.inputs = {}
          entry.inputs[portName] = rewriteRefs(expr)
        }
      }
    }

    prog.instances[instName] = entry
  }

  // Output expressions — loadProgramDef reads these from process.outputs
  const processOutputs: Record<string, ExprNode> = {}
  for (const [outName, ref] of Object.entries(outputs)) {
    // Use nested_out referencing an internal instance via its alias name
    processOutputs[outName] = { op: 'nested_out', ref: ref.instance, output: ref.output }
  }
  prog.process = { outputs: processOutputs }

  // Set input_defaults from current wiring of exposed ports
  const inputDefaults: Record<string, ExprNode> = {}
  for (const [inputName, target] of Object.entries(inputs)) {
    const currentExpr = session.inputExprNodes.get(target)
    if (currentExpr !== undefined) {
      inputDefaults[inputName] = rewriteRefs(currentExpr)
    }
  }
  if (Object.keys(inputDefaults).length > 0) {
    prog.input_defaults = inputDefaults
  }

  return prog
}
