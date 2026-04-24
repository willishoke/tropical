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
import { loadProgramDef, resolveProgramType } from './session.js'
import { applyFlatPlan } from './apply_plan.js'
import { Param, Trigger } from './runtime/param.js'
import { ProgramType } from './program_types.js'
import { exprDependencies, reachableInstances, buildDependencyGraph, topologicalSort } from './compiler.js'
import type { RawTypeArgs } from './specialize.js'
import { Float, portTypeEqual, type PortType } from './term.js'
import type { Bounds } from './program_types.js'

// ─────────────────────────────────────────────────────────────
// Program schema
// ─────────────────────────────────────────────────────────────

/** Compile-time shape dimension: a concrete int or a reference to an
 *  outer type parameter to be substituted during specialization. */
export type ShapeDim = number | { op: 'type_param'; name: string }

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

/** Port declaration: scalar/alias name or `{name, type?, default?, bounds?}`. */
export interface ProgramPortSpec {
  name: string
  type?: PortTypeDecl
  default?: ExprNode
  bounds?: [number | null, number | null]
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

/** Top-level session metadata carried alongside a root program in a v2 file. */
export interface ProgramTopLevel {
  params?: Array<{ name: string; value?: number; time_const?: number; type?: 'param' | 'trigger' }>
  audio_outputs?: Array<
    | { instance: string; output: string | number }
    | { expr: ExprNode }
  >
  config?: { buffer_length?: number; sample_rate?: number }
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
  gateable?: boolean
  gate_input?: ExprNode
}> {
  for (const d of prog.body?.decls ?? []) {
    if (typeof d !== 'object' || d === null || Array.isArray(d)) continue
    const obj = d as Record<string, unknown>
    if (obj.op !== 'instance_decl') continue
    yield {
      name: obj.name as string,
      program: obj.program as string,
      inputs: obj.inputs as Record<string, ExprNode> | undefined,
      type_args: obj.type_args as Record<string, number | ExprNode> | undefined,
      gateable: obj.gateable as boolean | undefined,
      gate_input: obj.gate_input as ExprNode | undefined,
    }
  }
}

/** Iterate program_decl entries in a ProgramNode's body. */
function* programDecls(prog: ProgramNode): Iterable<{ name: string; program: ProgramNode }> {
  for (const d of prog.body?.decls ?? []) {
    if (typeof d !== 'object' || d === null || Array.isArray(d)) continue
    const obj = d as Record<string, unknown>
    if (obj.op !== 'program_decl') continue
    yield { name: obj.name as string, program: obj.program as ProgramNode }
  }
}

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
  session.triggerRegistry.clear()
  session.inputExprNodes.clear()
  session._nameCounters.clear()
  session.typeAliasRegistry.clear()

  // Register type aliases from type_defs before anything else
  for (const td of prog.ports?.type_defs ?? []) {
    if (td.kind === 'alias') {
      session.typeAliasRegistry.set(td.name, { base: td.base, bounds: td.bounds })
    }
  }

  // Register inline program definitions (loadProgramAsType handles registration)
  for (const sub of programDecls(prog)) {
    loadProgramAsType({ ...sub.program, name: sub.name }, session)
  }

  // Create params and triggers before instances (instances may reference them)
  for (const p of topLevel.params ?? []) {
    if (p.type === 'trigger') {
      session.triggerRegistry.set(p.name, new Trigger())
    } else {
      session.paramRegistry.set(p.name, new Param(p.value ?? 0.0, p.time_const ?? 0.005))
    }
  }

  // Instantiate programs
  for (const inst of instanceDecls(prog)) {
    const { type, typeArgs } = resolveProgramType(session, inst.program, inst.type_args as RawTypeArgs | undefined, undefined)
    const instance = type.instantiateAs(inst.name, { baseTypeName: inst.program, typeArgs })
    if (inst.gateable) {
      if (inst.gate_input === undefined)
        throw new Error(`Instance '${inst.name}' has gateable=true but no gate_input expression.`)
      validateExpr(inst.gate_input, `${inst.name}.__gate__`)
      instance.gateable = true
      instance.gateInput = inst.gate_input
    }
    session.instanceRegistry.set(instance.name, instance)

    // Populate wiring from instance inputs
    if (inst.inputs) {
      for (const [input, expr] of Object.entries(inst.inputs)) {
        validateExpr(expr, `${inst.name}.${input}`)
        session.inputExprNodes.set(`${inst.name}:${input}`, expr)
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
  for (const out of topLevel.audio_outputs ?? []) {
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
 * Load a ProgramNode as a ProgramType (registerable in typeRegistry).
 * Programs with inline `program_decl` entries get their subprograms registered first.
 *
 * Generic programs (with `type_params`) are stored in `genericTemplates`
 * instead of being eagerly compiled; they materialize on instantiation via
 * `resolveProgramType`. For non-generic programs the ProgramType is both
 * registered in `typeRegistry` and returned.
 */
export function loadProgramAsType(
  prog: ProgramNode,
  session: Pick<SessionState, 'typeRegistry' | 'instanceRegistry' | 'paramRegistry' | 'triggerRegistry' | 'specializationCache' | 'genericTemplates'> & Partial<Pick<SessionState, 'typeAliasRegistry' | 'typeResolver'>>,
): ProgramType | undefined {
  // Register type aliases from type_defs before processing subprograms
  if (session.typeAliasRegistry) {
    for (const td of prog.ports?.type_defs ?? []) {
      if (td.kind === 'alias') {
        session.typeAliasRegistry.set(td.name, { base: td.base, bounds: td.bounds })
      }
    }
  }

  // Register inline subprograms first (each handles its own registration)
  for (const sub of programDecls(prog)) {
    loadProgramAsType({ ...sub.program, name: sub.name }, session)
  }

  // Generic: stash the template, defer compilation to instantiation time.
  if (prog.type_params && Object.keys(prog.type_params).length > 0) {
    session.genericTemplates.set(prog.name, prog)
    return undefined
  }

  const type = loadProgramDef(prog, session)
  session.typeRegistry.set(prog.name, type)
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
  for (const p of topLevel.params ?? []) {
    if (session.paramRegistry.has(p.name) || session.triggerRegistry.has(p.name))
      throw new Error(`merge collision: param/trigger '${p.name}' already exists.`)
  }

  // Register type aliases from type_defs (additive)
  for (const td of prog.ports?.type_defs ?? []) {
    if (td.kind === 'alias') {
      session.typeAliasRegistry.set(td.name, { base: td.base, bounds: td.bounds })
    }
  }

  // Register inline program definitions (loadProgramAsType handles registration)
  for (const sub of programDecls(prog)) {
    loadProgramAsType({ ...sub.program, name: sub.name }, session)
  }

  // Create params and triggers
  for (const p of topLevel.params ?? []) {
    if (p.type === 'trigger') {
      session.triggerRegistry.set(p.name, new Trigger())
    } else {
      session.paramRegistry.set(p.name, new Param(p.value ?? 0.0, p.time_const ?? 0.005))
    }
  }

  // Instantiate programs
  for (const inst of instanceDecls(prog)) {
    const { type, typeArgs } = resolveProgramType(session, inst.program, inst.type_args as RawTypeArgs | undefined, undefined)
    const instance = type.instantiateAs(inst.name, { baseTypeName: inst.program, typeArgs })
    if (inst.gateable) {
      if (inst.gate_input === undefined)
        throw new Error(`Instance '${inst.name}' has gateable=true but no gate_input expression.`)
      validateExpr(inst.gate_input, `${inst.name}.__gate__`)
      instance.gateable = true
      instance.gateInput = inst.gate_input
    }
    session.instanceRegistry.set(instance.name, instance)

    // Populate wiring from instance inputs
    if (inst.inputs) {
      for (const [input, expr] of Object.entries(inst.inputs)) {
        validateExpr(expr, `${inst.name}.${input}`)
        session.inputExprNodes.set(`${inst.name}:${input}`, expr)
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
  for (const out of topLevel.audio_outputs ?? []) {
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
import { loadStdlibFromMap } from './stdlib_loader.js'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

/**
 * Load all stdlib program files (tropical_program_2) into a type registry.
 * Reads `../stdlib/*.json` from disk, then delegates to `loadStdlibFromMap`
 * for registration. Browser builds use `loadStdlibFromMap` directly with a
 * bundled JSON map (see compiler/stdlib_bundled.ts).
 */
export function loadStdlib(
  target: Map<string, ProgramType> | Pick<SessionState, 'typeRegistry' | 'instanceRegistry' | 'paramRegistry' | 'triggerRegistry' | 'specializationCache' | 'genericTemplates'>,
): void {
  const stdlibDir = join(__dirname, '../stdlib')
  const files = readdirSync(stdlibDir).filter(f => f.endsWith('.json')).sort()

  const rawByName = new Map<string, unknown>()
  for (const file of files) {
    const path = join(stdlibDir, file)
    const raw = JSON.parse(readFileSync(path, 'utf-8')) as { schema?: string; name?: string }
    if (typeof raw.name !== 'string') throw new Error(`${path}: missing 'name' field`)
    rawByName.set(raw.name, raw)
  }

  loadStdlibFromMap(target, rawByName)
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
  for (const [name, inst] of session.instanceRegistry) {
    const entry: Record<string, unknown> = { op: 'instance_decl', name, program: inst.typeName }
    if (inst.typeArgs) entry.type_args = inst.typeArgs
    if (inst.gateable) {
      entry.gateable = true
      if (inst.gateInput !== undefined) entry.gate_input = inst.gateInput
    }

    // Merge wiring for this instance
    const inputs: Record<string, ExprNode> = {}
    for (const portName of inst.inputNames) {
      const key = `${name}:${portName}`
      const expr = session.inputExprNodes.get(key)
      if (expr !== undefined) inputs[portName] = expr
    }
    if (Object.keys(inputs).length > 0) entry.inputs = inputs
    decls.push(entry as ExprNode)
  }

  const node: ProgramNode = {
    op: 'program',
    name: 'patch',
    body: { op: 'block', decls },
  }

  const topLevel: ProgramTopLevel = {}
  if (session.graphOutputs.length) {
    topLevel.audio_outputs = session.graphOutputs.map(o => ({
      instance: o.instance, output: o.output,
    }))
  }

  const params: NonNullable<ProgramTopLevel['params']> = []
  for (const [name, p] of session.paramRegistry) {
    params.push({ name, value: p.value, time_const: 0.005 })
  }
  for (const [name] of session.triggerRegistry) {
    params.push({ name, type: 'trigger' })
  }
  if (params.length) topLevel.params = params

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

  // Gather port type + bounds metadata from the source instances so exported
  // inputs/outputs round-trip through re-parse.
  const isDefaultPortType = (t: PortType | undefined): boolean =>
    t === undefined || portTypeEqual(t, Float)
  const boundsProvided = (b: Bounds | null | undefined): b is Bounds =>
    !!b && (b[0] !== null || b[1] !== null)

  const portTypeToDecl = (t: PortType): PortTypeDecl => {
    switch (t.tag) {
      case 'scalar': return t.scalar
      case 'array': {
        if (t.element.tag !== 'scalar') {
          throw new Error(`export: cannot serialize nested array element type (${t.element.tag})`)
        }
        return { kind: 'array', element: t.element.scalar, shape: t.shape }
      }
      case 'struct': return t.name
      case 'sum': return t.name
      case 'unit': return 'unit'
      case 'product':
        throw new Error(`export: product port types cannot be serialized`)
    }
  }

  // Collect per-port defaults from current wiring of exposed ports
  const inputDefaults: Record<string, ExprNode> = {}
  for (const [inputName, target] of Object.entries(inputs)) {
    const currentExpr = session.inputExprNodes.get(target)
    if (currentExpr !== undefined) {
      inputDefaults[inputName] = rewriteRefs(currentExpr)
    }
  }

  const inputEntries: Array<string | ProgramPortSpec> = inputNames.map(inputName => {
    const target = inputs[inputName]
    const [instName, portName] = target.split(':')
    const inst = session.instanceRegistry.get(instName)!
    const idx = inst.inputIndex(portName)
    const pt = inst.inputPortType(idx)
    const bnds = inst._def.inputBounds[idx]
    const dflt = inputDefaults[inputName]
    const entry: ProgramPortSpec = { name: inputName }
    if (!isDefaultPortType(pt)) entry.type = portTypeToDecl(pt!)
    if (boundsProvided(bnds)) entry.bounds = bnds
    if (dflt !== undefined) entry.default = dflt
    return entry.type === undefined && entry.bounds === undefined && entry.default === undefined
      ? inputName
      : entry
  })

  const outputEntries: Array<string | ProgramPortSpec> = outputNames.map(outName => {
    const ref = outputs[outName]
    const inst = session.instanceRegistry.get(ref.instance)!
    const idx = inst.outputIndex(ref.output)
    const pt = inst.outputPortType(idx)
    const bnds = inst._def.outputBounds[idx]
    const entry: ProgramPortSpec = { name: outName }
    if (!isDefaultPortType(pt)) entry.type = portTypeToDecl(pt!)
    if (boundsProvided(bnds)) entry.bounds = bnds
    return entry.type === undefined && entry.bounds === undefined ? outName : entry
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
      op: 'instance_decl',
      name: instName,
      program: inst.typeName,
    }
    if (inst.typeArgs) entry.type_args = inst.typeArgs
    if (inst.gateable) {
      entry.gateable = true
      if (inst.gateInput !== undefined) entry.gate_input = rewriteRefs(inst.gateInput)
    }

    // Copy wiring, rewriting exposed ports to {op:"input", name:...}
    // and ref→nested_out for sibling instances
    const instInputs: Record<string, ExprNode> = {}
    for (const portName of inst.inputNames) {
      const key = `${instName}:${portName}`
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
      op: 'output_assign',
      name: outName,
      expr: { op: 'nested_out', ref: ref.instance, output: ref.output },
    } as ExprNode)
  }

  return {
    op: 'program',
    name,
    ports: { inputs: inputEntries, outputs: outputEntries },
    body: { op: 'block', decls, assigns },
  }
}
