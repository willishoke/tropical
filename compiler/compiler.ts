/**
 * compiler.ts — Compile a patch into a well-typed categorical Term.
 *
 * Phase 1 of the compilation pipeline:
 *   CompilerInput → compilePatch() → CompiledPatch (containing a Term)
 *
 * The compiler:
 * 1. Converts module instances to morphism/trace terms
 * 2. Builds a dependency graph from input expressions
 * 3. Topologically sorts modules (Kahn's with level grouping)
 * 4. Detects feedback cycles (Tarjan's SCC)
 * 5. Assembles a well-typed Term: tensor within levels, compose between levels,
 *    wiring morphisms route signals through the pipeline
 */

import type { ExprNode, SessionState } from './patch'
import {
  type PortType, type Term, type MorphismBody, type StateInit,
  Float, Int, Bool, Unit, ArrayType,
  product, morphism, compose, tensor, trace, tensorAll, composeAll, id,
} from './term'
import { TypeRegistry } from './type_registry.js'
import { inferType } from './type_check'

// ─────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────

export class CompilerError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'CompilerError'
  }
}

// ─────────────────────────────────────────────────────────────
// Port type conversion
// ─────────────────────────────────────────────────────────────

/**
 * Convert a string type annotation (from ModuleDef) to a PortType.
 *
 * Supports:
 *   'float', 'int', 'bool', 'unit'       — scalars
 *   'float[4]', 'float[4,4]'             — arrays with static shapes
 *   'array'                                — float[1] (legacy compat)
 *   'matrix'                               — float[1,1] (legacy compat)
 *   anything else                          — named type (elaborated via registry)
 *
 * If a TypeRegistry is provided, named types are elaborated to structural
 * PortTypes. Otherwise, named types produce a product([]) placeholder.
 */
export function portTypeFromString(s: string | undefined, registry?: TypeRegistry): PortType {
  if (s === undefined) return Float

  // Check for array syntax: type[d1,d2,...] e.g. float[8], int[4,4]
  const arrayMatch = s.match(/^(\w+)\[([^\]]+)\]$/)
  if (arrayMatch) {
    const elementType = portTypeFromString(arrayMatch[1], registry)
    const shape = arrayMatch[2].split(',').map(d => {
      const n = parseInt(d.trim(), 10)
      if (isNaN(n) || n <= 0) throw new Error(`Invalid array dimension '${d.trim()}' in type '${s}'`)
      return n
    })
    return ArrayType(elementType, shape)
  }

  switch (s) {
    case 'float': return Float
    case 'int':   return Int
    case 'bool':  return Bool
    case 'unit':  return Unit
    case 'array': return ArrayType(Float, [1])  // legacy: bare 'array' — shape from init value
    case 'matrix': return ArrayType(Float, [1, 1])  // legacy: bare 'matrix'
    default: {
      if (registry && registry.has(s)) {
        return registry.toPortType(s)
      }
      // Unknown named type without registry — treat as single float (backward compat)
      return Float
    }
  }
}

// ─────────────────────────────────────────────────────────────
// Module info
// ─────────────────────────────────────────────────────────────

/** Structural type information for a module instance — pure data, no C handles. */
export interface ModuleInfo {
  name: string
  typeName: string
  inputNames: string[]
  outputNames: string[]
  registerNames: string[]
  inputTypes: PortType[]
  outputTypes: PortType[]
  registerTypes: PortType[]
}

/** The subset of ModuleDef the compiler reads. */
interface ModuleDefLike {
  typeName: string
  inputNames: string[]
  outputNames: string[]
  registerNames: string[]
  inputPortTypes: (string | undefined)[]
  outputPortTypes: (string | undefined)[]
  registerPortTypes: (string | undefined)[]
}

/** Extract ModuleInfo from a module definition. */
export function extractModuleInfo(name: string, def: ModuleDefLike): ModuleInfo {
  return {
    name,
    typeName: def.typeName,
    inputNames: [...def.inputNames],
    outputNames: [...def.outputNames],
    registerNames: [...def.registerNames],
    inputTypes: def.inputPortTypes.map(portTypeFromString),
    outputTypes: def.outputPortTypes.map(portTypeFromString),
    registerTypes: def.registerPortTypes.map(portTypeFromString),
  }
}

// ─────────────────────────────────────────────────────────────
// Expression dependency extraction
// ─────────────────────────────────────────────────────────────

/** Extract module names referenced via 'ref' ops in an ExprNode tree. */
export function exprDependencies(node: ExprNode): Set<string> {
  const deps = new Set<string>()
  walkExprRefs(node, deps)
  return deps
}

function walkExprRefs(node: ExprNode, deps: Set<string>): void {
  if (typeof node === 'number' || typeof node === 'boolean') return
  if (Array.isArray(node)) {
    for (const item of node) walkExprRefs(item, deps)
    return
  }
  const n = node as { op: string; [k: string]: unknown }
  if (n.op === 'ref') {
    deps.add(n.module as string)
    return
  }
  // Walk standard args
  for (const arg of ((n.args as ExprNode[]) ?? [])) walkExprRefs(arg, deps)
  // Walk special nested expression forms
  if (n.op === 'construct_struct')
    for (const f of ((n.fields as ExprNode[]) ?? [])) walkExprRefs(f, deps)
  if (n.op === 'field_access')
    walkExprRefs(n.struct_expr as ExprNode, deps)
  if (n.op === 'construct_variant')
    for (const p of ((n.payload as ExprNode[]) ?? [])) walkExprRefs(p, deps)
  if (n.op === 'match_variant') {
    walkExprRefs(n.scrutinee as ExprNode, deps)
    for (const b of ((n.branches as ExprNode[]) ?? [])) walkExprRefs(b, deps)
  }
}

// ─────────────────────────────────────────────────────────────
// Dependency graph
// ─────────────────────────────────────────────────────────────

/**
 * Build a dependency graph: module → set of modules it depends on.
 * Dependencies come from 'ref' ops in inputExprNodes.
 */
export function buildDependencyGraph(
  moduleNames: Iterable<string>,
  inputExprNodes: Map<string, ExprNode>,
): Map<string, Set<string>> {
  const nameSet = new Set(moduleNames)
  const graph = new Map<string, Set<string>>()
  for (const name of nameSet) graph.set(name, new Set())

  for (const [key, expr] of inputExprNodes) {
    const moduleName = key.split(':')[0]
    if (!nameSet.has(moduleName)) continue
    const refs = exprDependencies(expr)
    const deps = graph.get(moduleName)!
    for (const ref of refs) {
      if (nameSet.has(ref) && ref !== moduleName) deps.add(ref)
    }
  }

  return graph
}

// ─────────────────────────────────────────────────────────────
// Topological sort (Kahn's algorithm with level grouping)
// ─────────────────────────────────────────────────────────────

export interface TopologicalResult {
  /** Modules in topological execution order. */
  order: string[]
  /** Modules grouped by parallel execution level. */
  levels: string[][]
  /** True if all modules were sorted (false means cycles exist). */
  complete: boolean
}

export function topologicalSort(
  deps: Map<string, Set<string>>,
): TopologicalResult {
  const inDegree = new Map<string, number>()
  const consumers = new Map<string, Set<string>>()

  for (const name of deps.keys()) {
    inDegree.set(name, 0)
    consumers.set(name, new Set())
  }
  for (const [consumer, producers] of deps) {
    inDegree.set(consumer, producers.size)
    for (const producer of producers) {
      consumers.get(producer)?.add(consumer)
    }
  }

  const order: string[] = []
  const levels: string[][] = []
  let queue = [...inDegree.entries()]
    .filter(([_, d]) => d === 0)
    .map(([n]) => n)
    .sort()

  while (queue.length > 0) {
    levels.push([...queue])
    order.push(...queue)
    const next: string[] = []
    for (const node of queue) {
      for (const c of consumers.get(node) ?? []) {
        const d = inDegree.get(c)! - 1
        inDegree.set(c, d)
        if (d === 0) next.push(c)
      }
    }
    queue = next.sort()
  }

  return { order, levels, complete: order.length === deps.size }
}

// ─────────────────────────────────────────────────────────────
// Tarjan's SCC — cycle detection
// ─────────────────────────────────────────────────────────────

/** Find strongly connected components. Cycles are SCCs with more than one member. */
export function tarjanSCC(deps: Map<string, Set<string>>): string[][] {
  let idx = 0
  const indices = new Map<string, number>()
  const lowlinks = new Map<string, number>()
  const onStack = new Set<string>()
  const stack: string[] = []
  const sccs: string[][] = []

  function visit(v: string): void {
    indices.set(v, idx)
    lowlinks.set(v, idx)
    idx++
    stack.push(v)
    onStack.add(v)

    for (const w of deps.get(v) ?? []) {
      if (!indices.has(w)) {
        visit(w)
        lowlinks.set(v, Math.min(lowlinks.get(v)!, lowlinks.get(w)!))
      } else if (onStack.has(w)) {
        lowlinks.set(v, Math.min(lowlinks.get(v)!, indices.get(w)!))
      }
    }

    if (lowlinks.get(v) === indices.get(v)) {
      const scc: string[] = []
      let w: string
      do {
        w = stack.pop()!
        onStack.delete(w)
        scc.push(w)
      } while (w !== v)
      sccs.push(scc)
    }
  }

  for (const v of deps.keys()) {
    if (!indices.has(v)) visit(v)
  }

  return sccs
}

// ─────────────────────────────────────────────────────────────
// Module → Term conversion
// ─────────────────────────────────────────────────────────────

/**
 * Convert a module's type info to a Term.
 *
 * Stateless (no registers): morphism(inputs → outputs)
 * Stateful (has registers):  trace(S, morphism(inputs⊗S → outputs⊗S))
 *
 * The morphism body is a stub referencing the module name — actual
 * process expressions live in the ModuleDef and are resolved when
 * generating execution plans.
 */
export function moduleToTerm(info: ModuleInfo): Term {
  const dom = product(info.inputTypes)
  const cod = product(info.outputTypes)

  const body: MorphismBody = {
    tag: 'expr',
    inputNames: info.inputNames,
    outputExprs: {},
  }

  if (info.registerNames.length === 0) {
    return morphism(info.name, dom, cod, body)
  }

  // Stateful: trace(S, morphism(I⊗S → O⊗S))
  const stateType = product(info.registerTypes)
  const bodyDom = product([...info.inputTypes, ...info.registerTypes])
  const bodyCod = product([...info.outputTypes, ...info.registerTypes])
  const innerBody: MorphismBody = {
    tag: 'expr',
    inputNames: [...info.inputNames, ...info.registerNames],
    outputExprs: {},
  }
  const inner = morphism(info.name, bodyDom, bodyCod, innerBody)
  const init: StateInit = info.registerNames.length === 1 ? 0 : new Array(info.registerNames.length).fill(0) as number[]
  return trace(stateType, init, inner)
}

// ─────────────────────────────────────────────────────────────
// Signal environment
// ─────────────────────────────────────────────────────────────

interface Wire {
  module: string
  output: string
  type: PortType
}

interface SignalEnv {
  wires: Wire[]
}

function envType(env: SignalEnv): PortType {
  return product(env.wires.map(w => w.type))
}

// ─────────────────────────────────────────────────────────────
// Wiring morphism construction
// ─────────────────────────────────────────────────────────────

/**
 * Build the initial wiring for level 0: Unit → level_0_inputs.
 * All inputs are constants, params, or special forms (no module refs).
 */
function buildInitialWiring(
  levelInfos: ModuleInfo[],
  inputExprNodes: Map<string, ExprNode>,
): Term {
  const inputTypes: PortType[] = []
  const wiringExprs: Record<string, ExprNode> = {}

  for (const info of levelInfos) {
    for (let i = 0; i < info.inputNames.length; i++) {
      inputTypes.push(info.inputTypes[i])
      const key = `${info.name}:${info.inputNames[i]}`
      wiringExprs[`${info.name}.${info.inputNames[i]}`] = inputExprNodes.get(key) ?? 0
    }
  }

  const to = product(inputTypes)
  return morphism('wiring', Unit, to, {
    tag: 'expr',
    inputNames: [],
    outputExprs: wiringExprs,
  })
}

/**
 * Build a wiring morphism for subsequent levels:
 *   envType → (envType ⊗ level_inputs)
 *
 * The passthrough comes first (preserving env order), then new module inputs.
 * This matches the tensor order: tensor(id(env), level_modules).
 */
function buildWiringMorphism(
  env: SignalEnv,
  levelInfos: ModuleInfo[],
  inputExprNodes: Map<string, ExprNode>,
): Term {
  const wiringExprs: Record<string, ExprNode> = {}

  // Passthrough: identity on each env signal
  for (const w of env.wires) {
    wiringExprs[`_pass.${w.module}.${w.output}`] = {
      op: 'ref', module: w.module, output: w.output,
    }
  }

  // Level inputs: from expressions or defaults
  const inputTypes: PortType[] = []
  for (const info of levelInfos) {
    for (let i = 0; i < info.inputNames.length; i++) {
      inputTypes.push(info.inputTypes[i])
      const key = `${info.name}:${info.inputNames[i]}`
      wiringExprs[`${info.name}.${info.inputNames[i]}`] = inputExprNodes.get(key) ?? 0
    }
  }

  const from = envType(env)
  const to = product([...env.wires.map(w => w.type), ...inputTypes])

  return morphism('wiring', from, to, {
    tag: 'expr',
    inputNames: env.wires.map(w => `${w.module}.${w.output}`),
    outputExprs: wiringExprs,
  })
}

/**
 * Build an output projection morphism: envType → graphOutputType.
 * Extracts the specified graph outputs from the accumulated signal environment.
 */
function buildOutputProjection(
  env: SignalEnv,
  graphOutputs: Array<{ module: string; output: string }>,
  modules: Map<string, ModuleInfo>,
): Term {
  if (graphOutputs.length === 0) {
    return morphism('output', envType(env), Unit, {
      tag: 'expr', inputNames: [], outputExprs: {},
    })
  }

  const outputTypes: PortType[] = []
  const projExprs: Record<string, ExprNode> = {}

  for (const { module, output } of graphOutputs) {
    const info = modules.get(module)
    if (!info) throw new CompilerError(`Output references unknown module '${module}'`)
    const idx = info.outputNames.indexOf(output)
    if (idx === -1) throw new CompilerError(`Output references unknown output '${module}.${output}'`)
    outputTypes.push(info.outputTypes[idx])
    projExprs[`${module}.${output}`] = { op: 'ref', module, output }
  }

  return morphism('output', envType(env), product(outputTypes), {
    tag: 'expr',
    inputNames: env.wires.map(w => `${w.module}.${w.output}`),
    outputExprs: projExprs,
  })
}

// ─────────────────────────────────────────────────────────────
// Compiler input/output
// ─────────────────────────────────────────────────────────────

/** Pure-data input to the compiler (no C handles). */
export interface CompilerInput {
  modules: Map<string, ModuleInfo>
  inputExprNodes: Map<string, ExprNode>
  graphOutputs: Array<{ module: string; output: string }>
}

/** The result of compilation. */
export interface CompiledPatch {
  /** The compiled term representing the entire patch. */
  term: Term
  /** Module info, keyed by instance name. */
  modules: Map<string, ModuleInfo>
  /** Topological execution levels. */
  levels: string[][]
  /** Detected feedback cycles (each is a list of module names). */
  cycles: string[][]
  /** Input wiring expressions (carried through for plan generation). */
  inputExprNodes: Map<string, ExprNode>
  /** Graph outputs (carried through for plan generation). */
  graphOutputs: Array<{ module: string; output: string }>
}

// ─────────────────────────────────────────────────────────────
// Main compilation
// ─────────────────────────────────────────────────────────────

/**
 * Compile a patch into a well-typed categorical Term.
 *
 * The term structure for a 3-level acyclic patch:
 *
 *   compose(
 *     compose(wiring₀, level₀_modules),         — Unit → level₀_outputs
 *     compose(wiring₁, tensor(id_env, level₁)),  — → env ⊗ level₁_outputs
 *     compose(wiring₂, tensor(id_env, level₂)),  — → env ⊗ level₂_outputs
 *     output_projection                           — → graph_outputs
 *   )
 *
 * Throws CompilerError on feedback cycles or invalid references.
 */
export function compilePatch(input: CompilerInput): CompiledPatch {
  const { modules, inputExprNodes, graphOutputs } = input

  // Handle empty patch
  if (modules.size === 0) {
    return {
      term: id(Unit),
      modules,
      levels: [],
      cycles: [],
      inputExprNodes,
      graphOutputs,
    }
  }

  // 1. Build dependency graph
  const deps = buildDependencyGraph(modules.keys(), inputExprNodes)

  // 2. Detect cycles
  const sccs = tarjanSCC(deps)
  const cycles = sccs.filter(scc => scc.length > 1)
  if (cycles.length > 0) {
    throw new CompilerError(
      `Feedback cycles detected (auto-trace not yet implemented): ` +
      cycles.map(c => c.join(' \u2194 ')).join('; ')
    )
  }

  // 3. Topological sort
  const { levels, complete } = topologicalSort(deps)
  if (!complete) {
    throw new CompilerError('Topological sort incomplete — hidden cycle?')
  }

  // 4. Assemble term level by level
  let env: SignalEnv = { wires: [] }
  const steps: Term[] = []

  for (let i = 0; i < levels.length; i++) {
    const level = levels[i]
    const levelInfos = level.map(n => modules.get(n)!)

    // Build module terms for this level
    const moduleTerms = levelInfos.map(info => moduleToTerm(info))
    const levelModules = moduleTerms.length === 1 ? moduleTerms[0] : tensorAll(moduleTerms)

    if (i === 0) {
      // First level: wiring from Unit → level_inputs, then modules
      const wiring = buildInitialWiring(levelInfos, inputExprNodes)
      steps.push(compose(wiring, levelModules))
    } else {
      // Subsequent levels: wiring from env → (env ⊗ level_inputs), then tensor(id_env, modules)
      const wiring = buildWiringMorphism(env, levelInfos, inputExprNodes)
      const eT = envType(env)
      const withPassthrough = eT.tag === 'unit'
        ? levelModules
        : tensor(id(eT), levelModules)
      steps.push(compose(wiring, withPassthrough))
    }

    // Update env: append this level's outputs
    for (const info of levelInfos) {
      for (let j = 0; j < info.outputNames.length; j++) {
        env.wires.push({
          module: info.name,
          output: info.outputNames[j],
          type: info.outputTypes[j],
        })
      }
    }
  }

  // 5. Output projection
  if (graphOutputs.length > 0) {
    steps.push(buildOutputProjection(env, graphOutputs, modules))
  }

  // 6. Compose all steps
  const term = steps.length === 0
    ? id(Unit)
    : steps.length === 1
      ? steps[0]
      : composeAll(steps)

  // 7. Verify the compiled term type-checks
  try {
    inferType(term)
  } catch (e) {
    throw new CompilerError(
      `Compiled term failed type check: ${e instanceof Error ? e.message : String(e)}`
    )
  }

  return { term, modules, levels, cycles, inputExprNodes, graphOutputs }
}

// ─────────────────────────────────────────────────────────────
// SessionState adapter
// ─────────────────────────────────────────────────────────────

/** Build a CompilerInput from a SessionState. */
export function compilerInputFromSession(session: SessionState): CompilerInput {
  const modules = new Map<string, ModuleInfo>()
  for (const [name, inst] of session.instanceRegistry) {
    modules.set(name, extractModuleInfo(name, inst._def))
  }
  return {
    modules,
    inputExprNodes: session.inputExprNodes,
    graphOutputs: session.graphOutputs,
  }
}
