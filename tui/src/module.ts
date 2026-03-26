/**
 * Module spec builder DSL. Port of egress/module.py.
 *
 * Key differences from Python:
 * - No operator overloading; arithmetic uses named functions from expr.ts
 * - define_module takes a process callback instead of using a context manager
 * - delay() reads from a module-level context stack (same pattern as Python,
 *   works fine since Node.js is single-threaded during synchronous callbacks)
 * - ModuleType.call() is used for nested calls inside process bodies;
 *   ModuleType.instantiate(graph) is used for top-level instantiation
 */

import * as b from './bindings.js'
import {
  SignalExpr, ExprCoercible, coerce,
  inputExpr, registerExpr, nestedOutputExpr, delayValueExpr, refExpr,
} from './expr.js'
import type { Graph } from './graph.js'

// ---------- Internal build context stack ----------

interface DelayState {
  nodeId: number
  initVal: number
  updateExpr: SignalExpr
}

interface NestedEntry {
  nodeId: number
  nestedH: unknown
}

class _BuildContext {
  readonly inputCount: number
  readonly sampleRate: number
  readonly delayStates: DelayState[] = []
  readonly nestedModules: NestedEntry[] = []
  private _delayNodeCounter = 0

  constructor(inputCount: number, sampleRate: number) {
    this.inputCount = inputCount
    this.sampleRate = sampleRate
  }

  allocateDelayNodeId(): number {
    return this._delayNodeCounter++
  }

  addDelay(nodeId: number, initVal: number, updateExpr: SignalExpr): void {
    this.delayStates.push({ nodeId, initVal, updateExpr })
  }
}

const _contextStack: _BuildContext[] = []

function _currentContext(): _BuildContext | null {
  return _contextStack.at(-1) ?? null
}

// ---------- SymbolMap ----------

export class SymbolMap {
  private _kind: string
  private _slots: Map<string, number>

  constructor(kind: string, names: string[]) {
    this._kind = kind
    this._slots = new Map(names.map((n, i) => [n, i]))
  }

  get(name: string): SignalExpr {
    const slotId = this._slots.get(name)
    if (slotId === undefined) {
      throw new Error(`Unknown ${this._kind} symbol '${name}'.`)
    }
    return this._kind === 'input' ? inputExpr(slotId) : registerExpr(slotId)
  }

  names(): string[] {
    return [...this._slots.keys()]
  }

  size(): number {
    return this._slots.size
  }
}

// ---------- ArrayStateSpec ----------

export class ArrayStateSpec {
  constructor(
    public readonly inputName: string,
    public readonly init: number = 0.0,
  ) {}
}

/** Declare a dynamic array register whose size mirrors a named input. */
export function arrayState(inputName: string, init = 0.0): ArrayStateSpec {
  return new ArrayStateSpec(inputName, init)
}

// ---------- delay() DSL function ----------

/** Create a one-sample delay. Only valid inside defineModule process callbacks. */
export function delay(value: ExprCoercible, init = 0.0): SignalExpr {
  const ctx = _currentContext()
  if (!ctx) {
    throw new Error('delay() may only be called inside defineModule process bodies.')
  }
  const sig = coerce(value)
  const nodeId = ctx.allocateDelayNodeId()
  ctx.addDelay(nodeId, init, sig)
  return delayValueExpr(nodeId)
}

// ---------- Value helpers ----------

export type ValueCoercible = boolean | number | number[] | number[][]

function scalarValueHandle(v: boolean | number): unknown {
  if (typeof v === 'boolean') return b.check(b.egress_value_bool(v), 'value_bool')
  if (Number.isInteger(v))   return b.check(b.egress_value_int(v),   'value_int')
  return b.check(b.egress_value_float(v), 'value_float')
}

export function valueHandle(v: ValueCoercible): unknown {
  if (typeof v === 'boolean' || typeof v === 'number') return scalarValueHandle(v)
  if (Array.isArray(v)) {
    if (v.length > 0 && Array.isArray(v[0])) {
      // Matrix
      const rows = v as number[][]
      const nRows = rows.length
      const nCols = rows[0].length
      const items = rows.flat().map(x => scalarValueHandle(x as number))
      const h = b.check(b.egress_value_matrix(items, nRows, nCols), 'value_matrix')
      items.forEach(ih => b.egress_value_free(ih))
      return h
    } else {
      // Flat array
      const items = (v as number[]).map(x => scalarValueHandle(x))
      const h = b.check(b.egress_value_array(items, items.length), 'value_array')
      items.forEach(ih => b.egress_value_free(ih))
      return h
    }
  }
  throw new TypeError(`Cannot convert ${typeof v} to egress value`)
}

// ---------- Definition shape ----------

interface RegisterSpec {
  bodyH: unknown | null
  initH: unknown
  arraySpec: { sourceInputId: number; init: number } | null
}

interface DelaySpecHandle {
  nodeId: number
  initH: unknown
  updateH: unknown
}

interface ModuleDef {
  typeName: string
  inputNames: string[]
  outputNames: string[]
  sampleRate: number
  rawInputDefaults: Record<string, ExprCoercible>
  inputDefaults: (SignalExpr | null)[]
  outputExprHandles: unknown[]
  registerSpecs: RegisterSpec[]
  delaySpecHandles: DelaySpecHandle[]
  nestedSpecHandles: NestedEntry[]
  // Keep JS objects alive to protect their koffi handles from FinalizationRegistry
  _liveOutputExprs: SignalExpr[]
  _liveRegUpdateExprs: (SignalExpr | null)[]
  _liveDelayUpdateExprs: SignalExpr[]
  _liveValueHandles: unknown[]   // egress_value_t — managed manually
}

// ---------- ModuleType ----------

export class ModuleType {
  readonly _def: ModuleDef

  constructor(def: ModuleDef) {
    this._def = def
  }

  get name(): string { return this._def.typeName }

  /**
   * Call inside a defineModule process body to nest this module.
   * Returns a single SignalExpr (one output) or an array (multiple outputs).
   */
  call(...args: ExprCoercible[]): SignalExpr | SignalExpr[] {
    const ctx = _currentContext()
    if (!ctx) {
      throw new Error(
        `ModuleType.call() used outside a defineModule body. ` +
        `Use .instantiate(graph) for top-level instantiation.`
      )
    }
    return this._nestedCall(ctx, args)
  }

  /** Top-level instantiation onto a Graph (auto-generates a name). */
  instantiate(graph: Graph): ModuleInstance {
    return this._instantiate(graph)
  }

  /** Instantiate with an explicit instance name (used by the MCP server). */
  instantiateAs(graph: Graph, name: string): ModuleInstance {
    const d = this._def
    const specH = this._buildSpec()
    try {
      const ok = graph.addModule(name, specH)
      if (!ok) throw new Error(`Failed to add module '${name}' to graph.`)
    } finally {
      b.egress_module_spec_free(specH)
    }
    for (let i = 0; i < d.inputDefaults.length; i++) {
      const def = d.inputDefaults[i]
      if (def !== null) graph.setInputExpr(name, i, def)
    }
    return new ModuleInstance(d, graph, name)
  }

  private _instantiate(graph: Graph): ModuleInstance {
    const d = this._def
    const name = graph.nextName(d.typeName)
    const specH = this._buildSpec()
    try {
      const ok = graph.addModule(name, specH)
      if (!ok) throw new Error(`Failed to add module '${name}' to graph.`)
    } finally {
      b.egress_module_spec_free(specH)
    }

    // Set input defaults
    for (let i = 0; i < d.inputDefaults.length; i++) {
      const def = d.inputDefaults[i]
      if (def !== null) graph.setInputExpr(name, i, def)
    }

    return new ModuleInstance(d, graph, name)
  }

  private _nestedCall(ctx: _BuildContext, args: ExprCoercible[]): SignalExpr | SignalExpr[] {
    const d = this._def
    const { inputNames, inputDefaults } = d

    if (args.length > inputNames.length) {
      throw new TypeError(`Module call expects at most ${inputNames.length} arguments.`)
    }

    // Coerce provided args; fill missing with defaults
    const callArgs: SignalExpr[] = args.map(a => coerce(a))
    for (let i = callArgs.length; i < inputNames.length; i++) {
      const def = inputDefaults[i]
      if (def !== null) {
        callArgs.push(def)
      } else {
        throw new TypeError(`Missing argument for module input '${inputNames[i]}'.`)
      }
    }

    const nestedH = b.check(
      b.egress_nested_spec_new(inputNames.length, d.sampleRate),
      'nested_spec_new',
    )
    const nodeId = b.egress_nested_spec_node_id(nestedH) as number

    for (const arg of callArgs) {
      b.egress_nested_spec_add_input_expr(nestedH, arg._h)
    }
    for (const exprH of d.outputExprHandles) {
      b.egress_nested_spec_add_output(nestedH, exprH)
    }
    for (const { bodyH, initH, arraySpec } of d.registerSpecs) {
      if (arraySpec !== null) {
        b.egress_nested_spec_add_register_array(nestedH, arraySpec.sourceInputId, initH)
      } else {
        b.egress_nested_spec_add_register(nestedH, bodyH, initH)
      }
    }
    for (const { initH, updateH } of d.delaySpecHandles) {
      b.egress_nested_spec_add_delay_state(nestedH, initH, updateH)
    }
    for (const { nestedH: innerH } of d.nestedSpecHandles) {
      b.egress_nested_spec_add_nested(nestedH, innerH)
    }

    ctx.nestedModules.push({ nodeId, nestedH })

    const outputs = d.outputNames.map((_, outId) => nestedOutputExpr(nodeId, outId))
    return outputs.length === 1 ? outputs[0] : outputs
  }

  _buildSpec(): unknown {
    const d = this._def
    const specH = b.check(
      b.egress_module_spec_new(d.inputNames.length, d.sampleRate),
      'module_spec_new',
    )
    for (const exprH of d.outputExprHandles) {
      b.egress_module_spec_add_output(specH, exprH)
    }
    for (const { bodyH, initH, arraySpec } of d.registerSpecs) {
      if (arraySpec !== null) {
        b.egress_module_spec_add_register_array(specH, arraySpec.sourceInputId, initH)
      } else {
        b.egress_module_spec_add_register(specH, bodyH, initH)
      }
    }
    for (const { initH, updateH } of d.delaySpecHandles) {
      b.egress_module_spec_add_delay_state(specH, initH, updateH)
    }
    for (const { nestedH } of d.nestedSpecHandles) {
      b.egress_module_spec_add_nested(specH, nestedH)
    }
    return specH
  }
}

// ---------- ModuleInstance ----------

export class ModuleInstance {
  readonly _def: ModuleDef
  readonly _graph: Graph
  readonly name: string

  constructor(def: ModuleDef, graph: Graph, name: string) {
    this._def = def
    this._graph = graph
    this.name = name
  }

  get inputNames(): string[] { return this._def.inputNames }
  get outputNames(): string[] { return this._def.outputNames }
  get typeName(): string { return this._def.typeName }

  inputIndex(name: string): number {
    const idx = this._def.inputNames.indexOf(name)
    if (idx === -1) throw new Error(`Unknown input '${name}' on module '${this.name}'.`)
    return idx
  }

  outputIndex(name: string): number {
    const idx = this._def.outputNames.indexOf(name)
    if (idx === -1) throw new Error(`Unknown output '${name}' on module '${this.name}'.`)
    return idx
  }

  input(name: string): InputPort {
    return new InputPort(this._graph, this.name, this.inputIndex(name))
  }

  output(name: string): OutputPort {
    return new OutputPort(this._graph, this.name, this.outputIndex(name))
  }

  setInput(name: string, value: ExprCoercible): void {
    this._graph.setInputExpr(this.name, this.inputIndex(name), coerce(value))
  }
}

// ---------- Port types ----------

export class OutputPort {
  constructor(
    private _graph: Graph,
    public readonly moduleName: string,
    public readonly outputId: number,
  ) {}

  /** Return a ref expression pointing to this output. */
  asExpr(): SignalExpr {
    return refExpr(this.moduleName, this.outputId)
  }
}

export class InputPort {
  constructor(
    private _graph: Graph,
    public readonly moduleName: string,
    public readonly inputId: number,
  ) {}

  getExpr(): SignalExpr | null {
    return this._graph.getInputExpr(this.moduleName, this.inputId)
  }

  assign(value: ExprCoercible): void {
    this._graph.setInputExpr(this.moduleName, this.inputId, coerce(value))
  }
}

// ---------- defineModule ----------

type RegsInit = Record<string, ValueCoercible | ArrayStateSpec>

interface ProcessResult {
  outputs: Record<string, ExprCoercible>
  nextRegs: Record<string, ExprCoercible>
}

type ProcessFn = (inputs: SymbolMap, regs: SymbolMap) => ProcessResult

export function defineModule(
  name: string,
  inputs: string[],
  outputs: string[],
  regs: RegsInit,
  process: ProcessFn,
  sampleRate = 44100.0,
  inputDefaults?: Record<string, ExprCoercible>,
): ModuleType {
  const inputNames = [...inputs]
  const outputNames = [...outputs]

  // Parse regs
  const regNames: string[] = []
  const regInitValues: (ValueCoercible)[] = []
  const regArraySpecs: ({ sourceInputId: number; init: number } | null)[] = []

  for (const [regName, regInit] of Object.entries(regs)) {
    regNames.push(regName)
    if (regInit instanceof ArrayStateSpec) {
      const srcId = inputNames.indexOf(regInit.inputName)
      if (srcId === -1) throw new Error(`arrayState input '${regInit.inputName}' not found.`)
      regInitValues.push(0.0)
      regArraySpecs.push({ sourceInputId: srcId, init: regInit.init })
    } else {
      regInitValues.push(regInit as ValueCoercible)
      regArraySpecs.push(null)
    }
  }

  // Parse input defaults
  const parsedDefaults: (SignalExpr | null)[] = new Array(inputNames.length).fill(null)
  if (inputDefaults) {
    for (const [k, v] of Object.entries(inputDefaults)) {
      const idx = inputNames.indexOf(k)
      if (idx === -1) throw new Error(`Input default references unknown input '${k}'.`)
      parsedDefaults[idx] = v !== null ? coerce(v) : null
    }
  }

  // Build symbol maps
  const inputsMap = new SymbolMap('input', inputNames)
  const regsMap = new SymbolMap('register', regNames)

  // Run process inside a build context
  const ctx = new _BuildContext(inputNames.length, sampleRate)
  _contextStack.push(ctx)
  let result: ProcessResult
  try {
    result = process(inputsMap, regsMap)
  } finally {
    _contextStack.pop()
  }

  // Collect output expressions
  const outputExprs: SignalExpr[] = new Array(outputNames.length).fill(null)
  for (const [outName, outVal] of Object.entries(result.outputs)) {
    const idx = outputNames.indexOf(outName)
    if (idx === -1) throw new Error(`Output '${outName}' not in outputs list.`)
    outputExprs[idx] = coerce(outVal)
  }
  for (let i = 0; i < outputNames.length; i++) {
    if (outputExprs[i] === null) throw new Error(`Output '${outputNames[i]}' was not assigned.`)
  }

  // Collect register update expressions
  const regUpdateExprs: (SignalExpr | null)[] = new Array(regNames.length).fill(null)
  for (const [regName, regVal] of Object.entries(result.nextRegs)) {
    const idx = regNames.indexOf(regName)
    if (idx === -1) throw new Error(`nextRegs references unknown register '${regName}'.`)
    if (regArraySpecs[idx] === null) {
      regUpdateExprs[idx] = coerce(regVal)
    }
  }

  // Build C value handles for register initial values
  const liveValueHandles: unknown[] = []
  const registerSpecs: RegisterSpec[] = []

  for (let i = 0; i < regNames.length; i++) {
    const arraySpec = regArraySpecs[i]
    const initVal = arraySpec !== null ? arraySpec.init : (regInitValues[i] as ValueCoercible)
    const initH = valueHandle(initVal as ValueCoercible)
    liveValueHandles.push(initH)

    if (arraySpec !== null) {
      registerSpecs.push({ bodyH: null, initH, arraySpec })
    } else {
      registerSpecs.push({ bodyH: regUpdateExprs[i]?._h ?? null, initH, arraySpec: null })
    }
  }

  // Build delay spec handles
  const delaySpecHandles: DelaySpecHandle[] = []
  for (const { nodeId, initVal, updateExpr } of ctx.delayStates) {
    const initH = valueHandle(initVal)
    liveValueHandles.push(initH)
    delaySpecHandles.push({ nodeId, initH, updateH: updateExpr._h })
  }

  const def: ModuleDef = {
    typeName: name,
    inputNames,
    outputNames,
    sampleRate,
    rawInputDefaults: inputDefaults ?? {},
    inputDefaults: parsedDefaults,
    outputExprHandles: outputExprs.map(e => e._h),
    registerSpecs,
    delaySpecHandles,
    nestedSpecHandles: ctx.nestedModules.map(({ nodeId, nestedH }) => ({ nodeId, nestedH })),
    _liveOutputExprs: outputExprs,
    _liveRegUpdateExprs: regUpdateExprs,
    _liveDelayUpdateExprs: ctx.delayStates.map(d => d.updateExpr),
    _liveValueHandles: liveValueHandles,
  }

  return new ModuleType(def)
}

// ---------- PureFunction / definePureFunction ----------

export class PureFunction {
  private _def: {
    inputNames: string[]
    outputNames: string[]
    outputExprHandles: unknown[]
    _liveOutputExprs: SignalExpr[]
  }

  constructor(def: PureFunction['_def']) {
    this._def = def
  }

  call(...args: ExprCoercible[]): SignalExpr | SignalExpr[] {
    const d = this._def
    if (args.length !== d.inputNames.length) {
      throw new TypeError(`PureFunction expects ${d.inputNames.length} arguments.`)
    }
    const callArgs = args.map(a => coerce(a))
    const argHandles = callArgs.map(a => a._h)

    const outputs: SignalExpr[] = []
    for (const bodyH of d.outputExprHandles) {
      const fnH = b.check(b.egress_expr_function(d.inputNames.length, bodyH), 'expr_function')
      const callH = b.check(b.egress_expr_call(fnH, argHandles, argHandles.length), 'expr_call')
      b.egress_expr_free(fnH)
      outputs.push(SignalExpr.fromHandle(callH))
    }
    return outputs.length === 1 ? outputs[0] : outputs
  }
}

type PureFn = (inputs: SymbolMap) => Record<string, ExprCoercible>

export function definePureFunction(
  inputs: string[],
  outputs: string[],
  process: PureFn,
): PureFunction {
  const inputNames = [...inputs]
  const outputNames = [...outputs]

  const inputsMap = new SymbolMap('input', inputNames)
  const result = process(inputsMap)

  const outputExprs: SignalExpr[] = new Array(outputNames.length).fill(null)
  for (const [outName, outVal] of Object.entries(result)) {
    const idx = outputNames.indexOf(outName)
    if (idx === -1) throw new Error(`Output '${outName}' not in outputs list.`)
    outputExprs[idx] = coerce(outVal)
  }
  for (let i = 0; i < outputNames.length; i++) {
    if (outputExprs[i] === null) throw new Error(`Output '${outputNames[i]}' was not assigned.`)
  }

  return new PureFunction({
    inputNames,
    outputNames,
    outputExprHandles: outputExprs.map(e => e._h),
    _liveOutputExprs: outputExprs,
  })
}
