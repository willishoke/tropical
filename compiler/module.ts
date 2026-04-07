/**
 * Module spec builder DSL. Port of tropical/module.py.
 *
 * Module types are defined purely in TypeScript. The process function builds
 * expression trees (ExprNode) that are serialized to JSON and sent to FlatRuntime.
 * No C API calls — all module instantiation is TS-only.
 */

import {
  SignalExpr, ExprCoercible, coerce, ExprNode,
  inputExpr, registerExpr, nestedOutputExpr, delayValueExpr,
} from './expr.js'

// ---------- Internal build context stack ----------

interface DelayState {
  nodeId: number
  initVal: number
  updateExpr: SignalExpr
}

interface NestedCallState {
  nodeId: number
  moduleDef: ModuleDef
  callArgNodes: ExprNode[]
}

class _BuildContext {
  readonly inputCount: number
  readonly sampleRate: number
  readonly delayStates: DelayState[] = []
  readonly nestedCalls: NestedCallState[] = []
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

// ---------- Typed register helpers ----------

/** A register initialiser: either a bare value (backwards compatible) or { init, type }. */
export type RegInit = ValueCoercible | { init: ValueCoercible; type: string }

/** Infer a type name from a register initial value. */
function inferTypeName(v: ValueCoercible): string {
  if (typeof v === 'boolean') return 'bool'
  if (typeof v === 'number') return 'float'
  if (Array.isArray(v)) {
    if (v.length > 0 && Array.isArray(v[0])) return 'matrix'
    return 'array'
  }
  return 'float'
}

function isTypedRegInit(v: RegInit): v is { init: ValueCoercible; type: string } {
  return typeof v === 'object' && v !== null && !Array.isArray(v) && 'init' in v
}

/** Validate that a declared type is compatible with the initial value. */
function validateRegType(name: string, declaredType: string, initValue: ValueCoercible): void {
  const inferred = inferTypeName(initValue)
  const compatible =
    declaredType === inferred ||
    (declaredType === 'float' && inferred === 'int') ||
    (declaredType === 'int' && inferred === 'float')
  if (!compatible) {
    throw new TypeError(
      `Register '${name}': declared type '${declaredType}' is incompatible ` +
      `with initial value type '${inferred}'.`,
    )
  }
}

// ---------- feedback() DSL helper ----------

/** Descriptor returned by feedback(): bundles init, type, and update morphism. */
export interface FeedbackSpec {
  init: ValueCoercible
  type: string
  update: (current: SignalExpr) => SignalExpr
}

/**
 * Define a feedback register specification.  The morphism `f` maps the current
 * state to the next state.  Type is inferred from `init`.
 *
 * Usage inside defineModule:
 *   const phase = feedback(p => mod(add(p, dt), 1.0), 0.0)
 *   // In regs:     { phase: { init: phase.init, type: phase.type } }
 *   // In nextRegs: { phase: phase.update(regs.get('phase')) }
 */
export function feedback(
  f: (current: SignalExpr) => SignalExpr,
  init: ValueCoercible,
): FeedbackSpec {
  return { init, type: inferTypeName(init), update: f }
}

// ---------- Definition shape ----------

interface ModuleDef {
  typeName: string
  inputNames: string[]
  outputNames: string[]
  inputPortTypes: (string | undefined)[]
  outputPortTypes: (string | undefined)[]
  registerNames: string[]
  registerPortTypes: (string | undefined)[]
  registerInitValues: ValueCoercible[]
  sampleRate: number
  rawInputDefaults: Record<string, ExprCoercible>
  inputDefaults: (SignalExpr | null)[]
  delayInitValues: number[]
  // ExprNode trees (JSON-serializable) for each output and register update
  outputExprNodes: ExprNode[]
  registerExprNodes: (ExprNode | null)[]
  // Delay update expressions
  delayUpdateNodes: ExprNode[]
  // Nested module call info — each entry records the nested module's def and the call arguments
  nestedCalls: NestedCallDef[]
}

/** Captured metadata for a nested ModuleType.call() inside a defineModule body. */
export interface NestedCallDef {
  moduleDef: ModuleDef
  callArgNodes: ExprNode[]
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
        `Use .instantiateAs(name) for top-level instantiation.`
      )
    }
    return this._nestedCall(ctx, args)
  }

  /** Instantiate with an explicit instance name. Returns a TS-only ModuleInstance. */
  instantiateAs(name: string): ModuleInstance {
    return new ModuleInstance(this._def, name)
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

    // Allocate a node ID for this nested call and capture metadata
    const nodeId = ctx.nestedCalls.length
    ctx.nestedCalls.push({
      nodeId,
      moduleDef: d,
      callArgNodes: callArgs.map(a => a._node),
    })

    const outputs = d.outputNames.map((_, outId) => nestedOutputExpr(nodeId, outId))
    return outputs.length === 1 ? outputs[0] : outputs
  }
}

// ---------- ModuleInstance ----------

export class ModuleInstance {
  readonly _def: ModuleDef
  readonly name: string

  constructor(def: ModuleDef, name: string) {
    this._def = def
    this.name = name
  }

  get inputNames(): string[] { return this._def.inputNames }
  get outputNames(): string[] { return this._def.outputNames }
  get registerNames(): string[] { return this._def.registerNames }
  get typeName(): string { return this._def.typeName }

  inputPortType(idx: number): string | undefined { return this._def.inputPortTypes[idx] }
  outputPortType(idx: number): string | undefined { return this._def.outputPortTypes[idx] }
  registerPortType(idx: number): string | undefined { return this._def.registerPortTypes[idx] }

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
}

// ---------- defineModule ----------

/** A port spec is either a plain name string or an object with optional type annotation. */
export type PortSpec = string | { name: string; type?: string }

function _portName(s: PortSpec): string { return typeof s === 'string' ? s : s.name }
function _portType(s: PortSpec): string | undefined { return typeof s === 'string' ? undefined : s.type }

type RegsInit = Record<string, RegInit>

interface ProcessResult {
  outputs: Record<string, ExprCoercible>
  nextRegs: Record<string, ExprCoercible>
}

type ProcessFn = (inputs: SymbolMap, regs: SymbolMap) => ProcessResult

export function defineModule(
  name: string,
  inputs: PortSpec[],
  outputs: PortSpec[],
  regs: RegsInit,
  process: ProcessFn,
  sampleRate = 44100.0,
  inputDefaults?: Record<string, ExprCoercible>,
): ModuleType {
  const inputNames = inputs.map(_portName)
  const outputNames = outputs.map(_portName)
  const inputPortTypes = inputs.map(_portType)
  const outputPortTypes = outputs.map(_portType)

  // Parse regs (supports bare values or { init, type } objects)
  const regNames: string[] = []
  const regInitValues: ValueCoercible[] = []
  const regPortTypes: (string | undefined)[] = []

  for (const [regName, regSpec] of Object.entries(regs)) {
    regNames.push(regName)
    if (isTypedRegInit(regSpec)) {
      if (regSpec.type !== undefined) {
        validateRegType(regName, regSpec.type, regSpec.init)
      }
      regInitValues.push(regSpec.init)
      regPortTypes.push(regSpec.type)
    } else {
      regInitValues.push(regSpec as ValueCoercible)
      regPortTypes.push(undefined)
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
    regUpdateExprs[idx] = coerce(regVal)
  }

  const def: ModuleDef = {
    typeName: name,
    inputNames,
    outputNames,
    inputPortTypes,
    outputPortTypes,
    registerNames: regNames,
    registerPortTypes: regPortTypes,
    registerInitValues: regInitValues,
    sampleRate,
    rawInputDefaults: inputDefaults ?? {},
    inputDefaults: parsedDefaults,
    delayInitValues: ctx.delayStates.map(d => d.initVal),
    outputExprNodes: outputExprs.map(e => e._node),
    registerExprNodes: regUpdateExprs.map(e => e?._node ?? null),
    delayUpdateNodes: ctx.delayStates.map(d => d.updateExpr._node),
    nestedCalls: ctx.nestedCalls.map(nc => ({
      moduleDef: nc.moduleDef,
      callArgNodes: nc.callArgNodes,
    })),
  }

  return new ModuleType(def)
}

// ---------- PureFunction / definePureFunction ----------
//
// PureFunction is now a thin wrapper around a stateless ModuleType.
// Using ModuleType.call() means outputs are resolved as nested_output refs
// (one per call site, cached in resolveNestedOutputs) rather than as
// inline function/call nodes that expand exponentially on shared DAGs.

export class PureFunction {
  private _moduleType: ModuleType

  constructor(moduleType: ModuleType) {
    this._moduleType = moduleType
  }

  call(...args: ExprCoercible[]): SignalExpr | SignalExpr[] {
    return this._moduleType.call(...args)
  }
}

type PureFn = (inputs: SymbolMap) => Record<string, ExprCoercible>

export function definePureFunction(
  inputs: string[],
  outputs: string[],
  process: PureFn,
): PureFunction {
  const moduleType = defineModule(
    '_pure',
    inputs,
    outputs,
    {},
    (inp) => ({ outputs: process(inp), nextRegs: {} }),
  )
  return new PureFunction(moduleType)
}
