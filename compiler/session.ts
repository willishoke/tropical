/**
 * Session state, expression pretty-printer, JSON loading, and ProgramNode → ProgramDef builder.
 */

import {
  type SignalExpr, type ExprCoercible, coerce, type ExprNode, validateExpr,
} from './expr.js'
import {
  ProgramType, ProgramInstance,
  type ProgramDef, type NestedCall, type ValueCoercible, type Bounds,
} from './program_types.js'
import { Runtime } from './runtime/runtime.js'
import { loadProgramAsSession, type PortTypeDecl, type ProgramNode, type ProgramPortSpec, type ProgramTopLevel } from './program.js'
import { expandDeclGenerators } from './lower_arrays.js'
import { parseProgramV2 } from './schema.js'
import { Param, Trigger } from './runtime/param.js'
import {
  specializeProgramNode, specializationCacheKey, resolveTypeArgs,
  type RawTypeArgs, type ResolvedTypeArgs,
} from './specialize.js'
import {
  type PortType, type ScalarKind, type SumTypeMeta,
  Float, Int, Bool, Unit, ArrayType, StructType, SumType,
} from './term.js'

// ─────────────────────────────────────────────────────────────
// JSON schema types
// ─────────────────────────────────────────────────────────────

// ExprNode is defined in expr.ts and re-exported here for backward compatibility.
export type { ExprNode } from './expr.js'



export interface TypeDefFieldJSON {
  name: string
  /** Scalar kind: 'float', 'int', or 'bool'. */
  scalar_type: ScalarKind
}

export interface StructTypeDefJSON {
  kind: 'struct'
  name: string
  fields: TypeDefFieldJSON[]
}

export interface SumVariantJSON {
  name: string
  payload: TypeDefFieldJSON[]
}

export interface SumTypeDefJSON {
  kind: 'sum'
  name: string
  variants: SumVariantJSON[]
}

export interface AliasTypeDefJSON {
  kind: 'alias'
  name: string
  base: string
  bounds: [number | null, number | null]
}

export type TypeDefJSON = StructTypeDefJSON | SumTypeDefJSON | AliasTypeDefJSON


// ─────────────────────────────────────────────────────────────
// Session state (shared by patch load/save and MCP server)
// ─────────────────────────────────────────────────────────────

export interface SessionState {
  bufferLength: number
  dac: import('./runtime/audio.js').DAC | null  // lazy type import to avoid circular dep
  typeRegistry: Map<string, ProgramType>
  typeAliasRegistry: Map<string, { base: string; bounds: Bounds }>
  /** Registered sum types from `ports.type_defs` entries with kind === 'sum'.
   *  Keyed by name; values carry the variant + payload metadata used for bundle decomposition. */
  sumTypeRegistry: Map<string, SumTypeMeta>
  /** Registered struct types from `ports.type_defs` entries with kind === 'struct'.
   *  Keyed by name; values carry the field metadata. Currently retained for type-system
   *  completeness; struct values themselves have no expression-level operations. */
  structTypeRegistry: Map<string, { fields: Array<{ name: string; scalar: ScalarKind }> }>
  instanceRegistry: Map<string, ProgramInstance>
  graphOutputs: Array<{ instance: string; output: string }>
  paramRegistry: Map<string, Param>
  triggerRegistry: Map<string, Trigger>
  /** Canonical input wiring: key is `${instance}:${input}`, value is the ExprNode for round-trip save. */
  inputExprNodes: Map<string, ExprNode>  // key: `${instance}:${input}`
  /** FlatRuntime — all audio goes through this. */
  runtime: Runtime
  /** Thin proxy over runtime that matches the old Graph interface for tests and legacy callers. */
  graph: { primeJit(): void; process(): void; readonly outputBuffer: Float64Array; dispose(): void }
  /** On-demand type resolver (set by loadStdlib for lazy loading). */
  typeResolver?: (name: string) => ProgramType | undefined
  /** Monomorphized specializations of generic programs, keyed by `Type<k1=v1,k2=v2>`. */
  specializationCache: Map<string, ProgramType>
  /** ProgramNode templates for generic programs (pre-specialization). Keyed by type name.
   *  Only populated for programs declaring type_params. */
  genericTemplates: Map<string, import('./program.js').ProgramNode>
  /** Name counter for auto-generated instance names. */
  _nameCounters: Map<string, number>
}

export function makeSession(bufferLength = 512): SessionState {
  const runtime = new Runtime(bufferLength)
  return {
    bufferLength,
    dac: null,
    typeRegistry: new Map(),
    typeAliasRegistry: new Map(),
    sumTypeRegistry: new Map(),
    structTypeRegistry: new Map(),
    instanceRegistry: new Map(),
    graphOutputs: [],
    paramRegistry: new Map(),
    triggerRegistry: new Map(),
    inputExprNodes: new Map(),
    specializationCache: new Map(),
    genericTemplates: new Map(),
    runtime,
    graph: {
      primeJit: () => {},
      process: () => runtime.process(),
      get outputBuffer() { return runtime.outputBuffer },
      dispose: () => runtime.dispose(),
    },
    _nameCounters: new Map(),
  }
}

/** Generate a unique instance name from a type prefix. */
export function nextName(session: SessionState, prefix: string): string {
  const count = (session._nameCounters.get(prefix) ?? 0) + 1
  session._nameCounters.set(prefix, count)
  return `${prefix}${count}`
}

// ─────────────────────────────────────────────────────────────
// Op name sets (used by pretty-printer)
// ─────────────────────────────────────────────────────────────

const BINARY_OPS = new Set([
  'add', 'sub', 'mul', 'div', 'floor_div', 'mod', 'pow',
  'lt', 'lte', 'gt', 'gte', 'eq', 'neq',
  'bit_and', 'bit_or', 'bit_xor', 'lshift', 'rshift',
])

const UNARY_OPS = new Set([
  'neg', 'abs', 'sin', 'cos', 'exp', 'log', 'tanh', 'not', 'bit_not',
])

// ─────────────────────────────────────────────────────────────
// Module loader
// ─────────────────────────────────────────────────────────────
// Module loader
// ─────────────────────────────────────────────────────────────

/**
 * Convert a name-based JSON ExprNode to a slot-based ExprNode.
 * Pure tree walk — no side effects, no context stack.
 */
function slottifyExpr(
  node: ExprNode,
  inputNames: string[],
  regNames: string[],
  delayNameToId: Map<string, number>,
  nestedAliasToId: Map<string, number>,
  nestedAliasDef: Map<string, ProgramDef>,
): ExprNode {
  if (typeof node === 'number' || typeof node === 'boolean') return node
  if (Array.isArray(node)) return node.map(n => slottifyExpr(n, inputNames, regNames, delayNameToId, nestedAliasToId, nestedAliasDef))

  const obj = node as Record<string, unknown>
  const op = obj.op as string
  const recurse = (n: ExprNode) => slottifyExpr(n, inputNames, regNames, delayNameToId, nestedAliasToId, nestedAliasDef)

  if (op === 'input') {
    const name = obj.name as string | undefined
    if (name !== undefined) {
      const id = inputNames.indexOf(name)
      if (id === -1) throw new Error(`Unknown input '${name}'. Available: ${inputNames.join(', ')}`)
      return { op: 'input', id }
    }
    return node // already slot-based
  }

  if (op === 'reg') {
    const name = obj.name as string | undefined
    if (name !== undefined) {
      const id = regNames.indexOf(name)
      if (id === -1) throw new Error(`Unknown register '${name}'. Available: ${regNames.join(', ')}`)
      return { op: 'reg', id }
    }
    return node
  }

  if (op === 'delay_ref') {
    const id = obj.id as string
    const nodeId = delayNameToId.get(id)
    if (nodeId === undefined) throw new Error(`delay_ref: no delay with id '${id}'.`)
    return { op: 'delay_value', node_id: nodeId }
  }

  if (op === 'nested_out') {
    const ref = obj.ref as string
    const nodeId = nestedAliasToId.get(ref)
    if (nodeId === undefined) throw new Error(`nested_out: no nested program named '${ref}'.`)
    const nestedDef = nestedAliasDef.get(ref)!
    const output = obj.output as string | number
    const outputId = typeof output === 'number' ? output : nestedDef.outputNames.indexOf(output)
    if (outputId === -1) throw new Error(`nested_out: unknown output '${output}' on '${ref}'.`)
    return { op: 'nested_output', node_id: nodeId, output_id: outputId }
  }

  // Generic op with args — recurse
  if ('args' in obj) {
    const args = (obj.args as ExprNode[]).map(recurse)
    const result: Record<string, unknown> = {}
    for (const [k, v] of Object.entries(obj)) {
      if (k === 'args') continue
      result[k] = v
    }
    result.args = args
    return result as ExprNode
  }

  // Handle array-like containers that aren't in 'args'
  if (op === 'array') {
    const items = (obj.items as ExprNode[]).map(recurse)
    return { ...obj, items } as unknown as ExprNode
  }

  // Combinator forms with nested expr fields
  if (op === 'let') {
    const bind = obj.bind as Record<string, ExprNode>
    const converted: Record<string, ExprNode> = {}
    for (const [k, v] of Object.entries(bind)) converted[k] = recurse(v)
    return { ...obj, bind: converted, in: recurse(obj.in as ExprNode) } as unknown as ExprNode
  }
  if (op === 'generate' || op === 'chain' || op === 'iterate') {
    return { ...obj, ...(obj.init !== undefined ? { init: recurse(obj.init as ExprNode) } : {}), body: recurse(obj.body as ExprNode) } as unknown as ExprNode
  }
  if (op === 'fold' || op === 'scan') {
    return { ...obj, over: recurse(obj.over as ExprNode), init: recurse(obj.init as ExprNode), body: recurse(obj.body as ExprNode) } as unknown as ExprNode
  }
  if (op === 'map2') {
    return { ...obj, over: recurse(obj.over as ExprNode), body: recurse(obj.body as ExprNode) } as unknown as ExprNode
  }
  if (op === 'zip_with') {
    return { ...obj, a: recurse(obj.a as ExprNode), b: recurse(obj.b as ExprNode), body: recurse(obj.body as ExprNode) } as unknown as ExprNode
  }

  // Sum-type wiring expressions. Recurse into payload fields (tag) and
  // scrutinee + per-arm body (match). Per-arm bind names are leaf strings —
  // not ExprNodes — so they pass through unchanged.
  if (op === 'tag') {
    const payload = obj.payload as Record<string, ExprNode> | undefined
    if (payload === undefined) return node
    const converted: Record<string, ExprNode> = {}
    for (const [k, v] of Object.entries(payload)) converted[k] = recurse(v)
    return { ...obj, payload: converted } as unknown as ExprNode
  }
  if (op === 'match') {
    const arms = obj.arms as Record<string, { bind?: string | string[]; body: ExprNode }>
    const newArms: Record<string, { bind?: string | string[]; body: ExprNode }> = {}
    for (const [variant, arm] of Object.entries(arms)) {
      newArms[variant] = arm.bind === undefined
        ? { body: recurse(arm.body) }
        : { bind: arm.bind, body: recurse(arm.body) }
    }
    return { ...obj, scrutinee: recurse(obj.scrutinee as ExprNode), arms: newArms } as unknown as ExprNode
  }

  // Leaf ops (sample_rate, sample_index, binding, float, int, bool, matrix, etc.)
  return node
}

// ─────────────────────────────────────────────────────────────
// Bounded type aliases
// ─────────────────────────────────────────────────────────────

/** Built-in type aliases that map semantic names to a base type + bounds. */
export const BOUNDED_TYPE_ALIASES: Record<string, { base: string; bounds: Bounds }> = {
  signal:   { base: 'float', bounds: [-1, 1] },
  bipolar:  { base: 'float', bounds: [-1, 1] },
  unipolar: { base: 'float', bounds: [0, 1] },
  phase:    { base: 'float', bounds: [0, 1] },
  freq:     { base: 'float', bounds: [0, null] },
}

type AliasMap = Map<string, { base: string; bounds: Bounds }>

/** Resolve a type string to its base type (stripping alias). Checks user aliases first. */
export function resolveBaseType(typeStr: string | undefined, userAliases?: AliasMap): string | undefined {
  if (!typeStr) return typeStr
  const user = userAliases?.get(typeStr)
  if (user) return user.base
  if (typeStr in BOUNDED_TYPE_ALIASES) return BOUNDED_TYPE_ALIASES[typeStr].base
  return typeStr
}

/**
 * Convert a scalar or alias name to a PortType.
 *
 * Resolution order:
 *   1. Built-in scalar names ('float', 'int', 'bool', 'unit')
 *   2. Registered sum types (when `sumTypes` is provided)
 *   3. Fallback to `StructType(name)` for unknown names
 *
 * Sum types are preferred over the struct fallback because they describe wire
 * types that flatten to bundles of scalar wires; struct refs are an opaque
 * fallback for any other named type.
 */
function scalarNameToPortType(name: string, sumTypes?: ReadonlySet<string>): PortType {
  switch (name) {
    case 'float': return Float
    case 'int':   return Int
    case 'bool':  return Bool
    case 'unit':  return Unit
    default:
      if (sumTypes?.has(name)) return SumType(name)
      return StructType(name)
  }
}

/** Decode a structured port type declaration to a PortType, resolving aliases.
 *  Throws if the shape still contains an unresolved type_param ref — callers that
 *  use type_params must run `specializeProgramNode` first.
 *
 *  @param sumTypes Optional set of registered sum-type names. When provided, an
 *                  unknown type name that matches a registered sum resolves to
 *                  `SumType(name)` rather than the `StructType(name)` fallback.
 */
export function decodePortTypeDecl(
  t: PortTypeDecl,
  aliases: AliasMap | undefined,
  contextName: string,
  sumTypes?: ReadonlySet<string>,
): PortType {
  if (typeof t === 'string') {
    return scalarNameToPortType(resolveBaseType(t, aliases) ?? t, sumTypes)
  }
  const elemName = resolveBaseType(t.element, aliases) ?? t.element
  const elem = scalarNameToPortType(elemName, sumTypes)
  const shape = t.shape.map(dim => {
    if (typeof dim === 'number') return dim
    throw new Error(
      `${contextName}: array port type shape contains unresolved type_param '${dim.name}'. ` +
      `This should have been substituted at specialization time.`,
    )
  })
  return ArrayType(elem, shape)
}

/** Extract bounds from a port spec. Explicit bounds override alias bounds. Checks user aliases first.
 *  Only string type names can carry alias-derived bounds; structured array types do not. */
export function resolveBounds(
  spec: string | { name: string; type?: PortTypeDecl; bounds?: [number | null, number | null] },
  userAliases?: AliasMap,
): Bounds | null {
  if (typeof spec === 'string') return null
  if (spec.bounds) return spec.bounds
  if (!spec.type || typeof spec.type !== 'string') return null
  const user = userAliases?.get(spec.type)
  if (user) return user.bounds
  if (spec.type in BOUNDED_TYPE_ALIASES) return BOUNDED_TYPE_ALIASES[spec.type].bounds
  return null
}

// ─────────────────────────────────────────────────────────────
// Generic program resolution
// ─────────────────────────────────────────────────────────────

type ResolveSession = Pick<SessionState, 'typeRegistry' | 'specializationCache' | 'genericTemplates' | 'instanceRegistry' | 'paramRegistry' | 'triggerRegistry'> &
  Partial<Pick<SessionState, 'typeResolver' | 'typeAliasRegistry'>>

/**
 * Resolve a (baseName, type_args) pair to a concrete ProgramType.
 * Generic types monomorphize on demand, keyed by fully-resolved integer args.
 * Non-generic types reject non-empty type_args.
 */
export function resolveProgramType(
  session: ResolveSession,
  baseName: string,
  rawTypeArgs: RawTypeArgs | undefined,
  outerArgs: ResolvedTypeArgs | undefined,
): { type: ProgramType; typeArgs?: ResolvedTypeArgs } {
  const template = session.genericTemplates.get(baseName)
  if (template) {
    const resolved = resolveTypeArgs(rawTypeArgs, outerArgs, template.type_params, `instance of '${baseName}'`)
    const key = specializationCacheKey(baseName, resolved)
    const cached = session.specializationCache.get(key)
    if (cached) return { type: cached, typeArgs: resolved }
    const specialized = specializeProgramNode(template, resolved)
    specialized.name = key
    const type = loadProgramDef(specialized, session)
    session.specializationCache.set(key, type)
    return { type, typeArgs: resolved }
  }

  const type = session.typeRegistry.get(baseName) ?? session.typeResolver?.(baseName)
  if (!type) {
    const known = [
      ...session.typeRegistry.keys(),
      ...session.genericTemplates.keys(),
    ].join(', ')
    throw new Error(`Unknown program type '${baseName}'. Known: ${known || '(none)'}`)
  }
  if (rawTypeArgs && Object.keys(rawTypeArgs).length > 0) {
    throw new Error(`Program '${baseName}' does not declare type_params; got type_args: ${Object.keys(rawTypeArgs).join(', ')}`)
  }
  return { type }
}

// ─────────────────────────────────────────────────────────────
// ProgramNode → ProgramDef
// ─────────────────────────────────────────────────────────────

/**
 * Elaborate a v2 ProgramNode into a slot-indexed ProgramDef.
 *
 * Walks `body.decls` once to assign register/delay/instance IDs in source
 * order (preserving insertion-order semantics), then walks `body.assigns`
 * to attach output and next-state expressions. Nested `program_decl`
 * entries are ignored here — `loadProgramAsType` registers them before
 * this function runs.
 */
export function loadProgramDef(
  def: ProgramNode,
  session: Pick<SessionState, 'typeRegistry' | 'instanceRegistry' | 'paramRegistry' | 'triggerRegistry' | 'specializationCache' | 'genericTemplates'> & Partial<Pick<SessionState, 'typeAliasRegistry' | 'typeResolver'>>,
): ProgramType {
  const aliases = session.typeAliasRegistry
  const ports = def.ports ?? {}
  const inputSpecs  = ports.inputs  ?? []
  const outputSpecs = ports.outputs ?? []
  const inputNames  = inputSpecs.map(i => typeof i === 'string' ? i : i.name)
  const outputNames = outputSpecs.map(o => typeof o === 'string' ? o : o.name)
  const decodeType = (t: PortTypeDecl | undefined): PortType | undefined => {
    if (t === undefined) return undefined
    return decodePortTypeDecl(t, aliases, def.name)
  }
  const inputPortTypes  = inputSpecs.map(i => decodeType(typeof i === 'string' ? undefined : i.type))
  const outputPortTypes = outputSpecs.map(o => decodeType(typeof o === 'string' ? undefined : o.type))
  const inputBounds     = inputSpecs.map(s => resolveBounds(s, aliases))
  const outputBounds    = outputSpecs.map(s => resolveBounds(s, aliases))

  // ── First pass over decls: assign IDs in source order ──
  const body = expandDeclGenerators(def.body)
  const regNames: string[] = []
  const regInitValues: ValueCoercible[] = []
  const regPortTypes: (PortType | undefined)[] = []
  const delayNames: string[] = []
  const delayInitValues: number[] = []
  const delayUpdateByName = new Map<string, ExprNode>()
  const nestedAliases: string[] = []
  const nestedSpecByAlias = new Map<string, {
    program: string
    inputs?: Record<string, ExprNode>
    type_args?: Record<string, number | ExprNode>
  }>()

  for (const rawDecl of body.decls ?? []) {
    if (typeof rawDecl !== 'object' || rawDecl === null || Array.isArray(rawDecl))
      throw new Error(`${def.name}: block.decls entries must be objects`)
    const d = rawDecl as Record<string, unknown>
    const op = d.op as string

    if (op === 'reg_decl') {
      const name = d.name as string
      regNames.push(name)
      const init = d.init as unknown
      const typeDecl = d.type as PortTypeDecl | undefined
      if (typeof init === 'object' && init !== null && !Array.isArray(init) && 'zeros' in (init as Record<string, unknown>)) {
        const n = (init as { zeros: number }).zeros
        regInitValues.push(new Array(n).fill(0))
        regPortTypes.push(ArrayType(Float, [n]))
      } else if (typeDecl !== undefined) {
        regInitValues.push(init as ValueCoercible)
        regPortTypes.push(decodePortTypeDecl(typeDecl, aliases, def.name))
      } else {
        regInitValues.push(init as ValueCoercible)
        regPortTypes.push(undefined)
      }
    } else if (op === 'delay_decl') {
      const name = d.name as string
      delayNames.push(name)
      delayInitValues.push((d.init as number | undefined) ?? 0)
      if (d.update !== undefined) delayUpdateByName.set(name, d.update as ExprNode)
    } else if (op === 'instance_decl') {
      const alias = d.name as string
      nestedAliases.push(alias)
      nestedSpecByAlias.set(alias, {
        program: d.program as string,
        inputs: d.inputs as Record<string, ExprNode> | undefined,
        type_args: d.type_args as Record<string, number | ExprNode> | undefined,
      })
    } else if (op === 'program_decl') {
      // Registered by loadProgramAsType; nothing to do here.
    } else {
      throw new Error(`${def.name}: unexpected decl op '${op}' in block.decls`)
    }
  }

  const delayNameToId = new Map(delayNames.map((name, i) => [name, i]))
  const nestedAliasToId = new Map(nestedAliases.map((alias, i) => [alias, i]))
  const nestedAliasDef = new Map<string, ProgramDef>()
  const nestedCalls: NestedCall[] = []

  // Resolve nested instance types up-front, monomorphizing generics.
  // Callers above us (for a generic outer program) have already specialized
  // the outer frame, so spec.type_args here contains only concrete integers.
  const nestedResolved = new Map<string, ProgramType>()
  for (const alias of nestedAliases) {
    const spec = nestedSpecByAlias.get(alias)!
    const { type } = resolveProgramType(session, spec.program, spec.type_args as RawTypeArgs | undefined, undefined)
    nestedResolved.set(alias, type)
    nestedAliasDef.set(alias, type._def)
  }

  // Second pass: build call arg nodes (may reference any sibling via nested_out)
  for (const alias of nestedAliases) {
    const spec = nestedSpecByAlias.get(alias)!
    const type = nestedResolved.get(alias)!

    const callArgNodes: ExprNode[] = type._def.inputNames.map((name, idx) => {
      if (name in (spec.inputs ?? {})) {
        return slottifyExpr((spec.inputs ?? {})[name], inputNames, regNames, delayNameToId, nestedAliasToId, nestedAliasDef)
      }
      const defaultExpr = type._def.inputDefaults[idx]
      if (defaultExpr) return defaultExpr._node
      throw new Error(`Missing input '${name}' for nested module '${alias}'.`)
    })

    nestedCalls.push({ programDef: type._def, callArgNodes })
  }

  // ── Walk assigns: collect output_assign + next_update entries ──
  const outputExprByName = new Map<string, ExprNode>()
  const registerExprByName = new Map<string, ExprNode>()

  for (const rawAssign of def.body?.assigns ?? []) {
    if (typeof rawAssign !== 'object' || rawAssign === null || Array.isArray(rawAssign))
      throw new Error(`${def.name}: block.assigns entries must be objects`)
    const a = rawAssign as Record<string, unknown>
    const op = a.op as string

    if (op === 'output_assign') {
      outputExprByName.set(a.name as string, a.expr as ExprNode)
    } else if (op === 'next_update') {
      const target = a.target as { kind: string; name: string }
      if (target.kind === 'reg') {
        registerExprByName.set(target.name, a.expr as ExprNode)
      } else if (target.kind === 'delay') {
        delayUpdateByName.set(target.name, a.expr as ExprNode)
      } else {
        throw new Error(`${def.name}: next_update target.kind must be 'reg' or 'delay', got '${target.kind}'`)
      }
    } else {
      throw new Error(`${def.name}: unexpected assign op '${op}' in block.assigns`)
    }
  }

  // ── Convert delay update expressions ──
  const delayUpdateNodes = delayNames.map(name => {
    const update = delayUpdateByName.get(name)
    if (update === undefined) throw new Error(`${def.name}: delay '${name}' has no update expression`)
    return slottifyExpr(update, inputNames, regNames, delayNameToId, nestedAliasToId, nestedAliasDef)
  })

  // ── Convert output expressions ──
  const outputExprNodes = outputNames.map(name => {
    const node = outputExprByName.get(name)
    if (node === undefined) throw new Error(`${def.name}: Output '${name}' missing from block.assigns.`)
    return slottifyExpr(node, inputNames, regNames, delayNameToId, nestedAliasToId, nestedAliasDef)
  })

  // ── Convert register update expressions ──
  const registerExprNodes: (ExprNode | null)[] = regNames.map(name => {
    const node = registerExprByName.get(name)
    if (node === undefined) return null
    return slottifyExpr(node, inputNames, regNames, delayNameToId, nestedAliasToId, nestedAliasDef)
  })

  // ── Parse input defaults (carried on port specs in v2) ──
  const rawInputDefaults: Record<string, ExprNode> = {}
  const inputDefaults: (SignalExpr | null)[] = new Array(inputNames.length).fill(null)
  for (let i = 0; i < inputSpecs.length; i++) {
    const spec = inputSpecs[i] as string | ProgramPortSpec
    if (typeof spec === 'string') continue
    if (spec.default === undefined) continue
    const name = spec.name
    rawInputDefaults[name] = spec.default as ExprNode
    inputDefaults[i] = coerce(spec.default as ExprCoercible)
  }

  const programDef: ProgramDef = {
    typeName: def.name,
    inputNames,
    outputNames,
    inputPortTypes,
    outputPortTypes,
    registerNames: regNames,
    registerPortTypes: regPortTypes,
    registerInitValues: regInitValues,
    sampleRate: def.sample_rate ?? 44100.0,
    rawInputDefaults,
    inputDefaults,
    delayInitValues,
    outputExprNodes,
    registerExprNodes,
    delayUpdateNodes,
    nestedCalls,
    breaksCycles: def.breaks_cycles ?? false,
    inputBounds,
    outputBounds,
  }

  return new ProgramType(programDef)
}

// ─────────────────────────────────────────────────────────────
// Program file I/O
// ─────────────────────────────────────────────────────────────

/** Parse a tropical_program_2 file, split it into program + top-level metadata. */
export function normalizeProgramFile(
  raw: { schema?: string; [k: string]: unknown },
): { node: ProgramNode; topLevel: ProgramTopLevel } {
  if (raw.schema !== 'tropical_program_2') {
    throw new Error(`Unknown schema '${raw.schema}'. Expected 'tropical_program_2'.`)
  }
  const v2 = parseProgramV2(raw) as Record<string, unknown> & {
    schema: 'tropical_program_2'
    params?: ProgramTopLevel['params']
    audio_outputs?: ProgramTopLevel['audio_outputs']
    config?: ProgramTopLevel['config']
  }
  const { schema: _schema, params, audio_outputs, config, ...progFields } = v2
  void _schema
  const node: ProgramNode = { op: 'program', ...progFields } as unknown as ProgramNode
  const topLevel: ProgramTopLevel = {}
  if (params !== undefined)        topLevel.params        = params
  if (audio_outputs !== undefined) topLevel.audio_outputs = audio_outputs
  if (config !== undefined)        topLevel.config        = config
  return { node, topLevel }
}

/** Wrap a v2 program node + top-level metadata as a serializable v2 file. */
export function v2NodeToFile(
  node: ExprNode,
  topLevel: ProgramTopLevel = {},
): { schema: 'tropical_program_2'; [k: string]: unknown } {
  if (typeof node !== 'object' || node === null || Array.isArray(node))
    throw new Error('v2NodeToFile: expected program object')
  const p = node as Record<string, unknown>
  const { op: _op, ...fields } = p
  void _op
  const file: Record<string, unknown> = { schema: 'tropical_program_2', ...fields }
  if (topLevel.params !== undefined)        file.params        = topLevel.params
  if (topLevel.audio_outputs !== undefined) file.audio_outputs = topLevel.audio_outputs
  if (topLevel.config !== undefined)        file.config        = topLevel.config
  return file as { schema: 'tropical_program_2'; [k: string]: unknown }
}

// ─────────────────────────────────────────────────────────────
// Expression pretty-printer
// ─────────────────────────────────────────────────────────────

/** Infix symbols for binary ops. Ops in BINARY_OPS but absent here fall back to `op(l, r)`. */
const BINARY_INFIX: Record<string, string> = {
  add: '+', sub: '-', mul: '*', div: '/', floor_div: '//', mod: '%', pow: '**', matmul: '@',
  lt: '<', lte: '<=', gt: '>', gte: '>=', eq: '==', neq: '!=',
  bit_and: '&', bit_or: '|', bit_xor: '^', lshift: '<<', rshift: '>>',
}

/** Prefix symbols for unary ops. Ops in UNARY_OPS but absent here use `op(x)` notation. */
const UNARY_PREFIX: Record<string, string> = { neg: '-' }

/**
 * Render an ExprNode as a human-readable string.
 * Refs appear as `Module.output`; math appears as infix expressions.
 * instanceRegistry is used to resolve numeric output indices to port names.
 */
export function prettyExpr(
  node: ExprNode,
  instanceRegistry: Map<string, ProgramInstance>,
): string {
  if (typeof node === 'number') return String(node)
  if (typeof node === 'boolean') return String(node)
  if (Array.isArray(node)) return `[${node.map(n => prettyExpr(n, instanceRegistry)).join(', ')}]`

  const n = node as { op: string; [k: string]: unknown }
  const op = n.op
  const args = (n.args as ExprNode[] | undefined) ?? []

  if (op === 'ref') {
    const mod = n.instance as string
    const out = n.output
    const inst = instanceRegistry.get(mod)
    const outName = inst && typeof out === 'number' ? (inst.outputNames[out] ?? String(out)) : String(out)
    return `${mod}.${outName}`
  }
  if (op === 'input')     return `input(${n.name})`
  if (op === 'param')     return `param(${n.name})`
  if (op === 'trigger')   return `trigger(${n.name})`
  if (op === 'binding')   return `$${n.name}`
  if (op === 'sample_rate')  return 'sample_rate'
  if (op === 'sample_index') return 'sample_index'
  if (op === 'float' || op === 'int')  return String(n.value)
  if (op === 'bool')  return String(n.value)

  if (BINARY_OPS.has(op)) {
    const sym = BINARY_INFIX[op]
    const l = prettyExpr(args[0], instanceRegistry)
    const r = prettyExpr(args[1], instanceRegistry)
    return sym ? `(${l} ${sym} ${r})` : `${op}(${l}, ${r})`
  }
  if (UNARY_OPS.has(op)) {
    const pfx = UNARY_PREFIX[op]
    const x = prettyExpr(args[0], instanceRegistry)
    return pfx ? `${pfx}${x}` : `${op}(${x})`
  }

  if (op === 'clamp')  return `clamp(${args.map(a => prettyExpr(a, instanceRegistry)).join(', ')})`
  if (op === 'select') return `select(${args.map(a => prettyExpr(a, instanceRegistry)).join(', ')})`
  if (op === 'index')  return `${prettyExpr(args[0], instanceRegistry)}[${prettyExpr(args[1], instanceRegistry)}]`
  if (op === 'array_set') return `array_set(${args.map(a => prettyExpr(a, instanceRegistry)).join(', ')})`
  if (op === 'array') return `[${(n.items as ExprNode[]).map(i => prettyExpr(i, instanceRegistry)).join(', ')}]`
  if (op === 'matrix') return `matrix(${JSON.stringify(n.rows)})`
  if (op === 'delay') return `delay(${prettyExpr(args[0], instanceRegistry)}, ${n.init ?? 0})`
  if (op === 'delay_ref') return `delay_ref(${n.id})`
  if (op === 'nested_out') return `${n.ref}.${n.output}`
  if (op === 'tag') {
    const payload = n.payload as Record<string, ExprNode> | undefined
    const fields = payload === undefined
      ? ''
      : `{${Object.entries(payload).map(([k, v]) => `${k}: ${prettyExpr(v, instanceRegistry)}`).join(', ')}}`
    return `${n.type}::${n.variant}${fields}`
  }
  if (op === 'match') {
    const arms = n.arms as Record<string, { bind?: string | string[]; body: ExprNode }>
    const armStrs = Object.entries(arms).map(([variant, arm]) => {
      const bindStr = arm.bind === undefined
        ? ''
        : ` bind ${typeof arm.bind === 'string' ? arm.bind : `(${arm.bind.join(', ')})`}`
      return `${variant}${bindStr}: ${prettyExpr(arm.body, instanceRegistry)}`
    })
    return `match(${prettyExpr(n.scrutinee as ExprNode, instanceRegistry)}, type=${n.type}){${armStrs.join(', ')}}`
  }
  // Should never reach here given the finite op set, but keep a safe fallback
  throw new Error(`prettyExpr: unhandled op '${op}'`)
}

// ─────────────────────────────────────────────────────────────
// Patch loader
// ─────────────────────────────────────────────────────────────

/**
 * Load any supported JSON schema into a session.
 * Detects schema version and delegates to the appropriate loader.
 */
export function loadJSON(json: { schema: string; [k: string]: unknown }, session: SessionState): void {
  const { node, topLevel } = normalizeProgramFile(json)
  loadProgramAsSession(node, topLevel, session)
}

