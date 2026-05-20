/**
 * Session state, expression pretty-printer, JSON loading, and generic-program
 * resolution. The strata pipeline (compiler/ir/) handles ProgramNode →
 * ResolvedProgram → ProgramDef; this module owns the session-state shell
 * and the small library of port-type utilities used by both strata and
 * pretty-printing.
 */

import { type ExprNode } from './expr.js'
import {
  type Compiled, type Instance,
  outputNames, outputPortType,
  // Note: re-exports below pick up Compiled/Instance + helpers for
  // callers that already import from session.js.
} from './program_types.js'
import { Runtime } from './runtime/runtime.js'
import { loadProgramAsSession, type PortTypeDecl, type ProgramNode, type ProgramTopLevel } from './program.js'
import { parseProgramV2 } from './schema.js'
import { Param, Trigger } from './runtime/param.js'
import {
  specializationCacheKey, resolveTypeArgs,
  type RawTypeArgs, type ResolvedTypeArgs,
} from './specialize.js'
import {
  type PortType, type ScalarKind, type SumTypeMeta,
  Float, Int, Bool, Unit, ArrayType, StructType, SumType,
} from './term.js'
import type { PortType as IRPortType, ScalarKind as IRScalarKind } from './ir/nodes.js'
import { programTypeFromResolved } from './ir/strata.js'
import type { TypeParamDecl } from './ir/nodes.js'
import {
  type PortRef, type WireKey, type SlotKey,
  type InstanceName,
  wireKey, slotKey,
} from './ir/branded_names.js'

// ─────────────────────────────────────────────────────────────
// JSON schema types
// ─────────────────────────────────────────────────────────────

// ExprNode is defined in expr.ts and re-exported here for backward compatibility.
export type { ExprNode } from './expr.js'

// Re-export Compiled/Instance + free-function helpers for callers that
// already import from session.js.
export {
  type Compiled, type Instance,
  makeCompiled, instantiate,
  inputNames, outputNames, registerNames,
  inputPortTypes, outputPortTypes, registerPortTypes,
  inputPortType, outputPortType, registerPortType,
  inputIndex, outputIndex,
  rawInputDefaults,
} from './program_types.js'

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
}

export type TypeDefJSON = StructTypeDefJSON | SumTypeDefJSON | AliasTypeDefJSON


// ─────────────────────────────────────────────────────────────
// Session state (shared by patch load/save and MCP server)
// ─────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────
// Slot model — additive state for the inter-module slot array.
// All five fields below start empty and stay empty until the slot-mode
// compile path (M4+) starts populating them. Until then, the existing
// inputExprNodes / paramRegistry path remains the source of truth.
// ─────────────────────────────────────────────────────────────

/** Metadata for one declared output port, derived from its post-strata
 *  PortType. Captured at allocate time so wire() can typecheck combine
 *  arguments and look up scalar slot indices without re-walking the
 *  PortType tree.
 *
 *  Uses the post-strata `IRPortType` (from `ir/nodes.ts`) because it is
 *  derived from a `Compiled`'s already-stratified port declarations.
 *  After strata, sums/structs/products/units are gone — only scalar,
 *  alias, and array remain. */
export interface WirePortMeta {
  /** Branded SlotKey per scalar slot — `${instance}.${port}` for
   *  scalar ports, `${instance}.${port}[0]` etc. for array elements. */
  scalarSlotNames: SlotKey[]
  /** One ScalarKind per slot; length === scalarSlotNames.length. */
  scalarTypes:     IRScalarKind[]
  /** The original IR PortType, retained for combine typecheck. */
  portType:        IRPortType
}

export interface SessionState {
  bufferLength: number
  dac: import('./runtime/audio.js').DAC | null  // lazy type import to avoid circular dep
  typeRegistry: Map<string, Compiled>
  typeAliasRegistry: Map<string, { base: string }>
  /** Registered sum types from `ports.type_defs` entries with kind === 'sum'.
   *  Keyed by name; values carry the variant + payload metadata used for bundle decomposition. */
  sumTypeRegistry: Map<string, SumTypeMeta>
  /** Registered struct types from `ports.type_defs` entries with kind === 'struct'.
   *  Keyed by name; values carry the field metadata. Currently retained for type-system
   *  completeness; struct values themselves have no expression-level operations. */
  structTypeRegistry: Map<string, { fields: Array<{ name: string; scalar: ScalarKind }> }>
  instanceRegistry: Map<string, Instance>
  graphOutputs: Array<{ instance: string; output: string }>
  paramRegistry: Map<string, Param>
  triggerRegistry: Map<string, Trigger>
  /** Canonical input wiring: key is the branded `${instance}:${port}`
   *  wire identity, value is the ExprNode for round-trip save. Only
   *  `setWireExpr` (via `wireKey(portRef(...))`) may construct keys;
   *  raw strings are rejected at the type level. */
  inputExprNodes: Map<WireKey, ExprNode>

  // ── Slot model state (populated by M3+; empty in legacy path) ───────────
  /** Branded `${instance}.${slotName}` → slot index in the shared
   *  slot[] array. Constructed only via `slotKey(instName, slotName)`. */
  outputSlotRegistry: Map<SlotKey, number>
  /** Param/trigger name → slot index. Same flat slot[] as outputs.
   *  Param names are plain strings (no instance context to brand
   *  against). */
  paramSlotRegistry:  Map<string, number>
  /** Per-output-port metadata captured at allocate time. Keyed by the
   *  port's SlotKey (`${instance}.${port}`). */
  outputPortMeta:     Map<SlotKey, WirePortMeta>
  /** Next slot index to allocate. Always equals outputSlotRegistry.size + paramSlotRegistry.size. */
  slotCount:          number
  /** Input expressions keyed by scalar-slot name "${instance}:${scalarSlotName}".
   *  Coexists with inputExprNodes during the migration; M8 unifies them. */
  inputExprs:         Map<string, ExprNode>
  /** FlatRuntime — all audio goes through this. */
  runtime: Runtime
  /** Thin proxy over runtime that matches the old Graph interface for tests and legacy callers. */
  graph: { primeJit(): void; process(): void; readonly outputBuffer: Float64Array; dispose(): void }
  /** On-demand type resolver (set by loadStdlib for lazy loading). */
  typeResolver?: (name: string) => Compiled | undefined
  /** Monomorphized specializations of generic programs, keyed by `Type<k1=v1,k2=v2>`. */
  specializationCache: Map<string, Compiled>
  /** ResolvedProgram templates for generic programs. Keyed by type name.
   *  Only populated for programs declaring type_params. The strata pipeline's
   *  `specializeProgram` consumes these at instantiation time, producing a
   *  fresh `Compiled` per (template, type-args) pair via the
   *  specialization cache. */
  genericTemplatesResolved: Map<string, import('./ir/nodes.js').ResolvedProgram>
  /** Pre-strata `ResolvedProgram` for every non-generic registered type,
   *  keyed by type name. The elaborator consults this when an inline
   *  `instanceDecl` references a previously registered sibling — without
   *  it, follow-up `define_program` calls couldn't resolve cross-program
   *  references. Generic templates live in `genericTemplatesResolved`. */
  resolvedRegistry: Map<string, import('./ir/nodes.js').ResolvedProgram>
  /** Name counter for auto-generated instance names. */
  nameCounters: Map<string, number>
  /** Delay slots extracted from MCP-auto-delayed wires. Populated by
   *  `extractSessionDelays` (compile-time pre-pass). One entry per
   *  unit-delay wire; the `sourceExpr` is translated into a `WriteSlot`
   *  in the scheduler's `state_evolution` phase, and the wire itself
   *  is rewritten to `{op: 'sessionSlot', index: slotIdx}`. */
  delaySlotRegistry: DelaySlotEntry[]
}

/** One extracted unit-delay wire. */
export interface DelaySlotEntry {
  /** Index in the unified slot array. */
  slotIdx:    number
  /** Stable slot name (used for hot-swap state transfer). */
  slotName:   string
  /** Initial slot value (set in `slot_defaults`). */
  init:       number
  /** Un-delayed source expression; lowered to NInstrs in
   *  `state_evolution` each sample. */
  sourceExpr: ExprNode
  /** Scalar type for the WriteSlot (always 'float' today; slot
   *  array is float64-backed). */
  scalarType: 'float' | 'int' | 'bool'
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
    outputSlotRegistry: new Map(),
    paramSlotRegistry:  new Map(),
    outputPortMeta:     new Map(),
    slotCount:          0,
    inputExprs:         new Map(),
    specializationCache: new Map(),
    genericTemplatesResolved: new Map(),
    resolvedRegistry: new Map(),
    runtime,
    graph: {
      primeJit: () => {},
      process: () => runtime.process(),
      get outputBuffer() { return runtime.outputBuffer },
      dispose: () => runtime.dispose(),
    },
    nameCounters: new Map(),
    delaySlotRegistry: [],
  }
}

/** Generate a unique instance name from a type prefix. */
export function nextName(session: SessionState, prefix: string): string {
  const count = (session.nameCounters.get(prefix) ?? 0) + 1
  session.nameCounters.set(prefix, count)
  return `${prefix}${count}`
}

// ─────────────────────────────────────────────────────────────
// Wire storage — auto-delay convention
// ─────────────────────────────────────────────────────────────
//
// Every wire stored via `setWireExpr` is wrapped in a session-level
// unit delay: `{op: 'delay', args: [rawExpr], init: 0, id: ...}`. This
// is the entire structural mechanism by which the MCP-built IR stays
// acyclic — cycles in user wiring are broken at the wire layer, so the
// compiler's instance dep graph (built by `computeInstanceTopoOrder`)
// is acyclic by construction. The compile-time pre-emit pass
// `extractSessionDelays` lowers each `delay()`-wrapped wire to a
// session-level module slot whose update lives in the scheduler's
// dedicated state-evolution phase, giving every MCP wire exactly one
// sample of latency (VCV-Rack semantics).
//
// The auto-wrap is on this path only. Two wire kinds bypass it
// intentionally:
//
//   - DAC wires (`session.graphOutputs`) — direct stitch into the
//     audio mix; no extra latency on the output side.
//   - Alive inputs (`Instance.aliveInput`) — already implicitly
//     one-sample-delayed by the scheduler preamble mechanism.

/** Options for `setWireExpr`. Both fields control the synthesized
 *  delay-slot's identity: `init` becomes the slot's initial value
 *  (defaults to 0); `id` becomes the stable name used for state
 *  transfer across hot-swap. */
export interface WireExprOpts {
  readonly init?: number
  readonly id?:   string
}

/** Wrap a raw wire expression in a session-level unit delay and store
 *  it in `session.inputExprNodes` at the consumer port `ref`. Every
 *  MCP wire-setting tool routes through this helper. Callers must
 *  construct the `PortRef` via `portRef(instanceName, portName)` —
 *  the brand on `WireKey` makes raw-string keys impossible. */
export function setWireExpr(
  session: SessionState,
  ref:     PortRef,
  rawExpr: ExprNode,
  opts:    WireExprOpts = {},
): void {
  const key = wireKey(ref)
  const id  = opts.id ?? `__autodelay:${key}`
  session.inputExprNodes.set(key, wrapInUnitDelay(rawExpr, opts.init ?? 0, id))
}

/** Strip a top-level `delay()` wrapper if present, returning the raw
 *  source expression. Used by combiner logic (fan-in, `wire` with
 *  `combine`) to compose new sources with previously-stored wires
 *  without double-wrapping the existing's outer delay inside the
 *  combiner's new outer delay. */
export function unwrapDelay(expr: ExprNode): ExprNode {
  if (typeof expr !== 'object' || expr === null || Array.isArray(expr)) {
    return expr
  }
  const obj = expr as { op?: unknown; args?: unknown }
  if (obj.op === 'delay' && Array.isArray(obj.args) && obj.args.length === 1) {
    return obj.args[0] as ExprNode
  }
  return expr
}

/** Construct the delay envelope. Pure. */
function wrapInUnitDelay(expr: ExprNode, init: number, id: string): ExprNode {
  return { op: 'delay', args: [expr], init, id } as ExprNode
}

// ─────────────────────────────────────────────────────────────
// Slot allocation (M2 — additive helpers, no callers yet)
// ─────────────────────────────────────────────────────────────

/** Expand a post-strata port type to its scalar-slot names + types.
 *
 *    scalar  → 1 slot, suffix ""
 *    alias   → treated as 1 opaque scalar (the wrapped scalar kind comes
 *              from the alias's underlying type — defaults to 'float')
 *    array   → product(shape) slots, suffix "[i]" for each linear index;
 *              element must be scalar (post-strata guarantees this for
 *              array-of-scalar ports)
 *
 *  Uses the IR (post-strata) PortType. Sum/struct/product/unit don't
 *  appear here because strata lowered them all. */
export function expandPortToSlots(
  baseName: SlotKey,
  type: IRPortType,
): { names: SlotKey[]; types: IRScalarKind[] } {
  switch (type.kind) {
    case 'scalar':
      return { names: [baseName], types: [type.scalar] }
    case 'alias':
      // Aliases wrap a scalar (e.g. an opaque sum-bundle handle). For
      // slot allocation purposes, treat as a single float slot. The
      // alias's actual underlying scalar layout is computed elsewhere
      // when the slot is read/written; here we only need shape and count.
      return { names: [baseName], types: ['float'] }
    case 'array': {
      // Element is ScalarKind | AliasTypeDef per ir/port_type.ts.
      const elemKind: IRScalarKind = typeof type.element === 'string'
        ? type.element
        : 'float'  // alias element treated as opaque scalar
      // Shape dims should be concrete numbers post-specialize. If any
      // dim is still a TypeParamDecl, specialize wasn't run on the
      // owning program — that's a bug in the caller, not something to
      // silently accept.
      let total = 1
      for (const dim of type.shape) {
        if (typeof dim !== 'number') {
          throw new Error(
            `expandPortToSlots: array port '${baseName}' has unresolved ` +
            `type-param dimension; ensure specialize ran on the owning program`,
          )
        }
        total *= dim
      }
      const names: SlotKey[] = []
      const types: IRScalarKind[] = []
      for (let i = 0; i < total; i++) {
        // `${baseName}[i]` preserves the SlotKey shape — the baseName
        // is already `inst.port`, and `inst.port[0]` is a valid SlotKey
        // (slotKey's slotName parameter explicitly allows brackets).
        names.push(`${baseName}[${i}]` as SlotKey)
        types.push(elemKind)
      }
      return { names, types }
    }
  }
}

/** Default port type used when a Compiled doesn't carry an explicit
 *  PortType for an output — happens for unannotated stdlib outputs
 *  (e.g. `Delay`'s `y`). The pipeline already treats these as scalar
 *  float at codegen, so do the same here. */
const DEFAULT_OUTPUT_PORT_TYPE: IRPortType = { kind: 'scalar', scalar: 'float' }

/** Allocate slot indices for every output port of an instance. Records
 *  slot indices in `outputSlotRegistry` and full PortMeta in
 *  `outputPortMeta`. Also reserves an `__alive__` slot — every
 *  session instance has one, defaulting to literal `1.0` so the
 *  default behavior matches "always alive" with zero runtime cost
 *  after LLVM folds the conditional. Idempotent. */
export function allocateOutputSlots(
  session: SessionState,
  instName: InstanceName,
  type: Compiled,
): void {
  const portNames = outputNames(type)
  for (let idx = 0; idx < portNames.length; idx++) {
    const portName = portNames[idx]
    const portKey = slotKey(instName, portName)
    if (session.outputPortMeta.has(portKey)) continue  // idempotent
    const portType = outputPortType(type, idx) ?? DEFAULT_OUTPUT_PORT_TYPE
    const { names, types } = expandPortToSlots(portKey, portType)
    for (let i = 0; i < names.length; i++) {
      session.outputSlotRegistry.set(names[i], session.slotCount + i)
    }
    session.outputPortMeta.set(portKey, {
      scalarSlotNames: names,
      scalarTypes:     types,
      portType,
    })
    session.slotCount += names.length
  }

  const aliveKey = slotKey(instName, '__alive__')
  if (!session.outputSlotRegistry.has(aliveKey)) {
    session.outputSlotRegistry.set(aliveKey, session.slotCount)
    session.slotCount += 1
  }
}

/** Allocate a single param/trigger slot owned by the control plane.
 *  Returns the assigned slot index. Idempotent for a given name (returns
 *  the existing index if already allocated). */
export function allocateParamSlot(
  session: SessionState,
  name: string,
): number {
  const existing = session.paramSlotRegistry.get(name)
  if (existing !== undefined) return existing
  const idx = session.slotCount
  session.paramSlotRegistry.set(name, idx)
  session.slotCount += 1
  return idx
}

// ─────────────────────────────────────────────────────────────
// Op name sets (used by pretty-printer)
// ─────────────────────────────────────────────────────────────

const BINARY_OPS = new Set([
  'add', 'sub', 'mul', 'div', 'floorDiv', 'mod',
  'lt', 'lte', 'gt', 'gte', 'eq', 'neq',
  'bitAnd', 'bitOr', 'bitXor', 'lshift', 'rshift',
])

const UNARY_OPS = new Set([
  'neg', 'abs', 'sin', 'cos', 'exp', 'log', 'tanh', 'not', 'bitNot',
])

// ─────────────────────────────────────────────────────────────
// Builtin port-type aliases
// ─────────────────────────────────────────────────────────────

/** Built-in type aliases that map semantic names to a base scalar type. The
 *  alias is purely a user-facing name for the underlying scalar — there is
 *  no separate metadata. (Bounds were removed in Phase D P0.4.) */
export const BUILTIN_TYPE_ALIASES: Record<string, { base: string }> = {
  signal:   { base: 'float' },
  bipolar:  { base: 'float' },
  unipolar: { base: 'float' },
  phase:    { base: 'float' },
  freq:     { base: 'float' },
}

type AliasMap = Map<string, { base: string }>

/** Resolve a type string to its base type (stripping alias). Checks user aliases first. */
export function resolveBaseType(typeStr: string | undefined, userAliases?: AliasMap): string | undefined {
  if (!typeStr) return typeStr
  const user = userAliases?.get(typeStr)
  if (user) return user.base
  if (typeStr in BUILTIN_TYPE_ALIASES) return BUILTIN_TYPE_ALIASES[typeStr].base
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

// ─────────────────────────────────────────────────────────────
// Generic program resolution
// ─────────────────────────────────────────────────────────────

type ResolveSession = Pick<SessionState, 'typeRegistry' | 'specializationCache' | 'genericTemplatesResolved' | 'instanceRegistry' | 'paramRegistry' | 'triggerRegistry'> &
  Partial<Pick<SessionState, 'typeResolver' | 'typeAliasRegistry'>>

/**
 * Resolve a (baseName, type_args) pair to a concrete Compiled.
 * Generic types monomorphize on demand, keyed by fully-resolved integer args.
 * Non-generic types reject non-empty type_args.
 *
 * The strata pipeline is the only path: generic templates live in
 * `genericTemplatesResolved` as `ResolvedProgram`s; instantiation routes
 * through `programTypeFromResolved` to produce a fresh `Compiled`.
 */
export function resolveProgramType(
  session: ResolveSession,
  baseName: string,
  rawTypeArgs: RawTypeArgs | undefined,
  outerArgs: ResolvedTypeArgs | undefined,
): { type: Compiled; typeArgs?: ResolvedTypeArgs } {
  const specializeFromResolvedTemplate = (template: import('./ir/nodes.js').ResolvedProgram) => {
    // Mirror the legacy `type_params` shape that resolveTypeArgs expects.
    const typeParamsByName: Record<string, { type: 'int'; default?: number }> = {}
    for (const tp of template.typeParams) {
      const entry: { type: 'int'; default?: number } = { type: 'int' }
      if (tp.default !== undefined) entry.default = tp.default
      typeParamsByName[tp.name] = entry
    }
    const resolved = resolveTypeArgs(rawTypeArgs, outerArgs, typeParamsByName, `instance of '${baseName}'`)
    const key = specializationCacheKey(baseName, resolved)
    const cached = session.specializationCache.get(key)
    if (cached) return { type: cached, typeArgs: resolved }
    // Build the TypeParamDecl-keyed substitution map expected by specializeProgram.
    const subst = new Map<TypeParamDecl, number>()
    for (const [name, value] of Object.entries(resolved)) {
      const decl = template.typeParams.find(p => p.name === name)
      if (!decl) {
        throw new Error(`resolveProgramType: unknown type-param '${name}' on '${baseName}'`)
      }
      subst.set(decl, value)
    }
    const type = programTypeFromResolved(template, subst, { displayName: key })
    session.specializationCache.set(key, type)
    return { type, typeArgs: resolved }
  }

  const resolvedTemplate = session.genericTemplatesResolved.get(baseName)
  if (resolvedTemplate) return specializeFromResolvedTemplate(resolvedTemplate)

  // typeResolver may register baseName lazily (returning undefined for the
  // generic-template case — the resolver populates `genericTemplatesResolved`
  // as a side effect). Re-check after the resolver fires so the lazy-load
  // path works.
  const concrete = session.typeRegistry.get(baseName) ?? session.typeResolver?.(baseName)
  if (concrete === undefined) {
    const lateResolved = session.genericTemplatesResolved.get(baseName)
    if (lateResolved) return specializeFromResolvedTemplate(lateResolved)
  }
  if (!concrete) {
    const known = [
      ...session.typeRegistry.keys(),
      ...session.genericTemplatesResolved.keys(),
    ].join(', ')
    throw new Error(`Unknown program type '${baseName}'. Known: ${known || '(none)'}`)
  }
  if (rawTypeArgs && Object.keys(rawTypeArgs).length > 0) {
    throw new Error(`Program '${baseName}' does not declare type_params; got type_args: ${Object.keys(rawTypeArgs).join(', ')}`)
  }
  return { type: concrete }
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
  }
  const { schema: _schema, params, audio_outputs, ...progFields } = v2
  void _schema
  const node: ProgramNode = { op: 'program', ...progFields } as unknown as ProgramNode
  const topLevel: ProgramTopLevel = {}
  if (params !== undefined)        topLevel.params        = params
  if (audio_outputs !== undefined) topLevel.audio_outputs = audio_outputs
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
  instanceRegistry: Map<string, Instance>,
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
    const outName = inst && typeof out === 'number' ? (outputNames(inst)[out] ?? String(out)) : String(out)
    return `${mod}.${outName}`
  }
  if (op === 'input')     return `input(${n.name})`
  if (op === 'param')     return `param(${n.name})`
  if (op === 'trigger')   return `trigger(${n.name})`
  if (op === 'binding')   return `$${n.name}`
  if (op === 'sampleRate')  return 'sampleRate'
  if (op === 'sampleIndex') return 'sampleIndex'
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
  if (op === 'arraySet') return `array_set(${args.map(a => prettyExpr(a, instanceRegistry)).join(', ')})`
  if (op === 'array') return `[${(n.items as ExprNode[]).map(i => prettyExpr(i, instanceRegistry)).join(', ')}]`
  if (op === 'matrix') return `matrix(${JSON.stringify(n.rows)})`
  if (op === 'delay') return `delay(${prettyExpr(args[0], instanceRegistry)}, ${n.init ?? 0})`
  if (op === 'delayRef') return `delay_ref(${n.id})`
  if (op === 'nestedOut') return `${n.ref}.${n.output}`
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
