/**
 * program_types.ts — Compiled / Instance.
 *
 * Records over a post-strata `ResolvedProgram`. The helpers are free
 * functions; slot-derived fields are memoized into the record's
 * slotsCache field on first access. The wrapper is metadata; the IR
 * is the value.
 *
 * `Compiled` carries an immutable `displayName` populated at construction
 * (defaults to `prog.name`). The session's specialization cache passes
 * `displayName: 'Type<N=8>'` so downstream serializers print the
 * specialized key without mutating `prog.name`.
 */

import type { ExprNode } from './expr.js'
import type { PortType as LegacyPortType } from './term.js'
import { Float, Int, Bool, ArrayType } from './term.js'
import type {
  ResolvedProgram, ResolvedExpr, RegDecl, AliasTypeDef, ScalarKind,
  PortType as ResolvedPortType,
} from './ir/nodes.js'
import { buildSlotMaps } from './ir/slots.js'

// ---------- Value helpers ----------

export type ValueCoercible = boolean | number | number[] | number[][]

/** A register initialiser: either a bare value or { init, type }. */
export type RegInit = ValueCoercible | { init: ValueCoercible; type: string }

// ---------- Compiled ----------

interface SlotsCache {
  names: string[]
  types: (LegacyPortType | undefined)[]
}

/** A post-strata ResolvedProgram with display metadata. */
export type Compiled = {
  readonly prog: ResolvedProgram
  readonly displayName: string
  slotsCache?: SlotsCache
  defaultsCache?: Record<string, ExprNode>
}

export function makeCompiled(
  prog: ResolvedProgram,
  opts?: { displayName?: string },
): Compiled {
  return { prog, displayName: opts?.displayName ?? prog.name }
}

/** Accessors are polymorphic over Compiled | Instance — callers pass
 *  either the type or an instance, and the helpers project to the
 *  underlying Compiled. */
type CompiledOrInstance = Compiled | Instance
const _c = (x: CompiledOrInstance): Compiled =>
  'compiled' in x ? x.compiled : x

export const inputNames      = (x: CompiledOrInstance): string[] =>
  _c(x).prog.ports.inputs.map(d => d.name)
export const outputNames     = (x: CompiledOrInstance): string[] =>
  _c(x).prog.ports.outputs.map(d => d.name)
export const inputPortTypes  = (x: CompiledOrInstance): (LegacyPortType | undefined)[] =>
  _c(x).prog.ports.inputs.map(d => convertPortType(d.type))
export const outputPortTypes = (x: CompiledOrInstance): (LegacyPortType | undefined)[] =>
  _c(x).prog.ports.outputs.map(d => convertPortType(d.type))

export const inputPortType   = (x: CompiledOrInstance, idx: number): LegacyPortType | undefined =>
  inputPortTypes(x)[idx]
export const outputPortType  = (x: CompiledOrInstance, idx: number): LegacyPortType | undefined =>
  outputPortTypes(x)[idx]

function ensureSlots(c: Compiled): SlotsCache {
  if (c.slotsCache) return c.slotsCache
  const { regDecls } = buildSlotMaps(c.prog)
  const names = regDecls.map(d => d.name)
  const types: (LegacyPortType | undefined)[] = regDecls.map(d => regPortType(d))
  // Override register port types for array-init regs — lift the shape
  // from the resolved init's array literal / `zeros{count}` form.
  for (let i = 0; i < regDecls.length; i++) {
    const init = regDecls[i].init
    if (Array.isArray(init)) {
      types[i] = ArrayType(Float, [init.length])
    } else if (typeof init === 'object' && init !== null && (init as { op?: string }).op === 'zeros') {
      const count = (init as { count: unknown }).count
      if (typeof count === 'number') types[i] = ArrayType(Float, [count])
    }
  }
  c.slotsCache = { names, types }
  return c.slotsCache
}

export const registerNames     = (x: CompiledOrInstance): string[] =>
  ensureSlots(_c(x)).names
export const registerPortTypes = (x: CompiledOrInstance): (LegacyPortType | undefined)[] =>
  ensureSlots(_c(x)).types
export const registerPortType  = (x: CompiledOrInstance, idx: number): LegacyPortType | undefined =>
  ensureSlots(_c(x)).types[idx]

/** Default expressions per input port name; seeded into
 *  `session.inputExprNodes` by `loadProgramAsSession` when the user
 *  doesn't wire that input. */
export function rawInputDefaults(x: CompiledOrInstance): Record<string, ExprNode> {
  const c = _c(x)
  if (c.defaultsCache) return c.defaultsCache
  const out: Record<string, ExprNode> = {}
  for (const d of c.prog.ports.inputs) {
    if (d.default !== undefined) out[d.name] = literalDefault(d.default, d.name)
  }
  c.defaultsCache = out
  return out
}

// ---------- Instance ----------

export type Instance = {
  readonly compiled: Compiled
  readonly name: string
  /** Base (pre-specialization) type name. Equals compiled.displayName for
   *  non-generic types. */
  readonly baseTypeName: string
  /** Resolved compile-time args if this instance was specialized. */
  readonly typeArgs?: Record<string, number>
  /** Per-usage gating; the materializer wraps outputs in
   *  `select(gate, raw, fallback)`. Mutability scoped to per-usage flags. */
  gateable: boolean
  gateInput?: ExprNode
}

export function instantiate(
  c: Compiled,
  name: string,
  opts?: { baseTypeName?: string; typeArgs?: Record<string, number> },
): Instance {
  return {
    compiled: c,
    name,
    baseTypeName: opts?.baseTypeName ?? c.displayName,
    typeArgs: opts?.typeArgs,
    gateable: false,
    gateInput: undefined,
  }
}

export function inputIndex(i: Instance, name: string): number {
  const idx = inputNames(i.compiled).indexOf(name)
  if (idx === -1) throw new Error(`Unknown input '${name}' on instance '${i.name}'.`)
  return idx
}

export function outputIndex(i: Instance, name: string): number {
  const idx = outputNames(i.compiled).indexOf(name)
  if (idx === -1) throw new Error(`Unknown output '${name}' on instance '${i.name}'.`)
  return idx
}

// ─── helpers (resolved → legacy port type, default-expr lowering) ───────────

function convertPortType(pt: ResolvedPortType | undefined): LegacyPortType | undefined {
  if (pt === undefined) return undefined
  switch (pt.kind) {
    case 'scalar': return scalarToLegacy(pt.scalar)
    case 'alias':  return scalarToLegacy(pt.alias.base)
    case 'array': {
      const elem = typeof pt.element === 'string' ? scalarToLegacy(pt.element) : scalarToLegacy(pt.element.base)
      const shape = pt.shape.map(d => {
        if (typeof d !== 'number') {
          throw new Error(
            `Compiled: array shape contains unresolved type-param '${d.name}'. ` +
            `Run specializeProgram first.`,
          )
        }
        return d
      })
      return ArrayType(elem, shape)
    }
  }
}

function scalarToLegacy(s: ScalarKind): LegacyPortType {
  switch (s) {
    case 'float': return Float
    case 'int':   return Int
    case 'bool':  return Bool
  }
}

function regPortType(d: RegDecl): LegacyPortType | undefined {
  if (d.type === undefined) return undefined
  if (typeof d.type === 'string') return scalarToLegacy(d.type)
  return scalarToLegacy((d.type as AliasTypeDef).base)
}

/** Lower a `ResolvedExpr` input-default to the MCP wire-format `ExprNode`
 *  shape that gets spliced into `session.inputExprNodes`. Defaults are
 *  ref-free (they don't read other instances' outputs or program-internal
 *  decls), so this walker only needs to handle literals + arithmetic +
 *  clamp/select. The bounds-lowering pass (`parse/lower_bounds.ts`) wraps
 *  bounded defaults in `clamp` ops, which is why we accept that op here. */
function literalDefault(expr: ResolvedExpr, portName: string): ExprNode {
  if (typeof expr === 'number' || typeof expr === 'boolean') return expr
  if (Array.isArray(expr)) return expr.map(e => literalDefault(e, portName))
  if (typeof expr !== 'object' || expr === null) {
    throw new Error(`Compiled: input '${portName}' default has unexpected shape`)
  }
  const obj = expr as { op: string; args?: unknown[]; count?: ResolvedExpr }
  switch (obj.op) {
    case 'add': case 'sub': case 'mul': case 'div': case 'mod':
    case 'floorDiv': case 'ldexp':
    case 'lt': case 'lte': case 'gt': case 'gte': case 'eq': case 'neq':
    case 'and': case 'or':
    case 'bitAnd': case 'bitOr': case 'bitXor': case 'lshift': case 'rshift':
    case 'neg': case 'not': case 'bitNot': case 'sqrt': case 'abs':
    case 'floor': case 'ceil': case 'round': case 'floatExponent':
    case 'toInt': case 'toBool': case 'toFloat':
    case 'clamp': case 'select': case 'index': case 'arraySet': {
      const args = (obj.args ?? []) as ResolvedExpr[]
      return { op: obj.op, args: args.map(a => literalDefault(a, portName)) }
    }
    case 'sampleRate': case 'sampleIndex':
      return { op: obj.op }
    case 'zeros': {
      const c = literalDefault(obj.count as ResolvedExpr, portName)
      return { op: 'zeros', count: c }
    }
  }
  throw new Error(
    `Compiled: input '${portName}' default has op '${obj.op}' that's not a literal-class form; ` +
    `defaults shouldn't reference decls or run combinators`,
  )
}
