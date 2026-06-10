/**
 * resolved_codec.ts — Resolved⇄JSON codec (Phase 4, stage 1 of the
 * Lean port).
 *
 * `encodeResolved` serializes a `ResolvedProgram` to a plain
 * JSON-serializable value; `decodeResolved` rebuilds a fully valid
 * `ResolvedProgram` (typed decl tables projected through `mkProgram`,
 * `programRegistry` restored with sharing). The pair is property-gated
 * by `resolved_codec.test.ts`: round-trip structural equality,
 * encode∘decode∘encode = encode, and recompile-equality (byte-identical
 * plan JSON) across the stdlib / generic / sum / nested / session-root
 * corpus.
 *
 * ## Encoding shape — three identity pools + positional refs
 *
 * The in-memory IR is index-keyed everywhere a position-table exists
 * (branded `RegIdx` / `InputIdx` / … refs into the program's decl
 * tables), but four reference families are still object pointers, and
 * compile behavior depends on their *identity*, not just their shape:
 *
 *   1. **Programs.** `programRegistry` values and `ProgramDecl.program`
 *      may be shared across registries (the stdlib loader's topological
 *      relink, specialize's shared sub-program pass-through). Encoded
 *      as a flat pool collected by object identity in post-order DFS
 *      from the root (children strictly before parents — the IR is
 *      acyclic by construction; a defensive in-flight set throws on
 *      cyclic references). References are pool indices; registry
 *      entries are `[key, poolIdx]` pairs in Map insertion order.
 *
 *   2. **Type defs.** `sum_lower` compares variants and payload fields
 *      by `===` (`variants.indexOf(tag.variant)`, `slot.variant ===
 *      initTag.variant`, `payload.find(p => p.field === slot.field)`),
 *      and the elaborator's `lookupTypeDef` walks the *scope chain* —
 *      a nested program's `Tag`/alias may reference a TypeDef declared
 *      in an enclosing program's `ports.typeDefs`. Worse, after
 *      `inlineInstances` a lifted reg's alias `type` can outlive every
 *      `typeDefs` table that ever held it. So per-program table
 *      indices are NOT sufficient: typeDefs get their own flat pool
 *      collected by object identity across all pooled programs and all
 *      inline refs. `SumVariant.parent` back-pointers are implicit in
 *      the pool entry (decode rebuilds the parent↔variant cycle);
 *      `Tag.variant` / `MatchArm.variant` encode as a variant index
 *      into the pooled SumTypeDef; `Tag.payload[].field` as a field
 *      index into the variant's payload. Alias deps are pooled before
 *      their dependents, so decode is a single forward pass.
 *
 *   3. **Type params.** `specializeProgram` keys its substitution on
 *      `TypeParamDecl` identity (`Set(prog.typeParams)`,
 *      `subst.get(shapeDim)`), and `resolveShapeDim` walks the scope
 *      chain — a nested program's array shape dim may point at an
 *      *enclosing* program's TypeParamDecl. Same treatment: a flat
 *      identity pool; `prog.typeParams` and `ShapeDim` entries encode
 *      as pool indices. (`TypeParamRef.idx` in expression position is
 *      already a positional `TypeParamIdx` and encodes as a number.)
 *
 *   4. **Decl tables.** `regs` / `params` / `instances` must be the
 *      same objects as the decls in `body.decls`. Decode routes every
 *      program through `mkProgram` (decl_tables.ts), which projects
 *      the tables from the decoded `body.decls` — the invariant holds
 *      by construction and the registry coverage check runs for free.
 *
 * Expression-level DAG sharing is NOT preserved: a subexpression
 * referenced from two parents decodes as two structurally equal trees.
 * This is safe for plan equality because `emit_resolved` keys CSE on a
 * bottom-up *structural* id (not node identity), so duplicates collapse
 * to the same instruction; the substitution passes key their maps on
 * branded indices (`BinderIdx`, `InputIdx`), never on expression
 * pointers. The recompile-equality gate holds the codec to this claim.
 *
 * Branded indices encode as plain numbers and decode through the brand
 * constructors (`regIdx(...)` etc.) — no `as` casts. Non-finite number
 * literals are rejected at encode time (JSON.stringify would silently
 * corrupt them to `null`).
 *
 * Encode never mutates its input; decode builds fresh objects only.
 */

import type {
  ResolvedProgram, ResolvedExpr, ResolvedExprOp,
  BodyDecl, BodyAssign, OutputAssign,
  InputDecl, OutputDecl, RegDecl, ParamDecl, InstanceDecl, ProgramDecl,
  TypeParamDecl, BinderDecl,
  TypeDef, StructTypeDef, SumTypeDef, SumVariant, AliasTypeDef, StructField,
  PortType, ShapeDim, ScalarKind,
  MatchArm,
  BinaryOpTag, UnaryOpTag,
  ProgramKey,
} from './nodes.js'
import {
  regIdx, inputIdx, outputIdx, paramIdx, instanceIdx, typeParamIdx, binderIdx,
  programKey,
} from './nodes.js'
import { mkProgram } from './decl_tables.js'

// ─────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────

export class ResolvedCodecError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'ResolvedCodecError'
  }
}

const fail = (msg: string): never => { throw new ResolvedCodecError(msg) }

// ─────────────────────────────────────────────────────────────
// Encoded shapes (the wire form — plain JSON)
// ─────────────────────────────────────────────────────────────

export const RESOLVED_CODEC_SCHEMA = 'tropical_resolved_1'

export interface EncTypeParam {
  name: string
  default?: number
}

export interface EncField {
  name: string
  /** Scalar kind string, or `{ alias: poolIdx }`. */
  type: EncScalarOrAlias
}

export type EncScalarOrAlias = ScalarKind | { alias: number }

export type EncTypeDef =
  | { kind: 'alias'; name: string; base: ScalarKind }
  | { kind: 'struct'; name: string; fields: EncField[] }
  | { kind: 'sum'; name: string; variants: Array<{ name: string; payload: EncField[] }> }

export type EncShapeDim = number | { typeParam: number }

export type EncPortType =
  | { kind: 'scalar'; scalar: ScalarKind }
  | { kind: 'alias'; alias: number }
  | { kind: 'array'; element: EncScalarOrAlias; shape: EncShapeDim[] }

export interface EncInputDecl {
  name: string
  type?: EncPortType
  default?: EncExpr
}

export interface EncOutputDecl {
  name: string
  type?: EncPortType
}

export interface EncBinder {
  name: string
  idx: number
}

/** Expression wire form. Numbers / booleans / arrays pass through;
 *  op nodes mirror the in-memory shape with pointer refs rewritten to
 *  pool indices / positional indices. */
export type EncExpr = number | boolean | EncExpr[] | EncExprOp

export type EncExprOp =
  | { op: BinaryOpTag; args: [EncExpr, EncExpr] }
  | { op: UnaryOpTag; args: [EncExpr] }
  | { op: 'clamp' | 'select' | 'arraySet'; args: [EncExpr, EncExpr, EncExpr] }
  | { op: 'index'; args: [EncExpr, EncExpr] }
  | { op: 'zeros'; count: EncExpr }
  | { op: 'inputRef' | 'regRef' | 'paramRef' | 'typeParamRef' | 'bindingRef'; idx: number }
  | { op: 'nestedOut'; instance: number; output: number }
  | { op: 'sampleRate' | 'sampleIndex' }
  | { op: 'fold' | 'scan'; over: EncExpr; init: EncExpr; acc: EncBinder; elem: EncBinder; body: EncExpr }
  | { op: 'generate'; count: EncExpr; iter: EncBinder; body: EncExpr }
  | { op: 'iterate' | 'chain'; count: EncExpr; init: EncExpr; iter: EncBinder; body: EncExpr }
  | { op: 'map2'; over: EncExpr; elem: EncBinder; body: EncExpr }
  | { op: 'zipWith'; a: EncExpr; b: EncExpr; x: EncBinder; y: EncBinder; body: EncExpr }
  | { op: 'let'; binders: Array<{ binder: EncBinder; value: EncExpr }>; in: EncExpr }
  | { op: 'tag'; def: number; variant: number; payload: Array<{ field: number; value: EncExpr }> }
  | {
      op: 'match'; def: number; scrutinee: EncExpr
      arms: Array<{ variant: number; binders: EncBinder[]; body: EncExpr }>
    }

export type EncBodyDecl =
  | { op: 'regDecl'; name: string; init: EncExpr; update?: EncExpr; type?: EncScalarOrAlias; liftedFrom?: string }
  | { op: 'paramDecl'; name: string; value?: number }
  | {
      op: 'instanceDecl'; name: string; typeKey: string
      typeArgs: Array<{ param: number; value: number }>
      inputs: Array<{ port: number; value: EncExpr }>
    }
  | { op: 'programDecl'; name: string; program: number }

export interface EncOutputAssign {
  /** OutputIdx into the program's `ports.outputs[]`, or the dac leaf. */
  target: number | { kind: 'dac' }
  expr: EncExpr
}

export interface EncProgram {
  name: string
  /** typeParam pool indices, in `prog.typeParams` order. */
  typeParams: number[]
  inputs: EncInputDecl[]
  outputs: EncOutputDecl[]
  /** typeDef pool indices, in `ports.typeDefs` order. */
  typeDefs: number[]
  decls: EncBodyDecl[]
  assigns: EncOutputAssign[]
  binderCount: number
  /** `[key, programPoolIdx]` pairs in registry insertion order. */
  registry: Array<[string, number]>
}

export interface EncodedResolvedProgram {
  schema: typeof RESOLVED_CODEC_SCHEMA
  typeParamPool: EncTypeParam[]
  /** Topologically ordered: alias deps strictly before dependents. */
  typeDefPool: EncTypeDef[]
  /** Post-order DFS from the root: referenced programs strictly before
   *  referencing programs. The root is the final entry. */
  programPool: EncProgram[]
  root: number
}

// ─────────────────────────────────────────────────────────────
// Exhaustive op-tag tables (Record trick: adding a tag to the union
// without adding it here is a compile error, and vice versa)
// ─────────────────────────────────────────────────────────────

const BINARY_OP_TAGS: Record<BinaryOpTag, true> = {
  add: true, sub: true, mul: true, div: true, mod: true,
  lt: true, lte: true, gt: true, gte: true, eq: true, neq: true,
  and: true, or: true,
  bitAnd: true, bitOr: true, bitXor: true, lshift: true, rshift: true,
  floorDiv: true, ldexp: true,
}

const UNARY_OP_TAGS: Record<UnaryOpTag, true> = {
  neg: true, not: true, bitNot: true,
  sqrt: true, abs: true, floor: true, ceil: true, round: true,
  floatExponent: true, toInt: true, toBool: true, toFloat: true,
}

const SCALAR_KINDS: Record<ScalarKind, true> = { float: true, int: true, bool: true }

const isBinaryOpTag = (s: string): s is BinaryOpTag =>
  Object.prototype.hasOwnProperty.call(BINARY_OP_TAGS, s)
const isUnaryOpTag = (s: string): s is UnaryOpTag =>
  Object.prototype.hasOwnProperty.call(UNARY_OP_TAGS, s)
const isScalarKind = (s: string): s is ScalarKind =>
  Object.prototype.hasOwnProperty.call(SCALAR_KINDS, s)

// ─────────────────────────────────────────────────────────────
// Encode
// ─────────────────────────────────────────────────────────────

export function encodeResolved(prog: ResolvedProgram): unknown {
  const enc = new Encoder()
  const root = enc.programId(prog)
  const out: EncodedResolvedProgram = {
    schema: RESOLVED_CODEC_SCHEMA,
    typeParamPool: enc.typeParamPool,
    typeDefPool: enc.typeDefPool,
    programPool: enc.programPool,
    root,
  }
  return out
}

class Encoder {
  readonly typeParamPool: EncTypeParam[] = []
  readonly typeDefPool: EncTypeDef[] = []
  readonly programPool: EncProgram[] = []

  private readonly typeParamIds = new Map<TypeParamDecl, number>()
  private readonly typeDefIds = new Map<TypeDef, number>()
  private readonly programIds = new Map<ResolvedProgram, number>()
  /** Programs currently being encoded — the defensive cycle check. */
  private readonly inFlight = new Set<ResolvedProgram>()

  // ── Pools ──

  programId(p: ResolvedProgram): number {
    const got = this.programIds.get(p)
    if (got !== undefined) return got
    if (this.inFlight.has(p)) {
      return fail(
        `encodeResolved: cyclic program reference through '${p.name}' ` +
        `(program reachable from itself via programRegistry / programDecl). ` +
        `The resolved IR is acyclic by construction; this is an upstream registry-build bug.`,
      )
    }
    this.inFlight.add(p)
    const encoded = this.encodeProgram(p)
    this.inFlight.delete(p)
    const idx = this.programPool.length
    this.programPool.push(encoded)
    this.programIds.set(p, idx)
    return idx
  }

  private typeParamId(d: TypeParamDecl): number {
    const got = this.typeParamIds.get(d)
    if (got !== undefined) return got
    const entry: EncTypeParam = { name: d.name }
    if (d.default !== undefined) entry.default = finite(d.default, `typeParam '${d.name}' default`)
    const idx = this.typeParamPool.length
    this.typeParamPool.push(entry)
    this.typeParamIds.set(d, idx)
    return idx
  }

  private typeDefId(td: TypeDef): number {
    const got = this.typeDefIds.get(td)
    if (got !== undefined) return got
    // Encode dependencies (alias refs inside fields) BEFORE pushing
    // self, so the pool stays topologically ordered for the decoder's
    // single forward pass. Aliases have no deps; def→def references
    // other than field aliases don't exist in the IR.
    let entry: EncTypeDef
    switch (td.op) {
      case 'aliasTypeDef':
        entry = { kind: 'alias', name: td.name, base: td.base }
        break
      case 'structTypeDef':
        entry = { kind: 'struct', name: td.name, fields: td.fields.map(f => this.encodeField(f)) }
        break
      case 'sumTypeDef':
        entry = {
          kind: 'sum',
          name: td.name,
          variants: td.variants.map(v => {
            if (v.parent !== td) {
              return fail(
                `encodeResolved: variant '${v.name}' of sum '${td.name}' has a foreign parent ` +
                `('${v.parent.name}') — the parent back-pointer invariant is broken upstream`,
              )
            }
            return { name: v.name, payload: v.payload.map(f => this.encodeField(f)) }
          }),
        }
        break
    }
    const idx = this.typeDefPool.length
    this.typeDefPool.push(entry)
    this.typeDefIds.set(td, idx)
    return idx
  }

  private encodeField(f: StructField): EncField {
    return { name: f.name, type: this.encodeScalarOrAlias(f.type) }
  }

  private encodeScalarOrAlias(t: ScalarKind | AliasTypeDef): EncScalarOrAlias {
    return typeof t === 'string' ? t : { alias: this.typeDefId(t) }
  }

  /** `{ def, variant }` ref via the variant's parent back-pointer.
   *  Identity-checked: the variant must be a member of its parent's
   *  `variants[]` (the exact `===` sum_lower depends on). */
  private variantRef(v: SumVariant): { def: number; variant: number } {
    const def = this.typeDefId(v.parent)
    const idx = v.parent.variants.indexOf(v)
    if (idx === -1) {
      return fail(
        `encodeResolved: variant '${v.name}' is not (by identity) a member of its ` +
        `parent sum '${v.parent.name}' — variant identity is broken upstream`,
      )
    }
    return { def, variant: idx }
  }

  // ── Program ──

  private encodeProgram(p: ResolvedProgram): EncProgram {
    const registry: Array<[string, number]> = []
    for (const [key, target] of p.programRegistry) {
      registry.push([key, this.programId(target)])
    }
    return {
      name: p.name,
      typeParams: p.typeParams.map(tp => this.typeParamId(tp)),
      inputs: p.ports.inputs.map(d => this.encodeInputDecl(d)),
      outputs: p.ports.outputs.map(d => this.encodeOutputDecl(d)),
      typeDefs: p.ports.typeDefs.map(td => this.typeDefId(td)),
      decls: p.body.decls.map(d => this.encodeBodyDecl(d)),
      assigns: p.body.assigns.map(a => this.encodeAssign(a)),
      binderCount: p.binderCount,
      registry,
    }
  }

  private encodeInputDecl(d: InputDecl): EncInputDecl {
    const out: EncInputDecl = { name: d.name }
    if (d.type !== undefined) out.type = this.encodePortType(d.type)
    if (d.default !== undefined) out.default = this.encodeExpr(d.default)
    return out
  }

  private encodeOutputDecl(d: OutputDecl): EncOutputDecl {
    const out: EncOutputDecl = { name: d.name }
    if (d.type !== undefined) out.type = this.encodePortType(d.type)
    return out
  }

  private encodePortType(pt: PortType): EncPortType {
    switch (pt.kind) {
      case 'scalar': return { kind: 'scalar', scalar: pt.scalar }
      case 'alias':  return { kind: 'alias', alias: this.typeDefId(pt.alias) }
      case 'array':  return {
        kind: 'array',
        element: this.encodeScalarOrAlias(pt.element),
        shape: pt.shape.map(d => this.encodeShapeDim(d)),
      }
    }
  }

  private encodeShapeDim(d: ShapeDim): EncShapeDim {
    if (typeof d === 'number') return finite(d, 'shape dim')
    return { typeParam: this.typeParamId(d) }
  }

  private encodeBodyDecl(d: BodyDecl): EncBodyDecl {
    switch (d.op) {
      case 'regDecl': {
        const out: Extract<EncBodyDecl, { op: 'regDecl' }> =
          { op: 'regDecl', name: d.name, init: this.encodeExpr(d.init) }
        if (d.update !== undefined) out.update = this.encodeExpr(d.update)
        if (d.type !== undefined) out.type = this.encodeScalarOrAlias(d.type)
        if (d._liftedFrom !== undefined) out.liftedFrom = d._liftedFrom
        return out
      }
      case 'paramDecl': {
        const out: Extract<EncBodyDecl, { op: 'paramDecl' }> = { op: 'paramDecl', name: d.name }
        if (d.value !== undefined) out.value = finite(d.value, `param '${d.name}' value`)
        return out
      }
      case 'instanceDecl':
        return {
          op: 'instanceDecl',
          name: d.name,
          typeKey: d.typeKey,
          typeArgs: d.typeArgs.map(a => ({ param: a.param, value: finite(a.value, `instance '${d.name}' typeArg`) })),
          inputs: d.inputs.map(i => ({ port: i.port, value: this.encodeExpr(i.value) })),
        }
      case 'programDecl':
        return { op: 'programDecl', name: d.name, program: this.programId(d.program) }
    }
  }

  private encodeAssign(a: BodyAssign): EncOutputAssign {
    return {
      target: typeof a.target === 'number' ? a.target : { kind: 'dac' },
      expr: this.encodeExpr(a.expr),
    }
  }

  // ── Expressions ──

  private encodeBinder(b: BinderDecl): EncBinder {
    return { name: b.name, idx: b.idx }
  }

  private encodeExpr(e: ResolvedExpr): EncExpr {
    if (typeof e === 'number') return finite(e, 'number literal')
    if (typeof e === 'boolean') return e
    if (Array.isArray(e)) return e.map(x => this.encodeExpr(x))
    return this.encodeOpNode(e)
  }

  private encodeOpNode(node: ResolvedExprOp): EncExprOp {
    const recur = (x: ResolvedExpr): EncExpr => this.encodeExpr(x)

    switch (node.op) {
      // Refs — already positional; copy the (branded) numbers.
      case 'inputRef': case 'regRef': case 'paramRef':
      case 'typeParamRef': case 'bindingRef':
        return { op: node.op, idx: node.idx }
      case 'nestedOut':
        return { op: 'nestedOut', instance: node.instance, output: node.output }
      case 'sampleRate': case 'sampleIndex':
        return { op: node.op }

      // Uniform binary ops.
      case 'add': case 'sub': case 'mul': case 'div': case 'mod':
      case 'lt': case 'lte': case 'gt': case 'gte': case 'eq': case 'neq':
      case 'and': case 'or':
      case 'bitAnd': case 'bitOr': case 'bitXor': case 'lshift': case 'rshift':
      case 'floorDiv': case 'ldexp':
        return { op: node.op, args: [recur(node.args[0]), recur(node.args[1])] }

      // Unary ops.
      case 'neg': case 'not': case 'bitNot':
      case 'sqrt': case 'abs': case 'floor': case 'ceil': case 'round':
      case 'floatExponent': case 'toInt': case 'toBool': case 'toFloat':
        return { op: node.op, args: [recur(node.args[0])] }

      // Ternary / fixed-arity ops.
      case 'clamp': case 'select': case 'arraySet':
        return { op: node.op, args: [recur(node.args[0]), recur(node.args[1]), recur(node.args[2])] }
      case 'index':
        return { op: 'index', args: [recur(node.args[0]), recur(node.args[1])] }
      case 'zeros':
        return { op: 'zeros', count: recur(node.count) }

      // Combinators.
      case 'fold': case 'scan':
        return {
          op: node.op,
          over: recur(node.over), init: recur(node.init),
          acc: this.encodeBinder(node.acc), elem: this.encodeBinder(node.elem),
          body: recur(node.body),
        }
      case 'generate':
        return { op: 'generate', count: recur(node.count), iter: this.encodeBinder(node.iter), body: recur(node.body) }
      case 'iterate': case 'chain':
        return {
          op: node.op,
          count: recur(node.count), init: recur(node.init),
          iter: this.encodeBinder(node.iter), body: recur(node.body),
        }
      case 'map2':
        return { op: 'map2', over: recur(node.over), elem: this.encodeBinder(node.elem), body: recur(node.body) }
      case 'zipWith':
        return {
          op: 'zipWith',
          a: recur(node.a), b: recur(node.b),
          x: this.encodeBinder(node.x), y: this.encodeBinder(node.y),
          body: recur(node.body),
        }
      case 'let':
        return {
          op: 'let',
          binders: node.binders.map(b => ({ binder: this.encodeBinder(b.binder), value: recur(b.value) })),
          in: recur(node.in),
        }

      // ADT expressions.
      case 'tag': {
        const ref = this.variantRef(node.variant)
        return {
          op: 'tag',
          def: ref.def,
          variant: ref.variant,
          payload: node.payload.map(p => {
            const fieldIdx = node.variant.payload.indexOf(p.field)
            if (fieldIdx === -1) {
              return fail(
                `encodeResolved: tag '${node.variant.name}' payload field '${p.field.name}' is not ` +
                `(by identity) a member of the variant's payload — field identity is broken upstream`,
              )
            }
            return { field: fieldIdx, value: recur(p.value) }
          }),
        }
      }
      case 'match':
        return {
          op: 'match',
          def: this.typeDefId(node.type),
          scrutinee: recur(node.scrutinee),
          arms: node.arms.map(arm => {
            const variantIdx = node.type.variants.indexOf(arm.variant)
            if (variantIdx === -1) {
              return fail(
                `encodeResolved: match arm variant '${arm.variant.name}' is not (by identity) ` +
                `a member of sum '${node.type.name}' — variant identity is broken upstream`,
              )
            }
            return {
              variant: variantIdx,
              binders: arm.binders.map(b => this.encodeBinder(b)),
              body: recur(arm.body),
            }
          }),
        }
    }
  }
}

function finite(n: number, what: string): number {
  if (!Number.isFinite(n)) {
    return fail(`encodeResolved: non-finite ${what} (${n}) — JSON.stringify would corrupt it to null`)
  }
  return n
}

// ─────────────────────────────────────────────────────────────
// Decode — narrowing helpers
// ─────────────────────────────────────────────────────────────

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v)
}

function reqRecord(v: unknown, ctx: string): Record<string, unknown> {
  if (!isRecord(v)) return fail(`decodeResolved: ${ctx}: expected an object, got ${describe(v)}`)
  return v
}

function reqArray(v: unknown, ctx: string): unknown[] {
  if (!Array.isArray(v)) return fail(`decodeResolved: ${ctx}: expected an array, got ${describe(v)}`)
  return v
}

function reqString(v: unknown, ctx: string): string {
  if (typeof v !== 'string') return fail(`decodeResolved: ${ctx}: expected a string, got ${describe(v)}`)
  return v
}

function reqNumber(v: unknown, ctx: string): number {
  if (typeof v !== 'number' || !Number.isFinite(v)) {
    return fail(`decodeResolved: ${ctx}: expected a finite number, got ${describe(v)}`)
  }
  return v
}

function reqInt(v: unknown, ctx: string): number {
  const n = reqNumber(v, ctx)
  if (!Number.isInteger(n)) return fail(`decodeResolved: ${ctx}: expected an integer, got ${n}`)
  return n
}

function reqScalarKind(v: unknown, ctx: string): ScalarKind {
  const s = reqString(v, ctx)
  if (!isScalarKind(s)) return fail(`decodeResolved: ${ctx}: unknown scalar kind '${s}'`)
  return s
}

function describe(v: unknown): string {
  if (v === null) return 'null'
  if (Array.isArray(v)) return 'an array'
  return typeof v
}

// ─────────────────────────────────────────────────────────────
// Decode
// ─────────────────────────────────────────────────────────────

export function decodeResolved(json: unknown): ResolvedProgram {
  const top = reqRecord(json, 'top level')
  const schema = reqString(top.schema, 'schema')
  if (schema !== RESOLVED_CODEC_SCHEMA) {
    return fail(`decodeResolved: unsupported schema '${schema}' (expected '${RESOLVED_CODEC_SCHEMA}')`)
  }
  const dec = new Decoder(
    reqArray(top.typeParamPool, 'typeParamPool'),
    reqArray(top.typeDefPool, 'typeDefPool'),
    reqArray(top.programPool, 'programPool'),
  )
  return dec.programAt(reqInt(top.root, 'root'), 'root')
}

class Decoder {
  private readonly typeParams: TypeParamDecl[]
  private readonly typeDefs: TypeDef[]
  private readonly programs: ResolvedProgram[] = []

  constructor(typeParamPool: unknown[], typeDefPool: unknown[], programPool: unknown[]) {
    // 1. Type params — leaves, no internal refs.
    this.typeParams = typeParamPool.map((raw, i) => {
      const r = reqRecord(raw, `typeParamPool[${i}]`)
      const decl: TypeParamDecl = { op: 'typeParamDecl', name: reqString(r.name, `typeParamPool[${i}].name`) }
      if (r.default !== undefined) decl.default = reqNumber(r.default, `typeParamPool[${i}].default`)
      return decl
    })

    // 2. Type defs — single forward pass; alias refs point strictly
    //    earlier in the pool (the encoder's topological guarantee).
    this.typeDefs = []
    for (let i = 0; i < typeDefPool.length; i++) {
      this.typeDefs.push(this.decodeTypeDef(reqRecord(typeDefPool[i], `typeDefPool[${i}]`), i))
    }

    // 3. Programs — single forward pass; program refs point strictly
    //    earlier in the pool (post-order DFS guarantee).
    for (let i = 0; i < programPool.length; i++) {
      this.programs.push(this.decodeProgram(reqRecord(programPool[i], `programPool[${i}]`), i))
    }
  }

  // ── Pool lookups ──

  programAt(idx: number, ctx: string): ResolvedProgram {
    const p = this.programs[idx]
    if (p === undefined) {
      return fail(
        `decodeResolved: ${ctx}: program pool index ${idx} is out of range ` +
        `(decoded so far: ${this.programs.length}); forward references violate the post-order pool contract`,
      )
    }
    return p
  }

  private typeParamAt(idx: number, ctx: string): TypeParamDecl {
    const tp = this.typeParams[idx]
    if (tp === undefined) return fail(`decodeResolved: ${ctx}: typeParam pool index ${idx} out of range`)
    return tp
  }

  private typeDefAt(idx: number, ctx: string): TypeDef {
    const td = this.typeDefs[idx]
    if (td === undefined) {
      return fail(
        `decodeResolved: ${ctx}: typeDef pool index ${idx} is out of range ` +
        `(decoded so far: ${this.typeDefs.length})`,
      )
    }
    return td
  }

  private aliasAt(idx: number, ctx: string): AliasTypeDef {
    const td = this.typeDefAt(idx, ctx)
    if (td.op !== 'aliasTypeDef') return fail(`decodeResolved: ${ctx}: typeDef '${td.name}' is not an alias`)
    return td
  }

  private sumAt(idx: number, ctx: string): SumTypeDef {
    const td = this.typeDefAt(idx, ctx)
    if (td.op !== 'sumTypeDef') return fail(`decodeResolved: ${ctx}: typeDef '${td.name}' is not a sum type`)
    return td
  }

  // ── Type defs ──

  private decodeTypeDef(r: Record<string, unknown>, i: number): TypeDef {
    const ctx = `typeDefPool[${i}]`
    const kind = reqString(r.kind, `${ctx}.kind`)
    const name = reqString(r.name, `${ctx}.name`)
    switch (kind) {
      case 'alias':
        return { op: 'aliasTypeDef', name, base: reqScalarKind(r.base, `${ctx}.base`) }
      case 'struct': {
        const def: StructTypeDef = {
          op: 'structTypeDef',
          name,
          fields: reqArray(r.fields, `${ctx}.fields`).map((f, j) => this.decodeField(f, `${ctx}.fields[${j}]`)),
        }
        return def
      }
      case 'sum': {
        // Build the parent first, then push variants carrying the
        // back-pointer — the same construction order the elaborator
        // uses; the decl↔variant cycle is restored here.
        const sum: SumTypeDef = { op: 'sumTypeDef', name, variants: [] }
        for (const [j, rawV] of reqArray(r.variants, `${ctx}.variants`).entries()) {
          const v = reqRecord(rawV, `${ctx}.variants[${j}]`)
          const variant: SumVariant = {
            op: 'sumVariant',
            name: reqString(v.name, `${ctx}.variants[${j}].name`),
            payload: reqArray(v.payload, `${ctx}.variants[${j}].payload`)
              .map((f, k) => this.decodeField(f, `${ctx}.variants[${j}].payload[${k}]`)),
            parent: sum,
          }
          sum.variants.push(variant)
        }
        return sum
      }
      default:
        return fail(`decodeResolved: ${ctx}: unknown typeDef kind '${kind}'`)
    }
  }

  private decodeField(raw: unknown, ctx: string): StructField {
    const r = reqRecord(raw, ctx)
    return {
      op: 'structField',
      name: reqString(r.name, `${ctx}.name`),
      type: this.decodeScalarOrAlias(r.type, `${ctx}.type`),
    }
  }

  private decodeScalarOrAlias(raw: unknown, ctx: string): ScalarKind | AliasTypeDef {
    if (typeof raw === 'string') return reqScalarKind(raw, ctx)
    const r = reqRecord(raw, ctx)
    return this.aliasAt(reqInt(r.alias, `${ctx}.alias`), ctx)
  }

  // ── Programs ──

  private decodeProgram(r: Record<string, unknown>, i: number): ResolvedProgram {
    const ctx = `programPool[${i}]`
    const name = reqString(r.name, `${ctx}.name`)

    const typeParams = reqArray(r.typeParams, `${ctx}.typeParams`)
      .map((tp, j) => this.typeParamAt(reqInt(tp, `${ctx}.typeParams[${j}]`), `${ctx}.typeParams[${j}]`))

    const inputs = reqArray(r.inputs, `${ctx}.inputs`)
      .map((d, j) => this.decodeInputDecl(d, `${ctx}.inputs[${j}]`))
    const outputs = reqArray(r.outputs, `${ctx}.outputs`)
      .map((d, j) => this.decodeOutputDecl(d, `${ctx}.outputs[${j}]`))
    const typeDefs = reqArray(r.typeDefs, `${ctx}.typeDefs`)
      .map((td, j) => this.typeDefAt(reqInt(td, `${ctx}.typeDefs[${j}]`), `${ctx}.typeDefs[${j}]`))

    const decls = reqArray(r.decls, `${ctx}.decls`)
      .map((d, j) => this.decodeBodyDecl(d, `${ctx}.decls[${j}]`))
    const assigns = reqArray(r.assigns, `${ctx}.assigns`)
      .map((a, j) => this.decodeAssign(a, `${ctx}.assigns[${j}]`))

    const registry = new Map<ProgramKey, ResolvedProgram>()
    for (const [j, rawEntry] of reqArray(r.registry, `${ctx}.registry`).entries()) {
      const entry = reqArray(rawEntry, `${ctx}.registry[${j}]`)
      if (entry.length !== 2) return fail(`decodeResolved: ${ctx}.registry[${j}]: expected a [key, poolIdx] pair`)
      const key = reqString(entry[0], `${ctx}.registry[${j}][0]`)
      const target = this.programAt(reqInt(entry[1], `${ctx}.registry[${j}][1]`), `${ctx}.registry[${j}]`)
      registry.set(programKey(key), target)
    }

    // mkProgram projects the typed decl tables from body.decls (the
    // table↔body identity invariant) and validates registry coverage.
    return mkProgram({
      name,
      typeParams,
      ports: { inputs, outputs, typeDefs },
      body: { op: 'block', decls, assigns },
      binderCount: reqInt(r.binderCount, `${ctx}.binderCount`),
      programRegistry: registry,
    })
  }

  private decodeInputDecl(raw: unknown, ctx: string): InputDecl {
    const r = reqRecord(raw, ctx)
    const decl: InputDecl = { op: 'inputDecl', name: reqString(r.name, `${ctx}.name`) }
    if (r.type !== undefined) decl.type = this.decodePortType(r.type, `${ctx}.type`)
    if (r.default !== undefined) decl.default = this.decodeExpr(r.default, `${ctx}.default`)
    return decl
  }

  private decodeOutputDecl(raw: unknown, ctx: string): OutputDecl {
    const r = reqRecord(raw, ctx)
    const decl: OutputDecl = { op: 'outputDecl', name: reqString(r.name, `${ctx}.name`) }
    if (r.type !== undefined) decl.type = this.decodePortType(r.type, `${ctx}.type`)
    return decl
  }

  private decodePortType(raw: unknown, ctx: string): PortType {
    const r = reqRecord(raw, ctx)
    const kind = reqString(r.kind, `${ctx}.kind`)
    switch (kind) {
      case 'scalar': return { kind: 'scalar', scalar: reqScalarKind(r.scalar, `${ctx}.scalar`) }
      case 'alias':  return { kind: 'alias', alias: this.aliasAt(reqInt(r.alias, `${ctx}.alias`), ctx) }
      case 'array':  return {
        kind: 'array',
        element: this.decodeScalarOrAlias(r.element, `${ctx}.element`),
        shape: reqArray(r.shape, `${ctx}.shape`).map((d, j) => this.decodeShapeDim(d, `${ctx}.shape[${j}]`)),
      }
      default:
        return fail(`decodeResolved: ${ctx}: unknown port-type kind '${kind}'`)
    }
  }

  private decodeShapeDim(raw: unknown, ctx: string): ShapeDim {
    if (typeof raw === 'number') return reqNumber(raw, ctx)
    const r = reqRecord(raw, ctx)
    return this.typeParamAt(reqInt(r.typeParam, `${ctx}.typeParam`), ctx)
  }

  private decodeBodyDecl(raw: unknown, ctx: string): BodyDecl {
    const r = reqRecord(raw, ctx)
    const op = reqString(r.op, `${ctx}.op`)
    const name = reqString(r.name, `${ctx}.name`)
    switch (op) {
      case 'regDecl': {
        const decl: RegDecl = { op: 'regDecl', name, init: this.decodeExpr(r.init, `${ctx}.init`) }
        if (r.update !== undefined) decl.update = this.decodeExpr(r.update, `${ctx}.update`)
        if (r.type !== undefined) decl.type = this.decodeScalarOrAlias(r.type, `${ctx}.type`)
        if (r.liftedFrom !== undefined) decl._liftedFrom = reqString(r.liftedFrom, `${ctx}.liftedFrom`)
        return decl
      }
      case 'paramDecl': {
        const decl: ParamDecl = { op: 'paramDecl', name }
        if (r.value !== undefined) decl.value = reqNumber(r.value, `${ctx}.value`)
        return decl
      }
      case 'instanceDecl': {
        const decl: InstanceDecl = {
          op: 'instanceDecl',
          name,
          typeKey: programKey(reqString(r.typeKey, `${ctx}.typeKey`)),
          typeArgs: reqArray(r.typeArgs, `${ctx}.typeArgs`).map((a, j) => {
            const e = reqRecord(a, `${ctx}.typeArgs[${j}]`)
            return {
              param: typeParamIdx(reqInt(e.param, `${ctx}.typeArgs[${j}].param`)),
              value: reqNumber(e.value, `${ctx}.typeArgs[${j}].value`),
            }
          }),
          inputs: reqArray(r.inputs, `${ctx}.inputs`).map((w, j) => {
            const e = reqRecord(w, `${ctx}.inputs[${j}]`)
            return {
              port: inputIdx(reqInt(e.port, `${ctx}.inputs[${j}].port`)),
              value: this.decodeExpr(e.value, `${ctx}.inputs[${j}].value`),
            }
          }),
        }
        return decl
      }
      case 'programDecl': {
        const decl: ProgramDecl = {
          op: 'programDecl',
          name,
          program: this.programAt(reqInt(r.program, `${ctx}.program`), `${ctx}.program`),
        }
        return decl
      }
      default:
        return fail(`decodeResolved: ${ctx}: unknown body-decl op '${op}'`)
    }
  }

  private decodeAssign(raw: unknown, ctx: string): BodyAssign {
    const r = reqRecord(raw, ctx)
    const target: OutputAssign['target'] = typeof r.target === 'number'
      ? outputIdx(reqInt(r.target, `${ctx}.target`))
      : decodeDacTarget(r.target, ctx)
    const assign: OutputAssign = { op: 'outputAssign', target, expr: this.decodeExpr(r.expr, `${ctx}.expr`) }
    return assign
  }

  // ── Expressions ──

  private decodeBinder(raw: unknown, ctx: string): BinderDecl {
    const r = reqRecord(raw, ctx)
    return {
      op: 'binderDecl',
      name: reqString(r.name, `${ctx}.name`),
      idx: binderIdx(reqInt(r.idx, `${ctx}.idx`)),
    }
  }

  private decodeExpr(raw: unknown, ctx: string): ResolvedExpr {
    if (typeof raw === 'number') return reqNumber(raw, ctx)
    if (typeof raw === 'boolean') return raw
    if (Array.isArray(raw)) return raw.map((x, j) => this.decodeExpr(x, `${ctx}[${j}]`))
    return this.decodeOpNode(reqRecord(raw, ctx), ctx)
  }

  private decodeOpNode(r: Record<string, unknown>, ctx: string): ResolvedExprOp {
    const op = reqString(r.op, `${ctx}.op`)
    const expr = (key: string): ResolvedExpr => this.decodeExpr(r[key], `${ctx}.${key}`)

    if (isBinaryOpTag(op)) {
      const args = reqArray(r.args, `${ctx}.args`)
      if (args.length !== 2) return fail(`decodeResolved: ${ctx}: '${op}' expects 2 args, got ${args.length}`)
      return { op, args: [this.decodeExpr(args[0], `${ctx}.args[0]`), this.decodeExpr(args[1], `${ctx}.args[1]`)] }
    }
    if (isUnaryOpTag(op)) {
      const args = reqArray(r.args, `${ctx}.args`)
      if (args.length !== 1) return fail(`decodeResolved: ${ctx}: '${op}' expects 1 arg, got ${args.length}`)
      return { op, args: [this.decodeExpr(args[0], `${ctx}.args[0]`)] }
    }

    switch (op) {
      case 'clamp': case 'select': case 'arraySet': {
        const args = reqArray(r.args, `${ctx}.args`)
        if (args.length !== 3) return fail(`decodeResolved: ${ctx}: '${op}' expects 3 args, got ${args.length}`)
        return {
          op,
          args: [
            this.decodeExpr(args[0], `${ctx}.args[0]`),
            this.decodeExpr(args[1], `${ctx}.args[1]`),
            this.decodeExpr(args[2], `${ctx}.args[2]`),
          ],
        }
      }
      case 'index': {
        const args = reqArray(r.args, `${ctx}.args`)
        if (args.length !== 2) return fail(`decodeResolved: ${ctx}: 'index' expects 2 args, got ${args.length}`)
        return { op: 'index', args: [this.decodeExpr(args[0], `${ctx}.args[0]`), this.decodeExpr(args[1], `${ctx}.args[1]`)] }
      }
      case 'zeros':
        return { op: 'zeros', count: expr('count') }

      case 'inputRef':     return { op, idx: inputIdx(reqInt(r.idx, `${ctx}.idx`)) }
      case 'regRef':       return { op, idx: regIdx(reqInt(r.idx, `${ctx}.idx`)) }
      case 'paramRef':     return { op, idx: paramIdx(reqInt(r.idx, `${ctx}.idx`)) }
      case 'typeParamRef': return { op, idx: typeParamIdx(reqInt(r.idx, `${ctx}.idx`)) }
      case 'bindingRef':   return { op, idx: binderIdx(reqInt(r.idx, `${ctx}.idx`)) }
      case 'nestedOut':
        return {
          op: 'nestedOut',
          instance: instanceIdx(reqInt(r.instance, `${ctx}.instance`)),
          output: outputIdx(reqInt(r.output, `${ctx}.output`)),
        }
      case 'sampleRate': case 'sampleIndex':
        return { op }

      case 'fold': case 'scan':
        return {
          op,
          over: expr('over'), init: expr('init'),
          acc: this.decodeBinder(r.acc, `${ctx}.acc`),
          elem: this.decodeBinder(r.elem, `${ctx}.elem`),
          body: expr('body'),
        }
      case 'generate':
        return { op: 'generate', count: expr('count'), iter: this.decodeBinder(r.iter, `${ctx}.iter`), body: expr('body') }
      case 'iterate': case 'chain':
        return {
          op,
          count: expr('count'), init: expr('init'),
          iter: this.decodeBinder(r.iter, `${ctx}.iter`),
          body: expr('body'),
        }
      case 'map2':
        return { op: 'map2', over: expr('over'), elem: this.decodeBinder(r.elem, `${ctx}.elem`), body: expr('body') }
      case 'zipWith':
        return {
          op: 'zipWith',
          a: expr('a'), b: expr('b'),
          x: this.decodeBinder(r.x, `${ctx}.x`),
          y: this.decodeBinder(r.y, `${ctx}.y`),
          body: expr('body'),
        }
      case 'let':
        return {
          op: 'let',
          binders: reqArray(r.binders, `${ctx}.binders`).map((b, j) => {
            const e = reqRecord(b, `${ctx}.binders[${j}]`)
            return {
              binder: this.decodeBinder(e.binder, `${ctx}.binders[${j}].binder`),
              value: this.decodeExpr(e.value, `${ctx}.binders[${j}].value`),
            }
          }),
          in: expr('in'),
        }

      case 'tag': {
        const sum = this.sumAt(reqInt(r.def, `${ctx}.def`), `${ctx}.def`)
        const variant = this.variantAt(sum, reqInt(r.variant, `${ctx}.variant`), ctx)
        return {
          op: 'tag',
          variant,
          payload: reqArray(r.payload, `${ctx}.payload`).map((p, j) => {
            const e = reqRecord(p, `${ctx}.payload[${j}]`)
            const fieldIdx = reqInt(e.field, `${ctx}.payload[${j}].field`)
            const field = variant.payload[fieldIdx]
            if (field === undefined) {
              return fail(
                `decodeResolved: ${ctx}.payload[${j}]: field index ${fieldIdx} out of range for ` +
                `variant '${variant.name}' (${variant.payload.length} fields)`,
              )
            }
            return { field, value: this.decodeExpr(e.value, `${ctx}.payload[${j}].value`) }
          }),
        }
      }
      case 'match': {
        const sum = this.sumAt(reqInt(r.def, `${ctx}.def`), `${ctx}.def`)
        const arms: MatchArm[] = reqArray(r.arms, `${ctx}.arms`).map((a, j) => {
          const e = reqRecord(a, `${ctx}.arms[${j}]`)
          return {
            variant: this.variantAt(sum, reqInt(e.variant, `${ctx}.arms[${j}].variant`), `${ctx}.arms[${j}]`),
            binders: reqArray(e.binders, `${ctx}.arms[${j}].binders`)
              .map((b, k) => this.decodeBinder(b, `${ctx}.arms[${j}].binders[${k}]`)),
            body: this.decodeExpr(e.body, `${ctx}.arms[${j}].body`),
          }
        })
        return { op: 'match', type: sum, scrutinee: expr('scrutinee'), arms }
      }

      default:
        return fail(`decodeResolved: ${ctx}: unknown expression op '${op}'`)
    }
  }

  private variantAt(sum: SumTypeDef, idx: number, ctx: string): SumVariant {
    const v = sum.variants[idx]
    if (v === undefined) {
      return fail(
        `decodeResolved: ${ctx}: variant index ${idx} out of range for sum '${sum.name}' ` +
        `(${sum.variants.length} variants)`,
      )
    }
    return v
  }
}

function decodeDacTarget(raw: unknown, ctx: string): { kind: 'dac' } {
  const r = reqRecord(raw, `${ctx}.target`)
  const kind = reqString(r.kind, `${ctx}.target.kind`)
  if (kind !== 'dac') return fail(`decodeResolved: ${ctx}.target: unknown target kind '${kind}'`)
  return { kind: 'dac' }
}
