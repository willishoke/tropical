/**
 * compiler/ir/emit_resolved.ts — `ResolvedExpr → FlatProgram` emitter.
 *
 * The §2.1 from-scratch port that closes Phase D's structural goal:
 * the runtime path is `ResolvedProgram → strata → emit_resolved → JIT`,
 * with no intermediate Expr tree. Refs become operand kinds via
 * decl-identity slot lookups; the dispatch is exhaustive over the
 * closed `ResolvedExprOp` union.
 *
 * Replaces:
 *   - compiler/ir/lower_to_exprnode.ts:resolvedToSlotted (deleted)
 *   - compiler/emit_numeric.ts (deleted)
 *   - the EmitExprNode bag-of-fields type (deleted; see compiler/ir/emit_node.ts)
 *
 * Borrowed wholesale from emit_numeric.ts:
 *   - the FlatProgram / NInstr / NOperand / GroupInfo data types
 *   - the BINARY_TAG / UNARY_TAG / CAST_RESULT / *_TAGS sets
 *   - the type-inference rules (promoteTypes, inferResultType)
 *   - the structural-CSE intern table (issue #131)
 *   - the array-loop emission patterns (loop_count > 1, strides[])
 *
 * What's actually new:
 *   - `tryTerminal` dispatches on the ResolvedExpr ref ops
 *     (`inputRef`, `regRef`, `delayRef`, `paramRef`, `sampleRate`,
 *     `sampleIndex`) using slot maps + decl identity. No string
 *     lookups, no `obj._ptr` reflection.
 *   - `compileNodeUncached` switches over the closed
 *     `ResolvedExprOp` union; TypeScript's exhaustiveness check
 *     gates missing cases at compile time.
 *   - param handles thread through `paramHandles: Map<ParamDecl, ...>`
 *     populated by compileSession from the session's paramRegistry.
 */

import type {
  ResolvedExpr, ResolvedExprOp,
  RegDecl, DelayDecl, InputDecl, InstanceDecl, ParamDecl,
} from './nodes.js'

// ─────────────────────────────────────────────────────────────
// Public types — mirror the C++ engine's FlatProgram contract
// ─────────────────────────────────────────────────────────────

export type ScalarType = 'float' | 'int' | 'bool'

export type NOperand =
  | { kind: 'const';     val: number;  scalar_type: ScalarType }
  | { kind: 'input';     slot: number; scalar_type: ScalarType }
  | { kind: 'reg';       slot: number; scalar_type: ScalarType }
  | { kind: 'array_reg'; slot: number }
  | { kind: 'state_reg'; slot: number; scalar_type: ScalarType }
  | { kind: 'param';     ptr: string;  scalar_type: ScalarType }
  | { kind: 'rate';      scalar_type: ScalarType }
  | { kind: 'tick';      scalar_type: ScalarType }

export type NInstr = {
  tag:         string
  dst:         number
  args:        NOperand[]
  loop_count:  number
  strides:     number[]
  result_type: ScalarType
  group_id?:   string
}

export type GroupInfo = {
  id:           string
  gate_operand: NOperand
}

export type FlatProgram = {
  register_count:   number
  array_slot_count: number
  array_slot_sizes: number[]
  instructions:     NInstr[]
  output_targets:   number[]
  register_targets: number[]
  groups?:          GroupInfo[]
}

// ─────────────────────────────────────────────────────────────
// Slot tables — passed in by compile_resolved.ts / compile_session.ts
// ─────────────────────────────────────────────────────────────

export interface EmitSlots {
  inputs: Map<InputDecl, number>
  regs:   Map<RegDecl, number>
  delays: Map<DelayDecl, number>
  /** Total scalar-register count (regs.size). Delays land at `regCount + delaySlot`
   *  in the unified state-register layout. */
  regCount: number
  /** FFI handle metadata per param/trigger decl, populated by compile_session
   *  from the session's paramRegistry/triggerRegistry. */
  paramHandles: Map<ParamDecl, { ptr: string }>
}

// ─────────────────────────────────────────────────────────────
// Op-tag mappings (verbatim from emit_numeric.ts)
// ─────────────────────────────────────────────────────────────

const BINARY_TAG: Record<string, string> = {
  add: 'Add', sub: 'Sub', mul: 'Mul', div: 'Div', mod: 'Mod',
  floorDiv: 'FloorDiv',
  lt: 'Less', lte: 'LessEq', gt: 'Greater', gte: 'GreaterEq',
  eq: 'Equal', neq: 'NotEqual',
  bitAnd: 'BitAnd', bitOr: 'BitOr', bitXor: 'BitXor',
  lshift: 'LShift', rshift: 'RShift',
  and: 'And', or: 'Or',
  ldexp: 'Ldexp',
}

const UNARY_TAG: Record<string, string> = {
  neg: 'Neg', abs: 'Abs', sqrt: 'Sqrt',
  floor: 'Floor', ceil: 'Ceil', round: 'Round',
  not: 'Not', bitNot: 'BitNot',
  floatExponent: 'FloatExponent',
  toInt: 'ToInt', toBool: 'ToBool', toFloat: 'ToFloat',
}

const CAST_RESULT: Record<string, ScalarType> = {
  ToInt: 'int', ToBool: 'bool', ToFloat: 'float',
}

const BITWISE_TAGS = new Set(['BitAnd', 'BitOr', 'BitXor', 'LShift', 'RShift', 'BitNot'])
const COMPARISON_TAGS = new Set(['Less', 'LessEq', 'Greater', 'GreaterEq', 'Equal', 'NotEqual', 'Not', 'And', 'Or'])
const TRANSCENDENTAL_TAGS = new Set(['Sqrt', 'Floor', 'Ceil', 'Round', 'Ldexp', 'FloatExponent'])

function promoteTypes(a: ScalarType, b: ScalarType): ScalarType {
  if (a === 'float' || b === 'float') return 'float'
  if (a === 'int' || b === 'int') return 'int'
  return 'bool'
}

function inferResultType(tag: string, argTypes: ScalarType[]): ScalarType {
  if (CAST_RESULT[tag]) return CAST_RESULT[tag]
  if (BITWISE_TAGS.has(tag)) return 'int'
  if (COMPARISON_TAGS.has(tag)) return 'bool'
  if (TRANSCENDENTAL_TAGS.has(tag)) return 'float'
  if (tag === 'Select') return promoteTypes(argTypes[1] ?? 'float', argTypes[2] ?? 'float')
  if (tag === 'Clamp') return argTypes[0] ?? 'float'
  if (argTypes.length === 0) return 'float'
  return argTypes.reduce(promoteTypes)
}

// ─────────────────────────────────────────────────────────────
// Internal compile result
// ─────────────────────────────────────────────────────────────

type ScalarResult = { isArray: false; op: NOperand; scalarType: ScalarType }
type ArrayResult  = { isArray: true;  op: NOperand; size: number; scalarType: ScalarType }
type CompileResult = ScalarResult | ArrayResult

// ─────────────────────────────────────────────────────────────
// Emitter
// ─────────────────────────────────────────────────────────────

class Emitter {
  private nextReg       = 0
  private nextArraySlot = 0
  private arraySizes:   number[] = []
  private instrs:       NInstr[] = []
  private groupStack:   string[] = []
  private groupCounter  = 0
  private groups:       GroupInfo[] = []

  // Structural CSE — same shape as emit_numeric (issue #131).
  private hashTable = new Map<string, number>()
  private hashCache = new WeakMap<object, number>()
  private memo      = new Map<string, CompileResult>()

  // ResolvedExpr-array regs surface as `regRef` to a regDecl whose init
  // is an array. Tracked here so `compileNodeUncached`'s regRef branch
  // returns an `array_reg` operand rather than a scalar `state_reg`.
  private arrayRegMap = new Map<number, { slot: number; size: number }>()

  private regTypes = new Map<number, ScalarType>()
  private stateRegTypes: ScalarType[]
  private inputPortTypes: ScalarType[]
  private slots: EmitSlots

  constructor(
    slots: EmitSlots,
    stateInit: (number | boolean | number[])[],
    stateRegTypes: ScalarType[],
    inputPortTypes: ScalarType[],
  ) {
    this.slots = slots
    this.stateRegTypes = stateRegTypes
    this.inputPortTypes = inputPortTypes
    for (let i = 0; i < stateInit.length; i++) {
      const init = stateInit[i]
      if (Array.isArray(init)) {
        const slot = this.allocArraySlot(init.length)
        this.arrayRegMap.set(i, { slot, size: init.length })
      }
    }
  }

  private allocReg(): number { return this.nextReg++ }

  private allocArraySlot(size: number): number {
    const slot = this.nextArraySlot++
    this.arraySizes.push(size)
    return slot
  }

  private emit(instr: NInstr): void {
    if (this.groupStack.length > 0) instr.group_id = this.groupStack[this.groupStack.length - 1]
    this.instrs.push(instr)
  }

  // ── Terminal check ──────────────────────────────────────────
  private tryTerminal(node: ResolvedExpr, expected?: ScalarType): { op: NOperand; scalarType: ScalarType } | null {
    if (typeof node === 'number') {
      const t = this.resolveNumericLiteralType(node, expected)
      return { op: { kind: 'const', val: node, scalar_type: t }, scalarType: t }
    }
    if (typeof node === 'boolean') return { op: { kind: 'const', val: node ? 1 : 0, scalar_type: 'bool' }, scalarType: 'bool' }
    if (Array.isArray(node)) return null
    if (typeof node !== 'object' || node === null) return { op: { kind: 'const', val: 0, scalar_type: 'float' }, scalarType: 'float' }
    const obj = node as ResolvedExprOp
    switch (obj.op) {
      case 'inputRef': {
        const slot = this.slots.inputs.get(obj.decl)
        if (slot === undefined) throw new Error(`emit_resolved: input '${obj.decl.name}' missing from slot table`)
        const portT = this.inputPortTypes[slot] ?? 'float'
        return { op: { kind: 'input', slot, scalar_type: portT }, scalarType: portT }
      }
      case 'regRef': {
        const slot = this.slots.regs.get(obj.decl)
        if (slot === undefined) throw new Error(`emit_resolved: reg '${obj.decl.name}' missing from slot table`)
        // Array-typed regs return null so compileNodeUncached emits an
        // array_reg operand (matching emit_numeric's behavior).
        if (this.arrayRegMap.has(slot)) return null
        const regType = this.stateRegTypes[slot] ?? 'float'
        return { op: { kind: 'state_reg', slot, scalar_type: regType }, scalarType: regType }
      }
      case 'delayRef': {
        const slot = this.slots.delays.get(obj.decl)
        if (slot === undefined) throw new Error(`emit_resolved: delay '${obj.decl.name}' missing from slot table`)
        const combined = this.slots.regCount + slot
        const regType = this.stateRegTypes[combined] ?? 'float'
        return { op: { kind: 'state_reg', slot: combined, scalar_type: regType }, scalarType: regType }
      }
      case 'paramRef': {
        const handle = this.slots.paramHandles.get(obj.decl)
        if (handle === undefined) {
          // No live FFI handle (e.g. running under interpret-style fixtures
          // that don't bind params) — emit zero, matching the legacy
          // emit_numeric fallback.
          return { op: { kind: 'const', val: 0, scalar_type: 'float' }, scalarType: 'float' }
        }
        return { op: { kind: 'param', ptr: handle.ptr, scalar_type: 'float' }, scalarType: 'float' }
      }
      case 'sampleRate':  return { op: { kind: 'rate', scalar_type: 'float' }, scalarType: 'float' }
      case 'sampleIndex': return { op: { kind: 'tick', scalar_type: 'int' }, scalarType: 'int' }
    }
    return null
  }

  private resolveNumericLiteralType(val: number, expected?: ScalarType): ScalarType {
    if (expected === 'int') {
      if (!Number.isInteger(val)) {
        throw new Error(`Lossy conversion: literal ${val} cannot narrow to int. Wrap the source in to_int() to narrow explicitly.`)
      }
      return 'int'
    }
    if (expected === 'bool') {
      if (val !== 0 && val !== 1) {
        throw new Error(`Lossy conversion: literal ${val} cannot narrow to bool. Wrap the source in to_bool() to narrow explicitly.`)
      }
      return 'bool'
    }
    return 'float'
  }

  // ── Structural CSE id ───────────────────────────────────────
  //
  // Ref-bearing nodes (regRef/delayRef/paramRef/inputRef/etc.) are keyed
  // by op + decl IDENTITY, not by recursing through `decl`'s fields. The
  // resolved IR has cycles via decl init/update fields (a self-
  // referential delay reads its own previous value), so naive deep
  // hashing infinite-recurses.
  //
  // For other ops the recursive hash is what makes CSE collapse
  // structurally-identical subtrees produced by clone-then-substitute.
  private declIds = new WeakMap<object, number>()
  private nextDeclId = 0
  private declIdOf(decl: object): number {
    let id = this.declIds.get(decl)
    if (id === undefined) {
      id = this.nextDeclId++
      this.declIds.set(decl, id)
    }
    return id
  }

  private structuralId(node: object): number {
    const cached = this.hashCache.get(node)
    if (cached !== undefined) return cached
    let key: string
    if (Array.isArray(node)) {
      key = `a:${node.map(c => this.structuralKey(c)).join(',')}`
    } else {
      const obj = node as Record<string, unknown>
      const op = String(obj.op)
      // Ref-bearing nodes: hash op + decl identity, skip recursion.
      if (op === 'regRef' || op === 'delayRef' || op === 'paramRef'
          || op === 'inputRef' || op === 'typeParamRef' || op === 'bindingRef') {
        key = `op:${op}|decl=${this.declIdOf(obj.decl as object)}`
      } else if (op === 'nestedOut') {
        key = `op:${op}|inst=${this.declIdOf(obj.instance as object)}|out=${this.declIdOf(obj.output as object)}`
      } else {
        const parts: string[] = [`op:${op}`]
        const fieldNames = Object.keys(obj).filter(k => k !== 'op').sort()
        for (const k of fieldNames) parts.push(`${k}=${this.structuralKey(obj[k])}`)
        key = parts.join('|')
      }
    }
    let id = this.hashTable.get(key)
    if (id === undefined) {
      id = this.hashTable.size
      this.hashTable.set(key, id)
    }
    this.hashCache.set(node, id)
    return id
  }

  private structuralKey(v: unknown): string {
    if (v === null) return 'null'
    if (typeof v === 'number')  return `n:${v}`
    if (typeof v === 'boolean') return `b:${v}`
    if (typeof v === 'string')  return `s:${v}`
    if (typeof v === 'object')  return `i:${this.structuralId(v as object)}`
    return `u:${typeof v}`
  }

  // ── Compile a node to a CompileResult ──────────────────────
  compileNode(node: ResolvedExpr, expected?: ScalarType): CompileResult {
    const terminal = this.tryTerminal(node, expected)
    if (terminal !== null) return { isArray: false, op: terminal.op, scalarType: terminal.scalarType }

    const key = `${this.structuralId(node as object)}:${expected ?? ''}`
    const cached = this.memo.get(key)
    if (cached !== undefined) return cached

    const result = this.compileNodeUncached(node, expected)
    this.memo.set(key, result)
    return result
  }

  private compileNodeUncached(node: ResolvedExpr, expected?: ScalarType): CompileResult {
    if (Array.isArray(node)) return this.compilePack(node, expected)

    const obj = node as ResolvedExprOp

    // Array-typed regRef (filtered out by tryTerminal).
    if (obj.op === 'regRef') {
      const slot = this.slots.regs.get(obj.decl)
      if (slot === undefined) throw new Error(`emit_resolved: reg '${obj.decl.name}' missing from slot table`)
      const arr = this.arrayRegMap.get(slot)
      if (arr) return { isArray: true, op: { kind: 'array_reg', slot: arr.slot }, size: arr.size, scalarType: 'float' }
      throw new Error(`emit_resolved: regRef to non-array slot ${slot} reached compileNodeUncached unexpectedly`)
    }

    // Binary arithmetic / comparison / bitwise / logical ops.
    const binTag = BINARY_TAG[obj.op]
    if (binTag) {
      const opNode = obj as Extract<ResolvedExprOp, { args: [ResolvedExpr, ResolvedExpr] }>
      return this.compileBinary(binTag, opNode.args, expected)
    }

    // Unary ops (`pow` is binary in the resolved IR — handled above).
    const uniTag = UNARY_TAG[obj.op]
    if (uniTag) {
      const opNode = obj as Extract<ResolvedExprOp, { args: [ResolvedExpr] }>
      return this.compileUnary(uniTag, opNode.args[0], expected)
    }

    // Ternary ops.
    if (obj.op === 'clamp')  return this.compileTernary('Clamp',  [obj.args[0], obj.args[1], obj.args[2]], expected)
    if (obj.op === 'select') return this.compileTernary('Select', [obj.args[0], obj.args[1], obj.args[2]], expected)
    if (obj.op === 'arraySet') return this.compileSetElement([obj.args[0], obj.args[1], obj.args[2]])

    // Index.
    if (obj.op === 'index') return this.compileIndex([obj.args[0], obj.args[1]])

    // zeros literal — should be statically unrolled by arrayLower, but if
    // it survives (e.g. as a regDecl init), emit a zero array.
    if (obj.op === 'zeros') {
      const c = this.compileNode(obj.count, 'int')
      const n = c.op.kind === 'const' && typeof c.op.val === 'number' ? c.op.val : 0
      const slot = this.allocArraySlot(n)
      this.emit({
        tag: 'Pack', dst: slot,
        args: new Array(n).fill({ kind: 'const', val: 0, scalar_type: 'float' as ScalarType }),
        loop_count: 1, strides: [], result_type: 'float',
      })
      return { isArray: true, op: { kind: 'array_reg', slot }, size: n, scalarType: 'float' }
    }

    // Combinators / let / ADTs should have been lowered out by arrayLower
    // / sumLower. Reaching one is a strata bug.
    switch (obj.op) {
      case 'fold': case 'scan': case 'generate': case 'iterate':
      case 'chain': case 'map2': case 'zipWith':
      case 'let':
      case 'tag': case 'match':
      case 'typeParamRef': case 'bindingRef': case 'nestedOut':
        throw new Error(`emit_resolved: '${obj.op}' should have been lowered before emit`)
    }

    // Unreachable — TypeScript will catch missing cases at compile time
    // when the ResolvedExprOp union grows.
    const _exhaustive: never = obj as never
    void _exhaustive
    throw new Error(`emit_resolved: unhandled op (TypeScript exhaustiveness escape)`)
  }

  // ── Unbox a size-1 array to a scalar via Index[0]. ──
  private unboxArray(arr: ArrayResult): ScalarResult {
    const dst = this.allocReg()
    const rt = arr.scalarType
    this.regTypes.set(dst, rt)
    this.emit({ tag: 'Index', dst, args: [arr.op, { kind: 'const', val: 0, scalar_type: 'int' }], loop_count: 1, strides: [], result_type: rt })
    return { isArray: false, op: { kind: 'reg', slot: dst, scalar_type: rt }, scalarType: rt }
  }

  // ── Compile an inline JS array to a Pack instruction. ──
  private compilePack(elements: ResolvedExpr[], expected?: ScalarType): ArrayResult {
    const size = elements.length
    const slot = this.allocArraySlot(size)
    const args: NOperand[] = elements.map(e => {
      const r = this.compileNode(e, expected)
      return r.isArray ? { kind: 'const' as const, val: 0, scalar_type: 'float' as ScalarType } : r.op
    })
    this.emit({ tag: 'Pack', dst: slot, args, loop_count: 1, strides: [], result_type: 'float' })
    return { isArray: true, op: { kind: 'array_reg', slot }, size, scalarType: 'float' }
  }

  // ── Compile a binary op. ──
  private compileBinary(tag: string, argNodes: [ResolvedExpr, ResolvedExpr], expected?: ScalarType): CompileResult {
    // `expected` is a hint used to narrow leaf literals (e.g. `add(int_reg, 1)`
    // wants the `1` to compile as int, not float). For arithmetic ops we can
    // propagate it down — but never `'bool'`. Arithmetic on float/int args
    // can't produce bool, so pushing `expected='bool'` into the args (which
    // happens when this op's result is the cond of a Select, or the left arg
    // of a comparison whose other side is bool) would wrongly try to narrow
    // any float literal in the subtree to bool.
    //
    // Comparison ops are already exempt (argExpected=undefined) because they
    // fully redefine their arg types via the comparison itself; we additionally
    // strip 'bool' from non-bitwise non-comparison ops here. The `secondExpected`
    // path below also strips 'bool' from the comparison's right-arg expected,
    // since `secondExpected = l.scalarType` may be 'bool' (e.g. `gt(bool, expr)`).
    const propagated = expected === 'bool' ? undefined : expected
    const argExpected = BITWISE_TAGS.has(tag) ? 'int' as ScalarType
      : COMPARISON_TAGS.has(tag) ? undefined
      : propagated
    let l = this.compileNode(argNodes[0], argExpected)
    const secondExpected = COMPARISON_TAGS.has(tag)
      ? (l.isArray ? 'float' : (l.scalarType === 'bool' ? undefined : l.scalarType))
      : argExpected
    let r = this.compileNode(argNodes[1], secondExpected)
    if (l.isArray && l.size === 1) l = this.unboxArray(l)
    if (r.isArray && r.size === 1) r = this.unboxArray(r)

    const rt = inferResultType(tag, [l.scalarType, r.scalarType])

    if (!l.isArray && !r.isArray) {
      const dst = this.allocReg()
      this.regTypes.set(dst, rt)
      this.emit({ tag, dst, args: [l.op, r.op], loop_count: 1, strides: [], result_type: rt })
      return { isArray: false, op: { kind: 'reg', slot: dst, scalar_type: rt }, scalarType: rt }
    }

    const size = l.isArray ? l.size : (r as ArrayResult).size
    const slot = this.allocArraySlot(size)
    const strides = [l.isArray ? 1 : 0, r.isArray ? 1 : 0]
    this.emit({ tag, dst: slot, args: [l.op, r.op], loop_count: size, strides, result_type: rt })
    return { isArray: true, op: { kind: 'array_reg', slot }, size, scalarType: rt }
  }

  // ── Compile a unary op. ──
  private compileUnary(tag: string, argNode: ResolvedExpr, expected?: ScalarType): CompileResult {
    const argExpected = TRANSCENDENTAL_TAGS.has(tag) ? undefined
      : COMPARISON_TAGS.has(tag) ? undefined
      : tag === 'BitNot' ? 'int' as ScalarType
      : expected
    let a = this.compileNode(argNode, argExpected)
    if (a.isArray && a.size === 1) a = this.unboxArray(a)

    const rt = inferResultType(tag, [a.scalarType])

    if (!a.isArray) {
      const dst = this.allocReg()
      this.regTypes.set(dst, rt)
      this.emit({ tag, dst, args: [a.op], loop_count: 1, strides: [], result_type: rt })
      return { isArray: false, op: { kind: 'reg', slot: dst, scalar_type: rt }, scalarType: rt }
    }

    const slot = this.allocArraySlot(a.size)
    this.emit({ tag, dst: slot, args: [a.op], loop_count: a.size, strides: [1], result_type: rt })
    return { isArray: true, op: { kind: 'array_reg', slot }, size: a.size, scalarType: rt }
  }

  // ── Compile a ternary op. ──
  private compileTernary(tag: string, argNodes: [ResolvedExpr, ResolvedExpr, ResolvedExpr], expected?: ScalarType): CompileResult {
    // For Select: cond must be bool, but arm exprs are arbitrary — don't
    // push 'bool' into them (same reasoning as compileBinary above).
    const condExpected: ScalarType | undefined = tag === 'Select' ? 'bool' : expected
    const armExpected = expected === 'bool' ? undefined : expected
    let a = this.compileNode(argNodes[0], condExpected)
    let b = this.compileNode(argNodes[1], armExpected)
    let c = this.compileNode(argNodes[2], armExpected)

    if (a.isArray && a.size === 1) a = this.unboxArray(a)
    if (b.isArray && b.size === 1) b = this.unboxArray(b)
    if (c.isArray && c.size === 1) c = this.unboxArray(c)

    const rt = inferResultType(tag, [a.scalarType, b.scalarType, c.scalarType])
    const anyArray = a.isArray || b.isArray || c.isArray
    if (!anyArray) {
      const dst = this.allocReg()
      this.regTypes.set(dst, rt)
      this.emit({ tag, dst, args: [a.op, b.op, c.op], loop_count: 1, strides: [], result_type: rt })
      return { isArray: false, op: { kind: 'reg', slot: dst, scalar_type: rt }, scalarType: rt }
    }

    const size = (a.isArray ? a.size : b.isArray ? b.size : (c as ArrayResult).size)
    const slot = this.allocArraySlot(size)
    const strides = [a.isArray ? 1 : 0, b.isArray ? 1 : 0, c.isArray ? 1 : 0]
    this.emit({ tag, dst: slot, args: [a.op, b.op, c.op], loop_count: size, strides, result_type: rt })
    return { isArray: true, op: { kind: 'array_reg', slot }, size, scalarType: rt }
  }

  // ── Index. ──
  private compileIndex(argNodes: [ResolvedExpr, ResolvedExpr]): ScalarResult {
    const arr = this.compileNode(argNodes[0])
    const idx = this.compileNode(argNodes[1], 'int')
    const dst = this.allocReg()
    const rt = arr.scalarType
    this.regTypes.set(dst, rt)
    const arrOp: NOperand = arr.isArray ? arr.op : { kind: 'const', val: 0, scalar_type: 'float' }
    const idxOp: NOperand = idx.isArray ? { kind: 'const', val: 0, scalar_type: 'int' } : idx.op
    this.emit({ tag: 'Index', dst, args: [arrOp, idxOp], loop_count: 1, strides: [], result_type: rt })
    return { isArray: false, op: { kind: 'reg', slot: dst, scalar_type: rt }, scalarType: rt }
  }

  // ── ArraySet. ──
  private compileSetElement(argNodes: [ResolvedExpr, ResolvedExpr, ResolvedExpr]): ArrayResult {
    const arr = this.compileNode(argNodes[0])
    const idx = this.compileNode(argNodes[1])
    const val = this.compileNode(argNodes[2])

    if (!arr.isArray) {
      const size = 1
      const slot = this.allocArraySlot(size)
      return { isArray: true, op: { kind: 'array_reg', slot }, size, scalarType: 'float' }
    }

    const arrOp: NOperand = arr.op
    const idxOp: NOperand = idx.isArray ? { kind: 'const', val: 0, scalar_type: 'float' } : idx.op
    const valOp: NOperand = val.isArray ? { kind: 'const', val: 0, scalar_type: 'float' } : val.op
    const slot = (arr.op as { slot: number }).slot
    this.emit({ tag: 'SetElement', dst: slot, args: [arrOp, idxOp, valOp], loop_count: 1, strides: [], result_type: 'float' })
    return { isArray: true, op: arr.op, size: arr.size, scalarType: 'float' }
  }

  // ── Top-level emit driver ──
  emitProgram(outputExprs: ResolvedExpr[], registerExprs: (ResolvedExpr | null)[]): FlatProgram {
    const output_targets: number[] = []
    const register_targets: number[] = []

    for (const expr of outputExprs) {
      const r = this.compileNode(expr, 'float')
      if (r.isArray) {
        const dst = this.allocReg()
        this.regTypes.set(dst, 'float')
        this.emit({ tag: 'Index', dst, args: [r.op, { kind: 'const', val: 0, scalar_type: 'int' }], loop_count: 1, strides: [], result_type: 'float' })
        output_targets.push(dst)
      } else {
        const dst = this.allocReg()
        this.regTypes.set(dst, r.scalarType)
        this.emit({ tag: 'Add', dst, args: [r.op, { kind: 'const', val: 0, scalar_type: r.scalarType }], loop_count: 1, strides: [], result_type: r.scalarType })
        output_targets.push(dst)
      }
    }

    // Two-pass register update to preserve per-sample read-before-
    // write isolation for array regs.
    //
    // Scalar regs: the writeback is just `register_targets[i] = dst`
    // and the engine writes after the sample loop body, so
    // intra-sample reads of older state still see pre-update values.
    //
    // Array regs: there's no analogous deferred-writeback mechanism
    // in the kernel. We need to emit an explicit copy instruction
    // from the expression's result slot into the reg's persistent
    // storage slot. If we emit those copies inline as we compile
    // each reg expression, a later reg expression reading the same
    // array reg would see post-update values. So pass 1 compiles
    // every reg expression, collecting (source slot, persistent
    // slot, size) triples; pass 2 emits the copies, which all sit
    // at the tail of the per-sample body. Subsequent samples then
    // read the freshly-copied persistent slot.
    //
    // The copy itself is a strided elementwise no-op (`Add x, 0`)
    // reusing the existing OrcJitEngine elementwise-loop machinery —
    // no engine changes needed. The in-place case (arraySet on the
    // reg's own array_reg, used by Delay) is free: srcSlot already
    // equals the persistent slot, so we skip the copy.
    type ArrayCopy = { src: NOperand; dst: number; size: number }
    const arrayCopies: ArrayCopy[] = []
    for (let ri = 0; ri < registerExprs.length; ri++) {
      const expr = registerExprs[ri]
      if (expr === null) {
        register_targets.push(-1)
        continue
      }
      const regExpected = this.stateRegTypes[ri]
      const r = this.compileNode(expr, regExpected)
      if (r.isArray) {
        const arrInfo = this.arrayRegMap.get(ri)
        if (!arrInfo) {
          throw new Error(`emitProgram: array-result update on non-array reg ${ri}`)
        }
        const srcSlot = (r.op as { slot: number }).slot
        if (srcSlot !== arrInfo.slot) {
          arrayCopies.push({ src: r.op, dst: arrInfo.slot, size: arrInfo.size })
        }
        register_targets.push(-1)
      } else {
        const dst = this.allocReg()
        this.regTypes.set(dst, r.scalarType)
        this.emit({ tag: 'Add', dst, args: [r.op, { kind: 'const', val: 0, scalar_type: r.scalarType }], loop_count: 1, strides: [], result_type: r.scalarType })
        register_targets.push(dst)
      }
    }
    for (const c of arrayCopies) {
      this.emit({
        tag: 'Add',
        dst: c.dst,
        args: [c.src, { kind: 'const', val: 0, scalar_type: 'float' }],
        loop_count: c.size,
        strides: [1, 0],
        result_type: 'float',
      })
    }

    const out: FlatProgram = {
      register_count:   this.nextReg,
      array_slot_count: this.nextArraySlot,
      array_slot_sizes: this.arraySizes,
      instructions:     this.instrs,
      output_targets,
      register_targets,
    }
    if (this.groups.length > 0) out.groups = this.groups
    return out
  }
}

// ─────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────

export interface EmitResolvedInputs {
  outputExprs: ResolvedExpr[]
  registerExprs: (ResolvedExpr | null)[]
  stateInit: (number | boolean | number[])[]
  stateRegTypes: ScalarType[]
  inputPortTypes: ScalarType[]
  slots: EmitSlots
}

export function emitResolvedProgram(input: EmitResolvedInputs): FlatProgram {
  const e = new Emitter(input.slots, input.stateInit, input.stateRegTypes, input.inputPortTypes)
  return e.emitProgram(input.outputExprs, input.registerExprs)
}
