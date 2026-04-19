/**
 * emit_numeric.ts — TS instruction emitter for tropical_plan_4.
 *
 * Replaces the C++ PlanParser + ExprCompiler + NumericProgramBuilder pipeline.
 * Walks ExprNode trees (post-flatten, post-lower_arrays) and emits a flat
 * FlatProgram instruction stream consumed by OrcJitEngine::compile_flat_program().
 *
 * Terminals (literals, inputs, registers, params) become Operands embedded
 * inside instructions — no separate pseudo-ops for them.
 *
 * loop_count > 1 indicates an elementwise loop; strides[i] = 1 means arg i
 * advances with the loop index (array), 0 means it broadcasts (scalar).
 *
 * Every operand carries a scalar_type ('float' | 'int' | 'bool') and every
 * instruction carries a result_type. The C++ JIT uses these to emit native
 * typed LLVM IR (i64 for int, i1 for bool) instead of f64 round-trips.
 */

import type { ExprNode } from './expr.js'

// ─────────────────────────────────────────────────────────────
// Public types (mirror OrcJitEngine.hpp FlatProgram / FlatInstr)
// ─────────────────────────────────────────────────────────────

export type ScalarType = 'float' | 'int' | 'bool'

export type NOperand =
  | { kind: 'const';     val: number;  scalar_type: ScalarType }
  | { kind: 'input';     slot: number; scalar_type: ScalarType }
  | { kind: 'reg';       slot: number; scalar_type: ScalarType }      // scalar temp register
  | { kind: 'array_reg'; slot: number }      // array slot (element type TBD)
  | { kind: 'state_reg'; slot: number; scalar_type: ScalarType }      // persistent module register
  | { kind: 'param';     ptr: string;  scalar_type: ScalarType }       // ControlParam ptr as decimal string
  | { kind: 'rate';      scalar_type: ScalarType }
  | { kind: 'tick';      scalar_type: ScalarType }

export type NInstr = {
  tag:         string       // OpTag name (e.g. 'Add', 'Sin', 'Pack')
  dst:         number       // scalar temp index (loop_count==1) or array slot (loop_count>1 / Pack)
  args:        NOperand[]
  loop_count:  number       // 1 = scalar; N > 1 = elementwise loop over N elements
  strides:     number[]     // per-arg: 1 = iterate (arr[i]), 0 = broadcast
  result_type: ScalarType   // scalar type of the result written to dst
}

export type FlatProgram = {
  register_count:   number     // number of scalar temp registers needed
  array_slot_count: number     // number of array slots needed
  array_slot_sizes: number[]   // element count per array slot
  instructions:     NInstr[]
  output_targets:   number[]   // temps[output_targets[i]] holds output i
  register_targets: number[]   // temps[register_targets[i]] holds new value for register i; -1 = unchanged
}

// ─────────────────────────────────────────────────────────────
// Op-string → OpTag mapping
// ─────────────────────────────────────────────────────────────

const BINARY_TAG: Record<string, string> = {
  add: 'Add', sub: 'Sub', mul: 'Mul', div: 'Div', mod: 'Mod',
  floor_div: 'FloorDiv', floorDiv: 'FloorDiv',
  lt: 'Less', lte: 'LessEq', gt: 'Greater', gte: 'GreaterEq',
  eq: 'Equal', neq: 'NotEqual',
  bit_and: 'BitAnd', bit_or: 'BitOr', bit_xor: 'BitXor',
  bitAnd: 'BitAnd', bitOr: 'BitOr', bitXor: 'BitXor',
  lshift: 'LShift', rshift: 'RShift',
  and: 'And', or: 'Or',
  ldexp: 'Ldexp',
}

const UNARY_TAG: Record<string, string> = {
  neg: 'Neg', abs: 'Abs', sqrt: 'Sqrt',
  floor: 'Floor', ceil: 'Ceil', round: 'Round',
  not: 'Not', bit_not: 'BitNot',
  float_exponent: 'FloatExponent',
}

// ─────────────────────────────────────────────────────────────
// Type inference
// ─────────────────────────────────────────────────────────────

const BITWISE_TAGS = new Set(['BitAnd', 'BitOr', 'BitXor', 'LShift', 'RShift', 'BitNot'])
const COMPARISON_TAGS = new Set(['Less', 'LessEq', 'Greater', 'GreaterEq', 'Equal', 'NotEqual', 'Not', 'And', 'Or'])
const TRANSCENDENTAL_TAGS = new Set(['Sqrt', 'Floor', 'Ceil', 'Round', 'Ldexp', 'FloatExponent'])

/** Promotion order: float > int > bool. */
function promoteTypes(a: ScalarType, b: ScalarType): ScalarType {
  if (a === 'float' || b === 'float') return 'float'
  if (a === 'int' || b === 'int') return 'int'
  return 'bool'
}

/** Infer the result ScalarType of an op from its tag and argument types. */
function inferResultType(tag: string, argTypes: ScalarType[]): ScalarType {
  if (BITWISE_TAGS.has(tag)) return 'int'
  if (COMPARISON_TAGS.has(tag)) return 'bool'
  if (TRANSCENDENTAL_TAGS.has(tag)) return 'float'
  // Select: type of then/else branches (args[1], args[2])
  if (tag === 'Select') return promoteTypes(argTypes[1] ?? 'float', argTypes[2] ?? 'float')
  // Clamp: type of value (args[0])
  if (tag === 'Clamp') return argTypes[0] ?? 'float'
  // Arithmetic (Add, Sub, Mul, Div, Mod, FloorDiv, Neg, Abs): promote all args
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
// Emitter class
// ─────────────────────────────────────────────────────────────

class Emitter {
  private nextReg       = 0
  private nextArraySlot = 0
  private arraySizes:   number[] = []
  private instrs:       NInstr[] = []
  // CSE: memoize compiled results by object identity (WeakMap keyed on the ExprNode object).
  // Identity-based lookup is O(1) vs O(JSON size) for structural hashing, which matters
  // when expression DAGs have many shared subexpressions (e.g. nested module call chains).
  // Terminals are not memoized — they allocate nothing and return an Operand directly.
  private memo = new WeakMap<object, CompileResult>()
  // Map register ID → pre-allocated array slot for registers whose init value is an array.
  private arrayRegMap = new Map<number, { slot: number, size: number }>()
  // Type tracking: temp register slot → ScalarType
  private regTypes = new Map<number, ScalarType>()
  // Type annotations for state registers (from module registerPortTypes)
  private stateRegTypes: ScalarType[]

  constructor(
    stateInit?: (number | boolean | number[])[],
    stateRegTypes?: ScalarType[],
  ) {
    this.stateRegTypes = stateRegTypes ?? []
    if (stateInit) {
      for (let i = 0; i < stateInit.length; i++) {
        const init = stateInit[i]
        if (Array.isArray(init)) {
          const slot = this.allocArraySlot(init.length)
          this.arrayRegMap.set(i, { slot, size: init.length })
        }
      }
    }
  }

  // Allocate a new scalar temporary register.
  private allocReg(): number { return this.nextReg++ }

  // Allocate a new array slot of `size` elements.
  private allocArraySlot(size: number): number {
    const slot = this.nextArraySlot++
    this.arraySizes.push(size)
    return slot
  }

  private emit(instr: NInstr): void { this.instrs.push(instr) }

  // ── Terminal check: if node is a leaf, return its Operand + type directly. ──
  private tryTerminal(node: ExprNode): { op: NOperand; scalarType: ScalarType } | null {
    if (typeof node === 'number')  return { op: { kind: 'const', val: node, scalar_type: 'float' }, scalarType: 'float' }
    if (typeof node === 'boolean') return { op: { kind: 'const', val: node ? 1 : 0, scalar_type: 'bool' }, scalarType: 'bool' }
    if (typeof node !== 'object' || node === null) return { op: { kind: 'const', val: 0, scalar_type: 'float' }, scalarType: 'float' }
    const obj = node as { op: string; [k: string]: unknown }
    switch (obj.op) {
      case 'input':        return { op: { kind: 'input', slot: obj.id as number, scalar_type: 'float' }, scalarType: 'float' }
      case 'reg': {
        if (this.arrayRegMap.has(obj.id as number)) return null  // array register — handled in compileNodeUncached
        const regType = this.stateRegTypes[obj.id as number] ?? 'float'
        return { op: { kind: 'state_reg', slot: obj.id as number, scalar_type: regType }, scalarType: regType }
      }
      case 'sample_rate':  return { op: { kind: 'rate', scalar_type: 'float' }, scalarType: 'float' }
      case 'sample_index': return { op: { kind: 'tick', scalar_type: 'int' }, scalarType: 'int' }
      case 'smoothed_param':
      case 'trigger_param':
        // Params embed their C++ pointer. Serialize as decimal string (JSON-safe).
        if (obj._ptr && obj._handle != null) {
          return { op: { kind: 'param', ptr: String(obj._handle), scalar_type: 'float' }, scalarType: 'float' }
        }
        return { op: { kind: 'const', val: 0, scalar_type: 'float' }, scalarType: 'float' }  // no live handle
    }
    return null  // not a terminal
  }

  // ── Compile a node to an operand (emitting instructions as needed). ──
  compileNode(node: ExprNode): CompileResult {
    // Terminal shortcut — no allocation, skip memo
    const terminal = this.tryTerminal(node)
    if (terminal !== null) return { isArray: false, op: terminal.op, scalarType: terminal.scalarType }

    // CSE: check memo before allocating anything
    const cached = this.memo.get(node as object)
    if (cached !== undefined) return cached

    const result = this.compileNodeUncached(node)
    this.memo.set(node as object, result)
    return result
  }

  private compileNodeUncached(node: ExprNode): CompileResult {
    // Inline JS array → Pack instruction
    if (Array.isArray(node)) {
      return this.compilePack(node as ExprNode[])
    }

    const obj = node as { op: string; [k: string]: unknown }

    // Array register reference (skipped by tryTerminal)
    if (obj.op === 'reg') {
      const arrInfo = this.arrayRegMap.get(obj.id as number)
      if (arrInfo) return { isArray: true, op: { kind: 'array_reg', slot: arrInfo.slot }, size: arrInfo.size, scalarType: 'float' }
    }

    // {"op":"array","items":[...]} — JSON patch format for inline arrays
    if (obj.op === 'array' && Array.isArray(obj.items)) {
      return this.compilePack(obj.items as ExprNode[])
    }

    // Binary ops
    const binTag = BINARY_TAG[obj.op]
    if (binTag) return this.compileBinary(binTag, obj.args as ExprNode[])

    // Unary ops
    const uniTag = UNARY_TAG[obj.op]
    if (uniTag) return this.compileUnary(uniTag, (obj.args as ExprNode[])[0])

    // Ternary ops
    if (obj.op === 'clamp')  return this.compileTernary('Clamp',  obj.args as ExprNode[])
    if (obj.op === 'select') return this.compileTernary('Select', obj.args as ExprNode[])

    // Array ops
    if (obj.op === 'index')     return this.compileIndex(obj.args as ExprNode[])
    if (obj.op === 'array_set') return this.compileSetElement(obj.args as ExprNode[])

    // Matrix literal → flatten rows → Pack
    if (obj.op === 'matrix') {
      const rows = obj.rows as ExprNode[][]
      const flat = rows.flat() as ExprNode[]
      return this.compilePack(flat)
    }

    // broadcast_to surviving lower_arrays (dynamic, non-literal src)
    if (obj.op === 'broadcast_to') return this.compileBroadcastTo(obj)

    // Fallthrough: emit a zero constant (unknown op — safe stub)
    console.warn(`emit_numeric: unhandled op '${obj.op}', substituting 0`)
    return { isArray: false, op: { kind: 'const', val: 0, scalar_type: 'float' }, scalarType: 'float' }
  }

  // ── Unbox a size-1 array to a scalar via Index[0].
  // Needed because loop_count == 1 is the scalar signal to the JIT — using it for
  // a size-1 array loop would cause the JIT's scalar path to receive ArrayReg args
  // (which resolve_scalar returns nullptr for), causing a null-deref / segfault.
  private unboxArray(arr: ArrayResult): ScalarResult {
    const dst = this.allocReg()
    const rt = arr.scalarType
    this.regTypes.set(dst, rt)
    this.emit({ tag: 'Index', dst, args: [arr.op, { kind: 'const', val: 0, scalar_type: 'int' }], loop_count: 1, strides: [], result_type: rt })
    return { isArray: false, op: { kind: 'reg', slot: dst, scalar_type: rt }, scalarType: rt }
  }

  // ── Compile an inline JS array to a Pack instruction. ──
  private compilePack(elements: ExprNode[]): ArrayResult {
    const size = elements.length
    const slot = this.allocArraySlot(size)
    const args: NOperand[] = elements.map(e => {
      const r = this.compileNode(e)
      // Pack expects scalar operands; if element is array, flatten is unsupported here
      return r.isArray ? { kind: 'const' as const, val: 0, scalar_type: 'float' as ScalarType } : r.op
    })
    this.emit({ tag: 'Pack', dst: slot, args, loop_count: 1, strides: [], result_type: 'float' })
    return { isArray: true, op: { kind: 'array_reg', slot }, size, scalarType: 'float' }
  }

  // ── Compile a binary op, routing to scalar or elementwise. ──
  private compileBinary(tag: string, argNodes: ExprNode[]): CompileResult {
    let l = this.compileNode(argNodes[0])
    let r = this.compileNode(argNodes[1])

    // Unbox size-1 arrays: loop_count==1 is the scalar signal; don't emit ArrayReg into it.
    if (l.isArray && l.size === 1) l = this.unboxArray(l)
    if (r.isArray && r.size === 1) r = this.unboxArray(r)

    const rt = inferResultType(tag, [l.scalarType, r.scalarType])

    if (!l.isArray && !r.isArray) {
      // Scalar × scalar → scalar instruction
      const dst = this.allocReg()
      this.regTypes.set(dst, rt)
      this.emit({ tag, dst, args: [l.op, r.op], loop_count: 1, strides: [], result_type: rt })
      return { isArray: false, op: { kind: 'reg', slot: dst, scalar_type: rt }, scalarType: rt }
    }

    // At least one array (size > 1) → elementwise loop
    const size = l.isArray ? l.size : (r as ArrayResult).size
    const slot = this.allocArraySlot(size)
    const strides = [l.isArray ? 1 : 0, r.isArray ? 1 : 0]
    this.emit({ tag, dst: slot, args: [l.op, r.op], loop_count: size, strides, result_type: rt })
    return { isArray: true, op: { kind: 'array_reg', slot }, size, scalarType: rt }
  }

  private compileUnary(tag: string, argNode: ExprNode): CompileResult {
    let a = this.compileNode(argNode)
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

  private compileTernary(tag: string, argNodes: ExprNode[]): CompileResult {
    let a = this.compileNode(argNodes[0])
    let b = this.compileNode(argNodes[1])
    let c = this.compileNode(argNodes[2])

    // Unbox size-1 arrays before checking array path
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

  // ── index(arr, idx) → scalar element ──
  private compileIndex(argNodes: ExprNode[]): ScalarResult {
    const arr = this.compileNode(argNodes[0])
    const idx = this.compileNode(argNodes[1])
    const dst = this.allocReg()
    const rt = arr.scalarType  // element type of the array
    this.regTypes.set(dst, rt)
    const arrOp: NOperand = arr.isArray ? arr.op : { kind: 'const', val: 0, scalar_type: 'float' }
    const idxOp: NOperand = idx.isArray ? { kind: 'const', val: 0, scalar_type: 'int' } : idx.op
    this.emit({ tag: 'Index', dst, args: [arrOp, idxOp], loop_count: 1, strides: [], result_type: rt })
    return { isArray: false, op: { kind: 'reg', slot: dst, scalar_type: rt }, scalarType: rt }
  }

  // ── array_set(arr, idx, val) → same array slot (side-effect mutation) ──
  private compileSetElement(argNodes: ExprNode[]): ArrayResult {
    const arr = this.compileNode(argNodes[0])
    const idx = this.compileNode(argNodes[1])
    const val = this.compileNode(argNodes[2])

    if (!arr.isArray) {
      // Degenerate — shouldn't happen after lower_arrays; return a fresh zero array
      const size = 1
      const slot = this.allocArraySlot(size)
      return { isArray: true, op: { kind: 'array_reg', slot }, size, scalarType: 'float' }
    }

    const arrOp: NOperand = arr.op
    const idxOp: NOperand = idx.isArray ? { kind: 'const', val: 0, scalar_type: 'float' } : idx.op
    const valOp: NOperand = val.isArray ? { kind: 'const', val: 0, scalar_type: 'float' } : val.op
    // dst is ignored for SetElement (side-effect only) but must be a valid array slot
    const slot = (arr.op as { slot: number }).slot
    this.emit({ tag: 'SetElement', dst: slot, args: [arrOp, idxOp, valOp], loop_count: 1, strides: [], result_type: 'float' })
    return { isArray: true, op: arr.op, size: arr.size, scalarType: 'float' }
  }

  // ── broadcast_to surviving lower_arrays (dynamic array source) ──
  private compileBroadcastTo(obj: Record<string, unknown>): CompileResult {
    const src = this.compileNode((obj.args as ExprNode[])[0])
    const shape = obj.shape as number[]
    const targetSize = shape.reduce((a, b) => a * b, 1)

    if (!src.isArray) {
      // Scalar broadcast: elementwise copy with stride=0
      const slot = this.allocArraySlot(targetSize)
      this.emit({ tag: 'Add', dst: slot, args: [src.op, { kind: 'const', val: 0, scalar_type: 'float' }], loop_count: targetSize, strides: [0, 0], result_type: 'float' })
      return { isArray: true, op: { kind: 'array_reg', slot }, size: targetSize, scalarType: 'float' }
    }

    if (src.size === targetSize) return src  // already the right size

    // Repeat-tile (src.size divides targetSize, each src element repeats targetSize/src.size times)
    // Implement as elementwise: dst[i] = src[i % src.size]
    // Can't do modulo indexing in the current loop scheme; fall back to Pack unrolling if small enough
    if (targetSize <= 64) {
      const elems: NOperand[] = []
      for (let i = 0; i < targetSize; i++) {
        const idxReg = this.allocReg()
        this.emit({ tag: 'Index', dst: idxReg, args: [src.op, { kind: 'const', val: i % src.size, scalar_type: 'float' }], loop_count: 1, strides: [], result_type: 'float' })
        elems.push({ kind: 'reg', slot: idxReg, scalar_type: 'float' })
      }
      const slot = this.allocArraySlot(targetSize)
      this.emit({ tag: 'Pack', dst: slot, args: elems, loop_count: 1, strides: [], result_type: 'float' })
      return { isArray: true, op: { kind: 'array_reg', slot }, size: targetSize, scalarType: 'float' }
    }

    // Large dynamic broadcast — stub with zero array
    console.warn(`emit_numeric: large dynamic broadcast_to (${src.size} → ${targetSize}), substituting zeros`)
    const slot = this.allocArraySlot(targetSize)
    const zeroArgs: NOperand[] = new Array(targetSize).fill(null).map(() => ({ kind: 'const' as const, val: 0, scalar_type: 'float' as ScalarType }))
    this.emit({ tag: 'Pack', dst: slot, args: zeroArgs, loop_count: 1, strides: [], result_type: 'float' })
    return { isArray: true, op: { kind: 'array_reg', slot }, size: targetSize, scalarType: 'float' }
  }

  // ── Entry point ──
  emitProgram(
    outputExprs: ExprNode[],
    registerExprs: (ExprNode | null)[],
  ): FlatProgram {
    const output_targets: number[] = []
    const register_targets: number[] = []

    for (const expr of outputExprs) {
      const r = this.compileNode(expr)
      if (r.isArray) {
        // Array output → index element 0 as scalar output (always float for audio)
        const dst = this.allocReg()
        this.regTypes.set(dst, 'float')
        this.emit({ tag: 'Index', dst, args: [r.op, { kind: 'const', val: 0, scalar_type: 'int' }], loop_count: 1, strides: [], result_type: 'float' })
        output_targets.push(dst)
      } else {
        // Copy to output temp, preserving float for audio path
        const dst = this.allocReg()
        this.regTypes.set(dst, r.scalarType)
        this.emit({ tag: 'Add', dst, args: [r.op, { kind: 'const', val: 0, scalar_type: r.scalarType }], loop_count: 1, strides: [], result_type: r.scalarType })
        output_targets.push(dst)
      }
    }

    for (const expr of registerExprs) {
      if (expr === null) {
        register_targets.push(-1)
        continue
      }
      const r = this.compileNode(expr)
      if (r.isArray) {
        register_targets.push(-1)  // array registers not supported as state reg targets yet
      } else {
        // Register update: preserve the expression type
        const dst = this.allocReg()
        this.regTypes.set(dst, r.scalarType)
        this.emit({ tag: 'Add', dst, args: [r.op, { kind: 'const', val: 0, scalar_type: r.scalarType }], loop_count: 1, strides: [], result_type: r.scalarType })
        register_targets.push(dst)
      }
    }

    return {
      register_count:   this.nextReg,
      array_slot_count: this.nextArraySlot,
      array_slot_sizes: this.arraySizes,
      instructions:     this.instrs,
      output_targets,
      register_targets,
    }
  }
}

// ─────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────

/**
 * Compile flat ExprNode trees into a FlatProgram instruction stream.
 *
 * @param outputExprs     One expression per output port.
 * @param registerExprs   One expression per state register; null = no update.
 * @param stateInit       Initial values for state registers.
 * @param stateRegTypes   ScalarType per state register (from module registerPortTypes).
 */
export function emitNumericProgram(
  outputExprs: ExprNode[],
  registerExprs: (ExprNode | null)[],
  stateInit?: (number | boolean | number[])[],
  stateRegTypes?: ScalarType[],
): FlatProgram {
  return new Emitter(stateInit, stateRegTypes).emitProgram(outputExprs, registerExprs)
}
