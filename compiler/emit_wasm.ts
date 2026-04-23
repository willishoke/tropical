/**
 * emit_wasm.ts — tropical_plan_4 → WebAssembly binary module.
 *
 * Web counterpart to engine/jit/OrcJitEngine.cpp. Takes a FlatPlan, emits a
 * standalone WASM module whose exported `process(buffer_length, start_sample_index)`
 * function runs the kernel.
 *
 * Kernel structure mirrors the native loop (OrcJitEngine.cpp:730-987):
 *   for s in 0..buffer_length:
 *     sample_idx = start_sample_index + s
 *     (run all instructions, writing to temps[])
 *     register writeback: temps[register_targets[i]] → registers[i]
 *     output[s] = sum(temps[output_targets[outputs[i]]]) / 20
 *
 * The module exports a single shared memory. Layout is fixed at emit time
 * by wasm_memory_layout.ts:
 *   inputs | registers | temps | arrays | param_table | param_frame | output
 *
 * Storage:
 *   - registers[]  → i64 cells (float bitcast; int signed; bool zext 0/1)
 *   - temps[]      → i64 cells (same encoding)
 *   - arrays[]     → f64 cells (state transfer stays WASM-only)
 *   - param_table  → f64 cells written by the host
 *   - param_frame  → f64 cells snapshotted per block for TriggerParam
 */

import type { FlatPlan } from './flatten.js'
import type { NInstr, NOperand, ScalarType } from './emit_numeric.js'
import { type WasmLayout, computeLayout, collectParamPtrs } from './wasm_memory_layout.js'

// ─────────────────────────────────────────────────────────────
// Byte writer + LEB128
// ─────────────────────────────────────────────────────────────

class ByteWriter {
  private buf: number[] = []
  get length(): number { return this.buf.length }
  bytes(): Uint8Array { return Uint8Array.from(this.buf) }

  u8(b: number): void { this.buf.push(b & 0xff) }
  raw(src: Uint8Array | number[]): void { for (const b of src) this.buf.push(b & 0xff) }

  uleb(n: number | bigint): void {
    let v = typeof n === 'bigint' ? n : BigInt(n)
    do {
      let b = Number(v & 0x7fn)
      v >>= 7n
      if (v !== 0n) b |= 0x80
      this.buf.push(b)
    } while (v !== 0n)
  }

  sleb(n: number | bigint): void {
    let v = typeof n === 'bigint' ? n : BigInt(n)
    while (true) {
      const b = Number(v & 0x7fn)
      const sign = b & 0x40
      v >>= 7n
      if ((v === 0n && sign === 0) || (v === -1n && sign !== 0)) {
        this.buf.push(b)
        return
      }
      this.buf.push(b | 0x80)
    }
  }

  f64(x: number): void {
    const dv = new DataView(new ArrayBuffer(8))
    dv.setFloat64(0, x, true)
    for (let i = 0; i < 8; i++) this.buf.push(dv.getUint8(i))
  }

  name(s: string): void {
    const u = new TextEncoder().encode(s)
    this.uleb(u.length)
    for (const b of u) this.buf.push(b)
  }
}

// ─────────────────────────────────────────────────────────────
// Value types + opcode table (only what we use)
// ─────────────────────────────────────────────────────────────

const VT = { i32: 0x7f, i64: 0x7e, f64: 0x7c } as const
type VType = typeof VT[keyof typeof VT]

const OP = {
  BLOCK: 0x02, LOOP: 0x03, IF: 0x04, ELSE: 0x05, END: 0x0b,
  BR: 0x0c, BR_IF: 0x0d, DROP: 0x1a, SELECT: 0x1b,
  LOCAL_GET: 0x20, LOCAL_SET: 0x21, LOCAL_TEE: 0x22,
  I64_LOAD: 0x29, F64_LOAD: 0x2b,
  I64_STORE: 0x37, F64_STORE: 0x39,
  I32_CONST: 0x41, I64_CONST: 0x42, F64_CONST: 0x44,
  I32_EQZ: 0x45, I32_EQ: 0x46, I32_NE: 0x47,
  I32_LT_U: 0x49, I32_GE_U: 0x4f,
  I64_EQZ: 0x50, I64_EQ: 0x51, I64_NE: 0x52,
  I64_LT_S: 0x53, I64_GT_S: 0x55, I64_LE_S: 0x57, I64_GE_S: 0x59,
  F64_EQ: 0x61, F64_NE: 0x62, F64_LT: 0x63, F64_GT: 0x64, F64_LE: 0x65, F64_GE: 0x66,
  I32_ADD: 0x6a, I32_SUB: 0x6b, I32_MUL: 0x6c, I32_AND: 0x71, I32_OR: 0x72,
  I64_ADD: 0x7c, I64_SUB: 0x7d, I64_MUL: 0x7e, I64_DIV_S: 0x7f,
  I64_REM_S: 0x81, I64_AND: 0x83, I64_OR: 0x84, I64_XOR: 0x85,
  I64_SHL: 0x86, I64_SHR_S: 0x87,
  F64_ABS: 0x99, F64_NEG: 0x9a, F64_CEIL: 0x9b, F64_FLOOR: 0x9c,
  F64_TRUNC: 0x9d, F64_NEAREST: 0x9e, F64_SQRT: 0x9f,
  F64_ADD: 0xa0, F64_SUB: 0xa1, F64_MUL: 0xa2, F64_DIV: 0xa3,
  I32_WRAP_I64: 0xa7, I64_EXTEND_I32_S: 0xac, I64_EXTEND_I32_U: 0xad,
  F64_CONVERT_I32_U: 0xb8, F64_CONVERT_I64_S: 0xb9,
  I64_REINTERPRET_F64: 0xbd, F64_REINTERPRET_I64: 0xbf,
  FC: 0xfc, I64_TRUNC_SAT_F64_S: 0x06,
} as const

// ─────────────────────────────────────────────────────────────
// Code emitter
// ─────────────────────────────────────────────────────────────

class Code {
  w = new ByteWriter()
  u8(b: number): void { this.w.u8(b) }

  end(): void { this.u8(OP.END) }
  drop(): void { this.u8(OP.DROP) }
  select(): void { this.u8(OP.SELECT) }

  block(): void { this.u8(OP.BLOCK); this.u8(0x40) }
  loop(): void  { this.u8(OP.LOOP);  this.u8(0x40) }
  if_(): void   { this.u8(OP.IF);    this.u8(0x40) }
  else_(): void { this.u8(OP.ELSE) }
  br(d: number): void   { this.u8(OP.BR);    this.w.uleb(d) }
  brIf(d: number): void { this.u8(OP.BR_IF); this.w.uleb(d) }

  localGet(i: number): void { this.u8(OP.LOCAL_GET); this.w.uleb(i) }
  localSet(i: number): void { this.u8(OP.LOCAL_SET); this.w.uleb(i) }
  localTee(i: number): void { this.u8(OP.LOCAL_TEE); this.w.uleb(i) }

  i32c(n: number): void { this.u8(OP.I32_CONST); this.w.sleb(n) }
  i64c(n: number | bigint): void { this.u8(OP.I64_CONST); this.w.sleb(n) }
  f64c(x: number): void { this.u8(OP.F64_CONST); this.w.f64(x) }

  private memarg(align: number, offset: number): void {
    this.w.uleb(align)
    this.w.uleb(offset)
  }
  i64Load(offset: number): void { this.u8(OP.I64_LOAD); this.memarg(3, offset) }
  f64Load(offset: number): void { this.u8(OP.F64_LOAD); this.memarg(3, offset) }
  i64Store(offset: number): void { this.u8(OP.I64_STORE); this.memarg(3, offset) }
  f64Store(offset: number): void { this.u8(OP.F64_STORE); this.memarg(3, offset) }

  i64TruncSat(): void { this.u8(OP.FC); this.w.uleb(OP.I64_TRUNC_SAT_F64_S) }

  bytes(): Uint8Array { return this.w.bytes() }
}

// ─────────────────────────────────────────────────────────────
// Module builder
// ─────────────────────────────────────────────────────────────

type FuncDef = { typeIdx: number; localGroups: { count: number; type: VType }[]; body: Uint8Array }

class Module {
  types: { params: VType[]; results: VType[] }[] = []
  funcs: FuncDef[] = []
  mem: { min: number; max: number | null } | null = null
  exports: { name: string; kind: 0 | 2; idx: number }[] = []

  addType(params: VType[], results: VType[]): number {
    this.types.push({ params, results })
    return this.types.length - 1
  }
  addFunc(f: FuncDef): number { this.funcs.push(f); return this.funcs.length - 1 }
  addMem(min: number, max: number | null = null): void { this.mem = { min, max } }
  addExport(name: string, kind: 0 | 2, idx: number): void { this.exports.push({ name, kind, idx }) }

  build(): Uint8Array {
    const out = new ByteWriter()
    out.raw([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00])

    const emitSection = (id: number, payload: Uint8Array) => {
      if (payload.length === 0) return
      out.u8(id)
      out.uleb(payload.length)
      out.raw(payload)
    }

    // 1: Type
    {
      const p = new ByteWriter()
      p.uleb(this.types.length)
      for (const t of this.types) {
        p.u8(0x60)
        p.uleb(t.params.length); for (const v of t.params) p.u8(v)
        p.uleb(t.results.length); for (const v of t.results) p.u8(v)
      }
      emitSection(1, p.bytes())
    }
    // 3: Function
    {
      const p = new ByteWriter()
      p.uleb(this.funcs.length)
      for (const f of this.funcs) p.uleb(f.typeIdx)
      emitSection(3, p.bytes())
    }
    // 5: Memory
    if (this.mem) {
      const p = new ByteWriter()
      p.uleb(1)
      if (this.mem.max !== null) {
        p.u8(1); p.uleb(this.mem.min); p.uleb(this.mem.max)
      } else {
        p.u8(0); p.uleb(this.mem.min)
      }
      emitSection(5, p.bytes())
    }
    // 7: Export
    {
      const p = new ByteWriter()
      p.uleb(this.exports.length)
      for (const e of this.exports) {
        p.name(e.name)
        p.u8(e.kind)
        p.uleb(e.idx)
      }
      emitSection(7, p.bytes())
    }
    // 10: Code
    {
      const p = new ByteWriter()
      p.uleb(this.funcs.length)
      for (const f of this.funcs) {
        const body = new ByteWriter()
        body.uleb(f.localGroups.length)
        for (const g of f.localGroups) { body.uleb(g.count); body.u8(g.type) }
        body.raw(f.body)
        const bb = body.bytes()
        p.uleb(bb.length)
        p.raw(bb)
      }
      emitSection(10, p.bytes())
    }
    return out.bytes()
  }
}

// ─────────────────────────────────────────────────────────────
// Local indices
// ─────────────────────────────────────────────────────────────
//
// Function params (indices 0,1) + 11 additional locals.
// Layout: [i32 buflen][i64 start_idx] [i32 s][i32 tmp_i32] [i64 sidx][i64 ewi][i64 ai][i64 bi][i64 ci] [f64 af][f64 bf][f64 cf][f64 mixed]

const L_BUFLEN = 0
const L_START  = 1
const L_S      = 2
const L_TI32   = 3
const L_SIDX   = 4
const L_EWI    = 5
const L_AI     = 6
const L_BI     = 7
const L_CI     = 8
const L_AF     = 9
const L_BF     = 10
const L_CF     = 11
const L_MIX    = 12

const LOCALS = [
  { count: 2, type: VT.i32 as VType }, // s, tmp_i32
  { count: 5, type: VT.i64 as VType }, // sidx, ewi, ai, bi, ci
  { count: 4, type: VT.f64 as VType }, // af, bf, cf, mixed
]

// ─────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────

export type EmitWasmResult = {
  bytes: Uint8Array
  layout: WasmLayout
  paramPtrs: string[]
}

export type EmitWasmOptions = {
  inputCount?: number
  maxBlockSize?: number
  sampleRate?: number
}

export function emitWasm(plan: FlatPlan, opts: EmitWasmOptions = {}): EmitWasmResult {
  const inputCount = opts.inputCount ?? 0
  const maxBlockSize = opts.maxBlockSize ?? 2048
  const sampleRate = opts.sampleRate ?? plan.config.sample_rate

  const flatProgram = {
    register_count: plan.register_count,
    array_slot_count: plan.array_slot_count,
    array_slot_sizes: plan.array_slot_sizes,
    instructions: plan.instructions,
    output_targets: plan.output_targets,
    register_targets: plan.register_targets,
  }
  const paramPtrs = collectParamPtrs(flatProgram)
  const paramIndex = new Map<string, number>()
  paramPtrs.forEach((p, i) => paramIndex.set(p, i))

  const layout = computeLayout({ plan: flatProgram, inputCount, paramPtrs, maxBlockSize })
  const ctx: EmitCtx = { layout, paramIndex, sampleRate }

  const c = new Code()

  // s = 0; block { loop { if s >= buflen br 1; body; s++; br 0 } }
  c.i32c(0); c.localSet(L_S)
  c.block()
  c.loop()
  c.localGet(L_S); c.localGet(L_BUFLEN); c.u8(OP.I32_GE_U); c.brIf(1)

  // sidx = start + s
  c.localGet(L_START)
  c.localGet(L_S); c.u8(OP.I64_EXTEND_I32_U)
  c.u8(OP.I64_ADD)
  c.localSet(L_SIDX)

  for (const instr of plan.instructions) emitInstruction(c, instr, ctx)

  // Register writeback
  for (let ri = 0; ri < plan.register_targets.length; ri++) {
    const ti = plan.register_targets[ri]!
    if (ti < 0) continue
    const regType = plan.register_types[ri] ?? 'float'
    const tempOffset = layout.tempsOffset + ti * 8
    const regOffset  = layout.registersOffset + ri * 8

    if (regType === 'bool') {
      // Temp slot holds bool stored as i64 (0 or 1). Normalize and store.
      c.i32c(0); c.i64Load(tempOffset); c.i64c(0); c.u8(OP.I64_NE); c.u8(OP.I64_EXTEND_I32_U)
      c.localSet(L_AI)
      c.i32c(0); c.localGet(L_AI); c.i64Store(regOffset)
    } else {
      // Float/int: temp slot is already in the right bitwise form.
      c.i32c(0); c.i64Load(tempOffset); c.localSet(L_AI)
      c.i32c(0); c.localGet(L_AI); c.i64Store(regOffset)
    }
  }

  // Output mix: output[s] = sum(temps[output_targets[outputs[i]]]) / 20
  c.f64c(0); c.localSet(L_MIX)
  for (const outIdx of plan.outputs) {
    if (outIdx >= plan.output_targets.length) continue
    const tempSlot = plan.output_targets[outIdx]!
    c.localGet(L_MIX)
    c.i32c(0); c.f64Load(layout.tempsOffset + tempSlot * 8)
    c.u8(OP.F64_ADD)
    c.localSet(L_MIX)
  }
  // addr = outputOffset + s*8
  c.localGet(L_S); c.i32c(8); c.u8(OP.I32_MUL); c.i32c(layout.outputOffset); c.u8(OP.I32_ADD)
  c.localGet(L_MIX); c.f64c(20); c.u8(OP.F64_DIV)
  c.f64Store(0)

  // s++
  c.localGet(L_S); c.i32c(1); c.u8(OP.I32_ADD); c.localSet(L_S)
  c.br(0)
  c.end() // loop
  c.end() // block
  c.end() // function

  const m = new Module()
  const tIdx = m.addType([VT.i32, VT.i64], [])
  m.addMem(layout.pageCount)
  const fIdx = m.addFunc({ typeIdx: tIdx, localGroups: LOCALS, body: c.bytes() })
  m.addExport('memory', 2, 0)
  m.addExport('process', 0, fIdx)

  return { bytes: m.build(), layout, paramPtrs }
}

// ─────────────────────────────────────────────────────────────
// Instruction emission
// ─────────────────────────────────────────────────────────────

type EmitCtx = {
  layout: WasmLayout
  paramIndex: Map<string, number>
  sampleRate: number
}

function emitInstruction(c: Code, instr: NInstr, ctx: EmitCtx): void {
  if (instr.tag === 'SmoothParam') return emitSmoothParam(c, instr, ctx)
  if (instr.tag === 'TriggerParam') return emitTriggerParam(c, instr, ctx)
  if (instr.tag === 'Pack') return emitPack(c, instr, ctx)
  if (instr.tag === 'Index') return emitIndex(c, instr, ctx)
  if (instr.tag === 'SetElement') return emitSetElement(c, instr, ctx)
  if (instr.loop_count > 1) return emitElementwise(c, instr, ctx)
  return emitScalar(c, instr, ctx)
}

// ── Operand emitter: pushes a value in its native WASM type, returns the ScalarType ──
function pushOperand(c: Code, op: NOperand, ctx: EmitCtx): ScalarType {
  switch (op.kind) {
    case 'const': {
      if (op.scalar_type === 'float') c.f64c(op.val)
      else if (op.scalar_type === 'int') c.i64c(BigInt(Math.trunc(op.val)))
      else c.i32c(op.val ? 1 : 0)
      return op.scalar_type
    }
    case 'input': {
      c.i32c(0); c.f64Load(ctx.layout.inputsOffset + op.slot * 8)
      return 'float' // inputs always f64 in memory; caller coerces if needed
    }
    case 'reg': {
      const off = ctx.layout.tempsOffset + op.slot * 8
      if (op.scalar_type === 'float') { c.i32c(0); c.f64Load(off); return 'float' }
      if (op.scalar_type === 'int')   { c.i32c(0); c.i64Load(off); return 'int' }
      c.i32c(0); c.i64Load(off); c.u8(OP.I32_WRAP_I64)
      return 'bool'
    }
    case 'state_reg': {
      const off = ctx.layout.registersOffset + op.slot * 8
      if (op.scalar_type === 'float') { c.i32c(0); c.f64Load(off); return 'float' }
      if (op.scalar_type === 'int')   { c.i32c(0); c.i64Load(off); return 'int' }
      c.i32c(0); c.i64Load(off); c.u8(OP.I32_WRAP_I64)
      return 'bool'
    }
    case 'array_reg':
      // Arrays aren't scalar — shouldn't be read as a scalar value. Push 0 as safety.
      c.f64c(0); return 'float'
    case 'param': {
      const idx = ctx.paramIndex.get(op.ptr) ?? 0
      c.i32c(0); c.f64Load(ctx.layout.paramTableOffset + idx * 8)
      return 'float'
    }
    case 'rate': {
      c.f64c(ctx.sampleRate); return 'float'
    }
    case 'tick': {
      c.localGet(L_SIDX); return 'int'
    }
  }
}

// Coerce TOS from `from` to `to`.
function coerce(c: Code, from: ScalarType, to: ScalarType): void {
  if (from === to) return
  if (from === 'float') {
    if (to === 'int')  c.i64TruncSat()
    else { c.f64c(0); c.u8(OP.F64_NE) } // → bool
  } else if (from === 'int') {
    if (to === 'float') c.u8(OP.F64_CONVERT_I64_S)
    else { c.i64c(0); c.u8(OP.I64_NE) } // → bool
  } else { // bool (i32 0/1)
    if (to === 'float') c.u8(OP.F64_CONVERT_I32_U)
    else c.u8(OP.I64_EXTEND_I32_U)  // → int
  }
}

function pushAs(c: Code, op: NOperand, want: ScalarType, ctx: EmitCtx): void {
  const src = pushOperand(c, op, ctx)
  coerce(c, src, want)
}

// Store TOS (in native WASM type for `type`) into temps[dst] as 8-byte cell.
function storeTempAt(c: Code, dst: number, type: ScalarType, ctx: EmitCtx): void {
  const absOffset = ctx.layout.tempsOffset + dst * 8
  if (type === 'float') {
    c.localSet(L_AF); c.i32c(0); c.localGet(L_AF); c.f64Store(absOffset)
  } else if (type === 'int') {
    c.localSet(L_AI); c.i32c(0); c.localGet(L_AI); c.i64Store(absOffset)
  } else {
    c.u8(OP.I64_EXTEND_I32_U)
    c.localSet(L_AI); c.i32c(0); c.localGet(L_AI); c.i64Store(absOffset)
  }
}

// ─────────────────────────────────────────────────────────────
// Specialized ops
// ─────────────────────────────────────────────────────────────

function emitSmoothParam(c: Code, instr: NInstr, ctx: EmitCtx): void {
  const pOp = instr.args[0]!, sOp = instr.args[1]!, kOp = instr.args[2]!
  if (pOp.kind !== 'param' || sOp.kind !== 'state_reg' || kOp.kind !== 'const') {
    throw new Error('SmoothParam: unexpected arg shape')
  }
  const pIdx = ctx.paramIndex.get(pOp.ptr) ?? 0
  const regOff = ctx.layout.registersOffset + sOp.slot * 8
  const paramOff = ctx.layout.paramTableOffset + pIdx * 8

  // cur = registers[slot]; tgt = param
  c.i32c(0); c.f64Load(regOff)
  c.localTee(L_AF)   // cur
  c.i32c(0); c.f64Load(paramOff)       // tgt
  c.localGet(L_AF)                     // cur
  c.u8(OP.F64_SUB)                     // tgt - cur
  c.f64c(kOp.val)
  c.u8(OP.F64_MUL)                     // k*(tgt-cur)
  c.u8(OP.F64_ADD)                     // cur + k*(tgt-cur)  [new]
  c.localTee(L_BF)                     // save new
  // store → registers[slot]
  c.localSet(L_AF)
  c.i32c(0); c.localGet(L_AF); c.f64Store(regOff)
  // store → temps[dst]
  c.i32c(0); c.localGet(L_BF); c.f64Store(ctx.layout.tempsOffset + instr.dst * 8)
}

function emitTriggerParam(c: Code, instr: NInstr, ctx: EmitCtx): void {
  const pOp = instr.args[0]!
  if (pOp.kind !== 'param') throw new Error('TriggerParam: arg must be param')
  const pIdx = ctx.paramIndex.get(pOp.ptr) ?? 0
  c.i32c(0); c.f64Load(ctx.layout.paramFrameOffset + pIdx * 8)
  c.localSet(L_AF)
  c.i32c(0); c.localGet(L_AF); c.f64Store(ctx.layout.tempsOffset + instr.dst * 8)
}

function emitPack(c: Code, instr: NInstr, ctx: EmitCtx): void {
  const arrOff = ctx.layout.arrayOffsets[instr.dst]!
  for (let i = 0; i < instr.args.length; i++) {
    pushAs(c, instr.args[i]!, 'float', ctx)
    c.localSet(L_AF)
    c.i32c(0); c.localGet(L_AF); c.f64Store(arrOff + i * 8)
  }
}

function emitIndex(c: Code, instr: NInstr, ctx: EmitCtx): void {
  const arrOp = instr.args[0]!
  if (arrOp.kind !== 'array_reg') throw new Error('Index: arg0 must be array_reg')
  const arrOff = ctx.layout.arrayOffsets[arrOp.slot]!
  const arrSize = ctx.layout.arraySizes[arrOp.slot]!

  pushAs(c, instr.args[1]!, 'int', ctx)
  c.localSet(L_AI)

  // in_range = (ai >= 0) && (ai < arrSize)
  c.localGet(L_AI); c.i64c(0); c.u8(OP.I64_GE_S)
  c.localGet(L_AI); c.i64c(BigInt(arrSize)); c.u8(OP.I64_LT_S)
  c.u8(OP.I32_AND)
  c.if_()
  // arrays[slot][ai]
  c.localGet(L_AI); c.i64c(8); c.u8(OP.I64_MUL); c.u8(OP.I32_WRAP_I64)
  c.i32c(arrOff); c.u8(OP.I32_ADD)
  c.f64Load(0)
  c.localSet(L_AF)
  c.else_()
  c.f64c(0); c.localSet(L_AF)
  c.end()
  c.i32c(0); c.localGet(L_AF); c.f64Store(ctx.layout.tempsOffset + instr.dst * 8)
}

function emitSetElement(c: Code, instr: NInstr, ctx: EmitCtx): void {
  const arrOp = instr.args[0]!
  if (arrOp.kind !== 'array_reg') throw new Error('SetElement: arg0 must be array_reg')
  const arrOff = ctx.layout.arrayOffsets[arrOp.slot]!
  const arrSize = ctx.layout.arraySizes[arrOp.slot]!

  pushAs(c, instr.args[1]!, 'int', ctx); c.localSet(L_AI)
  pushAs(c, instr.args[2]!, 'float', ctx); c.localSet(L_AF)

  c.localGet(L_AI); c.i64c(0); c.u8(OP.I64_GE_S)
  c.localGet(L_AI); c.i64c(BigInt(arrSize)); c.u8(OP.I64_LT_S)
  c.u8(OP.I32_AND)
  c.if_()
  c.localGet(L_AI); c.i64c(8); c.u8(OP.I64_MUL); c.u8(OP.I32_WRAP_I64)
  c.i32c(arrOff); c.u8(OP.I32_ADD)
  c.localGet(L_AF); c.f64Store(0)
  c.end()
}

function emitElementwise(c: Code, instr: NInstr, ctx: EmitCtx): void {
  const dstOff = ctx.layout.arrayOffsets[instr.dst]!
  const n = instr.loop_count
  const nargs = instr.args.length

  c.i64c(0); c.localSet(L_EWI)
  c.block()
  c.loop()
  c.localGet(L_EWI); c.i64c(BigInt(n)); c.u8(OP.I64_GE_S); c.brIf(1)

  // Push each arg as f64 (elementwise ops always operate on f64 elements)
  for (let i = 0; i < nargs; i++) {
    const stride = instr.strides[i] ?? 0
    const arg = instr.args[i]!
    if (stride === 1) {
      if (arg.kind !== 'array_reg') throw new Error('elementwise: stride=1 needs array_reg')
      const srcOff = ctx.layout.arrayOffsets[arg.slot]!
      c.localGet(L_EWI); c.i64c(8); c.u8(OP.I64_MUL); c.u8(OP.I32_WRAP_I64)
      c.i32c(srcOff); c.u8(OP.I32_ADD)
      c.f64Load(0)
    } else {
      pushAs(c, arg, 'float', ctx)
    }
  }
  // Run the op with all args as float
  emitInlineFloatOp(c, instr.tag)
  // Store to dst[ewi]
  c.localSet(L_AF)
  c.localGet(L_EWI); c.i64c(8); c.u8(OP.I64_MUL); c.u8(OP.I32_WRAP_I64)
  c.i32c(dstOff); c.u8(OP.I32_ADD)
  c.localGet(L_AF); c.f64Store(0)

  c.localGet(L_EWI); c.i64c(1); c.u8(OP.I64_ADD); c.localSet(L_EWI)
  c.br(0)
  c.end()
  c.end()
}

// For elementwise use only — args have been pushed as f64, result is f64.
function emitInlineFloatOp(c: Code, tag: string): void {
  switch (tag) {
    case 'Add': c.u8(OP.F64_ADD); return
    case 'Sub': c.u8(OP.F64_SUB); return
    case 'Mul': c.u8(OP.F64_MUL); return
    case 'Div': c.u8(OP.F64_DIV); return
    case 'Neg': c.u8(OP.F64_NEG); return
    case 'Abs': c.u8(OP.F64_ABS); return
    case 'Sqrt': c.u8(OP.F64_SQRT); return
    case 'Floor': c.u8(OP.F64_FLOOR); return
    case 'Ceil': c.u8(OP.F64_CEIL); return
    case 'Round': c.u8(OP.F64_NEAREST); return
    default:
      throw new Error(`emit_wasm: elementwise op ${tag} not supported`)
  }
}

// ─────────────────────────────────────────────────────────────
// Scalar op emission — materializes args into locals, then emits op
// ─────────────────────────────────────────────────────────────

function emitScalar(c: Code, instr: NInstr, ctx: EmitCtx): void {
  const tag = instr.tag
  const rt = instr.result_type

  // ── Select (ternary) ──
  if (tag === 'Select') {
    // cond truthy, then/else in result type
    pushAs(c, instr.args[0]!, 'bool', ctx)
    c.localSet(L_TI32)
    pushAs(c, instr.args[1]!, rt, ctx)
    pushAs(c, instr.args[2]!, rt, ctx)
    c.localGet(L_TI32)
    c.select()
    storeTempAt(c, instr.dst, rt, ctx)
    return
  }

  // ── Clamp ──
  if (tag === 'Clamp') {
    pushAs(c, instr.args[0]!, rt, ctx)
    if (rt === 'int') c.localSet(L_AI); else c.localSet(L_AF)
    pushAs(c, instr.args[1]!, rt, ctx)
    if (rt === 'int') c.localSet(L_BI); else c.localSet(L_BF)
    pushAs(c, instr.args[2]!, rt, ctx)
    if (rt === 'int') c.localSet(L_CI); else c.localSet(L_CF)
    if (rt === 'int') {
      // max(a, lo) = a > lo ? a : lo
      c.localGet(L_AI); c.localGet(L_BI)
      c.localGet(L_AI); c.localGet(L_BI); c.u8(OP.I64_GT_S)
      c.select()
      c.localTee(L_AI)
      c.localGet(L_CI)
      c.localGet(L_AI); c.localGet(L_CI); c.u8(OP.I64_LT_S)
      c.select()
    } else {
      c.localGet(L_AF); c.localGet(L_BF)
      c.localGet(L_AF); c.localGet(L_BF); c.u8(OP.F64_GT)
      c.select()
      c.localTee(L_AF)
      c.localGet(L_CF)
      c.localGet(L_AF); c.localGet(L_CF); c.u8(OP.F64_LT)
      c.select()
    }
    storeTempAt(c, instr.dst, rt, ctx)
    return
  }

  // ── Comparison (result bool; op type decided by arg types) ──
  if (COMPARISON_OPS.has(tag)) {
    const aT = operandScalarType(instr.args[0]!), bT = operandScalarType(instr.args[1]!)
    const useInt = (aT === 'int' || bT === 'int')
    const cmpT: ScalarType = useInt ? 'int' : 'float'
    pushAs(c, instr.args[0]!, cmpT, ctx)
    pushAs(c, instr.args[1]!, cmpT, ctx)
    if (useInt) {
      switch (tag) {
        case 'Less':      c.u8(OP.I64_LT_S); break
        case 'LessEq':    c.u8(OP.I64_LE_S); break
        case 'Greater':   c.u8(OP.I64_GT_S); break
        case 'GreaterEq': c.u8(OP.I64_GE_S); break
        case 'Equal':     c.u8(OP.I64_EQ);   break
        case 'NotEqual':  c.u8(OP.I64_NE);   break
      }
    } else {
      switch (tag) {
        case 'Less':      c.u8(OP.F64_LT); break
        case 'LessEq':    c.u8(OP.F64_LE); break
        case 'Greater':   c.u8(OP.F64_GT); break
        case 'GreaterEq': c.u8(OP.F64_GE); break
        case 'Equal':     c.u8(OP.F64_EQ); break
        case 'NotEqual':  c.u8(OP.F64_NE); break
      }
    }
    storeTempAt(c, instr.dst, 'bool', ctx)
    return
  }

  // ── Logical And/Or (bool) ──
  if (tag === 'And' || tag === 'Or') {
    pushAs(c, instr.args[0]!, 'bool', ctx)
    pushAs(c, instr.args[1]!, 'bool', ctx)
    c.u8(tag === 'And' ? OP.I32_AND : OP.I32_OR)
    storeTempAt(c, instr.dst, 'bool', ctx)
    return
  }

  // ── Not ──
  if (tag === 'Not') {
    pushAs(c, instr.args[0]!, 'bool', ctx)
    c.u8(OP.I32_EQZ)
    storeTempAt(c, instr.dst, 'bool', ctx)
    return
  }

  // ── Bitwise ops (int, i64) ──
  if (BITWISE_OPS.has(tag)) {
    if (tag === 'BitNot') {
      pushAs(c, instr.args[0]!, 'int', ctx)
      c.i64c(-1); c.u8(OP.I64_XOR)
    } else {
      pushAs(c, instr.args[0]!, 'int', ctx)
      pushAs(c, instr.args[1]!, 'int', ctx)
      switch (tag) {
        case 'BitAnd': c.u8(OP.I64_AND); break
        case 'BitOr':  c.u8(OP.I64_OR);  break
        case 'BitXor': c.u8(OP.I64_XOR); break
        case 'LShift': c.u8(OP.I64_SHL); break
        case 'RShift': c.u8(OP.I64_SHR_S); break
      }
    }
    storeTempAt(c, instr.dst, 'int', ctx)
    return
  }

  // ── Casts ──
  if (tag === 'ToInt' || tag === 'ToBool' || tag === 'ToFloat') {
    const target: ScalarType = tag === 'ToInt' ? 'int' : tag === 'ToBool' ? 'bool' : 'float'
    pushAs(c, instr.args[0]!, target, ctx)
    storeTempAt(c, instr.dst, target, ctx)
    return
  }

  // ── Float-only unary ops ──
  if (FLOAT_ONLY_OPS.has(tag)) {
    if (tag === 'Ldexp') {
      pushAs(c, instr.args[0]!, 'float', ctx) // x
      pushAs(c, instr.args[1]!, 'float', ctx) // n_float
      c.i64TruncSat()
      c.i64c(1023); c.u8(OP.I64_ADD)
      c.i64c(52); c.u8(OP.I64_SHL)
      c.u8(OP.F64_REINTERPRET_I64)
      c.u8(OP.F64_MUL)
    } else if (tag === 'FloatExponent') {
      pushAs(c, instr.args[0]!, 'float', ctx)
      c.u8(OP.I64_REINTERPRET_F64)
      c.i64c(52); c.u8(OP.I64_SHR_S)
      c.i64c(1023); c.u8(OP.I64_SUB)
      c.u8(OP.F64_CONVERT_I64_S)
    } else {
      pushAs(c, instr.args[0]!, 'float', ctx)
      switch (tag) {
        case 'Sqrt':  c.u8(OP.F64_SQRT); break
        case 'Floor': c.u8(OP.F64_FLOOR); break
        case 'Ceil':  c.u8(OP.F64_CEIL); break
        case 'Round': c.u8(OP.F64_NEAREST); break
      }
    }
    storeTempAt(c, instr.dst, 'float', ctx)
    return
  }

  // ── Arithmetic: Add/Sub/Mul/Div/Mod/FloorDiv/Neg/Abs ──
  // Native OrcJitEngine falls through non-int result_types to float and
  // coerces back. We do the same: computation happens in `opT` (int or
  // float), then we coerce from opT to rt before storing.
  const opT: ScalarType = rt === 'int' ? 'int' : 'float'

  if (tag === 'Neg') {
    pushAs(c, instr.args[0]!, opT, ctx)
    if (opT === 'int') { c.i64c(-1); c.u8(OP.I64_MUL) }
    else               { c.u8(OP.F64_NEG) }
    coerce(c, opT, rt)
    storeTempAt(c, instr.dst, rt, ctx)
    return
  }
  if (tag === 'Abs') {
    pushAs(c, instr.args[0]!, opT, ctx)
    if (opT === 'int') {
      c.localTee(L_AI)
      c.i64c(0); c.u8(OP.I64_LT_S)
      c.if_()
      c.i64c(0); c.localGet(L_AI); c.u8(OP.I64_SUB); c.localSet(L_AI)
      c.end()
      c.localGet(L_AI)
    } else {
      c.u8(OP.F64_ABS)
    }
    coerce(c, opT, rt)
    storeTempAt(c, instr.dst, rt, ctx)
    return
  }
  // Binary arithmetic: use locals to safely guard Div/Mod against zero.
  {
    pushAs(c, instr.args[0]!, opT, ctx)
    if (opT === 'int') c.localSet(L_AI); else c.localSet(L_AF)
    pushAs(c, instr.args[1]!, opT, ctx)
    if (opT === 'int') c.localSet(L_BI); else c.localSet(L_BF)

    switch (tag) {
      case 'Add':
        if (opT === 'int') { c.localGet(L_AI); c.localGet(L_BI); c.u8(OP.I64_ADD) }
        else               { c.localGet(L_AF); c.localGet(L_BF); c.u8(OP.F64_ADD) }
        break
      case 'Sub':
        if (opT === 'int') { c.localGet(L_AI); c.localGet(L_BI); c.u8(OP.I64_SUB) }
        else               { c.localGet(L_AF); c.localGet(L_BF); c.u8(OP.F64_SUB) }
        break
      case 'Mul':
        if (opT === 'int') { c.localGet(L_AI); c.localGet(L_BI); c.u8(OP.I64_MUL) }
        else               { c.localGet(L_AF); c.localGet(L_BF); c.u8(OP.F64_MUL) }
        break
      case 'Div':
        if (opT === 'int') {
          // (b==0) ? 0 : a/b  — WASM i64.div_s traps on b==0, so guard via if/else.
          c.localGet(L_BI); c.u8(OP.I64_EQZ)
          c.if_()
          c.i64c(0); c.localSet(L_CI)
          c.else_()
          c.localGet(L_AI); c.localGet(L_BI); c.u8(OP.I64_DIV_S); c.localSet(L_CI)
          c.end()
          c.localGet(L_CI)
        } else {
          // float div: b==0 → 0, else a/b (f64.div doesn't trap but produces NaN/Inf; match native guard)
          c.localGet(L_BF); c.f64c(0); c.u8(OP.F64_EQ)
          c.if_()
          c.f64c(0); c.localSet(L_CF)
          c.else_()
          c.localGet(L_AF); c.localGet(L_BF); c.u8(OP.F64_DIV); c.localSet(L_CF)
          c.end()
          c.localGet(L_CF)
        }
        break
      case 'Mod':
        if (opT === 'int') {
          c.localGet(L_BI); c.u8(OP.I64_EQZ)
          c.if_()
          c.i64c(0); c.localSet(L_CI)
          c.else_()
          c.localGet(L_AI); c.localGet(L_BI); c.u8(OP.I64_REM_S); c.localSet(L_CI)
          c.end()
          c.localGet(L_CI)
        } else {
          // fmod: b==0 → 0, else a - floor(a/b)*b
          c.localGet(L_BF); c.f64c(0); c.u8(OP.F64_EQ)
          c.if_()
          c.f64c(0); c.localSet(L_CF)
          c.else_()
          c.localGet(L_AF)                                                    // a
          c.localGet(L_AF); c.localGet(L_BF); c.u8(OP.F64_DIV); c.u8(OP.F64_FLOOR) // floor(a/b)
          c.localGet(L_BF); c.u8(OP.F64_MUL)                                  // floor(a/b) * b
          c.u8(OP.F64_SUB)                                                    // a - floor(a/b)*b
          c.localSet(L_CF)
          c.end()
          c.localGet(L_CF)
        }
        break
      case 'FloorDiv':
        if (opT === 'int') {
          c.localGet(L_BI); c.u8(OP.I64_EQZ)
          c.if_()
          c.i64c(0); c.localSet(L_CI)
          c.else_()
          c.localGet(L_AI); c.localGet(L_BI); c.u8(OP.I64_DIV_S); c.localSet(L_CI)
          c.end()
          c.localGet(L_CI)
        } else {
          c.localGet(L_BF); c.f64c(0); c.u8(OP.F64_EQ)
          c.if_()
          c.f64c(0); c.localSet(L_CF)
          c.else_()
          c.localGet(L_AF); c.localGet(L_BF); c.u8(OP.F64_DIV); c.u8(OP.F64_FLOOR); c.localSet(L_CF)
          c.end()
          c.localGet(L_CF)
        }
        break
      default:
        throw new Error(`emit_wasm: scalar op ${tag} not supported`)
    }
    coerce(c, opT, rt)
    storeTempAt(c, instr.dst, rt, ctx)
  }
}

function operandScalarType(op: NOperand): ScalarType {
  if (op.kind === 'array_reg') return 'float'
  if (op.kind === 'rate') return 'float'
  if (op.kind === 'tick') return 'int'
  return op.scalar_type
}

const COMPARISON_OPS = new Set(['Less', 'LessEq', 'Greater', 'GreaterEq', 'Equal', 'NotEqual'])
const BITWISE_OPS = new Set(['BitAnd', 'BitOr', 'BitXor', 'LShift', 'RShift', 'BitNot'])
const FLOAT_ONLY_OPS = new Set(['Sqrt', 'Floor', 'Ceil', 'Round', 'Ldexp', 'FloatExponent'])
