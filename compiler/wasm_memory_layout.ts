/**
 * wasm_memory_layout.ts — byte layout of the WASM kernel's linear memory.
 *
 * Shared between `emit_wasm.ts` (codegen) and `web/worklet/runtime.ts` so
 * offsets stay in sync. All regions are aligned to 8 bytes and stored
 * contiguously starting at offset 0.
 *
 * Type encoding mirrors the native engine:
 *   - f64 slots store doubles directly.
 *   - i64 slots store signed ints, or f64s reinterpreted via bitcast,
 *     or bool zero-extended. The op's result_type tells codegen which
 *     WASM type to load/store.
 */

import type { FlatProgram } from './emit_numeric.js'

export type WasmLayout = {
  /** bytes */
  inputsOffset:      number
  registersOffset:   number
  tempsOffset:       number
  arraysOffset:      number
  arrayOffsets:      number[]     // per array slot: absolute byte offset
  paramTableOffset:  number       // f64[paramCount]
  paramFrameOffset:  number       // f64[paramCount] — trigger snapshot per block
  outputOffset:      number       // f64[maxBlockSize]
  totalBytes:        number
  /** Number of 64 KiB WASM pages needed */
  pageCount:         number
  /** Counts — for codegen loop bounds & state-transfer diffing */
  inputCount:        number
  registerCount:     number
  arraySlotCount:    number
  arraySizes:        number[]
  paramCount:        number
  maxBlockSize:      number
}

export const WASM_PAGE_SIZE = 65536

/** Round up to 8-byte alignment. */
function align8(n: number): number {
  return (n + 7) & ~7
}

/** Build a layout for the given plan + ordered param list. */
export function computeLayout(args: {
  plan: FlatProgram
  inputCount: number
  /** Ordered list of param "ptr" strings (decimal index for web, native ptr for desktop). */
  paramPtrs: string[]
  /** Max audio block size the kernel will be asked to render in one call. */
  maxBlockSize: number
}): WasmLayout {
  const { plan, inputCount, paramPtrs, maxBlockSize } = args
  const f64 = 8
  const i64 = 8

  let cursor = 0
  const inputsOffset = cursor
  cursor = align8(cursor + inputCount * f64)

  const registersOffset = cursor
  cursor = align8(cursor + plan.register_count * i64)

  const tempsOffset = cursor
  cursor = align8(cursor + plan.register_count * i64)

  const arraysOffset = cursor
  const arrayOffsets: number[] = []
  for (const size of plan.array_slot_sizes) {
    arrayOffsets.push(cursor)
    cursor = align8(cursor + size * f64)
  }

  const paramTableOffset = cursor
  cursor = align8(cursor + paramPtrs.length * f64)

  const paramFrameOffset = cursor
  cursor = align8(cursor + paramPtrs.length * f64)

  const outputOffset = cursor
  cursor = align8(cursor + maxBlockSize * f64)

  const totalBytes = cursor
  const pageCount = Math.max(1, Math.ceil(totalBytes / WASM_PAGE_SIZE))

  return {
    inputsOffset,
    registersOffset,
    tempsOffset,
    arraysOffset,
    arrayOffsets,
    paramTableOffset,
    paramFrameOffset,
    outputOffset,
    totalBytes,
    pageCount,
    inputCount,
    registerCount: plan.register_count,
    arraySlotCount: plan.array_slot_count,
    arraySizes: plan.array_slot_sizes.slice(),
    paramCount: paramPtrs.length,
    maxBlockSize,
  }
}

/**
 * Collect unique param pointers in first-appearance order across the plan's
 * instructions. Matches the C++ engine's canonical param table ordering
 * (OrcJitEngine.cpp:304-315).
 */
export function collectParamPtrs(plan: FlatProgram): string[] {
  const seen = new Map<string, number>()
  for (const instr of plan.instructions) {
    for (const arg of instr.args) {
      if (arg.kind === 'param' && !seen.has(arg.ptr)) {
        seen.set(arg.ptr, seen.size)
      }
    }
  }
  return [...seen.keys()]
}
