/**
 * wasm_vs_jit_equiv.test.ts — compare WASM output against the native JIT.
 *
 * Runs real stdlib-based patches through both backends and asserts the
 * output buffers match within a tight tolerance. This is the forcing
 * function that keeps emit_wasm.ts faithful to OrcJitEngine.cpp.
 *
 * Requires `libtropical.dylib` available to koffi (either built in-tree
 * or via the symlinked ../build directory when running from a worktree).
 */

import { describe, test, expect } from 'bun:test'
import { makeSession, loadJSON, type ProgramFile } from './session'
import { loadStdlib } from './program'
import { flattenSession, type FlatPlan } from './flatten'
import { emitWasm } from './emit_wasm'

// Load state_init values into the WASM module's register region.
function initWasmState(memory: WebAssembly.Memory, regOffset: number, stateInit: (number | boolean)[], regTypes: string[]): void {
  const dv = new DataView(memory.buffer)
  for (let i = 0; i < stateInit.length; i++) {
    const v = stateInit[i]
    const t = regTypes[i] ?? 'float'
    const off = regOffset + i * 8
    if (typeof v === 'boolean') {
      dv.setBigInt64(off, v ? 1n : 0n, true)
    } else if (t === 'int') {
      dv.setBigInt64(off, BigInt(Math.trunc(v as number)), true)
    } else if (t === 'bool') {
      dv.setBigInt64(off, (v as number) !== 0 ? 1n : 0n, true)
    } else {
      dv.setFloat64(off, v as number, true)
    }
  }
}

async function runWasm(plan: FlatPlan, samples: number): Promise<Float64Array> {
  const { bytes, layout } = emitWasm(plan, { maxBlockSize: samples })
  const mod = await WebAssembly.compile(bytes)
  const instance = await WebAssembly.instantiate(mod, {})
  const memory = instance.exports.memory as WebAssembly.Memory
  const process_ = instance.exports.process as (blen: number, sidx: bigint) => void

  // Initialize state_init (ignoring array-slot entries which are arrays themselves)
  const scalarStateInit = plan.state_init.map((v) => (Array.isArray(v) ? 0 : v)) as (number | boolean)[]
  initWasmState(memory, layout.registersOffset, scalarStateInit, plan.register_types)

  process_(samples, 0n)
  return new Float64Array(memory.buffer, layout.outputOffset, samples).slice()
}

function runNative(plan: FlatPlan, samples: number): Float64Array {
  const session = makeSession(samples)
  try {
    session.runtime.loadPlan(JSON.stringify(plan))
    session.runtime.process()
    return new Float64Array(session.runtime.outputBuffer)
  } finally {
    session.runtime.dispose()
  }
}

function makeSinOscPlan(freqHz: number): FlatPlan {
  const session = makeSession(64)
  try {
    loadStdlib(session.typeRegistry)
    const prog: ProgramFile = {
      schema: 'tropical_program_2',
      name: 'eq_sinosc',
      body: { op: 'block', decls: [
        { op: 'instance_decl', name: 'osc', program: 'SinOsc', inputs: { freq: freqHz } },
      ]},
      audio_outputs: [{ instance: 'osc', output: 'sine' }],
    }
    loadJSON(prog, session)
    return flattenSession(session)
  } finally {
    session.runtime.dispose()
  }
}

function makeOnePolePlan(cutoff: number): FlatPlan {
  const session = makeSession(64)
  try {
    loadStdlib(session.typeRegistry)
    const prog: ProgramFile = {
      schema: 'tropical_program_2',
      name: 'eq_onepole',
      body: { op: 'block', decls: [
        { op: 'instance_decl', name: 'osc', program: 'SinOsc', inputs: { freq: 220 } },
        { op: 'instance_decl', name: 'lp', program: 'OnePole', inputs: {
          signal: { op: 'ref', instance: 'osc', output: 'sine' },
          cutoff,
        }},
      ]},
      audio_outputs: [{ instance: 'lp', output: 'out' }],
    }
    loadJSON(prog, session)
    return flattenSession(session)
  } finally {
    session.runtime.dispose()
  }
}

const TOL = 1e-9 // generous; expect bit-near-identical since both use IEEE f64 arithmetic

describe('wasm vs native JIT', () => {
  test('SinOsc 440 Hz — first 64 samples', async () => {
    const plan = makeSinOscPlan(440)
    const N = 64
    const nat = runNative(plan, N)
    const wasm = await runWasm(plan, N)
    for (let i = 0; i < N; i++) {
      expect(Math.abs(wasm[i]! - nat[i]!)).toBeLessThan(TOL)
    }
  })

  test('SinOsc 880 Hz — 128 samples', async () => {
    const plan = makeSinOscPlan(880)
    const N = 128
    const nat = runNative(plan, N)
    const wasm = await runWasm(plan, N)
    for (let i = 0; i < N; i++) {
      expect(Math.abs(wasm[i]! - nat[i]!)).toBeLessThan(TOL)
    }
  })

  test('SinOsc → OnePole(1000 Hz) — 256 samples', async () => {
    const plan = makeOnePolePlan(1000)
    const N = 256
    const nat = runNative(plan, N)
    const wasm = await runWasm(plan, N)
    for (let i = 0; i < N; i++) {
      expect(Math.abs(wasm[i]! - nat[i]!)).toBeLessThan(TOL)
    }
  })
})
