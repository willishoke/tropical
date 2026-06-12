/**
 * wasm_vs_jit.test.ts — compare WASM output against the native JIT.
 *
 * Runs real stdlib-based patches through both backends and asserts the
 * output buffers match within a tight tolerance. This is the forcing
 * function that keeps web/wasm/emit_wasm.ts faithful to OrcJitEngine.cpp.
 *
 * Both sides come off the SAME `tropical_plan_5` wire plan produced by
 * the Lean compiler (`diffcli compile`):
 *   - WASM side: emit_wasm (relocated to web/wasm/) on the parsed plan.
 *   - Native side: `diffcli render-bytes` (the Lean engine's JIT).
 * No TS-compiler import and no native FFI — fully decoupled post-Phase-8.
 */

import { describe, test, expect } from 'bun:test'
import { spawnSync } from 'bun'
import { writeFileSync } from 'fs'
import { resolve, dirname } from 'path'
import { fileURLToPath } from 'url'
import { parseWirePlan, type FlatPlan } from '../../web/wasm/flat_plan'
import { emitWasm } from '../../web/wasm/emit_wasm'

const __dirname = dirname(fileURLToPath(import.meta.url))
// tests/web/ is two levels under the repo root, like tests/equiv/ was.
const repoRoot = resolve(__dirname, '../..')
const diffcli = resolve(repoRoot, 'lean/.lake/build/bin/diffcli')
// diffcli reads stdlib/parsed/manifest.json relative to its cwd, so run
// from the repo root and put elan on PATH for the Lean toolchain.
const cliEnv = { ...process.env, PATH: `${process.env.HOME}/.elan/bin:${process.env.PATH}` }

/** A `tropical_program_2` patch authored inline below. */
interface ProgramFile {
  schema: 'tropical_program_2'
  name: string
  body: { op: 'block'; decls: unknown[] }
  audio_outputs: { instance: string; output: string }[]
}

/** Compile a `ProgramFile` to a `tropical_plan_5` WIRE plan JSON string
 *  via the Lean `diffcli compile` front door. */
function compileViaLean(program: ProgramFile): string {
  const tmp = '/tmp/wvj-program.json'
  writeFileSync(tmp, JSON.stringify(program))
  const r = spawnSync([diffcli, 'compile', tmp, '--mode=fused'], { cwd: repoRoot, env: cliEnv })
  if (r.exitCode !== 0) {
    throw new Error(`diffcli compile failed (exit ${r.exitCode}): ${r.stderr?.toString()}`)
  }
  return r.stdout.toString().trim()
}

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

/** Native side: render the wire plan through the Lean engine's JIT via
 *  `diffcli render-bytes`. Takes the wire-plan JSON STRING and returns
 *  the N rendered f64 samples. */
function runNative(wirePlan: string, samples: number): Float64Array {
  const tmp = '/tmp/wvj-native.plan.json'
  writeFileSync(tmp, wirePlan)
  const r = spawnSync([diffcli, 'render-bytes', tmp, '--frames', '1', '--buffer', String(samples)], { cwd: repoRoot, env: cliEnv })
  if (r.exitCode !== 0) {
    throw new Error(`diffcli render-bytes failed (exit ${r.exitCode}): ${r.stderr?.toString()}`)
  }
  const b = r.stdout
  return new Float64Array(b.buffer.slice(b.byteOffset, b.byteOffset + samples * 8))
}

function sinOscProgram(freqHz: number): ProgramFile {
  return {
    schema: 'tropical_program_2',
    name: 'eq_sinosc',
    body: { op: 'block', decls: [
      { op: 'instanceDecl', name: 'osc', program: 'SinOsc', inputs: { freq: freqHz } },
    ]},
    audio_outputs: [{ instance: 'osc', output: 'sine' }],
  }
}

function onePoleProgram(cutoff: number): ProgramFile {
  return {
    schema: 'tropical_program_2',
    name: 'eq_onepole',
    body: { op: 'block', decls: [
      { op: 'instanceDecl', name: 'osc', program: 'SinOsc', inputs: { freq: 220 } },
      { op: 'instanceDecl', name: 'lp', program: 'OnePole', inputs: {
        input: { op: 'ref', instance: 'osc', output: 'sine' },
        g: cutoff,
      }},
    ]},
    audio_outputs: [{ instance: 'lp', output: 'out' }],
  }
}

const TOL = 1e-9 // generous; expect bit-near-identical since both use IEEE f64 arithmetic

describe('wasm vs native JIT', () => {
  test('SinOsc 440 Hz — first 64 samples', async () => {
    const wire = compileViaLean(sinOscProgram(440))
    const N = 64
    const nat = runNative(wire, N)
    const wasm = await runWasm(parseWirePlan(JSON.parse(wire)), N)
    for (let i = 0; i < N; i++) {
      expect(Math.abs(wasm[i]! - nat[i]!)).toBeLessThan(TOL)
    }
  })

  test('SinOsc 880 Hz — 128 samples', async () => {
    const wire = compileViaLean(sinOscProgram(880))
    const N = 128
    const nat = runNative(wire, N)
    const wasm = await runWasm(parseWirePlan(JSON.parse(wire)), N)
    for (let i = 0; i < N; i++) {
      expect(Math.abs(wasm[i]! - nat[i]!)).toBeLessThan(TOL)
    }
  })

  test('SinOsc → OnePole(1000 Hz) — 256 samples', async () => {
    const wire = compileViaLean(onePoleProgram(1000))
    const N = 256
    const nat = runNative(wire, N)
    const wasm = await runWasm(parseWirePlan(JSON.parse(wire)), N)
    for (let i = 0; i < N; i++) {
      expect(Math.abs(wasm[i]! - nat[i]!)).toBeLessThan(TOL)
    }
  })

  // Phase B cross-check: bring a wholesale-array-writeback shape into
  // the WASM equivalence loop so all three backends agree on at least
  // one Phase B shape. This is the original `array_reg_zipwith_writeback`
  // EDGE_FIXTURES program, re-expressed as a `programDecl` + `instanceDecl`
  // patch so the Lean `diffcli compile` front door can ingest it.
  test('Phase B array-zipWith wholesale writeback — WASM matches JIT', async () => {
    const program: ProgramFile = {
      schema: 'tropical_program_2',
      name: 'eq_arrayzip',
      body: { op: 'block', decls: [
        { op: 'programDecl', name: 'ArrayRegZipWithWriteback', program: {
          op: 'program',
          name: 'ArrayRegZipWithWriteback',
          ports: { inputs: [], outputs: ['out'] },
          body: { op: 'block',
            decls: [
              { op: 'regDecl', name: 'arr', init: [0, 0, 0, 0] },
            ],
            assigns: [
              { op: 'outputAssign', name: 'out',
                expr: { op: 'index', args: [{ op: 'reg', name: 'arr' }, 2] } },
              { op: 'nextUpdate', target: { kind: 'reg', name: 'arr' },
                expr: { op: 'zipWith',
                  a: [1, 2, 3, 4],
                  b: [10, 20, 30, 40],
                  x_var: 'x', y_var: 'y',
                  body: { op: 'add', args: [
                    { op: 'binding', name: 'x' },
                    { op: 'binding', name: 'y' },
                  ]},
                }},
            ],
          },
        }},
        { op: 'instanceDecl', name: 'inst', program: 'ArrayRegZipWithWriteback', inputs: {} },
      ]},
      audio_outputs: [{ instance: 'inst', output: 'out' }],
    }
    const wire = compileViaLean(program)
    const N = 64
    const nat = runNative(wire, N)
    const wasm = await runWasm(parseWirePlan(JSON.parse(wire)), N)
    for (let i = 0; i < N; i++) {
      expect(Math.abs(wasm[i]! - nat[i]!)).toBeLessThan(TOL)
    }
  })
})

// Root-program lowering (Option A): the whole session compiles to one
// synthetic root kernel whose `instance_functions[0]` has an empty body
// and the real instance bodies as nested `children`. This is the shape
// that forced emit_wasm to recurse like the C++ `emit_kernel_block`
// (preamble → per-child {pre_input, child} → body → writebacks). Same
// patches as above, asserting WASM still matches the native JIT on the
// nested plan.
describe('wasm vs native JIT (root-program lowering)', () => {
  test('SinOsc 440 Hz — root plan', async () => {
    const wire = compileViaLean(sinOscProgram(440))
    const N = 64
    const nat = runNative(wire, N)
    const wasm = await runWasm(parseWirePlan(JSON.parse(wire)), N)
    for (let i = 0; i < N; i++) {
      expect(Math.abs(wasm[i]! - nat[i]!)).toBeLessThan(TOL)
    }
  })

  test('SinOsc → OnePole(1000 Hz) — root plan (nested children)', async () => {
    const wire = compileViaLean(onePoleProgram(1000))
    const N = 256
    const nat = runNative(wire, N)
    const wasm = await runWasm(parseWirePlan(JSON.parse(wire)), N)
    for (let i = 0; i < N; i++) {
      expect(Math.abs(wasm[i]! - nat[i]!)).toBeLessThan(TOL)
    }
  })
})
