/**
 * wasm_vs_jit.test.ts — compare WASM output against the native JIT.
 *
 * Runs real stdlib-based patches through both backends and asserts the output
 * buffers match. Post-Phase-3 the WASM is the *same LLVM IR* the JIT runs, just
 * lowered to wasm32 by the same LLVM — so agreement is structural; this gate's
 * job is to catch any FP-target divergence (fp-contract/fast-math).
 *
 * Both sides come off the SAME source compiled by the Lean engine:
 *   - WASM side: `diffcli compile-wasm` (in-process LLVM+lld) → WasmKernel.
 *   - Native side: `diffcli render-bytes` (the engine's ORC JIT).
 */

import { describe, test, expect } from 'bun:test'
import { spawnSync } from 'bun'
import { writeFileSync, readFileSync } from 'fs'
import { resolve, dirname } from 'path'
import { fileURLToPath } from 'url'
import { WasmKernel, type KernelManifest } from '../../web/runtime/index'

const __dirname = dirname(fileURLToPath(import.meta.url))
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

const PROG_TMP = '/tmp/wvj-program.json'

/** Compile a `ProgramFile` to a `tropical_plan_5` wire plan JSON string via
 *  the Lean `diffcli compile` front door (native oracle + manifest source). */
function compileViaLean(program: ProgramFile): string {
  writeFileSync(PROG_TMP, JSON.stringify(program))
  const r = spawnSync([diffcli, 'compile', PROG_TMP, '--mode=fused'], { cwd: repoRoot, env: cliEnv })
  if (r.exitCode !== 0) throw new Error(`diffcli compile failed (exit ${r.exitCode}): ${r.stderr?.toString()}`)
  return r.stdout.toString().trim()
}

/** Trim a plan_5 JSON to the KernelManifest the runtime consumes. */
function manifestFromPlan(planJson: string): KernelManifest {
  const p = JSON.parse(planJson)
  return {
    sampleRate:     p.config?.sampleRate ?? 44100,
    registerCount:  p.register_count ?? 0,
    registerTypes:  p.register_types ?? [],
    stateInit:      p.state_init ?? [],
    arraySlotSizes: p.array_slot_sizes ?? [],
    slotCount:      p.slot_count ?? (p.slot_defaults?.length ?? 0),
    slotDefaults:   p.slot_defaults ?? [],
  }
}

/** Native side: render the wire plan through the engine's JIT. */
function runNative(wirePlan: string, samples: number): Float64Array {
  const tmp = '/tmp/wvj-native.plan.json'
  writeFileSync(tmp, wirePlan)
  const r = spawnSync([diffcli, 'render-bytes', tmp, '--frames', '1', '--buffer', String(samples)], { cwd: repoRoot, env: cliEnv })
  if (r.exitCode !== 0) throw new Error(`diffcli render-bytes failed (exit ${r.exitCode}): ${r.stderr?.toString()}`)
  const b = r.stdout
  return new Float64Array(b.buffer.slice(b.byteOffset, b.byteOffset + samples * 8))
}

/** WASM side: lower the same source to wasm32 in-process and drive it through
 *  the runtime package (the artifacts the browser ships). */
async function runWasm(program: ProgramFile, plan: string, samples: number): Promise<Float64Array> {
  writeFileSync(PROG_TMP, JSON.stringify(program))
  const out = '/tmp/wvj-program.wasm'
  const r = spawnSync([diffcli, 'compile-wasm', PROG_TMP, `--out=${out}`], { cwd: repoRoot, env: cliEnv })
  if (r.exitCode !== 0) throw new Error(`diffcli compile-wasm failed (exit ${r.exitCode}): ${r.stderr?.toString()}`)
  const k = await WasmKernel.instantiate(new Uint8Array(readFileSync(out)), manifestFromPlan(plan), samples)
  return k.render(samples).slice()
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

// Same IR, two LLVM targets → bit-exact in practice; keep a tolerance for safety.
const TOL = 1e-12

describe('wasm vs native JIT', () => {
  test('SinOsc 440 Hz — first 64 samples', async () => {
    const prog = sinOscProgram(440)
    const wire = compileViaLean(prog)
    const N = 64
    const nat = runNative(wire, N)
    const wasm = await runWasm(prog, wire, N)
    for (let i = 0; i < N; i++) expect(Math.abs(wasm[i]! - nat[i]!)).toBeLessThan(TOL)
  })

  test('SinOsc 880 Hz — 128 samples', async () => {
    const prog = sinOscProgram(880)
    const wire = compileViaLean(prog)
    const N = 128
    const nat = runNative(wire, N)
    const wasm = await runWasm(prog, wire, N)
    for (let i = 0; i < N; i++) expect(Math.abs(wasm[i]! - nat[i]!)).toBeLessThan(TOL)
  })

  test('SinOsc → OnePole(1000 Hz) — 256 samples', async () => {
    const prog = onePoleProgram(1000)
    const wire = compileViaLean(prog)
    const N = 256
    const nat = runNative(wire, N)
    const wasm = await runWasm(prog, wire, N)
    for (let i = 0; i < N; i++) expect(Math.abs(wasm[i]! - nat[i]!)).toBeLessThan(TOL)
  })

  // Array shape: wholesale array-register writeback via zipWith. Keeps an
  // array op (Pack / Index / elementwise) in the wasm equivalence loop.
  test('array-zipWith wholesale writeback — WASM matches JIT', async () => {
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
    const wasm = await runWasm(program, wire, N)
    for (let i = 0; i < N; i++) expect(Math.abs(wasm[i]! - nat[i]!)).toBeLessThan(TOL)
  })
})
