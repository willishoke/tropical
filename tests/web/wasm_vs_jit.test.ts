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

import { describe, test, expect, setDefaultTimeout } from 'bun:test'
import { spawnSync } from 'bun'
import { writeFileSync, readFileSync } from 'fs'
import { resolve, dirname } from 'path'
import { fileURLToPath } from 'url'
import { WasmKernel, type KernelManifest } from '../../web/runtime/index'
import { expectAudible } from './energy_floor'

// Each case spawns three sequential heavy subprocesses — `diffcli compile`,
// `render-bytes`, and `compile-wasm` (the last loads LLVM+lld in-process to
// lower wasm32). That runs ~5s locally and tips over bun's 5000ms default on a
// slower/cold CI runner. Give the file realistic headroom; it's an integration
// gate, not a unit test.
setDefaultTimeout(60_000)

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
 *  the Lean `diffcli compile` front door (native oracle + manifest source).
 *  `--fixtures` registers the builder-authored test programs (`OpZoo`) on top
 *  of the stdlib so patch-bay fixtures can instantiate them by name. */
function compileViaLean(program: ProgramFile): string {
  writeFileSync(PROG_TMP, JSON.stringify(program))
  const r = spawnSync([diffcli, 'compile', PROG_TMP, '--mode=fused', '--fixtures'], { cwd: repoRoot, env: cliEnv })
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
  const r = spawnSync([diffcli, 'compile-wasm', PROG_TMP, `--out=${out}`, '--fixtures'], { cwd: repoRoot, env: cliEnv })
  if (r.exitCode !== 0) throw new Error(`diffcli compile-wasm failed (exit ${r.exitCode}): ${r.stderr?.toString()}`)
  const k = await WasmKernel.instantiate(new Uint8Array(readFileSync(out)), manifestFromPlan(plan), samples)
  return k.render(samples).slice()
}

function sinOscProgram(freqHz: number): ProgramFile {
  return {
    schema: 'tropical_program_2',
    name: 'eq_sinosc',
    body: { op: 'block', decls: [
      { op: 'instanceDecl', name: 'osc', program: 'FixedSinOsc', inputs: { freq: freqHz } },
    ]},
    audio_outputs: [{ instance: 'osc', output: 'sine' }],
  }
}

// Two-instance ref-wired chain (osc → shaper) across both backends. CF-only has
// no IIR filter; SoftClip is the closed-form signal-shaper stand-in — the point
// here is cross-backend agreement on a multi-instance graph, not the DSP itself.
function softClipChainProgram(drive: number): ProgramFile {
  return {
    schema: 'tropical_program_2',
    name: 'eq_softclip',
    body: { op: 'block', decls: [
      { op: 'instanceDecl', name: 'osc', program: 'FixedSinOsc', inputs: { freq: 220 } },
      { op: 'instanceDecl', name: 'sc', program: 'SoftClip', inputs: {
        input: { op: 'ref', instance: 'osc', output: 'sine' },
        drive,
      }},
    ]},
    audio_outputs: [{ instance: 'sc', output: 'out' }],
  }
}

// A single self-contained program exercising the rare ops at *runtime* (seeded
// from sampleIndex so LLVM can't constant-fold them away): the full
// bitwise/shift cluster, both float↔int conversions, mod/floorDiv,
// clamp/select, comparisons, and the rounding intrinsics. Confirms every op
// lowers correctly on wasm32 (esp. fptosi → i64.trunc_sat, the one op with
// target-divergent lowering), not just the arithmetic the audio patches hit.
// The program body is authored in Lean (`EmitArrow.opZooSig`, registered by
// `--fixtures`); this file drives it BY NAME — the schema cannot carry program
// bodies over the wire.
function opZooProgram(): ProgramFile {
  return {
    schema: 'tropical_program_2',
    name: 'eq_opzoo',
    body: { op: 'block', decls: [
      { op: 'instanceDecl', name: 'inst', program: 'OpZoo', inputs: {} },
    ]},
    audio_outputs: [{ instance: 'inst', output: 'out' }],
  }
}

// Same IR, two LLVM targets → bit-exact in practice; keep a tolerance for safety.
const TOL = 1e-12

describe('wasm vs native JIT', () => {
  test('FixedSinOsc 440 Hz — first 64 samples', async () => {
    const prog = sinOscProgram(440)
    const wire = compileViaLean(prog)
    const N = 64
    const nat = runNative(wire, N)
    expectAudible('FixedSinOsc 440 Hz', nat)
    const wasm = await runWasm(prog, wire, N)
    for (let i = 0; i < N; i++) expect(Math.abs(wasm[i]! - nat[i]!)).toBeLessThan(TOL)
  })

  test('FixedSinOsc 880 Hz — 128 samples', async () => {
    const prog = sinOscProgram(880)
    const wire = compileViaLean(prog)
    const N = 128
    const nat = runNative(wire, N)
    expectAudible('FixedSinOsc 880 Hz', nat)
    const wasm = await runWasm(prog, wire, N)
    for (let i = 0; i < N; i++) expect(Math.abs(wasm[i]! - nat[i]!)).toBeLessThan(TOL)
  })

  test('FixedSinOsc → SoftClip(drive 4) — 256 samples', async () => {
    const prog = softClipChainProgram(4)
    const wire = compileViaLean(prog)
    const N = 256
    const nat = runNative(wire, N)
    expectAudible('FixedSinOsc → SoftClip(drive 4)', nat)
    const wasm = await runWasm(prog, wire, N)
    for (let i = 0; i < N; i++) expect(Math.abs(wasm[i]! - nat[i]!)).toBeLessThan(TOL)
  })

  // Program definitions over the wire are retired (elaborator retirement,
  // phase 5): a tropical_program_2 file carrying a programDecl — whatever its
  // body spells — is refused AT INGEST with the retirement message; the wire
  // is a patch bay (instances + wiring + params of registered types). This
  // pins that refusal cross-process. The raise-level combinator refusal
  // ('fold' is not a trunk construct) is pinned in-process by tropicaltest's
  // retired-front-door gate. Summing indexed families are authored as
  // Sig.bankSum in Lean, which this schema cannot spell.
  test('program definitions over the wire — refused at ingest', () => {
    const program: ProgramFile = {
      schema: 'tropical_program_2',
      name: 'eq_arraycf',
      body: { op: 'block', decls: [
        { op: 'programDecl', name: 'ArrayCF', program: {
          op: 'program', name: 'ArrayCF',
          ports: { inputs: [], outputs: ['out'] },
          body: { op: 'block', decls: [],
            assigns: [{ op: 'outputAssign', name: 'out', expr: {
              op: 'fold',
              over: { op: 'zipWith',
                a: { op: 'generate', count: 4, var: 'i',
                     body: { op: 'to_float', args: [{ op: 'binding', name: 'i' }] } },
                b: [10, 20, 30, 40],
                x_var: 'x', y_var: 'y',
                body: { op: 'mul', args: [
                  { op: 'add', args: [{ op: 'binding', name: 'x' }, { op: 'binding', name: 'y' }] },
                  { op: 'to_float', args: [{ op: 'bitAnd', args: [
                    { op: 'toInt', args: [{ op: 'sampleIndex' }] }, 255] }] } ] } },
              init: 0, acc_var: 'acc', elem_var: 'e',
              body: { op: 'add', args: [{ op: 'binding', name: 'acc' }, { op: 'binding', name: 'e' }] }
            }}],
            value: null } } },
        { op: 'instanceDecl', name: 'inst', program: 'ArrayCF', inputs: {} }
      ]},
      audio_outputs: [{ instance: 'inst', output: 'out' }],
    }
    expect(() => compileViaLean(program)).toThrow(/program definitions over the wire are retired/)
  })

  test('op-zoo (bitwise / shifts / float↔int casts / mod / clamp / select) — WASM matches JIT', async () => {
    const prog = opZooProgram()
    const wire = compileViaLean(prog)
    const N = 256
    const nat = runNative(wire, N)
    expectAudible('op-zoo', nat)
    const wasm = await runWasm(prog, wire, N)
    for (let i = 0; i < N; i++) expect(Math.abs(wasm[i]! - nat[i]!)).toBeLessThan(TOL)
  })
})
