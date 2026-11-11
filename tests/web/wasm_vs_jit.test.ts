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
  body: { op: 'block'; decls: unknown[]; assigns: unknown[] }
}

const PROG_TMP = '/tmp/wvj-program.json'
const dacOut = (instance: string, output: string) => ({
  op: 'outputAssign',
  name: 'dac.out',
  expr: { op: 'ref', instance, output },
})

/** Compile a `ProgramFile` to a `tropical_plan_7` wire plan JSON string via
 *  the Lean `diffcli compile` front door (native oracle + manifest source).
 *  `--fixtures` registers the builder-authored test programs (`OpZoo`) on top
 *  of the stdlib so patch-bay fixtures can instantiate them by name. */
function compileViaLean(program: ProgramFile): string {
  writeFileSync(PROG_TMP, JSON.stringify(program))
  const r = spawnSync([diffcli, 'compile', PROG_TMP, '--mode=fused', '--fixtures'], { cwd: repoRoot, env: cliEnv })
  if (r.exitCode !== 0) throw new Error(`diffcli compile failed (exit ${r.exitCode}): ${r.stderr?.toString()}`)
  return r.stdout.toString().trim()
}

/** The `tropical_kernel_1` manifest the runtime consumes, for the same
 *  source, via `diffcli compile-kernel` (128 frames per call: the worklet
 *  quantum, and what `compile-wasm` defaults to). */
function kernelManifestViaLean(program: ProgramFile): KernelManifest {
  writeFileSync(PROG_TMP, JSON.stringify(program))
  const r = spawnSync([diffcli, 'compile-kernel', PROG_TMP, '--fixtures'], { cwd: repoRoot, env: cliEnv })
  if (r.exitCode !== 0) throw new Error(`diffcli compile-kernel failed (exit ${r.exitCode}): ${r.stderr?.toString()}`)
  return JSON.parse(r.stdout.toString()) as KernelManifest
}

function outputChannelsOfPlan(planJson: string): number {
  const p = JSON.parse(planJson)
  return p.output_channel_count ??
    (p.sinks ?? []).reduce(
      (count: number, sink: { target?: number }) =>
        Math.max(count, (sink.target ?? 0) + 1),
      1,
    )
}

/** Native side: render the wire plan through the engine's JIT. */
function runNative(wirePlan: string, samples: number): Float64Array {
  const tmp = '/tmp/wvj-native.plan.json'
  writeFileSync(tmp, wirePlan)
  const r = spawnSync([diffcli, 'render-bytes', tmp, '--frames', '1', '--buffer', String(samples)], { cwd: repoRoot, env: cliEnv })
  if (r.exitCode !== 0) throw new Error(`diffcli render-bytes failed (exit ${r.exitCode}): ${r.stderr?.toString()}`)
  const b = r.stdout
  const outputSamples = samples * outputChannelsOfPlan(wirePlan)
  return new Float64Array(
    b.buffer.slice(b.byteOffset, b.byteOffset + outputSamples * 8))
}

/** WASM side: lower the same source to wasm32 in-process and drive it through
 *  the runtime package (the artifacts the browser ships). */
async function runWasm(program: ProgramFile, _plan: string, samples: number): Promise<Float64Array> {
  const manifest = kernelManifestViaLean(program)
  writeFileSync(PROG_TMP, JSON.stringify(program))
  const out = '/tmp/wvj-program.wasm'
  const r = spawnSync([diffcli, 'compile-wasm', PROG_TMP, `--out=${out}`, '--fixtures'], { cwd: repoRoot, env: cliEnv })
  if (r.exitCode !== 0) throw new Error(`diffcli compile-wasm failed (exit ${r.exitCode}): ${r.stderr?.toString()}`)
  // The kernel's shape is static (128 frames); `render` chunks a longer
  // window over the same closed form, so the sample count is free.
  const k = await WasmKernel.instantiate(new Uint8Array(readFileSync(out)), manifest)
  return k.render(samples).slice()
}

function sinOscProgram(freqHz: number): ProgramFile {
  return {
    schema: 'tropical_program_2',
    name: 'eq_sinosc',
    body: { op: 'block', decls: [
      { op: 'instanceDecl', name: 'osc', program: 'FixedSinOsc', inputs: { freq: freqHz } },
    ], assigns: [dacOut('osc', 'sine')] },
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
    ], assigns: [dacOut('sc', 'out')] },
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
    ], assigns: [dacOut('inst', 'out')] },
  }
}

// The spectral fixtures (`Testing/SpectralFixtures.lean`, registered by
// `--fixtures`): a 200-partial family (one `bankSum` region over a coefficient
// table) and the same family under a complex SVF-lowpass envelope whose cutoff
// is wired from an audio-rate LFO instance — the per-partial complex gain, its
// division, and the soft edges all run on both targets.
function spectralSawProgram(f0: number): ProgramFile {
  return {
    schema: 'tropical_program_2',
    name: 'eq_spectral_saw',
    body: { op: 'block', decls: [
      { op: 'instanceDecl', name: 'saw', program: 'SpectralSaw', inputs: { f0, phase: 0.25 } },
    ], assigns: [dacOut('saw', 'out')] },
  }
}

function spectralLowpassProgram(f0: number, lfoHz: number, q: number): ProgramFile {
  return {
    schema: 'tropical_program_2',
    name: 'eq_spectral_lowpass',
    body: { op: 'block', decls: [
      { op: 'instanceDecl', name: 'lfo', program: 'ExpCutoffLfo', inputs: { c0: 800, octaves: 2, rate: lfoHz } },
      { op: 'instanceDecl', name: 'saw', program: 'SpectralLowpassSaw', inputs: {
        f0, cutoff: { op: 'ref', instance: 'lfo', output: 'out' }, q } },
    ], assigns: [dacOut('saw', 'out')] },
  }
}

// The int32 kernels: `FixedSinOscQ15` (int32 throughout, an int32 output),
// `Int32Zoo` (every int32 op at runtime, seeded from the sample index's low
// word; registered by `--fixtures`), and `Int32Zoo` with its int32 input wired
// from the LFO's int32 output — a module-slot crossing at 32 bits. Integer
// semantics are exact, so these compare bit for bit, not within TOL.
function q15Program(freqHz: number, phase: number): ProgramFile {
  return {
    schema: 'tropical_program_2',
    name: 'eq_q15',
    body: { op: 'block', decls: [
      { op: 'instanceDecl', name: 'lfo', program: 'FixedSinOscQ15', inputs: { freq: freqHz, phase } },
    ], assigns: [dacOut('lfo', 'out')] },
  }
}
function int32ZooProgram(wired: boolean): ProgramFile {
  const decls: unknown[] = wired
    ? [
        { op: 'instanceDecl', name: 'lfo', program: 'FixedSinOscQ15', inputs: { freq: 1234.567 } },
        { op: 'instanceDecl', name: 'zoo', program: 'Int32Zoo', inputs: {
          x: { op: 'ref', instance: 'lfo', output: 'out' } } },
      ]
    : [{ op: 'instanceDecl', name: 'zoo', program: 'Int32Zoo', inputs: {} }]
  return {
    schema: 'tropical_program_2',
    name: wired ? 'eq_int32zoo_wired' : 'eq_int32zoo',
    body: { op: 'block', decls, assigns: [dacOut('zoo', 'out')] },
  }
}

// The integer edge-value programs (`IntEdge64`, `IntEdge32`, `IntEdgeConv`,
// registered by `--fixtures`; `EmitArrow.intEdgeSig`): every `int` and `int32`
// op on MIN, MAX, -1, 0, 1, runtime divisors 0 and -1, shift amounts past the
// word, and NaN/±∞/out-of-range floats into the conversions, read out as exact
// 16-bit pieces. `tropicaltest`'s `int-edge` gate holds the JIT to the
// reference semantics on these programs; here WASM is held to the JIT bit for
// bit (before the total semantics, the zero and MIN/-1 divisors trapped).
// Frames: 26 ops × 106 pairs × 4 pieces, and 2 ops × 22 floats × 4 pieces.
const INT_EDGE: Array<[string, number]> = [
  ['IntEdge64', 26 * 106 * 4], ['IntEdge32', 26 * 106 * 4], ['IntEdgeConv', 2 * 22 * 4]]
function intEdgeProgram(program: string): ProgramFile {
  return {
    schema: 'tropical_program_2',
    name: `eq_${program.toLowerCase()}`,
    body: { op: 'block', decls: [{ op: 'instanceDecl', name: 'e', program, inputs: {} }],
            assigns: [dacOut('e', 'out')] },
  }
}

async function expectBitIdentical(name: string, prog: ProgramFile, N: number): Promise<void> {
  const wire = compileViaLean(prog)
  const nat = runNative(wire, N)
  expectAudible(name, nat)
  const wasm = await runWasm(prog, wire, N)
  expect(wasm.length).toBe(nat.length)
  for (let i = 0; i < nat.length; i++) expect(wasm[i]).toBe(nat[i]!)
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
    expect(wasm.length).toBe(nat.length)
    for (let i = 0; i < nat.length; i++) expect(Math.abs(wasm[i]! - nat[i]!)).toBeLessThan(TOL)
  })

  test('FixedSinOsc 880 Hz — 128 samples', async () => {
    const prog = sinOscProgram(880)
    const wire = compileViaLean(prog)
    const N = 128
    const nat = runNative(wire, N)
    expectAudible('FixedSinOsc 880 Hz', nat)
    const wasm = await runWasm(prog, wire, N)
    expect(wasm.length).toBe(nat.length)
    for (let i = 0; i < nat.length; i++) expect(Math.abs(wasm[i]! - nat[i]!)).toBeLessThan(TOL)
  })

  test('FixedSinOsc → SoftClip(drive 4) — 256 samples', async () => {
    const prog = softClipChainProgram(4)
    const wire = compileViaLean(prog)
    const N = 256
    const nat = runNative(wire, N)
    expectAudible('FixedSinOsc → SoftClip(drive 4)', nat)
    const wasm = await runWasm(prog, wire, N)
    expect(wasm.length).toBe(nat.length)
    for (let i = 0; i < nat.length; i++) expect(Math.abs(wasm[i]! - nat[i]!)).toBeLessThan(TOL)
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
      ], assigns: [dacOut('inst', 'out')] },
    }
    expect(() => compileViaLean(program)).toThrow(/program definitions over the wire are retired/)
  })

  test('SpectralSaw (200-partial family) — WASM matches JIT', async () => {
    const prog = spectralSawProgram(110)
    const wire = compileViaLean(prog)
    const N = 512
    const nat = runNative(wire, N)
    expectAudible('SpectralSaw', nat)
    const wasm = await runWasm(prog, wire, N)
    expect(wasm.length).toBe(nat.length)
    for (let i = 0; i < nat.length; i++) expect(Math.abs(wasm[i]! - nat[i]!)).toBeLessThan(TOL)
  })

  test('SpectralLowpassSaw, cutoff from a 220 Hz LFO (complex envelope) — WASM matches JIT', async () => {
    const prog = spectralLowpassProgram(82.5, 220, 12)
    const wire = compileViaLean(prog)
    const N = 512
    const nat = runNative(wire, N)
    expectAudible('SpectralLowpassSaw', nat)
    const wasm = await runWasm(prog, wire, N)
    expect(wasm.length).toBe(nat.length)
    for (let i = 0; i < nat.length; i++) expect(Math.abs(wasm[i]! - nat[i]!)).toBeLessThan(TOL)
  })

  test('FixedSinOscQ15 (int32 datapath, int32 output) — WASM ≡ JIT bit for bit', async () => {
    await expectBitIdentical('FixedSinOscQ15', q15Program(1234.567, 0.3), 512)
  })

  test('int32 op zoo — WASM ≡ JIT bit for bit', async () => {
    await expectBitIdentical('Int32Zoo', int32ZooProgram(false), 512)
  })

  test('int32 op zoo, input wired from an int32 output (32-bit slot) — WASM ≡ JIT bit for bit', async () => {
    await expectBitIdentical('Int32Zoo ← FixedSinOscQ15', int32ZooProgram(true), 512)
  })

  for (const [program, frames] of INT_EDGE) {
    test(`${program} (integer edge values, total semantics) — WASM ≡ JIT bit for bit`, async () => {
      await expectBitIdentical(program, intEdgeProgram(program), frames)
    })
  }

  test('op-zoo (bitwise / shifts / float↔int casts / mod / clamp / select) — WASM matches JIT', async () => {
    const prog = opZooProgram()
    const wire = compileViaLean(prog)
    const N = 256
    const nat = runNative(wire, N)
    expectAudible('op-zoo', nat)
    const wasm = await runWasm(prog, wire, N)
    expect(wasm.length).toBe(nat.length)
    for (let i = 0; i < nat.length; i++) expect(Math.abs(wasm[i]! - nat[i]!)).toBeLessThan(TOL)
  })
})
