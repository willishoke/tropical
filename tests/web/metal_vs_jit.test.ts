/**
 * metal_vs_jit.test.ts — compare the METAL backend against the native JIT.
 *
 * Sibling of wasm_vs_jit, but the comparison is SNR, not bytes: the GPU
 * kernel is f32-valued (Apple GPUs have no f64), so every output sample
 * quantizes to 24 bits (~-144 dB) — full-render byte-exactness is
 * IMPOSSIBLE by construction. What IS exact is the integer datapath
 * (clocks, phases, the Q2.30 sine — design/fixed-carrier.md): EmitMsl
 * folds literal landings in f64 to exact i64 constants, so a
 * literal-frequency patch's phase axis is bit-identical to the CPU's and
 * the SNR sits at the f32 value floor (~140 dB) FLAT IN TAU — the long-τ
 * cases render at sample 2^40 (~288 days in) and must not lose a dB.
 *
 * Known, deliberate exceptions (documented so nobody hunts a "flaky gate"):
 *  - SLOT-DRIVEN frequencies compute their landing in f32 in-kernel: the
 *    increment sits a few grid units off the CPU's, so phase error grows
 *    linearly with render length. Gate those on short windows only.
 *  - modal_heavy64 is the pre-scope-A benchmark relic: it evaluates
 *    sin(ω·t) on UNREDUCED float radians, the exact pathology scope A
 *    fixed in production paths. In f32 the π range-reduction bleeds with
 *    the argument size; it is kept here as the canary for that boundary,
 *    gated loosely on a short window.
 *
 * Both sides come off the SAME plan compiled by the Lean engine:
 *   - Native: `diffcli render-bytes` (f64, ORC JIT) — the reference.
 *   - Metal:  `diffcli render-metal` (EmitMsl → newLibraryWithSource →
 *     per-block GPU dispatch; the JIT is dual-loaded but the output is
 *     the GPU's).
 *
 * Requires libtropical built with TROPICAL_METAL on a Metal device —
 * otherwise the whole suite skips.
 */

import { describe, test, expect, setDefaultTimeout } from 'bun:test'
import { spawnSync } from 'bun'
import { resolve, dirname } from 'path'
import { fileURLToPath } from 'url'
import { expectAudible } from './energy_floor'

setDefaultTimeout(120_000)

const __dirname = dirname(fileURLToPath(import.meta.url))
const repoRoot = resolve(__dirname, '../..')
const diffcli = resolve(repoRoot, 'lean/.lake/build/bin/diffcli')
const cliEnv = { ...process.env, PATH: `${process.env.HOME}/.elan/bin:${process.env.PATH}` }

function compilePatch(patchPath: string): string {
  const r = spawnSync([diffcli, 'compile', resolve(repoRoot, patchPath), '--mode=fused'],
    { cwd: repoRoot, env: cliEnv })
  if (r.exitCode !== 0) throw new Error(`diffcli compile ${patchPath} failed: ${r.stderr?.toString()}`)
  return r.stdout.toString().trim()
}

function render(verb: 'render-bytes' | 'render-metal', planPath: string,
                samples: number, start: bigint = 0n): Float64Array {
  const buffer = 512
  const frames = Math.ceil(samples / buffer)
  const args = [diffcli, verb, planPath, '--frames', String(frames), '--buffer', String(buffer)]
  if (start !== 0n) args.push('--start', start.toString())
  const r = spawnSync(args, { cwd: repoRoot, env: cliEnv })
  if (r.exitCode !== 0) throw new Error(`diffcli ${verb} failed: ${r.stderr?.toString()}`)
  const b = r.stdout
  const n = frames * buffer
  return new Float64Array(b.buffer.slice(b.byteOffset, b.byteOffset + n * 8))
}

function snrDb(ref: Float64Array, got: Float64Array): number {
  let sig = 0, err = 0
  const n = Math.min(ref.length, got.length)
  for (let i = 0; i < n; i++) {
    sig += ref[i]! * ref[i]!
    const d = ref[i]! - got[i]!
    err += d * d
  }
  if (err === 0) return Infinity
  return 10 * Math.log10(sig / err)
}

import { writeFileSync } from 'fs'

function planFile(name: string, planJson: string): string {
  const p = `/tmp/mvj-${name}.plan.json`
  writeFileSync(p, planJson)
  return p
}

/** Probe: is the Metal backend actually available in this build/device? */
function metalAvailable(): boolean {
  if (process.platform !== 'darwin') return false
  try {
    const plan = planFile('probe', compilePatch('web/patches/pure-sine-440.json'))
    const r = spawnSync([diffcli, 'render-metal', plan, '--frames', '1', '--buffer', '64'],
      { cwd: repoRoot, env: cliEnv })
    return r.exitCode === 0
  } catch {
    return false
  }
}

const METAL = metalAvailable()
const FAR = 2n ** 40n   // ~288 days of playback at 44.1k

function gate(name: string, patch: string, samples: number, minDb: number, start: bigint = 0n) {
  const plan = planFile(name.replace(/[^a-z0-9-]/gi, '_'), compilePatch(patch))
  const ref = render('render-bytes', plan, samples, start)
  if (start === 0n) expectAudible(name, ref)
  const gpu = render('render-metal', plan, samples, start)
  const snr = snrDb(ref, gpu)
  console.log(`    ${name}: SNR ${snr.toFixed(1)} dB (floor ${minDb})`)
  expect(snr).toBeGreaterThan(minDb)
}

/** Render a playground PatchGraph through the TYPED session split
 *  (`diffcli render-graph`): banked coefficient columns hoist to the
 *  stage-0 coefficient kernel, and on `--metal` cross to the GPU via the
 *  packed `coeff_columns` buffer(3) — the same path a live session runs
 *  on TROPICAL_BACKEND=metal. Returns the samples plus the hoisted-column
 *  count the verb reports on stderr, so the gate can assert the crossing
 *  was actually exercised (a zero-column render would pass vacuously). */
function renderGraph(graphPath: string, metal: boolean,
                     samples: number): { out: Float64Array, columns: number } {
  const buffer = 512
  const frames = Math.ceil(samples / buffer)
  const args = [diffcli, 'render-graph', graphPath,
                '--frames', String(frames), '--buffer', String(buffer)]
  if (metal) args.push('--metal')
  const r = spawnSync(args, { cwd: repoRoot, env: cliEnv })
  if (r.exitCode !== 0) throw new Error(`diffcli render-graph failed: ${r.stderr?.toString()}`)
  const m = /hoisted columns=(\d+)/.exec(r.stderr?.toString() ?? '')
  const b = r.stdout
  const n = frames * buffer
  return { out: new Float64Array(b.buffer.slice(b.byteOffset, b.byteOffset + n * 8)),
           columns: m ? Number(m[1]) : 0 }
}

describe.skipIf(!METAL)('metal vs native JIT (SNR gates)', () => {
  test('pure-sine-440 — literal landing, f32 value floor', () => {
    gate('pure-sine-440', 'web/patches/pure-sine-440.json', 4096, 125)
  })

  test('pure-sine-440 @ τ+2^40 — long-τ FLAT (the integer-phase proof)', () => {
    gate('pure-sine-440 far', 'web/patches/pure-sine-440.json', 4096, 125, FAR)
  })

  test('tz-flanger — warped reads, reduced phases', () => {
    gate('tz-flanger', 'web/patches/tz-flanger.json', 4096, 125)
  })

  test('tz-flanger @ τ+2^40 — warps stay flat', () => {
    gate('tz-flanger far', 'web/patches/tz-flanger.json', 4096, 115, FAR)
  })

  test('reverse_reverb — future taps, modal voice', () => {
    gate('reverse_reverb', 'patches/reverse_reverb.json', 4096, 125)
  })

  test('scrub_reverb — the scrub-transport patch', () => {
    gate('scrub_reverb', 'patches/scrub_reverb.json', 4096, 125)
  })

  test('ring-mod — product of voices', () => {
    gate('ring-mod', 'web/patches/ring-mod.json', 4096, 125)
  })

  test('banked resonator — hoisted coefficient columns cross to the GPU (WS4)', () => {
    // The banked default lowers the resonator's mode bank to `bankSum`,
    // and the typed split hoists its LIVE coefficient columns (incr←freq,
    // sigma←decay) to the stage-0 kernel — array data the GPU can only
    // see through the coeff_columns buffer(3) crossing. The columns cross
    // f64→f32, so this is slot-driven by construction (the documented
    // exception class: phase error grows with the window). Measured
    // ~113 dB at 4096 samples on M-series; gate at 100 as the boundary.
    const graph = planFile('banked-resonator-graph', JSON.stringify({
      nodes: [
        { id: 'res', kind: 'resonator', params: { freq: 220, decay: 4 } },
        { id: 'out', kind: 'out', in: { in: ['res'] } },
      ],
      out: 'out',
    }))
    const ref = renderGraph(graph, false, 4096)
    expect(ref.columns).toBeGreaterThan(0)   // the split actually banked — no vacuous pass
    expectAudible('banked resonator', ref.out)
    const gpu = renderGraph(graph, true, 4096)
    expect(gpu.columns).toBe(ref.columns)
    const snr = snrDb(ref.out, gpu.out)
    console.log(`    banked resonator: ${ref.columns} hoisted column(s), SNR ${snr.toFixed(1)} dB (floor 100)`)
    expect(snr).toBeGreaterThan(100)
  })

  test('modal_heavy64 — the pre-scope-A unreduced-radian canary (short window only)', () => {
    // sin(ω·t) on a growing float argument: f32 π-reduction bleeds with
    // arg size — the pathology scope A fixed in production paths. ~92 dB
    // at 0.37 s on M1 Pro; gate at 80 as the documented boundary.
    gate('modal_heavy64', 'benchmarks/metal_patch/modal_heavy64.json', 16384, 80)
  })
})

describe.skipIf(METAL)('metal vs native JIT', () => {
  test.skip('skipped — libtropical built without TROPICAL_METAL or no Metal device', () => {})
})
