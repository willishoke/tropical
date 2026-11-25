/**
 * runtime_params.test.ts — the player's live knob writes (`ParamWriter`,
 * web/runtime/params.ts) against design/host-param-dispatch.md.
 *
 *  1. An anchored frequency (a knob wired straight into a `harmonic`'s freq)
 *     re-pitched ten hours in stays continuous: the `#phase` companion takes
 *     the jump `Δinc·now` exactly, and the new pitch is the knob's. The same
 *     write done raw jumps, so the check discriminates.
 *  2. A raw write is the value.
 *  3. Companions and disciplines the player does not implement are refused by
 *     name, never written raw.
 */
import { describe, test, expect, setDefaultTimeout } from 'bun:test'
import { spawnSync } from 'bun'
import { writeFileSync, readFileSync, mkdtempSync } from 'fs'
import { tmpdir } from 'os'
import { join, resolve, dirname } from 'path'
import { fileURLToPath } from 'url'
import { WasmKernel, ParamWriter, kernelParams, type KernelManifest } from '../../web/runtime/index'

setDefaultTimeout(120_000)

const __dirname = dirname(fileURLToPath(import.meta.url))
const repoRoot = resolve(__dirname, '../..')
const diffcli = process.env.TROPICAL_DIFFCLI
  ? resolve(process.env.TROPICAL_DIFFCLI)
  : resolve(repoRoot, 'lean/.lake/build/bin/diffcli')
const dir = mkdtempSync(join(tmpdir(), 'runtime-params-'))

function run(args: string[]): Buffer {
  const r = spawnSync([diffcli, ...args], { cwd: repoRoot })
  if (r.exitCode !== 0) throw new Error(`diffcli ${args[0]} failed: ${r.stderr.toString()}`)
  return r.stdout as Buffer
}

async function load(graph: unknown): Promise<{ k: WasmKernel; manifest: KernelManifest }> {
  const path = join(dir, `${Math.random().toString(36).slice(2)}.json`)
  writeFileSync(path, JSON.stringify(graph))
  const wasmPath = path.replace(/\.json$/, '.wasm')
  run(['compile-wasm', path, `--out=${wasmPath}`])
  const manifest = JSON.parse(run(['compile-kernel', path]).toString()) as KernelManifest
  return { k: await WasmKernel.instantiate(new Uint8Array(readFileSync(wasmPath)), manifest), manifest }
}

function zeroCrossingHz(x: Float64Array, rate: number): number {
  const ups: number[] = []
  for (let i = 1; i < x.length; i++) {
    const a = x[i - 1]!, b = x[i]!
    if (a < 0 && b >= 0) ups.push(i - 1 + a / (a - b))
  }
  return ((ups.length - 1) * rate) / (ups[ups.length - 1]! - ups[0]!)
}

// one partial: a pure sine at f0, so continuity is a bound on the step
const sineFamily = {
  nodes: [
    { id: 'f0', kind: 'knob', params: { value: 220 } },
    { id: 'h', kind: 'harmonic', params: { partials_max: 1, partials: 1 }, in: { freq: ['f0'] } },
  ],
  out: 'h',
}

describe('ParamWriter', () => {
  test('anchor: a re-pitch ten hours in is continuous and plays the new pitch; raw jumps', async () => {
    const far = 10 * 3600 * 44100 + 12345
    const n = 4096
    const run1 = async (anchored: boolean) => {
      const { k, manifest } = await load(sineFamily)
      const writer = new ParamWriter(manifest)
      expect(writer.check('param:f0.value')).toBeNull()
      k.seek(far)
      const before = k.render(n).slice()
      if (anchored) writer.write(k, 'param:f0.value', 330, k.frame)
      else k.setParam(kernelParams(manifest).findIndex((p) => p.name === 'param:f0.value'), 330)
      const after = k.render(n).slice()
      return { before, after, rate: manifest.rate }
    }
    const a = await run1(true)
    const peak = a.before.reduce((m, v) => Math.max(m, Math.abs(v)), 0)
    // the largest step a sine of this peak takes at 330 Hz, with margin
    const bound = 1.05 * peak * 2 * Math.PI * 330 / a.rate
    const step = Math.abs(a.after[0]! - a.before[n - 1]!)
    const hz = zeroCrossingHz(a.after, a.rate)
    const r = await run1(false)
    const rawStep = Math.abs(r.after[0]! - r.before[n - 1]!)
    console.log(`  anchored step ${step.toExponential(3)} (bound ${bound.toExponential(3)}), raw step ${rawStep.toExponential(3)}, pitch after ${hz.toFixed(4)} Hz`)
    expect(step).toBeLessThan(bound)
    expect(rawStep).toBeGreaterThan(bound)
    expect(Math.abs(hz - 330)).toBeLessThan(0.01)
  })

  test('anchor on the master clock: bit-exact at the write under tau_base, velocity and a negative frequency', async () => {
    // The playground anchor declares `clock: master.velocity`: the phasor
    // reads trunc(tau_base·rate·2¹⁶) + scaleClockQ(trunc(velocity·2³²), τ),
    // not τ. A seamless write leaves the sample AT the write frame unchanged
    // (the output there is a function of the phase alone) and changes the
    // next one (the pitch moved). The clock slots are set raw: the player
    // does not dispatch velocity writes, and the setup is not the subject.
    const { k, manifest } = await load({ nodes: [{ id: 'osc', kind: 'sine', params: { freq: 440 } }], out: 'osc' })
    const writer = new ParamWriter(manifest)
    const names = kernelParams(manifest).map((p) => p.name)
    const set = (name: string, v: number) => k.setParam(names.indexOf(name), v)
    const at = (frame: number) => { k.seek(frame); return k.render(2) }
    const now = 9_876_543
    for (const [velocity, tauBase, f1] of [[1.5, 0.37, 660], [-1, 12.5, 523.25], [0.75, 3.1, -317.3]] as const) {
      set('param:master.velocity', velocity)
      set('param:master.tau_base', tauBase)
      set('param:osc.freq', 440)
      set('param:osc.freq#phase', 0)
      const before = at(now)
      writer.write(k, 'param:osc.freq', f1, now)
      const after = at(now)
      expect(Object.is(after[0], before[0])).toBe(true)
      expect(after[1]).not.toBe(before[1])
    }
    // A clock that is missing or names no velocity param is refused when the
    // writer is built, never resolved at write time.
    const withClock = (clock: string | undefined) => {
      const m = structuredClone(manifest) as KernelManifest
      const port = m.inputs.find((p) => p.name === 'param:osc.freq') as { discipline?: { clock?: string } }
      if (clock === undefined) delete port.discipline!.clock
      else port.discipline!.clock = clock
      return m
    }
    expect(() => new ParamWriter(withClock(undefined))).toThrow(/exactly on an anchor_clock/)
    expect(() => new ParamWriter(withClock('master.gain'))).toThrow(/not a velocity param/)
    expect(() => new ParamWriter(withClock(''))).toThrow(/not a velocity param/)
    // So is a discipline the contract does not name, and a clock (the empty
    // string included) on anything but an anchor_clock.
    const withDiscipline = (discipline: string, clock: string | undefined) => {
      const m = withClock(clock)
      const port = m.inputs.find((p) => p.name === 'param:osc.freq') as { discipline?: { discipline: string } }
      port.discipline!.discipline = discipline
      return m
    }
    expect(() => new ParamWriter(withDiscipline('anchor_v2', undefined))).toThrow(/unknown discipline 'anchor_v2'/)
    expect(() => new ParamWriter(withDiscipline('anchor', ''))).toThrow(/exactly on an anchor_clock/)
    expect(() => new ParamWriter(withDiscipline('anchor', undefined))).not.toThrow()
  })

  test('a patch-bay program: an anchored re-pitch of a spline voice is continuous, a raw one jumps', async () => {
    // design/host-param-dispatch.md, "Patch-bay params": `pitch` declares
    // `anchor` on its paramDecl and is wired into ClockPhasor.freq, whose
    // `clk` is τ. The compile adds `pitch#phase` on the phasor's `offset`.
    // The sample at the write frame is a function of the phase alone: an
    // anchored write leaves it unchanged, a raw one moves it.
    const patch = (discipline?: string) => ({
      schema: 'tropical_program_2', name: 'spline_voice',
      body: { op: 'block', decls: [
        { op: 'paramDecl', name: 'pitch', value: 220, ...(discipline ? { discipline } : {}) },
        { op: 'instanceDecl', name: 'ph', program: 'ClockPhasor',
          inputs: { clk: { op: 'clock' }, freq: { op: 'param', name: 'pitch' } } },
        { op: 'instanceDecl', name: 'wave', program: 'PeriodicCatmullRom8', inputs: {
          phase: { op: 'toInt', args: [{ op: 'mul', args: [{ op: 'ref', instance: 'ph', output: 'phase' }, 4294967296] }] },
          c0: 0, c1: 0.5, c2: 0.6, c3: 0.2, c4: -0.1, c5: -0.5, c6: -0.4, c7: -0.2 } },
      ], assigns: [{ op: 'outputAssign', name: 'dac.out', expr: { op: 'ref', instance: 'wave', output: 'out' } }] },
    })
    const { k, manifest } = await load(patch('anchor'))
    const port = manifest.inputs.find((p) => p.name === 'param:pitch') as { discipline?: { discipline: string; companions?: string[] } }
    expect(port.discipline?.discipline).toBe('anchor')
    expect(port.discipline?.companions).toEqual(['pitch#phase'])
    const writer = new ParamWriter(manifest)
    expect(writer.check('param:pitch')).toBeNull()
    const names = kernelParams(manifest).map((p) => p.name)
    const at = (frame: number) => { k.seek(frame); return k.render(2) }
    // ~265 days in, then a negative frequency
    for (const [now, f1] of [[2 ** 40 + 7, 523.25], [2 ** 40 + 99_991, -110]] as const) {
      const before = at(now)
      writer.write(k, 'param:pitch', f1, now)
      const after = at(now)
      expect(Object.is(after[0], before[0])).toBe(true)
      expect(after[1]).not.toBe(before[1])
    }
    // The same write raw (the companion untouched) jumps.
    const before = at(2 ** 40 + 7)
    k.setParam(names.indexOf('param:pitch'), 330)
    const after = at(2 ** 40 + 7)
    expect(after[0]).not.toBe(before[0])
    // Undeclared, the param is raw and has no companion.
    const plain = await load(patch())
    expect(kernelParams(plain.manifest).map((p) => p.name)).not.toContain('param:pitch#phase')
  })

  test('raw: the write is the value; companions and unimplemented disciplines are refused', async () => {
    const { k, manifest } = await load({ ...sineFamily, nodes: [...sineFamily.nodes,
      { id: 'g', kind: 'knob', params: { value: 0.5 } },
      { id: 'v', kind: 'ring', in: { in: ['h', 'g'] } }], out: 'v' })
    const writer = new ParamWriter(manifest)
    const names = kernelParams(manifest).map((p) => p.name)
    writer.write(k, 'param:g.value', 0.25, k.frame)
    expect(k.getParam(names.indexOf('param:g.value'))).toBe(0.25)
    expect(writer.check('param:f0.value#phase')).toContain('discipline companion')
    expect(writer.check('param:master.velocity')).toContain("discipline 'velocity'")
    expect(writer.check('param:nope.value')).toContain('no param input')
    expect(() => writer.write(k, 'param:master.velocity', 2, 0)).toThrow(/velocity/)
  })
})
