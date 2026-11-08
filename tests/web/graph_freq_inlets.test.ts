/**
 * graph_freq_inlets.test.ts — frequency and rate inlets take only wires that
 * are constant within a kernel call.
 *
 * A generator's phase is `inc·τ`, its increment times the clock. That is exact
 * for a constant increment. For one that moves over time it is not the
 * integral of the frequency: a 440 Hz sine with ±1 Hz of vibrato wired into
 * its freq drifts by `Δinc·τ` and reaches kilohertz hours in. So every
 * frequency or rate inlet (`Metadata.isUniform`) accepts only a stage-0 wire —
 * a knob, arithmetic on knobs, a constant — and the compiler refuses anything
 * that varies over time, pointing at `pm` and `fm`, which modulate exactly.
 *
 *  1. A time-varying wire into each such inlet fails to compile, on the wasm
 *     verbs and on the JIT, with the message.
 *  2. Knob arithmetic into each generator's freq compiles and plays the
 *     arithmetic's pitch: measured by zero crossings near frame 0 on the JIT
 *     and ten hours in on the wasm kernel, and again after a knob write.
 *  3. A knob wired straight into a freq inlet is anchored: the manifest
 *     declares its `#phase` companion, and an expression of knobs gets none.
 */
import { describe, test, expect, setDefaultTimeout } from 'bun:test'
import { spawnSync } from 'bun'
import { writeFileSync, readFileSync, mkdtempSync } from 'fs'
import { tmpdir } from 'os'
import { join, resolve, dirname } from 'path'
import { fileURLToPath } from 'url'
import { WasmKernel, kernelParams, type KernelManifest } from '../../web/runtime/index'

setDefaultTimeout(120_000)

const __dirname = dirname(fileURLToPath(import.meta.url))
const repoRoot = resolve(__dirname, '../..')
const diffcli = process.env.TROPICAL_DIFFCLI
  ? resolve(process.env.TROPICAL_DIFFCLI)
  : resolve(repoRoot, 'lean/.lake/build/bin/diffcli')
const dir = mkdtempSync(join(tmpdir(), 'graph-freq-'))

function write(name: string, graph: unknown): string {
  const path = join(dir, `${name}.json`)
  writeFileSync(path, JSON.stringify(graph))
  return path
}

function run(args: string[]): Buffer {
  const r = spawnSync([diffcli, ...args], { cwd: repoRoot })
  if (r.exitCode !== 0) throw new Error(`diffcli ${args[0]} failed: ${r.stderr.toString()}`)
  return r.stdout as Buffer
}

/** The pitch of channel `ch` from its upward zero crossings, each placed by
 *  linear interpolation: (crossings − 1) periods over the span between the
 *  first and the last. */
function zeroCrossingHz(x: Float64Array, channels: number, ch: number, rate: number): number {
  const ups: number[] = []
  for (let i = 1; i < x.length / channels; i++) {
    const a = x[(i - 1) * channels + ch]!, b = x[i * channels + ch]!
    if (a < 0 && b >= 0) ups.push(i - 1 + a / (a - b))
  }
  return ((ups.length - 1) * rate) / (ups[ups.length - 1]! - ups[0]!)
}

/** `id` = knob `center` + `span`·(the 3 Hz sine `lfo`): vibrato around a knob. */
const lfoAround = (id: string, center: number, span: number): object[] => [
  { id: `${id}_c`, kind: 'knob', params: { value: center } },
  { id: `${id}_s`, kind: 'knob', params: { value: span } },
  { id: `${id}_m`, kind: 'ring', in: { in: ['lfo', `${id}_s`] } },
  { id, kind: 'mix', in: { in: [`${id}_c`, `${id}_m`] } },
]

// Every frequency or rate inlet: [kind, port, the inlet's knob value, the
// node's other inlets]. The modal kinds take a resonator as input.
const inlets: [kind: string, port: string, value: number, other: Record<string, string[]>][] = [
  ['source', 'freq', 440, {}],
  ['sine', 'freq', 440, {}],
  ['saw', 'freq', 440, {}],
  ['pulse', 'freq', 440, {}],
  ['harmonic', 'freq', 440, {}],
  ['harmonic', 'partials', 16, {}],
  ['pluck', 'freq', 440, {}],
  ['pluck', 'event_rate', 4, {}],
  ['fm', 'carrier', 440, { in: ['lfo'] }],
  ['phaser', 'rate', 0.5, { in: ['res'] }],
  ['modal_allpass_tail', 'rate', 0.5, { in: ['res'] }],
]
const pitchPorts = new Set(['freq', 'carrier'])

describe('a time-varying wire into a frequency or rate inlet is refused', () => {
  for (const [kind, port, value, other] of inlets) {
    // three time-varying drivers: a bare LFO, vibrato around a knob, and a
    // modal source realized at the inlet
    const drivers: [label: string, id: string, nodes: object[]][] = [
      ['an LFO', 'lfo', []],
      ['an LFO around a knob', 'vib', lfoAround('vib', value, value / 100)],
      ['a realized resonator', 'res', []],
    ]
    for (const [label, driver, nodes] of drivers) {
      test(`${kind}.${port} ← ${label}`, () => {
        const graph = { nodes: [
          { id: 'lfo', kind: 'sine', params: { freq: 3 } },
          { id: 'res', kind: 'resonator', params: { freq: 220, decay: 4 } },
          ...nodes,
          { id: 'x', kind, in: { ...other, [port]: [driver] } }], out: 'x' }
        const path = write(`refuse-${kind}-${port}-${driver}`, graph)
        const verbs = driver === 'vib'
          ? [['compile-kernel', path], ['render-graph', path, '--frames', '1', '--buffer', '64']]
          : [['compile-kernel', path]]
        for (const args of verbs) {
          const r = spawnSync([diffcli, ...args], { cwd: repoRoot })
          const err = r.stderr.toString()
          expect(r.exitCode).not.toBe(0)
          expect(r.stdout.length).toBe(0)
          // the phaser is a module over six all-pass atoms: the message names
          // the module the author placed, and the atom the wire reached
          const label = kind === 'phaser' ? `'x' (a module; its modal_allpass_tail 'tail')` : `'x' (${kind})`
          expect(err).toContain(`time-varying ${port}: ${label} inlet '${port}' is wired from '${driver}', which varies over time`)
          expect(err).toContain('constant within a kernel call: a knob, arithmetic on knobs, or a constant')
          if (kind === 'harmonic') {
            // a family's fundamental: pitch moves through a clock warp after it
            if (port === 'freq') expect(err).toContain('a phase-modulated family')
            if (port === 'freq') expect(err).toContain("a `delay` whose `amount` is the signal")
            if (port === 'partials') expect(err).toContain("trip count")
          } else if (pitchPorts.has(port)) {
            expect(err).toContain("a `source`'s `pm` inlet")
            expect(err).toContain("an `fm` node's `in`")
          }
        }
      })
    }
  }

  test('the same graphs compile with a knob, or knob arithmetic, in place of the time-varying wire', () => {
    for (const [kind, port, value, other] of inlets) {
      for (const [driver, nodes] of [
        ['k', [{ id: 'k', kind: 'knob', params: { value } }]],
        ['c', [{ id: 'a', kind: 'knob', params: { value: value / 2 } },
          { id: 'two', kind: 'knob', params: { value: 2 } },
          { id: 'c', kind: 'ring', in: { in: ['a', 'two'] } }]],
      ] as [string, object[]][]) {
        const path = write(`admit-${kind}-${port}-${driver}`, { nodes: [
          { id: 'lfo', kind: 'sine', params: { freq: 3 } },
          { id: 'res', kind: 'resonator', params: { freq: 220, decay: 4 } },
          ...nodes,
          { id: 'x', kind, in: { ...other, [port]: [driver] } }], out: 'x' })
        const manifest = JSON.parse(run(['compile-kernel', path]).toString()) as KernelManifest
        expect(manifest.schema).toBe('tropical_kernel_1')
      }
    }
  })
})

describe('knob arithmetic into a freq inlet plays its pitch', () => {
  // Each generator's freq is driven by a ring (110 × 2) or a mix (200 + 20) of
  // knobs: 220 Hz either way. The node's own freq param (999) is ignored when
  // the inlet is wired. The fm node's modulator is a zero knob (no warp).
  const generators: [kind: string, params: Record<string, number>, other: Record<string, string[]>][] = [
    ['sine', {}, {}],
    ['saw', {}, {}],
    ['pulse', { width: 0.5 }, {}],
    ['source', { morph: 0 }, {}],
    ['fm', { depth: 0 }, { in: ['zero'] }],
    // the partial family: a 64-partial band-limited saw, one upward zero
    // crossing per period (at the ramp's reset)
    ['harmonic', {}, {}],
  ]
  const cones: [label: string, nodes: object[]][] = [
    ['110 × 2', [
      { id: 'a', kind: 'knob', params: { value: 110 } },
      { id: 'b', kind: 'knob', params: { value: 2 } },
      { id: 'f', kind: 'ring', in: { in: ['a', 'b'] } }]],
    ['200 + 20', [
      { id: 'a', kind: 'knob', params: { value: 200 } },
      { id: 'b', kind: 'knob', params: { value: 20 } },
      { id: 'f', kind: 'mix', in: { in: ['a', 'b'] } }]],
  ]
  const tenHours = 10 * 3600 * 44100
  for (const [kind, params, other] of generators) {
    for (const [label, nodes] of cones) {
      test(`${kind}.${kind === 'fm' ? 'carrier' : 'freq'} ← ${label}: 220 Hz at frame 0 (JIT) and ten hours in (wasm)`, async () => {
        const port = kind === 'fm' ? 'carrier' : 'freq'
        const path = write(`pitch-${kind}-${label.replace(/\W+/g, '')}`, { nodes: [
          ...nodes,
          { id: 'zero', kind: 'knob', params: { value: 0 } },
          { id: 'x', kind, params: { ...params, [port]: 999 }, in: { ...other, [port]: ['f'] } }],
          out: 'x' })
        const n = 16384
        const b = run(['render-graph', path, '--frames', '1', '--buffer', String(n)])
        const jit = new Float64Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength))
        const hzJit = zeroCrossingHz(jit, 1, 0, 44100)

        const wasmPath = join(dir, `pitch-${kind}-${Math.random().toString(36).slice(2)}.wasm`)
        run(['compile-wasm', path, `--out=${wasmPath}`])
        const manifest = JSON.parse(run(['compile-kernel', path]).toString()) as KernelManifest
        const k = await WasmKernel.instantiate(new Uint8Array(readFileSync(wasmPath)), manifest)
        k.seek(tenHours)
        const far = k.render(n).slice()
        const hzFar = zeroCrossingHz(far, k.outputChannelCount, 0, manifest.rate)
        // a knob write re-pitches with no relower: a = 165 → 330 Hz (ring), 310 → 330 Hz (mix)
        const names = kernelParams(manifest).map((p) => p.name)
        k.setParam(names.indexOf('param:a.value'), label.startsWith('110') ? 165 : 310)
        k.seek(tenHours)
        const retuned = k.render(n).slice()
        const hzRetuned = zeroCrossingHz(retuned, k.outputChannelCount, 0, manifest.rate)
        console.log(`  ${kind} ← ${label}: ${hzJit.toFixed(4)} Hz at frame 0, ${hzFar.toFixed(4)} Hz ten hours in, ${hzRetuned.toFixed(4)} Hz after the write`)
        expect(Math.abs(hzJit - 220)).toBeLessThan(0.01)
        expect(Math.abs(hzFar - 220)).toBeLessThan(0.01)
        expect(Math.abs(hzRetuned - 330)).toBeLessThan(0.01)
      })
    }
  }

  test('pluck.freq ← 110 × 2: the plucked voice sounds at 220 Hz', () => {
    const path = write('pitch-pluck', { nodes: [
      { id: 'a', kind: 'knob', params: { value: 110 } },
      { id: 'b', kind: 'knob', params: { value: 2 } },
      { id: 'f', kind: 'ring', in: { in: ['a', 'b'] } },
      { id: 'x', kind: 'pluck', params: { freq: 999, morph: 0, event_rate: 2 }, in: { freq: ['f'] } }],
      out: 'x' })
    const n = 8192
    const b = run(['render-graph', path, '--frames', '1', '--buffer', String(n)])
    const x = new Float64Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength))
    const hz = zeroCrossingHz(x, 1, 0, 44100)
    console.log(`  pluck ← 110 × 2: ${hz.toFixed(4)} Hz`)
    expect(Math.abs(hz - 220)).toBeLessThan(0.05)
  })
})

describe('a knob wired straight into a freq inlet is anchored', () => {
  test('the knob slot gets the anchor discipline and a #phase companion; an expression of knobs does not', () => {
    const disciplines = (graph: unknown) => {
      const manifest = JSON.parse(run(['compile-kernel', write(`anchor-${Math.random().toString(36).slice(2)}`, graph)]).toString()) as KernelManifest
      return new Map(kernelParams(manifest).map((p) => [p.name, p]))
    }
    const straight = disciplines({ nodes: [
      { id: 'k', kind: 'knob', params: { value: 330 } },
      { id: 's', kind: 'saw', in: { freq: ['k'] } },
      { id: 'q', kind: 'sine', in: { freq: ['k'] } },
      { id: 'h', kind: 'harmonic', in: { freq: ['k'] } },
      { id: 'm', kind: 'mix', in: { in: ['s', 'q', 'h'] } }], out: 'm' })
    expect(JSON.stringify(straight.get('param:k.value'))).toContain('"anchor"')
    expect(JSON.stringify(straight.get('param:k.value'))).toContain('k.value#phase')
    expect(straight.has('param:k.value#phase')).toBe(true)
    const cone = disciplines({ nodes: [
      { id: 'k', kind: 'knob', params: { value: 330 } },
      { id: 'one', kind: 'knob', params: { value: 1 } },
      { id: 'c', kind: 'ring', in: { in: ['k', 'one'] } },
      { id: 's', kind: 'saw', in: { freq: ['c'] } }], out: 's' })
    expect(JSON.stringify(cone.get('param:k.value'))).toContain('"raw"')
    expect(cone.has('param:k.value#phase')).toBe(false)
  })
})
