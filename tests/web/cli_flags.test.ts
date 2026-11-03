/**
 * cli_flags.test.ts — the compiler front doors' shared contract.
 *
 * `tropicalc` (no runtime linked) and `diffcli` read their flags with one
 * parser (`Tropical.CliArgs`) and detect a playground graph with one detector
 * (`Tropical.Playground.isGraphDocument`). Checked here: a v3 hierarchy
 * document compiles through both binaries to the plan `render-graph` runs;
 * `--graph` forces the graph door on tropicalc as on diffcli; `--rate N` and
 * `--rate=N` are one flag on both, and `diffcli compile` honours it as
 * `tropicalc --emit=plan` does; an unknown flag or a malformed value is an
 * error on both.
 */
import { describe, test, expect, setDefaultTimeout } from 'bun:test'
import { spawnSync } from 'bun'
import { writeFileSync, readFileSync, mkdtempSync } from 'fs'
import { tmpdir } from 'os'
import { join, resolve, dirname } from 'path'
import { fileURLToPath } from 'url'

setDefaultTimeout(60_000)

const __dirname = dirname(fileURLToPath(import.meta.url))
const repoRoot = resolve(__dirname, '../..')
const bin = (name: string) => resolve(repoRoot, `lean/.lake/build/bin/${name}`)
const diffcli = process.env.TROPICAL_DIFFCLI ? resolve(process.env.TROPICAL_DIFFCLI) : bin('diffcli')
const tropicalc = bin('tropicalc')
const dir = mkdtempSync(join(tmpdir(), 'cli-flags-'))

function exec(binary: string, args: string[]) {
  const r = spawnSync([binary, ...args], { cwd: repoRoot })
  return { code: r.exitCode, out: r.stdout.toString(), err: r.stderr.toString() }
}
function ok(binary: string, args: string[]): string {
  const r = exec(binary, args)
  if (r.code !== 0) throw new Error(`${binary} ${args.join(' ')} failed: ${r.err}`)
  return r.out
}

const v3 = {
  version: 3,
  definitions: [{
    id: 'test.lowpass', version: 1, input: 'i', output: 'o',
    parameters: [{ name: 'cutoff', default: 900 }],
    nodes: [
      { id: 'i', kind: 'module_input' },
      { id: 'f', kind: 'filter', params: { resonance: 0.6 }, bindings: { cutoff: 'cutoff' }, in: { in: ['i'] } },
      { id: 'o', kind: 'module_output', in: { in: ['f'] } },
    ],
  }],
  scene: {
    nodes: [
      { id: 'saw', kind: 'saw', params: { freq: 110 } },
      { id: 'res', kind: 'resonator', params: { freq: 220, decay: 4 } },
      { id: 'lp', kind: 'module', definition: 'test.lowpass', definition_version: 1,
        params: { cutoff: 1200 }, in: { in: ['res'] } },
      { id: 'mix', kind: 'mix', in: { in: ['saw', 'lp'] } },
    ],
    out: 'mix',
  },
  taps: ['saw'],
}
const v3Path = join(dir, 'v3.json')
writeFileSync(v3Path, JSON.stringify(v3))
const patch = 'web/patches/pure-sine-440.json'

describe('a v3 hierarchy document through tropicalc and diffcli', () => {
  test('tropicalc --emit=plan ≡ diffcli compile ≡ the plan render-graph runs', () => {
    const pure = ok(tropicalc, [v3Path, '--emit=plan']).trim()
    expect(ok(diffcli, ['compile', v3Path]).trim()).toBe(pure)
    const dumped = join(dir, 'v3.plan.json')
    exec(diffcli, ['render-graph', v3Path, `--dump-plan=${dumped}`])
    expect(readFileSync(dumped, 'utf-8').trim()).toBe(pure)
    expect(JSON.parse(pure).output_channel_count).toBe(2)
  })

  test('tropicalc --emit=kernel ≡ diffcli compile-kernel at the same frames and rate', () => {
    // compile-kernel types the split kernel's audio plan; tropicalc types the
    // whole plan — so compare what the type says, not the nested plan.
    const shape = (m: string) => {
      const j = JSON.parse(m)
      return JSON.stringify({ rate: j.rate, inputs: j.inputs, outputs: j.outputs })
    }
    const pure = ok(tropicalc, [v3Path, '--emit=kernel', '--frames=128', '--rate', '48000'])
    const engine = ok(diffcli, ['compile-kernel', v3Path, '--frames', '128', '--rate=48000'])
    expect(JSON.parse(pure).rate).toBe(48000)
    expect(shape(pure)).toBe(shape(engine))
  })

  test('tropicalc --out writes the kernel artifact for a v3 document', () => {
    const out = join(dir, 'v3-artifact')
    ok(tropicalc, [v3Path, `--out=${out}`, '--frames=64'])
    expect(readFileSync(join(out, 'kernel.ll'), 'utf-8')).toContain('@tropical_kernel')
    expect(JSON.parse(readFileSync(join(out, 'manifest.json'), 'utf-8')).outputs[0].shape).toEqual([64, 2])
  })
})

describe('--graph forces the graph door', () => {
  // A graph that names a `schema` is not detected as one; --graph forces it.
  const forced = join(dir, 'forced.json')
  writeFileSync(forced, JSON.stringify({
    schema: 'not_a_patch', nodes: [{ id: 's', kind: 'saw', params: { freq: 110 } }], out: 's' }))
  const plain = join(dir, 'plain.json')
  writeFileSync(plain, JSON.stringify({ nodes: [{ id: 's', kind: 'saw', params: { freq: 110 } }], out: 's' }))

  test('tropicalc and diffcli compile take --graph, and compile it as the plain graph', () => {
    expect(exec(tropicalc, [forced, '--emit=plan']).code).not.toBe(0)
    expect(exec(diffcli, ['compile', forced]).code).not.toBe(0)
    const want = ok(tropicalc, [plain, '--emit=plan'])
    expect(ok(tropicalc, [forced, '--emit=plan', '--graph'])).toBe(want)
    expect(ok(diffcli, ['compile', forced, '--graph'])).toBe(want)
  })
})

describe('--rate', () => {
  test('both spellings, on both binaries, for a patch and a graph', () => {
    for (const src of [patch, v3Path]) {
      const spaced = ok(tropicalc, [src, '--emit=plan', '--rate', '48000'])
      expect(ok(tropicalc, [src, '--emit=plan', '--rate=48000'])).toBe(spaced)
      expect(JSON.parse(spaced).config.sampleRate).toBe(48000)
      // diffcli compile honours --rate exactly as tropicalc does
      expect(ok(diffcli, ['compile', src, '--rate', '48000'])).toBe(spaced)
      expect(ok(diffcli, ['compile', src, '--rate=48000'])).toBe(spaced)
      // and without it both say the default rate
      expect(JSON.parse(ok(diffcli, ['compile', src])).config.sampleRate).toBe(44100)
    }
  })

  test('a malformed or non-positive rate is an error, never ignored', () => {
    for (const bad of [['--rate=0'], ['--rate', '0'], ['--rate=-48000'], ['--rate=48k'], ['--rate=4.8e4'], ['--rate']]) {
      for (const [binary, args] of [
        [tropicalc, [patch, '--emit=plan', ...bad]],
        [diffcli, ['compile', patch, ...bad]],
        [diffcli, ['compile-kernel', patch, ...bad]],
        [diffcli, ['render-graph', v3Path, '--frames', '1', '--buffer', '8', ...bad]],
      ] as [string, string[]][]) {
        const r = exec(binary, args)
        expect(r.code).not.toBe(0)
        expect(r.err).toContain('--rate')
        expect(r.out).toBe('')
      }
    }
  })
})

describe('unknown flags are errors', () => {
  test('on every verb', () => {
    const cases: [string, string[]][] = [
      [tropicalc, [patch, '--emit=plan', '--rates=48000']],
      [tropicalc, [patch, '--emit=plan', '--metall']],
      [diffcli, ['compile', patch, '--fixture']],
      [diffcli, ['compile-kernel', patch, '--frame', '128']],
      [diffcli, ['compile-wasm', patch, `--out=${join(dir, 'x.wasm')}`, '--sample-rate=48000']],
      [diffcli, ['render-graph', v3Path, '--allchannels']],
      [diffcli, ['emit-ir', patch, '--rate=48000']],
    ]
    for (const [binary, args] of cases) {
      const r = exec(binary, args)
      expect(r.code).not.toBe(0)
      expect(r.err).toContain('unknown flag')
    }
    // a value flag given twice, and a switch given a value
    expect(exec(tropicalc, [patch, '--emit=plan', '--rate=48000', '--rate=44100']).err).toContain('given twice')
    expect(exec(diffcli, ['compile', patch, '--graph=yes']).err).toContain('takes no value')
  })
})
