/**
 * runtime_typed_ports.test.ts — the player's manifest reader and param writer
 * on typed ports (design/typed-kernel.md, "Typed ports").
 *
 *  1. A typed manifest reads: an `i32` Q16.16 param (its default is the word),
 *     an `i32` Q17.15 output (widened to `word · 2^-15`), the layout's 4-byte
 *     output cells, and `setParam`/`getParam` on the word.
 *  2. `encodeWord` is the kernel's conversion: truncate toward zero, saturate,
 *     NaN → 0, exact power-of-two scaling, a BigInt for an `i64` word.
 *  3. The reader refuses what it cannot bind (an `f16` param, a non-scalar
 *     param, a word with more fraction bits than bits, a default that is not
 *     a word it can hold exactly — an i64 past 2^53), and `ParamWriter`
 *     refuses a discipline other than `raw` on an integer port and a value
 *     outside an integer port's format.
 *  4. An existing f64 manifest reads exactly as before (f64 cells and output).
 *  5. A binary32 artifact (`--float f32`): f32 param cells and an f32 output
 *     (widened exactly), a binary32 default, writes rounded to binary32, a
 *     value past binary32 refused, a non-raw discipline on an f32 param
 *     refused; an unknown `realization.float`, a non-binary32 default and an
 *     f64 port in an f32 artifact refused by name.
 */
import { describe, test, expect, setDefaultTimeout } from 'bun:test'
import { spawnSync } from 'bun'
import { writeFileSync, readFileSync, mkdtempSync } from 'fs'
import { tmpdir } from 'os'
import { join, resolve, dirname } from 'path'
import { fileURLToPath } from 'url'
import {
  WasmKernel, ParamWriter, computeLayout, encodeWord, kernelParams, kernelFloatFormat,
  kernelOutputChannelCount, kernelOutputWordScale, type KernelManifest, type KernelPort,
} from '../../web/runtime/index'

setDefaultTimeout(120_000)

const __dirname = dirname(fileURLToPath(import.meta.url))
const repoRoot = resolve(__dirname, '../..')
const diffcli = process.env.TROPICAL_DIFFCLI
  ? resolve(process.env.TROPICAL_DIFFCLI)
  : resolve(repoRoot, 'lean/.lake/build/bin/diffcli')
const dir = mkdtempSync(join(tmpdir(), 'runtime-typed-ports-'))

function run(args: string[]): Buffer {
  const r = spawnSync([diffcli, ...args], { cwd: repoRoot })
  if (r.exitCode !== 0) throw new Error(`diffcli ${args[0]} failed: ${r.stderr.toString()}`)
  return r.stdout as Buffer
}

async function load(patch: unknown, float: 'f64' | 'f32' = 'f64'): Promise<{ k: WasmKernel; manifest: KernelManifest }> {
  const path = join(dir, `${Math.random().toString(36).slice(2)}.json`)
  writeFileSync(path, JSON.stringify(patch))
  const wasmPath = path.replace(/\.json$/, '.wasm')
  run(['compile-wasm', path, `--out=${wasmPath}`, '--float', float])
  const manifest = JSON.parse(run(['compile-kernel', path, '--float', float]).toString()) as KernelManifest
  return { k: await WasmKernel.instantiate(new Uint8Array(readFileSync(wasmPath)), manifest), manifest }
}

const knob = (hz: number) => ({
  schema: 'tropical_program_2', name: 'knob',
  body: { op: 'block', decls: [
    { op: 'paramDecl', name: 'freq', type: 'int32', frac_bits: 16, value: hz },
    { op: 'instanceDecl', name: 'lfo', program: 'SinOscQ15', inputs: { freq: { op: 'param', name: 'freq' } } },
  ], assigns: [{ op: 'outputAssign', name: 'dac.out', expr: { op: 'ref', instance: 'lfo', output: 'out' } }] },
})

const port = (elem: KernelPort['elem'], frac_bits?: number): KernelPort =>
  ({ name: 'param:x', elem, shape: [], role: 'param', frac_bits })

describe('typed ports in the web runtime', () => {
  test('an i32 Q16.16 param and an i32 Q17.15 output read, render and write', async () => {
    const { k, manifest } = await load(knob(2))
    const [freq] = kernelParams(manifest)
    expect(freq!.elem).toBe('i32')
    expect(freq!.frac_bits).toBe(16)
    expect(freq!.default).toBe(131072)
    expect(manifest.outputs[0]!.elem).toBe('i32')
    expect(manifest.outputs[0]!.frac_bits).toBe(15)
    expect(kernelOutputWordScale(manifest)).toBe(2 ** -15)
    const layout = computeLayout(manifest, 0)
    expect(layout.endByte - layout.output).toBe(Math.ceil((layout.frames * 4) / 8) * 8)
    expect(k.getParam(0)).toBe(2)
    const out = k.render(4096)
    // a Q1.15 sine widened to its real value: within ±32767/32768, on the 2⁻¹⁵ grid
    let peak = 0
    for (const v of out) {
      expect(Number.isInteger(v * 32768)).toBe(true)
      peak = Math.max(peak, Math.abs(v))
    }
    expect(peak).toBeLessThanOrEqual(32767 / 32768)
    expect(peak).toBeGreaterThan(0.1)
    k.setParam(0, 1234.567)
    expect(k.getParam(0)).toBe(Math.trunc(1234.567 * 65536) / 65536)
    const w = new ParamWriter(manifest)
    expect(w.check('param:freq')).toBeNull()
    w.write(k, 'param:freq', -440.25, k.frame)
    expect(k.getParam(0)).toBe(-440.25)
  })

  test('encodeWord is the kernel conversion', () => {
    const q16 = port('i32', 16)
    expect(encodeWord(q16, 2)).toBe(131072)
    expect(encodeWord(q16, -1.5)).toBe(-98304)
    expect(encodeWord(q16, 1 / 3)).toBe(21845)          // truncated, not rounded
    expect(encodeWord(q16, -1 / 3)).toBe(-21845)        // toward zero
    expect(encodeWord(q16, 40000)).toBe(2147483647)     // saturated
    expect(encodeWord(q16, -40000)).toBe(-2147483648)
    expect(encodeWord(q16, NaN)).toBe(0)
    expect(encodeWord(port('i32', 32), 0.25)).toBe(1073741824)
    expect(encodeWord(port('i64', 0), 3221225472)).toBe(3221225472n)
    expect(encodeWord(port('i64', 0), 1e30)).toBe((1n << 63n) - 1n)
    expect(encodeWord(port('i64', 32), -0.5)).toBe(-2147483648n)
  })

  test('the reader refuses what it cannot bind; the writer refuses non-raw on a word', () => {
    const base: KernelManifest = {
      schema: 'tropical_kernel_1', rate: 48000,
      inputs: [{ name: 'tau', elem: 'i64q16', shape: [16], role: 'signal' }, port('i32', 16)],
      outputs: [{ name: 'out', elem: 'i32', shape: [16, 1], frac_bits: 15 }],
    }
    expect(kernelOutputChannelCount(base)).toBe(1)
    const bad = (p: KernelPort) => ({ ...base, inputs: [base.inputs[0]!, p] })
    expect(() => kernelParams(bad({ ...port('f64'), elem: 'f16' as KernelPort['elem'] }))).toThrow(/binds f64, f32, i32 and i64 params/)
    expect(() => kernelParams(bad({ ...port('i32', 16), shape: [2] }))).toThrow(/not a scalar/)
    expect(() => kernelParams(bad(port('i32', 33)))).toThrow(/more fraction bits/)
    const anchored = bad({ ...port('i32', 16), discipline: { discipline: 'anchor', companions: ['x#phase'] } })
    expect(new ParamWriter(anchored).check('param:x')).toMatch(/integer word; discipline 'anchor' is not defined/)
    expect(() => kernelOutputChannelCount({ ...base, outputs: [{ name: 'out', elem: 'i64', shape: [16, 1] }] }))
      .toThrow(/f64, f32 or i32/)
    // a default word the JSON double cannot carry exactly is refused, not wrapped
    expect(() => kernelParams(bad({ ...port('i64', 0), default: 9223372036854775807 })))
      .toThrow(/past ±\(2\^53 − 1\)/)
    expect(() => kernelParams(bad({ ...port('i64', 0), default: 2 ** 53 }))).toThrow(/does not carry it exactly/)
    expect(kernelParams(bad({ ...port('i64', 0), default: 2 ** 53 - 1 }))[0]!.default).toBe(2 ** 53 - 1)
    expect(() => kernelParams(bad({ ...port('i32', 16), default: 2147483648 }))).toThrow(/not an i32 word/)
    expect(() => kernelParams(bad({ ...port('i32', 16), default: 0.5 }))).toThrow(/not an i32 word/)
  })

  test('the writer refuses a value outside an integer param\'s format', async () => {
    const { k, manifest } = await load(knob(2))
    const w = new ParamWriter(manifest)
    expect(() => w.write(k, 'param:freq', 100000, k.frame))
      .toThrow("param 'param:freq': value 100000 is outside the range of int32 Q16.16, [-32768, 32768)")
    expect(() => w.write(k, 'param:freq', -32768.5, k.frame)).toThrow(/outside the range/)
    expect(k.getParam(0)).toBe(2)                       // unchanged
    w.write(k, 'param:freq', -32768, k.frame)
    expect(k.getParam(0)).toBe(-32768)
  })

  test('an f64 manifest reads as before', async () => {
    const { k, manifest } = await load({
      schema: 'tropical_program_2', name: 'f64',
      body: { op: 'block', decls: [
        { op: 'paramDecl', name: 'f', value: 220 },
        { op: 'instanceDecl', name: 'o', program: 'FixedSinOsc', inputs: { freq: { op: 'param', name: 'f' } } },
      ], assigns: [{ op: 'outputAssign', name: 'dac.out', expr: { op: 'ref', instance: 'o', output: 'sine' } }] },
    })
    expect(kernelParams(manifest)[0]!.elem).toBe('f64')
    expect(kernelParams(manifest)[0]!.frac_bits).toBeUndefined()
    expect(kernelOutputWordScale(manifest)).toBeNull()
    expect(k.getParam(0)).toBe(220)
    k.setParam(0, 330.125)
    expect(k.getParam(0)).toBe(330.125)
  })

  test('a binary32 artifact reads, renders and writes f32 cells', async () => {
    const drive = (d: number) => ({
      schema: 'tropical_program_2', name: 'f32',
      body: { op: 'block', decls: [
        { op: 'paramDecl', name: 'drive', value: d },
        { op: 'instanceDecl', name: 'o', program: 'FixedSinOsc', inputs: { freq: 220 } },
        { op: 'instanceDecl', name: 's', program: 'SoftClip', inputs: {
          input: { op: 'ref', instance: 'o', output: 'sine' }, drive: { op: 'param', name: 'drive' } } },
      ], assigns: [{ op: 'outputAssign', name: 'dac.out', expr: { op: 'ref', instance: 's', output: 'out' } }] },
    })
    const { k, manifest } = await load(drive(0.1), 'f32')
    expect(kernelFloatFormat(manifest)).toBe('f32')
    const [d] = kernelParams(manifest)
    expect(d!.elem).toBe('f32')
    expect(d!.default).toBe(Math.fround(0.1))       // the binary32 itself, written exactly
    expect(manifest.outputs[0]!.elem).toBe('f32')
    expect(kernelOutputWordScale(manifest)).toBeNull()
    const layout = computeLayout(manifest, 0)
    expect(layout.endByte - layout.output).toBe(Math.ceil((layout.frames * 4) / 8) * 8)
    expect(k.getParam(0)).toBe(Math.fround(0.1))
    const out = k.render(1024)
    for (const v of out) expect(Math.fround(v)).toBe(v)
    expect(out.some((v) => Math.abs(v) > 0.05)).toBe(true)
    k.setParam(0, 1 / 3)
    expect(k.getParam(0)).toBe(Math.fround(1 / 3))
    // a written value ≡ the artifact compiled at it
    const fresh = await load(drive(2.5), 'f32')
    const w = new ParamWriter(manifest)
    w.write(k, 'param:drive', 2.5, k.frame)
    k.seek(0)
    const a = k.render(512), b = fresh.k.render(512)
    for (let i = 0; i < a.length; i++) expect(a[i]).toBe(b[i]!)
    expect(() => w.write(k, 'param:drive', 1e39, k.frame)).toThrow(/outside the range of binary32/)
    expect(k.getParam(0)).toBe(2.5)
  })

  test('the reader refuses an unknown float format and a port at another format', () => {
    const f32: KernelManifest = {
      schema: 'tropical_kernel_1', rate: 48000, realization: { mode: 'fused', float: 'f32' },
      inputs: [{ name: 'tau', elem: 'i64q16', shape: [16], role: 'signal' },
               { name: 'param:x', elem: 'f32', shape: [], role: 'param', default: 0.5 }],
      outputs: [{ name: 'out', elem: 'f32', shape: [16, 1] }],
    }
    expect(kernelParams(f32)[0]!.default).toBe(0.5)
    expect(kernelOutputChannelCount(f32)).toBe(1)
    expect(() => kernelParams({ ...f32, realization: { mode: 'fused', float: 'f16' } }))
      .toThrow("KernelManifest: unknown float format 'f16'; this host reads f64 and f32 artifacts")
    expect(() => kernelOutputChannelCount({ ...f32, outputs: [{ name: 'out', elem: 'f64', shape: [16, 1] }] }))
      .toThrow("KernelManifest: port 'out' is f64 in an artifact whose float format is f32")
    expect(() => kernelParams({ ...f32, realization: undefined }))
      .toThrow("port 'param:x' is f32 in an artifact whose float format is f64")
    expect(() => kernelParams({ ...f32, inputs: [f32.inputs[0]!, { ...f32.inputs[1]!, default: 0.1 }] }))
      .toThrow(/default 0.1 is not a binary32 value/)
    // f32 with a non-raw discipline is refused once, at load: on a port, and
    // in the plan's table (a glided knob has no port of its own)
    const anchored = { ...f32, inputs: [f32.inputs[0]!,
      { ...f32.inputs[1]!, discipline: { discipline: 'anchor', companions: ['x#phase'] } }] }
    expect(() => kernelParams(anchored)).toThrow("param 'param:x' has the 'anchor' discipline in a binary32 artifact")
    expect(() => new ParamWriter(anchored)).toThrow(/binary32 artifact's params are written raw/)
    const glided = { ...f32, plan: { param_disciplines: [
      { name: 'k', discipline: 'glide', companions: ['k#v0', 'k#v1', 'k#t0'] }] } }
    expect(() => kernelParams(glided)).toThrow("param 'k' has the 'glide' discipline in a binary32 artifact")
    expect(kernelParams({ ...glided, realization: { mode: 'fused', float: 'f64' },
      inputs: [f32.inputs[0]!, { ...f32.inputs[1]!, elem: 'f64' }],
      outputs: [{ name: 'out', elem: 'f64', shape: [16, 1] }] })).toHaveLength(1)
  })

  test('a module and a manifest compiled at different --float are refused', async () => {
    const patch = {
      schema: 'tropical_program_2', name: 'mismatch',
      body: { op: 'block', decls: [
        { op: 'instanceDecl', name: 'o', program: 'FixedSinOsc', inputs: { freq: 220 } },
      ], assigns: [{ op: 'outputAssign', name: 'dac.out', expr: { op: 'ref', instance: 'o', output: 'sine' } }] },
    }
    const path = join(dir, 'mismatch.json')
    writeFileSync(path, JSON.stringify(patch))
    for (const [wasmFloat, manifestFloat] of [['f64', 'f32'], ['f32', 'f64']] as const) {
      const wasmPath = join(dir, `mismatch-${wasmFloat}.wasm`)
      run(['compile-wasm', path, `--out=${wasmPath}`, '--float', wasmFloat])
      const manifest = JSON.parse(run(['compile-kernel', path, '--float', manifestFloat]).toString()) as KernelManifest
      await expect(WasmKernel.instantiate(new Uint8Array(readFileSync(wasmPath)), manifest))
        .rejects.toThrow(`the module is compiled at ${wasmFloat} but the manifest's float format is ${manifestFloat}`)
    }
  })
})
