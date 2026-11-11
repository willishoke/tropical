/**
 * embedded_object.test.ts — `diffcli compile-object` at a float format per
 * target (design/typed-kernel.md, "The float format per target").
 *
 *  1. The Cortex-M4F fixture (`tests/fixtures/embedded_f32_softclip.json`, an
 *     int32 LFO into a float `SoftClip` with a float param) compiles for
 *     `thumbv7em-none-eabihf` at `--float f32` with no libcalls; its manifest
 *     declares f32 ports, the FPU (binary32, no binary64) and the hard-float
 *     ABI.
 *  2. The same program as a host object and a wasm32 object at f32.
 *  3. Refusals, by name, with no object written: `--float f32` on the
 *     FPU-less M0+; a float program at f64 on the M4F (soft binary64); a
 *     program whose `int` → binary32 conversion is a libcall on the M4F.
 *     `--allow-libcalls` writes that object and lists the routine.
 */
import { describe, test, expect, setDefaultTimeout } from 'bun:test'
import { spawnSync } from 'bun'
import { existsSync, readFileSync, mkdtempSync } from 'fs'
import { tmpdir } from 'os'
import { join, resolve, dirname } from 'path'
import { fileURLToPath } from 'url'

setDefaultTimeout(60_000)

const __dirname = dirname(fileURLToPath(import.meta.url))
const repoRoot = resolve(__dirname, '../..')
const diffcli = process.env.TROPICAL_DIFFCLI
  ? resolve(process.env.TROPICAL_DIFFCLI)
  : resolve(repoRoot, 'lean/.lake/build/bin/diffcli')
const dir = mkdtempSync(join(tmpdir(), 'embedded-object-'))
const fixture = 'tests/fixtures/embedded_f32_softclip.json'
const M4F = ['--triple', 'thumbv7em-none-eabihf', '--cpu', 'cortex-m4']
const M0P = ['--triple', 'thumbv6m-none-eabi', '--cpu', 'cortex-m0plus']

function compileObject(patch: string, out: string, args: string[]) {
  const r = spawnSync([diffcli, 'compile-object', patch, '--out', out, '--frames', '16',
    '--rate', '48000', ...args], { cwd: repoRoot })
  return { code: r.exitCode, err: r.stderr.toString() }
}

function manifestOf(out: string): any {
  return JSON.parse(readFileSync(`${out}.manifest.json`, 'utf8'))
}

describe('compile-object at a float format per target', () => {
  test('the M4F fixture compiles at f32 with no libcalls', () => {
    const out = join(dir, 'm4f.o')
    const r = compileObject(fixture, out, [...M4F, '--float', 'f32'])
    expect(r.err).toContain('compile-object: wrote')
    expect(r.code).toBe(0)
    const m = manifestOf(out)
    expect(m.realization.float).toBe('f32')
    expect(m.object.triple).toBe('thumbv7em-none-eabihf')
    expect(m.object.float).toBe('f32')
    expect(m.object.libcalls).toEqual([])
    expect(m.object.float_hardware).toEqual({ binary32: true, binary64: false })
    expect(m.object.float_abi).toBe('hard')
    expect(m.object.numerics.float).toBe('ieee754-binary32')
    const ports = Object.fromEntries(m.object.ports.map((p: any) => [p.name, p]))
    expect(ports['param:drive'].elem).toBe('f32')
    expect(ports['param:drive'].bytes).toBe(4)
    expect(ports['param:freq'].format).toBe('Q16.16')
    expect(ports.out.elem).toBe('f32')
    expect(ports.out.format).toBe('ieee754-binary32')
  })

  test('the same program as a host object and a wasm32 object at f32', () => {
    for (const triple of ['', 'wasm32-unknown-unknown']) {
      const out = join(dir, `other-${triple || 'host'}.o`)
      const r = compileObject(fixture, out, [...(triple ? ['--triple', triple] : []), '--float', 'f32'])
      expect(r.code).toBe(0)
      expect(manifestOf(out).object.libcalls).toEqual([])
    }
  })

  test('--float f32 on the FPU-less M0+ is refused', () => {
    const out = join(dir, 'm0p-f32.o')
    const r = compileObject('tests/fixtures/embedded_lfo_knob.json', out, [...M0P, '--float', 'f32'])
    expect(r.code).toBe(1)
    expect(r.err).toContain('thumbv6m-none-eabi (cpu cortex-m0plus) has no binary32 FPU')
    expect(existsSync(out)).toBe(false)
  })

  test('a float program at f64 on the M4F is refused: binary64 is soft there', () => {
    const out = join(dir, 'm4f-f64.o')
    const r = compileObject(fixture, out, M4F)
    expect(r.code).toBe(1)
    expect(r.err).toContain('binary64 arithmetic on a target with no binary64 FPU')
    expect(existsSync(out)).toBe(false)
  })

  test('an int → binary32 conversion is a named libcall on the M4F; --allow-libcalls lists it', () => {
    const out = join(dir, 'm4f-sine.o')
    const sine = 'web/patches/pure-sine-440.json'
    const r = compileObject(sine, out, [...M4F, '--float', 'f32'])
    expect(r.code).toBe(1)
    expect(r.err).toContain('an `int` (64-bit) to binary32 conversion')
    expect(existsSync(out)).toBe(false)
    const allowed = compileObject(sine, out, [...M4F, '--float', 'f32', '--allow-libcalls'])
    expect(allowed.code).toBe(0)
    const libcalls: string[] = manifestOf(out).object.libcalls
    expect(libcalls.length).toBeGreaterThan(0)
    expect(libcalls.some((s) => /floatdisf|floatundisf|l2f/.test(s))).toBe(true)
  })
})
