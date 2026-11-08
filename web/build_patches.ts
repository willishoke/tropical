/**
 * build_patches.ts — precompile the curated demo patches via the Lean engine.
 *
 * Reads web/patches/<slug>.json (tropical_program_2) + web/patches/manifest.json
 * (slug/title/description). For each patch it lowers the Lean IR to a complete
 * wasm32 module in-process via `diffcli compile-wasm` (libtropical_wasm's
 * LLVM + lld, no external toolchain), and writes web/dist/patches/<slug>.wasm
 * plus <slug>.manifest.json, the kernel's `tropical_kernel_1` type (the
 * runtime's consumption contract). The browser fetches the .wasm + manifest
 * and instantiates — no emitter, no compiler.
 *
 * Requires libtropical_wasm built with TROPICAL_WASM_EMIT (cmake -DTROPICAL_WASM_EMIT=ON).
 *
 *   bun web/build_patches.ts
 */

import { writeFileSync, mkdirSync, readdirSync, unlinkSync, existsSync, readFileSync, statSync } from 'fs'
import { dirname, resolve, join } from 'path'
import { fileURLToPath } from 'url'
import { spawnSync } from 'bun'

const __dirname = dirname(fileURLToPath(import.meta.url))
const root = resolve(__dirname, '..')
const patchesDir = resolve(__dirname, 'patches')
const distDir = resolve(__dirname, 'dist/patches')
// TROPICAL_DIFFCLI points at a prebuilt diffcli (e.g. a worktree with no Lean
// build of its own). Run the binary directly — never `lake exe` (CLAUDE.md).
const diffcli = process.env.TROPICAL_DIFFCLI
  ? resolve(process.env.TROPICAL_DIFFCLI)
  : resolve(root, 'lean/.lake/build/bin/diffcli')

/** A knob the player shows for a patch: a `param:` slot (named without the
 *  prefix) and its slider range. The player writes it under the discipline
 *  the kernel manifest declares; only raw and anchor params are playable. */
type Knob = { param: string; label: string; min: number; max: number; log?: boolean; unit?: string }
type Entry = { slug: string; title: string; description: string; knobs?: Knob[] }

// Ensure the Lean compiler CLI exists (incremental; fast if already built).
if (!process.env.TROPICAL_DIFFCLI && !existsSync(diffcli)) {
  const b = spawnSync(['sh', '-c', `cd "${root}/lean" && PATH="$HOME/.elan/bin:$PATH" lake build diffcli`], {
    stdout: 'inherit', stderr: 'inherit',
  })
  if (b.exitCode !== 0 || !existsSync(diffcli)) throw new Error('failed to build lean diffcli')
}

const manifest = JSON.parse(readFileSync(resolve(patchesDir, 'manifest.json'), 'utf-8')) as Entry[]

mkdirSync(distDir, { recursive: true })
for (const f of readdirSync(distDir)) {
  if (f.endsWith('.plan.json') || f.endsWith('.wasm') || f.endsWith('.manifest.json')) {
    unlinkSync(join(distDir, f))
  }
}

const index: Array<Entry & { wasmPath: string; manifestPath: string }> = []
for (const { slug, title, description, knobs } of manifest) {
  const src = resolve(patchesDir, `${slug}.json`)

  // 1. Lower the source to a wasm32 module in-process (Lean IR → engine
  //    LLVM + lld), the port kernel for 128 frames per call. The wasm is the
  //    *only* codegen artifact the browser runs.
  const wasmPath = join(distDir, `${slug}.wasm`)
  const w = spawnSync([diffcli, 'compile-wasm', src, `--out=${wasmPath}`, '--frames', '128'], { cwd: root })
  if (w.exitCode !== 0) {
    console.warn(`SKIP  ${slug}: diffcli compile-wasm failed\n${w.stderr.toString()}`)
    continue
  }

  // 3. The runtime's manifest: the kernel's type (`tropical_kernel_1`) for
  //    the worklet's 128-frame quantum — the same frames `compile-wasm`
  //    compiled the module for.
  const k = spawnSync([diffcli, 'compile-kernel', src, '--frames', '128'], { cwd: root })
  if (k.exitCode !== 0) {
    console.warn(`SKIP  ${slug}: diffcli compile-kernel failed\n${k.stderr.toString()}`)
    continue
  }
  writeFileSync(join(distDir, `${slug}.manifest.json`), k.stdout.toString(), 'utf-8')

  // Every declared knob must name a param input the player can write: a
  // knob that names nothing, or a discipline the player does not implement,
  // fails the build rather than becoming a dead slider.
  const kernelType = JSON.parse(k.stdout.toString()) as {
    inputs: { name: string; discipline?: { discipline: string } }[]
  }
  for (const knob of knobs ?? []) {
    const port = kernelType.inputs.find((p) => p.name === `param:${knob.param}`)
    if (!port) throw new Error(`${slug}: knob '${knob.label}' names param '${knob.param}', which the kernel does not have`)
    const d = port.discipline?.discipline ?? 'raw'
    if (d !== 'raw' && d !== 'anchor')
      throw new Error(`${slug}: knob '${knob.label}' (${knob.param}) has discipline '${d}'; the player writes raw and anchor params only`)
    if (!(knob.min < knob.max) || (knob.log && knob.min <= 0))
      throw new Error(`${slug}: knob '${knob.label}' has a bad range [${knob.min}, ${knob.max}]`)
  }

  index.push({ slug, title, description, ...(knobs ? { knobs } : {}), wasmPath: `patches/${slug}.wasm`, manifestPath: `patches/${slug}.manifest.json` })
  console.log(`built ${slug.padEnd(22)} ${statSync(wasmPath).size.toString().padStart(7)} B wasm`)
}

writeFileSync(join(distDir, 'index.json'), JSON.stringify(index, null, 2), 'utf-8')
console.log(`\nmanifest: ${index.length} patches -> ${join(distDir, 'index.json')}`)
