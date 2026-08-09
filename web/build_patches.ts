/**
 * build_patches.ts — precompile the curated demo patches via the Lean engine.
 *
 * Reads web/patches/<slug>.json (tropical_program_2) + web/patches/manifest.json
 * (slug/title/description). For each patch it lowers the Lean IR to a complete
 * wasm32 module in-process via `diffcli compile-wasm` (engine LLVM + lld, no
 * toolchain), and writes web/dist/patches/<slug>.wasm plus a trimmed
 * <slug>.manifest.json (the runtime's consumption contract). The browser
 * fetches the .wasm + manifest and instantiates — no emitter, no compiler.
 *
 * Requires libtropical built with TROPICAL_WASM_EMIT (cmake -DTROPICAL_WASM_EMIT=ON).
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
const diffcli = resolve(root, 'lean/.lake/build/bin/diffcli')

type Entry = { slug: string; title: string; description: string }

// Ensure the Lean compiler CLI exists (incremental; fast if already built).
if (!existsSync(diffcli)) {
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
for (const { slug, title, description } of manifest) {
  const src = resolve(patchesDir, `${slug}.json`)

  // 1. Compile to plan_6 — the source of the runtime's manifest fields.
  const c = spawnSync([diffcli, 'compile', src, '--mode=fused'], { cwd: root })
  if (c.exitCode !== 0) {
    console.warn(`SKIP  ${slug}: diffcli compile failed\n${c.stderr.toString()}`)
    continue
  }
  const plan = JSON.parse(c.stdout.toString())

  // 2. Lower the same source to a wasm32 module in-process (Lean IR → engine
  //    LLVM + lld). The wasm is the *only* codegen artifact the browser runs.
  const wasmPath = join(distDir, `${slug}.wasm`)
  const w = spawnSync([diffcli, 'compile-wasm', src, `--out=${wasmPath}`], { cwd: root })
  if (w.exitCode !== 0) {
    console.warn(`SKIP  ${slug}: diffcli compile-wasm failed\n${w.stderr.toString()}`)
    continue
  }

  // 3. Trim plan_6 down to the KernelManifest the runtime consumes.
  writeFileSync(join(distDir, `${slug}.manifest.json`), JSON.stringify({
    sampleRate:     plan.config?.sampleRate ?? 44100,
    registerCount:  plan.register_count ?? 0,
    arraySlotSizes: plan.array_slot_sizes ?? [],
    slotCount:      plan.slot_count ?? (plan.slot_defaults?.length ?? 0),
    slotDefaults:   plan.slot_defaults ?? [],
  }), 'utf-8')

  index.push({ slug, title, description, wasmPath: `patches/${slug}.wasm`, manifestPath: `patches/${slug}.manifest.json` })
  console.log(`built ${slug.padEnd(22)} ${statSync(wasmPath).size.toString().padStart(7)} B wasm`)
}

writeFileSync(join(distDir, 'index.json'), JSON.stringify(index, null, 2), 'utf-8')
console.log(`\nmanifest: ${index.length} patches -> ${join(distDir, 'index.json')}`)
