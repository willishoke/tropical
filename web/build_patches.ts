/**
 * build_patches.ts — precompile the curated demo patches via the Lean engine.
 *
 * Reads web/patches/<slug>.json (tropical_program_2) + web/patches/manifest.json
 * (slug/title/description), compiles each to tropical_plan_5 with the Lean
 * `diffcli compile` CLI (the same engine that serves MCP), and writes
 * web/dist/patches/<slug>.plan.json + index.json. The browser fetches them and
 * runs only the WASM emitter + runtime — no TS compiler in the loop.
 *
 *   bun web/build_patches.ts
 */

import { writeFileSync, mkdirSync, readdirSync, unlinkSync, existsSync, readFileSync } from 'fs'
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
  if (f.endsWith('.plan.json')) unlinkSync(join(distDir, f))
}

const index: Array<Entry & { planPath: string }> = []
for (const { slug, title, description } of manifest) {
  const src = resolve(patchesDir, `${slug}.json`)
  const r = spawnSync([diffcli, 'compile', src, '--mode=fused'], { cwd: root })
  if (r.exitCode !== 0) {
    console.warn(`SKIP  ${slug}: diffcli compile failed\n${r.stderr.toString()}`)
    continue
  }
  const plan = r.stdout.toString().trim()
  writeFileSync(join(distDir, `${slug}.plan.json`), plan, 'utf-8')
  index.push({ slug, title, description, planPath: `patches/${slug}.plan.json` })
  console.log(`built ${slug.padEnd(22)} ${plan.length.toString().padStart(8)} bytes`)
}

writeFileSync(join(distDir, 'index.json'), JSON.stringify(index, null, 2), 'utf-8')
console.log(`\nmanifest: ${index.length} patches -> ${join(distDir, 'index.json')}`)
