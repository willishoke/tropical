/**
 * build.ts — bundles the web demo into web/dist/.
 *
 *   bun web/build.ts
 *
 * Steps:
 *  1. (re-)precompile patches via the Lean engine — web/build_patches.ts
 *  2. Bundle the AudioWorklet processor           — single self-contained file
 *  3. Bundle the main-thread app                  — app.js
 *  4. Copy index.html into dist/
 */

import { cpSync, existsSync, mkdirSync } from 'fs'
import { dirname, resolve } from 'path'
import { fileURLToPath } from 'url'
import { spawnSync } from 'bun'

const __dirname = dirname(fileURLToPath(import.meta.url))
const root = resolve(__dirname, '..')
const distDir = resolve(__dirname, 'dist')

function run(cmd: string[], desc: string): void {
  console.log(`▸ ${desc}`)
  const r = spawnSync(cmd, { cwd: root, stdout: 'inherit', stderr: 'inherit' })
  if (r.exitCode !== 0) {
    throw new Error(`${desc} failed (exit ${r.exitCode})`)
  }
}

mkdirSync(distDir, { recursive: true })

run(['bun', 'web/build_patches.ts'], 'precompile patches (Lean engine)')

run(
  ['bun', 'build', 'web/worklet/processor.ts', '--target=browser', '--outfile', `${distDir}/worklet.js`, '--format', 'esm'],
  'bundle AudioWorklet',
)

run(
  ['bun', 'build', 'web/site/app.ts', '--target=browser', '--outfile', `${distDir}/app.js`, '--format', 'esm'],
  'bundle main-thread app',
)

cpSync(resolve(__dirname, 'site/index.html'), resolve(distDir, 'index.html'))
console.log(`▸ copied index.html`)

if (!existsSync(resolve(distDir, 'patches', 'index.json'))) {
  throw new Error('patches/index.json missing; check build_patches.ts output')
}

console.log(`\n✓ build complete → ${distDir}`)
console.log(`  serve with: bun web/dev.ts  (or any static server with COEP/COOP headers)`)
