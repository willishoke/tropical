/**
 * convert_stdlib.ts — one-shot CLI: stdlib/*.json → stdlib/*.trop.
 *
 * For each stdlib JSON file:
 *   read → JSON.parse → normalizeProgramFile → ProgramNode
 *        → raiseProgram → ParsedProgram
 *        → printProgram → markdown text
 *        → write to stdlib/<name>.trop
 *
 * This is a migration utility, not a runtime path. Run via Bun:
 *
 *   bun run compiler/parse/convert_stdlib.ts
 *
 * Emits one confirmation line per file to stderr; stdout is left clean
 * for pipe-friendly behaviour. Exits non-zero on any failure, printing
 * the offending file path.
 */

import { readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { normalizeProgramFile } from '../session.js'
import { raiseProgram } from './raise.js'
import { printProgram } from './print.js'

function runConvert(): void {
  const __dirname = dirname(fileURLToPath(import.meta.url))
  const stdlibDir = join(__dirname, '../../stdlib')
  const files = readdirSync(stdlibDir).filter(f => f.endsWith('.json')).sort()

  if (files.length === 0) {
    process.stderr.write(`convert_stdlib: no .json files found in ${stdlibDir}\n`)
    process.exit(1)
  }

  for (const file of files) {
    const inPath = join(stdlibDir, file)
    try {
      const raw = JSON.parse(readFileSync(inPath, 'utf-8')) as { schema?: string; [k: string]: unknown }
      const { node } = normalizeProgramFile(raw)
      const parsed = raiseProgram(node)
      const out = printProgram(parsed)

      const outName = file.replace(/\.json$/, '.trop')
      const outPath = join(stdlibDir, outName)
      writeFileSync(outPath, out, 'utf-8')

      const size = statSync(outPath).size
      process.stderr.write(`${file} → ${outName} (${formatSize(size)})\n`)
    } catch (err) {
      process.stderr.write(`convert_stdlib: failed on ${inPath}\n`)
      process.stderr.write(`  ${(err as Error).message}\n`)
      process.exit(1)
    }
  }
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  return `${(bytes / 1024).toFixed(1)} KB`
}

// `import.meta.main` is Bun's "is this the entry script?" flag. Cast
// through `unknown` to satisfy plain ES2022 typings; safe because the
// script is only invoked under Bun.
if ((import.meta as unknown as { main?: boolean }).main === true) {
  runConvert()
}
