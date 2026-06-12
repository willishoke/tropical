/**
 * parse_cmd.ts — TS side of the surface-parse differential, for fixtures the
 * stdlib corpus doesn't cover (iterate/chain/map2/zipWith, number forms,
 * nested programs). Mirrors what `build_parsed_stdlib.ts` does per file:
 * extract the tropical block and run `parseProgram`, printing the
 * ParsedProgram JSON. Invoked as `bun run scripts/diff/parse_cmd.ts <file.md>`.
 */

import { readFileSync } from 'node:fs'
import { extractMarkdown } from '../../compiler/parse/markdown.js'
import { parseProgram } from '../../compiler/parse/declarations.js'

const file = process.argv[2]
if (!file) {
  console.error('usage: bun run scripts/diff/parse_cmd.ts <file.md>')
  process.exit(1)
}
const ext = extractMarkdown(readFileSync(file, 'utf-8'))
if (ext.blocks.length === 0) {
  console.log(JSON.stringify({ error: 'no tropical code block found' }))
  process.exit(0)
}
try {
  console.log(JSON.stringify(parseProgram(ext.blocks[0].source)))
} catch (e) {
  console.log(JSON.stringify({ error: (e as Error).message }))
}
