/**
 * stdlib_literate.test.ts — the structural gate for literate stdlib programs.
 *
 * Every stdlib program is a literate markdown document: an `# <Name>` title,
 * prose, at least one mermaid signal-flow diagram, and exactly one
 * ```tropical code block holding the source. This suite keeps the document
 * shape honest:
 *
 *   - exactly one tropical code block (the loader's invariant, asserted
 *     here too so a violation names the file instead of failing the load)
 *   - at least one mermaid diagram
 *   - the H1 title matches the filename stem, which matches the parsed
 *     program name
 *
 * Deliberately not enforced: structured interface metadata. The signature
 * in the code block is the interface; the prose is the documentation. If
 * a machine consumer (MCP get_info, UI tooltips) wants a summary later,
 * the convention is: the first paragraph after the H1 is the summary —
 * derive it, don't author it twice.
 */

import { readdirSync, readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, test, expect } from 'bun:test'
import { extractMarkdown } from './markdown.js'
import { parseProgram } from './declarations.js'

const __dirname = dirname(fileURLToPath(import.meta.url))
const stdlibDir = join(__dirname, '../../stdlib')

const programFiles = readdirSync(stdlibDir)
  .filter(f => f.endsWith('.md') && f !== 'README.md')
  .sort()

describe('stdlib literate gate — title, diagram, single code block', () => {
  test('stdlib contains program documents', () => {
    expect(programFiles.length).toBeGreaterThan(0)
  })

  for (const file of programFiles) {
    test(file, () => {
      const stem = file.replace(/\.md$/, '')
      const text = readFileSync(join(stdlibDir, file), 'utf-8')

      // Exactly one tropical code block (the loader's invariant).
      const ext = extractMarkdown(text)
      expect(ext.blocks.length).toBe(1)

      // Code stays within 80 columns. The grammar is layout-free —
      // newlines are whitespace everywhere — so there is never a reason
      // for a line to run long.
      for (const [i, line] of ext.blocks[0].source.split('\n').entries()) {
        if (line.length > 80) {
          throw new Error(`${file}: code line ${i + 1} is ${line.length} chars (max 80)`)
        }
      }

      // At least one mermaid diagram.
      expect(/^```mermaid\s*$/m.test(text)).toBe(true)

      // The document opens with `# <Name>`, and the code block's program
      // declaration agrees with the filename.
      expect(text.split('\n')[0]).toBe(`# ${stem}`)
      const parsed = parseProgram(ext.blocks[0].source)
      expect(parsed.name).toBe(stem)
    })
  }
})
