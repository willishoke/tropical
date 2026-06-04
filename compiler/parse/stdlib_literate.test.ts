/**
 * stdlib_literate.test.ts — the legibility gate for literate stdlib programs.
 *
 * Every stdlib program is a literate markdown document: YAML frontmatter
 * describing the interface, prose + mermaid describing the plumbing, and
 * exactly one ```tropical code block holding the source. This suite keeps
 * the human/machine-facing layer honest against the code:
 *
 *   - frontmatter exists and parses as YAML
 *   - `program` matches both the filename stem and the parsed program name
 *   - frontmatter inputs/outputs match the parsed ports exactly, in order
 *   - at least one mermaid diagram is present
 *   - exactly one tropical code block (the loader's invariant, asserted
 *     here too so a violation names the file instead of failing the load)
 *
 * A failure here means the documentation lies about the program — fix the
 * frontmatter, not the test.
 */

import { readdirSync, readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, test, expect } from 'bun:test'
import { parse as parseYaml } from 'yaml'
import { extractMarkdown, splitFrontmatter } from './markdown.js'
import { parseProgram } from './declarations.js'

const __dirname = dirname(fileURLToPath(import.meta.url))
const stdlibDir = join(__dirname, '../../stdlib')

const programFiles = readdirSync(stdlibDir)
  .filter(f => f.endsWith('.md') && f !== 'README.md')
  .sort()

interface FrontmatterPort {
  name: string
  type?: string
  default?: unknown
  description: string
}

interface Frontmatter {
  program: string
  summary: string
  inputs: FrontmatterPort[]
  outputs: FrontmatterPort[]
  state?: Array<{ name: string; description: string }>
  uses?: string[]
  type_params?: Array<{ name: string; type?: string; default?: unknown; description: string }>
  breaks_cycles?: boolean
}

describe('stdlib literate gate — frontmatter, diagrams, single code block', () => {
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

      // At least one mermaid diagram.
      expect(/^```mermaid\s*$/m.test(text)).toBe(true)

      // Frontmatter exists and parses.
      const fmText = splitFrontmatter(text)
      expect(fmText).not.toBeNull()
      const fm = parseYaml(fmText!) as Frontmatter
      expect(fm.program).toBe(stem)
      expect(typeof fm.summary).toBe('string')
      expect(fm.summary.length).toBeGreaterThan(0)

      // The code block parses, and the documented interface matches the
      // real one exactly (names, order).
      const parsed = parseProgram(ext.blocks[0].source)
      expect(parsed.name).toBe(stem)

      // ProgramPort is `string | ProgramPortSpec` — bare un-annotated
      // ports parse as plain strings.
      const portName = (p: string | { name: string }) =>
        typeof p === 'string' ? p : p.name
      const fmInputs = (fm.inputs ?? []).map(p => p.name)
      const fmOutputs = (fm.outputs ?? []).map(p => p.name)
      const declaredInputs = (parsed.ports?.inputs ?? []).map(portName)
      const declaredOutputs = (parsed.ports?.outputs ?? []).map(portName)
      expect(fmInputs).toEqual(declaredInputs)
      expect(fmOutputs).toEqual(declaredOutputs)

      // Every documented port carries a non-empty description.
      for (const p of [...(fm.inputs ?? []), ...(fm.outputs ?? [])]) {
        expect(typeof p.description).toBe('string')
        expect(p.description.length).toBeGreaterThan(0)
      }
    })
  }
})
