/**
 * markdown.ts — extract `tropical` code blocks from a literate `.trop` file.
 *
 * A `.trop` file is a markdown document. The compiler reads only the fenced
 * code blocks tagged ```` ```tropical ````; prose, headings, and other
 * fenced blocks (e.g. mermaid) are ignored. Each extracted block carries
 * its line offset so error spans can be mapped back to the source file.
 *
 * Scope of recognition:
 *   - Backtick fences (3 or more backticks). Tilde fences (`~~~`) are not
 *     supported.
 *   - The info string after the opening fence is split on whitespace; the
 *     first word is the language tag. Only `tropical` is extracted.
 *   - Fences must appear at column 0 (no indented fence support; matches
 *     the markdown spec for indented code blocks not being fenced).
 *   - A closing fence is a line starting with N or more backticks, where N
 *     is the opening fence's count, and nothing else after the backticks
 *     (whitespace allowed).
 *
 * Anything outside a recognized tropical block (prose, mermaid blocks,
 * other languages) is preserved in `rest` for the pretty-printer's
 * format-preserving save path. MVP loaders ignore `rest`.
 */

export interface CodeBlock {
  /** Source text of the code block (between the fences, no fence lines). */
  source: string
  /** 0-indexed line number in the original `.trop` file where the block's
   *  first line of content sits — i.e., one line after the opening fence.
   *  Use this to translate parser errors back to file:line:col. */
  lineOffset: number
}

export interface MarkdownExtraction {
  /** All `tropical`-tagged code blocks, in source order. */
  blocks: CodeBlock[]
  /** The original source split into lines. Out-of-block lines are kept
   *  verbatim (matching markdown's view of them as prose); in-block lines
   *  appear here too — the printer can reconstruct the exact file by
   *  consulting `blocks` for canonical content and using these lines for
   *  prose between blocks. MVP loaders may ignore this. */
  lines: string[]
}

const FENCE_LINE_RE = /^(`{3,})\s*([^\s]*)\s*(.*?)\s*$/
const CLOSING_FENCE_RE = /^(`{3,})\s*$/

/** Parse a `.trop` document, extracting tropical-tagged code blocks. */
export function extractMarkdown(source: string): MarkdownExtraction {
  const lines = source.split('\n')
  const blocks: CodeBlock[] = []

  let i = 0
  while (i < lines.length) {
    const line = lines[i]
    const open = FENCE_LINE_RE.exec(line)
    if (!open) { i++; continue }

    const fence = open[1]
    const lang = open[2]
    const isTropical = lang === 'tropical'

    // Find the matching closing fence: a line starting with at least
    // `fence.length` backticks and nothing else (whitespace allowed).
    let close = -1
    for (let j = i + 1; j < lines.length; j++) {
      const m = CLOSING_FENCE_RE.exec(lines[j])
      if (m && m[1].length >= fence.length) { close = j; break }
    }

    if (close === -1) {
      // Unterminated fence: per CommonMark spec, an unterminated fenced
      // block extends to the end of the document. We adopt that policy
      // for tropical blocks too — the body parser will report missing
      // closing constructs from inside the block.
      close = lines.length
    }

    if (isTropical) {
      const body = lines.slice(i + 1, close).join('\n')
      blocks.push({ source: body, lineOffset: i + 1 })
    }
    // For non-tropical fenced blocks (mermaid, etc.), skip past them
    // without extraction; they remain in `lines` for round-tripping.

    i = close + 1
  }

  return { blocks, lines }
}

/** Convenience: concatenate all tropical blocks with line-tracking comments
 *  preserved. The single string is suitable for handing to the lexer; line
 *  numbers in the resulting tokens map back via `blockOffsetForLine`.
 *
 *  Currently each block becomes a separate parsed unit; this helper exists
 *  for forthcoming sub-phases where the elaborator needs a unified view. */
export function joinBlocks(extraction: MarkdownExtraction): string {
  return extraction.blocks.map(b => b.source).join('\n\n')
}
