/**
 * raise_cmd.ts — the TS side of the raise differential gate.
 *
 * Two verbs, mirroring `lean/.lake/build/bin/diffcli`:
 *
 *   raise <file.json>             tropical_program_2 file →
 *                                 normalizeProgramFile + raiseProgram →
 *                                 `{program, params?, audio_outputs?}` on
 *                                 stdout, or `{error: <message>}` when the
 *                                 ingest path throws (the error string is
 *                                 part of the gated contract — the Lean
 *                                 port must reproduce it).
 *
 *   parsed-roundtrip <file.json>  serialized ParsedProgram JSON echoed
 *                                 verbatim (parse + re-stringify). The
 *                                 Lean side decodes into its typed AST and
 *                                 re-encodes; agreement gates the codec.
 *
 * Usage:  bun run scripts/diff/raise_cmd.ts <verb> <file.json>
 */

import { readFileSync } from 'node:fs'
import { normalizeProgramFile } from '../../compiler/session.js'
import { raiseProgram } from '../../compiler/parse/raise.js'

function main(argv: string[]): number {
  const [verb, file] = argv
  if ((verb !== 'raise' && verb !== 'parsed-roundtrip') || file === undefined) {
    console.error('usage: raise_cmd.ts raise|parsed-roundtrip <file.json>')
    return 1
  }
  const text = readFileSync(file, 'utf-8')

  if (verb === 'parsed-roundtrip') {
    process.stdout.write(JSON.stringify(JSON.parse(text)) + '\n')
    return 0
  }

  let out: unknown
  try {
    const { node, topLevel } = normalizeProgramFile(
      JSON.parse(text) as { schema?: string; [k: string]: unknown },
    )
    const program = raiseProgram(node)
    out = {
      program,
      ...(topLevel.params !== undefined ? { params: topLevel.params } : {}),
      ...(topLevel.audio_outputs !== undefined ? { audio_outputs: topLevel.audio_outputs } : {}),
    }
  } catch (e) {
    out = { error: e instanceof Error ? e.message : String(e) }
  }
  process.stdout.write(JSON.stringify(out) + '\n')
  return 0
}

process.exit(main(process.argv.slice(2)))
