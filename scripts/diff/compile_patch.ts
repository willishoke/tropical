/**
 * compile_patch.ts — one patch in, one tropical_plan_5 out.
 *
 * The TS side of the Lean-port differential harness. Loads a
 * tropical_program_2 patch into a fresh session and prints the compiled
 * plan JSON to stdout. The contract is deliberately tiny — `<cmd>
 * <patch.json> [--mode=<m>]` → plan JSON on stdout — so the Lean
 * pipeline can implement the same contract (`lake exe diffcli compile`)
 * and slot into diff_plan.ts / diff_audio.ts as the other side.
 *
 * Usage:  bun run scripts/diff/compile_patch.ts <patch.json> [--mode=fused|microkernel|microkernel-deep]
 */

import { readFileSync } from 'node:fs'
import { makeSession, loadJSON } from '../../compiler/session.js'
import { loadStdlib } from '../../compiler/program.js'
import { compileSession, type CompileSessionOptions } from '../../compiler/ir/compile_session.js'
import { toWirePlan } from '../../compiler/flat_plan.js'

const BUFFER_LEN = 256

function parseArgs(argv: string[]): { patch: string; options: CompileSessionOptions } {
  const positional: string[] = []
  const options: CompileSessionOptions = {}
  for (const arg of argv) {
    if (arg.startsWith('--mode=')) {
      const mode = arg.slice('--mode='.length)
      if (mode !== 'fused' && mode !== 'microkernel' && mode !== 'microkernel-deep') {
        throw new Error(`unknown compilation mode: ${mode}`)
      }
      options.compilation_mode = mode
    } else {
      positional.push(arg)
    }
  }
  if (positional.length !== 1) {
    throw new Error('usage: compile_patch.ts <patch.json> [--mode=<m>]')
  }
  return { patch: positional[0], options }
}

const { patch, options } = parseArgs(process.argv.slice(2))

const session = makeSession(BUFFER_LEN)
try {
  const json = JSON.parse(readFileSync(patch, 'utf-8'))
  loadStdlib(session)
  loadJSON(json, session)
  const plan = compileSession(session, options)
  process.stdout.write(JSON.stringify(toWirePlan(plan)) + '\n')
} finally {
  session.graph.dispose()
}
