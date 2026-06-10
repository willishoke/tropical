/**
 * diff_audio.ts — render two pipelines' plans and byte-compare the audio.
 *
 * The engine-level arbiter of the Lean-port harness: even if two plans
 * differ structurally (slot order, instruction order), what must agree
 * is the sound. Each side's plan is loaded into its own fresh
 * tropical_runtime_t and rendered for 16×256 samples; the buffers are
 * compared byte-for-byte and the first divergent sample index is
 * reported.
 *
 * Sides implement the compile_patch contract (plan JSON on stdout) —
 * same as diff_plan.ts, so `--b="lake exe diffcli compile"` works here
 * unchanged once the Lean emitter exists.
 *
 * Usage:
 *   bun run scripts/diff/diff_audio.ts                  # TS vs TS, patches/*.json
 *   bun run scripts/diff/diff_audio.ts --b="<cmd>" [--mode=<m>] [patch...]
 */

import { readdirSync } from 'node:fs'
import { resolve } from 'node:path'
import { Runtime } from '../../compiler/runtime/runtime.js'
import { compileVia } from './diff_plan.js'

const FRAMES = 16
const BUFFER_LEN = 256
const TS_COMPILE = 'bun run scripts/diff/compile_patch.ts'

interface Args { a: string; b: string; mode: string | null; patches: string[] }

function parseArgs(argv: string[]): Args {
  const args: Args = { a: TS_COMPILE, b: TS_COMPILE, mode: null, patches: [] }
  for (const arg of argv) {
    if (arg.startsWith('--a=')) args.a = arg.slice(4)
    else if (arg.startsWith('--b=')) args.b = arg.slice(4)
    else if (arg.startsWith('--mode=')) args.mode = arg.slice(7)
    else args.patches.push(arg)
  }
  if (args.patches.length === 0) {
    args.patches = readdirSync('patches')
      .filter(f => f.endsWith('.json'))
      .map(f => resolve('patches', f))
  }
  return args
}

/** Load a plan into a fresh runtime and render FRAMES buffers. */
function render(plan: unknown): Float64Array {
  const runtime = new Runtime(BUFFER_LEN)
  try {
    runtime.loadPlan(JSON.stringify(plan))
    const samples = new Float64Array(BUFFER_LEN * FRAMES)
    for (let i = 0; i < FRAMES; i++) {
      runtime.process()
      samples.set(runtime.outputBuffer, i * BUFFER_LEN)
    }
    return samples
  } finally {
    runtime.dispose()
  }
}

/** Index of the first sample whose bits differ, or -1 if identical. */
function firstDivergence(a: Float64Array, b: Float64Array): number {
  const ba = new BigUint64Array(a.buffer, a.byteOffset, a.length)
  const bb = new BigUint64Array(b.buffer, b.byteOffset, b.length)
  for (let i = 0; i < ba.length; i++) {
    if (ba[i] !== bb[i]) return i
  }
  return -1
}

function main(): number {
  const { a, b, mode, patches } = parseArgs(process.argv.slice(2))
  let failed = 0

  for (const patch of patches) {
    const name = patch.split('/').pop() ?? patch
    try {
      const samplesA = render(compileVia(a, patch, mode))
      const samplesB = render(compileVia(b, patch, mode))
      const idx = firstDivergence(samplesA, samplesB)
      if (idx === -1) {
        console.log(`  PASS   ${name}`)
      } else {
        console.log(
          `  FAIL   ${name.padEnd(28)}  first divergence at sample ${idx}: ` +
          `a=${samplesA[idx]} b=${samplesB[idx]}`)
        failed++
      }
    } catch (e) {
      console.log(`  ERROR  ${name.padEnd(28)}  ${e instanceof Error ? e.message.split('\n')[0] : String(e)}`)
      failed++
    }
  }

  console.log()
  console.log(failed === 0 ? `all ${patches.length} renders agree byte-for-byte` : `${failed}/${patches.length} diverged`)
  return failed === 0 ? 0 : 1
}

if (import.meta.main) process.exit(main())
