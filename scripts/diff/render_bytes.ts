/**
 * render_bytes.ts — the TS twin of `diffcli render-bytes`.
 *
 * Loads a tropical_plan_5 JSON file into a fresh runtime via the koffi
 * FFI path and writes frames×buffer raw little-endian float64 samples
 * to stdout. The render gate pipes both implementations through
 * `shasum` and demands identical digests: the Lean FFI path must be
 * sample-for-sample the koffi path.
 *
 * Usage:  bun run scripts/diff/render_bytes.ts <plan.json> [--frames N] [--buffer N]
 */

import { readFileSync } from 'node:fs'
import { Runtime } from '../../compiler/runtime/runtime.js'

function natFlag(args: string[], flag: string, fallback: number): number {
  const i = args.indexOf(flag)
  if (i === -1 || i + 1 >= args.length) return fallback
  const n = parseInt(args[i + 1], 10)
  return Number.isNaN(n) ? fallback : n
}

const args = process.argv.slice(2)
const planPath = args.find(a => !a.startsWith('--') && a.endsWith('.json'))
if (!planPath) {
  console.error('usage: render_bytes.ts <plan.json> [--frames N] [--buffer N]')
  process.exit(1)
}
const frames = natFlag(args, '--frames', 16)
const buffer = natFlag(args, '--buffer', 256)

const runtime = new Runtime(buffer)
try {
  runtime.loadPlan(readFileSync(planPath, 'utf-8'))
  for (let i = 0; i < frames; i++) {
    runtime.process()
    const out = runtime.outputBuffer
    process.stdout.write(new Uint8Array(out.buffer, out.byteOffset, out.byteLength))
  }
} finally {
  runtime.dispose()
}
