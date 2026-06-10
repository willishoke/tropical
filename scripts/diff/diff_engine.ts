/**
 * diff_engine.ts — replay recorded tool scripts against two engines and
 * diff the responses.
 *
 * Each side is a command speaking the ir_service protocol (newline
 * JSON-RPC on stdio, method = tool name). Both sides default to the TS
 * service, so the harness is green TS-vs-TS from day one; when the Lean
 * engine lands (Phase 1) side B becomes the Lean binary's `--rpc` mode
 * and the same scripts become its behavioral spec.
 *
 * Scripts live in tests/fixtures/mcp_scripts/*.json — each a JSON array
 * of `{method, params}` steps, replayed serially (engines are stateful;
 * order matters). Responses are normalized before comparison: the MCP
 * TextContent wrapper is unwrapped (`result.content[0].text` parsed as
 * JSON) so the diff is over the ToolResult, not its serialization.
 *
 * Usage:
 *   bun run scripts/diff/diff_engine.ts                    # TS vs TS, all scripts
 *   bun run scripts/diff/diff_engine.ts --b="<cmd>" [script.json ...]
 */

import { spawn, type ChildProcess } from 'node:child_process'
import { readdirSync, readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { structuralDiff, formatDiffs } from './structural.js'

const TS_SERVICE = 'bun run mcp/ir_service.ts'
const SCRIPT_DIR = resolve(import.meta.dir, '..', '..', 'tests', 'fixtures', 'mcp_scripts')

interface Step { method: string; params: Record<string, unknown> }

interface Args { a: string; b: string; scripts: string[] }

function parseArgs(argv: string[]): Args {
  const args: Args = { a: TS_SERVICE, b: TS_SERVICE, scripts: [] }
  for (const arg of argv) {
    if (arg.startsWith('--a=')) args.a = arg.slice(4)
    else if (arg.startsWith('--b=')) args.b = arg.slice(4)
    else args.scripts.push(arg)
  }
  if (args.scripts.length === 0) {
    args.scripts = readdirSync(SCRIPT_DIR)
      .filter(f => f.endsWith('.json'))
      .map(f => resolve(SCRIPT_DIR, f))
  }
  return args
}

class EngineClient {
  private proc: ChildProcess
  private buf = ''
  private pending = new Map<number, (v: unknown) => void>()
  private nextId = 1

  constructor(cmd: string) {
    const parts = cmd.split(' ').filter(s => s.length > 0)
    this.proc = spawn(parts[0], parts.slice(1), {
      stdio: ['pipe', 'pipe', 'inherit'],
      cwd: resolve(import.meta.dir, '..', '..'),
    })
    this.proc.stdout!.on('data', (chunk: Buffer) => this.onData(chunk.toString()))
  }

  private onData(s: string): void {
    this.buf += s
    let nl: number
    while ((nl = this.buf.indexOf('\n')) >= 0) {
      const line = this.buf.slice(0, nl).trim()
      this.buf = this.buf.slice(nl + 1)
      if (line === '') continue
      const msg = JSON.parse(line) as { id?: number }
      if (typeof msg.id === 'number') {
        const resolve = this.pending.get(msg.id)
        if (resolve !== undefined) {
          this.pending.delete(msg.id)
          resolve(msg)
        }
      }
    }
  }

  call(method: string, params: Record<string, unknown>): Promise<unknown> {
    const id = this.nextId++
    const req = JSON.stringify({ jsonrpc: '2.0', id, method, params })
    return new Promise(resolve => {
      this.pending.set(id, resolve)
      this.proc.stdin!.write(req + '\n')
    })
  }

  close(): void {
    this.proc.stdin!.end()
    this.proc.kill()
  }
}

/** Strip the JSON-RPC frame and unwrap the MCP TextContent envelope. */
function normalize(msg: unknown): unknown {
  if (typeof msg !== 'object' || msg === null) return msg
  const m = msg as Record<string, unknown>
  if ('error' in m) return { rpcError: m.error }
  const result = m.result
  if (typeof result === 'object' && result !== null && 'content' in result) {
    const content = (result as { content: unknown }).content
    if (Array.isArray(content) && content.length > 0) {
      const first = content[0] as { type?: string; text?: string }
      if (first.type === 'text' && typeof first.text === 'string') {
        try {
          return JSON.parse(first.text)
        } catch {
          return first.text
        }
      }
    }
  }
  return result
}

async function replay(cmd: string, steps: Step[]): Promise<unknown[]> {
  const client = new EngineClient(cmd)
  try {
    const responses: unknown[] = []
    for (const step of steps) {
      responses.push(normalize(await client.call(step.method, step.params)))
    }
    return responses
  } finally {
    client.close()
  }
}

async function main(): Promise<number> {
  const { a, b, scripts } = parseArgs(process.argv.slice(2))
  let failed = 0

  for (const script of scripts) {
    const name = script.split('/').pop() ?? script
    const steps = JSON.parse(readFileSync(script, 'utf-8')) as Step[]
    const [resA, resB] = [await replay(a, steps), await replay(b, steps)]

    const divergent: string[] = []
    for (let i = 0; i < steps.length; i++) {
      const diffs = structuralDiff(resA[i], resB[i])
      if (diffs.length > 0) {
        divergent.push(`  step ${i} (${steps[i].method}):\n${formatDiffs(diffs)}`)
      }
    }

    if (divergent.length === 0) {
      console.log(`  PASS   ${name}  (${steps.length} steps)`)
    } else {
      console.log(`  FAIL   ${name.padEnd(28)}  ${divergent.length}/${steps.length} steps diverged`)
      for (const d of divergent) console.log(d)
      failed++
    }
  }

  console.log()
  console.log(failed === 0 ? `all ${scripts.length} scripts agree` : `${failed}/${scripts.length} scripts diverged`)
  return failed === 0 ? 0 : 1
}

process.exit(await main())
