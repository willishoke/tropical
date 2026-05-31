/**
 * IR service — exposes the tropical IR engine over plain newline-delimited
 * JSON-RPC on stdio, WITHOUT the MCP protocol layer.
 *
 * The Lean front-door (Turnstile) spawns this as a subprocess and relays
 * validated tool calls to it. Each line is one request; each response is one
 * line. `method` is the tool name, `params` is the tool's argument object —
 * i.e. exactly what `handleTool` consumes.
 *
 *   Request:  {"jsonrpc":"2.0","id":N,"method":"<tool>","params":{...}}
 *   Response: {"jsonrpc":"2.0","id":N,"result":<handleTool output>}
 *          |  {"jsonrpc":"2.0","id":N,"error":{"code":-32603,"message":"..."}}
 *
 * Run with:  bun run mcp/ir_service.ts
 */

import { handleTool, listResources, readResource, listPrompts, getPrompt } from './engine.js'

function send(obj: unknown): void {
  process.stdout.write(JSON.stringify(obj) + '\n')
}

process.stdin.setEncoding('utf8')
let buf = ''

for await (const chunk of process.stdin) {
  buf += String(chunk)
  let nl: number
  while ((nl = buf.indexOf('\n')) !== -1) {
    const line = buf.slice(0, nl).trim()
    buf = buf.slice(nl + 1)
    if (line === '') continue

    let id: unknown = null
    try {
      const req = JSON.parse(line) as { id?: unknown; method: string; params?: unknown }
      id = req.id ?? null
      const params = (req.params ?? {}) as Record<string, unknown>
      let result: unknown
      switch (req.method) {
        case 'resources/list': result = { resources: listResources() }; break
        case 'resources/read': result = { text: readResource(String(params.uri ?? '')) }; break
        case 'prompts/list':   result = { prompts: listPrompts() }; break
        case 'prompts/get':    result = getPrompt(String(params.name ?? '')); break
        default:               result = await handleTool(req.method, params)
      }
      send({ jsonrpc: '2.0', id, result })
    } catch (e) {
      send({
        jsonrpc: '2.0',
        id,
        error: { code: -32603, message: e instanceof Error ? e.message : String(e) },
      })
    }
  }
}
