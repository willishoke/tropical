/**
 * wire_dac.test.ts — A1: dac.out boundary-leaf wiring via the unified `wire` tool.
 *
 * Verifies that `wire` accepts `instance: "dac", input: "out"` as a destination,
 * routes ref-shaped expressions into session.graphOutputs (mix-bus append),
 * supports mass-remove, reserves `dac` as an instance name, and rejects
 * non-ref expressions for the dac destination.
 */

import { describe, test, expect, beforeAll, afterAll } from 'bun:test'
import { spawn, type ChildProcess } from 'node:child_process'
import { resolve } from 'node:path'

type Envelope = {
  code: string
  message: string
  retryable: boolean
  param?: string
  value?: unknown
}

type ToolResult =
  | { status: 'ok'; data: unknown }
  | { status: 'error'; error: Envelope }

class Client {
  private proc: ChildProcess
  private buf = ''
  private pending = new Map<number, (v: unknown) => void>()
  private nextId = 1

  constructor() {
    const serverPath = resolve(import.meta.dir, 'server.ts')
    this.proc = spawn('bun', ['run', serverPath], {
      stdio: ['pipe', 'pipe', 'pipe'],
      cwd: resolve(import.meta.dir, '..'),
    })
    this.proc.stdout!.on('data', (chunk: Buffer) => this.onData(chunk.toString()))
  }

  private onData(s: string) {
    this.buf += s
    let idx: number
    while ((idx = this.buf.indexOf('\n')) >= 0) {
      const line = this.buf.slice(0, idx).trim()
      this.buf = this.buf.slice(idx + 1)
      if (!line) continue
      try {
        const msg = JSON.parse(line)
        if (typeof msg.id === 'number') {
          const r = this.pending.get(msg.id)
          if (r) { this.pending.delete(msg.id); r(msg) }
        }
      } catch { /* ignore */ }
    }
  }

  private request(method: string, params: unknown): Promise<any> {
    const id = this.nextId++
    return new Promise((res, rej) => {
      this.pending.set(id, res)
      this.proc.stdin!.write(JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n')
      setTimeout(() => {
        if (this.pending.has(id)) {
          this.pending.delete(id)
          rej(new Error(`Timeout on ${method}`))
        }
      }, 5000)
    })
  }

  private notify(method: string, params: unknown) {
    this.proc.stdin!.write(JSON.stringify({ jsonrpc: '2.0', method, params }) + '\n')
  }

  async init() {
    await this.request('initialize', {
      protocolVersion: '2024-11-05',
      capabilities: {},
      clientInfo: { name: 'wire-dac-test', version: '0' },
    })
    this.notify('notifications/initialized', {})
  }

  async call(toolName: string, args: Record<string, unknown> = {}): Promise<{
    isError?: boolean
    result: ToolResult
  }> {
    const resp = await this.request('tools/call', { name: toolName, arguments: args })
    const content = resp.result?.content?.[0]?.text
    if (typeof content !== 'string') throw new Error(`No text content in response: ${JSON.stringify(resp)}`)
    return { isError: resp.result.isError, result: JSON.parse(content) as ToolResult }
  }

  async callOk(toolName: string, args: Record<string, unknown> = {}): Promise<any> {
    const { result } = await this.call(toolName, args)
    if (result.status !== 'ok') {
      throw new Error(`Expected ok, got error: ${JSON.stringify(result)}`)
    }
    return result.data
  }

  async callError(toolName: string, args: Record<string, unknown> = {}): Promise<Envelope> {
    const { isError, result } = await this.call(toolName, args)
    expect(isError).toBe(true)
    if (result.status !== 'error') {
      throw new Error(`Expected error, got ok: ${JSON.stringify(result)}`)
    }
    return result.error
  }

  close() {
    this.proc.kill('SIGTERM')
  }
}

let client: Client

beforeAll(async () => {
  client = new Client()
  await client.init()
})

afterAll(() => {
  client?.close()
})

let uniq = 0
const unique = (prefix: string) => `${prefix}_${++uniq}`

describe('wire to dac.out — basic flow', () => {
  test('ref-shaped expression to dac.out is accepted and routed', async () => {
    const inst = unique('osc')
    await client.callOk('add_instance', { program: 'OnePole', instance_name: inst })

    const data = await client.callOk('wire', {
      set: [{
        instance: 'dac',
        input: 'out',
        expr: { op: 'ref', instance: inst, output: 0 },
      }],
    })

    // The handler returns the dac wires under a separate `dac` key.
    expect(data.dac).toEqual([
      { instance: 'dac', input: 'out', expr: { op: 'ref', instance: inst, output: 0 } },
    ])
  })

  test('multiple wires to dac.out accumulate (mix-bus)', async () => {
    const a = unique('a')
    const b = unique('b')
    await client.callOk('add_instance', { program: 'OnePole', instance_name: a })
    await client.callOk('add_instance', { program: 'OnePole', instance_name: b })

    // Clear any prior dac wires from earlier tests.
    await client.callOk('wire', {
      remove: [{ instance: 'dac', input: 'out' }],
    })

    await client.callOk('wire', {
      set: [
        { instance: 'dac', input: 'out', expr: { op: 'ref', instance: a, output: 0 } },
        { instance: 'dac', input: 'out', expr: { op: 'ref', instance: b, output: 0 } },
      ],
    })

    // graphOutputs surfaces via save → program.audio_outputs
    const saved = await client.callOk('save')
    expect(Array.isArray(saved.program?.audio_outputs)).toBe(true)
    const outs = saved.program.audio_outputs
    expect(outs.length).toBe(2)
    expect(outs.some((o: any) => o.instance === a)).toBe(true)
    expect(outs.some((o: any) => o.instance === b)).toBe(true)
  })

  test('remove clears all dac wires at once', async () => {
    // Establish at least one dac wire.
    const inst = unique('rm')
    await client.callOk('add_instance', { program: 'OnePole', instance_name: inst })
    await client.callOk('wire', {
      set: [{ instance: 'dac', input: 'out', expr: { op: 'ref', instance: inst, output: 0 } }],
    })

    const data = await client.callOk('wire', {
      remove: [{ instance: 'dac', input: 'out' }],
    })
    expect(data.dacRemoved).toBeGreaterThanOrEqual(1)

    const saved = await client.callOk('save')
    expect(saved.program?.audio_outputs ?? []).toEqual([])
  })
})

describe('wire to dac.out — error envelopes', () => {
  test('non-ref expression rejected', async () => {
    const env = await client.callError('wire', {
      set: [{ instance: 'dac', input: 'out', expr: 0.5 }],
    })
    expect(env.code).toBe('invalid_value')
    expect(env.param).toBe('expr')
  })

  test('wrong dac port name rejected', async () => {
    const inst = unique('w')
    await client.callOk('add_instance', { program: 'OnePole', instance_name: inst })
    const env = await client.callError('wire', {
      set: [{
        instance: 'dac',
        input: 'left',
        expr: { op: 'ref', instance: inst, output: 0 },
      }],
    })
    expect(env.code).toBe('unknown_output')
    expect(env.param).toBe('set[].input')
  })

  test('ref to unknown instance rejected with helpful options', async () => {
    const env = await client.callError('wire', {
      set: [{
        instance: 'dac',
        input: 'out',
        expr: { op: 'ref', instance: 'does_not_exist', output: 0 },
      }],
    })
    expect(env.code).toBe('unknown_instance')
  })

  test('ref output index out of range rejected', async () => {
    const inst = unique('oor')
    await client.callOk('add_instance', { program: 'OnePole', instance_name: inst })
    const env = await client.callError('wire', {
      set: [{
        instance: 'dac',
        input: 'out',
        expr: { op: 'ref', instance: inst, output: 99 },
      }],
    })
    expect(env.code).toBe('unknown_output')
  })

  test('remove with wrong dac port name rejected', async () => {
    const env = await client.callError('wire', {
      remove: [{ instance: 'dac', input: 'left' }],
    })
    expect(env.code).toBe('unknown_output')
    expect(env.param).toBe('remove[].input')
  })
})

describe('dac is a reserved instance name', () => {
  test('add_instance with name "dac" rejected', async () => {
    const env = await client.callError('add_instance', {
      program: 'OnePole',
      instance_name: 'dac',
    })
    expect(env.code).toBe('invalid_value')
    expect(env.param).toBe('instance_name')
  })

  test('replicate with name_prefix "dac" rejected', async () => {
    const env = await client.callError('replicate', {
      program: 'OnePole',
      count: 2,
      name_prefix: 'dac',
    })
    expect(env.code).toBe('invalid_value')
    expect(env.param).toBe('name_prefix')
  })
})

describe('set_output is removed', () => {
  test('calling set_output returns unknown-tool error', async () => {
    const { isError, result } = await client.call('set_output', {
      outputs: [{ instance: 'whatever', output: 'out' }],
    })
    expect(isError).toBe(true)
    if (result.status === 'error') {
      // Server rejects the unknown tool name; either MethodNotFound at the
      // protocol level or an internal_error from the dispatcher fall-through.
      expect(typeof result.error.message).toBe('string')
    }
  })
})
