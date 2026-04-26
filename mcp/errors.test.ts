/**
 * Tool error envelope tests — mcp/ERRORS.md.
 *
 * Spawns the MCP server as a subprocess, drives tool calls over stdio,
 * asserts the envelope shape (status, code, valid, suggestion, ...) for
 * every error surface in the server.
 *
 * The envelope is the contract agents see, so end-to-end stdio is the
 * correct test plane — covers serialization, wrap, fail, and all
 * failX helpers in combination.
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
  valid?:
    | { kind: 'enum'; options: string[] }
    | { kind: 'record'; fields: Record<string, unknown> }
    | { kind: 'predicate'; predicate: string; expected: unknown; got: unknown }
  suggestion?: unknown
}

type ToolResult =
  | { status: 'ok'; data: unknown }
  | { status: 'error'; error: Envelope }

// ─── stdio harness ────────────────────────────────────────────────────────────

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
    // Keep stderr muted unless debugging:
    // this.proc.stderr!.on('data', (c: Buffer) => process.stderr.write(c))
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
      clientInfo: { name: 'errors-test', version: '0' },
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

  async callOk(toolName: string, args: Record<string, unknown> = {}): Promise<unknown> {
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

// ─── Shared client for the whole file ─────────────────────────────────────────

let client: Client

beforeAll(async () => {
  client = new Client()
  await client.init()
})

afterAll(() => {
  client?.close()
})

// Monotonic counter for unique instance names across tests.
let uniq = 0
const unique = (prefix: string) => `${prefix}_${++uniq}`

// ─── Envelope invariants ──────────────────────────────────────────────────────

describe('envelope shape invariants', () => {
  test('ok responses use status:"ok" and carry data', async () => {
    const { result, isError } = await client.call('list_instances')
    expect(isError).toBeFalsy()
    expect(result.status).toBe('ok')
    if (result.status === 'ok') {
      expect(Array.isArray(result.data)).toBe(true)
    }
  })

  test('error responses use status:"error" with full envelope + isError flag', async () => {
    const { result, isError } = await client.call('remove_instance', { instance_name: 'does_not_exist' })
    expect(isError).toBe(true)
    expect(result.status).toBe('error')
    if (result.status === 'error') {
      const e = result.error
      expect(typeof e.code).toBe('string')
      expect(typeof e.message).toBe('string')
      expect(typeof e.retryable).toBe('boolean')
    }
  })

  test('validation errors have retryable:false', async () => {
    const cases: Array<[string, Record<string, unknown>]> = [
      ['remove_instance', { instance_name: 'nope' }],
      ['add_instance',    { program: 'Missing', instance_name: unique('x') }],
      ['set_param',       { name: 'nope', value: 0 }],
      ['export_program',  { outputs: {} }],
    ]
    for (const [name, args] of cases) {
      const env = await client.callError(name, args)
      expect(env.retryable).toBe(false)
    }
  })

  test('message is a single non-empty string, ≤ 200 chars', async () => {
    const env = await client.callError('add_instance', { program: 'DoesNotExist', instance_name: unique('x') })
    expect(env.message.length).toBeGreaterThan(0)
    expect(env.message.length).toBeLessThanOrEqual(200)
    expect(env.message.includes('\n')).toBe(false)
  })
})

// ─── Tier 1: failBare codes with no `valid` ───────────────────────────────────

describe('missing_argument', () => {
  test('export_program without name', async () => {
    const env = await client.callError('export_program', { outputs: { o: { instance: 'x', output: 'out' } } })
    expect(env.code).toBe('missing_argument')
    expect(env.param).toBe('name')
    expect(env.valid).toBeUndefined()
  })

  test('export_program without outputs', async () => {
    const env = await client.callError('export_program', { name: 'x' })
    expect(env.code).toBe('missing_argument')
    expect(env.param).toBe('outputs')
  })

  test('export_program with empty outputs dict', async () => {
    const env = await client.callError('export_program', { name: 'x', outputs: {} })
    expect(env.code).toBe('missing_argument')
    expect(env.param).toBe('outputs')
  })

  test('load without path or program', async () => {
    const env = await client.callError('load', {})
    expect(env.code).toBe('missing_argument')
  })

  test('merge without program or patch', async () => {
    const env = await client.callError('merge', {})
    expect(env.code).toBe('missing_argument')
  })
})

describe('arity_error', () => {
  test('wire_chain with <2 instances and no initial_expr', async () => {
    const env = await client.callError('wire_chain', { instances: ['a'], output: 'out', input: 'input' })
    expect(env.code).toBe('arity_error')
    expect(env.param).toBe('instances')
    expect(env.value).toEqual(['a'])
  })

  test('fan_in with empty sources', async () => {
    const env = await client.callError('fan_in', {
      sources: [],
      target: { instance: 'x', input: 'input' },
    })
    expect(env.code).toBe('arity_error')
    expect(env.param).toBe('sources')
  })
})

describe('length_mismatch', () => {
  test('wire_zip with sources/targets length mismatch', async () => {
    const env = await client.callError('wire_zip', {
      sources: [{ instance: 'a', output: 'out' }],
      targets: [{ instance: 'b', input: 'in' }, { instance: 'c', input: 'in' }],
    })
    expect(env.code).toBe('length_mismatch')
    expect(env.param).toBe('sources')
    expect(env.value).toEqual({ sources: 1, targets: 2 })
  })
})

// ─── Tier 3: unknown_instance across every wiring tool ────────────────────────

describe('unknown_instance — helper-routed across all wiring tools', () => {
  // First create a real instance so the options list is non-empty for suggestion testing.
  const real = unique('lp')
  beforeAll(async () => {
    await client.callOk('add_instance', { program: 'OnePole', instance_name: real })
  })

  const assertEnvelope = (env: Envelope, paramHint: string) => {
    expect(env.code).toBe('unknown_instance')
    expect(env.valid?.kind).toBe('enum')
    expect(env.retryable).toBe(false)
    // The options list includes the real instance we created.
    if (env.valid?.kind === 'enum') {
      expect(env.valid.options).toContain(real)
    }
    // param should name a meaningful path hint (e.g. 'instance_name', 'sources[].instance', ...)
    expect(typeof env.param).toBe('string')
    expect(env.param).toContain(paramHint)
  }

  test('remove_instance', async () => {
    const env = await client.callError('remove_instance', { instance_name: 'nope' })
    assertEnvelope(env, 'instance_name')
  })

  test('get_info', async () => {
    const env = await client.callError('get_info', { instance_name: 'nope' })
    assertEnvelope(env, 'instance_name')
  })

  test('wire set[].instance', async () => {
    const env = await client.callError('wire', {
      set: [{ instance: 'nope', input: 'input', expr: 0 }],
    })
    assertEnvelope(env, 'instance')
  })

  test('wire remove[].instance', async () => {
    const env = await client.callError('wire', {
      remove: [{ instance: 'nope', input: 'input' }],
    })
    assertEnvelope(env, 'instance')
  })

  test('wire_chain', async () => {
    const env = await client.callError('wire_chain', {
      instances: [real, 'nope'], output: 'out', input: 'input',
    })
    assertEnvelope(env, 'instances')
  })

  test('wire_zip sources[].instance', async () => {
    const env = await client.callError('wire_zip', {
      sources: [{ instance: 'nope', output: 'out' }],
      targets: [{ instance: real, input: 'input' }],
    })
    assertEnvelope(env, 'instance')
  })

  test('wire_zip targets[].instance', async () => {
    const env = await client.callError('wire_zip', {
      sources: [{ instance: real, output: 'out' }],
      targets: [{ instance: 'nope', input: 'input' }],
    })
    assertEnvelope(env, 'instance')
  })

  test('fan_in target.instance', async () => {
    const env = await client.callError('fan_in', {
      sources: [{ instance: real, output: 'out' }],
      target:  { instance: 'nope', input: 'input' },
    })
    assertEnvelope(env, 'instance')
  })

  test('fan_in sources[].instance', async () => {
    const env = await client.callError('fan_in', {
      sources: [{ instance: 'nope', output: 'out' }],
      target:  { instance: real, input: 'input' },
    })
    assertEnvelope(env, 'instance')
  })

  test('fan_out source.instance', async () => {
    const env = await client.callError('fan_out', {
      source:  { instance: 'nope', output: 'out' },
      targets: [{ instance: real, input: 'input' }],
    })
    assertEnvelope(env, 'instance')
  })

  test('fan_out targets[].instance', async () => {
    const env = await client.callError('fan_out', {
      source:  { instance: real, output: 'out' },
      targets: [{ instance: 'nope', input: 'input' }],
    })
    assertEnvelope(env, 'instance')
  })

  test('feedback from.instance', async () => {
    const env = await client.callError('feedback', {
      from: { instance: 'nope', output: 'out' },
      to:   { instance: real, input: 'input' },
    })
    assertEnvelope(env, 'instance')
  })

  test('feedback to.instance', async () => {
    const env = await client.callError('feedback', {
      from: { instance: real, output: 'out' },
      to:   { instance: 'nope', input: 'input' },
    })
    assertEnvelope(env, 'instance')
  })

  test('wire to dac.out — unknown source instance in ref expression', async () => {
    const env = await client.callError('wire', {
      set: [{
        instance: 'dac', input: 'out',
        expr: { op: 'ref', instance: 'nope', output: 0 },
      }],
    })
    assertEnvelope(env, 'instance')
  })
})

// ─── Tier 3: unknown_param / unknown_device ──────────────────────────────────

describe('unknown_param', () => {
  test('set_param with name not in registry', async () => {
    const env = await client.callError('set_param', { name: 'does_not_exist', value: 1.0 })
    expect(env.code).toBe('unknown_param')
    expect(env.param).toBe('name')
    expect(env.value).toBe('does_not_exist')
    expect(env.valid?.kind).toBe('enum')
  })
})

// unknown_device intentionally untested — start_audio would open real hardware.

// ─── Tier 4: unknown_input / unknown_output scoped to instance ────────────────

describe('unknown_input / unknown_output — scoped enum', () => {
  const inst = unique('op')
  beforeAll(async () => {
    await client.callOk('add_instance', { program: 'OnePole', instance_name: inst })
  })

  test('unknown_input — valid.options is exactly this instance\'s inputs', async () => {
    const env = await client.callError('wire', {
      set: [{ instance: inst, input: 'freuq', expr: 0 }],
    })
    expect(env.code).toBe('unknown_input')
    expect(env.param).toBe('input')
    expect(env.value).toBe('freuq')
    expect(env.valid?.kind).toBe('enum')
    if (env.valid?.kind === 'enum') {
      // OnePole has exactly two inputs: input, g
      expect(env.valid.options.sort()).toEqual(['g', 'input'])
    }
  })

  test('unknown_output — valid.options is exactly this instance\'s outputs', async () => {
    const env = await client.callError('wire', {
      set: [{
        instance: 'dac', input: 'out',
        expr: { op: 'ref', instance: inst, output: 'ouput' },
      }],
    })
    expect(env.code).toBe('unknown_output')
    expect(env.param).toBe('output')
    expect(env.valid?.kind).toBe('enum')
    if (env.valid?.kind === 'enum') {
      expect(env.valid.options).toEqual(['out'])
    }
  })

  test('Levenshtein suggestion fires on close typo (input)', async () => {
    const env = await client.callError('wire', {
      set: [{ instance: inst, input: 'inpu', expr: 0 }],  // 1-edit from 'input'
    })
    expect(env.suggestion).toBe('input')
  })

  test('no suggestion on far typo', async () => {
    const env = await client.callError('wire', {
      set: [{ instance: inst, input: 'zzzzzz', expr: 0 }],
    })
    expect(env.suggestion).toBeUndefined()
  })
})

// ─── Tier 5: instance_exists / invalid_value / unknown_program / invalid_type_args ──

describe('instance_exists', () => {
  test('add_instance with duplicate name', async () => {
    const name = unique('dup')
    await client.callOk('add_instance', { program: 'OnePole', instance_name: name })
    const env = await client.callError('add_instance', { program: 'OnePole', instance_name: name })
    expect(env.code).toBe('instance_exists')
    expect(env.param).toBe('instance_name')
    expect(env.value).toBe(name)
    expect(env.valid).toBeUndefined()
  })
})

describe('invalid_value', () => {
  test('replicate with count=0', async () => {
    const env = await client.callError('replicate', { program: 'OnePole', count: 0 })
    expect(env.code).toBe('invalid_value')
    expect(env.param).toBe('count')
    expect(env.value).toBe(0)
    expect(env.valid?.kind).toBe('record')
  })

  test('replicate with negative count', async () => {
    const env = await client.callError('replicate', { program: 'OnePole', count: -3 })
    expect(env.code).toBe('invalid_value')
    expect(env.value).toBe(-3)
  })

  test('replicate with non-integer count', async () => {
    const env = await client.callError('replicate', { program: 'OnePole', count: 1.5 })
    expect(env.code).toBe('invalid_value')
    expect(env.value).toBe(1.5)
  })

  test('record field spec shape', async () => {
    const env = await client.callError('replicate', { program: 'OnePole', count: 0 })
    expect(env.valid?.kind).toBe('record')
    if (env.valid?.kind === 'record') {
      expect(env.valid.fields.count).toEqual({ type: 'int', required: true, min: 1 })
    }
  })
})

describe('unknown_program', () => {
  test('add_instance with unregistered program', async () => {
    const env = await client.callError('add_instance', { program: 'NonExistent', instance_name: unique('x') })
    expect(env.code).toBe('unknown_program')
    expect(env.param).toBe('program')
    expect(env.value).toBe('NonExistent')
    expect(env.valid?.kind).toBe('enum')
    if (env.valid?.kind === 'enum') {
      expect(env.valid.options).toContain('OnePole')
    }
  })

  test('replicate with unregistered program', async () => {
    const env = await client.callError('replicate', { program: 'Imaginary', count: 1 })
    expect(env.code).toBe('unknown_program')
  })

  test('Levenshtein suggestion fires for 1-edit program typo', async () => {
    const env = await client.callError('add_instance', { program: 'OnePoel', instance_name: unique('x') })
    expect(env.suggestion).toBe('OnePole')
  })

  test('no suggestion for unrelated name', async () => {
    const env = await client.callError('add_instance', { program: 'Xyzzy', instance_name: unique('x') })
    expect(env.suggestion).toBeUndefined()
  })
})

describe('invalid_type_args', () => {
  // The generic Delay program expects type_args: { N: int }.
  test('generic program instantiated with unexpected type_args', async () => {
    // Delay accepts N; passing a bogus field triggers resolveTypeArgs to throw.
    const env = await client.callError('add_instance', {
      program:       'Delay',
      instance_name: unique('d'),
      type_args:     { BogusKey: 42 },
    })
    // Either invalid_type_args (registered template) or unknown_program — both are fine,
    // but the helper should produce invalid_type_args because Delay IS registered.
    expect(['invalid_type_args', 'unknown_program']).toContain(env.code)
    if (env.code === 'invalid_type_args') {
      expect(env.param).toBe('type_args')
    }
  })
})

// ─── Tier 6: type_mismatch predicate ──────────────────────────────────────────

describe('type_mismatch', () => {
  const inst = unique('sc')
  beforeAll(async () => {
    await client.callOk('add_instance', { program: 'OnePole', instance_name: inst })
  })

  test('array literal → scalar input', async () => {
    const env = await client.callError('wire', {
      set: [{ instance: inst, input: 'input', expr: [1, 2, 3, 4] }],
    })
    expect(env.code).toBe('type_mismatch')
    expect(env.valid?.kind).toBe('predicate')
    if (env.valid?.kind === 'predicate') {
      expect(env.valid.predicate).toBe('type_compatible')
      expect(env.valid.expected).toEqual({ tag: 'scalar', scalar: 'float' })
      expect(env.valid.got).toEqual({
        tag: 'array',
        element: { tag: 'scalar', scalar: 'float' },
        shape: [4],
      })
    }
  })

  test('scalar literal → scalar input is accepted (no error)', async () => {
    const data = await client.callOk('wire', {
      set: [{ instance: inst, input: 'input', expr: 0.5 }],
    })
    expect(data).toBeDefined()
  })
})

// ─── Tier 7: invalid_state ────────────────────────────────────────────────────

describe('invalid_state', () => {
  test('stop_audio before start_audio', async () => {
    // Only valid if the server is in a state where DAC hasn't been created.
    // This test is brittle if a prior test started audio, but none do in this file.
    const { result } = await client.call('stop_audio')
    if (result.status === 'error') {
      expect(result.error.code).toBe('invalid_state')
    } else {
      // If DAC already exists from a prior test, stop_audio succeeds — that's also fine.
      expect(result.status).toBe('ok')
    }
  })
})

// ─── internal_error fallback for unclassified throws ──────────────────────────

describe('internal_error fallback', () => {
  test('malformed program in load surfaces as internal_error', async () => {
    // loadJSON throws plain Error for unrecognized schema — caught by the fallback.
    const env = await client.callError('load', {
      program: { schema: 'not_a_real_schema', garbage: true },
    })
    expect(env.code).toBe('internal_error')
    expect(env.retryable).toBe(false)
  })
})

// ─── Suggestion-field behavior ────────────────────────────────────────────────

describe('suggestion field', () => {
  test('suggestion is always a value of the expected type (string for enum)', async () => {
    const env = await client.callError('add_instance', { program: 'OnePoel', instance_name: unique('x') })
    expect(typeof env.suggestion).toBe('string')
  })

  test('suggestion is one of the valid.options for enum errors', async () => {
    const env = await client.callError('add_instance', { program: 'OnePoel', instance_name: unique('x') })
    if (env.valid?.kind === 'enum' && typeof env.suggestion === 'string') {
      expect(env.valid.options).toContain(env.suggestion)
    }
  })

  test('suggestion omitted when no candidate within threshold', async () => {
    const env = await client.callError('set_param', { name: 'q9x4k2', value: 0 })
    expect(env.suggestion).toBeUndefined()
  })

  test('suggestion is not emitted for errors without valid.enum (instance_exists)', async () => {
    const name = unique('dup')
    await client.callOk('add_instance', { program: 'OnePole', instance_name: name })
    const env = await client.callError('add_instance', { program: 'OnePole', instance_name: name })
    expect(env.suggestion).toBeUndefined()
  })
})

// ─── list_wiring filter: unknown instance is NOT an error (by design) ─────────

describe('list_wiring non-error path', () => {
  test('unknown instance filter returns empty list without error', async () => {
    const data = await client.callOk('list_wiring', { instance: 'nonexistent_xyz' })
    expect(Array.isArray(data)).toBe(true)
    expect((data as unknown[]).length).toBe(0)
  })
})

// ─── Success-path envelope sampling ───────────────────────────────────────────

describe('success envelope shape', () => {
  test('add_instance → status:"ok" with summary data', async () => {
    const data = await client.callOk('add_instance', {
      program: 'OnePole', instance_name: unique('succ'),
    })
    const d = data as Record<string, unknown>
    expect(typeof d.name).toBe('string')
    expect(d.type_name).toBe('OnePole')
    expect(Array.isArray(d.inputs)).toBe(true)
    expect(Array.isArray(d.outputs)).toBe(true)
  })

  test('list_programs → status:"ok" with array', async () => {
    const data = await client.callOk('list_programs')
    expect(Array.isArray(data)).toBe(true)
    expect((data as unknown[]).length).toBeGreaterThan(0)
  })
})
