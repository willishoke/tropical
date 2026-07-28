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

import { describe, test, expect, beforeAll, afterAll, setDefaultTimeout } from 'bun:test'
import { spawn, type ChildProcess } from 'node:child_process'

// The boot beforeAll spawns and waits for the Lean engine (frontend + stdlib);
// that's seconds even warm, and tips over bun's 5000ms default on a loaded CI
// runner. Give the file's tests + lifecycle hooks realistic headroom.
setDefaultTimeout(60_000)
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
    // Engine under test: defaults to the TS ir_service; set
    // TROPICAL_ENGINE_CMD to point at another implementation of the same
    // protocol (e.g. "lean/.lake/build/bin/frontend --rpc" for the Lean
    // engine — the Phase 1 gate runs this suite against it unmodified).
    const cmd = process.env.TROPICAL_ENGINE_CMD?.split(' ').filter(Boolean)
      ?? ['bun', 'run', resolve(import.meta.dir, 'ir_service.ts')]
    this.proc = spawn(cmd[0], cmd.slice(1), {
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
      }, 30000)
    })
  }

  private notify(method: string, params: unknown) {
    this.proc.stdin!.write(JSON.stringify({ jsonrpc: '2.0', method, params }) + '\n')
  }

  async init() {
    // The IR service has no MCP handshake; one cheap call warms it up
    // (engine import + native lib load) before the real assertions run.
    await this.request('list_params', {})
  }

  async call(toolName: string, args: Record<string, unknown> = {}): Promise<{
    isError?: boolean
    result: ToolResult
  }> {
    const resp = await this.request(toolName, args)
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
    await client.callOk('add_instance', { program: 'SoftClip', instance_name: real })
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

  // (the 'feedback' helper was removed in the CF-only migration — there is no
  // per-wire delay subsystem — so its unknown-instance cases are gone too)

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

  test('retired parameter-method aliases are not RPC methods', async () => {
    for (const method of ['set_param_glide', 'set_param_freq', 'set_param_velocity']) {
      const env = await client.callError(method, { name: 'does_not_matter', value: 1.0 })
      expect(env.code).toBe('internal_error')
      expect(env.message).toContain(`Unknown tool: '${method}'`)
    }
  })
})

// unknown_device intentionally untested — start_audio would open real hardware.

// ─── Tier 4: unknown_input / unknown_output scoped to instance ────────────────

describe('unknown_input / unknown_output — scoped enum', () => {
  const inst = unique('op')
  beforeAll(async () => {
    await client.callOk('add_instance', { program: 'SoftClip', instance_name: inst })
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
      // SoftClip has exactly two inputs: input, drive
      expect(env.valid.options.sort()).toEqual(['drive', 'input'])
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
    await client.callOk('add_instance', { program: 'SoftClip', instance_name: name })
    const env = await client.callError('add_instance', { program: 'SoftClip', instance_name: name })
    expect(env.code).toBe('instance_exists')
    expect(env.param).toBe('instance_name')
    expect(env.value).toBe(name)
    expect(env.valid).toBeUndefined()
  })
})

describe('invalid_value', () => {
  test('replicate with count=0', async () => {
    const env = await client.callError('replicate', { program: 'SoftClip', count: 0 })
    expect(env.code).toBe('invalid_value')
    expect(env.param).toBe('count')
    expect(env.value).toBe(0)
    expect(env.valid?.kind).toBe('record')
  })

  test('replicate with negative count', async () => {
    const env = await client.callError('replicate', { program: 'SoftClip', count: -3 })
    expect(env.code).toBe('invalid_value')
    expect(env.value).toBe(-3)
  })

  test('replicate with non-integer count', async () => {
    const env = await client.callError('replicate', { program: 'SoftClip', count: 1.5 })
    expect(env.code).toBe('invalid_value')
    expect(env.value).toBe(1.5)
  })

  test('record field spec shape', async () => {
    const env = await client.callError('replicate', { program: 'SoftClip', count: 0 })
    expect(env.valid?.kind).toBe('record')
    if (env.valid?.kind === 'record') {
      expect(env.valid.fields.count).toEqual({ type: 'int', required: true, min: 1 })
    }
  })
})

// ─── The wire grammar as a refusal site ───────────────────────────────────────
//
// `Tropical.WireExpr`'s decoder is the ONE place a wire expression is admitted
// or refused, and it runs at the tool boundary — before the session store is
// touched, so a bad expression cannot poison a session and fail at the next
// compile. A refusal here is a bad ARGUMENT, so it carries `invalid_value` with
// `param`/`value`, not `internal_error` (which ERRORS.md reserves for
// unclassified throws).

describe('wire expression decode — refusal site', () => {
  const inst = unique('sc')
  beforeAll(async () => {
    await client.callOk('add_instance', { program: 'SoftClip', instance_name: inst })
  })

  const decodeError = (expr: unknown) =>
    client.callError('wire', { set: [{ instance: inst, input: 'input', expr }] })

  test('retired combinator is refused by name', async () => {
    const env = await decodeError({ op: 'fold', args: [] })
    expect(env.code).toBe('invalid_value')
    expect(env.param).toBe('set[].expr')
    expect(env.message).toContain("unknown op 'fold'")
  })

  test('state op is refused with the closed-form-only message', async () => {
    const env = await decodeError({ op: 'delay', args: [0] })
    expect(env.code).toBe('invalid_value')
    expect(env.message).toContain('closed-form-only')
    expect(env.message).toContain('no per-sample state')
  })

  test('wrong arity is refused at the tool, not at the next compile', async () => {
    const env = await decodeError({ op: 'mul', args: [1, 2, 3] })
    expect(env.code).toBe('invalid_value')
    expect(env.message).toContain("'mul' requires exactly 2 args, got 3")
  })

  test('a bare string is not an expression', async () => {
    const env = await decodeError('hello')
    expect(env.code).toBe('invalid_value')
    expect(env.message).toContain('got string')
  })

  test('the offending expression is echoed verbatim in value', async () => {
    const env = await decodeError({ op: 'matmul', args: [1, 2] })
    expect(env.value).toEqual({ op: 'matmul', args: [1, 2] })
  })

  // Three constructors are internal compiler forms: broadcastTo (the wiring
  // adapter builds it), input / nestedOut (export's serializer). Admitting one
  // would write it to the store and detonate at the next syncCompile,
  // permanently poisoning the session and persisting through `save`.
  for (const expr of [
    { op: 'input', name: 'zzz' },
    { op: 'nestedOut', ref: 'x', output: 'out' },
    { op: 'broadcastTo', args: [1], shape: [4] },
  ]) {
    test(`engine-internal form '${expr.op}' is refused, not stored`, async () => {
      const env = await client.callError('wire', {
        set: [{ instance: inst, input: 'input', expr }],
      })
      expect(env.code).toBe('invalid_value')
      expect(env.param).toBe('set[].expr')
      expect(env.message).toContain(`'${expr.op}' is not a wire a patch can carry`)

      // The session must still be usable — the old failure mode made every
      // later mutation return the same error until the session was reset.
      const ok = await client.callOk('wire', {
        set: [{ instance: inst, input: 'input', expr: 0.25 }],
      })
      expect(ok).toBeDefined()
    })
  }

  test('a retired alias nested inside a legal expression is caught', async () => {
    const env = await client.callError('wire', {
      set: [{ instance: inst, input: 'input',
              expr: { op: 'add', args: [1, { op: 'paramExpr', name: 'pitch' }] } }],
    })
    expect(env.code).toBe('invalid_value')
    expect(env.message).toContain("unknown op 'paramExpr'")
  })

  // handleWire commits each set[] entry as it goes, so a later refusal leaves
  // the earlier ones applied (and skips syncCompile). This pins the refusal
  // itself, not atomicity: the FAILING entry must not land.
  test('the refused entry of a multi-entry set does not land', async () => {
    await client.callError('wire', {
      set: [
        { instance: inst, input: 'drive', expr: 0.75 },
        { instance: inst, input: 'input', expr: { op: 'scan', args: [] } },
      ],
    })
    const wiring = (await client.callOk('list_wiring', { instance: inst })) as Array<{
      instance: string; input: string; expr: string
    }>
    expect(wiring.find(w => w.input === 'drive')?.expr).toBe('0.75')
    expect(wiring.find(w => w.input === 'input')?.expr).not.toContain('scan')
  })

  // Previously ungated surfaces: both of these bypassed validation entirely and
  // failed later, from inside the compile.
  test('wire_chain initial_expr is decoded', async () => {
    const env = await client.callError('wire_chain', {
      instances: [inst], output: 'out', input: 'input',
      initial_expr: { op: 'fold', args: [] },
    })
    expect(env.code).toBe('invalid_value')
    expect(env.param).toBe('initial_expr')
    expect(env.message).toContain("unknown op 'fold'")
  })

  test('fan_out source is decoded', async () => {
    const env = await client.callError('fan_out', {
      source: { op: 'reg', name: 'x' },
      targets: [{ instance: inst, input: 'input' }],
    })
    expect(env.code).toBe('invalid_value')
    expect(env.param).toBe('source')
    expect(env.message).toContain('closed-form-only')
  })

  // A `ref` whose `output` is present but neither a string nor an index is a
  // fixable argument, not an engine fault.
  test('non-scalar ref.output on a dac wire is classified', async () => {
    const env = await client.callError('wire', {
      set: [{ instance: 'dac', input: 'out', expr: { op: 'ref', instance: inst, output: true } }],
    })
    expect(env.code).toBe('invalid_value')
    expect(env.param).toBe('set[].expr')
    expect(env.message).toContain("'ref' output must be a string port name or index")
  })
})

describe('wire expression decode — retired aliases', () => {
  const inst = unique('sc')
  beforeAll(async () => {
    await client.callOk('add_instance', { program: 'SoftClip', instance_name: inst })
  })

  for (const op of [
    'paramExpr',
    'triggerParamExpr',
    'trigger',
    'array',
    'arrayLiteral',
    'sampleClock',
    'sample_clock',
    'array_set',
    'sessionSlot',
    'sessionArraySlot',
  ]) {
    test(`${op} is rejected instead of canonicalized`, async () => {
      const env = await client.callError('wire', {
        set: [{ instance: inst, input: 'input', expr: { op } }],
      })
      expect(env.code).toBe('invalid_value')
      expect(env.param).toBe('set[].expr')
      expect(env.message).toContain(`unknown op '${op}'`)
    })
  }
})

describe('wire combine', () => {
  const inst = unique('sc')
  beforeAll(async () => {
    await client.callOk('add_instance', { program: 'SoftClip', instance_name: inst })
    await client.callOk('wire', { set: [{ instance: inst, input: 'input', expr: 1 }] })
  })

  test('a binary wire op combines with the existing wire, existing on the left', async () => {
    const data = (await client.callOk('wire', {
      set: [{ instance: inst, input: 'input', expr: 2, combine: 'add' }],
    })) as { set: Array<{ expr: unknown }> }
    expect(data.set[0].expr).toEqual({ op: 'add', args: [1, 2] })
  })

  // Storing an arbitrary string used to defer the failure to the next compile.
  test('a ternary op is not a combine', async () => {
    const env = await client.callError('wire', {
      set: [{ instance: inst, input: 'input', expr: 2, combine: 'select' }],
    })
    expect(env.code).toBe('invalid_value')
    expect(env.param).toBe('set[].combine')
    expect(env.value).toBe('select')
  })

  test('an unknown op is not a combine', async () => {
    const env = await client.callError('wire', {
      set: [{ instance: inst, input: 'input', expr: 2, combine: 'frobnicate' }],
    })
    expect(env.code).toBe('invalid_value')
    expect(env.message).toContain('must be a binary wire op')
  })

  // combine only fires when there is an existing wire to combine WITH; on a
  // bare port it is ignored entirely, so the validation is never reached.
  test('combine on an unwired port is ignored, not validated', async () => {
    const data = await client.callOk('wire', {
      set: [{ instance: inst, input: 'drive', expr: 3, combine: 'frobnicate' }],
    })
    expect(data).toBeDefined()
  })
})

// The literal-source arm of fan_out (isPortRefTS = false) is the one this PR
// newly routed through the decoder; nothing pinned it SUCCEEDING.
describe('fan_out literal source', () => {
  const t1 = unique('sc')
  const t2 = unique('sc')
  beforeAll(async () => {
    await client.callOk('add_instance', { program: 'SoftClip', instance_name: t1 })
    await client.callOk('add_instance', { program: 'SoftClip', instance_name: t2 })
  })

  test('an expression source fans out to every target', async () => {
    const data = (await client.callOk('fan_out', {
      source: { op: 'mul', args: [2, 3] },
      targets: [{ instance: t1, input: 'input' }, { instance: t2, input: 'input' }],
    })) as { linked: string[] }
    expect(data.linked.length).toBe(2)
    const wiring = (await client.callOk('list_wiring', { instance: t1 })) as Array<{
      input: string; expr: string
    }>
    expect(wiring.find(w => w.input === 'input')?.expr).toBe('(2 * 3)')
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
      expect(env.valid.options).toContain('SoftClip')
    }
  })

  test('replicate with unregistered program', async () => {
    const env = await client.callError('replicate', { program: 'Imaginary', count: 1 })
    expect(env.code).toBe('unknown_program')
  })

  test('Levenshtein suggestion fires for 1-edit program typo', async () => {
    const env = await client.callError('add_instance', { program: 'SoftClp', instance_name: unique('x') })
    expect(env.suggestion).toBe('SoftClip')
  })

  test('no suggestion for unrelated name', async () => {
    const env = await client.callError('add_instance', { program: 'Xyzzy', instance_name: unique('x') })
    expect(env.suggestion).toBeUndefined()
  })
})

// ─── Tier 6: type_mismatch predicate ──────────────────────────────────────────

describe('type_mismatch', () => {
  const inst = unique('sc')
  beforeAll(async () => {
    await client.callOk('add_instance', { program: 'SoftClip', instance_name: inst })
  })

  test('array literal → scalar input', async () => {
    const env = await client.callError('wire', {
      set: [{ instance: inst, input: 'input', expr: [1, 2, 3, 4] }],
    })
    expect(env.code).toBe('type_mismatch')
    expect(env.valid?.kind).toBe('predicate')
    if (env.valid?.kind === 'predicate') {
      expect(env.valid.predicate).toBe('type_compatible')
      expect(env.valid.expected).toEqual({ kind: 'scalar', scalar: 'float' })
      expect(env.valid.got).toEqual({
        kind: 'array',
        element: 'float',
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

// ─── export_program: an instance outlives a re-registration of its type ───────
//
// `export_program` is the only route that registers a program type at runtime,
// and it can bind a name that already has live instances. Those instances keep
// their own port snapshot, so the export must not resolve ports the *current*
// target no longer declares — unless the port is actually being exported.
//
// This pins the ABSENCE OF A SPURIOUS ABORT, nothing more. A separate,
// pre-existing bug lives in the same scenario: the exported decl binds the
// instance by type NAME, so the crystallized program carries the REBOUND body
// rather than the snapshot the session plays. That is not fixed here and this
// test does not bless it — see the note at `resolveType` in ProgramIO.lean.

describe('export_program over a re-registered type name', () => {
  const osc = unique('osc')
  const osc2 = unique('osc')
  const voice = unique('Voice')
  const v1 = unique('v')

  test('a stale unwired port does not abort a later export', async () => {
    await client.callOk('add_instance', { program: 'FixedSinOsc', instance_name: osc })
    // Register `voice` with `freq` exposed, then instantiate it.
    await client.callOk('export_program', {
      name: voice,
      outputs: { out: { instance: osc, output: 'sine' } },
      inputs: { freq: `${osc}:freq` },
    })
    await client.callOk('add_instance', { program: voice, instance_name: v1 })

    // Re-register the SAME name with a different exposed set. `v1`'s snapshot
    // still says `freq`; the new target only declares `phase`.
    await client.callOk('add_instance', { program: 'FixedSinOsc', instance_name: osc2 })
    await client.callOk('export_program', {
      name: voice,
      outputs: { out: { instance: osc2, output: 'sine' } },
      inputs: { phase: `${osc2}:phase` },
    })

    // `v1.freq` is neither wired nor exposed, so it never lands in the exported
    // decl and must not be resolved.
    const data = await client.callOk('export_program', {
      name: unique('Outer'),
      outputs: { o: { instance: v1, output: 'out' } },
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

  // The documented split: a refused wire is invalid_value at a TOOL boundary
  // (a named argument the agent can fix) and internal_error on INGEST (a
  // property of the whole document). Without this pin, a future "classify
  // everything" pass would erase the split silently.
  const loadWith = (expr: unknown) => client.callError('load', {
    program: {
      schema: 'tropical_program_2', name: 'p',
      body: {
        op: 'block',
        decls: [{ op: 'instanceDecl', name: 'sc', program: 'SoftClip', inputs: { input: expr } }],
        assigns: [],
      },
    },
  })

  test('a wire the grammar refuses is internal_error on the ingest path', async () => {
    const env = await loadWith({ op: 'fold', args: [] })
    expect(env.code).toBe('internal_error')
    expect(env.param).toBeUndefined()
    expect(env.message).toContain("unknown op 'fold'")
  })

  test('a retired alias is refused on the ingest path too', async () => {
    const env = await loadWith({ op: 'sessionSlot', index: 7 })
    expect(env.code).toBe('internal_error')
    expect(env.message).toContain("unknown op 'sessionSlot'")
  })

  for (const [field, message] of [
    ['params', 'top-level params are retired'],
    ['audio_outputs', 'audio_outputs is retired'],
    ['breaks_cycles', 'breaks_cycles is retired'],
  ] as const) {
    test(`retired root field '${field}' is refused on ingest`, async () => {
      const env = await client.callError('load', {
        program: {
          schema: 'tropical_program_2',
          name: 'retired_root',
          body: { op: 'block', decls: [], assigns: [] },
          [field]: [],
        },
      })
      expect(env.code).toBe('internal_error')
      expect(env.message).toContain(message)
    })
  }
})

// ─── Suggestion-field behavior ────────────────────────────────────────────────

describe('suggestion field', () => {
  test('suggestion is always a value of the expected type (string for enum)', async () => {
    const env = await client.callError('add_instance', { program: 'SoftClp', instance_name: unique('x') })
    expect(typeof env.suggestion).toBe('string')
  })

  test('suggestion is one of the valid.options for enum errors', async () => {
    const env = await client.callError('add_instance', { program: 'SoftClp', instance_name: unique('x') })
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
    await client.callOk('add_instance', { program: 'SoftClip', instance_name: name })
    const env = await client.callError('add_instance', { program: 'SoftClip', instance_name: name })
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
      program: 'SoftClip', instance_name: unique('succ'),
    })
    const d = data as Record<string, unknown>
    expect(typeof d.name).toBe('string')
    expect(d.type_name).toBe('SoftClip')
    expect(Array.isArray(d.inputs)).toBe(true)
    expect(Array.isArray(d.outputs)).toBe(true)
  })

  test('list_programs → status:"ok" with array', async () => {
    const data = await client.callOk('list_programs')
    expect(Array.isArray(data)).toBe(true)
    expect((data as unknown[]).length).toBeGreaterThan(0)
  })
})
