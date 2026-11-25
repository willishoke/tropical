/**
 * type_family.test.ts — a registered type at a size, over the tool surface.
 *
 * A type family (`lean/Tropical/Family.lean`, `design/type-family.md`) is
 * instantiated by naming a member, `<family><size>`: `add_instance`, `load`
 * and `save` carry the member name, `list_programs` lists the families, and a
 * family spelling that names no member is refused with its reason.
 */

import { describe, test, expect, beforeAll, afterAll, setDefaultTimeout } from 'bun:test'
import { spawn, type ChildProcess } from 'node:child_process'
import { resolve } from 'node:path'

setDefaultTimeout(60_000)

type Envelope = {
  code: string
  message: string
  retryable: boolean
  param?: string
  value?: unknown
  valid?: { kind: string; predicate?: string; expected?: unknown; got?: unknown }
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
    // The IR service has no MCP handshake; one cheap call warms it up.
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

const controls = (k: number) =>
  Object.fromEntries(Array.from({ length: k }, (_, j) => [`c${j}`, (j % 3) / 4]))

function patch(decls: unknown[]) {
  return { schema: 'tropical_program_2', name: 'family', body: { op: 'block', decls } }
}

describe('type families — listing and instantiation', () => {
  test('list_programs lists each family with its size range', async () => {
    const programs = (await client.callOk('list_programs')) as Array<Record<string, any>>
    const cr = programs.find((p) => p.program_name === 'PeriodicCatmullRom')
    expect(cr?.family).toMatchObject({ member: 'PeriodicCatmullRom<size>', size_min: 1, size_max: 2147483647 })
    expect(programs.find((p) => p.program_name === 'PeriodicBSpline')?.family?.size_min).toBe(1)
    // The size-8 members are boot-registered concrete types, as before.
    expect(programs.find((p) => p.program_name === 'PeriodicCatmullRom8')?.inputs?.length).toBe(9)
  })

  test('add_instance of a member: ports derived from the size', async () => {
    const inst = unique('wave')
    const data = await client.callOk('add_instance', { program: 'PeriodicCatmullRom12', instance_name: inst })
    expect(data.type_name).toBe('PeriodicCatmullRom12')
    expect(data.inputs).toEqual(['phase', ...Array.from({ length: 12 }, (_, j) => `c${j}`)])
    expect(data.outputs).toEqual(['out'])
    // Now registered: listed as a concrete type, and a second instance reuses it.
    const programs = (await client.callOk('list_programs')) as Array<Record<string, any>>
    expect(programs.filter((p) => p.program_name === 'PeriodicCatmullRom12').length).toBe(1)
    const again = await client.callOk('add_instance', { program: 'PeriodicCatmullRom12', instance_name: unique('wave') })
    expect(again.inputs.length).toBe(13)
    await client.callOk('remove_instance', { instance_name: inst })
  })

  test('save → load keeps the member name, so the size survives', async () => {
    await client.callOk('load', { program: patch([
      { op: 'instanceDecl', name: 'lfo', program: 'PeriodicBSpline5', inputs: {
        phase: { op: 'mul', args: [{ op: 'sampleIndex' }, 89478485] }, ...controls(5) } },
    ]) })
    const saved = await client.callOk('save')
    const decl = (saved.program.body.decls as Array<Record<string, any>>).find((d) => d.name === 'lfo')
    expect(decl?.program).toBe('PeriodicBSpline5')
    await client.callOk('load', { program: saved.program })
    const instances = (await client.callOk('list_instances')) as Array<Record<string, any>>
    expect(instances).toHaveLength(1)
    expect(instances[0]).toMatchObject({ name: 'lfo', type_name: 'PeriodicBSpline5' })
    expect(instances[0]!.inputs).toEqual(['phase', 'c0', 'c1', 'c2', 'c3', 'c4'])
  })
})

describe('type families — refusals', () => {
  test('bare family name: invalid_value, needs a size', async () => {
    const env = await client.callError('add_instance', { program: 'PeriodicCatmullRom', instance_name: unique('x') })
    expect(env.code).toBe('invalid_value')
    expect(env.param).toBe('program')
    expect(env.message).toContain('needs a size')
  })

  test('size out of range: invalid_value with the range as a predicate', async () => {
    const env = await client.callError('add_instance', { program: 'PeriodicCatmullRom0', instance_name: unique('x') })
    expect(env.code).toBe('invalid_value')
    expect(env.valid?.kind).toBe('predicate')
    expect(env.valid?.expected).toEqual({ min: 1, max: 2147483647 })
    expect(env.valid?.got).toBe(0)
  })

  test('leading zeros: invalid_value naming the canonical member', async () => {
    const env = await client.callError('add_instance', { program: 'PeriodicBSpline012', instance_name: unique('x') })
    expect(env.code).toBe('invalid_value')
    expect(env.message).toContain("'PeriodicBSpline12'")
  })

  test('a size on a type that is not a family: invalid_value', async () => {
    const env = await client.callError('add_instance', { program: 'VCA3', instance_name: unique('x') })
    expect(env.code).toBe('invalid_value')
    expect(env.message).toContain("'VCA' is not a type family")
  })

  test('unknown family: unknown_program, and the message lists the families', async () => {
    const env = await client.callError('add_instance', { program: 'PeriodicHermite12', instance_name: unique('x') })
    expect(env.code).toBe('unknown_program')
    expect(env.message).toContain('PeriodicCatmullRom (1 to 2147483647 control points)')
  })

  test('load of a refused spelling: internal_error with the reason', async () => {
    const env = await client.callError('load', { program: patch([
      { op: 'instanceDecl', name: 'x', program: 'PeriodicCatmullRom2147483648' }]) })
    expect(env.code).toBe('internal_error')
    expect(env.message).toContain('size 2147483648 is out of range')
  })

  test('export_program may not take a family name or member spelling', async () => {
    const inst = unique('vca')
    await client.callOk('add_instance', { program: 'VCA', instance_name: inst })
    for (const name of ['PeriodicCatmullRom', 'PeriodicCatmullRom7']) {
      const env = await client.callError('export_program', {
        name, inputs: {}, outputs: { out: { instance: inst, output: 'out' } } })
      expect(env.code).toBe('invalid_value')
      expect(env.param).toBe('name')
      expect(env.message).toContain("reserved by the type family 'PeriodicCatmullRom'")
    }
  })
})
