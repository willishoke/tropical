/** Atomic load_patch_graph handshake: one reply identifies the vocabulary,
 * exact immutable runtime generation, realized graph facts, and complete
 * observation bindings. list_scope_taps remains a compatibility diagnostic. */

import { describe, expect, setDefaultTimeout, test } from 'bun:test'
import { spawn, type ChildProcess } from 'node:child_process'
import { existsSync } from 'node:fs'
import { connect, type Socket } from 'node:net'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

setDefaultTimeout(120_000)

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '../..')
const frontendBin = resolve(repoRoot, 'lean/.lake/build/bin/frontend')

type Pending = { resolve: (value: any) => void; reject: (error: Error) => void }
let engineClientSerial = 0

class EngineClient {
  private readonly process: ChildProcess
  private readonly socketPath: string
  private socket: Socket | null = null
  private buffer = ''
  private readonly pending = new Map<number, Pending>()
  private readonly queued: string[] = []
  private nextId = 1
  private stopped = false

  constructor() {
    this.socketPath = join(
      tmpdir(), `tropical-handshake-${process.pid}-${engineClientSerial++}.sock`,
    )
    this.process = spawn(frontendBin, ['--serve', this.socketPath], {
      cwd: repoRoot,
      stdio: ['ignore', 'ignore', 'ignore'],
    })
    this.process.on('exit', () => this.failAll(new Error('engine exited')))
    void this.connectLoop()
  }

  private async connectLoop() {
    for (let attempt = 0; attempt < 300 && !this.stopped; attempt++) {
      if (existsSync(this.socketPath)) {
        const connected = await new Promise<boolean>((done) => {
          const socket = connect({ path: this.socketPath })
          socket.on('connect', () => {
            this.socket = socket
            for (const line of this.queued) socket.write(line)
            this.queued.length = 0
            done(true)
          })
          socket.on('data', (chunk: Buffer) => this.onData(chunk.toString()))
          socket.on('error', () => {
            socket.destroy()
            done(false)
          })
        })
        if (connected) return
      }
      await new Promise((done) => setTimeout(done, 100))
    }
    if (!this.stopped) this.failAll(new Error(`could not connect to ${this.socketPath}`))
  }

  private failAll(error: Error) {
    for (const pending of this.pending.values()) pending.reject(error)
    this.pending.clear()
  }

  private onData(text: string) {
    this.buffer += text
    let newline: number
    while ((newline = this.buffer.indexOf('\n')) >= 0) {
      const line = this.buffer.slice(0, newline).trim()
      this.buffer = this.buffer.slice(newline + 1)
      if (!line) continue
      const message = JSON.parse(line)
      const pending = this.pending.get(message.id)
      if (!pending) continue
      this.pending.delete(message.id)
      if (message.error) {
        pending.reject(new Error(JSON.stringify(message.error)))
        continue
      }
      const textEnvelope = message.result?.content?.[0]?.text
      if (typeof textEnvelope === 'string') {
        const inner = JSON.parse(textEnvelope)
        inner.status === 'ok'
          ? pending.resolve(inner.data)
          : pending.reject(new Error(`${inner.error?.code}: ${inner.error?.message}`))
      } else {
        pending.resolve(message.result)
      }
    }
  }

  call(method: string, params: any = {}): Promise<any> {
    const id = this.nextId++
    const line = JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n'
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject })
      if (this.socket) this.socket.write(line)
      else this.queued.push(line)
      setTimeout(() => {
        if (this.pending.delete(id)) reject(new Error(`timeout: ${method}`))
      }, 60_000)
    })
  }

  stop() {
    this.stopped = true
    this.socket?.destroy()
    this.process.kill()
  }
}

const graph = {
  nodes: [
    { id: 'src', kind: 'source', params: { freq: 220, morph: 0 }, sel: {}, in: {} },
    { id: 'scopeProbe', kind: 'string', params: { freq: 220, decay: 1, t: 0 }, sel: {}, in: {} },
    { id: 'out', kind: 'out', params: {}, sel: {}, in: { in: ['src'] } },
  ],
  out: 'out',
  // Exercise both an audible node and a disconnected observation-only node.
  // Put the disconnected probe first: it must not replace the authored final
  // mix's stable `out` binding in the separate observation artifact.
  taps: ['scopeProbe', 'src'],
}

describe('load_patch_graph compile handshake', () => {
  test('one response binds realized state and observations to its exact generation', async () => {
    if (!existsSync(frontendBin))
      throw new Error(`frontend binary missing: ${frontendBin} (run \`make build lean\`)`)

    const client = new EngineClient()
    try {
      const vocabulary = await client.call('get_vocabulary')
      const report = await client.call('load_patch_graph', graph)

      expect(report.ok).toBe(true)
      expect(report.vocabulary_fingerprint).toBe(vocabulary.fingerprint)
      expect(report.program_version).toBeGreaterThan(0)
      expect(report.control_version).toBeGreaterThan(0)

      expect(report.nodes).toContainEqual({ id: 'src', kind: 'source', status: 'active' })
      expect(report.nodes).toContainEqual({ id: 'scopeProbe', kind: 'string', status: 'excluded' })
      expect(report.inputs).toContainEqual({ node: 'out', port: 'in', state: 'wired', sources: ['src'] })
      expect(report.params).toContainEqual({ name: 'src.freq', value: 220, discipline: 'anchor' })

      for (const tap of report.taps) {
        expect(Object.keys(tap).sort()).toEqual(['instance', 'name', 'output', 'slot'])
        expect(tap.slot).toBe(`${tap.instance}.${tap.output}`)
      }
      expect(report.taps.map((tap: any) => tap.name)).toEqual(['out', 'scopeProbe', 'src'])

      const rendered = await client.call('render_window', {
        start: 0,
        count: 32,
        slots: report.taps.map((tap: any) => tap.slot),
      })
      expect(rendered.program_version).toBe(report.program_version)
      expect(rendered.control_version).toBe(report.control_version)
      expect(rendered.values[0]).not.toEqual(rendered.values[1])

      const listed = await client.call('list_scope_taps')
      expect(listed.taps).toEqual(report.taps)

      // Removing projections selects the single audio artifact. Its authored
      // final mix must be byte-identical to the dual artifact's `out` binding.
      const audioOnly = await client.call('load_patch_graph', { ...graph, taps: [] })
      expect(audioOnly.taps.map((tap: any) => tap.name)).toEqual(['out'])
      const audioOnlyRendered = await client.call('render_window', {
        start: 0,
        count: 32,
        slots: audioOnly.taps.map((tap: any) => tap.slot),
      })
      expect(audioOnlyRendered.values[0]).toEqual(rendered.values[0])

      const next = await client.call('load_patch_graph', graph)
      expect(next.program_version).toBeGreaterThan(audioOnly.program_version)
      const nextRendered = await client.call('render_window', {
        start: 0,
        count: 1,
        slots: next.taps.map((tap: any) => tap.slot),
      })
      expect(nextRendered.program_version).toBe(next.program_version)
      expect(nextRendered.control_version).toBe(next.control_version)
    } finally {
      client.stop()
    }
  })

  test('ordinary signal and modal inlets provide typed implicit fan-in', async () => {
    if (!existsSync(frontendBin))
      throw new Error(`frontend binary missing: ${frontendBin} (run \`make build lean\`)`)

    const client = new EngineClient()
    try {
      const vocabulary = await client.call('get_vocabulary')
      const port = (kind: string, name: string) =>
        vocabulary.kinds.find((entry: any) => entry.kind === kind)
          ?.ports.find((entry: any) => entry.name === name)

      expect(port('delay', 'in')?.multi).toBe(true)
      expect(port('phaser', 'in')).toMatchObject({ accepts: ['modal'], multi: true })
      expect(port('reverb', 'rt60')?.multi).toBe(false)

      const implicitSignal = {
        nodes: [
          { id: 'a', kind: 'source', params: { freq: 220 }, in: {} },
          { id: 'b', kind: 'source', params: { freq: 330 }, in: {} },
          { id: 'delay', kind: 'delay', params: { amount: 0.004 }, in: { in: ['a', 'b'] } },
          { id: 'out', kind: 'out', params: {}, in: { in: ['delay'] } },
        ],
        out: 'out',
        taps: [],
      }
      const explicitSignal = {
        nodes: [
          { id: 'a', kind: 'source', params: { freq: 220 }, in: {} },
          { id: 'b', kind: 'source', params: { freq: 330 }, in: {} },
          { id: 'mix', kind: 'mix', params: {}, in: { in: ['a', 'b'] } },
          { id: 'delay', kind: 'delay', params: { amount: 0.004 }, in: { in: ['mix'] } },
          { id: 'out', kind: 'out', params: {}, in: { in: ['delay'] } },
        ],
        out: 'out',
        taps: [],
      }

      const implicitReport = await client.call('load_patch_graph', implicitSignal)
      expect(implicitReport.inputs).toContainEqual({
        node: 'delay', port: 'in', state: 'wired', sources: ['a', 'b'],
      })
      const implicitRendered = await client.call('render_window', {
        start: 0, count: 64, slots: [implicitReport.taps[0].slot],
      })

      const explicitReport = await client.call('load_patch_graph', explicitSignal)
      const explicitRendered = await client.call('render_window', {
        start: 0, count: 64, slots: [explicitReport.taps[0].slot],
      })
      expect(implicitRendered.values[0]).toEqual(explicitRendered.values[0])

      const modalReport = await client.call('load_patch_graph', {
        nodes: [
          { id: 'low', kind: 'resonator', params: { freq: 180, decay: 4 }, in: {} },
          { id: 'high', kind: 'resonator', params: { freq: 360, decay: 3 }, in: {} },
          {
            id: 'phaser', kind: 'phaser',
            params: { center: 700, sweep: 1.5, rate: 0.2, mix: 0.5 },
            in: { in: ['low', 'high'] },
          },
          { id: 'out', kind: 'out', params: {}, in: { in: ['phaser'] } },
        ],
        out: 'out',
        taps: [],
      })
      expect(modalReport.inputs).toContainEqual({
        node: 'phaser', port: 'in', state: 'wired', sources: ['low', 'high'],
      })
      const modalRendered = await client.call('render_window', {
        start: 0, count: 64, slots: [modalReport.taps[0].slot],
      })
      expect(modalRendered.values[0].every(Number.isFinite)).toBe(true)

      await expect(client.call('load_patch_graph', {
        nodes: [
          { id: 'a', kind: 'source', params: { freq: 1 }, in: {} },
          { id: 'b', kind: 'source', params: { freq: 2 }, in: {} },
          { id: 'res', kind: 'resonator', params: { freq: 180, decay: 4 }, in: {} },
          {
            id: 'room', kind: 'reverb', params: {},
            in: { in: ['res'], rt60: ['a', 'b'] },
          },
          { id: 'out', kind: 'out', params: {}, in: { in: ['room'] } },
        ],
        out: 'out',
      })).rejects.toThrow('accepts one source')
    } finally {
      client.stop()
    }
  })
})
