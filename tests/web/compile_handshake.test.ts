/** Atomic load_patch_graph handshake: one reply identifies the vocabulary,
 * exact immutable runtime generation, realized graph facts, and complete tap
 * bindings. list_scope_taps remains a compatible diagnostic, not a required
 * second phase of adoption. */

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
    this.socketPath = join(tmpdir(), `tropical-handshake-${process.pid}.sock`)
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
    // A disconnected, structural-only scope projection mirrors the production
    // scene's dedicated modal probes and exercises atomic dual-artifact load.
    { id: 'scopeProbe', kind: 'string', params: { freq: 220, decay: 1, t: 0 }, sel: {}, in: {} },
    { id: 'out', kind: 'out', params: {}, sel: {}, in: { in: ['src'] } },
  ],
  out: 'out',
  taps: ['scopeProbe'],
}

describe('load_patch_graph compile handshake', () => {
  test('one response binds realized state and taps to its exact generation', async () => {
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
      expect(report.taps.map((tap: any) => tap.name)).toEqual(['out', 'scopeProbe'])

      // The first scope frame proves these bindings and versions identify the
      // very immutable image published by the compile, with no list call.
      const rendered = await client.call('render_window', {
        start: 0,
        count: 4,
        slots: report.taps.map((tap: any) => tap.slot),
      })
      expect(rendered.program_version).toBe(report.program_version)
      expect(rendered.control_version).toBe(report.control_version)

      // Retain the old verb as a compatible diagnostic for older clients.
      const listed = await client.call('list_scope_taps')
      expect(listed.taps).toEqual(report.taps)

      const next = await client.call('load_patch_graph', graph)
      expect(next.program_version).toBeGreaterThan(report.program_version)
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
})
