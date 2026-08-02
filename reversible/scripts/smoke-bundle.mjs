#!/usr/bin/env node

import { cpSync, existsSync, mkdtempSync, readdirSync, rmSync } from 'node:fs'
import { connect } from 'node:net'
import { tmpdir } from 'node:os'
import { basename, dirname, join, resolve } from 'node:path'
import { spawn } from 'node:child_process'

const scriptDir = dirname(new URL(import.meta.url).pathname)
const repoRoot = resolve(scriptDir, '../..')
const sourceApp = resolve(process.argv[2] ?? join(repoRoot, 'build/Reversible.app'))
const temporaryRoot = mkdtempSync(join(tmpdir(), 'reversible-bundle-smoke-'))
const relocatedApp = join(temporaryRoot, basename(sourceApp))
const tropicalRoot = join(relocatedApp, 'Contents/Resources/Tropical')
const frontend = join(tropicalRoot, 'frontend')
const carrier = join(
  tropicalRoot,
  'playground/assets/grouped-room/clouds-current-radii-mono-v1-44100.tgrm',
)
const manifest = join(
  tropicalRoot,
  'playground/assets/grouped-room/clouds-current-radii-mono-v1-44100.json',
)
const socketPath = join(temporaryRoot, 'engine.sock')

function requireFile(path) {
  if (!existsSync(path)) throw new Error(`bundle input is missing: ${path}`)
}

function filesBelow(path) {
  return readdirSync(path, { withFileTypes: true }).flatMap((entry) => {
    const child = join(path, entry.name)
    return entry.isDirectory() ? filesBelow(child) : [child]
  })
}

class RPCClient {
  constructor(socket) {
    this.socket = socket
    this.buffer = ''
    this.nextID = 1
    this.pending = new Map()
    socket.on('data', (chunk) => this.receive(chunk.toString()))
    socket.on('error', (error) => this.fail(error))
    socket.on('close', () => this.fail(new Error('engine socket closed')))
  }

  receive(text) {
    this.buffer += text
    let newline
    while ((newline = this.buffer.indexOf('\n')) >= 0) {
      const line = this.buffer.slice(0, newline).trim()
      this.buffer = this.buffer.slice(newline + 1)
      if (!line) continue
      const message = JSON.parse(line)
      const request = this.pending.get(message.id)
      if (!request) continue
      this.pending.delete(message.id)
      clearTimeout(request.timer)
      try {
        if (message.error) throw new Error(JSON.stringify(message.error))
        const envelope = message.result?.content?.[0]?.text
        if (typeof envelope !== 'string') request.resolve(message.result)
        else {
          const inner = JSON.parse(envelope)
          if (inner.status !== 'ok') {
            throw new Error(`${inner.error?.code}: ${inner.error?.message}`)
          }
          request.resolve(inner.data)
        }
      } catch (error) {
        request.reject(error)
      }
    }
  }

  call(method, params = {}) {
    const id = this.nextID++
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id)
        reject(new Error(`timeout: ${method}`))
      }, 60_000)
      this.pending.set(id, { resolve, reject, timer })
      this.socket.write(`${JSON.stringify({ jsonrpc: '2.0', id, method, params })}\n`)
    })
  }

  fail(error) {
    for (const request of this.pending.values()) {
      clearTimeout(request.timer)
      request.reject(error)
    }
    this.pending.clear()
  }

  close() {
    this.socket.destroy()
    this.fail(new Error('client closed'))
  }
}

async function connectWhenReady(engine) {
  for (let attempt = 0; attempt < 300; attempt += 1) {
    if (engine.exitCode !== null) throw new Error(`embedded engine exited ${engine.exitCode}`)
    if (existsSync(socketPath)) {
      const socket = await new Promise((resolveSocket) => {
        const candidate = connect({ path: socketPath })
        candidate.once('connect', () => resolveSocket(candidate))
        candidate.once('error', () => {
          candidate.destroy()
          resolveSocket(null)
        })
      })
      if (socket) return new RPCClient(socket)
    }
    await new Promise((resolveDelay) => setTimeout(resolveDelay, 100))
  }
  throw new Error(`could not connect to embedded engine at ${socketPath}`)
}

const modes = [
  [211, 7.5, 1, 0], [211, 95.5, 1, 0],
  [433, 8.65, 1, 0], [433, 100.35, 1, 0],
  [887, 9.8, 1, 0], [887, 105.2, 1, 0],
  [1511, 10.95, 1, 0], [1511, 110.05, 1, 0],
  [2837, 12.1, 1, 0], [2837, 114.9, 1, 0],
  [5081, 13.25, 1, 0], [5081, 119.75, 1, 0],
]

const node = (id, kind, params, inputs) => ({ id, kind, params, sel: {}, in: inputs })
const graph = {
  nodes: [
    node('hit', 'string', { t: 0, modes }, {}),
    node('position', 'glideknob', { value: 1 }, {}),
    node(
      'room',
      'groupedroom',
      { profile: 'clouds-current-radii-mono-v1' },
      { in: ['hit'], position: ['position'] },
    ),
    node('out', 'out', {}, { in: ['room'] }),
  ],
  out: 'out',
}

let engine
let client
try {
  requireFile(sourceApp)
  cpSync(sourceApp, relocatedApp, { recursive: true })
  for (const path of [frontend, join(tropicalRoot, 'libtropical.dylib'), carrier, manifest]) {
    requireFile(path)
  }
  const forbidden = filesBelow(relocatedApp).filter((path) => (
    path.endsWith('-scene-44100.f32le') || path.includes('groupedroomcache')
  ))
  if (forbidden.length) throw new Error(`bundle contains fixed wet-score data: ${forbidden}`)

  engine = spawn(frontend, ['--serve', socketPath], {
    cwd: tropicalRoot,
    stdio: ['ignore', 'ignore', 'inherit'],
  })
  client = await connectWhenReady(engine)
  const vocabulary = await client.call('get_vocabulary')
  const realized = await client.call('load_patch_graph', graph)
  const room = realized.nodes?.find((item) => item.id === 'room')
  if (vocabulary.fingerprint !== realized.vocabulary_fingerprint) {
    throw new Error('compile fingerprint differs from decoded vocabulary')
  }
  if (room?.kind !== 'groupedroom' || room?.status !== 'active') {
    throw new Error(`live room was not active: ${JSON.stringify(room)}`)
  }
  console.log(JSON.stringify({
    schema: 'reversible_bundle_smoke_1',
    relocated_app: relocatedApp,
    vocabulary_fingerprint: vocabulary.fingerprint,
    program_version: realized.program_version,
    control_version: realized.control_version,
    taps: realized.taps?.map((tap) => tap.name),
    room,
    fixed_wet_score_files: 0,
    status: 'pass',
  }))
} finally {
  client?.close()
  engine?.kill('SIGTERM')
  rmSync(temporaryRoot, { recursive: true, force: true })
}
