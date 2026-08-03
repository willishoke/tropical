#!/usr/bin/env node

import { cpSync, existsSync, mkdtempSync, readdirSync, rmSync } from 'node:fs'
import { connect } from 'node:net'
import { tmpdir } from 'node:os'
import { basename, dirname, join, relative, resolve } from 'node:path'
import { spawn } from 'node:child_process'

const scriptDir = dirname(new URL(import.meta.url).pathname)
const repoRoot = resolve(scriptDir, '../..')
const sourceApp = resolve(process.argv[2] ?? join(repoRoot, 'build/Reversible.app'))
const temporaryRoot = mkdtempSync(join(tmpdir(), 'reversible-bundle-smoke-'))
const relocatedApp = join(temporaryRoot, basename(sourceApp))
const tropicalRoot = join(relocatedApp, 'Contents/Resources/Tropical')
const frontend = join(tropicalRoot, 'frontend')
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

const node = (id, kind, params, inputs) => ({ id, kind, params, sel: {}, in: inputs })
const graph = {
  nodes: [
    node('tone', 'source', { freq: 220, morph: 0 }, {}),
    node('out', 'out', {}, { in: ['tone'] }),
  ],
  out: 'out',
}

let engine
let client
try {
  requireFile(sourceApp)
  cpSync(sourceApp, relocatedApp, { recursive: true })
  for (const path of [frontend, join(tropicalRoot, 'libtropical.dylib')]) {
    requireFile(path)
  }
  const runtimeFiles = filesBelow(tropicalRoot)
    .map((path) => relative(tropicalRoot, path))
    .sort()
  const expectedRuntimeFiles = ['frontend', 'libtropical.dylib']
  if (JSON.stringify(runtimeFiles) !== JSON.stringify(expectedRuntimeFiles)) {
    throw new Error(`unexpected embedded runtime files: ${runtimeFiles}`)
  }

  engine = spawn(frontend, ['--serve', socketPath], {
    cwd: tropicalRoot,
    stdio: ['ignore', 'ignore', 'inherit'],
  })
  client = await connectWhenReady(engine)
  const vocabulary = await client.call('get_vocabulary')
  const realized = await client.call('load_patch_graph', graph)
  const tone = realized.nodes?.find((item) => item.id === 'tone')
  if (typeof vocabulary.fingerprint !== 'string' || !vocabulary.fingerprint.startsWith('fnv1a64:')) {
    throw new Error(`invalid vocabulary fingerprint: ${vocabulary.fingerprint}`)
  }
  if (realized.ok !== true || tone?.kind !== 'source' || tone?.status !== 'active') {
    throw new Error(`canonical source patch was not active: ${JSON.stringify(realized)}`)
  }
  console.log(JSON.stringify({
    schema: 'reversible_bundle_smoke_1',
    relocated_app: relocatedApp,
    vocabulary_fingerprint: vocabulary.fingerprint,
    taps: realized.taps?.map((tap) => tap.name),
    node: tone,
    packaged_runtime_files: runtimeFiles,
    status: 'pass',
  }))
} finally {
  client?.close()
  engine?.kill('SIGTERM')
  rmSync(temporaryRoot, { recursive: true, force: true })
}
