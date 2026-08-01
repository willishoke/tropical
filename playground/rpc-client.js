'use strict'

const { connect: nodeConnect } = require('node:net')

function unwrapResult(message) {
  if (message.error) throw new Error(JSON.stringify(message.error))
  const result = message.result
  const text = result?.content?.[0]?.text
  if (typeof text !== 'string') return result
  const inner = JSON.parse(text)
  if (inner.status === 'ok') return inner.data
  throw new Error(`${inner.error?.code}: ${inner.error?.message}`)
}

// One reconnecting newline-JSON-RPC lane. Instances deliberately own their
// request ids, buffers, pending promises, and reconnect state independently:
// a long scope response can never serialize a later control line in Electron.
class JsonLineRpcClient {
  constructor({
    path,
    connect = nodeConnect,
    retryMs = 100,
    requestTimeout = (method) => method === 'load_patch_graph' ? 300000 : 15000,
  }) {
    this.path = path
    this.connectSocket = connect
    this.retryMs = retryMs
    this.requestTimeout = requestTimeout
    this.socket = null
    this.ready = false
    this.closed = false
    this.retryTimer = null
    this.buffer = ''
    this.nextId = 1
    this.pending = new Map()
    this.outbox = []
  }

  start() {
    if (this.closed || this.socket || this.retryTimer) return
    let socket
    try {
      socket = this.connectSocket({ path: this.path })
    } catch {
      this.scheduleReconnect()
      return
    }
    this.socket = socket
    this.ready = false

    socket.on('connect', () => {
      if (this.closed || this.socket !== socket) return
      this.ready = true
      const queued = this.outbox
      this.outbox = []
      for (const request of queued) {
        if (this.pending.has(request.id)) socket.write(request.line)
      }
    })
    socket.on('data', (chunk) => this.handleData(socket, chunk))
    socket.on('error', () => this.disconnect(socket))
    socket.on('close', () => this.disconnect(socket))
  }

  scheduleReconnect() {
    if (this.closed || this.retryTimer) return
    this.retryTimer = setTimeout(() => {
      this.retryTimer = null
      this.start()
    }, this.retryMs)
  }

  disconnect(socket) {
    if (this.socket !== socket) return
    this.socket = null
    this.ready = false
    this.buffer = ''
    try { socket.destroy() } catch {}
    this.scheduleReconnect()
  }

  handleData(socket, chunk) {
    if (this.socket !== socket) return
    this.buffer += chunk.toString()
    let newline
    while ((newline = this.buffer.indexOf('\n')) >= 0) {
      const line = this.buffer.slice(0, newline).trim()
      this.buffer = this.buffer.slice(newline + 1)
      if (!line) continue
      let message
      try { message = JSON.parse(line) } catch { continue }
      const request = this.pending.get(message.id)
      if (!request) continue
      this.pending.delete(message.id)
      clearTimeout(request.timer)
      try { request.resolve(unwrapResult(message)) }
      catch (error) { request.reject(error) }
    }
  }

  call(method, params = {}) {
    if (this.closed) return Promise.reject(new Error('RPC client closed'))
    const id = this.nextId++
    const line = JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n'
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        if (!this.pending.delete(id)) return
        this.outbox = this.outbox.filter((request) => request.id !== id)
        reject(new Error(`timeout: ${method}`))
      }, this.requestTimeout(method))
      this.pending.set(id, { resolve, reject, timer })
      if (this.ready) this.socket.write(line)
      else {
        this.outbox.push({ id, line })
        this.start()
      }
    })
  }

  rejectAll(error) {
    for (const request of this.pending.values()) {
      clearTimeout(request.timer)
      request.reject(error)
    }
    this.pending.clear()
    this.outbox = []
  }

  close(error = new Error('RPC client closed')) {
    if (this.closed) return
    this.closed = true
    if (this.retryTimer) clearTimeout(this.retryTimer)
    this.retryTimer = null
    const socket = this.socket
    this.socket = null
    this.ready = false
    try { socket?.destroy() } catch {}
    this.rejectAll(error)
  }
}

module.exports = { JsonLineRpcClient, unwrapResult }
