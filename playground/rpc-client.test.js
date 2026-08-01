'use strict'

const test = require('node:test')
const assert = require('node:assert/strict')
const { EventEmitter } = require('node:events')
const { JsonLineRpcClient } = require('./rpc-client')

class FakeSocket extends EventEmitter {
  constructor() {
    super()
    this.writes = []
    this.destroyed = false
  }

  write(line) { this.writes.push(line) }
  destroy() { this.destroyed = true }
}

function lane(socket, options = {}) {
  return new JsonLineRpcClient({
    path: '/tmp/test.sock',
    connect: () => socket,
    retryMs: 1,
    requestTimeout: () => 1000,
    ...options,
  })
}

test('independent lanes own ids, buffering, and chunked responses', async () => {
  const controlSocket = new FakeSocket()
  const scopeSocket = new FakeSocket()
  const control = lane(controlSocket)
  const scope = lane(scopeSocket)
  const controlResult = control.call('set_param', { name: 'x', value: 1 })
  const scopeResult = scope.call('render_window', { start: 0, count: 4 })
  assert.equal(controlSocket.writes.length, 0)
  assert.equal(scopeSocket.writes.length, 0)

  controlSocket.emit('connect')
  scopeSocket.emit('connect')
  assert.equal(JSON.parse(controlSocket.writes[0]).id, 1)
  assert.equal(JSON.parse(scopeSocket.writes[0]).id, 1)
  controlSocket.emit('data', '{"jsonrpc":"2.0","id":1,"result":{"value":')
  controlSocket.emit('data', '1}}\n')
  scopeSocket.emit('data', '{"jsonrpc":"2.0","id":1,"result":{"values":[[0]]}}\n')
  assert.deepEqual(await controlResult, { value: 1 })
  assert.deepEqual(await scopeResult, { values: [[0]] })
  control.close()
  scope.close()
})

test('disconnect reconnects future work without replaying an ambiguous request', async () => {
  const first = new FakeSocket()
  const second = new FakeSocket()
  const sockets = [first, second]
  const client = new JsonLineRpcClient({
    path: '/tmp/test.sock',
    connect: () => sockets.shift(),
    retryMs: 1,
    requestTimeout: () => 1000,
  })
  client.start()
  first.emit('connect')
  first.emit('error', new Error('lost'))
  await new Promise((resolve) => setTimeout(resolve, 5))
  second.emit('connect')
  const result = client.call('get_telemetry')
  assert.equal(JSON.parse(second.writes[0]).id, 1)
  second.emit('data', '{"jsonrpc":"2.0","id":1,"result":{"ok":true}}\n')
  assert.deepEqual(await result, { ok: true })
  client.close()
})

test('closing a lane rejects every pending request', async () => {
  const socket = new FakeSocket()
  const client = lane(socket)
  const pending = client.call('set_param')
  client.close(new Error('engine exited'))
  await assert.rejects(pending, /engine exited/)
})
