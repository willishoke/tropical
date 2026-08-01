const test = require('node:test')
const assert = require('node:assert/strict')

const { LatestValueSender } = require('./renderer/latest-value-sender.js')

function deferred() {
  let resolve
  let reject
  const promise = new Promise((yes, no) => { resolve = yes; reject = no })
  return { promise, resolve, reject }
}

async function until(predicate) {
  for (let attempt = 0; attempt < 50; attempt += 1) {
    if (predicate()) return
    await new Promise((resolve) => setImmediate(resolve))
  }
  assert.fail('condition did not become true')
}

test('the first value sends immediately and an in-flight drag keeps only its latest value', async () => {
  const calls = []
  const waits = []
  const sender = new LatestValueSender((_key, value) => {
    calls.push(value)
    const wait = deferred()
    waits.push(wait)
    return wait.promise
  })

  sender.submit('veil', 100)
  assert.deepEqual(calls, [100])

  sender.submit('veil', 200)
  sender.submit('veil', 300)
  assert.deepEqual(calls, [100])

  waits[0].resolve()
  await until(() => calls.length === 2)
  assert.deepEqual(calls, [100, 300])

  waits[1].resolve()
  await sender.whenIdle('veil')
  assert.deepEqual(calls, [100, 300])
})

test('different parameters have independent in-flight lanes', async () => {
  const calls = []
  const waits = new Map()
  const sender = new LatestValueSender((key, value) => {
    calls.push([key, value])
    const wait = deferred()
    waits.set(key, wait)
    return wait.promise
  })

  sender.submit('veil', 700)
  sender.submit('edge', 0.4)
  assert.deepEqual(calls, [['veil', 700], ['edge', 0.4]])

  waits.get('veil').resolve()
  waits.get('edge').resolve()
  await Promise.all([sender.whenIdle('veil'), sender.whenIdle('edge')])
})

test('a fast out-and-back gesture preserves its turning point and endpoint', async () => {
  const calls = []
  const waits = []
  const sender = new LatestValueSender((_key, update) => {
    calls.push(update.targetValue)
    const wait = deferred()
    waits.push(wait)
    return wait.promise
  }, () => {}, (update) => update.targetValue)

  sender.submit('edge', { targetValue: 0.24 })
  sender.submit('edge', { targetValue: 0.82 })
  sender.submit('edge', { targetValue: 0.22 })
  assert.deepEqual(calls, [0.24])

  waits[0].resolve()
  await until(() => calls.length === 2)
  assert.deepEqual(calls, [0.24, 0.82])
  waits[1].resolve()
  await until(() => calls.length === 3)
  assert.deepEqual(calls, [0.24, 0.82, 0.22])
  waits[2].resolve()
  await sender.whenIdle('edge')
})

test('a failed value reports once and does not strand a newer value', async () => {
  const calls = []
  const errors = []
  const first = deferred()
  const sender = new LatestValueSender(async (_key, value) => {
    calls.push(value)
    if (value === 1) return first.promise
  }, (error, key, value) => errors.push([error.message, key, value]))

  sender.submit('length', 1)
  sender.submit('length', 2)
  first.reject(new Error('epoch refused'))
  await sender.whenIdle('length')

  assert.deepEqual(calls, [1, 2])
  assert.deepEqual(errors, [['epoch refused', 'length', 1]])
})
