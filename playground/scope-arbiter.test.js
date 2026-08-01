'use strict'

const test = require('node:test')
const assert = require('node:assert/strict')
const { ScopeArbiter } = require('./renderer/scope-arbiter')

test('pointer, sender, and quiet windows freeze-hold scope work', () => {
  let now = 1000
  const arbiter = new ScopeArbiter({ quietMs: 100, now: () => now })
  assert.equal(arbiter.isHeld(), false)
  arbiter.begin('pointer:1')
  assert.equal(arbiter.isHeld(), true)
  arbiter.setSenderBusy(true)
  arbiter.end('pointer:1')
  assert.equal(arbiter.isHeld(), true)
  arbiter.setSenderBusy(false)
  now = 1099
  assert.equal(arbiter.isHeld(), true)
  now = 1100
  assert.equal(arbiter.isHeld(), false)
})

test('cancel and blur cannot strand a transient hold', () => {
  let now = 0
  const arbiter = new ScopeArbiter({ quietMs: 100, now: () => now })
  arbiter.begin('pointer:7')
  arbiter.begin('key:ArrowRight')
  arbiter.clearTransient()
  assert.equal(arbiter.isHeld(), true)
  now = 100
  assert.equal(arbiter.isHeld(), false)
})

test('a surfaced fault keeps the last complete scope frame frozen', () => {
  const arbiter = new ScopeArbiter({ quietMs: 100, now: () => 1000 })
  arbiter.fault()
  assert.equal(arbiter.isHeld(), true)
})
