'use strict'

const test = require('node:test')
const assert = require('node:assert/strict')
const phaseLock = require('./renderer/phase-lock')

function sine(frequency, phase, amplitude = 1, count = 512) {
  return Array.from({ length: count }, (_unused, index) => (
    amplitude * Math.sin(phase + index * frequency)
  ))
}

test('each overlaid mode receives its own positive-going phase lock', () => {
  const low = phaseLock.window(sine(0.11, 2.2), 256, 128)
  const high = phaseLock.window(sine(0.37, -1.4), 256, 128)
  assert.ok(Math.abs(low.values[0]) < 1e-12)
  assert.ok(Math.abs(high.values[0]) < 1e-12)
  assert.ok(low.values[1] > 0)
  assert.ok(high.values[1] > 0)
  assert.notEqual(low.offset, high.offset)
})

test('relative arming keeps a quiet modal tail phase locked', () => {
  const loud = phaseLock.window(sine(0.19, 1.7, 1), 256, 128)
  const quiet = phaseLock.window(sine(0.19, 1.7, 1e-7), 256, 128)
  assert.ok(Math.abs(loud.offset - quiet.offset) < 1e-12)
  assert.ok(Math.abs(quiet.values[0]) < 1e-18)
})

test('the freshest eligible crossing keeps the envelope close to audible-now', () => {
  const locked = phaseLock.window(sine(0.11, 2.2), 256, 128)
  const period = Math.PI * 2 / 0.11
  assert.equal(locked.locked, true)
  assert.ok(locked.offset > 256 - period - 2)
  assert.ok(locked.offset < 256)
})

test('a fixed scale sees the modal envelope decay across locked frames', () => {
  const loud = phaseLock.window(sine(0.19, 1.7, 1), 256, 128)
  const tail = phaseLock.window(sine(0.19, 1.7, 0.25), 256, 128)
  assert.equal(loud.locked, true)
  assert.equal(tail.locked, true)
  assert.ok(Math.abs(tail.peak / loud.peak - 0.25) < 1e-12)
})

test('silence and DC cannot masquerade as phase-locked modes', () => {
  const silent = phaseLock.window(Array(512).fill(0), 256, 128)
  const dc = phaseLock.window(Array(512).fill(0.02), 256, 128)
  assert.equal(silent.active, false)
  assert.equal(silent.locked, false)
  assert.equal(dc.active, true)
  assert.equal(dc.locked, false)
})

test('centered locks put every positive crossing at the middle graticule', () => {
  const low = phaseLock.centeredWindow(sine(0.04, 1.2, 1, 2048), 896, 1200)
  const high = phaseLock.centeredWindow(sine(0.22, -2.1, 1, 2048), 896, 430)
  for (const frame of [low, high]) {
    assert.equal(frame.locked, true)
    assert.ok(Math.abs(frame.centerValue) < 1e-12)
    assert.ok(frame.values[447] < 0)
    assert.ok(frame.values[448] > 0)
  }
})
