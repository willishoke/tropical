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
