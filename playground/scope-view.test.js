'use strict'

const test = require('node:test')
const assert = require('node:assert/strict')

const scopeView = require('./renderer/scope-view')
const scopeProfile = require('./renderer/scope-profile')

test('fixed volts-per-division preserves modal envelope amplitude', () => {
  assert.equal(scopeView.normalizedAmplitude(0.04, 0.04), 1)
  assert.equal(scopeView.normalizedAmplitude(0.01, 0.04), 0.25)
  assert.equal(scopeView.normalizedAmplitude(-0.01, 0.04), -0.25)
})

test('only active phase-locked traces are drawable', () => {
  const live = {
    active: true, locked: true, peak: 0.02, displayEnvelope: 0.02, values: [0, 0.02],
  }
  const dc = {
    active: true, locked: false, peak: 0.02, displayEnvelope: 0.02, values: [0.02, 0.02],
  }
  const silent = {
    active: false, locked: false, peak: 0, displayEnvelope: 0, values: [0, 0],
  }
  assert.deepEqual(scopeView.visibleFrames([live, dc, silent]), [live])
  assert.equal(scopeView.envelopeFraction([live, dc, silent], 0.04), 0.5)
})

test('log-period density narrows the pitch spread without erasing it', () => {
  assert.equal(scopeView.visibleCycles(55), 1.5)
  assert.equal(scopeView.visibleCycles(110), 2.25)
  assert.equal(scopeView.visibleCycles(220), 3)
  assert.ok(scopeView.visibleCycles(371.25) < 3.6)
})

test('trigger-cycle peak jitter cannot modulate the displayed envelope', () => {
  const envelope = 0.018
  assert.equal(scopeView.envelopeScaledValue(0.01, 0.02, envelope), 0.009)
  assert.equal(scopeView.envelopeScaledValue(0.015, 0.03, envelope), 0.009)
})

test('display-synchronized scheduling caps a 120 Hz panel at 60 scope frames', () => {
  let deadline = 0
  let frames = 0
  for (let index = 0; index <= 120; index += 1) {
    const decision = scopeView.frameDecision(deadline, index * 1000 / 120, 60)
    deadline = decision.next
    if (decision.due) frames += 1
  }
  assert.ok(frames >= 60 && frames <= 61, `got ${frames} frames`)
})

test('projection scope requests retain every source sample', () => {
  assert.equal(
    scopeProfile.pointBudget,
    scopeProfile.displaySamples + scopeProfile.searchSamples + scopeProfile.warmupSamples,
  )
  assert.equal(scopeProfile.fps, 60)
})
