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
  const live = { active: true, locked: true, peak: 0.02, values: [0, 0.02] }
  const dc = { active: true, locked: false, peak: 0.02, values: [0.02, 0.02] }
  const silent = { active: false, locked: false, peak: 0, values: [0, 0] }
  assert.deepEqual(scopeView.visibleFrames([live, dc, silent]), [live])
  assert.equal(scopeView.envelopeFraction([live, dc, silent], 0.04), 0.5)
})

test('projection scope requests retain every source sample', () => {
  assert.equal(
    scopeProfile.pointBudget,
    scopeProfile.displaySamples + scopeProfile.searchSamples + scopeProfile.warmupSamples,
  )
})
