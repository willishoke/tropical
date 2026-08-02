'use strict'

const test = require('node:test')
const assert = require('node:assert/strict')

const scopeView = require('./renderer/scope-view')
const scopeProfile = require('./renderer/scope-profile')
const scopeFrame = require('./renderer/scope-frame')
const scene = require('./scene')

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

test('pointwise demodulation removes envelope slope before phase locking', () => {
  const carrier = Array.from({ length: 256 }, (_unused, index) => (
    Math.sin(0.11 * index + 0.7)
  ))
  const envelope = carrier.map((_unused, index) => (
    0.004 + 0.026 * Math.exp(-index / 47)
  ))
  const projection = carrier.map((value, index) => value * envelope[index])
  const extracted = scopeView.extractCarrier(projection, envelope)
  extracted.forEach((value, index) => {
    assert.ok(Math.abs(value - carrier[index]) < 1e-12)
  })
  assert.equal(scopeView.applyEnvelope(0.5, 0.018), 0.009)
})

test('envelope extraction refuses silent and invalid source samples', () => {
  assert.deepEqual(
    scopeView.extractCarrier([1, 2, NaN, 4], [0, 1e-14, 1, Infinity], 1e-12),
    [0, 0, 0, 0],
  )
})

test('batched scope envelopes equal the exact scalar envelope', () => {
  const first = 0.041
  const step = 1 / scene.SAMPLE_RATE
  const envelopes = scene.scopeEnvelopeSamples(0, 0, first, step, 2048)
  envelopes.forEach((value, index) => {
    assert.ok(Math.abs(
      value - scene.scopeEnvelopeAt(0, 0, first + index * step)
    ) < 1e-15)
  })
})

test('the shared production transform yields one invariant carrier', () => {
  const frequency = scene.voiceFrequency(scene.CHORDS[0], scene.CHORDS[0].voices[0])
  const tauBase = 0.4
  const values = scene.scopeEnvelopeSamples(
    0, 0, tauBase, 1 / scene.SAMPLE_RATE, scopeProfile.pointBudget,
  ).map((envelope, index) => (
    envelope * Math.sin(Math.PI * 2 * frequency
      * (tauBase + index / scene.SAMPLE_RATE) + 0.7)
  ))
  const frame = scopeFrame.fromProjection({
    values,
    responseStart: 0,
    stride: 1,
    warmupSamples: scopeProfile.warmupSamples,
    displaySamples: scopeProfile.displaySamples,
    frequency,
    chordIndex: 0,
    voiceIndex: 0,
    tauBase,
    velocity: 1,
    playbackPosition: scopeProfile.pointBudget,
  })
  const expected = frame.values.map((_value, index) => Math.sin(
    Math.PI * 2 * frame.cycles
      * (index - (frame.values.length - 1) / 2)
      / (frame.values.length - 1),
  ))
  frame.values.forEach((value, index) => {
    assert.ok(Math.abs(value - expected[index]) < 1e-4)
  })
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
