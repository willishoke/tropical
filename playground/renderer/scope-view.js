'use strict';

(function installScopeView(root, factory) {
  const api = factory()
  if (typeof module === 'object' && module.exports) module.exports = api
  if (root) root.ModalScopeView = api
}(typeof globalThis === 'undefined' ? this : globalThis, () => {
  function visibleFrames(frames) {
    return frames.filter((frame) => (
      frame?.active === true
      && frame?.locked === true
      && frame?.displayEnvelope > 1e-12
      && Array.isArray(frame.values)
      && frame.values.length >= 2
    ))
  }

  function visibleCycles(frequency, sampleRate, displaySamples, timeCompression = 1) {
    if (!(frequency > 0) || !(sampleRate > 0)
      || !(displaySamples > 0) || !(timeCompression > 0)) return 0
    return frequency * displaySamples * timeCompression / sampleRate
  }

  function normalizedAmplitude(value, fullScale) {
    if (!Number.isFinite(value) || !(fullScale > 0)) return 0
    return Math.max(-1, Math.min(1, value / fullScale))
  }

  function envelopeFraction(frames, fullScale) {
    if (!(fullScale > 0)) return 0
    const peak = visibleFrames(frames).reduce(
      (maximum, frame) => Math.max(maximum, frame.displayEnvelope || 0), 0,
    )
    return Math.max(0, Math.min(1, peak / fullScale))
  }

  function extractCarrier(values, envelopes, envelopeFloor = 1e-12) {
    if (!Array.isArray(values) || !Array.isArray(envelopes)) return []
    return values.map((value, index) => {
      const envelope = envelopes[index]
      if (!Number.isFinite(value) || !Number.isFinite(envelope)
        || Math.abs(envelope) <= envelopeFloor) return 0
      return value / envelope
    })
  }

  function applyEnvelope(carrierValue, displayEnvelope) {
    if (!Number.isFinite(carrierValue) || !(displayEnvelope >= 0)) return 0
    return carrierValue * displayEnvelope
  }

  function frameDecision(deadline, timestamp, fps) {
    if (!Number.isFinite(timestamp) || !(fps > 0)) {
      return { due: false, next: deadline || 0 }
    }
    const period = 1000 / fps
    const target = deadline || timestamp
    if (timestamp + 0.5 < target) return { due: false, next: target }
    return {
      due: true,
      next: timestamp - target > period ? timestamp + period : target + period,
    }
  }

  return {
    visibleFrames,
    visibleCycles,
    normalizedAmplitude,
    envelopeFraction,
    extractCarrier,
    applyEnvelope,
    frameDecision,
  }
}))
