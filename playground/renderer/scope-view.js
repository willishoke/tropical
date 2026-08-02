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

  function visibleCycles(frequency, baseFrequency = 55) {
    if (!(frequency > 0) || !(baseFrequency > 0)) return 1.5
    return Math.max(1.5, Math.min(
      3.75,
      1.5 + 0.75 * Math.log2(frequency / baseFrequency),
    ))
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

  function envelopeScaledValue(value, observedPeak, displayEnvelope) {
    if (!Number.isFinite(value) || !(observedPeak > 0)
      || !(displayEnvelope >= 0)) return 0
    return value / observedPeak * displayEnvelope
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
    envelopeScaledValue,
    frameDecision,
  }
}))
