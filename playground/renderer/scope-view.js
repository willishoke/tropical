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
      && Array.isArray(frame.values)
      && frame.values.length >= 2
    ))
  }

  function normalizedAmplitude(value, fullScale) {
    if (!Number.isFinite(value) || !(fullScale > 0)) return 0
    return Math.max(-1, Math.min(1, value / fullScale))
  }

  function envelopeFraction(frames, fullScale) {
    if (!(fullScale > 0)) return 0
    const peak = visibleFrames(frames).reduce(
      (maximum, frame) => Math.max(maximum, frame.peak || 0), 0,
    )
    return Math.max(0, Math.min(1, peak / fullScale))
  }

  return { visibleFrames, normalizedAmplitude, envelopeFraction }
}))
