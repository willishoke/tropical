'use strict';

(function installScopeProfile(root, factory) {
  const profile = factory()
  if (typeof module === 'object' && module.exports) module.exports = profile
  if (root) root.ModalScopeProfile = profile
}(typeof globalThis === 'undefined' ? this : globalThis, () => {
  const displaySamples = 896
  const timeCompression = 2
  // The visible span is twice the 896-sample base width. Another full base
  // width leaves more than one 55 Hz period of slack in which the centered
  // window can select a positive crossing without changing the time scale.
  const searchSamples = displaySamples * timeCompression
  const warmupSamples = 64
  return Object.freeze({
    displaySamples,
    searchSamples,
    warmupSamples,
    timeCompression,
    // The wider projection-only window remains below 10 ms p99 at full resolution.
    // Keeping stride=1 removes decimation/interpolation shimmer from the locks.
    pointBudget: displaySamples + searchSamples + warmupSamples,
    // requestAnimationFrame drives the product view; this cap follows ordinary
    // 60 Hz displays and avoids doubling scope work on 120 Hz panels.
    fps: 60,
  })
}))
