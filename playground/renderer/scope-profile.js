'use strict';

(function installScopeProfile(root, factory) {
  const profile = factory()
  if (typeof module === 'object' && module.exports) module.exports = profile
  if (root) root.ModalScopeProfile = profile
}(typeof globalThis === 'undefined' ? this : globalThis, () => {
  const displaySamples = 896
  const searchSamples = 896
  const warmupSamples = 64
  return Object.freeze({
    displaySamples,
    searchSamples,
    warmupSamples,
    // The projection-only artifact measures about 4 ms p99 at full resolution.
    // Keeping stride=1 removes decimation/interpolation shimmer from the locks.
    pointBudget: displaySamples + searchSamples + warmupSamples,
    fps: 24,
  })
}))
