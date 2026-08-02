'use strict';

(function installScopeProfile(root, factory) {
  const profile = factory()
  if (typeof module === 'object' && module.exports) module.exports = profile
  if (root) root.ModalScopeProfile = profile
}(typeof globalThis === 'undefined' ? this : globalThis, () => {
  const displaySamples = 896
  // 1,152 plus the 896-point display leaves more than one complete 55 Hz
  // period in which a centered 1.5-cycle window can choose its crossing.
  const searchSamples = 1152
  const warmupSamples = 64
  return Object.freeze({
    displaySamples,
    searchSamples,
    warmupSamples,
    // The projection-only artifact measures about 4 ms p99 at full resolution.
    // Keeping stride=1 removes decimation/interpolation shimmer from the locks.
    pointBudget: displaySamples + searchSamples + warmupSamples,
    // requestAnimationFrame drives the product view; this cap follows ordinary
    // 60 Hz displays and avoids doubling scope work on 120 Hz panels.
    fps: 60,
  })
}))
