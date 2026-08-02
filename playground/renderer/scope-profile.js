'use strict';

(function installScopeProfile(root, factory) {
  const profile = factory()
  if (typeof module === 'object' && module.exports) module.exports = profile
  if (root) root.ModalScopeProfile = profile
}(typeof globalThis === 'undefined' ? this : globalThis, () => Object.freeze({
  displaySamples: 896,
  searchSamples: 896,
  warmupSamples: 64,
  pointBudget: 384,
  fps: 24,
})))
