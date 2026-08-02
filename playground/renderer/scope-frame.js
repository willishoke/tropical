'use strict';

(function installScopeFrame(root, factory) {
  const api = factory(
    root?.ModalScene ?? (typeof require === 'function' ? require('../scene.js') : null),
    root?.ModalPhaseLock
      ?? (typeof require === 'function' ? require('./phase-lock.js') : null),
    root?.ModalScopeView
      ?? (typeof require === 'function' ? require('./scope-view.js') : null),
  )
  if (typeof module === 'object' && module.exports) module.exports = api
  if (root) root.ModalScopeFrame = api
}(typeof globalThis === 'undefined' ? this : globalThis, (scene, phaseLock, scopeView) => {
  function fromProjection({
    values,
    responseStart,
    stride,
    warmupSamples,
    displaySamples,
    frequency,
    chordIndex,
    voiceIndex,
    tauBase,
    velocity,
    playbackPosition,
  }) {
    const safeStride = stride ?? 1
    const warmupPoints = Math.ceil(warmupSamples / safeStride)
    const displayPoints = Math.ceil(displaySamples / safeStride)
    const raw = (values || []).slice(warmupPoints)
    const firstSample = responseStart + warmupPoints * safeStride
    const envelopes = scene.scopeEnvelopeSamples(
      chordIndex,
      voiceIndex,
      tauBase + velocity * firstSample / scene.SAMPLE_RATE,
      velocity * safeStride / scene.SAMPLE_RATE,
      raw.length,
    )
    const carrier = scopeView.extractCarrier(
      raw, envelopes, scene.SCOPE_FULL_SCALE * 1e-6,
    )
    const cycles = scopeView.visibleCycles(frequency, scene.BASE_HZ)
    const periodPoints = scene.SAMPLE_RATE / frequency / safeStride
    return {
      stride: safeStride,
      cycles,
      displayEnvelope: scene.scopeEnvelopeAt(
        chordIndex,
        voiceIndex,
        tauBase + velocity * playbackPosition / scene.SAMPLE_RATE,
      ),
      ...phaseLock.centeredWindow(
        carrier, displayPoints, cycles * periodPoints,
      ),
    }
  }

  return { fromProjection }
}))
