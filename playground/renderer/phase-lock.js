'use strict';

(function installPhaseLock(root, factory) {
  const api = factory()
  if (typeof module === 'object' && module.exports) module.exports = api
  if (root) root.ModalPhaseLock = api
}(typeof globalThis === 'undefined' ? this : globalThis, () => {
  function positiveZeroCrossing(values, searchLength, windowLength) {
    const lastStart = Math.max(0, values.length - windowLength - 1)
    const limit = Math.min(searchLength, lastStart + 1)
    if (limit < 2) return -1

    let peak = 0
    for (let index = 0; index < limit; index += 1) {
      const value = Number.isFinite(values[index]) ? values[index] : 0
      peak = Math.max(peak, Math.abs(value))
    }
    if (peak < 1e-12) return -1

    // Arm below a relative floor so late, quiet modal tails lock as reliably
    // as fresh attacks. The actual lock remains the interpolated zero, not the
    // threshold crossing.
    const armLevel = peak * 0.08
    let armed = (Number.isFinite(values[0]) ? values[0] : 0) < -armLevel
    let crossing = -1
    for (let index = 1; index < limit; index += 1) {
      const previous = Number.isFinite(values[index - 1]) ? values[index - 1] : 0
      const current = Number.isFinite(values[index]) ? values[index] : 0
      if (current < -armLevel) armed = true
      if (armed && previous < 0 && current >= 0) {
        const span = current - previous
        crossing = (index - 1) + (span === 0 ? 0 : -previous / span)
      }
    }
    // The freshest crossing leaves the display closest to audible-now. Using
    // the first crossing wastes most of the search window and adds a nearly
    // full display-window of envelope latency.
    return crossing
  }

  function sampleLinear(values, position) {
    const left = Math.max(0, Math.min(values.length - 1, Math.floor(position)))
    const right = Math.min(values.length - 1, left + 1)
    const fraction = position - left
    const leftValue = Number.isFinite(values[left]) ? values[left] : 0
    const rightValue = Number.isFinite(values[right]) ? values[right] : 0
    return leftValue + (rightValue - leftValue) * fraction
  }

  function window(values, searchLength, windowLength) {
    if (!values || values.length < 2 || windowLength < 1) {
      return { offset: 0, values: [], peak: 0, active: false, locked: false }
    }
    const length = Math.min(windowLength, values.length)
    const crossing = positiveZeroCrossing(values, searchLength, length)
    const lastStart = Math.max(0, values.length - length - 1)
    const offset = crossing >= 0 ? crossing : Math.min(searchLength, lastStart)
    const lockedValues = Array.from({ length }, (_unused, index) => (
      sampleLinear(values, offset + index)
    ))
    const peak = lockedValues.reduce(
      (maximum, value) => Math.max(maximum, Math.abs(value)), 0,
    )
    return {
      offset,
      values: lockedValues,
      peak,
      active: peak >= 1e-12,
      locked: crossing >= 0,
    }
  }

  return { positiveZeroCrossing, window }
}))
