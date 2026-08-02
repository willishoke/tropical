'use strict';

(function installPhaseLock(root, factory) {
  const api = factory()
  if (typeof module === 'object' && module.exports) module.exports = api
  if (root) root.ModalPhaseLock = api
}(typeof globalThis === 'undefined' ? this : globalThis, () => {
  function positiveZeroCrossing(values, searchLength, windowLength) {
    const lastStart = Math.max(0, values.length - windowLength - 1)
    const limit = Math.min(searchLength, lastStart + 1)
    if (limit < 2) return 0

    let peak = 0
    for (let index = 0; index < limit; index += 1) {
      peak = Math.max(peak, Math.abs(values[index] || 0))
    }
    if (peak < 1e-12) return 0

    // Arm below a relative floor so late, quiet modal tails lock as reliably
    // as fresh attacks. The actual lock remains the interpolated zero, not the
    // threshold crossing.
    const armLevel = peak * 0.08
    let armed = (values[0] || 0) < -armLevel
    for (let index = 1; index < limit; index += 1) {
      const previous = values[index - 1] || 0
      const current = values[index] || 0
      if (current < -armLevel) armed = true
      if (armed && previous < 0 && current >= 0) {
        const span = current - previous
        return (index - 1) + (span === 0 ? 0 : -previous / span)
      }
    }
    return Math.min(searchLength, lastStart)
  }

  function sampleLinear(values, position) {
    const left = Math.max(0, Math.min(values.length - 1, Math.floor(position)))
    const right = Math.min(values.length - 1, left + 1)
    const fraction = position - left
    return values[left] + (values[right] - values[left]) * fraction
  }

  function window(values, searchLength, windowLength) {
    if (!values || values.length < 2 || windowLength < 1) {
      return { offset: 0, values: [] }
    }
    const length = Math.min(windowLength, values.length)
    const offset = positiveZeroCrossing(values, searchLength, length)
    return {
      offset,
      values: Array.from({ length }, (_unused, index) => (
        sampleLinear(values, offset + index)
      )),
    }
  }

  return { positiveZeroCrossing, window }
}))
