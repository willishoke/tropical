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

  function centeredWindow(values, outputLength, spanLength) {
    if (!values || values.length < 2 || outputLength < 2 || !(spanLength > 0)) {
      return {
        offset: 0,
        center: 0,
        span: 0,
        values: [],
        peak: 0,
        active: false,
        locked: false,
      }
    }

    const length = Math.max(2, Math.floor(outputLength))
    const span = Math.min(spanLength, values.length - 1)
    const halfSpan = span / 2
    const target = (values.length - 1) / 2
    const minimumCenter = halfSpan
    const maximumCenter = values.length - 1 - halfSpan

    let searchPeak = 0
    for (const raw of values) {
      const value = Number.isFinite(raw) ? raw : 0
      searchPeak = Math.max(searchPeak, Math.abs(value))
    }

    let crossing = -1
    if (searchPeak >= 1e-12 && minimumCenter <= maximumCenter) {
      const armLevel = searchPeak * 0.08
      let armed = (Number.isFinite(values[0]) ? values[0] : 0) < -armLevel
      let bestDistance = Infinity
      for (let index = 1; index < values.length; index += 1) {
        const previous = Number.isFinite(values[index - 1]) ? values[index - 1] : 0
        const current = Number.isFinite(values[index]) ? values[index] : 0
        if (current < -armLevel) armed = true
        if (armed && previous < 0 && current >= 0) {
          const width = current - previous
          const candidate = (index - 1) + (width === 0 ? 0 : -previous / width)
          if (candidate >= minimumCenter && candidate <= maximumCenter) {
            const distance = Math.abs(candidate - target)
            if (distance < bestDistance) {
              crossing = candidate
              bestDistance = distance
            }
          }
        }
      }
      // A steep attack can put every in-bounds negative lobe below the
      // full-window hysteresis even though its zero is perfectly usable.
      // Fall back to an unarmed sign crossing; DC still cannot pass because it
      // never changes sign.
      if (crossing < 0) {
        for (let index = 1; index < values.length; index += 1) {
          const previous = Number.isFinite(values[index - 1]) ? values[index - 1] : 0
          const current = Number.isFinite(values[index]) ? values[index] : 0
          if (previous < 0 && current >= 0) {
            const width = current - previous
            const candidate = (index - 1) + (width === 0 ? 0 : -previous / width)
            if (candidate >= minimumCenter && candidate <= maximumCenter
              && Math.abs(candidate - target) < bestDistance) {
              crossing = candidate
              bestDistance = Math.abs(candidate - target)
            }
          }
        }
      }
    }

    const center = crossing >= 0
      ? crossing
      : Math.max(minimumCenter, Math.min(maximumCenter, target))
    const start = center - halfSpan
    const step = span / (length - 1)
    const lockedValues = Array.from({ length }, (_unused, index) => (
      sampleLinear(values, start + index * step)
    ))
    const peak = lockedValues.reduce(
      (maximum, value) => Math.max(maximum, Math.abs(value)), 0,
    )
    return {
      offset: start,
      center,
      span,
      centerValue: sampleLinear(values, center),
      values: lockedValues,
      peak,
      active: peak >= 1e-12,
      locked: crossing >= 0,
    }
  }

  return { positiveZeroCrossing, window, centeredWindow }
}))
