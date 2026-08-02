import Foundation

/// Fixed sampling geometry of the qualified five-mode phase view.
///
/// The canvas contains 896 points spanning twice that base width: 1,792 source
/// samples, or 40.6349206 ms at 44.1 kHz. The extra 1,792-sample search region
/// lets each mode choose its own centered positive-going zero crossing without
/// changing the shared linear time axis.
struct ModalScopeProfile: Equatable, Sendable {
    let sampleRate: Double
    let displaySamples: Int
    let searchSamples: Int
    let warmupSamples: Int
    let timeCompression: Double
    let pointBudget: Int
    let framesPerSecond: Double

    static let qualified = ModalScopeProfile(
        sampleRate: 44_100,
        displaySamples: 896,
        searchSamples: 1_792,
        warmupSamples: 64,
        timeCompression: 2,
        pointBudget: 896 + 1_792 + 64,
        framesPerSecond: 60
    )

    var linearSpanSamples: Double {
        Double(displaySamples) * timeCompression
    }

    var linearSpanSeconds: Double {
        linearSpanSamples / sampleRate
    }
}

/// The exact +a/-a exponential pair used by one projected fundamental.
struct ModalPairedEnvelope: Equatable, Sendable {
    let strikeTime: Double
    let slowDecay: Double
    let fastDecay: Double
    let amplitude: Double

    func value(at sceneTime: Double) -> Double {
        guard sceneTime.isFinite else { return 0 }
        let age = sceneTime - strikeTime
        guard age > 0 else { return 0 }
        return abs(amplitude) * (
            exp(-slowDecay * age) - exp(-fastDecay * age)
        )
    }

    func samples(firstSceneTime: Double, sceneStep: Double, count: Int) -> [Double] {
        guard firstSceneTime.isFinite, sceneStep.isFinite, count > 0 else { return [] }
        return (0..<count).map { index in
            value(at: firstSceneTime + Double(index) * sceneStep)
        }
    }

    var peak: Double {
        guard slowDecay > 0, fastDecay > slowDecay else { return 0 }
        let peakTime = log(fastDecay / slowDecay) / (fastDecay - slowDecay)
        return abs(amplitude) * (
            exp(-slowDecay * peakTime) - exp(-fastDecay * peakTime)
        )
    }
}

struct ModalScopeLock: Equatable, Sendable {
    let offset: Double
    let center: Double
    let span: Double
    let centerValue: Double
    let values: [Double]
    let peak: Double
    let active: Bool
    let locked: Bool

    static let empty = ModalScopeLock(
        offset: 0,
        center: 0,
        span: 0,
        centerValue: 0,
        values: [],
        peak: 0,
        active: false,
        locked: false
    )
}

struct ModalScopeFrame: Equatable, Sendable {
    let stride: Int
    let cycles: Double
    let timeSpanSeconds: Double
    let displayEnvelope: Double
    let lock: ModalScopeLock

    /// Restore the exact audible-now envelope after phase selection.
    var displayedValues: [Double] {
        lock.values.map { ModalScopeMath.applyEnvelope($0, displayEnvelope: displayEnvelope) }
    }
}

enum ModalScopeMath {
    /// The accepted canvas keeps eight percent of fixed headroom. It never
    /// derives a new scale from a frame, so modal amplitude remains visible.
    static let displayHeadroom = 1.08

    static func displayScale(sharedFullScale: Double) -> Double {
        sharedFullScale * displayHeadroom
    }

    static func visibleCycles(
        frequency: Double,
        profile: ModalScopeProfile = .qualified
    ) -> Double {
        guard frequency > 0, profile.sampleRate > 0,
              profile.displaySamples > 0, profile.timeCompression > 0
        else { return 0 }
        return frequency * profile.linearSpanSamples / profile.sampleRate
    }

    static func normalizedAmplitude(_ value: Double, fullScale: Double) -> Double {
        guard value.isFinite, fullScale > 0 else { return 0 }
        return max(-1, min(1, value / fullScale))
    }

    static func extractCarrier(
        values: [Double],
        envelopes: [Double],
        envelopeFloor: Double = 1e-12
    ) -> [Double] {
        values.enumerated().map { index, value in
            guard envelopes.indices.contains(index) else { return 0 }
            let envelope = envelopes[index]
            guard value.isFinite, envelope.isFinite, abs(envelope) > envelopeFloor
            else { return 0 }
            return value / envelope
        }
    }

    static func applyEnvelope(_ carrierValue: Double, displayEnvelope: Double) -> Double {
        guard carrierValue.isFinite, displayEnvelope >= 0 else { return 0 }
        return carrierValue * displayEnvelope
    }

    /// Linear interpolation with the same endpoint clamping as the JS oracle.
    static func sampleLinear(_ values: [Double], at position: Double) -> Double {
        guard !values.isEmpty else { return 0 }
        let left = max(0, min(values.count - 1, Int(floor(position))))
        let right = min(values.count - 1, left + 1)
        let fraction = position - Double(left)
        let leftValue = values[left].isFinite ? values[left] : 0
        let rightValue = values[right].isFinite ? values[right] : 0
        return leftValue + (rightValue - leftValue) * fraction
    }

    /// Choose the nearest eligible positive-going zero crossing to the middle
    /// of the available projection, then sample a linear, centered window.
    /// Each overlaid mode calls this independently.
    static func centeredWindow(
        _ values: [Double],
        outputLength: Int,
        spanLength: Double
    ) -> ModalScopeLock {
        guard values.count >= 2, outputLength >= 2, spanLength > 0 else {
            return .empty
        }

        let length = max(2, outputLength)
        let span = min(spanLength, Double(values.count - 1))
        let halfSpan = span / 2
        let target = Double(values.count - 1) / 2
        let minimumCenter = halfSpan
        let maximumCenter = Double(values.count - 1) - halfSpan

        var searchPeak = 0.0
        for raw in values {
            let value = raw.isFinite ? raw : 0
            searchPeak = max(searchPeak, abs(value))
        }

        var crossing = -1.0
        if searchPeak >= 1e-12, minimumCenter <= maximumCenter {
            let armLevel = searchPeak * 0.08
            var armed = (values[0].isFinite ? values[0] : 0) < -armLevel
            var bestDistance = Double.infinity

            for index in 1..<values.count {
                let previous = values[index - 1].isFinite ? values[index - 1] : 0
                let current = values[index].isFinite ? values[index] : 0
                if current < -armLevel { armed = true }
                if armed, previous < 0, current >= 0 {
                    let width = current - previous
                    let candidate = Double(index - 1) + (width == 0 ? 0 : -previous / width)
                    if candidate >= minimumCenter, candidate <= maximumCenter {
                        let distance = abs(candidate - target)
                        if distance < bestDistance {
                            crossing = candidate
                            bestDistance = distance
                        }
                    }
                }
            }

            // A steep attack can miss the relative arming threshold even when
            // its zero is usable. DC still fails because it never changes sign.
            if crossing < 0 {
                for index in 1..<values.count {
                    let previous = values[index - 1].isFinite ? values[index - 1] : 0
                    let current = values[index].isFinite ? values[index] : 0
                    guard previous < 0, current >= 0 else { continue }
                    let width = current - previous
                    let candidate = Double(index - 1) + (width == 0 ? 0 : -previous / width)
                    let distance = abs(candidate - target)
                    if candidate >= minimumCenter, candidate <= maximumCenter,
                       distance < bestDistance {
                        crossing = candidate
                        bestDistance = distance
                    }
                }
            }
        }

        let center = crossing >= 0
            ? crossing
            : max(minimumCenter, min(maximumCenter, target))
        let start = center - halfSpan
        let step = span / Double(length - 1)
        let lockedValues = (0..<length).map { index in
            sampleLinear(values, at: start + Double(index) * step)
        }
        let peak = lockedValues.reduce(0) { max($0, abs($1)) }

        return ModalScopeLock(
            offset: start,
            center: center,
            span: span,
            centerValue: sampleLinear(values, at: center),
            values: lockedValues,
            peak: peak,
            active: peak >= 1e-12,
            locked: crossing >= 0
        )
    }

    /// Qualified projection transform: remove the exact paired envelope at
    /// every source point, phase-lock the carrier, and retain the audible-now
    /// envelope for fixed-scale restoration by the canvas.
    static func frame(
        values: [Double],
        responseStart: Double,
        stride: Int = 1,
        frequency: Double,
        envelope: ModalPairedEnvelope,
        tauBase: Double,
        velocity: Double,
        playbackPosition: Double,
        sharedFullScale: Double,
        profile: ModalScopeProfile = .qualified
    ) -> ModalScopeFrame {
        let safeStride = max(1, stride)
        let warmupPoints = Int(ceil(Double(profile.warmupSamples) / Double(safeStride)))
        let displayPoints = Int(ceil(Double(profile.displaySamples) / Double(safeStride)))
        let raw = warmupPoints < values.count ? Array(values.dropFirst(warmupPoints)) : []
        let firstSample = responseStart + Double(warmupPoints * safeStride)
        let envelopes = envelope.samples(
            firstSceneTime: tauBase + velocity * firstSample / profile.sampleRate,
            sceneStep: velocity * Double(safeStride) / profile.sampleRate,
            count: raw.count
        )
        let carrier = extractCarrier(
            values: raw,
            envelopes: envelopes,
            envelopeFloor: sharedFullScale * 1e-6
        )
        let spanPoints = profile.linearSpanSamples / Double(safeStride)
        let lock = centeredWindow(
            carrier,
            outputLength: displayPoints,
            spanLength: spanPoints
        )
        let audibleNow = tauBase + velocity * playbackPosition / profile.sampleRate

        return ModalScopeFrame(
            stride: safeStride,
            cycles: visibleCycles(frequency: frequency, profile: profile),
            timeSpanSeconds: profile.linearSpanSeconds,
            displayEnvelope: envelope.value(at: audibleNow),
            lock: lock
        )
    }
}
