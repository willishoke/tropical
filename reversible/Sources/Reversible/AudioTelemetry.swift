import Foundation

/// One cumulative DAC counter snapshot. The engine reports cumulative means;
/// `AudioLoadMeter` differences their implied totals to recover the recent
/// window instead of displaying a lifetime maximum forever.
struct AudioTelemetrySample: Equatable {
    let backend: String
    let sampleRate: Double
    let bufferFrames: Int
    let callbackCount: Int
    let averageCallbackMs: Double
    let underrunCount: Int
    let overrunCount: Int

    init?(status: JSONValue) {
        guard let stats = status["stats"], stats != .null,
              let sampleRate = status["sampleRate"]?.doubleValue,
              sampleRate.isFinite, sampleRate > 0,
              let bufferFrames = JSONExact.int(status["bufferFrames"]),
              bufferFrames > 0,
              let callbackCount = JSONExact.int(stats["callbackCount"]),
              let averageCallbackMs = stats["avgCallbackMs"]?.doubleValue,
              averageCallbackMs.isFinite, averageCallbackMs >= 0,
              let underrunCount = JSONExact.int(stats["underrunCount"]),
              let overrunCount = JSONExact.int(stats["overrunCount"])
        else { return nil }
        self.backend = status["backend"]?.stringValue ?? "unknown"
        self.sampleRate = sampleRate
        self.bufferFrames = bufferFrames
        self.callbackCount = callbackCount
        self.averageCallbackMs = averageCallbackMs
        self.underrunCount = underrunCount
        self.overrunCount = overrunCount
    }
}

struct AudioLoadReading: Equatable {
    let backend: String
    let percent: Int
    let xruns: Int
    let deadlineMisses: Int
}

struct AudioLoadMeter {
    private var previousCallbackCount = 0
    private var previousTotalCallbackMs = 0.0
    private var underrunBase = 0
    private var overrunBase = 0
    private var hasBaseline = false

    mutating func reset(to sample: AudioTelemetrySample) {
        previousCallbackCount = sample.callbackCount
        previousTotalCallbackMs = sample.averageCallbackMs
            * Double(sample.callbackCount)
        underrunBase = sample.underrunCount
        overrunBase = sample.overrunCount
        hasBaseline = true
    }

    mutating func update(_ sample: AudioTelemetrySample) -> AudioLoadReading? {
        guard hasBaseline else {
            reset(to: sample)
            return nil
        }
        let totalCallbackMs = sample.averageCallbackMs
            * Double(sample.callbackCount)
        guard sample.callbackCount >= previousCallbackCount,
              totalCallbackMs >= previousTotalCallbackMs,
              sample.underrunCount >= underrunBase,
              sample.overrunCount >= overrunBase
        else {
            // A DAC restart resets its counters. Adopt the new epoch rather
            // than manufacturing a negative/huge interval.
            reset(to: sample)
            return nil
        }

        let callbackDelta = sample.callbackCount - previousCallbackCount
        guard callbackDelta > 0 else { return nil }
        let recentAverageMs = (totalCallbackMs - previousTotalCallbackMs)
            / Double(callbackDelta)
        previousCallbackCount = sample.callbackCount
        previousTotalCallbackMs = totalCallbackMs

        let deadlineMs = Double(sample.bufferFrames) / sample.sampleRate * 1_000
        let percent = Int((recentAverageMs / deadlineMs * 100).rounded())
        return AudioLoadReading(
            backend: sample.backend,
            percent: max(0, percent),
            xruns: sample.underrunCount - underrunBase,
            deadlineMisses: sample.overrunCount - overrunBase
        )
    }
}
