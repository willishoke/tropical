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
    let metalRenderTimeMs: Double?
    let metalRenderedFrames: Int?
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
        if let renderTimeNs = stats["metalRenderTimeNs"]?.doubleValue,
           renderTimeNs.isFinite, renderTimeNs >= 0,
           let renderedFrames = JSONExact.int(stats["metalRenderedFrames"]),
           renderedFrames >= 0 {
            self.metalRenderTimeMs = renderTimeNs / 1_000_000
            self.metalRenderedFrames = renderedFrames
        } else {
            self.metalRenderTimeMs = nil
            self.metalRenderedFrames = nil
        }
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
    private var previousMetalRenderMs: Double?
    private var previousMetalRenderedFrames: Int?
    private var underrunBase = 0
    private var overrunBase = 0
    private var hasBaseline = false

    mutating func reset(to sample: AudioTelemetrySample) {
        previousCallbackCount = sample.callbackCount
        previousTotalCallbackMs = sample.averageCallbackMs
            * Double(sample.callbackCount)
        previousMetalRenderMs = sample.metalRenderTimeMs
        previousMetalRenderedFrames = sample.metalRenderedFrames
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
        let loadRatio: Double
        if sample.backend == "metal",
           let renderMs = sample.metalRenderTimeMs,
           let renderedFrames = sample.metalRenderedFrames,
           let previousRenderMs = previousMetalRenderMs,
           let previousRenderedFrames = previousMetalRenderedFrames {
            guard renderMs >= previousRenderMs,
                  renderedFrames >= previousRenderedFrames
            else {
                reset(to: sample)
                return nil
            }
            let frameDelta = renderedFrames - previousRenderedFrames
            guard frameDelta > 0 else { return nil }
            let renderedAudioMs = Double(frameDelta) / sample.sampleRate * 1_000
            loadRatio = (renderMs - previousRenderMs) / renderedAudioMs
        } else {
            guard callbackDelta > 0 else { return nil }
            let recentAverageMs = (totalCallbackMs - previousTotalCallbackMs)
                / Double(callbackDelta)
            let deadlineMs = Double(sample.bufferFrames) / sample.sampleRate * 1_000
            loadRatio = recentAverageMs / deadlineMs
        }
        previousCallbackCount = sample.callbackCount
        previousTotalCallbackMs = totalCallbackMs
        previousMetalRenderMs = sample.metalRenderTimeMs
        previousMetalRenderedFrames = sample.metalRenderedFrames

        let percent = Int((loadRatio * 100).rounded())
        return AudioLoadReading(
            backend: sample.backend,
            percent: max(0, percent),
            xruns: sample.underrunCount - underrunBase,
            deadlineMisses: sample.overrunCount - overrunBase
        )
    }
}
