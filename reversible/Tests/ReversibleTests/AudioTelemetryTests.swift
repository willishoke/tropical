import XCTest
@testable import Reversible

final class AudioTelemetryTests: XCTestCase {
    func testUsesRecentAverageAndNegotiatedDeadline() throws {
        var meter = AudioLoadMeter()
        meter.reset(to: try sample(
            backend: "jit", frames: 256, rate: 48_000,
            callbacks: 100, averageMs: 2, underruns: 3, overruns: 4
        ))

        // First 100 callbacks total 200 ms. The next ten total 40 ms, so the
        // recent load is 4 / (256/48000*1000) = 75%, independent of any old max.
        let reading = meter.update(try sample(
            backend: "jit", frames: 256, rate: 48_000,
            callbacks: 110, averageMs: 240.0 / 110.0,
            underruns: 5, overruns: 7
        ))
        XCTAssertEqual(reading, .init(
            backend: "jit", percent: 75, xruns: 2, deadlineMisses: 3
        ))
    }

    func testMetalUsesRenderWorkerUtilizationInsteadOfCopyCallback() throws {
        var meter = AudioLoadMeter()
        meter.reset(to: try sample(
            backend: "metal", frames: 256, rate: 48_000,
            callbacks: 100, averageMs: 0.006,
            metalRenderMs: 200, metalRenderedFrames: 25_600,
            underruns: 0, overruns: 0
        ))

        // Ten 256-frame tiles represent 53.333 ms of audio. Forty ms of
        // synchronous Metal render work is 75% even though the callback that
        // copies those prepared samples rounds to 0%.
        let reading = meter.update(try sample(
            backend: "metal", frames: 256, rate: 48_000,
            callbacks: 110, averageMs: 0.006,
            metalRenderMs: 240, metalRenderedFrames: 28_160,
            underruns: 0, overruns: 0
        ))
        XCTAssertEqual(reading?.percent, 75)
    }

    func testCounterResetStartsANewWindow() throws {
        var meter = AudioLoadMeter()
        meter.reset(to: try sample(callbacks: 50, averageMs: 3, underruns: 2, overruns: 2))
        XCTAssertNil(meter.update(try sample(
            callbacks: 1, averageMs: 1, underruns: 0, overruns: 0
        )))
        XCTAssertEqual(
            meter.update(try sample(
                callbacks: 2, averageMs: 1.5, underruns: 0, overruns: 0
            ))?.percent,
            17
        )
    }

    private func sample(
        backend: String = "jit",
        frames: Int = 512,
        rate: Double = 44_100,
        callbacks: Int,
        averageMs: Double,
        metalRenderMs: Double? = nil,
        metalRenderedFrames: Int? = nil,
        underruns: Int,
        overruns: Int
    ) throws -> AudioTelemetrySample {
        var stats: [String: JSONValue] = [
            "callbackCount": .number(Double(callbacks)),
            "avgCallbackMs": .number(averageMs),
            "maxCallbackMs": .number(999),
            "underrunCount": .number(Double(underruns)),
            "overrunCount": .number(Double(overruns)),
        ]
        if let metalRenderMs, let metalRenderedFrames {
            stats["metalRenderTimeNs"] = .number(metalRenderMs * 1_000_000)
            stats["metalRenderedFrames"] = .number(Double(metalRenderedFrames))
        }
        return try XCTUnwrap(AudioTelemetrySample(status: .object([
            "backend": .string(backend),
            "sampleRate": .number(rate),
            "bufferFrames": .number(Double(frames)),
            "stats": .object(stats),
        ])))
    }
}
