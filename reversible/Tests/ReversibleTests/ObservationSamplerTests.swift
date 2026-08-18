import Foundation
import XCTest
@testable import Reversible

private actor ObservationRPCStub: ObservationRPC {
    private var replies: [JSONValue]
    private var calls: [(String, JSONValue)] = []
    private let delay: Duration?

    init(replies: [JSONValue], delay: Duration? = nil) {
        self.replies = replies
        self.delay = delay
    }

    func call(_ method: String, _ params: JSONValue) async throws -> JSONValue {
        calls.append((method, params))
        if let delay { try await Task.sleep(for: delay) }
        guard method == "render_window", !replies.isEmpty else {
            throw EngineError.rpc("unexpected observation call")
        }
        return replies.removeFirst()
    }

    func recordedCalls() -> [(String, JSONValue)] { calls }
}

final class ObservationSamplerTests: XCTestCase {
    private func binding() throws -> ObservationBinding {
        try ObservationBinding(
            expectedGeneration: .init(programVersion: 41, controlVersion: 77),
            span: 6,
            pointBudget: 3,
            channels: [
                .init(key: "scope1.a", slot: "__root__.tap:src"),
                .init(key: "scope1.b", slot: "__root__.out"),
            ]
        )
    }

    func testOneShotPlaybackRequestDecodesOneImmutableFrame() async throws {
        let binding = try binding()
        let stub = ObservationRPCStub(replies: [try fixtureJSON()])
        let worker = ObservationWorker(rpc: stub)
        let fetched = try await worker.fetch(binding)
        let frame = try XCTUnwrap(fetched)

        XCTAssertEqual(
            frame.generation,
            .init(programVersion: 41, controlVersion: 80)
        )
        XCTAssertEqual(frame.playbackPosition, 100)
        XCTAssertEqual(frame.effectiveStart, 94)
        XCTAssertEqual(frame.effectiveSampleIndex, 88)
        XCTAssertEqual(frame.count, 3)
        XCTAssertEqual(frame.stride, 2)
        XCTAssertEqual(frame.channels.map(\.channel), binding.channels)
        XCTAssertEqual(frame.channels[0].values, [0.25, 0.5, 0.75])

        let calls = await stub.recordedCalls()
        XCTAssertEqual(calls.map(\.0), ["render_window"])
        XCTAssertEqual(calls[0].1["anchor"], .string("playback"))
        XCTAssertEqual(calls[0].1["count"], .number(6))
        XCTAssertEqual(calls[0].1["point_budget"], .number(3))
        XCTAssertEqual(
            calls[0].1["slots"],
            .array(binding.channels.map { .string($0.slot) })
        )
    }

    func testPreemptedWrongProgramAndStaleControlDropWholeFrames() async throws {
        let binding = try binding()
        let valid = try fixtureJSON()
        let stub = ObservationRPCStub(replies: [
            response(valid, replacing: ["control_version": .number(80)]),
            response(valid, replacing: ["control_version": .number(79)]),
            response(valid, replacing: [
                "program_version": .number(42),
                "control_version": .number(100),
            ]),
            response(valid, replacing: [
                "count": .number(0),
                "preempted": .bool(true),
                "values": .array([]),
            ]),
            response(valid, replacing: ["control_version": .number(81)]),
        ])
        let worker = ObservationWorker(rpc: stub)

        let first = try await worker.fetch(binding)
        let stale = try await worker.fetch(binding)
        let wrongProgram = try await worker.fetch(binding)
        let preempted = try await worker.fetch(binding)
        let newer = try await worker.fetch(binding)
        XCTAssertNotNil(first)
        XCTAssertNil(stale)
        XCTAssertNil(wrongProgram)
        XCTAssertNil(preempted)
        XCTAssertEqual(
            newer?.generation,
            .init(programVersion: 41, controlVersion: 81)
        )
    }

    func testMalformedOrPartialResponseNeverPublishes() async throws {
        let binding = try binding()
        let valid = try fixtureJSON()
        guard case .object(var object) = valid,
              case .array(var channels)? = object["values"],
              case .array(var last)? = channels.last
        else { return XCTFail("fixture shape") }
        last.removeLast()
        channels[channels.count - 1] = .array(last)
        object["values"] = .array(channels)

        let partialWorker = ObservationWorker(
            rpc: ObservationRPCStub(replies: [.object(object)])
        )
        do {
            _ = try await partialWorker.fetch(binding)
            XCTFail("partial response must throw")
        } catch let error as ObservationError {
            XCTAssertEqual(error, .partialChannel(1))
        }

        let inexact = response(valid, replacing: [
            "effective_sample_index": .number(9_007_199_254_740_992),
        ])
        let inexactWorker = ObservationWorker(
            rpc: ObservationRPCStub(replies: [inexact])
        )
        do {
            _ = try await inexactWorker.fetch(binding)
            XCTFail("inexact protocol counter must throw")
        } catch let error as ObservationError {
            XCTAssertEqual(error, .invalidField("effective_sample_index"))
        }
    }

    func testSessionEpochAllowsRestartToReuseProgramAndControlVersions() async throws {
        let binding = try binding()
        let valid = try fixtureJSON()
        let stub = ObservationRPCStub(replies: [
            response(valid, replacing: ["control_version": .number(80)]),
            response(valid, replacing: ["control_version": .number(77)]),
        ])
        let worker = ObservationWorker(rpc: stub)

        let beforeRestart = try await worker.fetch(binding, sessionEpoch: 0)
        let afterRestart = try await worker.fetch(binding, sessionEpoch: 1)
        XCTAssertEqual(
            beforeRestart?.generation.controlVersion,
            80
        )
        XCTAssertEqual(
            afterRestart?.generation.controlVersion,
            77
        )
    }

    @MainActor
    func testSamplerAdmitsOnlyOneRequestAtATime() async throws {
        let binding = try binding()
        let sampler = ObservationSampler(rpc: ObservationRPCStub(
            replies: [try fixtureJSON()],
            delay: .milliseconds(80)
        ))
        let published = expectation(description: "one complete frame published")
        sampler.onFrame = { _ in published.fulfill() }
        sampler.adoptCompileGeneration(binding.expectedGeneration)

        sampler.request(binding)
        sampler.request(binding)
        XCTAssertTrue(sampler.hasRequestInFlight)
        XCTAssertEqual(sampler.requestCount, 1)
        XCTAssertEqual(sampler.droppedFrameCount, 1)

        await fulfillment(of: [published], timeout: 2)
        XCTAssertFalse(sampler.hasRequestInFlight)
        XCTAssertNotNil(sampler.latestFrame)
    }

    @MainActor
    func testClearDropsOldSessionReplyBeforeSameGenerationRestart() async throws {
        let binding = try binding()
        let valid = try fixtureJSON()
        let sampler = ObservationSampler(rpc: ObservationRPCStub(
            replies: [
                response(valid, replacing: ["control_version": .number(80)]),
                response(valid, replacing: ["control_version": .number(77)]),
            ],
            delay: .milliseconds(50)
        ))
        sampler.adoptCompileGeneration(binding.expectedGeneration)
        sampler.request(binding)
        sampler.clear()
        sampler.adoptCompileGeneration(binding.expectedGeneration)

        for _ in 0..<200 where sampler.hasRequestInFlight {
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertFalse(sampler.hasRequestInFlight)
        XCTAssertNil(sampler.latestFrame)

        let restarted = expectation(description: "restarted session frame")
        sampler.onFrame = { _ in restarted.fulfill() }
        sampler.request(binding)
        await fulfillment(of: [restarted], timeout: 2)
        XCTAssertEqual(sampler.latestFrame?.generation.controlVersion, 77)
    }

    func testMixedWindowBatchPreservesEachChannelsOwnHorizon() {
        let sharedBatch = (0...11).reversed().map(Double.init)
        XCTAssertEqual(
            ScopeTraceProjection.triggeredSamples(
                sharedBatch,
                displaySpanSamples: 3,
                acquisitionSpanSamples: 6,
                stride: 1
            ),
            [2, 1, 0]
        )
        XCTAssertEqual(
            ScopeTraceProjection.triggeredSamples(
                sharedBatch,
                displaySpanSamples: 6,
                acquisitionSpanSamples: 12,
                stride: 1
            ),
            [5, 4, 3, 2, 1, 0]
        )
    }

    func testOffsetWaveformPhaseLocksAcrossPlaybackAnchors() {
        func sine(start: Int) -> [Double] {
            (0..<80).map { index in
                5 + sin(2 * .pi * Double(start + index) / 20)
            }
        }
        let first = ScopeTraceProjection.triggeredSamples(
            sine(start: 0),
            displaySpanSamples: 40,
            acquisitionSpanSamples: 80,
            stride: 1
        )
        let advanced = ScopeTraceProjection.triggeredSamples(
            sine(start: 7),
            displaySpanSamples: 40,
            acquisitionSpanSamples: 80,
            stride: 1
        )

        XCTAssertEqual(first.count, 40)
        XCTAssertEqual(advanced.count, 40)
        XCTAssertEqual(first[0], 5, accuracy: 1e-12)
        XCTAssertGreaterThan(first[1], first[0])
        for (lhs, rhs) in zip(first, advanced) {
            XCTAssertEqual(lhs, rhs, accuracy: 1e-12)
        }
    }

    func testLongAcquisitionUsesBoundedDeterministicStride() throws {
        let binding = try ObservationBinding(
            expectedGeneration: .init(programVersion: 1, controlVersion: 1),
            span: 100_000,
            pointBudget: ObservationBinding.maximumPointBudget,
            channels: [.init(key: "scope1.ch1", slot: "out")]
        )

        XCTAssertEqual(binding.expectedStride, 7)
        XCTAssertEqual(binding.expectedCount, 14_286)
    }

    func testDisplayCadenceCapsOneHundredTwentyHertzAtSixty() {
        var cadence = ObservationCadence(framesPerSecond: 60)
        var accepted = 0
        for tick in 0..<120 {
            if cadence.accepts(timestamp: Double(tick) / 120) { accepted += 1 }
        }
        XCTAssertEqual(accepted, 60)
    }

    private func response(
        _ base: JSONValue,
        replacing fields: [String: JSONValue]
    ) -> JSONValue {
        guard case .object(var object) = base else { return base }
        for (key, value) in fields { object[key] = value }
        return .object(object)
    }

    private func fixtureJSON() throws -> JSONValue {
        let url = try XCTUnwrap(Bundle.module.url(
            forResource: "observation-frame-valid",
            withExtension: "json"
        ))
        return try JSONDecoder().decode(
            JSONValue.self,
            from: Data(contentsOf: url)
        )
    }
}
