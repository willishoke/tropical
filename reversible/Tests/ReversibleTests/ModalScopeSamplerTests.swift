import XCTest
@testable import Reversible

private actor ModalScopeRPCStub: ModalScopeRPC {
    private var calls: [(String, JSONValue)] = []
    private var renderReplies: [JSONValue]
    private let playbackPosition: Double

    init(playbackPosition: Double, renderReplies: [JSONValue]) {
        self.playbackPosition = playbackPosition
        self.renderReplies = renderReplies
    }

    func call(_ method: String, _ params: JSONValue) async throws -> JSONValue {
        calls.append((method, params))
        if method == "playback_position" {
            return .object(["position": .number(playbackPosition)])
        }
        guard method == "render_window", !renderReplies.isEmpty else {
            throw EngineError.rpc("unexpected scope stub call: \(method)")
        }
        return renderReplies.removeFirst()
    }

    func recordedCalls() -> [(String, JSONValue)] { calls }
}

final class ModalScopeSamplerTests: XCTestCase {
    private static let frequencies = [55.0, 82.5, 132.0, 198.0, 880.0 / 3.0]
    private static let phases = [0.2, 1.1, 2.0, 2.9, 3.8]

    private func binding() throws -> ModalScopeBinding {
        let modes = Self.frequencies.enumerated().map { index, frequency in
            ModalScopeModeDescriptor(
                id: "mode\(index + 1)",
                slot: "__root__.tap:mode\(index + 1)",
                frequency: frequency,
                envelope: ModalPairedEnvelope(
                    strikeTime: 0,
                    slowDecay: 0.42,
                    fastDecay: 5.22,
                    amplitude: 0.055 * (1 - Double(index) * 0.08)
                )
            )
        }
        return try XCTUnwrap(ModalScopeBinding(
            modes: modes,
            sharedFullScale: modes.map(\.envelope.peak).max()!,
            clock: .forward
        ))
    }

    private func response(
        binding: ModalScopeBinding,
        playbackPosition: Double,
        programVersion: UInt64,
        controlVersion: UInt64,
        effectiveSampleIndex: Double? = nil,
        preempted: Bool = false,
        partialLastChannel: Bool = false
    ) -> JSONValue {
        let profile = ModalScopeProfile.qualified
        let start = max(0, Int(playbackPosition) - profile.pointBudget)
        if preempted {
            return .object([
                "start": .number(Double(start)),
                "count": .number(0),
                "span": .number(Double(profile.pointBudget)),
                "stride": .number(1),
                "preempted": .bool(true),
                "program_version": .number(Double(programVersion)),
                "control_version": .number(Double(controlVersion)),
                "effective_sample_index": .number(
                    effectiveSampleIndex ?? playbackPosition
                ),
                "values": .array([]),
            ])
        }
        let channels = binding.modes.enumerated().map { modeIndex, mode in
            let count = partialLastChannel && modeIndex == 4
                ? profile.pointBudget - 1
                : profile.pointBudget
            return JSONValue.array((0..<count).map { index in
                let sceneTime = Double(start + index) / profile.sampleRate
                let value = mode.envelope.value(at: sceneTime) * cos(
                    2 * Double.pi * mode.frequency * sceneTime
                        + Self.phases[modeIndex]
                )
                return .number(value)
            })
        }
        return .object([
            "start": .number(Double(start)),
            "count": .number(Double(profile.pointBudget)),
            "span": .number(Double(profile.pointBudget)),
            "stride": .number(1),
            "preempted": .bool(false),
            "program_version": .number(Double(programVersion)),
            "control_version": .number(Double(controlVersion)),
            "effective_sample_index": .number(
                effectiveSampleIndex ?? playbackPosition
            ),
            "values": .array(channels),
        ])
    }

    func testOneCompletedRPCProducesFiveIndependentLocks() async throws {
        let binding = try binding()
        let position = 8_000.0
        let stub = ModalScopeRPCStub(
            playbackPosition: position,
            renderReplies: [response(
                binding: binding,
                playbackPosition: position,
                programVersion: 7,
                controlVersion: 19,
                effectiveSampleIndex: position - 32
            )]
        )
        let worker = ModalScopeWorker(rpc: stub)
        let frame = try XCTUnwrap(try await worker.fetch(binding: binding))

        XCTAssertEqual(frame.generation, ModalScopeGeneration(
            programVersion: 7, controlVersion: 19
        ))
        XCTAssertEqual(frame.responseCount, ModalScopeProfile.qualified.pointBudget)
        XCTAssertEqual(frame.responseSpan, ModalScopeProfile.qualified.pointBudget)
        XCTAssertEqual(frame.stride, 1)
        XCTAssertEqual(frame.playbackPosition, position - 32)
        XCTAssertEqual(frame.effectiveSampleIndex, UInt64(position - 32))
        XCTAssertEqual(frame.modes.count, 5)
        XCTAssertEqual(frame.displayScale, ModalScopeMath.displayScale(
            sharedFullScale: binding.sharedFullScale
        ))
        XCTAssertTrue(frame.modes.allSatisfy { $0.frame.lock.locked })
        XCTAssertTrue(frame.modes.allSatisfy {
            $0.frame.lock.values.count == ModalScopeProfile.qualified.displaySamples
        })
        XCTAssertEqual(
            Set(frame.modes.map { Int(floor($0.frame.lock.center)) }).count,
            5
        )

        let calls = await stub.recordedCalls()
        XCTAssertEqual(calls.map(\.0), ["playback_position", "render_window"])
        let request = try XCTUnwrap(calls.last?.1)
        XCTAssertEqual(
            request["count"]?.doubleValue,
            Double(ModalScopeProfile.qualified.pointBudget)
        )
        XCTAssertEqual(request["point_budget"], request["count"])
        guard case .array(let slots)? = request["slots"] else {
            return XCTFail("one render_window call must carry all five slots")
        }
        XCTAssertEqual(slots, binding.modes.map { .string($0.slot) })
    }

    func testGenerationRegressionAndPreemptionDropWholeFrames() async throws {
        let binding = try binding()
        let position = 8_000.0
        let stub = ModalScopeRPCStub(
            playbackPosition: position,
            renderReplies: [
                response(
                    binding: binding, playbackPosition: position,
                    programVersion: 9, controlVersion: 4
                ),
                response(
                    binding: binding, playbackPosition: position,
                    programVersion: 8, controlVersion: 99
                ),
                response(
                    binding: binding, playbackPosition: position,
                    programVersion: 10, controlVersion: 1,
                    preempted: true
                ),
            ]
        )
        let worker = ModalScopeWorker(rpc: stub)

        XCTAssertNotNil(try await worker.fetch(binding: binding))
        XCTAssertNil(try await worker.fetch(binding: binding))
        XCTAssertNil(try await worker.fetch(binding: binding))
    }

    func testPartialChannelRejectsInsteadOfPublishingMixedFrame() async throws {
        let binding = try binding()
        let position = 8_000.0
        let stub = ModalScopeRPCStub(
            playbackPosition: position,
            renderReplies: [response(
                binding: binding,
                playbackPosition: position,
                programVersion: 1,
                controlVersion: 1,
                partialLastChannel: true
            )]
        )
        let worker = ModalScopeWorker(rpc: stub)

        do {
            _ = try await worker.fetch(binding: binding)
            XCTFail("partial five-mode response must not publish")
        } catch let error as ModalScopeSamplerError {
            XCTAssertEqual(
                error,
                .malformedResponse("scope response contained a partial mode channel")
            )
        }
    }

    func testGenerationOutsideJSONExactIntegerRangeIsMalformed() async throws {
        let binding = try binding()
        let position = 8_000.0
        var invalid = response(
            binding: binding,
            playbackPosition: position,
            programVersion: 1,
            controlVersion: 1
        )
        guard case .object(var fields) = invalid else {
            return XCTFail("fixture response must be an object")
        }
        fields["program_version"] = .number(9_007_199_254_740_992)
        invalid = .object(fields)
        let stub = ModalScopeRPCStub(
            playbackPosition: position,
            renderReplies: [invalid]
        )
        let worker = ModalScopeWorker(rpc: stub)

        do {
            _ = try await worker.fetch(binding: binding)
            XCTFail("inexact JSON generation must not convert or publish")
        } catch let error as ModalScopeSamplerError {
            XCTAssertEqual(
                error,
                .malformedResponse(
                    "scope response was not one complete stride-1 five-slot projection"
                )
            )
        }
    }

    func testBindingRequiresFiveUniqueSlotsAndFixedPositiveScale() throws {
        let valid = try binding()
        XCTAssertNil(ModalScopeBinding(
            modes: Array(valid.modes.dropLast()),
            sharedFullScale: valid.sharedFullScale,
            clock: valid.clock
        ))
        var duplicate = valid.modes
        duplicate[4] = ModalScopeModeDescriptor(
            id: duplicate[4].id,
            slot: duplicate[0].slot,
            frequency: duplicate[4].frequency,
            envelope: duplicate[4].envelope
        )
        XCTAssertNil(ModalScopeBinding(
            modes: duplicate,
            sharedFullScale: valid.sharedFullScale,
            clock: valid.clock
        ))
        XCTAssertNil(ModalScopeBinding(
            modes: valid.modes,
            sharedFullScale: 0,
            clock: valid.clock
        ))
    }

    func testDisplayCadenceCapsOneHundredTwentyHertzTicksAtSixty() {
        var cadence = ModalScopeCadence(framesPerSecond: 60)
        var accepted = 0
        for tick in 0..<120 {
            if cadence.accepts(timestamp: Double(tick) / 120) { accepted += 1 }
        }
        XCTAssertEqual(accepted, 60)
    }
}
