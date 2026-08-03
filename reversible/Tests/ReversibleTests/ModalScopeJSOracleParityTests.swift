import Foundation
import XCTest
@testable import Reversible

private struct ModalScopeOracleFixture: Decodable {
    struct Profile: Decodable {
        let sampleRate: Double
        let displaySamples: Int
        let searchSamples: Int
        let warmupSamples: Int
        let timeCompression: Double
        let pointBudget: Int
        let fps: Double
        let fullScale: Double
        let displayScale: Double
    }

    struct Case: Decodable {
        struct Input: Decodable {
            let frequency: Double
            let slowDecay: Double
            let fastDecay: Double
            let amplitude: Double
            let phase: Double
            let strikeTime: Double
            let responseStart: Double
            let playbackPosition: Double
            let stride: Int
        }

        struct Expected: Decodable {
            struct Sample: Decodable {
                let index: Int
                let value: Double
            }

            let offset: Double
            let center: Double
            let span: Double
            let centerValue: Double
            let peak: Double
            let active: Bool
            let locked: Bool
            let displayEnvelope: Double
            let cycles: Double
            let timeSpanSeconds: Double
            let samples: [Sample]
        }

        let name: String
        let input: Input
        let expected: Expected
    }

    let schema: String
    let sourceCommit: String
    let profile: Profile
    let sampleIndices: [Int]
    let cases: [Case]
}

final class ModalScopeJSOracleParityTests: XCTestCase {
    private static func loadFixture() throws -> ModalScopeOracleFixture {
        let url = try XCTUnwrap(
            Bundle.module.url(
                forResource: "modal-scope-js-oracle-v1",
                withExtension: "json"
            )
        )
        return try JSONDecoder().decode(
            ModalScopeOracleFixture.self,
            from: Data(contentsOf: url)
        )
    }

    private func projectionValues(
        for item: ModalScopeOracleFixture.Case,
        profile: ModalScopeProfile
    ) -> [Double] {
        let input = item.input
        let envelope = ModalPairedEnvelope(
            strikeTime: input.strikeTime,
            slowDecay: input.slowDecay,
            fastDecay: input.fastDecay,
            amplitude: input.amplitude
        )
        return (0..<profile.pointBudget).map { index in
            let sceneTime = (input.responseStart + Double(index)) / profile.sampleRate
            return envelope.value(at: sceneTime) * cos(
                2 * Double.pi * input.frequency * sceneTime + input.phase
            )
        }
    }

    private func frame(
        for item: ModalScopeOracleFixture.Case,
        fixture: ModalScopeOracleFixture
    ) -> ModalScopeFrame {
        let profile = ModalScopeProfile.qualified
        return ModalScopeMath.frame(
            values: projectionValues(for: item, profile: profile),
            responseStart: item.input.responseStart,
            stride: item.input.stride,
            frequency: item.input.frequency,
            envelope: ModalPairedEnvelope(
                strikeTime: item.input.strikeTime,
                slowDecay: item.input.slowDecay,
                fastDecay: item.input.fastDecay,
                amplitude: item.input.amplitude
            ),
            tauBase: 0,
            velocity: 1,
            playbackPosition: item.input.playbackPosition,
            sharedFullScale: fixture.profile.fullScale,
            profile: profile
        )
    }

    func testQualifiedProfileMatchesAcceptedJSGeometry() throws {
        let fixture = try Self.loadFixture()
        let expected = fixture.profile
        let actual = ModalScopeProfile.qualified

        XCTAssertEqual(fixture.schema, "modal_scope_js_oracle_1")
        XCTAssertEqual(fixture.sourceCommit, "07a6b2517f8d24c8822d67e0337c0fd99d016bd8")
        XCTAssertEqual(actual.sampleRate, expected.sampleRate)
        XCTAssertEqual(actual.displaySamples, expected.displaySamples)
        XCTAssertEqual(actual.searchSamples, expected.searchSamples)
        XCTAssertEqual(actual.warmupSamples, expected.warmupSamples)
        XCTAssertEqual(actual.timeCompression, expected.timeCompression)
        XCTAssertEqual(actual.pointBudget, expected.pointBudget)
        XCTAssertEqual(actual.framesPerSecond, expected.fps)
        XCTAssertEqual(actual.linearSpanSamples, 1_792)
        XCTAssertEqual(actual.linearSpanSeconds, 1_792.0 / 44_100.0)
        XCTAssertEqual(
            ModalScopeMath.displayScale(sharedFullScale: expected.fullScale),
            expected.displayScale,
            accuracy: 1e-15
        )
    }

    func testFiveIndependentModeFramesAgreeWithJSOracle() throws {
        let fixture = try Self.loadFixture()
        XCTAssertEqual(fixture.cases.count, 5)
        var centers: [Double] = []

        for item in fixture.cases {
            let actual = frame(for: item, fixture: fixture)
            let expected = item.expected
            let lock = actual.lock
            centers.append(lock.center)

            XCTAssertEqual(actual.stride, item.input.stride, item.name)
            XCTAssertEqual(actual.cycles, expected.cycles, accuracy: 1e-12, item.name)
            XCTAssertEqual(
                actual.timeSpanSeconds,
                expected.timeSpanSeconds,
                accuracy: 1e-15,
                item.name
            )
            XCTAssertEqual(
                actual.displayEnvelope,
                expected.displayEnvelope,
                accuracy: 1e-12,
                item.name
            )
            XCTAssertEqual(lock.offset, expected.offset, accuracy: 2e-7, item.name)
            XCTAssertEqual(lock.center, expected.center, accuracy: 2e-7, item.name)
            XCTAssertEqual(lock.span, expected.span, accuracy: 1e-12, item.name)
            XCTAssertEqual(lock.peak, expected.peak, accuracy: 2e-7, item.name)
            XCTAssertEqual(lock.active, expected.active, item.name)
            XCTAssertEqual(lock.locked, expected.locked, item.name)
            XCTAssertEqual(lock.values.count, ModalScopeProfile.qualified.displaySamples)
            XCTAssertLessThan(abs(lock.centerValue), 1e-12, item.name)
            XCTAssertLessThan(abs(lock.centerValue - expected.centerValue), 1e-12, item.name)
            XCTAssertLessThan(lock.values[447], 0, item.name)
            XCTAssertGreaterThan(lock.values[448], 0, item.name)

            for sample in expected.samples {
                XCTAssertEqual(
                    lock.values[sample.index],
                    sample.value,
                    accuracy: 2e-7,
                    "\(item.name) sample \(sample.index)"
                )
            }

            let restoredPeak = actual.displayedValues.reduce(0) { max($0, abs($1)) }
            XCTAssertEqual(
                restoredPeak,
                lock.peak * actual.displayEnvelope,
                accuracy: 1e-14,
                item.name
            )
            XCTAssertLessThan(
                abs(restoredPeak / actual.displayEnvelope - 1),
                0.001,
                item.name
            )
        }

        // A shared canvas does not imply a shared trigger: all five carriers
        // choose their own interpolated center.
        XCTAssertEqual(Set(centers.map { Int(floor($0)) }).count, 5)
    }

    func testPointwiseEnvelopeRemovalRecoversCarrierBeforeLock() throws {
        let fixture = try Self.loadFixture()
        let item = try XCTUnwrap(fixture.cases.first)
        let profile = ModalScopeProfile.qualified
        let envelope = ModalPairedEnvelope(
            strikeTime: item.input.strikeTime,
            slowDecay: item.input.slowDecay,
            fastDecay: item.input.fastDecay,
            amplitude: item.input.amplitude
        )
        let values = projectionValues(for: item, profile: profile)
        let warmup = profile.warmupSamples
        let raw = Array(values.dropFirst(warmup))
        let firstSceneTime = (item.input.responseStart + Double(warmup)) / profile.sampleRate
        let envelopes = envelope.samples(
            firstSceneTime: firstSceneTime,
            sceneStep: 1 / profile.sampleRate,
            count: raw.count
        )
        let carrier = ModalScopeMath.extractCarrier(
            values: raw,
            envelopes: envelopes,
            envelopeFloor: fixture.profile.fullScale * 1e-6
        )

        for index in stride(from: 0, to: carrier.count, by: 137) {
            let sceneTime = firstSceneTime + Double(index) / profile.sampleRate
            let expected = cos(
                2 * Double.pi * item.input.frequency * sceneTime + item.input.phase
            )
            XCTAssertEqual(carrier[index], expected, accuracy: 1e-12)
        }
    }

    func testSilenceAndDCCannotMasqueradeAsCenteredLocks() {
        let silence = ModalScopeMath.centeredWindow(
            Array(repeating: 0, count: 2_688),
            outputLength: 896,
            spanLength: 1_792
        )
        let dc = ModalScopeMath.centeredWindow(
            Array(repeating: 0.02, count: 2_688),
            outputLength: 896,
            spanLength: 1_792
        )

        XCTAssertFalse(silence.active)
        XCTAssertFalse(silence.locked)
        XCTAssertTrue(dc.active)
        XCTAssertFalse(dc.locked)
        XCTAssertEqual(
            ModalScopeMath.extractCarrier(
                values: [1, 2, .nan, 4],
                envelopes: [0, 1e-14, 1, .infinity],
                envelopeFloor: 1e-12
            ),
            [0, 0, 0, 0]
        )
    }
}
