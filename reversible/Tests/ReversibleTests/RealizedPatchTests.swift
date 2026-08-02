import Foundation
import XCTest
@testable import Reversible

final class RealizedPatchTests: XCTestCase {
    func testDecodesCompleteAtomicHandshakeWithoutSemanticCopies() throws {
        let response = try decodeFixture()
        guard case .published(let patch) = response else {
            return XCTFail("expected a published patch")
        }

        XCTAssertEqual(patch.vocabularyFingerprint, "fnv1a64:0123456789abcdef")
        XCTAssertEqual(patch.generation, .init(programVersion: 41, controlVersion: 77))
        XCTAssertEqual(
            patch.node(id: "src"),
            .init(
                id: "src",
                kind: try NodeKindID(validating: "future_source"),
                status: .active
            )
        )
        XCTAssertEqual(patch.node(id: "parked")?.status, .excluded)
        XCTAssertEqual(
            patch.inlet(nodeID: "out", port: "in")?.state,
            .wired(sources: ["src"])
        )
        XCTAssertEqual(
            patch.inlet(nodeID: "src", port: "frequency")?.state,
            .normalled
        )

        let futureDiscipline = try WriteDisciplineID(validating: "future_anchor")
        XCTAssertEqual(
            patch.parameterStatus(nodeID: "src", port: "frequency"),
            .live(discipline: futureDiscipline)
        )
        XCTAssertEqual(
            patch.parameterStatus(nodeID: "src", port: "structural_rows"),
            .structural
        )
        XCTAssertEqual(
            patch.parameterStatus(nodeID: "parked", port: "anything"),
            .inactive
        )
        XCTAssertEqual(
            patch.taps,
            [
                .init(name: "out", instance: "__root__", output: "out", slot: "__root__.out"),
                .init(name: "src", instance: "__root__", output: "tap:src", slot: "__root__.tap:src"),
            ]
        )

        // The old diagnostic shape is derived byte-for-fact from this same
        // response; no second server query is needed to bind a scope.
        XCTAssertEqual(
            patch.legacyTapList["taps"],
            .array(patch.taps.map { tap in
                .object([
                    "name": .string(tap.name),
                    "instance": .string(tap.instance),
                    "output": .string(tap.output),
                    "slot": .string(tap.slot),
                ])
            })
        )
    }

    func testSupersededAndStaleResponsesNeverReplaceRunningTruth() throws {
        guard case .published(let first) = try decodeFixture() else {
            return XCTFail("expected a published patch")
        }
        var store = RealizedPatchStore()
        XCTAssertEqual(store.adopt(.published(first)), .adopted(first))

        XCTAssertEqual(
            store.adopt(.superseded),
            .superseded(activeGeneration: first.generation)
        )
        XCTAssertEqual(store.current, first)

        let stale = RealizedPatch(
            vocabularyFingerprint: first.vocabularyFingerprint,
            generation: .init(programVersion: 40, controlVersion: 999),
            nodes: first.nodes,
            inlets: first.inlets,
            liveParameters: first.liveParameters,
            taps: first.taps
        )
        XCTAssertEqual(
            store.adopt(.published(stale)),
            .stale(candidate: stale.generation, active: first.generation)
        )
        XCTAssertEqual(store.current, first)

        let next = RealizedPatch(
            vocabularyFingerprint: first.vocabularyFingerprint,
            generation: .init(programVersion: 42, controlVersion: 78),
            nodes: first.nodes,
            inlets: first.inlets,
            liveParameters: first.liveParameters,
            taps: first.taps
        )
        XCTAssertEqual(store.adopt(.published(next)), .adopted(next))
        XCTAssertEqual(store.current, next)
    }

    func testDecodesExplicitSupersededNonPublication() throws {
        XCTAssertEqual(
            try PatchCompileResponse.decode(.object(["superseded": .bool(true)])),
            .superseded
        )
    }

    func testRejectsIncompleteAndIncoherentBindings() throws {
        XCTAssertThrowsError(
            try PatchCompileResponse.decode(.object(["ok": .bool(true)]))
        ) { error in
            XCTAssertEqual(
                error as? RealizedPatchDecodeError,
                .missingField("vocabulary_fingerprint")
            )
        }

        var raw = try fixtureJSON()
        guard case .object(var object) = raw,
              case .array(var taps)? = object["taps"],
              case .object(var first)? = taps.first
        else { return XCTFail("fixture shape") }
        first["slot"] = .string("different.slot")
        taps[0] = .object(first)
        object["taps"] = .array(taps)
        raw = .object(object)
        XCTAssertThrowsError(try PatchCompileResponse.decode(raw)) { error in
            XCTAssertEqual(
                error as? RealizedPatchDecodeError,
                .invalidField("taps[0].slot")
            )
        }
    }

    func testRejectsGenerationBeyondJSONValuesExactIntegerRange() throws {
        var raw = try fixtureJSON()
        guard case .object(var object) = raw else { return XCTFail("fixture shape") }
        object["program_version"] = .number(9_007_199_254_740_992)
        XCTAssertThrowsError(try PatchCompileResponse.decode(.object(object))) { error in
            XCTAssertEqual(
                error as? RealizedPatchDecodeError,
                .invalidField("program_version")
            )
        }
    }

    private func decodeFixture() throws -> PatchCompileResponse {
        try PatchCompileResponse.decode(fixtureJSON())
    }

    private func fixtureJSON() throws -> JSONValue {
        let url = try XCTUnwrap(
            Bundle.module.url(forResource: "realized-patch-valid", withExtension: "json")
        )
        return try JSONDecoder().decode(JSONValue.self, from: Data(contentsOf: url))
    }
}
