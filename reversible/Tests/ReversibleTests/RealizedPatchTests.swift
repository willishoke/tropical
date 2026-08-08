import Foundation
import XCTest
@testable import Reversible

final class RealizedPatchTests: XCTestCase {
    func testObservationTapSelectionIsStableDeduplicatedAndEngineOnly() {
        XCTAssertEqual(
            ObservationTapSelection.stableSourceIDs(
                candidates: ["src2", "out", "src1", "src2", "scope", "src1"],
                allowedEngineNodeIDs: ["src1", "src2"]
            ),
            ["src2", "src1"]
        )
        XCTAssertEqual(
            ObservationTapSelection.expectedHandshakeNames(
                requestedSourceIDs: []
            ),
            ["out"]
        )
    }

    func testDecodesCompleteHandshakeAgainstExactSemanticExpectations() throws {
        guard case .published(let patch) = try PatchCompileResponse.decode(
            fixtureJSON(),
            expectedTapNames: ["out", "src"],
            expectedVocabularyFingerprint: "fnv1a64:0123456789abcdef"
        ) else { return XCTFail("expected published patch") }

        XCTAssertEqual(
            patch.generation,
            .init(programVersion: 41, controlVersion: 77)
        )
        XCTAssertEqual(patch.node(id: "parked")?.status, .excluded)
        XCTAssertEqual(
            patch.inlet(nodeID: "out", port: "in")?.state,
            .wired(sources: ["src"])
        )
        let futureDiscipline = try WriteDisciplineID(
            validating: "future_anchor"
        )
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
        XCTAssertEqual(patch.tap(named: "src")?.slot, "__root__.tap:src")
    }

    func testRejectsVocabularyAndCompleteTapTableMismatch() throws {
        XCTAssertThrowsError(try PatchCompileResponse.decode(
            fixtureJSON(),
            expectedTapNames: ["out", "src"],
            expectedVocabularyFingerprint: "fnv1a64:ffffffffffffffff"
        )) { error in
            XCTAssertEqual(
                error as? RealizedPatchDecodeError,
                .vocabularyMismatch(
                    expected: "fnv1a64:ffffffffffffffff",
                    actual: "fnv1a64:0123456789abcdef"
                )
            )
        }
        XCTAssertThrowsError(try PatchCompileResponse.decode(
            fixtureJSON(),
            expectedTapNames: ["out", "different"]
        )) { error in
            XCTAssertEqual(
                error as? RealizedPatchDecodeError,
                .invalidField("taps")
            )
        }
    }

    func testRejectsInexactGenerationAndIncoherentReferences() throws {
        guard case .object(var object) = try fixtureJSON() else {
            return XCTFail("fixture shape")
        }
        object["program_version"] = .number(9_007_199_254_740_992)
        XCTAssertThrowsError(try PatchCompileResponse.decode(.object(object))) {
            XCTAssertEqual(
                $0 as? RealizedPatchDecodeError,
                .invalidField("program_version")
            )
        }

        guard case .object(var valid) = try fixtureJSON(),
              case .array(var inputs)? = valid["inputs"],
              case .object(var wired)? = inputs.last
        else { return XCTFail("fixture input shape") }
        wired["sources"] = .array([.string("missing")])
        inputs[inputs.count - 1] = .object(wired)
        valid["inputs"] = .array(inputs)
        XCTAssertThrowsError(try PatchCompileResponse.decode(.object(valid))) {
            XCTAssertEqual(
                $0 as? RealizedPatchDecodeError,
                .invalidField("inputs[1].sources[0]")
            )
        }
    }

    func testCompileAndObservationTruthAdvanceAtomically() throws {
        guard case .published(let patch) = try PatchCompileResponse.decode(
            fixtureJSON()
        ) else { return XCTFail("expected published patch") }
        let graph: JSONValue = .object(["nodes": .array([])])
        var truth = PatchTruthStore()
        XCTAssertTrue(truth.adoptCompile(
            .adopted(patch),
            authoredRevision: 7,
            loweringRevision: 3,
            graph: graph
        ))
        XCTAssertEqual(truth.realized?.authoredRevision, 7)
        XCTAssertEqual(truth.realized?.loweringRevision, 3)
        XCTAssertEqual(truth.runningGeneration, patch.generation)

        XCTAssertTrue(truth.adoptObservation(.init(
            programVersion: 41,
            controlVersion: 80
        )))
        XCTAssertFalse(truth.adoptObservation(.init(
            programVersion: 41,
            controlVersion: 79
        )))
        XCTAssertFalse(truth.adoptObservation(.init(
            programVersion: 42,
            controlVersion: 81
        )))
        XCTAssertEqual(
            truth.runningGeneration,
            .init(programVersion: 41, controlVersion: 80)
        )

        XCTAssertFalse(truth.adoptCompile(
            .superseded(activeGeneration: patch.generation),
            authoredRevision: 8,
            loweringRevision: 4,
            graph: .null
        ))
        XCTAssertEqual(truth.realized?.graph, graph)
        truth.retagCurrent(authoredRevision: 8, loweringRevision: 3)
        XCTAssertEqual(truth.realized?.authoredRevision, 8)
    }

    private func fixtureJSON() throws -> JSONValue {
        let url = try XCTUnwrap(Bundle.module.url(
            forResource: "realized-patch-valid",
            withExtension: "json"
        ))
        return try JSONDecoder().decode(
            JSONValue.self,
            from: Data(contentsOf: url)
        )
    }
}
