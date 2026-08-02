import Foundation
import XCTest
@testable import Reversible

final class PatchDocumentV2LosslessTests: XCTestCase {
    func testV2RoundTripPreservesUnknownFieldsAndAuthoredOrder() throws {
        let document = try PatchDocumentV2.decode(from: fixture("document-v2-valid"))
        let report = document.validate(against: try vocabulary())

        XCTAssertTrue(report.canCompile, "\(report.blockers)")
        XCTAssertEqual(document.presentation.order, [
            "source1", "control2", "processor3", "out4", "scope5",
        ])
        XCTAssertEqual(document.nodes[2].inputs["in"], ["source1"])
        XCTAssertEqual(document.unknownFields["vendor_top"], .object(["kept": .array([.number(1), .number(2), .number(3)])]))
        XCTAssertEqual(document.nodes[0].unknownFields["vendor_node"], .object(["kept": .bool(true)]))
        XCTAssertEqual(document.monitors[0].unknownFields["vendor_monitor"], .string("kept"))

        let roundTrip = try PatchDocumentV2.decode(from: document.encoded())
        XCTAssertEqual(roundTrip, document)
    }

    func testV1MigrationSeparatesMonitorAndRequiresExplicitSave() throws {
        let result = try PatchDocumentV1Migrator.migrate(
            data: fixture("document-v1-valid"),
            vocabulary: try vocabulary()
        )
        let repeated = try PatchDocumentV1Migrator.migrate(
            data: fixture("document-v1-valid"),
            vocabulary: try vocabulary()
        )

        XCTAssertTrue(result.requiresExplicitV2Save)
        XCTAssertEqual(repeated.report, result.report)
        XCTAssertTrue(result.report.canCompile, "\(result.report.blockers)")
        XCTAssertEqual(result.document.nodes.map(\.id), ["source1", "out2"])
        XCTAssertEqual(result.document.monitors.map(\.id), ["scope3"])
        XCTAssertEqual(result.document.monitors[0].kind.rawValue, "client.monitor.scope")
        XCTAssertEqual(result.document.monitors[0].inputs["ch1"], ["source1"])
        XCTAssertEqual(result.document.nodes[0].structuralParams["future_value"], .object([
            "curve": .array([.number(0), .number(1), .number(0)]),
        ]))
        XCTAssertEqual(result.document.unknownFields["vendor_top"], .object([
            "author": .string("future-client"),
        ]))
        XCTAssertEqual(result.document.nodes[0].unknownFields["vendor_node"], .object([
            "kept": .bool(true),
        ]))
        XCTAssertEqual(result.report.facts.last?.code, .explicitSaveRequired)
        XCTAssertEqual(
            result.report.facts.filter { $0.code == .preservedUnknownNodeField }.map(\.path),
            ["$.nodes[0].vendor_node"]
        )
    }

    func testV1UnknownDataStaysSerializableInBlockedV2Document() throws {
        let result = try PatchDocumentV1Migrator.migrate(
            data: fixture("document-v1-unknown"),
            vocabulary: try vocabulary()
        )
        let codes = Set(result.report.blockers.map(\.code))

        XCTAssertTrue(codes.contains(.unknownKind))
        XCTAssertTrue(codes.contains(.unknownPort))
        XCTAssertTrue(codes.contains(.danglingEdge))
        XCTAssertTrue(codes.contains(.selfEdge))
        XCTAssertFalse(result.report.canCompile)

        let mystery = try XCTUnwrap(result.document.nodes.first { $0.id == "mystery3" })
        XCTAssertEqual(mystery.kind.rawValue, "future_unserved_kind")
        XCTAssertEqual(mystery.inputs["future_port"], ["missing9", "mystery3"])
        XCTAssertEqual(mystery.structuralParams["future_structural"], .object([
            "rows": .array([
                .array([.number(1), .number(2)]),
                .array([.number(3), .number(4)]),
            ]),
        ]))
        XCTAssertEqual(mystery.unknownFields["future_node_field"], .object([
            "must": .string("survive"),
        ]))
        XCTAssertEqual(result.document.unknownFields["unknown_top_field"], .object([
            "must": .string("survive"),
        ]))

        let roundTrip = try PatchDocumentV2.decode(from: result.document.encoded())
        XCTAssertEqual(roundTrip, result.document)
    }

    func testCycleIsReportedWithoutMutatingTopology() throws {
        let document = try PatchDocumentV2.decode(from: fixture("document-v2-cycle"))
        let report = document.validate(against: try vocabulary())

        XCTAssertEqual(report.blockers.filter { $0.code == .cycle }.count, 1)
        XCTAssertEqual(document.nodes[0].inputs["in"], ["processor2"])
        XCTAssertEqual(document.nodes[1].inputs["in"], ["processor1"])
    }

    func testDescriptorArityColorFiniteAndOutputValidation() throws {
        let engineVocabulary = try vocabulary()
        var document = try PatchDocumentV2.decode(from: fixture("document-v2-valid"))
        let processorIndex = try XCTUnwrap(document.nodes.firstIndex { $0.id == "processor3" })

        document.nodes[processorIndex].inputs["in"] = ["source1", "source1"]
        XCTAssertTrue(document.validate(against: engineVocabulary).blockers.contains {
            $0.code == .arityExceeded
        })

        document.nodes[processorIndex].inputs["in"] = ["control2"]
        XCTAssertTrue(document.validate(against: engineVocabulary).blockers.contains {
            $0.code == .incompatibleConnection
        })

        document.nodes[processorIndex].structuralParams["not_finite"] = .number(.infinity)
        document.output = nil
        let blockers = document.validate(against: engineVocabulary).blockers
        XCTAssertTrue(blockers.contains { $0.code == .nonFiniteNumber })
        XCTAssertTrue(blockers.contains { $0.code == .invalidOutput })
    }

    private func vocabulary() throws -> EngineVocabulary {
        try EngineVocabulary.decode(from: fixture("document-vocabulary-v1"))
    }

    private func fixture(_ name: String) throws -> Data {
        let url = try XCTUnwrap(Bundle.module.url(forResource: name, withExtension: "json"))
        return try Data(contentsOf: url)
    }
}
