import Foundation
import XCTest
@testable import Reversible

final class EngineVocabularyTests: XCTestCase {
    func testDecodesUnknownFutureKindWithoutSemanticEdit() throws {
        let withheld = Set([try NodeKindID(validating: "bloomgong")])
        let vocabulary = try EngineVocabulary.decode(
            from: fixture(named: "vocabulary-valid-future-kind"),
            withholding: withheld
        )

        XCTAssertEqual(vocabulary.schemaID, "tropical_vocabulary")
        XCTAssertEqual(vocabulary.schemaVersion, 1)
        XCTAssertEqual(vocabulary.fingerprint, "fnv1a64:0123456789abcdef")
        XCTAssertEqual(vocabulary.colors.map(\.rawValue), ["signal", "modal", "control"])

        let futureID = try NodeKindID(validating: "future_modal_cloud")
        let future = try XCTUnwrap(vocabulary.descriptor(for: futureID))
        XCTAssertEqual(future.outlet?.rawValue, "modal")
        XCTAssertEqual(future.ports.map(\.name), ["frequency", "rate"])

        let frequency = try XCTUnwrap(future.port(named: "frequency"))
        XCTAssertEqual(frequency.accepts.map(\.rawValue), ["control"])
        XCTAssertFalse(frequency.multi)
        XCTAssertEqual(frequency.defaultValue, 440)
        XCTAssertEqual(frequency.discipline?.rawValue, "future_anchor")
        XCTAssertEqual(frequency.min, 20)
        XCTAssertEqual(frequency.max, 20_000)
        XCTAssertEqual(frequency.logarithmic, true)
        XCTAssertEqual(frequency.unit, "Hz")

        let rate = try XCTUnwrap(future.port(named: "rate"))
        XCTAssertEqual(rate.owner, "frequency")
    }

    func testConnectionTypeAndArityComeOnlyFromDescriptors() throws {
        let vocabulary = try EngineVocabulary.decode(
            from: fixture(named: "vocabulary-valid-future-kind")
        )
        let control = try NodeKindID(validating: "control_source")
        let future = try NodeKindID(validating: "future_modal_cloud")
        let sink = try NodeKindID(validating: "modal_sink")

        XCTAssertEqual(
            vocabulary.connectionDecision(
                from: control,
                to: future,
                port: "frequency",
                existingConnectionCount: 0
            ),
            .allowed
        )
        XCTAssertEqual(
            vocabulary.connectionDecision(
                from: future,
                to: sink,
                port: "in",
                existingConnectionCount: 0
            ),
            .allowed
        )
        XCTAssertEqual(
            vocabulary.connectionDecision(
                from: future,
                to: sink,
                port: "in",
                existingConnectionCount: 1
            ),
            .inletAtCapacity(kind: sink, port: "in")
        )
        XCTAssertEqual(
            vocabulary.connectionDecision(
                from: control,
                to: sink,
                port: "in",
                existingConnectionCount: 0
            ),
            .incompatibleColor(
                outlet: try PortColorID(validating: "control"),
                accepts: [try PortColorID(validating: "modal")]
            )
        )
        XCTAssertEqual(
            vocabulary.connectionDecision(
                from: sink,
                to: future,
                port: "frequency",
                existingConnectionCount: 0
            ),
            .sourceHasNoOutlet(sink)
        )
    }

    func testRejectsDuplicateKinds() throws {
        XCTAssertThrowsError(
            try EngineVocabulary.decode(from: fixture(named: "vocabulary-invalid-duplicate-kind"))
        ) { error in
            XCTAssertEqual(
                error as? EngineVocabularyError,
                .duplicateKind(try! NodeKindID(validating: "duplicate"))
            )
        }
    }

    func testRejectsDuplicatePorts() throws {
        XCTAssertThrowsError(
            try EngineVocabulary.decode(from: fixture(named: "vocabulary-invalid-duplicate-port"))
        ) { error in
            XCTAssertEqual(
                error as? EngineVocabularyError,
                .duplicatePort(
                    kind: try! NodeKindID(validating: "duplicated_ports"),
                    port: "in"
                )
            )
        }
    }

    func testRejectsSuppliedWithheldKindIfServed() throws {
        let withheld = Set([try NodeKindID(validating: "bloomgong")])
        XCTAssertThrowsError(
            try EngineVocabulary.decode(
                from: fixture(named: "vocabulary-invalid-withheld-kind"),
                withholding: withheld
            )
        ) { error in
            XCTAssertEqual(
                error as? EngineVocabularyError,
                .withheldKindServed(try! NodeKindID(validating: "bloomgong"))
            )
        }
    }

    func testRejectsUnsupportedSchemaVersion() throws {
        let data = Data(
            """
            {
              "schema": "tropical_vocabulary",
              "schema_version": 2,
              "fingerprint": "fnv1a64:0123456789abcdef",
              "rule": "engine-owned",
              "colors": ["signal"],
              "kinds": [{"kind": "future", "outlet": "signal", "ports": []}]
            }
            """.utf8
        )
        XCTAssertThrowsError(try EngineVocabulary.decode(from: data)) { error in
            XCTAssertEqual(error as? EngineVocabularyError, .unsupportedSchemaVersion(2))
        }
    }

    func testClientMonitorsAndThemesRemainPresentationOnly() throws {
        XCTAssertThrowsError(try ClientMonitorKindID(validating: "scope"))

        let scopeID = try ClientMonitorKindID(validating: "client.monitor.scope")
        let monitor = ClientMonitorDescriptor(
            kind: scopeID,
            displayName: "Scope",
            theme: NodeThemeOverride(displayName: nil, accentRGB: 0x7FE08C)
        )
        let monitors = try ClientMonitorCatalog(monitors: [monitor])
        XCTAssertEqual(monitors.descriptor(for: scopeID), monitor)

        let future = try NodeKindID(validating: "future_modal_cloud")
        let themes = EngineNodeThemeCatalog()
        XCTAssertNil(themes.override(for: future))
    }

    private func fixture(named name: String) throws -> Data {
        let url = try XCTUnwrap(Bundle.module.url(forResource: name, withExtension: "json"))
        return try Data(contentsOf: url)
    }
}
