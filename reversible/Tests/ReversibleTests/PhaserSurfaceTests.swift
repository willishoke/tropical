import Foundation
import XCTest
@testable import Reversible

final class PhaserSurfaceTests: XCTestCase {
    func testModalPhaserDemoLoadsItsCompleteConnectionChain() {
        let graph = FactoryPatches.modalPhaser

        XCTAssertEqual(graph.order, [
            "source7", "resonator1", "reverb2", "phaser3",
            "reverb4", "out5", "scope6",
        ])
        XCTAssertEqual(graph.nodes["source7"]?.values["freq"], 0.63)
        XCTAssertEqual(graph.nodes["resonator1"]?.inputs["addr"], ["source7"])
        XCTAssertEqual(graph.nodes["reverb2"]?.inputs["in"], ["resonator1"])
        XCTAssertEqual(graph.nodes["phaser3"]?.inputs["in"], ["reverb2"])
        XCTAssertEqual(graph.nodes["reverb4"]?.inputs["in"], ["phaser3"])
        XCTAssertEqual(graph.nodes["out5"]?.inputs["in"], ["reverb4"])
        XCTAssertEqual(graph.nodes["scope6"]?.inputs["ch1"], ["out5"])
        XCTAssertTrue(graph.nodes["phaser3"]?.kind.spec.modal == true)
    }

    func testConnectedInletMenuDoesNotClaimItHasNoSources() {
        XCTAssertEqual(
            InletMenuCopy.noLegalSources(hasExistingConnections: true),
            "No additional legal sources"
        )
        XCTAssertEqual(
            InletMenuCopy.noLegalSources(hasExistingConnections: false),
            "No legal sources"
        )
    }

    func testNativePhaserSurfaceMatchesServedContract() {
        let spec = NodeKind.phaser.spec

        XCTAssertEqual(spec.title, "Phaser")
        XCTAssertTrue(spec.modal)
        XCTAssertEqual(spec.inlets, ["in"])
        XCTAssertEqual(spec.outlets, ["out"])
        XCTAssertEqual(spec.knobs.map(\.name), ["center", "sweep", "rate", "mix"])
        XCTAssertEqual(spec.knobs.map(\.min), [40, 0, 0.02, 0])
        XCTAssertEqual(spec.knobs.map(\.max), [4000, 3, 8, 1])
        XCTAssertEqual(spec.knobs.map(\.def), [700, 1.5, 0.2, 0.5])
        XCTAssertEqual(spec.knobs.map(\.log), [true, false, true, false])
        XCTAssertTrue(NodeKind.phaser.wantsModal(inlet: "in"))
        XCTAssertFalse(NodeKind.phaser.wantsModal(inlet: "center"))
    }

    func testPatchDocumentRoundTripPreservesPhaserControlsAndModalEdge() throws {
        let document = PatchDocument(
            nodes: [
                .init(
                    id: "res1", kind: .resonator, x: 20, y: 30, hue: 30,
                    values: ["freq": 220, "decay": 4], inputs: ["addr": []]
                ),
                .init(
                    id: "ph2", kind: .phaser, x: 120, y: 30, hue: 167,
                    values: ["center": 810, "sweep": 1.75, "rate": 0.31, "mix": 0.42],
                    inputs: ["in": ["res1"]]
                ),
            ],
            order: ["res1", "ph2"], velocity: -1,
            panX: 11, panY: -7, autoArrange: true
        )

        let data = try JSONEncoder().encode(document)
        let decoded = try JSONDecoder().decode(PatchDocument.self, from: data)
        let phaser = try XCTUnwrap(decoded.nodes.first { $0.id == "ph2" })

        XCTAssertEqual(phaser.kind, .phaser)
        XCTAssertEqual(phaser.inputs["in"], ["res1"])
        XCTAssertEqual(phaser.values["center"], 810)
        XCTAssertEqual(phaser.values["sweep"], 1.75)
        XCTAssertEqual(phaser.values["rate"], 0.31)
        XCTAssertEqual(phaser.values["mix"], 0.42)
        XCTAssertEqual(decoded.order, ["res1", "ph2"])
        XCTAssertEqual(decoded.velocity, -1)
    }
}
