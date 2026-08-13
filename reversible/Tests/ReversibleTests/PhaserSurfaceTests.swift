import Foundation
import XCTest
@testable import Reversible

final class PhaserSurfaceTests: XCTestCase {
    func testModalPhaserDemoLoadsItsCompleteConnectionChain() {
        let graph = FactoryPatches.modalPhaser

        XCTAssertEqual(graph.order, [
            "source1", "resonator2", "phaser3", "reverb4", "out5", "scope6",
        ])
        XCTAssertEqual(graph.nodes["source1"]?.values["freq"], 0.63)
        XCTAssertEqual(graph.nodes["resonator2"]?.inputs["addr"], ["source1"])
        XCTAssertEqual(graph.nodes["phaser3"]?.inputs["in"], ["resonator2"])
        XCTAssertEqual(graph.nodes["reverb4"]?.inputs["in"], ["phaser3"])
        XCTAssertEqual(graph.nodes["out5"]?.inputs["in"], ["reverb4"])
        XCTAssertEqual(graph.nodes["scope6"]?.inputs["ch1"], ["out5"])
        XCTAssertEqual(
            graph.nodes["scope6"]?.values["window"],
            ScopeSignalKnowledge.visibleCycles / 220
        )
        XCTAssertTrue(graph.nodes["phaser3"]?.kind.spec.modal == true)
    }

    func testReverbExposesOnlyItsUsefulDecayControl() {
        XCTAssertEqual(NodeKind.reverb.spec.knobs.map(\.name), ["rt60"])
    }

    func testScopeDerivesTimebaseFromConnectedSignalKnowledge() throws {
        var graph = FactoryPatches.modalPhaser

        // The sub-Hz source addresses/retriggers the resonator; it is not the
        // pitch seen at the terminal scope. Follow the modal/audio chain to
        // the resonator fundamental instead.
        XCTAssertEqual(
            ScopeSignalKnowledge.frequencies(
                for: "out5",
                nodes: graph.nodes
            ),
            [220]
        )
        let originalWindow = try XCTUnwrap(
            ScopeSignalKnowledge.recommendedWindow(
                for: ["out5"],
                nodes: graph.nodes
            )
        )
        XCTAssertEqual(originalWindow, 4.0 / 220, accuracy: 1e-12)

        var resonator = try XCTUnwrap(graph.nodes["resonator2"])
        resonator.values["freq"] = 40
        graph.nodes[resonator.id] = resonator
        let retunedWindow = try XCTUnwrap(
            ScopeSignalKnowledge.recommendedWindow(
                for: ["out5"],
                nodes: graph.nodes
            )
        )
        XCTAssertEqual(retunedWindow, 0.1, accuracy: 1e-12)
        XCTAssertTrue(ScopeSignalKnowledge.isAutomaticWindow(
            0.1,
            sourceIDs: ["out5"],
            nodes: graph.nodes
        ))
        XCTAssertFalse(ScopeSignalKnowledge.isAutomaticWindow(
            0.08,
            sourceIDs: ["out5"],
            nodes: graph.nodes
        ))
    }

    func testScopeTimebaseClampsForSubHertzSignals() throws {
        let graph = FactoryPatches.modalPhaser
        let window = try XCTUnwrap(
            ScopeSignalKnowledge.recommendedWindow(
                for: ["source1"],
                nodes: graph.nodes
            )
        )
        XCTAssertEqual(window, ScopeSignalKnowledge.maximumWindow)
        XCTAssertGreaterThan(
            ScopeSignalKnowledge.acquisitionSpanSamples(
                displaySpanSamples: Int(
                    ScopeSignalKnowledge.maximumWindow * PatchModel.sampleRate
                ),
                frequency: 0.63,
                sampleRate: PatchModel.sampleRate
            ),
            ObservationBinding.maximumPointBudget
        )
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

    func testLiveParameterFailuresNeverRequestGraphRelowering() {
        let failures: [EngineError] = [
            .notRunning,
            .exited,
            .timeout(method: "set_param"),
            .pendingLimit(lane: "control", limit: 8),
            .rpc("socket closed"),
            .engine(code: "unknown_param", message: "missing slot"),
        ]

        for failure in failures {
            XCTAssertEqual(
                ParamWriteFailurePolicy.recovery(for: failure),
                .reportOnly
            )
        }
        XCTAssertEqual(
            ParamWriteFailurePolicy.status(
                name: "master.velocity",
                error: EngineError.timeout(method: "set_param")
            ),
            "parameter master.velocity: timeout: set_param"
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
