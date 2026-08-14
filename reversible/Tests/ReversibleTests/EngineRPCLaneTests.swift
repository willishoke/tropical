import Foundation
import XCTest
@testable import Reversible

final class EngineRPCLaneTests: XCTestCase {
    func testDirectSwiftPMLaunchFindsEngineInOwningCheckout() {
        let candidates = Engine.binaryCandidates(
            resourceURL: nil,
            environment: [:],
            executableURL: URL(
                fileURLWithPath: "/work/tropical/reversible/.build/arm64-apple-macosx/debug/Reversible"
            )
        )

        XCTAssertEqual(
            candidates.map(\.path),
            ["/work/tropical/lean/.lake/build/bin/frontend"]
        )
    }

    func testEngineLookupDoesNotInferCheckoutForUnrelatedExecutable() {
        XCTAssertTrue(Engine.binaryCandidates(
            resourceURL: nil,
            environment: [:],
            executableURL: URL(fileURLWithPath: "/Applications/Other.app/Contents/MacOS/Other")
        ).isEmpty)
    }

    func testChildDefaultsToMetalButPreservesExplicitJITOverride() {
        let product = Engine.childEnvironment(inherited: [:])
        XCTAssertEqual(product["TROPICAL_BACKEND"], "metal")
        XCTAssertEqual(product["TROPICAL_ARROW"], "1")
        XCTAssertEqual(product["TROPICAL_BUFFER_LENGTH"], "128")
        XCTAssertEqual(product["TROPICAL_METAL_RENDER_TILE_FRAMES"], "128")

        let diagnostic = Engine.childEnvironment(inherited: [
            "TROPICAL_BACKEND": "jit",
            "TROPICAL_BUFFER_LENGTH": "512",
            "TROPICAL_METAL_RENDER_TILE_FRAMES": "1024",
        ])
        XCTAssertEqual(diagnostic["TROPICAL_BACKEND"], "jit")
        XCTAssertEqual(diagnostic["TROPICAL_BUFFER_LENGTH"], "512")
        XCTAssertEqual(diagnostic["TROPICAL_METAL_RENDER_TILE_FRAMES"], "1024")
    }

    func testTrafficClassesHaveIndependentConnections() {
        XCTAssertEqual(EngineRPCLane.lane(for: "load_patch_graph"), .graph)
        XCTAssertEqual(EngineRPCLane.lane(for: "get_vocabulary"), .graph)
        XCTAssertEqual(EngineRPCLane.lane(for: "set_param"), .control)
        XCTAssertEqual(EngineRPCLane.lane(for: "playback_position"), .control)
        XCTAssertEqual(EngineRPCLane.lane(for: "render_window"), .scope)
        XCTAssertEqual(EngineRPCLane.lane(for: "get_telemetry"), .telemetry)
        XCTAssertEqual(EngineRPCLane.lane(for: "audio_status"), .telemetry)
    }
}
