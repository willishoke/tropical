import XCTest
@testable import Reversible

final class EngineRPCLaneTests: XCTestCase {
    func testTrafficClassesHaveIndependentConnections() {
        XCTAssertEqual(EngineRPCLane.lane(for: "load_patch_graph"), .graph)
        XCTAssertEqual(EngineRPCLane.lane(for: "get_vocabulary"), .graph)
        XCTAssertEqual(EngineRPCLane.lane(for: "set_param"), .control)
        XCTAssertEqual(EngineRPCLane.lane(for: "playback_position"), .scope)
        XCTAssertEqual(EngineRPCLane.lane(for: "render_window"), .scope)
        XCTAssertEqual(EngineRPCLane.lane(for: "get_telemetry"), .telemetry)
        XCTAssertEqual(EngineRPCLane.lane(for: "audio_status"), .telemetry)
    }
}
