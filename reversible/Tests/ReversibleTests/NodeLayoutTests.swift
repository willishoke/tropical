import XCTest
@testable import Reversible

final class NodeLayoutTests: XCTestCase {
    func testCompactNodesReserveIdentityHeaderWidth() {
        for kind in NodeKind.allCases where kind != .scope {
            XCTAssertGreaterThanOrEqual(
                kind.spec.gridSize.w, 6,
                "\(kind) must fit its title, truth badge, and full node id"
            )
        }
    }

    func testKnobAndJackBodiesReserveTwoLineHeaderHeight() {
        XCTAssertGreaterThanOrEqual(NodeKind.source.spec.gridSize.h, 7)
        XCTAssertGreaterThanOrEqual(NodeKind.reverb.spec.gridSize.h, 7)
        XCTAssertGreaterThanOrEqual(NodeKind.out.spec.gridSize.h, 4)
    }
}
