import XCTest
@testable import Reversible

final class ImplicitFanInSurfaceTests: XCTestCase {
    func testOrdinaryInputsUseImplicitFanIn() {
        XCTAssertTrue(NodeKind.delay.allowsMultiple(inlet: "in"))
        XCTAssertTrue(NodeKind.reverb.allowsMultiple(inlet: "in"))
        XCTAssertTrue(NodeKind.phaser.allowsMultiple(inlet: "in"))
        XCTAssertTrue(NodeKind.out.allowsMultiple(inlet: "in"))

        XCTAssertFalse(NodeKind.source.allowsMultiple(inlet: "freq"))
        XCTAssertFalse(NodeKind.sflange.allowsMultiple(inlet: "mod"))
        XCTAssertFalse(NodeKind.resonator.allowsMultiple(inlet: "addr"))
        XCTAssertFalse(NodeKind.scope.allowsMultiple(inlet: "ch1"))
    }

    func testPlumbingReducersRemainDecodableButLeaveTheAddMenu() {
        XCTAssertFalse(NodeKind.mix.appearsInAddMenu)
        XCTAssertFalse(NodeKind.modalmix.appearsInAddMenu)
        XCTAssertTrue(NodeKind.delay.appearsInAddMenu)
        XCTAssertTrue(NodeKind.phaser.appearsInAddMenu)
        XCTAssertFalse(NodeKind.out.appearsInAddMenu)
    }
}
