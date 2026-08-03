import XCTest
@testable import Reversible

final class LatestValueBufferTests: XCTestCase {
    func testFirstAndFinalAreRetainedDuringMonotonicGesture() {
        var buffer = LatestValueBuffer(first: 0)
        buffer.offer(0.2)
        buffer.offer(0.4)
        buffer.offer(0.9)

        XCTAssertEqual(buffer.pendingCount, 2)
        XCTAssertEqual(buffer.pop(), 0)
        XCTAssertEqual(buffer.pop(), 0.9)
        XCTAssertNil(buffer.pop())
    }

    func testOneTurningPointAndFinalAreRetained() {
        var buffer = LatestValueBuffer(first: 0)
        buffer.offer(0.4)
        buffer.offer(0.8)
        buffer.offer(0.7)
        buffer.offer(0.3)

        XCTAssertEqual(buffer.pendingCount, 3)
        XCTAssertEqual(buffer.pop(), 0)
        XCTAssertEqual(buffer.pop(), 0.8)
        XCTAssertEqual(buffer.pop(), 0.3)
        XCTAssertNil(buffer.pop())
    }

    func testQueueStaysBoundedAcrossAdversarialOscillation() {
        var buffer = LatestValueBuffer(first: 0.5)
        for index in 0..<10_000 {
            buffer.offer(index.isMultiple(of: 2) ? 1 : 0)
            XCTAssertLessThanOrEqual(buffer.pendingCount, 3)
        }
    }

    func testNonFiniteEventsAreIgnored() {
        var buffer = LatestValueBuffer(first: 0.25)
        buffer.offer(.nan)
        buffer.offer(.infinity)
        XCTAssertEqual(buffer.pop(), 0.25)
    }
}
