import Foundation
import XCTest
@testable import Reversible

final class PackageSmokeTests: XCTestCase {
    func testFixtureResourcesAreBundledAndDecodable() throws {
        let url = try XCTUnwrap(
            Bundle.module.url(
                forResource: "package-smoke",
                withExtension: "json"
            )
        )
        let value = try JSONDecoder().decode(
            JSONValue.self,
            from: Data(contentsOf: url)
        )

        XCTAssertEqual(value["schema"]?.stringValue, "reversible-package-smoke")
        XCTAssertEqual(value["version"]?.doubleValue, 1)
        XCTAssertEqual(value["enabled"]?.boolValue, true)
    }
}
