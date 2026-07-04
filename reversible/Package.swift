// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "Reversible",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "Reversible",
            path: "Sources/Reversible"
        )
    ]
)
