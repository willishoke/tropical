import SwiftUI

@main
struct ReversibleApp: App {
    var body: some Scene {
        WindowGroup("tropical playground") {
            RootView()
                .frame(minWidth: 1100, minHeight: 700)
                .preferredColorScheme(.dark)
        }
        .defaultSize(width: 1400, height: 900)
    }
}

/// Placeholder root until the canvas lands (commit 4). Kept buildable so
/// every commit in the port runs.
struct RootView: View {
    var body: some View {
        ZStack {
            Theme.bg.ignoresSafeArea()
            Text("tropical")
                .font(.system(size: 13, weight: .bold, design: .monospaced))
                .foregroundStyle(Theme.muted)
        }
    }
}
