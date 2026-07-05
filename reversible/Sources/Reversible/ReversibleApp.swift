import SwiftUI

@main
struct ReversibleApp: App {
    // A bare SwiftPM executable is not an .app bundle, so macOS launches it
    // as a background process — no Dock icon, window never activated. Claim
    // regular-app status and take focus, or `swift run` shows nothing.
    init() {
        DispatchQueue.main.async {
            NSApp.setActivationPolicy(.regular)
            NSApp.activate(ignoringOtherApps: true)
            NSApp.windows.first?.makeKeyAndOrderFront(nil)
        }
    }

    var body: some Scene {
        WindowGroup("tropical playground") {
            RootView()
                .frame(minWidth: 1100, minHeight: 700)
                .preferredColorScheme(.dark)
        }
        .defaultSize(width: 1400, height: 900)
    }
}

struct RootView: View {
    @StateObject private var model = PatchModel()

    var body: some View {
        VStack(spacing: 0) {
            CanvasView()
            footer
        }
        .background(Theme.bg)
        .toolbar { PatchToolbar(model: model) }
        .environmentObject(model)
    }

    private var footer: some View {
        Text("＋ on an inlet patches a source (only legal ones are offered) · click a color chip to disconnect · an outlet's color marks its destinations · drag a knob vertically · the patch lowers through the arrow slide on every edit")
            .font(Theme.monoSmall)
            .foregroundStyle(Theme.muted)
            .lineLimit(1)
            .truncationMode(.tail)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 5)
            .background(Theme.panel)
            .overlay(Rectangle().frame(height: 1).foregroundStyle(Theme.edge), alignment: .top)
    }
}
