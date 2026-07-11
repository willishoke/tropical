import SwiftUI

/// The app spawns the engine as a CHILD, but the kernel doesn't reap children
/// with their parent: any exit that skips teardown leaves an orphaned engine
/// holding the DAC — audio that nothing can stop. Funnel every exit path
/// (Cmd-Q, last window closed, SIGTERM/SIGINT) through willTerminate, where
/// the model has registered the engine's synchronous child-kill.
final class AppDelegate: NSObject, NSApplicationDelegate {
    static var onTerminate: (() -> Void)?

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    func applicationWillTerminate(_ notification: Notification) {
        Self.onTerminate?()
    }
}

@main
struct ReversibleApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var delegate

    // Signal sources must stay referenced or the handler dies with them.
    private static var signalSources: [DispatchSourceSignal] = []

    // A bare SwiftPM executable is not an .app bundle, so macOS launches it
    // as a background process — no Dock icon, window never activated. Claim
    // regular-app status and take focus, or `swift run` shows nothing.
    init() {
        DispatchQueue.main.async {
            NSApp.setActivationPolicy(.regular)
            NSApp.activate(ignoringOtherApps: true)
            NSApp.windows.first?.makeKeyAndOrderFront(nil)
        }
        // A raw signal's default action exits without running willTerminate,
        // orphaning the engine — reroute SIGTERM/SIGINT into the normal
        // termination path (signal handlers can't touch AppKit; a dispatch
        // source runs on the main queue and can).
        for sig in [SIGTERM, SIGINT] {
            signal(sig, SIG_IGN)
            let src = DispatchSource.makeSignalSource(signal: sig, queue: .main)
            src.setEventHandler { NSApp.terminate(nil) }
            src.resume()
            Self.signalSources.append(src)
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
