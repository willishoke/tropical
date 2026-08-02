import SwiftUI

/// Async-signal-safe last resort. A fatal crash (SIGSEGV/SIGABRT/…) never
/// reaches applicationWillTerminate, so reap the engine right here: `kill` is
/// on the async-signal-safe list, unlike the Process API or any dispatch hop.
/// Then restore the default disposition and re-raise so the crash still cores
/// and reports normally.
private func reapEngineOnCrash(_ sig: Int32) {
    let pid = gEngineChildPID
    if pid > 0 { kill(pid_t(pid), SIGKILL) }
    signal(sig, SIG_DFL)
    raise(sig)
}

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
    // The model lives at APP scope, not in RootView: the File menu commands
    // are scene-level and need the same object the canvas is editing.
    @StateObject private var model = PatchModel()

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
        // Crashes are catchable (unlike an uncatchable SIGKILL of the app,
        // which only the startup sweep covers); reap the engine before dying.
        for sig in [SIGSEGV, SIGABRT, SIGILL, SIGFPE, SIGBUS] {
            signal(sig, reapEngineOnCrash)
        }
    }

    var body: some Scene {
        WindowGroup("tropical playground") {
            RootView(model: model)
                .frame(minWidth: 1100, minHeight: 700)
                .preferredColorScheme(.dark)
        }
        .defaultSize(width: 1400, height: 900)
        .commands {
            // Stock File-menu verbs with their stock shortcuts — a patch is a
            // document, so it should behave like one.
            CommandGroup(replacing: .newItem) {
                Button("New Patch") { Task { await model.newPatch() } }
                    .keyboardShortcut("n")
                Button("Open Patch…") { model.open() }
                    .keyboardShortcut("o")
                Menu("Open Recent") {
                    if model.recentDocumentURLs.isEmpty {
                        Text("No Recent Patches")
                    } else {
                        ForEach(model.recentDocumentURLs, id: \.self) { url in
                            Button(url.deletingPathExtension().lastPathComponent) {
                                model.openRecent(url)
                            }
                            .help(url.path)
                        }
                        Divider()
                        Button("Clear Menu") { model.clearRecentDocuments() }
                    }
                }
            }
            CommandGroup(replacing: .saveItem) {
                Button("Save") { model.save() }
                    .keyboardShortcut("s")
                Button("Save As…") { model.saveAs() }
                    .keyboardShortcut("s", modifiers: [.command, .shift])
            }
        }
    }
}

struct RootView: View {
    @ObservedObject var model: PatchModel

    var body: some View {
        VStack(spacing: 0) {
            CanvasView()
            footer
        }
        .background(Theme.bg)
        .toolbar { PatchToolbar(model: model) }
        .environmentObject(model)
        // The title and native close-button dot share the same canonical dirty
        // comparison, so every authored surface tells one document-state truth.
        .navigationTitle(model.documentURL?.deletingPathExtension().lastPathComponent
                         ?? "untitled patch")
        .background(
            WindowDocumentState(
                representedURL: model.documentURL,
                isEdited: model.isDocumentDirty)
            .frame(width: 0, height: 0))
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

/// Bridge document state into the hosting NSWindow without creating a second
/// document/model owner. AppKit supplies the standard edited indicator and
/// represented-file behavior while SwiftUI continues to own the scene.
private struct WindowDocumentState: NSViewRepresentable {
    let representedURL: URL?
    let isEdited: Bool

    func makeNSView(context: Context) -> WindowDocumentStateView {
        WindowDocumentStateView()
    }

    func updateNSView(_ view: WindowDocumentStateView, context: Context) {
        view.representedURL = representedURL
        view.isEdited = isEdited
        view.apply()
    }
}

private final class WindowDocumentStateView: NSView {
    var representedURL: URL?
    var isEdited = false

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        apply()
    }

    func apply() {
        window?.representedURL = representedURL
        window?.isDocumentEdited = isEdited
    }
}
