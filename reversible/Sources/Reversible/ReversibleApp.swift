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

struct RootView: View {
    @StateObject private var model = PatchModel()

    var body: some View {
        VStack(spacing: 0) {
            ToolbarView()
            CanvasView()
            footer
        }
        .background(Theme.bg)
        .environmentObject(model)
    }

    private var footer: some View {
        Text("drag from an outlet (right) to an inlet (left) to patch downstream · drag a knob vertically to tweak · double-click a node to delete · the patch lowers through the arrow slide on every edit")
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
