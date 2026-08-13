import SwiftUI

// Connections render as COLOR IDENTITY, not cables. Every node has a stable
// hue; an outlet wears its node's color, an inlet wears a chip per connected
// source. Fan-out reads as the same color at many inlets; fan-in as many
// chips on one inlet. Patching is selection, not drag: an inlet's menu lists
// ONLY the type-legal sources (control inlets → knobs, modal inlets → modal
// nodes, cycle-formers omitted) — the discipline enforced by what's offered,
// never by rejection.
struct CanvasView: View {
    @EnvironmentObject var model: PatchModel
    // The plane is infinite in both axes: node positions are WORLD
    // coordinates, model.pan maps world → view (on the model so arrange
    // can bring the graph back into view).
    @State private var panOrigin: CGSize?

    var body: some View {
        ZStack(alignment: .topLeading) {
            DotGrid(pan: model.pan)
                .gesture(
                    DragGesture(minimumDistance: 1)
                        .onChanged { g in
                            let o = panOrigin ?? model.pan
                            panOrigin = o
                            model.pan = CGSize(width: o.width + g.translation.width,
                                               height: o.height + g.translation.height)
                        }
                        .onEnded { _ in panOrigin = nil }
                )
            ForEach(model.order, id: \.self) { id in
                if let node = model.nodes[id] {
                    NodeView(node: node)
                        .offset(x: node.position.x + model.pan.width,
                                y: node.position.y + model.pan.height)
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .clipped()
    }
}

/// The 22px dot grid backdrop, phase-shifted by the pan so the plane reads
/// as infinite (dots scroll with the world).
struct DotGrid: View {
    let pan: CGSize

    var body: some View {
        Canvas { ctx, size in
            let step = Grid.unit
            let ox = pan.width.truncatingRemainder(dividingBy: step)
            let oy = pan.height.truncatingRemainder(dividingBy: step)
            var y = oy - step
            while y < size.height {
                var x = ox - step
                while x < size.width {
                    ctx.fill(
                        Path(ellipseIn: CGRect(x: x - 1, y: y - 1, width: 2, height: 2)),
                        with: .color(Theme.dotGrid))
                    x += step
                }
                y += step
            }
        }
        .background(Theme.bg)
        .contentShape(Rectangle())
    }
}

// ── Node ────────────────────────────────────────────────────────────────────
struct NodeView: View {
    @EnvironmentObject var model: PatchModel
    let node: PatchNode
    @State private var dragOrigin: CGPoint?

    private var spec: NodeSpec { node.kind.spec }

    private var truthBadge: (label: String, color: Color, help: String)? {
        switch model.presentationStatus(for: node) {
        case .active:
            ("active", Theme.truthActive, "active in the running program")
        case .excluded:
            ("excluded", Theme.truthExcluded, "authored but excluded from the running program")
        case .pending:
            ("pending", Theme.truthPending, "awaiting realized compile truth")
        case nil: nil
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            titleBar
            if node.kind == .scope {
                ScopeBody(node: node)
            } else {
                knobRow
                Spacer(minLength: 0)
                ports
            }
        }
        // VCV-style: every module has a defined footprint in grid units.
        .frame(width: Double(spec.gridSize.w) * Grid.unit,
               height: Double(spec.gridSize.h) * Grid.unit,
               alignment: .topLeading)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(Theme.edge))
        .shadow(color: .black.opacity(0.35), radius: 9, y: 6)
    }

    private var titleBar: some View {
        VStack(spacing: 3) {
            HStack(spacing: 6) {
                Circle().fill(node.color).frame(width: 9, height: 9)
                Text(spec.title)
                    .font(Theme.mono.bold())
                    .foregroundStyle(Theme.text)
                    .lineLimit(1)
                    .fixedSize(horizontal: true, vertical: false)
                Spacer(minLength: 4)
                if node.kind.isModule {
                    Button {
                        model.openModule(node.id)
                    } label: {
                        Image(systemName: "square.grid.3x3.square")
                            .foregroundStyle(Theme.muted)
                    }
                    .buttonStyle(.plain)
                    .help("open module")
                }
                if !spec.fixed {
                    Button {
                        model.deleteNode(node.id)
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(Theme.muted)
                    }
                    .buttonStyle(.plain)
                    .help("remove module")
                }
            }

            HStack(spacing: 6) {
                if let truthBadge {
                    Text(truthBadge.label)
                        .font(.system(size: 8, weight: .semibold, design: .monospaced))
                        .foregroundStyle(truthBadge.color)
                        .padding(.horizontal, 4).padding(.vertical, 2)
                        .background(
                            truthBadge.color.opacity(0.13),
                            in: Capsule()
                        )
                        .fixedSize()
                        .help(truthBadge.help)
                }
                Spacer(minLength: 4)
                Text(node.id)
                    .font(Theme.monoSmall)
                    .foregroundStyle(Theme.muted)
                    .lineLimit(1)
                    .fixedSize(horizontal: true, vertical: false)
            }
        }
        .padding(.horizontal, 9).padding(.vertical, 5)
        .overlay(Rectangle().frame(height: 1).foregroundStyle(Theme.edge), alignment: .bottom)
        .contentShape(Rectangle())
        .gesture(
            DragGesture(minimumDistance: 1, coordinateSpace: .global)
                .onChanged { g in
                    let origin = dragOrigin ?? node.position
                    dragOrigin = origin
                    // Snap live: the module lands where it reads. World
                    // coords are unbounded — the plane is infinite.
                    model.nodes[node.id]?.position = Grid.snap(CGPoint(
                        x: origin.x + g.translation.width,
                        y: origin.y + g.translation.height))
                }
                .onEnded { _ in dragOrigin = nil }
        )
    }

    @ViewBuilder
    private var knobRow: some View {
        // A knob whose name matches a wired inlet is overridden by the patch
        // (e.g. `freq` driven by a Knob node), so hide it.
        let shown = spec.knobs.filter {
            (node.inputs[$0.name] ?? []).isEmpty && node.bindings[$0.name] == nil
        }
        if !shown.isEmpty {
            HStack(alignment: .top, spacing: 8) {
                ForEach(shown, id: \.name) { k in
                    KnobView(node: node, knob: k)
                }
            }
            .padding(.horizontal, 9).padding(.vertical, 8)
        }
    }

    private var ports: some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 4) {
                ForEach(spec.inlets, id: \.self) { port in
                    InletView(node: node, port: port)
                }
            }
            Spacer(minLength: 12)
            VStack(alignment: .trailing, spacing: 4) {
                ForEach(spec.outlets, id: \.self) { port in
                    OutletView(node: node, port: port)
                }
            }
        }
        .padding(.horizontal, 6).padding(.bottom, 6)
    }
}

// ── Ports ───────────────────────────────────────────────────────────────────
/// An inlet: a menu of legal sources plus one color chip per connection.
/// Ordinary `in` ports retain every compatible selection as an implicit typed
/// fan-in; control/address/modulation ports replace their single source. Tap a
/// chip to disconnect. The menu is the whole patching gesture.
struct InletView: View {
    @EnvironmentObject var model: PatchModel
    let node: PatchNode
    let port: String

    private var sources: [String] { node.inputs[port] ?? [] }

    private func sourceTitle(_ sourceID: String) -> String {
        guard let source = model.nodes[sourceID] else { return sourceID }
        return "\(source.kind.spec.title) · \(sourceID)"
    }

    var body: some View {
        HStack(spacing: 4) {
            Menu {
                let legal = model.legalSources(for: node.id, port: port)
                if !sources.isEmpty {
                    Section("Connected to \(port)") {
                        ForEach(sources, id: \.self) { sourceID in
                            Button(role: .destructive) {
                                model.deleteEdge(
                                    to: node.id,
                                    port: port,
                                    from: sourceID
                                )
                            } label: {
                                Label(
                                    sourceTitle(sourceID),
                                    systemImage: "xmark.circle"
                                )
                            }
                        }
                    }
                }
                Section(sources.isEmpty ? "Connect source" : "Add source") {
                    if legal.isEmpty {
                        Text(InletMenuCopy.noLegalSources(
                            hasExistingConnections: !sources.isEmpty
                        ))
                    } else {
                        ForEach(legal) { src in
                            Button("\(src.kind.spec.title) · \(src.id)") {
                                model.connect(from: src.id, to: node.id, port: port)
                            }
                        }
                    }
                }
            } label: {
                Image(systemName: "plus.circle")
                    .font(.system(size: 11))
                    .foregroundStyle(Theme.muted)
            }
            .menuStyle(.borderlessButton)
            .menuIndicator(.hidden)
            .fixedSize()
            .help(sources.isEmpty
                  ? "patch a source into \(port)"
                  : "view, add, or disconnect sources for \(port)")

            ForEach(Array(sources.prefix(2)), id: \.self) { sourceID in
                HStack(spacing: 3) {
                    Circle()
                        .fill(model.nodes[sourceID]?.color ?? Theme.jack)
                        .frame(width: 10, height: 10)
                    Text(sourceID)
                        .font(.system(size: 8, design: .monospaced))
                        .foregroundStyle(Theme.text)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .frame(maxWidth: 62)
                }
                .padding(.horizontal, 4).padding(.vertical, 2)
                .background(
                    (model.nodes[sourceID]?.color ?? Theme.jack).opacity(0.13),
                    in: Capsule()
                )
                .contentShape(Capsule())
                .onTapGesture {
                    model.deleteEdge(to: node.id, port: port, from: sourceID)
                }
                .help("\(sourceTitle(sourceID)) → \(port) · click to disconnect")
            }
            if sources.count > 2 {
                Text("+\(sources.count - 2)")
                    .font(.system(size: 8, design: .monospaced))
                    .foregroundStyle(Theme.muted)
                    .help("\(sources.count - 2) more connected sources; open the menu to inspect")
            }

            Text(port).font(Theme.monoSmall).foregroundStyle(Theme.muted)
        }
    }
}

enum InletMenuCopy {
    static func noLegalSources(hasExistingConnections: Bool) -> String {
        hasExistingConnections
            ? "No additional legal sources"
            : "No legal sources"
    }
}

/// An outlet wears its node's identity color — find this color on other
/// modules' inlets to read the patch.
struct OutletView: View {
    let node: PatchNode
    let port: String

    var body: some View {
        HStack(spacing: 5) {
            Text(port).font(Theme.monoSmall).foregroundStyle(Theme.muted)
            Circle()
                .fill(node.color)
                .frame(width: 10, height: 10)
                .help("\(node.id) — this color marks its destinations")
        }
    }
}
