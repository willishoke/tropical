import SwiftUI

// ── Jack geometry ───────────────────────────────────────────────────────────
// Jack dots report their centers (in canvas space) up through a preference;
// the wire layer reads the dictionary. The SwiftUI analog of app.js's
// jackCenter() DOM query.
struct JackID: Hashable {
    let node: String
    let dir: JackDir
    let port: String
}

enum JackDir: Hashable { case input, output }

struct JackCentersKey: PreferenceKey {
    static var defaultValue: [JackID: CGPoint] = [:]
    static func reduce(value: inout [JackID: CGPoint], nextValue: () -> [JackID: CGPoint]) {
        value.merge(nextValue()) { _, new in new }
    }
}

/// A wire drag in flight: from an outlet, tracking the pointer.
struct PendingWire {
    let fromNode: String
    var cursor: CGPoint
}

struct CanvasView: View {
    @EnvironmentObject var model: PatchModel
    @State private var jackCenters: [JackID: CGPoint] = [:]
    @State private var pendingWire: PendingWire?
    // The plane is infinite in both axes: node positions are WORLD
    // coordinates, `pan` maps world → view. Nodes render at world + pan, so
    // jack centers (measured in the named space) already include the pan and
    // wires/hit-tests need no second transform.
    @State private var pan: CGSize = .zero
    @State private var panOrigin: CGSize?

    var body: some View {
        ZStack(alignment: .topLeading) {
            DotGrid(pan: pan)
                .gesture(
                    DragGesture(minimumDistance: 1)
                        .onChanged { g in
                            let o = panOrigin ?? pan
                            panOrigin = o
                            pan = CGSize(width: o.width + g.translation.width,
                                         height: o.height + g.translation.height)
                        }
                        .onEnded { _ in panOrigin = nil }
                )
            WiresView(jackCenters: jackCenters, pendingWire: pendingWire)
            ForEach(model.order, id: \.self) { id in
                if let node = model.nodes[id] {
                    NodeView(node: node, pendingWire: $pendingWire)
                        .offset(x: node.position.x + pan.width,
                                y: node.position.y + pan.height)
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .clipped()
        .coordinateSpace(name: "canvas")
        .onPreferenceChange(JackCentersKey.self) {
            jackCenters = $0
            model.jackGeometry = $0
        }
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

// ── Wires ───────────────────────────────────────────────────────────────────
struct WiresView: View {
    @EnvironmentObject var model: PatchModel
    let jackCenters: [JackID: CGPoint]
    let pendingWire: PendingWire?

    private struct Edge: Identifiable {
        let from: String
        let to: String
        let port: String
        var id: String { "\(from)→\(to):\(port)" }
    }

    private var edges: [Edge] {
        model.order.compactMap { model.nodes[$0] }.flatMap { n in
            n.inputs.flatMap { port, srcs in
                srcs.map { Edge(from: $0, to: n.id, port: port) }
            }
        }
    }

    var body: some View {
        ZStack {
            ForEach(edges) { e in
                if let a = jackCenters[JackID(node: e.from, dir: .output, port: "out")],
                   let b = jackCenters[JackID(node: e.to, dir: .input, port: e.port)] {
                    let path = wirePath(a, b)
                    path.stroke(Theme.wire, lineWidth: 2.5)
                        .opacity(0.85)
                        // Fat invisible stroke = the click target for delete.
                        .contentShape(path.strokedPath(StrokeStyle(lineWidth: 12)))
                        .onTapGesture {
                            model.deleteEdge(to: e.to, port: e.port, from: e.from)
                        }
                }
            }
            if let p = pendingWire,
               let a = jackCenters[JackID(node: p.fromNode, dir: .output, port: "out")] {
                wirePath(a, p.cursor)
                    .stroke(Theme.jackHot, style: StrokeStyle(lineWidth: 2.5, dash: [5, 4]))
                    .allowsHitTesting(false)
            }
        }
    }

    private func wirePath(_ a: CGPoint, _ b: CGPoint) -> Path {
        let dx = max(40, abs(b.x - a.x) * 0.5)
        var p = Path()
        p.move(to: a)
        p.addCurve(
            to: b,
            control1: CGPoint(x: a.x + dx, y: a.y),
            control2: CGPoint(x: b.x - dx, y: b.y))
        return p
    }
}

// ── Node ────────────────────────────────────────────────────────────────────
struct NodeView: View {
    @EnvironmentObject var model: PatchModel
    let node: PatchNode
    @Binding var pendingWire: PendingWire?
    @State private var dragOrigin: CGPoint?

    private var spec: NodeSpec { node.kind.spec }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            titleBar
            body_
            Spacer(minLength: 0)
            jacks
        }
        // VCV-style: every module has a defined footprint in grid units.
        .frame(width: Double(spec.gridSize.w) * Grid.unit,
               height: Double(spec.gridSize.h) * Grid.unit,
               alignment: .topLeading)
        .background(Theme.panel)
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(Theme.edge))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .shadow(color: .black.opacity(0.35), radius: 9, y: 6)
        .onTapGesture(count: 2) { model.deleteNode(node.id) }
    }

    private var titleBar: some View {
        HStack(spacing: 6) {
            Circle().fill(spec.accent).frame(width: 9, height: 9)
            Text(spec.title).font(Theme.mono.bold()).foregroundStyle(Theme.text)
            Spacer(minLength: 8)
            Text(node.id).font(Theme.monoSmall).foregroundStyle(Theme.muted)
        }
        .padding(.horizontal, 9).padding(.vertical, 6)
        .background(Theme.panel)
        .overlay(Rectangle().frame(height: 1).foregroundStyle(Theme.edge), alignment: .bottom)
        .contentShape(Rectangle())
        .gesture(
            DragGesture(minimumDistance: 1, coordinateSpace: .named("canvas"))
                .onChanged { g in
                    let origin = dragOrigin ?? node.position
                    dragOrigin = origin
                    // Snap live (not on release): the module lands where it
                    // reads, and wires re-route against the settled position.
                    // World coords are unbounded — the plane is infinite.
                    model.nodes[node.id]?.position = Grid.snap(CGPoint(
                        x: origin.x + g.translation.width,
                        y: origin.y + g.translation.height))
                }
                .onEnded { _ in dragOrigin = nil }
        )
    }

    @ViewBuilder
    private var body_: some View {
        // A knob whose name matches a wired inlet is overridden by the patch
        // cord (e.g. `freq` driven by a Knob node), so hide it.
        let shown = spec.knobs.filter { (node.inputs[$0.name] ?? []).isEmpty }
        if !shown.isEmpty {
            HStack(alignment: .top, spacing: 8) {
                ForEach(shown, id: \.name) { k in
                    KnobView(node: node, knob: k)
                }
            }
            .padding(.horizontal, 9).padding(.vertical, 8)
        }
    }

    private var jacks: some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 7) {
                ForEach(spec.inlets, id: \.self) { port in
                    JackView(node: node, port: port, dir: .input, pendingWire: $pendingWire)
                }
            }
            Spacer(minLength: 16)
            VStack(alignment: .trailing, spacing: 7) {
                ForEach(spec.outlets, id: \.self) { port in
                    JackView(node: node, port: port, dir: .output, pendingWire: $pendingWire)
                }
            }
        }
        .padding(.horizontal, 4).padding(.bottom, 6)
    }
}

// ── Jack ────────────────────────────────────────────────────────────────────
struct JackView: View {
    @EnvironmentObject var model: PatchModel
    let node: PatchNode
    let port: String
    let dir: JackDir
    @Binding var pendingWire: PendingWire?
    @State private var armed = false

    var body: some View {
        HStack(spacing: 5) {
            if dir == .input { dot; label } else { label; dot }
        }
    }

    private var label: some View {
        Text(port).font(Theme.monoSmall).foregroundStyle(Theme.muted)
    }

    private var dot: some View {
        Circle()
            .fill(armed ? Theme.jackHot : Theme.jack)
            .overlay(Circle().stroke(Color(hex: 0x555F72)))
            .frame(width: 12, height: 12)
            .background(
                GeometryReader { geo in
                    Color.clear.preference(
                        key: JackCentersKey.self,
                        value: [
                            JackID(node: node.id, dir: dir, port: port):
                                CGPoint(
                                    x: geo.frame(in: .named("canvas")).midX,
                                    y: geo.frame(in: .named("canvas")).midY)
                        ])
                })
            .gesture(dir == .output ? wireDrag : nil)
    }

    /// Drag from an outlet; drop on an inlet dot. The drop target is resolved
    /// by hit-testing the reported jack centers (nearest inlet within reach) —
    /// the SwiftUI analog of elementFromPoint.
    private var wireDrag: some Gesture {
        DragGesture(minimumDistance: 0, coordinateSpace: .named("canvas"))
            .onChanged { g in
                armed = true
                pendingWire = PendingWire(fromNode: node.id, cursor: g.location)
            }
            .onEnded { g in
                armed = false
                defer { pendingWire = nil }
                if let hit = model.nearestInlet(to: g.location) {
                    model.connect(from: node.id, to: hit.node, port: hit.port)
                }
            }
    }
}

extension PatchModel {
    /// Inlet hit-testing against the live jack geometry (12px dot + slop).
    /// Centers live on CanvasView state; mirrored here each preference pass.
    func nearestInlet(to point: CGPoint) -> (node: String, port: String)? {
        var best: (JackID, CGFloat)?
        for (id, c) in jackGeometry where id.dir == .input {
            let d = hypot(c.x - point.x, c.y - point.y)
            if d <= 14, d < (best?.1 ?? .infinity) { best = (id, d) }
        }
        guard let (id, _) = best else { return nil }
        return (id.node, id.port)
    }
}
