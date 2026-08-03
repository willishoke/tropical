import SwiftUI

/// The Scope body: a CRT well fed by the model's poll loop, plus a footer of
/// channel inlets and the window knob. Traces wear their SOURCE's identity
/// color — the same hue that marks that node's outlet everywhere else — so a
/// multi-trace scope reads with no legend.
struct ScopeBody: View {
    @EnvironmentObject var model: PatchModel
    let node: PatchNode

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ScopeDisplay(node: node)
                .padding(.horizontal, 8).padding(.top, 8)
            footer
        }
    }

    private var footer: some View {
        HStack(alignment: .center, spacing: 12) {
            let inlets = node.kind.spec.inlets
            VStack(alignment: .leading, spacing: 4) {
                InletView(node: node, port: inlets[0])
                InletView(node: node, port: inlets[1])
            }
            VStack(alignment: .leading, spacing: 4) {
                InletView(node: node, port: inlets[2])
                InletView(node: node, port: inlets[3])
            }
            Spacer(minLength: 8)
            if let knob = node.kind.spec.knobs.first {
                KnobView(node: node, knob: knob)
            }
        }
        .padding(.horizontal, 9).padding(.vertical, 6)
    }
}

struct ScopeDisplay: View {
    @EnvironmentObject var model: PatchModel
    let node: PatchNode

    /// (source color, samples) per connected channel, port order.
    private var traces: [(Color, [Double])] {
        node.kind.spec.inlets.compactMap { port in
            guard let src = node.inputs[port]?.first,
                  let s = model.scopeTraces["\(node.id).\(port)"], s.count > 1
            else { return nil }
            return (model.nodes[src]?.color ?? Theme.text, s)
        }
    }

    var body: some View {
        Canvas { ctx, size in
            let mid = size.height / 2
            var axis = Path()
            axis.move(to: CGPoint(x: 0, y: mid))
            axis.addLine(to: CGPoint(x: size.width, y: mid))
            ctx.stroke(axis, with: .color(Theme.edge.opacity(0.7)), lineWidth: 1)

            let traces = traces
            // One shared scale (unity floor): relative levels stay honest,
            // and a hot signal autoscales instead of clipping the well.
            let peak = max(1.0, traces.flatMap(\.1).reduce(0) { max($0, abs($1)) })
            let half = mid - 3
            for (color, s) in traces {
                ctx.stroke(Self.tracePath(s, size: size, mid: mid, half: half, peak: peak),
                           with: .color(color), lineWidth: 1.3)
            }
        }
        .background(Color(hex: 0x0C0F14), in: RoundedRectangle(cornerRadius: 6))
        .overlay(RoundedRectangle(cornerRadius: 6).stroke(Theme.edge))
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    /// A wide window overruns the pixel budget, so bin to per-column min/max
    /// strips (the classic scope raster — envelope survives, aliasing doesn't);
    /// a short window draws the plain polyline.
    static func tracePath(_ s: [Double], size: CGSize, mid: Double, half: Double, peak: Double) -> Path {
        var p = Path()
        let n = s.count
        let cols = max(1, Int(size.width))
        let y: (Double) -> Double = { mid - ($0 / peak) * half }
        if n <= cols * 2 {
            for (i, v) in s.enumerated() {
                let pt = CGPoint(x: size.width * Double(i) / Double(n - 1), y: y(v))
                if i == 0 { p.move(to: pt) } else { p.addLine(to: pt) }
            }
        } else {
            for c in 0..<cols {
                let a = c * n / cols, b = max(a + 1, (c + 1) * n / cols)
                var lo = s[a], hi = s[a]
                for v in s[a..<b] { lo = min(lo, v); hi = max(hi, v) }
                let x = (Double(c) + 0.5) * size.width / Double(cols)
                // A flat column still needs ink: pad to a minimum strip.
                p.move(to: CGPoint(x: x, y: y(hi)))
                p.addLine(to: CGPoint(x: x, y: max(y(lo), y(hi) + 0.8)))
            }
        }
        return p
    }
}
