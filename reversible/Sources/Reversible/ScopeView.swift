import SwiftUI

private struct ModalScopeBindingKey: EnvironmentKey {
    static let defaultValue: ModalScopeBinding? = nil
}

extension EnvironmentValues {
    /// The release scene injects its active chord's five immutable tap
    /// bindings here. Keeping scene selection outside the sampler avoids
    /// hard-coding product node ids into the reusable scope implementation.
    var modalScopeBinding: ModalScopeBinding? {
        get { self[ModalScopeBindingKey.self] }
        set { self[ModalScopeBindingKey.self] = newValue }
    }
}

extension View {
    func modalScopeBinding(_ binding: ModalScopeBinding?) -> some View {
        environment(\.modalScopeBinding, binding)
    }
}

/// The Scope body: one CRT well for either the qualified five-mode binding or
/// the legacy patcher channels, plus the monitor-only footer controls.
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
    @Environment(\.modalScopeBinding) private var modalBinding
    let node: PatchNode
    @StateObject private var sampler = ModalScopeSampler()
    @StateObject private var displayLink = ModalScopeDisplayLink()

    private static let modeColors: [Color] = [
        Color(hex: 0x72B6A1), Color(hex: 0xD5C66F),
        Color(hex: 0xD49A6A), Color(hex: 0x8FA7D1),
        Color(hex: 0xBD8FB4),
    ]

    /// Legacy patcher channels remain available until a five-mode release
    /// binding is present. The qualified path below never reads these values.
    private var legacyTraces: [(Color, [Double])] {
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
            Self.drawGraticule(in: &ctx, size: size, mid: mid)

            if let frame = sampler.latestFrame, modalBinding != nil {
                // One immutable response, five independent local locks, one
                // fixed paired-envelope volts/div. No trace derives scale from
                // the current frame, so audible decay remains visible.
                let half = max(1, mid - 5)
                for (index, mode) in frame.visibleModes.enumerated() {
                    let color = Self.modeColors[index % Self.modeColors.count]
                    ctx.stroke(
                        Self.phaseTracePath(
                            mode.frame.displayedValues,
                            size: size,
                            mid: mid,
                            half: half,
                            displayScale: frame.displayScale
                        ),
                        with: .color(color),
                        lineWidth: 1.25
                    )
                }
            } else if modalBinding == nil {
                let traces = legacyTraces
                let peak = max(
                    1.0,
                    traces.flatMap(\.1).reduce(0) { max($0, abs($1)) }
                )
                let half = mid - 3
                for (color, samples) in traces {
                    ctx.stroke(
                        Self.tracePath(
                            samples, size: size, mid: mid,
                            half: half, peak: peak
                        ),
                        with: .color(color), lineWidth: 1.3
                    )
                }
            }
        }
        .background(Color(hex: 0x0C0F14), in: RoundedRectangle(cornerRadius: 6))
        .overlay(RoundedRectangle(cornerRadius: 6).stroke(Theme.edge))
        .overlay(alignment: .bottomLeading) {
            if let frame = sampler.latestFrame, modalBinding != nil {
                Text(Self.metadata(frame))
                    .font(Theme.monoTiny)
                    .foregroundStyle(Theme.muted)
                    .lineLimit(1)
                    .padding(.horizontal, 7).padding(.bottom, 5)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .onAppear { restartModalSampling() }
        .onChange(of: modalBinding) { _, _ in restartModalSampling() }
        .onDisappear { displayLink.stop() }
    }

    private func restartModalSampling() {
        displayLink.stop()
        guard let modalBinding else {
            sampler.clear()
            return
        }
        sampler.configure(rpc: model.engine)
        sampler.request(modalBinding)
        displayLink.start { [weak sampler] in
            sampler?.request(modalBinding)
        }
    }

    private static func drawGraticule(
        in context: inout GraphicsContext,
        size: CGSize,
        mid: Double
    ) {
        var axes = Path()
        axes.move(to: CGPoint(x: 0, y: mid))
        axes.addLine(to: CGPoint(x: size.width, y: mid))
        axes.move(to: CGPoint(x: size.width / 2, y: 0))
        axes.addLine(to: CGPoint(x: size.width / 2, y: size.height))
        context.stroke(
            axes, with: .color(Theme.edge.opacity(0.72)), lineWidth: 1
        )

        var ticks = Path()
        for division in 1..<8 {
            let x = size.width * Double(division) / 8
            ticks.move(to: CGPoint(x: x, y: mid - 2))
            ticks.addLine(to: CGPoint(x: x, y: mid + 2))
        }
        context.stroke(
            ticks, with: .color(Theme.edge.opacity(0.48)), lineWidth: 1
        )
    }

    /// The accepted phase view always draws all 896 interpolated samples on a
    /// linear x axis. The centered graticule is therefore the local positive
    /// crossing selected independently for this mode.
    static func phaseTracePath(
        _ samples: [Double],
        size: CGSize,
        mid: Double,
        half: Double,
        displayScale: Double
    ) -> Path {
        var path = Path()
        guard samples.count > 1, displayScale > 0 else { return path }
        for (index, value) in samples.enumerated() {
            let x = size.width * Double(index) / Double(samples.count - 1)
            let normalized = ModalScopeMath.normalizedAmplitude(
                value, fullScale: displayScale
            )
            let point = CGPoint(x: x, y: mid - normalized * half)
            if index == 0 { path.move(to: point) }
            else { path.addLine(to: point) }
        }
        return path
    }

    static func metadata(_ frame: ModalScopeOverlayFrame) -> String {
        let visible = frame.visibleModes
        let cycles = visible.map(\.frame.cycles)
        let cycleText: String
        if let low = cycles.min(), let high = cycles.max() {
            cycleText = String(format: "%.1f–%.1f cyc", low, high)
        } else {
            cycleText = "no signal"
        }
        let milliseconds = ModalScopeProfile.qualified.linearSpanSeconds * 1_000
        return String(
            format: "five locks · %.1f ms / 2× time · %@ · gen %llu/%llu",
            milliseconds,
            cycleText,
            frame.generation.programVersion,
            frame.generation.controlVersion
        )
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
