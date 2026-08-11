import SwiftUI

/// Knob: 38px dial, pointer sweeping −135°…+135° (styles.css .knob).
/// A vertical drag (180px = full sweep) edits authored scalar truth. The
/// realized handshake decides whether that scalar is live (`set_param`) or
/// structural (relower); the client never guesses from a hard-coded knob list.
/// For live values the loaded plan owns the write discipline: raw, glide,
/// phase-anchor, or velocity rebase.
struct KnobView: View {
    @EnvironmentObject var model: PatchModel
    let node: PatchNode
    let knob: KnobSpec
    @State private var dragStartNorm: Double?

    private var value: Double { node.values[knob.name] ?? knob.def }

    private var truthBadge: (label: String, color: Color, help: String) {
        switch model.parameterStatus(for: node, knob: knob) {
        case .live(let discipline):
            ("live", Theme.truthActive, "live · \(discipline.rawValue)")
        case .structural:
            ("struct", Theme.truthExcluded, "structural · relowers on edit")
        case .inactive:
            ("inactive", Theme.truthPending, "inactive in the running patch")
        }
    }

    var body: some View {
        VStack(spacing: 2) {
            Text(truthBadge.label)
                .font(.system(size: 7, weight: .semibold, design: .monospaced))
                .foregroundStyle(truthBadge.color)
                .padding(.horizontal, 3).padding(.vertical, 1)
                .background(
                    truthBadge.color.opacity(0.16),
                    in: Capsule()
                )
                .fixedSize()
                .help(truthBadge.help)
            dial
                .contentShape(Circle())
                .gesture(
                    DragGesture(minimumDistance: 0)
                        .onChanged { g in
                            let t0 = dragStartNorm ?? knob.toNorm(value)
                            dragStartNorm = t0
                            let v = knob.fromNorm(t0 - g.translation.height / 180)
                            model.setKnob(node, knob, v)
                        }
                        .onEnded { _ in
                            dragStartNorm = nil
                            model.setKnob(node, knob, value)
                        }
                )
            Text(knob.name)
                .font(Theme.monoTiny)
                .foregroundStyle(Theme.muted)
                .lineLimit(1)
            Text(knob.format(value))
                .font(Theme.monoTiny)
                .foregroundStyle(Theme.text)
                .lineLimit(1)
                .minimumScaleFactor(0.8)
        }
        .frame(width: 54)
    }

    private var dial: some View {
        let angle = -135 + knob.toNorm(value) * 270
        return ZStack(alignment: .top) {
            Circle()
                .fill(
                    RadialGradient(
                        colors: [Color(hex: 0x39404E), Theme.panel],
                        center: UnitPoint(x: 0.5, y: 0.38),
                        startRadius: 0, endRadius: 27))
                .overlay(Circle().stroke(Theme.edge))
            // Pointer sits 5px in from the top edge; the ROTATED view is the
            // full 38px square, so the pivot is the dial center (CSS
            // transform-origin 50% 14px ≡ 5 + 14 = 19 = center).
            RoundedRectangle(cornerRadius: 2)
                .fill(Theme.jackHot)
                .frame(width: 2, height: 13)
                .offset(y: 5)
                .frame(width: 38, height: 38, alignment: .top)
                .rotationEffect(.degrees(angle), anchor: .center)
        }
        .frame(width: 38, height: 38)
    }
}
