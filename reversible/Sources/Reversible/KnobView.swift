import SwiftUI

/// Knob: 38px dial, pointer sweeping −135°…+135° (styles.css .knob).
/// Every continuous knob is a live param slot `<id>.<knob>` — a vertical
/// drag (180px = full sweep) drives the running kernel with no relower
/// (only topology edits recompile). A glided knob eases via
/// set_param_glide; freq-anchored knobs go phase-continuous via
/// set_param_freq; the rest do a raw set_param write.
struct KnobView: View {
    @EnvironmentObject var model: PatchModel
    let node: PatchNode
    let knob: KnobSpec
    @State private var dragStartNorm: Double?

    private var value: Double { node.values[knob.name] ?? knob.def }

    var body: some View {
        VStack(spacing: 2) {
            dial
                .contentShape(Circle())
                .gesture(
                    DragGesture(minimumDistance: 0)
                        .onChanged { g in
                            let t0 = dragStartNorm ?? knob.toNorm(value)
                            dragStartNorm = t0
                            let v = knob.fromNorm(t0 - g.translation.height / 180)
                            model.nodes[node.id]?.values[knob.name] = v
                            model.sendKnob(node, knob, v)
                        }
                        .onEnded { _ in
                            dragStartNorm = nil
                            model.sendKnob(node, knob, value)
                        }
                )
            Text(knob.name).font(Theme.monoTiny).foregroundStyle(Theme.muted)
            Text(knob.format(value)).font(Theme.monoTiny).foregroundStyle(Theme.text)
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
