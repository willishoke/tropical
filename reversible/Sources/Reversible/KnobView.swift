import SwiftUI

/// Knob: 38px dial, pointer sweeping −135°…+135° (styles.css .knob).
/// A vertical drag (180px = full sweep) edits the authored value. The realized
/// handshake decides whether that edit can use `set_param` immediately or
/// must be committed as a structural edit and relowered. For live values, the
/// loaded plan—not the client—owns the write discipline.
struct KnobView: View {
    @EnvironmentObject var model: PatchModel
    let node: PatchNode
    let knob: KnobSpec
    @State private var dragStartNorm: Double?

    private var value: Double { node.values[knob.name] ?? knob.def }
    private var truth: TruthBadgePresentation { model.parameterTruth(for: node, knob: knob) }
    private var canWriteLive: Bool { model.canWriteLive(node, knob: knob) }

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
                            if canWriteLive { model.sendKnob(node, knob, v) }
                        }
                        .onEnded { _ in
                            dragStartNorm = nil
                            let finalValue = model.nodes[node.id]?.values[knob.name] ?? value
                            if canWriteLive {
                                model.sendKnob(node, knob, finalValue)
                            } else if !node.kind.isMonitor {
                                model.commitStructuralKnobEdit(nodeID: node.id)
                            }
                        }
                )
            Text(knob.name).font(Theme.monoTiny).foregroundStyle(Theme.muted)
            Text(knob.format(value)).font(Theme.monoTiny).foregroundStyle(Theme.text)
            Label(truth.label, systemImage: truth.symbol)
                .truthPill(truth.color)
                .labelStyle(.titleAndIcon)
                .help(truth.detail)
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
