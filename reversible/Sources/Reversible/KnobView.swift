import SwiftUI

/// Knob face: 38px dial, pointer sweeping −135°…+135° (styles.css .knob).
/// Interaction (vertical drag → live param drive) lands in the next commit;
/// this renders value + angle so the canvas is visually complete.
struct KnobView: View {
    @EnvironmentObject var model: PatchModel
    let node: PatchNode
    let knob: KnobSpec

    private var value: Double { node.values[knob.name] ?? knob.def }

    var body: some View {
        VStack(spacing: 2) {
            dial
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
            RoundedRectangle(cornerRadius: 2)
                .fill(Theme.jackHot)
                .frame(width: 2, height: 13)
                .offset(y: 5)
                .rotationEffect(.degrees(angle), anchor: .center)
        }
        .frame(width: 38, height: 38)
    }
}
