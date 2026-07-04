import SwiftUI

struct ToolbarView: View {
    @EnvironmentObject var model: PatchModel

    var body: some View {
        HStack(spacing: 10) {
            Text("tropical")
                .font(Theme.mono.bold())
                .foregroundStyle(Color(hex: 0xAEB8C8))
            palette
            Spacer()
            ClockView()
            transport
            Text(model.status)
                .font(Theme.mono)
                .foregroundStyle(model.statusIsError ? Theme.err : Theme.muted)
                .lineLimit(1)
                .frame(minWidth: 220, alignment: .trailing)
        }
        .padding(.horizontal, 12).padding(.vertical, 8)
        .background(Theme.panel)
        .overlay(Rectangle().frame(height: 1).foregroundStyle(Theme.edge), alignment: .bottom)
    }

    private var palette: some View {
        HStack(spacing: 6) {
            ForEach(NodeKind.allCases.filter { !$0.spec.fixed }, id: \.self) { kind in
                Button {
                    model.addNode(kind)
                } label: {
                    HStack(spacing: 5) {
                        Circle().fill(kind.spec.accent).frame(width: 8, height: 8)
                        Text(kind.spec.title).font(Theme.mono)
                    }
                    .padding(.horizontal, 9).padding(.vertical, 5)
                    .background(Theme.panel2)
                    .overlay(RoundedRectangle(cornerRadius: 6).stroke(Theme.edge))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                }
                .buttonStyle(.plain)
            }
        }
    }

    private var transport: some View {
        Button {
            Task { await model.toggleAudio() }
        } label: {
            Text(model.audioOn ? "■ stop audio" : "▶ start audio")
                .font(Theme.mono.bold())
                .foregroundStyle(Color(hex: 0xEAF6EE))
                .padding(.horizontal, 14).padding(.vertical, 6)
                .background(model.audioOn ? Theme.transportOn : Theme.transportOff)
                .overlay(RoundedRectangle(cornerRadius: 6)
                    .stroke(model.audioOn ? Color(hex: 0xB25151) : Color(hex: 0x3C8A5E)))
                .clipShape(RoundedRectangle(cornerRadius: 6))
        }
        .buttonStyle(.plain)
    }
}

/// Master clock (global time-warp scrub). One control rides
/// `master.velocity`, the base clock of EVERY generator, so a scrub
/// reverses / freezes / varispeeds the whole patch — voices, envelopes, and
/// the modal reverb tail — coherently. `set_param_velocity` re-bases the
/// τ-origin (`master.tau_base`) so the change is value-continuous
/// (click-free). Closed-form: nothing downstream holds history, so reverse
/// is just f(τ) read at receding τ.
/// 1 = play · 0 = freeze · −1 = reverse · |v| > 1 = varispeed.
struct ClockView: View {
    @EnvironmentObject var model: PatchModel

    var body: some View {
        HStack(spacing: 6) {
            Text("clock").font(Theme.monoSmall).foregroundStyle(Theme.muted)
            clockButton("⏴", on: model.velocity < 0, help: "reverse (−1×)") {
                model.setVelocity(model.velocity < 0 ? 1 : -1)   // toggle reverse
            }
            clockButton("⏸", on: model.velocity == 0, help: "freeze (0×)") {
                model.setVelocity(model.velocity == 0 ? 1 : 0)
            }
            clockButton("⏵", on: model.velocity == 1, help: "play (1×)") {
                model.setVelocity(1)
            }
            Slider(
                value: Binding(
                    get: { model.velocity },
                    set: { model.setVelocity(detented($0)) }),
                in: -2...2)
                .frame(width: 120)
                .tint(Theme.scrubAccent)
            Text(String(format: "%.2f×", model.velocity))
                .font(Theme.monoSmall)
                .foregroundStyle(
                    model.velocity < 0 ? Theme.scrubAccent :
                    model.velocity == 0 ? Theme.muted : Theme.text)
                .monospacedDigit()
                .frame(minWidth: 44, alignment: .trailing)
        }
        .padding(.horizontal, 8).padding(.vertical, 3)
        .background(Theme.panel2)
        .overlay(RoundedRectangle(cornerRadius: 6).stroke(Theme.edge))
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .help("global time-warp — scrubs every voice, envelope, and reverb tail coherently (closed-form, no state)")
    }

    /// Snap near the musically meaningful rates (mirrors app.js detents).
    private func detented(_ v: Double) -> Double {
        for d in [0.0, 1.0, -1.0, 2.0, -2.0] where abs(v - d) < 0.06 { return d }
        return v
    }

    private func clockButton(_ label: String, on: Bool, help: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(label)
                .font(Theme.mono)
                .foregroundStyle(on ? Color(hex: 0xEAF3F6) : Theme.text)
                .padding(.horizontal, 7).padding(.vertical, 2)
                .background(on ? Theme.clockOn : Theme.panel)
                .overlay(RoundedRectangle(cornerRadius: 5)
                    .stroke(on ? Color(hex: 0x3F86A8) : Theme.edge))
                .clipShape(RoundedRectangle(cornerRadius: 5))
        }
        .buttonStyle(.plain)
        .help(help)
    }
}
