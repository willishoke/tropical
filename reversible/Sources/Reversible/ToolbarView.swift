import SwiftUI

/// Native window-toolbar content: add-module menu, master clock, transport,
/// status. Custom chrome only where no native widget exists (the clock
/// cluster); everything else is stock.
struct PatchToolbar: ToolbarContent {
    @ObservedObject var model: PatchModel

    var body: some ToolbarContent {
        ToolbarItem(placement: .navigation) {
            Menu {
                ForEach(model.addableKinds, id: \.self) { kind in
                    Button(kind.spec.title) { model.addNode(kind) }
                }
            } label: {
                Label("Add Module", systemImage: "plus")
            }
            .help("add a module to the patch")
        }

        ToolbarItem(placement: .navigation) {
            HStack(spacing: 6) {
                Button {
                    model.exitModule()
                } label: {
                    Image(systemName: "chevron.left")
                }
                .disabled(model.isAtScene)
                .help("return to containing graph")
                Text(model.hierarchyTitle)
                    .font(Theme.monoSmall)
                    .lineLimit(1)
            }
        }

        // Rank layout: one row per topological rank, sources at the top,
        // dac at the bottom — the DAG's own order as geometry. `auto`
        // re-runs it on every topology edit.
        ToolbarItem {
            ControlGroup {
                Button {
                    model.arrangeByRank()
                } label: {
                    Label("Arrange", systemImage: "square.stack.3d.down.right")
                }
                .help("lay out modules in rows by topological rank")
                Toggle(isOn: $model.autoArrange) {
                    Label("Auto", systemImage: "arrow.trianglehead.2.clockwise")
                }
                .help("re-arrange on every topology edit")
            }
        }

        ToolbarItem { ClockView() }

        ToolbarItem {
            Button {
                Task { await model.toggleAudio() }
            } label: {
                Label(model.audioOn ? "Stop" : "Start",
                      systemImage: model.audioOn ? "stop.fill" : "play.fill")
            }
            .buttonStyle(.borderedProminent)
            .tint(model.audioOn ? Theme.transportOn : Theme.transportOff)
            .help(model.audioOn ? "stop audio" : "start audio")
        }

        ToolbarItem {
            Text(truthSummary)
                .font(Theme.monoSmall)
                .foregroundStyle(Theme.muted)
                .monospacedDigit()
                .help("authored revision · realized revision · latest acknowledged program/control generation")
        }

        ToolbarItem {
            Text(model.status)
                .font(Theme.monoSmall)
                .foregroundStyle(model.statusIsError ? Theme.err : Theme.muted)
                .lineLimit(1)
                .frame(minWidth: 180, alignment: .trailing)
                .help(model.status)
        }
    }

    private var truthSummary: String {
        let realizedRevision = model.realizedPatch.map {
            String($0.authoredRevision)
        } ?? "—"
        guard let generation = model.runningGeneration else {
            return "A\(model.authoredRevision) · R\(realizedRevision) · P—/C—"
        }
        return "A\(model.authoredRevision) · R\(realizedRevision) · P\(generation.programVersion)/C\(generation.controlVersion)"
    }
}

/// Master clock (global time-warp scrub). One control rides
/// `master.velocity`, the base clock of EVERY generator, so a scrub
/// reverses / freezes / varispeeds the whole patch — voices, envelopes, and
/// the modal reverb tail—coherently. The engine's `set_param` velocity
/// discipline re-bases the
/// τ-origin (`master.tau_base`) so the change is value-continuous
/// (click-free). Closed-form: nothing downstream holds history, so reverse
/// is just f(τ) read at receding τ.
/// 1 = play · 0 = freeze · −1 = reverse · |v| > 1 = varispeed.
struct ClockView: View {
    @EnvironmentObject var model: PatchModel

    var body: some View {
        HStack(spacing: 8) {
            Picker("clock", selection: transportBinding) {
                Image(systemName: "backward.fill").tag(Transport.reverse)
                Image(systemName: "pause.fill").tag(Transport.freeze)
                Image(systemName: "play.fill").tag(Transport.play)
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .fixedSize()
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
        .help("global time-warp — scrubs every voice, envelope, and reverb tail coherently (closed-form, no state)")
    }

    private enum Transport: Hashable { case reverse, freeze, play, free }

    /// The segmented control reflects the three detents; a slider value off
    /// any detent selects nothing (`.free`).
    private var transportBinding: Binding<Transport> {
        Binding(
            get: {
                switch model.velocity {
                case -1: .reverse
                case 0: .freeze
                case 1: .play
                default: .free
                }
            },
            set: { t in
                switch t {
                case .reverse: model.setVelocity(-1)
                case .freeze: model.setVelocity(0)
                case .play: model.setVelocity(1)
                case .free: break
                }
            })
    }

    /// Snap near the musically meaningful rates (mirrors app.js detents).
    private func detented(_ v: Double) -> Double {
        for d in [0.0, 1.0, -1.0, 2.0, -2.0] where abs(v - d) < 0.06 { return d }
        return v
    }
}
