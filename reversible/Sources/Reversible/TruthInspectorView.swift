import SwiftUI

/// Presentation-only projections of authored and engine-acknowledged truth.
/// These helpers never infer semantic facts from node names or descriptors.
struct TruthBadgePresentation {
    let label: String
    let detail: String
    let symbol: String
    let color: Color
}

extension PatchModel {
    var runningGenerationText: String {
        guard let generation = realizedPatch?.generation else { return "run —" }
        return "run P\(generation.programVersion)/C\(generation.controlVersion)"
    }

    var compileTruth: TruthBadgePresentation {
        switch compileState {
        case .idle:
            return .init(
                label: "idle", detail: "No compile is active.",
                symbol: "circle", color: Theme.muted)
        case .compiling(let revision):
            return .init(
                label: "compiling A\(revision)",
                detail: "Authored revision \(revision) is compiling; \(runningGenerationText) remains audible.",
                symbol: "arrow.triangle.2.circlepath", color: .yellow)
        case .running(let revision, let program, let control):
            let generation = program.map(String.init) ?? "—"
            let controls = control.map(String.init) ?? "—"
            return .init(
                label: "running A\(revision)",
                detail: "Authored revision \(revision) is realized as program \(generation), control \(controls).",
                symbol: "checkmark.circle.fill", color: .green)
        case .failed(let revision, let message):
            return .init(
                label: "failed A\(revision)",
                detail: "Authored revision \(revision) failed: \(message). \(runningGenerationText) is unchanged.",
                symbol: "xmark.octagon.fill", color: Theme.err)
        case .superseded(let compiled, let current):
            return .init(
                label: "superseded A\(compiled)→A\(current)",
                detail: "Compile for authored revision \(compiled) was superseded by revision \(current); \(runningGenerationText) is unchanged.",
                symbol: "arrow.uturn.forward.circle.fill", color: .orange)
        }
    }

    func nodeTruth(for node: PatchNode) -> TruthBadgePresentation {
        if node.kind.isMonitor {
            return .init(
                label: "surface monitor",
                detail: "Client-owned, read-only monitor; it is never an engine graph node.",
                symbol: "display", color: Theme.scrubAccent)
        }
        if node.kind.spec.unavailable {
            return .init(
                label: "unavailable",
                detail: "Authored kind '\(node.kind.rawValue)' is not served by the running vocabulary. It remains visible and serializable.",
                symbol: "questionmark.diamond.fill", color: .orange)
        }
        guard let authoredKind = node.kind.engineID,
              let realizedNode = realizedPatch?.patch.node(id: node.id),
              realizedNode.kind == authoredKind
        else {
            return .init(
                label: "authored only",
                detail: "This authored node is not present in \(runningGenerationText).",
                symbol: "pencil.circle.fill", color: .yellow)
        }
        switch realizedNode.status {
        case .active:
            return .init(
                label: "active",
                detail: "Active in \(runningGenerationText).",
                symbol: "checkmark.circle.fill", color: .green)
        case .excluded:
            return .init(
                label: "excluded",
                detail: "Acknowledged but excluded from \(runningGenerationText).",
                symbol: "minus.circle.fill", color: Theme.err)
        }
    }

    func inletTruth(for node: PatchNode, port: String) -> TruthBadgePresentation {
        if node.kind.isMonitor {
            return .init(
                label: "surface",
                detail: "Client-owned observation input; it does not alter the audio graph.",
                symbol: "eye.fill", color: Theme.scrubAccent)
        }
        guard node.kind.descriptor?.port(named: port)?.isInlet == true else {
            return .init(
                label: "unknown",
                detail: "Authored port '\(port)' is not a served inlet. Its authored sources are preserved.",
                symbol: "questionmark.diamond.fill", color: .orange)
        }
        guard let authoredKind = node.kind.engineID,
              realizedPatch?.patch.node(id: node.id)?.kind == authoredKind,
              let inlet = realizedPatch?.patch.inlet(nodeID: node.id, port: port)
        else {
            return .init(
                label: "authored",
                detail: "No realized inlet fact exists for \(runningGenerationText).",
                symbol: "pencil", color: .yellow)
        }
        switch inlet.state {
        case .normalled:
            return .init(
                label: "normalled",
                detail: "The running inlet uses its engine-owned normal in \(runningGenerationText).",
                symbol: "arrow.down.to.line.compact", color: .blue)
        case .wired(let sources):
            return .init(
                label: "wired",
                detail: "Running sources: \(sources.joined(separator: ", ")).",
                symbol: "link", color: .green)
        }
    }

    func parameterTruth(for node: PatchNode, knob: KnobSpec) -> TruthBadgePresentation {
        if node.kind.isMonitor {
            return .init(
                label: "surface",
                detail: "Client-owned view state; no engine parameter write is sent.",
                symbol: "display", color: Theme.scrubAccent)
        }
        guard let authoredKind = node.kind.engineID,
              realizedPatch?.patch.node(id: node.id)?.kind == authoredKind,
              let realizedPatch
        else {
            return .init(
                label: "pending",
                detail: "No realized parameter fact exists for \(runningGenerationText).",
                symbol: "clock.fill", color: .yellow)
        }
        switch realizedPatch.parameterStatus(nodeID: node.id, port: knob.name) {
        case .live(let discipline):
            return .init(
                label: "live",
                detail: "Live in \(runningGenerationText); engine discipline '\(discipline.rawValue)'.",
                symbol: "bolt.fill", color: .green)
        case .structural:
            return .init(
                label: "structural",
                detail: "Not a live parameter in \(runningGenerationText); changing it requires a compile.",
                symbol: "hammer.fill", color: .orange)
        case .inactive:
            return .init(
                label: "inactive",
                detail: "The owning node is inactive in \(runningGenerationText).",
                symbol: "pause.circle.fill", color: Theme.err)
        }
    }

    func canWriteLive(_ node: PatchNode, knob: KnobSpec) -> Bool {
        guard let realizedPatch,
              realizedPatch.authoredRevision == authoredRevision,
              case .live = realizedPatch.parameterStatus(nodeID: node.id, port: knob.name)
        else { return false }
        return true
    }
}

struct TruthInspectorView: View {
    @ObservedObject var model: PatchModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header
            Divider()
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 12) {
                    compileSection
                    nodeSection
                    blockerSection
                    migrationSection
                }
            }
        }
        .padding(14)
        .frame(width: 540, height: 560)
        .background(Theme.panel)
    }

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: model.compileTruth.symbol)
                .foregroundStyle(model.compileTruth.color)
            VStack(alignment: .leading, spacing: 2) {
                Text("Patch truth").font(Theme.mono.bold())
                Text("authored A\(model.authoredRevision) · \(model.runningGenerationText)")
                    .font(Theme.monoSmall)
                    .foregroundStyle(Theme.muted)
            }
            Spacer()
            Text(model.compileTruth.label)
                .truthPill(model.compileTruth.color)
        }
    }

    private var compileSection: some View {
        TruthSection(title: "Compile") {
            Text(model.compileTruth.detail)
                .font(Theme.monoSmall)
                .foregroundStyle(Theme.text)
                .textSelection(.enabled)
            if let fingerprint = model.realizedPatch?.vocabularyFingerprint {
                Text("vocabulary \(fingerprint)")
                    .font(Theme.monoTiny)
                    .foregroundStyle(Theme.muted)
                    .textSelection(.enabled)
            }
        }
    }

    private var nodeSection: some View {
        TruthSection(title: "Authored nodes") {
            if model.order.isEmpty {
                Text("No authored nodes").font(Theme.monoSmall).foregroundStyle(Theme.muted)
            } else {
                ForEach(model.order, id: \.self) { id in
                    if let node = model.nodes[id] {
                        let truth = model.nodeTruth(for: node)
                        HStack(spacing: 7) {
                            Image(systemName: truth.symbol)
                                .foregroundStyle(truth.color)
                                .frame(width: 14)
                            Text(node.id).font(Theme.monoSmall)
                            Text(node.kind.rawValue)
                                .font(Theme.monoTiny)
                                .foregroundStyle(Theme.muted)
                            Spacer()
                            Text(truth.label).truthPill(truth.color)
                        }
                        .help(truth.detail)
                    }
                }
            }
        }
    }

    private var blockerSection: some View {
        TruthSection(title: "Document blockers (\(model.documentBlockers.count))") {
            if model.documentBlockers.isEmpty {
                Label("No document compile blockers", systemImage: "checkmark.circle.fill")
                    .font(Theme.monoSmall)
                    .foregroundStyle(.green)
            } else {
                ForEach(Array(model.documentBlockers.enumerated()), id: \.offset) { _, blocker in
                    VStack(alignment: .leading, spacing: 2) {
                        Text(blocker.code.rawValue).truthPill(Theme.err)
                        Text(blocker.path).font(Theme.monoTiny).foregroundStyle(Theme.muted)
                        Text(blocker.detail).font(Theme.monoSmall).textSelection(.enabled)
                    }
                }
            }
        }
    }

    @ViewBuilder
    private var migrationSection: some View {
        if let report = model.migrationReport {
            TruthSection(title: "V1 migration facts (\(report.facts.count))") {
                if model.requiresExplicitV2Save {
                    Label("Explicit v2 Save As required", systemImage: "square.and.arrow.down")
                        .font(Theme.monoSmall)
                        .foregroundStyle(.orange)
                }
                ForEach(Array(report.facts.enumerated()), id: \.offset) { index, fact in
                    VStack(alignment: .leading, spacing: 2) {
                        Text("\(index + 1). \(fact.code.rawValue)")
                            .font(Theme.monoSmall)
                            .foregroundStyle(Theme.text)
                        Text(fact.path).font(Theme.monoTiny).foregroundStyle(Theme.muted)
                        Text(fact.detail).font(Theme.monoTiny).textSelection(.enabled)
                    }
                }
            }
        }
    }
}

private struct TruthSection<Content: View>: View {
    let title: String
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            Text(title).font(Theme.monoSmall.bold()).foregroundStyle(Theme.muted)
            content
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.panel2, in: RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(Theme.edge))
    }
}

extension View {
    func truthPill(_ color: Color) -> some View {
        self
            .font(Theme.monoTiny.bold())
            .foregroundStyle(color)
            .padding(.horizontal, 5)
            .padding(.vertical, 2)
            .background(color.opacity(0.13), in: Capsule())
            .overlay(Capsule().stroke(color.opacity(0.45)))
    }
}
