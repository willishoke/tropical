import AppKit
import SwiftUI
import UniformTypeIdentifiers

extension PatchDocumentV2 {
    static let fileExtension = "rvpatch"

    static var contentType: UTType {
        UTType(filenameExtension: fileExtension) ?? .json
    }
}

enum PatchDocumentIOError: LocalizedError {
    case vocabularyUnavailable
    case missingVersion

    var errorDescription: String? {
        switch self {
        case .vocabularyUnavailable:
            return "the engine vocabulary is not available yet"
        case .missingVersion:
            return "patch document has no integer version"
        }
    }
}

extension PatchModel {
    /// Materialize the authored surface as v2. Existing typed and unknown v2
    /// fields are used as templates so opening and saving does not collapse a
    /// richer document to what the current UI happens to render.
    func makeDocument() throws -> PatchDocumentV2 {
        guard let vocabulary else { throw PatchDocumentIOError.vocabularyUnavailable }
        let template = documentTemplate
        let modelCannotRepresentExactly = documentBlockers.contains {
            $0.code == .duplicateID || $0.code == .invalidOrder || $0.code == .invalidPosition
        }
        // The live dictionaries/order cannot faithfully represent these
        // malformed shapes. Keep the exact decoded document serializable and
        // blocked rather than silently repairing it during a save or push.
        if modelCannotRepresentExactly, let template { return template }
        let oldNodes = Dictionary(
            (template?.nodes ?? []).map { ($0.id, $0) },
            uniquingKeysWith: { first, _ in first })
        let oldMonitors = Dictionary(
            (template?.monitors ?? []).map { ($0.id, $0) },
            uniquingKeysWith: { first, _ in first })

        let engineNodes: [PatchNodeV2] = order.compactMap { id in
            guard let node = nodes[id], let kind = node.kind.engineID else { return nil }
            return PatchNodeV2(
                id: id,
                kind: kind,
                structuralParams: node.structuralParams,
                inputs: node.inputs,
                unknownFields: oldNodes[id]?.unknownFields ?? [:])
        }
        let monitors: [PatchClientMonitorV2] = order.compactMap { id in
            guard let node = nodes[id], let kind = node.kind.monitorID else { return nil }
            return PatchClientMonitorV2(
                id: id,
                kind: kind,
                state: node.structuralParams,
                inputs: node.inputs,
                unknownFields: oldMonitors[id]?.unknownFields ?? [:])
        }

        var positions: [String: PatchPositionV2] = [:]
        for id in order {
            guard let node = nodes[id] else { continue }
            positions[id] = PatchPositionV2(
                x: node.position.x,
                y: node.position.y,
                hue: node.hue,
                unknownFields: template?.presentation.positions[id]?.unknownFields ?? [:])
        }

        let vocabularyReference = template?.vocabulary
            ?? PatchVocabularyReferenceV2(vocabulary: vocabulary)
        return PatchDocumentV2(
            vocabulary: vocabularyReference,
            sceneMetadata: template?.sceneMetadata ?? [:],
            nodes: engineNodes,
            monitors: monitors,
            output: outputNodeID,
            transport: PatchTransportV2(
                velocity: velocity,
                position: template?.transport.position,
                unknownFields: template?.transport.unknownFields ?? [:]),
            presentation: PatchPresentationV2(
                order: order,
                positions: positions,
                panX: pan.width,
                panY: pan.height,
                autoArrange: autoArrange,
                unknownFields: template?.presentation.unknownFields ?? [:]),
            unknownFields: template?.unknownFields ?? [:])
    }

    /// Adopt every authored item before validation. Unknown kinds, ports and
    /// fields remain visible/serializable; blockers prevent only realization.
    func apply(_ document: PatchDocumentV2) async {
        let validation = document.validate(against: vocabulary)
        var fresh: [String: PatchNode] = [:]
        for node in document.nodes {
            let descriptor = vocabulary?.descriptor(for: node.kind)
            let kind = NodeKind.engine(
                id: node.kind,
                descriptor: descriptor,
                theme: themes.override(for: node.kind),
                authoredInputNames: node.inputs.keys)
            let position = document.presentation.positions[node.id]
            fresh[node.id] = PatchNode(
                id: node.id,
                kind: kind,
                position: CGPoint(x: position?.x ?? 0, y: position?.y ?? 0),
                structuralParams: node.structuralParams,
                inputs: node.inputs,
                hue: position?.hue ?? 0)
        }
        for monitor in document.monitors {
            let kind = NodeKind.monitor(
                id: monitor.kind, authoredInputNames: monitor.inputs.keys)
            let position = document.presentation.positions[monitor.id]
            fresh[monitor.id] = PatchNode(
                id: monitor.id,
                kind: kind,
                position: CGPoint(x: position?.x ?? 0, y: position?.y ?? 0),
                structuralParams: monitor.state,
                inputs: monitor.inputs,
                hue: position?.hue ?? 0)
        }

        nodes = fresh
        // Keep every authored item visible while retaining the authored order
        // blocker in the template/report when the source order was malformed.
        var seen = Set<String>()
        order = document.presentation.order.filter {
            fresh[$0] != nil && seen.insert($0).inserted
        } + (document.nodes.map(\.id) + document.monitors.map(\.id)).filter {
            !seen.contains($0)
        }
        pan = CGSize(
            width: document.presentation.panX,
            height: document.presentation.panY)
        autoArrange = document.presentation.autoArrange
        velocity = document.transport.velocity
        outputNodeID = document.output
        documentTemplate = document
        documentBlockers = validation.blockers
        authoredRevision += 1
        adoptCounter(from: fresh.keys)

        if validation.canCompile {
            await pushGraph()
            setVelocity(document.transport.velocity)
        } else {
            compileState = .failed(
                authoredRevision: authoredRevision,
                message: validation.blockers.first?.detail ?? "document is not compilable")
            setStatus(
                "loaded authored patch · \(validation.blockers.count) compile blocker\(validation.blockers.count == 1 ? "" : "s")",
                isError: true)
        }
    }

    func newPatch() async {
        nodes = [:]
        order = []
        pan = .zero
        resetCounter()
        documentTemplate = nil
        documentBlockers = []
        migrationReport = nil
        requiresExplicitV2Save = false
        outputNodeID = nil
        seedBootPatch()
        documentURL = nil
        savedDocumentData = nil
        await pushGraph()
        setVelocity(velocity)
    }

    func write(to url: URL) throws {
        let document = try makeDocument()
        try document.encoded().write(to: url, options: .atomic)
        documentTemplate = document
        requiresExplicitV2Save = false
        documentURL = url
        savedDocumentData = try document.encoded(prettyPrinted: false)
        noteRecentDocument(url)
        setStatus("saved v2 · \(url.lastPathComponent)", isError: false)
    }

    func read(from url: URL) async throws {
        guard let vocabulary else { throw PatchDocumentIOError.vocabularyUnavailable }
        let data = try Data(contentsOf: url)
        let raw = try JSONDecoder().decode(JSONValue.self, from: data)
        guard let versionNumber = raw["version"]?.doubleValue,
              versionNumber.rounded(.towardZero) == versionNumber
        else { throw PatchDocumentIOError.missingVersion }

        switch Int(versionNumber) {
        case PatchDocumentV2.currentVersion:
            let document = try PatchDocumentV2.decode(from: data)
            migrationReport = nil
            requiresExplicitV2Save = false
            await apply(document)
            documentURL = url
            savedDocumentData = try document.encoded(prettyPrinted: false)
            noteRecentDocument(url)
            if documentBlockers.isEmpty {
                setStatus("loaded v2 · \(url.lastPathComponent)", isError: false)
            }
        case 1:
            let result = try PatchDocumentV1Migrator.migrate(
                data: data, vocabulary: vocabulary)
            migrationReport = result.report
            requiresExplicitV2Save = result.requiresExplicitV2Save
            await apply(result.document)
            // Migration is memory-only and must never write through the v1 URL.
            documentURL = nil
            savedDocumentData = nil
            setStatus(
                "migrated v1 in memory · Save As required",
                isError: !result.report.canCompile)
        default:
            throw PatchDocumentV2Error.unsupportedVersion(Int(versionNumber))
        }
    }

    func save() {
        guard !requiresExplicitV2Save, let url = documentURL else { return saveAs() }
        do { try write(to: url) } catch {
            setStatus("save: \(error.localizedDescription)", isError: true)
        }
    }

    func saveAs() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [PatchDocumentV2.contentType]
        panel.nameFieldStringValue =
            documentURL?.lastPathComponent ?? "patch.\(PatchDocumentV2.fileExtension)"
        panel.canCreateDirectories = true
        guard panel.runModal() == .OK, var url = panel.url else { return }
        if url.pathExtension.isEmpty {
            url.appendPathExtension(PatchDocumentV2.fileExtension)
        }
        do { try write(to: url) } catch {
            setStatus("save: \(error.localizedDescription)", isError: true)
        }
    }

    func open() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [PatchDocumentV2.contentType, .json]
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        Task {
            do { try await read(from: url) } catch {
                setStatus("open: \(error.localizedDescription)", isError: true)
            }
        }
    }

    func openRecent(_ url: URL) {
        Task {
            do { try await read(from: url) } catch {
                setStatus("open: \(error.localizedDescription)", isError: true)
                refreshRecentDocuments()
            }
        }
    }

    func clearRecentDocuments() {
        NSDocumentController.shared.clearRecentDocuments(nil)
        refreshRecentDocuments()
    }

    private func noteRecentDocument(_ url: URL) {
        NSDocumentController.shared.noteNewRecentDocumentURL(url)
        refreshRecentDocuments()
    }

    private func refreshRecentDocuments() {
        recentDocumentURLs = Array(
            NSDocumentController.shared.recentDocumentURLs.prefix(10))
    }
}
