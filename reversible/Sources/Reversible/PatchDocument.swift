import AppKit
import SwiftUI
import UniformTypeIdentifiers

/// The patch as a FILE. `PatchModel.serialize()` targets the engine — it drops
/// monitors, positions, and hues because the kernel has no use for them. A
/// document is the SURFACE: everything you'd lose by quitting, including the
/// scope you left watching the mix and where you put it.
///
/// Deliberately a flat, hand-readable JSON term (same shape family as the
/// engine graph) rather than an opaque archive: a patch is small, diffable,
/// and worth being able to hand-edit.
struct PatchDocument: Codable {
    struct Node: Codable {
        let id: String
        let kind: NodeKind
        let x: Double
        let y: Double
        let hue: Double
        let values: [String: Double]
        /// inlet port → ordered source node ids
        let inputs: [String: [String]]
    }

    /// Bumped only on a breaking shape change; a reader refuses a version it
    /// doesn't know rather than silently loading half a patch.
    static let currentVersion = 1

    var version = PatchDocument.currentVersion
    var nodes: [Node]
    /// stable z/render order (also the order inlet menus list sources in)
    var order: [String]
    var velocity: Double
    var panX: Double
    var panY: Double
    var autoArrange: Bool

    static let fileExtension = "rvpatch"

    /// A dynamic UTI is fine here — the panels only need it to filter and to
    /// stamp the extension, and a bare SwiftPM binary has no Info.plist to
    /// declare an exported type in.
    static var contentType: UTType {
        UTType(filenameExtension: fileExtension) ?? .json
    }
}

enum PatchDocumentError: LocalizedError {
    case unsupportedVersion(Int)

    var errorDescription: String? {
        switch self {
        case .unsupportedVersion(let v):
            return "patch version \(v) is newer than this build (\(PatchDocument.currentVersion))"
        }
    }
}

// ── Model ↔ document ──────────────────────────────────────────────────────
extension PatchModel {
    func makeDocument() -> PatchDocument {
        PatchDocument(
            nodes: order.compactMap { nodes[$0] }.map { n in
                PatchDocument.Node(
                    id: n.id, kind: n.kind,
                    x: n.position.x, y: n.position.y, hue: n.hue,
                    values: n.values, inputs: n.inputs)
            },
            order: order,
            velocity: velocity,
            panX: pan.width, panY: pan.height,
            autoArrange: autoArrange)
    }

    /// Adopt a document as the live patch. Everything is rebuilt through the
    /// same invariants `addNode` maintains — a knob absent from the file takes
    /// its spec default, an inlet the spec no longer has is dropped, and an
    /// edge to a node that isn't in the file is dropped — so a patch written by
    /// an older build (or hand-edited) loads as the nearest VALID patch rather
    /// than a half-wired one.
    func apply(_ doc: PatchDocument) async throws {
        guard doc.version <= PatchDocument.currentVersion else {
            throw PatchDocumentError.unsupportedVersion(doc.version)
        }
        let known = Set(doc.nodes.map(\.id))
        var fresh: [String: PatchNode] = [:]
        for d in doc.nodes {
            let spec = d.kind.spec
            var values: [String: Double] = [:]
            for k in spec.knobs { values[k.name] = d.values[k.name] ?? k.def }
            var inputs: [String: [String]] = [:]
            for p in spec.inlets {
                inputs[p] = (d.inputs[p] ?? []).filter { known.contains($0) && $0 != d.id }
            }
            fresh[d.id] = PatchNode(
                id: d.id, kind: d.kind,
                position: CGPoint(x: d.x, y: d.y),
                values: values, inputs: inputs, hue: d.hue)
        }
        nodes = fresh
        // The file's order is authoritative, but never let it lose or invent a
        // node (a hand-edit that forgets an id would make it unrenderable).
        var seen = Set<String>()
        order = doc.order.filter { known.contains($0) && seen.insert($0).inserted }
            + doc.nodes.map(\.id).filter { !seen.contains($0) }
        pan = CGSize(width: doc.panX, height: doc.panY)
        autoArrange = doc.autoArrange
        velocity = doc.velocity
        advanceAuthoredRevision()
        // New nodes must not collide with loaded ids: resume the counter past
        // the highest numeric suffix in the file.
        adoptCounter(from: known)

        await pushGraph()
        // pushGraph re-applies a non-default velocity after a COLD compile, but
        // a load whose graph happens to match what's already running skips the
        // compile entirely — so drive the clock here regardless.
        setVelocity(doc.velocity, force: true)
    }

    /// Reset to the boot patch (Out + one Osc), forgetting the file.
    func newPatch() async {
        nodes = [:]
        order = []
        pan = .zero
        resetCounter()
        seedBootPatch()
        documentURL = nil
        await pushGraph()
        setVelocity(velocity)
    }

    // ── File I/O ──────────────────────────────────────────────────────────
    func write(to url: URL) throws {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        try enc.encode(makeDocument()).write(to: url, options: .atomic)
        documentURL = url
        setStatus("saved · \(url.lastPathComponent)", isError: false)
    }

    func read(from url: URL) async throws {
        let doc = try JSONDecoder().decode(PatchDocument.self, from: Data(contentsOf: url))
        try await apply(doc)
        documentURL = url
        setStatus("loaded · \(url.lastPathComponent)", isError: false)
    }

    /// Save to the open file, or ask where if there isn't one yet.
    func save() {
        guard let url = documentURL else { return saveAs() }
        do { try write(to: url) } catch {
            setStatus("save: \(error.localizedDescription)", isError: true)
        }
    }

    func saveAs() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [PatchDocument.contentType]
        panel.nameFieldStringValue =
            documentURL?.lastPathComponent ?? "patch.\(PatchDocument.fileExtension)"
        panel.canCreateDirectories = true
        guard panel.runModal() == .OK, var url = panel.url else { return }
        if url.pathExtension.isEmpty {
            url.appendPathExtension(PatchDocument.fileExtension)
        }
        do { try write(to: url) } catch {
            setStatus("save: \(error.localizedDescription)", isError: true)
        }
    }

    func open() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [PatchDocument.contentType, .json]
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        Task {
            do { try await read(from: url) } catch {
                setStatus("open: \(error.localizedDescription)", isError: true)
            }
        }
    }
}
