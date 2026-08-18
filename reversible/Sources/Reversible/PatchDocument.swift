import AppKit
import SwiftUI
import UniformTypeIdentifiers

/// Complete authoring document. V3 separates the root scene from reusable
/// definition graphs; the engine receives a presentation-free projection and
/// owns its hygienic expansion. The decoder retains a deterministic v1 migration.
struct PatchDocument: Codable {
    struct Node: Codable {
        let id: String
        let kind: NodeKind
        let x: Double
        let y: Double
        let hue: Double
        let values: [String: Double]
        let inputs: [String: [String]]
        var module: ModuleReference?
        var bindings: [String: String]

        init(
            id: String,
            kind: NodeKind,
            x: Double,
            y: Double,
            hue: Double,
            values: [String: Double],
            inputs: [String: [String]],
            module: ModuleReference? = nil,
            bindings: [String: String] = [:]
        ) {
            self.id = id
            self.kind = kind
            self.x = x
            self.y = y
            self.hue = hue
            self.values = values
            self.inputs = inputs
            self.module = module
            self.bindings = bindings
        }

        private enum CodingKeys: String, CodingKey {
            case id, kind, x, y, hue, values, inputs, module, bindings
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            id = try c.decode(String.self, forKey: .id)
            kind = try c.decode(NodeKind.self, forKey: .kind)
            x = try c.decode(Double.self, forKey: .x)
            y = try c.decode(Double.self, forKey: .y)
            hue = try c.decode(Double.self, forKey: .hue)
            values = try c.decodeIfPresent([String: Double].self, forKey: .values) ?? [:]
            inputs = try c.decodeIfPresent([String: [String]].self, forKey: .inputs) ?? [:]
            module = try c.decodeIfPresent(ModuleReference.self, forKey: .module)
            bindings = try c.decodeIfPresent([String: String].self, forKey: .bindings) ?? [:]
        }
    }

    struct Graph: Codable {
        var nodes: [Node]
        var order: [String]
        var panX: Double
        var panY: Double
    }

    struct Definition: Codable {
        var id: String
        var version: Int
        var title: String
        var input: String
        var output: String
        var parameters: [ModuleParameter]
        var graph: Graph
    }

    static let currentVersion = 3

    var version: Int
    var scene: Graph
    var definitions: [Definition]
    var velocity: Double
    var autoArrange: Bool

    // Compatibility projections keep existing callers source-compatible while
    // every newly encoded file uses the explicit v3 scene envelope.
    var nodes: [Node] { scene.nodes }
    var order: [String] { scene.order }
    var panX: Double { scene.panX }
    var panY: Double { scene.panY }

    init(
        nodes: [Node],
        order: [String],
        velocity: Double,
        panX: Double,
        panY: Double,
        autoArrange: Bool,
        definitions: [Definition] = []
    ) {
        version = Self.currentVersion
        scene = .init(nodes: nodes, order: order, panX: panX, panY: panY)
        self.definitions = definitions
        self.velocity = velocity
        self.autoArrange = autoArrange
        migrateLegacyPhasers()
    }

    private enum CodingKeys: String, CodingKey {
        case version, scene, definitions
        case nodes, order, velocity, panX, panY, autoArrange
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        version = try c.decodeIfPresent(Int.self, forKey: .version) ?? 1
        guard version <= Self.currentVersion else {
            throw PatchDocumentError.unsupportedVersion(version)
        }
        velocity = try c.decodeIfPresent(Double.self, forKey: .velocity) ?? 1
        autoArrange = try c.decodeIfPresent(Bool.self, forKey: .autoArrange) ?? false
        if version >= 3 {
            scene = try c.decode(Graph.self, forKey: .scene)
            definitions = try c.decodeIfPresent([Definition].self, forKey: .definitions) ?? []
        } else {
            scene = .init(
                nodes: try c.decode([Node].self, forKey: .nodes),
                order: try c.decode([String].self, forKey: .order),
                panX: try c.decodeIfPresent(Double.self, forKey: .panX) ?? 0,
                panY: try c.decodeIfPresent(Double.self, forKey: .panY) ?? 0
            )
            definitions = []
            migrateLegacyPhasers()
            version = Self.currentVersion
        }
    }

    private mutating func migrateLegacyPhasers() {
        for index in scene.nodes.indices where scene.nodes[index].kind == .phaser {
            guard scene.nodes[index].module == nil else { continue }
            scene.nodes[index].module = StandardModuleLibrary.phaser
        }
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(Self.currentVersion, forKey: .version)
        try c.encode(scene, forKey: .scene)
        try c.encode(definitions, forKey: .definitions)
        try c.encode(velocity, forKey: .velocity)
        try c.encode(autoArrange, forKey: .autoArrange)
    }

    static let fileExtension = "rvpatch"

    static var contentType: UTType {
        UTType(filenameExtension: fileExtension) ?? .json
    }
}

enum PatchDocumentError: LocalizedError {
    case unsupportedVersion(Int)

    var errorDescription: String? {
        switch self {
        case .unsupportedVersion(let version):
            "patch version \(version) is newer than this build (\(PatchDocument.currentVersion))"
        }
    }
}

extension PatchModel {
    private func documentNode(_ node: PatchNode) -> PatchDocument.Node {
        .init(
            id: node.id,
            kind: node.kind,
            x: node.position.x,
            y: node.position.y,
            hue: node.hue,
            values: node.values,
            inputs: node.inputs,
            module: node.module,
            bindings: node.bindings
        )
    }

    private func documentGraph(_ graph: PatchGraphState) -> PatchDocument.Graph {
        .init(
            nodes: graph.order.compactMap { graph.nodes[$0] }.map(documentNode),
            order: graph.order,
            panX: graph.pan.width,
            panY: graph.pan.height
        )
    }

    func makeDocument() -> PatchDocument {
        snapshotActiveGraph()
        let encodedDefinitions = definitions.values.sorted {
            ($0.reference.id, $0.reference.version) <
                ($1.reference.id, $1.reference.version)
        }.map { definition in
            PatchDocument.Definition(
                id: definition.reference.id,
                version: definition.reference.version,
                title: definition.title,
                input: definition.inputNodeID,
                output: definition.outputNodeID,
                parameters: definition.parameters,
                graph: documentGraph(definition.graph)
            )
        }
        return PatchDocument(
            nodes: documentGraph(sceneGraph).nodes,
            order: sceneGraph.order,
            velocity: velocity,
            panX: sceneGraph.pan.width,
            panY: sceneGraph.pan.height,
            autoArrange: autoArrange,
            definitions: encodedDefinitions
        )
    }

    private func modelGraph(_ graph: PatchDocument.Graph) -> PatchGraphState {
        let known = Set(graph.nodes.map(\.id))
        var fresh: [String: PatchNode] = [:]
        for stored in graph.nodes {
            let spec = stored.kind.spec
            var values: [String: Double] = [:]
            for knob in spec.knobs {
                values[knob.name] = stored.values[knob.name] ?? knob.def
            }
            var inputs: [String: [String]] = [:]
            for port in spec.inlets {
                inputs[port] = (stored.inputs[port] ?? []).filter {
                    known.contains($0) && $0 != stored.id
                }
            }
            fresh[stored.id] = PatchNode(
                id: stored.id,
                kind: stored.kind,
                position: CGPoint(x: stored.x, y: stored.y),
                values: values,
                inputs: inputs,
                hue: stored.hue,
                module: stored.module,
                bindings: stored.bindings
            )
        }
        var seen = Set<String>()
        let stableOrder = graph.order.filter {
            known.contains($0) && seen.insert($0).inserted
        } + graph.nodes.map(\.id).filter { seen.insert($0).inserted }
        return .init(
            nodes: fresh,
            order: stableOrder,
            pan: CGSize(width: graph.panX, height: graph.panY)
        )
    }

    func apply(_ document: PatchDocument) async throws {
        guard document.version <= PatchDocument.currentVersion else {
            throw PatchDocumentError.unsupportedVersion(document.version)
        }

        var loadedDefinitions: [ModuleReference: ModuleDefinitionState] = [:]
        for stored in document.definitions {
            let reference = ModuleReference(id: stored.id, version: stored.version)
            loadedDefinitions[reference] = .init(
                reference: reference,
                title: stored.title,
                inputNodeID: stored.input,
                outputNodeID: stored.output,
                parameters: stored.parameters,
                graph: modelGraph(stored.graph)
            )
        }

        var root = modelGraph(document.scene)
        // Defensive compatibility for programmatic documents that bypassed
        // PatchDocument's decoder migration.
        for id in root.order {
            guard var node = root.nodes[id], node.kind == .phaser,
                  node.module == nil else { continue }
            node.module = StandardModuleLibrary.phaser
            root.nodes[id] = node
        }

        resetHierarchy(to: root)
        definitions = loadedDefinitions
        autoArrange = document.autoArrange
        velocity = document.velocity
        advanceAuthoredRevision()
        adoptCounter(from: root.nodes.keys)

        await pushGraph()
        setVelocity(document.velocity, force: true)
    }

    func newPatch() async {
        resetHierarchy(to: .init(nodes: [:], order: [], pan: .zero))
        resetCounter()
        seedBootPatch()
        snapshotActiveGraph()
        documentURL = nil
        await pushGraph()
        setVelocity(velocity)
    }

    func newPhaserDemo() async {
        resetCounter()
        let demo = FactoryPatches.modalPhaser
        resetHierarchy(to: demo)
        adoptCounter(from: demo.nodes.keys)
        advanceAuthoredRevision()
        documentURL = nil
        await pushGraph()
        setVelocity(velocity)
    }

    func write(to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(makeDocument()).write(to: url, options: .atomic)
        documentURL = url
        setStatus("saved · \(url.lastPathComponent)", isError: false)
    }

    func read(from url: URL) async throws {
        let document = try JSONDecoder().decode(
            PatchDocument.self,
            from: Data(contentsOf: url)
        )
        try await apply(document)
        documentURL = url
        setStatus("loaded · \(url.lastPathComponent)", isError: false)
    }

    func save() {
        guard let url = documentURL else { return saveAs() }
        do { try write(to: url) }
        catch { setStatus("save: \(error.localizedDescription)", isError: true) }
    }

    func saveAs() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [PatchDocument.contentType]
        panel.nameFieldStringValue =
            documentURL?.lastPathComponent ?? "patch.\(PatchDocument.fileExtension)"
        panel.canCreateDirectories = true
        guard panel.runModal() == .OK, var url = panel.url else { return }
        if url.pathExtension.isEmpty { url.appendPathExtension(PatchDocument.fileExtension) }
        do { try write(to: url) }
        catch { setStatus("save: \(error.localizedDescription)", isError: true) }
    }

    func open() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [PatchDocument.contentType, .json]
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        Task {
            do { try await read(from: url) }
            catch { setStatus("open: \(error.localizedDescription)", isError: true) }
        }
    }
}

/// Built-in scenes are complete immutable values before the model adopts
/// them. Loading one therefore publishes its nodes and edges together instead
/// of exposing a sequence of partially connected live edits to the UI and
/// engine startup tasks.
enum FactoryPatches {
    static var modalPhaser: PatchGraphState {
        func node(
            _ index: Int,
            _ kind: NodeKind,
            at position: CGPoint,
            values authoredValues: [String: Double] = [:],
            inputs authoredInputs: [String: [String]] = [:]
        ) -> PatchNode {
            let spec = kind.spec
            var values = Dictionary(uniqueKeysWithValues: spec.knobs.map {
                ($0.name, $0.def)
            })
            for (name, value) in authoredValues where values[name] != nil {
                values[name] = value
            }
            var inputs = Dictionary(uniqueKeysWithValues: spec.inlets.map {
                ($0, [String]())
            })
            for (port, sources) in authoredInputs where inputs[port] != nil {
                inputs[port] = sources
            }
            let module: ModuleReference? = switch kind {
            case .phaser: StandardModuleLibrary.phaser
            case .allpass: StandardModuleLibrary.allpass
            default: nil
            }
            return PatchNode(
                id: "\(kind.rawValue)\(index)",
                kind: kind,
                position: Grid.snap(position),
                values: values,
                inputs: inputs,
                hue: Double((index * 137) % 360),
                module: module
            )
        }

        let authored = [
            // An unaddressed resonator strikes only at master time zero. The
            // demo is hot-swapped after startup, so that strike may already
            // be in the past. This qualified sub-Hz saw repeatedly crosses
            // zero and therefore gives the scene an audible re-trigger.
            node(1, .source, at: CGPoint(x: 88, y: 420), values: [
                "freq": 0.63,
            ]),
            node(2, .resonator, at: CGPoint(x: 88, y: 132), inputs: [
                "addr": ["source1"],
            ]),
            node(3, .phaser, at: CGPoint(x: 360, y: 132), inputs: [
                "in": ["resonator2"],
            ]),
            node(4, .reverb, at: CGPoint(x: 700, y: 132), inputs: [
                "in": ["phaser3"],
            ]),
            node(5, .out, at: CGPoint(x: 980, y: 240), inputs: [
                "in": ["reverb4"],
            ]),
            node(
                6,
                .scope,
                at: CGPoint(x: 980, y: 430),
                values: ["window": ScopeSignalKnowledge.visibleCycles / 220],
                inputs: [
                    "ch1": ["out5"],
                ]
            ),
        ]
        return PatchGraphState(
            nodes: Dictionary(uniqueKeysWithValues: authored.map { ($0.id, $0) }),
            order: authored.map(\.id),
            pan: .zero
        )
    }
}
