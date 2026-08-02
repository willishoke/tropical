import Foundation

// MARK: - Version 2 authored document

struct PatchDocumentV2: Codable, Equatable {
    static let currentVersion = 2

    var vocabulary: PatchVocabularyReferenceV2
    var sceneMetadata: [String: JSONValue]
    var nodes: [PatchNodeV2]
    var monitors: [PatchClientMonitorV2]
    var output: String?
    var transport: PatchTransportV2
    var presentation: PatchPresentationV2
    var unknownFields: [String: JSONValue]

    init(
        vocabulary: PatchVocabularyReferenceV2,
        sceneMetadata: [String: JSONValue],
        nodes: [PatchNodeV2],
        monitors: [PatchClientMonitorV2],
        output: String?,
        transport: PatchTransportV2,
        presentation: PatchPresentationV2,
        unknownFields: [String: JSONValue]
    ) {
        self.vocabulary = vocabulary
        self.sceneMetadata = sceneMetadata
        self.nodes = nodes
        self.monitors = monitors
        self.output = output
        self.transport = transport
        self.presentation = presentation
        self.unknownFields = unknownFields
    }

    init(from decoder: Decoder) throws {
        let raw = try JSONValue(from: decoder)
        self = try Self.parse(raw, path: "$" )
    }

    func encode(to encoder: Encoder) throws {
        try jsonValue.encode(to: encoder)
    }

    static func decode(from data: Data) throws -> PatchDocumentV2 {
        try JSONDecoder().decode(PatchDocumentV2.self, from: data)
    }

    func encoded(prettyPrinted: Bool = true) throws -> Data {
        let encoder = JSONEncoder()
        if prettyPrinted {
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        } else {
            encoder.outputFormatting = [.sortedKeys]
        }
        return try encoder.encode(self)
    }

    private static func parse(_ raw: JSONValue, path: String) throws -> PatchDocumentV2 {
        var object = try PatchV2JSON.object(raw, path: path)
        let version = try PatchV2JSON.integer(
            PatchV2JSON.take("version", from: &object, path: path),
            path: "\(path).version"
        )
        guard version == currentVersion else {
            throw PatchDocumentV2Error.unsupportedVersion(version)
        }

        let vocabulary = try PatchVocabularyReferenceV2.parse(
            PatchV2JSON.take("vocabulary", from: &object, path: path),
            path: "\(path).vocabulary"
        )
        let sceneMetadata = try PatchV2JSON.object(
            PatchV2JSON.take("scene", from: &object, path: path),
            path: "\(path).scene"
        )
        let nodeValues = try PatchV2JSON.array(
            PatchV2JSON.take("nodes", from: &object, path: path),
            path: "\(path).nodes"
        )
        let nodes = try nodeValues.enumerated().map {
            try PatchNodeV2.parse($0.element, path: "\(path).nodes[\($0.offset)]")
        }
        let monitorValues = try PatchV2JSON.array(
            PatchV2JSON.take("monitors", from: &object, path: path),
            path: "\(path).monitors"
        )
        let monitors = try monitorValues.enumerated().map {
            try PatchClientMonitorV2.parse($0.element, path: "\(path).monitors[\($0.offset)]")
        }
        let outputValue = try PatchV2JSON.take("output", from: &object, path: path)
        let output: String?
        if outputValue == .null {
            output = nil
        } else {
            output = try PatchV2JSON.string(outputValue, path: "\(path).output")
        }
        let transport = try PatchTransportV2.parse(
            PatchV2JSON.take("transport", from: &object, path: path),
            path: "\(path).transport"
        )
        let presentation = try PatchPresentationV2.parse(
            PatchV2JSON.take("presentation", from: &object, path: path),
            path: "\(path).presentation"
        )

        return PatchDocumentV2(
            vocabulary: vocabulary,
            sceneMetadata: sceneMetadata,
            nodes: nodes,
            monitors: monitors,
            output: output,
            transport: transport,
            presentation: presentation,
            unknownFields: object
        )
    }

    var jsonValue: JSONValue {
        var object = unknownFields
        object["version"] = .number(Double(Self.currentVersion))
        object["vocabulary"] = vocabulary.jsonValue
        object["scene"] = .object(sceneMetadata)
        object["nodes"] = .array(nodes.map(\.jsonValue))
        object["monitors"] = .array(monitors.map(\.jsonValue))
        object["output"] = output.map(JSONValue.string) ?? .null
        object["transport"] = transport.jsonValue
        object["presentation"] = presentation.jsonValue
        return .object(object)
    }
}

struct PatchVocabularyReferenceV2: Equatable {
    var schemaID: String
    var schemaVersion: Int
    var fingerprint: String
    var unknownFields: [String: JSONValue]

    init(
        schemaID: String,
        schemaVersion: Int,
        fingerprint: String,
        unknownFields: [String: JSONValue]
    ) {
        self.schemaID = schemaID
        self.schemaVersion = schemaVersion
        self.fingerprint = fingerprint
        self.unknownFields = unknownFields
    }

    init(vocabulary: EngineVocabulary) {
        schemaID = vocabulary.schemaID
        schemaVersion = vocabulary.schemaVersion
        fingerprint = vocabulary.fingerprint
        unknownFields = [:]
    }

    fileprivate static func parse(_ raw: JSONValue, path: String) throws -> Self {
        var object = try PatchV2JSON.object(raw, path: path)
        let schemaID = try PatchV2JSON.string(
            PatchV2JSON.take("schema", from: &object, path: path),
            path: "\(path).schema"
        )
        let schemaVersion = try PatchV2JSON.integer(
            PatchV2JSON.take("schema_version", from: &object, path: path),
            path: "\(path).schema_version"
        )
        let fingerprint = try PatchV2JSON.string(
            PatchV2JSON.take("fingerprint", from: &object, path: path),
            path: "\(path).fingerprint"
        )
        return Self(
            schemaID: schemaID,
            schemaVersion: schemaVersion,
            fingerprint: fingerprint,
            unknownFields: object
        )
    }

    fileprivate var jsonValue: JSONValue {
        var object = unknownFields
        object["schema"] = .string(schemaID)
        object["schema_version"] = .number(Double(schemaVersion))
        object["fingerprint"] = .string(fingerprint)
        return .object(object)
    }
}

struct PatchNodeV2: Equatable {
    var id: String
    var kind: NodeKindID
    /// Structural JSON remains lossless; the client does not reduce it to
    /// numeric knobs or discard fields it cannot currently render.
    var structuralParams: [String: JSONValue]
    /// inlet name -> exact authored source order.
    var inputs: [String: [String]]
    var unknownFields: [String: JSONValue]

    fileprivate static func parse(_ raw: JSONValue, path: String) throws -> Self {
        var object = try PatchV2JSON.object(raw, path: path)
        let id = try PatchV2JSON.string(
            PatchV2JSON.take("id", from: &object, path: path),
            path: "\(path).id"
        )
        let kindText = try PatchV2JSON.string(
            PatchV2JSON.take("kind", from: &object, path: path),
            path: "\(path).kind"
        )
        let kind = try NodeKindID(validating: kindText)
        let params = try PatchV2JSON.object(
            PatchV2JSON.take("params", from: &object, path: path),
            path: "\(path).params"
        )
        let inputs = try PatchV2JSON.inputs(
            PatchV2JSON.take("in", from: &object, path: path),
            path: "\(path).in"
        )
        return Self(
            id: id,
            kind: kind,
            structuralParams: params,
            inputs: inputs,
            unknownFields: object
        )
    }

    fileprivate var jsonValue: JSONValue {
        var object = unknownFields
        object["id"] = .string(id)
        object["kind"] = .string(kind.rawValue)
        object["params"] = .object(structuralParams)
        object["in"] = PatchV2JSON.inputsValue(inputs)
        return .object(object)
    }
}

struct PatchClientMonitorV2: Equatable {
    var id: String
    var kind: ClientMonitorKindID
    var state: [String: JSONValue]
    var inputs: [String: [String]]
    var unknownFields: [String: JSONValue]

    fileprivate static func parse(_ raw: JSONValue, path: String) throws -> Self {
        var object = try PatchV2JSON.object(raw, path: path)
        let id = try PatchV2JSON.string(
            PatchV2JSON.take("id", from: &object, path: path),
            path: "\(path).id"
        )
        let kindText = try PatchV2JSON.string(
            PatchV2JSON.take("kind", from: &object, path: path),
            path: "\(path).kind"
        )
        let kind = try ClientMonitorKindID(validating: kindText)
        let state = try PatchV2JSON.object(
            PatchV2JSON.take("state", from: &object, path: path),
            path: "\(path).state"
        )
        let inputs = try PatchV2JSON.inputs(
            PatchV2JSON.take("in", from: &object, path: path),
            path: "\(path).in"
        )
        return Self(id: id, kind: kind, state: state, inputs: inputs, unknownFields: object)
    }

    fileprivate var jsonValue: JSONValue {
        var object = unknownFields
        object["id"] = .string(id)
        object["kind"] = .string(kind.rawValue)
        object["state"] = .object(state)
        object["in"] = PatchV2JSON.inputsValue(inputs)
        return .object(object)
    }
}

struct PatchTransportV2: Equatable {
    var velocity: Double
    var position: Double?
    var unknownFields: [String: JSONValue]

    fileprivate static func parse(_ raw: JSONValue, path: String) throws -> Self {
        var object = try PatchV2JSON.object(raw, path: path)
        let velocity = try PatchV2JSON.number(
            PatchV2JSON.take("velocity", from: &object, path: path),
            path: "\(path).velocity"
        )
        let position: Double?
        if let value = object.removeValue(forKey: "position") {
            position = value == .null ? nil : try PatchV2JSON.number(value, path: "\(path).position")
        } else {
            position = nil
        }
        return Self(velocity: velocity, position: position, unknownFields: object)
    }

    fileprivate var jsonValue: JSONValue {
        var object = unknownFields
        object["velocity"] = .number(velocity)
        if let position {
            object["position"] = .number(position)
        }
        return .object(object)
    }
}

struct PatchPositionV2: Equatable {
    var x: Double
    var y: Double
    var hue: Double
    var unknownFields: [String: JSONValue]

    fileprivate static func parse(_ raw: JSONValue, path: String) throws -> Self {
        var object = try PatchV2JSON.object(raw, path: path)
        let x = try PatchV2JSON.number(
            PatchV2JSON.take("x", from: &object, path: path),
            path: "\(path).x"
        )
        let y = try PatchV2JSON.number(
            PatchV2JSON.take("y", from: &object, path: path),
            path: "\(path).y"
        )
        let hue = try PatchV2JSON.number(
            PatchV2JSON.take("hue", from: &object, path: path),
            path: "\(path).hue"
        )
        return Self(x: x, y: y, hue: hue, unknownFields: object)
    }

    fileprivate var jsonValue: JSONValue {
        var object = unknownFields
        object["x"] = .number(x)
        object["y"] = .number(y)
        object["hue"] = .number(hue)
        return .object(object)
    }
}

struct PatchPresentationV2: Equatable {
    var order: [String]
    var positions: [String: PatchPositionV2]
    var panX: Double
    var panY: Double
    var autoArrange: Bool
    var unknownFields: [String: JSONValue]

    fileprivate static func parse(_ raw: JSONValue, path: String) throws -> Self {
        var object = try PatchV2JSON.object(raw, path: path)
        let order = try PatchV2JSON.strings(
            PatchV2JSON.take("order", from: &object, path: path),
            path: "\(path).order"
        )
        let rawPositions = try PatchV2JSON.object(
            PatchV2JSON.take("positions", from: &object, path: path),
            path: "\(path).positions"
        )
        var positions: [String: PatchPositionV2] = [:]
        for id in rawPositions.keys.sorted() {
            positions[id] = try PatchPositionV2.parse(
                rawPositions[id]!,
                path: "\(path).positions.\(id)"
            )
        }
        let panX = try PatchV2JSON.number(
            PatchV2JSON.take("pan_x", from: &object, path: path),
            path: "\(path).pan_x"
        )
        let panY = try PatchV2JSON.number(
            PatchV2JSON.take("pan_y", from: &object, path: path),
            path: "\(path).pan_y"
        )
        let autoArrange = try PatchV2JSON.bool(
            PatchV2JSON.take("auto_arrange", from: &object, path: path),
            path: "\(path).auto_arrange"
        )
        return Self(
            order: order,
            positions: positions,
            panX: panX,
            panY: panY,
            autoArrange: autoArrange,
            unknownFields: object
        )
    }

    fileprivate var jsonValue: JSONValue {
        var object = unknownFields
        object["order"] = .array(order.map(JSONValue.string))
        object["positions"] = .object(positions.mapValues(\.jsonValue))
        object["pan_x"] = .number(panX)
        object["pan_y"] = .number(panY)
        object["auto_arrange"] = .bool(autoArrange)
        return .object(object)
    }
}

// MARK: - Explicit v1 migration

struct PatchDocumentV1Migrator {
    static func migrate(
        data: Data,
        vocabulary: EngineVocabulary
    ) throws -> PatchDocumentMigrationResult {
        let raw = try JSONDecoder().decode(JSONValue.self, from: data)
        var root = try PatchV2JSON.object(raw, path: "$")
        let version = try PatchV2JSON.integer(
            PatchV2JSON.take("version", from: &root, path: "$"),
            path: "$.version"
        )
        guard version == 1 else {
            throw PatchDocumentV2Error.expectedV1(version)
        }

        let rawNodes = try PatchV2JSON.array(
            PatchV2JSON.take("nodes", from: &root, path: "$"),
            path: "$.nodes"
        )
        let order = try PatchV2JSON.strings(
            PatchV2JSON.take("order", from: &root, path: "$"),
            path: "$.order"
        )
        let velocity = try PatchV2JSON.number(
            PatchV2JSON.take("velocity", from: &root, path: "$"),
            path: "$.velocity"
        )
        let panX = try PatchV2JSON.number(
            PatchV2JSON.take("panX", from: &root, path: "$"),
            path: "$.panX"
        )
        let panY = try PatchV2JSON.number(
            PatchV2JSON.take("panY", from: &root, path: "$"),
            path: "$.panY"
        )
        let autoArrange = try PatchV2JSON.bool(
            PatchV2JSON.take("autoArrange", from: &root, path: "$"),
            path: "$.autoArrange"
        )

        var nodes: [PatchNodeV2] = []
        var monitors: [PatchClientMonitorV2] = []
        var positions: [String: PatchPositionV2] = [:]
        var facts: [PatchMigrationFact] = [
            .init(code: .sourceVersion, path: "$.version", detail: "decoded raw v1 JSON"),
        ]

        for (index, rawNode) in rawNodes.enumerated() {
            let path = "$.nodes[\(index)]"
            var object = try PatchV2JSON.object(rawNode, path: path)
            let id = try PatchV2JSON.string(
                PatchV2JSON.take("id", from: &object, path: path),
                path: "\(path).id"
            )
            let kindText = try PatchV2JSON.string(
                PatchV2JSON.take("kind", from: &object, path: path),
                path: "\(path).kind"
            )
            let x = try PatchV2JSON.number(
                PatchV2JSON.take("x", from: &object, path: path),
                path: "\(path).x"
            )
            let y = try PatchV2JSON.number(
                PatchV2JSON.take("y", from: &object, path: path),
                path: "\(path).y"
            )
            let hue = try PatchV2JSON.number(
                PatchV2JSON.take("hue", from: &object, path: path),
                path: "\(path).hue"
            )
            let values = try PatchV2JSON.object(
                PatchV2JSON.take("values", from: &object, path: path),
                path: "\(path).values"
            )
            let inputs = try PatchV2JSON.inputs(
                PatchV2JSON.take("inputs", from: &object, path: path),
                path: "\(path).inputs"
            )
            positions[id] = PatchPositionV2(x: x, y: y, hue: hue, unknownFields: [:])

            if kindText == "scope" {
                let monitorKind = try ClientMonitorKindID(validating: "client.monitor.scope")
                monitors.append(PatchClientMonitorV2(
                    id: id,
                    kind: monitorKind,
                    state: values,
                    inputs: inputs,
                    unknownFields: PatchV2JSON.preserveLegacyUnknown(
                        object,
                        collidingWith: ["id", "kind", "state", "in"]
                    )
                ))
                facts.append(.init(
                    code: .separatedClientMonitor,
                    path: path,
                    detail: "preserved legacy scope as client.monitor.scope"
                ))
            } else {
                let kind = try NodeKindID(validating: kindText)
                nodes.append(PatchNodeV2(
                    id: id,
                    kind: kind,
                    structuralParams: values,
                    inputs: inputs,
                    unknownFields: PatchV2JSON.preserveLegacyUnknown(
                        object,
                        collidingWith: ["id", "kind", "params", "in"]
                    )
                ))
                facts.append(.init(
                    code: .preservedAuthoredNode,
                    path: path,
                    detail: "preserved \(id) as kind \(kindText)"
                ))
            }
            for key in object.keys.sorted() {
                facts.append(.init(
                    code: .preservedUnknownNodeField,
                    path: "\(path).\(key)",
                    detail: "preserved unknown v1 node field"
                ))
            }
        }

        let outputCandidates = nodes.filter { $0.kind.rawValue == "out" }.map(\.id)
        let output = outputCandidates.count == 1 ? outputCandidates[0] : nil
        for key in root.keys.sorted() {
            facts.append(.init(
                code: .preservedUnknownTopLevelField,
                path: "$.\(key)",
                detail: "preserved unknown v1 top-level field"
            ))
        }
        facts.append(.init(
            code: .explicitSaveRequired,
            path: "$",
            detail: "migration is in memory; an explicit save is required to write v2"
        ))

        let document = PatchDocumentV2(
            vocabulary: PatchVocabularyReferenceV2(vocabulary: vocabulary),
            sceneMetadata: [
                "migration_source": .string("rvpatch-v1"),
            ],
            nodes: nodes,
            monitors: monitors,
            output: output,
            transport: PatchTransportV2(velocity: velocity, position: nil, unknownFields: [:]),
            presentation: PatchPresentationV2(
                order: order,
                positions: positions,
                panX: panX,
                panY: panY,
                autoArrange: autoArrange,
                unknownFields: [:]
            ),
            unknownFields: PatchV2JSON.preserveLegacyUnknown(
                root,
                collidingWith: [
                    "version", "vocabulary", "scene", "nodes", "monitors",
                    "output", "transport", "presentation",
                ]
            )
        )
        let validation = document.validate(against: vocabulary)
        return PatchDocumentMigrationResult(
            document: document,
            report: PatchMigrationReport(facts: facts, blockers: validation.blockers)
        )
    }
}

struct PatchDocumentMigrationResult: Equatable {
    var document: PatchDocumentV2
    var report: PatchMigrationReport
    /// Migration never writes through to the opened v1 URL. The UI may clear
    /// this state only after the user invokes an explicit v2 save operation.
    let requiresExplicitV2Save = true
}

struct PatchMigrationReport: Equatable {
    var facts: [PatchMigrationFact]
    var blockers: [PatchDocumentBlocker]

    var canCompile: Bool { blockers.isEmpty }
}

struct PatchMigrationFact: Equatable {
    enum Code: String, Equatable {
        case sourceVersion
        case preservedAuthoredNode
        case separatedClientMonitor
        case preservedUnknownTopLevelField
        case preservedUnknownNodeField
        case explicitSaveRequired
    }

    var code: Code
    var path: String
    var detail: String
}

// MARK: - Loss-aware validation

struct PatchDocumentValidationReport: Equatable {
    var blockers: [PatchDocumentBlocker]
    var canCompile: Bool { blockers.isEmpty }
}

struct PatchDocumentBlocker: Equatable {
    enum Code: String, Equatable {
        case vocabularyMismatch
        case invalidID
        case duplicateID
        case invalidOrder
        case invalidPosition
        case nonFiniteNumber
        case danglingEdge
        case selfEdge
        case monitorAsEngineSource
        case unknownKind
        case unknownPort
        case incompatibleConnection
        case arityExceeded
        case cycle
        case invalidOutput
    }

    var code: Code
    var path: String
    var detail: String
}

extension PatchDocumentV2 {
    func validate(against vocabulary: EngineVocabulary? = nil) -> PatchDocumentValidationReport {
        var blockers: [PatchDocumentBlocker] = []
        let block: (PatchDocumentBlocker.Code, String, String) -> Void = { code, path, detail in
            blockers.append(.init(code: code, path: path, detail: detail))
        }

        if let vocabulary,
           vocabulary.schemaID != self.vocabulary.schemaID
            || vocabulary.schemaVersion != self.vocabulary.schemaVersion
            || vocabulary.fingerprint != self.vocabulary.fingerprint {
            block(.vocabularyMismatch, "$.vocabulary", "document vocabulary does not match the running engine")
        }

        let allItems: [(id: String, path: String)] =
            nodes.enumerated().map { ($0.element.id, "$.nodes[\($0.offset)].id") }
            + monitors.enumerated().map { ($0.element.id, "$.monitors[\($0.offset)].id") }
        var seenIDs = Set<String>()
        for item in allItems {
            if item.id.isEmpty || item.id != item.id.trimmingCharacters(in: .whitespacesAndNewlines) {
                block(.invalidID, item.path, "ID must be nonempty with no surrounding whitespace")
            }
            if !seenIDs.insert(item.id).inserted {
                block(.duplicateID, item.path, "duplicate ID '\(item.id)'")
            }
        }
        let allIDs = Set(allItems.map(\.id))

        var seenOrder = Set<String>()
        for (index, id) in presentation.order.enumerated() {
            if !allIDs.contains(id) {
                block(.invalidOrder, "$.presentation.order[\(index)]", "order contains unknown ID '\(id)'")
            } else if !seenOrder.insert(id).inserted {
                block(.invalidOrder, "$.presentation.order[\(index)]", "order repeats ID '\(id)'")
            }
        }
        for id in allItems.map(\.id) where !seenOrder.contains(id) {
            block(.invalidOrder, "$.presentation.order", "order omits ID '\(id)'")
        }
        for id in presentation.positions.keys.sorted() where !allIDs.contains(id) {
            block(.invalidPosition, "$.presentation.positions.\(id)", "position belongs to unknown ID")
        }
        for item in allItems {
            guard let position = presentation.positions[item.id] else {
                block(.invalidPosition, "$.presentation.positions", "missing position for '\(item.id)'")
                continue
            }
            if !position.x.isFinite || !position.y.isFinite || !position.hue.isFinite {
                block(.invalidPosition, "$.presentation.positions.\(item.id)", "position/hue must be finite")
            }
        }
        if !presentation.panX.isFinite || !presentation.panY.isFinite {
            block(.invalidPosition, "$.presentation", "canvas pan must be finite")
        }
        if !transport.velocity.isFinite || transport.position?.isFinite == false {
            block(.nonFiniteNumber, "$.transport", "transport coordinates must be finite")
        }

        validateFiniteJSON(&blockers)

        let nodeIDs = Set(nodes.map(\.id))
        let monitorIDs = Set(monitors.map(\.id))
        for (nodeIndex, node) in nodes.enumerated() {
            for port in node.inputs.keys.sorted() {
                let sources = node.inputs[port] ?? []
                for (sourceIndex, source) in sources.enumerated() {
                    let path = "$.nodes[\(nodeIndex)].in.\(port)[\(sourceIndex)]"
                    if source == node.id {
                        block(.selfEdge, path, "self edge '\(source)' -> '\(node.id)'")
                    } else if monitorIDs.contains(source) {
                        block(.monitorAsEngineSource, path, "client monitor '\(source)' cannot feed an engine node")
                    } else if !nodeIDs.contains(source) {
                        block(.danglingEdge, path, "unknown source ID '\(source)'")
                    }
                }
            }
        }
        for (monitorIndex, monitor) in monitors.enumerated() {
            for port in monitor.inputs.keys.sorted() {
                for (sourceIndex, source) in (monitor.inputs[port] ?? []).enumerated() {
                    let path = "$.monitors[\(monitorIndex)].in.\(port)[\(sourceIndex)]"
                    if source == monitor.id {
                        block(.selfEdge, path, "monitor cannot observe itself")
                    } else if !nodeIDs.contains(source) {
                        block(.danglingEdge, path, "monitor source '\(source)' is not an engine node")
                    }
                }
            }
        }

        if let vocabulary {
            validateVocabularySemantics(vocabulary, blockers: &blockers)
        }
        validateOutput(vocabulary: vocabulary, blockers: &blockers)
        validateCycles(blockers: &blockers)
        return PatchDocumentValidationReport(blockers: blockers)
    }

    private func validateVocabularySemantics(
        _ vocabulary: EngineVocabulary,
        blockers: inout [PatchDocumentBlocker]
    ) {
        for (nodeIndex, node) in nodes.enumerated() {
            guard vocabulary.descriptor(for: node.kind) != nil else {
                blockers.append(.init(
                    code: .unknownKind,
                    path: "$.nodes[\(nodeIndex)].kind",
                    detail: "kind '\(node.kind.rawValue)' is not served by this engine"
                ))
                continue
            }
            for port in node.inputs.keys.sorted() {
                guard vocabulary.descriptor(for: node.kind)?.port(named: port)?.isInlet == true else {
                    blockers.append(.init(
                        code: .unknownPort,
                        path: "$.nodes[\(nodeIndex)].in.\(port)",
                        detail: "port '\(port)' is not a served inlet on '\(node.kind.rawValue)'"
                    ))
                    continue
                }
                for (sourceIndex, sourceID) in (node.inputs[port] ?? []).enumerated() {
                    guard let source = nodes.first(where: { $0.id == sourceID }) else { continue }
                    let decision = vocabulary.connectionDecision(
                        from: source.kind,
                        to: node.kind,
                        port: port,
                        existingConnectionCount: sourceIndex
                    )
                    switch decision {
                    case .allowed:
                        break
                    case .inletAtCapacity:
                        blockers.append(.init(
                            code: .arityExceeded,
                            path: "$.nodes[\(nodeIndex)].in.\(port)[\(sourceIndex)]",
                            detail: "single inlet has more than one authored source"
                        ))
                    case .unknownSourceKind, .unknownDestinationKind:
                        // The node-level unknown-kind blocker is more precise and
                        // the original edge remains serialized.
                        break
                    default:
                        blockers.append(.init(
                            code: .incompatibleConnection,
                            path: "$.nodes[\(nodeIndex)].in.\(port)[\(sourceIndex)]",
                            detail: "descriptor-derived connection rejection: \(decision)"
                        ))
                    }
                }
            }
        }
    }

    private func validateOutput(
        vocabulary: EngineVocabulary?,
        blockers: inout [PatchDocumentBlocker]
    ) {
        guard let output else {
            blockers.append(.init(
                code: .invalidOutput,
                path: "$.output",
                detail: "document must select exactly one output"
            ))
            return
        }
        guard let selected = nodes.first(where: { $0.id == output }) else {
            blockers.append(.init(
                code: .invalidOutput,
                path: "$.output",
                detail: "output '\(output)' is not an authored engine node"
            ))
            return
        }
        guard let vocabulary else { return }
        guard vocabulary.descriptor(for: selected.kind)?.outlet == nil else {
            blockers.append(.init(
                code: .invalidOutput,
                path: "$.output",
                detail: "selected output kind '\(selected.kind.rawValue)' is not an engine sink"
            ))
            return
        }
        let sinks = nodes.filter {
            vocabulary.descriptor(for: $0.kind)?.outlet == nil
        }
        if sinks.count != 1 || sinks[0].id != output {
            blockers.append(.init(
                code: .invalidOutput,
                path: "$.nodes",
                detail: "document must author one engine sink matching $.output"
            ))
        }
    }

    private func validateCycles(blockers: inout [PatchDocumentBlocker]) {
        let nodeByID = Dictionary(nodes.map { ($0.id, $0) }, uniquingKeysWith: { first, _ in first })
        var state: [String: Int] = [:]
        var stack: [String] = []
        var found = false

        func visit(_ id: String) {
            guard !found, let node = nodeByID[id] else { return }
            state[id] = 1
            stack.append(id)
            for port in node.inputs.keys.sorted() {
                for source in node.inputs[port] ?? [] where nodeByID[source] != nil {
                    if state[source] == 1 {
                        let start = stack.firstIndex(of: source) ?? 0
                        let cycle = Array(stack[start...]) + [source]
                        blockers.append(.init(
                            code: .cycle,
                            path: "$.nodes",
                            detail: cycle.joined(separator: " -> ")
                        ))
                        found = true
                        return
                    }
                    if state[source] == nil { visit(source) }
                    if found { return }
                }
            }
            _ = stack.popLast()
            state[id] = 2
        }

        for node in nodes where state[node.id] == nil {
            visit(node.id)
            if found { break }
        }
    }

    private func validateFiniteJSON(_ blockers: inout [PatchDocumentBlocker]) {
        func walk(_ value: JSONValue, path: String) {
            switch value {
            case .number(let number) where !number.isFinite:
                blockers.append(.init(
                    code: .nonFiniteNumber,
                    path: path,
                    detail: "JSON number must be finite"
                ))
            case .array(let values):
                for (index, value) in values.enumerated() {
                    walk(value, path: "\(path)[\(index)]")
                }
            case .object(let object):
                for key in object.keys.sorted() {
                    walk(object[key]!, path: "\(path).\(key)")
                }
            default:
                break
            }
        }

        walk(.object(sceneMetadata), path: "$.scene")
        walk(.object(unknownFields), path: "$.__unknown")
        walk(.object(vocabulary.unknownFields), path: "$.vocabulary.__unknown")
        walk(.object(transport.unknownFields), path: "$.transport.__unknown")
        walk(.object(presentation.unknownFields), path: "$.presentation.__unknown")
        for (index, node) in nodes.enumerated() {
            walk(.object(node.structuralParams), path: "$.nodes[\(index)].params")
            walk(.object(node.unknownFields), path: "$.nodes[\(index)].__unknown")
        }
        for (index, monitor) in monitors.enumerated() {
            walk(.object(monitor.state), path: "$.monitors[\(index)].state")
            walk(.object(monitor.unknownFields), path: "$.monitors[\(index)].__unknown")
        }
        for id in presentation.positions.keys.sorted() {
            if let position = presentation.positions[id] {
                walk(.object(position.unknownFields), path: "$.presentation.positions.\(id).__unknown")
            }
        }
    }
}

enum PatchDocumentV2Error: Error, Equatable, LocalizedError {
    case unsupportedVersion(Int)
    case expectedV1(Int)
    case missing(path: String)
    case type(path: String, expected: String)

    var errorDescription: String? {
        switch self {
        case .unsupportedVersion(let version):
            return "unsupported patch document version \(version); expected 2"
        case .expectedV1(let version):
            return "v1 migrator received document version \(version)"
        case .missing(let path):
            return "missing required patch field \(path)"
        case .type(let path, let expected):
            return "invalid patch field \(path); expected \(expected)"
        }
    }
}

// MARK: - Closed JSON shape helpers

private enum PatchV2JSON {
    /// V1 had no reserved knowledge of v2 field names. Preserve an unknown v1
    /// field even if its name now collides with a typed v2 key by nesting only
    /// those collisions in a migration envelope. If the source already used
    /// the envelope name, that original value is nested too.
    static func preserveLegacyUnknown(
        _ fields: [String: JSONValue],
        collidingWith reserved: Set<String>
    ) -> [String: JSONValue] {
        let envelope = "_legacy_v1_reserved_fields"
        var direct: [String: JSONValue] = [:]
        var collisions: [String: JSONValue] = [:]
        for key in fields.keys.sorted() {
            if reserved.contains(key) || key == envelope {
                collisions[key] = fields[key]
            } else {
                direct[key] = fields[key]
            }
        }
        if !collisions.isEmpty {
            direct[envelope] = .object(collisions)
        }
        return direct
    }

    static func take(
        _ key: String,
        from object: inout [String: JSONValue],
        path: String
    ) throws -> JSONValue {
        guard let value = object.removeValue(forKey: key) else {
            throw PatchDocumentV2Error.missing(path: "\(path).\(key)")
        }
        return value
    }

    static func object(_ value: JSONValue, path: String) throws -> [String: JSONValue] {
        guard case .object(let object) = value else {
            throw PatchDocumentV2Error.type(path: path, expected: "object")
        }
        return object
    }

    static func array(_ value: JSONValue, path: String) throws -> [JSONValue] {
        guard case .array(let array) = value else {
            throw PatchDocumentV2Error.type(path: path, expected: "array")
        }
        return array
    }

    static func string(_ value: JSONValue, path: String) throws -> String {
        guard case .string(let string) = value else {
            throw PatchDocumentV2Error.type(path: path, expected: "string")
        }
        return string
    }

    static func strings(_ value: JSONValue, path: String) throws -> [String] {
        let values = try array(value, path: path)
        return try values.enumerated().map {
            try string($0.element, path: "\(path)[\($0.offset)]")
        }
    }

    static func number(_ value: JSONValue, path: String) throws -> Double {
        guard case .number(let number) = value else {
            throw PatchDocumentV2Error.type(path: path, expected: "number")
        }
        return number
    }

    static func integer(_ value: JSONValue, path: String) throws -> Int {
        let number = try number(value, path: path)
        guard number.isFinite,
              number.rounded(.towardZero) == number,
              number >= Double(Int.min),
              number <= Double(Int.max)
        else {
            throw PatchDocumentV2Error.type(path: path, expected: "integer")
        }
        return Int(number)
    }

    static func bool(_ value: JSONValue, path: String) throws -> Bool {
        guard case .bool(let bool) = value else {
            throw PatchDocumentV2Error.type(path: path, expected: "boolean")
        }
        return bool
    }

    static func inputs(_ value: JSONValue, path: String) throws -> [String: [String]] {
        let object = try object(value, path: path)
        var inputs: [String: [String]] = [:]
        for port in object.keys.sorted() {
            inputs[port] = try strings(object[port]!, path: "\(path).\(port)")
        }
        return inputs
    }

    static func inputsValue(_ inputs: [String: [String]]) -> JSONValue {
        .object(inputs.mapValues { .array($0.map(JSONValue.string)) })
    }
}
