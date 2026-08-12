import Foundation

struct ModuleReference: Codable, Equatable, Hashable, Sendable {
    var id: String
    var version: Int
}

struct ModuleParameter: Codable, Equatable, Sendable {
    var name: String
    var defaultValue: Double

    enum CodingKeys: String, CodingKey {
        case name
        case defaultValue = "default"
    }
}

struct PatchGraphState {
    var nodes: [String: PatchNode]
    var order: [String]
    var pan: CGSize
}

struct ModuleDefinitionState {
    var reference: ModuleReference
    var title: String
    var inputNodeID: String
    var outputNodeID: String
    var parameters: [ModuleParameter]
    var graph: PatchGraphState
}

enum HierarchyLocation: Equatable {
    case scene
    case definition(ModuleReference)
}

struct HierarchyFrame: Equatable, Identifiable {
    let location: HierarchyLocation
    let title: String
    let instanceID: String?

    var id: String {
        switch location {
        case .scene: "scene"
        case .definition(let reference):
            "\(reference.id)@\(reference.version):\(instanceID ?? "")"
        }
    }
}

@MainActor
extension PatchModel {
    var hierarchyTitle: String {
        hierarchyPath.map(\.title).joined(separator: " / ")
    }

    var isAtScene: Bool { currentHierarchyLocation == .scene }

    var addableKinds: [NodeKind] {
        if isAtScene {
            return NodeKind.allCases.filter(\.appearsInAddMenu)
        }
        return [.allpass, .modalTail, .modalBlend, .modalmix]
    }

    func snapshotActiveGraph() {
        let graph = PatchGraphState(nodes: nodes, order: order, pan: pan)
        switch currentHierarchyLocation {
        case .scene:
            sceneGraph = graph
        case .definition(let reference):
            definitions[reference]?.graph = graph
        }
    }

    func activate(_ graph: PatchGraphState) {
        nodes = graph.nodes
        order = graph.order
        pan = graph.pan
        adoptCounter(from: graph.nodes.keys)
    }

    func openModule(_ nodeID: String) {
        guard var instance = nodes[nodeID], instance.kind.isModule,
              var reference = instance.module
        else { return }

        // Installed definitions are immutable. Entering one through an
        // editable canvas detaches exactly this instance first.
        if definitions[reference] == nil {
            guard let installed = moduleTemplate(for: reference) else {
                setStatus("module library unavailable · try again", isError: true)
                return
            }
            let detached = StandardModuleLibrary.detached(
                installed,
                instancePath: hierarchyPath.compactMap(\.instanceID) + [nodeID]
            )
            definitions[detached.reference] = detached
            reference = detached.reference
            instance.module = reference
            nodes[nodeID] = instance
            advanceAuthoredRevision()
            schedulePush()
        }

        snapshotActiveGraph()
        guard let definition = definitions[reference] else { return }
        hierarchyPath.append(.init(
            location: .definition(reference),
            title: "\(definition.title) · \(nodeID)",
            instanceID: nodeID
        ))
        currentHierarchyLocation = .definition(reference)
        activate(definition.graph)
    }

    func exitModule() {
        guard hierarchyPath.count > 1 else { return }
        snapshotActiveGraph()
        hierarchyPath.removeLast()
        let destination = hierarchyPath.last?.location ?? .scene
        currentHierarchyLocation = destination
        switch destination {
        case .scene:
            activate(sceneGraph)
        case .definition(let reference):
            if let definition = definitions[reference] {
                activate(definition.graph)
            }
        }
    }

    func resetHierarchy(to graph: PatchGraphState) {
        sceneGraph = graph
        definitions = [:]
        hierarchyPath = [.init(location: .scene, title: "Patch", instanceID: nil)]
        currentHierarchyLocation = .scene
        activate(graph)
    }
}

enum StandardModuleLibrary {
    static let phaser = ModuleReference(id: "tropical.modal.phaser", version: 1)
    static let allpass = ModuleReference(id: "tropical.modal.allpass1", version: 1)

    /// Clone an engine-served installed definition under an author-owned id.
    /// Swift supplies identity and presentation only; it never reconstructs
    /// the definition's DSP topology.
    static func detached(
        _ installed: ModuleDefinitionState,
        instancePath: [String]
    ) -> ModuleDefinitionState {
        let encoded = instancePath.map { "\($0.count)_\($0)" }.joined(separator: ".")
        let leaf = installed.reference.id.split(separator: ".").last.map(String.init)
            ?? "module"
        var definition = installed
        definition.reference = ModuleReference(
            id: "user.\(encoded).\(leaf)",
            version: 1
        )
        return definition
    }
}

enum ModuleLibraryDecodeError: Error, LocalizedError {
    case invalid(String)

    var errorDescription: String? {
        switch self {
        case .invalid(let path): "module library has invalid '\(path)'"
        }
    }
}

enum EngineModuleLibrary {
    static func decode(_ value: JSONValue) throws -> [ModuleReference: ModuleDefinitionState] {
        guard case .object(let root) = value,
              root["schema"]?.stringValue == "tropical_module_library",
              exactInt(root["schema_version"]) == 1,
              case .array(let rawDefinitions)? = root["definitions"]
        else { throw ModuleLibraryDecodeError.invalid("root") }

        var result: [ModuleReference: ModuleDefinitionState] = [:]
        for (definitionIndex, rawDefinition) in rawDefinitions.enumerated() {
            let path = "definitions[\(definitionIndex)]"
            guard case .object(let fields) = rawDefinition,
                  let id = fields["id"]?.stringValue,
                  let version = exactInt(fields["version"]),
                  let input = fields["input"]?.stringValue,
                  let output = fields["output"]?.stringValue,
                  case .array(let rawParameters)? = fields["parameters"],
                  case .array(let rawNodes)? = fields["nodes"]
            else { throw ModuleLibraryDecodeError.invalid(path) }
            let reference = ModuleReference(id: id, version: version)
            guard result[reference] == nil else {
                throw ModuleLibraryDecodeError.invalid("\(path).id/version")
            }
            let parameters = try rawParameters.enumerated().map { index, raw in
                guard case .object(let parameter) = raw,
                      let name = parameter["name"]?.stringValue,
                      let defaultValue = parameter["default"]?.doubleValue,
                      defaultValue.isFinite
                else {
                    throw ModuleLibraryDecodeError.invalid("\(path).parameters[\(index)]")
                }
                return ModuleParameter(name: name, defaultValue: defaultValue)
            }

            var nodes: [String: PatchNode] = [:]
            var order: [String] = []
            for (index, raw) in rawNodes.enumerated() {
                guard case .object(let nodeFields) = raw,
                      let nodeID = nodeFields["id"]?.stringValue,
                      let rawKind = nodeFields["kind"]?.stringValue
                else { throw ModuleLibraryDecodeError.invalid("\(path).nodes[\(index)]") }
                let module: ModuleReference?
                let kind: NodeKind
                if rawKind == "module" {
                    guard let definitionID = nodeFields["definition"]?.stringValue,
                          let definitionVersion = exactInt(nodeFields["definition_version"])
                    else {
                        throw ModuleLibraryDecodeError.invalid("\(path).nodes[\(index)].definition")
                    }
                    module = .init(id: definitionID, version: definitionVersion)
                    if definitionID == StandardModuleLibrary.allpass.id {
                        kind = .allpass
                    } else if definitionID == StandardModuleLibrary.phaser.id {
                        kind = .phaser
                    } else {
                        throw ModuleLibraryDecodeError.invalid(
                            "\(path).nodes[\(index)].definition"
                        )
                    }
                } else {
                    guard let decoded = NodeKind(rawValue: rawKind) else {
                        throw ModuleLibraryDecodeError.invalid("\(path).nodes[\(index)].kind")
                    }
                    module = nil
                    kind = decoded
                }
                let params = stringDoubleObject(nodeFields["params"])
                let inputs = try stringArrayObject(
                    nodeFields["in"],
                    path: "\(path).nodes[\(index)].in"
                )
                let bindings = stringStringObject(nodeFields["bindings"])
                var values: [String: Double] = [:]
                for knob in kind.spec.knobs {
                    values[knob.name] = params[knob.name] ?? knob.def
                }
                let column = Double(index)
                let node = PatchNode(
                    id: nodeID,
                    kind: kind,
                    position: CGPoint(
                        x: 44 + column * 184,
                        y: 116 + Double(index % 2) * 124
                    ),
                    values: values,
                    inputs: inputs,
                    hue: Double((index * 29 + 170) % 360),
                    module: module,
                    bindings: bindings
                )
                nodes[nodeID] = node
                order.append(nodeID)
            }
            guard nodes[input]?.kind == .moduleInput,
                  nodes[output]?.kind == .moduleOutput
            else { throw ModuleLibraryDecodeError.invalid("\(path).boundaries") }
            let title = reference == StandardModuleLibrary.phaser ? "Phaser" : "Allpass 1"
            result[reference] = .init(
                reference: reference,
                title: title,
                inputNodeID: input,
                outputNodeID: output,
                parameters: parameters,
                graph: .init(nodes: nodes, order: order, pan: .zero)
            )
        }
        return result
    }

    private static func exactInt(_ value: JSONValue?) -> Int? {
        guard let number = value?.doubleValue,
              number.isFinite,
              number.rounded(.towardZero) == number,
              number >= 0,
              number <= Double(Int.max)
        else { return nil }
        return Int(number)
    }

    private static func stringDoubleObject(_ value: JSONValue?) -> [String: Double] {
        guard case .object(let fields)? = value else { return [:] }
        return fields.reduce(into: [:]) { result, entry in
            if let number = entry.value.doubleValue, number.isFinite {
                result[entry.key] = number
            }
        }
    }

    private static func stringStringObject(_ value: JSONValue?) -> [String: String] {
        guard case .object(let fields)? = value else { return [:] }
        return fields.reduce(into: [:]) { result, entry in
            if let string = entry.value.stringValue { result[entry.key] = string }
        }
    }

    private static func stringArrayObject(
        _ value: JSONValue?,
        path: String
    ) throws -> [String: [String]] {
        guard case .object(let fields)? = value else { return [:] }
        var result: [String: [String]] = [:]
        for (port, rawSources) in fields {
            guard case .array(let sources) = rawSources else {
                throw ModuleLibraryDecodeError.invalid("\(path).\(port)")
            }
            result[port] = try sources.enumerated().map { index, source in
                guard let id = source.stringValue else {
                    throw ModuleLibraryDecodeError.invalid("\(path).\(port)[\(index)]")
                }
                return id
            }
        }
        return result
    }
}
