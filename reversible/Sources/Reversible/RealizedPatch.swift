import Foundation

/// Identity of an immutable runtime program and one of its control snapshots.
/// Program identity dominates ordering; live writes may advance only the
/// control component of the currently adopted program.
struct EngineGeneration: Equatable, Comparable, Sendable {
    let programVersion: UInt64
    let controlVersion: UInt64

    static func < (lhs: EngineGeneration, rhs: EngineGeneration) -> Bool {
        if lhs.programVersion != rhs.programVersion {
            return lhs.programVersion < rhs.programVersion
        }
        return lhs.controlVersion < rhs.controlVersion
    }
}

enum RealizedNodeStatus: String, Equatable, Sendable {
    case active
    case excluded
}

struct RealizedNode: Equatable, Sendable {
    let id: String
    let kind: NodeKindID
    let status: RealizedNodeStatus
}

enum RealizedInletState: Equatable, Sendable {
    case wired(sources: [String])
    case normalled
}

struct RealizedInlet: Equatable, Sendable {
    let nodeID: String
    let port: String
    let state: RealizedInletState
}

struct RealizedLiveParameter: Equatable, Sendable {
    let name: String
    let value: Double
    let discipline: WriteDisciplineID
}

enum RealizedParameterStatus: Equatable, Sendable {
    case live(discipline: WriteDisciplineID)
    case structural
    case inactive
}

struct ScopeTapBinding: Equatable, Sendable {
    let name: String
    let instance: String
    let output: String
    let slot: String
}

struct RealizedSourceLocation: Equatable, Sendable {
    let expandedID: String
    let definitionID: String
    let definitionVersion: Int
    let localID: String
    let instancePath: [String]
}

/// Complete realized truth published by one successful load_patch_graph call.
/// Nothing in this value is inferred from the authored graph or a follow-up
/// diagnostic request.
struct RealizedPatch: Equatable, Sendable {
    let vocabularyFingerprint: String
    let generation: EngineGeneration
    let nodes: [RealizedNode]
    let inlets: [RealizedInlet]
    let liveParameters: [RealizedLiveParameter]
    let taps: [ScopeTapBinding]
    var sourceMap: [RealizedSourceLocation] = []

    func node(id: String) -> RealizedNode? {
        nodes.first { $0.id == id }
    }

    func inlet(nodeID: String, port: String) -> RealizedInlet? {
        inlets.first { $0.nodeID == nodeID && $0.port == port }
    }

    func liveParameter(named name: String) -> RealizedLiveParameter? {
        liveParameters.first { $0.name == name }
    }

    func parameterStatus(nodeID: String, port: String) -> RealizedParameterStatus {
        guard node(id: nodeID)?.status == .active else { return .inactive }
        guard let parameter = liveParameter(named: "\(nodeID).\(port)") else {
            return .structural
        }
        return .live(discipline: parameter.discipline)
    }

    func tap(named name: String) -> ScopeTapBinding? {
        taps.first { $0.name == name }
    }

    func nodes(atInstancePath path: [String], localID: String?) -> [RealizedNode] {
        sourceMap.compactMap { location in
            guard location.instancePath == path,
                  localID.map({ location.localID == $0 }) ?? true
            else { return nil }
            return node(id: location.expandedID)
        }
    }
}

enum PatchCompileResponse: Equatable, Sendable {
    case published(RealizedPatch)
    case superseded

    /// Decode the complete handshake and, when supplied, require its tap table
    /// to be exactly the stable projection requested in the authored graph.
    static func decode(
        _ value: JSONValue,
        expectedTapNames: [String]? = nil,
        expectedVocabularyFingerprint: String? = nil
    ) throws -> PatchCompileResponse {
        guard case .object(let object) = value else {
            throw RealizedPatchDecodeError.invalidField("response")
        }
        if case .bool(true)? = object["superseded"] { return .superseded }
        guard case .bool(true)? = object["ok"] else {
            throw object["ok"] == nil
                ? RealizedPatchDecodeError.missingField("ok")
                : RealizedPatchDecodeError.invalidField("ok")
        }

        let fingerprint = try requiredString(
            object, "vocabulary_fingerprint", path: "vocabulary_fingerprint"
        )
        if let expectedVocabularyFingerprint,
           fingerprint != expectedVocabularyFingerprint {
            throw RealizedPatchDecodeError.vocabularyMismatch(
                expected: expectedVocabularyFingerprint,
                actual: fingerprint
            )
        }
        let generation = EngineGeneration(
            programVersion: try requiredExactUInt64(object, "program_version"),
            controlVersion: try requiredExactUInt64(object, "control_version")
        )
        guard generation.programVersion > 0, generation.controlVersion > 0 else {
            throw RealizedPatchDecodeError.invalidField("generation")
        }

        let nodeValues = try requiredArray(object, "nodes")
        var nodes: [RealizedNode] = []
        var nodeIDs = Set<String>()
        for (index, raw) in nodeValues.enumerated() {
            guard case .object(let fields) = raw else {
                throw RealizedPatchDecodeError.invalidField("nodes[\(index)]")
            }
            let id = try requiredString(fields, "id", path: "nodes[\(index)].id")
            let rawKind = try requiredString(
                fields, "kind", path: "nodes[\(index)].kind"
            )
            let rawStatus = try requiredString(
                fields, "status", path: "nodes[\(index)].status"
            )
            guard nodeIDs.insert(id).inserted else {
                throw RealizedPatchDecodeError.duplicate("node id '\(id)'")
            }
            guard let kind = NodeKindID(rawValue: rawKind),
                  let status = RealizedNodeStatus(rawValue: rawStatus)
            else { throw RealizedPatchDecodeError.invalidField("nodes[\(index)]") }
            nodes.append(.init(id: id, kind: kind, status: status))
        }

        let inletValues = try requiredArray(object, "inputs")
        var inlets: [RealizedInlet] = []
        var inletKeys = Set<String>()
        for (index, raw) in inletValues.enumerated() {
            guard case .object(let fields) = raw else {
                throw RealizedPatchDecodeError.invalidField("inputs[\(index)]")
            }
            let nodeID = try requiredString(
                fields, "node", path: "inputs[\(index)].node"
            )
            let port = try requiredString(
                fields, "port", path: "inputs[\(index)].port"
            )
            let rawState = try requiredString(
                fields, "state", path: "inputs[\(index)].state"
            )
            guard nodeIDs.contains(nodeID) else {
                throw RealizedPatchDecodeError.invalidField("inputs[\(index)].node")
            }
            let key = "\(nodeID)\u{0}\(port)"
            guard inletKeys.insert(key).inserted else {
                throw RealizedPatchDecodeError.duplicate("inlet '\(nodeID).\(port)'")
            }
            let state: RealizedInletState
            switch rawState {
            case "normalled":
                guard fields["sources"] == nil else {
                    throw RealizedPatchDecodeError.invalidField(
                        "inputs[\(index)].sources"
                    )
                }
                state = .normalled
            case "wired":
                let sourceValues = try requiredArray(fields, "sources")
                var seenSources = Set<String>()
                let sources = try sourceValues.enumerated().map { sourceIndex, source in
                    guard let id = source.stringValue, nodeIDs.contains(id),
                          seenSources.insert(id).inserted
                    else {
                        throw RealizedPatchDecodeError.invalidField(
                            "inputs[\(index)].sources[\(sourceIndex)]"
                        )
                    }
                    return id
                }
                guard !sources.isEmpty else {
                    throw RealizedPatchDecodeError.invalidField(
                        "inputs[\(index)].sources"
                    )
                }
                state = .wired(sources: sources)
            default:
                throw RealizedPatchDecodeError.invalidField("inputs[\(index)].state")
            }
            inlets.append(.init(nodeID: nodeID, port: port, state: state))
        }

        let parameterValues = try requiredArray(object, "params")
        var parameters: [RealizedLiveParameter] = []
        var parameterNames = Set<String>()
        for (index, raw) in parameterValues.enumerated() {
            guard case .object(let fields) = raw else {
                throw RealizedPatchDecodeError.invalidField("params[\(index)]")
            }
            let name = try requiredString(
                fields, "name", path: "params[\(index)].name"
            )
            let rawDiscipline = try requiredString(
                fields, "discipline", path: "params[\(index)].discipline"
            )
            guard parameterNames.insert(name).inserted else {
                throw RealizedPatchDecodeError.duplicate("parameter '\(name)'")
            }
            guard let value = fields["value"]?.doubleValue, value.isFinite,
                  let discipline = WriteDisciplineID(rawValue: rawDiscipline)
            else { throw RealizedPatchDecodeError.invalidField("params[\(index)]") }
            parameters.append(.init(name: name, value: value, discipline: discipline))
        }

        let tapValues = try requiredArray(object, "taps")
        var taps: [ScopeTapBinding] = []
        var tapNames = Set<String>()
        var tapSlots = Set<String>()
        for (index, raw) in tapValues.enumerated() {
            guard case .object(let fields) = raw else {
                throw RealizedPatchDecodeError.invalidField("taps[\(index)]")
            }
            let name = try requiredString(
                fields, "name", path: "taps[\(index)].name"
            )
            let instance = try requiredString(
                fields, "instance", path: "taps[\(index)].instance"
            )
            let output = try requiredString(
                fields, "output", path: "taps[\(index)].output"
            )
            let slot = try requiredString(
                fields, "slot", path: "taps[\(index)].slot"
            )
            guard name == "out" || nodeIDs.contains(name),
                  slot == "\(instance).\(output)"
            else { throw RealizedPatchDecodeError.invalidField("taps[\(index)]") }
            guard tapNames.insert(name).inserted else {
                throw RealizedPatchDecodeError.duplicate("tap name '\(name)'")
            }
            guard tapSlots.insert(slot).inserted else {
                throw RealizedPatchDecodeError.duplicate("tap slot '\(slot)'")
            }
            taps.append(.init(name: name, instance: instance, output: output, slot: slot))
        }
        if let expectedTapNames, taps.map(\.name) != expectedTapNames {
            throw RealizedPatchDecodeError.invalidField("taps")
        }


        var sourceMap: [RealizedSourceLocation] = []
        if case .array(let entries)? = object["source_map"] {
            for (index, raw) in entries.enumerated() {
                guard case .object(let fields) = raw else {
                    throw RealizedPatchDecodeError.invalidField("source_map[\(index)]")
                }
                let expanded = try requiredString(
                    fields, "expanded", path: "source_map[\(index)].expanded"
                )
                let definition = try requiredString(
                    fields, "definition", path: "source_map[\(index)].definition"
                )
                let local = try requiredString(
                    fields, "local", path: "source_map[\(index)].local"
                )
                guard let versionValue = fields["definition_version"],
                      let version64 = JSONExact.uint64(versionValue),
                      let version = Int(exactly: version64),
                      case .array(let pathValues)? = fields["path"]
                else {
                    throw RealizedPatchDecodeError.invalidField("source_map[\(index)]")
                }
                let path = try pathValues.enumerated().map { pathIndex, value in
                    guard let segment = value.stringValue, !segment.isEmpty else {
                        throw RealizedPatchDecodeError.invalidField(
                            "source_map[\(index)].path[\(pathIndex)]"
                        )
                    }
                    return segment
                }
                sourceMap.append(.init(
                    expandedID: expanded,
                    definitionID: definition,
                    definitionVersion: version,
                    localID: local,
                    instancePath: path
                ))
            }
        }

        return .published(.init(
            vocabularyFingerprint: fingerprint,
            generation: generation,
            nodes: nodes,
            inlets: inlets,
            liveParameters: parameters,
            taps: taps,
            sourceMap: sourceMap
        ))
    }
}

enum RealizedPatchDecodeError: Error, Equatable, LocalizedError, Sendable {
    case missingField(String)
    case invalidField(String)
    case duplicate(String)
    case vocabularyMismatch(expected: String, actual: String)

    var errorDescription: String? {
        switch self {
        case .missingField(let path): return "compile handshake missing '\(path)'"
        case .invalidField(let path): return "compile handshake has invalid '\(path)'"
        case .duplicate(let fact): return "compile handshake repeats \(fact)"
        case .vocabularyMismatch(let expected, let actual):
            return "compile handshake vocabulary '\(actual)' does not match adopted vocabulary '\(expected)'"
        }
    }
}

private func requiredString(
    _ object: [String: JSONValue], _ key: String, path: String
) throws -> String {
    guard let value = object[key] else {
        throw RealizedPatchDecodeError.missingField(path)
    }
    guard let string = value.stringValue,
          !string.isEmpty,
          string == string.trimmingCharacters(in: .whitespacesAndNewlines)
    else { throw RealizedPatchDecodeError.invalidField(path) }
    return string
}

private func requiredArray(
    _ object: [String: JSONValue], _ key: String
) throws -> [JSONValue] {
    guard let value = object[key] else {
        throw RealizedPatchDecodeError.missingField(key)
    }
    guard case .array(let array) = value else {
        throw RealizedPatchDecodeError.invalidField(key)
    }
    return array
}

private func requiredExactUInt64(
    _ object: [String: JSONValue], _ key: String
) throws -> UInt64 {
    guard let value = object[key] else {
        throw RealizedPatchDecodeError.missingField(key)
    }
    guard let exact = JSONExact.uint64(value) else {
        throw RealizedPatchDecodeError.invalidField(key)
    }
    return exact
}

enum CompileAdoption: Equatable, Sendable {
    case adopted(RealizedPatch)
    case superseded(activeGeneration: EngineGeneration?)
    case stale(candidate: EngineGeneration, active: EngineGeneration)
}

/// The engine-side generation guard. Failed, superseded, and late responses
/// never replace the last complete realized handshake.
struct RealizedPatchStore: Sendable {
    private(set) var current: RealizedPatch?

    mutating func adopt(_ response: PatchCompileResponse) -> CompileAdoption {
        switch response {
        case .superseded:
            return .superseded(activeGeneration: current?.generation)
        case .published(let candidate):
            if let current, candidate.generation <= current.generation {
                return .stale(
                    candidate: candidate.generation,
                    active: current.generation
                )
            }
            current = candidate
            return .adopted(candidate)
        }
    }
}

/// One authored graph and the complete runtime truth published for it.
struct RealizedPatchSnapshot: Equatable, Sendable {
    let authoredRevision: Int
    let loweringRevision: Int
    let graph: JSONValue
    let patch: RealizedPatch

    var generation: EngineGeneration { patch.generation }
}

/// Pure state transition used by PatchModel and its tests. Compile realization
/// is immutable; observations may only advance the control identity of that
/// same program.
struct PatchTruthStore: Equatable, Sendable {
    private(set) var realized: RealizedPatchSnapshot?
    private(set) var runningGeneration: EngineGeneration?

    mutating func adoptCompile(
        _ adoption: CompileAdoption,
        authoredRevision: Int,
        loweringRevision: Int,
        graph: JSONValue
    ) -> Bool {
        guard case .adopted(let patch) = adoption else { return false }
        realized = .init(
            authoredRevision: authoredRevision,
            loweringRevision: loweringRevision,
            graph: graph,
            patch: patch
        )
        runningGeneration = patch.generation
        return true
    }

    mutating func retagCurrent(
        authoredRevision: Int,
        loweringRevision: Int
    ) {
        guard let realized else { return }
        self.realized = .init(
            authoredRevision: authoredRevision,
            loweringRevision: loweringRevision,
            graph: realized.graph,
            patch: realized.patch
        )
    }

    mutating func adoptObservation(_ generation: EngineGeneration) -> Bool {
        guard let realized,
              generation.programVersion == realized.generation.programVersion,
              generation.controlVersion >= realized.generation.controlVersion,
              runningGeneration.map({ generation >= $0 }) ?? true
        else { return false }
        runningGeneration = generation
        return true
    }

    mutating func reset() {
        realized = nil
        runningGeneration = nil
    }
}
