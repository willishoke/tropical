import Foundation

// MARK: - Engine-owned protocol identifiers

/// A node-kind identifier served by the engine. This is deliberately a string
/// value rather than an exhaustive enum: a newly served kind must decode without
/// a Swift semantic edit.
struct NodeKindID: RawRepresentable, Codable, Hashable, Sendable {
    let rawValue: String

    init?(rawValue: String) {
        guard !rawValue.isEmpty,
              rawValue == rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        else { return nil }
        self.rawValue = rawValue
    }

    init(validating rawValue: String) throws {
        try Self.validate(rawValue, label: "node kind")
        self.rawValue = rawValue
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let value = try container.decode(String.self)
        do {
            try self.init(validating: value)
        } catch {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: error.localizedDescription
            )
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(rawValue)
    }

    private static func validate(_ value: String, label: String) throws {
        guard !value.isEmpty,
              value == value.trimmingCharacters(in: .whitespacesAndNewlines)
        else {
            throw EngineVocabularyError.invalidIdentifier(label: label, value: value)
        }
    }
}

/// Connection colors are also engine-owned identifiers. Keeping them open
/// avoids recreating the engine's signal/modal/control enum in Swift.
struct PortColorID: RawRepresentable, Codable, Hashable, Sendable {
    let rawValue: String

    init?(rawValue: String) {
        guard !rawValue.isEmpty,
              rawValue == rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        else { return nil }
        self.rawValue = rawValue
    }

    init(validating rawValue: String) throws {
        guard !rawValue.isEmpty,
              rawValue == rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        else {
            throw EngineVocabularyError.invalidIdentifier(label: "port color", value: rawValue)
        }
        self.rawValue = rawValue
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let value = try container.decode(String.self)
        do {
            try self.init(validating: value)
        } catch {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: error.localizedDescription
            )
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(rawValue)
    }
}

/// The write discipline is descriptive response data. The client always uses
/// the unified `set_param` method and never dispatches on this identifier.
struct WriteDisciplineID: RawRepresentable, Codable, Hashable, Sendable {
    let rawValue: String

    init?(rawValue: String) {
        guard !rawValue.isEmpty,
              rawValue == rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        else { return nil }
        self.rawValue = rawValue
    }

    init(validating rawValue: String) throws {
        guard !rawValue.isEmpty,
              rawValue == rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        else {
            throw EngineVocabularyError.invalidIdentifier(label: "write discipline", value: rawValue)
        }
        self.rawValue = rawValue
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let value = try container.decode(String.self)
        do {
            try self.init(validating: value)
        } catch {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: error.localizedDescription
            )
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(rawValue)
    }
}

// MARK: - Served descriptors

struct PortDescriptor: Decodable, Equatable, Sendable {
    let name: String
    let accepts: [PortColorID]
    let multi: Bool
    let defaultValue: Double?
    let discipline: WriteDisciplineID?
    let min: Double?
    let max: Double?
    let logarithmic: Bool?
    let unit: String?
    let owner: String?

    var isInlet: Bool { !accepts.isEmpty }
    var isParameter: Bool { defaultValue != nil }

    private enum CodingKeys: String, CodingKey {
        case name
        case accepts
        case multi
        case defaultValue = "default"
        case discipline
        case min
        case max
        case logarithmic = "log"
        case unit
        case owner
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        name = try container.decode(String.self, forKey: .name)

        if container.contains(.accepts) {
            accepts = try container.decode([PortColorID].self, forKey: .accepts)
            multi = try container.decode(Bool.self, forKey: .multi)
        } else {
            guard !container.contains(.multi) else {
                throw DecodingError.dataCorruptedError(
                    forKey: .multi,
                    in: container,
                    debugDescription: "multi is valid only when accepts is present"
                )
            }
            accepts = []
            multi = false
        }

        let parameterKeys: [CodingKeys] = [
            .defaultValue, .discipline, .min, .max, .logarithmic, .unit,
        ]
        if parameterKeys.contains(where: container.contains) {
            defaultValue = try container.decode(Double.self, forKey: .defaultValue)
            discipline = try container.decode(WriteDisciplineID.self, forKey: .discipline)
            min = try container.decode(Double.self, forKey: .min)
            max = try container.decode(Double.self, forKey: .max)
            logarithmic = try container.decode(Bool.self, forKey: .logarithmic)
            unit = try container.decode(String.self, forKey: .unit)
        } else {
            defaultValue = nil
            discipline = nil
            min = nil
            max = nil
            logarithmic = nil
            unit = nil
        }
        owner = try container.decodeIfPresent(String.self, forKey: .owner)
    }
}

struct NodeDescriptor: Decodable, Equatable, Sendable {
    let kind: NodeKindID
    let outlet: PortColorID?
    let ports: [PortDescriptor]

    func port(named name: String) -> PortDescriptor? {
        ports.first { $0.name == name }
    }

    var inlets: [PortDescriptor] { ports.filter(\.isInlet) }
    var parameters: [PortDescriptor] { ports.filter(\.isParameter) }

    private enum CodingKeys: String, CodingKey {
        case kind
        case outlet
        case ports
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        kind = try container.decode(NodeKindID.self, forKey: .kind)
        guard container.contains(.outlet) else {
            throw DecodingError.keyNotFound(
                CodingKeys.outlet,
                .init(codingPath: decoder.codingPath, debugDescription: "outlet is required; use null for a sink")
            )
        }
        outlet = try container.decodeIfPresent(PortColorID.self, forKey: .outlet)
        ports = try container.decode([PortDescriptor].self, forKey: .ports)
    }
}

struct EngineVocabulary: Decodable, Equatable, Sendable {
    static let supportedSchemaID = "tropical_vocabulary"
    static let supportedSchemaVersion = 1

    let schemaID: String
    let schemaVersion: Int
    let fingerprint: String
    let rule: String
    let colors: [PortColorID]
    let kinds: [NodeDescriptor]

    private enum CodingKeys: String, CodingKey {
        case schemaID = "schema"
        case schemaVersion = "schema_version"
        case fingerprint
        case rule
        case colors
        case kinds
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        schemaID = try container.decode(String.self, forKey: .schemaID)
        schemaVersion = try container.decode(Int.self, forKey: .schemaVersion)
        fingerprint = try container.decode(String.self, forKey: .fingerprint)
        rule = try container.decode(String.self, forKey: .rule)
        colors = try container.decode([PortColorID].self, forKey: .colors)
        kinds = try container.decode([NodeDescriptor].self, forKey: .kinds)
        try validate(withholding: [])
    }

    /// Decode and validate one complete response. Withheld kind IDs come from
    /// the caller's compatibility policy; this type contains no hard-coded
    /// engine kind list.
    static func decode(
        from data: Data,
        withholding withheldKinds: Set<NodeKindID> = []
    ) throws -> EngineVocabulary {
        let vocabulary = try JSONDecoder().decode(EngineVocabulary.self, from: data)
        try vocabulary.validate(withholding: withheldKinds)
        return vocabulary
    }

    func descriptor(for kind: NodeKindID) -> NodeDescriptor? {
        kinds.first { $0.kind == kind }
    }

    func validate(withholding withheldKinds: Set<NodeKindID>) throws {
        guard schemaID == Self.supportedSchemaID else {
            throw EngineVocabularyError.unsupportedSchemaID(schemaID)
        }
        guard schemaVersion == Self.supportedSchemaVersion else {
            throw EngineVocabularyError.unsupportedSchemaVersion(schemaVersion)
        }
        guard Self.isValidFingerprint(fingerprint) else {
            throw EngineVocabularyError.invalidFingerprint(fingerprint)
        }
        guard !rule.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw EngineVocabularyError.emptyRule
        }
        guard !colors.isEmpty else {
            throw EngineVocabularyError.emptyColors
        }
        guard !kinds.isEmpty else {
            throw EngineVocabularyError.emptyKinds
        }

        var seenColors = Set<PortColorID>()
        for color in colors where !seenColors.insert(color).inserted {
            throw EngineVocabularyError.duplicateColor(color)
        }

        var seenKinds = Set<NodeKindID>()
        for node in kinds {
            guard seenKinds.insert(node.kind).inserted else {
                throw EngineVocabularyError.duplicateKind(node.kind)
            }
            if withheldKinds.contains(node.kind) {
                throw EngineVocabularyError.withheldKindServed(node.kind)
            }
            if let outlet = node.outlet, !seenColors.contains(outlet) {
                throw EngineVocabularyError.undeclaredOutletColor(kind: node.kind, color: outlet)
            }
            try validate(node, declaredColors: seenColors)
        }
    }

    private func validate(
        _ node: NodeDescriptor,
        declaredColors: Set<PortColorID>
    ) throws {
        var seenPorts = Set<String>()
        for port in node.ports {
            guard !port.name.isEmpty,
                  port.name == port.name.trimmingCharacters(in: .whitespacesAndNewlines)
            else {
                throw EngineVocabularyError.invalidPortName(kind: node.kind, name: port.name)
            }
            guard seenPorts.insert(port.name).inserted else {
                throw EngineVocabularyError.duplicatePort(kind: node.kind, port: port.name)
            }

            var seenAccepts = Set<PortColorID>()
            for color in port.accepts {
                guard declaredColors.contains(color) else {
                    throw EngineVocabularyError.undeclaredAcceptedColor(
                        kind: node.kind,
                        port: port.name,
                        color: color
                    )
                }
                guard seenAccepts.insert(color).inserted else {
                    throw EngineVocabularyError.duplicateAcceptedColor(
                        kind: node.kind,
                        port: port.name,
                        color: color
                    )
                }
            }
            if port.multi && !port.isInlet {
                throw EngineVocabularyError.multiOnNonInlet(kind: node.kind, port: port.name)
            }

            if port.isParameter {
                guard let defaultValue = port.defaultValue,
                      let min = port.min,
                      let max = port.max,
                      let logarithmic = port.logarithmic,
                      defaultValue.isFinite,
                      min.isFinite,
                      max.isFinite,
                      min < max,
                      (min...max).contains(defaultValue)
                else {
                    throw EngineVocabularyError.invalidParameterRange(kind: node.kind, port: port.name)
                }
                if logarithmic && min <= 0 {
                    throw EngineVocabularyError.nonPositiveLogMinimum(kind: node.kind, port: port.name)
                }
            } else if port.owner != nil {
                throw EngineVocabularyError.ownerOnNonParameter(kind: node.kind, port: port.name)
            } else if !port.isInlet {
                throw EngineVocabularyError.inertPort(kind: node.kind, port: port.name)
            }
        }

        for port in node.ports {
            guard let owner = port.owner else { continue }
            guard owner != port.name,
                  let ownerPort = node.port(named: owner),
                  ownerPort.isInlet
            else {
                throw EngineVocabularyError.invalidOwner(
                    kind: node.kind,
                    port: port.name,
                    owner: owner
                )
            }
        }
    }

    private static func isValidFingerprint(_ value: String) -> Bool {
        let prefix = "fnv1a64:"
        guard value.hasPrefix(prefix) else { return false }
        let digest = value.dropFirst(prefix.count)
        return digest.count == 16 && digest.allSatisfy {
            ("0"..."9").contains(String($0))
                || ("a"..."f").contains(String($0))
        }
    }
}

enum EngineVocabularyError: Error, Equatable, LocalizedError {
    case invalidIdentifier(label: String, value: String)
    case unsupportedSchemaID(String)
    case unsupportedSchemaVersion(Int)
    case invalidFingerprint(String)
    case emptyRule
    case emptyColors
    case emptyKinds
    case duplicateColor(PortColorID)
    case duplicateKind(NodeKindID)
    case withheldKindServed(NodeKindID)
    case undeclaredOutletColor(kind: NodeKindID, color: PortColorID)
    case invalidPortName(kind: NodeKindID, name: String)
    case duplicatePort(kind: NodeKindID, port: String)
    case undeclaredAcceptedColor(kind: NodeKindID, port: String, color: PortColorID)
    case duplicateAcceptedColor(kind: NodeKindID, port: String, color: PortColorID)
    case multiOnNonInlet(kind: NodeKindID, port: String)
    case invalidParameterRange(kind: NodeKindID, port: String)
    case nonPositiveLogMinimum(kind: NodeKindID, port: String)
    case ownerOnNonParameter(kind: NodeKindID, port: String)
    case invalidOwner(kind: NodeKindID, port: String, owner: String)
    case inertPort(kind: NodeKindID, port: String)

    var errorDescription: String? {
        switch self {
        case .invalidIdentifier(let label, let value):
            return "invalid \(label) identifier: '\(value)'"
        case .unsupportedSchemaID(let value):
            return "unsupported vocabulary schema: '\(value)'"
        case .unsupportedSchemaVersion(let value):
            return "unsupported vocabulary schema version: \(value)"
        case .invalidFingerprint(let value):
            return "invalid vocabulary fingerprint: '\(value)'"
        case .emptyRule:
            return "vocabulary connection rule is empty"
        case .emptyColors:
            return "vocabulary declares no connection colors"
        case .emptyKinds:
            return "vocabulary declares no served kinds"
        case .duplicateColor(let color):
            return "duplicate vocabulary color: '\(color.rawValue)'"
        case .duplicateKind(let kind):
            return "duplicate served kind: '\(kind.rawValue)'"
        case .withheldKindServed(let kind):
            return "withheld kind was served: '\(kind.rawValue)'"
        case .undeclaredOutletColor(let kind, let color):
            return "kind '\(kind.rawValue)' uses undeclared outlet color '\(color.rawValue)'"
        case .invalidPortName(let kind, let name):
            return "kind '\(kind.rawValue)' has invalid port name '\(name)'"
        case .duplicatePort(let kind, let port):
            return "kind '\(kind.rawValue)' has duplicate port '\(port)'"
        case .undeclaredAcceptedColor(let kind, let port, let color):
            return "kind '\(kind.rawValue)' port '\(port)' accepts undeclared color '\(color.rawValue)'"
        case .duplicateAcceptedColor(let kind, let port, let color):
            return "kind '\(kind.rawValue)' port '\(port)' repeats accepted color '\(color.rawValue)'"
        case .multiOnNonInlet(let kind, let port):
            return "kind '\(kind.rawValue)' non-inlet port '\(port)' is multi"
        case .invalidParameterRange(let kind, let port):
            return "kind '\(kind.rawValue)' parameter '\(port)' has an invalid default/range"
        case .nonPositiveLogMinimum(let kind, let port):
            return "kind '\(kind.rawValue)' logarithmic parameter '\(port)' has a non-positive minimum"
        case .ownerOnNonParameter(let kind, let port):
            return "kind '\(kind.rawValue)' non-parameter port '\(port)' declares an owner"
        case .invalidOwner(let kind, let port, let owner):
            return "kind '\(kind.rawValue)' parameter '\(port)' has invalid owner '\(owner)'"
        case .inertPort(let kind, let port):
            return "kind '\(kind.rawValue)' port '\(port)' is neither an inlet nor a parameter"
        }
    }
}

// MARK: - Descriptor-derived connection checks

enum ConnectionTypeDecision: Equatable, Sendable {
    case allowed
    case invalidExistingConnectionCount(Int)
    case unknownSourceKind(NodeKindID)
    case unknownDestinationKind(NodeKindID)
    case sourceHasNoOutlet(NodeKindID)
    case unknownDestinationPort(kind: NodeKindID, port: String)
    case destinationPortIsNotInlet(kind: NodeKindID, port: String)
    case incompatibleColor(outlet: PortColorID, accepts: [PortColorID])
    case inletAtCapacity(kind: NodeKindID, port: String)
}

extension EngineVocabulary {
    /// Type and arity only. Node identity, duplicate-edge, and cycle checks
    /// remain authored-graph concerns; every fact used here comes from the
    /// decoded descriptors.
    func connectionDecision(
        from sourceKind: NodeKindID,
        to destinationKind: NodeKindID,
        port portName: String,
        existingConnectionCount: Int
    ) -> ConnectionTypeDecision {
        guard existingConnectionCount >= 0 else {
            return .invalidExistingConnectionCount(existingConnectionCount)
        }
        guard let source = descriptor(for: sourceKind) else {
            return .unknownSourceKind(sourceKind)
        }
        guard let destination = descriptor(for: destinationKind) else {
            return .unknownDestinationKind(destinationKind)
        }
        guard let outlet = source.outlet else {
            return .sourceHasNoOutlet(sourceKind)
        }
        guard let port = destination.port(named: portName) else {
            return .unknownDestinationPort(kind: destinationKind, port: portName)
        }
        guard port.isInlet else {
            return .destinationPortIsNotInlet(kind: destinationKind, port: portName)
        }
        guard port.accepts.contains(outlet) else {
            return .incompatibleColor(outlet: outlet, accepts: port.accepts)
        }
        guard port.multi || existingConnectionCount == 0 else {
            return .inletAtCapacity(kind: destinationKind, port: portName)
        }
        return .allowed
    }

    func canConnect(
        from sourceKind: NodeKindID,
        to destinationKind: NodeKindID,
        port portName: String,
        existingConnectionCount: Int
    ) -> Bool {
        connectionDecision(
            from: sourceKind,
            to: destinationKind,
            port: portName,
            existingConnectionCount: existingConnectionCount
        ) == .allowed
    }
}

// MARK: - Client-only presentation namespace

/// Presentation-only overrides. Absence means the UI uses its generic visual
/// treatment; it never supplies ports, ranges, defaults, or connection rules.
struct NodeThemeOverride: Equatable, Sendable {
    var displayName: String?
    var accentRGB: UInt32?
}

struct EngineNodeThemeCatalog: Sendable {
    private let overrides: [NodeKindID: NodeThemeOverride]

    init(overrides: [NodeKindID: NodeThemeOverride] = [:]) {
        self.overrides = overrides
    }

    func override(for kind: NodeKindID) -> NodeThemeOverride? {
        overrides[kind]
    }
}

/// Client monitor IDs cannot collide with served engine kinds because they use
/// a distinct type and a mandatory `client.monitor.` namespace.
struct ClientMonitorKindID: RawRepresentable, Codable, Hashable, Sendable {
    static let namespace = "client.monitor."

    let rawValue: String

    init?(rawValue: String) {
        let localName = rawValue.dropFirst(Self.namespace.count)
        guard rawValue.hasPrefix(Self.namespace),
              !localName.isEmpty,
              rawValue == rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        else { return nil }
        self.rawValue = rawValue
    }

    init(validating rawValue: String) throws {
        let localName = rawValue.dropFirst(Self.namespace.count)
        guard rawValue.hasPrefix(Self.namespace),
              !localName.isEmpty,
              rawValue == rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        else {
            throw ClientMonitorCatalogError.invalidMonitorID(rawValue)
        }
        self.rawValue = rawValue
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let value = try container.decode(String.self)
        do {
            try self.init(validating: value)
        } catch {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: error.localizedDescription
            )
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(rawValue)
    }
}

struct ClientMonitorDescriptor: Equatable, Sendable {
    let kind: ClientMonitorKindID
    let displayName: String
    let theme: NodeThemeOverride?
}

struct ClientMonitorCatalog: Sendable {
    let monitors: [ClientMonitorDescriptor]

    init(monitors: [ClientMonitorDescriptor]) throws {
        var seen = Set<ClientMonitorKindID>()
        for monitor in monitors {
            guard seen.insert(monitor.kind).inserted else {
                throw ClientMonitorCatalogError.duplicateMonitor(monitor.kind)
            }
        }
        self.monitors = monitors
    }

    func descriptor(for kind: ClientMonitorKindID) -> ClientMonitorDescriptor? {
        monitors.first { $0.kind == kind }
    }
}

enum ClientMonitorCatalogError: Error, Equatable, LocalizedError {
    case invalidMonitorID(String)
    case duplicateMonitor(ClientMonitorKindID)

    var errorDescription: String? {
        switch self {
        case .invalidMonitorID(let value):
            return "client monitor ID must begin with '\(ClientMonitorKindID.namespace)': '\(value)'"
        case .duplicateMonitor(let kind):
            return "duplicate client monitor: '\(kind.rawValue)'"
        }
    }
}
