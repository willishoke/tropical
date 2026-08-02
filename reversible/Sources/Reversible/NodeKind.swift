import SwiftUI

struct KnobSpec {
    let name: String
    let min: Double
    let max: Double
    let def: Double
    var log = false
    var unit = ""
}

struct NodeSpec {
    let title: String
    let accent: Color
    /// Inlet names are served by the engine for engine nodes. A monitor's
    /// ports are client-owned and live in the separate monitor namespace.
    let inlets: [String]
    let outlets: [String]
    let knobs: [KnobSpec]
    let multiInlets: Set<String>
    var fixed = false
    var monitor = false
    var unavailable = false
    var gridOverride: (w: Int, h: Int)? = nil

    var gridSize: (w: Int, h: Int) {
        if let o = gridOverride { return o }
        let width = Double(knobs.count) * 54
            + Double(max(0, knobs.count - 1)) * 8 + 18
        return (max(4, Int(ceil(width / Grid.unit))), knobs.isEmpty ? 3 : 6)
    }
}

enum Grid {
    static let unit: Double = 22

    static func snap(_ p: CGPoint) -> CGPoint {
        CGPoint(
            x: (p.x / unit).rounded() * unit,
            y: (p.y / unit).rounded() * unit)
    }
}

/// A UI value backed either by an engine-served node descriptor or by a
/// client monitor ID. This replaces the former exhaustive semantic enum: new
/// engine kinds flow through without a Swift case or fallback semantics.
struct NodeKind: Hashable {
    enum Identity: Hashable {
        case engine(NodeKindID)
        case monitor(ClientMonitorKindID)
    }

    let identity: Identity
    let descriptor: NodeDescriptor?
    let spec: NodeSpec

    var rawValue: String {
        switch identity {
        case .engine(let id): id.rawValue
        case .monitor(let id): id.rawValue
        }
    }

    var engineID: NodeKindID? {
        guard case .engine(let id) = identity else { return nil }
        return id
    }

    var monitorID: ClientMonitorKindID? {
        guard case .monitor(let id) = identity else { return nil }
        return id
    }

    var isMonitor: Bool { monitorID != nil }
    var isEngineSink: Bool { descriptor != nil && descriptor?.outlet == nil }

    static func == (lhs: Self, rhs: Self) -> Bool { lhs.identity == rhs.identity }

    func hash(into hasher: inout Hasher) { hasher.combine(identity) }

    static func engine(
        id: NodeKindID,
        descriptor: NodeDescriptor?,
        theme: NodeThemeOverride? = nil,
        authoredInputNames: some Sequence<String> = []
    ) -> Self {
        let servedInlets = descriptor?.inlets.map(\.name) ?? []
        let extraInlets = Set(authoredInputNames).subtracting(servedInlets).sorted()
        let inlets = servedInlets + extraInlets
        let knobs = descriptor?.parameters.compactMap { port -> KnobSpec? in
            guard let minimum = port.min,
                  let maximum = port.max,
                  let defaultValue = port.defaultValue
            else { return nil }
            return KnobSpec(
                name: port.name,
                min: minimum,
                max: maximum,
                def: defaultValue,
                log: port.logarithmic ?? false,
                unit: port.unit ?? "")
        } ?? []
        let multi = Set(descriptor?.inlets.filter(\.multi).map(\.name) ?? [])
        let title = theme?.displayName ?? displayName(for: id.rawValue)
        let accent = theme?.accentRGB.map(Color.init(hex:)) ?? Theme.wire
        let spec = NodeSpec(
            title: title,
            accent: accent,
            inlets: inlets,
            outlets: descriptor?.outlet == nil ? [] : ["out"],
            knobs: knobs,
            multiInlets: multi,
            fixed: descriptor != nil && descriptor?.outlet == nil,
            unavailable: descriptor == nil)
        return Self(identity: .engine(id), descriptor: descriptor, spec: spec)
    }

    /// Scope is a surface monitor rather than an engine node. Its client-owned
    /// ports and state are intentionally kept out of EngineVocabulary.
    static func scope(authoredInputNames: some Sequence<String> = []) -> Self {
        let id = try! ClientMonitorKindID(validating: "client.monitor.scope")
        let standard = ["ch1", "ch2", "ch3", "ch4"]
        let inlets = standard + Set(authoredInputNames).subtracting(standard).sorted()
        return Self(
            identity: .monitor(id),
            descriptor: nil,
            spec: NodeSpec(
                title: "Scope",
                accent: Color(hex: 0x7FE08C),
                inlets: inlets,
                outlets: [],
                knobs: [KnobSpec(
                    name: "window", min: 0.002, max: 0.35,
                    def: 0.02, log: true, unit: "s")],
                multiInlets: [],
                monitor: true,
                gridOverride: (w: 16, h: 12)))
    }

    static func monitor(
        id: ClientMonitorKindID,
        authoredInputNames: some Sequence<String>
    ) -> Self {
        if id.rawValue == "client.monitor.scope" {
            return .scope(authoredInputNames: authoredInputNames)
        }
        let inlets = Array(Set(authoredInputNames)).sorted()
        return Self(
            identity: .monitor(id),
            descriptor: nil,
            spec: NodeSpec(
                title: displayName(for: String(id.rawValue.dropFirst(ClientMonitorKindID.namespace.count))),
                accent: Theme.wire,
                inlets: inlets,
                outlets: [],
                knobs: [],
                multiInlets: [],
                monitor: true,
                unavailable: true))
    }

    private static func displayName(for id: String) -> String {
        id.replacingOccurrences(of: "_", with: " ")
            .replacingOccurrences(of: "-", with: " ")
            .split(separator: " ")
            .map { $0.prefix(1).uppercased() + $0.dropFirst() }
            .joined(separator: " ")
    }
}

extension KnobSpec {
    func toNorm(_ v: Double) -> Double {
        if log {
            return (Foundation.log(v) - Foundation.log(min))
                / (Foundation.log(max) - Foundation.log(min))
        }
        return (v - min) / (max - min)
    }

    func fromNorm(_ t: Double) -> Double {
        let t = Swift.min(1, Swift.max(0, t))
        if log {
            return exp(Foundation.log(min) + t * (Foundation.log(max) - Foundation.log(min)))
        }
        return min + t * (max - min)
    }

    func format(_ v: Double) -> String {
        switch unit {
        case "s": return String(format: "%.\(v < 0.001 ? 2 : 1)fms", v * 1000)
        case "sec": return String(format: "%.2fs", v)
        case "Hz": return v >= 100 ? String(format: "%.0fHz", v) : String(format: "%.2fHz", v)
        default: return String(format: "%.2f", v)
        }
    }
}
