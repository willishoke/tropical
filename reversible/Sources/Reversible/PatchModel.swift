import SwiftUI

struct PatchNode: Identifiable {
    let id: String
    let kind: NodeKind
    var position: CGPoint
    var values: [String: Double]
    /// inlet port → ordered source node ids
    var inputs: [String: [String]]
    /// Identity hue (degrees). Connections render as COLOR, not cables: an
    /// inlet wears the hues of its sources, so fan-out is the same color
    /// appearing at many inlets. Golden-angle spacing keeps neighbors apart.
    let hue: Double
    var module: ModuleReference? = nil
    /// local knob name → containing module parameter name
    var bindings: [String: String] = [:]

    var allInputs: [String] { inputs.values.flatMap { $0 } }

    var color: Color {
        Color(hue: hue / 360, saturation: 0.62, brightness: 0.88)
    }
}

enum ScopeTraceProjection {
    /// Preserve each channel's own trigger-search horizon when several
    /// timebases share one generation-pinned RPC. Source-sample spans are
    /// converted through the response stride so a slow known signal can use
    /// a long acquisition without making the display itself enormous.
    static func triggeredSamples(
        _ values: [Double],
        displaySpanSamples: Int,
        acquisitionSpanSamples: Int,
        stride: Int
    ) -> [Double] {
        guard displaySpanSamples > 0, !values.isEmpty else { return [] }
        let safeStride = max(1, stride)
        let displayCount = min(
            values.count,
            max(2, (displaySpanSamples + safeStride - 1) / safeStride)
        )
        let acquisitionCount = min(
            values.count,
            max(displayCount, (acquisitionSpanSamples + safeStride - 1) / safeStride)
        )
        let suffix = Array(values.suffix(acquisitionCount))
        guard suffix.count > displayCount else { return suffix }
        guard let trigger = triggerOffset(suffix, display: displayCount) else {
            // An untriggerable/DC/one-shot signal should still be live. The
            // old index-zero fallback showed the oldest half of the fetch.
            return Array(suffix.suffix(displayCount))
        }
        return (0..<displayCount).map {
            sampleLinear(suffix, at: trigger + Double($0))
        }
    }

    /// DC-offset-tolerant rising level trigger. The threshold is the observed
    /// range midpoint rather than absolute zero, and the newest crossing with
    /// a complete display after it wins. Linear crossing interpolation keeps
    /// sub-sample phase from wobbling as the playback anchor advances.
    static func triggerOffset(_ samples: [Double], display: Int) -> Double? {
        let searchEnd = samples.count - display
        guard searchEnd > 0 else { return nil }
        let finite = samples.filter(\.isFinite)
        guard let low = finite.min(), let high = finite.max(),
              high - low >= 1e-12
        else { return nil }
        let threshold = (low + high) / 2
        let armLevel = threshold - (high - low) * 0.08
        var armed = samples[0].isFinite && samples[0] <= armLevel
        var best: Double?

        for index in 1...searchEnd {
            let previous = samples[index - 1]
            let current = samples[index]
            guard previous.isFinite, current.isFinite else {
                armed = false
                continue
            }
            if current <= armLevel { armed = true }
            guard armed, previous < threshold, current >= threshold else { continue }
            let width = current - previous
            let crossing = Double(index - 1) +
                (width == 0 ? 0 : (threshold - previous) / width)
            if crossing <= Double(searchEnd) {
                best = crossing
                armed = false
            }
        }
        if best != nil { return best }

        // A steep or very asymmetric waveform can skip the arming band while
        // retaining a perfectly useful rising midpoint crossing.
        for index in 1...searchEnd {
            let previous = samples[index - 1]
            let current = samples[index]
            guard previous.isFinite, current.isFinite,
                  previous < threshold, current >= threshold
            else { continue }
            let width = current - previous
            best = Double(index - 1) +
                (width == 0 ? 0 : (threshold - previous) / width)
        }
        return best
    }

    private static func sampleLinear(_ values: [Double], at position: Double) -> Double {
        let left = max(0, min(values.count - 1, Int(floor(position))))
        let right = min(values.count - 1, left + 1)
        let fraction = position - Double(left)
        return values[left] + (values[right] - values[left]) * fraction
    }
}

enum ScopeSignalKnowledge {
    static let defaultWindow = 0.02
    static let minimumWindow = 0.002
    static let maximumWindow = 0.35
    static let visibleCycles = 4.0

    /// The surface graph already knows oscillator and modal fundamentals. Use
    /// that knowledge for a useful shared timebase, following audio/modal
    /// inputs but deliberately ignoring temporal/control inputs such as a
    /// resonator's `addr` signal.
    static func frequencies(
        for sourceID: String,
        nodes: [String: PatchNode]
    ) -> [Double] {
        var visiting = Set<String>()
        return frequencies(for: sourceID, nodes: nodes, visiting: &visiting)
    }

    static func recommendedWindow(
        for sourceIDs: [String],
        nodes: [String: PatchNode]
    ) -> Double? {
        let frequency = sourceIDs.flatMap { frequencies(for: $0, nodes: nodes) }
            .filter { $0.isFinite && $0 > 0 }
            .min()
        guard let frequency else { return nil }
        return min(maximumWindow, max(minimumWindow, visibleCycles / frequency))
    }

    static func acquisitionSpanSamples(
        displaySpanSamples: Int,
        frequency: Double?,
        sampleRate: Double
    ) -> Int {
        guard let frequency, frequency.isFinite, frequency > 0 else {
            return min(
                ObservationBinding.maximumSpan,
                max(displaySpanSamples + 1, displaySpanSamples * 2)
            )
        }
        let onePeriod = Int(ceil(sampleRate / frequency)) + 2
        return min(
            ObservationBinding.maximumSpan,
            max(displaySpanSamples + 1, displaySpanSamples + onePeriod)
        )
    }

    static func isAutomaticWindow(
        _ window: Double,
        sourceIDs: [String],
        nodes: [String: PatchNode]
    ) -> Bool {
        let expected = recommendedWindow(for: sourceIDs, nodes: nodes) ?? defaultWindow
        return abs(window - expected) <= max(1e-9, expected * 1e-6)
    }

    private static func frequencies(
        for sourceID: String,
        nodes: [String: PatchNode],
        visiting: inout Set<String>
    ) -> [Double] {
        guard visiting.insert(sourceID).inserted,
              let node = nodes[sourceID]
        else { return [] }
        defer { visiting.remove(sourceID) }

        switch node.kind {
        case .source:
            if let controlID = node.inputs["freq"]?.first,
               let control = nodes[controlID], control.kind == .knob {
                return [control.values["value"] ?? 220]
            }
            return [node.values["freq"] ?? 220]
        case .resonator:
            return [node.values["freq"] ?? 220]
        case .fm:
            return [node.values["carrier"] ?? 330]
        case .knob, .scope, .moduleInput:
            return []
        case .modalBlend:
            return ["dry", "wet"].flatMap { port in
                (node.inputs[port] ?? []).flatMap {
                    frequencies(for: $0, nodes: nodes, visiting: &visiting)
                }
            }
        case .flange, .sflange, .delay, .reverse, .mix, .ring,
             .reverb, .phaser, .modalmix, .allpass, .modalTail,
             .moduleOutput, .out:
            // `sflange.mod` and similar non-audio inlets describe motion, not
            // the trace's carrier; all of these kinds carry audio/modal data
            // through their `in` inlet.
            return (node.inputs["in"] ?? []).flatMap {
                frequencies(for: $0, nodes: nodes, visiting: &visiting)
            }
        }
    }
}

enum ObservationTapSelection {
    static func stableSourceIDs(
        candidates: [String],
        allowedEngineNodeIDs: Set<String>
    ) -> [String] {
        var seen = Set<String>()
        return candidates.filter {
            allowedEngineNodeIDs.contains($0) && seen.insert($0).inserted
        }
    }

    static func expectedHandshakeNames(requestedSourceIDs: [String]) -> [String] {
        ["out"] + requestedSourceIDs
    }
}

enum PatchCompileState: Equatable, Sendable {
    case idle
    case compiling(authoredRevision: Int)
    case running(authoredRevision: Int, generation: EngineGeneration)
    case failed(authoredRevision: Int, message: String)
    case superseded(compiledRevision: Int, currentRevision: Int)
}

enum NodePresentationStatus: Equatable, Sendable {
    case active
    case excluded
    case pending
}

/// The patch graph + engine session, one object. The realized compile handshake
/// owns scalar routing: live parameters drive `set_param`; structural values
/// and topology edits relower and hot-swap. Authored revision and lowering
/// compatibility remain separate so acknowledged live editing does not invent
/// a structural invalidation.
@MainActor
final class PatchModel: ObservableObject {
    @Published var nodes: [String: PatchNode] = [:]
    @Published var order: [String] = []   // stable z/render order
    @Published var definitions: [ModuleReference: ModuleDefinitionState] = [:]
    @Published var hierarchyPath: [HierarchyFrame] = [
        .init(location: .scene, title: "Patch", instanceID: nil)
    ]
    var currentHierarchyLocation: HierarchyLocation = .scene
    var sceneGraph = PatchGraphState(nodes: [:], order: [], pan: .zero)
    @Published var status = "booting engine…"
    @Published var statusIsError = false
    private var telemetryOwnsStatus = false
    @Published var audioOn = false
    @Published var velocity: Double = 1
    /// World → view offset for the infinite plane. Lives on the model so
    /// arrange can bring the graph back into view.
    @Published var pan: CGSize = .zero
    /// When on, every topology edit re-runs the rank layout.
    @Published var autoArrange = false
    /// The file this patch came from / saves to (nil = never saved). Drives
    /// the window title and whether ⌘S needs to ask.
    @Published var documentURL: URL?
    @Published private(set) var vocabulary: EngineVocabulary?
    @Published private(set) var installedDefinitions: [
        ModuleReference: ModuleDefinitionState
    ] = [:]
    @Published private(set) var authoredRevision = 0
    @Published private(set) var realizedPatch: RealizedPatchSnapshot?
    @Published private(set) var runningGeneration: EngineGeneration?
    @Published private(set) var compileState: PatchCompileState = .idle

    /// Sources an inlet may legally take — the connection UI shows ONLY
    /// these (the type discipline enforced by omission: control inlets
    /// list knobs, modal inlets list modal nodes, cycle-formers absent).
    func legalSources(for nodeId: String, port: String) -> [PatchNode] {
        order.compactMap { nodes[$0] }
            .filter { canConnect(from: $0.id, to: nodeId, port: port) }
    }

    func presentationStatus(for node: PatchNode) -> NodePresentationStatus? {
        guard !node.kind.spec.monitor else { return nil }
        guard let realizedPatch,
              realizedPatch.loweringRevision == loweringRevision
        else { return .pending }
        let realizedNode: RealizedNode?
        if isAtScene {
            realizedNode = realizedPatch.patch.node(id: node.id)
        } else {
            let path = hierarchyPath.compactMap(\.instanceID)
            let candidates: [RealizedNode]
            if node.kind.isModule {
                candidates = realizedPatch.patch.nodes(
                    atInstancePath: path + [node.id],
                    localID: nil
                )
            } else {
                candidates = realizedPatch.patch.nodes(
                    atInstancePath: path,
                    localID: node.id
                )
            }
            realizedNode = candidates.first { $0.status == .active } ?? candidates.first
        }
        guard let realizedNode else { return nil }
        return realizedNode.status == .active ? .active : .excluded
    }

    func parameterStatus(
        for node: PatchNode,
        knob: KnobSpec
    ) -> RealizedParameterStatus {
        if node.kind.spec.monitor { return .structural }
        guard let realizedPatch,
              realizedPatch.loweringRevision == loweringRevision
        else { return .inactive }
        return realizedPatch.patch.parameterStatus(
            nodeID: node.id,
            port: knob.name
        )
    }

    // ── Topological layout ────────────────────────────────────────────────
    /// Rank = longest path from the sources (0 = no inputs). Total on a DAG;
    /// the graph is acyclic by construction (canConnect rejects back-edges),
    /// and the visit guard keeps a hypothetical bug from recursing forever.
    func topologicalRanks() -> [String: Int] {
        var memo: [String: Int] = [:]
        func rank(_ id: String) -> Int {
            if let r = memo[id] { return r }
            memo[id] = 0
            guard let n = nodes[id] else { return 0 }
            let r = n.allInputs.map { rank($0) + 1 }.max() ?? 0
            memo[id] = r
            return r
        }
        for id in nodes.keys { _ = rank(id) }
        return memo
    }

    /// One row per rank, sources at the top, dac at the bottom; within a row
    /// modules keep their current left-to-right order (the mental map
    /// survives the tidy). Positions land on the grid by construction.
    func arrangeByRank() {
        let ranks = topologicalRanks()
        let rows = Dictionary(grouping: nodes.values) { ranks[$0.id] ?? 0 }
        let margin = 2.0 * Grid.unit
        let gap = 2.0 * Grid.unit
        let rowPitch = 8.0 * Grid.unit   // tallest ordinary module (7u) + breathing room
        withAnimation(.snappy(duration: 0.35)) {
            for (r, members) in rows {
                var x = margin
                for n in members.sorted(by: { $0.position.x < $1.position.x }) {
                    nodes[n.id]?.position = CGPoint(x: x, y: margin + Double(r) * rowPitch)
                    x += Double(n.kind.spec.gridSize.w) * Grid.unit + gap
                }
            }
            pan = .zero   // bring the arranged graph back into view
        }
    }

    private func autoArrangeIfOn() {
        if autoArrange { arrangeByRank() }
    }

    let engine: Engine
    private let observationSampler: ObservationSampler
    private let observationDisplayLink: ObservationDisplayLink
    private var truth = PatchTruthStore()
    private var loweringRevision = 0
    private var counter = 0

    // Node ids are "<kind><n>" and must stay unique for the lifetime of a
    // patch — a loaded file carries ids the counter has never issued, so the
    // counter resumes past the highest one rather than restarting at 0.
    func adoptCounter(from ids: some Sequence<String>) {
        counter = ids.reduce(0) { hi, id in
            max(hi, Int(id.drop(while: { !$0.isNumber })) ?? 0)
        }
    }

    func resetCounter() { counter = 0 }

    private var pushTask: Task<Void, Never>?
    private var pushInFlight = false
    private var pushAgain = false

    init() {
        // Bounce engine events onto the main actor; Engine is created with a
        // plain sendable callback so the actor never touches UI state itself.
        let box = EventBox()
        let forward: @Sendable (EngineEvent) -> Void = { [box] event in
            Task { @MainActor [box] in box.handler?(event) }
        }
        let engine = Engine(events: forward)
        self.engine = engine
        observationSampler = ObservationSampler(rpc: engine)
        observationDisplayLink = ObservationDisplayLink()
        box.handler = { [weak self] event in self?.handle(event) }
        observationSampler.onFrame = { [weak self] frame in
            self?.adoptObservationFrame(frame)
        }
        // Every exit path must reap the child engine or it orphans with the
        // DAC running (see AppDelegate).
        AppDelegate.onTerminate = { [engine] in engine.terminateChild() }

        startEngine()
        seedBootPatch()
        snapshotActiveGraph()
        setStatus("engine up · patch something", isError: false)
        startScopePolling()
    }

    private func startEngine() {
        let e = engine
        Task { [weak self] in
            do { try await e.start() } catch {
                self?.setStatus("engine: \(error.localizedDescription)", isError: true)
            }
        }
    }

    private func handle(_ event: EngineEvent) {
        switch event {
        case .up:
            setStatus("engine up", isError: false)
            Task { await synchronizeEngineState() }
        case .exit(let code):
            vocabulary = nil
            installedDefinitions = [:]
            lastPushed = nil
            truth.reset()
            publishTruth()
            observationSampler.clear()
            scopeTraces = [:]
            compileState = .idle
            setStatus("engine exited (code \(code))", isError: true)
        case .stderr(let text):
            // Diagnostic only — the Electron app console.warn'd these.
            FileHandle.standardError.write(Data("[engine] \(text)".utf8))
        }
    }

    private func synchronizeEngineState() async {
        do {
            vocabulary = try await engine.loadVocabulary()
            installedDefinitions = try await engine.loadModuleLibrary()
            await syncAudioState()
            schedulePush()
        } catch {
            setStatus("engine vocabulary: \(error.localizedDescription)", isError: true)
        }
    }

    func moduleTemplate(for reference: ModuleReference) -> ModuleDefinitionState? {
        installedDefinitions[reference]
    }

    private func publishTruth() {
        realizedPatch = truth.realized
        runningGeneration = truth.runningGeneration
    }

    func advanceAuthoredRevision(requiresCompile: Bool = true) {
        authoredRevision += 1
        if requiresCompile { loweringRevision += 1 }
    }

    func setStatus(_ text: String, isError: Bool) {
        status = text
        statusIsError = isError
        telemetryOwnsStatus = false
    }

    private func setTelemetryStatus(_ text: String, isError: Bool) {
        status = text
        statusIsError = isError
        telemetryOwnsStatus = true
    }

    /// The patch you start with, and the one `New Patch` returns to.
    func seedBootPatch() {
        addNode(.out, at: CGPoint(x: 1180, y: 360))
        addNode(.source, at: CGPoint(x: 120, y: 200))
    }

    // ── Node lifecycle ───────────────────────────────────────────────────
    @discardableResult
    func addNode(_ kind: NodeKind, at point: CGPoint? = nil) -> PatchNode? {
        let spec = kind.spec
        if spec.fixed && nodes.values.contains(where: { $0.kind == kind }) { return nil }
        counter += 1
        let id = "\(kind.rawValue)\(counter)"
        let pos = Grid.snap(point ?? CGPoint(
            x: 80 + Double(counter % 6) * 34,
            y: 90 + Double(counter % 6) * 30))
        var values: [String: Double] = [:]
        for k in spec.knobs { values[k.name] = k.def }
        var inputs: [String: [String]] = [:]
        for p in spec.inlets { inputs[p] = [] }
        var module: ModuleReference?
        if kind == .phaser {
            module = StandardModuleLibrary.phaser
        } else if kind == .allpass {
            module = StandardModuleLibrary.allpass
        }
        let n = PatchNode(
            id: id, kind: kind, position: pos, values: values, inputs: inputs,
            hue: Double((counter * 137) % 360), module: module)
        nodes[id] = n
        order.append(id)
        autoArrangeIfOn()
        advanceAuthoredRevision(requiresCompile: !kind.spec.monitor)
        schedulePush()
        return n
    }

    /// Remember which scopes are still following graph-derived defaults
    /// before a graph/parameter edit. A user-turned window no longer equals
    /// its prior recommendation and is therefore left alone.
    private func automaticallyTunedScopeIDs(
        in graph: [String: PatchNode]
    ) -> Set<String> {
        Set(graph.values.compactMap { scope in
            guard scope.kind == .scope,
                  ScopeSignalKnowledge.isAutomaticWindow(
                    scope.values["window"] ?? ScopeSignalKnowledge.defaultWindow,
                    sourceIDs: scope.allInputs,
                    nodes: graph
                  )
            else { return nil }
            return scope.id
        })
    }

    private func retuneScopes(_ scopeIDs: Set<String>) {
        for id in scopeIDs {
            guard var scope = nodes[id], scope.kind == .scope else { continue }
            let recommended = ScopeSignalKnowledge.recommendedWindow(
                for: scope.allInputs,
                nodes: nodes
            ) ?? ScopeSignalKnowledge.defaultWindow
            guard scope.values["window"] != recommended else { continue }
            scope.values["window"] = recommended
            nodes[id] = scope
        }
    }

    func deleteNode(_ id: String) {
        guard let n = nodes[id], !n.kind.spec.fixed else { return }
        let automaticScopes = automaticallyTunedScopeIDs(in: nodes)
        nodes.removeValue(forKey: id)
        order.removeAll { $0 == id }
        for var m in nodes.values {
            for p in m.inputs.keys { m.inputs[p]?.removeAll { $0 == id } }
            nodes[m.id] = m
        }
        retuneScopes(automaticScopes)
        autoArrangeIfOn()
        advanceAuthoredRevision()
        schedulePush()
    }

    // ── Connection discipline ────────────────────────────────────────────
    /// reachability for the cycle check (across all inlets)
    private func reaches(_ a: String, _ target: String, _ seen: inout Set<String>) -> Bool {
        if a == target { return true }
        if seen.contains(a) { return false }
        seen.insert(a)
        guard let na = nodes[a] else { return false }
        return na.allInputs.contains { reaches($0, target, &seen) }
    }

    func canConnect(from fromId: String, to toId: String, port: String) -> Bool {
        guard fromId != toId,
              let from = nodes[fromId], let to = nodes[toId],
              to.kind.spec.inlets.contains(port),
              !(to.inputs[port] ?? []).contains(fromId)
        else { return false }
        let fromIsKnob = from.kind == .knob
        // A sink (Out, Scope) has no outlet, so it is never a source — except
        // the final mix, which is always tappable (`__root__.out`), so Out may
        // feed a Scope channel.
        if from.kind.spec.outlets.isEmpty && !(to.kind == .scope && from.kind == .out) { return false }
        // A knob is a control value: it may only drive a control inlet (freq/mod).
        if fromIsKnob && !NodeKind.controlInlets.contains(port) { return false }
        // `freq` is control-only — audio-rate into the pitch port is not FM.
        if !fromIsKnob && port == "freq" { return false }
        // A modal inlet (Reverb/Phaser/Modal∪) carries poles: only a modal node may
        // drive it. (A modal outlet INTO a Sig inlet is fine — it realizes at
        // the seam.)
        if to.kind.wantsModal(inlet: port) && !from.kind.spec.modal { return false }
        var seen = Set<String>()
        if reaches(fromId, toId, &seen) { return false }   // would form a cycle
        return true
    }

    func connect(from fromId: String, to toId: String, port: String) {
        guard canConnect(from: fromId, to: toId, port: port), var to = nodes[toId] else { return }
        let automaticScopes = automaticallyTunedScopeIDs(in: nodes)
        let servedMulti: Bool? = {
            guard let vocabulary,
                  let kind = NodeKindID(rawValue: to.kind.rawValue),
                  let descriptor = vocabulary.descriptor(for: kind),
                  let inlet = descriptor.port(named: port)
            else { return nil }
            return inlet.multi
        }()
        if !(servedMulti ?? to.kind.allowsMultiple(inlet: port)) {
            to.inputs[port] = []
        }
        to.inputs[port, default: []].append(fromId)
        nodes[toId] = to
        retuneScopes(automaticScopes)
        autoArrangeIfOn()
        advanceAuthoredRevision()
        schedulePush()
    }

    func deleteEdge(to toId: String, port: String, from fromId: String) {
        guard var to = nodes[toId] else { return }
        let automaticScopes = automaticallyTunedScopeIDs(in: nodes)
        to.inputs[port]?.removeAll { $0 == fromId }
        nodes[toId] = to
        retuneScopes(automaticScopes)
        autoArrangeIfOn()
        advanceAuthoredRevision()
        schedulePush()
    }

    // ── Serialization → load_patch_graph ─────────────────────────────────
    /// Stable, explicit observation projection. Monitor order and inlet order
    /// define the request; repeated sources are kept once, and Out is omitted
    /// because its final-mix binding is always published as `out`.
    var requestedTapNodeIDs: [String] {
        snapshotActiveGraph()
        var candidates: [String] = []
        for id in sceneGraph.order {
            guard let monitor = sceneGraph.nodes[id], monitor.kind.spec.monitor else { continue }
            for port in monitor.kind.spec.inlets {
                candidates.append(contentsOf: monitor.inputs[port] ?? [])
            }
        }
        let allowed = Set(sceneGraph.nodes.values.compactMap { node in
            node.kind != .out && !node.kind.spec.monitor ? node.id : nil
        })
        return ObservationTapSelection.stableSourceIDs(
            candidates: candidates,
            allowedEngineNodeIDs: allowed
        )
    }

    private var expectedTapNames: [String] {
        ObservationTapSelection.expectedHandshakeNames(
            requestedSourceIDs: requestedTapNodeIDs
        )
    }

    func serialize() -> JSONValue {
        snapshotActiveGraph()

        func engineNodes(_ graph: PatchGraphState) -> [JSONValue] {
            var result: [JSONValue] = []
            for id in graph.order {
                guard let node = graph.nodes[id], !node.kind.spec.monitor else { continue }
                let spec = node.kind.spec
                var params: [String: JSONValue] = [:]
                for knob in spec.knobs {
                    params[knob.name] = .number(node.values[knob.name] ?? knob.def)
                }
                var inputs: [String: JSONValue] = [:]
                for port in spec.inlets {
                    inputs[port] = .array((node.inputs[port] ?? []).map(JSONValue.string))
                }
                var encoded: [String: JSONValue] = [
                    "id": .string(node.id),
                    "kind": .string(node.module == nil ? node.kind.rawValue : "module"),
                    "params": .object(params),
                    "sel": .object([:]),
                    "in": .object(inputs),
                ]
                if let module = node.module {
                    encoded["definition"] = .string(module.id)
                    encoded["definition_version"] = .number(Double(module.version))
                }
                if !node.bindings.isEmpty {
                    encoded["bindings"] = .object(
                        node.bindings.mapValues(JSONValue.string)
                    )
                }
                result.append(.object(encoded))
            }
            return result
        }

        let definitionValues: [JSONValue] = definitions.values
            .sorted {
                ($0.reference.id, $0.reference.version) <
                    ($1.reference.id, $1.reference.version)
            }
            .map { definition in
                .object([
                    "id": .string(definition.reference.id),
                    "version": .number(Double(definition.reference.version)),
                    "input": .string(definition.inputNodeID),
                    "output": .string(definition.outputNodeID),
                    "input_domain": .string("modal"),
                    "output_domain": .string("modal"),
                    "parameters": .array(definition.parameters.map { parameter in
                        .object([
                            "name": .string(parameter.name),
                            "default": .number(parameter.defaultValue),
                        ])
                    }),
                    "nodes": .array(engineNodes(definition.graph)),
                ])
            }
        let outNode = sceneGraph.nodes.values.first { $0.kind == .out }
        return .object([
            "version": .number(3),
            "definitions": .array(definitionValues),
            "scene": .object([
                "nodes": .array(engineNodes(sceneGraph)),
                "out": outNode.map { .string($0.id) } ?? .null,
            ]),
            "taps": .array(requestedTapNodeIDs.map(JSONValue.string)),
        ])
    }

    // ── Relower push (debounced, single-in-flight) ────────────────────────
    func schedulePush(after delay: Duration = .zero) {
        pushTask?.cancel()
        pushTask = Task { [weak self] in
            if delay > .zero { try? await Task.sleep(for: delay) }
            guard !Task.isCancelled else { return }
            await self?.pushGraph()
        }
    }

    private var lastPushed: JSONValue?

    func pushGraph() async {
        guard let vocabulary else { return }
        if pushInFlight { pushAgain = true; return }
        pushInFlight = true
        defer {
            pushInFlight = false
            if pushAgain { pushAgain = false; schedulePush() }
        }
        // A monitor-only edit (e.g. scoping the Out mix) leaves the engine
        // graph byte-identical — skip the recompile, not just debounce it.
        let graph = serialize()
        let revision = authoredRevision
        let compiledLoweringRevision = loweringRevision
        if graph == lastPushed {
            truth.retagCurrent(
                authoredRevision: revision,
                loweringRevision: compiledLoweringRevision
            )
            publishTruth()
            if let runningGeneration {
                compileState = .running(
                    authoredRevision: revision,
                    generation: runningGeneration
                )
            }
            return
        }
        do {
            // The compile+JIT is synchronous on the engine's single control
            // thread, so a heavy node (e.g. a modal reverb: ~10⁵ IR lines →
            // seconds of LLVM) blocks here with no other signal. Show it, so
            // "compiling" is distinguishable from "hung" — the acute failure
            // mode of the live-patch loop.
            setStatus("compiling…", isError: false)
            compileState = .compiling(authoredRevision: revision)
            let adoption = try await engine.loadPatchGraph(
                graph,
                expectedTapNames: expectedTapNames,
                vocabularyFingerprint: vocabulary.fingerprint
            )
            guard truth.adoptCompile(
                adoption,
                authoredRevision: revision,
                loweringRevision: compiledLoweringRevision,
                graph: graph
            ) else {
                switch adoption {
                case .superseded, .stale:
                    compileState = .superseded(
                        compiledRevision: revision,
                        currentRevision: authoredRevision
                    )
                    setStatus(
                        "compile superseded · running patch unchanged",
                        isError: false
                    )
                case .adopted:
                    break
                }
                return
            }
            lastPushed = graph
            publishTruth()
            if let realizedPatch {
                observationSampler.adoptCompileGeneration(
                    realizedPatch.generation
                )
            }
            scopeTraces = [:]
            // A relower transfers slots by name, so a live scrub survives it —
            // except a COLD first compile, which seeds master.velocity at its
            // default (1). Re-apply a non-default scrub so the slider and the
            // running clock agree (no-op if equal).
            if velocity != 1 { params.send("master.velocity", velocity) }
            if revision == authoredRevision, let runningGeneration {
                compileState = .running(
                    authoredRevision: revision,
                    generation: runningGeneration
                )
                setStatus(audioOn ? "playing" : "compiled", isError: false)
            } else {
                compileState = .superseded(
                    compiledRevision: revision,
                    currentRevision: authoredRevision
                )
                setStatus("compiled revision superseded by edits", isError: false)
            }
        } catch {
            compileState = .failed(
                authoredRevision: revision,
                message: error.localizedDescription
            )
            setStatus("compile: \(error.localizedDescription)", isError: true)
        }
    }

    // ── Live param drive ──────────────────────────────────────────────────
    lazy var params = ParamSender(model: self)

    func setKnob(_ node: PatchNode, _ knob: KnobSpec, _ value: Double) {
        applyKnob(
            nodeID: node.id,
            knob: knob,
            value: value,
            guaranteeDelivery: false
        )
    }

    /// End-of-gesture delivery does not trust the view's node snapshot. Even
    /// when authored truth already equals the final pointer value, enqueue it
    /// once more so coalescing cannot leave the engine at an intermediate
    /// value when the gesture produces no further events.
    func finishKnobGesture(nodeID: String, knob: KnobSpec, value: Double) {
        applyKnob(
            nodeID: nodeID,
            knob: knob,
            value: value,
            guaranteeDelivery: true
        )
    }

    private func applyKnob(
        nodeID: String,
        knob: KnobSpec,
        value: Double,
        guaranteeDelivery: Bool
    ) {
        guard value.isFinite, var current = nodes[nodeID] else { return }
        let changed = current.values[knob.name] != value
        guard changed || guaranteeDelivery else { return }
        if changed {
            let automaticScopes = current.kind.spec.monitor
                ? Set<String>()
                : automaticallyTunedScopeIDs(in: nodes)
            current.values[knob.name] = value
            nodes[nodeID] = current
            retuneScopes(automaticScopes)
        }

        // A monitor's knobs (Scope `window`) are authored view state, not
        // parameter slots and do not invalidate lowering compatibility.
        guard !current.kind.spec.monitor else {
            if changed { advanceAuthoredRevision(requiresCompile: false) }
            return
        }
        let name = "\(nodeID).\(knob.name)"
        guard let realizedPatch,
              realizedPatch.loweringRevision == loweringRevision,
              case .live = realizedPatch.patch.parameterStatus(
                nodeID: nodeID,
                port: knob.name
              )
        else {
            guard changed else { return }
            // Absent live truth means the edit is structural (or belongs to a
            // not-yet-realized lowering revision).
            advanceAuthoredRevision()
            schedulePush(after: .milliseconds(90))
            return
        }
        if changed { advanceAuthoredRevision(requiresCompile: false) }
        params.send(name, value)
    }

    func setVelocity(_ v: Double, force: Bool = false) {
        guard v.isFinite else { return }
        let changed = velocity != v
        if changed {
            velocity = v
            advanceAuthoredRevision(requiresCompile: false)
        }
        if changed || force { params.send("master.velocity", v) }
    }

    // ── Transport ─────────────────────────────────────────────────────────
    func toggleAudio() async {
        do {
            if !audioOn {
                try await engine.call("start_audio")
                audioOn = true
                setStatus("playing", isError: false)
                startTelemetry()
            } else {
                try await engine.call("stop_audio")
                audioOn = false
                setStatus("stopped", isError: false)
                stopTelemetry()
            }
        } catch {
            setStatus(error.localizedDescription, isError: true)
        }
    }

    /// The engine (audio + DAC) is a separate process that OUTLIVES a window
    /// reload — a fresh surface must not assume audio is off. Ask the engine
    /// its real state and adopt it, otherwise the transport shows "start"
    /// while sound keeps playing with no way to stop it.
    func syncAudioState() async {
        guard let st = try? await engine.call("audio_status"),
              st["is_running"]?.boolValue == true, !audioOn else { return }
        audioOn = true
        startTelemetry()
    }

    // ── Live audio-load telemetry ─────────────────────────────────────────
    // Compile latency is one failure mode; runtime deadline cost is the
    // other. Difference the DAC's cumulative callback totals into recent
    // windows; a lifetime max is useful qualification evidence but a bad UI
    // meter because one spike otherwise remains at (say) 230% forever.
    private var telemetryTask: Task<Void, Never>?
    private var telemetryInFlight = false
    private var audioLoadMeter = AudioLoadMeter()

    func stopTelemetry() {
        telemetryTask?.cancel()
        telemetryTask = nil
    }

    func startTelemetry() {
        stopTelemetry()
        telemetryTask = Task { [weak self] in
            if let self {
                if let status = try? await self.engine.call("audio_status"),
                   let sample = AudioTelemetrySample(status: status) {
                    self.audioLoadMeter.reset(to: sample)
                }
            }
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(750))
                guard !Task.isCancelled else { return }
                await self?.pollTelemetry()
            }
        }
    }

    // ── Generation-pinned observations ────────────────────────────────────
    /// Latest complete trace set, keyed "<scopeId>.<port>". The dictionary is
    /// replaced only from one immutable ObservationFrame.
    @Published var scopeTraces: [String: [Double]] = [:]

    private struct ScopeProjection: Equatable {
        let binding: ObservationBinding
        let channels: [String: ScopeChannelProjection]
    }

    private struct ScopeChannelProjection: Equatable {
        let displaySpanSamples: Int
        let acquisitionSpanSamples: Int
    }

    func scopeSlot(for sourceId: String) -> String? {
        guard let realizedPatch else { return nil }
        let tapName = nodes[sourceId]?.kind == .out ? "out" : sourceId
        return realizedPatch.patch.tap(named: tapName)?.slot
    }

    static let sampleRate = 44100.0
    private func startScopePolling() {
        observationDisplayLink.start { [weak self] in
            self?.pollScopes()
        }
    }

    private func pollScopes() {
        guard let projection = makeScopeProjection() else {
            if !scopeTraces.isEmpty { scopeTraces = [:] }
            return
        }
        observationSampler.request(projection.binding)
    }

    private func makeScopeProjection() -> ScopeProjection? {
        guard let realizedPatch else { return nil }
        let scopes = order.compactMap { nodes[$0] }
            .filter { $0.kind == .scope && !$0.allInputs.isEmpty }
        guard !scopes.isEmpty else { return nil }

        var channels: [ObservationChannel] = []
        var channelProjections: [String: ScopeChannelProjection] = [:]
        var desiredMaximumFetch = 0
        var highestKnownFrequency = 0.0
        var hasUnknownFrequency = false
        for scope in scopes {
            let window = scope.values["window"] ?? ScopeSignalKnowledge.defaultWindow
            let displaySpanSamples = max(128, Int(window * Self.sampleRate))
            for port in scope.kind.spec.inlets {
                guard let src = scope.inputs[port]?.first,
                      let slot = scopeSlot(for: src) else { continue }
                let key = "\(scope.id).\(port)"
                let frequencies = ScopeSignalKnowledge.frequencies(
                    for: src,
                    nodes: nodes
                ).filter { $0.isFinite && $0 > 0 }
                let fundamental = frequencies.min()
                if let channelHigh = frequencies.max() {
                    highestKnownFrequency = max(highestKnownFrequency, channelHigh)
                } else {
                    hasUnknownFrequency = true
                }
                let acquisitionSpanSamples =
                    ScopeSignalKnowledge.acquisitionSpanSamples(
                        displaySpanSamples: displaySpanSamples,
                        frequency: fundamental,
                        sampleRate: Self.sampleRate
                    )
                channels.append(.init(key: key, slot: slot))
                channelProjections[key] = .init(
                    displaySpanSamples: displaySpanSamples,
                    acquisitionSpanSamples: acquisitionSpanSamples
                )
                desiredMaximumFetch = max(
                    desiredMaximumFetch,
                    acquisitionSpanSamples
                )
            }
        }
        // Keep at least eight observation points per cycle of the fastest
        // known channel. A very slow channel may ask for seconds of trigger
        // history, but it must not alias a fast trace sharing the RPC.
        let maximumStride: Int
        if hasUnknownFrequency || highestKnownFrequency <= 0 {
            maximumStride = 1
        } else {
            maximumStride = max(
                1,
                Int(floor(Self.sampleRate / (highestKnownFrequency * 8)))
            )
        }
        let maximumFetch = min(
            desiredMaximumFetch,
            ObservationBinding.maximumPointBudget * maximumStride
        )
        for (key, projection) in channelProjections {
            channelProjections[key] = .init(
                displaySpanSamples: projection.displaySpanSamples,
                acquisitionSpanSamples: min(
                    projection.acquisitionSpanSamples,
                    maximumFetch
                )
            )
        }
        guard maximumFetch > 0, !channels.isEmpty,
              let binding = try? ObservationBinding(
                expectedGeneration: realizedPatch.generation,
                span: maximumFetch,
                pointBudget: min(
                    maximumFetch,
                    ObservationBinding.maximumPointBudget
                ),
                channels: channels
              )
        else { return nil }
        return .init(binding: binding, channels: channelProjections)
    }

    private func adoptObservationFrame(_ frame: ObservationFrame) {
        if truth.adoptObservation(frame.generation) {
            publishTruth()
            if case .running(let revision, _) = compileState,
               let runningGeneration {
                compileState = .running(
                    authoredRevision: revision,
                    generation: runningGeneration
                )
            }
        }

        // The authored monitor wiring may have changed while the socket request
        // was in flight. Never publish old channel identities into a new view.
        guard let projection = makeScopeProjection(),
              projection.binding == frame.binding
        else { return }
        var fresh: [String: [Double]] = [:]
        for channel in frame.channels {
            guard let channelProjection = projection.channels[channel.channel.key]
            else { continue }
            fresh[channel.channel.key] = ScopeTraceProjection.triggeredSamples(
                channel.values,
                displaySpanSamples: channelProjection.displaySpanSamples,
                acquisitionSpanSamples: channelProjection.acquisitionSpanSamples,
                stride: frame.stride
            )
        }
        guard fresh.count == frame.channels.count else { return }
        scopeTraces = fresh
    }

    private func pollTelemetry() async {
        // Single-in-flight: if a poll is delayed we must not enqueue another.
        // Keep polling after a warning so load can recover; `statusIsError`
        // used to freeze the first over-budget reading permanently.
        if !audioOn || pushInFlight || telemetryInFlight ||
            (statusIsError && !telemetryOwnsStatus) { return }
        telemetryInFlight = true
        defer { telemetryInFlight = false }
        guard let status = try? await engine.call("audio_status"),
              let sample = AudioTelemetrySample(status: status),
              let reading = audioLoadMeter.update(sample) else { return }
        let hasDeadlineProblem = reading.xruns > 0 || reading.deadlineMisses > 0
        let issueText: String
        if reading.xruns > 0 {
            issueText = " · ⚠ \(reading.xruns) xrun\(reading.xruns == 1 ? "" : "s") — dropout"
        } else if reading.deadlineMisses > 0 {
            issueText = " · ⚠ \(reading.deadlineMisses) deadline miss\(reading.deadlineMisses == 1 ? "" : "es")"
        } else {
            issueText = ""
        }
        setTelemetryStatus(
            "\(reading.backend) · load \(reading.percent)%\(issueText)",
            isError: reading.percent > 90 || hasDeadlineProblem
        )
    }
}

/// Breaks the init-order knot between the Engine's @Sendable event callback
/// and the @MainActor model that wants to receive the events.
private final class EventBox: @unchecked Sendable {
    @MainActor var handler: ((EngineEvent) -> Void)?

    @MainActor init() {}
}

enum ParamWriteFailureRecovery: Equatable {
    case reportOnly
}

enum ParamWriteFailurePolicy {
    /// A failed write says nothing about the authored graph. In particular,
    /// transport timeouts and publication failures must not turn a live
    /// control gesture into an unrelated graph compile.
    static func recovery(for _: Error) -> ParamWriteFailureRecovery {
        .reportOnly
    }

    static func status(name: String, error: Error) -> String {
        "parameter \(name): \(error.localizedDescription)"
    }
}

/// Live param drive: set the `param:<name>` slot on the running kernel — no
/// recompile. Rapid drags coalesce to the latest value per name. Cold-start
/// lowering is decided by `setKnob` before a write reaches this sender; once
/// a control is live, an RPC failure is reported and never relowers the graph.
@MainActor
final class ParamSender {
    private var pending: [String: LatestValueBuffer] = [:]
    private var busy: Set<String> = []
    private weak var model: PatchModel?

    init(model: PatchModel) { self.model = model }

    func send(_ name: String, _ value: Double) {
        guard value.isFinite else { return }
        if busy.contains(name) {
            pending[name]?.offer(value)
            return
        }
        busy.insert(name)
        pending[name] = LatestValueBuffer(first: value)
        Task { [weak self] in
            defer {
                self?.busy.remove(name)
                self?.pending.removeValue(forKey: name)
            }
            while let v = self?.pending[name]?.pop() {
                guard let engine = self?.model?.engine else { return }
                do {
                    _ = try await engine.call("set_param", .object([
                        "name": .string(name), "value": .number(v)
                    ]))
                } catch {
                    guard let model = self?.model else { return }
                    switch ParamWriteFailurePolicy.recovery(for: error) {
                    case .reportOnly:
                        model.setStatus(
                            ParamWriteFailurePolicy.status(name: name, error: error),
                            isError: true
                        )
                    }
                    return
                }
            }
        }
    }
}
