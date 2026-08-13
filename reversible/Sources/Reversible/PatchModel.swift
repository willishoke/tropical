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
    /// Preserve each scope's own 2× trigger-search horizon when several
    /// different window sizes share one larger generation-pinned RPC.
    static func triggeredSamples(_ values: [Double], displayCount: Int) -> [Double] {
        guard displayCount > 0, !values.isEmpty else { return [] }
        let horizon = min(values.count, min(16_384, displayCount * 2))
        let suffix = Array(values.suffix(horizon))
        let trigger = triggerIndex(suffix, display: displayCount)
        return Array(suffix[trigger..<min(trigger + displayCount, suffix.count)])
    }

    /// Rising level-trigger with hysteresis (the playground scope's port of
    /// tui/scope/trigger.ts).
    static func triggerIndex(_ samples: [Double], display: Int) -> Int {
        let searchEnd = samples.count - display
        guard searchEnd > 0 else { return 0 }
        let peak = samples.reduce(0) { max($0, abs($1)) }
        let hysteresis = peak * 0.05
        guard hysteresis > 0 else { return 0 }
        var armed = false
        for index in 0..<searchEnd {
            if samples[index] < -hysteresis { armed = true }
            else if armed && samples[index] >= hysteresis { return index }
        }
        return 0
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

    func deleteNode(_ id: String) {
        guard let n = nodes[id], !n.kind.spec.fixed else { return }
        nodes.removeValue(forKey: id)
        order.removeAll { $0 == id }
        for var m in nodes.values {
            for p in m.inputs.keys { m.inputs[p]?.removeAll { $0 == id } }
            nodes[m.id] = m
        }
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
        autoArrangeIfOn()
        advanceAuthoredRevision()
        schedulePush()
    }

    func deleteEdge(to toId: String, port: String, from fromId: String) {
        guard var to = nodes[toId] else { return }
        to.inputs[port]?.removeAll { $0 == fromId }
        nodes[toId] = to
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
        guard value.isFinite, var current = nodes[node.id],
              current.values[knob.name] != value
        else { return }
        current.values[knob.name] = value
        nodes[node.id] = current

        // A monitor's knobs (Scope `window`) are authored view state, not
        // parameter slots and do not invalidate lowering compatibility.
        guard !node.kind.spec.monitor else {
            advanceAuthoredRevision(requiresCompile: false)
            return
        }
        let name = "\(node.id).\(knob.name)"
        guard let realizedPatch,
              realizedPatch.loweringRevision == loweringRevision,
              case .live = realizedPatch.patch.parameterStatus(
                nodeID: node.id,
                port: knob.name
              )
        else {
            // Absent live truth means the edit is structural (or belongs to a
            // not-yet-realized lowering revision).
            advanceAuthoredRevision()
            schedulePush(after: .milliseconds(90))
            return
        }
        advanceAuthoredRevision(requiresCompile: false)
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
        let displayCounts: [String: Int]
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
        var displayCounts: [String: Int] = [:]
        var maximumFetch = 0
        for scope in scopes {
            let window = scope.values["window"] ?? 0.02
            let displayCount = min(
                8_192,
                max(128, Int(window * Self.sampleRate))
            )
            maximumFetch = max(maximumFetch, min(16_384, displayCount * 2))
            for port in scope.kind.spec.inlets {
                guard let src = scope.inputs[port]?.first,
                      let slot = scopeSlot(for: src) else { continue }
                let key = "\(scope.id).\(port)"
                channels.append(.init(key: key, slot: slot))
                displayCounts[key] = displayCount
            }
        }
        guard maximumFetch > 0, !channels.isEmpty,
              let binding = try? ObservationBinding(
                expectedGeneration: realizedPatch.generation,
                span: maximumFetch,
                pointBudget: maximumFetch,
                channels: channels
              )
        else { return nil }
        return .init(binding: binding, displayCounts: displayCounts)
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
            guard let displayCount = projection.displayCounts[channel.channel.key]
            else { continue }
            fresh[channel.channel.key] = ScopeTraceProjection.triggeredSamples(
                channel.values,
                displayCount: displayCount
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
