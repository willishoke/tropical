import SwiftUI

struct PatchNode: Identifiable {
    let id: String
    let kind: NodeKind
    var position: CGPoint
    /// Exact authored parameters. Numeric values are projected through
    /// `values` for the current knob UI, without erasing unknown JSON fields.
    var structuralParams: [String: JSONValue]
    /// inlet port → ordered source node ids
    var inputs: [String: [String]]
    /// Identity hue (degrees). Connections render as COLOR, not cables: an
    /// inlet wears the hues of its sources, so fan-out is the same color
    /// appearing at many inlets. Golden-angle spacing keeps neighbors apart.
    let hue: Double

    var values: [String: Double] {
        get { structuralParams.compactMapValues(\.doubleValue) }
        set {
            for (name, value) in newValue where value.isFinite {
                structuralParams[name] = .number(value)
            }
        }
    }

    var allInputs: [String] { inputs.values.flatMap { $0 } }

    var color: Color {
        Color(hue: hue / 360, saturation: 0.62, brightness: 0.88)
    }
}

/// The engine-acknowledged graph is separate from the mutable authored graph.
/// A failed or superseded compile never replaces either side with a hybrid.
struct RealizedPatchSnapshot: Equatable {
    let authoredRevision: Int
    let graph: JSONValue
    let patch: RealizedPatch

    var vocabularyFingerprint: String { patch.vocabularyFingerprint }
    var generation: EngineGeneration { patch.generation }
    var nodes: [RealizedNode] { patch.nodes }
    var inlets: [RealizedInlet] { patch.inlets }
    var taps: [ScopeTapBinding] { patch.taps }

    func parameterStatus(nodeID: String, port: String) -> RealizedParameterStatus {
        patch.parameterStatus(nodeID: nodeID, port: port)
    }
}

enum PatchCompileState: Equatable {
    case idle
    case compiling(authoredRevision: Int)
    case running(authoredRevision: Int, programVersion: Int?, controlVersion: Int?)
    case failed(authoredRevision: Int, message: String)
    case superseded(compiledRevision: Int, currentRevision: Int)
}

/// The patch graph + engine session, one object. TOPOLOGY RELOWERS, SCALARS
/// ARE LIVE: every continuous knob is a live `param:<id>.<knob>` module slot —
/// turning it drives `set_param` on the running kernel, no recompile. Only
/// structural edits (add/remove/rewire) change the graph topology and trigger
/// a relower + hot-swap.
@MainActor
final class PatchModel: ObservableObject {
    @Published var nodes: [String: PatchNode] = [:]
    @Published var order: [String] = []   // stable z/render order
    @Published var status = "booting engine…"
    @Published var statusIsError = false
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
    @Published var authoredRevision = 0
    @Published private(set) var realizedPatch: RealizedPatchSnapshot?
    @Published var compileState: PatchCompileState = .idle
    @Published var documentBlockers: [PatchDocumentBlocker] = []
    @Published var migrationReport: PatchMigrationReport?

    var documentTemplate: PatchDocumentV2?
    var outputNodeID: String?
    var requiresExplicitV2Save = false
    let themes = EngineNodeThemeCatalog()

    var availableKinds: [NodeKind] {
        guard let vocabulary else { return [] }
        return vocabulary.kinds.map {
            .engine(id: $0.kind, descriptor: $0, theme: themes.override(for: $0.kind))
        }
    }

    var monitorKinds: [NodeKind] { [.scope()] }

    /// Sources an inlet may legally take — the connection UI shows ONLY
    /// these (the type discipline enforced by omission: control inlets
    /// list knobs, modal inlets list modal nodes, cycle-formers absent).
    func legalSources(for nodeId: String, port: String) -> [PatchNode] {
        order.compactMap { nodes[$0] }
            .filter { canConnect(from: $0.id, to: nodeId, port: port) }
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
        let rowPitch = 8.0 * Grid.unit   // tallest module (6u) + breathing room
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
        var forward: (@Sendable (EngineEvent) -> Void)!
        let box = EventBox()
        forward = { event in Task { @MainActor in box.handler?(event) } }
        engine = Engine(events: forward)
        box.handler = { [weak self] event in self?.handle(event) }
        // Every exit path must reap the child engine or it orphans with the
        // DAC running (see AppDelegate).
        AppDelegate.onTerminate = { [engine] in engine.terminateChild() }

        startEngine()
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
            Task { await loadVocabularyAndBoot() }
        case .exit(let code):
            setStatus("engine exited (code \(code))", isError: true)
        case .stderr(let text):
            // Diagnostic only — the Electron app console.warn'd these.
            FileHandle.standardError.write(Data("[engine] \(text)".utf8))
        }
    }

    private func loadVocabularyAndBoot() async {
        do {
            let raw = try await engine.call("get_vocabulary")
            let decoded = try EngineVocabulary.decode(from: JSONEncoder().encode(raw))
            vocabulary = decoded
            if nodes.isEmpty {
                seedBootPatch()
                await pushGraph()
            }
            await syncAudioState()
            if realizedPatch == nil {
                setStatus("engine up · patch something", isError: false)
            }
        } catch {
            setStatus("vocabulary: \(error.localizedDescription)", isError: true)
        }
    }

    func setStatus(_ text: String, isError: Bool) {
        status = text
        statusIsError = isError
    }

    /// The patch you start with, and the one `New Patch` returns to.
    func seedBootPatch() {
        guard let vocabulary else { return }
        // The engine vocabulary determines both roles. No Swift kind name is
        // needed to construct a bootable authored graph.
        guard let sink = vocabulary.kinds.first(where: { $0.outlet == nil }) else {
            setStatus("vocabulary has no output sink", isError: true)
            return
        }
        let source = vocabulary.kinds.first {
            $0.outlet != nil && $0.inlets.isEmpty
        } ?? vocabulary.kinds.first { $0.outlet != nil }
        let sinkKind = NodeKind.engine(
            id: sink.kind, descriptor: sink, theme: themes.override(for: sink.kind))
        outputNodeID = addNode(sinkKind, at: CGPoint(x: 1180, y: 360), schedule: false)?.id
        if let source {
            let sourceKind = NodeKind.engine(
                id: source.kind, descriptor: source,
                theme: themes.override(for: source.kind))
            _ = addNode(sourceKind, at: CGPoint(x: 120, y: 200), schedule: false)
        }
        authoredRevision += 1
    }

    // ── Node lifecycle ───────────────────────────────────────────────────
    @discardableResult
    func addNode(
        _ kind: NodeKind,
        at point: CGPoint? = nil,
        schedule shouldSchedule: Bool = true
    ) -> PatchNode? {
        let spec = kind.spec
        if spec.fixed && nodes.values.contains(where: { $0.kind == kind }) { return nil }
        counter += 1
        let id = "\(kind.rawValue)\(counter)"
        let pos = Grid.snap(point ?? CGPoint(
            x: 80 + Double(counter % 6) * 34,
            y: 90 + Double(counter % 6) * 30))
        var structuralParams: [String: JSONValue] = [:]
        for k in spec.knobs { structuralParams[k.name] = .number(k.def) }
        var inputs: [String: [String]] = [:]
        for p in spec.inlets { inputs[p] = [] }
        let n = PatchNode(
            id: id, kind: kind, position: pos,
            structuralParams: structuralParams, inputs: inputs,
            hue: Double((counter * 137) % 360))
        nodes[id] = n
        order.append(id)
        autoArrangeIfOn()
        if shouldSchedule {
            authoredRevision += 1
            schedulePush()
        }
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
        authoredRevision += 1
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
        guard let vocabulary,
              fromId != toId,
              let from = nodes[fromId], let to = nodes[toId],
              let sourceKind = from.kind.engineID,
              !(to.inputs[port] ?? []).contains(fromId)
        else { return false }
        if to.kind.isMonitor {
            return to.kind.spec.inlets.contains(port)
                && (vocabulary.descriptor(for: sourceKind)?.outlet != nil || fromId == outputNodeID)
                && (to.inputs[port]?.isEmpty ?? true)
        }
        guard let destinationKind = to.kind.engineID else { return false }
        guard vocabulary.canConnect(
            from: sourceKind,
            to: destinationKind,
            port: port,
            existingConnectionCount: to.inputs[port]?.count ?? 0)
        else { return false }
        var seen = Set<String>()
        if reaches(fromId, toId, &seen) { return false }   // would form a cycle
        return true
    }

    func connect(from fromId: String, to toId: String, port: String) {
        guard canConnect(from: fromId, to: toId, port: port), var to = nodes[toId] else { return }
        if to.kind.isMonitor || to.kind.descriptor?.port(named: port)?.multi != true {
            to.inputs[port] = []
        }
        to.inputs[port, default: []].append(fromId)
        nodes[toId] = to
        autoArrangeIfOn()
        authoredRevision += 1
        schedulePush()
    }

    func deleteEdge(to toId: String, port: String, from fromId: String) {
        guard var to = nodes[toId] else { return }
        to.inputs[port]?.removeAll { $0 == fromId }
        nodes[toId] = to
        autoArrangeIfOn()
        authoredRevision += 1
        schedulePush()
    }

    // ── Serialization → load_patch_graph ─────────────────────────────────
    /// Exact, deterministic tap request. The final output already has
    /// `__root__.out`, so only monitored intermediate engine nodes are listed.
    var requestedTapNodeIDs: [String] {
        var result: [String] = []
        var seen = Set<String>()
        for id in order {
            guard let monitor = nodes[id], monitor.kind.isMonitor else { continue }
            for port in monitor.kind.spec.inlets {
                for source in monitor.inputs[port] ?? []
                    where source != outputNodeID && nodes[source]?.kind.engineID != nil
                {
                    if seen.insert(source).inserted { result.append(source) }
                }
            }
        }
        return result
    }

    func serialize() -> JSONValue {
        var out: [JSONValue] = []
        for id in order {
            guard let n = nodes[id], let kind = n.kind.engineID else { continue }
            let inputs: [String: JSONValue] = n.inputs.mapValues {
                JSONValue.array($0.map(JSONValue.string))
            }
            out.append(.object([
                "id": .string(n.id),
                "kind": .string(kind.rawValue),
                "params": .object(n.structuralParams),
                "sel": .object([:]),
                "in": .object(inputs),
            ]))
        }
        return .object([
            "nodes": .array(out),
            "out": outputNodeID.map(JSONValue.string) ?? .null,
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
        guard vocabulary != nil else { return }
        guard let authoredDocument = try? makeDocument() else { return }
        let validation = authoredDocument.validate(against: vocabulary)
        documentBlockers = validation.blockers
        guard validation.canCompile else {
            compileState = .failed(
                authoredRevision: authoredRevision,
                message: validation.blockers.first?.detail ?? "document is not compilable")
            setStatus(
                "blocked · \(validation.blockers.count) authored issue\(validation.blockers.count == 1 ? "" : "s")",
                isError: true)
            return
        }
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
        if graph == lastPushed {
            if let realizedPatch {
                self.realizedPatch = RealizedPatchSnapshot(
                    authoredRevision: revision,
                    graph: realizedPatch.graph,
                    patch: realizedPatch.patch)
                compileState = .running(
                    authoredRevision: revision,
                    programVersion: Int(exactly: realizedPatch.generation.programVersion),
                    controlVersion: Int(exactly: realizedPatch.generation.controlVersion))
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
            let adoption = try await engine.loadPatchGraph(graph)
            guard case .adopted(let patch) = adoption else {
                switch adoption {
                case .superseded:
                    compileState = .superseded(
                        compiledRevision: revision, currentRevision: authoredRevision)
                    setStatus("compile superseded · running patch unchanged", isError: false)
                case .stale:
                    compileState = .superseded(
                        compiledRevision: revision, currentRevision: authoredRevision)
                    setStatus("stale compile ignored · running patch unchanged", isError: false)
                case .adopted:
                    break
                }
                return
            }
            lastPushed = graph
            realizedPatch = RealizedPatchSnapshot(
                authoredRevision: revision, graph: graph, patch: patch)
            scopeTapSlots = Dictionary(
                patch.taps.map { ($0.name, $0.slot) },
                uniquingKeysWith: { first, _ in first })
            let programVersion = Int(exactly: patch.generation.programVersion)
            let controlVersion = Int(exactly: patch.generation.controlVersion)
            // A relower transfers slots by name, so a live scrub survives it —
            // except a COLD first compile, which seeds master.velocity at its
            // default (1). Re-apply a non-default scrub so the slider and the
            // running clock agree (no-op if equal).
            if velocity != 1 { params.send("master.velocity", velocity) }
            if revision == authoredRevision {
                compileState = .running(
                    authoredRevision: revision,
                    programVersion: programVersion,
                    controlVersion: controlVersion)
                setStatus(audioOn ? "playing" : "compiled", isError: false)
            } else {
                compileState = .superseded(
                    compiledRevision: revision, currentRevision: authoredRevision)
                setStatus("compiled revision superseded by edits", isError: false)
            }
        } catch {
            compileState = .failed(
                authoredRevision: revision, message: error.localizedDescription)
            setStatus("compile: \(error.localizedDescription)", isError: true)
        }
    }

    // ── Live param drive ──────────────────────────────────────────────────
    lazy var params = ParamSender(model: self)

    func sendKnob(_ node: PatchNode, _ knob: KnobSpec, _ value: Double) {
        // A monitor's knobs (Scope `window`) are view state, not param slots.
        guard !node.kind.spec.monitor,
              let realizedPatch,
              realizedPatch.authoredRevision == authoredRevision,
              case .live = realizedPatch.parameterStatus(
                nodeID: node.id, port: knob.name)
        else { return }
        let name = "\(node.id).\(knob.name)"
        params.send(name, value)
    }

    func setVelocity(_ v: Double) {
        velocity = v
        params.send("master.velocity", v)
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
    // Compile latency is one failure mode; RUNTIME cost is the other, and
    // it's the one that's silent. A heavy kernel (a modal reverb: ~200 modes
    // × transcendentals per sample) can blow the ~11.6 ms buffer deadline
    // (512 frames @ 44.1k), and RtAudio responds to a missed deadline by
    // emitting a zero buffer → "no sound" with a perfectly healthy compile.
    // The DAC already counts it (`audio_status`); poll it so the meter reads
    // "load 130% · ⚠ xruns" instead of leaving you guessing.
    private static let bufferMs = (512.0 / 44100.0) * 1000   // ≈ 11.61 ms deadline
    private var telemetryTask: Task<Void, Never>?
    private var telemetryInFlight = false
    private var underrunBase = 0

    func stopTelemetry() {
        telemetryTask?.cancel()
        telemetryTask = nil
    }

    func startTelemetry() {
        stopTelemetry()
        telemetryTask = Task { [weak self] in
            if let self {
                let st = try? await self.engine.call("audio_status")
                self.underrunBase = Int(st?["stats"]?["underrunCount"]?.doubleValue ?? 0)
            }
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(750))
                guard !Task.isCancelled else { return }
                await self?.pollTelemetry()
            }
        }
    }

    // ── Scope taps ────────────────────────────────────────────────────────
    // A scope channel names a NODE; the engine names a SLOT. An adopted
    // compile handshake atomically replaces this binding table (a node that
    // failed to lower has no tap and its trace goes dark). The final mix is
    // always `__root__.out`, so it needs no requested tap.
    private var scopeTapSlots: [String: String] = [:]

    /// Latest triggered trace per scope channel, keyed "<scopeId>.<port>".
    @Published var scopeTraces: [String: [Double]] = [:]

    func scopeSlot(for sourceId: String) -> String? {
        if sourceId == outputNodeID { return "__root__.out" }
        return scopeTapSlots[sourceId]
    }

    // ── Scope poll loop ───────────────────────────────────────────────────
    // `render_window` + `playback_position` are data-plane methods: answered
    // synchronously in C++ off the audio thread, never queued behind the Lean
    // control thread — so the trace stays live through a compile and works
    // with the transport stopped (the kernel is closed-form; a frozen clock
    // just yields a still waveform). One serial loop, so a slow frame can
    // never pile up requests.
    static let sampleRate = 44100.0
    private var scopeTask: Task<Void, Never>?

    private func startScopePolling() {
        scopeTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(33))
                guard !Task.isCancelled else { return }
                await self?.pollScopes()
            }
        }
    }

    private func pollScopes() async {
        let scopes = order.compactMap { nodes[$0] }
            .filter { $0.kind.isMonitor && !$0.allInputs.isEmpty }
        guard !scopes.isEmpty else {
            if !scopeTraces.isEmpty { scopeTraces = [:] }
            return
        }
        guard let pos = try? await engine.call("playback_position"),
              let position = pos["position"]?.doubleValue else { return }
        var fresh: [String: [Double]] = [:]
        for scope in scopes {
            let window = scope.values["window"] ?? 0.02
            let count = min(8192, max(128, Int(window * Self.sampleRate)))
            // Fetch 2× the display span so the trigger can align a full
            // window ending at the live head.
            let fetch = min(16384, count * 2)
            let start = max(0, Int(position) - fetch)
            var slots: [String] = [], keys: [String] = []
            for port in scope.kind.spec.inlets {
                guard let src = scope.inputs[port]?.first,
                      let slot = scopeSlot(for: src) else { continue }
                slots.append(slot)
                keys.append("\(scope.id).\(port)")
            }
            guard !slots.isEmpty,
                  let r = try? await engine.call("render_window", .object([
                      "start": .number(Double(start)),
                      "count": .number(Double(fetch)),
                      "slots": .array(slots.map(JSONValue.string)),
                  ])),
                  case .array(let chans)? = r["values"] else { continue }
            for (i, key) in keys.enumerated() {
                guard i < chans.count, case .array(let raw) = chans[i] else { continue }
                let samples = raw.compactMap(\.doubleValue)
                let t = Self.triggerIndex(samples, display: count)
                fresh[key] = Array(samples[t..<min(t + count, samples.count)])
            }
        }
        scopeTraces = fresh
    }

    /// Rising level-trigger with hysteresis (the playground scope's port of
    /// tui/scope/trigger.ts): arm below −h, fire on the first climb past +h,
    /// searching only far enough back that a full display window remains.
    static func triggerIndex(_ s: [Double], display: Int) -> Int {
        let searchEnd = s.count - display
        guard searchEnd > 0 else { return 0 }
        let peak = s.reduce(0) { max($0, abs($1)) }
        let h = peak * 0.05
        guard h > 0 else { return 0 }
        var armed = false
        for i in 0..<searchEnd {
            if s[i] < -h { armed = true }
            else if armed && s[i] >= h { return i }
        }
        return 0
    }

    private func pollTelemetry() async {
        // Single-in-flight: the control thread is shared with compiles and
        // knobs, so if a poll is stuck behind a long op we must NOT keep
        // enqueuing more — that turns a slow compile into a pile-up.
        if !audioOn || pushInFlight || telemetryInFlight || statusIsError { return }
        telemetryInFlight = true
        defer { telemetryInFlight = false }
        guard let st = try? await engine.call("audio_status"),
              let stats = st["stats"], stats != .null,
              let maxMs = stats["maxCallbackMs"]?.doubleValue else { return }
        let pct = Int((maxMs / Self.bufferMs * 100).rounded())
        let xruns = max(0, Int(stats["underrunCount"]?.doubleValue ?? 0) - underrunBase)
        let over = pct > 90 || xruns > 0
        let xrunText = xruns > 0 ? " · ⚠ \(xruns) xrun\(xruns > 1 ? "s" : "") — dropout" : ""
        setStatus("playing · load \(pct)%\(xrunText)", isError: over)
    }
}

/// Breaks the init-order knot between the Engine's @Sendable event callback
/// and the @MainActor model that wants to receive the events.
private final class EventBox: @unchecked Sendable {
    @MainActor var handler: ((EngineEvent) -> Void)?
}

/// Live param drive: set the `param:<name>` slot on the running kernel — no
/// recompile. Rapid drags coalesce to the latest value per name; a cold-start
/// miss (slot not compiled yet) falls back to a relower that bakes the value
/// as the slot default, after which subsequent turns are live.
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
                    self?.model?.schedulePush()
                    return
                }
            }
        }
    }
}
