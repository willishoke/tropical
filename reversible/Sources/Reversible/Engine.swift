import Foundation

/// PID of the live child engine, or 0. Read from an async-signal handler on a
/// fatal crash (see reapEngineOnCrash), so it must be a `sig_atomic_t`, not a
/// lock-guarded value — locking isn't async-signal-safe. Written only on
/// spawn and reap.
var gEngineChildPID: sig_atomic_t = 0

/// Lifecycle/diagnostic events from the engine process — mirrors the
/// Electron main-process `onStatus` channel.
enum EngineEvent {
    case up
    case stderr(String)
    case exit(Int32)
}

enum EngineError: Error, LocalizedError {
    case binaryNotFound([String])
    case notRunning
    case exited
    case timeout(method: String)
    case pendingLimit(lane: String, limit: Int)
    case rpc(String)
    case engine(code: String, message: String)
    case compileSuperseded(activeGeneration: EngineGeneration?)
    case staleCompile(candidate: EngineGeneration, active: EngineGeneration)

    var errorDescription: String? {
        switch self {
        case .binaryNotFound(let paths):
            return "Tropical engine not found. Checked: \(paths.joined(separator: ", "))"
        case .notRunning: return "engine not started"
        case .exited: return "engine exited"
        case .timeout(let m): return "timeout: \(m)"
        case .pendingLimit(let lane, let limit):
            return "RPC \(lane) has \(limit) pending requests"
        case .rpc(let s): return "RPC \(s)"
        case .engine(let code, let message): return "\(code): \(message)"
        case .compileSuperseded(let active):
            if let active {
                return "compile superseded before publication; running program \(active.programVersion) unchanged"
            }
            return "compile superseded before any program was published"
        case .staleCompile(let candidate, let active):
            return "stale compile program \(candidate.programVersion) ignored; running program \(active.programVersion) is newer"
        }
    }
}

// MARK: - Atomic compile handshake

/// The immutable runtime image published by one successful compile. Program
/// identity dominates; control identity names the initial control snapshot
/// created in the same publication.
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

/// Surface truth for an authored parameter. Excluded nodes are inactive even
/// if the compiler report retains their authored value as diagnostic data.
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

/// Everything a client needs to replace its last realized snapshot. No field
/// is filled from authored graph guesses or a second RPC.
struct RealizedPatch: Equatable, Sendable {
    let vocabularyFingerprint: String
    let generation: EngineGeneration
    let nodes: [RealizedNode]
    let inlets: [RealizedInlet]
    let liveParameters: [RealizedLiveParameter]
    let taps: [ScopeTapBinding]

    func node(id: String) -> RealizedNode? {
        nodes.first { $0.id == id }
    }

    func inlet(nodeID: String, port: String) -> RealizedInlet? {
        inlets.first { $0.nodeID == nodeID && $0.port == port }
    }

    func liveParameter(named name: String) -> RealizedLiveParameter? {
        liveParameters.first { $0.name == name }
    }

    func parameterStatus(named name: String) -> RealizedParameterStatus {
        guard let parameter = liveParameter(named: name) else { return .structural }
        return .live(discipline: parameter.discipline)
    }

    func parameterStatus(nodeID: String, port: String) -> RealizedParameterStatus {
        guard node(id: nodeID)?.status == .active else { return .inactive }
        return parameterStatus(named: "\(nodeID).\(port)")
    }

    /// Wire-compatible diagnostic response for legacy callers. Engine serves
    /// this from the adopted handshake rather than issuing list_scope_taps.
    var legacyTapList: JSONValue {
        .object([
            "taps": .array(taps.map { tap in
                .object([
                    "name": .string(tap.name),
                    "instance": .string(tap.instance),
                    "output": .string(tap.output),
                    "slot": .string(tap.slot),
                ])
            }),
        ])
    }
}

enum PatchCompileResponse: Equatable, Sendable {
    case published(RealizedPatch)
    /// The server coalesced this queued authored snapshot; no runtime image was
    /// published and the previous realized generation remains authoritative.
    case superseded

    static func decode(_ value: JSONValue) throws -> PatchCompileResponse {
        guard case .object(let object) = value else {
            throw RealizedPatchDecodeError.invalidField("response")
        }
        if object["superseded"]?.boolValue == true { return .superseded }
        guard object["ok"]?.boolValue == true else {
            throw RealizedPatchDecodeError.invalidField("ok")
        }

        let fingerprint = try requiredString(
            object, "vocabulary_fingerprint", path: "vocabulary_fingerprint"
        )
        let generation = EngineGeneration(
            programVersion: try requiredVersion(object, "program_version"),
            controlVersion: try requiredVersion(object, "control_version")
        )
        guard generation.programVersion > 0, generation.controlVersion > 0 else {
            throw RealizedPatchDecodeError.invalidField("generation")
        }

        let nodeValues = try requiredArray(object, "nodes")
        var nodes: [RealizedNode] = []
        var nodeIDs = Set<String>()
        for (index, value) in nodeValues.enumerated() {
            guard case .object(let fields) = value else {
                throw RealizedPatchDecodeError.invalidField("nodes[\(index)]")
            }
            let id = try requiredString(fields, "id", path: "nodes[\(index)].id")
            let rawKind = try requiredString(fields, "kind", path: "nodes[\(index)].kind")
            let rawStatus = try requiredString(fields, "status", path: "nodes[\(index)].status")
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
        for (index, value) in inletValues.enumerated() {
            guard case .object(let fields) = value else {
                throw RealizedPatchDecodeError.invalidField("inputs[\(index)]")
            }
            let nodeID = try requiredString(fields, "node", path: "inputs[\(index)].node")
            let port = try requiredString(fields, "port", path: "inputs[\(index)].port")
            let rawState = try requiredString(fields, "state", path: "inputs[\(index)].state")
            let key = "\(nodeID)\u{0}\(port)"
            guard inletKeys.insert(key).inserted else {
                throw RealizedPatchDecodeError.duplicate("inlet '\(nodeID).\(port)'")
            }
            let state: RealizedInletState
            switch rawState {
            case "normalled":
                guard fields["sources"] == nil else {
                    throw RealizedPatchDecodeError.invalidField("inputs[\(index)].sources")
                }
                state = .normalled
            case "wired":
                let sourceValues = try requiredArray(fields, "sources")
                let sources = try sourceValues.enumerated().map { sourceIndex, source in
                    guard let id = source.stringValue, !id.isEmpty else {
                        throw RealizedPatchDecodeError.invalidField(
                            "inputs[\(index)].sources[\(sourceIndex)]"
                        )
                    }
                    return id
                }
                guard !sources.isEmpty else {
                    throw RealizedPatchDecodeError.invalidField("inputs[\(index)].sources")
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
        for (index, value) in parameterValues.enumerated() {
            guard case .object(let fields) = value else {
                throw RealizedPatchDecodeError.invalidField("params[\(index)]")
            }
            let name = try requiredString(fields, "name", path: "params[\(index)].name")
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
        var tapNames = Set<String>(), tapSlots = Set<String>()
        for (index, value) in tapValues.enumerated() {
            guard case .object(let fields) = value else {
                throw RealizedPatchDecodeError.invalidField("taps[\(index)]")
            }
            let name = try requiredString(fields, "name", path: "taps[\(index)].name")
            let instance = try requiredString(fields, "instance", path: "taps[\(index)].instance")
            let output = try requiredString(fields, "output", path: "taps[\(index)].output")
            let slot = try requiredString(fields, "slot", path: "taps[\(index)].slot")
            guard slot == "\(instance).\(output)" else {
                throw RealizedPatchDecodeError.invalidField("taps[\(index)].slot")
            }
            guard tapNames.insert(name).inserted else {
                throw RealizedPatchDecodeError.duplicate("tap name '\(name)'")
            }
            guard tapSlots.insert(slot).inserted else {
                throw RealizedPatchDecodeError.duplicate("tap slot '\(slot)'")
            }
            taps.append(.init(name: name, instance: instance, output: output, slot: slot))
        }

        return .published(.init(
            vocabularyFingerprint: fingerprint,
            generation: generation,
            nodes: nodes,
            inlets: inlets,
            liveParameters: parameters,
            taps: taps
        ))
    }
}

enum RealizedPatchDecodeError: Error, Equatable, LocalizedError {
    case missingField(String)
    case invalidField(String)
    case duplicate(String)

    var errorDescription: String? {
        switch self {
        case .missingField(let path): return "compile handshake missing '\(path)'"
        case .invalidField(let path): return "compile handshake has invalid '\(path)'"
        case .duplicate(let fact): return "compile handshake repeats \(fact)"
        }
    }
}

private func requiredString(
    _ object: [String: JSONValue], _ key: String, path: String
) throws -> String {
    guard let value = object[key] else { throw RealizedPatchDecodeError.missingField(path) }
    guard let string = value.stringValue,
          !string.isEmpty,
          string == string.trimmingCharacters(in: .whitespacesAndNewlines)
    else { throw RealizedPatchDecodeError.invalidField(path) }
    return string
}

private func requiredArray(
    _ object: [String: JSONValue], _ key: String
) throws -> [JSONValue] {
    guard let value = object[key] else { throw RealizedPatchDecodeError.missingField(key) }
    guard case .array(let array) = value else {
        throw RealizedPatchDecodeError.invalidField(key)
    }
    return array
}

private func requiredVersion(
    _ object: [String: JSONValue], _ key: String
) throws -> UInt64 {
    guard let value = object[key] else { throw RealizedPatchDecodeError.missingField(key) }
    // JSONValue currently stores a Double. Refuse values beyond its exact
    // integer range rather than silently adopting a rounded generation.
    guard let number = value.doubleValue,
          number.isFinite,
          number >= 0,
          number <= 9_007_199_254_740_991,
          number.rounded(.towardZero) == number,
          let version = UInt64(exactly: number)
    else { throw RealizedPatchDecodeError.invalidField(key) }
    return version
}

enum CompileAdoption: Equatable, Sendable {
    case adopted(RealizedPatch)
    case superseded(activeGeneration: EngineGeneration?)
    case stale(candidate: EngineGeneration, active: EngineGeneration)
}

/// Generation guard shared by typed and legacy callers. Failed, superseded,
/// and late responses never replace the last successfully published facts.
struct RealizedPatchStore: Sendable {
    private(set) var current: RealizedPatch?

    mutating func adopt(_ response: PatchCompileResponse) -> CompileAdoption {
        switch response {
        case .superseded:
            return .superseded(activeGeneration: current?.generation)
        case .published(let candidate):
            if let current, candidate.generation <= current.generation {
                return .stale(candidate: candidate.generation, active: current.generation)
            }
            current = candidate
            return .adopted(candidate)
        }
    }
}

enum EngineRPCLane: String, Sendable {
    case graph
    case control
    case scope
    case telemetry

    static func lane(for method: String) -> EngineRPCLane {
        switch method {
        case "set_param": return .control
        case "render_window", "playback_position": return .scope
        case "get_telemetry", "audio_status": return .telemetry
        default: return .graph
        }
    }
}

/// Owns the native `frontend` binary (the Lean compiler + session + runtime
/// FFI) over its Unix-socket JSON-RPC surface (`--serve`). Audio plays
/// out of the HOST's native device (RtAudio); the app is purely a control
/// surface.
///
/// Transport mirrors playground/main.js: one JSON object per line, replies
/// matched by `id`. One supervisor owns the process/socket namespace, while
/// graph, control, scope, and telemetry each use an independent connection
/// and bounded request table. The server's C++ data-plane methods therefore
/// never sit behind a compile response's bytes on the client. Control-plane
/// replies wrap an inner {status,data} in result.content[0].text; data-plane
/// replies are a plain `result`. Each lane unwraps both forms.
actor Engine {
    private var process: Process?
    private var sockPath = ""
    /// The child engine OUTLIVES the app unless someone reaps it — an orphaned
    /// engine keeps the DAC (and whatever it was playing) alive forever. The
    /// box holds the Process outside actor isolation so the app-termination
    /// path (a synchronous, can't-await context) can always reach it.
    private nonisolated let procBox = ProcessBox()
    private var graphRPC: RPCConnection?
    private var controlRPC: RPCConnection?
    private var scopeRPC: RPCConnection?
    private var telemetryRPC: RPCConnection?
    private var realizedPatches = RealizedPatchStore()
    private let events: @Sendable (EngineEvent) -> Void

    /// Resolution order: bundled engine, then an explicit developer/test
    /// override. Development launchers set the override from their own repo
    /// location; production never guesses a checkout under the user's home.
    static func resolveBinary() throws -> URL {
        var checked: [URL] = []
        if let resources = Bundle.main.resourceURL {
            let bundled = resources
                .appendingPathComponent("Tropical", isDirectory: true)
                .appendingPathComponent("frontend", isDirectory: false)
            checked.append(bundled)
            if FileManager.default.isExecutableFile(atPath: bundled.path) {
                return bundled
            }
        }
        if let p = ProcessInfo.processInfo.environment["TROPICAL_ENGINE_BIN"] {
            let explicit = URL(fileURLWithPath: p).standardizedFileURL
            checked.append(explicit)
            if FileManager.default.isExecutableFile(atPath: explicit.path) {
                return explicit
            }
        }
        throw EngineError.binaryNotFound(checked.map(\.path))
    }

    static func workingDirectory(for binary: URL) -> URL {
        if let resources = Bundle.main.resourceURL {
            let tropicalResources = resources.appendingPathComponent(
                "Tropical", isDirectory: true
            ).standardizedFileURL
            if binary.standardizedFileURL.path.hasPrefix(tropicalResources.path + "/") {
                return tropicalResources
            }
        }
        // Explicit developer override points at
        // <repo>/lean/.lake/build/bin/frontend.
        return binary.deletingLastPathComponent()  // bin/
            .deletingLastPathComponent()            // build/
            .deletingLastPathComponent()            // .lake/
            .deletingLastPathComponent()            // lean/
            .deletingLastPathComponent()            // repo root
    }

    init(events: @escaping @Sendable (EngineEvent) -> Void) {
        self.events = events
    }

    func start() async throws {
        realizedPatches = RealizedPatchStore()
        // Reap any engine orphaned by a PREVIOUS instance that died before
        // teardown (SIGKILL, crash) — the one gap the graceful path can't close.
        Engine.sweepOrphans()
        let bin = try Engine.resolveBinary()
        // The engine resolves production assets relative to this owned root.
        let root = Engine.workingDirectory(for: bin)
        // sockaddr_un caps the path at ~104 bytes, so the per-user temp dir
        // (short on macOS) rather than the app scratch dir.
        sockPath = FileManager.default.temporaryDirectory
            .appendingPathComponent("reversible-\(ProcessInfo.processInfo.processIdentifier).sock").path
        let p = Process()
        p.executableURL = bin
        p.arguments = ["--serve", sockPath]
        p.currentDirectoryURL = root
        // The arrow patch-graph path does not route through syncCompile, but we
        // set the var anyway so any session-path compile in this process also
        // takes the session → resolved-root (arrow) path. Inert if ignored.
        var env = ProcessInfo.processInfo.environment
        env["TROPICAL_ARROW"] = "1"
        p.environment = env

        let outPipe = Pipe(), errPipe = Pipe()
        p.standardInput = FileHandle.nullDevice   // serve mode never reads stdin
        p.standardOutput = outPipe
        p.standardError = errPipe

        let events = self.events
        // Serve mode replies on the socket; anything on stdout/stderr is
        // diagnostic (e.g. "serving on <addr>").
        for pipe in [outPipe, errPipe] {
            pipe.fileHandleForReading.readabilityHandler = { h in
                let data = h.availableData
                guard !data.isEmpty, let s = String(data: data, encoding: .utf8) else { return }
                events(.stderr(s))
            }
        }
        p.terminationHandler = { [weak self] proc in
            events(.exit(proc.terminationStatus))
            Task { await self?.childDidExit() }
        }

        try p.run()
        process = p
        procBox.process = p
        gEngineChildPID = sig_atomic_t(p.processIdentifier)
        do {
            try await connectLanes()
        } catch {
            terminateChild()
            await closeLanes(error: error)
            throw error
        }
        events(.up)
    }

    /// Synchronous child teardown for app exit (Cmd-Q, SIGTERM). Safe from
    /// any thread; idempotent.
    nonisolated func terminateChild() {
        gEngineChildPID = 0
        procBox.take()?.terminate()
    }

    /// Reap engines orphaned by a PREVIOUS reversible instance that died
    /// without running teardown (the app was SIGKILLed, or crashed before the
    /// crash handler fired). A socket name embeds the OWNER app's pid
    /// (`reversible-<pid>.sock`), so an orphan is an engine whose owner is no
    /// longer alive. We skip live siblings (owner still running) and can't
    /// touch our own engine (not spawned yet at sweep time). Best-effort: any
    /// parse/exec failure just leaves the sweep a no-op.
    nonisolated static func sweepOrphans() {
        let ps = Process()
        ps.executableURL = URL(fileURLWithPath: "/bin/ps")
        ps.arguments = ["-axo", "pid=,command="]
        let pipe = Pipe()
        ps.standardOutput = pipe
        ps.standardError = FileHandle.nullDevice
        guard (try? ps.run()) != nil else { return }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        ps.waitUntilExit()
        guard let out = String(data: data, encoding: .utf8) else { return }

        let me = getpid()
        for raw in out.split(separator: "\n") {
            let line = String(raw).trimmingCharacters(in: .whitespaces)
            guard line.range(of: "frontend --serve") != nil,
                  let sockRange = line.range(
                    of: #"/\S*reversible-[0-9]+\.sock"#, options: .regularExpression)
            else { continue }
            let sockPath = String(line[sockRange])
            // Owner pid = the run of digits in `reversible-<pid>.sock`.
            let owner = pid_t(sockPath
                .drop(while: { !$0.isNumber })
                .prefix(while: { $0.isNumber })) ?? -1
            guard owner > 0, owner != me else { continue }
            // owner reachable (alive, or alive-but-not-ours) → a live instance
            // still owns this engine; leave it be.
            if Darwin.kill(owner, 0) == 0 || errno == EPERM { continue }
            // Engine's own pid is the first ps column.
            guard let sp = line.firstIndex(of: " "),
                  let childPID = pid_t(line[..<sp]) else { continue }
            Darwin.kill(childPID, SIGKILL)
            unlink(sockPath)
        }
    }

    /// The socket node appears once Engine.boot finishes and binds. Establish
    /// the compiler lane while checking child liveness, then attach the three
    /// independent data lanes to the ready server.
    private func connectLanes() async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(20))
        while ContinuousClock.now < deadline {
            guard process?.isRunning == true else { throw EngineError.exited }
            if let fd = RPCConnection.connectOnce(path: sockPath) {
                graphRPC = RPCConnection(lane: .graph, fileDescriptor: fd)
                controlRPC = try await RPCConnection.connect(
                    lane: .control, path: sockPath
                )
                scopeRPC = try await RPCConnection.connect(
                    lane: .scope, path: sockPath
                )
                telemetryRPC = try await RPCConnection.connect(
                    lane: .telemetry, path: sockPath
                )
                return
            }
            try await Task.sleep(for: .milliseconds(100))
        }
        throw EngineError.timeout(method: "connect \(sockPath)")
    }

    func kill() async {
        gEngineChildPID = 0
        process?.terminationHandler = nil
        process?.terminate()
        process = nil
        procBox.take()
        realizedPatches = RealizedPatchStore()
        await closeLanes(error: EngineError.exited)
        unlink(sockPath)
    }

    private func childDidExit() async {
        gEngineChildPID = 0
        process = nil
        procBox.take()
        realizedPatches = RealizedPatchStore()
        await closeLanes(error: EngineError.exited)
        unlink(sockPath)
    }

    private func closeLanes(error: Error) async {
        let lanes = [graphRPC, controlRPC, scopeRPC, telemetryRPC]
        graphRPC = nil
        controlRPC = nil
        scopeRPC = nil
        telemetryRPC = nil
        for lane in lanes { await lane?.close(error: error) }
    }

    /// Typed compiler lane. A superseded request is an explicit non-publication;
    /// a late generation is reported but cannot roll the active snapshot back.
    @discardableResult
    func loadPatchGraph(_ graph: JSONValue) async throws -> CompileAdoption {
        let raw = try await callRaw("load_patch_graph", graph)
        return realizedPatches.adopt(try PatchCompileResponse.decode(raw))
    }

    func currentRealizedPatch() -> RealizedPatch? {
        realizedPatches.current
    }

    @discardableResult
    func call(_ method: String, _ params: JSONValue = .object([:])) async throws -> JSONValue {
        if method == "list_scope_taps", let current = realizedPatches.current {
            // Compatibility only: this is the tap table from the atomic compile
            // response, not a second server request that can cross a hot-swap.
            return current.legacyTapList
        }
        let result = try await callRaw(method, params)
        if method == "load_patch_graph" {
            switch realizedPatches.adopt(try PatchCompileResponse.decode(result)) {
            case .adopted:
                break
            case .superseded(let activeGeneration):
                throw EngineError.compileSuperseded(activeGeneration: activeGeneration)
            case .stale(let candidate, let active):
                throw EngineError.staleCompile(candidate: candidate, active: active)
            }
        }
        return result
    }

    private func callRaw(
        _ method: String,
        _ params: JSONValue = .object([:])
    ) async throws -> JSONValue {
        let connection: RPCConnection? = switch EngineRPCLane.lane(for: method) {
        case .graph: graphRPC
        case .control: controlRPC
        case .scope: scopeRPC
        case .telemetry: telemetryRPC
        }
        guard let connection else { throw EngineError.notRunning }
        return try await connection.call(method, params)
    }
}

/// Cross-isolation handle to the child process for the synchronous
/// termination path. `take()` swaps out under the lock so terminate is
/// sent at most once.
private final class ProcessBox: @unchecked Sendable {
    private let lock = NSLock()
    private var proc: Process?

    var process: Process? {
        get { lock.withLock { proc } }
        set { lock.withLock { proc = newValue } }
    }

    @discardableResult
    func take() -> Process? {
        lock.withLock { let p = proc; proc = nil; return p }
    }
}
