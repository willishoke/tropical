import Foundation

/// PID of the live child engine, or 0. Read from an async-signal handler on a
/// fatal crash (see reapEngineOnCrash), so it must be a `sig_atomic_t`, not a
/// lock-guarded value — locking isn't async-signal-safe. Written only on
/// spawn and reap.
nonisolated(unsafe) var gEngineChildPID: sig_atomic_t = 0

/// Lifecycle/diagnostic events from the engine process — mirrors the
/// Electron main-process `onStatus` channel.
enum EngineEvent: Sendable {
    case up
    case stderr(String)
    case exit(Int32)
}

enum EngineError: Error, LocalizedError, Sendable {
    case binaryNotFound([String])
    case notRunning
    case exited
    case timeout(method: String)
    case pendingLimit(lane: String, limit: Int)
    case rpc(String)
    case engine(code: String, message: String)

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
        case "set_param", "playback_position": return .control
        case "render_window": return .scope
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

    static func childEnvironment(
        inherited: [String: String]
    ) -> [String: String] {
        var environment = inherited
        environment["TROPICAL_ARROW"] = "1"
        if environment["TROPICAL_BACKEND"] == nil {
            environment["TROPICAL_BACKEND"] = "metal"
        }
        return environment
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
        // Preserve explicit developer overrides while selecting the qualified
        // live backend for a normal product launch.
        let env = Engine.childEnvironment(
            inherited: ProcessInfo.processInfo.environment
        )
        // Reversible is the live product surface for the routed modal backend.
        // Its heavyweight two-room kernels are qualified for Metal and exceed
        // the real-time deadline on the scalar f64 JIT. Keep an explicit
        // developer override (`TROPICAL_BACKEND=jit`) for diagnosis, but do not
        // silently launch the product on the reference backend.
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
            guard let engine = self else { return }
            Task { await engine.childDidExit() }
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

    func loadVocabulary() async throws -> EngineVocabulary {
        let raw = try await callRaw("get_vocabulary")
        let data = try JSONEncoder().encode(raw)
        return try EngineVocabulary.decode(from: data)
    }

    /// Typed compiler lane. The expected vocabulary and ordered tap names come
    /// from the model's already-adopted authored semantics; a mismatch is
    /// rejected before the engine-side realized store advances.
    @discardableResult
    func loadPatchGraph(
        _ graph: JSONValue,
        expectedTapNames: [String],
        vocabularyFingerprint: String
    ) async throws -> CompileAdoption {
        let raw = try await callRaw("load_patch_graph", graph)
        let response = try PatchCompileResponse.decode(
            raw,
            expectedTapNames: expectedTapNames,
            expectedVocabularyFingerprint: vocabularyFingerprint
        )
        return realizedPatches.adopt(response)
    }

    func currentRealizedPatch() -> RealizedPatch? {
        realizedPatches.current
    }

    @discardableResult
    func call(_ method: String, _ params: JSONValue = .object([:])) async throws -> JSONValue {
        try await callRaw(method, params)
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
