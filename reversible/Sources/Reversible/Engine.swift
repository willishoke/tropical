import Foundation

/// Lifecycle/diagnostic events from the engine process — mirrors the
/// Electron main-process `onStatus` channel.
enum EngineEvent {
    case up
    case stderr(String)
    case exit(Int32)
}

enum EngineError: Error, LocalizedError {
    case notRunning
    case exited
    case timeout(method: String)
    case rpc(String)
    case engine(code: String, message: String)

    var errorDescription: String? {
        switch self {
        case .notRunning: return "engine not started"
        case .exited: return "engine exited"
        case .timeout(let m): return "timeout: \(m)"
        case .rpc(let s): return "RPC \(s)"
        case .engine(let code, let message): return "\(code): \(message)"
        }
    }
}

/// Owns the native `frontend` binary (the Lean compiler + session + runtime
/// FFI) over its Unix-socket JSON-RPC surface (`--serve`). Audio plays
/// out of the HOST's native device (RtAudio); the app is purely a control
/// surface.
///
/// Transport mirrors playground/main.js: one JSON object per line on the
/// socket; replies are matched by `id`. The socket has a control/data plane
/// SPLIT (tropical_socket.hpp): `set_param` / `render_window` /
/// `playback_position` / `get_telemetry` are answered synchronously in C++
/// and never queue behind the single Lean control thread — so knob writes
/// and scope frames stay live through a long compile. Control-plane replies
/// wrap an inner {status,data} in result.content[0].text; data-plane replies
/// are a plain `result`. We unwrap both.
actor Engine {
    private var process: Process?
    private var sock: FileHandle?
    private var sockPath = ""
    /// The child engine OUTLIVES the app unless someone reaps it — an orphaned
    /// engine keeps the DAC (and whatever it was playing) alive forever. The
    /// box holds the Process outside actor isolation so the app-termination
    /// path (a synchronous, can't-await context) can always reach it.
    private nonisolated let procBox = ProcessBox()
    private var buf = ""
    private var pending: [Int: CheckedContinuation<JSONValue, Error>] = [:]
    private var nextId = 1
    private let events: @Sendable (EngineEvent) -> Void

    /// Resolution order: TROPICAL_ENGINE_BIN, then the default checkout.
    static func resolveBinary() -> URL {
        if let p = ProcessInfo.processInfo.environment["TROPICAL_ENGINE_BIN"] {
            return URL(fileURLWithPath: p)
        }
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("tropical/lean/.lake/build/bin/frontend")
    }

    init(events: @escaping @Sendable (EngineEvent) -> Void) {
        self.events = events
    }

    func start() async throws {
        let bin = Engine.resolveBinary()
        // The engine expects to run from the repo root (stdlib paths etc.):
        // bin is <root>/lean/.lake/build/bin/frontend, so root is 5 up.
        let root = bin.deletingLastPathComponent()  // bin/
            .deletingLastPathComponent()            // build/
            .deletingLastPathComponent()            // .lake/
            .deletingLastPathComponent()            // lean/
            .deletingLastPathComponent()            // root
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
            Task { await self?.failAllPending() }
        }

        try p.run()
        process = p
        procBox.process = p
        try await connectSocket()
        events(.up)
    }

    /// Synchronous child teardown for app exit (Cmd-Q, SIGTERM). Safe from
    /// any thread; idempotent.
    nonisolated func terminateChild() {
        procBox.take()?.terminate()
    }

    /// The socket node appears once Engine.boot finishes and binds, so retry
    /// until it accepts (or the process dies under us).
    private func connectSocket() async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(20))
        while ContinuousClock.now < deadline {
            guard process?.isRunning == true else { throw EngineError.exited }
            if let fd = Engine.connectOnce(path: sockPath) {
                let h = FileHandle(fileDescriptor: fd, closeOnDealloc: true)
                h.readabilityHandler = { [weak self] h in
                    let data = h.availableData
                    if data.isEmpty { h.readabilityHandler = nil; return }  // EOF
                    guard let s = String(data: data, encoding: .utf8) else { return }
                    Task { await self?.consume(s) }
                }
                sock = h
                return
            }
            try await Task.sleep(for: .milliseconds(100))
        }
        throw EngineError.timeout(method: "connect \(sockPath)")
    }

    private static func connectOnce(path: String) -> Int32? {
        let fd = Darwin.socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else { return nil }
        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let bytes = Array(path.utf8CString)
        guard bytes.count <= MemoryLayout.size(ofValue: addr.sun_path) else {
            close(fd)
            return nil
        }
        withUnsafeMutableBytes(of: &addr.sun_path) { dst in
            bytes.withUnsafeBytes { src in
                dst.copyMemory(from: src)
            }
        }
        let r = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { sa in
                Darwin.connect(fd, sa, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        guard r == 0 else {
            close(fd)
            return nil
        }
        return fd
    }

    func kill() {
        process?.terminationHandler = nil
        process?.terminate()
        process = nil
        procBox.take()
        sock?.readabilityHandler = nil
        sock = nil
        unlink(sockPath)
    }

    private func failAllPending() {
        let waiting = pending.values
        pending.removeAll()
        for c in waiting { c.resume(throwing: EngineError.exited) }
        process = nil
        procBox.take()
        sock?.readabilityHandler = nil
        sock = nil
        unlink(sockPath)
    }

    private func consume(_ chunk: String) {
        buf += chunk
        while let i = buf.firstIndex(of: "\n") {
            let line = String(buf[..<i]).trimmingCharacters(in: .whitespaces)
            buf = String(buf[buf.index(after: i)...])
            guard !line.isEmpty, let data = line.data(using: .utf8),
                  let msg = try? JSONDecoder().decode(JSONValue.self, from: data),
                  let id = msg["id"]?.doubleValue,
                  let cont = pending.removeValue(forKey: Int(id))
            else { continue }
            cont.resume(with: unwrap(msg))
        }
    }

    /// Control-plane replies nest {status,data} as text in result.content[0];
    /// data-plane replies are the bare result. Same unwrap as main.js.
    private func unwrap(_ msg: JSONValue) -> Result<JSONValue, Error> {
        if let err = msg["error"], err != .null {
            let enc = (try? JSONEncoder().encode(err)).flatMap { String(data: $0, encoding: .utf8) }
            return .failure(EngineError.rpc(enc ?? "unknown"))
        }
        let result = msg["result"] ?? .null
        guard case .array(let content)? = result["content"],
              let text = content.first?["text"]?.stringValue
        else { return .success(result) }
        guard let data = text.data(using: .utf8),
              let inner = try? JSONDecoder().decode(JSONValue.self, from: data)
        else { return .failure(EngineError.rpc("unparseable inner reply")) }
        if inner["status"]?.stringValue == "ok" {
            return .success(inner["data"] ?? .null)
        }
        return .failure(EngineError.engine(
            code: inner["error"]?["code"]?.stringValue ?? "unknown",
            message: inner["error"]?["message"]?.stringValue ?? "unknown"))
    }

    @discardableResult
    func call(_ method: String, _ params: JSONValue = .object([:])) async throws -> JSONValue {
        guard let sock else { throw EngineError.notRunning }
        let id = nextId
        nextId += 1
        let req = JSONValue.object([
            "jsonrpc": .string("2.0"), "id": .number(Double(id)),
            "method": .string(method), "params": params,
        ])
        var line = try JSONEncoder().encode(req)
        line.append(0x0A)

        // The compile+JIT is synchronous on the engine's one control thread, and
        // a heavy kernel (a modal reverb) can take tens of seconds. A short
        // timeout gives up while the engine is still grinding (the thread stays
        // blocked, the late reply is dropped), so heavy compiles get a long leash
        // while other CONTROL calls still fail fast. Data-plane methods
        // (set_param/render_window/playback_position) answer in C++ and never
        // wait on that thread. NOTE: a diagnostic band-aid — the real fix is
        // compiling off the control thread.
        let timeout: Duration = method == "load_patch_graph" ? .seconds(180) : .seconds(30)
        let timer = Task { [weak self] in
            try await Task.sleep(for: timeout)
            await self?.timeOut(id: id, method: method)
        }
        defer { timer.cancel() }

        return try await withCheckedThrowingContinuation { cont in
            pending[id] = cont
            do { try sock.write(contentsOf: line) } catch {
                pending.removeValue(forKey: id)
                cont.resume(throwing: error)
            }
        }
    }

    private func timeOut(id: Int, method: String) {
        guard let cont = pending.removeValue(forKey: id) else { return }
        cont.resume(throwing: EngineError.timeout(method: method))
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
