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
/// FFI) over its newline-delimited JSON-RPC surface (`--rpc`). Audio plays
/// out of the HOST's native device (RtAudio); the app is purely a control
/// surface.
///
/// Transport mirrors playground/main.js: one JSON object per line on stdin;
/// replies arrive on stdout, matched by `id`. Control-plane replies wrap an
/// inner {status,data} in result.content[0].text; data-plane replies
/// (set_param) are a plain `result`. We unwrap both.
actor Engine {
    private var process: Process?
    private var stdin: FileHandle?
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

    func start() throws {
        let bin = Engine.resolveBinary()
        // The engine expects to run from the repo root (stdlib paths etc.):
        // bin is <root>/lean/.lake/build/bin/frontend, so root is 5 up.
        let root = bin.deletingLastPathComponent()  // bin/
            .deletingLastPathComponent()            // build/
            .deletingLastPathComponent()            // .lake/
            .deletingLastPathComponent()            // lean/
            .deletingLastPathComponent()            // root
        let p = Process()
        p.executableURL = bin
        p.arguments = ["--rpc"]
        p.currentDirectoryURL = root
        // The arrow patch-graph path does not route through syncCompile, but we
        // set the var anyway so any session-path compile in this process also
        // takes the session → resolved-root (arrow) path. Inert if ignored.
        var env = ProcessInfo.processInfo.environment
        env["TROPICAL_ARROW"] = "1"
        p.environment = env

        let inPipe = Pipe(), outPipe = Pipe(), errPipe = Pipe()
        p.standardInput = inPipe
        p.standardOutput = outPipe
        p.standardError = errPipe

        outPipe.fileHandleForReading.readabilityHandler = { [weak self] h in
            let data = h.availableData
            guard !data.isEmpty, let s = String(data: data, encoding: .utf8) else { return }
            Task { await self?.consume(s) }
        }
        let events = self.events
        errPipe.fileHandleForReading.readabilityHandler = { h in
            let data = h.availableData
            guard !data.isEmpty, let s = String(data: data, encoding: .utf8) else { return }
            events(.stderr(s))
        }
        p.terminationHandler = { [weak self] proc in
            events(.exit(proc.terminationStatus))
            Task { await self?.failAllPending() }
        }

        try p.run()
        process = p
        stdin = inPipe.fileHandleForWriting
        events(.up)
    }

    func kill() {
        process?.terminationHandler = nil
        process?.terminate()
        process = nil
        stdin = nil
    }

    private func failAllPending() {
        let waiting = pending.values
        pending.removeAll()
        for c in waiting { c.resume(throwing: EngineError.exited) }
        process = nil
        stdin = nil
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
        guard let stdin else { throw EngineError.notRunning }
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
        // while every other call still fails fast. NOTE: a diagnostic band-aid —
        // the real fix is compiling off the control thread.
        let timeout: Duration = method == "load_patch_graph" ? .seconds(180) : .seconds(30)
        let timer = Task { [weak self] in
            try await Task.sleep(for: timeout)
            await self?.timeOut(id: id, method: method)
        }
        defer { timer.cancel() }

        return try await withCheckedThrowingContinuation { cont in
            pending[id] = cont
            do { try stdin.write(contentsOf: line) } catch {
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
