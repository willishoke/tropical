import Foundation

/// One framed JSON-RPC socket with its own request IDs, pending bound, timeout
/// state, and read buffer. Compiler, control, scope, and telemetry each own an
/// instance, so a large response or blocked control-plane compile on one
/// connection cannot sit in front of another lane's bytes.
actor RPCConnection {
    private let lane: EngineRPCLane
    private let pendingLimit: Int
    private var socket: FileHandle?
    private var buffer = ""
    private var pending: [Int: CheckedContinuation<JSONValue, Error>] = [:]
    private var nextID = 1
    private var terminalError: Error?

    init(
        lane: EngineRPCLane,
        fileDescriptor: Int32,
        pendingLimit: Int = 128
    ) {
        self.lane = lane
        self.pendingLimit = pendingLimit
        let handle = FileHandle(fileDescriptor: fileDescriptor, closeOnDealloc: true)
        socket = handle
        handle.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            if data.isEmpty {
                handle.readabilityHandler = nil
                Task { await self?.close(error: EngineError.exited) }
                return
            }
            guard let text = String(data: data, encoding: .utf8) else { return }
            Task { await self?.consume(text) }
        }
    }

    static func connect(
        lane: EngineRPCLane,
        path: String,
        timeout: Duration = .seconds(5)
    ) async throws -> RPCConnection {
        let deadline = ContinuousClock.now.advanced(by: timeout)
        while ContinuousClock.now < deadline {
            if let fd = connectOnce(path: path) {
                return RPCConnection(lane: lane, fileDescriptor: fd)
            }
            try await Task.sleep(for: .milliseconds(25))
        }
        throw EngineError.timeout(method: "connect \(lane.rawValue)")
    }

    nonisolated static func connectOnce(path: String) -> Int32? {
        let fd = Darwin.socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else { return nil }
        var address = sockaddr_un()
        address.sun_family = sa_family_t(AF_UNIX)
        let bytes = Array(path.utf8CString)
        guard bytes.count <= MemoryLayout.size(ofValue: address.sun_path) else {
            Darwin.close(fd)
            return nil
        }
        withUnsafeMutableBytes(of: &address.sun_path) { destination in
            bytes.withUnsafeBytes { source in
                destination.copyMemory(from: source)
            }
        }
        let result = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { pointer in
                Darwin.connect(
                    fd,
                    pointer,
                    socklen_t(MemoryLayout<sockaddr_un>.size)
                )
            }
        }
        guard result == 0 else {
            Darwin.close(fd)
            return nil
        }
        return fd
    }

    @discardableResult
    func call(
        _ method: String,
        _ params: JSONValue = .object([:])
    ) async throws -> JSONValue {
        if let terminalError { throw terminalError }
        guard let socket else { throw EngineError.notRunning }
        guard pending.count < pendingLimit else {
            throw EngineError.pendingLimit(lane: lane.rawValue, limit: pendingLimit)
        }

        let id = nextID
        nextID += 1
        let request = JSONValue.object([
            "jsonrpc": .string("2.0"),
            "id": .number(Double(id)),
            "method": .string(method),
            "params": params,
        ])
        var line = try JSONEncoder().encode(request)
        line.append(0x0A)

        let timeout: Duration = method == "load_patch_graph"
            ? .seconds(180) : .seconds(30)
        let timer = Task { [weak self] in
            try await Task.sleep(for: timeout)
            await self?.timeOut(id: id, method: method)
        }
        defer { timer.cancel() }

        return try await withCheckedThrowingContinuation { continuation in
            pending[id] = continuation
            do {
                try socket.write(contentsOf: line)
            } catch {
                pending.removeValue(forKey: id)
                continuation.resume(throwing: error)
            }
        }
    }

    func close(error: Error) {
        guard terminalError == nil else { return }
        terminalError = error
        socket?.readabilityHandler = nil
        socket = nil
        let waiters = pending.values
        pending.removeAll()
        for waiter in waiters { waiter.resume(throwing: error) }
    }

    private func timeOut(id: Int, method: String) {
        guard let continuation = pending.removeValue(forKey: id) else { return }
        continuation.resume(throwing: EngineError.timeout(method: method))
    }

    private func consume(_ chunk: String) {
        buffer += chunk
        while let newline = buffer.firstIndex(of: "\n") {
            let line = String(buffer[..<newline])
                .trimmingCharacters(in: .whitespaces)
            buffer = String(buffer[buffer.index(after: newline)...])
            guard !line.isEmpty,
                  let data = line.data(using: .utf8),
                  let message = try? JSONDecoder().decode(JSONValue.self, from: data),
                  let wireID = message["id"]?.doubleValue,
                  let continuation = pending.removeValue(forKey: Int(wireID))
            else { continue }
            continuation.resume(with: unwrap(message))
        }
    }

    /// Control-plane replies nest `{status,data}` as text while data-plane
    /// replies are bare JSON-RPC results.
    private func unwrap(_ message: JSONValue) -> Result<JSONValue, Error> {
        if let error = message["error"], error != .null {
            let encoded = (try? JSONEncoder().encode(error)).flatMap {
                String(data: $0, encoding: .utf8)
            }
            return .failure(EngineError.rpc(encoded ?? "unknown"))
        }
        let result = message["result"] ?? .null
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
            message: inner["error"]?["message"]?.stringValue ?? "unknown"
        ))
    }
}
