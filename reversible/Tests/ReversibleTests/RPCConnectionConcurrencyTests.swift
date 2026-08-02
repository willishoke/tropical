import Foundation
import XCTest
@testable import Reversible

final class RPCConnectionConcurrencyTests: XCTestCase {
    func testRequestIDsAndFragmentBuffersArePerConnection() async throws {
        let (left, leftPeer) = try makePair(lane: .graph)
        let (right, rightPeer) = try makePair(lane: .control)
        defer {
            leftPeer.close()
            rightPeer.close()
        }

        let leftCall = Task { try await left.call("left") }
        let rightCall = Task { try await right.call("right") }
        let leftRequest = try await leftPeer.nextJSON()
        let rightRequest = try await rightPeer.nextJSON()

        // IDs and partial-line buffers belong to one connection, not Engine.
        XCTAssertEqual(leftRequest["id"]?.doubleValue, 1)
        XCTAssertEqual(rightRequest["id"]?.doubleValue, 1)

        try leftPeer.reply(
            id: 1,
            result: .object(["lane": .string("left")]),
            fragments: [1, 3, 7]
        )
        try rightPeer.reply(
            id: 1,
            result: .object(["lane": .string("right")]),
            fragments: [2, 5]
        )

        let leftValue = try await leftCall.value
        let rightValue = try await rightCall.value
        XCTAssertEqual(leftValue["lane"]?.stringValue, "left")
        XCTAssertEqual(rightValue["lane"]?.stringValue, "right")
        await left.close(error: EngineError.exited)
        await right.close(error: EngineError.exited)
    }

    func testOutOfOrderRepliesCorrelateByRequestID() async throws {
        let (rpc, peer) = try makePair(lane: .graph)
        defer { peer.close() }

        let alpha = Task { try await rpc.call("alpha") }
        let beta = Task { try await rpc.call("beta") }
        let firstRequest = try await peer.nextJSON()
        let secondRequest = try await peer.nextJSON()
        let requests = [firstRequest, secondRequest]
        let byMethod = Dictionary(uniqueKeysWithValues: requests.map {
            ($0["method"]!.stringValue!, Int($0["id"]!.doubleValue!))
        })

        try peer.reply(
            id: try XCTUnwrap(byMethod["beta"]),
            result: .string("beta-result")
        )
        try peer.reply(
            id: try XCTUnwrap(byMethod["alpha"]),
            result: .string("alpha-result")
        )

        let alphaValue = try await alpha.value
        let betaValue = try await beta.value
        XCTAssertEqual(alphaValue.stringValue, "alpha-result")
        XCTAssertEqual(betaValue.stringValue, "beta-result")
        await rpc.close(error: EngineError.exited)
    }

    func testPendingBoundAndTerminalTimeoutFailEveryWaiterExactlyOnce() async throws {
        let (rpc, peer) = try makePair(lane: .scope, pendingLimit: 128)
        defer { peer.close() }

        let calls = (0..<128).map { index in
            Task { () -> Result<JSONValue, Error> in
                do { return .success(try await rpc.call("pending-\(index)")) }
                catch { return .failure(error) }
            }
        }
        // Seeing every line proves all 128 continuations are installed before
        // the overflow attempt; no scheduler timing assumption is involved.
        for _ in 0..<128 { _ = try await peer.nextJSON() }

        do {
            _ = try await rpc.call("overflow")
            XCTFail("129th pending request was accepted")
        } catch EngineError.pendingLimit(let lane, let limit) {
            XCTAssertEqual(lane, EngineRPCLane.scope.rawValue)
            XCTAssertEqual(limit, 128)
        } catch {
            XCTFail("unexpected overflow error: \(error)")
        }

        // Inject the same error the request timer uses through the real close
        // path, then close again with a different error. Idempotence plus 128
        // completed tasks proves every waiter resumed once and only once.
        await rpc.close(error: EngineError.timeout(method: "fixture-timeout"))
        await rpc.close(error: EngineError.exited)
        var timeoutCount = 0
        for call in calls {
            switch await call.value {
            case .success:
                XCTFail("pending request unexpectedly succeeded")
            case .failure(EngineError.timeout(let method)):
                XCTAssertEqual(method, "fixture-timeout")
                timeoutCount += 1
            case .failure(let error):
                XCTFail("unexpected terminal error: \(error)")
            }
        }
        XCTAssertEqual(timeoutCount, 128)
    }

    func testDelayedGraphReplyDoesNotBlockControlLane() async throws {
        let (graph, graphPeer) = try makePair(lane: .graph)
        let (control, controlPeer) = try makePair(lane: .control)
        let graphCompletion = CompletionProbe()
        defer {
            graphPeer.close()
            controlPeer.close()
        }

        let graphCall = Task {
            defer { graphCompletion.markCompleted() }
            return try await graph.call("load_patch_graph")
        }
        let controlCall = Task { try await control.call("set_param") }
        let graphRequest = try await graphPeer.nextJSON()
        let controlRequest = try await controlPeer.nextJSON()

        // Leave the graph request outstanding and answer control immediately.
        try controlPeer.reply(
            id: Int(try XCTUnwrap(controlRequest["id"]?.doubleValue)),
            result: .object(["accepted": .bool(true)])
        )
        let controlValue = try await controlCall.value
        XCTAssertEqual(controlValue["accepted"]?.boolValue, true)
        XCTAssertFalse(graphCompletion.completed)

        try graphPeer.reply(
            id: Int(try XCTUnwrap(graphRequest["id"]?.doubleValue)),
            result: .object(["ok": .bool(true)])
        )
        let graphValue = try await graphCall.value
        XCTAssertEqual(graphValue["ok"]?.boolValue, true)
        XCTAssertTrue(graphCompletion.completed)
        await graph.close(error: EngineError.exited)
        await control.close(error: EngineError.exited)
    }

    func testConnectTimeoutIsBoundedAndTyped() async throws {
        let missingPath = FileManager.default.temporaryDirectory
            .appendingPathComponent("reversible-missing-\(UUID().uuidString).sock")
            .path
        do {
            _ = try await RPCConnection.connect(
                lane: .telemetry,
                path: missingPath,
                timeout: .milliseconds(10)
            )
            XCTFail("connected to a missing socket")
        } catch EngineError.timeout(let method) {
            XCTAssertEqual(method, "connect telemetry")
        } catch {
            XCTFail("unexpected connect error: \(error)")
        }
    }

    private func makePair(
        lane: EngineRPCLane,
        pendingLimit: Int = 128
    ) throws -> (RPCConnection, FakeLinePeer) {
        var descriptors = [Int32](repeating: -1, count: 2)
        let result = descriptors.withUnsafeMutableBufferPointer { pointer in
            Darwin.socketpair(AF_UNIX, SOCK_STREAM, 0, pointer.baseAddress!)
        }
        guard result == 0 else {
            throw POSIXError(.init(rawValue: errno) ?? .EIO)
        }
        return (
            RPCConnection(
                lane: lane,
                fileDescriptor: descriptors[0],
                pendingLimit: pendingLimit
            ),
            FakeLinePeer(fileDescriptor: descriptors[1])
        )
    }
}

private final class CompletionProbe: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false

    var completed: Bool {
        lock.lock()
        defer { lock.unlock() }
        return value
    }

    func markCompleted() {
        lock.lock()
        value = true
        lock.unlock()
    }
}

/// A deterministic newline-JSON peer. It deliberately supports fragmented
/// writes so tests exercise RPCConnection's real per-socket receive buffer.
private final class FakeLinePeer: @unchecked Sendable {
    private let handle: FileHandle
    private let lock = NSLock()
    private var buffer = ""
    private var lines: [String] = []
    private var waiters: [CheckedContinuation<String, Never>] = []

    init(fileDescriptor: Int32) {
        handle = FileHandle(fileDescriptor: fileDescriptor, closeOnDealloc: true)
        handle.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else {
                return
            }
            self?.ingest(text)
        }
    }

    func close() {
        handle.readabilityHandler = nil
        try? handle.close()
    }

    func nextJSON() async throws -> JSONValue {
        let line = await withCheckedContinuation { continuation in
            lock.lock()
            if lines.isEmpty {
                waiters.append(continuation)
                lock.unlock()
            } else {
                let line = lines.removeFirst()
                lock.unlock()
                continuation.resume(returning: line)
            }
        }
        return try JSONDecoder().decode(JSONValue.self, from: Data(line.utf8))
    }

    func reply(
        id: Int,
        result: JSONValue,
        fragments: [Int] = []
    ) throws {
        var data = try JSONEncoder().encode(JSONValue.object([
            "jsonrpc": .string("2.0"),
            "id": .number(Double(id)),
            "result": result,
        ]))
        data.append(0x0A)

        var offset = 0
        for length in fragments where offset < data.count {
            let end = min(offset + length, data.count)
            try handle.write(contentsOf: data[offset..<end])
            offset = end
        }
        if offset < data.count {
            try handle.write(contentsOf: data[offset..<data.count])
        }
    }

    private func ingest(_ text: String) {
        var deliveries: [(CheckedContinuation<String, Never>, String)] = []
        lock.lock()
        buffer += text
        while let newline = buffer.firstIndex(of: "\n") {
            lines.append(String(buffer[..<newline]))
            buffer = String(buffer[buffer.index(after: newline)...])
        }
        while !lines.isEmpty, !waiters.isEmpty {
            deliveries.append((waiters.removeFirst(), lines.removeFirst()))
        }
        lock.unlock()
        for (waiter, line) in deliveries { waiter.resume(returning: line) }
    }
}
