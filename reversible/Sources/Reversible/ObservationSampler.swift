import CoreVideo
import Foundation
import SwiftUI

struct ObservationChannel: Equatable, Hashable, Sendable {
    let key: String
    let slot: String
}

/// One version-pinned projection request. `span` is the full sample history;
/// pointBudget permits deterministic engine-side striding without changing
/// the identity or completeness of the returned channel set.
struct ObservationBinding: Equatable, Sendable {
    let expectedGeneration: EngineGeneration
    let span: Int
    let pointBudget: Int
    let channels: [ObservationChannel]

    init(
        expectedGeneration: EngineGeneration,
        span: Int,
        pointBudget: Int,
        channels: [ObservationChannel]
    ) throws {
        guard expectedGeneration.programVersion > 0,
              expectedGeneration.controlVersion > 0,
              (1...16_384).contains(span),
              (1...16_384).contains(pointBudget),
              !channels.isEmpty,
              Set(channels.map(\.key)).count == channels.count,
              channels.allSatisfy({ channel in
                  !channel.key.isEmpty && !channel.slot.isEmpty
                      && channel.key == channel.key.trimmingCharacters(
                          in: .whitespacesAndNewlines
                      )
                      && channel.slot == channel.slot.trimmingCharacters(
                          in: .whitespacesAndNewlines
                      )
              })
        else { throw ObservationError.invalidBinding }
        self.expectedGeneration = expectedGeneration
        self.span = span
        self.pointBudget = pointBudget
        self.channels = channels
    }

    var requestJSON: JSONValue {
        .object([
            "anchor": .string("playback"),
            "count": .number(Double(span)),
            "point_budget": .number(Double(pointBudget)),
            "slots": .array(channels.map { .string($0.slot) }),
        ])
    }

    var expectedStride: Int {
        (span + pointBudget - 1) / pointBudget
    }

    var expectedCount: Int {
        (span + expectedStride - 1) / expectedStride
    }
}

struct ObservationChannelFrame: Equatable, Sendable {
    let channel: ObservationChannel
    let values: [Double]
}

/// A complete playback-anchored response. All channels, timing metadata, and
/// generation identity are published together or not at all.
struct ObservationFrame: Equatable, Sendable {
    let binding: ObservationBinding
    let generation: EngineGeneration
    let playbackPosition: UInt64
    /// The engine's effective start for this playback-anchored projection
    /// (`start` on the wire).
    let effectiveStart: UInt64
    let effectiveSampleIndex: UInt64
    let count: Int
    let stride: Int
    let channels: [ObservationChannelFrame]
}

enum ObservationDecodedResponse: Equatable, Sendable {
    case frame(ObservationFrame)
    case preempted(generation: EngineGeneration)
}

enum ObservationError: Error, Equatable, LocalizedError, Sendable {
    case invalidBinding
    case missingField(String)
    case invalidField(String)
    case partialChannel(Int)

    var errorDescription: String? {
        switch self {
        case .invalidBinding: return "invalid observation binding"
        case .missingField(let path): return "observation response missing '\(path)'"
        case .invalidField(let path): return "observation response has invalid '\(path)'"
        case .partialChannel(let index):
            return "observation response channel \(index) is partial or malformed"
        }
    }
}

enum ObservationResponseDecoder {
    static func decode(
        _ value: JSONValue,
        for binding: ObservationBinding
    ) throws -> ObservationDecodedResponse {
        guard case .object(let object) = value else {
            throw ObservationError.invalidField("response")
        }
        guard case .string("playback")? = object["anchor"] else {
            throw object["anchor"] == nil
                ? ObservationError.missingField("anchor")
                : ObservationError.invalidField("anchor")
        }
        let playbackPosition = try exactUInt64(object, "playback_position")
        let effectiveStart = try exactUInt64(object, "start")
        let effectiveSampleIndex = try exactUInt64(
            object, "effective_sample_index"
        )
        let count = try exactInt(object, "count")
        let span = try exactInt(object, "span")
        let stride = try exactInt(object, "stride")
        let generation = EngineGeneration(
            programVersion: try exactUInt64(object, "program_version"),
            controlVersion: try exactUInt64(object, "control_version")
        )
        guard generation.programVersion > 0, generation.controlVersion > 0 else {
            throw ObservationError.invalidField("generation")
        }
        guard span == binding.span else {
            throw ObservationError.invalidField("span")
        }
        guard stride == binding.expectedStride else {
            throw ObservationError.invalidField("stride")
        }
        let expectedStart = playbackPosition >= UInt64(binding.span)
            ? playbackPosition - UInt64(binding.span)
            : 0
        guard effectiveStart == expectedStart else {
            throw ObservationError.invalidField("start")
        }
        guard case .bool(let preempted)? = object["preempted"] else {
            throw object["preempted"] == nil
                ? ObservationError.missingField("preempted")
                : ObservationError.invalidField("preempted")
        }
        guard case .array(let rawChannels)? = object["values"] else {
            throw object["values"] == nil
                ? ObservationError.missingField("values")
                : ObservationError.invalidField("values")
        }

        if preempted {
            guard count == 0, rawChannels.isEmpty else {
                throw ObservationError.invalidField("preempted")
            }
            return .preempted(generation: generation)
        }

        guard count == binding.expectedCount else {
            throw ObservationError.invalidField("count")
        }
        guard rawChannels.count == binding.channels.count else {
            throw ObservationError.invalidField("values")
        }
        var channels: [ObservationChannelFrame] = []
        channels.reserveCapacity(rawChannels.count)
        for (index, rawChannel) in rawChannels.enumerated() {
            guard case .array(let rawValues) = rawChannel,
                  rawValues.count == count
            else { throw ObservationError.partialChannel(index) }
            var values: [Double] = []
            values.reserveCapacity(count)
            for rawValue in rawValues {
                guard let sample = rawValue.doubleValue, sample.isFinite else {
                    throw ObservationError.partialChannel(index)
                }
                values.append(sample)
            }
            channels.append(.init(
                channel: binding.channels[index],
                values: values
            ))
        }
        return .frame(.init(
            binding: binding,
            generation: generation,
            playbackPosition: playbackPosition,
            effectiveStart: effectiveStart,
            effectiveSampleIndex: effectiveSampleIndex,
            count: count,
            stride: stride,
            channels: channels
        ))
    }

    private static func exactUInt64(
        _ object: [String: JSONValue], _ key: String
    ) throws -> UInt64 {
        guard let value = object[key] else {
            throw ObservationError.missingField(key)
        }
        guard let exact = JSONExact.uint64(value) else {
            throw ObservationError.invalidField(key)
        }
        return exact
    }

    private static func exactInt(
        _ object: [String: JSONValue], _ key: String
    ) throws -> Int {
        guard let value = object[key] else {
            throw ObservationError.missingField(key)
        }
        guard let exact = JSONExact.int(value) else {
            throw ObservationError.invalidField(key)
        }
        return exact
    }
}

protocol ObservationRPC: AnyObject, Sendable {
    func call(_ method: String, _ params: JSONValue) async throws -> JSONValue
}

extension Engine: ObservationRPC {}

/// Owns response validation and monotonic generation adoption away from the
/// main actor. The wrapper below additionally refuses to enqueue concurrent
/// display ticks.
actor ObservationWorker {
    private let rpc: any ObservationRPC
    private var lastAcceptedGeneration: EngineGeneration?
    private var activeSessionEpoch: UInt64 = 0

    init(rpc: any ObservationRPC) {
        self.rpc = rpc
    }

    func fetch(
        _ binding: ObservationBinding,
        sessionEpoch: UInt64 = 0
    ) async throws -> ObservationFrame? {
        guard sessionEpoch >= activeSessionEpoch else { return nil }
        if sessionEpoch > activeSessionEpoch {
            activeSessionEpoch = sessionEpoch
            lastAcceptedGeneration = nil
        }
        let raw = try await rpc.call("render_window", binding.requestJSON)
        guard sessionEpoch == activeSessionEpoch else { return nil }
        switch try ObservationResponseDecoder.decode(raw, for: binding) {
        case .preempted:
            return nil
        case .frame(let frame):
            guard frame.generation.programVersion
                    == binding.expectedGeneration.programVersion,
                  frame.generation.controlVersion
                    >= binding.expectedGeneration.controlVersion
            else { return nil }
            if let lastAcceptedGeneration,
               lastAcceptedGeneration.programVersion == frame.generation.programVersion,
               frame.generation.controlVersion
                    < lastAcceptedGeneration.controlVersion {
                return nil
            }
            lastAcceptedGeneration = frame.generation
            return frame
        }
    }
}

/// Main-actor publication seam. At most one socket request exists at a time;
/// a hot-swap changes the expected program immediately, so an old in-flight
/// reply cannot become visible even if it completes successfully.
@MainActor
final class ObservationSampler: ObservableObject {
    @Published private(set) var latestFrame: ObservationFrame?
    @Published private(set) var droppedFrameCount = 0
    private(set) var requestCount = 0
    private(set) var hasRequestInFlight = false
    var onFrame: ((ObservationFrame) -> Void)?

    private let worker: ObservationWorker
    private var expectedGeneration: EngineGeneration?
    private var sessionEpoch: UInt64 = 0

    init(rpc: any ObservationRPC) {
        worker = ObservationWorker(rpc: rpc)
    }

    func adoptCompileGeneration(_ generation: EngineGeneration) {
        expectedGeneration = generation
        latestFrame = nil
    }

    func clear() {
        sessionEpoch &+= 1
        expectedGeneration = nil
        latestFrame = nil
    }

    func request(_ binding: ObservationBinding) {
        guard expectedGeneration == binding.expectedGeneration,
              !hasRequestInFlight
        else {
            droppedFrameCount += 1
            return
        }
        hasRequestInFlight = true
        requestCount += 1
        let requestEpoch = sessionEpoch
        Task { [weak self] in
            guard let self else { return }
            defer { hasRequestInFlight = false }
            do {
                guard let frame = try await worker.fetch(
                    binding,
                    sessionEpoch: requestEpoch
                ),
                      requestEpoch == sessionEpoch,
                      expectedGeneration == binding.expectedGeneration
                else {
                    droppedFrameCount += 1
                    return
                }
                latestFrame = frame
                onFrame?(frame)
            } catch {
                droppedFrameCount += 1
            }
        }
    }
}

/// Pure request cadence used by the production display-link adapter and tests.
struct ObservationCadence: Equatable, Sendable {
    let framesPerSecond: Double
    private(set) var nextDeadline: Double = 0

    init(framesPerSecond: Double = 60) {
        self.framesPerSecond = framesPerSecond
    }

    mutating func accepts(timestamp: Double) -> Bool {
        guard timestamp.isFinite, framesPerSecond > 0 else { return false }
        let period = 1 / framesPerSecond
        let target = nextDeadline == 0 ? timestamp : nextDeadline
        guard timestamp + 0.0005 >= target else { return false }
        nextDeadline = timestamp - target > period
            ? timestamp + period
            : target + period
        return true
    }
}

/// CVDisplayLink is only a wake-up source. The callback never performs RPC or
/// touches observable state; accepted ticks hop to MainActor, where the
/// sampler's in-flight gate enforces backpressure.
@MainActor
final class ObservationDisplayLink {
    private var displayLink: CVDisplayLink?
    private let callbackState = CallbackState()

    func start(_ action: @escaping @MainActor () -> Void) {
        stop()
        callbackState.install(action)
        var candidate: CVDisplayLink?
        guard CVDisplayLinkCreateWithActiveCGDisplays(&candidate) == kCVReturnSuccess,
              let candidate else { return }
        displayLink = candidate
        CVDisplayLinkSetOutputCallback(
            candidate,
            { _, _, outputTime, _, _, context in
                guard let context else { return kCVReturnError }
                let state = Unmanaged<CallbackState>
                    .fromOpaque(context).takeUnretainedValue()
                state.receive(outputTime.pointee)
                return kCVReturnSuccess
            },
            Unmanaged.passUnretained(callbackState).toOpaque()
        )
        CVDisplayLinkStart(candidate)
    }

    func stop() {
        if let displayLink { CVDisplayLinkStop(displayLink) }
        displayLink = nil
        callbackState.clear()
    }

    deinit {
        if let displayLink { CVDisplayLinkStop(displayLink) }
    }

    private final class CallbackState {
        private let lock = NSLock()
        private var cadence = ObservationCadence()
        private var enqueued = false
        private var action: (@MainActor () -> Void)?

        func install(_ action: @escaping @MainActor () -> Void) {
            lock.withLock {
                cadence = ObservationCadence()
                enqueued = false
                self.action = action
            }
        }

        func clear() {
            lock.withLock {
                enqueued = false
                action = nil
            }
        }

        func receive(_ timestamp: CVTimeStamp) {
            let seconds: Double
            if timestamp.videoTimeScale > 0 {
                seconds = Double(timestamp.videoTime) / Double(timestamp.videoTimeScale)
            } else {
                seconds = Double(timestamp.hostTime) / CVGetHostClockFrequency()
            }
            let accepted = lock.withLock {
                guard !enqueued, action != nil, cadence.accepts(timestamp: seconds)
                else { return false }
                enqueued = true
                return true
            }
            guard accepted else { return }
            Task { @MainActor [weak self] in
                guard let self else { return }
                let action = lock.withLock { () -> (@MainActor () -> Void)? in
                    self.enqueued = false
                    return self.action
                }
                action?()
            }
        }
    }
}
