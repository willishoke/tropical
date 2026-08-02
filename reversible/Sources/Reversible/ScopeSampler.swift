import CoreVideo
import Foundation
import SwiftUI

/// Program/control identity of one immutable scope snapshot.
struct ModalScopeGeneration: Equatable, Sendable {
    let programVersion: UInt64
    let controlVersion: UInt64

    func isOlder(than other: ModalScopeGeneration) -> Bool {
        programVersion < other.programVersion
            || (programVersion == other.programVersion
                && controlVersion < other.controlVersion)
    }
}

/// The master-clock mapping captured by the owner at one display tick. The
/// later release-scene adapter supplies this beside the five tap bindings; the
/// sampler never reads UI state piecemeal while transforming a response.
struct ModalScopeClockMapping: Equatable, Sendable {
    let tauBase: Double
    let velocity: Double

    static let forward = ModalScopeClockMapping(tauBase: 0, velocity: 1)
}

/// Everything needed to turn one projected engine slot into one mode trace.
/// `id` is presentation identity only; `slot` is the exact returned tap binding.
struct ModalScopeModeDescriptor: Identifiable, Equatable, Sendable {
    let id: String
    let slot: String
    let frequency: Double
    let envelope: ModalPairedEnvelope
}

/// A fixed five-mode scope binding. Construction fails rather than silently
/// accepting four channels, duplicate tap slots, or a per-frame autoscale.
struct ModalScopeBinding: Equatable, Sendable {
    let modes: [ModalScopeModeDescriptor]
    let sharedFullScale: Double
    let clock: ModalScopeClockMapping

    init?(
        modes: [ModalScopeModeDescriptor],
        sharedFullScale: Double,
        clock: ModalScopeClockMapping
    ) {
        guard modes.count == 5,
              Set(modes.map(\.id)).count == 5,
              Set(modes.map(\.slot)).count == 5,
              modes.allSatisfy({ $0.frequency.isFinite && $0.frequency > 0 }),
              sharedFullScale.isFinite, sharedFullScale > 0,
              clock.tauBase.isFinite, clock.velocity.isFinite
        else { return nil }
        self.modes = modes
        self.sharedFullScale = sharedFullScale
        self.clock = clock
    }
}

struct ModalScopeModeFrame: Equatable, Sendable {
    let descriptor: ModalScopeModeDescriptor
    let frame: ModalScopeFrame
}

/// A complete frame is published as one value. There is no observable state
/// in which some mode traces belong to a different runtime generation.
struct ModalScopeOverlayFrame: Equatable, Sendable {
    let generation: ModalScopeGeneration
    let responseStart: Double
    let responseCount: Int
    let responseSpan: Int
    let stride: Int
    let effectiveSampleIndex: UInt64
    let playbackPosition: Double
    let displayScale: Double
    let modes: [ModalScopeModeFrame]

    var visibleModes: [ModalScopeModeFrame] {
        modes.filter {
            $0.frame.lock.active && $0.frame.lock.locked
                && $0.frame.displayEnvelope > 1e-12
                && $0.frame.displayedValues.count >= 2
        }
    }
}

enum ModalScopeSamplerError: Error, Equatable {
    case malformedPlaybackPosition
    case malformedResponse(String)
}

/// The small seam used by the sampler tests. Engine already owns a dedicated
/// scope connection; conforming here does not add or reroute an RPC method.
protocol ModalScopeRPC: AnyObject {
    func call(_ method: String, _ params: JSONValue) async throws -> JSONValue
}

extension Engine: ModalScopeRPC {}

/// Fetches and transforms away from MainActor. The actor admits one operation
/// at a time; ModalScopeSampler additionally refuses to enqueue a second tick
/// while this one is suspended in RPC.
actor ModalScopeWorker {
    private let rpc: any ModalScopeRPC
    private let profile: ModalScopeProfile
    private var adoptedGeneration: ModalScopeGeneration?

    init(rpc: any ModalScopeRPC, profile: ModalScopeProfile = .qualified) {
        self.rpc = rpc
        self.profile = profile
    }

    func fetch(binding: ModalScopeBinding) async throws -> ModalScopeOverlayFrame? {
        let positionReply = try await rpc.call("playback_position", .object([:]))
        guard let playbackPosition = positionReply["position"]?.doubleValue,
              playbackPosition.isFinite, playbackPosition >= 0
        else { throw ModalScopeSamplerError.malformedPlaybackPosition }

        // The requested projection includes 64 warm-up points plus the 896
        // display points and 1,792-point crossing search. Its visible linear
        // span remains exactly 1,792 samples; point_budget=count forces stride 1.
        let requestedCount = profile.pointBudget
        let start = max(0, Int(playbackPosition) - requestedCount)
        let response = try await rpc.call("render_window", .object([
            "start": .number(Double(start)),
            "count": .number(Double(requestedCount)),
            "point_budget": .number(Double(requestedCount)),
            "slots": .array(binding.modes.map { .string($0.slot) }),
        ]))

        if response["preempted"]?.boolValue == true { return nil }
        guard response["preempted"]?.boolValue == false,
              let responseStart = response["start"]?.doubleValue,
              let responseCount = Self.exactInt(response["count"]),
              let responseSpan = Self.exactInt(response["span"]),
              let stride = Self.exactInt(response["stride"]),
              let programVersion = Self.exactUInt64(response["program_version"]),
              let controlVersion = Self.exactUInt64(response["control_version"]),
              let effectiveSampleIndex = Self.exactUInt64(
                response["effective_sample_index"]
              ),
              case .array(let channels)? = response["values"],
              responseStart.isFinite,
              responseCount == requestedCount,
              responseSpan == requestedCount,
              stride == 1,
              channels.count == binding.modes.count
        else {
            throw ModalScopeSamplerError.malformedResponse(
                "scope response was not one complete stride-1 five-slot projection"
            )
        }

        var values: [[Double]] = []
        values.reserveCapacity(channels.count)
        for channel in channels {
            guard case .array(let cells) = channel, cells.count == requestedCount else {
                throw ModalScopeSamplerError.malformedResponse(
                    "scope response contained a partial mode channel"
                )
            }
            let decoded = cells.compactMap(\.doubleValue)
            guard decoded.count == requestedCount, decoded.allSatisfy(\.isFinite) else {
                throw ModalScopeSamplerError.malformedResponse(
                    "scope response contained a non-finite mode sample"
                )
            }
            values.append(decoded)
        }

        let generation = ModalScopeGeneration(
            programVersion: programVersion,
            controlVersion: controlVersion
        )
        if let adoptedGeneration, generation.isOlder(than: adoptedGeneration) {
            return nil
        }

        // The first playback-position read chooses only the window address.
        // Audible-now math belongs to the pinned render response, whose
        // effective index shares its program/control generation.
        let authoritativePosition = Double(effectiveSampleIndex)
        let frames = zip(binding.modes, values).map { descriptor, samples in
            ModalScopeModeFrame(
                descriptor: descriptor,
                frame: ModalScopeMath.frame(
                    values: samples,
                    responseStart: responseStart,
                    stride: stride,
                    frequency: descriptor.frequency,
                    envelope: descriptor.envelope,
                    tauBase: binding.clock.tauBase,
                    velocity: binding.clock.velocity,
                    playbackPosition: authoritativePosition,
                    sharedFullScale: binding.sharedFullScale,
                    profile: profile
                )
            )
        }
        guard frames.count == 5 else {
            throw ModalScopeSamplerError.malformedResponse(
                "scope transform did not complete all five modes"
            )
        }

        let frame = ModalScopeOverlayFrame(
            generation: generation,
            responseStart: responseStart,
            responseCount: responseCount,
            responseSpan: responseSpan,
            stride: stride,
            effectiveSampleIndex: effectiveSampleIndex,
            playbackPosition: authoritativePosition,
            displayScale: ModalScopeMath.displayScale(
                sharedFullScale: binding.sharedFullScale
            ),
            modes: frames
        )
        adoptedGeneration = generation
        return frame
    }

    private static func exactInt(_ value: JSONValue?) -> Int? {
        guard let number = value?.doubleValue, number.isFinite,
              number >= 0, number.rounded(.towardZero) == number,
              number <= 9_007_199_254_740_991,
              let exact = Int(exactly: number)
        else { return nil }
        return exact
    }

    private static func exactUInt64(_ value: JSONValue?) -> UInt64? {
        guard let number = value?.doubleValue, number.isFinite,
              number >= 0, number.rounded(.towardZero) == number,
              number <= 9_007_199_254_740_991,
              let exact = UInt64(exactly: number)
        else { return nil }
        return exact
    }
}

/// MainActor owns only the latest immutable frame and the in-flight bit. JSON
/// framing, socket waiting, validation, and five transforms stay on the worker.
@MainActor
final class ModalScopeSampler: ObservableObject {
    @Published private(set) var latestFrame: ModalScopeOverlayFrame?
    @Published private(set) var droppedFrameCount = 0
    private(set) var requestCount = 0
    private var worker: ModalScopeWorker?
    private var inFlight = false

    var hasRequestInFlight: Bool { inFlight }

    func configure(rpc: any ModalScopeRPC) {
        guard worker == nil else { return }
        worker = ModalScopeWorker(rpc: rpc)
    }

    func clear() {
        latestFrame = nil
    }

    func request(_ binding: ModalScopeBinding) {
        guard let worker, !inFlight else {
            if inFlight { droppedFrameCount += 1 }
            return
        }
        inFlight = true
        requestCount += 1
        Task { [weak self] in
            do {
                let frame = try await worker.fetch(binding: binding)
                guard let self else { return }
                if let frame { self.latestFrame = frame }
                else { self.droppedFrameCount += 1 }
                self.inFlight = false
            } catch {
                guard let self else { return }
                self.droppedFrameCount += 1
                self.inFlight = false
            }
        }
    }
}

/// Pure cadence port of the qualified requestAnimationFrame gate.
struct ModalScopeCadence: Equatable, Sendable {
    let framesPerSecond: Double
    private(set) var nextDeadline: Double = 0

    init(framesPerSecond: Double = ModalScopeProfile.qualified.framesPerSecond) {
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

/// CVDisplayLink is a wake-up source only. It never waits for RPC or mutates
/// view state on its callback thread; accepted ticks hop to MainActor and the
/// sampler's in-flight gate drops any superseded request.
@MainActor
final class ModalScopeDisplayLink: ObservableObject {
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
        private var cadence = ModalScopeCadence()
        private var enqueued = false
        private var action: (@MainActor () -> Void)?

        func install(_ action: @escaping @MainActor () -> Void) {
            lock.withLock {
                cadence = ModalScopeCadence()
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
            let accepted: Bool = lock.withLock {
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
