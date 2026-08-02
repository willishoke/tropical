import Foundation

/// Bounded coalescing for one continuous gesture while a write is in flight.
/// The first value is stored on initialization. Later pointer events collapse
/// to the newest value, except that the first direction-turning point is kept
/// ahead of the final value. At most three unsent values exist before the first
/// request is issued, and at most two while that request is in flight.
struct LatestValueBuffer {
    private var firstPending: Double?
    private var latest: Double?
    private var reversal: Double?
    private var lastObserved: Double
    private var direction = 0
    private var capturedReversal = false

    init(first: Double) {
        firstPending = first
        lastObserved = first
    }

    var pendingCount: Int {
        (firstPending == nil ? 0 : 1)
            + (reversal == nil ? 0 : 1)
            + (latest == nil ? 0 : 1)
    }

    mutating func offer(_ value: Double) {
        guard value.isFinite else { return }
        let delta = value - lastObserved
        let nextDirection = delta == 0 ? 0 : (delta > 0 ? 1 : -1)
        if !capturedReversal, direction != 0, nextDirection != 0,
           nextDirection != direction {
            reversal = lastObserved
            capturedReversal = true
        }
        if nextDirection != 0 { direction = nextDirection }
        lastObserved = value
        latest = value
    }

    mutating func pop() -> Double? {
        if let firstPending {
            self.firstPending = nil
            return firstPending
        }
        if let reversal {
            self.reversal = nil
            if latest == reversal { latest = nil }
            return reversal
        }
        defer { latest = nil }
        return latest
    }
}
