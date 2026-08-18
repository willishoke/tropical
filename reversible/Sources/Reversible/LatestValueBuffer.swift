import Foundation

/// Bounded coalescing for one continuous gesture while a write is in flight.
/// The first value is stored on initialization. Later pointer events collapse
/// to the newest value. Retaining a stale turning point makes the audible value
/// travel away from the pointer after a fast reversal, so only the first and
/// current values survive. At most two unsent values exist before the first
/// request is issued, and one while that request is in flight.
struct LatestValueBuffer {
    private var firstPending: Double?
    private var latest: Double?

    init(first: Double) {
        firstPending = first
    }

    var pendingCount: Int {
        (firstPending == nil ? 0 : 1)
            + (latest == nil ? 0 : 1)
    }

    mutating func offer(_ value: Double) {
        guard value.isFinite else { return }
        latest = value
    }

    mutating func pop() -> Double? {
        if let firstPending {
            self.firstPending = nil
            return firstPending
        }
        defer { latest = nil }
        return latest
    }
}
