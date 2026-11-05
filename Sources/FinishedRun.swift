import Foundation

/// What the queue did between starting and going idle, summed up for the "finished" notification.
struct FinishedRun {
    var started = Date()
    var jobNames: [String] = []  // never the prompt: this shows on the lock screen
    var images: [URL] = []
    private(set) var failures = 0
    private(set) var lastError: String?

    mutating func recordFailures(_ count: Int, error: String?) {
        guard count > 0 else { return }  // can be negative if failed jobs were dismissed or retried meanwhile
        failures += count
        lastError = error
    }

    /// A cancelled run produced nothing, and whoever cancelled it knows.
    var isWorthAnnouncing: Bool { !images.isEmpty || failures > 0 }

    var title: String {
        let ready = images.count == 1 ? "Image ready" : "\(images.count) images ready"
        let failed = failures == 1 ? "1 job failed" : "\(failures) jobs failed"
        if failures == 0 { return ready }
        return images.isEmpty ? failed : "\(ready), \(failed)"
    }

    func body(now: Date = Date()) -> String {
        if images.isEmpty, let line = lastError?.split(whereSeparator: \.isNewline).first {
            return String(line.prefix(200))
        }
        let what = jobNames.count == 1 ? jobNames[0] : "\(jobNames.count) jobs"
        return "\(what) · \(Self.duration(now.timeIntervalSince(started)))"
    }

    private static func duration(_ seconds: TimeInterval) -> String {
        let total = max(1, Int(seconds.rounded()))
        return total < 60 ? "\(total) s" : "\(total / 60) min \(total % 60) s"
    }
}
