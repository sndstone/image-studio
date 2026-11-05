import Foundation

@main
struct FinishedRunTests {
    static func main() {
        let start = Date(timeIntervalSince1970: 1000)
        var run = FinishedRun(started: start)
        precondition(!run.isWorthAnnouncing, "A cancelled run stays silent")

        run.jobNames = ["FLUX.2 klein 4B"]
        run.images = [URL(fileURLWithPath: "/tmp/a.png")]
        precondition(run.title == "Image ready")
        precondition(run.body(now: start.addingTimeInterval(12)) == "FLUX.2 klein 4B · 12 s")

        run.recordFailures(-1, error: "ignored")  // failed jobs dismissed during the run
        precondition(run.failures == 0 && run.lastError == nil)

        run.jobNames.append("Upscale")
        run.images.append(URL(fileURLWithPath: "/tmp/b.png"))
        run.recordFailures(1, error: "Out of memory\nmore detail")
        precondition(run.title == "2 images ready, 1 job failed")
        precondition(run.body(now: start.addingTimeInterval(125)) == "2 jobs · 2 min 5 s")

        var failed = FinishedRun(started: start)
        failed.jobNames = ["GPT Image 2.6"]
        failed.recordFailures(1, error: "Out of memory\nmore detail")
        precondition(failed.isWorthAnnouncing && failed.title == "1 job failed")
        precondition(failed.body() == "Out of memory", "A failure shows the error's first line")
        print("Finish notification wording checks passed.")
    }
}
