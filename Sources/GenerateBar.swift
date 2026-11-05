import AppKit
import SwiftUI
import UniformTypeIdentifiers
import ImageIO

struct GenerateBar: View {
    @EnvironmentObject var model: AppModel
    @State private var jobEstimate: Estimate?  // estimate for the running job's own request, not the editor's settings

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let error = model.errorMessage { StudioNote(.error, error) }
            // The pre-job check and the engine often report the same memory warning; show it once.
            let notes = [model.status?.memory_warning, model.preflightNote].compactMap { $0 }
            ForEach(Array(Set(notes)).sorted(), id: \.self) { note in StudioNote(.warning, note) }
            if model.models.isEmpty, let error = model.modelError { StudioNote(.error, error) }
            if let error = model.modelErrors[model.modelID] ?? model.currentModel?.status.error { StudioNote(.error, error) }
            if model.isGenerating {
                ProgressView(value: progress) {
                    Text(progressLabel).font(.caption)
                }
                HStack {
                    Button(role: .cancel) { model.cancel() } label: {
                        Text("Cancel").frame(maxWidth: .infinity)
                    }
                    .help("Stop this image; queued jobs still run")
                    Button { model.generate() } label: {
                        Label("Add to Queue", systemImage: "text.badge.plus").frame(maxWidth: .infinity)
                    }
                    .disabled(model.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                              || model.currentModel?.status.downloaded != true)
                    .keyboardShortcut(.return, modifiers: .command)
                    .help("Queue the current settings to run after this one (⌘↩)")
                }
                .controlSize(.large)

            } else if let info = model.currentModel, info.status.downloading {
                ProgressView(value: info.status.fraction ?? 0) {
                    Text("Downloading \(info.name): \(formatBytes(info.status.bytes_present))"
                         + (info.status.bytes_total.map { " of \(formatBytes($0))" } ?? ""))
                        .font(.caption).monospacedDigit()
                }
                Button(role: .cancel) { model.cancelDownload(info) } label: {
                    Text("Cancel Download").frame(maxWidth: .infinity)
                }
                .controlSize(.large)
            } else if let info = model.currentModel, info.isCloud, !info.status.downloaded {
                Button { model.openCloudSettings() } label: {
                    Label("Add \(cloudProviderName(info)) API Key…", systemImage: "key").frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
            } else if let info = model.currentModel, !info.status.downloaded {
                Button { model.download(info) } label: {
                    Label("Download \(info.name)" + (info.status.bytes_total.map { " (\(formatBytes($0)))" } ?? ""),
                          systemImage: "arrow.down.circle")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
            } else {
                Button { model.generate() } label: {
                    Label("Generate", systemImage: "sparkles").frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .disabled(model.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                .keyboardShortcut(.return, modifiers: .command)
                HStack {
                    if let info = model.currentModel, info.isCloud {
                        Label(cloudCostLine(info), systemImage: "cloud")
                            .font(.caption).foregroundStyle(.secondary)
                            .help("An estimate before the run. The actual cost shows with each image and in Costs (⇧⌘K).")
                    } else {
                        EstimateLine(estimate: model.estimate, count: model.count)
                    }
                    Spacer()
                }
            }
            QueueList()
            if !model.isGenerating {
                HStack {
                    Button { model.showCompareSheet = true } label: {
                        Label("Compare…", systemImage: "square.split.2x1")
                    }
                    .help("Run this prompt on several models and see the results side by side (⇧⌘C)")
                    Button { model.variations() } label: {
                        Label("Variations ×4", systemImage: "square.grid.2x2")
                    }
                    if model.currentModel?.isCloud != true {
                        Button { model.showSweep = true } label: {
                            Label("Sweep…", systemImage: "slider.horizontal.3")
                        }
                    }
                }.buttonStyle(StudioButtonStyle()).disabled(model.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .sheet(isPresented: $model.showCompareSheet) { CompareSheet() }
        .sheet(isPresented: $model.showSweep) { SweepView() }
        .task(id: model.currentJob?.id) {
            // Estimate the running job's own request: queued jobs can differ from what the editor shows now.
            jobEstimate = nil
            guard let job = model.currentJob, case .generate(let request) = job.kind else { return }
            let estimate = try? await model.backend.estimate(request)
            guard !Task.isCancelled, model.currentJob?.id == job.id else { return }
            jobEstimate = estimate
        }
        .task(id: model.estimateKey) {
            try? await Task.sleep(nanoseconds: 250_000_000)  // settle while a slider is dragged
            guard !Task.isCancelled else { return }
            await model.refreshEstimate()
        }
    }

    /// One image's expected price, or plainly that it isn't known until a run has reported it.
    private func cloudCostLine(_ info: ModelInfo) -> String {
        if model.effectiveOutputFormat(for: info) == "svg" { return "Billed by OpenAI · an SVG's cost isn't known here: see your OpenAI usage" }
        guard let cost = model.estimate?.cost_usd else { return "Billed by \(cloudProviderName(info)) · cost shows after the first run" }
        let total = cost * Double(max(1, model.count))
        let basis = model.estimate?.cost_basis == "list" ? "list price" : "your recent runs"
        return "≈ \(formatUSD(total))" + (model.count > 1 ? " for \(model.count) images" : " per image") + " · \(basis)"
    }

    private var progress: Double? {
        if let s = model.status, s.phase == "converting", s.total_steps > 0 {
            return min(0.95, max(0, Double(s.step) / Double(s.total_steps)))
        }
        guard let s = model.status, ["encoding", "denoising", "decoding", "saving"].contains(s.phase),
              s.total_steps > 0, s.image_count > 0 else { return nil }
        let perImage = Double(s.step) / Double(s.total_steps)
        return min(0.95, max(0, (Double(s.image_index - 1) + perImage) / Double(s.image_count)))
    }

    /// Remaining time from the learned estimate and the progress so far; blank until both are known.
    private var timeLeft: String {
        guard let seconds = jobEstimate?.seconds, let progress, progress > 0.05 else { return "" }
        let left = seconds * (1 - progress)  // the estimate already covers the whole batch
        return left >= 2 ? " · ≈ \(formatDuration(left)) left" : ""
    }

    private var progressLabel: String {
        guard let s = model.status else { return "Starting…" }
        if s.phase == "loading" { return s.message }
        let prefix = s.image_count > 1 ? "Image \(s.image_index)/\(s.image_count) · " : ""
        switch s.phase {
        case "converting": return "Converting model for first use (one time)…"
        case "cloud": return prefix + s.message
        case "upscaling": return "Upscaling…"
        case "optimizing": return "Optimizing model…"
        case "encoding": return prefix + "Reading prompt and input images…"
        case "denoising":
            return prefix + "Step \(s.step)/\(s.total_steps)" + ((s.passes ?? 1) > 1 ? " · 2 passes per step" : "") + timeLeft
        case "decoding": return prefix + "Decoding image…"
        case "saving": return prefix + "Saving…"
        default: return s.message
        }
    }
}

struct EstimateLine: View {
    var estimate: Estimate?
    var count: Int

    var body: some View {
        if let estimate {
            HStack(spacing: 4) {
                Image(systemName: "clock")
                Text(text(estimate))
            }
            .font(.caption).foregroundStyle(.secondary)
            .help("Learned from earlier runs of this model on this Mac; updates after every generation.")
        }
    }

    private func text(_ e: Estimate) -> String {
        let work = "\(e.steps_run) step\(e.steps_run == 1 ? "" : "s")" + (e.passes > 1 ? " × 2 passes" : "")
        guard let seconds = e.seconds else {
            return "\(work) · no time estimate yet (first run of this model on this Mac)"
        }
        var text = (e.basis == "similar" ? "roughly " : "≈ ") + formatDuration(seconds)
        if count > 1 { text += " for \(count) images" }
        text += " · \(work)"
        if let load = e.load_seconds, load > 1 { text += " · +\(formatDuration(load)) to load the model" }
        else if e.load_seconds == nil { text += " · plus loading the model" }
        return text
    }
}

func formatDuration(_ seconds: Double) -> String {
    let s = Int(seconds.rounded())
    if s < 60 { return "\(max(1, s)) s" }
    let m = s / 60, r = s % 60
    return r == 0 || m >= 10 ? "\(m) min" : "\(m) min \(r) s"
}

