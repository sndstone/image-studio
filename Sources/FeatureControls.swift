import AppKit
import SwiftUI
import UniformTypeIdentifiers

extension AppModel {
    func referenceIsUsed(at index: Int) -> Bool {
        guard let info = currentModel, info.edit != "none", index < info.max_refs else { return false }
        return imageMode != "init" || info.family == "qwen21" || index == 0
    }
}

struct LorasView: View {
    @EnvironmentObject var model: AppModel
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(model.loras.indices, id: \.self) { index in
                VStack {
                    HStack {
                        Text(URL(fileURLWithPath: model.loras[index].path).lastPathComponent).lineLimit(1).truncationMode(.middle)
                        Spacer()
                        Button { model.loras.remove(at: index) } label: { Image(systemName: "minus.circle") }
                    }
                    HStack {
                        Slider(value: $model.loras[index].scale, in: 0...2, step: 0.05)
                        Text(model.loras[index].scale.formatted(.number.precision(.fractionLength(2)))).monospacedDigit()
                    }
                }
            }
            Button("Add LoRA…") { model.pickLora() }
            Text("Use LoRAs trained for \(model.currentModel?.name ?? "this model"). Switching models clears this list.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }
}

struct ReferenceReorderDrop: DropDelegate {
    let model: AppModel
    let target: URL
    func performDrop(info: DropInfo) -> Bool {
        guard let provider = info.itemProviders(for: [.plainText]).first else { return false }
        _ = provider.loadObject(ofClass: NSString.self) { object, _ in
            guard let value = object as? String, value.hasPrefix("reference:") else { return }
            Task { @MainActor in model.moveReference(URL(fileURLWithPath: String(value.dropFirst(10))), before: target) }
        }
        return true
    }
}

struct QueueReorderDrop: DropDelegate {
    let model: AppModel
    let target: UUID
    func performDrop(info: DropInfo) -> Bool {
        guard let provider = info.itemProviders(for: [.plainText]).first else { return false }
        _ = provider.loadObject(ofClass: NSString.self) { object, _ in
            guard let value = object as? String, let id = UUID(uuidString: value) else { return }
            Task { @MainActor in model.moveJob(id, before: target) }
        }
        return true
    }
}

struct QueueList: View {
    @EnvironmentObject var model: AppModel
    var body: some View {
        if !model.queue.isEmpty || !model.failedJobs.isEmpty {
            VStack(alignment: .leading, spacing: 5) {
                HStack {
                    Text(model.currentJob == nil ? "Queued (\(model.queue.count))"
                         : "Generating 1 · Queued \(model.queue.count)").font(.caption.bold())
                    Spacer()
                    if model.queuePaused || model.holdRestored {
                        Button("Resume") { model.resumeQueue() }
                            .help(model.holdRestored ? "Run the jobs left over from last time" : "Continue the queue")
                    } else {
                        Button("Pause") { model.queuePaused = true }
                    }
                    Button("Clear") { model.clearQueue() }
                }.buttonStyle(.link).font(.caption)
                ScrollView {
                    VStack(alignment: .leading, spacing: 5) {
                        if let job = model.currentJob {
                            // The running job stays at the top, so the list reads in run order.
                            HStack {
                                ProgressView().controlSize(.mini)
                                Text(job.title).lineLimit(1)
                                Spacer()
                                Text(model.currentJobLabel).foregroundStyle(.secondary).lineLimit(1)
                                Button { model.cancel() } label: { Image(systemName: "stop.circle") }
                                    .buttonStyle(.borderless)
                                    .help("Stop this job; queued jobs still run")
                            }
                            .contentShape(Rectangle())
                            .onTapGesture { model.showCurrentJob() }
                            .help("Show the image being generated")
                        }
                        ForEach(model.queue) { job in
                            HStack {
                                Image(systemName: "line.3.horizontal").foregroundStyle(.secondary)
                                Text(job.title).lineLimit(1)
                                Spacer()
                                Button { model.removeFromQueue(job) } label: { Image(systemName: "xmark.circle") }
                                    .buttonStyle(.borderless)
                            }
                            .onDrag { NSItemProvider(object: job.id.uuidString as NSString) }
                            .onDrop(of: [.plainText], delegate: QueueReorderDrop(model: model, target: job.id))
                        }
                        if !model.failedJobs.isEmpty { Text("Failed").font(.caption.bold()).foregroundStyle(.red) }
                        ForEach(model.failedJobs) { failed in
                            VStack(alignment: .leading) {
                                HStack {
                                    Text(failed.job.title).lineLimit(1)
                                    Spacer()
                                    Button("Retry") { model.retryFailed(failed) }
                                    Button { model.failedJobs.removeAll { $0.id == failed.id } } label: { Image(systemName: "xmark.circle") }
                                }
                                Text(failed.error).foregroundStyle(.red).textSelection(.enabled)
                            }
                        }
                    }.font(.caption)
                }.frame(maxHeight: 160)
            }.padding(8).background(Color.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 6))
        }
    }
}

struct SweepView: View {
    @EnvironmentObject var model: AppModel
    @Environment(\.dismiss) var dismiss
    @State private var parameter = "seed"
    @State private var values = "42, 43, 44, 45"
    @State private var error: String?
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Parameter Sweep").font(.headline)
            Picker("Parameter", selection: $parameter) {
                Text("Seed").tag("seed"); Text("Steps").tag("steps")
                Text("Guidance").tag("guidance"); Text("Change").tag("change")
            }
            TextField("Comma-separated values", text: $values).textFieldStyle(.roundedBorder)
            Text("Each value adds one job with the current settings. Results appear in the comparison grid.").font(.caption).foregroundStyle(.secondary)
            if let error { Text(error).foregroundStyle(.red).font(.caption) }
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Queue Sweep") {
                    error = model.sweep(parameter: parameter, values: values)
                    if error == nil { dismiss() }
                }.keyboardShortcut(.defaultAction)
            }
        }.padding(20).frame(width: 440)
    }
}

struct UpscaleOptionsView: View {
    @EnvironmentObject var model: AppModel
    var item: HistoryItem
    @Environment(\.dismiss) var dismiss
    private func allowed(_ scale: Double) -> Bool {
        guard let size = AppModel.displaySize(of: item.url) else { return false }
        return size.width * size.height * scale * scale <= 4_500_000
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Upscale").font(.headline)
            Picker("Scale", selection: $model.upscaleScale) {
                ForEach([1.5, 2, 3, 4], id: \.self) { scale in
                    Text("\(scale.formatted())×\(allowed(scale) ? "" : " (over 4.5 MP)")").tag(scale)
                }
            }
            LabeledContent("Softness") {
                Slider(value: $model.upscaleSoftness, in: 0...1)
                Text(model.upscaleSoftness.formatted(.number.precision(.fractionLength(2))))
            }
            Text("Output must be at most 4.5 megapixels.").font(.caption).foregroundStyle(.secondary)
            Button("Upscale") { model.upscale(item); dismiss() }.disabled(!allowed(model.upscaleScale))
        }.padding(16).frame(width: 300)
    }
}

/// Leaves text editing shortcuts alone while making image navigation work without focusing a thumbnail.
struct HistoryKeyboard: NSViewRepresentable {
    let model: AppModel
    class Coordinator {
        var observer: Any?
    }
    func makeCoordinator() -> Coordinator { Coordinator() }
    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        context.coordinator.observer = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak view, weak model] event in
            guard let model, let window = view?.window, window == NSApp.keyWindow else { return event }
            let handled = MainActor.assumeIsolated {
                let command = event.modifierFlags.intersection(.deviceIndependentFlagsMask).contains(.command)
                if command && event.charactersIgnoringModifiers == "v" && NSPasteboard.general.canReadObject(forClasses: [NSImage.self]) {
                    model.pasteReference(); return true
                }
                if window.firstResponder is NSTextView { return false }
                if command && event.keyCode == 51, let item = model.selected { model.delete(item); return true }
                if !command {
                    switch event.keyCode {
                    case 123: model.selectHistory(offset: -1); return true
                    case 124: model.selectHistory(offset: 1); return true
                    case 49: if model.selected != nil { model.showFullView.toggle(); return true }
                    default: break
                    }
                }
                return false
            }
            return handled ? nil : event
        }
        return view
    }
    func updateNSView(_ view: NSView, context: Context) {}
    static func dismantleNSView(_ view: NSView, coordinator: Coordinator) {
        if let observer = coordinator.observer { NSEvent.removeMonitor(observer) }
        coordinator.observer = nil
    }
}
