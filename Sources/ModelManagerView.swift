import AppKit
import SwiftUI

func formatBytes(_ bytes: Int) -> String {
    ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file)
}

func quantLabel(_ quant: Int) -> String { quant == 0 ? "Full" : "\(quant)-bit" }

struct ModelManagerView: View {
    @EnvironmentObject var model: AppModel
    @Environment(\.dismiss) private var dismiss
    @State private var showAddCustom = false
    @State private var showStorage = false

    var body: some View {
        VStack(spacing: 0) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Models").font(.title2.bold())
                    if let system = model.system {
                        Text("\(formatBytes(system.disk_free)) free on disk · \(formatBytes(system.memory_total)) memory · "
                             + (system.memory_in_use > 0 ? "\(formatBytes(system.memory_in_use)) used by the loaded model" : "no model loaded"))
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
                Spacer()
                Button("Add Custom Model…") { showAddCustom = true }
                Button {
                    if let dir = model.system?.models_dir { NSWorkspace.shared.open(URL(fileURLWithPath: dir)) }
                } label: { Image(systemName: "folder") }
                .help("Open the model cache folder")
            }
            .padding(16)

            if let error = model.modelError {
                Text(error).font(.caption).foregroundStyle(.red).textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading).padding(.horizontal, 16).padding(.bottom, 8)
            }
            Divider()

            ScrollViewReader { proxy in
            ScrollView {
                VStack(spacing: 10) {
                    cards(model.models.filter { !$0.isTool && !$0.isCloud })
                    let cloud = model.models.filter(\.isCloud)
                    if !cloud.isEmpty {
                        HStack(alignment: .firstTextBaseline) {
                            Text("Cloud").font(.headline)
                            Text("Run on the provider's servers with your API key, billed per image.")
                                .font(.caption).foregroundStyle(.secondary)
                            Spacer()
                            Button("API Keys…") { model.openCloudSettings() }.controlSize(.small)
                        }
                        .padding(.top, 6)
                        cards(cloud)
                    }
                    let tools = model.models.filter(\.isTool)
                    if !tools.isEmpty {
                        Text("Tools").font(.headline).frame(maxWidth: .infinity, alignment: .leading).padding(.top, 6)
                        ForEach(tools) { info in ModelRow(info: info) }
                    }
                }
                .padding(16)
            }
            .task(id: model.models.count) {
                // UI-test hook: scroll to a card, e.g. "group:FLUX.3 klein".
                let env = ProcessInfo.processInfo.environment
                if let target = env["IMAGE_STUDIO_TEST_MANAGER_SCROLL"] {
                    try? await Task.sleep(for: .seconds(1))
                    proxy.scrollTo(target, anchor: env["IMAGE_STUDIO_TEST_MANAGER_ANCHOR"] == "bottom" ? .bottom : .top)
                }
            }
            }

            Divider()
            HStack {
                Text("Models download to the Hugging Face cache and are shared with other apps that use it. Gated models need your Hugging Face token (API key) and an accepted license.")
                    .font(.caption).foregroundStyle(.secondary)
                Spacer()
                Button("Clean Up Storage…") { showStorage = true }
                    .help("Find unfinished downloads, outdated conversions, caches and unused models")
                Button("Done") { dismiss() }.keyboardShortcut(.defaultAction)
            }
            .padding(16)
        }
        .frame(minWidth: 760, idealWidth: 820, minHeight: 560, idealHeight: 680)
        .sheet(isPresented: $showAddCustom) { AddCustomModelView() }
        .sheet(isPresented: $showStorage) { StorageView() }
        .task {
            // Sizes of models that were never downloaded are looked up in the background.
            try? await model.backend.fetchModelSizes()
            await model.refreshModels()
        }
    }
}

extension ModelManagerView {
    fileprivate func cards(_ models: [ModelInfo]) -> some View {
        ForEach(ModelCard.cards(models)) { card in
            Group {
                switch card {
                case .single(let info): ModelRow(info: info)
                case .group(let name, let variants): ModelGroupCard(name: name, variants: variants)
                }
            }
            .id(card.id)
        }
    }
}

/// One card in the Model Manager: a model, or the versions of one model (its precisions) together.
enum ModelCard: Identifiable {
    case single(ModelInfo)
    case group(String, [ModelInfo])

    var id: String {
        switch self {
        case .single(let info): return info.id
        case .group(let name, _): return "group:" + name
        }
    }

    static func cards(_ models: [ModelInfo]) -> [ModelCard] {
        var cards: [ModelCard] = []
        var seen = Set<String>()
        for info in models {
            guard let group = info.group else { cards.append(.single(info)); continue }
            if seen.insert(group).inserted { cards.append(.group(group, models.filter { $0.group == group })) }
        }
        return cards
    }
}

extension ModelInfo {
    var capabilitiesText: String {
        switch edit {
        case "multi": return "Text→image, edit (up to \(max_refs) images)"
        case "single": return "Text→image, image→image"
        default: return "Text→image"
        }
    }

    var memoryText: String? {
        let parts = quant_options.compactMap { q in memory(for: q).map { "\(quantLabel(q)) ~\(Int($0)) GB" } }
        return parts.isEmpty ? nil : parts.joined(separator: " · ")
    }
}

struct ModelRow: View {
    @EnvironmentObject var model: AppModel
    var info: ModelInfo

    var body: some View {
        HStack(alignment: .top, spacing: 14) {
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 6) {
                    Text(info.name).font(.headline)
                    if info.is_new { Badge(text: "NEW", color: .green) }
                    GatedBadge(info: info)
                    if info.custom { Badge(text: "Custom", color: .purple) }
                    if info.loaded { Badge(text: "Loaded · \(quantLabel(info.loaded_quant ?? 0))", color: .blue) }
                }
                Text(info.description).font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                HStack(spacing: 12) {
                    if !info.released.isEmpty { Label(info.released, systemImage: "calendar") }
                    LicenseLabel(info: info)
                    Label(info.capabilitiesText, systemImage: "wand.and.stars")
                    if let mem = info.memoryText { Label(mem, systemImage: "memorychip") }
                    ModelPageLink(info: info)
                }
                .font(.caption).foregroundStyle(.secondary)
                ModelStatusLine(info: info)
                if let error = model.modelErrors[info.id] { Text(error).font(.caption).foregroundStyle(.red).textSelection(.enabled) }
            }
            Spacer(minLength: 8)
            ModelActions(info: info).frame(width: 150, alignment: .trailing)
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 10).fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(info.id == model.modelID ? Color.accentColor : .clear, lineWidth: 2))
    }
}

/// The versions of one model (e.g. INT4 / INT8 / BF16) in one card: each downloads on its own, or all at once.
struct ModelGroupCard: View {
    @EnvironmentObject var model: AppModel
    var name: String
    var variants: [ModelInfo]

    private var first: ModelInfo { variants[0] }
    private var missing: [ModelInfo] {
        variants.filter { !$0.status.downloaded && !$0.status.downloading && !model.modelBusy.contains($0.id) }
    }
    /// Bytes still to download (files the versions share counted once), when known.
    private var missingBytes: Int? { first.group_missing_bytes }
    /// Details every version shares go in the header; the rest per version.
    private func shared<T: Hashable>(_ value: (ModelInfo) -> T) -> Bool { Set(variants.map(value)).count == 1 }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Text(name).font(.headline)
                if variants.contains(where: \.is_new) { Badge(text: "NEW", color: .green) }
                if let loaded = variants.first(where: \.loaded) {
                    Badge(text: "Loaded · \(loaded.variant ?? quantLabel(loaded.loaded_quant ?? 0))", color: .blue)
                }
                Spacer()
                if !first.isCloud && !missing.isEmpty && variants.count > 1 {
                    Button((missing.count == variants.count ? "Download All" : "Download Remaining")
                           + (missingBytes.map { " (\(formatBytes($0)))" } ?? "")) { downloadAll() }
                        .controlSize(.small)
                        .disabled(model.modelBusy.contains("group:" + name))
                        .help("Download every version of \(name) that isn't on this Mac yet. You only need one to use the model; "
                              + "files the versions share are downloaded once.")
                }
            }
            Text(first.group_description ?? first.description)
                .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            HStack(spacing: 12) {
                if shared(\.released) && !first.released.isEmpty { Label(first.released, systemImage: "calendar") }
                if shared(\.license) { LicenseLabel(info: first) }
                if shared(\.capabilitiesText) { Label(first.capabilitiesText, systemImage: "wand.and.stars") }
            }
            .font(.caption).foregroundStyle(.secondary)
            if let error = model.modelErrors["group:" + name] {
                Text(error).font(.caption).foregroundStyle(.red).textSelection(.enabled)
            }
            VStack(spacing: 6) {
                ForEach(variants) { info in
                    ModelVariantRow(info: info, showRelease: !shared(\.released), showLicense: !shared(\.license),
                                    showCapabilities: !shared(\.capabilitiesText))
                }
            }
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 10).fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(RoundedRectangle(cornerRadius: 10)
            .stroke(variants.contains { $0.id == model.modelID } ? Color.accentColor.opacity(0.5) : .clear, lineWidth: 1))
    }

    private func downloadAll() {
        let free = model.system?.disk_free ?? 0
        if let needed = missingBytes, free > 0, needed > free {
            model.modelErrors["group:" + name] = "Not enough disk space for all versions: needs \(formatBytes(needed)), "
                + "\(formatBytes(free)) free. Download only the version you need."
            return
        }
        model.downloadGroup(name, missing)
    }
}

struct ModelVariantRow: View {
    @EnvironmentObject var model: AppModel
    var info: ModelInfo
    var showRelease = false
    var showLicense = false
    var showCapabilities = false

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 6) {
                    Text(info.variant ?? info.name).font(.subheadline.bold())
                    GatedBadge(info: info)
                    if info.loaded { Badge(text: "Loaded · \(quantLabel(info.loaded_quant ?? 0))", color: .blue) }
                    if info.id == model.modelID { Badge(text: "In use", color: .accentColor) }
                }
                Text(info.description).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                HStack(spacing: 12) {
                    if showRelease && !info.released.isEmpty { Label(info.released, systemImage: "calendar") }
                    if showLicense { LicenseLabel(info: info) }
                    if showCapabilities { Label(info.capabilitiesText, systemImage: "wand.and.stars") }
                    if let mem = info.memoryText { Label(mem, systemImage: "memorychip") }
                    ModelPageLink(info: info)
                }
                .font(.caption).foregroundStyle(.secondary)
                ModelStatusLine(info: info)
                if let error = model.modelErrors[info.id] { Text(error).font(.caption).foregroundStyle(.red).textSelection(.enabled) }
            }
            Spacer(minLength: 8)
            ModelActions(info: info).frame(width: 150, alignment: .trailing)
        }
        .padding(10)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color(nsColor: .windowBackgroundColor).opacity(0.6)))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(info.id == model.modelID ? Color.accentColor : .clear, lineWidth: 2))
    }
}

struct LicenseLabel: View {
    var info: ModelInfo

    var body: some View {
        Label(info.license, systemImage: info.commercial ? "checkmark.seal" : "exclamationmark.shield")
            .help(info.commercial ? "Commercial use allowed" : "Not licensed for commercial use")
    }
}

struct GatedBadge: View {
    @EnvironmentObject var model: AppModel
    var info: ModelInfo

    var body: some View {
        if info.gated {
            switch model.gatedAccess[info.id] ?? nil {
            case true?: Badge(text: "Unlocked", color: .green)
            case false?: Badge(text: "Accept license", color: .orange)
                .help("Open the model page and accept its license with your Hugging Face account")
            case nil: Badge(text: "Needs token", color: .orange)
            }
        }
    }
}

struct ModelPageLink: View {
    var info: ModelInfo

    var body: some View {
        if let repo = info.sources.first?.repo, let url = URL(string: "https://huggingface.co/\(repo)") {
            Link(destination: url) { Label("Model page", systemImage: "safari") }
        }
    }
}

struct ModelStatusLine: View {
    var info: ModelInfo

    var body: some View {
        let s = info.status
        if info.isCloud {
            if s.downloaded {
                Label("API key saved", systemImage: "checkmark.circle.fill").font(.caption).foregroundStyle(.green)
            } else {
                Label("Needs a \(cloudProviderName(info)) API key", systemImage: "key").font(.caption).foregroundStyle(.secondary)
            }
        } else if s.downloading {
            VStack(alignment: .leading, spacing: 3) {
                ProgressView(value: s.fraction ?? 0)
                Text("Downloading \(formatBytes(s.bytes_present))" + (s.bytes_total.map { " of \(formatBytes($0))" } ?? ""))
                    .font(.caption).foregroundStyle(.secondary).monospacedDigit()
            }
        } else if s.downloaded {
            Label(info.model_path.map { $0.hasPrefix("/") ? "Local folder" : "Downloaded" } ?? "Downloaded"
                  + (info.disk_bytes > 0 ? " · \(formatBytes(info.disk_bytes)) on disk" : ""),
                  systemImage: "checkmark.circle.fill")
                .font(.caption).foregroundStyle(.green)
        } else {
            VStack(alignment: .leading, spacing: 3) {
                let size = s.bytes_total.map(formatBytes) ?? "unknown size"
                Label(s.bytes_present > 0 ? "Partially downloaded (\(formatBytes(s.bytes_present)) of \(size))"
                                          : "Not downloaded · \(size)",
                      systemImage: "arrow.down.circle")
                    .font(.caption).foregroundStyle(.secondary)
                if let error = s.error, error != "Cancelled" {
                    Text(error).font(.caption).foregroundStyle(.red).textSelection(.enabled)
                }
            }
        }
    }
}

/// Download / load / use / optimize / delete buttons for one model or version, with the delete dialogs.
struct ModelActions: View {
    @EnvironmentObject var model: AppModel
    var info: ModelInfo
    @State private var confirmDelete = false
    @State private var confirmOwnDelete = false

    private var busy: Bool { model.modelBusy.contains(info.id) }

    var body: some View {
        let s = info.status
        VStack(alignment: .trailing, spacing: 6) {
            if info.isCloud {
                if s.downloaded {
                    Button("Use") { model.modelID = info.id }.disabled(info.id == model.modelID)
                } else {
                    Button("Add API Key…") { model.openCloudSettings() }.buttonStyle(.borderedProminent)
                }
            } else if busy {
                ProgressView().controlSize(.small)
            } else if s.downloading {
                Button("Cancel") { model.cancelDownload(info) }
            } else if s.downloaded {
                if info.loaded {
                    Button("Unload") { model.unloadModel() }
                        .help("Free the memory this model uses")
                } else if !info.isTool {
                    Button("Load") { model.loadModel(info) }
                        .help("Load into memory now so the first image starts faster")
                        .disabled(model.isGenerating)
                }
                if !info.isTool {
                    Button("Use") { model.modelID = info.id }
                        .disabled(info.id == model.modelID)
                }
                if info.optimizable == true {
                    Menu("Optimize…") {
                        ForEach(info.quant_options.filter { $0 > 0 }, id: \.self) { quant in
                            Button("\(quantLabel(quant))\(info.optimized?[String(quant)] == true ? " (saved)" : "")") { model.optimizeModel(info, quant: quant) }
                                .disabled(info.optimized?[String(quant)] == true)
                        }
                    }.disabled(model.isGenerating)
                    Text("Extra disk space: several GB for a model copy at the chosen precision.").font(.caption2).foregroundStyle(.secondary)
                }
                Button("Delete…", role: .destructive) { confirmDelete = true }
                    .disabled(model.isGenerating)
                if !info.shared_with.isEmpty {
                    Button("Remove only this variant…", role: .destructive) { confirmOwnDelete = true }
                        .help("Frees about \(formatBytes(info.own_bytes ?? 0)) without removing shared repositories")
                        .disabled(model.isGenerating)
                }
            } else {
                Button(s.bytes_present > 0 ? "Resume" : "Download") { model.download(info) }
                    .buttonStyle(.borderedProminent)
                if info.custom {
                    Button("Remove", role: .destructive) { confirmDelete = true }
                } else if s.bytes_present > 0 {
                    Button("Delete…", role: .destructive) { confirmDelete = true }
                        .help("Remove the partial download")
                }
            }
        }
        .controlSize(.small)
        .confirmationDialog("Delete \(info.name)?", isPresented: $confirmDelete) {
            Button("Delete", role: .destructive) { model.deleteModel(info) }
            if !info.shared_with.isEmpty {
                Button("Remove only this variant (\(formatBytes(info.own_bytes ?? 0)))", role: .destructive) { model.deleteModel(info, scope: "own") }
            }
        } message: {
            Text(deleteMessage)
        }
        .confirmationDialog("Remove only \(info.name)?", isPresented: $confirmOwnDelete) {
            Button("Remove only this variant", role: .destructive) { model.deleteModel(info, scope: "own") }
        } message: {
            Text("Frees about \(formatBytes(info.own_bytes ?? 0)). Shared repositories used by other models are preserved.")
        }
    }

    private var deleteMessage: String {
        var text = "Frees about \(formatBytes(info.disk_bytes)). You can download it again later."
        let siblings = info.shared_with.compactMap { id in model.models.first { $0.id == id && $0.status.downloaded }?.name }
        if !siblings.isEmpty { text += " This also removes \(siblings.joined(separator: ", ")), which uses the same files." }
        if info.custom { text += " The custom entry is removed from the list." }
        return text
    }
}

struct Badge: View {
    var text: String
    var color: Color
    var body: some View {
        Text(text).font(.caption2.bold())
            .padding(.horizontal, 6).padding(.vertical, 2)
            .background(color.opacity(0.18), in: Capsule())
            .foregroundStyle(color)
    }
}

struct AddCustomModelView: View {
    @EnvironmentObject var model: AppModel
    @Environment(\.dismiss) private var dismiss
    @State private var name = ""
    @State private var path = ""
    @State private var architecture = "flux2-klein-4b"
    @State private var error: String?
    @State private var saving = false

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Add Custom Model").font(.title3.bold())
            Text("Use a fine-tuned or pre-quantized checkpoint from Hugging Face, or a model folder on this Mac (for example one saved with mflux-save). It must use one of the supported architectures.")
                .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            Form {
                TextField("Name", text: $name, prompt: Text("My model"))
                HStack {
                    TextField("Source", text: $path, prompt: Text("owner/repo or /path/to/folder"))
                    Button("Choose Folder…") {
                        let panel = NSOpenPanel()
                        panel.canChooseDirectories = true
                        panel.canChooseFiles = false
                        if panel.runModal() == .OK, let url = panel.url {
                            path = url.path
                            if name.isEmpty { name = url.lastPathComponent }
                        }
                    }
                }
                Picker("Architecture", selection: $architecture) {
                    ForEach(model.architectures) { Text($0.name).tag($0.id) }
                }
            }
            if let error { Text(error).font(.caption).foregroundStyle(.red) }
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Add") {
                    saving = true
                    Task {
                        error = await model.addCustomModel(name: name, path: path, architecture: architecture)
                        saving = false
                        if error == nil { dismiss() }
                    }
                }
                .keyboardShortcut(.defaultAction)
                .disabled(path.trimmingCharacters(in: .whitespaces).isEmpty || saving)
            }
        }
        .padding(20)
        .frame(width: 520)
    }
}

struct StorageView: View {
    @EnvironmentObject var model: AppModel
    @Environment(\.dismiss) private var dismiss
    @State private var items: [StorageItem]?
    @State private var chosen: Set<String> = []
    @State private var message: String?
    @State private var working = false

    private var chosenBytes: Int { items?.filter { chosen.contains($0.id) }.reduce(0) { $0 + $1.bytes } ?? 0 }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Clean Up Storage").font(.title3.bold())
            Text("Downloaded models are managed in their rows; this finds everything else Image Studio can free.")
                .font(.caption).foregroundStyle(.secondary)
            if let items {
                if items.isEmpty {
                    Label("Nothing to clean up.", systemImage: "checkmark.circle").foregroundStyle(.green)
                        .frame(maxWidth: .infinity, minHeight: 120)
                } else {
                    ScrollView {
                        VStack(alignment: .leading, spacing: 14) {
                            ForEach(categories(items), id: \.self) { category in
                                VStack(alignment: .leading, spacing: 6) {
                                    Text(category).font(.headline)
                                    if category == "Not used by Image Studio" {
                                        Label("Shared with other apps that use Hugging Face models. Only delete what you know is unused.",
                                              systemImage: "exclamationmark.triangle")
                                            .font(.caption).foregroundStyle(.orange)
                                    }
                                    ForEach(items.filter { $0.category == category }) { item in row(item) }
                                }
                            }
                        }
                        .padding(.vertical, 4)
                    }
                    .frame(minHeight: 260)
                }
            } else {
                ProgressView("Scanning…").frame(maxWidth: .infinity, minHeight: 200)
            }
            if let message { Text(message).font(.caption).foregroundStyle(.secondary) }
            HStack {
                Spacer()
                Button("Close") { dismiss() }.keyboardShortcut(.cancelAction)
                Button(chosen.isEmpty ? "Free Up Space" : "Free \(formatBytes(chosenBytes))") { clean() }
                    .buttonStyle(.borderedProminent)
                    .disabled(chosen.isEmpty || working || model.isGenerating)
            }
        }
        .padding(20)
        .frame(width: 600)
        .task { await scan() }
    }

    private func row(_ item: StorageItem) -> some View {
        Toggle(isOn: Binding(get: { chosen.contains(item.id) },
                             set: { if $0 { chosen.insert(item.id) } else { chosen.remove(item.id) } })) {
            VStack(alignment: .leading, spacing: 2) {
                HStack {
                    Text(item.label).lineLimit(1).truncationMode(.middle)
                    Spacer()
                    Text(formatBytes(item.bytes)).monospacedDigit().foregroundStyle(.secondary)
                }
                Text(item.detail).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
        }
        .toggleStyle(.checkbox)
    }

    private func categories(_ items: [StorageItem]) -> [String] {
        var seen: [String] = []
        for item in items where !seen.contains(item.category) { seen.append(item.category) }
        return seen
    }

    private func scan() async {
        do {
            let found = try await model.scanStorage()
            items = found
            chosen = Set(found.filter(\.recommended).map(\.id))
        } catch {
            items = []
            message = error.localizedDescription
        }
    }

    private func clean() {
        working = true
        Task {
            defer { working = false }
            do {
                let result = try await model.cleanStorage(ids: Array(chosen))
                message = "Freed \(formatBytes(result.freed))."
                    + (result.failed.isEmpty ? "" : " Couldn't remove: \(result.failed.joined(separator: "; "))")
                items = nil
                await scan()
            } catch {
                message = error.localizedDescription
            }
        }
    }
}
