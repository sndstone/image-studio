import AppKit
import Combine
import ImageIO
import SwiftUI
import UniformTypeIdentifiers

struct FailedJob: Identifiable, Codable {
    var id: UUID { job.id }
    var job: QueuedJob
    var error: String
}

struct SavedQueue: Codable {
    var jobs: [QueuedJob]
    var failed: [FailedJob]
    var paused: Bool

    init(jobs: [QueuedJob], failed: [FailedJob], paused: Bool) {
        self.jobs = jobs; self.failed = failed; self.paused = paused
    }

    enum CodingKeys: String, CodingKey { case jobs, failed, paused }
    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        jobs = try values.decode([QueuedJob].self, forKey: .jobs)
        failed = try values.decodeIfPresent([FailedJob].self, forKey: .failed) ?? []
        paused = try values.decodeIfPresent(Bool.self, forKey: .paused) ?? false
    }
}

extension AppModel {
    var engineSettings: EngineSettings { EngineSettings(embed_metadata: embedMetadata, low_memory: lowMemory, offline: offline) }

    func updateEngineSettings() {
        UserDefaults.standard.set(embedMetadata, forKey: "embedMetadata")
        UserDefaults.standard.set(lowMemory, forKey: "lowMemory")
        UserDefaults.standard.set(offline, forKey: "offline")
        guard phase == .ready else { return }
        Task {
            do { try await backend.setSettings(engineSettings) }
            catch { errorMessage = error.localizedDescription }
        }
    }

    func normalizedDimensions(for model: ModelInfo) -> (Int, Int) {
        let multiple = max(1, model.size_multiple)
        let minimum = ((256 + multiple - 1) / multiple) * multiple
        var w = max(minimum, min(2048, width) / multiple * multiple)
        var h = max(minimum, min(2048, height) / multiple * multiple)
        let cap = (model.max_megapixels ?? 2.0) * 1_000_000
        if Double(w * h) > cap {
            let ratio = sqrt(cap / Double(w * h))
            w = max(minimum, Int(Double(w) * ratio) / multiple * multiple)
            h = max(minimum, Int(Double(h) * ratio) / multiple * multiple)
            while Double(w * h) > cap && (w > minimum || h > minimum) {
                if w >= h && w > minimum { w -= multiple } else if h > minimum { h -= multiple } else { w -= multiple }
            }
        }
        return (w, h)
    }

    func normalizeSize() {
        guard !restoringSettings else { return }
        guard let model = currentModel else { width = min(2048, max(256, width)); height = min(2048, max(256, height)); return }
        (width, height) = normalizedDimensions(for: model)
    }

    func saveGenerationSettings(model id: String) {
        guard !restoringSettings, booted, let info = models.first(where: { $0.id == id }) else { return }
        if let data = try? JSONEncoder().encode(makeRequest(for: info)),
           var json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            // Save the editor, including inputs the current model cannot use.
            json["image_paths"] = references.map(\.path)
            json["negative_prompt"] = negativePrompt
            json["strength"] = strength
            json["width"] = width
            json["height"] = height
            json["count"] = count
            json["steps"] = steps
            json["guidance"] = guidance
            json["image_mode"] = imageMode
            json["fast"] = fast
            json["enhance_prompt"] = enhancePrompt
            json["mask_path"] = maskPath?.path
            json["auto_mask"] = autoMask
            json["ui_seed"] = seed
            json["ui_random_seed"] = randomSeed
            json["ui_auto_size"] = autoSize
            json["ui_size_reference"] = sizeReference
            json["ui_size_reference_auto"] = sizeReferenceIsAuto
            json["ui_selected_path"] = selected?.url.path
            if references.indices.contains(sizeReference) {
                json["ui_size_reference_path"] = references[sizeReference].path
            }
            guard let saved = try? JSONSerialization.data(withJSONObject: json) else { return }
            UserDefaults.standard.set(saved, forKey: "generation.session")
        }
    }

    func removeLegacyGenerationSettings() {
        let defaults = UserDefaults.standard
        for key in defaults.dictionaryRepresentation().keys
            where (key.hasPrefix("generation.") && key != "generation.session") || key.hasPrefix("autoSize.") {
            defaults.removeObject(forKey: key)
        }
    }

    func restoreGenerationSettings() {
        defer { removeLegacyGenerationSettings() }
        let defaults = UserDefaults.standard
        let legacyKey = "generation." + modelID
        let data = defaults.data(forKey: "generation.session") ?? defaults.data(forKey: legacyKey)
        guard let data, var json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            normalizeSize()
            return
        }
        if json["ui_auto_size"] == nil {
            json["ui_auto_size"] = defaults.object(forKey: "autoSize." + modelID) as? Bool ?? false
        }
        let wasRestoring = restoringSettings
        restoringSettings = true
        restoreSettings(json)
        if let value = json["ui_seed"] as? Int { seed = value }
        if let value = json["ui_random_seed"] as? Bool { randomSeed = value }
        autoSize = json["ui_auto_size"] as? Bool ?? false
        if let path = json["ui_selected_path"] as? String, FileManager.default.fileExists(atPath: path) {
            let url = URL(fileURLWithPath: path)
            let settings = Self.settingsJSON(at: url)
            selected = HistoryItem(url: url, seed: settings?["seed"] as? Int ?? 0, seconds: settings?["seconds"] as? Double ?? 0)
        }
        restoringSettings = wasRestoring
        normalizeSize()
        if let saved = try? JSONSerialization.data(withJSONObject: json) {
            defaults.set(saved, forKey: "generation.session")
        }
    }

    func resetModelDefaults() {
        applyModelDefaults()
        saveGenerationSettings(model: modelID)
    }

    var windowIsVisible: Bool {
        NSApp.windows.contains { $0.canBecomeMain && $0.isVisible && !$0.isMiniaturized && $0.occlusionState.contains(.visible) }
    }

    var hasRunningWork: Bool {
        currentJob != nil || isGenerating || status?.busy != nil || !downloadRequests.isEmpty || models.contains { $0.status.downloading }
    }

    func updateActivity() {
        let needed = !isShuttingDown && (hasRunningWork || (!queuePaused && !queue.isEmpty))
        if needed, activity == nil {
            activity = ProcessInfo.processInfo.beginActivity(options: [.userInitiated, .idleSystemSleepDisabled], reason: "Generating images or downloading models")
        } else if !needed, let activity {
            ProcessInfo.processInfo.endActivity(activity)
            self.activity = nil
        }
    }

    func startPendingItems(for job: QueuedJob) {
        let count: Int
        let upscale: Bool
        switch job.kind {
        case .generate(let request): count = request.count; upscale = false
        case .upscale: count = 1; upscale = true
        }
        pendingItems = (0..<max(1, count)).map { PendingItem(index: $0, isUpscale: upscale) }
        selected = nil
        selectedPending = pendingItems[0].id
        previewPending = pendingItems.first?.id
        followJob = true
    }

    func selectPending(_ id: UUID) {
        guard pendingItems.contains(where: { $0.id == id && $0.result == nil }) else { return }
        followJob = id == previewPending  // clicking the running image resumes following
        selected = nil
        selectedPending = id
        closeCompare()
    }

    var selectedPendingItem: PendingItem? {
        pendingItems.first { $0.id == selectedPending && $0.result == nil }
    }

    func pendingLabel(_ item: PendingItem) -> String {
        guard item.id == previewPending else { return "Waiting" }
        if item.isUpscale { return "Upscaling" }
        let batch = pendingItems.count > 1 ? "\(item.index + 1)/\(pendingItems.count) · " : ""
        if let status, status.total_steps > 0, status.step > 0 {
            return batch + "Step \(status.step)/\(status.total_steps)"
        }
        return batch + "Generating"
    }

    /// Progress of the running job for the queue list, e.g. "2/4 · Step 5/8".
    var currentJobLabel: String {
        let running = pendingItems.first { $0.id == previewPending && $0.result == nil }
        return running.map(pendingLabel) ?? "Generating"
    }

    func showCurrentJob() {
        if let id = previewPending ?? pendingItems.first(where: { $0.result == nil })?.id { selectPending(id) }
    }

    /// Position 1 is the image being generated; finished images (including earlier ones of the same batch) follow,
    /// newest first, like the rest of History.
    var galleryItems: [GalleryItem] {
        let running = pendingItems.first { $0.id == previewPending && $0.result == nil }
            ?? pendingItems.first { $0.result == nil }
        let head = running.map { [GalleryItem(id: .pending($0.id), pending: $0)] } ?? []
        return head + filteredHistory.map { GalleryItem(id: .result($0.url), result: $0) }
    }

    /// IMAGE_STUDIO_DEBUG=1 prints gallery/selection state changes (for UI tests).
    func debugGallery(_ event: String) {
        guard ProcessInfo.processInfo.environment["IMAGE_STUDIO_DEBUG"] != nil else { return }
        let slots = pendingItems.map { "\($0.index):\($0.result?.url.lastPathComponent ?? "-")\($0.id == selectedPending ? "*" : "")\($0.id == previewPending ? "p" : "")" }
        let tiles = galleryItems.prefix(4).map { $0.pending != nil ? "P" : ($0.result?.url.lastPathComponent ?? "?") }
        FileHandle.standardError.write(Data("[gallery] \(event) slots=\(slots) selected=\(selected?.url.lastPathComponent ?? "nil") tiles=\(tiles)\n".utf8))
    }

    func finishPendingItems() {
        let wasPending = selectedPending != nil
        pendingItems = []
        previewPending = nil
        if previewImage != nil { previewImage = nil }
        selectedPending = nil
        if wasPending { selected = history.first }
    }

    func prepareJob(_ request: GenerateRequest) async throws -> GenerateRequest? {
        var request = request
        let flight: Preflight
        do { flight = try await retryBusy { try await self.backend.preflight(request) } }
        catch let error as BackendError where error.statusCode == 404 { return request }
        if jobCancelRequested { throw CancellationError() }
        preflightNote = flight.level == "ok" ? nil : flight.message
        guard flight.level == "critical" else { return request }
        let alert = NSAlert()
        alert.messageText = "Memory may be too low for this job"
        alert.informativeText = flight.message
        alert.alertStyle = .warning
        alert.addButton(withTitle: "Generate anyway")
        let offerFour = flight.suggest_quant == 4 && models.first(where: { $0.id == request.model })?.quant_options.contains(4) == true
        if offerFour { alert.addButton(withTitle: "Use 4-bit") }
        alert.addButton(withTitle: "Cancel")
        let answer = alert.runModal()
        if answer == .alertFirstButtonReturn { return request }
        if offerFour && answer == .alertSecondButtonReturn { request.quantize = 4; return request }
        return nil
    }

    func retryBusy<T>(_ work: () async throws -> T) async throws -> T {
        let deadline = Date().addingTimeInterval(180)
        var lastBusy: BackendError?
        while true {
            try Task.checkCancellation()
            if jobCancelRequested { throw CancellationError() }
            if let lastBusy, Date() >= deadline { throw lastBusy }
            do { return try await work() }
            catch let error as BackendError where error.isBusyConflict && Date() < deadline {
                lastBusy = error
                preflightNote = "Waiting for the image engine: " + error.message
                let wait = max(0, min(5, deadline.timeIntervalSinceNow))
                try await Task.sleep(nanoseconds: UInt64(wait * 1_000_000_000))
            }
        }
    }

    func acceptResults(_ items: [HistoryItem], job: QueuedJob) {
        guard currentJob?.id == job.id else { return }
        let changed = items.enumerated().filter { index, item in
            pendingItems.indices.contains(index) && pendingItems[index].result != item
                && FileManager.default.fileExists(atPath: item.url.path)
        }
        guard !changed.isEmpty else { return }
        historyRevision += 1
        debugGallery("accept \(changed.map { $0.element.url.lastPathComponent })")
        let existing = Set(history.map(\.url))
        let fresh = changed.map(\.element).filter { !existing.contains($0.url) }
        for (index, item) in changed {
            let id = pendingItems[index].id
            let previous = pendingItems[index].result
            pendingItems[index].result = item
            if selectedPending == id || (previous != nil && selected == previous) { selected = item }
            if previewPending == id { if previewImage != nil { previewImage = nil }; previewPending = nil }
            invalidateMetadata(at: item.url)
            loadMetadata(for: item.url)
        }
        let results = pendingItems.compactMap(\.result).filter { FileManager.default.fileExists(atPath: $0.url.path) }.reversed()
        let urls = Set(results.map(\.url))
        history = results + history.filter { !urls.contains($0.url) }
        historyIndex = results + historyIndex.filter { !urls.contains($0.url) }
        if let group = job.compareGroup, group == activeCompareGroup {
            let compared = Set(compareItems.map(\.url))
            compareItems = compareItems.map { old in items.first { $0.url == old.url } ?? old }
                + fresh.filter { !compared.contains($0.url) }
        }
    }

    func restartEngine() {
        guard !isStartingBackend else { return }
        Task {
            phase = .starting
            do {
                // Reuse the token already in memory; restarting doesn't require another Keychain read.
                try await startBackend(token: hasSavedToken ? tokenDraft : nil)
                await refreshModels()
                try requireLiveBackend()
                phase = .ready
                runNextJob()
            } catch { phase = .failed(error.localizedDescription) }
        }
    }

    var queueURL: URL { Backend.runtimeDir.appendingPathComponent("queue.json") }
    func persistQueue() {
        guard !restoringQueue, !isShuttingDown else { return }
        let saved = SavedQueue(jobs: (currentJob.map { [$0] } ?? []) + queue, failed: failedJobs, paused: queuePaused)
        let directory = Backend.runtimeDir
        let file = queueURL
        let previous = queueWriteTask
        // Chain snapshots so a slower, older write can never overwrite the latest queue.
        let write = Task.detached(priority: .utility) { () -> String? in
            _ = await previous?.value
            do {
                try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
                try JSONEncoder().encode(saved).write(to: file, options: .atomic)
                return nil
            } catch { return error.localizedDescription }
        }
        queueWriteTask = write
        Task {
            if let message = await write.value { errorMessage = "Could not save the queue: " + message }
        }
    }

    func restoreQueue() async {
        let file = queueURL
        let previous = queueWriteTask
        restoringQueue = true
        defer { restoringQueue = false }
        do {
            let saved = try await Task.detached(priority: .utility) { () throws -> SavedQueue? in
                _ = await previous?.value
                guard FileManager.default.fileExists(atPath: file.path) else { return nil }
                return try JSONDecoder().decode(SavedQueue.self, from: Data(contentsOf: file))
            }.value
            guard let saved else { return }
            // Pending jobs pause to avoid silently repeating a job interrupted by a relaunch.
            queuePaused = saved.paused
            holdRestored = !saved.jobs.isEmpty
            queue = saved.jobs.map { var job = $0; job.restored = true; return job }
            failedJobs = saved.failed
            if !queue.isEmpty {
                preflightNote = "\(queue.count) job\(queue.count == 1 ? "" : "s") from last time are waiting (paused). New images still start right away; press Resume to run them."
            }
        } catch { errorMessage = "Could not restore the queue: " + error.localizedDescription }
    }

    func retryFailed(_ failed: FailedJob) {
        failedJobs.removeAll { $0.id == failed.id }
        enqueue(failed.job)
    }

    func moveJob(_ id: UUID, before target: UUID) {
        guard id != target, let source = queue.firstIndex(where: { $0.id == id }), queue.contains(where: { $0.id == target }) else { return }
        let job = queue.remove(at: source)
        if let destination = queue.firstIndex(where: { $0.id == target }) { queue.insert(job, at: destination) }
    }

    func variations(_ item: HistoryItem? = nil) {
        if let item {
            guard Self.settingsJSON(at: item.url) != nil else { errorMessage = "No saved settings found in this image."; return }
            reuseSettings(of: item)
        }
        guard let info = currentModel, !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        let group = UUID()
        activeCompareGroup = group; compareItems = []
        let base = min(Int.max - 4, max(0, item?.seed ?? (randomSeed ? Int.random(in: 0..<Int(Int32.max)) : seed)))
        for offset in 1...4 {
            var request = makeRequest(for: info)
            request.seed = base + offset; request.count = 1
            enqueue(QueuedJob(kind: .generate(request), title: "Variation \(offset) · \(info.name)", compareGroup: group))
        }
    }

    func sweep(parameter: String, values: String) -> String? {
        guard let info = currentModel, !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return "Enter a prompt first." }
        let tokens = values.split(separator: ",", omittingEmptySubsequences: false)
        let numbers = tokens.compactMap { Double($0.trimmingCharacters(in: .whitespaces)) }
        guard !numbers.isEmpty, numbers.count == tokens.count, numbers.count <= 64, numbers.allSatisfy({ $0.isFinite }) else { return "Enter up to 64 comma-separated numbers." }
        for value in numbers {
            switch parameter {
            case "seed": if value < 0 || value >= Double(Int.max - 1) || value.rounded() != value { return "Seeds must be nonnegative integers." }
            case "steps": if value < 1 || value > 100 || value.rounded() != value { return "Steps must be integers from 1 to 100." }
            case "guidance": if value < 0 || value > 20 { return "Guidance must be from 0 to 20." }
            default: if value < 0.1 || value > 0.95 { return "Change must be from 0.1 to 0.95." }
            }
        }
        let group = UUID(); activeCompareGroup = group; compareItems = []
        let sharedSeed = randomSeed ? Int.random(in: 0..<Int(Int32.max)) : seed
        for value in numbers {
            var request = makeRequest(for: info); request.count = 1; request.seed = sharedSeed
            switch parameter {
            case "seed": request.seed = Int(value)
            case "steps": request.steps = Int(value)
            case "guidance": request.guidance = value
            default: request.strength = value
            }
            enqueue(QueuedJob(kind: .generate(request), title: "\(parameter) \(value.formatted()) · \(info.name)", compareGroup: group))
        }
        return nil
    }

    nonisolated static func scanHistory(_ directory: URL) -> HistoryScan {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(at: directory, includingPropertiesForKeys: [.contentModificationDateKey]) else { return HistoryScan(items: [], descriptions: [:], metadata: [:]) }
        var result: [(Date, HistoryItem)] = []
        var descriptions: [String: HistoryDescription] = [:]
        var metadata: [URL: HistoryMetadata] = [:]
        for case let url as URL in enumerator where url.pathExtension.lowercased() == "png" {
            let version = HistoryMetadataVersion(at: url)
            let json = settingsJSON(at: url)
            var record = readMetadata(at: url, json: json)
            record.version = version
            metadata[url] = record
            descriptions[url.path] = record.description
            let date = (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            result.append((date, HistoryItem(url: url, seed: json?["seed"] as? Int ?? 0, seconds: json?["seconds"] as? Double ?? 0)))
        }
        return HistoryScan(items: result.sorted { $0.0 > $1.0 }.map(\.1), descriptions: descriptions, metadata: metadata)
    }

    nonisolated static func settingsJSON(at url: URL) -> [String: Any]? {
        let sidecar = url.deletingPathExtension().appendingPathExtension("json")
        if let data = try? Data(contentsOf: sidecar), let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] { return json }
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let exif = properties[kCGImagePropertyExifDictionary] as? [CFString: Any] else { return nil }
        let comment = exif[kCGImagePropertyExifUserComment]
        var text = comment as? String
        if let data = comment as? Data { text = String(data: data, encoding: .utf8) }
        guard let value = text, let start = value.firstIndex(of: "{"), let end = value.lastIndex(of: "}"),
              let data = String(value[start...end]).data(using: .utf8) else { return nil }
        return try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    }

    func importSettings(from url: URL) -> Bool {
        guard url.pathExtension.lowercased() == "png", let json = Self.settingsJSON(at: url), json["prompt"] != nil, json["model"] != nil else { return false }
        restoreSettings(json)
        return true
    }

    func moveReference(_ url: URL, before target: URL) {
        guard url != target, let index = references.firstIndex(of: url) else { return }
        var reordered = references
        reordered.remove(at: index)
        guard let destination = reordered.firstIndex(of: target) else { return }
        reordered.insert(url, at: destination)
        references = reordered
    }

    func pasteReference() {
        let pasteboard = NSPasteboard.general
        let urls = pasteboard.readObjects(forClasses: [NSURL.self]) as? [URL] ?? []
        let data = (pasteboard.types ?? []).filter { ImageImports.isImageType($0.rawValue) }.compactMap { pasteboard.data(forType: $0) }
        let image = data.isEmpty ? (pasteboard.readObjects(forClasses: [NSImage.self]) as? [NSImage])?.first : nil
        let snapshot = image?.copy() as? NSImage
        let directory = Backend.runtimeDir.appendingPathComponent("dropped", isDirectory: true)
        enqueueReferenceImport {
            ImageImports.paste(urls: urls, data: data, image: snapshot, directory: directory)
        }
    }

    /// A dropped file: restores its settings if it is one of our PNGs, otherwise adds it as a reference.
    /// One queued operation, so a later paste or drop can't overtake it.
    func importDroppedFile(_ url: URL) {
        enqueueReferenceWork({ () -> ([String: Any]?, [URL]) in
            if url.pathExtension.lowercased() == "png", let json = Self.settingsJSON(at: url),
               json["prompt"] != nil, json["model"] != nil {
                return (json, [])
            }
            return (nil, ImageImports.validURLs([url]))
        }) { [weak self] json, urls in
            guard let self else { return }
            if let json { self.restoreSettings(json, invalidateImports: false); return }
            let fresh = urls.filter { !self.references.contains($0) }
            if !fresh.isEmpty { self.references.append(contentsOf: fresh) }
        }
    }

    func pickUpscaleFile() {
        let panel = NSOpenPanel(); panel.allowedContentTypes = [.image]
        if panel.runModal() == .OK, let url = panel.url { upscaleFileItem = HistoryItem(url: url, seed: 0, seconds: 0) }
    }

    func updateFilteredHistory() {
        let search = historySearch.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let items = search.isEmpty && !favoritesOnly ? history : historyIndex
        let filtered = items.filter { item in
            if favoritesOnly && !favorites.contains(item.url.path) { return false }
            if search.isEmpty { return true }
            let description = historyDescriptions[item.url.path]
            let name = modelSearchNames[description?.model ?? ""] ?? ""
            return ((description?.searchText ?? "") + " " + name).contains(search)
        }
        if filteredHistory != filtered { filteredHistory = filtered }
    }

    /// macOS memory-pressure notifications, forwarded to the engine (it shrinks its cache, skips previews and
    /// unloads an idle model on critical pressure).
    func startPressureMonitor() {
        guard pressureSource == nil else { return }
        let source = DispatchSource.makeMemoryPressureSource(eventMask: [.normal, .warning, .critical], queue: .main)
        source.setEventHandler { [weak self] in
            let event = source.data
            let level = event.contains(.critical) ? "critical" : event.contains(.warning) ? "warning" : "normal"
            if level != "normal" { ImageThumbnails.purge() }
            Task { @MainActor in try? await self?.backend.setPressure(level) }
        }
        source.resume()
        pressureSource = source
    }

    func selectHistory(offset: Int) {
        followJob = false
        let items = galleryItems
        guard !items.isEmpty else { return }
        let index = items.firstIndex { entry in
            if let pending = entry.pending { return pending.id == selectedPending }
            return entry.result?.url == selected?.url
        } ?? 0
        let entry = items[min(items.count - 1, max(0, index + offset))]
        if let pending = entry.pending { selectPending(pending.id) }
        else { selected = entry.result }
        closeCompare()
    }

    func toggleFavorite(_ item: HistoryItem) {
        if favorites.contains(item.url.path) { favorites.remove(item.url.path) } else { favorites.insert(item.url.path) }
        do {
            try FileManager.default.createDirectory(at: Backend.runtimeDir, withIntermediateDirectories: true)
            try JSONEncoder().encode(favorites).write(to: Backend.runtimeDir.appendingPathComponent("favorites.json"), options: .atomic)
        } catch { errorMessage = "Could not save favorites: " + error.localizedDescription }
    }

    func restoreLibraryState() {
        let file = Backend.runtimeDir.appendingPathComponent("favorites.json")
        guard FileManager.default.fileExists(atPath: file.path) else { return }
        do { favorites = try JSONDecoder().decode(Set<String>.self, from: Data(contentsOf: file)) }
        catch { errorMessage = "Could not read favorites: " + error.localizedDescription }
    }

    nonisolated static func scanAppDateFolders(_ directory: URL) -> [URL] {
        let fm = FileManager.default
        return ((try? fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: [.isDirectoryKey])) ?? []).filter { folder in
            guard folder.lastPathComponent.range(of: #"^\d{4}-\d{2}-\d{2}$"#, options: .regularExpression) != nil,
                  (try? folder.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true else { return false }
            let files = (try? fm.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? []
            let images = files.filter { $0.pathExtension.lowercased() == "png" }
            guard !images.isEmpty, images.allSatisfy({ image in
                let json = Self.settingsJSON(at: image)
                return json?["model"] != nil && json?["prompt"] != nil && json?["seed"] != nil
                    && fm.fileExists(atPath: image.deletingPathExtension().appendingPathExtension("json").path)
            }) else { return false }
            let allowed = Set(images.flatMap { [$0.lastPathComponent, $0.deletingPathExtension().appendingPathExtension("json").lastPathComponent] })
            return files.allSatisfy { allowed.contains($0.lastPathComponent) }
        }
    }

    func requestClearHistory() {
        let directory = outputDir
        Task {
            let count = await Task.detached(priority: .utility) {
                Self.scanAppDateFolders(directory).reduce(0) { count, folder in
                    count + (((try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? []).filter { $0.pathExtension.lowercased() == "png" }.count)
                }
            }.value
            clearHistoryCount = count
            confirmClearHistory = true
        }
    }

    func comparisonSource(for item: HistoryItem) -> URL? {
        historyMetadata[item.url]?.comparisonSource
    }
}
