import AppKit
import Foundation
import Combine
import ImageIO
import SwiftUI
import UniformTypeIdentifiers

struct AspectPreset: Identifiable, Hashable {
    var id: String { label }
    var label: String
    var width: Int
    var height: Int
}

let aspectPresets = [
    AspectPreset(label: "1:1", width: 1024, height: 1024),
    AspectPreset(label: "4:3", width: 1152, height: 864),
    AspectPreset(label: "3:4", width: 864, height: 1152),
    AspectPreset(label: "3:2", width: 1216, height: 816),
    AspectPreset(label: "2:3", width: 816, height: 1216),
    AspectPreset(label: "16:9", width: 1360, height: 768),
    AspectPreset(label: "9:16", width: 768, height: 1360),
]

struct HistoryDescription {
    var prompt: String
    var model: String
    var searchText: String

    init(prompt: String, model: String) {
        self.prompt = prompt
        self.model = model
        searchText = (prompt + " " + model).lowercased()
    }
}

struct HistoryScan {
    var items: [HistoryItem]
    var descriptions: [String: HistoryDescription]
    var metadata: [URL: HistoryMetadata]
}

struct HistoryItem: Identifiable, Hashable, Codable {
    var id: URL { url }
    var url: URL
    var seed: Int
    var seconds: Double
}

/// One unit of work in the queue: a generation (possibly part of a model comparison) or an upscale.
struct QueuedJob: Identifiable, Hashable, Codable {
    enum Kind: Hashable, Codable {
        case generate(GenerateRequest)
        case upscale(URL, Double, Double)
    }

    var id = UUID()
    var kind: Kind
    var title: String  // what the queue list shows
    var compareGroup: UUID?
    /// Restored from a previous session: waits for Resume, while jobs started now still run.
    var restored = false

    enum CodingKeys: String, CodingKey { case id, kind, title, compareGroup, restored }

    init(kind: Kind, title: String, compareGroup: UUID? = nil) {
        self.kind = kind
        self.title = title
        self.compareGroup = compareGroup
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(UUID.self, forKey: .id)
        kind = try c.decode(Kind.self, forKey: .kind)
        title = try c.decode(String.self, forKey: .title)
        compareGroup = try c.decodeIfPresent(UUID.self, forKey: .compareGroup)
        restored = try c.decodeIfPresent(Bool.self, forKey: .restored) ?? false
    }
}

struct PendingItem: Identifiable {
    var id = UUID()
    var index: Int
    var isUpscale: Bool
    var result: HistoryItem?
}

struct GalleryItem: Identifiable {
    enum ID: Hashable { case pending(UUID), result(URL) }
    var id: ID
    var result: HistoryItem?
    var pending: PendingItem?
}

struct EditorSettingsSnapshot: Equatable {
    var request: GenerateRequest
    var randomSeed: Bool
    var autoSize: Bool
    var sizeReference: Int
    var sizeReferenceIsAuto: Bool
    var selectedURL: URL?
}

struct EstimateKey: Hashable {
    var request: GenerateRequest?
    var modelID: String
    var loadedModel: String?
    var historyCount: Int
    var ready: Bool
}

struct HistoryMetadata {
    var version: HistoryMetadataVersion
    var hasSettings: Bool
    var model: String?
    var steps: Int?
    var upscale: Double?
    var seconds: Double
    var comparisonSource: URL?
    var description: HistoryDescription
}

@MainActor
final class AppModel: ObservableObject {
    enum Phase: Equatable { case settingUp, starting, ready, failed(String) }

    // Generation settings
    @Published var prompt = "" { didSet { editorSettingsChanged() } }
    @Published var references: [URL] = [] {
        didSet {
            defer { editorSettingsChanged() }
            guard !restoringSettings else { return }
            if references.first != oldValue.first { maskPath = nil }
            // Composing needs every image rather than anchoring the edit to image 1.
            if references.count >= 2, oldValue.count < 2, imageMode == "init",
               currentModel?.image_modes.contains("reference") == true {
                imageMode = "reference"
                referenceNotice = "Switched to Edit / combine so every image is used."
            }
            if references.count < 2 { referenceNotice = nil }
            let selectedURL = oldValue.indices.contains(sizeReference) ? oldValue[sizeReference] : nil
            if !sizeReferenceIsAuto, let selectedURL, let index = references.firstIndex(of: selectedURL) {
                sizeReference = index
            } else {
                sizeReferenceIsAuto = true
                sizeReference = max(0, references.count - 1)
            }
            if autoSize, references != oldValue { matchFirstReferenceSize() }
        }
    }
    /// Which image the output's shape follows; automatic selection follows the last reference.
    @Published var sizeReference = 0 {
        didSet { if !restoringSettings, autoSize, sizeReference != oldValue { matchFirstReferenceSize() }; editorSettingsChanged() }
    }
    @Published var sizeReferenceIsAuto = true { didSet { editorSettingsChanged() } }
    @Published var referenceNotice: String?
    @Published var imageMode = "reference" { didSet { editorSettingsChanged() } }  // reference: edit/combine images | init: start from image 1
    @Published var autoSize = true {
        didSet { if !restoringSettings, autoSize, autoSize != oldValue { matchFirstReferenceSize() }; editorSettingsChanged() }
    }
    @Published var modelID = UserDefaults.standard.string(forKey: "modelID") ?? "flux2-klein-4b" {
        didSet {
            guard modelID != oldValue else { return }
            defer { editorSettingsChanged() }
            UserDefaults.standard.set(modelID, forKey: "modelID")
            guard !restoringSettings else { return }
            applyModelDefaults()
            normalizeSize()
        }
    }
    @Published var quantize: Int = 8 { didSet { editorSettingsChanged() } }  // 0 = full precision
    @Published var negativePrompt = "" { didSet { editorSettingsChanged() } }
    @Published var strength = 0.6 { didSet { editorSettingsChanged() } }  // how much to change the input image
    @Published var width = 1024 { didSet { editorSettingsChanged() } }
    @Published var height = 1024 { didSet { editorSettingsChanged() } }
    @Published var steps = 4 { didSet { editorSettingsChanged() } }
    @Published var guidance = 1.0 { didSet { editorSettingsChanged() } }
    @Published var randomSeed = true { didSet { editorSettingsChanged() } }
    @Published var seed = 42 { didSet { editorSettingsChanged() } }
    @Published var count = 1 { didSet { editorSettingsChanged() } }
    @Published var loraPath: URL? { didSet { editorSettingsChanged() } }
    @Published var loraScale = 1.0 { didSet { editorSettingsChanged() } }

    @Published var fast = false { didSet { editorSettingsChanged() } }
    @Published var enhancePrompt = false { didSet { editorSettingsChanged() } }
    @Published var autoMask = "" { didSet { editorSettingsChanged() } }
    @Published var maskPath: URL? { didSet { editorSettingsChanged() } }
    @Published var loras: [LoraSetting] = [] { didSet { editorSettingsChanged() } }
    @Published var showMaskEditor = false
    @Published var showSweep = false
    @Published var showFullView = false
    @Published var upscaleFileItem: HistoryItem?
    @Published var upscaleScale = 2.0
    @Published var upscaleSoftness = 0.0
    @Published var engineUnreachable = false
    @Published var engineUnresponsive = false
    @Published var previewNote: String?
    @Published var preflightNote: String?
    @Published var historySearch = "" { didSet { if historySearch != oldValue { updateFilteredHistory() } } }
    @Published var favoritesOnly = false { didSet { if favoritesOnly != oldValue { updateFilteredHistory() } } }
    @Published var favorites: Set<String> = [] { didSet { if favorites != oldValue { updateFilteredHistory() } } }
    var historyDescriptions: [String: HistoryDescription] = [:] { didSet { updateFilteredHistory() } }
    @Published var historyMetadata: [URL: HistoryMetadata] = [:]
    var metadataLoads: [URL: UUID] = [:]
    @Published var filteredHistory: [HistoryItem] = []
    var modelSearchNames: [String: String] = [:]
    @Published var clearHistoryCount = 0
    @Published var historyIndex: [HistoryItem] = [] { didSet { if historyIndex != oldValue { updateFilteredHistory() } } }
    @Published var queuePaused = false { didSet { persistQueue(); if !queuePaused { runNextJob() }; updateActivity() } }
    /// True after a relaunch restored queued jobs, until the user resumes them.
    @Published var holdRestored = false

    /// Resume: runs the queue, including jobs restored from last time.
    func resumeQueue() {
        holdRestored = false
        if queuePaused { queuePaused = false } else { runNextJob() }
    }
    @Published var failedJobs: [FailedJob] = [] { didSet { persistQueue() } }
    @Published var modelBusy: Set<String> = []
    @Published var modelErrors: [String: String] = [:]
    @Published var embedMetadata = UserDefaults.standard.bool(forKey: "embedMetadata") { didSet { updateEngineSettings() } }
    @Published var lowMemory = UserDefaults.standard.string(forKey: "lowMemory") ?? "auto" { didSet { updateEngineSettings() } }
    @Published var offline = UserDefaults.standard.bool(forKey: "offline") { didSet { updateEngineSettings() } }
    var restoringQueue = false
    var restoringSettings = false

    // Runtime state
    @Published var phase: Phase = .starting
    @Published var setupLog = ""
    @Published var isGenerating = false
    @Published var status: ServerStatus?
    @Published var errorMessage: String?
    @Published var history: [HistoryItem] = [] { didSet { if history != oldValue { updateFilteredHistory() } } }
    @Published var confirmClearHistory = false
    @Published var tokenDraft = ""
    @Published var hasSavedToken = false
    @Published var tokenStatus: String?
    @Published var tokenOK = false
    @Published var isCheckingToken = false
    @Published var gatedAccess: [String: Bool?] = [:]  // model id -> license accepted (nil = unknown)
    @Published var selected: HistoryItem? {
        didSet { if selected != nil { selectedPending = nil }; if let selected { loadMetadata(for: selected.url) }; editorSettingsChanged() }
    }
    @Published var selectedPending: UUID?
    /// While a job runs and the user hasn't clicked another tile, the view follows the image being made.
    var followJob = false
    var pressureSource: DispatchSourceMemoryPressure?
    @Published var pendingItems: [PendingItem] = []
    var previewPending: UUID?
    @Published var estimate: Estimate?
    @Published var queue: [QueuedJob] = [] { didSet { persistQueue(); updateActivity() } }
    @Published var currentJob: QueuedJob?
    @Published var previewImage: NSImage?  // live preview of the image being generated
    @Published var compareItems: [HistoryItem] = [] { didSet { for item in compareItems { loadMetadata(for: item.url) } } }  // shown side by side when non-empty
    @Published var compareSelection: Set<URL> = []  // history items picked with ⌘-click
    @Published var showCompareSheet = false
    @Published var livePreview = UserDefaults.standard.object(forKey: "livePreview") as? Bool ?? true {
        didSet {
            UserDefaults.standard.set(livePreview, forKey: "livePreview")
            guard !applyingPreviewStatus else { return }
            if livePreview { previewNote = nil }
            previewSettingsUpdating += 1
            let requested = livePreview
            Task {
                defer { previewSettingsUpdating -= 1 }
                do { try await backend.setLivePreview(requested) }
                catch { errorMessage = "Could not update live previews: " + error.localizedDescription }
            }
        }
    }
    var activeCompareGroup: UUID?
    var lastPreviewID: Int?
    var applyingPreviewStatus = false
    var previewSettingsUpdating = 0
    /// Unload the model after this many idle minutes (0 = never) so other apps get the memory back.
    @Published var idleUnloadMinutes = UserDefaults.standard.object(forKey: "idleUnloadMinutes") as? Int ?? 10 {
        didSet {
            UserDefaults.standard.set(idleUnloadMinutes, forKey: "idleUnloadMinutes")
            Task { try? await backend.setIdleUnload(minutes: idleUnloadMinutes) }
        }
    }

    // Models
    @Published var models: [ModelInfo] = [] {
        didSet {
            let names = Dictionary(models.map { ($0.id, $0.name.lowercased()) }, uniquingKeysWith: { _, new in new })
            if names != modelSearchNames { modelSearchNames = names; updateFilteredHistory() }
        }
    }
    @Published var system: SystemInfo?
    @Published var architectures: [Architecture] = []
    @Published var showModelManager = false
    @Published var modelActionInProgress: String?  // model id being loaded/deleted
    @Published var modelError: String?

    let backend = Backend()
    let monitor = SystemMonitor()
    let zoom = ZoomState()
    var jobCancelRequested = false
    var jobTask: Task<Void, Never>?
    var pollTask: Task<Void, Never>?
    var modelPollTask: Task<Void, Never>?
    var isStartingBackend = false
    var recentRestarts: [Date] = []
    var deferredBackendExit: Int32?
    var queueWriteTask: Task<String?, Never>?
    var isClearingHistory = false
    var isShuttingDown = false
    var historyRevision = 0
    var activity: NSObjectProtocol?
    var downloadRequests: Set<String> = []

    var outputDir: URL {
        FileManager.default.urls(for: .picturesDirectory, in: .userDomainMask)[0].appendingPathComponent("Image Studio")
    }

    var booted = false
    var booting = false
    var lastEditorSettings: EditorSettingsSnapshot?
    var settingsSaveTask: Task<Void, Never>?
    var referenceImportTask: Task<Void, Never>?
    var referenceImportRevision = UUID()
    var lastStatusSuccess = Date()
    var lastSystemFetch = Date.distantPast

    func boot() async {
        guard !booted, !booting else { return }
        booting = true
        defer { booting = false }
        do {
            if !Backend.isRuntimeInstalled {
                phase = .settingUp
                try await Backend.installRuntime { text in
                    Task { @MainActor in self.setupLog += text }
                }
            }
            phase = .starting
            // Off the main thread: macOS may show a Keychain permission dialog after an app update,
            // and the window must stay responsive while it does. IMAGE_STUDIO_SKIP_KEYCHAIN is for testing.
            let saved = ProcessInfo.processInfo.environment["IMAGE_STUDIO_SKIP_KEYCHAIN"] != nil
                ? nil : await Task.detached { Keychain.read() }.value
            hasSavedToken = saved != nil
            tokenDraft = saved ?? ""
            try await startBackend(token: saved)
            await refreshModels()
            if currentModel?.status.downloaded != true,
               let fallback = models.first(where: { $0.status.downloaded && !$0.isTool }) {
                modelID = fallback.id
            } else {
                applyModelDefaults()
            }
            restoreGenerationSettings()
            loadHistory()
            restoreLibraryState()
            await restoreQueue()
            try requireLiveBackend()
            phase = .ready
            applyTestOverrides()
            booted = true
            editorSettingsChanged()
            runNextJob()
            startModelPolling()
            startPolling()
            architectures = (try? await backend.architectures()) ?? []
            if let saved { await applyToken(saved) }
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    /// UI-test hooks: preselect a model, reference images and image mode without clicking.
    func applyTestOverrides() {
        let env = ProcessInfo.processInfo.environment
        if let id = env["IMAGE_STUDIO_TEST_MODEL"] { modelID = id }
        if let paths = env["IMAGE_STUDIO_TEST_REFERENCES"] {
            references = paths.split(separator: ":").map { URL(fileURLWithPath: String($0)) }
        }
        if let mode = env["IMAGE_STUDIO_TEST_MODE"] { imageMode = mode }
        if let g = env["IMAGE_STUDIO_TEST_GUIDANCE"].flatMap(Double.init) { guidance = g }
        if let path = env["IMAGE_STUDIO_TEST_SELECT"] {
            let item = HistoryItem(url: URL(fileURLWithPath: path), seed: 0, seconds: 0)
            history.insert(item, at: 0)
            selected = item
            if let z = env["IMAGE_STUDIO_TEST_ZOOM"].flatMap(Double.init) {
                Task {
                    try? await Task.sleep(nanoseconds: 1_500_000_000)
                    zoom.set(CGFloat(z), centeredAt: nil)
                }
            }
        }
        if let n = env["IMAGE_STUDIO_TEST_COUNT"].flatMap(Int.init) { count = n }
        if let text = env["IMAGE_STUDIO_TEST_GENERATE"] {  // types this prompt and presses Generate
            prompt = text
            Task {
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                generate()
                // Then queue more jobs behind it, to check the queue list shows the running one.
                for _ in 0..<(env["IMAGE_STUDIO_TEST_QUEUE"].flatMap(Int.init) ?? 0) { generate() }
                // Then switch model, to check the prompt/images/size survive a switch.
                if let other = env["IMAGE_STUDIO_TEST_SWITCH_TO"] {
                    try? await Task.sleep(nanoseconds: 2_000_000_000)
                    modelID = other
                }
            }
        }
        if let paths = env["IMAGE_STUDIO_TEST_COMPARE"] {
            compareItems = paths.split(separator: ":").map { HistoryItem(url: URL(fileURLWithPath: String($0)), seed: 0, seconds: 0) }
        }
    }

    func shutdown() {
        guard !isShuttingDown else { return }
        settingsSaveTask?.cancel()
        jobTask?.cancel()
        pollTask?.cancel()
        modelPollTask?.cancel()
        saveGenerationSettings(model: modelID)
        persistQueue()
        isShuttingDown = true
        monitor.stop()
        if let activity { ProcessInfo.processInfo.endActivity(activity); self.activity = nil }
        Task { await backend.stop() }
    }

    func startBackend(token: String?) async throws {
        backend.onUnexpectedExit = { [weak self] code in
            guard let self else { return }
            if self.isStartingBackend {
                self.deferredBackendExit = code
            } else {
                Task { await self.restartBackend(exitCode: code) }
            }
        }
        isStartingBackend = true
        defer { isStartingBackend = false; deferredBackendExit = nil }
        while true {
            deferredBackendExit = nil
            try await backend.start(token: token)
            // start() can retry a port conflict internally; discard exits from those old attempts.
            if backend.isAlive { deferredBackendExit = nil }
            try? await backend.setIdleUnload(minutes: idleUnloadMinutes)
            try? await backend.setLivePreview(livePreview)
            try? await backend.setSettings(engineSettings)
            if backend.isAlive, deferredBackendExit == nil { break }
            // A successful /health response is not enough: the process can exit during settings.
            try recordBackendExit(deferredBackendExit ?? backend.exitCode ?? -1)
            try await Task.sleep(nanoseconds: UInt64(recentRestarts.count) * 1_000_000_000)
        }
        lastStatusSuccess = Date()
        engineUnreachable = false
        engineUnresponsive = false
        monitor.backendPID = backend.pid
        monitor.shouldSample = { [weak self] in
            guard let self else { return false }
            return self.hasRunningWork || self.windowIsVisible
        }
        monitor.start()
        startPressureMonitor()
    }

    func requireLiveBackend() throws {
        guard backend.isAlive else {
            engineUnreachable = true
            throw BackendError(message: "The image engine stopped during startup. Retry to restart it. See " + Backend.logFile.path)
        }
    }

    /// Restarts a crashed backend, at most 3 times in 2 minutes so a broken install can't loop forever.
    func recordBackendExit(_ exitCode: Int32) throws {
        recentRestarts = recentRestarts.filter { $0.timeIntervalSinceNow > -120 } + [Date()]
        let log = "~/Library/Application Support/Image Studio/server.previous.log"
        guard recentRestarts.count <= 3 else {
            throw BackendError(message: "The image engine keeps stopping (exit \(exitCode)). Details: \(log)")
        }
        errorMessage = "The image engine stopped unexpectedly (exit \(exitCode)) and was restarted. Details: \(log)"
    }

    func restartBackend(exitCode: Int32) async {
        guard !isStartingBackend else { deferredBackendExit = exitCode; return }
        phase = .starting
        do {
            try recordBackendExit(exitCode)
            try await Task.sleep(nanoseconds: UInt64(recentRestarts.count) * 1_000_000_000)
            // Reuse the token already in memory when recovering from an exit.
            try await startBackend(token: hasSavedToken ? tokenDraft : nil)
            await refreshModels()
            try requireLiveBackend()
            phase = .ready
            runNextJob()
        } catch {
            phase = .failed("Could not restart the image engine: \(error.localizedDescription)")
        }
    }

    /// Generates with the current settings, or queues it behind the running job.
    func generate() {
        guard phase == .ready, !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        guard let model = currentModel, model.status.downloaded, !model.isTool else {
            showModelManager = true
            return
        }
        let missing = references.filter { !FileManager.default.fileExists(atPath: $0.path) }
        if !missing.isEmpty {
            references.removeAll { missing.contains($0) }
            errorMessage = "Removed \(missing.count) input image\(missing.count == 1 ? "" : "s") that no longer exist\(missing.count == 1 ? "s" : ""). Check the images and press Generate again."
            return
        }
        normalizeSize()
        enqueue(QueuedJob(kind: .generate(makeRequest(for: model)), title: jobTitle(model.name)))
    }

    /// Runs the same prompt, images, size and seed on several models, each with its own recommended
    /// steps/guidance, and shows the results side by side.
    func compare(modelIDs: [String]) {
        let chosen = models.filter { modelIDs.contains($0.id) && $0.status.downloaded && !$0.isTool }
        guard chosen.count >= 2 else { return }
        let group = UUID()
        let sharedSeed = randomSeed ? Int.random(in: 0..<Int(Int32.max)) : seed
        activeCompareGroup = group
        compareItems = []
        for info in chosen {
            var request = makeRequest(for: info)
            request.steps = info.default_steps
            request.guidance = max(info.min_guidance, info.default_guidance)
            request.strength = references.isEmpty ? nil : info.default_strength
            request.quantize = info.quant_options.contains(quantize) ? quantize : info.default_quant
            request.negative_prompt = info.supports_negative && !negativePrompt.isEmpty ? negativePrompt : nil
            request.seed = sharedSeed
            request.count = 1
            request.loras = []
            request.mask_path = nil
            request.auto_mask = nil
            request.lora_path = nil  // a LoRA only fits the model it was trained for
            enqueue(QueuedJob(kind: .generate(request), title: "Compare · \(info.name)", compareGroup: group))
        }
    }

    func upscale(_ item: HistoryItem) {
        guard let upscaler = models.first(where: { $0.family == "seedvr2" }) else { return }
        guard upscaler.status.downloaded else {
            errorMessage = "Upscaling needs \(upscaler.name) (\(upscaler.status.bytes_total.map(formatBytes) ?? "a one-time download")). Download it in the Model Manager."
            showModelManager = true
            return
        }
        guard let size = Self.displaySize(of: item.url), size.width * size.height * upscaleScale * upscaleScale <= 4_500_000 else {
            errorMessage = "This scale exceeds the upscaler’s 4.5 MP limit. Choose a smaller scale."
            return
        }
        enqueue(QueuedJob(kind: .upscale(item.url, upscaleScale, upscaleSoftness), title: "Upscale \(upscaleScale.formatted())× · \(item.url.lastPathComponent)"))
    }

    func removeFromQueue(_ job: QueuedJob) { queue.removeAll { $0.id == job.id } }
    func clearQueue() { queue.removeAll() }

    func jobTitle(_ modelName: String) -> String {
        let text = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        return "\(modelName) · \(text.count > 60 ? text.prefix(60) + "…" : text)"
    }

    func enqueue(_ job: QueuedJob) {
        queue.append(job)
        runNextJob()
    }

    func runNextJob() {
        guard !isShuttingDown, !restoringQueue, !isClearingHistory, !isGenerating, phase == .ready, !queue.isEmpty else { return }
        guard !queuePaused else { return }
        // Jobs restored from a previous session wait until the user resumes them; new ones run now.
        guard let next = holdRestored ? queue.firstIndex(where: { !$0.restored }) : queue.startIndex else { return }
        if next != queue.startIndex { queue.insert(queue.remove(at: next), at: queue.startIndex) }
        restoringQueue = true
        let job = queue[0]
        currentJob = job
        queue.removeFirst()
        // A normal generation must show its result: close a comparison left open from earlier, which
        // otherwise keeps covering the result area while new images only land in History.
        if job.compareGroup == nil || job.compareGroup != activeCompareGroup { closeCompare() }
        restoringQueue = false
        persistQueue()
        isGenerating = true
        jobCancelRequested = false
        errorMessage = nil
        if previewImage != nil { previewImage = nil }
        lastPreviewID = nil
        status = nil
        startPendingItems(for: job)
        updateActivity()
        if pollTask != nil { startPolling() }
        jobTask = Task {
            do {
                let items: [HistoryItem]
                switch job.kind {
                case .generate(let request):
                    guard let checked = try await prepareJob(request) else { throw CancellationError() }
                    let response = try await retryBusy { try await self.backend.generate(checked) }
                    if let error = response.error {
                        errorMessage = error
                        failedJobs.append(FailedJob(job: job, error: error))
                    }
                    items = response.images.map {
                        HistoryItem(url: URL(fileURLWithPath: $0.path), seed: $0.seed, seconds: $0.seconds)
                    }
                    if !randomSeed, job.compareGroup == nil, let last = items.last { seed = last.seed }
                case .upscale(let url, let scale, let softness):
                    let result = try await retryBusy { try await self.backend.upscale(path: url.path, scale: scale, softness: softness) }
                    if result.cancelled == true { throw CancellationError() }
                    guard let path = result.path, let seed = result.seed else {
                        throw BackendError(message: "The upscaler returned no output image.")
                    }
                    items = [HistoryItem(url: URL(fileURLWithPath: path), seed: seed, seconds: result.seconds)]
                }
                acceptResults(items, job: job)
            } catch {
                if !Task.isCancelled && !(error is CancellationError) {
                    errorMessage = error.localizedDescription
                    failedJobs.append(FailedJob(job: job, error: error.localizedDescription))
                }
            }
            finishPendingItems()
            isGenerating = false
            currentJob = nil
            jobTask = nil
            persistQueue()
            if previewImage != nil { previewImage = nil }
            await refreshModels()
            runNextJob()
            updateActivity()
        }
    }

    func cancel() {
        jobCancelRequested = true
        Task { await backend.cancel() }
    }

    func makeRequest(for model: ModelInfo) -> GenerateRequest {
        GenerateRequest(
            prompt: prompt,
            negative_prompt: model.supports_negative && !negativePrompt.isEmpty ? negativePrompt : nil,
            image_paths: model.edit == "none" ? [] : Array(references.prefix(model.max_refs)).map(\.path),
            model: model.id,
            quantize: quantize,  // 0 = full precision; the server maps it to "no quantization"
            width: normalizedDimensions(for: model).0,
            height: normalizedDimensions(for: model).1,
            steps: min(100, max(1, steps)),
            guidance: min(20, max(0, guidance)),
            strength: model.edit != "none" && !references.isEmpty ? min(0.95, max(0.1, strength)) : nil,
            image_mode: model.image_modes.contains(imageMode) ? imageMode : (model.image_modes.first ?? "reference"),
            seed: randomSeed ? nil : max(0, seed),
            count: min(16, max(1, count)),
            lora_path: loraPath?.path,
            lora_scale: loraScale,
            fast: model.supports_step_cache == true && fast,
            mask_path: model.supports_mask == true && !references.isEmpty ? maskPath?.path : nil,
            auto_mask: model.supports_mask == true && !references.isEmpty && !autoMask.isEmpty ? autoMask : nil,
            enhance_prompt: model.supports_enhance == true && model.edit != "none" && model.max_refs > 0 && !references.isEmpty && enhancePrompt,
            loras: loras
        )
    }

    // MARK: Time estimate

    /// Everything the estimate depends on; the generate bar re-estimates when it changes.
    var estimateKey: EstimateKey {
        EstimateKey(request: currentModel.map { makeRequest(for: $0) }, modelID: modelID,
                    loadedModel: status?.loaded_model, historyCount: history.count, ready: phase == .ready)
    }

    func refreshEstimate() async {
        guard phase == .ready, let model = currentModel, model.status.downloaded else { estimate = nil; return }
        estimate = try? await backend.estimate(makeRequest(for: model))
    }

    /// Puts a result in as the image to edit, ready for the next instruction.
    func editImage(_ item: HistoryItem) {
        invalidateReferenceImports()
        autoSize = true  // keep the edited image's size and shape
        references = [item.url]
        matchFirstReferenceSize()
        let modes = currentModel?.image_modes ?? ["reference"]
        imageMode = modes.contains("reference") ? "reference" : (modes.first ?? "reference")
        prompt = ""
    }

    // MARK: Hugging Face token

    func saveToken() {
        let token = tokenDraft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !token.isEmpty else { return clearToken() }
        do { try Keychain.save(token) }
        catch { tokenOK = false; tokenStatus = "Could not save the token: " + error.localizedDescription; return }
        hasSavedToken = true
        Task { await applyToken(token) }
    }

    func clearToken() {
        do { try Keychain.delete() }
        catch { tokenStatus = "Could not remove the token: " + error.localizedDescription; return }
        tokenDraft = ""
        hasSavedToken = false
        tokenOK = false
        tokenStatus = nil
        gatedAccess = [:]
        Task { _ = try? await backend.setToken(nil) }
    }

    func applyToken(_ token: String) async {
        isCheckingToken = true
        defer { isCheckingToken = false }
        do {
            let status = try await backend.setToken(token)
            tokenOK = status.valid == true
            gatedAccess = status.gated_access
            guard status.valid == true, let user = status.user else {
                tokenStatus = status.valid == false
                    ? "Hugging Face rejected this token. Check that it was copied completely."
                    : "Saved, but couldn't reach Hugging Face to verify it."
                return
            }
            let name = { (id: String) in self.models.first { $0.id == id }?.name ?? id }
            let unlocked = status.gated_access.filter { $0.value == true }.keys.sorted().map(name)
            let locked = status.gated_access.filter { $0.value == false }.keys.sorted().map(name)
            var text = "Signed in as \(user)."
            if !unlocked.isEmpty { text += " Unlocked: \(unlocked.joined(separator: ", "))." }
            if !locked.isEmpty { text += " Accept the license on Hugging Face for: \(locked.joined(separator: ", "))." }
            tokenStatus = text
        } catch {
            tokenOK = false
            tokenStatus = error.localizedDescription
        }
    }

    // MARK: Models

    var currentModel: ModelInfo? { models.first { $0.id == modelID } }

    func applyModelDefaults() {
        guard let model = currentModel else { return }
        var notes: [String] = []
        if loraPath != nil || !loras.isEmpty {
            notes.append("LoRAs were cleared because they belong to the previous model.")
        }
        loraPath = nil
        loras = []
        if model.supports_mask != true, maskPath != nil || !autoMask.isEmpty {
            maskPath = nil
            autoMask = ""
            notes.append("The mask was cleared because this model does not support masks.")
        }
        if !notes.isEmpty { errorMessage = notes.joined(separator: " ") }
        steps = model.default_steps
        guidance = max(model.min_guidance, model.default_guidance)
        strength = model.default_strength
        quantize = model.default_quant
        fast = false
        enhancePrompt = false
        if !model.image_modes.contains(imageMode) { imageMode = model.image_modes.first ?? "reference" }
    }

    func refreshModels() async {
        do { let updated = try await backend.models(); if models != updated { models = updated }; updateActivity() } catch { modelError = "Could not read the model list: " + error.localizedDescription }
    }

    /// Refreshes download progress: every second while downloads run or the manager is open.
    func startModelPolling() {
        modelPollTask?.cancel()
        modelPollTask = Task { [weak self] in
            var tick = 0
            while !Task.isCancelled, let self {
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                tick += 1
                let active = showModelManager || models.contains { $0.status.downloading }
                if active || tick % 10 == 0 { await refreshModels() }
            }
        }
    }

    func modelAction(_ id: String, _ work: @escaping () async throws -> Void) {
        guard !modelBusy.contains(id) else { return }
        modelBusy.insert(id)
        modelActionInProgress = id
        modelErrors[id] = nil
        modelError = nil
        Task {
            defer { modelBusy.remove(id); modelActionInProgress = nil }
            do { try await work() } catch { modelErrors[id] = error.localizedDescription; modelError = error.localizedDescription }
            await refreshModels()
        }
    }

    /// Inputs remain protected while jobs are queued, running, or retained for retry after failure.
    var pathsInUse: [String] {
        var paths = references.map(\.path)
        if let maskPath { paths.append(maskPath.path) }
        if let upscaleFileItem { paths.append(upscaleFileItem.url.path) }
        for job in queue + (currentJob.map { [$0] } ?? []) + failedJobs.map(\.job) {
            switch job.kind {
            case .generate(let request): paths += request.image_paths; if let mask = request.mask_path { paths.append(mask) }
            case .upscale(let url, _, _): paths.append(url.path)
            }
        }
        return paths
    }

    func scanStorage() async throws -> [StorageItem] {
        try await backend.storageScan(keep: pathsInUse)
    }

    func cleanStorage(ids: [String]) async throws -> StorageCleanResult {
        let result = try await backend.storageClean(ids: ids, keep: pathsInUse)
        await refreshModels()
        return result
    }

    func download(_ model: ModelInfo) {
        guard !modelBusy.contains(model.id) else { return }
        downloadRequests.insert(model.id)
        updateActivity()
        modelAction(model.id) {
            defer { self.downloadRequests.remove(model.id); self.updateActivity() }
            try await self.backend.downloadModel(model.id)
            await self.refreshModels()
        }
    }
    func cancelDownload(_ model: ModelInfo) { modelAction(model.id) { try await self.backend.cancelDownload(model.id) } }
    func deleteModel(_ model: ModelInfo, scope: String = "all") { modelAction(model.id) { try await self.backend.deleteModel(model.id, scope: scope) } }

    func optimizeModel(_ model: ModelInfo, quant: Int) { modelAction(model.id) { try await self.backend.optimizeModel(model.id, quantize: quant) } }
    func unloadModel() {
        let id = status?.loaded_model ?? models.first(where: { $0.loaded })?.id ?? ""
        modelAction(id) { try await self.backend.unloadModel() }
    }

    func loadModel(_ model: ModelInfo) {
        let quant = model.id == modelID && model.quant_options.contains(quantize) ? quantize : model.default_quant
        modelAction(model.id) {
            try await self.backend.loadModel(model.id, quantize: quant, loraPath: nil, loraScale: 1)
            self.modelID = model.id
        }
    }

    func addCustomModel(name: String, path: String, architecture: String) async -> String? {
        do {
            try await backend.addCustomModel(name: name, path: path, architecture: architecture)
            await refreshModels()
            return nil
        } catch {
            return error.localizedDescription
        }
    }

    /// Keeps generation progress and the backend's memory figures current for the status bar.
    func startPolling() {
        pollTask?.cancel()
        pollTask = Task {
            while !Task.isCancelled {
                let polledJob = currentJob?.id
                if let s = try? await backend.status() {
                    guard !Task.isCancelled else { break }
                    guard polledJob == currentJob?.id else { continue }
                    if status != s { status = s }
                    lastStatusSuccess = Date()
                    if engineUnreachable { engineUnreachable = false }
                    if engineUnresponsive { engineUnresponsive = false }
                    if previewSettingsUpdating == 0, let actual = s.live_preview, actual != livePreview {
                        if !actual { previewNote = "The image engine turned previews off after a preview failed." }
                        applyingPreviewStatus = true
                        livePreview = actual
                        applyingPreviewStatus = false
                    }
                    if let job = currentJob, s.busy == "generate" || s.busy == "upscale" {
                        if let done = s.images_done {
                            acceptResults(done.map { HistoryItem(url: URL(fileURLWithPath: $0.path), seed: $0.seed, seconds: $0.seconds) }, job: job)
                        }
                        let pending = pendingItems.first { $0.result == nil && $0.index == max(0, s.image_index - 1) }
                        if previewPending != pending?.id {
                            if previewImage != nil { previewImage = nil }
                            previewPending = pending?.id
                            // Batch: move on to the next image's live preview unless the user looked elsewhere.
                            if followJob, let pending { selected = nil; selectedPending = pending.id }
                            debugGallery("previewPending -> index \(pending?.index ?? -1) (image_index \(s.image_index))")
                        }
                        if let pending, let id = s.preview_id, id != lastPreviewID,
                           let data = try? await backend.preview() {
                            let image = await Task.detached(priority: .utility) { NSImage(data: data) }.value
                            guard !Task.isCancelled, currentJob?.id == job.id, previewPending == pending.id,
                                  pendingItems.contains(where: { $0.id == pending.id && $0.result == nil }) else { continue }
                            lastPreviewID = id
                            if previewImage !== image { previewImage = image }
                        }
                    }
                } else {
                    guard !Task.isCancelled else { break }
                    if status != nil { status = nil }
                    if !engineUnreachable { engineUnreachable = true }
                    if !engineUnresponsive, backend.isAlive && Date().timeIntervalSince(lastStatusSuccess) > 45 { engineUnresponsive = true }
                }
                updateActivity()
                let systemInterval: TimeInterval = windowIsVisible ? (showModelManager ? 2 : 5) : 30
                if Date().timeIntervalSince(lastSystemFetch) >= systemInterval {
                    lastSystemFetch = Date()
                    if let info = try? await backend.system(), system != info { system = info }
                }
                try? await Task.sleep(nanoseconds: isGenerating ? 400_000_000 : (hasRunningWork || windowIsVisible ? 1_000_000_000 : 5_000_000_000))
            }
        }
    }

    // MARK: References

    func addReferences(_ urls: [URL]) {
        enqueueReferenceImport { ImageImports.validURLs(urls) }
    }

    /// Images without a file representation are persisted by the same ordered worker as file imports.
    func addReference(image: NSImage) {
        let snapshot = image.copy() as? NSImage ?? image
        let directory = Backend.runtimeDir.appendingPathComponent("dropped", isDirectory: true)
        enqueueReferenceImport { ImageImports.persist(image: snapshot, directory: directory).map { [$0] } ?? [] }
    }

    func addReference(data: Data) {
        let directory = Backend.runtimeDir.appendingPathComponent("dropped", isDirectory: true)
        enqueueReferenceImport { ImageImports.persist(data: data, directory: directory).map { [$0] } ?? [] }
    }

    func enqueueReferenceImport(_ work: @escaping () -> [URL]) {
        enqueueReferenceWork(work) { [weak self] urls in
            guard let self else { return }
            let fresh = urls.filter { !self.references.contains($0) }
            if !fresh.isEmpty { self.references.append(contentsOf: fresh) }
        }
    }

    func invalidateReferenceImports() {
        referenceImportRevision = UUID()
        referenceImportTask?.cancel()
        referenceImportTask = nil
    }

    @discardableResult
    func enqueueReferenceWork<T>(_ work: @escaping () -> T, apply: @escaping (T) -> Void) -> Task<Void, Never> {
        let previous = referenceImportTask
        let revision = referenceImportRevision
        let task = Task { [weak self] in
            await previous?.value
            guard !Task.isCancelled, self?.referenceImportRevision == revision else { return }
            let result = await Task.detached(priority: .userInitiated) { work() }.value
            guard !Task.isCancelled, self?.referenceImportRevision == revision else { return }
            apply(result)
        }
        referenceImportTask = task
        return task
    }

    func pickReferences() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.allowsMultipleSelection = true
        if panel.runModal() == .OK { addReferences(panel.urls) }
    }

    func pickLora() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.init(filenameExtension: "safetensors") ?? .data]
        if panel.runModal() == .OK, let url = panel.url { loras.append(LoraSetting(path: url.path, scale: 1)) }
    }

    /// Output size: the chosen reference's shape at about one megapixel.
    func matchFirstReferenceSize() {
        guard !restoringSettings, !references.isEmpty else { return }
        let url = references[min(sizeReference, references.count - 1)]
        guard let size = Self.displaySize(of: url) else { return }
        let w = size.width, h = size.height
        // Keep roughly one megapixel at the reference's aspect ratio.
        let scale = (1024.0 * 1024.0 / (w * h)).squareRoot()
        let m = Double(currentModel?.size_multiple ?? 16)
        width = Int((w * scale / m).rounded() * m)
        height = Int((h * scale / m).rounded() * m)
        normalizeSize()
    }

    /// Pixel size as displayed, i.e. with EXIF rotation applied (phone photos are often stored sideways).
    nonisolated static func displaySize(of url: URL) -> CGSize? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let w = props[kCGImagePropertyPixelWidth] as? Double, let h = props[kCGImagePropertyPixelHeight] as? Double,
              w > 0, h > 0 else { return nil }
        let orientation = props[kCGImagePropertyOrientation] as? Int ?? 1
        return orientation >= 5 ? CGSize(width: h, height: w) : CGSize(width: w, height: h)
    }

    // MARK: History

    func loadHistory() {
        let directory = outputDir
        let revision = historyRevision
        Task {
            let found = await Task.detached(priority: .utility) { Self.scanHistory(directory) }.value
            guard revision == historyRevision else {
                if !isClearingHistory { loadHistory() }
                return
            }
            acceptHistoryScan(found)
        }
    }

    func acceptHistoryScan(_ found: HistoryScan) {
        let current = Dictionary(history.map { ($0.url, $0) }, uniquingKeysWith: { _, latest in latest })
        let scanned = Set(found.items.map(\.url))
        let new = history.filter { !scanned.contains($0.url) }
        let scannedItems = found.items.map { item in
            historyMetadata[item.url]?.version == found.metadata[item.url]?.version ? (current[item.url] ?? item) : item
        }
        for url in found.metadata.keys { metadataLoads[url] = nil }
        historyDescriptions.merge(found.descriptions) { _, scanned in scanned }
        historyMetadata.merge(found.metadata) { _, scanned in scanned }
        historyIndex = new + scannedItems
        history = new + Array(scannedItems.prefix(100))
        if selected == nil && selectedPending == nil { selected = history.first }
    }

    func loadMoreHistory() {
        let visible = Set(history.map(\.url))
        history.append(contentsOf: historyIndex.filter { !visible.contains($0.url) }.prefix(100))
    }

    func reuseSettings(of item: HistoryItem) {
        historyRevision += 1
        invalidateMetadata(at: item.url)
        loadMetadata(for: item.url)
        guard let json = Self.settingsJSON(at: item.url) else { errorMessage = "No saved settings found in this image."; return }
        restoreSettings(json)
    }

    func restoreSettings(_ json: [String: Any], invalidateImports: Bool = true) {
        if invalidateImports { invalidateReferenceImports() }
        let wasRestoring = restoringSettings
        restoringSettings = true
        defer { restoringSettings = wasRestoring; editorSettingsChanged() }
        autoSize = false
        if let v = json["model"] as? String { modelID = v }
        if let v = json["prompt"] as? String { prompt = v }
        negativePrompt = json["negative_prompt"] as? String ?? ""
        if let v = json["strength"] as? Double { strength = v }
        if let v = json["image_mode"] as? String { imageMode = v }
        if let v = json["width"] as? Int { width = v }
        if let v = json["height"] as? Int { height = v }
        if let v = json["steps"] as? Int { steps = v }
        if let v = json["guidance"] as? Double { guidance = v }
        quantize = json["quantize"] as? Int ?? 0
        count = json["count"] as? Int ?? 1
        randomSeed = json["seed"] as? Int == nil
        if let v = json["seed"] as? Int { seed = v }
        fast = json["fast"] as? Bool ?? false
        enhancePrompt = json["enhance_prompt"] as? Bool ?? false
        autoMask = json["auto_mask"] as? String ?? ""
        loras = (json["loras"] as? [[String: Any]] ?? []).compactMap { value in
            guard let path = value["path"] as? String else { return nil }
            return LoraSetting(path: path, scale: value["scale"] as? Double ?? 1)
        }
        loraPath = nil
        if loras.isEmpty, let path = json["lora_path"] as? String {
            loras = [LoraSetting(path: path, scale: json["lora_scale"] as? Double ?? 1)]
        }
        if let paths = json["image_paths"] as? [String] {
            references = paths.map { URL(fileURLWithPath: $0) }.filter { FileManager.default.fileExists(atPath: $0.path) }
        }
        maskPath = (json["mask_path"] as? String).map { URL(fileURLWithPath: $0) }
        sizeReferenceIsAuto = json["ui_size_reference_auto"] as? Bool ?? true
        let savedIndex = json["ui_size_reference"] as? Int ?? max(0, references.count - 1)
        if !sizeReferenceIsAuto, let path = json["ui_size_reference_path"] as? String,
           let index = references.firstIndex(where: { $0.path == path }) {
            sizeReference = index
        } else if !sizeReferenceIsAuto, references.indices.contains(savedIndex), json["ui_size_reference_path"] == nil {
            sizeReference = savedIndex
        } else {
            sizeReferenceIsAuto = true
            sizeReference = max(0, references.count - 1)
        }
        referenceNotice = nil
    }

    func toggleCompareSelection(_ item: HistoryItem) {
        if compareSelection.contains(item.url) { compareSelection.remove(item.url) } else { compareSelection.insert(item.url) }
    }

    func compareSelected() {
        compareItems = history.filter { compareSelection.contains($0.url) }
        activeCompareGroup = nil
        compareSelection = []
    }

    func closeCompare() {
        compareItems = []
        activeCompareGroup = nil
    }

    /// Model name, steps and time from the cached settings record.
    func caption(for item: HistoryItem) -> String {
        guard let metadata = historyMetadata[item.url], metadata.hasSettings else { return item.url.lastPathComponent }
        var parts: [String] = []
        if let id = metadata.model { parts.append(models.first { $0.id == id }?.name ?? id) }
        if let scale = metadata.upscale { parts.append("upscaled \(scale.formatted())×") }
        else if let steps = metadata.steps { parts.append("\(steps) steps") }
        let seconds = item.seconds > 0 ? item.seconds : metadata.seconds
        if seconds > 0 { parts.append(formatDuration(seconds)) }
        return parts.joined(separator: " · ")
    }

    /// Moves every generated image (and its settings file) to the Trash.
    func clearHistory() {
        guard !isClearingHistory else { return }
        isClearingHistory = true
        historyRevision += 1
        let directory = outputDir
        Task {
            defer { isClearingHistory = false; runNextJob() }
            let folders = await Task.detached(priority: .utility) { Self.scanAppDateFolders(directory) }.value
            let protected = Set(pathsInUse)
            guard !folders.contains(where: { folder in protected.contains { $0.hasPrefix(folder.path + "/") } }) else {
                errorMessage = "History contains inputs used by reference images or retained jobs. Remove those inputs first."
                return
            }
            let knownURLs = Set((history + historyIndex + compareItems).map(\.url))
                .union(compareSelection).union(selected.map { [$0.url] } ?? [])
            let result = await Task.detached(priority: .utility) { () -> (Set<URL>, [String], [URL]) in
                let fm = FileManager.default
                var failures: [String] = []
                var removedFolders: [URL] = []
                for folder in folders {
                    do {
                        try fm.trashItem(at: folder, resultingItemURL: nil)
                        removedFolders.append(folder)
                    }
                    catch { failures.append(folder.lastPathComponent + ": " + error.localizedDescription) }
                }
                return (Set(knownURLs.filter { !fm.fileExists(atPath: $0.path) }), failures, removedFolders)
            }.value
            func removed(_ url: URL) -> Bool {
                result.0.contains(url) || result.2.contains { url.path.hasPrefix($0.path + "/") }
            }
            let removedURLs = Set(historyMetadata.keys.filter { removed($0) }).union(metadataLoads.keys.filter { removed($0) })
            for url in removedURLs { invalidateMetadata(at: url, removed: true) }
            historyDescriptions = historyDescriptions.filter { !removed(URL(fileURLWithPath: $0.key)) }
            ImageThumbnails.purge()
            history.removeAll { removed($0.url) }
            historyIndex.removeAll { removed($0.url) }
            compareItems.removeAll { removed($0.url) }
            compareSelection = compareSelection.filter { !removed($0) }
            if let selected, removed(selected.url) { self.selected = history.first }
            if !result.1.isEmpty { errorMessage = "Could not clear some history folders: " + result.1.joined(separator: "\n") }
        }
    }

    func delete(_ item: HistoryItem) {
        guard !pathsInUse.contains(item.url.path) else { errorMessage = "This image is used by reference images or a queued/running job. Remove that input first."; return }
        do { try FileManager.default.trashItem(at: item.url, resultingItemURL: nil) }
        catch { errorMessage = "Could not trash the image: " + error.localizedDescription; return }
        let sidecar = item.url.deletingPathExtension().appendingPathExtension("json")
        if FileManager.default.fileExists(atPath: sidecar.path) {
            do { try FileManager.default.trashItem(at: sidecar, resultingItemURL: nil) }
            catch { errorMessage = "The image was trashed, but its settings could not be removed: " + error.localizedDescription }
        }
        historyRevision += 1
        invalidateMetadata(at: item.url, removed: true)
        historyDescriptions[item.url.path] = nil
        ImageThumbnails.purge()
        history.removeAll { $0.url == item.url }
        historyIndex.removeAll { $0.url == item.url }
        compareItems.removeAll { $0.url == item.url }
        compareSelection.remove(item.url)
        if selected == item { selected = history.first }
    }
}
