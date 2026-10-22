import Foundation
import Security
import Darwin

struct LoraSetting: Codable, Hashable {
    var path: String
    var scale: Double
}

struct Preflight: Decodable {
    var level: String
    var message: String
    var suggest_quant: Int?
}

struct EngineSettings: Codable {
    var embed_metadata: Bool
    var low_memory: String
    var offline: Bool
}

struct GenerateRequest: Codable, Hashable {
    var prompt: String
    var negative_prompt: String?
    var image_paths: [String]
    var model: String
    var quantize: Int
    var width: Int
    var height: Int
    var steps: Int
    var guidance: Double
    var strength: Double?
    var image_mode: String
    var seed: Int?
    var count: Int
    var lora_path: String?
    var lora_scale: Double
    var fast: Bool? = nil
    var mask_path: String? = nil
    var auto_mask: String? = nil
    var enhance_prompt: Bool? = nil
    var loras: [LoraSetting]? = nil
}

/// Expected duration learned from earlier runs on this Mac.
struct Estimate: Decodable, Equatable {
    var seconds: Double?  // nil until this model has run once
    var load_seconds: Double?  // time to load the model first; nil = unknown, 0 = already loaded
    var basis: String  // measured | similar (other kind of run, same model) | none
    var steps_run: Int
    var passes: Int
}

struct StorageItem: Decodable, Identifiable, Hashable {
    var id: String
    var category: String
    var label: String
    var detail: String
    var bytes: Int
    var recommended: Bool
}

struct StorageCleanResult: Decodable {
    var freed: Int
    var removed: [String]
    var failed: [String]
}

struct GeneratedImageInfo: Codable, Hashable {
    var path: String
    var seed: Int
    var seconds: Double
}

struct GenerateResponse: Decodable {
    var images: [GeneratedImageInfo]
    var cancelled: Bool
    var seconds: Double
    var error: String?
}

struct ServerStatus: Decodable, Equatable {
    var phase: String
    var message: String
    var step: Int
    var total_steps: Int
    var passes: Int?  // model passes per step: 2 when guidance runs a negative pass
    var preview_id: Int?  // changes whenever a new live preview is available (nil = none)
    var preview_step: Int?
    var auto_unloaded: Bool?  // the model was unloaded after the idle timeout
    var idle_unload_minutes: Int?
    var image_index: Int
    var image_count: Int
    var loaded_model: String?
    var mlx_active: Int?
    var images_done: [GeneratedImageInfo]?
    var live_preview: Bool?
    var busy: String?
    var memory_warning: String?
    var output_dir: String
}

struct ModelSource: Decodable, Hashable {
    var repo: String
    var patterns: [String]
}

struct DownloadStatus: Decodable, Hashable {
    var downloaded: Bool
    var downloading: Bool
    var bytes_present: Int
    var bytes_total: Int?
    var error: String?

    var fraction: Double? {
        guard let total = bytes_total, total > 0 else { return nil }
        return min(1, Double(bytes_present) / Double(total))
    }
}

/// One entry of the backend's model catalog (built-in or custom), with its download/load state.
struct ModelInfo: Decodable, Identifiable, Hashable {
    var id: String
    var name: String
    var family: String
    var released: String
    var license: String
    var commercial: Bool
    var gated: Bool
    var description: String
    var sources: [ModelSource]
    var edit: String  // multi | single | none
    var max_refs: Int
    var image_modes: [String]
    var default_strength: Double
    var default_steps: Int
    var default_guidance: Double
    var min_guidance: Double
    var fixed_steps: Bool
    var supports_negative: Bool
    var supports_strength: Bool
    var supports_guidance: Bool?  // false: guidance is built in, the slider does nothing
    var distilled: Bool?  // tuned for guidance 1.0
    var double_pass_when: String?  // "guidance>1" | "guidance!=1" | "never"
    var size_multiple: Int
    var quant_options: [Int]
    var default_quant: Int
    var memory_gb: [String: Double]
    var memory_measured: [Int]?  // precisions whose figure is this Mac's measured peak (else an estimate)
    var is_new: Bool
    var custom: Bool
    var model_path: String?
    var status: DownloadStatus
    var shared_with: [String]
    var disk_bytes: Int
    var loaded: Bool
    var loaded_quant: Int?
    var tool: Bool?  // the upscaler: downloadable, but not an image generator

    var max_megapixels: Double?
    var supports_step_cache: Bool?
    var supports_mask: Bool?
    var supports_enhance: Bool?
    var optimizable: Bool?
    var optimized: [String: Bool]?
    var own_bytes: Int?

    var isTool: Bool { tool == true }

    var downloadBytes: Int? { status.bytes_total }
    var usesGuidance: Bool { supports_guidance != false }

    /// Guidance adds a second (negative) model pass to every step, doubling the time per step.
    func passesPerStep(guidance: Double) -> Int {
        switch double_pass_when {
        case "guidance!=1": return guidance != 1 ? 2 : 1
        case "guidance>1": return guidance > 1 ? 2 : 1
        default: return 1
        }
    }
    func memory(for quant: Int) -> Double? { memory_gb[String(quant)] }
}

struct SystemInfo: Decodable, Equatable {
    var memory_total: Int
    var memory_in_use: Int
    var mlx_cache: Int
    var mlx_peak: Int
    var pid: Int32
    var disk_free: Int
    var models_dir: String
}

struct Architecture: Decodable, Identifiable, Hashable {
    var id: String
    var name: String
}

struct TokenStatus: Decodable {
    var valid: Bool?  // nil = could not verify (offline)
    var user: String?
    var gated_access: [String: Bool?]
}

struct BackendError: LocalizedError {
    var message: String
    var statusCode: Int? = nil
    var errorDescription: String? { message }
    var isBusyConflict: Bool {
        guard statusCode == 409 else { return false }
        return ["Generating images", "Upscaling an image", "Loading a model", "Optimizing a model",
                "Cleaning up storage", "Unloading a model", "An operation is running"]
            .contains { message == $0 + ". Please wait." }
    }
}

/// Owns the Python runtime: first-run setup, launching the server, and the HTTP calls to it.
@MainActor
final class Backend {
    static let runtimeDir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("Image Studio")
    static var python: URL { runtimeDir.appendingPathComponent("venv/bin/python") }
    static var logFile: URL { runtimeDir.appendingPathComponent("server.log") }

    private var process: Process?
    private(set) var port = 0
    private var stopping = false
    private var apiToken = ""
    var isAlive: Bool { process?.isRunning == true }
    var exitCode: Int32? { guard let process, !process.isRunning else { return nil }; return process.terminationStatus }
    /// Called on the main queue when the backend exits without being asked to.
    var onUnexpectedExit: ((Int32) -> Void)?
    var pid: Int32? { process?.processIdentifier }
    private var baseURL: URL { URL(string: "http://127.0.0.1:\(port)")! }

    private lazy var session = Self.makeSession()

    private static func makeSession() -> URLSession {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 60 * 60 * 2
        config.timeoutIntervalForResource = 60 * 60 * 2
        return URLSession(configuration: config)
    }

    /// Bump together with RUNTIME_VERSION in setup.sh when the Python dependencies change.
    static let runtimeVersion = "3"

    static var isRuntimeInstalled: Bool {
        let marker = runtimeDir.appendingPathComponent("runtime-version")
        let version = (try? String(contentsOf: marker, encoding: .utf8))?.trimmingCharacters(in: .whitespacesAndNewlines)
        return FileManager.default.isExecutableFile(atPath: python.path) && version == runtimeVersion
    }

    /// Runs the bundled setup script, streaming its output line by line.
    static func installRuntime(log: @escaping (String) -> Void) async throws {
        guard let script = Bundle.main.url(forResource: "setup", withExtension: "sh") else {
            throw BackendError(message: "setup.sh missing from app bundle")
        }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = [script.path, runtimeDir.path]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        pipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if let text = String(data: data, encoding: .utf8), !text.isEmpty { log(text) }
        }
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            process.terminationHandler = { _ in cont.resume() }
            do { try process.run() } catch { process.terminationHandler = nil; cont.resume(throwing: error) }
        }
        pipe.fileHandleForReading.readabilityHandler = nil
        if process.terminationStatus != 0 {
            throw BackendError(message: "Runtime setup failed (exit \(process.terminationStatus))")
        }
    }

    func start(token: String?) async throws {
        guard let script = Bundle.main.url(forResource: "server", withExtension: "py") else {
            throw BackendError(message: "server.py missing from app bundle")
        }
        await stop()
        let previous = Self.runtimeDir.appendingPathComponent("server.previous.log")
        try? FileManager.default.removeItem(at: previous)
        try? FileManager.default.moveItem(at: Self.logFile, to: previous)
        FileManager.default.createFile(atPath: Self.logFile.path, contents: nil)
        let log = try FileHandle(forWritingTo: Self.logFile)
        defer { try? log.close() }
        for attempt in 0..<3 {
            port = Self.freePort()
            var bytes = [UInt8](repeating: 0, count: 32)
            guard SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes) == errSecSuccess else {
                throw BackendError(message: "Could not create the engine API token")
            }
            apiToken = bytes.map { String(format: "%02x", $0) }.joined()
            let child = Process()
            child.qualityOfService = .userInitiated
            child.executableURL = Self.python
            child.arguments = [script.path, "--port", String(port)]
            var env = ProcessInfo.processInfo.environment
            env["PYTHONUNBUFFERED"] = "1"
            env.removeValue(forKey: "HF_TOKEN")
            env.removeValue(forKey: "IMAGE_STUDIO_HF_TOKEN")
            env["IMAGE_STUDIO_TOKEN"] = apiToken
            if let token, !token.isEmpty { env["IMAGE_STUDIO_HF_TOKEN"] = token }
            child.environment = env
            child.standardOutput = log
            child.standardError = log
            stopping = false
            child.terminationHandler = { [weak self] proc in
                Task { @MainActor in
                    guard let self, !self.stopping, self.process === proc else { return }
                    self.onUnexpectedExit?(proc.terminationStatus)
                }
            }
            process = child
            try child.run()
            for _ in 0..<240 {
                if !child.isRunning { break }
                if (try? await get("/health")) != nil { return }
                try await Task.sleep(nanoseconds: 250_000_000)
            }
            let retry = !child.isRunning && child.terminationStatus == 98 && attempt < 2
            await stop()
            if !retry { throw BackendError(message: "Backend did not start. See \(Self.logFile.path)") }
        }
    }

    func stop() async {
        stopping = true
        guard let old = process else { return }
        if old.isRunning { old.terminate() }
        let deadline = Date().addingTimeInterval(5)
        while old.isRunning && Date() < deadline {
            try? await Task.sleep(nanoseconds: 50_000_000)
        }
        if old.isRunning { kill(old.processIdentifier, SIGKILL) }
        if old.isRunning { await Task.detached { old.waitUntilExit() }.value }
        if process === old { process = nil }
        session.invalidateAndCancel()
        session = Self.makeSession()
    }

    func preflight(_ request: GenerateRequest) async throws -> Preflight {
        try JSONDecoder().decode(Preflight.self, from: try await post("/preflight", body: request))
    }

    func setSettings(_ settings: EngineSettings) async throws { _ = try await post("/settings", body: settings) }

    func optimizeModel(_ id: String, quantize: Int) async throws {
        _ = try await post("/models/\(id)/optimize", body: ["quantize": quantize])
    }

    func status() async throws -> ServerStatus {
        try JSONDecoder().decode(ServerStatus.self, from: try await get("/status"))
    }

    func generate(_ request: GenerateRequest) async throws -> GenerateResponse {
        var urlRequest = URLRequest(url: baseURL.appendingPathComponent("generate"))
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        urlRequest.httpBody = try JSONEncoder().encode(request)
        urlRequest.setValue(apiToken, forHTTPHeaderField: "X-Image-Studio-Token")
        let (data, response) = try await session.data(for: urlRequest)
        try Self.check(data, response)
        return try JSONDecoder().decode(GenerateResponse.self, from: data)
    }

    func estimate(_ request: GenerateRequest) async throws -> Estimate {
        try JSONDecoder().decode(Estimate.self, from: try await post("/estimate", body: request))
    }

    func setIdleUnload(minutes: Int) async throws {
        _ = try await post("/settings", body: ["idle_unload_minutes": minutes])
    }

    func setLivePreview(_ on: Bool) async throws {
        _ = try await post("/settings", body: ["live_preview": on])
    }

    /// Forwards macOS memory-pressure events so the engine can shrink caches or unload while idle.
    func setPressure(_ level: String) async throws {
        _ = try await post("/pressure", body: ["level": level])
    }

    func preview() async throws -> Data { try await get("/preview") }

    func storageScan(keep: [String]) async throws -> [StorageItem] {
        struct Body: Encodable { var keep: [String] }
        struct Response: Decodable { var items: [StorageItem] }
        return try JSONDecoder().decode(Response.self, from: try await post("/storage/scan", body: Body(keep: keep))).items
    }

    func storageClean(ids: [String], keep: [String]) async throws -> StorageCleanResult {
        struct Body: Encodable { var ids: [String]; var keep: [String] }
        return try JSONDecoder().decode(StorageCleanResult.self, from: try await post("/storage/clean", body: Body(ids: ids, keep: keep)))
    }

    struct UpscaleResult: Decodable {
        var path: String?
        var seconds: Double
        var width: Int?
        var height: Int?
        var seed: Int?
        var cancelled: Bool?
    }

    func upscale(path: String, scale: Double = 2, softness: Double = 0) async throws -> UpscaleResult {
        struct Body: Encodable { var path: String; var scale: Double; var softness: Double }
        return try JSONDecoder().decode(UpscaleResult.self, from: try await post("/upscale", body: Body(path: path, scale: scale, softness: softness)))
    }

    func setToken(_ token: String?) async throws -> TokenStatus {
        var urlRequest = URLRequest(url: baseURL.appendingPathComponent("token"))
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        urlRequest.httpBody = try JSONEncoder().encode(["token": token])
        urlRequest.setValue(apiToken, forHTTPHeaderField: "X-Image-Studio-Token")
        let (data, response) = try await session.data(for: urlRequest)
        try Self.check(data, response)
        return try JSONDecoder().decode(TokenStatus.self, from: data)
    }

    func models() async throws -> [ModelInfo] {
        try JSONDecoder().decode([ModelInfo].self, from: try await get("/models"))
    }

    func system() async throws -> SystemInfo {
        try JSONDecoder().decode(SystemInfo.self, from: try await get("/system"))
    }

    func architectures() async throws -> [Architecture] {
        try JSONDecoder().decode([Architecture].self, from: try await get("/architectures"))
    }

    func downloadModel(_ id: String) async throws { _ = try await post("/models/\(id)/download") }
    func cancelDownload(_ id: String) async throws { _ = try await post("/models/\(id)/cancel") }
    func deleteModel(_ id: String, scope: String = "all") async throws { _ = try await post("/models/\(id)/delete", body: ["scope": scope]) }
    func unloadModel() async throws { _ = try await post("/unload") }

    func loadModel(_ id: String, quantize: Int, loraPath: String?, loraScale: Double) async throws {
        struct Body: Encodable { var quantize: Int; var lora_path: String?; var lora_scale: Double }
        _ = try await post("/models/\(id)/load", body: Body(quantize: quantize, lora_path: loraPath, lora_scale: loraScale))
    }

    func addCustomModel(name: String, path: String, architecture: String) async throws {
        struct Body: Encodable { var name: String; var path: String; var architecture: String }
        _ = try await post("/models/custom", body: Body(name: name, path: path, architecture: architecture))
    }

    private func post(_ path: String, body: (any Encodable)? = nil) async throws -> Data {
        var urlRequest = URLRequest(url: URL(string: baseURL.absoluteString + path)!)
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let body { urlRequest.httpBody = try JSONEncoder().encode(body) }
        urlRequest.setValue(apiToken, forHTTPHeaderField: "X-Image-Studio-Token")
        let (data, response) = try await session.data(for: urlRequest)
        try Self.check(data, response)
        return data
    }

    func cancel() async {
        var urlRequest = URLRequest(url: baseURL.appendingPathComponent("cancel"))
        urlRequest.httpMethod = "POST"
        urlRequest.setValue(apiToken, forHTTPHeaderField: "X-Image-Studio-Token")
        _ = try? await session.data(for: urlRequest)
    }

    private func get(_ path: String) async throws -> Data {
        var urlRequest = URLRequest(url: baseURL.appendingPathComponent(path))
        urlRequest.timeoutInterval = path == "/models" ? 60 : 5
        urlRequest.setValue(apiToken, forHTTPHeaderField: "X-Image-Studio-Token")
        let (data, response) = try await session.data(for: urlRequest)
        try Self.check(data, response)
        return data
    }

    static func check(_ data: Data, _ response: URLResponse) throws {
        guard let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) else { return }
        let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        var detail = object?["detail"] as? String
        if let entries = object?["detail"] as? [[String: Any]] {
            detail = entries.map { entry in
                let loc = (entry["loc"] as? [Any])?.map { String(describing: $0) }.joined(separator: ".") ?? "Request"
                return loc + ": " + (entry["msg"] as? String ?? "Invalid value")
            }.joined(separator: "\n")
        }
        throw BackendError(message: detail ?? "Backend error \(http.statusCode)", statusCode: http.statusCode)
    }

    private static func freePort() -> Int {
        let sock = socket(AF_INET, SOCK_STREAM, 0)
        defer { close(sock) }
        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_addr.s_addr = inet_addr("127.0.0.1")
        addr.sin_port = 0
        var len = socklen_t(MemoryLayout<sockaddr_in>.size)
        _ = withUnsafeMutablePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(sock, $0, len) }
        }
        _ = withUnsafeMutablePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(sock, $0, &len) }
        }
        return Int(UInt16(bigEndian: addr.sin_port))
    }
}
