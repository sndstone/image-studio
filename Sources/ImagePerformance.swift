import AppKit
import Darwin
import ImageIO
import SwiftUI
import UniformTypeIdentifiers

struct FileIdentity: Equatable {
    var device: dev_t
    var inode: ino_t
    var size: off_t
    var modifiedSeconds: Int
    var modifiedNanoseconds: Int
    var changedSeconds: Int
    var changedNanoseconds: Int

    init?(at url: URL) {
        var info = stat()
        guard url.withUnsafeFileSystemRepresentation({ path in
            path.map { stat($0, &info) == 0 } ?? false
        }) else { return nil }
        device = info.st_dev
        inode = info.st_ino
        size = info.st_size
        modifiedSeconds = info.st_mtimespec.tv_sec
        modifiedNanoseconds = info.st_mtimespec.tv_nsec
        changedSeconds = info.st_ctimespec.tv_sec
        changedNanoseconds = info.st_ctimespec.tv_nsec
    }
}

struct HistoryMetadataVersion: Equatable {
    var image: FileIdentity?
    var sidecar: FileIdentity?

    init(at url: URL) {
        image = FileIdentity(at: url)
        sidecar = FileIdentity(at: url.deletingPathExtension().appendingPathExtension("json"))
    }
}

extension AppModel {
    var editorSettingsSnapshot: EditorSettingsSnapshot {
        let request = GenerateRequest(prompt: prompt, negative_prompt: negativePrompt, image_paths: references.map(\.path),
                                      model: modelID, quantize: quantize, width: width, height: height, steps: steps,
                                      guidance: guidance, strength: strength, image_mode: imageMode, seed: seed, count: count,
                                      lora_path: loraPath?.path, lora_scale: loraScale, fast: fast, mask_path: maskPath?.path,
                                      auto_mask: autoMask, enhance_prompt: enhancePrompt, loras: loras)
        return EditorSettingsSnapshot(request: request, randomSeed: randomSeed, autoSize: autoSize,
                                      sizeReference: sizeReference, sizeReferenceIsAuto: sizeReferenceIsAuto, selectedURL: selected?.url)
    }

    func editorSettingsChanged() {
        guard booted, !restoringSettings else { return }
        let snapshot = editorSettingsSnapshot
        guard snapshot != lastEditorSettings else { return }
        lastEditorSettings = snapshot
        settingsSaveTask?.cancel()
        settingsSaveTask = Task { @MainActor [weak self] in
            do { try await Task.sleep(nanoseconds: 700_000_000) } catch { return }
            guard let self else { return }
            settingsSaveTask = nil
            saveGenerationSettings(model: modelID)
        }
    }

    nonisolated static func readMetadata(at url: URL, json: [String: Any]?) -> HistoryMetadata {
        let hasSettings = json != nil
        let json = json ?? [:]
        let path = json["upscaled_from"] as? String ?? (json["image_paths"] as? [String])?.first
        let source = path.flatMap { FileManager.default.fileExists(atPath: $0) ? URL(fileURLWithPath: $0) : nil }
        return HistoryMetadata(version: HistoryMetadataVersion(at: url), hasSettings: hasSettings, model: json["model"] as? String, steps: json["steps"] as? Int,
                               upscale: json["upscale"] as? Double, seconds: json["seconds"] as? Double ?? 0,
                               comparisonSource: source,
                               description: HistoryDescription(prompt: json["prompt"] as? String ?? "", model: json["model"] as? String ?? ""))
    }

    func loadMetadata(for url: URL) {
        guard metadataLoads[url] == nil else { return }
        let token = UUID()
        let cachedVersion = historyMetadata[url]?.version
        metadataLoads[url] = token
        Task {
            let record = await Task.detached(priority: .utility) { () -> HistoryMetadata? in
                let version = HistoryMetadataVersion(at: url)
                guard cachedVersion != version else { return nil }
                var record = Self.readMetadata(at: url, json: Self.settingsJSON(at: url))
                // A write during the read must also cause the next request to refresh.
                record.version = version
                return record
            }.value
            guard metadataLoads[url] == token else { return }
            metadataLoads[url] = nil
            guard let record else { return }
            historyMetadata[url] = record
            historyDescriptions[url.path] = record.description
        }
    }

    func invalidateMetadata(at url: URL, removed: Bool = false) {
        historyMetadata[url] = nil
        metadataLoads[url] = nil
        // Sources may themselves be history items that were just deleted or overwritten.
        guard removed else { return }
        for key in Array(historyMetadata.keys) where historyMetadata[key]?.comparisonSource == url {
            historyMetadata[key]?.comparisonSource = nil
            // Re-check the source on the next selection, in case it is restored at the same path.
            historyMetadata[key]?.version.image = nil
        }
    }
}

/// Runs only on an image worker; file/data imports retain their original encoding and color metadata.
enum ImageImports {
    private static let imageTypes = Set(CGImageSourceCopyTypeIdentifiers() as! [String])

    static func isImageType(_ identifier: String) -> Bool { imageTypes.contains(identifier) }

    static func validURLs(_ urls: [URL]) -> [URL] {
        urls.filter { url in
            guard url.isFileURL,
                  let source = CGImageSourceCreateWithURL(url as CFURL, [kCGImageSourceShouldCache: false] as CFDictionary),
                  CGImageSourceGetCount(source) > 0, let type = CGImageSourceGetType(source) else { return false }
            return isImageType(type as String)
        }
    }

    static func paste(urls: [URL], data: [Data], image: NSImage?, directory: URL) -> [URL] {
        let images = validURLs(urls)
        if !images.isEmpty { return images }
        for representation in data {
            if let url = persist(data: representation, directory: directory) { return [url] }
        }
        return image.flatMap { persist(image: $0, directory: directory) }.map { [$0] } ?? []
    }

    private static func filenameExtension(for type: String) -> String {
        if let suffix = UTType(type)?.preferredFilenameExtension { return suffix }
        return ["public.png": "png", "public.jpeg": "jpg", "public.tiff": "tiff", "public.heic": "heic",
                "public.heif": "heif", "com.compuserve.gif": "gif", "org.webmproject.webp": "webp",
                "public.avif": "avif", "com.microsoft.bmp": "bmp"][type] ?? "image"
    }

    private static func destination(extension suffix: String, directory: URL) -> URL? {
        guard (try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)) != nil else { return nil }
        return directory.appendingPathComponent(UUID().uuidString).appendingPathExtension(suffix)
    }

    static func persist(data: Data, directory: URL) -> URL? {
        guard let source = CGImageSourceCreateWithData(data as CFData, [kCGImageSourceShouldCache: false] as CFDictionary),
              CGImageSourceGetCount(source) > 0, let type = CGImageSourceGetType(source),
              isImageType(type as String),
              let url = destination(extension: filenameExtension(for: type as String), directory: directory) else { return nil }
        guard (try? data.write(to: url, options: .atomic)) != nil else { return nil }
        return url
    }

    static func persist(image: NSImage, directory: URL) -> URL? {
        guard let cg = image.cgImage(forProposedRect: nil, context: nil, hints: nil),
              let url = destination(extension: "png", directory: directory),
              let output = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(output, cg, nil)
        return CGImageDestinationFinalize(output) ? url : nil
    }
}

enum ImageExports {
    static func write(input: URL, output: URL, format: String, removeMetadata: Bool) -> String? {
        guard let source = CGImageSourceCreateWithURL(input as CFURL, [kCGImageSourceShouldCache: false] as CFDictionary),
              let image = CGImageSourceCreateImageAtIndex(source, 0, [kCGImageSourceShouldCacheImmediately: true] as CFDictionary),
              let destination = CGImageDestinationCreateWithURL(output as CFURL, format as CFString, 1, nil) else {
            return "This image format could not be exported."
        }
        var properties: [CFString: Any] = [:]
        if !removeMetadata {
            properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any] ?? [:]
            if let json = AppModel.settingsJSON(at: input), let data = try? JSONSerialization.data(withJSONObject: json, options: [.sortedKeys]), let text = String(data: data, encoding: .utf8) {
                var exif = properties[kCGImagePropertyExifDictionary] as? [CFString: Any] ?? [:]
                exif[kCGImagePropertyExifUserComment] = text
                properties[kCGImagePropertyExifDictionary] = exif
            }
        }
        properties[kCGImageDestinationLossyCompressionQuality] = 0.95
        CGImageDestinationAddImage(destination, image, properties as CFDictionary)
        return CGImageDestinationFinalize(destination) ? nil : "Could not write the exported image."
    }
}
