import AppKit
import ImageIO
import UniformTypeIdentifiers

enum ImageExports {
    static let original = "original"

    static func write(input: URL, output: URL, format: String, removeMetadata: Bool) -> String? {
        if format == original {
            do {
                if FileManager.default.fileExists(atPath: output.path) { try FileManager.default.removeItem(at: output) }
                try FileManager.default.copyItem(at: input, to: output)
                return nil
            } catch { return "Could not write the exported file: " + error.localizedDescription }
        }
        if input.isSVG {
            guard let image = SVGImages.render(input, maxPixels: 2048),
                  let destination = CGImageDestinationCreateWithURL(output as CFURL, format as CFString, 1, nil) else {
                return "This image format could not be exported."
            }
            CGImageDestinationAddImage(destination, image, [kCGImageDestinationLossyCompressionQuality: 0.95] as CFDictionary)
            return CGImageDestinationFinalize(destination) ? nil : "Could not write the exported image."
        }
        guard let source = CGImageSourceCreateWithURL(input as CFURL, [kCGImageSourceShouldCache: false] as CFDictionary) else {
            return "This image format could not be exported."
        }
        let image: CGImage?
        if removeMetadata {
            // Orientation is metadata too: bake it into the pixels before stripping it.
            let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any] ?? [:]
            let width = (properties[kCGImagePropertyPixelWidth] as? NSNumber)?.intValue ?? 0
            let height = (properties[kCGImagePropertyPixelHeight] as? NSNumber)?.intValue ?? 0
            image = CGImageSourceCreateThumbnailAtIndex(source, 0, [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceCreateThumbnailWithTransform: true,
                kCGImageSourceThumbnailMaxPixelSize: max(width, height),
                kCGImageSourceShouldCacheImmediately: true,
            ] as CFDictionary)
        } else {
            image = CGImageSourceCreateImageAtIndex(source, 0, [kCGImageSourceShouldCacheImmediately: true] as CFDictionary)
        }
        guard let image, let destination = CGImageDestinationCreateWithURL(output as CFURL, format as CFString, 1, nil) else {
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
