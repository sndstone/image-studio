import AppKit
import Foundation
import ImageIO

extension URL {
    private var lowercasedExtension: String { pathExtension.lowercased() }

    /// The file types a generation can produce.
    var isGeneratedMedia: Bool { ["png", "gif", "svg"].contains(lowercasedExtension) }
    var isGIF: Bool { lowercasedExtension == "gif" }
    var isSVG: Bool { lowercasedExtension == "svg" }

    /// Model inputs must be pixels, and a GIF's first frame is. An SVG isn't, by OpenAI's rules and ours.
    var canBeReference: Bool { !isSVG }
    var canUpscale: Bool { !isSVG && !isGIF }
    var formatBadge: String? { isGIF ? "GIF" : isSVG ? "SVG" : nil }
}

enum GIFFrames {
    struct Frame {
        var image: NSImage
        var delay: TimeInterval
    }

    /// Browsers treat a delay this short as a mistake in the file; so does this.
    private static let shortestDelay: TimeInterval = 0.02
    private static let defaultDelay: TimeInterval = 0.1

    static func decode(_ url: URL) -> [Frame] {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, [kCGImageSourceShouldCache: false] as CFDictionary) else { return [] }
        return (0..<CGImageSourceGetCount(source)).compactMap { index in
            guard let cg = CGImageSourceCreateImageAtIndex(source, index, [kCGImageSourceShouldCacheImmediately: true] as CFDictionary) else { return nil }
            let properties = CGImageSourceCopyPropertiesAtIndex(source, index, nil) as? [CFString: Any]
            let gif = properties?[kCGImagePropertyGIFDictionary] as? [CFString: Any]
            let stated = (gif?[kCGImagePropertyGIFUnclampedDelayTime] ?? gif?[kCGImagePropertyGIFDelayTime]) as? Double ?? defaultDelay
            return Frame(image: NSImage(cgImage: cg, size: NSSize(width: cg.width, height: cg.height)), delay: max(shortestDelay, stated))
        }
    }
}

/// ImageIO can't read SVG but NSImage can, without running anything in the file.
enum SVGImages {
    /// Vectors have no pixel size, so a generous one: 100% stays sharp whatever size the file declares.
    private static let displayLongSide: CGFloat = 2048

    static func displaySize(of url: URL) -> CGSize? {
        guard let size = NSImage(contentsOf: url)?.size, size.width > 0, size.height > 0 else { return nil }
        let scale = displayLongSide / max(size.width, size.height)
        return CGSize(width: size.width * scale, height: size.height * scale)
    }

    static func render(_ url: URL, maxPixels: Int) -> CGImage? {
        guard let image = NSImage(contentsOf: url), image.size.width > 0, image.size.height > 0 else { return nil }
        let scale = CGFloat(maxPixels) / max(image.size.width, image.size.height)
        let width = max(1, Int((image.size.width * scale).rounded())), height = max(1, Int((image.size.height * scale).rounded()))
        guard let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                                      space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        NSGraphicsContext.saveGraphicsState()
        defer { NSGraphicsContext.restoreGraphicsState() }
        NSGraphicsContext.current = NSGraphicsContext(cgContext: context, flipped: false)
        image.draw(in: NSRect(x: 0, y: 0, width: width, height: height))
        return context.makeImage()
    }
}
