import AppKit
import ImageIO
import UniformTypeIdentifiers

// Export only reads generation settings when metadata is retained.
enum AppModel {
    static func settingsJSON(at url: URL) -> [String: Any]? { ["prompt": "export regression"] }
}

@main
struct ImageExportsTests {
    static func main() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        // Six distinct colors make rotation and mirroring observable independently of dimensions.
        let colors: [[UInt8]] = [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255],
                                 [255, 255, 0, 255], [255, 0, 255, 255], [0, 255, 255, 255]]
        let pixels = Data(colors.flatMap { $0 })
        let provider = CGDataProvider(data: pixels as CFData)!
        let image = CGImage(width: 3, height: 2, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: 12,
                            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
        let expected = [[0, 1, 2, 3, 4, 5], [2, 1, 0, 5, 4, 3], [5, 4, 3, 2, 1, 0], [3, 4, 5, 0, 1, 2],
                        [0, 3, 1, 4, 2, 5], [3, 0, 4, 1, 5, 2], [5, 2, 4, 1, 3, 0], [2, 5, 1, 4, 0, 3]]
        for orientation in 1...8 {
            let input = directory.appendingPathComponent("source-\(orientation).tiff")
            let destination = CGImageDestinationCreateWithURL(input as CFURL, UTType.tiff.identifier as CFString, 1, nil)!
            CGImageDestinationAddImage(destination, image, [kCGImagePropertyOrientation: orientation,
                kCGImagePropertyExifDictionary: [kCGImagePropertyExifUserComment: "private comment"]] as CFDictionary)
            precondition(CGImageDestinationFinalize(destination))
            for removeMetadata in [true, false] {
                let output = directory.appendingPathComponent("export-\(orientation)-\(removeMetadata).png")
                let failure = ImageExports.write(input: input, output: output, format: UTType.png.identifier, removeMetadata: removeMetadata)
                precondition(failure == nil, failure ?? "")
                let source = CGImageSourceCreateWithURL(output as CFURL, nil)!
                let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as! [CFString: Any]
                let decoded = CGImageSourceCreateImageAtIndex(source, 0, nil)!
                if removeMetadata {
                    precondition(decoded.width == (orientation < 5 ? 3 : 2))
                    precondition(decoded.height == (orientation < 5 ? 2 : 3))
                    precondition((properties[kCGImagePropertyOrientation] as? NSNumber)?.intValue ?? 1 == 1)
                    let exif = properties[kCGImagePropertyExifDictionary] as? [CFString: Any]
                    precondition(exif?[kCGImagePropertyExifUserComment] == nil)
                    var bytes = [UInt8](repeating: 0, count: 24)
                    let context = CGContext(data: &bytes, width: decoded.width, height: decoded.height, bitsPerComponent: 8,
                                            bytesPerRow: decoded.width * 4, space: CGColorSpaceCreateDeviceRGB(),
                                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
                    context.draw(decoded, in: CGRect(x: 0, y: 0, width: decoded.width, height: decoded.height))
                    let expectedBytes = expected[orientation - 1].flatMap { colors[$0] }
                    precondition(zip(bytes, expectedBytes).allSatisfy { abs(Int($0) - Int($1)) <= 1 }, "Pixel transform failed for orientation \(orientation): \(bytes)")
                } else {
                    precondition(decoded.width == 3 && decoded.height == 2)
                    precondition((properties[kCGImagePropertyOrientation] as? NSNumber)?.intValue ?? 1 == orientation)
                    let exif = properties[kCGImagePropertyExifDictionary] as? [CFString: Any]
                    precondition((exif?[kCGImagePropertyExifUserComment] as? String)?.contains("export regression") == true)
                }
            }
        }
        print("Image export regression checks passed (all eight EXIF orientations, stripped and retained metadata).")
    }
}
