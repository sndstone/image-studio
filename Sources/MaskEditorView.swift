import AppKit
import ImageIO
import SwiftUI
import UniformTypeIdentifiers

struct MaskStroke {
    var points: [CGPoint]
    var size: CGFloat
}

struct MaskEditorView: View {
    @EnvironmentObject var model: AppModel
    @Environment(\.dismiss) var dismiss
    @State private var strokes: [MaskStroke] = []
    @State private var brushSize = 40.0
    @State private var drawing = false
    @State private var error: String?

    private var source: URL? { model.references.first }
    @State private var pixels = CGSize(width: 1024, height: 1024)
    @State private var resolvedSource: URL?

    var body: some View {
        VStack(spacing: 12) {
            Text("Paint the area to change in image 1").font(.headline)
            GeometryReader { geometry in
                let ratio = min(geometry.size.width / pixels.width, geometry.size.height / pixels.height)
                let size = CGSize(width: pixels.width * ratio, height: pixels.height * ratio)
                ZStack {
                    if let source { Thumbnail(url: source, maxPixels: 2048, fit: true) }
                    Canvas { context, _ in
                        for stroke in strokes {
                            var path = Path()
                            if let first = stroke.points.first { path.move(to: CGPoint(x: first.x * size.width, y: first.y * size.height)) }
                            for point in stroke.points.dropFirst() { path.addLine(to: CGPoint(x: point.x * size.width, y: point.y * size.height)) }
                            context.stroke(path, with: .color(.red.opacity(0.6)), style: StrokeStyle(lineWidth: stroke.size * ratio, lineCap: .round, lineJoin: .round))
                        }
                    }
                    .contentShape(Rectangle())
                    .gesture(DragGesture(minimumDistance: 0).onChanged { value in
                        let point = CGPoint(x: min(1, max(0, value.location.x / size.width)), y: min(1, max(0, value.location.y / size.height)))
                        if !drawing {
                            strokes.append(MaskStroke(points: [point, point], size: CGFloat(brushSize)))
                            drawing = true
                        } else if !strokes.isEmpty { strokes[strokes.count - 1].points.append(point) }
                    }.onEnded { _ in drawing = false })
                }.frame(width: size.width, height: size.height).position(x: geometry.size.width / 2, y: geometry.size.height / 2)
            }.frame(width: 700, height: 470)
            HStack {
                Text("Brush size")
                Slider(value: $brushSize, in: 5...200).frame(width: 180)
                Text("\(Int(brushSize)) px").monospacedDigit()
                Button("Clear") { strokes = [] }
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Save Mask") { save() }.disabled(strokes.isEmpty || resolvedSource != source).keyboardShortcut(.defaultAction)
            }
            if let error { Text(error).foregroundStyle(.red).font(.caption) }
        }.padding(20)
        .task(id: source) {
            let url = source
            resolvedSource = nil
            strokes = []
            let size = await Task.detached(priority: .utility) { url.flatMap(AppModel.displaySize) }.value
            guard !Task.isCancelled, source == url else { return }
            pixels = size ?? CGSize(width: 1024, height: 1024)
            resolvedSource = url
        }
    }

    private func save() {
        let pixels = pixels
        let w = Int(pixels.width), h = Int(pixels.height)
        guard let context = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w,
                                      space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue) else { error = "Could not create the mask."; return }
        context.setFillColor(gray: 0, alpha: 1); context.fill(CGRect(x: 0, y: 0, width: w, height: h))
        context.setStrokeColor(gray: 1, alpha: 1); context.setLineCap(.round); context.setLineJoin(.round)
        context.translateBy(x: 0, y: CGFloat(h)); context.scaleBy(x: 1, y: -1)
        for stroke in strokes {
            guard let first = stroke.points.first else { continue }
            context.setLineWidth(stroke.size); context.beginPath()
            context.move(to: CGPoint(x: first.x * pixels.width, y: first.y * pixels.height))
            for point in stroke.points.dropFirst() { context.addLine(to: CGPoint(x: point.x * pixels.width, y: point.y * pixels.height)) }
            context.strokePath()
            context.setFillColor(gray: 1, alpha: 1)
            context.fillEllipse(in: CGRect(x: first.x * pixels.width - stroke.size / 2, y: first.y * pixels.height - stroke.size / 2, width: stroke.size, height: stroke.size))
        }
        do {
            let directory = Backend.runtimeDir.appendingPathComponent("masks", isDirectory: true)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let url = directory.appendingPathComponent(UUID().uuidString + ".png")
            guard let image = context.makeImage(), let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else { throw BackendError(message: "Could not save the mask.") }
            CGImageDestinationAddImage(destination, image, nil)
            guard CGImageDestinationFinalize(destination) else { throw BackendError(message: "Could not write the mask PNG.") }
            model.maskPath = url
            dismiss()
        } catch { self.error = error.localizedDescription }
    }
}
