import AppKit
import ImageIO
import SwiftUI
import UniformTypeIdentifiers

/// What GPT Image can do beyond a prompt and a size. Options a model lacks stay visible, locked, with the reason.
struct OpenAIOptionsView: View {
    @EnvironmentObject var model: AppModel
    var info: ModelInfo

    private static let qualityNames = ["auto": "Auto", "low": "Low", "medium": "Medium", "high": "High", "xhigh": "Extra high", "max": "Max"]

    var body: some View {
        let format = model.effectiveOutputFormat(for: info)
        LabeledContent("Output") {
            ChoicePicker(selection: Binding(get: { format }, set: { model.outputFormat = $0 }), options: [
                ChoiceOption(value: "png", label: "Image"),
                ChoiceOption(value: "gif", label: "GIF", unsupported: animatedOrVectorBlock("Animated GIFs"), help: "An animation: describe the motion and the loop in the prompt."),
                ChoiceOption(value: "svg", label: "SVG", unsupported: animatedOrVectorBlock("SVGs"), help: "Editable vector shapes, for icons and illustrations."),
            ])
        }
        LabeledContent("Quality") {
            Picker("Quality", selection: Binding(
                get: { (info.cloud_qualities ?? []).contains(model.openaiQuality) ? model.openaiQuality : "auto" },
                set: { model.openaiQuality = $0 })) {
                ForEach(info.cloud_qualities ?? ["auto"], id: \.self) { Text(Self.qualityNames[$0] ?? $0).tag($0) }
            }
            .labelsHidden().fixedSize()
            .help(format == "svg" ? "Sets the quality of the reference image the SVG is drawn from." : "Higher quality costs noticeably more per image. Auto lets OpenAI choose.")
        }
        LabeledContent("Complexity") {
            ChoicePicker(selection: Binding(get: { info.cloud_complexity == true ? model.openaiComplexity : "high" }, set: { model.openaiComplexity = $0 }),
                         options: [("low", "Low"), ("medium", "Medium"), ("high", "High")].map { value, label in
                ChoiceOption(value: value, label: label, unsupported: info.cloud_complexity == true ? nil : "Complexity is available on GPT Image 2.6.",
                             help: "How much the model plans before rendering. No extra fee.")
            })
        }
        if format == "gif" { gifControls }
        if format == "svg" {
            Text("OpenAI chooses the size. An SVG's cost isn't shown here: it includes a drawing model whose price isn't published.")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }
    }

    @ViewBuilder private var gifControls: some View {
        LabeledContent("Frames") {
            Stepper("\(model.gifFrameCount)", value: $model.gifFrameCount, in: 2...16)
        }
        LabeledContent("Frame time") {
            HStack {
                Slider(value: Binding(get: { Double(model.gifFrameDuration) }, set: { model.gifFrameDuration = Int(($0 / 10).rounded()) * 10 }), in: 20...2000)
                Text("\(model.gifFrameDuration) ms").monospacedDigit().frame(width: 60, alignment: .trailing)
            }
        }
        Text("Plays for \(String(format: "%.1f", Double(model.gifFrameCount * model.gifFrameDuration) / 1000)) s per loop.")
            .font(.caption).foregroundStyle(.secondary)
        Toggle("Let OpenAI choose the frame size", isOn: $model.gifAutoSize)
            .help("All the frames share a pixel budget. Your own size may be refused if it's too large for this many frames.")
    }

    private func animatedOrVectorBlock(_ name: String) -> String? {
        if (info.cloud_formats ?? []).count < 3 { return "\(name) are available on GPT Image 2.6." }
        if !model.references.isEmpty { return "\(name) can't use reference images: remove them first." }
        return nil
    }
}

struct MarkupStroke {
    var points: [CGPoint]
    var size: CGFloat
    var color: Color
    var cgColor: CGColor
}

/// Annotations for OpenAI's markup-guided edits, saved as a transparent layer the size of image 1.
struct MarkupEditorView: View {
    @EnvironmentObject var model: AppModel
    @Environment(\.dismiss) var dismiss
    @State private var strokes: [MarkupStroke] = []
    @State private var brushSize = 12.0
    @State private var drawing = false
    @State private var colorIndex = 0
    @State private var error: String?
    @State private var pixels = CGSize(width: 1024, height: 1024)
    @State private var resolvedSource: URL?

    private static let palette: [(name: String, color: NSColor)] = [
        ("Red", .systemRed), ("Yellow", .systemYellow), ("Green", .systemGreen), ("Blue", .systemBlue), ("White", .white), ("Black", .black),
    ]
    private var source: URL? { model.references.first }

    var body: some View {
        VStack(spacing: 12) {
            Text("Draw on image 1 to show where the change goes").font(.headline)
            Text("Circle, sketch or point at things, then describe the change in the prompt.").font(.caption).foregroundStyle(.secondary)
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
                            context.stroke(path, with: .color(stroke.color), style: StrokeStyle(lineWidth: stroke.size * ratio, lineCap: .round, lineJoin: .round))
                        }
                    }
                    .contentShape(Rectangle())
                    .gesture(DragGesture(minimumDistance: 0).onChanged { value in
                        let point = CGPoint(x: min(1, max(0, value.location.x / size.width)), y: min(1, max(0, value.location.y / size.height)))
                        if !drawing {
                            let choice = Self.palette[colorIndex].color
                            strokes.append(MarkupStroke(points: [point, point], size: CGFloat(brushSize), color: Color(nsColor: choice), cgColor: choice.usingColorSpace(.sRGB)?.cgColor ?? choice.cgColor))
                            drawing = true
                        } else if !strokes.isEmpty { strokes[strokes.count - 1].points.append(point) }
                    }.onEnded { _ in drawing = false })
                }.frame(width: size.width, height: size.height).position(x: geometry.size.width / 2, y: geometry.size.height / 2)
            }.frame(width: 700, height: 470)
            HStack(spacing: 6) {
                ForEach(Self.palette.indices, id: \.self) { index in
                    Button { colorIndex = index } label: {
                        Circle().fill(Color(nsColor: Self.palette[index].color)).frame(width: 18, height: 18)
                            .overlay(Circle().stroke(colorIndex == index ? Color.accentColor : Color.secondary.opacity(0.5), lineWidth: colorIndex == index ? 3 : 1))
                    }
                    .buttonStyle(.plain)
                    .help(Self.palette[index].name)
                    .accessibilityLabel(Self.palette[index].name)
                    .accessibilityAddTraits(colorIndex == index ? .isSelected : [])
                }
                Text("Size").padding(.leading, 8)
                Slider(value: $brushSize, in: 2...80).frame(width: 130)
                Text("\(Int(brushSize)) px").monospacedDigit().frame(width: 44, alignment: .leading)
                Button("Undo") { _ = strokes.popLast() }.disabled(strokes.isEmpty).keyboardShortcut("z", modifiers: .command)
                Button("Clear") { strokes = [] }.disabled(strokes.isEmpty)
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Use Markup") { save() }.disabled(strokes.isEmpty || resolvedSource != source).keyboardShortcut(.defaultAction)
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
            if ProcessInfo.processInfo.environment["IMAGE_STUDIO_TEST_MARKUP"] == "draw" {  // UI tests: top-left to the middle
                let color = NSColor.systemRed
                strokes = [MarkupStroke(points: [CGPoint(x: 0.1, y: 0.1), CGPoint(x: 0.5, y: 0.5)], size: 40, color: Color(nsColor: color), cgColor: color.usingColorSpace(.sRGB)?.cgColor ?? color.cgColor)]
                save()
            }
        }
    }

    private func save() {
        let w = Int(pixels.width), h = Int(pixels.height)
        guard let context = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                                      space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { error = "Could not create the markup."; return }
        context.setLineCap(.round); context.setLineJoin(.round)
        context.translateBy(x: 0, y: CGFloat(h)); context.scaleBy(x: 1, y: -1)
        for stroke in strokes {
            guard let first = stroke.points.first else { continue }
            context.setStrokeColor(stroke.cgColor)
            context.setLineWidth(stroke.size); context.beginPath()
            context.move(to: CGPoint(x: first.x * pixels.width, y: first.y * pixels.height))
            for point in stroke.points.dropFirst() { context.addLine(to: CGPoint(x: point.x * pixels.width, y: point.y * pixels.height)) }
            context.strokePath()
        }
        do {
            let directory = Backend.runtimeDir.appendingPathComponent("masks", isDirectory: true)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let url = directory.appendingPathComponent(UUID().uuidString + ".png")
            guard let image = context.makeImage(), let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else { throw BackendError(message: "Could not save the markup.") }
            CGImageDestinationAddImage(destination, image, nil)
            guard CGImageDestinationFinalize(destination) else { throw BackendError(message: "Could not write the markup PNG.") }
            model.markupPath = url
            dismiss()
        } catch { self.error = error.localizedDescription }
    }
}
