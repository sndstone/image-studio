import AppKit
import SwiftUI
import UniformTypeIdentifiers
import ImageIO

// MARK: - Controls

/// Grey caption for an option the selected model doesn't support.
struct UnsupportedNote: View {
    let text: String
    init(_ text: String) { self.text = text }

    var body: some View {
        Label(text, systemImage: "nosign").font(.caption).foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
    }
}

struct ChoiceOption<T: Hashable> {
    var value: T
    var label: String
    var unsupported: String? = nil  // reason shown as a tooltip; the option is greyed out
    var help: String? = nil
}

/// Segmented control whose options can be individually greyed out (SwiftUI's can't).
struct ChoicePicker<T: Hashable>: View {
    @Binding var selection: T
    let options: [ChoiceOption<T>]
    @Environment(\.isEnabled) private var isEnabled

    var body: some View {
        HStack(spacing: 2) {
            ForEach(options, id: \.label) { option in
                let available = option.unsupported == nil
                let selected = option.value == selection && available
                Button { selection = option.value } label: {
                    Text(option.label)
                        .strikethrough(!available)
                        .lineLimit(1)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 3)
                        .padding(.horizontal, 6)
                        .foregroundStyle(selected && isEnabled ? Color.white : available ? .primary : .secondary)
                        .background(RoundedRectangle(cornerRadius: 5)
                            .fill(selected ? Color.accentColor.opacity(isEnabled ? 1 : 0.35) : .clear))
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .disabled(!available)
                .help(option.unsupported ?? option.help ?? "")
            }
        }
        .padding(2)
        .background(RoundedRectangle(cornerRadius: 7).fill(Color.secondary.opacity(0.12)))
        .opacity(isEnabled ? 1 : 0.5)
    }
}

struct ControlsView: View {
    @EnvironmentObject var model: AppModel
    @State private var showAdvanced = false
    @State private var showToken = false

    var body: some View {
        VStack(spacing: 0) {
            ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    section("Prompt") {
                        TextEditor(text: $model.prompt)
                            .font(.body)
                            .frame(minHeight: 110)
                            .scrollContentBackground(.hidden)
                            .padding(6)
                            .background(Color(nsColor: .textBackgroundColor))
                            .clipShape(RoundedRectangle(cornerRadius: 6))
                            .overlay(RoundedRectangle(cornerRadius: 6).stroke(.separator))
                        Text("With reference images, describe the edit or how to combine them, e.g. “put the person from image 1 in the room from image 2”.")
                            .font(.caption).foregroundStyle(.secondary)
                        let negativeOK = model.currentModel?.supports_negative == true
                        TextField("Negative prompt (what to avoid)", text: $model.negativePrompt, axis: .vertical)
                            .lineLimit(1...3)
                            .textFieldStyle(.roundedBorder)
                            .disabled(!negativeOK)
                        if !negativeOK {
                            UnsupportedNote("Negative prompt isn't supported by \(model.currentModel?.name ?? "this model").")
                        } else if model.currentModel?.passesPerStep(guidance: model.guidance) != 2 && !model.negativePrompt.isEmpty {
                            Text(model.currentModel?.double_pass_when == "guidance!=1"
                                 ? "The negative prompt takes effect whenever guidance differs from 1."
                                 : "The negative prompt only takes effect with guidance above 1.")
                                .font(.caption).foregroundStyle(.orange)
                        }
                    }

                    section("Reference images") { ReferencesView() }

                    section("Model") { ModelSectionView() }

                    section("Size") { SizeView() }

                    section("Generation") {
                        Color.clear.frame(height: 0).id("generation")
                        let info = model.currentModel
                        let fixedSteps = info?.fixed_steps == true
                        LabeledContent("Steps") {
                            Stepper("\(model.steps)", value: $model.steps, in: 1...100)
                                .disabled(fixedSteps)
                        }
                        if fixedSteps {
                            UnsupportedNote("Fixed at \(model.steps) steps: this turbo model was trained for exactly that many.")
                        }
                        LabeledContent("Guidance") {
                            HStack {
                                Slider(value: $model.guidance, in: (info?.min_guidance ?? 0)...10, step: 0.5)
                                Text(String(format: "%.1f", model.guidance)).monospacedDigit().frame(width: 32)
                            }
                        }
                        .disabled(info?.usesGuidance == false)
                        if let info, !info.usesGuidance {
                            UnsupportedNote("Guidance isn't used by \(info.name): it's built into the model.")
                        } else if let info, info.passesPerStep(guidance: model.guidance) == 2 {
                            let steps = fixedSteps ? info.default_steps : model.steps
                            Text("Guidance \(String(format: "%.1f", model.guidance)) runs the model twice per step "
                                 + "(with and without your prompt), so \(steps) steps = \(steps * 2) model passes, "
                                 + "about twice the time. Guidance 1.0 runs one pass per step"
                                 + (info.distilled == true ? "; this model is tuned for 1.0 and rarely benefits from more." : " but follows the prompt less closely."))
                                .font(.caption)
                                .foregroundStyle(info.distilled == true ? .orange : .secondary)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        if info?.supports_step_cache == true { Toggle("Fast", isOn: $model.fast) }
                        if info?.supports_enhance == true { Toggle("Improve prompt", isOn: $model.enhancePrompt).disabled(model.references.isEmpty) }
                        Button("Reset to Model Defaults") { model.resetModelDefaults() }.font(.caption)
                        LabeledContent("Images") {
                            Stepper("\(model.count)", value: $model.count, in: 1...16)
                        }
                        LabeledContent("Seed") {
                            HStack {
                                Toggle("Random", isOn: $model.randomSeed).toggleStyle(.checkbox)
                                TextField("Seed", value: $model.seed, format: .number.grouping(.never))
                                    .frame(width: 110)
                                    .disabled(model.randomSeed)
                            }
                        }
                    }

                    DisclosureGroup("LoRAs", isExpanded: $showAdvanced) {
                        LorasView().padding(.top, 6)
                    }

                    DisclosureGroup("API key (Hugging Face)", isExpanded: $showToken) { TokenView().padding(.top, 6) }
                }
                .padding(16)
            }
            .task {
                // UI-test hook: show the Generation section without scrolling by hand.
                guard ProcessInfo.processInfo.environment["IMAGE_STUDIO_TEST_SCROLL"] == "1" else { return }
                try? await Task.sleep(for: .seconds(6))
                proxy.scrollTo("generation", anchor: .top)
            }
            }

            Divider()
            GenerateBar().padding(16)
        }
    }

    @ViewBuilder
    private func section<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title).font(.headline)
            content()
        }
    }
}

struct ModelSectionView: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Picker("Model", selection: $model.modelID) {
                    ForEach(model.models.filter { !$0.isTool }) { info in
                        Text(info.name + (info.is_new ? "  ✦ new" : "") + (info.status.downloaded ? "" : "  ↓"))
                            .tag(info.id)
                    }
                }
                .labelsHidden()
                Button("Manage…") { model.showModelManager = true }
                    .help("Download, load, delete or add models (⇧⌘M)")
            }
            if let info = model.currentModel {
                Text(info.description).font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                HStack(spacing: 8) {
                    Label(info.license, systemImage: info.commercial ? "checkmark.seal" : "exclamationmark.shield")
                    if !info.status.downloaded {
                        Label(info.status.downloading ? "Downloading" : "Not downloaded", systemImage: "arrow.down.circle")
                            .foregroundStyle(.orange)
                    } else if info.loaded {
                        Label("In memory", systemImage: "memorychip")
                    }
                }
                .font(.caption).foregroundStyle(.secondary)

                LabeledContent("Precision") {
                    ChoicePicker(selection: $model.quantize, options: [8, 4, 0].map { q in
                        ChoiceOption(value: q, label: quantLabel(q), unsupported: info.quant_options.contains(q) ? nil
                            : "\(quantLabel(q)) isn't available for \(info.name): too large for this Mac's memory.")
                    })
                }
                if let mem = info.memory(for: model.quantize) {
                    let total = Double(model.system?.memory_total ?? 0) / 1_073_741_824
                    let tight = total > 0 && mem > total * 0.5
                    let measured = info.memory_measured?.contains(model.quantize) == true
                    Text("Needs about \(Int(mem.rounded())) GB of memory at \(quantLabel(model.quantize))"
                         + (measured ? " (measured on this Mac)." : " (estimate until the first run).")
                         + (tight ? " That's most of this Mac's \(Int(total.rounded())) GB: close other large apps, or it will swap and slow down a lot." : ""))
                        .font(.caption)
                        .foregroundStyle(tight ? .orange : .secondary)
                }
            }
        }
    }
}

struct TokenView: View {
    @EnvironmentObject var model: AppModel
    @State private var reveal = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Group {
                    if reveal {
                        TextField("hf_…", text: $model.tokenDraft)
                    } else {
                        SecureField("hf_…", text: $model.tokenDraft)
                    }
                }
                .textFieldStyle(.roundedBorder)
                .font(.system(.body, design: .monospaced))
                .onSubmit { model.saveToken() }
                Button { reveal.toggle() } label: { Image(systemName: reveal ? "eye.slash" : "eye") }
                    .buttonStyle(.borderless)
                    .help(reveal ? "Hide token" : "Show token")
            }
            HStack {
                Button("Save") { model.saveToken() }
                    .disabled(model.tokenDraft.trimmingCharacters(in: .whitespaces).isEmpty)
                Button("Remove", role: .destructive) { model.clearToken() }
                    .disabled(!model.hasSavedToken && model.tokenDraft.isEmpty)
                if model.isCheckingToken { ProgressView().controlSize(.small) }
                Spacer()
                Link("Get a token", destination: URL(string: "https://huggingface.co/settings/tokens")!)
                    .font(.caption)
            }
            if let status = model.tokenStatus {
                Label(status, systemImage: model.tokenOK ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(model.tokenOK ? .green : .orange)
            }
            Text("Stored in your macOS Keychain and only sent to Hugging Face to download gated models (e.g. klein 9B). A read-only token is enough. Images are still generated locally.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }
}

struct ReferencesView: View {
    @EnvironmentObject var model: AppModel
    @State private var targeted = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 72), spacing: 8)], spacing: 8) {
                ForEach(Array(model.references.enumerated()), id: \.element) { index, url in
                    ZStack(alignment: .topTrailing) {
                        Thumbnail(url: url).frame(width: 72, height: 72)
                            .opacity(model.referenceIsUsed(at: index) ? 1 : 0.35)
                        Button { model.references.removeAll { $0 == url } } label: {
                            Image(systemName: "xmark.circle.fill").symbolRenderingMode(.palette)
                                .foregroundStyle(.white, .black.opacity(0.6))
                        }
                        .buttonStyle(.plain)
                        .padding(3)
                        Text("\(index + 1)").font(.caption2.bold()).padding(.horizontal, 5).padding(.vertical, 1)
                            .background(.black.opacity(0.6), in: Capsule()).foregroundStyle(.white)
                            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomLeading).padding(3)
                    }
                    .onDrag { NSItemProvider(object: ("reference:" + url.path) as NSString) }
                    .onDrop(of: [.plainText], delegate: ReferenceReorderDrop(model: model, target: url))
                    .help(url.lastPathComponent)
                }
                Button { model.pickReferences() } label: {
                    VStack(spacing: 4) {
                        Image(systemName: "plus").font(.title3)
                        Text("Add").font(.caption)
                    }
                    .frame(width: 72, height: 72)
                    .background(RoundedRectangle(cornerRadius: 6).strokeBorder(style: StrokeStyle(lineWidth: 1, dash: [4])))
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
            .padding(6)
            .background(RoundedRectangle(cornerRadius: 8).fill(targeted ? Color.accentColor.opacity(0.15) : .clear))
            .onDrop(of: [.fileURL, .image], isTargeted: $targeted, perform: handleDrop)

            Text(hint).font(.caption).foregroundStyle(.secondary)
            if let notice = model.referenceNotice {
                Label(notice, systemImage: "info.circle").font(.caption).foregroundStyle(.secondary)
            }
            if let info = model.currentModel, model.references.count > 1, model.imageMode == "init",
               info.image_modes.contains("reference") {
                // Measured: in this mode klein ignored both images and Qwen returned image 1 almost unchanged.
                Text(info.family == "qwen21"
                     ? "Start from image 1 keeps image 1's layout, so things from the other images rarely get in. To put something from one image into another, use Edit / combine."
                     : "Start from image 1 uses only image 1 with \(info.name). To put something from one image into another, use Edit / combine.")
                    .font(.caption).foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let info = model.currentModel, model.references.count > info.max_refs, info.edit != "none" {
                Text("\(info.name) uses only the first \(info.max_refs) image\(info.max_refs == 1 ? "" : "s").")
                    .font(.caption).foregroundStyle(.orange)
            }
            if let info = model.currentModel, info.edit != "none" {
                let hasImages = !model.references.isEmpty
                let mode = info.image_modes.contains(model.imageMode) ? model.imageMode : (info.image_modes.first ?? "reference")
                LabeledContent("Use images to") {
                    ChoicePicker(selection: $model.imageMode, options: [
                        ChoiceOption(value: "reference", label: "Edit / combine",
                                     unsupported: info.image_modes.contains("reference") ? nil
                                        : "\(info.name) can't edit or combine images; it can only start from image 1.",
                                     help: "The model looks at the images and follows your instructions."),
                        ChoiceOption(value: "init", label: "Start from image 1",
                                     unsupported: info.image_modes.contains("init") ? nil
                                        : "\(info.name) can't start from an image.",
                                     help: "Image 1 is redrawn; its layout and colors are kept."),
                    ])
                }
                .disabled(!hasImages)
                if hasImages {
                    Text(modeExplanation(info)).font(.caption).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                    if !info.image_modes.contains("reference") {
                        UnsupportedNote("Edit / combine isn't supported by \(info.name).")
                    }
                }
                if info.supports_mask == true {
                    TextField("Only change…", text: $model.autoMask).textFieldStyle(.roundedBorder).disabled(!hasImages)
                    HStack {
                        Button("Paint Mask…") { model.showMaskEditor = true }.disabled(!hasImages)
                        if model.maskPath != nil { Button("Clear Mask") { model.maskPath = nil } }
                    }
                }
                LabeledContent("Change") {
                    HStack {
                        Slider(value: $model.strength, in: 0.1...0.95, step: 0.05)
                        Text("\(Int(model.strength * 100))%").monospacedDigit().frame(width: 40)
                    }
                }
                .help("How much the result may differ from image 1.")
                .disabled(!hasImages || mode != "init")
                if hasImages && mode == "init" {
                    Text("Lower keeps image 1's layout and colors; higher follows the prompt more. Around \(Int(info.default_strength * 100))% works well for \(info.name).")
                        .font(.caption).foregroundStyle(.secondary)
                } else if hasImages {
                    UnsupportedNote("Change only applies to Start from image 1.")
                }
            }
        }
        .sheet(isPresented: $model.showMaskEditor) { MaskEditorView() }
    }

    private var hint: String {
        let info = model.currentModel
        if info?.edit == "none" { return "\(info?.name ?? "This model") is text-to-image only; reference images are ignored." }
        if model.references.isEmpty {
            if info?.edit == "single" {
                return "Optional. \(info?.name ?? "This model") can start from one image (image-to-image). No image = text-to-image."
            }
            return "Optional. Drop images here to edit them or combine them (up to \(info?.max_refs ?? 4)). No images = text-to-image."
        }
        return "Drag to add more, × to remove. Numbers show the order (image 1, image 2…)."
    }

    private func modeExplanation(_ info: ModelInfo) -> String {
        let initOnly = !info.image_modes.contains("reference")
        if model.imageMode == "init" || initOnly {
            let others = model.references.count > 1
                ? (info.family == "qwen21" ? " Other images still act as references." : " Other images are ignored.")
                : ""
            return "Image 1 is the canvas: the result keeps its layout, pose and colors, and the Change slider "
                + "sets how far it may drift. Best for restyling or changing the mood of one photo. Describe "
                + "the result you want (\"a snowy winter scene\").\(others)"
        }
        // Tested: composing works when the output takes the scene's shape (the size picker defaults to the
        // last image) and the prompt names everything to move ("the person" alone dropped their motorcycle).
        return "The model looks at the images and follows instructions: change the style, add or remove things, "
            + "or combine them (\"put the man and his motorcycle from image 1 on the square in image 2\"). Refer "
            + "to them as image 1, image 2… and name everything you want moved. When putting something into a "
            + "scene, let the size match the scene's image."
    }

    private func handleDrop(_ providers: [NSItemProvider]) -> Bool {
        let revision = model.referenceImportRevision
        Task {
            // Await each provider in order; asynchronous callbacks otherwise scramble image 1.
            for provider in providers {
                if provider.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) {
                    let url: URL? = await withCheckedContinuation { continuation in
                        _ = provider.loadObject(ofClass: URL.self) { url, _ in continuation.resume(returning: url) }
                    }
                    guard revision == model.referenceImportRevision else { return }
                    if let url { model.importDroppedFile(url) }
                } else {
                    if let type = provider.registeredTypeIdentifiers.first(where: { ImageImports.isImageType($0) }) {
                        let data: Data? = await withCheckedContinuation { continuation in
                            _ = provider.loadDataRepresentation(forTypeIdentifier: type) { data, _ in continuation.resume(returning: data) }
                        }
                        guard revision == model.referenceImportRevision else { return }
                        if let data { model.addReference(data: data); continue }
                    }
                    if provider.canLoadObject(ofClass: NSImage.self) {
                        let image: NSImage? = await withCheckedContinuation { continuation in
                            _ = provider.loadObject(ofClass: NSImage.self) { image, _ in continuation.resume(returning: image as? NSImage) }
                        }
                        guard revision == model.referenceImportRevision else { return }
                        if let image { model.addReference(image: image) }
                    }
                }
            }
        }
        return true
    }

}

struct SizeView: View {
    @EnvironmentObject var model: AppModel
    @FocusState private var dimensionFocused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 4) {
                ForEach(aspectPresets) { preset in
                    let active = model.width == preset.width && model.height == preset.height
                    Button(preset.label) {
                        model.autoSize = false
                        model.width = preset.width
                        model.height = preset.height
                        model.normalizeSize()
                    }
                    .buttonStyle(.bordered)
                    .tint(active ? .accentColor : nil)
                    .controlSize(.small)
                }
            }
            HStack {
                TextField("W", value: $model.width, format: .number.grouping(.never)).frame(width: 64).focused($dimensionFocused).onSubmit { model.autoSize = false; model.normalizeSize() }
                Text("×")
                TextField("H", value: $model.height, format: .number.grouping(.never)).frame(width: 64).focused($dimensionFocused).onSubmit { model.autoSize = false; model.normalizeSize() }
                Text("px").foregroundStyle(.secondary)
                Spacer()
                Toggle(model.references.count >= 2 ? "Match image" : "Match image 1 (~1 MP)", isOn: $model.autoSize)
                    .toggleStyle(.checkbox)
                    .controlSize(.small)
                    .help("Use that image's shape at about one megapixel")
                if model.references.count >= 2 {
                    Picker("", selection: Binding(get: { model.sizeReference },
                                                  set: { model.sizeReferenceIsAuto = false; model.sizeReference = $0 })) {
                        ForEach(model.references.indices, id: \.self) { Text("\($0 + 1)").tag($0) }
                    }
                    .labelsHidden().frame(width: 52).controlSize(.small)
                    .disabled(!model.autoSize)
                    .help("Which image's shape the result takes. When putting something into a scene, pick the scene.")
                }
            }
        }
        .onChange(of: dimensionFocused) { _, focused in if !focused, !model.restoringSettings { model.autoSize = false; model.normalizeSize() } }
    }
}

