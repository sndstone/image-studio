import AppKit
import SwiftUI
import UniformTypeIdentifiers
import ImageIO

// MARK: - Results

struct ResultView: View {
    @EnvironmentObject var model: AppModel
    @State private var showInspector = false
    @State private var showExport = false
    @State private var showUpscale = false
    @State private var showWipe = false
    @State private var wipe = 0.5

    var body: some View {
        VStack(spacing: 0) {
            ZStack {
                Color(nsColor: .underPageBackgroundColor)
                if !model.compareItems.isEmpty {
                    CompareGrid()
                        .overlay(alignment: .bottomTrailing) {
                            if model.isGenerating, let preview = model.previewImage {
                                Image(nsImage: preview).resizable().scaledToFit().frame(width: 180, height: 180)
                                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8)).padding(16)
                            }
                        }
                } else if let pending = model.selectedPendingItem {
                    PendingResultView(item: pending)
                } else if let item = model.selected {
                    if model.showFullView {
                        Thumbnail(url: item.url, maxPixels: 1024, revision: item, fit: true)
                    } else {
                        ZoomableImage(url: item.url, beforeURL: showWipe ? model.comparisonSource(for: item) : nil, wipe: CGFloat(wipe), revision: item, zoom: model.zoom)
                            .contextMenu { ItemMenu(item: item) }
                            .overlay(alignment: .bottomTrailing) { ZoomControls(zoom: model.zoom).padding(12) }
                    }
                } else {
                    VStack(spacing: 8) {
                        Image(systemName: "photo.on.rectangle.angled").font(.system(size: 48)).foregroundStyle(.tertiary)
                        Text("Write a prompt, optionally add reference images, and press ⌘↩").foregroundStyle(.secondary)
                    }
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            if let item = model.selected {
                if model.comparisonSource(for: item) != nil {
                    HStack {
                        Toggle("Before / After", isOn: $showWipe).toggleStyle(.checkbox)
                        if showWipe { Text("Before"); Slider(value: $wipe, in: 0...1).frame(maxWidth: 240); Text("After") }
                    }.font(.caption).padding(.horizontal, 12).padding(.vertical, 4)
                }
                HStack {
                    Text(item.url.lastPathComponent).font(.caption).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
                    Text("seed \(String(item.seed))").font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                    if item.seconds > 0 {
                        Text(String(format: "%.1fs", item.seconds)).font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                }.padding(.horizontal, 12).padding(.vertical, 3)
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack {
                        Button("Upscale…") { showUpscale = true }
                            .popover(isPresented: $showUpscale) { UpscaleOptionsView(item: item) }
                            .help("Add detail and double the size with the SeedVR2 upscaler")
                        Button("Edit This Image") { model.editImage(item) }
                            .help("Use this result as image 1 and describe the next change")
                        Button("Use as Reference") { model.addReferences([item.url]) }
                        Button("Reuse Settings") { model.reuseSettings(of: item) }
                        Button { showInspector = true } label: { Image(systemName: "info.circle") }
                            .popover(isPresented: $showInspector) { ResultInspector(item: item) }
                        Button("Export…") { showExport = true }
                            .sheet(isPresented: $showExport) { ExportView(item: item) }
                        Button { model.toggleFavorite(item) } label: { Image(systemName: model.favorites.contains(item.url.path) ? "star.fill" : "star") }
                        Button { NSWorkspace.shared.activateFileViewerSelecting([item.url]) } label: {
                            Image(systemName: "folder")
                        }
                        .help("Show in Finder")
                    }
                    .controlSize(.small)
                    .padding(.horizontal, 12).padding(.vertical, 6)
                }
            }

            Divider()
            HStack {
                Text("History").font(.caption.bold()).foregroundStyle(.secondary)
                Text("\(model.historyIndex.count)").font(.caption).foregroundStyle(.tertiary)
                if model.compareSelection.count >= 2 {
                    Button("Compare \(model.compareSelection.count) Selected") { model.compareSelected() }
                    Button("Deselect") { model.compareSelection = [] }
                } else {
                    Text("⌘-click images to compare them").font(.caption).foregroundStyle(.tertiary)
                }
                Spacer()
                Button { NSWorkspace.shared.open(model.outputDir) } label: { Image(systemName: "folder") }
                    .help("Open output folder")
                Button("Clear History…", role: .destructive) { model.requestClearHistory() }
                    .disabled(model.history.isEmpty || model.isGenerating)
            }
            .buttonStyle(.borderless)
            .controlSize(.small)
            .padding(.horizontal, 12)
            .padding(.top, 6)
            HStack {
                TextField("Search prompt or model", text: $model.historySearch).textFieldStyle(.roundedBorder)
                Toggle("Favorites only", isOn: $model.favoritesOnly).toggleStyle(.checkbox)
            }.font(.caption).padding(.horizontal, 12).padding(.top, 4)
            ScrollViewReader { strip in
            ScrollView(.horizontal) {
                LazyHStack(spacing: 8) {
                    ForEach(model.galleryItems) { entry in
                        if let item = entry.result {
                            Thumbnail(url: item.url, revision: item)
                                .id(entry.id)
                                .frame(width: 84, height: 84)
                                .overlay(RoundedRectangle(cornerRadius: 6)
                                    .stroke(model.compareSelection.contains(item.url) ? Color.orange
                                            : model.selected == item ? Color.accentColor : .clear, lineWidth: 3))
                                .overlay(alignment: .topTrailing) {
                                    if model.favorites.contains(item.url.path) {
                                        Image(systemName: "star.fill").foregroundStyle(.yellow).padding(4)
                                    }
                                    if model.compareSelection.contains(item.url) {
                                        Image(systemName: "checkmark.circle.fill").foregroundStyle(.white, .orange).padding(4)
                                    }
                                }
                                .onTapGesture {
                                    NSApp.keyWindow?.makeFirstResponder(nil)
                                    if NSEvent.modifierFlags.contains(.command) {
                                        model.toggleCompareSelection(item)
                                    } else {
                                        model.followJob = false
                                        model.selected = item
                                        model.closeCompare()
                                    }
                                }
                                .onDrag { NSItemProvider(contentsOf: item.url) ?? NSItemProvider() }
                                .contextMenu { ItemMenu(item: item) }
                        } else if let pending = entry.pending {
                            PendingThumbnail(item: pending)
                                .id(entry.id)
                                .frame(width: 84, height: 84)
                                .overlay(RoundedRectangle(cornerRadius: 6)
                                    .stroke(model.selectedPending == pending.id ? Color.accentColor : .clear, lineWidth: 3))
                                .onTapGesture {
                                    NSApp.keyWindow?.makeFirstResponder(nil)
                                    model.selectPending(pending.id)
                                }
                        }
                    }
                    if model.history.count < model.historyIndex.count && model.historySearch.isEmpty && !model.favoritesOnly {
                        Button("Load More") { model.loadMoreHistory() }.frame(width: 84, height: 84)
                            .onAppear { model.loadMoreHistory() }
                    }
                }
                .padding(10)
            }
            .frame(height: 104)
            .onChange(of: model.selectedPending) { _, id in
                if let id { withAnimation { strip.scrollTo(GalleryItem.ID.pending(id), anchor: .leading) } }
            }
            .onChange(of: model.selected?.url) { _, url in
                if let entry = model.galleryItems.first(where: { $0.result?.url == url }) {
                    withAnimation { strip.scrollTo(entry.id) }
                }
            }
            }
        }
        .confirmationDialog("Move all \(model.clearHistoryCount) generated images to the Trash?",
                            isPresented: $model.confirmClearHistory) {
            Button("Move to Trash", role: .destructive) { model.clearHistory() }
        } message: {
            Text("This moves Image Studio’s date folders to the Trash. Other folders are preserved. You can restore the images from the Trash.")
        }
    }
}

struct ItemMenu: View {
    @EnvironmentObject var model: AppModel
    var item: HistoryItem

    var body: some View {
        Button("Edit This Image") { model.editImage(item) }
        Button("Upscale with Current Options") { model.upscale(item) }
        Button("Variations ×4") { model.variations(item) }
        Button(model.favorites.contains(item.url.path) ? "Remove Favorite" : "Favorite") { model.toggleFavorite(item) }
        Button("Use as Reference") { model.addReferences([item.url]) }
        Button("Reuse Settings") { model.reuseSettings(of: item) }
        Button("Copy Image") {
            if let image = NSImage(contentsOf: item.url) {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.writeObjects([image])
            }
        }
        Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([item.url]) }
        Divider()
        Button("Move to Trash", role: .destructive) { model.delete(item) }
    }
}

struct Thumbnail: View {
    var url: URL
    var maxPixels = 256
    var revision: HistoryItem? = nil
    var fit = false  // show the whole image (compare view) instead of filling a square
    @State private var image: NSImage?

    var body: some View {
        Color.secondary.opacity(fit ? 0 : 0.1)
            .overlay {
                if let image {
                    Image(nsImage: image).resizable().aspectRatio(contentMode: fit ? .fit : .fill)
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .contentShape(Rectangle())
        .task(id: ThumbnailKey(url: url, maxPixels: maxPixels, revision: revision)) {
            let size = maxPixels
            image = nil
            let loaded = await ImageThumbnails.load(url, maxPixels: size)
            guard !Task.isCancelled else { return }
            image = loaded
        }
        .onDisappear { image = nil }
    }
}

struct PendingResultView: View {
    @EnvironmentObject var model: AppModel
    var item: PendingItem

    var body: some View {
        ZStack {
            if item.id == model.previewPending, let preview = model.previewImage {
                Image(nsImage: preview).resizable().interpolation(.high).scaledToFit().padding(20)
                    .overlay(alignment: .topLeading) {
                        Text("Live preview · " + model.pendingLabel(item))
                            .font(.caption.bold()).padding(8)
                            .background(.ultraThinMaterial, in: Capsule()).padding(28)
                    }
            } else {
                VStack(spacing: 12) {
                    if item.id == model.previewPending { ProgressView() }
                    else { Image(systemName: "clock").font(.largeTitle).foregroundStyle(.secondary) }
                    Text(model.pendingLabel(item)).foregroundStyle(.secondary)
                }
            }
        }.frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

struct PendingThumbnail: View {
    @EnvironmentObject var model: AppModel
    var item: PendingItem

    var body: some View {
        RoundedRectangle(cornerRadius: 6).fill(Color.secondary.opacity(0.1))
            .overlay {
                if item.id == model.previewPending, let preview = model.previewImage {
                    Image(nsImage: preview).resizable().scaledToFill()
                }
            }
            .overlay {
                VStack(spacing: 5) {
                    if item.id == model.previewPending { ProgressView().controlSize(.small) }
                    else { Image(systemName: "clock") }
                    Text(model.pendingLabel(item)).font(.caption2).lineLimit(1)
                }.padding(6).background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 6))
            }
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .contentShape(Rectangle())
            .help(item.isUpscale ? "Upscaling" : "Image \(item.index + 1)")
    }
}

struct ThumbnailKey: Hashable {
    var url: URL
    var maxPixels: Int
    var revision: HistoryItem? = nil
}

/// Shared small decoded images; full-size results are never cached.
enum ImageThumbnails {
    private static let cache: NSCache<NSString, NSImage> = {
        let cache = NSCache<NSString, NSImage>()
        cache.countLimit = 300
        cache.totalCostLimit = 100 * 1024 * 1024
        return cache
    }()
    private static let coordination = DispatchQueue(label: "ImageStudio.thumbnails")
    private static let decodes: OperationQueue = {
        let queue = OperationQueue()
        queue.name = "ImageStudio.thumbnailDecodes"
        queue.maxConcurrentOperationCount = 4
        queue.qualityOfService = .utility
        return queue
    }()
    private static var inFlight: [String: [CheckedContinuation<NSImage?, Never>]] = [:]
    private static var generation = 0

    static func purge() {
        cache.removeAllObjects()
        coordination.async { generation += 1; cache.removeAllObjects() }
    }

    static func load(_ url: URL, maxPixels: Int = 256) async -> NSImage? {
        guard !Task.isCancelled else { return nil }
        return await withCheckedContinuation { continuation in
            coordination.async {
                var file = url
                file.removeCachedResourceValue(forKey: .contentModificationDateKey)
                let modified = (try? file.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate)?.timeIntervalSince1970 ?? 0
                let key = "\(url.absoluteString)|\(modified)|\(maxPixels)"
                if let image = cache.object(forKey: key as NSString) { continuation.resume(returning: image); return }
                if inFlight[key] != nil { inFlight[key]?.append(continuation); return }
                inFlight[key] = [continuation]
                let epoch = generation
                decodes.addOperation {
                    let decoded = autoreleasepool { decode(url, maxPixels: maxPixels) }
                    let image = decoded?.image
                    coordination.async {
                        if let decoded, epoch == generation {
                            cache.setObject(decoded.image, forKey: key as NSString, cost: decoded.cost)
                        }
                        let waiting = inFlight.removeValue(forKey: key) ?? []
                        for waiter in waiting { waiter.resume(returning: image) }
                    }
                }
            }
        }
    }

    private static func decode(_ url: URL, maxPixels: Int) -> (image: NSImage, cost: Int)? {
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceThumbnailMaxPixelSize: maxPixels,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceShouldCacheImmediately: true,
        ]
        guard let source = CGImageSourceCreateWithURL(url as CFURL, [kCGImageSourceShouldCache: false] as CFDictionary),
              let cg = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else { return nil }
        return (NSImage(cgImage: cg, size: NSSize(width: cg.width, height: cg.height)), cg.bytesPerRow * cg.height)
    }
}
