import AppKit
import SwiftUI
import UniformTypeIdentifiers
import ImageIO

// MARK: - Results

struct ResultView: View {
    @EnvironmentObject var model: AppModel
    @State private var showWipe = false
    @State private var wipe = 0.5
    @State private var columnHeight: CGFloat = 0

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
                    VStack(spacing: 12) {
                        Image(systemName: "photo.on.rectangle.angled")
                            .font(.system(size: 40, weight: .light)).foregroundStyle(.tertiary)
                        Text("Your next image starts here").font(.title3.weight(.medium))
                        Text("Write a prompt and choose Generate. Add reference images to edit or combine them.")
                            .font(.callout).foregroundStyle(.secondary)
                            .multilineTextAlignment(.center).frame(maxWidth: 320)
                        Text("⌘ Return to generate").font(.caption).foregroundStyle(.tertiary)
                    }.padding(32)
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
                    if let cost = model.historyMetadata[item.url]?.costUSD {
                        Button(formatUSD(cost)) { model.showCosts = true }
                            .buttonStyle(.link).font(.caption.monospacedDigit())
                            .help("What this image cost. Click to open Costs.")
                    }
                    if item.seconds > 0 {
                        Text(String(format: "%.1fs", item.seconds)).font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                }.padding(.horizontal, 12).padding(.vertical, 3)
                ResultActions(item: item)
                    .padding(.horizontal, 12).padding(.top, 6).padding(.bottom, 12)

            }

            // Leave the result about 200 pt plus its info line and action buttons (~150 pt) and the history header.
            HistoryPanel(maxGridHeight: max(120, columnHeight - 400))
        }
        .background(GeometryReader { geometry in
            Color.clear
                .onAppear { columnHeight = geometry.size.height }
                .onChange(of: geometry.size.height) { _, height in columnHeight = height }
        })
        .confirmationDialog("Move all \(model.clearHistoryCount) generated images to the Trash?",
                            isPresented: $model.confirmClearHistory) {
            Button("Move to Trash", role: .destructive) { model.clearHistory() }
        } message: {
            Text("This moves Image Studio’s date folders to the Trash. Other folders are preserved. You can restore the images from the Trash.")
        }
    }
}

/// The history strip under the result: a single row or a resizable grid, with search and favorites behind icons.
struct HistoryPanel: View {
    @EnvironmentObject var model: AppModel
    @AppStorage("historyLayout") private var layout = "strip"
    @AppStorage("historyGridHeight") private var gridHeight = 220.0
    /// The tallest the grid may be in the current window; the saved height is kept for larger windows.
    var maxGridHeight: CGFloat = 600
    @State private var dragStart: Double?
    @State private var cursorPushed = false
    @FocusState private var searchFocused: Bool

    private let tile: CGFloat = 84

    var body: some View {
        VStack(spacing: 0) {
            // Drag handle: resize the grid by pulling the divider.
            ZStack {
                Divider()
                Color.clear.frame(height: 7).contentShape(Rectangle())
            }
            .onHover { inside in setResizeCursor(inside && layout == "grid") }
            .onChange(of: layout) { _, _ in setResizeCursor(false) }
            .onDisappear { setResizeCursor(false) }
            .gesture(DragGesture(minimumDistance: 1).onChanged { value in
                guard layout == "grid" else { return }
                if dragStart == nil || dragStart! > maxGridHeight { dragStart = min(gridHeight, maxGridHeight) }
                gridHeight = min(600, maxGridHeight, max(120, (dragStart ?? gridHeight) - value.translation.height))
            }.onEnded { _ in dragStart = nil })
            .help(layout == "grid" ? "Drag to resize the history grid" : "")

            HStack(spacing: 8) {
                Text("History").font(.caption.bold()).foregroundStyle(.secondary)
                Text("\(model.historyIndex.count)").font(.caption).foregroundStyle(.tertiary)
                    .accessibilityLabel("\(model.historyIndex.count) images")
                if model.compareSelection.count >= 2 {
                    Button("Compare \(model.compareSelection.count) Selected") { model.compareSelected() }
                    Button("Deselect") { model.compareSelection = [] }
                }
                Spacer()
                if model.historySearchVisible {
                    TextField("Search prompt or model", text: $model.historySearch)
                        .textFieldStyle(.roundedBorder)
                        .controlSize(.small)
                        .frame(maxWidth: 260)
                        .focused($searchFocused)
                        .accessibilityLabel("Search history")
                        .onExitCommand { closeSearch() }
                }
                Picker("Layout", selection: $layout) {
                    Image(systemName: "rectangle.split.3x1").tag("strip").help("One row")
                    Image(systemName: "square.grid.3x3").tag("grid").help("Grid")
                }
                .pickerStyle(.segmented).labelsHidden().controlSize(.small).fixedSize()
                .help("Show history as one row or as a grid")
                Button {
                    if model.historySearchVisible { closeSearch() } else { model.historySearchVisible = true; searchFocused = true }
                } label: { Image(systemName: "magnifyingglass") }
                    .foregroundStyle(model.historySearchVisible || !model.historySearch.isEmpty ? Color.accentColor : .primary)
                    .help("Search prompt or model (⌘F)")
                    .accessibilityLabel("Search history")
                Button { model.favoritesOnly.toggle() } label: { Image(systemName: model.favoritesOnly ? "star.fill" : "star") }
                    .foregroundStyle(model.favoritesOnly ? Color.orange : .primary)
                    .help(model.favoritesOnly ? "Showing favorites only" : "Show favorites only")
                    .accessibilityLabel("Favorites only")
                    .accessibilityAddTraits(model.favoritesOnly ? .isSelected : [])
                Button { NSWorkspace.shared.open(model.outputDir) } label: { Image(systemName: "folder") }
                    .help("Open output folder")
                    .accessibilityLabel("Open output folder")
                Button(role: .destructive) { model.requestClearHistory() } label: { Image(systemName: "trash") }
                    .disabled(model.history.isEmpty || model.isGenerating)
                    .help("Clear history: move all generated images to the Trash")
                    .accessibilityLabel("Clear history")
            }
            .buttonStyle(.borderless)
            .controlSize(.small)
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .help("Click an image to view it, ⌘-click images to compare them, ← → to step through")

            ScrollViewReader { strip in
                Group {
                    if model.galleryItems.isEmpty {
                        // Outside the grid, so the message isn't squeezed into one tile's width.
                        Text(model.historySearch.isEmpty && !model.favoritesOnly
                             ? "Generated images will appear here."
                             : "No images match your filters.")
                            .font(.caption).foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.horizontal, 12)
                            .frame(height: layout == "grid" ? min(gridHeight, maxGridHeight) : tile + 20, alignment: .top)
                            .padding(.top, layout == "grid" ? 10 : 0)
                    } else if layout == "grid" {
                        ScrollView(.vertical) {
                            LazyVGrid(columns: [GridItem(.adaptive(minimum: tile), spacing: 8)], spacing: 8) { tiles }
                                .padding(10)
                        }
                        .frame(height: min(gridHeight, maxGridHeight))
                    } else {
                        ScrollView(.horizontal) {
                            LazyHStack(spacing: 8) { tiles }.padding(10)
                        }
                        .frame(height: tile + 20)
                    }
                }
                .onChange(of: model.selectedPending) { _, id in
                    if let id { withAnimation { strip.scrollTo(GalleryItem.ID.pending(id), anchor: layout == "grid" ? .top : .leading) } }
                }
                .onChange(of: model.selected?.url) { _, url in
                    if let entry = model.galleryItems.first(where: { $0.result?.url == url }) {
                        withAnimation { strip.scrollTo(entry.id) }
                    }
                }
            }
        }
        .onChange(of: model.historySearchFocusRequest) { _, _ in
            model.historySearchVisible = true
            searchFocused = true
        }
    }

    private func setResizeCursor(_ on: Bool) {
        guard on != cursorPushed else { return }
        cursorPushed = on
        if on { NSCursor.resizeUpDown.push() } else { NSCursor.pop() }
    }

    private func closeSearch() {
        model.historySearch = ""
        model.historySearchVisible = false
        searchFocused = false
    }

    @ViewBuilder private var tiles: some View {
        ForEach(model.galleryItems) { entry in
            if let item = entry.result {
                HistoryTile(item: item)
                    .id(entry.id)
                    .frame(width: tile, height: tile)
            } else if let pending = entry.pending {
                PendingThumbnail(item: pending)
                    .id(entry.id)
                    .frame(width: tile, height: tile)
                    .overlay(RoundedRectangle(cornerRadius: 6)
                        .stroke(model.selectedPending == pending.id ? Color.accentColor : .clear, lineWidth: 3))
                    .onTapGesture {
                        NSApp.keyWindow?.makeFirstResponder(nil)
                        model.selectPending(pending.id)
                    }
                    .accessibilityElement(children: .ignore)
                    .accessibilityLabel(model.pendingLabel(pending))
                    .accessibilityAddTraits(.isButton)
                    .accessibilityAction { model.selectPending(pending.id) }
            }
        }
        if model.history.count < model.historyIndex.count && model.historySearch.isEmpty && !model.favoritesOnly {
            Button("Load More") { model.loadMoreHistory() }.frame(width: tile, height: tile)
                .onAppear { model.loadMoreHistory() }
        }
    }
}

/// One finished image in the history: selectable, ⌘-click to pick it for comparison.
struct HistoryTile: View {
    @EnvironmentObject var model: AppModel
    var item: HistoryItem

    var body: some View {
        let favorite = model.favorites.contains(item.url.path)
        let comparing = model.compareSelection.contains(item.url)
        let selected = model.selected == item
        Thumbnail(url: item.url, revision: item)
            .overlay(RoundedRectangle(cornerRadius: 6)
                .stroke(comparing ? Color.orange : selected ? Color.accentColor : .clear, lineWidth: 3))
            .overlay(alignment: .bottomLeading) {
                if let badge = item.url.formatBadge {
                    Text(badge).font(.system(size: 9, weight: .bold)).foregroundStyle(.white)
                        .padding(.horizontal, 4).padding(.vertical, 1)
                        .background(Color.black.opacity(0.65), in: RoundedRectangle(cornerRadius: 3)).padding(4)
                }
            }
            .overlay(alignment: .topTrailing) {
                if favorite { Image(systemName: "star.fill").foregroundStyle(.yellow).padding(4) }
                if comparing { Image(systemName: "checkmark.circle.fill").foregroundStyle(.white, .orange).padding(4) }
            }
            .onTapGesture {
                NSApp.keyWindow?.makeFirstResponder(nil)
                if NSEvent.modifierFlags.contains(.command) {
                    model.toggleCompareSelection(item)
                } else {
                    select()
                }
            }
            .onDrag { NSItemProvider(contentsOf: item.url) ?? NSItemProvider() }
            .contextMenu { ItemMenu(item: item) }
            .help("\(item.url.lastPathComponent) · seed \(item.seed)")
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(accessibilityText(favorite: favorite))
            .accessibilityAddTraits(selected ? [.isButton, .isSelected] : .isButton)
            .accessibilityAction(named: "Select") { select() }
            .accessibilityAction(named: comparing ? "Remove from comparison" : "Add to comparison") { model.toggleCompareSelection(item) }
    }

    private func select() {
        model.followJob = false
        model.selected = item
        model.closeCompare()
    }

    private func accessibilityText(favorite: Bool) -> String {
        var parts = [item.url.lastPathComponent, "seed \(item.seed)"]
        if favorite { parts.append("favorite") }
        return parts.joined(separator: ", ")
    }
}

struct ResultActions: View {
    @EnvironmentObject var model: AppModel
    var item: HistoryItem
    @State private var showInspector = false
    @State private var showExport = false
    @State private var showUpscale = false

    var body: some View {
        HStack(spacing: 8) {
            Button { model.editImage(item) } label: {
                Label("Edit Image", systemImage: "slider.horizontal.3")
            }
            .buttonStyle(StudioButtonStyle(emphasized: true))
            .disabled(!item.url.canBeReference)
            .help(item.url.canBeReference ? "Use this result as image 1 and describe the next change" : "An SVG can't be edited: image models need a pixel image")
            Button { showUpscale = true } label: {
                Label("Upscale…", systemImage: "arrow.up.left.and.arrow.down.right")
            }
            .popover(isPresented: $showUpscale) { UpscaleOptionsView(item: item) }
            .disabled(!item.url.canUpscale)
            .help(item.url.canUpscale ? "Add detail with the SeedVR2 upscaler" : "Only still pixel images can be upscaled")
            Button { showExport = true } label: {
                Label("Export…", systemImage: "square.and.arrow.up")
            }
            .sheet(isPresented: $showExport) { ExportView(item: item) }
            Spacer(minLength: 0)
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 8) {
                    Button("Use as Reference") { model.addReferences([item.url]) }.disabled(!item.url.canBeReference)
                    Button("Reuse Settings") { model.reuseSettings(of: item) }
                }
                Color.clear.frame(width: 0, height: 0)
            }
            Button { model.toggleFavorite(item) } label: {
                Label(model.favorites.contains(item.url.path) ? "Remove Favorite" : "Favorite",
                      systemImage: model.favorites.contains(item.url.path) ? "star.fill" : "star")
                    .labelStyle(.iconOnly)
            }
            .foregroundStyle(model.favorites.contains(item.url.path) ? Color.orange : Color.primary)
            .help(model.favorites.contains(item.url.path) ? "Remove favorite" : "Add to favorites")
            Menu {
                Button("Use as Reference", systemImage: "photo.badge.plus") { model.addReferences([item.url]) }
                    .disabled(!item.url.canBeReference)
                Button("Reuse Settings", systemImage: "arrow.counterclockwise") { model.reuseSettings(of: item) }
                Divider()
                Button("Image Settings…", systemImage: "info.circle") { showInspector = true }
                Button("Show in Finder", systemImage: "folder") { NSWorkspace.shared.activateFileViewerSelecting([item.url]) }
            } label: {
                Label("More image actions", systemImage: "ellipsis").labelStyle(.iconOnly)
            }
            .menuIndicator(.hidden)
            .fixedSize()
            .help("More image actions")
            .popover(isPresented: $showInspector) { ResultInspector(item: item) }
        }
        .buttonStyle(StudioButtonStyle())
    }
}

struct ItemMenu: View {
    @EnvironmentObject var model: AppModel
    var item: HistoryItem

    var body: some View {
        Button("Edit This Image") { model.editImage(item) }.disabled(!item.url.canBeReference)
        Button("Upscale with Current Options") { model.upscale(item) }.disabled(!item.url.canUpscale)
        Button("Variations ×4") { model.variations(item) }
        Button(model.favorites.contains(item.url.path) ? "Remove Favorite" : "Favorite") { model.toggleFavorite(item) }
        Button("Use as Reference") { model.addReferences([item.url]) }.disabled(!item.url.canBeReference)
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
        if url.isSVG {
            return SVGImages.render(url, maxPixels: maxPixels).map {
                (NSImage(cgImage: $0, size: NSSize(width: $0.width, height: $0.height)), $0.bytesPerRow * $0.height)
            }
        }
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
