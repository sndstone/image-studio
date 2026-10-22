import AppKit
import SwiftUI
import ImageIO

/// Zoom level of the result viewer, shared with the menu commands and the on-screen controls.
@MainActor
final class ZoomState: ObservableObject {
    @Published private(set) var magnification: CGFloat = 1
    @Published private(set) var isFit = true  // follows the window size until the user zooms
    fileprivate weak var scrollView: ZoomScrollView?

    static let minimum: CGFloat = 0.05
    static let maximum: CGFloat = 16

    var percentText: String { "\(Int((magnification * 100).rounded()))%" }

    func zoomIn() { step(by: 1.5) }
    func zoomOut() { step(by: 1 / 1.5) }

    /// 100%: one image pixel per screen point, like Preview's Actual Size.
    func actualSize() { set(1, centeredAt: nil) }

    func fit() {
        guard let scrollView, let size = scrollView.documentView?.frame.size, size.width > 0, size.height > 0 else { return }
        scrollView.cancelPendingZoom()
        let visible = scrollView.contentSize
        let scale = min(visible.width / size.width, visible.height / size.height)
        scrollView.magnification = max(Self.minimum, min(Self.maximum, scale))
        if !isFit { isFit = true }
        if magnification != scrollView.magnification { magnification = scrollView.magnification }
        scrollView.ensureResolution(for: scrollView.magnification)
    }

    func set(_ value: CGFloat, centeredAt point: NSPoint?) {
        guard let scrollView, let doc = scrollView.documentView else { return }
        let clamped = max(Self.minimum, min(Self.maximum, value))
        let center = point ?? NSPoint(x: scrollView.contentView.bounds.midX, y: scrollView.contentView.bounds.midY)
        guard scrollView.prepareZoom(clamped, centeredAt: point) else { return }
        scrollView.setMagnification(clamped, centeredAt: doc.convert(center, from: scrollView.contentView))
        if isFit { isFit = false }
        if magnification != scrollView.magnification { magnification = scrollView.magnification }
        scrollView.ensureResolution(for: scrollView.magnification)
    }

    fileprivate func step(by factor: CGFloat, at point: NSPoint? = nil) {
        guard let scrollView else { return }
        set(scrollView.requestedMagnification * factor, centeredAt: point)
    }

    fileprivate func userMagnified() {
        guard let scrollView else { return }
        if isFit { isFit = false }
        if magnification != scrollView.magnification { magnification = scrollView.magnification }
        scrollView.ensureResolution(for: scrollView.magnification)
    }
}

/// A result image that can be zoomed (pinch, ⌘-scroll, buttons, ⌘= / ⌘- / ⌘0 / ⌘9) and panned (drag, scroll).
struct ZoomableImage: NSViewRepresentable {
    var url: URL
    var beforeURL: URL? = nil
    var wipe: CGFloat = 0.5
    var revision: HistoryItem? = nil
    @ObservedObject var zoom: ZoomState

    func makeNSView(context: Context) -> ZoomScrollView {
        let scrollView = ZoomScrollView()
        scrollView.contentView = CenteringClipView()
        scrollView.drawsBackground = false
        scrollView.hasHorizontalScroller = true
        scrollView.hasVerticalScroller = true
        scrollView.autohidesScrollers = true
        scrollView.allowsMagnification = true
        scrollView.minMagnification = ZoomState.minimum
        scrollView.maxMagnification = ZoomState.maximum
        scrollView.zoom = zoom
        let imageView = PanningImageView()
        imageView.imageScaling = .scaleAxesIndependently
        imageView.zoom = zoom
        scrollView.documentView = imageView
        zoom.scrollView = scrollView
        scrollView.magnifyObserver = NotificationCenter.default.addObserver(forName: NSScrollView.didEndLiveMagnifyNotification,
                                               object: scrollView, queue: .main) { [weak zoom] _ in
            MainActor.assumeIsolated { zoom?.userMagnified() }
        }
        return scrollView
    }

    func updateNSView(_ scrollView: ZoomScrollView, context: Context) {
        zoom.scrollView = scrollView
        guard let imageView = scrollView.documentView as? PanningImageView else { return }
        scrollView.loadBeforeImage(at: beforeURL)
        if imageView.wipe != wipe { imageView.wipe = wipe; imageView.needsDisplay = true }
        scrollView.loadImage(at: url, revision: revision)
    }
    static func dismantleNSView(_ scrollView: ZoomScrollView, coordinator: ()) {
        if let observer = scrollView.magnifyObserver { NotificationCenter.default.removeObserver(observer) }
        scrollView.magnifyObserver = nil
        scrollView.beforeLoadTask?.cancel()
        scrollView.imageLoadTask?.cancel()
        scrollView.cancelResolutionLoad()
        (scrollView.documentView as? PanningImageView)?.image = nil
        (scrollView.documentView as? PanningImageView)?.beforeImage = nil
        if scrollView.zoom?.scrollView === scrollView { scrollView.zoom?.scrollView = nil }
        scrollView.zoom = nil
    }
}

final class ZoomScrollView: NSScrollView {
    var magnifyObserver: NSObjectProtocol?
    var loadedURL: URL?
    private var loadedRevision: HistoryItem?
    private var beforeURL: URL?
    var imageLoadTask: Task<Void, Never>?
    var beforeLoadTask: Task<Void, Never>?
    private var sourceSize = CGSize.zero
    private var decodedPixels = 0
    private var requestedPixels = 0
    private var resolutionTask: Task<Void, Never>?
    private var pendingZoom: (scale: CGFloat, point: NSPoint?)?

    func loadImage(at url: URL, revision: HistoryItem? = nil) {
        guard loadedURL != url || loadedRevision != revision else { return }
        loadedURL = url
        loadedRevision = revision
        imageLoadTask?.cancel()
        resolutionTask?.cancel()
        pendingZoom = nil
        sourceSize = .zero
        decodedPixels = 0
        requestedPixels = 0
        guard let imageView = documentView as? PanningImageView else { return }
        imageView.image = nil
        imageLoadTask = Task { @MainActor [weak self] in
            let size = await Task.detached(priority: .userInitiated) { AppModel.displaySize(of: url) }.value
            guard !Task.isCancelled, let self, loadedURL == url, let size else { return }
            sourceSize = size
            imageView.frame = NSRect(origin: .zero, size: size)
            zoom?.fit()
            ensureResolution(for: magnification)
        }
    }

    var requestedMagnification: CGFloat { pendingZoom?.scale ?? magnification }

    func cancelResolutionLoad() { resolutionTask?.cancel(); pendingZoom = nil }
    func cancelPendingZoom() { pendingZoom = nil }

    func prepareZoom(_ scale: CGFloat, centeredAt point: NSPoint?) -> Bool {
        let fit = min(contentSize.width / max(1, sourceSize.width), contentSize.height / max(1, sourceSize.height))
        let full = Int(ceil(max(sourceSize.width, sourceSize.height)))
        guard scale > fit + 0.0001, decodedPixels < full else { pendingZoom = nil; return true }
        pendingZoom = (scale, point)
        ensureResolution(for: scale)
        return false
    }

    func ensureResolution(for scale: CGFloat) {
        guard let url = loadedURL, sourceSize.width > 0, sourceSize.height > 0 else { return }
        let full = Int(ceil(max(sourceSize.width, sourceSize.height)))
        let fit = min(contentSize.width / sourceSize.width, contentSize.height / sourceSize.height)
        let backing = window?.backingScaleFactor ?? NSScreen.main?.backingScaleFactor ?? 2
        // Fit keeps only the pixels the display can show. Any zoom beyond fit requests the original.
        let target = scale > fit + 0.0001 ? full : min(full, max(1, Int(ceil(CGFloat(full) * max(fit, 0.001) * backing))))
        guard target > decodedPixels, target > requestedPixels else { return }
        requestedPixels = target
        resolutionTask?.cancel()
        resolutionTask = Task { @MainActor [weak self] in
            let decode = Task.detached(priority: .userInitiated) { Self.decodeImage(url, maxPixels: target) }
            let image = await withTaskCancellationHandler(operation: { await decode.value }, onCancel: { decode.cancel() })
            guard !Task.isCancelled, let self, loadedURL == url else { return }
            requestedPixels = 0
            guard let image else { pendingZoom = nil; return }
            decodedPixels = target
            (documentView as? PanningImageView)?.image = image
            if let pending = pendingZoom {
                pendingZoom = nil
                zoom?.set(pending.scale, centeredAt: pending.point)
            }
        }
    }

    nonisolated private static func decodeImage(_ url: URL, maxPixels: Int) -> NSImage? {
        guard !Task.isCancelled,
              let source = CGImageSourceCreateWithURL(url as CFURL, [kCGImageSourceShouldCache: false] as CFDictionary) else { return nil }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceThumbnailMaxPixelSize: maxPixels,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceShouldCacheImmediately: true,
        ]
        guard let cg = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary), !Task.isCancelled else { return nil }
        return NSImage(cgImage: cg, size: NSSize(width: cg.width, height: cg.height))
    }

    func loadBeforeImage(at url: URL?) {
        guard beforeURL != url else { return }
        beforeURL = url
        beforeLoadTask?.cancel()
        guard let imageView = documentView as? PanningImageView else { return }
        imageView.beforeImage = nil
        guard let url else { return }
        // The wipe source only needs a display preview, not a second full-size decode.
        beforeLoadTask = Task { @MainActor [weak self] in
            let image = await ImageThumbnails.load(url, maxPixels: 1024)
            guard !Task.isCancelled, let self, self.beforeURL == url else { return }
            let currentView = self.documentView as? PanningImageView
            currentView?.beforeImage = image
            currentView?.needsDisplay = true
        }
    }
    weak var zoom: ZoomState?

    override func layout() {
        super.layout()
        if zoom?.isFit == true, pendingZoom == nil { MainActor.assumeIsolated { zoom?.fit() } }
    }

    override func viewDidChangeBackingProperties() {
        super.viewDidChangeBackingProperties()
        ensureResolution(for: magnification)
    }

    override func magnify(with event: NSEvent) {
        let base = pendingZoom?.scale ?? magnification
        let point = contentView.convert(event.locationInWindow, from: nil)
        zoom?.set(base * (1 + event.magnification), centeredAt: point)
    }

    /// ⌘-scroll zooms toward the pointer; plain scrolling pans.
    override func scrollWheel(with event: NSEvent) {
        guard event.modifierFlags.contains(.command), let zoom else { return super.scrollWheel(with: event) }
        let delta = event.hasPreciseScrollingDeltas ? event.scrollingDeltaY / 100 : event.scrollingDeltaY / 10
        guard delta != 0 else { return }
        let point = contentView.convert(event.locationInWindow, from: nil)
        MainActor.assumeIsolated { zoom.step(by: pow(2, delta), at: point) }
    }
}

/// Keeps an image smaller than the view centered instead of pinned to the corner.
final class CenteringClipView: NSClipView {
    override func constrainBoundsRect(_ proposedBounds: NSRect) -> NSRect {
        var rect = super.constrainBoundsRect(proposedBounds)
        guard let doc = documentView else { return rect }
        if rect.width > doc.frame.width { rect.origin.x = (doc.frame.width - rect.width) / 2 }
        if rect.height > doc.frame.height { rect.origin.y = (doc.frame.height - rect.height) / 2 }
        return rect
    }
}

/// Drag to pan when zoomed in; double-click toggles between fit and 100% at the clicked point.
final class PanningImageView: NSImageView {
    var beforeImage: NSImage? { didSet { if beforeImage !== oldValue { needsDisplay = true } } }
    var wipe: CGFloat = 0.5
    override func draw(_ dirtyRect: NSRect) {
        super.draw(dirtyRect)
        guard let beforeImage else { return }
        NSGraphicsContext.saveGraphicsState()
        NSBezierPath(rect: NSRect(x: 0, y: 0, width: bounds.width * wipe, height: bounds.height)).addClip()
        beforeImage.draw(in: bounds, from: .zero, operation: .sourceOver, fraction: 1)
        NSGraphicsContext.restoreGraphicsState()
        NSColor.white.setStroke()
        let divider = NSBezierPath()
        divider.move(to: NSPoint(x: bounds.width * wipe, y: 0))
        divider.line(to: NSPoint(x: bounds.width * wipe, y: bounds.height))
        divider.lineWidth = 2
        divider.stroke()
    }
    weak var zoom: ZoomState?
    private var dragStart: NSPoint?
    private var originAtStart: NSPoint?

    override var acceptsFirstResponder: Bool { true }

    private var canPan: Bool {
        guard let clip = enclosingScrollView?.contentView else { return false }
        return clip.bounds.width < frame.width || clip.bounds.height < frame.height
    }

    override func resetCursorRects() {
        if canPan { addCursorRect(visibleRect, cursor: .openHand) }
    }

    override func mouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
        if event.clickCount == 2, let zoom {
            let point = enclosingScrollView?.contentView.convert(event.locationInWindow, from: nil)
            MainActor.assumeIsolated {
                if zoom.isFit || zoom.magnification < 0.99 { zoom.set(1, centeredAt: point) } else { zoom.fit() }
            }
            window?.invalidateCursorRects(for: self)
            return
        }
        guard canPan, let clip = enclosingScrollView?.contentView else { return }
        dragStart = event.locationInWindow
        originAtStart = clip.bounds.origin
        NSCursor.closedHand.push()
    }

    override func mouseDragged(with event: NSEvent) {
        guard let start = dragStart, let origin = originAtStart, let scrollView = enclosingScrollView else { return }
        let scale = scrollView.magnification
        let dx = (event.locationInWindow.x - start.x) / scale
        let dy = (event.locationInWindow.y - start.y) / scale
        let clip = scrollView.contentView
        clip.scroll(to: clip.constrainBoundsRect(NSRect(origin: NSPoint(x: origin.x - dx, y: origin.y - dy),
                                                          size: clip.bounds.size)).origin)
        scrollView.reflectScrolledClipView(clip)
    }

    override func mouseUp(with event: NSEvent) {
        if dragStart != nil { NSCursor.pop() }
        dragStart = nil
        window?.invalidateCursorRects(for: self)
    }

    override func magnify(with event: NSEvent) {
        super.magnify(with: event)
        window?.invalidateCursorRects(for: self)
    }
}

/// − 100% + and Fit, shown over the result.
struct ZoomControls: View {
    @ObservedObject var zoom: ZoomState

    var body: some View {
        HStack(spacing: 2) {
            Button { zoom.zoomOut() } label: { Image(systemName: "minus.magnifyingglass") }
                .help("Zoom out (⌘-)")
            Button { zoom.actualSize() } label: {
                Text(zoom.percentText).monospacedDigit().frame(minWidth: 44)
            }
            .help("Actual size, 100% (⌘0)")
            Button { zoom.zoomIn() } label: { Image(systemName: "plus.magnifyingglass") }
                .help("Zoom in (⌘=). Also: pinch, ⌘-scroll, or double-click")
            Divider().frame(height: 14)
            Button { zoom.fit() } label: { Image(systemName: "arrow.down.right.and.arrow.up.left") }
                .help("Zoom to fit (⌘9)")
                .disabled(zoom.isFit)
        }
        .buttonStyle(.borderless)
        .padding(.horizontal, 8).padding(.vertical, 5)
        .background(.ultraThinMaterial, in: Capsule())
    }
}
