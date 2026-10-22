import AppKit
import ImageIO
import SwiftUI
import UniformTypeIdentifiers

struct ResultInspector: View {
    var item: HistoryItem
    @State private var json: [String: Any] = [:]
    private var all: String {
        guard let data = try? JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted, .sortedKeys]) else { return "No saved settings." }
        return String(data: data, encoding: .utf8) ?? ""
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Image Settings").font(.headline)
            ScrollView { Text(all).font(.system(.caption, design: .monospaced)).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading) }
            HStack {
                Button("Copy Prompt") { copy(json["prompt"] as? String ?? "") }
                Button("Copy All") { copy(all) }
            }
        }.padding(16).frame(width: 440, height: 400)
        .task(id: item) {
            let url = item.url
            let settings = await Task.detached(priority: .utility) { AppModel.settingsJSON(at: url) ?? [:] }.value
            guard !Task.isCancelled else { return }
            json = settings
        }
    }
    private func copy(_ text: String) { NSPasteboard.general.clearContents(); NSPasteboard.general.setString(text, forType: .string) }
}

struct ExportView: View {
    var item: HistoryItem
    @Environment(\.dismiss) var dismiss
    @State private var format = UTType.png.identifier
    @State private var removeMetadata = true
    @State private var isExporting = false
    @State private var error: String?
    private var formats: [(String, String)] {
        var values = [("PNG", UTType.png.identifier), ("JPEG", UTType.jpeg.identifier), ("HEIC", UTType.heic.identifier)]
        let supported = CGImageDestinationCopyTypeIdentifiers() as! [String]
        if supported.contains(UTType.webP.identifier) { values.append(("WebP", UTType.webP.identifier)) }
        return values
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Export Image").font(.headline)
            Picker("Format", selection: $format) { ForEach(formats, id: \.1) { Text($0.0).tag($0.1) } }
            Toggle("Remove metadata", isOn: $removeMetadata)
            if let error { Text(error).font(.caption).foregroundStyle(.red) }
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Export…") { export() }.disabled(isExporting).keyboardShortcut(.defaultAction)
            }
        }.padding(20).frame(width: 350)
    }
    private func export() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [UTType(format) ?? .png]
        panel.nameFieldStringValue = item.url.deletingPathExtension().lastPathComponent
        guard panel.runModal() == .OK, let url = panel.url else { return }
        guard url.standardizedFileURL != item.url.standardizedFileURL else { error = "Choose a new filename to preserve the original."; return }
        isExporting = true
        let input = item.url
        let outputFormat = format
        let stripMetadata = removeMetadata
        Task {
            defer { isExporting = false }
            let failure = await Task.detached(priority: .userInitiated) {
                ImageExports.write(input: input, output: url, format: outputFormat, removeMetadata: stripMetadata)
            }.value
            if let failure { error = failure } else { dismiss() }
        }
    }
}

struct FullResultView: View {
    @EnvironmentObject var model: AppModel
    @StateObject private var zoom = ZoomState()
    @Environment(\.dismiss) var dismiss
    var body: some View {
        VStack(spacing: 0) {
            if let pending = model.selectedPendingItem {
                PendingResultView(item: pending)
            } else if let item = model.selected {
                ZoomableImage(url: item.url, revision: item, zoom: zoom)
                    .overlay(alignment: .bottomTrailing) { ZoomControls(zoom: zoom).padding() }
            }
            if let error = model.errorMessage { Text(error).font(.caption).foregroundStyle(.red).padding(.horizontal) }
            HStack {
                Button { model.selectHistory(offset: -1) } label: { Image(systemName: "chevron.left") }.keyboardShortcut(.leftArrow, modifiers: [])
                Button { model.selectHistory(offset: 1) } label: { Image(systemName: "chevron.right") }.keyboardShortcut(.rightArrow, modifiers: [])
                if let item = model.selected { Button("Move to Trash") { model.delete(item) }.keyboardShortcut(.delete, modifiers: .command) }
                Spacer()
                Button("Close") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Close Full View") { dismiss() }.keyboardShortcut(.space, modifiers: []).hidden().frame(width: 0)
            }.padding(10)
        }.frame(minWidth: 800, minHeight: 600)
    }
}
