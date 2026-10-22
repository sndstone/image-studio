import AppKit
import SwiftUI
import UniformTypeIdentifiers
import ImageIO

struct CompareSheet: View {
    @EnvironmentObject var model: AppModel
    @Environment(\.dismiss) private var dismiss
    @State private var chosen: Set<String> = []

    private var candidates: [ModelInfo] { model.models.filter { $0.status.downloaded && !$0.isTool } }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Compare Models").font(.title3.bold())
            Text("Runs your prompt\(model.references.isEmpty ? "" : ", images") and size on each model with the same seed, "
                 + "using each model's recommended steps and guidance. Results appear side by side.")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            ForEach(candidates) { info in
                Toggle(isOn: Binding(get: { chosen.contains(info.id) },
                                     set: { if $0 { chosen.insert(info.id) } else { chosen.remove(info.id) } })) {
                    HStack {
                        Text(info.name)
                        Spacer()
                        Text("\(info.default_steps) steps").font(.caption).foregroundStyle(.secondary)
                    }
                }
                .toggleStyle(.checkbox)
            }
            if candidates.count < 2 {
                Text("Download at least two models to compare them.").font(.caption).foregroundStyle(.orange)
            }
            HStack {
                Text("Models run one after another; switching models adds loading time.").font(.caption).foregroundStyle(.secondary)
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Compare \(chosen.count) Models") {
                    model.compare(modelIDs: Array(chosen))
                    dismiss()
                }
                .buttonStyle(.borderedProminent)
                .disabled(chosen.count < 2)
            }
        }
        .padding(20)
        .frame(width: 480)
        .onAppear { chosen = [model.modelID] }
    }
}

/// Results side by side, each captioned with its model, steps and time.
struct CompareGrid: View {
    @EnvironmentObject var model: AppModel

    @State private var page = 0
    private var pages: Int { max(1, (model.compareItems.count + 8) / 9) }
    private var items: [HistoryItem] { Array(model.compareItems.dropFirst(min(page, pages - 1) * 9).prefix(9)) }

    var body: some View {
        VStack(spacing: 8) {
            HStack {
                Text("Comparing \(model.compareItems.count)").font(.headline)
                if let job = model.currentJob, job.compareGroup != nil || !model.queue.filter({ $0.compareGroup != nil }).isEmpty {
                    Text("· more on the way").foregroundStyle(.secondary)
                }
                Spacer()
                Button { page = max(0, page - 1) } label: { Image(systemName: "chevron.left") }.disabled(page == 0)
                Text("\(min(page, pages - 1) + 1) / \(pages)").font(.caption)
                Button { page = min(pages - 1, page + 1) } label: { Image(systemName: "chevron.right") }.disabled(page >= pages - 1)
                Button("Close") { model.closeCompare() }.keyboardShortcut(.cancelAction)
            }
            .padding(.horizontal, 20).padding(.top, 14)
            GeometryReader { geo in
                let n = max(1, items.count)
                let columns = n <= 3 ? n : Int(Double(n).squareRoot().rounded(.up))
                let rows = Int((Double(n) / Double(columns)).rounded(.up))
                let side = min((geo.size.width - CGFloat(columns + 1) * 12) / CGFloat(columns),
                               (geo.size.height - CGFloat(rows) * 34) / CGFloat(rows))
                ScrollView {
                LazyVGrid(columns: Array(repeating: GridItem(.fixed(max(80, side)), spacing: 12), count: columns), spacing: 12) {
                    ForEach(items) { item in
                        VStack(spacing: 4) {
                            Thumbnail(url: item.url, maxPixels: 1024, revision: item, fit: true)
                                .frame(width: max(80, side), height: max(80, side) - 22)
                                .onTapGesture(count: 2) { model.selected = item; model.closeCompare() }
                                .contextMenu { ItemMenu(item: item) }
                                .help("Double-click to open")
                            Text(model.caption(for: item)).font(.caption).lineLimit(1)
                        }
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            .padding(.bottom, 10)
        }
    }
}

