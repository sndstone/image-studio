import AppKit
import SwiftUI
import UniformTypeIdentifiers
import ImageIO

struct CompareSheet: View {
    @EnvironmentObject var model: AppModel
    @Environment(\.dismiss) private var dismiss
    @State private var chosen: Set<String> = []
    @State private var plan: ComparePlan?

    private var candidates: [ModelInfo] { model.models.filter { $0.status.downloaded && !$0.isTool } }
    private var local: [ModelInfo] { candidates.filter { !$0.isCloud } }
    private var cloud: [ModelInfo] { candidates.filter(\.isCloud) }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Compare Models").font(.title3.bold())
            Text("Runs your prompt\(model.references.isEmpty ? "" : ", images") and size on each model, using each model's "
                 + "recommended settings. Results appear side by side.")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            ScrollView {
                VStack(alignment: .leading, spacing: 6) {
                    if !local.isEmpty { group("On this Mac", local) }
                    if !cloud.isEmpty { group("Cloud", cloud) }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxHeight: 340)
            if candidates.count < 2 {
                StudioNote(.warning, "Download a model or add an API key: comparing takes at least two.")
            }
            if chosen.count >= 2, let plan { planView(plan) }
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Compare \(chosen.count) Models") {
                    model.compare(modelIDs: Array(chosen))
                    dismiss()
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .disabled(chosen.count < 2)
            }
        }
        .padding(20)
        .frame(width: 520)
        .onAppear {
            if candidates.contains(where: { $0.id == model.modelID }) { chosen = [model.modelID] }
            if let ids = ProcessInfo.processInfo.environment["IMAGE_STUDIO_TEST_COMPARE_CHOSEN"] { chosen = Set(ids.components(separatedBy: ",")) }  // UI tests
        }
        .task(id: chosen) {
            guard chosen.count >= 2 else { plan = nil; return }
            plan = try? await model.backend.comparePlan(models: Array(chosen), width: model.width, height: model.height, quantize: model.quantize,
                                                        quality: model.openaiQuality, complexity: model.openaiComplexity)
        }
    }

    private func group(_ title: String, _ models: [ModelInfo]) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.subheadline.bold()).padding(.top, 4)
            ForEach(models) { info in
                Toggle(isOn: Binding(get: { chosen.contains(info.id) },
                                     set: { if $0 { chosen.insert(info.id) } else { chosen.remove(info.id) } })) {
                    HStack {
                        Text(info.name)
                        Spacer()
                        if !info.isCloud { Text("\(info.default_steps) steps").font(.caption).foregroundStyle(.secondary) }
                    }
                }
                .toggleStyle(.checkbox)
                .padding(.trailing, 12)
            }
        }
    }

    /// How the chosen models will run: the engine checks memory again when each one starts.
    @ViewBuilder
    private func planView(_ plan: ComparePlan) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            if plan.cloud_models > 0 {
                Label(plan.cloud_models == 1 ? "The cloud model runs first." : "\(plan.cloud_models) cloud models run at the same time, first.", systemImage: "cloud")
            }
            if plan.local_models == 1 {
                Label("The local model runs on this Mac.", systemImage: "desktopcomputer")
            } else if plan.local_models > 1 {
                Label(plan.keep_loaded
                      ? "\(plan.local_models) local models take turns. All fit in free memory, so they stay loaded and switching is instant."
                      : "\(plan.local_models) local models take turns. They don't all fit in free memory, so each unloads before the next loads.",
                      systemImage: "desktopcomputer")
            }
            if plan.cloud_models > 0 {
                Label(costText(plan), systemImage: "dollarsign.circle")
            }
        }
        .font(.caption).foregroundStyle(.secondary)
        .fixedSize(horizontal: false, vertical: true)
    }

    private func costText(_ plan: ComparePlan) -> String {
        if plan.costs_unknown == plan.cloud_models { return "Billed per image; the cost shows after the first run of these models." }
        let more = plan.costs_unknown > 0 ? ", plus \(plan.costs_unknown) model\(plan.costs_unknown == 1 ? "" : "s") not priced yet" : ""
        return "About \(formatUSD(plan.cost_usd))\(more)."
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
                               (geo.size.height - CGFloat(rows) * 48) / CGFloat(rows))
                ScrollView {
                LazyVGrid(columns: Array(repeating: GridItem(.fixed(max(80, side)), spacing: 12), count: columns), spacing: 12) {
                    ForEach(items) { item in
                        VStack(spacing: 4) {
                            Thumbnail(url: item.url, maxPixels: 1024, revision: item, fit: true)
                                .frame(width: max(80, side), height: max(80, side) - 36)
                                .onTapGesture(count: 2) { model.selected = item; model.closeCompare() }
                                .contextMenu { ItemMenu(item: item) }
                                .help("Double-click to open")
                            Text(model.caption(for: item)).font(.caption).lineLimit(2).multilineTextAlignment(.center)
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

