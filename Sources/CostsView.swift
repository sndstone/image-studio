import AppKit
import Charts
import SwiftUI

func formatUSD(_ value: Double) -> String {
    if value == 0 { return "$0.00" }
    if value < 0.01 { return String(format: "$%.4f", value) }
    if value < 1 { return String(format: "$%.3f", value) }
    return String(format: "$%.2f", value)
}

struct CostEntry: Decodable, Identifiable {
    var time: Double
    var provider: String
    var model: String
    var kind: String?
    var cost_usd: Double?
    var cost_basis: String?
    var prompt: String?
    var width: Int?
    var height: Int?
    var path: String?
    var id: String { "\(time)|\(model)|\(path ?? "")" }
}

struct CostModelRow: Decodable, Identifiable {
    var provider: String
    var model: String
    var images: Int
    var cost_usd: Double
    var id: String { provider + "|" + model }
}

struct CostDayRow: Decodable, Identifiable {
    var day: String
    var provider: String
    var cost_usd: Double
    var id: String { day + "|" + provider }
}

struct CostPriceRow: Decodable, Identifiable {
    var provider: String
    var model: String
    var price: String
    var id: String { provider + "|" + model }
}

struct CostSummary: Decodable {
    var today_usd: Double
    var month_usd: Double
    var all_usd: Double
    var images: Int
    var unknown_cost_images: Int
    var by_provider: [String: Double]
    var by_model: [CostModelRow]
    var by_day: [CostDayRow]
    var days: [String]
    var entries: [CostEntry]
    var prices: [CostPriceRow]
    var prices_checked: String
}

/// One color per provider everywhere (validated for color blindness, light and dark; aqua on light needs the labels).
private func providerColor(_ provider: String) -> Color {
    let (light, dark): (UInt32, UInt32)
    switch provider {
    case "google": (light, dark) = (0x2a78d6, 0x3987e5)
    case "openai": (light, dark) = (0xeb6834, 0xd95926)
    default: (light, dark) = (0x1baf7a, 0x199e70)
    }
    return Color(nsColor: NSColor(name: nil) { appearance in
        let hex = appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua ? dark : light
        return NSColor(srgbRed: CGFloat(hex >> 16 & 255) / 255, green: CGFloat(hex >> 8 & 255) / 255, blue: CGFloat(hex & 255) / 255, alpha: 1)
    })
}

private func providerName(_ id: String) -> String { cloudProviders.first { $0.id == id }?.name ?? id }
private let providerOrder = ["google", "openai", "bfl"]

private let dayParser: DateFormatter = {
    let formatter = DateFormatter()
    formatter.calendar = Calendar(identifier: .gregorian)
    formatter.locale = Locale(identifier: "en_US_POSIX")
    formatter.dateFormat = "yyyy-MM-dd"
    return formatter
}()

struct CostsView: View {
    @EnvironmentObject var model: AppModel
    @Environment(\.dismiss) private var dismiss
    @State private var summary: CostSummary?
    @State private var failure: String?
    @State private var tab = "dashboard"
    @State private var credits: Double?

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Costs").font(.title2.bold())
                Spacer()
                Picker("View", selection: $tab) {
                    Text("Dashboard").tag("dashboard")
                    Text("Images").tag("images")
                    Text("Prices").tag("prices")
                }
                .pickerStyle(.segmented).labelsHidden().frame(width: 280)
                Spacer()
                Button("Done") { dismiss() }.keyboardShortcut(.defaultAction)
            }
            .padding(16)
            Divider()
            Group {
                if let summary {
                    switch tab {
                    case "images": CostImagesTab(summary: summary)
                    case "prices": CostPricesTab(summary: summary)
                    default: CostDashboardTab(summary: summary, credits: credits)
                    }
                } else if let failure {
                    StudioNote(.error, failure).padding(20)
                    Spacer()
                } else {
                    ProgressView().frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .frame(minWidth: 780, idealWidth: 860, minHeight: 600, idealHeight: 720)
        .onAppear { if let view = ProcessInfo.processInfo.environment["IMAGE_STUDIO_TEST_COSTS_TAB"] { tab = view } }  // UI tests
        .task(id: model.history.count) {
            do { summary = try await model.backend.cloudCosts(); failure = nil }
            catch { failure = "Could not read the costs: " + error.localizedDescription }
        }
        .task {
            guard model.cloudKeys["bfl"] != nil, !model.offline else { return }
            credits = (try? await model.backend.testCloudKey(provider: "bfl"))?.credits
        }
    }
}

private struct StatTile: View {
    var title: String
    var value: String
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            Text(value).font(.title2.weight(.semibold)).monospacedDigit().lineLimit(1).minimumScaleFactor(0.7)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 10).fill(Color(nsColor: .controlBackgroundColor)))
    }
}

private struct CostDashboardTab: View {
    @EnvironmentObject var model: AppModel
    var summary: CostSummary
    var credits: Double?
    @State private var hovered: Date?
    @State private var showTable = false

    /// Only what the chart shows: `by_provider` also has anything older than these 30 days.
    private var periodTotals: [String: Double] {
        summary.by_day.reduce(into: [:]) { totals, row in totals[row.provider, default: 0] += row.cost_usd }
    }
    private var providers: [String] { providerOrder.filter { periodTotals[$0] != nil } }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                HStack(spacing: 10) {
                    StatTile(title: "Today", value: formatUSD(summary.today_usd))
                    StatTile(title: "This month", value: formatUSD(summary.month_usd))
                    StatTile(title: "All time", value: formatUSD(summary.all_usd))
                    StatTile(title: "Cloud images", value: summary.images.formatted())
                    if let credits { StatTile(title: "BFL balance", value: formatUSD(credits * 0.01)) }
                }
                if summary.images == 0 {
                    Text("No cloud images in your library. What each one costs will show up here.")
                        .foregroundStyle(.secondary).frame(maxWidth: .infinity).padding(.vertical, 40)
                } else {
                    dailyChart
                    modelTable
                }
                Text("These totals cover the images still in your library: moving one to the Trash removes its cost from here. Deleted images, test images and requests cancelled or failed in flight may still have been charged, so only the provider's bill shows what you spent.")
                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                if summary.unknown_cost_images > 0 {
                    StudioNote(.warning, "\(summary.unknown_cost_images) image\(summary.unknown_cost_images == 1 ? " has" : "s have") no known cost, so the totals are too low: the provider's response didn't say.")
                }
            }
            .padding(16)
        }
    }

    private var dailyChart: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline) {
                Text("Last 30 days").font(.headline)
                Spacer()
                ForEach(providers, id: \.self) { provider in
                    HStack(spacing: 4) {
                        RoundedRectangle(cornerRadius: 2).fill(providerColor(provider)).frame(width: 10, height: 10)
                        Text("\(providerName(provider)) \(formatUSD(periodTotals[provider] ?? 0))")
                            .font(.caption).monospacedDigit().foregroundStyle(.secondary)
                    }
                }
                Toggle("Table", isOn: $showTable).toggleStyle(.checkbox).controlSize(.small)
            }
            if showTable { dayTable } else {
                Chart(summary.by_day) { row in
                    BarMark(x: .value("Day", dayParser.date(from: row.day) ?? .distantPast, unit: .day),
                            y: .value("Cost", row.cost_usd), width: .ratio(0.7))
                        .foregroundStyle(by: .value("Provider", row.provider))
                        .cornerRadius(2)
                    if let hovered, Calendar.current.isDate(hovered, inSameDayAs: dayParser.date(from: row.day) ?? .distantPast) {
                        RuleMark(x: .value("Day", hovered, unit: .day))
                            .foregroundStyle(Color.secondary.opacity(0.12))
                            .annotation(position: .top, overflowResolution: .init(x: .fit, y: .disabled)) { tooltip(hovered) }
                    }
                }
                .chartForegroundStyleScale(domain: providerOrder, range: providerOrder.map(providerColor))
                .chartLegend(.hidden)
                .chartXScale(domain: chartDomain)
                .chartXSelection(value: $hovered)
                .chartXAxis {
                    AxisMarks(values: .stride(by: .day, count: 7)) { _ in
                        AxisTick().foregroundStyle(.quaternary)
                        AxisValueLabel(format: .dateTime.month(.abbreviated).day())
                    }
                }
                .chartYAxis {
                    AxisMarks(position: .leading) { value in
                        AxisGridLine().foregroundStyle(.quaternary)
                        AxisValueLabel { if let amount = value.as(Double.self) { Text(formatUSD(amount)) } }
                    }
                }
                .frame(height: 220)
                .accessibilityLabel("Spending per day for the last 30 days, by provider. Turn on Table to read the figures.")
            }
        }
    }

    private var chartDomain: ClosedRange<Date> {
        let first = summary.days.first.flatMap(dayParser.date) ?? Date()
        let last = summary.days.last.flatMap(dayParser.date) ?? Date()
        return first...(Calendar.current.date(byAdding: .day, value: 1, to: last) ?? last)
    }

    private func tooltip(_ date: Date) -> some View {
        let rows = summary.by_day.filter { Calendar.current.isDate(dayParser.date(from: $0.day) ?? .distantPast, inSameDayAs: date) }
        return VStack(alignment: .leading, spacing: 2) {
            Text(date, format: .dateTime.month(.abbreviated).day()).font(.caption.bold())
            ForEach(rows) { row in
                Text("\(providerName(row.provider)) \(formatUSD(row.cost_usd))").font(.caption).monospacedDigit()
            }
        }
        .padding(6)
        .background(RoundedRectangle(cornerRadius: 6).fill(Color(nsColor: .windowBackgroundColor)).shadow(radius: 2))
    }

    private var dayTable: some View {
        Grid(alignment: .leading, horizontalSpacing: 20, verticalSpacing: 4) {
            GridRow { Text("Day"); Text("Provider"); Text("Cost").gridColumnAlignment(.trailing) }
                .font(.caption.bold()).foregroundStyle(.secondary)
            ForEach(summary.by_day.reversed()) { row in
                GridRow {
                    Text(row.day)
                    Text(providerName(row.provider))
                    Text(formatUSD(row.cost_usd))
                }
                .font(.caption).monospacedDigit()
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var modelTable: some View {
        let top = summary.by_model.map(\.cost_usd).max() ?? 0
        return VStack(alignment: .leading, spacing: 8) {
            Text("By model").font(.headline)
            Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 7) {
                GridRow {
                    Text("Model")
                    Text("Provider")
                    Text("Images").gridColumnAlignment(.trailing)
                    Text("Average").gridColumnAlignment(.trailing)
                    Text("Total").gridColumnAlignment(.trailing)
                    Color.clear.frame(height: 1)
                }
                .font(.caption.bold()).foregroundStyle(.secondary)
                ForEach(summary.by_model) { row in
                    GridRow {
                        Text(modelName(row.model)).lineLimit(1)
                        Text(providerName(row.provider)).foregroundStyle(.secondary).lineLimit(1)
                        Text(row.images.formatted())
                        Text(row.images > 0 ? formatUSD(row.cost_usd / Double(row.images)) : "–")
                        Text(formatUSD(row.cost_usd))
                        GeometryReader { geo in
                            RoundedRectangle(cornerRadius: 3).fill(providerColor(row.provider))
                                .frame(width: top > 0 ? max(2, geo.size.width * row.cost_usd / top) : 0, height: 8)
                                .frame(maxHeight: .infinity, alignment: .center)
                        }
                        .frame(minWidth: 120, maxWidth: .infinity, minHeight: 14, maxHeight: 14)
                        .accessibilityHidden(true)
                    }
                    .font(.callout).monospacedDigit()
                }
            }
        }
    }

    private func modelName(_ cloudModel: String) -> String {
        model.models.first { $0.id == "cloud-" + cloudModel }?.name ?? cloudModel
    }
}

private struct CostImagesTab: View {
    @EnvironmentObject var model: AppModel
    var summary: CostSummary

    var body: some View {
        if summary.entries.isEmpty {
            Text("No cloud images yet.").foregroundStyle(.secondary).frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            List(summary.entries) { entry in row(entry) }
        }
    }

    private func row(_ entry: CostEntry) -> some View {
        let url = entry.path.map { URL(fileURLWithPath: $0) }
        let exists = url.map { FileManager.default.fileExists(atPath: $0.path) } ?? false
        return HStack(spacing: 12) {
            Group {
                if let url, exists {
                    Thumbnail(url: url, maxPixels: 128)
                } else {
                    Image(systemName: "photo").foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            .frame(width: 44, height: 44)
            VStack(alignment: .leading, spacing: 2) {
                Text(entry.prompt ?? "").lineLimit(1)
                Text(details(entry)).font(.caption).foregroundStyle(.secondary).lineLimit(1)
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 2) {
                Text(entry.cost_usd.map(formatUSD) ?? "Unknown").monospacedDigit().fontWeight(.medium)
                Text(basisLabel(entry)).font(.caption).foregroundStyle(.secondary)
            }
        }
        .contextMenu {
            if let url, exists {
                Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([url]) }
            }
        }
    }

    private func details(_ entry: CostEntry) -> String {
        var parts = [model.models.first { $0.id == "cloud-" + entry.model }?.name ?? entry.model]
        if let width = entry.width, let height = entry.height { parts.append("\(width)×\(height)") }
        parts.append(Date(timeIntervalSince1970: entry.time).formatted(date: .abbreviated, time: .shortened))
        return parts.joined(separator: " · ")
    }

    /// Where the figure came from, since the providers don't all report one.
    private func basisLabel(_ entry: CostEntry) -> String {
        switch entry.cost_basis {
        case "reported": return "reported by \(providerName(entry.provider))"
        case "computed": return "from tokens used"
        case "list": return "list price"
        default: return "not reported"
        }
    }
}

private struct CostPricesTab: View {
    @EnvironmentObject var model: AppModel
    var summary: CostSummary

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 8) {
                    GridRow {
                        Text("Provider")
                        Text("Model")
                        Text("Price")
                    }
                    .font(.caption.bold()).foregroundStyle(.secondary)
                    ForEach(summary.prices) { row in
                        GridRow {
                            Text(providerName(row.provider))
                            Text(model.models.first { $0.id == "cloud-" + row.model }?.name ?? row.model)
                            Text(row.price).fixedSize(horizontal: false, vertical: true)
                        }
                        .font(.callout)
                    }
                }
                Text("Checked on \(summary.prices_checked); prices change, and these figures only update with the app. Reference images can add to the cost; what each image really cost is under Images.")
                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                HStack(spacing: 14) {
                    Link("Google pricing", destination: URL(string: "https://ai.google.dev/gemini-api/docs/pricing")!)
                    Link("OpenAI pricing", destination: URL(string: "https://developers.openai.com/api/docs/pricing")!)
                    Link("BFL pricing", destination: URL(string: "https://bfl.ai/pricing")!)
                }
                .font(.caption)
            }
            .padding(16)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}
