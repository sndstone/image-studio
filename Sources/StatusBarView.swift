import SwiftUI

/// Bottom bar: what the engine is doing, GPU load, and where the memory goes.
struct StatusBarView: View {
    @EnvironmentObject var model: AppModel
    @ObservedObject var monitor: SystemMonitor

    var body: some View {
        HStack(spacing: 16) {
            activity
            Spacer(minLength: 8)
            gpu
            Divider().frame(height: 16)
            engineMemory
            Divider().frame(height: 16)
            systemMemory
        }
        .font(.caption)
        .monospacedDigit()
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(.bar)
    }

    private var activity: some View {
        let s = model.status
        let busy = s.map { $0.phase != "idle" } ?? false
        let text: String = {
            guard let s else { return model.engineUnreachable ? "Engine unreachable" : "Starting…" }
            let name = model.models.first { $0.id == (s.loaded_model ?? "") }?.name
            switch s.phase {
            case "upscaling": return "Upscaling image…"
            case "optimizing": return "Optimizing model…"
            case "loading": return "Loading model into memory…"
            case "converting": return "Converting model for first use · \(s.step)/\(s.total_steps) tensors"
            case "encoding": return "Reading prompt and input images"
            case "denoising":
                return "Generating · step \(s.step)/\(s.total_steps)" + ((s.passes ?? 1) > 1 ? " (2 passes each: guidance)" : "")
            case "decoding": return "Decoding image"
            case "saving": return "Saving"
            default:
                if let downloading = model.models.first(where: { $0.status.downloading }) {
                    return "Downloading \(downloading.name)"
                }
                if s.auto_unloaded == true {
                    return "Idle · model unloaded after \(s.idle_unload_minutes ?? 0) idle min to free memory"
                }
                return name.map { "Idle · \($0) loaded" } ?? "Idle · no model loaded"
            }
        }()
        return HStack(spacing: 6) {
            Circle().fill(busy ? Color.green : Color.secondary.opacity(0.5)).frame(width: 7, height: 7)
            Text(text).foregroundStyle(busy ? .primary : .secondary).lineLimit(1)
        }
    }

    private var gpu: some View {
        HStack(spacing: 6) {
            Text("GPU").foregroundStyle(.secondary)
            Sparkline(values: monitor.gpuHistory, maxValue: 100)
                .frame(width: 70, height: 14)
            Text(monitor.current?.gpu.map { "\(Int($0))%" } ?? "–").frame(width: 34, alignment: .trailing)
        }
        .help("GPU utilization over the last minute (all apps)")
    }

    private var engineMemory: some View {
        let mlx = model.status?.mlx_active ?? model.system?.memory_in_use ?? 0
        let footprint = monitor.current?.backendFootprint
        return HStack(spacing: 4) {
            Text("Model").foregroundStyle(.secondary)
            Text(gb(UInt64(max(0, mlx))))
            if let footprint {
                Text("· engine \(gb(footprint))").foregroundStyle(.secondary)
            }
        }
        .help("Model: GPU memory held by the loaded model and its working buffers. Engine: total memory of Image Studio's image engine process.")
    }

    private var systemMemory: some View {
        let c = monitor.current
        let used = Double(c?.memoryUsed ?? 0), total = Double(c?.memoryTotal ?? 1)
        let color: Color = switch c?.pressure ?? 1 {
        case 4: .red
        case 2: .orange
        default: .green
        }
        return HStack(spacing: 6) {
            Text("Memory").foregroundStyle(.secondary)
            ProgressView(value: min(1, used / total))
                .tint(color)
                .frame(width: 60)
            Text("\(gb(c?.memoryUsed ?? 0)) / \(gb(c?.memoryTotal ?? 0))")
            if let swap = c?.swapUsed, swap > 0 || monitor.swapMBPerSecond > 0 {
                Text(String(format: "· swap %.1f MB/s · %@ total", monitor.swapMBPerSecond, gb(swap))).foregroundStyle(swap > 4 * 1_073_741_824 ? .orange : .secondary)
            }
        }
        .help(pressureHelp(c?.pressure ?? 1))
    }

    private func pressureHelp(_ level: Int) -> String {
        switch level {
        case 4: return "Memory pressure is critical: macOS is swapping heavily and generation will be very slow. Close other apps or unload the model."
        case 2: return "Memory pressure is elevated: generation may slow down. Closing other large apps helps."
        default: return "Memory pressure is normal."
        }
    }

    private func gb(_ bytes: UInt64) -> String {
        String(format: "%.1f GB", Double(bytes) / 1_073_741_824)
    }
}

struct Sparkline: View {
    var values: [Double]
    var maxValue: Double

    var body: some View {
        Canvas { context, size in
            guard values.count > 1 else { return }
            let step = size.width / CGFloat(max(1, 59))
            let offset = size.width - step * CGFloat(values.count - 1)
            var line = Path()
            var fill = Path()
            for (i, v) in values.enumerated() {
                let point = CGPoint(x: offset + CGFloat(i) * step,
                                    y: size.height * (1 - CGFloat(min(v, maxValue) / maxValue)))
                if i == 0 {
                    line.move(to: point)
                    fill.move(to: CGPoint(x: point.x, y: size.height))
                    fill.addLine(to: point)
                } else {
                    line.addLine(to: point)
                    fill.addLine(to: point)
                }
            }
            fill.addLine(to: CGPoint(x: size.width, y: size.height))
            fill.closeSubpath()
            context.fill(fill, with: .color(.accentColor.opacity(0.25)))
            context.stroke(line, with: .color(.accentColor), lineWidth: 1.2)
        }
        .background(RoundedRectangle(cornerRadius: 3).fill(Color.secondary.opacity(0.1)))
    }
}
