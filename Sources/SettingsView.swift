import AppKit
import SwiftUI
import UniformTypeIdentifiers
import ImageIO

struct SettingsView: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        Form {
            Picker("Free the model's memory after", selection: $model.idleUnloadMinutes) {
                Text("5 idle minutes").tag(5)
                Text("10 idle minutes").tag(10)
                Text("15 idle minutes").tag(15)
                Text("30 idle minutes").tag(30)
                Text("1 idle hour").tag(60)
                Text("Never").tag(0)
            }
            Text(loadedModelMemory + " Unloading gives that memory back to other apps; the next "
                 + "generation loads it again first (a few seconds to half a minute).")
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Toggle("Embed prompt and settings in saved images", isOn: $model.embedMetadata)
            Picker("Low-memory mode", selection: $model.lowMemory) {
                Text("Automatic").tag("auto")
                Text("Always").tag("always")
                Text("Never").tag("never")
            }
            Toggle("Offline mode", isOn: $model.offline)
            if let note = model.previewNote { Text(note).font(.caption).foregroundStyle(.orange) }
            Toggle("Show a live preview while generating", isOn: $model.livePreview)
            Text("Updates every few seconds so you can cancel a run that's going the wrong way. Previews are "
                 + "low resolution and cost a few percent of the generation time.")
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(20)
        .frame(width: 440)
    }

    /// The engine's live figure rather than a fixed range, since it differs per model and precision.
    private var loadedModelMemory: String {
        let inUse = model.status?.mlx_active ?? model.system?.memory_in_use ?? 0
        return inUse > 0 ? "The loaded model holds \(formatBytes(inUse)) right now."
            : "A loaded model holds most of the memory shown for it in the Model Manager."
    }
}

