import AppKit
import SwiftUI
import UniformTypeIdentifiers
import ImageIO

struct SettingsView: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        TabView(selection: $model.settingsTab) {
            GeneralSettingsView().tabItem { Label("General", systemImage: "gearshape") }.tag("general")
            CloudSettingsView().tabItem { Label("Cloud", systemImage: "cloud") }.tag("cloud")
            HuggingFaceSettingsView().tabItem { Label("Hugging Face", systemImage: "key") }.tag("huggingface")
        }
        .scenePadding(.minimum)
    }
}

/// A caption under a settings section: the explanation goes here so the controls stay one line each.
private struct SettingsNote: View {
    var text: String
    init(_ text: String) { self.text = text }
    var body: some View {
        Text(text).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
    }
}

struct GeneralSettingsView: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        Form {
            Section {
                Picker("Unload idle model after", selection: $model.idleUnloadMinutes) {
                    Text("5 minutes").tag(5)
                    Text("10 minutes").tag(10)
                    Text("15 minutes").tag(15)
                    Text("30 minutes").tag(30)
                    Text("1 hour").tag(60)
                    Text("Never").tag(0)
                }
                Picker("Low-memory mode", selection: $model.lowMemory) {
                    Text("Automatic").tag("auto")
                    Text("Always").tag("always")
                    Text("Never").tag("never")
                }
            } header: {
                Text("Memory")
            } footer: {
                SettingsNote(loadedModelMemory + " Unloading gives that memory back to other apps; the next image loads the model again first.")
            }

            Section {
                Toggle("Live preview", isOn: $model.livePreview)
                if let note = model.previewNote { Text(note).font(.caption).foregroundStyle(.orange) }
            } header: {
                Text("While generating")
            } footer: {
                SettingsNote("Updates every few seconds so you can cancel a run going the wrong way. Costs a few percent of the time. Local models only.")
            }

            Section {
                Toggle("Send a notification", isOn: $model.notifyWhenFinished)
                if model.notifyWhenFinished && model.notificationsBlocked {
                    HStack {
                        Text("Notifications are off for Image Studio.").font(.caption).foregroundStyle(.orange)
                        Spacer()
                        Button("Open System Settings…") {
                            if let url = URL(string: "x-apple.systempreferences:com.apple.Notifications-Settings.extension") { NSWorkspace.shared.open(url) }
                        }
                    }
                }
            } header: {
                Text("When finished")
            } footer: {
                SettingsNote("Only while you're in another app. A queue sends one notification at the end, and it says if any job failed. Prompts aren't shown in it.")
            }

            Section {
                Toggle("Embed prompt and settings", isOn: $model.embedMetadata)
            } header: {
                Text("Saved images")
            } footer: {
                SettingsNote("Writes them into each PNG, so they travel with the file. They're always kept beside the image in a settings file.")
            }

            Section {
                Toggle("Offline mode", isOn: $model.offline)
            } header: {
                Text("Network")
            } footer: {
                SettingsNote("Stops Hugging Face downloads and cloud models. Local models you've already downloaded keep working.")
            }
        }
        .formStyle(.grouped)
        .frame(width: 520, height: 590)
        .task { await model.refreshNotificationStatus() }
    }

    /// The engine's live figure rather than a fixed range, since it differs per model and precision.
    private var loadedModelMemory: String {
        let inUse = model.status?.mlx_active ?? model.system?.memory_in_use ?? 0
        return inUse > 0 ? "The loaded model holds \(formatBytes(inUse)) right now."
            : "A loaded model holds most of the memory shown for it in the Model Manager."
    }
}

struct HuggingFaceSettingsView: View {
    var body: some View {
        Form {
            Section {
                TokenView()
            } header: {
                Text("API key")
            } footer: {
                SettingsNote("Needed only to download gated models such as klein 9B.")
            }
        }
        .formStyle(.grouped)
        .frame(width: 520, height: 300)
    }
}
