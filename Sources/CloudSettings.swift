import AppKit
import SwiftUI

struct CloudProvider: Identifiable {
    var id: String
    var name: String
    var models: String
    var placeholder: String
    var keyPage: URL
}

let cloudProviders = [
    CloudProvider(id: "google", name: "Google", models: "Nano Banana", placeholder: "AIza…",
                  keyPage: URL(string: "https://aistudio.google.com/apikey")!),
    CloudProvider(id: "openai", name: "OpenAI", models: "GPT Image", placeholder: "sk-…",
                  keyPage: URL(string: "https://platform.openai.com/api-keys")!),
    CloudProvider(id: "bfl", name: "Black Forest Labs", models: "FLUX API", placeholder: "API key",
                  keyPage: URL(string: "https://dashboard.bfl.ai")!),
]

func cloudProviderName(_ info: ModelInfo) -> String {
    cloudProviders.first { $0.id == info.cloud_provider }?.name ?? "the provider"
}

extension AppModel {
    func openCloudSettings() {
        NSApp.activate(ignoringOtherApps: true)
        openSettingsWindow?()
        // A tab selected before the window exists is ignored on first open.
        Task {
            try? await Task.sleep(for: .milliseconds(150))
            settingsTab = "cloud"
        }
    }

    /// Off the main thread for the same reason as the Hugging Face token: macOS may ask for Keychain permission.
    func loadCloudKeys() async {
        if let key = ProcessInfo.processInfo.environment["IMAGE_STUDIO_TEST_CLOUD_KEY"] {  // UI tests: not saved
            cloudKeys = Dictionary(uniqueKeysWithValues: cloudProviders.map { ($0.id, key) })
            cloudDrafts = cloudKeys
            return
        }
        guard ProcessInfo.processInfo.environment["IMAGE_STUDIO_SKIP_KEYCHAIN"] == nil else { return }
        let saved = await Task.detached { Keychain.read(account: Keychain.cloudAccount) }.value
        guard let data = saved?.data(using: .utf8),
              let keys = try? JSONDecoder().decode([String: String].self, from: data) else { return }
        cloudKeys = keys
        cloudDrafts = keys
    }

    /// The engine keeps keys only in memory, so every start hands them over again.
    func pushCloudKeys() async {
        for (provider, key) in cloudKeys { try? await backend.setCloudKey(provider: provider, key: key) }
    }

    /// Applies a change to what's in the Keychain now: the item holds every provider's key, and `cloudKeys`
    /// is empty if reading it failed at launch, so writing that back would erase the others.
    private func updateStoredCloudKeys(_ change: (inout [String: String]) -> Void) throws -> [String: String] {
        var keys: [String: String] = [:]
        if let stored = try Keychain.lookup(account: Keychain.cloudAccount) {
            keys = try JSONDecoder().decode([String: String].self, from: Data(stored.utf8))
        }
        change(&keys)
        if keys.isEmpty {
            try Keychain.delete(account: Keychain.cloudAccount)
        } else {
            let json = String(decoding: try JSONEncoder().encode(keys), as: UTF8.self)
            try Keychain.save(json, account: Keychain.cloudAccount)
        }
        return keys
    }

    func saveCloudKey(_ provider: String) {
        let key = (cloudDrafts[provider] ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        // Checked before saving: the Keychain and the engine must not end up with different keys.
        guard !key.isEmpty, !cloudBusy.contains(provider) else { return }
        do { cloudKeys = try updateStoredCloudKeys { $0[provider] = key } } catch {
            cloudResults[provider] = CloudTestResult(ok: false, message: "Could not save the key: " + error.localizedDescription)
            return
        }
        cloudDrafts[provider] = key
        runCloudTest(provider) {
            try await self.backend.setCloudKey(provider: provider, key: key)
            await self.refreshModels()
            return try await self.backend.testCloudKey(provider: provider)
        }
    }

    func removeCloudKey(_ provider: String) {
        guard !cloudBusy.contains(provider) else { return }
        do { cloudKeys = try updateStoredCloudKeys { $0[provider] = nil } } catch {
            cloudResults[provider] = CloudTestResult(ok: false, message: "Could not remove the key: " + error.localizedDescription)
            return
        }
        cloudDrafts[provider] = nil
        cloudResults[provider] = nil
        Task {
            try? await backend.setCloudKey(provider: provider, key: nil)
            await refreshModels()
        }
    }

    func testCloudImage(_ provider: String) {
        runCloudTest(provider) { try await self.backend.testCloudImage(provider: provider) }
    }

    private func runCloudTest(_ provider: String, _ work: @escaping () async throws -> CloudTestResult) {
        guard !cloudBusy.contains(provider) else { return }
        cloudBusy.insert(provider)
        cloudResults[provider] = nil
        Task {
            defer { cloudBusy.remove(provider) }
            do { cloudResults[provider] = try await work() } catch {
                cloudResults[provider] = CloudTestResult(ok: false, message: error.localizedDescription)
            }
        }
    }
}

struct CloudSettingsView: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        Form {
            Section {
                if model.offline { StudioNote(.warning, "Offline mode is on, so cloud models and tests won't run.") }
            } footer: {
                Text("Cloud models run on the provider's servers: your prompt and reference images are sent there, and your account is billed per image. Keys stay in your macOS Keychain and only go to their own provider.")
                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            ForEach(cloudProviders) { provider in
                Section { CloudProviderRow(provider: provider) } header: {
                    HStack(alignment: .firstTextBaseline) {
                        Text(provider.name)
                        Text(provider.models).font(.caption).foregroundStyle(.secondary)
                        Spacer()
                        Link("Get a key", destination: provider.keyPage).font(.caption)
                    }
                }
            }
        }
        .formStyle(.grouped)
        .frame(width: 520, height: 560)
    }
}

struct CloudProviderRow: View {
    @EnvironmentObject var model: AppModel
    var provider: CloudProvider
    @State private var reveal = false

    private var draft: Binding<String> {
        Binding(get: { model.cloudDrafts[provider.id] ?? "" }, set: { model.cloudDrafts[provider.id] = $0 })
    }
    private var trimmed: String { draft.wrappedValue.trimmingCharacters(in: .whitespacesAndNewlines) }
    private var saved: String? { model.cloudKeys[provider.id] }
    private var busy: Bool { model.cloudBusy.contains(provider.id) }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Group {
                    if reveal {
                        TextField("API key", text: draft, prompt: Text(provider.placeholder))
                    } else {
                        SecureField("API key", text: draft, prompt: Text(provider.placeholder))
                    }
                }
                .textFieldStyle(.roundedBorder)
                .font(.system(.body, design: .monospaced))
                .accessibilityLabel("\(provider.name) API key")
                .onSubmit { model.saveCloudKey(provider.id) }
                Button { reveal.toggle() } label: { Image(systemName: reveal ? "eye.slash" : "eye") }
                    .buttonStyle(.borderless)
                    .help(reveal ? "Hide key" : "Show key")
                    .accessibilityLabel(reveal ? "Hide key" : "Show key")
            }
            HStack {
                Button("Save & Test") { model.saveCloudKey(provider.id) }
                    .disabled(trimmed.isEmpty || busy)
                    .help("Saves the key and checks it with a free call")
                Button("Test Image (billed)") { model.testCloudImage(provider.id) }
                    .disabled(saved == nil || saved != trimmed || busy)
                    .help("Generates one small image with the provider's cheapest model. It's charged to your account, and it's the only check that also catches missing billing, quota or account verification.")
                Button("Remove", role: .destructive) { model.removeCloudKey(provider.id) }
                    .disabled((saved == nil && trimmed.isEmpty) || busy)
                if busy { ProgressView().controlSize(.small) }
            }
            .controlSize(.small)
            if let result = model.cloudResults[provider.id] {
                HStack(alignment: .top, spacing: 8) {
                    if let thumbnail = result.thumbnail, let data = Data(base64Encoded: thumbnail), let image = NSImage(data: data) {
                        Image(nsImage: image).resizable().scaledToFit().frame(width: 64, height: 64)
                            .clipShape(RoundedRectangle(cornerRadius: 4))
                            .accessibilityLabel("Test image")
                    }
                    Label(result.message, systemImage: result.ok ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                        .font(.caption)
                        .foregroundStyle(result.ok ? .green : .orange)
                        .fixedSize(horizontal: false, vertical: true)
                        .textSelection(.enabled)
                }
            }
        }
    }
}
