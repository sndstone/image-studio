import AppKit
import SwiftUI
import UniformTypeIdentifiers

@main
struct ImageStudioApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var delegate
    @StateObject private var model = AppModel()

    var body: some Scene {
        Window("Image Studio", id: "main") {
            ContentView()
                .environmentObject(model)
                .frame(minWidth: 1000, minHeight: 680)
                .task {
                    delegate.model = model
                    await model.boot()
                }
        }
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("Generate") { model.generate() }
                    .keyboardShortcut(.return, modifiers: .command)
                    .disabled(model.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || model.phase != .ready)
                Button("Upscale Image File…") { model.pickUpscaleFile() }
                Button("Add Reference Images…") { model.pickReferences() }
                    .keyboardShortcut("o", modifiers: .command)
                Button("Model Manager…") { model.showModelManager = true }
                    .keyboardShortcut("m", modifiers: [.command, .shift])
                Divider()
                Button("Clear History…") { model.requestClearHistory() }
                    .disabled(model.history.isEmpty || model.isGenerating)
            }
            CommandGroup(after: .toolbar) {
                Button("Zoom In") { model.zoom.zoomIn() }.keyboardShortcut("=", modifiers: .command)
                Button("Zoom Out") { model.zoom.zoomOut() }.keyboardShortcut("-", modifiers: .command)
                Button("Actual Size") { model.zoom.actualSize() }.keyboardShortcut("0", modifiers: .command)
                Button("Zoom to Fit") { model.zoom.fit() }.keyboardShortcut("9", modifiers: .command)
                Divider()
            }
        }
        Settings {
            SettingsView().environmentObject(model)
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    var model: AppModel?
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard let model else { return .terminateNow }
        model.shutdown()
        let pendingWrite = model.queueWriteTask
        // Keep the app alive until the final ordered queue snapshot is on disk.
        Task { @MainActor in
            _ = await pendingWrite?.value
            sender.reply(toApplicationShouldTerminate: true)
        }
        return .terminateLater
    }
}

struct ContentView: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        switch model.phase {
        case .settingUp:
            SetupView()
        case .starting:
            ProgressView("Starting image engine…").frame(maxWidth: .infinity, maxHeight: .infinity)
        case .failed(let message):
            VStack(spacing: 12) {
                Image(systemName: "exclamationmark.triangle").font(.largeTitle).foregroundStyle(.orange)
                Text(message).multilineTextAlignment(.center).textSelection(.enabled)
                if !model.setupLog.isEmpty { LogView(text: model.setupLog).frame(height: 200) }
                Button("Retry") { if model.booted { model.restartEngine() } else { Task { await model.boot() } } }
            }
            .padding(40)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        case .ready:
            VStack(spacing: 0) {
                if model.engineUnresponsive {
                    HStack {
                        Label("The image engine isn't responding", systemImage: "exclamationmark.triangle")
                        Spacer()
                        Button("Restart") { model.restartEngine() }
                    }.padding(10).background(Color.orange.opacity(0.15))
                }
                HSplitView {
                    ControlsView().frame(minWidth: 320, idealWidth: 360, maxWidth: 460)
                    ResultView().frame(minWidth: 500)
                }
                Divider()
                StatusBarView(monitor: model.monitor)
            }
            .background(HistoryKeyboard(model: model).frame(width: 0, height: 0))
            .sheet(isPresented: $model.showModelManager) { ModelManagerView().environmentObject(model) }
            .sheet(item: $model.upscaleFileItem) { item in UpscaleOptionsView(item: item).environmentObject(model) }
            .sheet(isPresented: $model.showFullView) { FullResultView().environmentObject(model) }
        }
    }
}

struct SetupView: View {
    @EnvironmentObject var model: AppModel
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Setting up the local image engine").font(.title2.bold())
            Text("Installing Python, MLX and mflux into ~/Library/Application Support/Image Studio. This happens once.")
                .foregroundStyle(.secondary)
            ProgressView().progressViewStyle(.linear)
            LogView(text: model.setupLog)
        }
        .padding(30)
    }
}

struct LogView: View {
    var text: String
    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                Text(text).font(.system(.caption, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
                    .padding(8)
                Color.clear.frame(height: 1).id("end")
            }
            .background(Color(nsColor: .textBackgroundColor))
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .onChange(of: text) { _, _ in proxy.scrollTo("end") }
        }
    }
}

