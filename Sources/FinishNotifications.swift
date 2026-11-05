import AppKit
import UserNotifications

extension AppModel {
    /// Asks once, the first time there's something to notify about, rather than at launch.
    func prepareNotifications() {
        guard notifyWhenFinished else { return }
        Task {
            let center = UNUserNotificationCenter.current()
            if await center.notificationSettings().authorizationStatus == .notDetermined {
                do { _ = try await center.requestAuthorization(options: [.alert, .sound]) }
                catch { FileHandle.standardError.write(Data("Notification permission: \(error.localizedDescription)\n".utf8)) }
            }
            await refreshNotificationStatus()
        }
    }

    func refreshNotificationStatus() async {
        let blocked = await UNUserNotificationCenter.current().notificationSettings().authorizationStatus == .denied
        if notificationsBlocked != blocked { notificationsBlocked = blocked }
    }

    /// Queue titles contain the prompt, which a notification shouldn't show.
    func finishName(for job: QueuedJob) -> String {
        switch job.kind {
        case .generate(let request): return models.first { $0.id == request.model }?.name ?? request.model
        case .cloudBatch(let requests): return "\(requests.count) cloud models"
        case .upscale: return "Upscale"
        }
    }

    func announceFinishedRun() {
        defer { finishedRun = nil }
        guard let run = finishedRun, run.isWorthAnnouncing, notifyWhenFinished, !NSApp.isActive, !isShuttingDown else { return }
        let content = UNMutableNotificationContent()
        content.title = run.title
        content.body = run.body()
        content.sound = .default
        if let last = run.images.last { content.userInfo = ["path": last.path] }
        UNUserNotificationCenter.current().add(UNNotificationRequest(identifier: UUID().uuidString, content: content, trigger: nil)) { error in
            if let error { FileHandle.standardError.write(Data("Notification: \(error.localizedDescription)\n".utf8)) }
        }
    }

    /// The image may have been trashed since the notification arrived.
    func showFinishedImage(path: String) {
        guard let item = history.first(where: { $0.url.path == path }) else { return }
        selectedPending = nil
        selected = item
    }
}

extension AppDelegate: UNUserNotificationCenterDelegate {
    /// Clicking the notification: bring the window back and show the last image.
    nonisolated func userNotificationCenter(_ center: UNUserNotificationCenter, didReceive response: UNNotificationResponse,
                                            withCompletionHandler completionHandler: @escaping () -> Void) {
        let path = response.notification.request.content.userInfo["path"] as? String
        Task { @MainActor in
            NSApp.activate(ignoringOtherApps: true)
            NSApp.windows.first { $0.canBecomeMain }?.makeKeyAndOrderFront(nil)
            if let path { self.model?.showFinishedImage(path: path) }
            completionHandler()
        }
    }
}
