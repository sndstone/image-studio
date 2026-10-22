import Darwin
import Foundation
import IOKit

/// Samples GPU utilization, system memory, swap and the backend's memory once a second.
@MainActor
final class SystemMonitor: ObservableObject {
    struct Sample {
        var gpu: Double?  // 0...100
        var memoryUsed: UInt64
        var memoryTotal: UInt64
        var swapUsed: UInt64
        var pressure: Int  // 1 normal, 2 warning, 4 critical
        var backendFootprint: UInt64?
    }

    @Published private(set) var current: Sample?
    @Published private(set) var gpuHistory: [Double] = []
    @Published private(set) var swapMBPerSecond = 0.0
    private var swapSamples: [(Date, UInt64)] = []
    private var running = false
    var backendPID: Int32?
    var shouldSample: (() -> Bool)?
    private var timer: Timer?

    func start() {
        guard timer == nil else { return }
        running = true
        sample()
        timer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.sample() }
        }
    }

    func stop() {
        running = false
        timer?.invalidate()
        timer = nil
        swapSamples = []
    }

    private func sample() {
        guard shouldSample?() != false else { swapSamples = []; return }
        let pid = backendPID
        Task.detached(priority: .utility) {
            let sample = Sample(
                gpu: Self.gpuUtilization(),
                memoryUsed: Self.memoryUsed(),
                memoryTotal: ProcessInfo.processInfo.physicalMemory,
                swapUsed: Self.swapUsed(),
                pressure: Self.pressureLevel(),
                backendFootprint: pid.flatMap(Self.footprint)
            )
            await MainActor.run {
                guard self.running else { return }
                self.current = sample
                let now = Date()
                self.swapSamples.append((now, Self.swapTrafficBytes()))
                self.swapSamples.removeAll { now.timeIntervalSince($0.0) > 60 }
                if let first = self.swapSamples.first, let last = self.swapSamples.last, last.0 > first.0 {
                    self.swapMBPerSecond = Double(last.1 >= first.1 ? last.1 - first.1 : 0) / last.0.timeIntervalSince(first.0) / 1_048_576
                }
                self.gpuHistory.append(sample.gpu ?? 0)
                if self.gpuHistory.count > 60 { self.gpuHistory.removeFirst(self.gpuHistory.count - 60) }
            }
        }
    }

    /// "Device Utilization %" from the GPU's IOAccelerator performance statistics (no privileges needed).
    nonisolated static func gpuUtilization() -> Double? {
        var iterator = io_iterator_t()
        guard IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("IOAccelerator"), &iterator) == KERN_SUCCESS
        else { return nil }
        defer { IOObjectRelease(iterator) }
        var best: Double?
        while true {
            let entry = IOIteratorNext(iterator)
            if entry == 0 { break }
            defer { IOObjectRelease(entry) }
            guard let stats = IORegistryEntryCreateCFProperty(entry, "PerformanceStatistics" as CFString, kCFAllocatorDefault, 0)?
                .takeRetainedValue() as? [String: Any],
                let value = stats["Device Utilization %"] as? NSNumber else { continue }
            best = max(best ?? 0, value.doubleValue)
        }
        return best
    }

    /// Matches Activity Monitor's "Memory Used": app memory + wired + compressed.
    nonisolated static func memoryUsed() -> UInt64 {
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
        let host = mach_host_self()  // returns a send right that must be released
        defer { mach_port_deallocate(mach_task_self_, host) }
        let result = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(host, HOST_VM_INFO64, $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }
        let page = UInt64(vm_kernel_page_size)
        let app = UInt64(stats.internal_page_count) - UInt64(stats.purgeable_count)
        return (app + UInt64(stats.wire_count) + UInt64(stats.compressor_page_count)) * page
    }

    nonisolated static func swapTrafficBytes() -> UInt64 {
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
        let host = mach_host_self()
        defer { mach_port_deallocate(mach_task_self_, host) }
        let result = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(host, HOST_VM_INFO64, $0, &count)
            }
        }
        return result == KERN_SUCCESS ? (stats.swapins + stats.swapouts) * UInt64(vm_kernel_page_size) : 0
    }

    nonisolated static func swapUsed() -> UInt64 {
        var usage = xsw_usage()
        var size = MemoryLayout<xsw_usage>.size
        guard sysctlbyname("vm.swapusage", &usage, &size, nil, 0) == 0 else { return 0 }
        return usage.xsu_used
    }

    nonisolated static func pressureLevel() -> Int {
        var level: Int32 = 1
        var size = MemoryLayout<Int32>.size
        guard sysctlbyname("kern.memorystatus_vm_pressure_level", &level, &size, nil, 0) == 0 else { return 1 }
        return Int(level)
    }

    /// Physical footprint of a process (what Activity Monitor shows in its Memory column).
    nonisolated static func footprint(of pid: Int32) -> UInt64? {
        var info = rusage_info_v4()
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) {
                proc_pid_rusage(pid, RUSAGE_INFO_V4, $0)
            }
        }
        return result == 0 ? info.ri_phys_footprint : nil
    }
}
