"""Host memory and safe output helpers; no MLX or model loading."""

import ctypes
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path


def assert_mlx_thread():
    assert threading.current_thread().name.startswith("mlx"), "MLX work must run via on_mlx"


def atomic_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def unique_path(path: Path) -> Path:
    candidate, n = path, 2
    while candidate.exists() or candidate.with_suffix(".json").exists():
        candidate = path.with_name(f"{path.stem}_{n}{path.suffix}")
        n += 1
    return candidate


def verify_png(path: Path):
    from PIL import Image

    with Image.open(path) as image:
        if image.format != "PNG":
            raise ValueError("Saved output is not a PNG")
        image.verify()
    with Image.open(path) as image:
        image.load()
        return image.size


def save_png(image, path: Path):
    """Fast lossless Pillow save, verified before atomic publication."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            image.save(stream, format="PNG", compress_level=6, optimize=False)  # level 1 measured no faster here but ~45% larger
        size = verify_png(Path(temp))
        os.replace(temp, path)
        return size
    finally:
        Path(temp).unlink(missing_ok=True)


class DiskSpaceError(ValueError):
    pass


def require_disk(path: Path, needed: int):
    ancestor = path
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < needed + 512 * 1024**2:
        raise DiskSpaceError(f"Not enough free disk space: need about {needed / 1e9:.1f} GB plus 0.5 GB spare (507).")


def _command(*args):
    try:
        return subprocess.check_output(args, text=True, timeout=3)
    except (OSError, subprocess.SubprocessError):
        return ""


_memory_lock = threading.Lock()
_memory_cached = (float("-inf"), {})
_total_ram = None


class _VMStatistics64(ctypes.Structure):
    """<mach/vm_statistics.h> vm_statistics64 (HOST_VM_INFO64)."""
    _fields_ = ([(n, ctypes.c_uint32) for n in ("free_count", "active_count", "inactive_count", "wire_count")]
                + [(n, ctypes.c_uint64) for n in ("zero_fill_count", "reactivations", "pageins", "pageouts", "faults",
                                                  "cow_faults", "lookups", "hits", "purges")]
                + [(n, ctypes.c_uint32) for n in ("purgeable_count", "speculative_count")]
                + [(n, ctypes.c_uint64) for n in ("decompressions", "compressions", "swapins", "swapouts")]
                + [(n, ctypes.c_uint32) for n in ("compressor_page_count", "throttled_count", "external_page_count",
                                                  "internal_page_count")]
                + [("total_uncompressed_pages_in_compressor", ctypes.c_uint64)])


class _XswUsage(ctypes.Structure):
    """<sys/sysctl.h> xsw_usage (vm.swapusage)."""
    _fields_ = [("total", ctypes.c_uint64), ("avail", ctypes.c_uint64), ("used", ctypes.c_uint64),
                ("pagesize", ctypes.c_uint32), ("encrypted", ctypes.c_int32)]


try:
    _libc = ctypes.CDLL(None, use_errno=True)
    _libc.mach_host_self.restype = ctypes.c_uint32
    _host = _libc.mach_host_self()  # one send right for the process's lifetime
except (OSError, AttributeError):
    _libc = None


def _sysctl(name: str, value):
    size = ctypes.c_size_t(ctypes.sizeof(value))
    if _libc.sysctlbyname(name.encode(), ctypes.byref(value), ctypes.byref(size), None, ctypes.c_size_t(0)) != 0:
        raise OSError(ctypes.get_errno(), f"sysctlbyname({name}) failed")
    return value


def _read_memory_native() -> dict:
    """The same figures as sysctl/vm_stat, read in-process: spawning those tools forks this (large) process."""
    stats = _VMStatistics64()
    count = ctypes.c_uint32(ctypes.sizeof(stats) // 4)
    if _libc.host_statistics64(_host, 4, ctypes.byref(stats), ctypes.byref(count)) != 0:  # HOST_VM_INFO64
        raise OSError("host_statistics64 failed")
    page_size = ctypes.c_size_t()
    if _libc.host_page_size(_host, ctypes.byref(page_size)) != 0:
        raise OSError("host_page_size failed")
    pressure = _sysctl("kern.memorystatus_vm_pressure_level", ctypes.c_int32()).value
    swap = _sysctl("vm.swapusage", _XswUsage()).used
    total = _total_ram or _sysctl("hw.memsize", ctypes.c_uint64()).value
    # vm_stat's "Pages free" excludes speculative pages; inactive pages are reclaimable.
    pages = stats.free_count - stats.speculative_count + stats.inactive_count
    return {"pressure": {2: "warning", 4: "critical"}.get(pressure, "ok"),
            "free_bytes": pages * page_size.value, "swap_bytes": int(swap), "total_bytes": total}


def memory_snapshot() -> dict:
    global _memory_cached
    with _memory_lock:
        now = time.monotonic()
        if now - _memory_cached[0] >= 1:
            snapshot = _read_memory_snapshot()
            _memory_cached = (time.monotonic(), snapshot)
        # Callers add reusable bytes to their own snapshot, never to the cached object.
        return dict(_memory_cached[1])


def total_memory() -> int | None:
    """Physical RAM; read once, without refreshing the memory snapshot."""
    global _total_ram
    if _total_ram is None:
        _total_ram = memory_snapshot()["total_bytes"]
    return _total_ram


def _read_memory_snapshot() -> dict:
    global _total_ram
    if _libc is not None:
        try:
            snapshot = _read_memory_native()
            _total_ram = snapshot["total_bytes"]
            return snapshot
        except (OSError, AttributeError, ValueError):
            pass
    pressure = _command("/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level").strip()
    swap = _command("/usr/sbin/sysctl", "-n", "vm.swapusage")
    match = re.search(r"used\s*=\s*([\d.]+)([KMG])", swap)
    swap_bytes = float(match[1]) * {"K": 1024, "M": 1024**2, "G": 1024**3}[match[2]] if match else 0
    vm = _command("/usr/bin/vm_stat")
    page = re.search(r"page size of (\d+) bytes", vm)
    page_size = int(page[1]) if page else 4096
    # Inactive pages are reclaimable; speculative pages are already counted as free by macOS.
    pages = sum(int(m[1]) for label in ("Pages free", "Pages inactive")
                if (m := re.search(rf"{label}:\s+(\d+)", vm)))
    if _total_ram is None:
        total = _command("/usr/sbin/sysctl", "-n", "hw.memsize").strip()
        if total.isdigit():
            _total_ram = int(total)
    return {"pressure": {"2": "warning", "4": "critical"}.get(pressure, "ok"),
            "free_bytes": pages * page_size if vm else None,
            "swap_bytes": int(swap_bytes), "total_bytes": _total_ram}


def preflight_memory(memory_gb: dict, quant: int | None, snapshot: dict) -> dict:
    needed = memory_gb.get(str(quant or 0), max(memory_gb.values(), default=0)) * 1e9
    free = snapshot.get("free_bytes")
    level = snapshot["pressure"]
    # A loaded model already occupies host pages. Compare its advertised peak with host free +
    # reusable MLX allocations (provided by the caller), leaving headroom for VAE decode and apps.
    available = (free or 0) + snapshot.get("reusable_bytes", 0)
    if free is not None and needed > available:
        level = "critical" if needed > available * 1.5 else ("warning" if level == "ok" else level)
    if snapshot["swap_bytes"] > 2 * 1024**3 and level == "ok":
        level = "warning"
    suggest = 4 if quant != 4 and "4" in memory_gb and memory_gb["4"] * 1e9 < needed else None
    if level == "ok":
        message = "Memory looks sufficient."
    else:
        available_text = f"{available / 1e9:.1f} GB available" if free is not None else "free memory unavailable"
        message = (f"Memory pressure is {level}: this model needs about {needed / 1e9:.1f} GB; "
                   f"{available_text}, {snapshot['swap_bytes'] / 1e9:.1f} GB swap in use. "
                   "Close other large apps or use a smaller image or lower precision.")
    return {"level": level, "message": message, "suggest_quant": suggest}
