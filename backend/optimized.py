"""Pre-quantized base checkpoints shared by variants with the same weights."""

import hashlib
import json
import os
import shutil
import stat
import threading
import time
from importlib.metadata import version
from pathlib import Path

from core import assert_mlx_thread, atomic_json, require_disk

_poll_lock = threading.Lock()
_poll_cache = {}
_poll_epoch = 0


def invalidate_caches():
    global _poll_epoch
    with _poll_lock:
        _poll_epoch += 1
        _poll_cache.clear()


def polling_info(spec, quant):
    """Share validation and byte totals across catalog variants for ten seconds.

    Load/save/cleanup still validate directly with usable(), avoiding stale load decisions.
    Epoch checking prevents a concurrent mutation from repopulating stale polling results.
    """
    if not spec.optimizable:
        return False, 0
    root = folder(spec, quant)
    cache_key = (str(root), spec.family)
    now = time.monotonic()
    with _poll_lock:
        cached = _poll_cache.get(cache_key)
        if cached and now - cached[0] < 10:
            return cached[1]
        epoch = _poll_epoch
    result = (usable(spec, quant) is not None,
              sum(disk_bytes(copy) for copy in (root, root.with_name(root.name + ".partial"),
                                                root.with_name(root.name + ".old"))))
    with _poll_lock:
        if epoch == _poll_epoch:
            _poll_cache[cache_key] = (now, result)
    return result


def mflux_version():
    return version("mflux")


def base_identity(spec):
    return spec.model_path or spec.sources[0].repo


def model_ids(spec, models=None):
    import catalog

    return sorted(m.id for m in (catalog.all_models() if models is None else models)
                  if m.optimizable and base_identity(m) == base_identity(spec)
                  and m.mflux_name == spec.mflux_name)


def key(spec, quant):
    value = f"{spec.mflux_name}|{base_identity(spec)}|{quant or 0}"
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def folder(spec, quant):
    import catalog

    return catalog.APP_SUPPORT / "optimized" / key(spec, quant)


def base_snapshot(spec):
    from downloads import _snapshot

    if spec.model_path and os.path.isabs(spec.model_path):
        root = Path(spec.model_path).resolve()
        if not root.exists():
            return None
        entries = [(str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns)
                   for p in root.rglob("*") if p.is_file() and p.suffix in (".json", ".safetensors")]
        digest = hashlib.sha256(json.dumps(sorted(entries)).encode()).hexdigest()[:16]
        return f"{root}|{digest}"
    root = _snapshot(base_identity(spec))
    if root is None:
        return None
    revision = str(root.resolve())
    if spec.gguf:
        gguf_root = _snapshot(spec.gguf[0])
        if gguf_root:
            revision += f"|{(gguf_root / spec.gguf[1]).resolve()}"
    return revision


def checkpoint_valid(root, family):
    # mflux 0.22 ModelSaver uses the components' save subdirs. Krea's native
    # transformer uses the checkpoint root; the other supported families use transformer/.
    components = {
        "flux2": ("transformer", "text_encoder", "vae"),
        "qwen21": ("transformer", "text_encoder", "vae"),
        "zimage": ("transformer", "text_encoder", "vae"),
        "krea2": ("", "text_encoder", "vae"),
    }.get(family)
    if components is None:
        return False
    try:
        for part in components:
            index = root / part / "model.safetensors.index.json"
            weights = json.loads(index.read_text())["weight_map"]
            if not isinstance(weights, dict) or not weights or not all(isinstance(name, str) and Path(name).name == name
                                      and name.endswith(".safetensors")
                                      for name in weights.values()):
                return False
            for name in set(weights.values()):
                info = (index.parent / name).stat()
                if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
                    return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def usable(spec, quant):
    if not spec.optimizable:
        return None
    root = folder(spec, quant)
    try:
        stamp = json.loads((root / ".optimized.json").read_text())
        current = base_snapshot(spec)
        if (stamp["key"] != key(spec, quant) or stamp["quant"] != (quant or 0)
                or stamp["mflux_version"] != mflux_version()
                or (current is not None and stamp["base_snapshot"] != current)):
            return None
        return root if checkpoint_valid(root, spec.family) else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save(spec, quant, pipeline):
    assert_mlx_thread()
    bits = getattr(pipeline.model, "bits", None)
    if (bits or 0) != (quant or 0):
        raise ValueError(f"The checkpoint is already quantized at {bits or 0}-bit; cannot save it as {quant or 0}-bit.")
    target = folder(spec, quant)
    partial = target.with_name(target.name + ".partial")
    old = target.with_name(target.name + ".old")
    require_disk(target, int(spec.memory_gb.get(str(quant or 0), 20) * 1e9))
    shutil.rmtree(partial, ignore_errors=True)
    try:
        partial.mkdir(parents=True)
        pipeline.model.save_model(str(partial))
        if not checkpoint_valid(partial, spec.family):
            raise ValueError("The saved optimized checkpoint is incomplete")
        atomic_json(partial / ".optimized.json", {
            "key": key(spec, quant), "model_ids": model_ids(spec), "quant": quant or 0,
            "mflux_version": mflux_version(), "base_snapshot": base_snapshot(spec),
        })
        shutil.rmtree(old, ignore_errors=True)
        if target.exists():
            os.replace(target, old)
        try:
            os.replace(partial, target)
        except BaseException:
            if old.exists():
                os.replace(old, target)
            raise
    finally:
        shutil.rmtree(partial, ignore_errors=True)
        invalidate_caches()
    shutil.rmtree(old, ignore_errors=True)
    invalidate_caches()
    return target


def copies(spec):
    if not spec.optimizable:
        return []
    return [folder(spec, q) for q in spec.quant_options]


def disk_bytes(root):
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
        except FileNotFoundError:
            pass  # A download or cleanup may have replaced it while polling.
    return total
