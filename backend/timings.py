"""Learns how fast each model runs on this Mac, so the app can estimate a generation before it starts.

Denoising time is modelled as seconds per "work unit" = one model pass over one megapixel of tokens
(the target image plus, in edit mode, each reference image). Encoding/decoding/saving is kept as a
separate overhead that scales with the output size, and model loads are timed on their own.
"""

from __future__ import annotations

import json
import os
import threading

from catalog import APP_SUPPORT, ModelSpec, get

FILE = APP_SUPPORT / "timings.json"
_lock = threading.Lock()
_ALPHA = 0.5  # weight of the newest measurement (machines get busier/quieter; adapt quickly)


def _read() -> dict:
    try:
        return json.loads(FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write(data: dict) -> None:
    FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, FILE)


def _blend(old: float | None, new: float) -> float:
    return new if old is None else old + _ALPHA * (new - old)


def kind_of(image_paths: list, image_mode: str) -> str:
    if not image_paths:
        return "text"
    return "init" if image_mode == "init" else "edit"


def steps_run(spec: ModelSpec, steps: int, change: float | None, kind: str) -> int:
    """Denoising steps that actually run. Starting from an image skips the start of the schedule:
    mflux's Config.init_time_step = max(1, int(steps * image_strength)) whenever image_strength > 0."""
    if kind != "init":
        return steps
    change = spec.default_strength if change is None else change
    if spec.family == "qwen21":  # Pipeline passes strength=max(0.05, change) only when change < 1
        keep = 1 - max(0.05, change) if change < 1 else 0.0
    else:  # Pipeline passes image_strength = min(0.95, max(0, 1 - change))
        change = min(0.95, change)
        keep = min(0.95, max(0.0, 1 - change))
    return steps - max(1, int(steps * keep)) if keep > 0 else steps


def _units(steps: int, passes: int, megapixels: float, kind: str, refs: int, model_id: str = "") -> float:
    spec = get(model_id) if kind == "init" else None
    reference_tokens = kind == "edit" or (kind == "init" and spec is not None and spec.family == "qwen21")
    tokens = megapixels * (1 + refs) if reference_tokens else megapixels
    return max(1e-6, steps * passes * tokens)


def _key(model_id: str, quant: int | None, kind: str) -> str:
    return f"{model_id}|{quant or 0}|{kind}"


def record_generation(model_id: str, quant: int | None, kind: str, steps: int, passes: int, megapixels: float,
                      refs: int, denoise_seconds: float, other_seconds: float) -> None:
    if steps <= 0 or denoise_seconds <= 0:
        return
    with _lock:
        data = _read()
        entry = data.setdefault("gen", {}).setdefault(_key(model_id, quant, kind), {})
        entry["per_unit"] = _blend(entry.get("per_unit"), denoise_seconds / _units(steps, passes, megapixels, kind, refs, model_id))
        entry["overhead_per_mp"] = _blend(entry.get("overhead_per_mp"), max(0.0, other_seconds) / max(0.05, megapixels))
        entry["runs"] = entry.get("runs", 0) + 1
        _write(data)


def record_load(model_id: str, quant: int | None, seconds: float) -> None:
    with _lock:
        data = _read()
        key = f"{model_id}|{quant or 0}"
        loads = data.setdefault("load", {})
        loads[key] = _blend(loads.get(key), seconds)
        _write(data)


def estimate(model_id: str, quant: int | None, kind: str, steps: int, passes: int, megapixels: float, refs: int,
             count: int, needs_load: bool) -> dict:
    data = _read()
    gen = data.get("gen", {})
    entry = gen.get(_key(model_id, quant, kind))
    basis = "measured"
    if entry is None:
        # Prefer the same kind, then text, with matching precision breaking ties.
        candidates = [(k.split("|")[-1], k.split("|")[1], v) for k, v in gen.items() if k.startswith(f"{model_id}|")]
        candidates.sort(key=lambda item: (item[0] != kind, item[0] != "text", item[1] != str(quant or 0)))
        entry = candidates[0][2] if candidates else None
        basis = "similar" if entry else "none"
    load = data.get("load", {}).get(f"{model_id}|{quant or 0}") if needs_load else 0.0
    if entry is None:
        return {"seconds": None, "load_seconds": load, "basis": basis}
    per_image = entry["per_unit"] * _units(steps, passes, megapixels, kind, refs, model_id)
    per_image += entry.get("overhead_per_mp", 0.0) * megapixels
    return {"seconds": round(per_image * count, 1), "load_seconds": load, "basis": basis}


# --- Memory: the job's real MLX peak (load + encode + denoise + decode), per model, precision, input kind and
# memory mode. Text-only, edit and start-from-image jobs peak differently, and low-memory mode (text encoder freed
# before denoising) much lower, so one figure per model would be too low after a light job and too high after
# a heavy one.

_ACTIVATION_GB_PER_MP = 1.5  # how the peak grows with output size (rough, from klein/Qwen runs at 0.6-1 MP)
_PEAK_FORMAT = 2  # bump when the engine's memory use changes so peaks measured by an older engine are ignored


def _peak_key(model_id: str, quant: int | None, kind: str, low_memory: bool) -> str:
    return f"{model_id}|{quant or 0}|{kind}" + ("|low" if low_memory else "")


def record_peak(model_id: str, quant: int | None, peak_bytes: int, megapixels: float,
                kind: str = "text", low_memory: bool = False) -> None:
    if peak_bytes <= 0 or megapixels <= 0:
        return
    with _lock:
        data = _read()
        peaks = {k: v for k, v in data.get("peak", {}).items() if v.get("v") == _PEAK_FORMAT}
        peaks[_peak_key(model_id, quant, kind, low_memory)] = {
            "gb": round(peak_bytes / 1e9, 2), "mp": round(megapixels, 3), "v": _PEAK_FORMAT}
        data["peak"] = peaks
        _write(data)


def measured_peak_gb(model_id: str, quant: int | None, megapixels: float = 1.0, kind: str | None = None,
                     low_memory: bool = False, data: dict | None = None) -> float | None:
    """Expected peak for a job at this size, from the last measured job of this model and precision.
    Without a kind: the largest measured kind. Low memory falls back to the normal-mode peak (an upper bound)."""
    peaks = {k: v for k, v in (_read() if data is None else data).get("peak", {}).items()
             if v.get("v") == _PEAK_FORMAT}
    kinds = [kind] if kind else ["text", "edit", "init"]
    modes = [True, False] if low_memory else [False]
    for low in modes:
        entries = [peaks[key] for key in (_peak_key(model_id, quant, k, low) for k in kinds) if key in peaks]
        if entries:
            sized = [max(e["gb"] * 0.7, e["gb"] + (megapixels - e["mp"]) * _ACTIVATION_GB_PER_MP) for e in entries]
            return round(max(sized), 1)
    return None
