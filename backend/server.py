"""Image Studio backend: keeps one mflux (MLX) image model loaded and serves generation, model
management and download requests from the Swift app over a local-only HTTP API."""

import argparse
import gc
import hashlib
import hmac
import math
import socket
import io
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import mlx.core as mx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# The app supplies its Hugging Face token explicitly.
if "IMAGE_STUDIO_HF_TOKEN" in os.environ:
    os.environ["HF_TOKEN"] = os.environ["IMAGE_STUDIO_HF_TOKEN"]

import catalog
import optimized
from core import (DiskSpaceError, assert_mlx_thread, atomic_json, memory_snapshot, preflight_memory, total_memory,
                  require_disk, save_png, unique_path, verify_png)
from performance import materialize
import storage
import timings
from downloads import DownloadManager, disk_usage, shared_with, local_models_using, _repo_dir
from gguf_convert import staging_folders
from mflux.utils.exceptions import StopImageGenerationException

OUTPUT_ROOT = Path(os.environ.get("IMAGE_STUDIO_OUTPUT", Path.home() / "Pictures/Image Studio"))
REF_CACHE = catalog.APP_SUPPORT / "reference-cache"
log = logging.getLogger("image-studio")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _friendly_error(e: Exception) -> str:
    text = str(e)
    if "Insufficient Memory" in text or "OutOfMemory" in text or "out of memory" in text.lower():
        return ("Ran out of memory. Close other large apps, use a smaller size or 4-bit precision, "
                "or unload the current model in the Model Manager.")
    if "cannot identify image file" in text:
        return "Couldn't read an input image. Try saving it as JPEG or PNG."
    return text


def _clear_exception_frames(error):
    """Drop completed model frames, including chained errors retained by log handlers."""
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        traceback.clear_frames(current.__traceback__)
        current.__traceback__ = None
        pending.extend((current.__cause__, current.__context__))


def _reference_orientation(image, path):
    import struct
    from PIL import Image, PngImagePlugin

    if image.format == "PNG":
        # Pillow's PNG getexif() calls load() to find trailing metadata. Read only
        # metadata chunks, seeking over compressed pixels, with Pillow's own parsers.
        with open(path, "rb") as fp:
            fp.seek(8)
            png = PngImagePlugin.PngStream(fp)
            seen_pixels = False
            while True:
                try:
                    cid, pos, length = png.read()
                except (struct.error, SyntaxError):
                    break
                if cid == b"IEND" or (seen_pixels and cid == b"fcTL" and image.is_animated):
                    break
                seen_pixels |= cid == b"IDAT"
                if cid in (b"eXIf", b"tEXt", b"zTXt", b"iTXt"):
                    try:
                        png.call(cid, pos, length)
                    except UnicodeDecodeError:
                        break
                fp.seek(pos + length + 4)
            image.info.update(png.im_info)
        return Image.Image.getexif(image).get(0x0112, 1)
    return image.getexif().get(0x0112, 1)


def _prepare_reference(path: str, max_megapixels: float = 2.0) -> str:
    """Returns a path the pipelines can read: HEIC/AVIF/TIFF etc. are converted to PNG and EXIF
    rotation is applied (mflux's FLUX.2 loader ignores it). Cached by path + modification time."""
    from PIL import Image, ImageOps, UnidentifiedImageError

    src = Path(path)
    key = hashlib.sha1(f"{src.resolve()}|{src.stat().st_mtime_ns}|{max_megapixels}".encode()).hexdigest()[:16]
    out = REF_CACHE / f"{key}.png"
    if out.exists():
        return str(out)
    REF_CACHE.mkdir(parents=True, exist_ok=True)
    readable = src

    def open_image(path):
        image = Image.open(path)
        try:
            needs_resize = image.width * image.height > max_megapixels * 1e6
            passthrough = (not needs_resize and image.format in ("JPEG", "PNG", "WEBP")
                           and _reference_orientation(image, path) == 1 and image.mode in ("RGB", "RGBA", "L"))
            if not passthrough:
                image.load()
            return image, passthrough, needs_resize
        except BaseException:
            image.close()
            raise

    try:
        image, passthrough, needs_resize = open_image(src)
    except (UnidentifiedImageError, OSError):
        # Formats Pillow can't read (HEIC, some AVIF/RAW): let macOS convert them.
        converted = REF_CACHE / f"{key}-sips.png"
        result = subprocess.run(["/usr/bin/sips", "-s", "format", "png", str(src), "--out", str(converted)],
                                capture_output=True, text=True, timeout=30)
        if result.returncode != 0 or not converted.exists():
            raise HTTPException(400, f"Couldn't read the input image {src.name}. Try saving it as JPEG or PNG.")
        readable = converted
        image, passthrough, needs_resize = open_image(converted)
    if passthrough:
        image.close()
        return str(readable)
    with image:
        transposed = ImageOps.exif_transpose(image)
        with transposed:
            image = transposed.convert("RGBA" if "A" in transposed.getbands() else "RGB")
    if needs_resize:
        ratio = math.sqrt(max_megapixels * 1e6 / (image.width * image.height))
        image.thumbnail((max(1, int(image.width * ratio)), max(1, int(image.height * ratio))), Image.Resampling.LANCZOS)
    image.save(out, compress_level=1)
    image.close()
    return str(out)


class LoraRequest(BaseModel):
    path: str
    scale: float = 1.0


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str | None = None
    image_paths: list[str] = Field(default_factory=list)
    model: str = "flux2-klein-4b"
    quantize: int | None = 8
    width: int = 1024
    height: int = 1024
    steps: int = 4
    guidance: float = 1.0
    strength: float | None = None  # how much to change the input image (img2img/edit), 0..1
    image_mode: str = "reference"  # reference: edit/combine the images | init: start from image 1
    seed: int | None = None
    count: int = 1
    lora_path: str | None = None
    lora_scale: float = 1.0
    loras: list[LoraRequest] = Field(default_factory=list)
    fast: bool = False
    mask_path: str | None = None
    auto_mask: str | None = None
    enhance_prompt: bool = False


class UpscaleRequest(BaseModel):
    path: str
    scale: float = 2.0
    softness: float = 0.0  # 0..1: pre-smooths the input (helps noisy photos; generated images are clean)
    quantize: int | None = None


class StorageRequest(BaseModel):
    ids: list[str] = Field(default_factory=list)
    keep: list[str] = Field(default_factory=list)  # files the app still uses (current reference images)


class EstimateRequest(GenerateRequest):
    prompt: str = ""


class SettingsRequest(BaseModel):
    idle_unload_minutes: int | None = None  # 0 = keep the model loaded
    live_preview: bool | None = None
    embed_metadata: bool | None = None
    low_memory: str | None = None
    offline: bool | None = None


class PressureRequest(BaseModel):
    level: str


class LoadRequest(BaseModel):
    quantize: int | None = None
    lora_path: str | None = None
    lora_scale: float = 1.0


class OptimizeRequest(BaseModel):
    quantize: int


class DeleteRequest(BaseModel):
    scope: str = "all"


class CustomModelRequest(BaseModel):
    name: str
    path: str
    architecture: str


class State:
    def __init__(self):
        self.lock = threading.Lock()  # one model load / generation at a time
        self.model_status_lock = threading.Lock()  # coherent key + precision for status readers
        self.busy = None
        self.images_done = []
        self.memory_warning = None
        self.embed_metadata = False
        self.low_memory = "auto"
        self.offline = False
        self.encoder_evicted = False
        self.pressure = None  # Latest app notification; sysctl is the fallback until one arrives.
        self.mlx_job_active = False
        self.cache_limit = None
        self.job_low_memory = False
        self.pipeline: catalog.Pipeline | None = None
        self.model_key = None
        self.phase = "idle"  # idle | loading | encoding | denoising | decoding | saving
        self.message = "Ready"
        self.step = 0
        self.total_steps = 0
        self.passes = 1  # model passes per step (2 when guidance adds a negative pass)
        self.image_index = 0
        self.image_count = 0
        self.cancel = threading.Event()
        # MLX allocator counters, sampled on the MLX thread (reading them elsewhere races with it).
        self.mlx_active = 0
        self.mlx_cache = 0
        self.mlx_peak = 0
        # Timing of the current image's denoising loop, for time estimates.
        self.loop_started = 0.0
        self.loop_ended = 0.0
        self.loop_steps = 0
        self.converted = False  # the current load included a one-time GGUF conversion
        # Idle unloading frees the model's memory for other apps when nothing has run for a while.
        self.idle_unload_minutes = 10
        self.last_activity = time.time()
        self.auto_unloaded = False
        # Live preview: a small JPEG of the image mid-generation, refreshed every few seconds.
        self.live_preview = True
        self.preview: bytes | None = None
        self.preview_id = 0
        self.preview_step = 0
        self.last_preview = 0.0
        self.preview_cost = 0.0  # seconds one preview decode takes (smoothed)
        self.preview_time = 0.0  # preview time inside the current loop (excluded from timings)


state = State()
downloads = DownloadManager()
_shutdown = threading.Event()


@asynccontextmanager
async def lifespan(app):
    from mflux.utils.image_util import ImageUtil

    ImageUtil.embed_metadata_enabled = state.embed_metadata
    on_mlx(_init_mlx)
    _shutdown.clear()
    idle = threading.Thread(target=_idle_unloader, daemon=True)
    idle.start()
    try:
        yield
    finally:
        _shutdown.set()
        downloads.cancel_all()
        on_mlx(_unload)
        idle.join(timeout=1)


app = FastAPI(lifespan=lifespan)
app.state.port = 7861


def authorized(host: str, token: str | None, port: int) -> bool:
    expected = os.environ.get("IMAGE_STUDIO_TOKEN")
    return (host in (f"127.0.0.1:{port}", f"localhost:{port}")
            and (expected is None or (token is not None and hmac.compare_digest(token.encode(), expected.encode()))))


@app.middleware("http")
async def authenticate(request: Request, call_next):
    address = request.scope.get("server")
    port = (address[1] if address else None) or app.state.port
    if not authorized(request.headers.get("host", ""), request.headers.get("x-image-studio-token"), port):
        return JSONResponse(status_code=403, content={"detail": "Forbidden"})
    return await call_next(request)


@app.exception_handler(RequestValidationError)
async def invalid_request(request, exc):
    errors = "; ".join(f"{'.'.join(str(p) for p in e['loc'][1:])}: {e['msg']}" for e in exc.errors())
    return JSONResponse(status_code=400, content={"detail": errors})


def _acquire(kind):
    if not state.lock.acquire(blocking=False):
        text = {"generate": "Generating images", "upscale": "Upscaling an image", "load": "Loading a model",
                "optimize": "Optimizing a model", "cleanup": "Cleaning up storage", "unload": "Unloading a model"}
        raise HTTPException(409, f"{text.get(state.busy, 'An operation is running')}. Please wait.")
    state.busy = kind
    if kind in ("generate", "upscale", "load", "optimize"):
        try:
            on_mlx(_apply_memory_policy)
        except BaseException:
            state.busy = None
            state.lock.release()
            raise


def _release():
    try:
        # Settle pressure/cache policy while we still own the operation lock.
        on_mlx(_handle_idle_pressure)
    finally:
        state.busy = None
        state.lock.release()


def _check_cancel():
    if state.cancel.is_set():
        raise StopImageGenerationException("Cancelled")

# MLX streams belong to the thread that created them, and FastAPI runs each request on an arbitrary
# pool thread. Loading on one thread and generating on another fails with "There is no Stream(cpu, 0)
# in current thread", so every MLX operation runs on this single thread.
_mlx_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")


def on_mlx(fn, *args, **kwargs):
    return _mlx_thread.submit(fn, *args, **kwargs).result()


def _init_mlx():
    assert_mlx_thread()
    mx.set_memory_limit(16 * 1024**3)  # Allocator guideline, not a hard memory cap.
    mx.set_wired_limit(0)
    _apply_memory_policy()


def _pressure_level():
    if state.pressure is not None:
        return state.pressure
    level = memory_snapshot()["pressure"]
    return "normal" if level == "ok" else level


def _apply_memory_policy(active=None, flush=True):
    assert_mlx_thread()
    level = _pressure_level()
    if active is None:
        active = state.mlx_job_active or state.busy in ("generate", "upscale", "load", "optimize")
    limit = 0 if level != "normal" else (512 * 1024**2 if active else 2 * 1024**3)
    if state.cache_limit != limit:
        mx.set_cache_limit(limit)
        state.cache_limit = limit
        if flush:
            mx.clear_cache()
    if level != "normal":
        state.preview = None
    return level


def _handle_idle_pressure():
    assert_mlx_thread()
    if _apply_memory_policy(active=False) == "critical" and state.pipeline is not None:
        _unload()
        state.auto_unloaded = True
        state.message = "Model unloaded while memory pressure is critical"


@app.post("/pressure")
def pressure(req: PressureRequest):
    if req.level not in ("normal", "warning", "critical"):
        raise HTTPException(400, "level must be normal, warning or critical")
    state.pressure = req.level
    # Busy jobs consume the notification at their next safe callback; never queue an
    # unload behind a running job or interrupt it. The same lock protects idle unloads.
    if state.lock.acquire(blocking=False):
        state.busy = "pressure"
        try:
            on_mlx(_handle_idle_pressure)
        finally:
            _release()
    return {"ok": True, "pressure": state.pressure}


def _begin_job():
    assert_mlx_thread()
    state.mlx_job_active = True
    mx.set_memory_limit(16 * 1024**3)
    mx.set_wired_limit(0)
    _apply_memory_policy()


def _sample_memory():
    """Must run on the MLX thread."""
    assert_mlx_thread()
    state.mlx_active = int(mx.get_active_memory())
    state.mlx_cache = int(mx.get_cache_memory())
    state.mlx_peak = int(mx.get_peak_memory())




def _drop_text_encoders(model):
    """Drop mflux 0.22's encoder references; collect after all shared adapters."""
    for name in ("clip_text_encoder", "t5_text_encoder", "text_encoder", "image_encoder",
                 "image_embedder", "depth_pro", "qwen_vl_encoder"):
        if hasattr(model, name):
            setattr(model, name, None)
    tokenizers = getattr(model, "tokenizers", None)
    if tokenizers is not None and "qwen_vl" in tokenizers:
        tokenizers["qwen_vl"] = None


class ProgressCallback:
    skipped = 0  # schedule steps skipped by starting from an image

    def call_before_loop(self, seed, prompt, latents, config, **kwargs):
        assert_mlx_thread()
        _check_cancel()
        # All families finish prompt encoding before this callback. The edit FLUX pipeline
        # shares weights but has its own attributes, so evict encoders from both objects.
        cache_limit = state.cache_limit
        _apply_memory_policy(flush=False)
        encodings = getattr(state.pipeline, "prompt_encodings", None)
        safe_to_evict = state.image_index == state.image_count or (encodings is not None and encodings.reusable)
        if (safe_to_evict and state.pipeline is not None
                and getattr(state.pipeline.model, "text_encoder", None) is not None
                and (state.low_memory == "always" or
                     (state.low_memory == "auto" and (state.job_low_memory or _pressure_level() != "normal")))):
            before = int(mx.get_active_memory())
            for model in (state.pipeline.model, getattr(state.pipeline, "edit", None)):
                if model is not None:
                    _drop_text_encoders(model)
            model = None
            gc.collect()
            mx.clear_cache()
            _sample_memory()
            log.info("Encoder eviction MLX active: %d → %d bytes; evaluated job encodings: %d bytes",
                     before, state.mlx_active, encodings.bytes if encodings is not None else 0)
            state.encoder_evicted = True
            with state.model_status_lock:
                state.model_key = None
        elif state.cache_limit != cache_limit:
            mx.clear_cache()
        state.loop_started = state.last_preview = time.time()
        state.loop_steps = 0
        state.preview_time = 0.0
        state.phase = "denoising"
        # Starting from an image skips the start of the schedule; count only the steps that run.
        self.skipped = getattr(config, "init_time_step", 0) or 0
        self.prev = latents
        self.previews = 0
        state.total_steps = config.num_inference_steps - self.skipped
        state.step = 0

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps, denoised=None):
        assert_mlx_thread()
        _apply_memory_policy()
        _sample_memory()
        state.phase = "denoising"
        state.step = t + 1 - self.skipped
        state.loop_steps += 1
        state.total_steps = config.num_inference_steps - self.skipped
        if state.cancel.is_set():
            raise StopImageGenerationException("Cancelled")
        prev, self.prev = getattr(self, "prev", None), latents
        if state.live_preview and _pressure_level() == "normal" and state.step < state.total_steps:
            self._maybe_preview(prev, latents, config, t, denoised)

    @staticmethod
    def _predicted_image(prev, latents, config, t):
        """The model's current guess of the finished image. Mid-run latents are mostly noise; all our
        schedulers take Euler steps x' = x + (s' - s) * v, so v and then x0 = x' - s' * v follow from two
        consecutive latents."""
        assert_mlx_thread()
        sigmas = getattr(config.scheduler, "sigmas", None)
        sigmas = sigmas() if callable(sigmas) else sigmas
        if prev is None or sigmas is None or t + 1 >= len(sigmas) or prev.shape != latents.shape:
            return latents
        s_now, s_next = float(sigmas[t].item()), float(sigmas[t + 1].item())
        if abs(s_next - s_now) < 1e-6:
            return latents
        velocity = (latents - prev) / (s_next - s_now)
        return latents - s_next * velocity

    def _maybe_preview(self, prev, latents, config, t, denoised=None):
        assert_mlx_thread()
        now = time.time()
        # Keep previews cheap: at most every 2 s and every 5 decode-times, but show at least one by halfway.
        due = now - state.last_preview >= max(2.0, 5 * state.preview_cost)
        halfway = self.previews == 0 and state.step * 2 >= state.total_steps
        if not (due or halfway) or state.pipeline is None:
            return
        if _pressure_level() != "normal":
            state.memory_warning = "Live previews paused while memory pressure is high."
            state.last_preview = now
            return
        # Several models call back before evaluating the step; finish it first so the preview's own
        # cost (excluded from the speed measurements) doesn't include the step.
        mx.eval(latents)
        now = time.time()
        try:
            # Krea's sampler isn't a plain Euler step, but it hands over its own prediction.
            target = denoised if denoised is not None else self._predicted_image(prev, latents, config, t)
            image = state.pipeline.preview(target, config)
        except Exception:
            log.exception("Preview failed; turning previews off for this session")
            state.live_preview = False
            return
        if image is None:
            return
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, "JPEG", quality=80)
        cost = time.time() - now
        log.info("Preview at step %d/%d took %.2f s", state.step, state.total_steps, cost)
        state.preview_cost = cost if state.preview_cost == 0 else 0.5 * (state.preview_cost + cost)
        state.preview_time += cost
        state.preview = buffer.getvalue()
        state.preview_step = state.step
        state.preview_id += 1
        self.previews += 1
        state.last_preview = time.time()

    def call_after_loop(self, seed, prompt, latents, config):
        assert_mlx_thread()
        _apply_memory_policy()
        state.loop_ended = time.time()
        _sample_memory()
        state.phase = "decoding"


def _quant(spec: catalog.ModelSpec, quantize: int | None) -> int | None:
    q = spec.default_quant if quantize is None else quantize
    if q not in spec.quant_options:
        q = spec.default_quant
    return None if q == 0 else q


def _unload(*, cleanup=True) -> None:
    assert_mlx_thread()
    state.encoder_evicted = False
    _clear_job_references(state.pipeline)
    with state.model_status_lock:
        state.model_key = None
        state.pipeline = None
    if cleanup:
        gc.collect()
        _apply_memory_policy(flush=False)
        mx.clear_cache()
        _sample_memory()


def _conversion_progress(done: int, total: int) -> None:
    state.converted = True
    if done >= total:  # conversion finished; the weights are loaded next
        state.phase, state.message = "loading", "Loading…"
        return
    state.phase = "converting"
    state.message = "Converting the model for first use…"
    state.step, state.total_steps = done, total


def _load(spec: catalog.ModelSpec, quantize: int | None, lora: str | None, lora_scale: float,
          loras=None) -> catalog.Pipeline:
    assert_mlx_thread()
    _apply_memory_policy()
    q = _quant(spec, quantize)
    adapters = loras or ([{"path": lora, "scale": lora_scale}] if lora else [])
    key = _model_key(spec, q, adapters)
    if state.pipeline is not None and state.model_key == key:
        return state.pipeline
    checkpoint = optimized.usable(spec, q)
    checked_spec = _adapter_spec(spec) if checkpoint else spec
    status = downloads.status(checked_spec)
    if not status["downloaded"]:
        raise HTTPException(409, f"{spec.name} is not downloaded yet. Download it in the Model Manager.")
    state.phase = "loading"
    state.message = f"Loading {spec.name}…"
    _unload()
    state.converted = False
    state.preview_cost = 0.0  # preview decode cost depends on the model's VAE
    started = time.time()
    pipeline, failure = None, None
    try:
        pipeline = catalog.Pipeline(spec, q, None, 1.0, progress=_conversion_progress,
                                    loras=adapters, model_path=str(checkpoint) if checkpoint else None)
        # Weight loading and quantization are lazy. Materialize them before reporting Ready.
        materialize(pipeline.model)
    except Exception as e:
        log.exception("Loading %s failed", spec.id)
        failure = HTTPException(400 if isinstance(e, DiskSpaceError) else 500,
                                str(e) if isinstance(e, DiskSpaceError) else
                                f"Could not load {spec.name}: {_friendly_error(e)}")
        _clear_exception_frames(e)
        _clear_job_references(pipeline)
        pipeline = None
    if failure is not None:
        # Outside the except block, e and its traceback no longer retain model frames.
        _unload()
        raise failure from None
    pipeline.progress = ProgressCallback()
    pipeline.register(pipeline.progress)
    with state.model_status_lock:
        state.pipeline = pipeline
        state.model_key = key
    state.auto_unloaded = False
    state.last_activity = time.time()
    if not state.converted and not adapters:
        timings.record_load(spec.id, q, time.time() - started)
    _sample_memory()
    return pipeline


def _adapter_spec(spec):
    repos = {repo for repo, _ in spec.builtin_loras}
    return replace(spec, sources=[s for s in spec.sources if s.repo in repos], model_path=None)


def _converted_folders(spec: catalog.ModelSpec) -> list[Path]:
    if not spec.gguf:
        return []
    folder = catalog.CONVERTED_DIR / spec.id
    return [folder, *staging_folders(folder)]


def _converted_bytes(spec: catalog.ModelSpec) -> int:
    return sum(f.stat().st_size for folder in _converted_folders(spec) if folder.exists()
               for f in folder.rglob("*") if f.is_file() and not f.is_symlink())


def _loaded_snapshot():
    with state.model_status_lock:
        model_key = state.model_key
        pipeline = state.pipeline
        if model_key is None:
            return None, None
        bits = getattr(pipeline.model, "bits", None) if pipeline else None
        return model_key[0], bits if isinstance(bits, int) else (model_key[1] or 0)


def _model_json(spec: catalog.ModelSpec, usage: dict[str, int], loaded_snapshot=None,
                models=None, timing_data=None) -> dict:
    loaded_id, loaded_quant = _loaded_snapshot() if loaded_snapshot is None else loaded_snapshot
    data = spec.to_json()
    timing_data = timings._read() if timing_data is None else timing_data
    data["memory_gb"] = _memory_gb(spec, timing_data=timing_data)  # at ~1 MP
    data["memory_measured"] = [int(q) for q in spec.memory_gb
                               if timings.measured_peak_gb(spec.id, int(q), data=timing_data) is not None]
    data["status"] = downloads.status(spec)
    data["shared_with"] = shared_with(spec, models=models)
    checkpoint_info = {str(q): optimized.polling_info(spec, q) for q in spec.quant_options}
    checkpoints = {q: info[0] for q, info in checkpoint_info.items()}
    if any(checkpoints.values()) and downloads.status(_adapter_spec(spec))["downloaded"]:
        data["status"]["downloaded"] = True
    optimized_bytes = sum(info[1] for info in checkpoint_info.values())
    converted_bytes = _converted_bytes(spec)
    data["disk_bytes"] = sum(usage.get(s.repo, 0) for s in spec.sources) + converted_bytes + optimized_bytes
    data["own_bytes"] = sum(usage.get(s.repo, 0) for s in _own_sources(spec, models=models)) + converted_bytes
    if spec.optimizable and len(optimized.model_ids(spec, models=models)) == 1:
        data["own_bytes"] += optimized_bytes
    data["optimized"] = checkpoints
    loaded = loaded_id == spec.id
    data["loaded"] = loaded
    # A pre-quantized checkpoint keeps its own precision regardless of what was requested.
    data["loaded_quant"] = loaded_quant if loaded else None
    return data


# --- generic -----------------------------------------------------------------------------------


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/status")
def status():
    loaded_id, _ = _loaded_snapshot()
    return {
        "phase": state.phase,
        "message": state.message,
        "step": state.step,
        "total_steps": state.total_steps,
        "passes": state.passes,
        "preview_id": state.preview_id if state.preview else None,
        "preview_step": state.preview_step,
        "auto_unloaded": state.auto_unloaded,
        "idle_unload_minutes": state.idle_unload_minutes,
        "image_index": state.image_index,
        "image_count": state.image_count,
        "loaded_model": loaded_id,
        "mlx_active": state.mlx_active,
        "images_done": list(state.images_done),
        "live_preview": state.live_preview,
        "busy": state.busy,
        "memory_warning": state.memory_warning,
        "pressure": _pressure_level(),
        "output_dir": str(OUTPUT_ROOT),
    }


@app.get("/system")
def system():
    from huggingface_hub.constants import HF_HUB_CACHE

    hub = Path(HF_HUB_CACHE)
    hub.mkdir(parents=True, exist_ok=True)
    return {
        "memory_total": total_memory() or 0,
        "memory_in_use": state.mlx_active,
        "mlx_cache": state.mlx_cache,
        "mlx_peak": state.mlx_peak,
        "pid": os.getpid(),
        "disk_free": shutil.disk_usage(hub).free,
        "models_dir": str(hub),
    }


# --- models ------------------------------------------------------------------------------------


@app.get("/models")
def list_models():
    loaded_snapshot = _loaded_snapshot()
    usage = disk_usage()
    models = catalog.all_models()
    timing_data = timings._read()
    return [_model_json(spec, usage, loaded_snapshot, models, timing_data) for spec in models]


def _spec(model_id: str) -> catalog.ModelSpec:
    spec = catalog.get(model_id)
    if spec is None:
        raise HTTPException(404, f"Unknown model {model_id}")
    return spec


@app.post("/models/{model_id}/download")
def download_model(model_id: str):
    if state.offline:
        raise HTTPException(409, "Offline mode is on")
    spec = _spec(model_id)
    if not spec.sources:
        raise HTTPException(400, "This model is a local folder; there is nothing to download.")
    try:
        downloads.start(spec)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {"ok": True}


@app.post("/models/{model_id}/cancel")
def cancel_download(model_id: str):
    downloads.cancel(_spec(model_id))
    return {"ok": True}


def _own_sources(spec, models=None):
    shared = {source.repo for other in (catalog.all_models() if models is None else models)
              if other.id != spec.id for source in other.sources}
    return [source for source in spec.sources if source.repo not in shared
            and not local_models_using([_repo_dir(source.repo)], exclude=spec.id, models=models)]


@app.post("/models/{model_id}/delete")
def delete_model(model_id: str, req: DeleteRequest = DeleteRequest()):
    spec = _spec(model_id)
    if req.scope not in ("all", "own"):
        raise HTTPException(400, "scope must be all or own")
    _acquire("cleanup")
    try:
        targets = [spec] if req.scope == "own" else [spec, *filter(None, map(catalog.get, shared_with(spec)))]
        delete_spec = replace(spec, sources=_own_sources(spec)) if req.scope == "own" else spec
        paths = [_repo_dir(source.repo) for source in delete_spec.sources]
        for other in targets:
            paths.extend(_converted_folders(other))
            if req.scope == "own" and other.optimizable and len(optimized.model_ids(other)) > 1:
                continue
            for root in optimized.copies(other):
                paths.extend((root, root.with_name(root.name + ".partial"), root.with_name(root.name + ".old")))
        protected = local_models_using(paths, exclude=spec.id)
        if protected:
            names = ", ".join(m.name for m in protected)
            raise HTTPException(409, f"Cannot delete files used by local custom model(s): {names}. Remove those entries first.")
        model_key = state.model_key
        if model_key and model_key[0] in {m.id for m in targets}:
            on_mlx(_unload)
        result = downloads.delete(delete_spec)
        removed = set()
        for other in targets:
            for root in _converted_folders(other):
                result["freed"] += optimized.disk_bytes(root)
                shutil.rmtree(root, ignore_errors=True)
            if req.scope == "own" and other.optimizable and len(optimized.model_ids(other)) > 1:
                continue
            for root in optimized.copies(other):
                for copy in (root, root.with_name(root.name + ".partial"), root.with_name(root.name + ".old")):
                    if copy not in removed:
                        result["freed"] += optimized.disk_bytes(copy)
                        shutil.rmtree(copy, ignore_errors=True)
                        removed.add(copy)
        if spec.custom:
            catalog.remove_custom(model_id)
        return result
    finally:
        optimized.invalidate_caches()
        _release()


@app.post("/models/{model_id}/load")
def load_model(model_id: str, req: LoadRequest):
    spec = _spec(model_id)
    _acquire("load")
    try:
        on_mlx(_load, spec, req.quantize, req.lora_path, req.lora_scale)
        return {"ok": True, "memory_in_use": state.mlx_active}
    finally:
        state.phase, state.message = "idle", "Ready"
        _release()


@app.post("/models/{model_id}/optimize")
def optimize_model(model_id: str, req: OptimizeRequest):
    spec = _spec(model_id)
    if not spec.optimizable or req.quantize not in spec.quant_options:
        raise HTTPException(400, "This model cannot be optimized at that precision.")
    _acquire("optimize")
    try:
        return on_mlx(_optimize, spec, _quant(spec, req.quantize))
    finally:
        state.phase, state.message = "idle", "Ready"
        _release()


def _optimize(spec, quant):
    assert_mlx_thread()
    _begin_job()
    pipeline, failure = None, None
    try:
        existing = optimized.usable(spec, quant)
        if existing:
            return {"ok": True, "path": str(existing)}
        lora_repos = {repo for repo, _ in spec.builtin_loras}
        base_spec = replace(spec, sources=[s for s in spec.sources if s.repo not in lora_repos])
        if not downloads.status(base_spec)["downloaded"]:
            raise HTTPException(409, "Download the model before optimizing it.")
        require_disk(optimized.folder(spec, quant), int(spec.memory_gb.get(str(quant or 0), 20) * 1e9))
        _unload()
        state.phase, state.message = "optimizing", f"Optimizing {spec.name}…"
        # ModelSaver bakes LoRAs, so use a separate base-only pipeline for shared saves.
        pipeline = catalog.Pipeline(spec, quant, None, 1.0, progress=_conversion_progress, skip_loras=True)
        materialize(pipeline.model)
        state.phase = "optimizing"
        root = optimized.save(spec, quant, pipeline)
        pipeline = None
        if spec.gguf:
            for converted in _converted_folders(spec):
                shutil.rmtree(converted, ignore_errors=True)
        return {"ok": True, "path": str(root)}
    except HTTPException as e:
        failure = HTTPException(e.status_code, e.detail, headers=e.headers)
        _clear_exception_frames(e)
    except ValueError as e:
        failure = HTTPException(400, str(e))
        _clear_exception_frames(e)
    except Exception as e:
        log.exception("Optimization failed")
        failure = HTTPException(500, _friendly_error(e))
        _clear_exception_frames(e)
    finally:
        _clear_job_references(pipeline)
        pipeline = None
        state.mlx_job_active = False
        _unload()
        _apply_memory_policy(active=False)
        optimized.invalidate_caches()
        state.last_activity = time.time()
    raise failure from None


@app.post("/unload")
def unload_model():
    _acquire("unload")
    try:
        on_mlx(_unload)
        state.auto_unloaded = False
        return {"ok": True}
    finally:
        _release()


@app.post("/models/custom")
def add_custom_model(req: CustomModelRequest):
    _acquire("cleanup")
    try:
        try:
            spec = catalog.add_custom(req.name, req.path, req.architecture)
        except ValueError as e:
            raise HTTPException(400, str(e))
        model_key = state.model_key
        if model_key and model_key[0] == spec.id:
            on_mlx(_unload)
        return _model_json(spec, disk_usage())
    finally:
        _release()


@app.get("/architectures")
def architectures():
    return [{"id": k, "name": v["name"]} for k, v in catalog.ARCHITECTURES.items()]


# --- token -------------------------------------------------------------------------------------


class TokenRequest(BaseModel):
    token: str | None = None


@app.post("/token")
def set_token(req: TokenRequest):
    """Sets (or clears) the Hugging Face token used for downloads and reports what it can access."""
    from huggingface_hub import auth_check, whoami
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    token = (req.token or "").strip()
    if not token:
        os.environ.pop("HF_TOKEN", None)
        return {"valid": False, "user": None, "gated_access": {}}
    os.environ["HF_TOKEN"] = token
    try:
        user = whoami(token=token).get("name")
    except HfHubHTTPError as e:
        if e.response is not None and e.response.status_code == 401:
            # A rejected token would also break downloads of public models, so stop using it.
            os.environ.pop("HF_TOKEN", None)
            return {"valid": False, "user": None, "gated_access": {}}
        return {"valid": None, "user": None, "gated_access": {}}
    except Exception:  # offline: keep the token, it may be fine
        return {"valid": None, "user": None, "gated_access": {}}
    access = {}
    for spec in catalog.CATALOG:
        if not spec.gated:
            continue
        try:
            auth_check(spec.sources[0].repo, token=token)
            access[spec.id] = True
        except GatedRepoError:
            access[spec.id] = False
        except Exception:
            access[spec.id] = None
    return {"valid": True, "user": user, "gated_access": access}


# --- generation --------------------------------------------------------------------------------


def _model_key(spec, quant, loras):
    return (spec.id, quant, tuple((item["path"], item["scale"]) for item in loras))


def normalize_request(req: GenerateRequest, require_prompt: bool = True):
    spec = _spec(req.model)
    if spec.tool:
        raise HTTPException(400, f"{spec.name} is a tool, not an image generator.")
    if require_prompt and not req.prompt.strip():
        raise HTTPException(400, "Prompt is empty")
    for name, lo, hi in (("width", 256, 2048), ("height", 256, 2048), ("steps", 1, 100),
                         ("count", 1, 16), ("guidance", 0, 20)):
        value = getattr(req, name)
        if not math.isfinite(value) or not lo <= value <= hi:
            raise HTTPException(400, f"{name} must be between {lo} and {hi}")
    if req.width * req.height / 1e6 > spec.max_megapixels:
        raise HTTPException(400, f"{spec.name} supports up to {spec.max_megapixels:g} megapixels; reduce width or height.")
    if req.strength is not None and (not math.isfinite(req.strength) or not 0 <= req.strength <= 1):
        raise HTTPException(400, "strength must be between 0 and 1")
    if req.seed is not None and req.seed < 0:
        raise HTTPException(400, "seed must be at least 0 or null")
    paths = req.image_paths[:spec.max_refs] if spec.edit != "none" else []
    mode = req.image_mode if req.image_mode in spec.image_modes else spec.image_modes[0]
    if req.mask_path is not None or req.auto_mask is not None:
        if not spec.supports_mask:
            raise HTTPException(400, "This model does not support masks")
        if not paths:
            raise HTTPException(400, "A mask needs at least one reference image")
        if req.auto_mask is not None and not req.auto_mask.strip():
            raise HTTPException(400, "Mask description is empty")
    if req.enhance_prompt and spec.supports_enhance and not paths:
        raise HTTPException(400, "Prompt enhancement needs at least one reference image")
    loras = [item.model_dump() for item in req.loras]
    if not loras and req.lora_path:
        loras = [{"path": req.lora_path, "scale": req.lora_scale}]
    for item in loras:
        if not item["path"].strip() or not math.isfinite(item["scale"]):
            raise HTTPException(400, "Each LoRA needs a path and a finite scale")
    strength = spec.default_strength if req.strength is None else req.strength
    if paths and (mode == "init" or spec.family in ("zimage", "krea2")):
        strength = max(0.05, strength)
        if spec.family in ("flux2", "zimage", "krea2"):
            strength = min(0.95, strength)
    guidance = max(spec.min_guidance, req.guidance) if spec.supports_guidance else spec.default_guidance
    steps = spec.default_steps if spec.fixed_steps else max(2 if paths else 1, req.steps)
    normalized = req.model_copy(update={
        "width": req.width // spec.size_multiple * spec.size_multiple,
        "height": req.height // spec.size_multiple * spec.size_multiple,
        "steps": steps, "guidance": guidance, "strength": strength, "image_mode": mode,
        "quantize": _quant(spec, req.quantize) or 0, "loras": [LoraRequest(**item) for item in loras],
        "lora_path": loras[0]["path"] if len(loras) == 1 else None,
        "lora_scale": loras[0]["scale"] if len(loras) == 1 else 1.0,
        "fast": req.fast and spec.supports_step_cache,
        "enhance_prompt": req.enhance_prompt and spec.supports_enhance,
        "negative_prompt": req.negative_prompt if spec.supports_negative and spec.passes_per_step(guidance) > 1 else None,
    })
    return normalized, spec, paths


@app.post("/estimate")
def estimate(req: EstimateRequest):
    req, spec, paths = normalize_request(req, require_prompt=False)  # asked before a prompt is typed
    kind = timings.kind_of(paths, req.image_mode)
    passes = spec.passes_per_step(req.guidance)
    run = timings.steps_run(spec, req.steps, req.strength, kind)
    key = _model_key(spec, _quant(spec, req.quantize), [item.model_dump() for item in req.loras])
    result = timings.estimate(spec.id, _quant(spec, req.quantize), kind, run, passes,
                              req.width * req.height / 1e6, len(paths), req.count, needs_load=state.model_key != key)
    return {**result, "steps_run": run, "passes": passes}


def _preflight(req, spec, paths):
    snapshot = memory_snapshot()
    if state.pressure is not None:
        snapshot["pressure"] = "ok" if state.pressure == "normal" else state.pressure
    snapshot["reusable_bytes"] = state.mlx_active + state.mlx_cache
    kind = timings.kind_of(paths, req.image_mode)
    figures = _memory_gb(spec, req.width * req.height / 1e6, kind, low_memory=state.low_memory == "always")
    return preflight_memory(figures, _quant(spec, req.quantize), snapshot)


def _memory_gb(spec: catalog.ModelSpec, megapixels: float = 1.0, kind: str | None = None,
               low_memory: bool = False, timing_data: dict | None = None) -> dict:
    """Peak memory per precision: measured on this Mac where available (scaled to the size), else the catalog's."""
    figures = dict(spec.memory_gb)
    for quant in spec.memory_gb:
        measured = timings.measured_peak_gb(spec.id, int(quant), megapixels, kind, low_memory, data=timing_data)
        if measured is not None:
            figures[quant] = measured
    return figures


@app.post("/preflight")
def preflight(req: GenerateRequest):
    req, spec, paths = normalize_request(req)
    return _preflight(req, spec, paths)


@app.get("/preview")
def preview():
    if not state.preview:
        raise HTTPException(404, "No preview")
    return Response(content=state.preview, media_type="image/jpeg")


@app.get("/settings")
def get_settings():
    return {"idle_unload_minutes": state.idle_unload_minutes, "live_preview": state.live_preview,
            "embed_metadata": state.embed_metadata, "low_memory": state.low_memory, "offline": state.offline}


@app.post("/settings")
def set_settings(req: SettingsRequest):
    if req.low_memory is not None and req.low_memory not in ("auto", "always", "never"):
        raise HTTPException(400, "low_memory must be auto, always or never")
    if req.idle_unload_minutes is not None:
        state.idle_unload_minutes = max(0, req.idle_unload_minutes)
        state.last_activity = time.time()
    if req.live_preview is not None:
        state.live_preview = req.live_preview
    if req.embed_metadata is not None:
        from mflux.utils.image_util import ImageUtil

        state.embed_metadata = req.embed_metadata
        ImageUtil.embed_metadata_enabled = req.embed_metadata
    if req.low_memory is not None:
        state.low_memory = req.low_memory
    if req.offline is not None:
        from huggingface_hub import constants

        state.offline = req.offline
        if req.offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
        else:
            os.environ.pop("HF_HUB_OFFLINE", None)
        constants.HF_HUB_OFFLINE = req.offline
    return get_settings()


def _idle_unloader():
    """Frees the loaded model after the configured idle time; the next generation reloads it."""
    while not _shutdown.wait(15):
        minutes = state.idle_unload_minutes
        critical = _pressure_level() == "critical"
        if state.pipeline is None or (not critical and (not minutes or time.time() - state.last_activity < minutes * 60)):
            continue
        if not state.lock.acquire(blocking=False):
            continue
        state.busy = "unload"
        try:
            on_mlx(_handle_idle_pressure)
            # Re-check under the lock: a generation may have just finished or the setting changed.
            minutes = state.idle_unload_minutes
            if minutes and state.pipeline is not None and time.time() - state.last_activity >= minutes * 60:
                log.info("Unloading %s after %d idle minutes", state.model_key[0], minutes)
                on_mlx(_unload)
                state.auto_unloaded = True
                state.message = f"Model unloaded after {minutes} idle minutes to free memory"
        finally:
            _release()


@app.post("/cancel")
def cancel():
    state.cancel.set()
    return {"ok": True}


def _check_input(path):
    if not Path(path).is_file():
        raise HTTPException(400, f"Input image not found: {path}")


@app.post("/generate")
def generate(req: GenerateRequest):
    req, spec, paths = normalize_request(req)
    for path in [*paths, *([req.mask_path] if req.mask_path is not None else [])]:
        _check_input(path)
    _acquire("generate")
    started = time.time()
    try:
        state.cancel.clear()
        state.images_done = []
        state.preview = None
        state.image_index, state.image_count = 0, req.count
        state.step, state.total_steps = 0, req.steps
        state.passes = spec.passes_per_step(req.guidance)
        warning = _preflight(req, spec, paths)
        state.job_low_memory = warning["level"] != "ok"
        state.memory_warning = warning["message"] if warning["level"] != "ok" else None
        if state.memory_warning:
            log.warning(state.memory_warning)
        # Clear cancellation before reference preparation, and preserve the user's original paths.
        state.phase, state.message = "encoding", "Preparing reference images…"
        prepared = []
        for path in paths:
            _check_cancel()
            prepared.append(_prepare_reference(path, spec.max_megapixels))
        _check_cancel()
        return on_mlx(_generate, req, spec, prepared, req.image_mode)
    except StopImageGenerationException:
        return {"images": list(state.images_done), "cancelled": True,
                "seconds": round(time.time() - started, 1), "error": None}
    finally:
        state.phase, state.message = "idle", "Ready"
        state.passes = 1
        state.preview = None
        _release()


def _sidecar(req, spec, pipeline, seed, elapsed):
    bits = getattr(pipeline.model, "bits", _quant(spec, req.quantize))
    quant = bits if isinstance(bits, int) else 0
    data = {**req.model_dump(), "quantize": quant, "seed": seed, "seconds": round(elapsed, 1)}
    data["effective"] = {
        "quantize": quant, "guidance": req.guidance, "steps": req.steps, "strength": req.strength,
        "width": req.width, "height": req.height,
        "builtin_loras": [{"repo": repo, "filename": name, "scale": 1.0} for repo, name in spec.builtin_loras],
        "scheduler": spec.scheduler or {"flux2": "flow_match_euler_discrete", "qwen21": "linear",
                                        "zimage": "linear", "krea2": "er_sde"}.get(spec.family),
        "mflux_version": optimized.mflux_version(), "revision": optimized.base_snapshot(spec),
    }
    return data


def _clear_job_references(pipeline):
    assert_mlx_thread()
    if pipeline is not None:
        end = getattr(pipeline, "end_job", None)
        if end is not None:
            end()
        callback = getattr(pipeline, "progress", None)
        if callback is not None:
            callback.prev = None


def _finish_job():
    """Called only after the job's locals and callback/cache references are gone."""
    assert_mlx_thread()
    state.mlx_job_active = False
    state.job_low_memory = False
    if state.encoder_evicted or _pressure_level() == "critical":
        _unload(cleanup=False)
    state.last_activity = time.time()
    gc.collect()
    _apply_memory_policy(active=False, flush=False)
    mx.clear_cache()
    _sample_memory()


def _save_result(result, path):
    if state.embed_metadata:
        result.save(path=path, export_json_metadata=False, overwrite=False)
        return verify_png(path)
    return save_png(result.image, path)


def _generate(req: GenerateRequest, spec: catalog.ModelSpec, image_paths: list[str], image_mode: str) -> dict:
    assert_mlx_thread()
    _begin_job()
    mx.reset_peak_memory()
    pipeline = None
    results, error = [], None
    started = time.time()
    try:
        _check_cancel()
        pipeline = _load(spec, req.quantize, None, 1.0, [item.model_dump() for item in req.loras])
        if hasattr(pipeline, "begin_job"):
            pipeline.begin_job(image_paths)
        _check_cancel()
        base_seed = req.seed if req.seed is not None else random.randint(0, 2**31 - 1)
        out_dir = OUTPUT_ROOT / datetime.now().strftime("%Y-%m-%d")
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(req.count):
            _check_cancel()
            seed = base_seed + i
            state.image_index = i + 1
            state.step, state.total_steps = 0, req.steps
            state.phase, state.message = "encoding", f"Image {i + 1}/{req.count}"
            t0 = time.time()
            state.loop_started = state.loop_ended = 0.0
            state.loop_steps = 0
            state.preview = None
            try:
                image = pipeline.generate(
                    seed=seed, prompt=req.prompt, steps=req.steps, width=req.width, height=req.height,
                    guidance=req.guidance, image_paths=image_paths, strength=req.strength,
                    negative_prompt=req.negative_prompt, image_mode=image_mode,
                    fast=req.fast, mask_path=req.mask_path, auto_mask=req.auto_mask, enhance_prompt=req.enhance_prompt,
                )
                _check_cancel()
                state.phase = "saving"
                path = unique_path(out_dir / f"{datetime.now().strftime('%H%M%S')}_{seed}.png")
                _save_result(image, path)
                elapsed = time.time() - t0
                meta = _sidecar(req, spec, pipeline, seed, elapsed)
                if req.enhance_prompt:
                    meta["prompt"] = getattr(image, "prompt", req.prompt)
                atomic_json(path.with_suffix(".json"), meta)
                results.append({"path": str(path), "seed": seed, "seconds": round(elapsed, 1)})
                state.images_done = list(results)
                if state.loop_steps and state.loop_ended > state.loop_started > 0:
                    denoise = state.loop_ended - state.loop_started - state.preview_time
                    timings.record_generation(
                        spec.id, _quant(spec, req.quantize), timings.kind_of(image_paths, image_mode), state.loop_steps,
                        state.passes, req.width * req.height / 1e6, len(image_paths), denoise,
                        elapsed - state.preview_time - denoise)
                    # The real peak so far in this job (the counter is reset when the job starts) feeds
                    # the memory warnings instead of the catalog's rough figures.
                    timings.record_peak(spec.id, _quant(spec, req.quantize), int(mx.get_peak_memory()),
                                        req.width * req.height / 1e6, timings.kind_of(image_paths, image_mode),
                                        low_memory=state.encoder_evicted)
            except StopImageGenerationException:
                state.cancel.set()
                break
            except Exception as e:
                log.exception("Generation with %s failed", spec.id)
                error = f"Generation failed: {_friendly_error(e)}"
                _clear_exception_frames(e)
                break
    except StopImageGenerationException:
        state.cancel.set()
    finally:
        _clear_job_references(pipeline)
        pipeline = image = None
        _finish_job()
    return {"images": results, "cancelled": state.cancel.is_set(),
            "seconds": round(time.time() - started, 1), "error": error}


def _busy_repos() -> set[str]:
    return {repo for job in downloads._jobs() if job.running for repo in job.repos}


@app.post("/storage/scan")
def storage_scan(req: StorageRequest):
    return {"items": storage.public(storage.scan(set(req.keep), _busy_repos(), protected=_protected_paths()))}


@app.post("/storage/clean")
def storage_clean(req: StorageRequest):
    _acquire("cleanup")
    # Reserve every repo we might touch so no download can start on it while we delete.
    with downloads._lock:
        busy = {repo for job in downloads.jobs if job.running for repo in job.repos}
        reserved = {i.split(":", 1)[1] for i in req.ids if i.startswith(("partial:", "hf:"))} - busy
        downloads._deleting |= reserved
    try:
        return storage.clean(set(req.ids), set(req.keep), busy, protected=_protected_paths())
    finally:
        with downloads._lock:
            downloads._deleting -= reserved
        _release()


def _protected_paths() -> set[str]:
    """Files of custom models (local folders or anything inside the HF cache) are never cleanup targets."""
    return {str(Path(spec.model_path).resolve()) for spec in catalog.all_models()
            if spec.model_path and os.path.isabs(spec.model_path)}


@app.post("/upscale")
def upscale(req: UpscaleRequest):
    if not math.isfinite(req.scale) or not 1.5 <= req.scale <= 4:
        raise HTTPException(400, "scale must be between 1.5 and 4")
    if not math.isfinite(req.softness) or not 0 <= req.softness <= 1:
        raise HTTPException(400, "softness must be between 0 and 1")
    spec = next((m for m in catalog.CATALOG if m.family == "seedvr2"), None)
    source = Path(req.path)
    if spec is None or not source.is_file():
        raise HTTPException(400, f"Image not found: {req.path}")
    if not downloads.status(spec)["downloaded"]:
        raise HTTPException(409, f"Download {spec.name} in the Model Manager first.")
    from PIL import Image

    try:
        with Image.open(source) as im:
            w, h = im.size
    except OSError as e:
        raise HTTPException(400, _friendly_error(e))
    if w * h * req.scale * req.scale > 4.5e6:
        raise HTTPException(400, "Upscale output must be at most 4.5 megapixels; reduce the scale.")
    try:
        require_disk(source.parent, int(w * h * req.scale * req.scale * 16))
    except ValueError as e:
        raise HTTPException(400, str(e))
    _acquire("upscale")
    try:
        state.cancel.clear()
        state.images_done, state.preview = [], None
        state.memory_warning = None
        state.image_index, state.image_count = 1, 1
        return on_mlx(_upscale, spec, req, source)
    finally:
        state.phase, state.message = "idle", "Ready"
        state.passes, state.preview = 1, None
        _release()


def _upscale(spec: catalog.ModelSpec, req: UpscaleRequest, source: Path) -> dict:
    assert_mlx_thread()
    _begin_job()
    mx.reset_peak_memory()
    pipeline, failure = None, None
    t0 = time.time()
    try:
        _check_cancel()
        pipeline = _load(spec, req.quantize, None, 1.0)
        _check_cancel()
        prepared = _prepare_reference(str(source), spec.max_megapixels)
        _check_cancel()
        state.phase, state.message = "upscaling", f"Upscaling {req.scale:g}x…"
        state.step, state.total_steps = 0, 1
        result = pipeline.upscale(image_path=prepared, scale=req.scale,
                                  softness=req.softness, seed=random.randint(0, 2**31 - 1))
        _check_cancel()
        state.phase = "saving"
        out = unique_path(source.with_name(f"{source.stem}_up{req.scale:g}x.png"))
        width, height = _save_result(result, out)
        try:
            meta = json.loads(source.with_suffix(".json").read_text())
        except (OSError, ValueError):
            meta = {}
        elapsed = round(time.time() - t0, 1)
        meta.update({"upscaled_from": str(source), "upscale": req.scale, "softness": req.softness,
                     "output_width": width, "output_height": height, "seconds": elapsed})
        atomic_json(out.with_suffix(".json"), meta)
        state.images_done = [{"path": str(out), "seed": meta.get("seed", 0), "seconds": elapsed}]
        return {"path": str(out), "seconds": elapsed, "width": width, "height": height,
                "seed": meta.get("seed", 0), "cancelled": False}
    except StopImageGenerationException:
        state.cancel.set()
        return {"cancelled": True, "seconds": round(time.time() - t0, 1)}
    except Exception as e:
        log.exception("Upscaling %s failed", source)
        failure = HTTPException(500, f"Upscaling failed: {_friendly_error(e)}")
        _clear_exception_frames(e)
    finally:
        _clear_job_references(pipeline)
        pipeline = result = None
        _finish_job()
    raise failure from None


def _exit_when_parent_dies(parent_pid: int):
    while True:
        time.sleep(2)
        if os.getppid() != parent_pid:
            downloads.cancel_all()
            os._exit(0)


def _serve(port):
    app.state.port = port
    bound = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        bound.bind(("127.0.0.1", port))
        bound.listen(128)
    except OSError:
        bound.close()
        sys.exit(98)
    try:
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        uvicorn.Server(config).run(sockets=[bound])
    finally:
        bound.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()
    threading.Thread(target=_exit_when_parent_dies, args=(os.getppid(),), daemon=True).start()
    _serve(args.port)
