#!/usr/bin/env python3
"""Local-only count=2 model benchmark. --compare does not import MLX.

Use a fresh process/output directory for each run. Model defaults are used
unless overridden; init inputs run the app's default strength/skipped schedule.
"""

import argparse
import gc
import io
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, redirect_stdout
from pathlib import Path


MODELS = ("flux2-klein-4b", "flux2-klein-base-4b", "flux2-klein-9b", "qwen-image-2.1-turbo", "qwen-image-2.1",
          "qwen-image-2.1-uncensored-turbo", "qwen-image-2.1-uncensored", "z-image-turbo", "krea-2-turbo")


def pixel_difference(a, b):
    import numpy as np
    from PIL import Image

    with Image.open(a) as image:
        left = np.asarray(image.convert("RGB"), dtype=np.float64)
    with Image.open(b) as image:
        right = np.asarray(image.convert("RGB"), dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(f"Image dimensions differ: {a}: {left.shape}, {b}: {right.shape}")
    diff = np.abs(left - right)
    squared = float(np.sum(diff * diff))
    mse = squared / diff.size
    return {"max_abs_pixel_difference": int(diff.max()),
            "mean_abs_pixel_difference": float(diff.mean()),
            "psnr_db": 10 * math.log10(255**2 / mse) if mse else "Infinity",
            "identical": squared == 0, "values": int(diff.size),
            "absolute_sum": float(diff.sum()), "squared_sum": squared}


def compare_dirs(a, b):
    a, b = Path(a), Path(b)
    left = {p.relative_to(a) for p in a.rglob("*.png")}
    right = {p.relative_to(b) for p in b.rglob("*.png")}
    if not left or left != right:
        raise ValueError(f"PNG sets must be nonempty and identical; only A: {sorted(map(str, left-right))}; "
                         f"only B: {sorted(map(str, right-left))}")
    for name in sorted(left):
        result = pixel_difference(a / name, b / name)
        for key in ("values", "absolute_sum", "squared_sum"):
            result.pop(key)
        yield {"type": "comparison", "file": str(name), **result}


def find_references(paths):
    if paths:
        references = [Path(p).expanduser().resolve() for p in paths]
    else:
        references = sorted(Path("test-results").glob("*.jpg"))
        if len(references) < 2:
            root = Path.home() / "Pictures/Image Studio"
            references += sorted(p for p in root.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        references = [p.resolve() for p in references[:2]]
    for path in references:
        if not path.is_file():
            raise ValueError(f"Reference does not exist: {path}")
    return references


class RunTimings:
    def __init__(self, pipeline, mx, args):
        self.pipeline, self.mx, self.args = pipeline, mx, args
        self.preview_cost = 0.0
        self.index = 0

    def reset(self):
        self.started = time.perf_counter()
        self.encode_s = self.decode_s = self.preview_s = 0.0
        self.step_s, self.step_indices = [], []
        self.preview_count = 0
        self.prev = None
        self.in_preview = False

    def call_before_loop(self, latents, config, **kwargs):
        self.mx.eval(latents)
        encodings = getattr(self.pipeline, "prompt_encodings", None)
        reusable = encodings is not None and encodings.reusable
        if self.args.low_memory == "always" and (reusable or self.index == self.args.count - 1):
            from mflux.callbacks.instances.memory_saver import MemorySaver

            for model in (self.pipeline.model, getattr(self.pipeline, "edit", None)):
                if model is not None:
                    MemorySaver(model, cache_limit_bytes=None)._delete_text_encoders()
            gc.collect()
            self.mx.clear_cache()
        self.encode_s = time.perf_counter() - self.started
        self.skipped = getattr(config, "init_time_step", 0) or 0
        self.prev = latents
        self.last_preview = self.step_started = time.perf_counter()

    def call_in_loop(self, t, latents, config, denoised=None, **kwargs):
        self.mx.eval(latents)
        now = time.perf_counter()
        self.step_s.append(now - self.step_started)
        self.step_indices.append(t)
        prev, self.prev = self.prev, latents
        step, total = t + 1 - self.skipped, config.num_inference_steps - self.skipped
        due = now - self.last_preview >= max(2.0, 5 * self.preview_cost)
        halfway = self.preview_count == 0 and step * 2 >= total
        if self.args.live_preview and step < total and (due or halfway):
            start = time.perf_counter()
            target = denoised
            if target is None:
                target = latents
                sigmas = config.scheduler.sigmas
                sigmas = sigmas() if callable(sigmas) else sigmas
                s_now, s_next = float(sigmas[t].item()), float(sigmas[t + 1].item())
                if abs(s_next - s_now) >= 1e-6:
                    target = latents - s_next * ((latents - prev) / (s_next - s_now))
            self.in_preview = True
            try:
                image = self.pipeline.preview(target, config)
                buffer = io.BytesIO()
                image.convert("RGB").save(buffer, "JPEG", quality=80)
            finally:
                self.in_preview = False
            cost = time.perf_counter() - start
            self.preview_cost = cost if not self.preview_cost else (self.preview_cost + cost) / 2
            self.preview_s += cost
            self.preview_count += 1
            self.last_preview = time.perf_counter()
        self.step_started = time.perf_counter()

    def call_after_loop(self, latents, **kwargs):
        self.prev = None


@contextmanager
def timed_vae_decode(timer, mx):
    from mflux.models.common.vae.vae_util import VAEUtil

    descriptor = VAEUtil.__dict__["decode"]
    original = VAEUtil.decode

    def decode(*args, **kwargs):
        start = time.perf_counter()
        result = original(*args, **kwargs)
        mx.eval(result)
        if not timer.in_preview:
            timer.decode_s += time.perf_counter() - start
        return result

    VAEUtil.decode = staticmethod(decode)
    try:
        yield
    finally:
        VAEUtil.decode = descriptor


def local_model_path(spec, quant):
    import catalog
    import optimized
    from downloads import _snapshot

    checkpoint = optimized.usable(spec, quant)
    if checkpoint is not None:
        return checkpoint, checkpoint
    if spec.gguf:
        # Never call converted_folder/build_model_folder: even recovery can write.
        from gguf_convert import _usable, _stamp

        base, gguf = _snapshot(spec.sources[0].repo), _snapshot(spec.gguf[0])
        target = catalog.CONVERTED_DIR / spec.id
        if base and gguf and (gguf / spec.gguf[1]).is_file():
            if _usable(target, _stamp(base, (gguf / spec.gguf[1]).resolve())):
                return target, None
        raise ValueError("Z-Image needs an existing valid optimized or converted copy; no copies are created")
    root = _snapshot(spec.sources[0].repo)
    if root is None:
        raise ValueError(f"{spec.id} is not downloaded locally; this benchmark never downloads weights")
    return root, None


def validate_cases(spec, args):
    if spec.family in ("zimage", "krea2"):
        defaults = ["text", "init"]
    elif spec.id == "flux2-klein-9b":
        defaults = ["text", "edit1", "edit2"]
    else:
        defaults = ["text", "edit1"]
    cases = args.cases or defaults
    if len(cases) != len(set(cases)):
        raise ValueError("Each case may be selected only once")
    if args.low_memory == "always" and len(cases) != 1:
        raise ValueError("--low-memory always requires one --cases value; run each case in a fresh process")
    if "init" in cases and spec.family not in ("zimage", "krea2"):
        raise ValueError("init is supported here for Z-Image/Krea; use edit1 for FLUX/Qwen")
    if "edit2" in cases and spec.max_refs < 2:
        raise ValueError("edit2 requires a model supporting at least two references")
    return cases


def run_benchmark(args, references):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import mlx.core as mx
    import catalog
    from core import save_png
    from performance import materialize

    spec = catalog.get(args.model, include_custom=False)
    cases = validate_cases(spec, args)
    needed = 2 if "edit2" in cases else int(any(c != "text" for c in cases))
    if len(references) < needed:
        raise ValueError(f"Need {needed} reference images; supply --refs PATH ...")
    steps = args.steps if args.steps is not None else spec.default_steps
    guidance = args.guidance if args.guidance is not None else spec.default_guidance
    if not spec.supports_guidance:
        guidance = spec.default_guidance
    if spec.family == "qwen21":
        guidance = max(1.0, guidance)
    model_path, checkpoint = local_model_path(spec, 4)
    # Resolve builtin LoRAs locally even when HF was imported before this module.
    hub_download = catalog.hf_hub_download

    def local_download(*args, **kwargs):
        kwargs["local_files_only"] = True
        return hub_download(*args, **kwargs)

    for repo, filename in spec.builtin_loras:
        local_download(repo, filename)
    output = Path(args.output).expanduser().resolve()
    for case in cases:
        for index in range(args.count):
            filename = output / f"{case}_{index}_{args.seed + index}.png"
            if filename.exists():
                raise ValueError(f"Output already exists: {filename}; choose a new --output directory")
    output.mkdir(parents=True, exist_ok=True)
    mx.set_memory_limit(16 * 1024**3)
    mx.set_wired_limit(0)
    mx.set_cache_limit(args.cache_limit_mb * 1024**2)
    mx.reset_peak_memory()
    started = time.perf_counter()
    catalog.hf_hub_download = local_download
    try:
        with redirect_stdout(sys.stderr):
            pipeline = catalog.Pipeline(spec, 4, None, 1.0, model_path=str(model_path))
            materialize(pipeline.model)
    finally:
        catalog.hf_hub_download = hub_download
    load_s = time.perf_counter() - started
    load_peak = int(mx.get_peak_memory())
    timer = RunTimings(pipeline, mx, args)
    pipeline.register(timer)
    for case in cases:
        ref_count = 2 if case == "edit2" else 1
        paths = [] if case == "text" else [str(p) for p in references[:ref_count]]
        image_mode = "init" if spec.family in ("zimage", "krea2") else "reference"
        prompt = args.prompt if case == "text" or image_mode == "init" else args.edit_prompt
        pipeline.begin_job(paths)
        try:
            for index in range(args.count):
                seed = args.seed + index
                timer.index = index
                mx.reset_peak_memory()
                timer.reset()
                with timed_vae_decode(timer, mx), redirect_stdout(sys.stderr):
                    image = pipeline.generate(seed=seed, prompt=prompt, steps=steps, width=1024, height=1024,
                                              guidance=guidance, image_paths=paths, strength=args.strength,
                                              negative_prompt=args.negative_prompt, image_mode=image_mode)
                generation_s = time.perf_counter() - timer.started
                save_start = time.perf_counter()
                filename = output / f"{case}_{index}_{seed}.png"
                save_png(image.image, filename)
                save_s = time.perf_counter() - save_start
                charged_load = load_s if case == cases[0] and index == 0 else 0.0
                record = {"type": "run", "model": spec.id, "case": case, "image_mode": image_mode,
                          "index": index, "seed": seed, "count": args.count, "quantize": 4,
                          "steps": steps, "steps_run": len(timer.step_s), "step_indices": timer.step_indices,
                          "width": 1024, "height": 1024, "guidance": guidance,
                          "strength": args.strength if args.strength is not None else spec.default_strength,
                          "prompt": prompt, "negative_prompt": args.negative_prompt, "references": paths,
                          "live_preview": args.live_preview, "low_memory": args.low_memory,
                          "cache_limit_mb": args.cache_limit_mb, "model_path": str(model_path),
                          "optimized_copy": str(checkpoint) if checkpoint else None,
                          "load_s": charged_load, "pipeline_load_s": load_s, "load_peak_memory_bytes": load_peak,
                          "encode_s": timer.encode_s, "per_step_s": timer.step_s, "decode_s": timer.decode_s,
                          "preview_s": timer.preview_s, "preview_count": timer.preview_count,
                          "save_s": save_s, "generation_s": generation_s,
                          "image_total_s": generation_s + save_s, "total_s": charged_load + generation_s + save_s,
                          "peak_memory_bytes": int(mx.get_peak_memory()),
                          "active_memory_bytes": int(mx.get_active_memory()),
                          "cache_memory_bytes": int(mx.get_cache_memory()), "png": str(filename)}
                conditioning = getattr(pipeline, "job_conditioning", None)
                if conditioning is not None:
                    record.update(condition_cache_hits=conditioning.values.hits,
                                  condition_cache_misses=conditioning.values.misses,
                                  condition_cache_bytes=conditioning.values.bytes)
                encodings = getattr(pipeline, "prompt_encodings", None)
                if encodings is not None:
                    record["prompt_cache_bytes"] = encodings.bytes
                print(json.dumps(record, allow_nan=False), flush=True)
                image = None
        finally:
            pipeline.end_job()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"))
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--guidance", type=float, help="Use >1 for an explicit full-Qwen CFG test; catalog default is 1")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--strength", type=float)
    parser.add_argument("--refs", nargs="+")
    parser.add_argument("--cases", nargs="+", choices=("text", "edit1", "edit2", "init"))
    parser.add_argument("--output", default="bench-results/models-baseline")
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--live-preview", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--low-memory", choices=("never", "always"), default="never")
    parser.add_argument("--cache-limit-mb", type=int, default=512)
    parser.add_argument("--prompt", default="A small red ceramic teapot on a wooden table, soft window light, detailed photograph")
    parser.add_argument("--edit-prompt", default="Place the reference subject on a wooden table in soft window light, detailed photograph")
    args = parser.parse_args()
    try:
        if args.compare:
            for record in compare_dirs(*args.compare):
                print(json.dumps(record, allow_nan=False), flush=True)
            return
        if args.model is None:
            raise ValueError("--model is required when generating")
        if args.count < 1 or args.cache_limit_mb < 0 or (args.steps is not None and args.steps < 1):
            raise ValueError("count/steps must be positive and cache-limit-mb nonnegative")
        # Catalog is deliberately imported only inside generation mode, after offline flags.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import catalog

        cases = validate_cases(catalog.get(args.model, include_custom=False), args)
        references = find_references(args.refs) if any(c != "text" for c in cases) else []
        if any(c != "text" for c in cases) and not references:
            raise ValueError("Need a reference image; supply --refs PATH")
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-bench") as executor:
            executor.submit(run_benchmark, args, references).result()
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
