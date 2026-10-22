"""Model catalog: what the app offers, what each model needs on disk, and how to run it with mflux."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import dataclasses
from dataclasses import asdict, dataclass, field
from pathlib import Path

from huggingface_hub import hf_hub_download

from core import assert_mlx_thread

APP_SUPPORT = Path.home() / "Library/Application Support/Image Studio"
CUSTOM_MODELS_FILE = APP_SUPPORT / "custom_models.json"
_custom_lock = threading.Lock()

FLUX2_PATTERNS = [
    "vae/*.safetensors", "vae/*.json",
    "transformer/*.safetensors", "transformer/*.json",
    "text_encoder/*.safetensors", "text_encoder/*.json",
    "tokenizer/**", "added_tokens.json", "chat_template.jinja",
]
QWEN21_PATTERNS = [
    "model_index.json", "scheduler/*.json", "processor/**",
    "vae/*.json", "vae/*.safetensors",
    "transformer/*.json", "transformer/*.safetensors",
    "text_encoder/*.json", "text_encoder/*.safetensors",
]
KREA2_TURBO_PATTERNS = [
    "turbo.safetensors",
    "vae/*.safetensors", "vae/*.json",
    "text_encoder/*.safetensors", "text_encoder/*.json",
    "tokenizer/**",
]
VIGGLE_REPO = "Viggle/Qwen-Image-2.1-viggle-turbo"
VIGGLE_LORA = "Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r128.safetensors"
# abenzerps' "UC" files are bit-for-bit bf16(Qwen-Image 2.1 + this LoRA at scale 1.0), so the LoRA on the
# base we already have gives the same model without another 4-15 GB download.
UNCENSORED_REPO = "abenzerps/Qwen-Image-2.1-Uncensored-Quantized"
UNCENSORED_LORA = "qwen-image-2.1-uncensored-lora.safetensors"
ZIMAGE_REPO = "Tongyi-MAI/Z-Image-Turbo"
# The GGUF replaces the official 24.6 GB fp32 transformer; the text encoder and VAE come from Tongyi.
ZIMAGE_PATTERNS = [
    "model_index.json", "scheduler/*.json", "transformer/config.json",
    "vae/*.safetensors", "vae/*.json",
    "text_encoder/*.safetensors", "text_encoder/*.json",
    "tokenizer/*",
]
ZIMAGE_GGUF = ("unsloth/Z-Image-Turbo-GGUF", "z-image-turbo-Q8_0.gguf")
CONVERTED_DIR = APP_SUPPORT / "converted"
SEEDVR2_REPO = "numz/SeedVR2_comfyUI"

# Architectures a custom checkpoint can be based on; "base" is the catalog entry it inherits settings from.
ARCHITECTURES = {
    "flux2-klein-4b": {"name": "FLUX.2 klein 4B", "base": "flux2-klein-4b", "patterns": FLUX2_PATTERNS},
    "flux2-klein-9b": {"name": "FLUX.2 klein 9B", "base": "flux2-klein-9b", "patterns": FLUX2_PATTERNS},
    "qwen-image-2.1": {"name": "Qwen-Image 2.1", "base": "qwen-image-2.1", "patterns": QWEN21_PATTERNS},
    "krea-2": {"name": "Krea 2 Turbo", "base": "krea-2-turbo", "patterns": KREA2_TURBO_PATTERNS},
}


@dataclass
class Source:
    repo: str
    patterns: list[str]


@dataclass
class ModelSpec:
    id: str
    name: str
    family: str  # flux2 | qwen21 | krea2 | zimage | seedvr2 (upscaler)
    mflux_name: str  # mflux registry alias used for ModelConfig
    released: str
    license: str
    commercial: bool
    gated: bool
    description: str
    sources: list[Source]
    edit: str  # "multi" (reference editing) | "single" (img2img from image 1) | "none"
    max_refs: int
    default_steps: int
    default_guidance: float
    min_guidance: float = 0.0
    # How input images can be used: "reference" = edit/combine them, "init" = start from image 1 (img2img).
    image_modes: list[str] = field(default_factory=lambda: ["reference", "init"])
    default_strength: float = 0.6  # "init" mode: how much the result may change
    fixed_steps: bool = False
    supports_negative: bool = False
    supports_strength: bool = False
    size_multiple: int = 16
    quant_options: list[int] = field(default_factory=lambda: [8, 4, 0])  # 0 = full precision
    default_quant: int = 8
    memory_gb: dict[str, float] = field(default_factory=dict)  # quant -> approx. peak GB
    is_new: bool = False
    custom: bool = False
    model_path: str | None = None  # custom checkpoints: HF repo or local folder
    builtin_loras: list[tuple[str, str]] = field(default_factory=list)  # (repo, filename)
    scheduler: str | None = None
    supports_guidance: bool = True
    # Step/guidance-distilled: tuned for guidance 1.0; more doubles the time and rarely helps.
    distilled: bool = False
    gguf: tuple[str, str] | None = None  # (repo, filename) transformer converted on first load
    tool: bool = False  # not an image generator (e.g. the upscaler); hidden from the model picker

    max_megapixels: float = 2.0
    supports_step_cache: bool = False
    supports_mask: bool = False
    supports_enhance: bool = False
    optimizable: bool = True

    def __post_init__(self):
        self.max_megapixels = {"qwen21": 1.6, "krea2": 1.3, "seedvr2": 4.5}.get(self.family, 2.0)
        if self.family == "flux2" and "9b" in self.mflux_name:
            self.max_megapixels = 1.6
        self.supports_mask = self.supports_enhance = self.family == "qwen21"
        self.supports_step_cache = self.family == "qwen21" and self.default_steps == 40
        self.optimizable = not self.tool

    def passes_per_step(self, guidance: float) -> int:
        """Guidance runs a second (negative) model pass per step: mflux does it for guidance > 1, and for
        Krea for any guidance other than 1."""
        if not self.supports_guidance:
            return 1
        if self.family == "krea2":
            return 2 if guidance != 1 else 1
        return 2 if guidance > 1 else 1

    def to_json(self) -> dict:
        data = asdict(self)
        data["sources"] = [asdict(s) for s in self.sources]
        data["double_pass_when"] = (
            "never" if not self.supports_guidance else "guidance!=1" if self.family == "krea2" else "guidance>1"
        )
        return data


CATALOG: list[ModelSpec] = [
    ModelSpec(
        id="flux2-klein-4b", name="FLUX.2 klein 4B", family="flux2", mflux_name="flux2-klein-4b",
        released="2026-01-14", license="Apache 2.0", commercial=True, gated=False,
        description="Fast all-rounder: 4 steps, text-to-image plus editing with several reference images.",
        sources=[Source("black-forest-labs/FLUX.2-klein-4B", FLUX2_PATTERNS)],
        edit="multi", max_refs=4, default_steps=4, default_guidance=1.0, default_strength=0.5, distilled=True,
        # Distilled for guidance 1.0 (mflux's CLI rejects other values); 4-bit measured 12% faster and
        # 4 GB lighter than 8-bit at 1024² with no visible quality loss.
        supports_guidance=False, default_quant=4,
        memory_gb={"8": 14, "4": 12, "0": 18},  # measured peak 12.4 GB at 4-bit, 1 MP
    ),
    ModelSpec(
        id="flux2-klein-base-4b", name="FLUX.2 klein Base 4B", family="flux2", mflux_name="flux2-klein-base-4b",
        released="2026-01-14", license="Apache 2.0", commercial=True, gated=False,
        description="Undistilled klein with real guidance: each step runs the model twice, so 25 steps is 50 "
                    "passes (~1 min/image vs 8 s for klein 4B). More variety; good for "
                    "LoRAs; weaker and much slower at editing than the distilled 4B, and \"start from image\" "
                    "keeps the photo but mostly ignores edit instructions.",
        sources=[Source("black-forest-labs/FLUX.2-klein-base-4B", FLUX2_PATTERNS)],
        # 50 steps (mflux's default) take ~2 min at 768²; 25 halves that.
        edit="multi", max_refs=4, default_steps=25, default_guidance=1.5, default_strength=0.5,
        memory_gb={"8": 14, "4": 12, "0": 18},  # measured 12.3 GB at 4-bit, 1 MP edit
    ),
    ModelSpec(
        id="flux2-klein-9b", name="FLUX.2 klein 9B", family="flux2", mflux_name="flux2-klein-9b",
        released="2026-01-14", license="FLUX non-commercial", commercial=False, gated=True,
        description="Larger klein with better detail and prompt following. 4 steps.",
        sources=[Source("black-forest-labs/FLUX.2-klein-9B", FLUX2_PATTERNS)],
        edit="multi", max_refs=4, default_steps=4, default_guidance=1.0, default_strength=0.5, distilled=True,
        supports_guidance=False,  # distilled for guidance 1.0
        quant_options=[4], default_quant=4, memory_gb={"4": 17},  # measured 17.2 GB at 1 MP, text or edit
    ),
    ModelSpec(
        id="qwen-image-2.1-turbo", name="Qwen-Image 2.1 Turbo", family="qwen21", mflux_name="qwen-image-2.1",
        released="2026-09-22", license="Qwen Research (non-commercial)", commercial=False, gated=False,
        description="#1 open model on the Artificial Analysis arena, sped up to 6 steps with Viggle's "
                    "turbo LoRA. Excellent text rendering; edits with up to 10 reference images.",
        sources=[Source("Qwen/Qwen-Image-2.1", QWEN21_PATTERNS), Source(VIGGLE_REPO, [VIGGLE_LORA])],
        edit="multi", max_refs=10, default_steps=6, fixed_steps=True, default_guidance=1.0, min_guidance=1.0,
        default_strength=0.7, supports_guidance=False,
        supports_strength=True, size_multiple=32, quant_options=[4], default_quant=4,
        memory_gb={"4": 18}, is_new=True,  # measured 15.3 GB text / 17.5 GB edit at 1 MP
        builtin_loras=[(VIGGLE_REPO, VIGGLE_LORA)], scheduler="viggle_turbo",
    ),
    ModelSpec(
        id="qwen-image-2.1", name="Qwen-Image 2.1", family="qwen21", mflux_name="qwen-image-2.1",
        released="2026-09-14", license="Qwen Research (non-commercial)", commercial=False, gated=False,
        description="Full-quality Qwen-Image 2.1 (40 steps, slower). Same download as the Turbo version.",
        sources=[Source("Qwen/Qwen-Image-2.1", QWEN21_PATTERNS)],
        edit="multi", max_refs=10, default_steps=40, default_guidance=1.0, min_guidance=1.0,
        # Its shifted 40-step schedule barely touches the image at 0.7 (Turbo's isn't); 0.9 keeps layout.
        default_strength=0.9,
        supports_negative=True, supports_strength=True, size_multiple=32,
        quant_options=[4], default_quant=4, memory_gb={"4": 18}, is_new=True,
    ),
    ModelSpec(
        id="qwen-image-2.1-uncensored-turbo", name="Qwen-Image 2.1 Uncensored Turbo", family="qwen21",
        mflux_name="qwen-image-2.1",
        released="2026-10-12", license="Qwen Research (non-commercial)", commercial=False, gated=False,
        description="Qwen-Image 2.1 with abenzerps' uncensored LoRA and the 6-step turbo LoRA. No content "
                    "filter. Reuses the Qwen-Image 2.1 download (+30 MB).",
        sources=[Source("Qwen/Qwen-Image-2.1", QWEN21_PATTERNS), Source(VIGGLE_REPO, [VIGGLE_LORA]),
                 Source(UNCENSORED_REPO, [UNCENSORED_LORA])],
        edit="multi", max_refs=10, default_steps=6, fixed_steps=True, default_guidance=1.0, min_guidance=1.0,
        default_strength=0.7, supports_guidance=False,
        supports_strength=True, size_multiple=32, quant_options=[4], default_quant=4,
        memory_gb={"4": 18}, is_new=True,
        builtin_loras=[(VIGGLE_REPO, VIGGLE_LORA), (UNCENSORED_REPO, UNCENSORED_LORA)], scheduler="viggle_turbo",
    ),
    ModelSpec(
        id="qwen-image-2.1-uncensored", name="Qwen-Image 2.1 Uncensored", family="qwen21",
        mflux_name="qwen-image-2.1",
        released="2026-10-12", license="Qwen Research (non-commercial)", commercial=False, gated=False,
        description="Full-quality Qwen-Image 2.1 (40 steps) with abenzerps' uncensored LoRA. No content "
                    "filter. Reuses the Qwen-Image 2.1 download (+30 MB).",
        sources=[Source("Qwen/Qwen-Image-2.1", QWEN21_PATTERNS), Source(UNCENSORED_REPO, [UNCENSORED_LORA])],
        edit="multi", max_refs=10, default_steps=40, default_guidance=1.0, min_guidance=1.0,
        default_strength=0.9,
        supports_negative=True, supports_strength=True, size_multiple=32,
        quant_options=[4], default_quant=4, memory_gb={"4": 18}, is_new=True,
        builtin_loras=[(UNCENSORED_REPO, UNCENSORED_LORA)],
    ),
    ModelSpec(
        id="z-image-turbo", name="Z-Image Turbo (GGUF Q8)", family="zimage", mflux_name="z-image-turbo",
        released="2025-11-26", license="Apache 2.0", commercial=True, gated=False,
        description="Tongyi's fast 6B photoreal model, 9 steps, strong English and Chinese text. Uses "
                    "unsloth's Q8 GGUF (converted once on first load). Image input works as a starting image; "
                    "describe the result you want rather than an edit instruction.",
        sources=[Source(ZIMAGE_REPO, ZIMAGE_PATTERNS), Source(ZIMAGE_GGUF[0], [ZIMAGE_GGUF[1]])],
        edit="single", max_refs=1, default_steps=9, default_guidance=0.0, supports_guidance=False,
        # Keeps the subject up to ~0.4-0.5; at 0.6 it already swaps the person, at 0.8 the whole scene.
        image_modes=["init"], default_strength=0.5, supports_strength=True,
        # 4-bit measured 13% faster and 5 GB lighter than 8-bit at 1024², same quality.
        quant_options=[8, 4], default_quant=4, memory_gb={"8": 15, "4": 11},  # 4-bit measured 8.6 GB text / 11.3 GB from image
        gguf=ZIMAGE_GGUF,
    ),
    ModelSpec(
        id="seedvr2-3b", name="SeedVR2 Upscaler (3B)", family="seedvr2", mflux_name="seedvr2-3b",
        released="2025-06-05", license="Apache 2.0", commercial=True, gated=False,
        description="Upscales a result 2x with ByteDance's one-step diffusion upscaler, adding real detail "
                    "instead of just enlarging. Used by \"Upscale 2x\" on a result.",
        sources=[Source(SEEDVR2_REPO, ["seedvr2_ema_3b_fp16.safetensors", "ema_vae_fp16.safetensors"])],
        edit="none", max_refs=0, default_steps=1, default_guidance=1.0, supports_guidance=False,
        image_modes=["init"], quant_options=[8, 0], default_quant=8, memory_gb={"8": 16, "0": 20},
        tool=True, is_new=True,
    ),
    ModelSpec(
        id="krea-2-turbo", name="Krea 2 Turbo", family="krea2", mflux_name="krea-2",
        released="2026-06-18", license="Krea 2 license", commercial=False, gated=True,
        description="Photorealism and aesthetics specialist, 8 steps. Image input works as a starting "
                    "image (img2img), not multi-reference editing.",
        sources=[Source("krea/Krea-2-Turbo", KREA2_TURBO_PATTERNS)],
        edit="single", max_refs=1, default_steps=8, default_guidance=1.0, distilled=True,
        image_modes=["init"], default_strength=0.6,
        supports_negative=True, supports_strength=True, quant_options=[4], default_quant=4,
        memory_gb={"4": 18}, is_new=True,  # measured 17.4 GB text / 18.2 GB from image
    ),
]


def _read_custom() -> list[dict]:
    with _custom_lock:
        try:
            return json.loads(CUSTOM_MODELS_FILE.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return []


def _write_custom(entries: list[dict]) -> None:
    CUSTOM_MODELS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CUSTOM_MODELS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(entries, indent=2))
    os.replace(tmp, CUSTOM_MODELS_FILE)


def _custom_spec(entry: dict) -> ModelSpec | None:
    arch = ARCHITECTURES.get(entry.get("architecture", ""))
    base = get(arch["base"], include_custom=False) if arch else None
    if base is None:
        return None
    path = entry["path"]
    is_local = os.path.isabs(path)
    spec = dataclasses.replace(base)
    spec.id = entry["id"]
    spec.name = entry["name"]
    spec.description = f"Custom {arch['name']} checkpoint: {path}"
    spec.sources = [] if is_local else [Source(path, arch["patterns"])]
    spec.builtin_loras = []
    spec.scheduler = None
    spec.gated = False
    spec.is_new = False
    spec.custom = True
    spec.model_path = path
    spec.released = ""
    spec.license = "Unknown (check the model page)"
    spec.commercial = False
    return spec


def _load_custom() -> list[ModelSpec]:
    return [spec for spec in map(_custom_spec, _read_custom()) if spec is not None]


def all_models() -> list[ModelSpec]:
    return CATALOG + _load_custom()


def get(model_id: str, include_custom: bool = True) -> ModelSpec | None:
    for spec in (all_models() if include_custom else CATALOG):
        if spec.id == model_id:
            return spec
    return None


def add_custom(name: str, path: str, architecture: str) -> ModelSpec:
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {architecture}")
    path = os.path.expanduser(path.strip())
    if os.path.isabs(path):
        path = os.path.normpath(path)
        if not Path(path).is_dir():
            raise ValueError(f"Folder not found: {path}")
    elif path.count("/") != 1 or " " in path:
        raise ValueError("Enter a Hugging Face repo as owner/name, or an absolute folder path.")
    digest = hashlib.sha1(f"{architecture}|{path}".encode()).hexdigest()[:12]
    entry = {"id": f"custom-{digest}", "name": name.strip() or path, "path": path, "architecture": architecture}
    spec = _custom_spec(entry)
    if spec is None:
        raise ValueError(f"Unsupported architecture {architecture}")
    with _custom_lock:
        try:
            entries = json.loads(CUSTOM_MODELS_FILE.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            entries = []
        entries = [e for e in entries if e["id"] != entry["id"]] + [entry]
        _write_custom(entries)
    return spec


def remove_custom(model_id: str) -> None:
    with _custom_lock:
        try:
            entries = json.loads(CUSTOM_MODELS_FILE.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return
        _write_custom([e for e in entries if e["id"] != model_id])


# ---------------------------------------------------------------------------------------------
# Pipelines: a uniform generate() over the different mflux model classes.


def _preview_factor(shape, height, width, max_side):
    """Pool to the preview size, keeping at least four latent pixels per axis."""
    factor = 1
    while max(height, width) / factor > max_side and min(shape[-2:]) // (factor * 2) >= 4:
        factor *= 2
    return factor


class Pipeline:
    def __init__(self, spec: ModelSpec, quantize: int | None, lora_path: str | None, lora_scale: float,
                 progress=None, loras=None, skip_loras=False, model_path=None):
        assert_mlx_thread()
        from mflux.models.common.config.model_config import ModelConfig

        self.spec = spec
        config = ModelConfig.from_name(spec.mflux_name)
        user_loras = loras or ([{"path": lora_path, "scale": lora_scale}] if lora_path else [])
        loras = [] if skip_loras else [hf_hub_download(repo, filename) for repo, filename in spec.builtin_loras]
        scales = [1.0] * len(loras)
        if not skip_loras:
            loras.extend(item["path"] for item in user_loras)
            scales.extend(item["scale"] for item in user_loras)
        kwargs = dict(
            quantize=quantize,
            model_config=config,
            model_path=model_path or spec.model_path,
            lora_paths=loras or None,
            lora_scales=scales or None,
            # Baking a LoRA into quantized weights rounds most of it away (only ~40% of the Qwen turbo
            # LoRA survived at 4-bit), so keep LoRAs as a separate low-rank term on quantized models.
            bake_lora=quantize is None,
        )
        if spec.family == "flux2":
            from mflux.models.flux2.variants import Flux2Klein, Flux2KleinEdit

            self.model = Flux2Klein(**kwargs)
            self._decode_in_stages(self.model.vae.decoder)
            # The edit pipeline shares the same weights; it can't run without reference images.
            self.edit = Flux2KleinEdit.__new__(Flux2KleinEdit)
            dict.update(self.edit, self.model)
            self.edit.__dict__.update(self.model.__dict__)
        elif spec.family == "qwen21":
            from mflux.models.qwen21.variants.edit.qwen_image_21_edit import QwenImage21Edit

            # The edit pipeline quantizes the 17.5 GB text encoder too (the text-only one keeps it
            # in bf16), so it is the one that fits in 24 GB. It also handles text-to-image.
            self.model = QwenImage21Edit(**kwargs)
            self._decode_in_bf16(self.model.vae)
        elif spec.family == "krea2":
            import mflux.models.krea2  # noqa: F401 registers Krea's schedulers
            from mflux.models.krea2 import Krea2

            self.model = Krea2(**kwargs)
            self._decode_in_stages(self.model.vae.decoder)
        elif spec.family == "seedvr2":
            from mflux.models.seedvr2 import SeedVR2

            # Its initializer sets up VAE tiling itself; LoRAs don't apply.
            self.model = SeedVR2(quantize=quantize, model_config=config)
            return
        elif spec.family == "zimage":
            from mflux.models.z_image import ZImage

            if spec.gguf and not model_path:
                kwargs["model_path"] = str(converted_folder(spec, progress))
            self.model = ZImage(**kwargs)
        else:
            raise ValueError(f"Unsupported model family {spec.family}")

        self._enable_tiling()
        self._wrap_prompt_encodings()
        from speedups import install

        install(self)

    def _wrap_prompt_encodings(self):
        assert_mlx_thread()
        from performance import PromptEncodings

        self.prompt_encodings = PromptEncodings()
        if self.spec.family == "flux2":
            self.prompt_encodings.wrap(self.model, "_encode_prompt_pair", adapter="flux2-text")
            self.prompt_encodings.wrap(self.edit, "_encode_prompt_pair", adapter="flux2-edit", edit=True)
        elif self.spec.family == "zimage":
            self.prompt_encodings.wrap(self.model, "_encode_prompts", adapter="zimage")

    def begin_job(self, image_paths):
        assert_mlx_thread()
        cache = getattr(self, "prompt_encodings", None)
        if cache is not None:
            cache.begin(image_paths)
        for name in ("reference_conditioning", "job_predictors", "job_conditioning"):
            cache = getattr(self, name, None)
            if cache is not None:
                cache.begin()

    def end_job(self):
        assert_mlx_thread()
        cache = getattr(self, "prompt_encodings", None)
        if cache is not None:
            cache.clear()
        if getattr(getattr(self, "spec", None), "family", None) == "krea2":
            # Keep native prompt reuse within a batch, never across job boundaries.
            cache = getattr(self.model, "prompt_cache", None)
            if cache is not None:
                cache.clear()
        for name in ("reference_conditioning", "job_predictors", "job_conditioning"):
            cache = getattr(self, name, None)
            if cache is not None:
                cache.clear()
        callback = getattr(self, "progress", None)
        if callback is not None:
            callback.prev = None

    @staticmethod
    def _decode_in_bf16(vae) -> None:
        """Qwen's VAE ships in fp32. Decoding in bf16 gave the same images (PSNR ~50 dB in our test) with a
        2.4 GB lower peak, so cast its weights and feed it bf16 latents."""
        import mlx.core as mx
        from mlx.utils import tree_map

        vae.update(tree_map(lambda x: x.astype(mx.bfloat16) if x.dtype == mx.float32 else x, vae.parameters()))
        decode = vae.decode
        vae.decode = lambda latents, _decode=decode: _decode(latents.astype(mx.bfloat16))

    @staticmethod
    def _decode_in_stages(decoder) -> None:
        """Evaluate a VAE decoder (FLUX.2 or Qwen-style) block by block. Same operations and pixel-identical
        output, but MLX no longer keeps several full-size activations alive at once: the final decode was
        the peak-memory moment (klein about 2.5 GB lower at 1024², same speed)."""
        import mlx.core as mx

        def eval_after(module):
            cls = type(module)

            def call(self, *args, **kwargs):
                out = cls.__call__(self, *args, **kwargs)
                mx.eval(out)
                return out

            module.__class__ = type(cls.__name__, (cls,), {"__call__": call})

        blocks = getattr(decoder, "up_blocks", None) or [getattr(decoder, f"up_block{i}") for i in range(4)]
        stages = [decoder.conv_in, decoder.mid_block]
        for block in blocks:
            stages += [*block.resnets, *(block.upsamplers or [])]
        for stage in stages:
            eval_after(stage)

    def _enable_tiling(self) -> None:
        """Decode/encode images in tiles where the model's VAE does so without visible seams. The
        final decode is otherwise the peak-memory moment (Qwen's fp32 VAE took ~7 GB extra)."""
        assert_mlx_thread()
        from mflux.models.common.vae.tiling_config import TilingConfig

        if TilingConfig.may_tile_implicitly(self.model):
            self.model.tiling_config = TilingConfig()
            if getattr(self, "edit", None) is not None:
                self.edit.tiling_config = self.model.tiling_config

    def register(self, callback) -> None:
        assert_mlx_thread()
        callbacks = getattr(self.model, "callbacks", None)
        if callbacks is not None:
            callbacks.register(callback)

    def upscale(self, *, image_path: str, scale: float, softness: float, seed: int):
        assert_mlx_thread()
        from mflux.utils.scale_factor import ScaleFactor

        return self.model.generate_image(seed=seed, image_path=image_path,
                                         resolution=ScaleFactor.parse(f"{scale:g}x"), softness=softness)

    def preview(self, latents, config, max_side: int = 512):
        """Average latents to the preview size before VAE decoding. Runs on the MLX thread."""
        assert_mlx_thread()
        import mlx.core as mx
        from mflux.models.common.vae.vae_util import VAEUtil
        from mflux.utils.image_util import ImageUtil

        family, h, w = self.spec.family, config.height, config.width
        if family == "flux2":
            from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator

            z = self.model.vae.unpack_packed_latents(Flux2LatentCreator.unpack_latents(latents, h, w))
        elif family == "qwen21":
            from mflux.models.qwen21.latent_creator.qwen_image21_latent_creator import QwenImage21LatentCreator

            z = QwenImage21LatentCreator.unpack_latents(latents, h, w).astype(mx.float32)
        elif family == "zimage":
            from mflux.models.z_image.latent_creator.z_image_latent_creator import ZImageLatentCreator

            z = ZImageLatentCreator.unpack_latents(latents, h, w)
        elif family == "krea2":
            z = latents
        else:
            return None
        factor = _preview_factor(z.shape, h, w, max_side)
        lh, lw = z.shape[-2] // factor * factor, z.shape[-1] // factor * factor
        if factor > 1:
            z = z[..., :lh, :lw]
            z = z.reshape(*z.shape[:-2], lh // factor, factor, lw // factor, factor).mean(axis=(-3, -1))
        image = ImageUtil.to_pil(VAEUtil.decode(self.model.vae, z, getattr(self.model, "tiling_config", None)))
        image.thumbnail((max_side, max_side))
        return image

    def generate(self, *, seed, prompt, steps, width, height, guidance, image_paths, strength, negative_prompt,
                 image_mode="reference", fast=False, mask_path=None, auto_mask=None, enhance_prompt=False):
        assert_mlx_thread()
        spec = self.spec
        cache = getattr(self, "prompt_encodings", None)
        if cache is not None and cache.enabled:
            cache.set_references(image_paths)
        change = strength if strength is not None else spec.default_strength
        if image_paths and spec.family in ("flux2", "zimage", "krea2"):
            change = min(0.95, change)
        if not spec.supports_guidance:
            guidance = spec.default_guidance
        if spec.family == "flux2":
            common = dict(seed=seed, prompt=prompt, num_inference_steps=steps, height=height, width=width,
                          guidance=guidance)
            if image_paths and image_mode == "init":
                # img2img: start denoising from image 1. image_strength is how much of it to keep.
                return self.model.generate_image(**common, image_path=image_paths[0],
                                                 image_strength=min(0.95, max(0.0, 1 - change)))
            if image_paths:
                return self.edit.generate_image(**common, image_paths=image_paths)
            return self.model.generate_image(**common)

        if spec.family == "qwen21":
            guidance = max(1.0, guidance)
            kwargs = dict(
                seed=seed, prompt=prompt, num_inference_steps=steps, height=height, width=width,
                guidance=guidance, image_paths=image_paths or None,
                use_step_cache=fast and spec.supports_step_cache,
                mask_image=mask_path, auto_mask=auto_mask, enhance_prompt=enhance_prompt,
            )
            if guidance > 1:
                kwargs["negative_prompt"] = negative_prompt or ""
            if image_paths and image_mode == "init" and change < 1:
                # Start from image 1 (keeps its layout); the images still act as references.
                kwargs["strength"] = max(0.05, change)
            if spec.scheduler:
                kwargs["scheduler"] = spec.scheduler
            try:
                return self.model.generate_image(**kwargs)
            finally:
                cache = getattr(self.model, "prompt_cache", None)
                if cache is not None:
                    while len(cache) > 16:
                        cache.pop(next(iter(cache)))

        if spec.family == "zimage":
            # Turbo is guidance-distilled (mflux forces guidance 0); image input is img2img only.
            kwargs = dict(seed=seed, prompt=prompt, num_inference_steps=steps, height=height, width=width)
            if image_paths:
                kwargs["image_path"] = image_paths[0]
                kwargs["image_strength"] = min(0.95, max(0.0, 1 - change))
            return self.model.generate_image(**kwargs)

        if spec.family == "krea2":
            kwargs = dict(seed=seed, prompt=prompt, num_inference_steps=steps, height=height, width=width,
                          guidance=guidance)
            if negative_prompt and guidance != 1:
                kwargs["negative_prompt"] = negative_prompt
            if image_paths:
                # The app's strength is "how much to change"; mflux's image_strength is how much to keep.
                kwargs["image_path"] = image_paths[0]
                # 0 keeps nothing (plain text-to-image); mflux treats any positive value as img2img.
                kwargs["image_strength"] = min(0.95, max(0.0, 1 - change))
            return self.model.generate_image(**kwargs)

        raise ValueError(f"Unsupported model family {spec.family}")


def converted_folder(spec: ModelSpec, progress=None) -> Path:
    """Model folder with the GGUF transformer converted to safetensors (built once, then reused)."""
    assert_mlx_thread()
    from downloads import _snapshot
    from gguf_convert import build_model_folder

    base = _snapshot(spec.sources[0].repo)
    gguf_snap = _snapshot(spec.gguf[0])
    if base is None or gguf_snap is None or not (gguf_snap / spec.gguf[1]).exists():
        raise RuntimeError(f"{spec.name} is not fully downloaded.")
    return build_model_folder(base, (gguf_snap / spec.gguf[1]).resolve(), CONVERTED_DIR / spec.id, progress)
