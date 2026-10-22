"""Exact conditioning reuse and Qwen block compilation on the single MLX executor.

Caches live for one job. Scoped helper patches restore on failure/cancellation;
concurrent model calls or in-place weight changes during a job are unsupported.
"""

import hashlib
from contextlib import contextmanager
from pathlib import Path

import mlx.core as mx

from core import assert_mlx_thread
from performance import leaves


def reference_identity(paths):
    """Ordered file versions; replacing a file also invalidates the conditioning."""
    versions = []
    for path in paths or ():
        path = Path(path).resolve()
        stat = path.stat()
        versions.append((str(path), stat.st_dev, stat.st_ino, stat.st_size,
                         stat.st_mtime_ns, stat.st_ctime_ns))
    return tuple(versions)


class ReferenceConditioning:
    """One bounded, evaluated conditioning tuple, retained only within a job."""

    def __init__(self, limit=64 * 1024**2):
        self.limit = limit
        self.enabled = False
        self.clear()

    def clear(self):
        assert_mlx_thread()
        self.key = self.value = None
        self.pending = None
        self.bytes = 0
        self.hits = self.misses = 0
        self.enabled = False

    def begin(self):
        self.clear()
        self.enabled = True

    def prepare(self, original, **kwargs):
        assert_mlx_thread()
        key = (reference_identity(kwargs.get("image_paths")), kwargs.get("batch_size", 1),
               id(kwargs["vae"]), repr(kwargs.get("tiling_config")))
        if self.enabled and self.key == key:
            self.hits += 1
            return self.value
        # Drop the previous entry before evaluating a changed set of references.
        self.key = self.value = None
        self.pending = None
        self.bytes = 0
        self.misses += 1
        result = original(**kwargs)
        self.pending = (key, result)
        return result

    def materialize(self):
        assert_mlx_thread()
        if self.pending is None:
            return
        key, result = self.pending
        arrays = list(leaves(result))
        if arrays:
            mx.eval(*arrays)
            mx.clear_cache()
        size = sum(a.nbytes for a in arrays)
        if self.enabled and arrays and size <= self.limit:
            self.key, self.value, self.bytes = key, result, size
        self.pending = None

    @contextmanager
    def scope(self, helper):
        assert_mlx_thread()
        original = helper.prepare_reference_image_conditioning
        # Save the descriptor as well, so the class is restored exactly.
        descriptor = helper.__dict__["prepare_reference_image_conditioning"]
        helper.prepare_reference_image_conditioning = staticmethod(
            lambda **kwargs: self.prepare(original, **kwargs))
        try:
            yield
        finally:
            helper.prepare_reference_image_conditioning = descriptor


class JobPredictors:
    """Retain mflux's own compiled callable across images, never across jobs."""

    def __init__(self):
        self.enabled = False
        self.entries = {}

    def begin(self):
        self.clear()
        self.enabled = True

    def clear(self):
        assert_mlx_thread()
        self.entries.clear()
        self.enabled = False

    def wrap(self, model, adapter):
        assert_mlx_thread()
        original = model._predict

        def predict(transformer):
            assert_mlx_thread()
            if not self.enabled:
                return original(transformer)
            entry = self.entries.get(adapter)
            if entry is None or entry[0] is not transformer:
                entry = (transformer, original(transformer))
                self.entries[adapter] = entry
            return entry[1]

        model._predict = predict


def install_flux(pipeline):
    pipeline.reference_conditioning = ReferenceConditioning()
    generate = pipeline.edit.generate_image

    def edit_generate(**kwargs):
        from mflux.models.flux2.variants.edit.flux2_klein_edit_helpers import _Flux2KleinEditHelpers

        with pipeline.reference_conditioning.scope(_Flux2KleinEditHelpers):
            return generate(**kwargs)

    pipeline.edit.generate_image = edit_generate
    pipeline.job_predictors = JobPredictors()
    pipeline.job_predictors.wrap(pipeline.model, "text")
    pipeline.job_predictors.wrap(pipeline.edit, "edit")
    original_predict = pipeline.edit._predict

    def ready_predict(transformer):
        # before_loop can evict text weights before reference VAE materialization.
        pipeline.reference_conditioning.materialize()
        return original_predict(transformer)

    pipeline.edit._predict = ready_predict


class Values:
    """Bounded evaluated values, released at the next job boundary."""

    def __init__(self, limit=64 * 1024**2):
        self.limit = limit
        self.clear()

    def clear(self):
        assert_mlx_thread()
        self.entries = {}
        self.pending = []
        self.bytes = self.hits = self.misses = 0
        self.enabled = False

    def begin(self):
        self.clear()
        self.enabled = True

    def get(self, key, build, defer=False):
        assert_mlx_thread()
        if not self.enabled:
            return build()
        if key in self.entries:
            self.hits += 1
            return self.entries[key]
        self.misses += 1
        result = build()
        if defer:
            self.pending.append((key, result))
        else:
            self.admit(key, result)
        return result

    def admit(self, key, result):
        arrays = [a for a in leaves(result) if hasattr(a, "nbytes")]
        if arrays:
            mx.eval(*arrays)
        size = sum(a.nbytes for a in arrays)
        if self.enabled and key not in self.entries and arrays and size <= self.limit - self.bytes:
            self.entries[key] = result
            self.bytes += size

    def materialize(self):
        assert_mlx_thread()
        for key, result in self.pending:
            self.admit(key, result)
        self.pending.clear()
        mx.clear_cache()


def image_identity(images):
    # Hash the actual resized RGBA bytes, not a thumbnail: every reference pixel
    # can affect the VAE. No MLX/host synchronization is needed for these PIL inputs.
    return tuple((im.size, im.mode, hashlib.sha256(im.tobytes()).digest()) for im in images)


class QwenPrompts(Values):
    """Implements the existing prompt_encodings eviction interface for both branches."""

    def clear(self):
        super().clear()
        self.reusable = False
        self.references = ()
        self.required = []

    def begin(self, paths=()):
        super().begin()
        self.set_references(paths)

    def set_references(self, paths):
        assert_mlx_thread()
        self.references = reference_identity(paths)


class JobConditioning:
    def __init__(self):
        self.values = Values()
        self.clear()

    def begin(self):
        self.clear()
        self.enabled = True
        self.values.begin()

    def clear(self):
        assert_mlx_thread()
        self.enabled = False
        self.compiled = {}
        self.values.clear()

    def wrap_qwen_block(self, block):
        original = block.forward_reference

        def forward(x, modulation, target_mask, layout, rope, cached=None, extract=False, key_valid=None):
            assert_mlx_thread()
            if not self.enabled or cached is None or extract:
                return original(x, modulation, target_mask, layout, rope, cached, extract, key_valid)
            # The cached attention branch does not read layout. Pass arrays as
            # compile inputs, keeping the prefix cache and first-step mutation outside.
            if id(block) not in self.compiled:
                def target(x, modulation, target_mask, cos, sin, key, value, valid):
                    return original(x, modulation, target_mask, None, (cos, sin),
                                    (key, value), False, valid)[0]
                self.compiled[id(block)] = mx.compile(target)
            result = self.compiled[id(block)](x, modulation, target_mask, *rope, *cached, key_valid)
            return result, None

        block.forward_reference = forward

    @contextmanager
    def init_scope(self):
        from mflux.models.common.latent_creator.latent_creator import LatentCreator

        descriptor = LatentCreator.__dict__["encode_image"]
        original = LatentCreator.encode_image

        def encode(*, vae, image_path, height, width, tiling_config=None):
            # FLUX.2's VAE returns the posterior mean; seeded noise is created outside this helper.
            key = ("init", id(vae), reference_identity([image_path]), height, width, repr(tiling_config))
            return self.values.get(key, lambda: original(vae=vae, image_path=image_path, height=height,
                                                        width=width, tiling_config=tiling_config))

        LatentCreator.encode_image = staticmethod(encode)
        try:
            yield
        finally:
            LatentCreator.encode_image = descriptor


def wrap_qwen(pipeline, conditioning):
    model = pipeline.model
    prompts = pipeline.prompt_encodings = QwenPrompts()
    generate, encode = model.generate_image, model._encode_prompt

    def run(**kwargs):
        assert_mlx_thread()
        if not conditioning.enabled:
            return generate(**kwargs)
        prompts.required = []
        prompts.reusable = False
        identity = None
        ordinal = 0
        vae_encode = model.vae.encode
        start = model.callbacks.start

        def prompt_encode(prompt, images):
            nonlocal identity
            identity = image_identity(images)
            key = (prompt, identity, id(model.processor))
            result = prompts.get(key, lambda: encode(prompt, images))
            prompts.required.append(key)
            # Both positive AND negative embeddings must fit before early eviction.
            prompts.reusable = all(k in prompts.entries for k in prompts.required) and not any(
                kwargs.get(k) for k in ("enhance_prompt", "auto_mask", "verify"))
            return result

        def reference_encode(pixels):
            nonlocal ordinal
            index = ordinal
            ordinal += 1
            # Only reference encodes: later blend-source/mask encodes are untouched.
            if identity is None or index >= len(identity):
                return vae_encode(pixels)
            key = ("qwen-ref", id(model.vae), identity[index], tuple(pixels.shape), str(pixels.dtype))
            return conditioning.values.get(key, lambda: vae_encode(pixels), defer=True)

        def callbacks_start(*args, **kw):
            ctx = start(*args, **kw)
            before = ctx.before_loop

            def before_loop(*values, **options):
                # Let the app evict fully cached text/vision weights first. The
                # boundary is still before denoising, with no graph across images.
                result = before(*values, **options)
                conditioning.values.materialize()
                return result

            ctx.before_loop = before_loop
            return ctx

        model._encode_prompt = prompt_encode
        model.vae.encode = reference_encode
        model.callbacks.start = callbacks_start
        try:
            return generate(**kwargs)
        finally:
            model._encode_prompt = encode
            model.vae.encode = vae_encode
            model.callbacks.start = start
            conditioning.values.pending.clear()

    model.generate_image = run


def wrap_krea_prompts(pipeline):
    cache = pipeline.prompt_encodings = QwenPrompts()
    original = pipeline.model._encode_prompts

    def encode(*, prompt, negative_prompt, guidance):
        # Krea uses CFG for guidance != 1, including values below 1. FLUX's
        # existing PromptEncodings uses >1 and cannot serve this adapter exactly.
        branch = guidance != 1.0
        key = (prompt, negative_prompt if branch else None, branch)
        result = cache.get(key, lambda: original(prompt=prompt, negative_prompt=negative_prompt, guidance=guidance))
        cache.reusable = cache.enabled and key in cache.entries
        return result

    pipeline.model._encode_prompts = encode


def install(pipeline):
    assert_mlx_thread()
    family = pipeline.spec.family
    if family == "flux2":
        install_flux(pipeline)
    if family not in ("flux2", "qwen21", "zimage", "krea2"):
        return
    conditioning = pipeline.job_conditioning = JobConditioning()
    if family == "qwen21":
        wrap_qwen(pipeline, conditioning)
        for block in pipeline.model.transformer.transformer_blocks:
            conditioning.wrap_qwen_block(block)
        return
    if family == "krea2":
        wrap_krea_prompts(pipeline)
    generate = pipeline.model.generate_image

    def run(**kwargs):
        assert_mlx_thread()
        if not conditioning.enabled:
            return generate(**kwargs)
        with conditioning.init_scope():
            return generate(**kwargs)

    pipeline.model.generate_image = run
