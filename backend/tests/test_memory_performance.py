"""Memory and polling regressions with CPU tensors and temporary files only."""

import json
import struct
import sys
import tempfile
import types
import unittest
import weakref
import zlib
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image, ImageOps

if "test_core" not in sys.modules and "tests.test_core" not in sys.modules:
    from tests import test_core  # noqa: F401

import catalog
import gguf_convert
import optimized
import server
import timings
import speedups


class MemoryPerformanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state = patch.object(server, "state", server.State())
        self.state.start()
        server.state.pressure = "normal"

    def tearDown(self):
        self.state.stop()
        self.tmp.cleanup()

    def test_checkpoint_stats_each_unique_shard_once(self):
        for part in ("transformer", "text_encoder", "vae"):
            folder = self.root / part
            folder.mkdir()
            for name in ("a", "b"):
                (folder / f"{name}.safetensors").write_bytes(b"weights")
            (folder / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {str(i): f"{'a' if i % 2 else 'b'}.safetensors" for i in range(1000)}}))
        stat = Path.stat
        shards = []

        def counted(path, *args, **kwargs):
            if path.suffix == ".safetensors":
                shards.append(path)
            return stat(path, *args, **kwargs)

        with patch.object(Path, "stat", counted):
            self.assertTrue(optimized.checkpoint_valid(self.root, "flux2"))
        self.assertEqual(len(shards), 6)
        self.assertEqual(len(set(shards)), 6)
        shard = self.root / "vae/a.safetensors"
        shard.unlink()
        shard.mkdir()
        self.assertFalse(optimized.checkpoint_valid(self.root, "flux2"))

    def test_models_read_one_catalog_and_timing_snapshot_per_response(self):
        with patch.object(catalog, "CUSTOM_MODELS_FILE", self.root / "custom.json"), \
             patch.object(timings, "FILE", self.root / "timings.json"), \
             patch.object(catalog, "_read_custom", wraps=catalog._read_custom) as custom_read, \
             patch.object(timings, "_read", wraps=timings._read) as timing_read, \
             patch.object(server, "disk_usage", return_value={}), \
             patch.object(server.downloads, "status", return_value={"downloaded": False}), \
             patch.object(optimized, "polling_info", return_value=(False, 0)), \
             patch.object(server, "_converted_bytes", return_value=17) as converted:
            def poll():
                custom_read.reset_mock()
                timing_read.reset_mock()
                converted.reset_mock()
                rows = server.list_models()
                custom_read.assert_called_once_with()
                timing_read.assert_called_once_with()
                self.assertEqual([call.args[0].id for call in converted.call_args_list], [r["id"] for r in rows])
                self.assertTrue(all(r["own_bytes"] == r["disk_bytes"] == 17 for r in rows))
                return {r["id"]: r for r in rows}

            before = poll()
            spec = catalog.add_custom("Custom", "test/checkpoint", "flux2-klein-4b")
            timings.record_peak(spec.id, 4, 7_000_000_000, 1)
            after = poll()
            self.assertNotIn(spec.id, before)
            self.assertEqual(after[spec.id]["memory_gb"]["4"], 7)
            self.assertEqual(after[spec.id]["memory_measured"], [4])
            catalog.remove_custom(spec.id)
            self.assertNotIn(spec.id, poll())

    def test_reference_passthrough_does_not_load_pixels(self):
        for fmt in ("JPEG", "PNG", "WEBP"):
            with self.subTest(format=fmt):
                path = self.root / f"ref.{fmt.lower()}"
                Image.new("RGB", (32, 16), "red").save(path, format=fmt)
                opened = Image.open(path)
                with patch.object(Image, "open", return_value=opened), \
                     patch.object(opened, "load", side_effect=AssertionError("decoded pixels")), \
                     patch.object(server, "REF_CACHE", self.root / "cache"):
                    self.assertEqual(server._prepare_reference(str(path)), str(path))

    def test_reference_transpose_and_cache_png_are_lossless(self):
        path = self.root / "rotated.png"
        pixels = np.arange(32 * 16 * 3, dtype=np.uint8).reshape(16, 32, 3)
        image = Image.fromarray(pixels)
        exif = Image.Exif()
        exif[0x0112] = 6
        image.save(path, exif=exif)
        with Image.open(path) as source:
            expected = ImageOps.exif_transpose(source).convert("RGB")
        save = Image.Image.save
        levels = []

        def counted(im, fp, *args, **kwargs):
            levels.append(kwargs.get("compress_level"))
            return save(im, fp, *args, **kwargs)

        with patch.object(Image.Image, "save", counted), patch.object(server, "REF_CACHE", self.root / "cache"):
            out = server._prepare_reference(str(path))
            self.assertEqual(server._prepare_reference(str(path)), out)
        self.assertEqual(levels, [1])
        with Image.open(out) as actual:
            self.assertEqual((actual.size, actual.mode, actual.tobytes()),
                             (expected.size, expected.mode, expected.tobytes()))

    def test_png_trailing_orientation_without_pixel_decode(self):
        for orientation in (1, 6):
            path = self.root / f"trailing-{orientation}.png"
            Image.new("RGB", (32, 16)).save(path)
            data = path.read_bytes()
            exif = Image.Exif()
            exif[0x0112] = orientation
            payload = exif.tobytes()[6:]
            chunk = b"eXIf" + payload
            path.write_bytes(data[:-12] + struct.pack(">I", len(payload)) + chunk
                             + struct.pack(">I", zlib.crc32(chunk)) + data[-12:])
            with Image.open(path) as im, patch.object(im, "load", side_effect=AssertionError("decoded pixels")):
                self.assertEqual(server._reference_orientation(im, path), orientation)
            with patch.object(server, "REF_CACHE", self.root / "cache"):
                out = server._prepare_reference(str(path))
                if orientation == 1:
                    self.assertEqual(out, str(path))
                else:
                    with Image.open(out) as im:
                        self.assertEqual(im.size, (16, 32))

        path = self.root / "trailing-xmp.png"
        Image.new("RGB", (32, 16)).save(path)
        data = path.read_bytes()
        payload = b'XML:com.adobe.xmp\0\0\0\0\0<x tiff:Orientation="6"/>'
        chunk = b"iTXt" + payload
        path.write_bytes(data[:-12] + struct.pack(">I", len(payload)) + chunk
                         + struct.pack(">I", zlib.crc32(chunk)) + data[-12:])
        with Image.open(path) as im, patch.object(im, "load", side_effect=AssertionError("decoded pixels")):
            self.assertEqual(server._reference_orientation(im, path), 6)

    def test_encoder_eviction_collects_after_both_adapters(self):
        names = ("clip_text_encoder", "t5_text_encoder", "text_encoder", "image_encoder",
                 "image_embedder", "depth_pro", "qwen_vl_encoder")
        models = [NS(**{name: object() for name in names}, tokenizers={"qwen_vl": object(), "other": "keep"})
                  for _ in range(2)]
        pipeline = NS(model=models[0], edit=models[1])
        server.state.pipeline = pipeline
        server.state.low_memory = "always"
        server.state.image_index = server.state.image_count = 1
        server.state.cache_limit = 512 * 1024**2
        server.state.mlx_job_active = True

        def collect():
            self.assertTrue(all(getattr(model, name) is None for model in models for name in names))
            self.assertTrue(all(model.tokenizers == {"qwen_vl": None, "other": "keep"} for model in models))

        # Also change pressure policy at this boundary: it must share the flush.
        server.state.pressure = "warning"
        with patch.object(server.gc, "collect", side_effect=collect) as gc, \
             patch.object(server.mx, "clear_cache") as clear:
            server.on_mlx(server.ProgressCallback().call_before_loop, 1, "p", "latents", NS(num_inference_steps=4))
            gc.assert_called_once_with()
            clear.assert_called_once_with()

    def test_job_finish_collects_and_flushes_once_with_or_without_unload(self):
        for evicted in (False, True):
            with self.subTest(evicted=evicted):
                server.state.encoder_evicted = evicted
                server.state.cache_limit = 512 * 1024**2
                server.state.busy = "generate"
                server.state.pipeline = NS(end_job=Mock())
                with patch.object(server.gc, "collect") as gc, patch.object(server.mx, "clear_cache") as clear:
                    server.on_mlx(server._finish_job)
                    gc.assert_called_once_with()
                    clear.assert_called_once_with()
                self.assertEqual(server.state.pipeline is None, evicted)
                self.assertEqual(server.state.cache_limit, 2 * 1024**3)

    def test_preview_pool_factor_matches_target_and_minimum(self):
        for shape, height, width, expected in (
                ((1, 32, 128, 128), 1024, 1024, 2),
                ((1, 32, 64, 256), 512, 2048, 4),
                ((1, 32, 256, 64), 2048, 512, 4),
                ((1, 32, 512, 512), 4096, 4096, 8),
                ((1, 32, 64, 64), 512, 512, 1),
                ((1, 32, 8, 256), 64, 2048, 2),
                ((1, 32, 7, 256), 56, 2048, 1)):
            with self.subTest(shape=shape):
                self.assertEqual(catalog._preview_factor(shape, height, width, 512), expected)

    def test_preview_pools_before_decode_without_mutating_latents(self):
        latent = np.arange(1 * 4 * 64 * 256, dtype=np.float32).reshape(1, 4, 64, 256)
        original = latent.copy()
        pipeline = catalog.Pipeline.__new__(catalog.Pipeline)
        pipeline.spec = NS(family="krea2")
        pipeline.model = NS(vae=object())
        vae = types.ModuleType("mflux.models.common.vae.vae_util")
        util = types.ModuleType("mflux.utils.image_util")
        decode = Mock(return_value=object())
        vae.VAEUtil = NS(decode=decode)
        util.ImageUtil = NS(to_pil=Mock(return_value=Image.new("RGB", (512, 128))))
        with patch.dict(sys.modules, {vae.__name__: vae, util.__name__: util}):
            server.on_mlx(pipeline.preview, latent, NS(height=512, width=2048))
        pooled = decode.call_args.args[1]
        np.testing.assert_array_equal(pooled, latent.reshape(1, 4, 16, 4, 64, 4).mean(axis=(-3, -1)))
        np.testing.assert_array_equal(latent, original)

    def test_flux_init_returns_exact_array_and_preserves_seeded_noise(self):
        latent_module = types.ModuleType("mflux.models.common.latent_creator.latent_creator")
        path = self.root / "init.png"
        path.write_bytes(b"init")
        vae = object()
        encoded = np.arange(32, dtype=np.float32).reshape(1, 2, 4, 4)
        encode = Mock(side_effect=lambda **kw: encoded.copy())
        latent_module.LatentCreator = type("LatentCreator", (), {"encode_image": staticmethod(encode)})
        pipeline = catalog.Pipeline.__new__(catalog.Pipeline)
        pipeline.spec = NS(family="flux2")

        def generate(seed):
            noise = np.random.default_rng(seed).standard_normal(encoded.shape).astype(np.float32)
            init = latent_module.LatentCreator.encode_image(vae=vae, image_path=path, height=32, width=32)
            return init, init * .75 + noise * .25

        pipeline.model = NS(generate_image=generate, _predict=Mock())
        pipeline.edit = NS(generate_image=Mock(), _predict=Mock())

        def run():
            speedups.install(pipeline)
            uncached = [pipeline.model.generate_image(seed=i) for i in (42, 43)]
            encode.reset_mock()
            pipeline.begin_job([path])
            cached = [pipeline.model.generate_image(seed=i) for i in (42, 43)]
            encode.assert_called_once()
            self.assertIs(cached[0][0], cached[1][0])
            for a, b in zip(uncached, cached):
                for x, y in zip(a, b):
                    np.testing.assert_array_equal(x, y)
            pipeline.end_job()
            self.assertEqual(pipeline.job_conditioning.values.entries, {})

        with patch.dict(sys.modules, {latent_module.__name__: latent_module}):
            server.on_mlx(run)

    def test_gguf_conversion_small_shards_match_original_tensor_bits(self):
        import gguf

        names = ("layers.0.attention.qkv.weight", "x_embedder.weight", "layers.0.attention.out.weight")
        arrays = [np.linspace(-2, 3, rows * 32, dtype=np.float32).reshape(rows, 32) for rows in (6, 4, 4)]
        tensors = [NS(name=name, shape=array.shape[::-1], tensor_type=gguf.GGMLQuantizationType.Q8_0,
                      data=gguf.quants.quantize(array, gguf.GGMLQuantizationType.Q8_0))
                   for name, array in zip(names, arrays)]

        def bf16_bits(values):
            bits = np.asarray(values).view(np.uint32)
            return ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype(np.uint16)

        class Array:
            def __init__(self, values):
                self.values = values

            def astype(self, dtype):
                return bf16_bits(self.values)

        expected = {}
        for tensor, array in zip(tensors, arrays):
            values = gguf.quants.dequantize(tensor.data, tensor.tensor_type).reshape(array.shape).astype(np.float32)
            for name, rows in gguf_convert._target_names(tensor.name, array.shape[0]):
                expected[name] = bf16_bits(values if rows is None else values[rows])
        saved = {}
        sizes = []
        casts, temporaries = [], []

        class Values(np.ndarray):
            def astype(self, dtype, *args, **kwargs):
                casts.append(kwargs.get("copy", True))
                result = super().astype(dtype, *args, **kwargs)
                temporaries.append(weakref.ref(result))
                return result

        dequantize = gguf.quants.dequantize

        def dequant(*args):
            return dequantize(*args).view(Values)

        def save(path, shard):
            sizes.append(sum(a.nbytes for a in shard.values()))
            saved.update({name: array.copy() for name, array in shard.items()})

        with patch.object(gguf, "GGUFReader", return_value=NS(tensors=tensors)), \
             patch.object(gguf.quants, "dequantize", side_effect=dequant), \
             patch.object(server.mx, "array", Array, create=True), \
             patch.object(server.mx, "bfloat16", object(), create=True), \
             patch.object(server.mx, "save_safetensors", side_effect=save, create=True), \
             patch.object(gguf_convert, "SHARD_BYTES", 256):
            def progress(*args):
                self.assertTrue(all(ref() is None for ref in temporaries))
            server.on_mlx(gguf_convert.convert_transformer, "synthetic.gguf", self.root, progress)
        self.assertEqual(casts, [False] * len(tensors))
        self.assertGreater(len(sizes), 1)
        self.assertTrue(all(size <= 256 for size in sizes))
        self.assertEqual(saved.keys(), expected.keys())
        for name in expected:
            np.testing.assert_array_equal(saved[name], expected[name])
        index = json.loads((self.root / "diffusion_pytorch_model.safetensors.index.json").read_text())
        self.assertEqual(set(index["weight_map"]), set(expected))
        self.assertEqual(len(set(index["weight_map"].values())), len(sizes))


if __name__ == "__main__":
    unittest.main()
