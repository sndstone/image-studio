"""Core regression tests. No model weights or Metal operations are used."""

import json
import os
import sys
import tempfile
import threading
import time
import types
import unittest
import weakref
from pathlib import Path
from unittest.mock import MagicMock, patch

# Replace MLX before importing the server; even allocator calls must not touch Metal in tests.
_fake_mx = types.ModuleType("mlx.core")
for _name in ("set_cache_limit", "set_memory_limit", "set_wired_limit", "get_active_memory", "get_cache_memory", "get_peak_memory", "clear_cache",
              "reset_peak_memory", "eval"):
    setattr(_fake_mx, _name, MagicMock(return_value=0))
_fake_mlx = types.ModuleType("mlx")
_fake_mlx.core = _fake_mx
sys.modules["mlx"] = _fake_mlx
sys.modules["mlx.core"] = _fake_mx
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import catalog
import downloads
import core
import gguf_convert
import optimized
import performance
import server
import timings
from fastapi import HTTPException
from PIL import Image


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.state_patch = patch.object(server, "state", server.State())
        self.state_patch.start()
        self.memory_patch = patch.object(server, "memory_snapshot", return_value={
            "pressure": "ok", "free_bytes": 24 * 1024**3, "swap_bytes": 0, "total_bytes": 24 * 1024**3})
        self.memory_patch.start()

    def tearDown(self):
        self.memory_patch.stop()
        self.state_patch.stop()
        self.temp.cleanup()

    def req(self, **kwargs):
        return server.GenerateRequest(**{"prompt": "A tree", **kwargs})

    def test_catalog_sanity(self):
        for spec in catalog.CATALOG:
            with self.subTest(model=spec.id):
                self.assertIn(spec.default_quant, spec.quant_options)
                self.assertEqual(set(spec.memory_gb), {str(q) for q in spec.quant_options})
                self.assertGreater(spec.max_megapixels, 0)
                self.assertEqual(spec.supports_mask, spec.family == "qwen21")
                self.assertEqual(spec.supports_enhance, spec.family == "qwen21")
                self.assertEqual(spec.supports_step_cache, spec.family == "qwen21" and spec.default_steps == 40)
                self.assertEqual(spec.optimizable, not spec.tool)
        self.assertEqual(catalog.get("flux2-klein-9b").max_megapixels, 1.6)
        self.assertEqual(catalog.get("krea-2-turbo").max_megapixels, 1.3)

    def test_custom_license(self):
        spec = catalog._custom_spec({"architecture": "flux2-klein-4b", "path": "/tmp/custom",
                                     "id": "custom-test", "name": "custom"})
        self.assertEqual(spec.license, "Unknown (check the model page)")
        self.assertFalse(spec.commercial)

    def test_validation(self):
        for field, value in (("width", 255), ("width", 2049), ("height", 255), ("steps", 0),
                             ("steps", 101), ("count", 0), ("count", 17), ("guidance", -1),
                             ("guidance", 21), ("guidance", float("nan")), ("strength", -0.1),
                             ("strength", 1.1), ("seed", -1), ("prompt", "  ")):
            with self.subTest(field=field, value=value), self.assertRaises(HTTPException) as caught:
                server.normalize_request(self.req(**{field: value}))
            self.assertEqual(caught.exception.status_code, 400)
        with self.assertRaises(HTTPException):
            server.normalize_request(self.req(width=2048, height=2048))
        with self.assertRaises(HTTPException):
            server.normalize_request(self.req(mask_path="mask.png", image_paths=["image.png"]))
        with self.assertRaises(HTTPException):
            server.normalize_request(self.req(model="qwen-image-2.1", auto_mask="person"))
        with self.assertRaises(HTTPException):
            server.normalize_request(self.req(model="qwen-image-2.1", enhance_prompt=True))

    def test_normalization(self):
        req, spec, paths = server.normalize_request(self.req(model="qwen-image-2.1", width=1001, height=1001,
                steps=1, guidance=0, image_paths=["input.png"], image_mode="init", strength=0,
                lora_path="lora.safetensors", lora_scale=0.4, fast=True))
        self.assertEqual((req.width, req.height, req.steps, req.guidance, req.quantize), (992, 992, 2, 1, 4))
        self.assertEqual(req.strength, 0.05)
        self.assertTrue(req.fast)
        self.assertEqual(req.loras[0].model_dump(), {"path": "lora.safetensors", "scale": 0.4})
        req, _, _ = server.normalize_request(self.req(model="qwen-image-2.1-turbo", steps=40, fast=True))
        self.assertEqual(req.steps, 6)
        self.assertFalse(req.fast)
        for model in ("flux2-klein-4b", "z-image-turbo", "krea-2-turbo"):
            req, _, _ = server.normalize_request(self.req(model=model, image_paths=["input.png"],
                                                         image_mode="init", strength=1))
            self.assertEqual(req.strength, 0.95)
        req, _, _ = server.normalize_request(self.req(lora_path="ignored", loras=[{"path": "a", "scale": 0.5},
                                                                                 {"path": "b", "scale": 1}]))
        self.assertEqual([x.path for x in req.loras], ["a", "b"])

    def test_unique_filenames_and_png(self):
        path = self.root / "120000_42.png"
        self.assertEqual(core.unique_path(path), path)
        Image.new("RGB", (16, 16)).save(path)
        self.assertEqual(core.verify_png(path), (16, 16))
        self.assertEqual(core.unique_path(path).name, "120000_42_2.png")
        path.with_name("120000_42_2.json").write_text("{}")
        self.assertEqual(core.unique_path(path).name, "120000_42_3.png")
        bad = self.root / "bad.png"
        bad.write_text("not a png")
        with self.assertRaises(OSError):
            core.verify_png(bad)

    def test_atomic_sidecar_and_effective(self):
        req, spec, _ = server.normalize_request(self.req(width=1001, height=1001, guidance=9,
                                                         image_paths=["original.png"], image_mode="init", strength=1))
        pipeline = types.SimpleNamespace(model=types.SimpleNamespace(bits=8))
        with patch.object(optimized, "base_snapshot", return_value="revision"):
            data = server._sidecar(req, spec, pipeline, 42, 1.24)
        self.assertEqual(data["width"], 992)
        self.assertEqual(data["strength"], 0.95)
        self.assertEqual(data["image_paths"], ["original.png"])
        self.assertEqual(data["effective"]["revision"], "revision")
        self.assertEqual(data["effective"]["quantize"], 8)
        path = self.root / "settings.json"
        real_replace = os.replace
        observed = []
        def observe(temp, target):
            observed.append(json.loads(Path(temp).read_text()))
            self.assertEqual(Path(temp).parent, path.parent)
            real_replace(temp, target)
        with patch.object(core.os, "replace", side_effect=observe):
            core.atomic_json(path, data)
        self.assertEqual(observed, [data])
        self.assertEqual(json.loads(path.read_text()), data)
        with patch.object(core.os, "replace", side_effect=OSError("disk")), self.assertRaises(OSError):
            core.atomic_json(path, {"changed": True})
        self.assertEqual(json.loads(path.read_text()), data)
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_steps_and_estimates(self):
        flux = catalog.get("flux2-klein-4b")
        qwen = catalog.get("qwen-image-2.1")
        self.assertEqual(timings.steps_run(flux, 4, 0.5, "init"), 2)
        self.assertEqual(timings.steps_run(flux, 4, 1.0, "init"), 3)
        self.assertEqual(timings.steps_run(qwen, 40, 0.5, "init"), 20)
        self.assertEqual(timings.steps_run(qwen, 40, 1.0, "init"), 40)
        self.assertEqual(timings._units(10, 1, 1, "init", 2, qwen.id), 30)
        self.assertEqual(timings._units(10, 1, 1, "init", 2, flux.id), 10)
        data = {"gen": {"model|8|edit": {"per_unit": 9}, "model|8|text": {"per_unit": 2},
                         "model|4|init": {"per_unit": 3}}}
        with patch.object(timings, "_read", return_value=data):
            result = timings.estimate("model", 8, "init", 10, 1, 1, 0, 1, False)
            self.assertEqual(result["seconds"], 30)
            result = timings.estimate("model", 0, "other", 10, 1, 1, 0, 1, False)
            self.assertEqual(result["seconds"], 20)
        with patch.object(timings, "FILE", self.root / "timings.json"):
            timings.record_generation(qwen.id, 4, "init", 20, 1, 1, 1, 40, 5)
            self.assertEqual(timings.estimate(qwen.id, 4, "init", 20, 1, 1, 1, 1, False)["seconds"], 45)

    def test_measured_peaks_per_kind_and_memory_mode(self):
        file = self.root / "timings.json"
        file.write_text(json.dumps({"peak": {"m|4": {"gb": 18.4, "mp": 1.0}}}))  # recorded by an older engine
        with patch.object(timings, "FILE", file):
            self.assertIsNone(timings.measured_peak_gb("m", 4))
            timings.record_peak("m", 4, int(9e9), 1.0, "text")
            timings.record_peak("m", 4, int(12e9), 1.0, "init")
            timings.record_peak("m", 4, int(6e9), 1.0, "init", low_memory=True)
            self.assertNotIn("m|4", json.loads(file.read_text())["peak"])
            self.assertEqual(timings.measured_peak_gb("m", 4, kind="text"), 9.0)
            self.assertEqual(timings.measured_peak_gb("m", 4), 12.0)  # no kind: the heaviest one
            self.assertIsNone(timings.measured_peak_gb("m", 4, kind="edit"))
            self.assertEqual(timings.measured_peak_gb("m", 4, kind="init", low_memory=True), 6.0)
            # No low-memory run of this kind yet: the normal-mode peak is the safe upper bound.
            self.assertEqual(timings.measured_peak_gb("m", 4, kind="text", low_memory=True), 9.0)
            self.assertEqual(timings.measured_peak_gb("m", 4, 2.0, kind="text"), 10.5)

    def test_shared_normalization_endpoints(self):
        bad = self.req(width=2048, height=2048)
        for endpoint in (server.generate, server.estimate, server.preflight):
            with self.assertRaises(HTTPException) as caught:
                endpoint(bad)
            self.assertEqual(caught.exception.status_code, 400)

    def test_qkv_mapping(self):
        targets = gguf_convert._target_names("layers.0.attention.qkv.weight", 12)
        self.assertEqual([name for name, _ in targets], [f"layers.0.attention.to_{x}.weight" for x in "qkv"])
        self.assertEqual([(s.start, s.stop) for _, s in targets], [(0, 4), (4, 8), (8, 12)])
        self.assertEqual(gguf_convert._target_names("x_embedder.weight", 1), [("all_x_embedder.2-1.weight", None)])
        self.assertEqual(gguf_convert._target_names("final_layer.weight", 1), [("all_final_layer.2-1.weight", None)])
        self.assertEqual(gguf_convert._target_names("a.attention.out.weight", 1), [("a.attention.to_out.0.weight", None)])
        with self.assertRaises(ValueError):
            gguf_convert._target_names("a.attention.qkv.weight", 4)

    def test_preflight_sysctl_math(self):
        def command(args, **kwargs):
            return {"kern.memorystatus_vm_pressure_level": "2\n", "vm.swapusage": "total = 4096M used = 3072M free = 1024M",
                    "hw.memsize": str(24 * 1024**3), "/usr/bin/vm_stat":
                    "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 100000.\nPages inactive: 200000.\n"}[args[-1]]
        with patch.object(core, "_memory_cached", (float("-inf"), {})), patch.object(core, "_total_ram", None), \
             patch.object(core, "_libc", None), patch.object(core.subprocess, "check_output", side_effect=command):
            snapshot = core.memory_snapshot()
        self.assertEqual(snapshot["pressure"], "warning")
        self.assertEqual(snapshot["free_bytes"], 300000 * 16384)
        self.assertEqual(snapshot["swap_bytes"], 3 * 1024**3)
        result = core.preflight_memory({"8": 12, "4": 8}, 8, snapshot)
        self.assertEqual(result["level"], "critical")
        self.assertEqual(result["suggest_quant"], 4)
        snapshot.update(free_bytes=20 * 10**9, swap_bytes=0, pressure="ok")
        self.assertEqual(core.preflight_memory({"4": 8}, 4, snapshot)["level"], "ok")
        snapshot["pressure"] = "critical"
        self.assertEqual(core.preflight_memory({"4": 8}, 4, snapshot)["level"], "critical")

    def test_disk_guard(self):
        with patch.object(core.shutil, "disk_usage", return_value=types.SimpleNamespace(free=100)), self.assertRaises(ValueError):
            core.require_disk(self.root / "does-not-exist", 1000)

    def test_auth_middleware(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            with patch.dict(os.environ, {"IMAGE_STUDIO_TOKEN": "secret"}):
                self.assertFalse(server.authorized("localhost:7861", None, 7861))
                self.assertTrue(server.authorized("localhost:7861", "secret", 7861))
                self.assertFalse(server.authorized("evil:7861", "secret", 7861))
            return

        with patch.dict(os.environ, {"IMAGE_STUDIO_TOKEN": "secret"}):
            client = TestClient(server.app, base_url="http://127.0.0.1:7861")
            self.assertEqual(client.get("/health").status_code, 403)
            self.assertEqual(client.get("/health", headers={"X-Image-Studio-Token": "wrong"}).status_code, 403)
            good = {"X-Image-Studio-Token": "secret"}
            self.assertEqual(client.get("/health", headers=good).json(), {"ok": True})
            self.assertEqual(client.get("/health", headers={**good, "Host": "evil.example:7861"}).status_code, 403)
            self.assertEqual(client.get("/health", headers={**good, "Host": "localhost:7862"}).status_code, 403)
            self.assertEqual(client.post("/generate", headers=good, json={"prompt": "a", "steps": "invalid"}).status_code, 400)
        with patch.dict(os.environ, clear=True):
            self.assertTrue(server.authorized("localhost:7861", None, 7861))
            self.assertFalse(server.authorized("evil.example:7861", None, 7861))
        with patch.dict(os.environ, {"IMAGE_STUDIO_TOKEN": ""}):
            self.assertFalse(server.authorized("localhost:7861", None, 7861))

    def test_busy_custom_cannot_mutate(self):
        server._acquire("load")
        try:
            with patch.object(catalog, "add_custom") as add, self.assertRaises(HTTPException) as caught:
                server.add_custom_model(server.CustomModelRequest(name="a", path="owner/model", architecture="flux2-klein-4b"))
            add.assert_not_called()
            self.assertEqual(caught.exception.status_code, 409)
            self.assertIn("Loading a model", caught.exception.detail)
        finally:
            server._release()
        self.assertIsNone(server.state.busy)

    def test_load_materializes_on_mlx_thread(self):
        weight = types.SimpleNamespace(nbytes=128)
        pipeline = types.SimpleNamespace(model=types.SimpleNamespace(parameters=lambda: {"weight": weight}),
                                         register=MagicMock())
        with patch.object(server.downloads, "status", return_value={"downloaded": True}), \
             patch.object(optimized, "usable", return_value=None), patch.object(catalog, "Pipeline", return_value=pipeline), \
             patch.object(timings, "record_load") as timing:
            server.on_mlx(server._load, catalog.get("flux2-klein-4b"), 8, None, 1)
        _fake_mx.eval.assert_called_with(weight)
        timing.assert_called_once()
        self.assertIs(server.state.pipeline, pipeline)
        with self.assertRaises(AssertionError):
            server._unload()

    def test_reference_downscale_cache(self):
        source = self.root / "reference.png"
        Image.new("RGB", (2000, 2000)).save(source)
        with patch.object(server, "REF_CACHE", self.root / "cache"):
            out = server._prepare_reference(str(source), 1.6)
            with Image.open(out) as image:
                self.assertLessEqual(image.width * image.height, 1.6e6)
            self.assertEqual(server._prepare_reference(str(source), 1.6), out)
            self.assertNotEqual(server._prepare_reference(str(source), 1.3), out)

    def test_partial_batch_and_swallowed_save(self):
        class FakeImage:
            image = Image.new("RGB", (16, 16))
        generated = FakeImage()
        pipeline = types.SimpleNamespace(model=types.SimpleNamespace(bits=8), progress=types.SimpleNamespace(prev="latents"),
                                         generate=MagicMock(side_effect=[generated, RuntimeError("second failed")]))
        req, spec, _ = server.normalize_request(self.req(count=3))
        with patch.object(server, "OUTPUT_ROOT", self.root), patch.object(server, "_load", return_value=pipeline), \
             patch.object(optimized, "base_snapshot", return_value="snapshot"):
            with self.assertLogs(server.log, level="ERROR"):
                result = server.on_mlx(server._generate, req, spec, [], req.image_mode)
        self.assertEqual(len(result["images"]), 1)
        self.assertIn("second failed", result["error"])
        self.assertEqual(result["images"], server.state.images_done)
        self.assertEqual(core.verify_png(Path(result["images"][0]["path"])), (16, 16))
        self.assertIsNone(pipeline.progress.prev)
        with patch.object(server, "OUTPUT_ROOT", self.root), patch.object(server, "_load", return_value=pipeline):
            pipeline.generate.side_effect = None
            pipeline.generate.return_value = types.SimpleNamespace(save=lambda **kwargs: None)
            with self.assertLogs(server.log, level="ERROR"):
                result = server.on_mlx(server._generate, req, spec, [], req.image_mode)
        self.assertEqual(result["images"], [])
        self.assertIsNotNone(result["error"])

    def test_cancel_reference_and_after_load(self):
        source = self.root / "input.png"
        Image.new("RGB", (16, 16)).save(source)
        def prepare(*args):
            server.state.cancel.set()
            return str(source)
        with patch.object(server, "_prepare_reference", side_effect=prepare), \
             patch.object(server, "_preflight", return_value={"level": "ok"}), patch.object(server, "_load") as load:
            server.state.cancel.set()  # stale cancellation is cleared first
            result = server.generate(self.req(image_paths=[str(source)]))
            load.assert_not_called()
        self.assertTrue(result["cancelled"])
        self.assertEqual(result["images"], [])
        server.state.cancel.clear()
        pipeline = types.SimpleNamespace(progress=types.SimpleNamespace(prev="retained"))
        def load(*args):
            server.state.cancel.set()
            return pipeline
        with patch.object(server, "_load", side_effect=load):
            req, spec, _ = server.normalize_request(self.req())
            result = server.on_mlx(server._generate, req, spec, [], req.image_mode)
        self.assertTrue(result["cancelled"])
        self.assertIsNone(pipeline.progress.prev)

    def test_upscale_sidecar_and_cancel(self):
        source = self.root / "source.png"
        Image.new("RGB", (16, 16)).save(source)
        core.atomic_json(source.with_suffix(".json"), {"width": 768, "height": 768, "seed": 12})
        class FakeResult:
            image = Image.new("RGB", (32, 32))
        pipeline = types.SimpleNamespace(upscale=MagicMock(return_value=FakeResult()), progress=types.SimpleNamespace(prev="latent"))
        spec = catalog.get("seedvr2-3b")
        req = server.UpscaleRequest(path=str(source))
        with patch.object(server, "_load", return_value=pipeline), patch.object(server, "REF_CACHE", self.root / "cache"):
            result = server.on_mlx(server._upscale, spec, req, source)
        meta = json.loads(Path(result["path"]).with_suffix(".json").read_text())
        self.assertEqual((meta["width"], meta["height"]), (768, 768))
        self.assertEqual((meta["output_width"], meta["output_height"]), (32, 32))
        pipeline.upscale.side_effect = server.StopImageGenerationException("Cancelled")
        with patch.object(server, "_load", return_value=pipeline), patch.object(server, "REF_CACHE", self.root / "cache"):
            result = server.on_mlx(server._upscale, spec, req, source)
        self.assertTrue(result["cancelled"])
        self.assertIsNone(pipeline.progress.prev)
        for field, value in (("scale", 1), ("scale", 5), ("softness", -1), ("softness", 2)):
            with self.assertRaises(HTTPException) as caught:
                server.upscale(server.UpscaleRequest(**{"path": str(source), field: value}))
            self.assertEqual(caught.exception.status_code, 400)

    def test_optimized_shared_and_stamp_validation(self):
        qwen = catalog.get("qwen-image-2.1")
        turbo = catalog.get("qwen-image-2.1-uncensored-turbo")
        self.assertEqual(optimized.key(qwen, 4), optimized.key(turbo, 4))
        self.assertNotEqual(optimized.key(qwen, 4), optimized.key(qwen, 8))
        root = self.root / "checkpoint"
        for component in ("transformer", "text_encoder", "vae"):
            folder = root / component
            folder.mkdir(parents=True)
            (folder / "0.safetensors").write_bytes(b"fake")
            core.atomic_json(folder / "model.safetensors.index.json", {"weight_map": {"w": "0.safetensors"}})
        core.atomic_json(root / ".optimized.json", {"key": optimized.key(qwen, 4), "quant": 4,
                         "model_ids": optimized.model_ids(qwen), "mflux_version": optimized.mflux_version(), "base_snapshot": "base"})
        with patch.object(optimized, "folder", return_value=root), patch.object(optimized, "base_snapshot", return_value="base"):
            self.assertEqual(optimized.usable(turbo, 4), root)
            (root / "vae/0.safetensors").unlink()
            self.assertIsNone(optimized.usable(qwen, 4))

    def test_own_sources_preserve_shared(self):
        spec = catalog.get("qwen-image-2.1-uncensored-turbo")
        self.assertEqual(server._own_sources(spec), [])  # both adapter repos are shared by other variants
        self.assertEqual(server._own_sources(catalog.get("flux2-klein-4b")), catalog.get("flux2-klein-4b").sources)
        with patch.object(server.downloads, "delete", return_value={"freed": 0}) as delete, \
             patch.object(server, "_converted_folders", return_value=[]), patch.object(optimized, "copies", return_value=[]):
            server.delete_model(spec.id, server.DeleteRequest(scope="own"))
        self.assertEqual(delete.call_args.args[0].sources, [])

    def test_deletion_preserves_local_hf_snapshot(self):
        spec = catalog.get("flux2-klein-4b", include_custom=False)
        with patch.object(downloads, "HF_HUB_CACHE", str(self.root / "hub")):
            snapshot = downloads._repo_dir(spec.sources[0].repo) / "snapshots/rev"
            snapshot.mkdir(parents=True)
            weight = snapshot / "weight.safetensors"
            weight.write_bytes(b"needed")
            custom = catalog._custom_spec({"architecture":spec.id, "path":str(snapshot),
                                           "id":"local-snapshot", "name":"My snapshot"})
            with patch.object(catalog, "all_models", return_value=[spec, custom]), \
                 patch.object(downloads, "all_models", return_value=[spec, custom]), \
                 patch.object(server.downloads, "delete", return_value={"freed":0}) as delete, \
                 patch.object(optimized, "copies", return_value=[]), patch.object(server, "on_mlx") as mlx:
                self.assertEqual(downloads.shared_with(spec), [custom.id])
                self.assertEqual(server._own_sources(spec), [])
                server.delete_model(spec.id, server.DeleteRequest(scope="own"))
                self.assertEqual(delete.call_args.args[0].sources, [])
                delete.reset_mock()
                with self.assertRaises(HTTPException) as caught:
                    server.delete_model(spec.id)
                self.assertEqual(caught.exception.status_code, 409)
                self.assertIn(custom.name, caught.exception.detail)
                delete.assert_not_called()
                self.assertFalse(any(call.args[0] is server._unload for call in mlx.call_args_list))
                self.assertEqual(weight.read_bytes(), b"needed")
                self.assertIsNone(server.state.busy)

    def test_deletion_protects_local_optimized_folder(self):
        spec = catalog.get("flux2-klein-4b", include_custom=False)
        folder = self.root / "optimized"
        local = folder / "nested"
        local.mkdir(parents=True)
        custom = catalog._custom_spec({"architecture":spec.id, "path":str(local),
                                       "id":"local-copy", "name":"My optimized copy"})
        with patch.object(catalog, "all_models", return_value=[spec, custom]), \
             patch.object(downloads, "all_models", return_value=[spec, custom]), \
             patch.object(optimized, "copies", return_value=[folder]), \
             patch.object(server.downloads, "delete") as delete:
            for scope in ("all", "own"):
                with self.subTest(scope=scope), self.assertRaises(HTTPException) as caught:
                    server.delete_model(spec.id, server.DeleteRequest(scope=scope))
                self.assertEqual(caught.exception.status_code, 409)
                self.assertIn(custom.name, caught.exception.detail)
            delete.assert_not_called()
            self.assertTrue(local.exists())

    def test_status_snapshots_key_once(self):
        class ChangingState(server.State):
            @property
            def model_key(self):
                value = self.saved_key
                self.saved_key = None
                return value
            @model_key.setter
            def model_key(self, value):
                self.saved_key = value
        state = ChangingState()
        spec = catalog.get("flux2-klein-4b", include_custom=False)
        state.model_key = (spec.id, 8, ())
        state.pipeline = types.SimpleNamespace(model=types.SimpleNamespace(bits=4))
        with patch.object(server, "state", state):
            self.assertEqual(server.status()["loaded_model"], spec.id)
            self.assertIsNone(server.status()["loaded_model"])
            state.model_key = (spec.id, 8, ())
            with patch.object(server.downloads, "status", return_value={"downloaded":True}), \
                 patch.object(optimized, "usable", return_value=None), patch.object(optimized, "copies", return_value=[]):
                data = server._model_json(spec, {})
            self.assertTrue(data["loaded"])
            self.assertEqual(data["loaded_quant"], 4)

    def test_models_use_one_loaded_snapshot_for_all_rows(self):
        first = catalog.get("flux2-klein-4b", include_custom=False)
        second = catalog.get("krea-2-turbo", include_custom=False)
        server.state.model_key = (first.id, 8, ())
        server.state.pipeline = types.SimpleNamespace(model=types.SimpleNamespace(bits=4))
        def mutate(spec):
            with server.state.model_status_lock:
                server.state.model_key = (second.id, 8, ())
                server.state.pipeline = types.SimpleNamespace(model=types.SimpleNamespace(bits=8))
            return {"downloaded":True}
        with patch.object(catalog, "all_models", return_value=[first, second]), \
             patch.object(downloads, "all_models", return_value=[first, second]), \
             patch.object(server, "disk_usage", return_value={}), \
             patch.object(server.downloads, "status", side_effect=mutate), \
             patch.object(optimized, "usable", return_value=None), patch.object(optimized, "copies", return_value=[]):
            rows = server.list_models()
        self.assertEqual([(r["loaded"], r["loaded_quant"]) for r in rows], [(True,4), (False,None)])

    def test_optimized_component_layouts(self):
        for family in ("flux2", "qwen21", "zimage", "krea2"):
            with self.subTest(family=family):
                root = self.root / family
                parts = ("" if family == "krea2" else "transformer", "text_encoder", "vae")
                for part in parts:
                    folder = root / part; folder.mkdir(parents=True)
                    (folder / "0.safetensors").write_bytes(b"weights")
                    core.atomic_json(folder / "model.safetensors.index.json", {"weight_map":{"w":"0.safetensors"}})
                self.assertTrue(optimized.checkpoint_valid(root, family))
                # Missing component indices and shards must not be rescued by unrelated directories.
                index = root / parts[0] / "model.safetensors.index.json"
                index.unlink()
                self.assertFalse(optimized.checkpoint_valid(root, family))
                for name in ("missing.safetensors", "../vae/0.safetensors", "", "wrong.bin"):
                    core.atomic_json(index, {"weight_map":{"w":name}})
                    self.assertFalse(optimized.checkpoint_valid(root, family))
                core.atomic_json(index, {"weight_map":{"w":"0.safetensors"}})
                (root / parts[0] / "0.safetensors").write_bytes(b"")
                self.assertFalse(optimized.checkpoint_valid(root, family))
        self.assertFalse(optimized.checkpoint_valid(self.root / "flux2", "unknown"))

    def test_krea_optimized_save_and_load_root_shards(self):
        spec = catalog.get("krea-2-turbo", include_custom=False)
        target = self.root / "krea-copy"
        def save_model(root):
            for part in ("", "text_encoder", "vae"):
                folder = Path(root) / part
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "0.safetensors").write_bytes(b"weights")
                core.atomic_json(folder / "model.safetensors.index.json", {"weight_map":{"w":"0.safetensors"}})
        pipeline = types.SimpleNamespace(model=types.SimpleNamespace(bits=4, save_model=save_model))
        with patch.object(optimized, "folder", return_value=target), patch.object(optimized, "require_disk"), \
             patch.object(optimized, "base_snapshot", return_value="krea-snapshot"):
            self.assertEqual(server.on_mlx(optimized.save, spec, 4, pipeline), target)
            self.assertEqual(optimized.usable(spec, 4), target)
            self.assertFalse((target / "transformer").exists())

    def test_low_memory_evicts_after_last_encoding(self):
        evicted = []
        drop = server._drop_text_encoders
        def evict(model):
            evicted.append(model)
            drop(model)
        for setting, pressure, should_evict in (("always", "ok", True), ("auto", "critical", True),
                                               ("auto", "ok", False), ("never", "critical", False)):
            with self.subTest(setting=setting, pressure=pressure):
                server.state = server.State()
                evicted.clear()
                model = types.SimpleNamespace(text_encoder=object())
                edit = types.SimpleNamespace(text_encoder=model.text_encoder)
                pipeline = types.SimpleNamespace(model=model, edit=edit)
                callback = server.ProgressCallback()
                pipeline.progress = callback
                server.state.pipeline, server.state.model_key = pipeline, ("model", 4, ())
                server.state.low_memory = setting
                server.state.image_index, server.state.image_count = 1, 2
                config = types.SimpleNamespace(init_time_step=0, num_inference_steps=4)
                with patch.object(server, "_drop_text_encoders", side_effect=evict), \
                     patch.object(server, "memory_snapshot", return_value={"pressure": pressure}):
                    server.on_mlx(callback.call_before_loop, 1, "prompt", "latents", config)
                    self.assertEqual(evicted, [])
                    server.state.image_index = 2
                    server.on_mlx(callback.call_before_loop, 2, "prompt", "latents", config)
                self.assertEqual(len(evicted), 2 if should_evict else 0)
                self.assertEqual(server.state.encoder_evicted, should_evict)
                server.on_mlx(server._clear_job_references, pipeline)
                server.on_mlx(server._finish_job)
                self.assertIsNone(callback.prev)
                if should_evict:
                    self.assertIsNone(server.state.pipeline)
                    self.assertIsNone(server.state.model_key)

    def test_preview_skips_on_memory_pressure(self):
        callback = server.ProgressCallback()
        callback.previews = 0
        pipeline = types.SimpleNamespace(preview=MagicMock())
        server.state.pipeline = pipeline
        server.state.step, server.state.total_steps = 2, 4
        with patch.object(server, "memory_snapshot", return_value={"pressure": "warning"}):
            server.on_mlx(callback._maybe_preview, None, "latents", None, 0)
        pipeline.preview.assert_not_called()
        self.assertTrue(server.state.live_preview)
        self.assertIn("paused", server.state.memory_warning)

    def test_qwen_new_fields_and_prompt_cache_bound(self):
        pipeline = catalog.Pipeline.__new__(catalog.Pipeline)
        pipeline.spec = catalog.get("qwen-image-2.1")
        cache = {str(i): i for i in range(20)}
        pipeline.model = types.SimpleNamespace(prompt_cache=cache, generate_image=MagicMock(return_value="image"))
        result = server.on_mlx(pipeline.generate, seed=1, prompt="edit", steps=40, width=1024, height=1024,
                guidance=1, image_paths=["reference.png"], strength=0.5, negative_prompt=None,
                image_mode="init", fast=True, mask_path="mask.png", auto_mask="person", enhance_prompt=True)
        self.assertEqual(result, "image")
        kwargs = pipeline.model.generate_image.call_args.kwargs
        self.assertTrue(kwargs["use_step_cache"])
        self.assertTrue(kwargs["enhance_prompt"])
        self.assertEqual(kwargs["mask_image"], "mask.png")
        self.assertEqual(kwargs["auto_mask"], "person")
        self.assertEqual(kwargs["strength"], 0.5)
        self.assertEqual(len(cache), 16)
        self.assertEqual(list(cache)[0], "4")

    def test_optimized_save_base_only(self):
        spec = catalog.get("qwen-image-2.1-uncensored-turbo")
        pipeline = types.SimpleNamespace(model=types.SimpleNamespace(parameters=lambda: {}, bits=4))
        with patch.object(server.downloads, "status", return_value={"downloaded": True}), \
             patch.object(optimized, "usable", return_value=None), patch.object(catalog, "Pipeline", return_value=pipeline) as build, \
             patch.object(server, "require_disk"), patch.object(optimized, "save", return_value=self.root) as save:
            result = server.on_mlx(server._optimize, spec, 4)
        self.assertTrue(result["ok"])
        self.assertTrue(build.call_args.kwargs["skip_loras"])
        self.assertIsNone(build.call_args.args[2])
        save.assert_called_once_with(spec, 4, pipeline)
        with self.assertRaises(ValueError):
            server.on_mlx(optimized.save, spec, 8, pipeline)

    def test_optimized_load_skips_conversion(self):
        spec = catalog.get("z-image-turbo")
        pipeline = types.SimpleNamespace(model=types.SimpleNamespace(parameters=lambda: {}), register=MagicMock())
        with patch.object(optimized, "usable", return_value=self.root), \
             patch.object(server.downloads, "status", return_value={"downloaded": True}) as status, \
             patch.object(catalog, "Pipeline", return_value=pipeline) as build, patch.object(timings, "record_load"):
            server.on_mlx(server._load, spec, 8, None, 1)
        self.assertEqual(build.call_args.kwargs["model_path"], str(self.root))
        self.assertEqual(status.call_args.args[0].sources, [])
        self.assertIsNone(status.call_args.args[0].model_path)

    def test_optimized_atomic_save_and_invalid_stamp(self):
        spec = catalog.get("qwen-image-2.1")
        target = self.root / "optimized"
        def save_model(root):
            root = Path(root)
            for component in ("transformer", "text_encoder", "vae"):
                folder = root / component
                folder.mkdir(parents=True)
                (folder / "0.safetensors").write_bytes(b"weights")
                core.atomic_json(folder / "model.safetensors.index.json", {"weight_map": {"w": "0.safetensors"}})
        pipeline = types.SimpleNamespace(model=types.SimpleNamespace(bits=4, save_model=save_model))
        with patch.object(optimized, "folder", return_value=target), patch.object(optimized, "require_disk"), \
             patch.object(optimized, "base_snapshot", return_value="snapshot"):
            saved = server.on_mlx(optimized.save, spec, 4, pipeline)
            self.assertEqual(optimized.usable(spec, 4), saved)
            stamp = json.loads((target / ".optimized.json").read_text())
            self.assertEqual(stamp["model_ids"], optimized.model_ids(spec))
            self.assertFalse(target.with_name(target.name + ".partial").exists())
            self.assertFalse(target.with_name(target.name + ".old").exists())
            with patch.object(optimized, "base_snapshot", return_value="new-snapshot"):
                self.assertIsNone(optimized.usable(spec, 4))
            with patch.object(optimized, "mflux_version", return_value="different"):
                self.assertIsNone(optimized.usable(spec, 4))
            core.atomic_json(target / "vae/model.safetensors.index.json", {"weight_map": ["invalid"]})
            self.assertIsNone(optimized.usable(spec, 4))

    def test_settings_offline_and_metadata(self):
        from huggingface_hub import constants
        module = types.ModuleType("mflux.utils.image_util")
        module.ImageUtil = types.SimpleNamespace(embed_metadata_enabled=True)
        with patch.dict(sys.modules, {module.__name__: module}), patch.dict(os.environ), \
             patch.object(constants, "HF_HUB_OFFLINE", False):
            result = server.set_settings(server.SettingsRequest(embed_metadata=False, offline=True, low_memory="always"))
            self.assertFalse(module.ImageUtil.embed_metadata_enabled)
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertTrue(constants.HF_HUB_OFFLINE)
            self.assertEqual(result["low_memory"], "always")
            with self.assertRaises(HTTPException) as caught:
                server.download_model("flux2-klein-4b")
            self.assertEqual(caught.exception.detail, "Offline mode is on")
            server.set_settings(server.SettingsRequest(offline=False))
            self.assertNotIn("HF_HUB_OFFLINE", os.environ)
        with self.assertRaises(HTTPException):
            server.set_settings(server.SettingsRequest(low_memory="sometimes"))

    def test_bind_failure_exit_98(self):
        bound = MagicMock()
        bound.bind.side_effect = OSError("Address already in use")
        with patch.object(server.socket, "socket", return_value=bound), patch.object(server.uvicorn, "Server") as run, \
             self.assertRaises(SystemExit) as caught:
            server._serve(7861)
        self.assertEqual(caught.exception.code, 98)
        bound.close.assert_called_once()
        run.assert_not_called()

    def test_serves_reserved_socket(self):
        bound = MagicMock()
        with patch.object(server.socket, "socket", return_value=bound), patch.object(server.uvicorn, "Server") as run:
            server._serve(7861)
        bound.bind.assert_called_once_with(("127.0.0.1", 7861))
        run.return_value.run.assert_called_once_with(sockets=[bound])
        bound.close.assert_called_once()

    def test_status_contract(self):
        server.state.images_done = [{"path": "saved.png", "seed": 1, "seconds": 1}]
        server.state.busy = "generate"
        data = server.status()
        self.assertNotIn("downloaded_bytes", data)
        for key in ("images_done", "live_preview", "busy", "memory_warning"):
            self.assertIn(key, data)
        self.assertEqual(data["images_done"], server.state.images_done)


class FakeArray:
    def __init__(self, value, nbytes=64, dependency=None):
        self.value, self.nbytes = value, nbytes
        self.dependency = dependency
        self.evaluated = False


class PerformanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.patches = [patch.object(server, "state", server.State()),
                        patch.object(server, "memory_snapshot", return_value={
                            "pressure": "ok", "free_bytes": 24 * 1024**3, "swap_bytes": 0,
                            "total_bytes": 24 * 1024**3})]
        for p in self.patches:
            p.start()
        for name in ("eval", "clear_cache", "set_cache_limit", "set_memory_limit", "set_wired_limit"):
            getattr(_fake_mx, name).reset_mock()
        optimized.invalidate_caches()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.temp.cleanup()

    @staticmethod
    def evaluate(*arrays):
        core.assert_mlx_thread()
        for array in arrays:
            array.evaluated = True
            array.dependency = None  # Model lazy-graph ownership ends on evaluation.

    def test_evaluated_encoding_cache_branches_and_job_scope(self):
        for family, method in (("flux2", "_encode_prompt_pair"), ("zimage", "_encode_prompts")):
            with self.subTest(family=family):
                encoder = object()
                calls = []
                def encode(*, prompt, negative_prompt, guidance):
                    calls.append((prompt, negative_prompt, guidance))
                    positive = FakeArray(prompt, dependency=encoder)
                    negative = FakeArray(negative_prompt, dependency=encoder) if guidance > 1 else None
                    return {"positive": positive, "extra": [negative, None]}
                model = types.SimpleNamespace(**{method: encode})
                cache = performance.PromptEncodings()
                server.on_mlx(cache.wrap, model, method, adapter=family)
                wrapped = getattr(model, method)
                def run():
                    cache.begin()
                    first = wrapped(prompt="tree", negative_prompt="bad", guidance=2)
                    second = wrapped(prompt="tree", negative_prompt="bad", guidance=3)
                    self.assertIs(first, second)  # Same guidance branch, different scale.
                    self.assertTrue(all(a.evaluated and a.dependency is None for a in performance.leaves(first)))
                    # Seed-dependent downstream arithmetic is identical with/without reuse.
                    for seed in (1, 42):
                        uncached = encode(prompt="tree", negative_prompt="bad", guidance=2)
                        self.assertEqual([f"{a.value}:{seed}" for a in performance.leaves(first)],
                                         [f"{a.value}:{seed}" for a in performance.leaves(uncached)])
                    wrapped(prompt="tree", negative_prompt="other", guidance=2)
                    wrapped(prompt="tree", negative_prompt="ignored", guidance=1)
                    wrapped(prompt="tree", negative_prompt="also ignored", guidance=0)
                    self.assertEqual(len(cache.entries), 3)
                    cache.clear()
                    self.assertEqual((cache.bytes, cache.entries, cache.enabled), (0, {}, False))
                    cache.begin()
                    wrapped(prompt="tree", negative_prompt="bad", guidance=2)
                    self.assertEqual(len(calls), 6)
                with patch.object(_fake_mx, "eval", side_effect=self.evaluate):
                    server.on_mlx(run)
                with self.assertRaises(AssertionError):
                    wrapped(prompt="tree", negative_prompt=None, guidance=1)

    def test_cache_bounds_and_reference_identity(self):
        reference = self.root / "reference.png"
        reference.write_bytes(b"reference")
        encode = MagicMock(side_effect=lambda **kw: (FakeArray(kw["prompt"], nbytes=32), None))
        model = types.SimpleNamespace(encode=encode)
        cache = performance.PromptEncodings(limit=64)
        def run():
            cache.wrap(model, "encode", adapter="edit", edit=True)
            cache.begin([str(reference)])
            first = model.encode(prompt="a", negative_prompt=None, guidance=1)
            self.assertIs(first, model.encode(prompt="a", negative_prompt=None, guidance=1))
            old = reference.stat().st_mtime_ns
            os.utime(reference, ns=(old + 1, old + 1))
            cache.set_references([str(reference)])
            self.assertIsNot(first, model.encode(prompt="a", negative_prompt=None, guidance=1))
            self.assertEqual(cache.bytes, 64)
            model.encode(prompt="b", negative_prompt=None, guidance=1)
            self.assertFalse(cache.reusable)
            self.assertLessEqual(cache.bytes, cache.limit)
            model.encode(prompt="b", negative_prompt=None, guidance=1)
            self.assertEqual(encode.call_count, 4)
            cache.clear()
        with patch.object(_fake_mx, "eval", side_effect=self.evaluate):
            server.on_mlx(run)

    def test_pipeline_wrappers_and_batch_first_image_eviction(self):
        cleared = []
        drop = server._drop_text_encoders
        def evict(model):
            # The callback must never see lazy encodings retaining the encoder.
            self.assertTrue(all(a.evaluated for entry in server.state.pipeline.prompt_encodings.entries.values()
                                for a in performance.leaves(entry)))
            drop(model)
            cleared.append(model)
        for family, model_id, method in (("flux2", "flux2-klein-base-4b", "_encode_prompt_pair"),
                                         ("zimage", "z-image-turbo", "_encode_prompts")):
            for mode in (("reference", "init") if family == "flux2" else ("init",)):
                with self.subTest(family=family, mode=mode):
                    server.state = server.State()
                    server.state.low_memory = "always"
                    server.state.image_count = 2
                    cleared.clear()
                    pipeline = catalog.Pipeline.__new__(catalog.Pipeline)
                    pipeline.spec = catalog.get(model_id)
                    encodes, references = [], []
                    def make_model(name):
                        model = types.SimpleNamespace(text_encoder=object())
                        def encode(**kwargs):
                            self.assertIsNotNone(model.text_encoder)
                            encodes.append((name, kwargs))
                            return (FakeArray(kwargs["prompt"], dependency=model.text_encoder),
                                    FakeArray(kwargs["negative_prompt"], dependency=model.text_encoder)
                                    if kwargs["guidance"] > 1 else None)
                        setattr(model, method, encode)
                        def generate_image(**kwargs):
                            embeds = getattr(model, method)(prompt=kwargs["prompt"], negative_prompt=" ",
                                                           guidance=kwargs.get("guidance", 0))
                            if "image_paths" in kwargs:
                                references.append(tuple(kwargs["image_paths"]))
                            config = types.SimpleNamespace(num_inference_steps=4, init_time_step=0)
                            pipeline.progress.call_before_loop(kwargs["seed"], kwargs["prompt"], "latents", config)
                            return tuple((a.value, kwargs["seed"]) for a in performance.leaves(embeds))
                        model.generate_image = generate_image
                        return model
                    pipeline.model = make_model("text")
                    if family == "flux2":
                        pipeline.edit = make_model("edit")
                    pipeline.progress = server.ProgressCallback()
                    server.state.pipeline = pipeline
                    path = self.root / "ref.png"
                    path.write_bytes(b"ref")
                    def run():
                        pipeline._wrap_prompt_encodings()
                        pipeline.begin_job([str(path)])
                        output = []
                        for i in range(2):
                            server.state.image_index = i + 1
                            output.append(pipeline.generate(seed=42 + i, prompt="tree", steps=4, width=512, height=512,
                                guidance=2, image_paths=[str(path)], strength=0.5, negative_prompt=None, image_mode=mode))
                            self.assertIsNone(pipeline.model.text_encoder)
                            if family == "flux2":
                                self.assertIsNone(pipeline.edit.text_encoder)
                        self.assertEqual(len(encodes), 1)
                        self.assertEqual(output[0][0], ("tree", 42))
                        self.assertEqual(output[1][0], ("tree", 43))
                        if family == "flux2" and mode == "reference":
                            self.assertEqual(references, [(str(path),), (str(path),)])
                        pipeline.end_job()
                        self.assertEqual(pipeline.prompt_encodings.bytes, 0)
                    with patch.object(server, "_drop_text_encoders", side_effect=evict), \
                         patch.object(_fake_mx, "eval", side_effect=self.evaluate):
                        server.on_mlx(run)
                    self.assertEqual(len(cleared), 2 if family == "flux2" else 1)

    def test_materialization_groups_and_components(self):
        mib = 1024**2
        arrays = [FakeArray(i, nbytes=size * mib) for i, size in enumerate((100, 100, 100, 300, 10, 20))]
        model = types.SimpleNamespace(parameters=lambda: {"transformer": {"layers": arrays[:5]},
                                                          "text_encoder": {"weight": arrays[5]}})
        events = []
        def evaluate(*group):
            self.evaluate(*group)
            events.append(tuple(a.value for a in group))
            self.assertTrue(sum(a.nbytes for a in group) <= 256 * mib or len(group) == 1)
        def clear():
            core.assert_mlx_thread()
            events.append("clear")
        with patch.object(_fake_mx, "eval", side_effect=evaluate), patch.object(_fake_mx, "clear_cache", side_effect=clear):
            server.on_mlx(performance.materialize, model)
        self.assertEqual(events, [(0, 1), "clear", (2,), "clear", (3,), "clear", (4,), "clear", (5,), "clear"])
        self.assertTrue(all(a.evaluated for a in arrays))
        with self.assertRaises(AssertionError):
            performance.materialize(model)

    def test_memory_policy_pressure_and_busy_lock(self):
        def check_thread(*args):
            self.assertTrue(threading.current_thread().name.startswith("mlx"))
        with patch.object(_fake_mx, "set_cache_limit", side_effect=check_thread) as cache_limit, \
             patch.object(_fake_mx, "set_memory_limit", side_effect=check_thread) as memory_limit, \
             patch.object(_fake_mx, "set_wired_limit", side_effect=check_thread) as wired_limit:
            server.on_mlx(server._init_mlx)
            cache_limit.assert_called_with(2 * 1024**3)
            memory_limit.assert_called_with(16 * 1024**3)
            wired_limit.assert_called_with(0)
            server._acquire("generate")
            try:
                server.on_mlx(server._begin_job)
                cache_limit.assert_called_with(512 * 1024**2)
                pipeline = types.SimpleNamespace(progress=server.ProgressCallback())
                server.state.pipeline = pipeline
                server.state.preview = b"preview"
                server.pressure(server.PressureRequest(level="critical"))
                # Notification only writes host state while the job owns the lock.
                self.assertIs(server.state.pipeline, pipeline)
                cache_limit.assert_called_with(512 * 1024**2)
                config = types.SimpleNamespace(num_inference_steps=4, init_time_step=0)
                server.on_mlx(pipeline.progress.call_in_loop, 0, 1, "tree", "latents", config, [])
                cache_limit.assert_called_with(0)
                self.assertIsNone(server.state.preview)
                self.assertIs(server.state.pipeline, pipeline)
                server.on_mlx(server._clear_job_references, pipeline)
                server.on_mlx(server._finish_job)
                self.assertIsNone(server.state.pipeline)
            finally:
                server._release()
            server.pressure(server.PressureRequest(level="normal"))
            cache_limit.assert_called_with(2 * 1024**3)
            self.assertEqual(server.status()["pressure"], "normal")
            server.state.pipeline = types.SimpleNamespace(progress=types.SimpleNamespace(prev="latents"))
            server.pressure(server.PressureRequest(level="warning"))
            cache_limit.assert_called_with(0)
            self.assertIsNotNone(server.state.pipeline)
            server.pressure(server.PressureRequest(level="critical"))
            self.assertIsNone(server.state.pipeline)
            with self.assertRaises(HTTPException):
                server.pressure(server.PressureRequest(level="invalid"))

    def test_cleanup_drops_generation_and_upscale_locals_before_sampling(self):
        for upscale in (False, True):
            with self.subTest(upscale=upscale):
                server.state = server.State()
                server.state.pressure = "normal"
                refs = []
                class Pipeline:
                    def __init__(self):
                        self.model = types.SimpleNamespace(bits=4)
                        self.progress = types.SimpleNamespace(prev=self)
                    def generate(self, **kwargs):
                        # Exercise a result retaining the pipeline as well as callback.prev.
                        return types.SimpleNamespace(image=Image.new("RGB", (16, 16)), owner=self)
                    upscale = generate
                def load(*args):
                    pipeline = Pipeline()
                    refs.append(weakref.ref(pipeline))
                    server.state.pipeline = pipeline
                    server.state.encoder_evicted = True
                    return pipeline
                def sample():
                    core.assert_mlx_thread()
                    self.assertIsNone(refs[-1]())
                source = self.root / "source.png"
                Image.new("RGB", (16, 16)).save(source)
                with patch.object(server, "_load", side_effect=load), patch.object(server, "_sample_memory", side_effect=sample), \
                     patch.object(server, "OUTPUT_ROOT", self.root), patch.object(optimized, "base_snapshot", return_value="revision"):
                    if upscale:
                        result = server.on_mlx(server._upscale, catalog.get("seedvr2-3b"),
                                               server.UpscaleRequest(path=str(source)), source)
                        self.assertFalse(result["cancelled"])
                    else:
                        req, spec, _ = server.normalize_request(server.GenerateRequest(prompt="tree", count=2))
                        result = server.on_mlx(server._generate, req, spec, [], "reference")
                        self.assertEqual(len(result["images"]), 2)
                self.assertIsNone(refs[-1]())

    def test_native_host_snapshot_matches_command_line_tools(self):
        if core._libc is None:
            self.skipTest("no libc")
        with patch.object(core, "_total_ram", None):
            try:
                native = core._read_memory_native()
            except PermissionError:
                self.skipTest("sandbox blocks native memory sysctls")
            with patch.object(core, "_libc", None):
                tools = core._read_memory_snapshot()
        self.assertEqual(native["total_bytes"], tools["total_bytes"])
        self.assertIn(native["pressure"], ("ok", "warning", "critical"))
        # Free memory moves between the two reads; the units and accounting must agree.
        self.assertLess(abs(native["free_bytes"] - tools["free_bytes"]), 2 * 1024**3)
        self.assertLess(abs(native["swap_bytes"] - tools["swap_bytes"]), 256 * 1024**2)
        with patch.object(core, "_memory_cached", (float("-inf"), {})), patch.object(core, "_total_ram", None), \
             patch.object(core, "_command") as command:
            self.assertEqual(core.total_memory(), native["total_bytes"])
            core.memory_snapshot()
            command.assert_not_called()

    def test_host_snapshot_ttl_and_total_ram_cached_forever(self):
        with patch.object(core, "_memory_cached", (float("-inf"), {})), patch.object(core, "_total_ram", None), \
             patch.object(core, "_libc", None), patch.object(core.time, "monotonic", return_value=10) as clock, \
             patch.object(core, "_command", side_effect=lambda *args: str(24 * 1024**3) if args[-1] == "hw.memsize" else "") as command:
            first = core.memory_snapshot()
            first["reusable_bytes"] = 100
            clock.return_value = 10.9
            second = core.memory_snapshot()
            self.assertNotIn("reusable_bytes", second)
            self.assertEqual(command.call_count, 4)
            clock.return_value = 11.1
            core.memory_snapshot()
            self.assertEqual(command.call_count, 7)
            self.assertEqual(sum(c.args[-1] == "hw.memsize" for c in command.call_args_list), 1)

    def test_failure_tracebacks_do_not_retain_models_during_cleanup(self):
        source = self.root / "source.png"
        Image.new("RGB", (16, 16)).save(source)
        for operation in ("load", "optimize", "generate", "upscale"):
            with self.subTest(operation=operation):
                server.state = server.State()
                server.state.pressure = "normal"
                refs = []
                class Pipeline:
                    def __init__(self):
                        self.model = self
                        self.progress = types.SimpleNamespace(prev=self)
                    def parameters(self):
                        if operation == "load":
                            raise RuntimeError("materialization failed")
                        return {}
                    def generate(self, **kwargs):
                        raise RuntimeError("generation failed")
                    upscale = generate
                def build(*args, **kwargs):
                    pipeline = Pipeline()
                    refs.append(weakref.ref(pipeline))
                    if operation in ("generate", "upscale"):
                        server.state.pipeline = pipeline
                        server.state.encoder_evicted = True
                    return pipeline
                def sample():
                    if refs:
                        self.assertIsNone(refs[-1]())
                def save(spec, quant, pipeline):
                    raise RuntimeError("save failed")
                with patch.object(catalog, "Pipeline", new=build), patch.object(server, "_sample_memory", new=sample), \
                     patch.object(server.downloads, "status", return_value={"downloaded": True}), \
                     patch.object(optimized, "usable", return_value=None), patch.object(optimized, "save", new=save), \
                     patch.object(server, "require_disk"), patch.object(server, "OUTPUT_ROOT", self.root), \
                     patch.object(server, "_load", new=build) if operation in ("generate", "upscale") else patch.object(server, "REF_CACHE", self.root / "cache"):
                    with self.assertLogs(server.log, level="ERROR"):
                        if operation == "generate":
                            req, spec, _ = server.normalize_request(server.GenerateRequest(prompt="tree"))
                            result = server.on_mlx(server._generate, req, spec, [], "reference")
                            self.assertIn("generation failed", result["error"])
                        else:
                            with self.assertRaises(HTTPException):
                                if operation == "load":
                                    server.on_mlx(server._load, catalog.get("flux2-klein-4b"), 4, None, 1)
                                elif operation == "optimize":
                                    server.on_mlx(server._optimize, catalog.get("flux2-klein-4b"), 4)
                                else:
                                    server.on_mlx(server._upscale, catalog.get("seedvr2-3b"),
                                                  server.UpscaleRequest(path=str(source)), source)
                self.assertIsNone(refs[-1]())

    def test_oversized_cache_defers_eviction_and_auto_uses_preflight_warning(self):
        saver_module = types.ModuleType("mflux.callbacks.instances.memory_saver")
        class Saver:
            def __init__(self, model, **kwargs):
                self.model = model
            def _delete_text_encoders(self):
                self.model.text_encoder = None
        saver_module.MemorySaver = Saver
        cache = performance.PromptEncodings(limit=32)
        model = types.SimpleNamespace(text_encoder=object(), encode=lambda **kw: (FakeArray("tree", nbytes=64), None))
        pipeline = types.SimpleNamespace(model=model, prompt_encodings=cache, progress=server.ProgressCallback())
        server.state.pipeline = pipeline
        server.state.low_memory = "auto"
        server.state.job_low_memory = True  # Preflight detected low availability/swap, sysctl is normal.
        server.state.image_count = 2
        def run():
            cache.wrap(model, "encode", adapter="flux2")
            cache.begin()
            model.encode(prompt="tree", negative_prompt=None, guidance=1)
            self.assertFalse(cache.reusable)
            self.assertEqual(cache.bytes, 0)
            config = types.SimpleNamespace(num_inference_steps=4, init_time_step=0)
            server.state.image_index = 1
            pipeline.progress.call_before_loop(1, "tree", "latents", config)
            self.assertIsNotNone(model.text_encoder)
            server.state.image_index = 2
            pipeline.progress.call_before_loop(2, "tree", "latents", config)
            self.assertIsNone(model.text_encoder)
            server._clear_job_references(pipeline)
        with patch.dict(sys.modules, {saver_module.__name__: saver_module}), patch.object(_fake_mx, "eval", side_effect=self.evaluate):
            server.on_mlx(run)

    def test_optimized_polling_shared_ttl_and_mutation_epoch(self):
        spec = catalog.get("qwen-image-2.1")
        variant = catalog.get("qwen-image-2.1-uncensored-turbo")
        with patch.object(optimized, "folder", return_value=self.root), \
             patch.object(optimized, "usable", return_value=self.root) as usable, \
             patch.object(optimized, "disk_bytes", return_value=10) as size, \
             patch.object(optimized.time, "monotonic", return_value=10) as clock:
            self.assertEqual(optimized.polling_info(spec, 4), (True, 30))
            self.assertEqual(optimized.polling_info(variant, 4), (True, 30))
            self.assertEqual((usable.call_count, size.call_count), (1, 3))
            clock.return_value = 20.1
            optimized.polling_info(spec, 4)
            self.assertEqual(usable.call_count, 2)
            downloads.invalidate_caches()  # Download finish/delete/cleanup invalidate this too.
            optimized.polling_info(spec, 4)
            self.assertEqual(usable.call_count, 3)
            optimized.invalidate_caches()
            usable.side_effect = lambda *args: (optimized.invalidate_caches(), self.root)[1]
            optimized.polling_info(spec, 4)
            optimized.polling_info(spec, 4)
            self.assertEqual(usable.call_count, 5)  # A stale in-flight measurement was not cached.

    def test_fast_png_lossless_atomic_and_metadata_path(self):
        pixels = bytes((i * 37 + i // 257) % 256 for i in range(257 * 193 * 4))
        image = Image.frombytes("RGBA", (257, 193), pixels)
        path = self.root / "fast.png"
        started = time.perf_counter()
        with patch.object(image, "save", wraps=image.save) as save:
            self.assertEqual(core.save_png(image, path), image.size)
            self.assertEqual(save.call_args.kwargs, {"format": "PNG", "compress_level": 6, "optimize": False})
        elapsed = time.perf_counter() - started
        self.assertGreater(elapsed, 0)
        server.log.info("Lossless 257×193 RGBA atomic PNG save/verify took %.3f ms", elapsed * 1000)
        with Image.open(path) as restored:
            self.assertEqual((restored.mode, restored.size, restored.tobytes()), ("RGBA", image.size, pixels))
            self.assertEqual(restored.info, {})
        original = path.read_bytes()
        with patch.object(core.os, "replace", side_effect=OSError("disk")), self.assertRaises(OSError):
            core.save_png(Image.new("RGB", (16, 16)), path)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(self.root.glob("*.tmp")), [])
        with patch.object(core, "verify_png", side_effect=ValueError("bad PNG")), self.assertRaises(ValueError):
            core.save_png(image, path)
        self.assertEqual(path.read_bytes(), original)
        result = types.SimpleNamespace(image=image, save=MagicMock(side_effect=lambda **kw: image.save(kw["path"])))
        server._save_result(result, self.root / "plain.png")
        result.save.assert_not_called()
        server.state.embed_metadata = True
        self.assertEqual(server._save_result(result, self.root / "meta.png"), image.size)
        result.save.assert_called_once_with(path=self.root / "meta.png", export_json_metadata=False, overwrite=False)


if __name__ == "__main__":
    unittest.main()
