"""Pure-Python orchestration tests: MLX evaluation/compile/allocation are mocked."""

import io
import json
import sys
import tempfile
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

# Reuse the backend suite's CPU-only MLX fixture for standalone invocation too.
if "test_core" not in sys.modules and "tests.test_core" not in sys.modules:
    from tests import test_core  # noqa: F401

import catalog
import speedups as exp
import bench_models as bench


class ModelSpeedupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.root = Path(self.tmp.name)
        self.ref = self.root / "ref.png"
        Image.new("RGBA", (2, 2), "red").save(self.ref)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-test")
        self.patches = [patch.object(exp.mx, "eval"), patch.object(exp.mx, "clear_cache"),
                        patch.object(exp.mx, "compile", side_effect=lambda fn: fn, create=True),
                        patch.multiple(exp.mx, **{name: Mock(return_value=0) for name in (
                            "set_memory_limit", "set_wired_limit", "set_cache_limit", "reset_peak_memory",
                            "get_peak_memory", "get_active_memory", "get_cache_memory")})]
        for p in self.patches:
            p.start()
        latent = types.ModuleType("mflux.models.common.latent_creator.latent_creator")
        latent.LatentCreator = type("LatentCreator", (), {"encode_image": staticmethod(Mock())})
        vae = types.ModuleType("mflux.models.common.vae.vae_util")
        vae.VAEUtil = type("VAEUtil", (), {"decode": staticmethod(Mock())})
        self.modules = patch.dict(sys.modules, {latent.__name__: latent, vae.__name__: vae})
        self.modules.start()

    def tearDown(self):
        self.modules.stop()
        for p in reversed(self.patches):
            p.stop()
        self.executor.shutdown()
        self.tmp.cleanup()

    def run_mlx(self, fn):
        return self.executor.submit(fn).result()

    def test_values_bounds_evaluation_deferred_and_job_clear(self):
        build = Mock(return_value=(NS(nbytes=16), NS(nbytes=16)))

        def run():
            cache = exp.Values(limit=32)
            cache.begin()
            result = cache.get("a", build, defer=True)
            self.assertEqual(cache.bytes, 0)
            exp.mx.eval.assert_not_called()
            cache.materialize()
            self.assertIs(cache.get("a", build), result)
            exp.mx.eval.assert_called_once_with(*result)
            cache.get("b", build)
            self.assertNotIn("b", cache.entries)
            cache.clear()
            self.assertEqual((cache.entries, cache.pending, cache.bytes, cache.enabled), ({}, [], 0, False))
            cache.get("a", build)
            self.assertEqual(cache.entries, {})

        self.run_mlx(run)

    def test_image_identity_uses_every_pixel_order_mode_and_size(self):
        a = Image.new("RGBA", (2, 2), "red")
        b = a.copy()
        self.assertEqual(exp.image_identity([a]), exp.image_identity([b]))
        b.putpixel((1, 1), (255, 0, 0, 0))
        self.assertNotEqual(exp.image_identity([a]), exp.image_identity([b]))
        self.assertNotEqual(exp.image_identity([a, b]), exp.image_identity([b, a]))
        self.assertNotEqual(exp.image_identity([a]), exp.image_identity([a.convert("RGB")]))

    def qwen_pipeline(self, images, *, negative=False, fail=False):
        pipeline = catalog.Pipeline.__new__(catalog.Pipeline)
        pipeline.spec = NS(family="qwen21")
        pipeline.prompt_encodings = NS(begin=Mock(), clear=Mock())
        encoded = Mock(side_effect=lambda prompt, images: (NS(nbytes=16), NS(nbytes=4)))
        vae_encode = Mock(side_effect=lambda pixels: NS(nbytes=32))
        callback = Mock()
        ctx = NS(before_loop=callback)
        model = pipeline.model = NS(_encode_prompt=encoded, vae=NS(encode=vae_encode), processor=object(),
                                   callbacks=NS(start=Mock(return_value=ctx)),
                                   transformer=NS(transformer_blocks=[]))

        def generate(**kwargs):
            positive = model._encode_prompt(kwargs["prompt"], images)
            if negative:
                model._encode_prompt("negative", images)
            conditions = [model.vae.encode(NS(shape=(1, 4, 2, 2), dtype="float32")) for im in images]
            # Blend-source encode must never hit the reference cache.
            blend = model.vae.encode(NS(shape=(1, 4, 2, 2), dtype="float32"))
            model.callbacks.start().before_loop(None)
            if fail:
                raise RuntimeError("cancelled")
            return positive, conditions, blend

        model.generate_image = generate
        return pipeline, encoded, vae_encode, callback

    def test_qwen_cache_both_branches_reference_hits_and_lifecycle(self):
        images = [Image.new("RGBA", (2, 2), "red")]
        pipeline, encode, vae_encode, callback = self.qwen_pipeline(images, negative=True)

        def run():
            exp.install(pipeline)
            pipeline.begin_job([self.ref])
            first = pipeline.model.generate_image(prompt="prompt")
            second = pipeline.model.generate_image(prompt="prompt")
            self.assertIs(first[0], second[0])
            self.assertIs(first[1][0], second[1][0])
            self.assertIsNot(first[2], second[2])
            self.assertEqual(encode.call_count, 2)
            self.assertEqual(vae_encode.call_count, 3)
            self.assertTrue(pipeline.prompt_encodings.reusable)
            self.assertEqual(pipeline.job_conditioning.values.hits, 1)
            images[0].putpixel((1, 1), (0, 0, 0, 255))
            third = pipeline.model.generate_image(prompt="prompt")
            self.assertIsNot(first[1][0], third[1][0])
            self.assertEqual(encode.call_count, 4)
            pipeline.end_job()
            self.assertEqual(pipeline.job_conditioning.values.entries, {})
            self.assertEqual(pipeline.prompt_encodings.entries, {})
            self.assertIs(pipeline.model._encode_prompt, encode)
            self.assertIs(pipeline.model.vae.encode, vae_encode)

        self.run_mlx(run)

    def test_qwen_eviction_requires_both_branches_and_handles_failure(self):
        pipeline, encode, vae_encode, callback = self.qwen_pipeline([], negative=True, fail=True)

        def run():
            exp.install(pipeline)
            pipeline.prompt_encodings.limit = 20  # positive fits, negative doesn't
            pipeline.begin_job([])
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                pipeline.model.generate_image(prompt="prompt")
            self.assertFalse(pipeline.prompt_encodings.reusable)
            self.assertIs(pipeline.model._encode_prompt, encode)
            self.assertIs(pipeline.model.vae.encode, vae_encode)
            self.assertEqual(pipeline.job_conditioning.values.pending, [])
            pipeline.end_job()

        self.run_mlx(run)

    def test_qwen_reference_materialization_follows_callback(self):
        pipeline, encode, vae_encode, callback = self.qwen_pipeline([Image.new("RGBA", (2, 2))])
        events = []
        callback.side_effect = lambda *a: events.append("evict")

        def run():
            exp.install(pipeline)
            pipeline.begin_job([self.ref])
            with patch.object(exp.mx, "eval", side_effect=lambda *a: events.append("eval")):
                pipeline.model.generate_image(prompt="prompt")
            self.assertEqual(events, ["eval", "evict", "eval"])
            pipeline.end_job()

        self.run_mlx(run)

    def test_krea_prompt_cache_preserves_cfg_below_one_and_early_eviction(self):
        original = Mock(side_effect=lambda **kw: (NS(nbytes=16), NS(nbytes=16) if kw["guidance"] != 1 else None))
        pipeline = catalog.Pipeline.__new__(catalog.Pipeline)
        pipeline.spec = NS(family="krea2")
        native = {}
        pipeline.model = NS(_encode_prompts=original, prompt_cache=native)
        original.side_effect = lambda **kw: native.setdefault(
            (kw["prompt"], kw["guidance"] != 1),
            (NS(nbytes=16), NS(nbytes=16) if kw["guidance"] != 1 else None))

        def run():
            exp.wrap_krea_prompts(pipeline)
            pipeline.begin_job([])
            first = pipeline.model._encode_prompts(prompt="p", negative_prompt="n", guidance=1)
            below = pipeline.model._encode_prompts(prompt="p", negative_prompt="n", guidance=.5)
            above = pipeline.model._encode_prompts(prompt="p", negative_prompt="n", guidance=2)
            self.assertIsNone(first[1])
            self.assertIsNotNone(below[1])
            self.assertIs(below, above)  # only embeddings are cached, never guided predictions
            self.assertTrue(pipeline.prompt_encodings.reusable)
            self.assertEqual(original.call_count, 2)
            self.assertEqual(len(native), 2)
            pipeline.end_job()
            self.assertFalse(pipeline.prompt_encodings.reusable)
            self.assertEqual(native, {})

        self.run_mlx(run)

    def test_init_helper_invalidation_and_descriptor_restored(self):
        from mflux.models.common.latent_creator.latent_creator import LatentCreator
        original = Mock(side_effect=lambda **kw: NS(nbytes=32))
        vae = object()

        def run():
            conditioning = exp.JobConditioning()
            conditioning.begin()
            with patch.object(LatentCreator, "encode_image", staticmethod(original)):
                descriptor = LatentCreator.__dict__["encode_image"]
                with self.assertRaisesRegex(RuntimeError, "cancelled"):
                    with conditioning.init_scope():
                        kwargs = dict(vae=vae, image_path=self.ref, height=1024, width=1024)
                        first = LatentCreator.encode_image(**kwargs)
                        self.assertIs(first, LatentCreator.encode_image(**kwargs))
                        for change in ({"height": 512}, {"width": 512}, {"vae": object()},
                                       {"tiling_config": "tiled"}):
                            self.assertIsNot(first, LatentCreator.encode_image(**{**kwargs, **change}))
                        self.ref.write_bytes(b"replacement")
                        self.assertIsNot(first, LatentCreator.encode_image(**kwargs))
                        raise RuntimeError("cancelled")
                self.assertIs(LatentCreator.__dict__["encode_image"], descriptor)
            conditioning.clear()
            self.assertEqual(conditioning.values.entries, {})

        self.run_mlx(run)

    def test_qwen_compile_only_cached_branch_and_released_between_jobs(self):
        result = object()
        original = Mock(return_value=(result, None))
        block = NS(forward_reference=original)

        def run():
            conditioning = exp.JobConditioning()
            conditioning.wrap_qwen_block(block)
            args = (object(), object(), object(), object(), (object(), object()))
            block.forward_reference(*args, cached=(object(), object()))
            exp.mx.compile.assert_not_called()
            conditioning.begin()
            block.forward_reference(*args)
            exp.mx.compile.assert_not_called()
            cache = (object(), object())
            self.assertEqual(block.forward_reference(*args, cached=cache), (result, None))
            exp.mx.compile.assert_called_once()
            self.assertIsNone(original.call_args.args[3])  # dataclass is not a compile input
            block.forward_reference(*args, cached=cache)
            exp.mx.compile.assert_called_once()
            conditioning.clear()
            self.assertEqual(conditioning.compiled, {})

        self.run_mlx(run)

    def test_default_init_hooks_reuse_and_restore_on_failure_for_all_families(self):
        from mflux.models.common.latent_creator.latent_creator import LatentCreator

        for family in ("flux2", "zimage", "krea2"):
            with self.subTest(family=family):
                original = Mock(side_effect=lambda **kw: NS(nbytes=32))
                predict = Mock()
                pipeline = catalog.Pipeline.__new__(catalog.Pipeline)
                pipeline.spec = NS(family=family)
                pipeline.model = NS(_predict=predict, _encode_prompts=Mock())
                vae = object()

                def generate(**kwargs):
                    result = LatentCreator.encode_image(vae=vae, image_path=self.ref, height=1024, width=1024)
                    if kwargs.get("cancel"):
                        raise RuntimeError("cancelled")
                    return result

                pipeline.model.generate_image = generate
                if family == "flux2":
                    pipeline.edit = NS(_predict=Mock(), generate_image=Mock())

                def run():
                    exp.install(pipeline)
                    if family != "flux2":
                        self.assertIs(pipeline.model._predict, predict)
                    pipeline.begin_job([self.ref])
                    with patch.object(LatentCreator, "encode_image", staticmethod(original)):
                        descriptor = LatentCreator.__dict__["encode_image"]
                        first = pipeline.model.generate_image()
                        self.assertIs(first, pipeline.model.generate_image())
                        self.assertIs(LatentCreator.__dict__["encode_image"], descriptor)
                        original.assert_called_once()
                        with self.assertRaisesRegex(RuntimeError, "cancelled"):
                            try:
                                pipeline.model.generate_image(cancel=True)
                            finally:
                                pipeline.end_job()
                        self.assertIs(LatentCreator.__dict__["encode_image"], descriptor)
                    self.assertEqual(pipeline.job_conditioning.values.entries, {})
                    pipeline.begin_job([self.ref])
                    self.assertEqual(pipeline.job_conditioning.values.bytes, 0)
                    pipeline.end_job()

                self.run_mlx(run)

    def test_default_qwen_install_compiles_cached_blocks(self):
        pipeline, _, _, _ = self.qwen_pipeline([])
        block = NS(forward_reference=Mock(return_value=(object(), None)))
        pipeline.model.transformer.transformer_blocks = [block]

        def run():
            exp.install(pipeline)
            pipeline.begin_job([])
            args = (object(), object(), object(), object(), (object(), object()))
            block.forward_reference(*args, cached=(object(), object()))
            exp.mx.compile.assert_called_once()
            pipeline.end_job()
            self.assertEqual(pipeline.job_conditioning.compiled, {})

        self.run_mlx(run)

    def test_benchmark_case_validation(self):
        spec = catalog.get("flux2-klein-9b", include_custom=False)
        args = NS(cases=None, low_memory="never")
        self.assertEqual(bench.validate_cases(spec, args), ["text", "edit1", "edit2"])
        args.cases = ["edit1", "edit1"]
        with self.assertRaisesRegex(ValueError, "only once"):
            bench.validate_cases(spec, args)
        args.cases = ["edit2"]
        with self.assertRaisesRegex(ValueError, "at least two"):
            bench.validate_cases(catalog.get("krea-2-turbo", include_custom=False), args)
        args.cases = ["init"]
        with self.assertRaisesRegex(ValueError, "init is supported"):
            bench.validate_cases(spec, args)
        args.cases = None
        args.low_memory = "always"
        with self.assertRaisesRegex(ValueError, "one --cases"):
            bench.validate_cases(spec, args)

    def test_benchmark_all_families_json_phases_and_skipped_init_steps(self):
        from mflux.models.common.vae.vae_util import VAEUtil
        import performance

        for model_id in ("flux2-klein-4b", "flux2-klein-base-4b", "flux2-klein-9b", "qwen-image-2.1",
                         "qwen-image-2.1-turbo", "z-image-turbo", "krea-2-turbo"):
            spec = catalog.get(model_id, include_custom=False)
            args = NS(model=model_id, steps=4, guidance=None, cases=None, count=2, seed=42,
                      output=str(self.root / model_id), cache_limit_mb=512, low_memory="never",
                      prompt="text", edit_prompt="edit", live_preview=True, strength=None, negative_prompt="")
            pipeline = NS(model=NS(), prompt_encodings=NS(bytes=0, reusable=False), begin_job=Mock(), end_job=Mock(),
                          preview=Mock(return_value=Image.new("RGB", (2, 2))))
            pipeline.register = lambda callback: setattr(pipeline, "callback", callback)

            def generate(**kwargs):
                print("library noise goes to stderr")
                skipped = 1 if kwargs["image_paths"] and spec.family in ("zimage", "krea2") else 0
                config = NS(num_inference_steps=4, init_time_step=skipped,
                            scheduler=NS(sigmas=np.linspace(1, 0, 5)))
                latents = np.zeros((1, 1))
                pipeline.callback.call_before_loop(latents, config)
                for t in range(skipped, 4):
                    pipeline.callback.call_in_loop(t, latents, config,
                                                   denoised=latents if spec.family == "krea2" else None)
                pipeline.callback.call_after_loop(latents)
                VAEUtil.decode(vae=None, latent=latents)
                return NS(image=Image.new("RGB", (2, 2), (kwargs["seed"], 0, 0)))

            pipeline.generate = generate
            output = io.StringIO()
            with patch.object(catalog, "Pipeline", return_value=pipeline) as build, \
                 patch.object(bench, "local_model_path", return_value=(self.root, None)), \
                 patch.object(catalog, "hf_hub_download", return_value="local.safetensors") as download, \
                 patch.object(performance, "materialize"), \
                 patch.object(VAEUtil, "decode", return_value=np.zeros((1, 1))), \
                 redirect_stdout(output):
                self.run_mlx(lambda: bench.run_benchmark(args, [self.ref, self.ref]))
            records = [json.loads(line) for line in output.getvalue().splitlines()]
            case_count = 3 if model_id == "flux2-klein-9b" else 2
            self.assertEqual(len(records), case_count * 2)
            self.assertEqual(pipeline.end_job.call_count, case_count)
            if model_id == "flux2-klein-9b":
                self.assertEqual([len(r["references"]) for r in records], [0, 0, 1, 1, 2, 2])
            self.assertEqual(build.call_args.kwargs["model_path"], str(self.root))
            for record in records:
                self.assertTrue(Path(record["png"]).is_file())
                self.assertEqual(record["total_s"], record["load_s"] + record["image_total_s"])
                expected = 3 if record["case"] == "init" else 4
                self.assertEqual(record["steps_run"], expected)
                self.assertEqual(len(record["per_step_s"]), expected)
                self.assertEqual(record["preview_count"], 1)
            for call in download.call_args_list:
                self.assertTrue(call.kwargs["local_files_only"])

    def test_local_path_never_builds_converted_or_optimized_copies(self):
        import optimized
        import downloads
        import gguf_convert

        spec = catalog.get("z-image-turbo", include_custom=False)
        with patch.object(optimized, "usable", return_value=None), \
             patch.object(downloads, "_snapshot", return_value=self.root), \
             patch.object(gguf_convert, "build_model_folder") as build, \
             patch.object(optimized, "save") as save:
            with self.assertRaisesRegex(ValueError, "existing valid"):
                bench.local_model_path(spec, 4)
            build.assert_not_called()
            save.assert_not_called()
        spec = catalog.get("flux2-klein-4b", include_custom=False)
        with patch.object(optimized, "usable", return_value=None), patch.object(downloads, "_snapshot", return_value=None):
            with self.assertRaisesRegex(ValueError, "never downloads"):
                bench.local_model_path(spec, 4)


if __name__ == "__main__":
    unittest.main()
