"""Pure-Python adapter and comparison tests; no weights or Metal calls."""

import sys
import tempfile
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

# Load the suite's CPU-only MLX fixture for standalone invocation too.
if "test_core" not in sys.modules and "tests.test_core" not in sys.modules:
    from tests import test_core  # noqa: F401

import catalog
import speedups
from bench_models import compare_dirs, pixel_difference
from PIL import Image


class FluxSpeedupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.root = Path(self.tmp.name)
        self.a, self.b = self.root / "a.png", self.root / "b.png"
        self.a.write_bytes(b"a")
        self.b.write_bytes(b"b")
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-test")
        self.eval_patch = patch.object(speedups.mx, "eval")
        self.evaluate = self.eval_patch.start()
        self.clear_patch = patch.object(speedups.mx, "clear_cache")
        self.clear_patch.start()
        self.allocator_patch = patch.multiple(speedups.mx, **{
            name: Mock(return_value=0) for name in ("set_memory_limit", "set_wired_limit", "set_cache_limit",
                                                   "reset_peak_memory", "get_peak_memory", "get_active_memory",
                                                   "get_cache_memory")})
        self.allocator_patch.start()

    def tearDown(self):
        self.allocator_patch.stop()
        self.clear_patch.stop()
        self.eval_patch.stop()
        self.executor.shutdown()
        self.tmp.cleanup()

    def run_mlx(self, function):
        return self.executor.submit(function).result()

    def test_reference_versions_preserve_order_and_replacement(self):
        first = speedups.reference_identity([self.a, self.b])
        self.assertNotEqual(first, speedups.reference_identity([self.b, self.a]))
        self.assertEqual(first, speedups.reference_identity([str(self.a), str(self.b)]))
        self.a.write_bytes(b"changed")
        self.assertNotEqual(first, speedups.reference_identity([self.a, self.b]))

    def test_reference_reuse_invalidation_bounds_and_lifetime(self):
        encode = Mock(side_effect=lambda **kw: (NS(nbytes=32), NS(nbytes=16)))
        kwargs = dict(vae=object(), tiling_config=None, image_paths=[self.a, self.b], batch_size=1)

        def run():
            cache = speedups.ReferenceConditioning(limit=48)
            cache.begin()
            first = cache.prepare(encode, **kwargs)
            cache.materialize()
            self.assertIs(first, cache.prepare(encode, **kwargs))
            self.assertEqual((cache.bytes, cache.hits, cache.misses), (48, 1, 1))
            self.assertEqual(self.evaluate.call_count, 1)
            for change in ({"image_paths": [self.b, self.a]}, {"batch_size": 2},
                           {"vae": object()}, {"tiling_config": "tiled"}):
                self.assertIsNot(first, cache.prepare(encode, **{**kwargs, **change}))
            self.a.write_bytes(b"new contents")
            self.assertIsNot(first, cache.prepare(encode, **kwargs))
            cache.clear()
            self.assertEqual((cache.value, cache.bytes, cache.enabled), (None, 0, False))
            cache.begin()
            self.assertIsNot(first, cache.prepare(encode, **kwargs))
            cache.limit = 47
            cache.begin()
            cache.prepare(encode, **kwargs)
            cache.materialize()
            cache.prepare(encode, **kwargs)
            cache.materialize()
            self.assertEqual((cache.value, cache.bytes, cache.hits), (None, 0, 0))

        self.run_mlx(run)

    def test_reference_scope_exception_restoration(self):
        original = Mock(return_value=(NS(nbytes=1), NS(nbytes=1)))

        class Helper:
            prepare_reference_image_conditioning = staticmethod(original)

        descriptor = Helper.__dict__["prepare_reference_image_conditioning"]

        def run():
            cache = speedups.ReferenceConditioning()
            cache.begin()
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                with cache.scope(Helper):
                    kwargs = dict(vae=object(), tiling_config=None, image_paths=[self.a])
                    Helper.prepare_reference_image_conditioning(**kwargs)
                    Helper.prepare_reference_image_conditioning(**kwargs)
                    self.assertEqual(original.call_count, 2)
                    self.assertIsNone(cache.value)
                    raise RuntimeError("cancelled")
            self.assertIs(Helper.__dict__["prepare_reference_image_conditioning"], descriptor)

        self.run_mlx(run)

    def test_deferred_reference_eval_follows_before_loop_eviction_boundary(self):
        result = (NS(nbytes=2), NS(nbytes=2))
        original = Mock(return_value=result)

        def run():
            cache = speedups.ReferenceConditioning()
            cache.begin()
            kwargs = dict(vae=object(), image_paths=[self.a], tiling_config=None)
            self.assertIs(cache.prepare(original, **kwargs), result)
            self.evaluate.assert_not_called()
            self.assertIsNone(cache.value)
            # The app's before_loop callback may release the text encoder here.
            cache.materialize()
            self.evaluate.assert_called_once_with(*result)
            self.assertIs(cache.prepare(original, **kwargs), result)
            original.assert_called_once()
            cache.clear()
            self.assertIsNone(cache.pending)

        self.run_mlx(run)

    def test_predictors_use_original_callable_and_release_at_end_job(self):
        build = Mock(side_effect=lambda transformer: object())
        model = NS(_predict=build)
        transformer = object()

        def run():
            cache = speedups.JobPredictors()
            cache.wrap(model, "edit")
            self.assertIsNot(model._predict(transformer), model._predict(transformer))
            cache.begin()
            first = model._predict(transformer)
            self.assertIs(first, model._predict(transformer))
            self.assertIsNot(first, model._predict(object()))
            cache.clear()
            self.assertEqual(cache.entries, {})
            cache.begin()
            self.assertIsNot(first, model._predict(transformer))

        self.run_mlx(run)

    def test_combined_hooks_materialize_changed_refs_even_with_cached_predictor(self):
        encode = Mock(side_effect=lambda **kw: (NS(nbytes=2), NS(nbytes=2)))

        class Helper:
            prepare_reference_image_conditioning = staticmethod(encode)

        module = types.ModuleType("mflux.models.flux2.variants.edit.flux2_klein_edit_helpers")
        module._Flux2KleinEditHelpers = Helper
        build = Mock(side_effect=lambda transformer: object())
        transformer = object()
        pipeline = catalog.Pipeline.__new__(catalog.Pipeline)
        pipeline.spec = NS(family="flux2")
        pipeline.model = NS(_predict=build)
        pipeline.edit = NS(_predict=build)

        def generate(**kwargs):
            result = Helper.prepare_reference_image_conditioning(
                vae=pipeline.model, image_paths=kwargs["image_paths"], tiling_config=None)
            if kwargs.get("cancel"):
                raise RuntimeError("cancelled")
            # before_loop eviction would run here, before ready_predict materializes.
            predictor = pipeline.edit._predict(transformer)
            return result, predictor

        pipeline.edit.generate_image = generate

        def run():
            pipeline.model.generate_image = Mock()
            speedups.install(pipeline)
            pipeline.begin_job([self.a])
            first = pipeline.edit.generate_image(image_paths=[self.a])
            second = pipeline.edit.generate_image(image_paths=[self.a])
            self.assertIs(first[0], second[0])
            self.assertIs(first[1], second[1])
            self.a.write_bytes(b"changed")
            third = pipeline.edit.generate_image(image_paths=[self.a])
            self.assertIsNot(first[0], third[0])
            self.assertIs(first[1], third[1])
            self.assertEqual(self.evaluate.call_count, 2)
            self.assertIsNone(pipeline.reference_conditioning.pending)
            self.assertEqual(encode.call_count, 2)
            self.assertEqual(build.call_count, 1)
            pipeline.end_job()
            self.assertIsNone(pipeline.reference_conditioning.value)
            self.assertEqual(pipeline.job_predictors.entries, {})
            pipeline.begin_job([self.a])
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                try:
                    pipeline.edit.generate_image(image_paths=[self.a], cancel=True)
                finally:
                    pipeline.end_job()
            self.assertIsNone(pipeline.reference_conditioning.pending)
            self.assertIsNone(pipeline.reference_conditioning.value)
            self.assertEqual(pipeline.job_predictors.entries, {})
            self.assertIs(Helper.__dict__["prepare_reference_image_conditioning"], descriptor)

        descriptor = Helper.__dict__["prepare_reference_image_conditioning"]
        with patch.dict(sys.modules, {module.__name__: module}):
            self.run_mlx(run)

    def test_pipeline_job_lifecycle_clears_all_speedups(self):
        pipeline = catalog.Pipeline.__new__(catalog.Pipeline)
        pipeline.reference_conditioning = Mock()
        pipeline.job_predictors = Mock()
        pipeline.job_conditioning = Mock()

        def run():
            pipeline.begin_job([])
            pipeline.end_job()
            for cache in (pipeline.reference_conditioning, pipeline.job_predictors, pipeline.job_conditioning):
                cache.begin.assert_called_once_with()
                cache.clear.assert_called_once_with()

        self.run_mlx(run)

    def test_comparison_exact_known_psnr_mismatch_and_missing_files(self):
        a, b = self.root / "A", self.root / "B"
        a.mkdir()
        b.mkdir()
        Image.new("RGB", (2, 2), "black").save(a / "image.png")
        Image.new("RGB", (2, 2), "black").save(b / "image.png")
        result = list(compare_dirs(a, b))[0]
        self.assertTrue(result["identical"])
        self.assertEqual(result["psnr_db"], "Infinity")
        Image.new("RGB", (2, 2), "white").save(b / "image.png")
        result = pixel_difference(a / "image.png", b / "image.png")
        self.assertEqual(result["max_abs_pixel_difference"], 255)
        self.assertEqual(result["mean_abs_pixel_difference"], 255)
        self.assertEqual(result["psnr_db"], 0)
        Image.new("RGB", (3, 2)).save(b / "image.png")
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            list(compare_dirs(a, b))
        (b / "image.png").unlink()
        with self.assertRaisesRegex(ValueError, "PNG sets"):
            list(compare_dirs(a, b))


if __name__ == "__main__":
    unittest.main()
