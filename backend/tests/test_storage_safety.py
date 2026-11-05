"""Cleanup safety and polling I/O regressions using temporary files only."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

if "test_core" not in sys.modules and "tests.test_core" not in sys.modules:
    from tests import test_core  # noqa: F401

import catalog
import optimized
import storage


class StorageSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.patches = [
            patch.object(catalog, "APP_SUPPORT", self.root),
            patch.object(catalog, "CONVERTED_DIR", self.root / "converted"),
            patch.object(storage, "HF_HUB_CACHE", str(self.root / "hub")),
            patch.object(storage, "REF_CACHE", self.root / "reference-cache"),
            patch.object(storage, "DROPPED_DIR", self.root / "dropped"),
            patch.object(catalog, "all_models", return_value=[]),
            patch.object(storage, "running_logs", return_value=set()),
            patch.object(storage, "scan_cache_dir", return_value=NS(repos=[])),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def test_custom_files_in_caches_survive_cleanup(self):
        ref_model = storage.REF_CACHE / "custom-model"
        ref_model.mkdir(parents=True)
        weights = ref_model / "weights.safetensors"
        weights.write_bytes(b"weights")
        storage.DROPPED_DIR.mkdir()
        protected_image = storage.DROPPED_DIR / "custom.safetensors"
        protected_image.write_bytes(b"model")
        unused_image = storage.DROPPED_DIR / "unused.png"
        unused_image.write_bytes(b"unused")
        protected = {str(ref_model.resolve()), str(protected_image.resolve())}
        items = storage.scan(set(), set(), protected)
        self.assertEqual([item["id"] for item in items], ["dropped"])
        self.assertEqual(items[0]["bytes"], len(b"unused"))
        result = storage.clean({"refcache", "dropped"}, set(), set(), protected)
        self.assertEqual(result["removed"], ["dropped"])
        self.assertTrue(weights.is_file())
        self.assertTrue(protected_image.is_file())
        self.assertFalse(unused_image.exists())

    def test_scan_uses_one_catalog_snapshot(self):
        storage.scan(set(), set())
        catalog.all_models.assert_called_once_with()

    def test_size_ignores_symlink_targets_and_missing_files(self):
        cache = self.root / "cache"
        cache.mkdir()
        (cache / "weights").write_bytes(b"1234")
        outside = self.root / "outside"
        outside.write_bytes(b"do not count")
        link = cache / "linked"
        link.symlink_to(outside)
        self.assertEqual(optimized.disk_bytes(cache), 4)
        self.assertEqual(storage._size(cache), 4)
        self.assertEqual(storage._size(link), 0)
        self.assertEqual(storage._size(self.root / "missing"), 0)

    def test_disk_scan_stats_each_entry_once_and_tolerates_removal(self):
        (self.root / "weights").write_bytes(b"1234")
        vanishing = self.root / "vanishing"
        vanishing.write_bytes(b"gone")
        original = Path.lstat
        counted = []

        def read(path, *args, **kwargs):
            counted.append(path)
            if path == vanishing:
                raise FileNotFoundError(path)
            return original(path, *args, **kwargs)

        with patch.object(Path, "lstat", read):
            self.assertEqual(optimized.disk_bytes(self.root), 4)
        self.assertEqual(len(counted), 2)
        self.assertEqual(len(set(counted)), 2)


if __name__ == "__main__":
    unittest.main()
