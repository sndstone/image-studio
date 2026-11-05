"""Model Manager polling must tolerate concurrent download and cleanup writes."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import downloads


class DownloadStatusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = patch.object(downloads, "HF_HUB_CACHE", str(self.root))
        self.cache.start()
        self.addCleanup(self.cache.stop)
        self.revision = patch.object(downloads, "_revision", return_value=None)
        self.revision.start()
        self.addCleanup(self.revision.stop)

    def test_completed_blob_disappears_during_progress_poll(self):
        repo = downloads._repo_dir("org/model")
        blobs = repo / "blobs"
        blobs.mkdir(parents=True)
        vanished, active = blobs / "finished.incomplete", blobs / "active.incomplete"
        vanished.write_bytes(b"1234")
        active.write_bytes(b"12")
        original_stat = Path.stat

        def raced_stat(path, *args, **kwargs):
            if path == vanished:
                raise FileNotFoundError(path)
            return original_stat(path, *args, **kwargs)

        with patch.object(downloads, "manifest", return_value={"weights.bin": 10}), \
             patch.object(Path, "stat", raced_stat):
            self.assertEqual(downloads.source_progress("org/model", ["*.bin"]), (2, 10, False))

    def test_snapshot_removed_between_listing_and_stat(self):
        snapshots = downloads._repo_dir("org/model") / "snapshots"
        vanished, available = snapshots / "vanished", snapshots / "available"
        vanished.mkdir(parents=True)
        available.mkdir()
        original_stat = Path.stat

        def raced_stat(path, *args, **kwargs):
            if path == vanished:
                raise FileNotFoundError(path)
            return original_stat(path, *args, **kwargs)

        with patch.object(Path, "stat", raced_stat):
            self.assertEqual(downloads._snapshot("org/model"), available)

    def test_snapshot_fallback_ignores_non_directories(self):
        snapshots = downloads._repo_dir("org/model") / "snapshots"
        snapshots.mkdir(parents=True)
        (snapshots / "not-a-snapshot").write_bytes(b"stray file")
        self.assertIsNone(downloads._snapshot("org/model"))

    def test_removed_main_ref_falls_back_to_remaining_snapshot(self):
        repo = downloads._repo_dir("org/model")
        available = repo / "snapshots" / "available"
        available.mkdir(parents=True)
        ref = repo / "refs" / "main"
        ref.parent.mkdir()
        ref.write_text("available")
        original_read = Path.read_text

        def raced_read(path, *args, **kwargs):
            if path == ref:
                raise FileNotFoundError(path)
            return original_read(path, *args, **kwargs)

        with patch.object(Path, "read_text", raced_read):
            self.assertEqual(downloads._snapshot("org/model"), available)

    def test_directory_cannot_count_as_downloaded_weights(self):
        snapshots = downloads._repo_dir("org/model") / "snapshots" / "available"
        fake_weight = snapshots / "weights.bin"
        fake_weight.mkdir(parents=True)
        with patch.object(downloads, "manifest", return_value={"weights.bin": fake_weight.stat().st_size}):
            present, _, complete = downloads.source_progress("org/model", ["*.bin"])
        self.assertEqual(present, 0)
        self.assertFalse(complete)
