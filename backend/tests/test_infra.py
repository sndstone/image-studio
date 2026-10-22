"""Infrastructure regression tests; no network, subprocess downloads, or real cache writes."""
import json
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import downloads as d
import storage as s
import optimized
import catalog
import download_worker as worker
import httpx


class InfraTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.spec = NS(id='test', sources=[catalog.Source('org/model', ['*.bin'])], model_path=None, gguf=None)
        self.patches = [patch.object(d, 'APP_SUPPORT', self.root), patch.object(d, 'MANIFEST_FILE', self.root/'manifests.json'),
                        patch.object(d, 'HF_HUB_CACHE', str(self.root/'hub')), patch.object(s, 'HF_HUB_CACHE', str(self.root/'hub')),
                        patch.object(catalog, 'APP_SUPPORT', self.root), patch.object(catalog, 'CONVERTED_DIR', self.root/'converted'),
                        patch.object(s, 'REF_CACHE', self.root/'reference-cache'), patch.object(s, 'DROPPED_DIR', self.root/'dropped'),
                        patch.object(catalog, 'all_models', return_value=[self.spec]),
                        patch.object(s, 'scan_cache_dir', return_value=NS(repos=[]))]
        for p in self.patches: p.start()
        d.invalidate_caches()
    def tearDown(self):
        for p in reversed(self.patches): p.stop()
        self.tmp.cleanup()

    def test_status_cache_and_invalidation(self):
        manager = d.DownloadManager()
        with patch.object(d, 'source_progress', return_value=(1, 2, False)) as progress:
            manager.status(self.spec); manager.status(self.spec)
            self.assertEqual(progress.call_count, 1)
            manager.cancel(self.spec); manager.status(self.spec)
            self.assertEqual(progress.call_count, 2)
            d.invalidate_caches(); manager.status(self.spec)
            self.assertEqual(progress.call_count, 3)
            with patch.object(d.time, 'monotonic', return_value=10**12): manager.status(self.spec)
            self.assertEqual(progress.call_count, 4)

    def test_disk_cache(self):
        with patch.object(d, 'scan_cache_dir', return_value=NS(repos=[])) as scan:
            d.disk_usage(); d.disk_usage(); self.assertEqual(scan.call_count, 1)
            d.invalidate_caches(); d.disk_usage(); self.assertEqual(scan.call_count, 2)

    def test_sizes_and_no_network_poll(self):
        snap = d._repo_dir('org/model')/'snapshots'/'rev'; snap.mkdir(parents=True)
        (snap/'a.bin').write_bytes(b'123')
        d.MANIFEST_FILE.write_text(json.dumps({'org/model|*.bin': {'a.bin': 4}}))
        with patch.object(d, 'HfApi', side_effect=AssertionError('network during polling')):
            self.assertEqual(d.source_progress('org/model', ['*.bin']), (3, 4, False))
            (snap/'a.bin').write_bytes(b'1234')
            self.assertEqual(d.source_progress('org/model', ['*.bin']), (4, 4, True))

    def test_cancel_disappearing_process(self):
        job = d.DownloadJob.__new__(d.DownloadJob)
        job.process = Mock(pid=123, poll=Mock(return_value=None))
        with patch.object(d.os, 'killpg', side_effect=ProcessLookupError) as kill:
            job.cancel(); kill.assert_called_once_with(123, signal.SIGTERM)
        job.process.wait.side_effect = [subprocess.TimeoutExpired('download', 10), None]
        with patch.object(d.os, 'killpg', side_effect=[None, ProcessLookupError]): job.cancel()

    def test_log_handle_closed(self):
        with patch.object(d.subprocess, 'Popen', return_value=Mock(poll=Mock(return_value=0))) as popen:
            job = d.DownloadJob('test', self.spec.sources)
            self.assertTrue(popen.call_args.kwargs['stdout'].closed)

    def test_offline_and_disk_space(self):
        manager = d.DownloadManager()
        with patch.dict(d.os.environ, {'HF_HUB_OFFLINE': '1'}):
            with self.assertRaisesRegex(RuntimeError, '^Offline mode is on$'): manager.start(self.spec)
        with patch.object(d, 'manifest'), patch.object(d, 'source_progress', return_value=(3, 10, False)), \
             patch.object(d.shutil, 'disk_usage', return_value=NS(free=6)):
            with self.assertRaisesRegex(RuntimeError, 'Not enough disk space: needs 7 bytes, 6 bytes free'): manager.start(self.spec)

    def test_followup_and_failure_records(self):
        manager = d.DownloadManager()
        failed = NS(running=False, repos={'org/model'}, error=lambda: 'failed')
        running = NS(running=True, repos={'org/model'}, error=lambda: None)
        manager.jobs = [failed, running]
        with patch.object(d, 'manifest'), patch.object(d, 'source_progress', return_value=(0, 10, False)), \
             patch.object(manager, '_launch') as launch:
            manager.start(self.spec)
            self.assertEqual(manager._pending, [('test', self.spec.sources)])
            self.assertIn(failed, manager.jobs)
            running.running = False; running.process = Mock()
            manager._finished(running)
            launch.assert_called_once_with('test', self.spec.sources)

    def test_storage_protections(self):
        partial = d._repo_dir('org/model')/'blobs'/'a.incomplete'; partial.parent.mkdir(parents=True); partial.write_bytes(b'1')
        dropped = s.DROPPED_DIR; dropped.mkdir(); keep = dropped/'keep.png'; keep.write_bytes(b'k')
        log = self.root/'download-test.log'; log.write_text('running')
        with patch.object(s, 'running_logs', return_value={log}):
            items = s.scan({str(keep)}, {'org/model'})
            self.assertFalse(items)
            items = s.scan({str(keep)}, set(), {str(partial.parent.resolve())})
            self.assertFalse(items)

    def test_optimized_and_converted(self):
        self.spec = catalog.get('z-image-turbo', include_custom=False)
        catalog.all_models.return_value = [self.spec]
        converted = catalog.CONVERTED_DIR/'z-image-turbo'; converted.mkdir(parents=True); (converted/'weight').write_bytes(b'12')
        folder = optimized.folder(self.spec, 8)
        for part in ('transformer', 'text_encoder', 'vae'):
            component = folder/part; component.mkdir(parents=True)
            (component/'0.safetensors').write_bytes(b'weight')
            (component/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'w':'0.safetensors'}}))
        base = d._repo_dir(self.spec.sources[0].repo)/'snapshots'/'rev'; base.mkdir(parents=True)
        gguf = d._repo_dir(self.spec.gguf[0])/'snapshots'/'rev'/self.spec.gguf[1]
        gguf.parent.mkdir(parents=True); gguf.write_bytes(b'gguf')
        self.assertIn('|', optimized.base_snapshot(self.spec))
        stamp = folder/'.optimized.json'
        data = {'key': optimized.key(self.spec, 8), 'quant': 8, 'model_ids':[self.spec.id],
                'mflux_version': optimized.mflux_version(), 'base_snapshot': optimized.base_snapshot(self.spec)}
        stamp.write_text(json.dumps(data))
        item_id = f'optimized:{folder.name}'
        items = {i['id']: i for i in s.scan(set(), set())}
        self.assertEqual(items['converted:z-image-turbo']['detail'], 'No longer needed (optimized copy exists)')
        self.assertNotIn(item_id, items)
        data['base_snapshot'] = 'old|gguf'; stamp.write_text(json.dumps(data))
        items = {i['id']: i for i in s.scan(set(), set())}
        self.assertEqual(items[item_id]['detail'], 'Outdated optimized copy')
        self.assertNotIn(item_id, {i['id']:i for i in s.scan(set(), {self.spec.sources[0].repo})})
        data['model_ids'] = ['removed']; stamp.write_text(json.dumps(data))
        self.assertIn(item_id, {i['id']:i for i in s.scan(set(), set())})

    def test_local_optimized_identity_and_incomplete_cleanup(self):
        base = self.root/'custom'; base.mkdir(); (base/'weight.safetensors').write_bytes(b'base')
        spec = catalog._custom_spec({'architecture':'flux2-klein-4b', 'path':str(base), 'id':'custom', 'name':'Local'})
        catalog.all_models.return_value = [spec]
        folder = optimized.folder(spec, 8)
        for part in ('transformer', 'text_encoder', 'vae'):
            component = folder/part; component.mkdir(parents=True)
            (component/'0.safetensors').write_bytes(b'weight')
            (component/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'w':'0.safetensors'}}))
        data = {'key':optimized.key(spec, 8), 'quant':8, 'model_ids':[spec.id],
                'mflux_version':optimized.mflux_version(), 'base_snapshot':optimized.base_snapshot(spec)}
        (folder/'.optimized.json').write_text(json.dumps(data))
        self.assertIn('|', data['base_snapshot'])
        self.assertEqual(s.scan(set(), set()), [])
        # Loading and cleanup agree even after original source files have been removed.
        (base/'weight.safetensors').unlink(); base.rmdir()
        self.assertEqual(optimized.usable(spec, 8), folder)
        self.assertEqual(s.scan(set(), set()), [])
        (folder/'vae/0.safetensors').unlink()
        self.assertEqual([i['id'] for i in s.scan(set(), set())], [f'optimized:{folder.name}'])

    def test_retry_auth_and_transient(self):
        with patch.object(worker, 'snapshot_download', side_effect=[httpx.ConnectError('offline')]*3+[None]) as fetch, patch.object(worker.time, 'sleep') as sleep:
            worker.download({'repo':'r', 'patterns':['*']})
            self.assertEqual(fetch.call_count, 4)
            self.assertEqual([c.args[0] for c in sleep.call_args_list], [1,2,4])
        error = worker.HfHubHTTPError('denied', response=httpx.Response(403, request=httpx.Request('GET', 'https://example.com')))
        with patch.object(worker, 'snapshot_download', side_effect=error) as fetch, patch.object(worker.time, 'sleep'):
            with self.assertRaises(worker.HfHubHTTPError): worker.download({'repo':'r', 'patterns':['*']})
            self.assertEqual(fetch.call_count, 1)


if __name__ == '__main__': unittest.main()
