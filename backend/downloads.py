"""Model downloads: download jobs (cancellable subprocesses), progress, disk usage, deletion."""

from __future__ import annotations

import dataclasses
import fnmatch
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import weakref
from pathlib import Path

from huggingface_hub import HfApi, scan_cache_dir
from huggingface_hub.constants import HF_HUB_CACHE

from catalog import APP_SUPPORT, ModelSpec, Source, all_models

MANIFEST_FILE = APP_SUPPORT / "manifests.json"
WORKER = Path(__file__).with_name("download_worker.py")
_manifest_lock = threading.Lock()
CACHE_TTL = 3.0
_disk_lock = threading.Lock()
_disk_cached = (0.0, {})
_cache_epoch = 0
_active_jobs = weakref.WeakSet()


def invalidate_caches() -> None:
    global _disk_cached, _cache_epoch
    with _disk_lock:
        _cache_epoch += 1
        _disk_cached = (0.0, {})
    import optimized

    optimized.invalidate_caches()


def running_logs() -> set[Path]:
    return {job.log for job in list(_active_jobs) if job.running}



def _repo_dir(repo: str) -> Path:
    return Path(HF_HUB_CACHE) / ("models--" + repo.replace("/", "--"))


def _load_manifests() -> dict:
    try:
        return json.loads(MANIFEST_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def manifest(repo: str, patterns: list[str], refresh: bool = False, cached_only: bool = False) -> dict[str, int] | None:
    """Files (path -> size) of a repo that match the patterns. Cached on disk so it works offline."""
    key = repo + "|" + ",".join(patterns)
    with _manifest_lock:
        cached = _load_manifests()
        if key in cached and not refresh:
            return cached[key]
    if cached_only or os.environ.get("HF_HUB_OFFLINE") == "1":
        return cached.get(key)
    try:
        files = HfApi().list_repo_tree(repo, recursive=True, expand=False)
        # huggingface_hub's allow_patterns use fnmatch too, where "*" also matches "/".
        result = {
            f.path: f.size
            for f in files
            if getattr(f, "size", None) is not None and any(fnmatch.fnmatch(f.path, p) for p in patterns)
        }
    except Exception:
        return cached.get(key)
    with _manifest_lock:
        cached = _load_manifests()
        cached[key] = result
        MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = MANIFEST_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(cached))
        os.replace(tmp, MANIFEST_FILE)
    return result


def _snapshot(repo: str) -> Path | None:
    ref = _repo_dir(repo) / "refs" / "main"
    if ref.exists():
        snap = _repo_dir(repo) / "snapshots" / ref.read_text().strip()
        if snap.is_dir():
            return snap
    snaps_dir = _repo_dir(repo) / "snapshots"
    snaps = sorted(snaps_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True) if snaps_dir.exists() else []
    return snaps[0] if snaps else None


def source_progress(repo: str, patterns: list[str]) -> tuple[int, int | None, bool]:
    """(bytes present, total bytes or None if unknown, complete?)"""
    files = manifest(repo, patterns, cached_only=True)
    snap = _snapshot(repo)
    if files is None:
        # Offline and never seen: fall back to "does mflux consider it complete".
        from mflux.models.common.resolution.path_resolution import PathResolution

        done = PathResolution._find_complete_cached_snapshot(repo, patterns) is not None
        return 0, None, done
    if not files:  # nothing matched: wrong patterns or an incompatible repo, never "complete"
        return 0, 0, False
    present = 0
    complete = True
    for path, size in files.items():
        try:
            actual = (snap / path).stat().st_size if snap is not None else 0
        except OSError:
            actual = 0
        present += min(actual, size)
        if snap is None or actual != size or not (snap / path).is_file():
            complete = False
    # Partially downloaded files are written as *.incomplete in the repo's blobs folder.
    blobs = _repo_dir(repo) / "blobs"
    if not complete and blobs.exists():
        present += sum(f.stat().st_size for f in blobs.glob("*.incomplete"))
    total = sum(files.values())
    return min(present, total), total, complete


def _local_folder_ok(path: str) -> bool:
    folder = Path(path)
    return folder.is_dir() and any(folder.rglob("*.safetensors"))


class DownloadJob:
    def __init__(self, owner: str, sources: list[Source]):
        self.owner = owner
        self.repos = {s.repo for s in sources}
        self.cancelled = False
        self._saved_error = None
        self.log = APP_SUPPORT / f"download-{owner}-{time.time_ns()}.log"
        payload = json.dumps([{"repo": s.repo, "patterns": s.patterns} for s in sources])
        APP_SUPPORT.mkdir(parents=True, exist_ok=True)
        with self.log.open("w") as log_handle:
            self.process = subprocess.Popen(
                [sys.executable, str(WORKER), payload],
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=os.environ.copy(),  # carries HF_TOKEN when the user saved one
            )

        _active_jobs.add(self)

    @property
    def running(self) -> bool:
        return self.process.poll() is None

    def cancel(self) -> None:
        self.cancelled = True
        if self.running:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=10)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait()

    def error(self) -> str | None:
        if self._saved_error is not None:
            return self._saved_error
        if self.running or self.process.returncode == 0:
            return None
        if self.cancelled:
            return "Cancelled"
        try:
            lines = [l for l in self.log.read_text().splitlines() if l.strip()]
        except OSError:
            lines = []
        text = "\n".join(lines[-30:])
        if "GatedRepoError" in text or "gated" in text.lower() or " 401" in text or " 403" in text:
            repo = sorted(self.repos)[0]
            return (f"Access denied. Add your Hugging Face token (API key) and accept the license at "
                    f"https://huggingface.co/{repo}")
        if "No space left" in text:
            return "Not enough disk space."
        return lines[-1][-300:] if lines else f"Download failed (exit {self.process.returncode})"


class DownloadManager:
    """Jobs are tracked per repo set: entries that share a repo (Qwen-Image 2.1 and its Turbo)
    share the job downloading it, and cancel/status apply to every job touching their repos."""

    def __init__(self):
        self.jobs: list[DownloadJob] = []
        self._lock = threading.Lock()
        self._deleting: set[str] = set()
        self._status_cache = {}
        self._pending = []

    def invalidate(self) -> None:
        self._status_cache.clear()
        invalidate_caches()

    def _launch(self, owner, sources):
        job = DownloadJob(owner, sources)
        self.jobs.append(job)  # retain failed jobs for their error records
        threading.Thread(target=self._finished, args=(job,), daemon=True).start()

    def _finished(self, job):
        job.process.wait()
        job._saved_error = job.error()
        with self._lock:
            self.invalidate()
            busy = {r for j in self.jobs if j.running for r in j.repos}
            for owner, sources in list(self._pending):
                if not ({s.repo for s in sources} & (busy | self._deleting)):
                    self._pending.remove((owner, sources))
                    self._launch(owner, sources)
                    busy.update(s.repo for s in sources)

    def _jobs(self) -> list[DownloadJob]:
        with self._lock:
            return list(self.jobs)

    def start(self, spec: ModelSpec) -> None:
        if os.environ.get("HF_HUB_OFFLINE") == "1":
            raise RuntimeError("Offline mode is on")
        for source in spec.sources:
            manifest(source.repo, source.patterns, refresh=True)
        needed = sum(max(0, (total or 0) - present)
                     for present, total, _ in (source_progress(s.repo, s.patterns) for s in spec.sources))
        Path(HF_HUB_CACHE).mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(HF_HUB_CACHE).free
        if needed > free:
            raise RuntimeError(f"Not enough disk space: needs {needed:,} bytes, {free:,} bytes free")
        with self._lock:
            if self._deleting & {s.repo for s in spec.sources}:
                raise RuntimeError("This model is being deleted.")
            busy = {repo for job in self.jobs if job.running for repo in job.repos}
            ready = [s for s in spec.sources if s.repo not in busy]
            pending = [s for s in spec.sources if s.repo in busy]
            if pending and (spec.id, pending) not in self._pending:
                self._pending.append((spec.id, pending))
            if ready:
                self._launch(spec.id, ready)
            self.invalidate()

    def cancel(self, spec: ModelSpec) -> None:
        repos = {s.repo for s in spec.sources}
        with self._lock:
            self._pending = [(owner, sources) for owner, sources in self._pending
                             if not repos & {s.repo for s in sources}]
            self.invalidate()
        for job in self._jobs():
            if job.running and job.repos & repos:
                job.cancel()

    def cancel_all(self) -> None:
        with self._lock:
            self._pending.clear()
            self.invalidate()
        for job in self._jobs():
            if job.running:
                job.cancel()

    def status(self, spec: ModelSpec) -> dict:
        key = (_cache_epoch, spec.id, spec.model_path, tuple((s.repo, tuple(s.patterns)) for s in spec.sources))
        now = time.monotonic()
        with self._lock:
            cached = self._status_cache.get(key)
            if cached and now - cached[0] < CACHE_TTL:
                return dict(cached[1])
        present, total, complete = 0, 0, True
        unknown = False
        for source in spec.sources:
            p, t, c = source_progress(source.repo, source.patterns)
            present += p
            if t is None:
                unknown = True
            else:
                total += t
            complete = complete and c
        if spec.model_path and os.path.isabs(spec.model_path):
            complete = _local_folder_ok(spec.model_path)
        repos = {s.repo for s in spec.sources}
        jobs = [j for j in self._jobs() if j.repos & repos]
        downloading = any(j.running for j in jobs)
        with self._lock:
            downloading |= any(repos & {s.repo for s in sources} for _, sources in self._pending)
        error = None
        if not complete and not downloading:
            error = next((j.error() for j in reversed(jobs) if j.error()), None)
        result = {
            "downloaded": complete and not downloading,
            "downloading": downloading,
            "bytes_present": present,
            "bytes_total": None if unknown else total,
            "error": error,
        }

        with self._lock:
            # A mutation may have completed while this status was being measured.
            if key[0] == _cache_epoch:
                self._status_cache = {k: v for k, v in self._status_cache.items() if k[0] == _cache_epoch}
                self._status_cache[key] = (now, result)
        return dict(result)

    def delete(self, spec: ModelSpec) -> dict:
        """Removes the model's repos (and any partial downloads) from the Hugging Face cache."""
        repos = {s.repo for s in spec.sources}
        with self._lock:
            self._deleting |= repos
        try:
            self.cancel(spec)
            info = scan_cache_dir()
            # Scope to exactly these model repos: delete_revisions matches bare commit hashes
            # against every repo it is given.
            targets = frozenset(r for r in info.repos if r.repo_type == "model" and r.repo_id in repos)
            freed = 0
            revisions = [rev.commit_hash for r in targets for rev in r.revisions]
            if revisions:
                strategy = dataclasses.replace(info, repos=targets).delete_revisions(*revisions)
                freed = strategy.expected_freed_size
                strategy.execute()
            for repo in repos:
                folder = _repo_dir(repo)
                for partial in (folder / "blobs").glob("*.incomplete") if (folder / "blobs").exists() else []:
                    freed += partial.stat().st_size
                    partial.unlink(missing_ok=True)
                snapshots = folder / "snapshots"
                if folder.exists() and (not snapshots.exists() or not any(snapshots.iterdir())):
                    shutil.rmtree(folder, ignore_errors=True)
            return {"freed": freed}
        finally:
            with self._lock:
                self._deleting -= repos
                self.invalidate()


def local_models_using(paths, exclude=None, models=None) -> list[ModelSpec]:
    """Local checkpoint folders overlapping paths whose files would be removed."""
    roots = [Path(path).resolve() for path in paths]
    protected = []
    for model in (all_models() if models is None else models):
        if model.id == exclude or not model.model_path or not os.path.isabs(model.model_path):
            continue
        local = Path(model.model_path).resolve()
        if any(root == local or root in local.parents or local in root.parents for root in roots):
            protected.append(model)
    return protected


def shared_with(spec: ModelSpec, models=None) -> list[str]:
    models = all_models() if models is None else models
    repos = {s.repo for s in spec.sources}
    local = {m.id for m in local_models_using(map(_repo_dir, repos), exclude=spec.id, models=models)}
    return [m.id for m in models if m.id != spec.id and (m.id in local or repos & {s.repo for s in m.sources})]


def disk_usage() -> dict[str, int]:
    global _disk_cached
    with _disk_lock:
        now = time.monotonic()
        if now - _disk_cached[0] < CACHE_TTL:
            return dict(_disk_cached[1])
        try:
            result = {r.repo_id: r.size_on_disk for r in scan_cache_dir().repos if r.repo_type == "model"}
        except Exception:
            result = {}
        _disk_cached = (now, result)
        return dict(result)
