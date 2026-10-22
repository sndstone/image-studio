"""Finds disk space Image Studio can give back: unfinished downloads, stale conversions, caches, old logs,
and Hugging Face cache entries no Image Studio model uses (those may belong to other apps)."""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

from huggingface_hub import scan_cache_dir
from huggingface_hub.constants import HF_HUB_CACHE

import catalog
import optimized
from downloads import _repo_dir, _snapshot, running_logs, invalidate_caches
from gguf_convert import _stamp, _usable, staging_folders

DROPPED_DIR = catalog.APP_SUPPORT / "dropped"
REF_CACHE = catalog.APP_SUPPORT / "reference-cache"


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file() and not f.is_symlink())


def _item(id: str, category: str, label: str, detail: str, size: int, recommended: bool, paths=(), repo=None):
    return {"id": id, "category": category, "label": label, "detail": detail, "bytes": size,
            "recommended": recommended, "_paths": [str(p) for p in paths], "_repo": repo}


def _used_repos() -> set[str]:
    return {s.repo for spec in catalog.all_models() for s in spec.sources}


def _converted_ok(spec: catalog.ModelSpec, folder: Path) -> bool:
    base, gguf_snap = _snapshot(spec.sources[0].repo), _snapshot(spec.gguf[0]) if spec.gguf else None
    if base is None or gguf_snap is None or not (gguf_snap / spec.gguf[1]).exists():
        return False
    return _usable(folder, _stamp(base, (gguf_snap / spec.gguf[1]).resolve()))


def _inside(path: Path, protected: set[str]) -> bool:
    """True if path is, contains, or lies inside a protected path (custom model folders)."""
    p = str(path.resolve())
    return any(p == q or q.startswith(p + "/") or p.startswith(q + "/") for q in protected)


def scan(keep: set[str], busy_repos: set[str], protected: set[str] = frozenset()) -> list[dict]:
    items = []

    # Unfinished downloads (*.incomplete blobs), except ones being downloaded right now.
    hub = Path(HF_HUB_CACHE)
    for repo_dir in sorted(hub.glob("models--*")) if hub.exists() else []:
        repo = repo_dir.name[len("models--"):].replace("--", "/")
        partial = list((repo_dir / "blobs").glob("*.incomplete")) if (repo_dir / "blobs").exists() else []
        if partial and repo not in busy_repos and not _inside(repo_dir, protected):
            items.append(_item(f"partial:{repo}", "Unfinished downloads", repo,
                               "Left over from a cancelled or failed download; downloading again restarts it.",
                               sum(p.stat().st_size for p in partial), True, partial))

    models = {spec.id: spec for spec in catalog.all_models()}
    valid_optimized_ids = set()
    optimized_dir = catalog.APP_SUPPORT / "optimized"
    for folder in sorted(optimized_dir.iterdir()) if optimized_dir.exists() else []:
        if not folder.is_dir():
            continue
        try:
            stamp = json.loads((folder / ".optimized.json").read_text())
            ids = set(stamp["model_ids"])
            valid_ids = {id for id in ids & models.keys() for q in models[id].quant_options
                         if optimized.folder(models[id], q) == folder and optimized.usable(models[id], q) == folder}
            valid = bool(valid_ids)
        except (OSError, ValueError, KeyError, TypeError):
            ids, valid = set(), False
        if valid:
            valid_optimized_ids.update(valid_ids)
        repos = {source.repo for id in ids & models.keys() for source in models[id].sources}
        if not valid and not repos & busy_repos and not _inside(folder, protected):
            items.append(_item(f"optimized:{folder.name}", "Optimized models", folder.name,
                               "Outdated optimized copy", _size(folder), True, [folder]))

    # One-time GGUF conversions that are outdated, orphaned or half-written.
    specs = {spec.id: spec for spec in catalog.all_models() if spec.gguf}
    for folder in sorted(catalog.CONVERTED_DIR.iterdir()) if catalog.CONVERTED_DIR.exists() else []:
        if not folder.is_dir() or _inside(folder, protected):
            continue
        name = folder.name
        spec = models.get(name.removesuffix(".partial").removesuffix(".old"))
        if spec:
            repos = {source.repo for source in spec.sources}
            if spec.gguf:
                repos.add(spec.gguf[0])
            if repos & busy_repos:
                continue
        if name == "z-image-turbo" and name in valid_optimized_ids:
            detail = "No longer needed (optimized copy exists)"
        elif name.endswith((".partial", ".old")):
            detail = "Half-finished files from an interrupted conversion."
        elif name not in specs:
            detail = "Converted copy of a model that is no longer in the app."
        elif not _converted_ok(specs[name], folder):
            detail = "Outdated converted copy (the model was re-downloaded or deleted); rebuilt on next use."
        else:
            continue
        items.append(_item(f"converted:{name}", "Converted models", name, detail, _size(folder), True, [folder]))

    if REF_CACHE.exists() and any(REF_CACHE.iterdir()):
        items.append(_item("refcache", "Caches", "Converted input images",
                           "HEIC and rotated photos converted for the models; recreated when needed.",
                           _size(REF_CACHE), True, [REF_CACHE]))

    if DROPPED_DIR.exists():
        unused = [f for f in DROPPED_DIR.iterdir() if f.is_file() and str(f) not in keep]
        if unused:
            items.append(_item("dropped", "Caches", "Images dragged in from other apps",
                               f"{len(unused)} copies of images dropped from browsers or other apps; images "
                               "currently in your reference list are kept.",
                               sum(f.stat().st_size for f in unused), True, unused))

    active_logs = running_logs()
    logs = [p for p in [catalog.APP_SUPPORT / "server.previous.log", *catalog.APP_SUPPORT.glob("download-*.log")]
            if p.exists() and p not in active_logs and not _inside(p, protected) and str(p) not in keep
            and not (busy_repos and p.name.startswith("download-"))]
    if logs:
        items.append(_item("logs", "Logs", "Old logs", "Engine and download logs from earlier sessions.",
                           sum(p.stat().st_size for p in logs), True, logs))

    # Hugging Face cache repos that no Image Studio model uses. Other apps share this cache, so these
    # are listed but not recommended.
    used = _used_repos()
    try:
        repos = [r for r in scan_cache_dir().repos if r.repo_type == "model" and r.repo_id not in used]
    except Exception:
        repos = []
    for r in sorted(repos, key=lambda r: -r.size_on_disk):
        if r.repo_id in busy_repos or _inside(Path(r.repo_path), protected):
            continue
        items.append(_item(f"hf:{r.repo_id}", "Not used by Image Studio", r.repo_id,
                           "In the shared Hugging Face cache but not used by any Image Studio model. Other apps "
                           "(or your own scripts) may still need it.", r.size_on_disk, False, repo=r.repo_id))
    return items


def clean(ids: set[str], keep: set[str], busy_repos: set[str], protected: set[str] = frozenset()) -> dict:
    freed, removed, failed = 0, [], []
    for item in scan(keep, busy_repos, protected):
        if item["id"] not in ids:
            continue
        try:
            if item["_repo"]:
                _delete_repo(item["_repo"])
            else:
                for p in map(Path, item["_paths"]):
                    if p.is_dir():
                        shutil.rmtree(p)
                    else:
                        p.unlink(missing_ok=True)
            freed += item["bytes"]
            removed.append(item["id"])
        except OSError as e:
            failed.append(f"{item['label']}: {e}")
    invalidate_caches()
    return {"freed": freed, "removed": removed, "failed": failed}


def _delete_repo(repo: str) -> None:
    info = scan_cache_dir()
    targets = frozenset(r for r in info.repos if r.repo_type == "model" and r.repo_id == repo)
    revisions = [rev.commit_hash for r in targets for rev in r.revisions]
    if revisions:
        dataclasses.replace(info, repos=targets).delete_revisions(*revisions).execute()
    shutil.rmtree(_repo_dir(repo), ignore_errors=True)


def public(items: list[dict]) -> list[dict]:
    return [{k: v for k, v in item.items() if not k.startswith("_")} for item in items]
