"""Turns a ComfyUI-style Z-Image GGUF (e.g. unsloth/Z-Image-Turbo-GGUF) into the diffusers layout mflux loads.

The GGUF holds only the transformer, with ComfyUI names: fused q/k/v, `x_embedder`, `final_layer`, ...
We dequantize it tensor by tensor to bf16 safetensors and assemble a model folder next to the official
repo's text encoder, tokenizer and VAE (symlinked, not copied).
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np

from core import assert_mlx_thread, require_disk

SHARD_BYTES = 256 * 1024**2
VERSION = 1  # bump when the conversion changes so stale folders are rebuilt


def _target_names(name: str, rows: int) -> list[tuple[str, slice | None]]:
    """ComfyUI name -> [(diffusers name, row slice)]; fused qkv splits into three."""
    for old, new in (("x_embedder.", "all_x_embedder.2-1."), ("final_layer.", "all_final_layer.2-1.")):
        if name.startswith(old):
            return [(new + name[len(old):], None)]
    if name.endswith(".attention.qkv.weight"):
        prefix = name[: -len("qkv.weight")]
        if rows <= 0 or rows % 3:
            raise ValueError("Fused QKV rows must be divisible by three")
        third = rows // 3
        return [(f"{prefix}to_{part}.weight", slice(i * third, (i + 1) * third)) for i, part in enumerate("qkv")]
    name = name.replace(".attention.out.", ".attention.to_out.0.")
    name = name.replace(".attention.q_norm.", ".attention.norm_q.").replace(".attention.k_norm.", ".attention.norm_k.")
    return [(name, None)]


def convert_transformer(gguf_path: str | Path, out_dir: Path, progress=None) -> None:
    assert_mlx_thread()
    import gguf
    import mlx.core as mx

    reader = gguf.GGUFReader(str(gguf_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    shard: dict[str, mx.array] = {}
    size = 0
    files: list[str] = []
    weight_map: dict[str, str] = {}

    def flush():
        nonlocal shard, size
        if not shard:
            return
        fname = f"diffusion_pytorch_model-{len(files) + 1:05d}.safetensors"
        mx.save_safetensors(str(out_dir / fname), shard)
        files.append(fname)
        weight_map.update({k: fname for k in shard})
        shard, size = {}, 0

    total = len(reader.tensors)
    for i, tensor in enumerate(reader.tensors):
        # The reader's shape is in ggml order (fastest dimension first); torch order is the reverse.
        shape = [int(d) for d in reversed(tensor.shape)]
        values = gguf.quants.dequantize(tensor.data, tensor.tensor_type).reshape(shape).astype(np.float32, copy=False)
        for name, rows in _target_names(tensor.name, shape[0]):
            part = values if rows is None else values[rows]
            if shard and size + part.size * 2 > SHARD_BYTES:
                flush()
            shard[name] = mx.array(part).astype(mx.bfloat16)
            mx.eval(shard[name])
            size += part.size * 2
            del part
        del values
        if size >= SHARD_BYTES:
            flush()
        if progress:
            progress(i + 1, total)
    flush()
    (out_dir / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"source": Path(gguf_path).name}, "weight_map": weight_map}, indent=1)
    )


REQUIRED = ["model_index.json", "transformer/config.json", "tokenizer", "text_encoder", "vae", "scheduler"]


def _stamp(base_snapshot: Path, gguf_path: Path) -> dict:
    return {"base": str(base_snapshot.resolve()), "gguf": str(gguf_path), "size": gguf_path.stat().st_size,
            "version": VERSION}


def _usable(target: Path, stamp: dict) -> bool:
    try:
        if json.loads((target / ".converted.json").read_text()) != stamp:
            return False
        index = json.loads((target / "transformer/diffusion_pytorch_model.safetensors.index.json").read_text())
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return False
    weights = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weights, dict) or not weights:
        return False
    if not all(isinstance(f, str) and Path(f).name == f for f in weights.values()):
        return False
    shards = set(weights.values())
    links_ok = all((target / name).exists() for name in REQUIRED)  # exists() follows the symlinks
    return links_ok and all((target / "transformer" / f).is_file() for f in shards)


def staging_folders(target: Path) -> list[Path]:
    return [target.with_name(target.name + suffix) for suffix in (".partial", ".old")]


def build_model_folder(base_snapshot: Path, gguf_path: Path, target: Path, progress=None) -> Path:
    """target/ = base repo's text encoder, tokenizer, VAE, scheduler (symlinks) + converted transformer."""
    assert_mlx_thread()
    tmp, old = staging_folders(target)
    if not target.exists() and old.exists():  # interrupted swap: the previous folder is still intact
        os.replace(old, target)
    stamp = _stamp(base_snapshot, gguf_path)
    if _usable(target, stamp):
        return target
    require_disk(target, max(12 * 1024**3, gguf_path.stat().st_size * 2))
    missing = [name for name in REQUIRED if not (base_snapshot / name).exists()]
    if missing:
        raise RuntimeError(f"Download incomplete, missing {', '.join(missing)}. Download the model again.")
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        tmp.mkdir(parents=True)
        for entry in base_snapshot.iterdir():
            if entry.name != "transformer":
                os.symlink(entry.resolve() if entry.is_file() else entry, tmp / entry.name)
        transformer = tmp / "transformer"
        convert_transformer(gguf_path, transformer, progress)
        shutil.copy(base_snapshot / "transformer" / "config.json", transformer / "config.json")
        (tmp / ".converted.json").write_text(json.dumps(stamp))
        # Swap without a moment where neither folder exists.
        shutil.rmtree(old, ignore_errors=True)
        if target.exists():
            os.replace(target, old)
        os.replace(tmp, target)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)  # a partial conversion holds up to ~12 GB
        raise
    shutil.rmtree(old, ignore_errors=True)
    return target
