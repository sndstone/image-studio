"""App-owned MLX adapters; every operation here belongs on the MLX executor."""

import logging
from pathlib import Path

import mlx.core as mx

from core import assert_mlx_thread

log = logging.getLogger("image-studio")
ENCODING_LIMIT = 64 * 1024**2
MATERIALIZE_LIMIT = 256 * 1024**2


def leaves(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from leaves(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from leaves(child)
    elif value is not None:
        yield value


def materialize(model, limit=MATERIALIZE_LIMIT):
    """Evaluate one component at a time, grouping leaves up to about 256 MiB.

    A single larger leaf is indivisible; evaluate it alone rather than copying/slicing it.
    Traversing parameters does not evaluate their lazy loading/quantization graphs.
    """
    assert_mlx_thread()
    parameters = model.parameters()
    components = parameters.values() if isinstance(parameters, dict) else (parameters,)
    for component in components:
        group, size = [], 0
        for array in leaves(component):
            nbytes = array.nbytes
            if group and size + nbytes > limit:
                mx.eval(*group)
                group.clear()
                mx.clear_cache()
                size = 0
            group.append(array)
            size += nbytes
        if group:
            mx.eval(*group)
            group.clear()
        mx.clear_cache()


class PromptEncodings:
    """A bounded cache shared by a pipeline's adapters, enabled only during a job.

    mflux 0.22 Flux edit encodes references separately from _encode_prompt_pair;
    only the evaluated text tuple is retained here, never reference latents or KV state.
    An entry that cannot fit disables early eviction so later images can re-encode.
    """

    def __init__(self, limit=ENCODING_LIMIT):
        self.limit = limit
        self.enabled = False
        self.entries = {}
        self.bytes = 0
        self.references = ()
        self.reusable = False

    def begin(self, paths=()):
        assert_mlx_thread()
        self.clear()
        self.enabled = True
        self.set_references(paths)

    def set_references(self, paths):
        assert_mlx_thread()
        self.references = tuple((str(Path(p).resolve()), Path(p).stat().st_mtime_ns) for p in paths)

    def clear(self):
        assert_mlx_thread()
        self.entries.clear()
        self.bytes = 0
        self.references = ()
        self.reusable = False
        self.enabled = False

    def wrap(self, model, method, *, adapter, edit=False):
        assert_mlx_thread()
        encode = getattr(model, method)

        def evaluated_encode(*, prompt, negative_prompt, guidance):
            assert_mlx_thread()
            branch = guidance is not None and guidance > 1
            key = (adapter, prompt, negative_prompt if branch else None, branch,
                   self.references if edit else ())
            if self.enabled and key in self.entries:
                self.reusable = True
                return self.entries[key]
            result = encode(prompt=prompt, negative_prompt=negative_prompt, guidance=guidance)
            arrays = list(leaves(result))
            # Evaluate every returned array before the callback can drop encoder weights.
            if arrays:
                mx.eval(*arrays)
            size = sum(a.nbytes for a in arrays)
            self.reusable = False
            if self.enabled and size <= self.limit - self.bytes:
                self.entries[key] = result
                self.bytes += size
                self.reusable = True
            log.info("Evaluated %s prompt encodings: %d bytes (job cache %d bytes)", adapter, size, self.bytes)
            mx.clear_cache()
            return result

        setattr(model, method, evaluated_encode)
