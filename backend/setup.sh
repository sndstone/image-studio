#!/bin/bash
# Creates the Python runtime Image Studio uses. Safe to re-run.
set -euo pipefail
RUNTIME="${1:-$HOME/Library/Application Support/Image Studio}"
RUNTIME_VERSION=3  # keep in sync with Backend.runtimeVersion
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    brew install uv
  else
    UV_VERSION=0.13.0
    # SHA-256 of uv-aarch64-apple-darwin.tar.gz from that release (checked against the published .sha256
    # and a fresh download on 2026-10-13).
    UV_SHA256=a9c1b29002cf3c83f07fa9cd8a887a3be0107d90e23189721221e7257db8e3d6
    if [[ ! "$UV_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
      echo "Cannot install uv: pinned release checksum is unavailable. Install uv with Homebrew or add a verified SHA-256 to setup.sh." >&2
      exit 1
    fi
    UV_TMP=$(mktemp -d)
    trap 'rm -rf "$UV_TMP"' EXIT
    curl --fail --location --proto '=https' --tlsv1.2 \
      "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-aarch64-apple-darwin.tar.gz" \
      -o "$UV_TMP/uv.tar.gz"
    printf '%s  %s\n' "$UV_SHA256" "$UV_TMP/uv.tar.gz" | shasum -a 256 -c -
    tar -xzf "$UV_TMP/uv.tar.gz" -C "$UV_TMP"
    mkdir -p "$HOME/.local/bin"
    install -m 755 "$UV_TMP/uv-aarch64-apple-darwin/uv" "$HOME/.local/bin/uv"
  fi
fi

mkdir -p "$RUNTIME"
# Remove any old success marker: a failed upgrade must not look healthy.
rm -f "$RUNTIME/runtime-version"
echo "Creating Python 3.12 environment in $RUNTIME/venv…"
uv venv --allow-existing --python 3.12 "$RUNTIME/venv"
echo "Installing pinned model and server dependencies…"
uv pip install --python "$RUNTIME/venv/bin/python" -r "$SCRIPT_DIR/requirements.lock"
"$RUNTIME/venv/bin/python" - <<'PY'
import mflux, mlx.core, fastapi, uvicorn, starlette, pydantic
import huggingface_hub, gguf, numpy, PIL, safetensors, transformers, tokenizers
from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.utils.image_util import ImageUtil
assert callable(PathResolution._find_complete_cached_snapshot)
assert hasattr(ImageUtil, 'embed_metadata_enabled')
print('Runtime ready.')
PY
printf '%s\n' "$RUNTIME_VERSION" > "$RUNTIME/runtime-version"
