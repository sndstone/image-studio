# Image Studio

Local image generation and editing for Apple Silicon, using FLUX.2 klein, Qwen-Image 2.1,
Z-Image Turbo, Krea 2 and SeedVR2 through [mflux](https://github.com/mflux-community/mflux) (MLX).
Generation and editing run on your Mac.

- Prompt → image (text-to-image)
- Prompt + 1…N reference images → edit / combine ("put the cat from image 1 on the sofa in image 2")
- Size presets or custom size, steps, guidance, seed, batch count, 8/4-bit/full precision, optional LoRA
- Results are saved to `~/Pictures/Image Studio/<date>/` with a `.json` sidecar of the settings

## Install

Requires an Apple Silicon Mac with macOS 14 or later. The app shows each model's memory need; Qwen-Image
and Krea 2 want 24 GB or more.

1. Download `Image-Studio-<version>.dmg` from [Releases](../../releases) and drag **Image Studio** to Applications.
2. The app is not notarized. On first launch macOS blocks it: open **System Settings → Privacy & Security**
   and click **Open Anyway** (or run `xattr -dr com.apple.quarantine "/Applications/Image Studio.app"`).
3. On first launch the app sets up its Python runtime (a few minutes). Models download from the Model Manager.

## Layout

| Path | What |
| --- | --- |
| `Sources/` | SwiftUI app |
| `backend/server.py` | Local FastAPI server that keeps the model loaded (bundled into the app) |
| `backend/setup.sh` | Creates the Python runtime in `~/Library/Application Support/Image Studio/venv` (run automatically on first launch) |
| `build.sh` | Builds `build/Image Studio.app` |

Model weights are cached by Hugging Face in `~/.cache/huggingface/hub` (~16 GB for klein 4B).
Server log: `~/Library/Application Support/Image Studio/server.log`.

## Models

Open **Model Manager** (⇧⌘M or "Manage…" next to the model picker) to download, load/unload, delete
models, or add a custom checkpoint (Hugging Face repo or local folder, e.g. from `mflux-save`).

| Model | Released | Notes |
| --- | --- | --- |
| FLUX.2 klein 4B (default) | Jan 2026 | 4 steps, multi-reference edit, Apache 2.0, ~6 s/image |
| FLUX.2 klein Base 4B | Jan 2026 | 25 steps by default, Apache 2.0 |
| FLUX.2 klein 9B | Jan 2026 | gated (token + license), non-commercial |
| Qwen-Image 2.1 Turbo | Sep 2026 | Viggle 6-step LoRA, edits with up to 10 images, best text rendering, research license, 4-bit, ~25–45 s/image |
| Qwen-Image 2.1 | Sep 2026 | same weights, 40 steps |
| Qwen-Image 2.1 Uncensored (Turbo / full) | Oct 2026 | Qwen-Image 2.1 + abenzerps' uncensored LoRA (30 MB; identical to their merged files), no content filter |
| Z-Image Turbo (GGUF Q8) | Nov 2025 | unsloth GGUF transformer, converted once to bf16 in `~/Library/Application Support/Image Studio/converted/z-image-turbo`; 9 steps, ~25 s/image, img2img from one image, Apache 2.0 |
| Krea 2 Turbo | Jun 2026 | photorealism, img2img from one image, gated, 4-bit (~18 GB) |
| SeedVR2 Upscaler (3B) | tool | "Upscale 2×" on a result; 1024² → 2048² in ~55 s, ~16 GB peak; outputs up to ~2048² |

Backend modules: `catalog.py` (model list + mflux pipeline adapters), `downloads.py` /
`download_worker.py` (cancellable downloads, progress, deletion), `gguf_convert.py` (Z-Image GGUF),
`timings.py` (learned speed per model for time estimates), `storage.py` (Clean Up Storage).

## Features

- **Time estimate** under Generate, learned from earlier runs on this Mac (`timings.json`).
- **Live preview** while generating: the model's current guess of the final image, decoded at half
  resolution every ~2 s (~0.2 s each). Toggle in Settings (⌘,).
- **Queue**: "Add to Queue" (⌘↩) while a job runs. **Compare Models…** runs one prompt on several models
  with the same seed; ⌘-click history images to compare existing results.
- **Edit This Image** / **Upscale 2×** on any result.
- **Zoom** the result: pinch, ⌘-scroll, double-click (fit ↔ 100%), ⌘= / ⌘- / ⌘0 (actual size) / ⌘9 (fit), or
  the controls at the bottom right; drag to pan.
- **Idle unload**: frees the model's memory after 10 idle minutes (Settings).
- **Clean Up Storage…** in the Model Manager: unfinished downloads, outdated conversions, caches, logs, and
  Hugging Face models no Image Studio entry uses (unchecked by default; other apps may use them).

Optimized copies live in `~/Library/Application Support/Image Studio/optimized/<key>/` and load
without quantizing again. Once Z-Image has a valid optimized copy, Clean Up Storage can remove its
bf16 converted copy. Outdated optimized copies are also listed for cleanup.

- **Combining images** ("put X from image 1 into image 2"): use Edit / combine and let the size follow the
  scene's image (the size picker defaults to the last image). In "Start from image 1" klein ignores the other
  images and Qwen keeps image 1's layout; the app switches modes when a second image is added and warns.
- **Faster defaults (measured)**: klein 4B and Z-Image default to 4-bit (12–13% faster, 4–5 GB less memory, same
  look); Qwen decodes in bf16 (same image, 2.4 GB lower peak); Qwen 40-step "Fast" mode (step cache) took 36 s
  instead of 82 s with a nearly identical image.
- **Exact speed/memory defaults (M5 Pro, 1024², 4-bit, two images)**: FLUX.2 predictor reuse saves
  ~0.3–1 s/image; klein 9B reference caching saves 0.9/1.3 s on image 2 with one/two refs. Qwen edit
  prompt/reference caches save 1.4/0.7 s on image 2; compiled blocks cut later steps 6–8%
  (Turbo 3.85→3.54 s, full 3.48→3.28 s). Init caching saves 1.2 s (Z-Image) / 2.7 s (Krea) on image 2.
  Krea prompt caching lets low-memory mode release its 8 GB encoder (denoise peak 17.5→9.7 GB).
  Staged VAE decoding cuts klein/Krea decode peaks by 2.5/2 GB. These measured changes produced
  pixel-identical PNGs; the speed-only changes left peak memory unchanged.
- **Idle cost (2026-10-22)**: with a 300-image history the app used 1.7–3.5 s CPU per idle minute and a 111–187 MB
  footprint; now 0.2 s and 75–106 MB (poll results are only applied when they change, settings are saved on change, sidecar
  metadata is cached off the main thread, the viewer decodes at screen size until you zoom, thumbnail decodes are
  deduplicated, `-wmo` build). The engine reads memory statistics in-process (`host_statistics64`/`sysctlbyname`) instead
  of starting `sysctl`/`vm_stat` (5 ms → 4 µs). Krea's prompt cache is cleared after each job, FLUX.2 "start from image"
  reuses the image encoding across a batch, plain JPEG/PNG/WebP references are no longer decoded just to check them, and
  GGUF conversion holds 256 MB shards instead of 4 GB.
- **More**: memory check before each job, low-memory mode, Optimize (pre-quantized copy) per model, masked editing
  for Qwen (brush or "Only change…"), several LoRAs, variations and sweeps, persistent queue with pause/retry,
  before/after wipe, history search and favorites, settings inspector, export (PNG/JPEG/HEIC/WebP), offline mode.

## Privacy

Setup accesses Python/package sources (and Homebrew or GitHub if uv needs installing). Model downloads,
file manifests and token checks contact Hugging Face. No other app network use is intended; prompts,
reference images and generated images stay local. Offline mode disables Hugging Face downloads.
PNG prompt/path metadata is off by default; enable it in Settings if wanted. Settings still accompany
results in local JSON sidecars. A saved Hugging Face token is stored in the macOS Keychain item
service `local.imagestudio`, account `huggingface-token`.

## Rebuild

```sh
./build.sh
```

Builds are signed with a self-signed identity kept in `~/Library/Application Support/Image Studio Signing/` (a private keychain), so macOS remembers "Always Allow" for the app's Keychain item across rebuilds. Existing project `.signing/` material is moved there on the next build. Deleting
that signing directory creates a new identity and macOS asks once more.

Setup installs exact versions from the bundled `requirements.lock`; runtime version is 3.

UI test hooks (environment variables, ignored otherwise): `IMAGE_STUDIO_SKIP_KEYCHAIN`, `IMAGE_STUDIO_TEST_MODEL`,
`IMAGE_STUDIO_TEST_REFERENCES`, `IMAGE_STUDIO_TEST_MODE`, `IMAGE_STUDIO_TEST_GUIDANCE`, `IMAGE_STUDIO_TEST_SCROLL=1`,
`IMAGE_STUDIO_TEST_COMPARE`, `IMAGE_STUDIO_TEST_SELECT`, `IMAGE_STUDIO_TEST_ZOOM`, `IMAGE_STUDIO_TEST_GENERATE` (types a prompt and presses Generate).

## Development

```sh
scripts/install-hooks.sh   # once per clone: runs scripts/ci.sh before every git push
scripts/ci.sh              # shell syntax, backend tests, Swift build with warnings as errors, privacy scan
```

CI runs locally rather than on hosted runners. The privacy scan rejects home-folder paths, private IP
addresses, `.local` hostnames, tokens, and committed images other than the app icon; add personal patterns
(names, hostnames) to `.git/info/private-patterns`, which is never committed. Backend tests use the app's runtime
in `~/Library/Application Support/Image Studio/venv` (override with `IMAGE_STUDIO_PYTHON`).

### Releases

The version lives in `VERSION`. Feature releases bump the minor version (`0.2.0`); minor fixes bump the patch
version (`0.1.1`). Add a `## <version>` section to `CHANGELOG.md`, commit, then run `scripts/release.sh`: it runs CI,
builds an ad-hoc-signed app, packages `dist/Image-Studio-<version>.dmg` with a SHA-256 file, tags `v<version>`
and publishes the GitHub release with that changelog section as notes.
