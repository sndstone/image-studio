# Changelog

Versioning: `x.y.0` for feature releases, `x.y.z` (z ≥ 1) for minor fixes.

## 0.1.0

First public release.

- Text-to-image and multi-reference editing with FLUX.2 klein (4B, Base 4B, 9B), Qwen-Image 2.1 (Turbo, full,
  uncensored LoRA), Z-Image Turbo (GGUF) and Krea 2 Turbo through mflux/MLX; SeedVR2 2× upscaling.
- Model Manager: downloads, load/unload, pre-quantized optimized copies, custom checkpoints, Clean Up Storage.
- Live preview, queue with pause/retry, Compare Models, masked editing (Qwen), LoRAs, variations and sweeps,
  history search and favorites, before/after wipe, export to PNG/JPEG/HEIC/WebP, offline mode.
- Memory guard before each job, low-memory mode, idle unload, learned time estimates.
- Low idle cost: about 0.2 s CPU per idle minute with a 300-image history; memory statistics are read in-process.
