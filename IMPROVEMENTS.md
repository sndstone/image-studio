# Image Studio: improvement list

**Status 2026-10-13:** all items below were implemented (four gpt-6.1-sol workers + two fix rounds after two review rounds), except distribution-only items (hardened runtime/notarization: not needed for a locally built app). Benchmarks and the reference-composition investigation are summarized in README.md.

Merged from three independent read-only reviews on 2026-10-13: OpenAI gpt-6.1-sol (via Codex), Gemini 3.8 Flash
(High) and Claude Opus 5.5, plus notes from building and testing the app.
Tags: **O** = OpenAI, **G** = Gemini, **C** = Claude Opus, **me** = observed while building and testing.
✔ = checked against the code. Impact/effort: H/M/L.

## Top 10 (recommended order)

| # | Improvement | Found by | Impact / effort |
|---|---|---|---|
| 1 | "Load" doesn't really load: mflux loads and quantizes lazily, so the first image pays the full load, and estimates/memory figures are off ✔ | C | H / S |
| 2 | Opening a second window (⌘N) boots again and restarts the engine mid-work ✔ | O | H / S |
| 3 | Pin `mflux==0.22.0` and dependencies (lock file, no `curl \| sh`), plus a smoke test of the mflux internals we rely on | O G C | H / S |
| 4 | Limits on size/steps/count (typing 4096 → out of memory), plus a memory-pressure guard (also server-side for queued jobs) | O C G me | H / S-M |
| 5 | Free the text encoder after reading the prompt (mflux's own CLI does: "wastes 8–12 GB"), at least under memory pressure ✔ | C | H / M |
| 6 | Save pre-quantized copies (Qwen 4-bit, Z-Image 8-bit): loads read ~12 GB instead of 33 GB; drop the GGUF + bf16 copy (~12 GB disk) | C me | H / M |
| 7 | Reuse Settings bugs: an upscaled image sets size to 2048² (→ OOM); LoRA, LoRA strength and count aren't restored; switching model keeps an incompatible LoRA | C O | M / S |
| 8 | Queue robustness: retry instead of dropping jobs when the engine is briefly busy; keep finished batch images if a later one fails; verify saves (mflux swallows save errors ✔) | C O | M-H / S |
| 9 | Privacy: PNGs embed the prompt and absolute local paths by default ✔; localhost API has no token/Host check | C O G | M / S |
| 10 | Reorder reference images (image 1 matters) and remember settings between launches | O G C | M / S |

## 1. Bugs / correctness

- "Load" button and load timings: weights load lazily on first use (`_load` needs `mx.eval`) ✔ (C, H/S)
- ⌘N opens a new window whose boot restarts the backend and resets history ✔ (O, H/S)
- Image save errors are swallowed by mflux; the app reports a result that may not exist ✔ (O, H/S)
- No bounds on width/height/steps/count/guidance/upscale scale (O C, H/S)
- Reuse Settings on an upscaled image restores 2048×2048 (C, M/S)
- Reuse Settings omits LoRA path/strength and count; model switch keeps the old LoRA (O C, M/S)
- Queued jobs fail permanently on a 409 "busy" (Model Manager load, cleanup, idle unload) (C, M/S)
- Batch: if image 3 fails, images 1–2 are saved but never shown; results only appear at the end (O C, H/M)
- Output names `HHMMSS_seed.png` saved with overwrite: same seed in the same second overwrites (O C, L/S)
- Clear History trashes every subfolder of the output folder; the count shown is capped at 200 (O C, M/S)
- Trashing a result doesn't check queued jobs that use it as input (O, M/M)
- Cancel during reference preparation is lost; load/conversion/decode can't be cancelled (O C, M/M)
- "Start from image 1" at Change 100% silently becomes text-to-image (klein, Z-Image, Krea) (C, L/S)
- History after relaunch: seed parsed from the filename (upscales show seed 0 ✔), time shows 0 (O C, L/S)
- ⌘↩ with an empty prompt sends a request and shows a backend error (C, L/S)
- Download cancel: a second `killpg` after ProcessLookupError can raise (C, L/S)
- Sizes silently rounded to the model's multiple (32 for Qwen); presets not snapped (O C, L/S)
- If a preview fails, the backend turns previews off but Settings still shows them on (G, L/S)
- Settings files store requested, not effective, settings; no model revision/LoRA provenance (O, M/M)
- Re-adding a custom model while busy keeps serving the old weights (O, L/S)
- Time-estimate fallback uses an arbitrary run type; Qwen start-from-image omits reference cost (O C, L/S)
- Negative-prompt hint is wrong for Krea (it applies at any guidance ≠ 1) (O, L/S)
- Downloads: a busy repo blocks another model's missing files; revision not pinned; file sizes not verified (O, M/M)

## 2. Performance & memory

- Free text encoders after encoding, at least when memory pressure is high (C, H/M) ✔ mflux does this in its CLI
- Pre-quantized saved weights for faster loads, and less disk (C me, H/M)
- Memory-pressure guard: compare the model's memory need with free memory and swap; offer 4-bit or unload (all, H/M)
- Qwen 40-step "Fast" mode using mflux's step cache (`use_step_cache`), quality to be checked (O C, M-H/S)
- Planned benchmarks: Qwen VAE decode in bf16, compile Qwen's step (the other models are already compiled), 4-bit vs 8-bit (me, H/M)
- `/models` polling scans the whole HF cache, the 11 GB converted folder and manifests every 1–10 s; cache it (O C, M/S)
- Network calls (HF manifests) inside status polling; move them off the polling path with timeouts (O, H/M)
- Reuse prompt/reference encodings across images in a batch (O, H/M)
- Qwen reference images are VAE-encoded without tiling (memory spike with big inputs) (O, H/M)
- Qwen's prompt cache grows without limit in long sessions (O, M/S)
- Release the last latents held by the progress callback after each job (O, L/S)
- Upscaling unloads the image model; the next generation pays a reload (C, L-M/M)
- Full-size image decoding and history scanning on the main thread; no thumbnail cache (O C G, M/S-M)
- Make previews back off when memory pressure rises (O, M/M)

## 3. UX / UI

- Reorder reference images; keep drop order for multi-image drops (O G C, M/S-M)
- Remember size/steps/seed/count/precision/negative prompt/LoRA across launches, with Reset to Defaults (O C, M/S)
- Show each batch image as soon as it's done (C, M/M)
- Settings inspector on a result (prompt + all settings, copy) (G, M/S)
- Keyboard: arrow keys through history, ⌘⌫ to trash, ⌘V to paste a reference, Quick Look (O G C, M/S)
- Show download/load errors next to the button that caused them; readable 409/422 messages (O C, M/S)
- Progress: bar hits 100% before decoding; the status bar shows "Idle" during upscaling (O, M/S)
- Compare: scroll or page large sets; keep live preview visible while a comparison runs; select-all helpers (O C G, M/M)
- Deleting an "Uncensored" variant deletes the shared 33 GB Qwen base; offer "remove only the LoRA" (C, M/S)
- Model Manager: per-row busy state (one global spinner today) (O, M/M)
- Show swap activity (rate), not just total swap (G, M/S)
- Clarify "Match image 1" (same shape at about 1 MP) and mark extra images a model ignores (O, M/S)

## 4. New features

- Masked / region editing for Qwen (`mask_image`, text `auto_mask` already in mflux) (O C, H/L)
- Before/after wipe comparison with synced zoom (great for upscales and edits) (O, M/M)
- Searchable image library: filters (prompt/model/date), favorites, beyond 200 images; prompt history (O G, M/L)
- Persistent queue with pause, reorder, retry and failed-job records (O, H/L)
- Variations and parameter sweeps (seeds, strength, steps, guidance) (O C, M/M)
- Upscale options (scale, softness) and upscale any file, not only results (O G C, M/S)
- Export as JPEG/WebP, Save As, strip metadata; import settings from a PNG (O G C, M/S)
- Several LoRAs at once and a LoRA library linked to compatible models (O C, M/M)
- (Speculative) Qwen "enhance prompt" / "verify" options (C, L-M/S)

## 5. Reliability / error handling

- Watchdog for a hung engine and per-job deadlines (2 h URL timeout today) (O C, H/M)
- Wait for the old engine to exit before starting a new one (overlapping engines double memory) (O, H/M)
- Atomic image + settings-file writes (O G, H/M)
- Download retries with backoff; verify revision and file sizes (G O, M/M)
- Setup: `terminationHandler` set after `run()` can hang; check dependency health, not just a marker (O C, M/S)
- Disk-space checks for downloads, conversion and outputs (O, H/M)
- Port race at engine start (`freePort` closes the socket before uvicorn binds) (G C, L/S)
- Small leaks: download log file handles never closed, `sips` without timeout, conversion not cancellable (C, L/S)
- Show "engine unreachable" instead of stale status; surface decode errors (empty model list today) (O C, M/S)
- Validate saved JSON files and local checkpoints (O, M/M)

## 6. Code quality / tests

- Unit tests: timings, GGUF name mapping, catalog sanity, storage protection, queue and restart logic (all, H/M)
- Smoke test for fragile mflux internals (klein edit weight-sharing trick, private `PathResolution`, latent unpack for previews) (O C, M/S)
- One shared request-normalization path for `/estimate` and `/generate` (O C, M/M)
- Split `Views.swift` (~1000 lines) and `server.py`; stop reaching into private download internals; remove dead fields; FastAPI lifespan instead of `on_event` (all, M/M)
- Thread-safety ownership (`Backend` is `@unchecked Sendable`; status snapshots); assert MLX calls stay on the MLX thread (O C, M/M)
- Remove the zoom view's notification observer; give SystemMonitor a stop (O, L/S)

## 7. Packaging / distribution / security

- Pin mflux and dependencies with a lock file; install uv without `curl | sh` (all, H/S)
- Stop embedding prompt and absolute file paths in PNG metadata by default, make it opt-in ✔ (C O, M/S)
- Local API: per-launch token and Host-header check (cross-site requests, DNS rebinding, arbitrary file paths) (all, M/S)
- Keychain save deletes the old token before adding the new one (a failure loses it) ✔; use update (O, H/S)
- Signing key and its password sit in `~/Downloads/ImageStudio/.signing (old location)`: move to `~/Library`; fix `eval` quoting (O C, M/S)
- Explicit HF token handling (an inherited `HF_TOKEN` env var is used silently) (O G, M/M)
- Only if sharing the app: hardened runtime, notarization, entitlements, versioning (all, M/L)
- README drift, privacy wording ("nothing leaves the machine"), custom-model license shown as the base's (O C, M/S)
- Strict offline mode; `/system` builds the cache path itself instead of using `HF_HUB_CACHE` (O, M/M)

## Checked and rejected (the code already handles these)

- "Storage cleanup reads download jobs without a lock" (G): it runs inside `with downloads._lock` (server.py:745).
- "No watchdog for an orphaned engine" (G): `_exit_when_parent_dies` exists (server.py:836).
- "Embed settings in the PNG" (G): mflux already embeds them (which is the privacy item above).
- "Model steps aren't compiled" (G): klein, Z-Image and Krea use `mx.compile`; only Qwen doesn't (planned benchmark).
- "No custom sizes" (G): width/height fields already exist.
- "Offline marks downloaded models unusable" (G): it falls back to mflux's snapshot check; only the byte count shows 0.
