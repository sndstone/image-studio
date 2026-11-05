# Changelog

Versioning: `x.y.0` for feature releases, `x.y.z` (z ≥ 1) for minor fixes.

## 0.4.0

- A notification when generating finishes while you're in another app: one for a whole queue, saying how many images
  are ready and whether anything failed. Clicking it opens the last image. Settings → General turns it off.
- ⌘-scroll and pinch no longer change a result's zoom while another app is in front. macOS sends scrolling to the
  window under the pointer, so a gesture meant for another app could leave an image at a stray zoom such as 6% or 221%.

Not yet tested against the live APIs: the GPT Image GIF, SVG and markup options (GIF output also needs OpenAI's GIF
preview access) and reference-image edits on the cloud models. Plain text-to-image was tested live on all three providers.

## 0.3.0

- Cloud models in the model picker and Model Manager, next to the local ones: Google Nano Banana 2.1, 2, Pro and
  2 Lite; OpenAI GPT Image 2.6, 2.5 and 2; Black Forest Labs FLUX.3 Image, FLUX.3 klein and FLUX.2 max, pro, flex and
  klein. Text-to-image and editing with reference images; results land in the same history, with sidecars.
- Settings → Cloud: API keys per provider, stored in the Keychain. **Save & Test** checks a key with a free call and
  reports unlisted models (BFL: the credit balance). **Test Image (billed)** makes one small image to catch missing
  billing, quota or verification.
- GPT Image options in the Generation section when an OpenAI model is selected, locked with the reason when a model
  lacks one: animated GIF and SVG output and complexity (2.6), Extra high and Max quality (2.5, 2.6), and a markup
  editor for sketching on image 1 (2.5, 2.6). GIFs play in the viewer, and history tiles mark GIFs and SVGs.
- Compare (⇧⌘C) handles cloud and local models together: cloud models run at the same time, then local models take
  turns, staying loaded when the next one fits in free memory and unloading first when it doesn't. The window shows
  how the run will go and what it should cost; a model that fails can be retried without re-billing the others.
- Costs (⇧⌘K): a dashboard of spending by day, provider and model with the BFL balance, a list of each image's cost,
  and current prices. Each image shows its cost, and the generate bar shows what the next image should cost. Costs
  covers the images still in the library: trashing an image removes its entry, favorite and cached reference copies.
- Redesigned Settings: grouped sections with the explanations below them, so text no longer runs out of the window.
- Reference images are re-encoded before upload, so their EXIF (location, camera) stays on this Mac. Offline mode
  also blocks cloud models. The controls local models need (precision, LoRAs, fixed steps) hide for cloud models.

## 0.2.1

- Fixed the window layout breaking while generating: the memory warning in the generate bar made the window's
  content taller than the window, which hid the prompt, Cancel, the preview and the history.
- Failed jobs in the queue no longer push the generate bar off screen: their errors are cut to three lines
  (full text in the tooltip) and the list scrolls in a fixed height.
- The history grid never grows taller than the window allows; the result keeps its space in small windows.
- The time left while generating is estimated from the running job's own settings, so queued jobs with a
  different model, size or image count show the right countdown.
- The "Generated images will appear here" message no longer squeezes into one tile in the grid view.

## 0.2.0

- Reworked sidebar: a native sidebar with the model picker and Manage in the window toolbar (no more text
  showing through the title bar), the prompt pinned above the scrolling controls with a Tips popover, a
  "Reset to defaults" link in the Generation header, Steps as a slider with a number field, and the
  Hugging Face API key moved to Settings. Warnings use one note style with a "Why?" popover for the long
  explanation, and the memory advice is shown once. The window subtitle shows the model or the progress.
- History panel: switch between one row and a resizable grid, search behind a magnifier (⌘F), favorites
  filter and clear-history as icons; thumbnails have VoiceOver labels. ⌘L focuses the prompt, ⌘[ and ⌘]
  step through images. The queue box sizes to its rows instead of reserving empty space.
- Polished image actions with a clearer Edit Image button, consistent tool buttons, and secondary actions
  that move into a menu in narrower windows. Improved empty states, size controls, and reference labels.
- Fixed segmented option highlights and cramped image-mode controls in the sidebar.
- Download status polling tolerates files being renamed or removed; storage scans do less filesystem work
  and cleanup preserves custom models stored inside input-image caches.
- Exporting with metadata removed preserves all EXIF rotations and mirroring by orienting the pixels first.
- FLUX.3 Image klein (Black Forest Labs, Oct 2026): text-to-image and editing with up to 10 reference images.
  INT4 (5.3 GB), INT8 (9.9 GB) and BF16 (18.5 GB) share one Model Manager card: download just the precision
  you need, or all of them with Download All. Sizes of models not downloaded yet are looked up when the
  Model Manager opens. The BF16 version can also run 8-bit or 4-bit quantized at load.
- FLUX.3 can start from image 1 (img2img) with the Change slider, which mflux only offers for FLUX.2.
  Change defaults to 70%: the result keeps image 1's pose and framing while the prompt restyles the scene.
- Models with several versions (FLUX.2 klein 4B / Base 4B / 9B, FLUX.3 klein INT4 / INT8 / BF16, Qwen-Image 2.1
  Turbo / Full / Uncensored Turbo / Uncensored) each have one Model Manager card and one submenu in the model
  picker. Download one version, or all of them with Download All; files the versions share (the Qwen-Image 2.1
  base model) are downloaded and counted once.
- FLUX.3 runs its VAE convolutions in strips: the same images (54 dB PSNR against the normal decode), about
  4 GB less memory for the final decode and 1.5 GB less for each reference image at 1024².
- FLUX.3 low-memory mode now really frees the 3.9 GB text encoder: prompt encodings are evaluated when made,
  so they no longer keep the encoder's weights alive.
- FLUX.3 edit guidance: the prompt hint and image-mode text explain that FLUX.3 needs a full description of the
  result, with a warning for short edit prompts (a prompt that only describes the photo returns it unchanged).
- Model sizes stay visible after the mflux 0.23 revision pinning (fallback to the size list cached before the pin).
- mflux 0.23.0 (adds FLUX.3); the Python runtime is reinstalled once on first launch.

## 0.1.0

First public release.

- Text-to-image and multi-reference editing with FLUX.2 klein (4B, Base 4B, 9B), Qwen-Image 2.1 (Turbo, full,
  uncensored LoRA), Z-Image Turbo (GGUF) and Krea 2 Turbo through mflux/MLX; SeedVR2 2× upscaling.
- Model Manager: downloads, load/unload, pre-quantized optimized copies, custom checkpoints, Clean Up Storage.
- Live preview, queue with pause/retry, Compare Models, masked editing (Qwen), LoRAs, variations and sweeps,
  history search and favorites, before/after wipe, export to PNG/JPEG/HEIC/WebP, offline mode.
- Memory guard before each job, low-memory mode, idle unload, learned time estimates.
- Low idle cost: about 0.2 s CPU per idle minute with a 300-image history; memory statistics are read in-process.
