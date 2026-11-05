"""Cloud image models: Google (Nano Banana), OpenAI (GPT Image) and Black Forest Labs (FLUX API).

Keys are given to the running server by the app (which keeps them in the Keychain) and live only in this
module's memory: never in the environment (child processes would inherit it), logs, sidecars or error text.
"""

from __future__ import annotations

import base64
import io
import math
import threading
import time
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urlparse

import httpx

import costs

GOOGLE_BASE = "https://generativelanguage.googleapis.com/v1beta"
OPENAI_BASE = "https://api.openai.com/v1"
BFL_BASE = "https://api.bfl.ai/v1"

PROVIDERS = {"google": "Google", "openai": "OpenAI", "bfl": "Black Forest Labs"}
OPENAI_COMPLEXITIES = ("low", "medium", "high")

# Gemini takes a named shape and size instead of pixels. The Pro and Lite tables have no extreme ratios.
GEMINI_RATIOS = ("1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9")
GEMINI_WIDE_RATIOS = GEMINI_RATIOS + ("1:4", "4:1", "1:8", "8:1")
GEMINI_WIDE_MODELS = {"gemini-nano-banana-2.1", "gemini-3.1-flash-image"}
GEMINI_NO_SIZE_MODELS = {"gemini-3.1-flash-lite-image"}  # no size table is published for Lite

OPENAI_MIN_PIXELS = 655_360
BFL_WAITING = {"Pending", "Reasoning", "Generating", "Queued", "Processing"}
POLL_TIMEOUT = 600.0
TIMEOUT = httpx.Timeout(300.0, connect=20.0, write=120.0)
VERIFY_TIMEOUT = httpx.Timeout(20.0)

# The cheapest way to prove a key can really generate (billed); see test_image.
TEST_PROMPT = "A small red apple on a plain white table, studio photo"
TEST_MODELS = {"google": "gemini-3.1-flash-lite-image", "openai": "gpt-image-2.5-flare", "bfl": "flux-2-klein-4b"}

_keys: dict[str, str] = {}
_keys_lock = threading.Lock()
_transport: httpx.BaseTransport | None = None  # replaced in tests; no test reaches the network


class CloudError(Exception):
    """A failure worth showing to the user as it is."""


class Cancelled(Exception):
    pass


@dataclass
class Job:
    model: str
    prompt: str
    width: int
    height: int
    image_paths: list[str]
    seed: int | None = None
    steps: int | None = None
    guidance: float | None = None
    # OpenAI
    quality: str | None = None
    complexity: str | None = None
    output_format: str = "png"
    frame_count: int | None = None
    frame_duration_ms: int | None = None
    gif_auto_size: bool = True
    markup_path: str | None = None


def set_key(provider: str, key: str | None) -> None:
    key = (key or "").strip()
    with _keys_lock:
        if key:
            _keys[provider] = key
        else:
            _keys.pop(provider, None)


def has_key(provider: str) -> bool:
    with _keys_lock:
        return provider in _keys


def _key(provider: str) -> str:
    with _keys_lock:
        key = _keys.get(provider)
    if not key:
        raise CloudError(f"No {PROVIDERS[provider]} API key. Add one in Settings → Cloud.")
    return key


def _client(timeout, follow_redirects=False) -> httpx.Client:
    # API calls carry a key, so they never follow a redirect to another host.
    return httpx.Client(timeout=timeout, follow_redirects=follow_redirects, transport=_transport)


def _redact(text: str, key: str | None = None) -> str:
    with _keys_lock:
        secrets = [*_keys.values(), *([key] if key else [])]
    for secret in secrets:
        text = text.replace(secret, "••••")
    return text


def _interruptible(work: Callable[[], object], should_stop: Callable[[], bool] | None):
    """Runs a blocking request so Cancel works at once. The provider may still finish (and bill) the request."""
    if should_stop is None:
        return work()
    box: dict[str, object] = {}
    done = threading.Event()

    def run():
        try:
            box["value"] = work()
        except BaseException as error:
            box["error"] = error
        finally:
            done.set()

    threading.Thread(target=run, daemon=True).start()
    while not done.wait(0.1):
        if should_stop():
            raise Cancelled()
    if "error" in box:
        raise box["error"]  # type: ignore[misc]
    return box["value"]


def _detail(response: httpx.Response) -> str:
    """The provider's own explanation, from whichever shape its error body has."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:300].strip()
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        if isinstance(error, str):
            return error
        detail = body.get("detail")
        if isinstance(detail, str):
            return detail
        if isinstance(detail, list):  # FastAPI-style validation errors (BFL)
            return "; ".join(str(item.get("msg", item)) if isinstance(item, dict) else str(item) for item in detail)
        if body.get("message"):
            return str(body["message"])
    return str(body)[:300]


def _rejected_key(provider: str, response: httpx.Response) -> bool:
    if response.status_code in (401, 403):
        return True
    # Google answers 400 API_KEY_INVALID for a bad key.
    return provider == "google" and response.status_code == 400 and "API_KEY_INVALID" in response.text


def _raise_for_error(provider: str, response: httpx.Response, key: str) -> None:
    if response.is_success:
        return
    name, detail = PROVIDERS[provider], _redact(_detail(response), key)
    code = response.status_code
    if code == 402:
        raise CloudError(f"{name}: not enough credits on this account. {detail}".strip())
    if code == 429:
        raise CloudError(f"{name}: rate limit or quota reached. {detail}".strip())
    if _rejected_key(provider, response):
        raise CloudError(f"{name} refused the request ({code}). Check the key and your account's access. {detail}".strip())
    raise CloudError(f"{name} error {code}: {detail}".strip())


def _transport_error(provider: str, error: httpx.HTTPError, key: str | None = None) -> CloudError:
    kind = "timed out" if isinstance(error, httpx.TimeoutException) else "could not be reached"
    return CloudError(f"{PROVIDERS[provider]} {kind}. {_redact(str(error), key)}".strip())


def _post_json(provider, url, headers, payload, key, should_stop) -> dict:
    # Encoding references takes a while, and a request sent after Cancel would still be billed.
    if should_stop is not None and should_stop():
        raise Cancelled()

    def work():
        with _client(TIMEOUT) as client:
            return client.post(url, headers=headers, json=payload)

    try:
        response = _interruptible(work, should_stop)
    except httpx.HTTPError as error:
        raise _transport_error(provider, error, key)
    _raise_for_error(provider, response, key)
    try:
        return response.json()
    except ValueError:
        raise CloudError(f"{PROVIDERS[provider]} sent a response that isn't JSON.")


def encode_reference(path: str) -> bytes:
    """A JPEG of the input. Re-encoding also keeps its EXIF (location, camera) from leaving this Mac."""
    from PIL import Image

    with Image.open(path) as image:
        if "A" in image.getbands():
            rgba = image.convert("RGBA")
            flat = Image.new("RGB", rgba.size, "white")
            flat.paste(rgba, mask=rgba.getchannel("A"))
        else:
            flat = image.convert("RGB")
    out = io.BytesIO()
    flat.save(out, format="JPEG", quality=95)
    return out.getvalue()


def encode_markup(path: str, first_image: str) -> bytes:
    """The annotation layer as a transparent PNG the size of the first image as it's sent: OpenAI requires them
    to match, and large inputs are scaled down before sending."""
    from PIL import Image

    with Image.open(path) as image:
        layer = image.convert("RGBA")
    with Image.open(first_image) as first:
        size = first.size
    if layer.size != size:
        layer = layer.resize(size, Image.Resampling.LANCZOS)
    if layer.getchannel("A").getextrema()[0] == 255:
        raise CloudError("The markup layer has no transparent pixels. Draw it again in the markup editor.")
    out = io.BytesIO()
    layer.save(out, format="PNG")
    return out.getvalue()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


# --- Google --------------------------------------------------------------------------------------


def gemini_aspect(model: str, width: int, height: int) -> str:
    ratios = GEMINI_WIDE_RATIOS if model in GEMINI_WIDE_MODELS else GEMINI_RATIOS

    def distance(ratio: str) -> float:
        w, h = ratio.split(":")
        return abs(math.log((int(w) / int(h)) / (width / height)))

    return min(ratios, key=distance)


def _google(job: Job, should_stop) -> tuple[bytes, dict]:
    key = _key("google")
    parts: list[dict] = [{"text": job.prompt}]
    parts += [{"inlineData": {"mimeType": "image/jpeg", "data": _b64(encode_reference(p))}} for p in job.image_paths]
    image_config = {"aspectRatio": gemini_aspect(job.model, job.width, job.height)}
    if job.model not in GEMINI_NO_SIZE_MODELS:
        image_config["imageSize"] = "2K" if max(job.width, job.height) > 1400 else "1K"
    body = _post_json("google", f"{GOOGLE_BASE}/models/{job.model}:generateContent", {"x-goog-api-key": key},
                      {"contents": [{"role": "user", "parts": parts}],
                       "generationConfig": {"responseModalities": ["TEXT", "IMAGE"], "imageConfig": image_config}},
                      key, should_stop)
    candidates = body.get("candidates") or []
    found = None
    text = []
    for part in (candidates[0].get("content") or {}).get("parts") or [] if candidates else []:
        inline = part.get("inlineData") or part.get("inline_data")
        if part.get("thought"):  # interim images while the model thinks
            continue
        if inline and inline.get("data"):
            found = inline["data"]  # the last image is the finished one
        elif part.get("text"):
            text.append(part["text"])
    if found is None:
        block = (body.get("promptFeedback") or {}).get("blockReason")
        finish = candidates[0].get("finishReason") if candidates else None
        reason = f"Blocked: {block}." if block else f"Finished with {finish}." if finish else ""
        raise CloudError(" ".join(f"Google returned no image. {reason} {' '.join(text)[:300]}".split()))
    cost, basis = costs.google_cost(job.model, body.get("usageMetadata"), image_config.get("imageSize"))
    return base64.b64decode(found), {"aspect_ratio": image_config["aspectRatio"], "image_size": image_config.get("imageSize"),
                                     "cost_usd": cost, "cost_basis": basis}


# --- OpenAI --------------------------------------------------------------------------------------


def openai_size(width: int, height: int) -> tuple[int, int]:
    """Nearest size GPT Image accepts: multiples of 16, shape between 1:3 and 3:1, at least 655,360 pixels."""
    if width > 3 * height:
        width = 3 * height
    elif height > 3 * width:
        height = 3 * width
    scale = max(1.0, math.sqrt(OPENAI_MIN_PIXELS / (width * height)))
    return tuple(math.ceil(side * scale / 16) * 16 for side in (width, height))  # type: ignore[return-value]


def _openai(job: Job, should_stop) -> tuple[bytes, dict]:
    key = _key("openai")
    payload: dict = {"model": job.model, "prompt": job.prompt, "n": 1, "output_format": job.output_format}
    if job.output_format == "png":
        width, height = openai_size(job.width, job.height)
        payload["size"] = f"{width}x{height}"
    elif job.output_format == "gif":
        # The frames share a pixel budget OpenAI doesn't publish, so "auto" is the size that can't exceed it.
        if not job.gif_auto_size:
            payload["size"] = f"{math.ceil(job.width / 16) * 16}x{math.ceil(job.height / 16) * 16}"
        if job.frame_count is not None:
            payload["frame_count"] = job.frame_count
        if job.frame_duration_ms is not None:
            payload["frame_duration_ms"] = job.frame_duration_ms
    if job.quality and job.quality != "auto":
        payload["quality"] = job.quality
    if job.complexity:
        payload["complexity"] = job.complexity
    route = "generations"
    if job.image_paths:
        route = "edits"
        payload["images"] = [{"image_url": "data:image/jpeg;base64," + _b64(encode_reference(p))} for p in job.image_paths]
        if job.markup_path:
            payload["markup"] = {"image_url": "data:image/png;base64," + _b64(encode_markup(job.markup_path, job.image_paths[0]))}
    body = _post_json("openai", f"{OPENAI_BASE}/images/{route}", {"Authorization": f"Bearer {key}"}, payload, key, should_stop)
    data = (body.get("data") or [{}])[0].get("b64_json")
    if not data:
        raise CloudError("OpenAI returned no image.")
    # An SVG also bills an authoring model whose rate isn't published: a partial figure would understate it.
    cost, basis = (None, "unknown") if job.output_format == "svg" else costs.openai_cost(body.get("usage"))
    details = {"size": body.get("size") or payload.get("size"), "quality": body.get("quality") or job.quality,
               "cost_usd": cost, "cost_basis": basis}
    if body.get("svg_usage"):
        details["svg_usage"] = body["svg_usage"]
    return base64.b64decode(data), details


# --- Black Forest Labs ---------------------------------------------------------------------------


def _is_bfl_url(url: str) -> bool:
    parts = urlparse(url)
    host = parts.hostname or ""
    return parts.scheme == "https" and (host == "bfl.ai" or host.endswith(".bfl.ai"))


def _bfl_payload(job: Job) -> dict:
    payload: dict = {"prompt": job.prompt, "width": job.width, "height": job.height}
    images = [_b64(encode_reference(p)) for p in job.image_paths]
    if job.model.startswith("flux-3-"):
        # Unknown fields are a 422 on these endpoints: FLUX.3 Image has no seed or output_format.
        if images:
            payload["images"] = images
        if job.model == "flux-3-image-klein":
            payload["output_format"] = "png"
            if job.seed is not None:
                payload["seed"] = job.seed % 2**32
        return payload
    for index, image in enumerate(images):
        payload["input_image" if index == 0 else f"input_image_{index + 1}"] = image
    payload["output_format"] = "png"
    if job.seed is not None:
        payload["seed"] = job.seed
    if job.model == "flux-2-flex":
        if job.steps is not None:
            payload["steps"] = min(50, max(1, job.steps))
        if job.guidance is not None:
            payload["guidance"] = min(10.0, max(1.5, job.guidance))
    return payload


def _bfl(job: Job, should_stop) -> tuple[bytes, dict]:
    key = _key("bfl")
    headers = {"x-key": key}
    task = _post_json("bfl", f"{BFL_BASE}/{job.model}", headers, _bfl_payload(job), key, should_stop)
    polling_url = task.get("polling_url")
    # The key goes to this URL, so it has to be BFL's own whatever the response says.
    if not isinstance(polling_url, str) or not _is_bfl_url(polling_url):
        raise CloudError("Black Forest Labs returned an unexpected polling address.")
    cost, basis = costs.bfl_cost(task.get("cost"))
    meta = {"cost_credits": task.get("cost"), "task_id": task.get("id"), "cost_usd": cost, "cost_basis": basis}
    deadline = time.monotonic() + POLL_TIMEOUT
    failures = 0
    with _client(httpx.Timeout(30.0)) as client:
        while True:
            if should_stop is not None and should_stop():
                raise Cancelled()
            if time.monotonic() > deadline:
                raise CloudError("Black Forest Labs did not finish in time.")
            time.sleep(0.5)
            try:
                response = _interruptible(lambda: client.get(polling_url, headers=headers), should_stop)
            except httpx.HTTPError as error:
                failures += 1
                if failures >= 5:
                    raise _transport_error("bfl", error, key)
                continue
            if response.status_code >= 500 or response.status_code == 429:
                failures += 1
                if failures >= 5:
                    _raise_for_error("bfl", response, key)
                continue
            _raise_for_error("bfl", response, key)
            failures = 0
            result = response.json()
            status = result.get("status")
            if status == "Ready":
                sample = (result.get("result") or {}).get("sample")
                if not isinstance(sample, str):
                    raise CloudError("Black Forest Labs finished without an image.")
                break
            if status in ("Request Moderated", "Content Moderated"):
                raise CloudError(f"Black Forest Labs: {status}. {_redact(str(result.get('details') or ''), key)}".strip())
            if status not in BFL_WAITING:
                raise CloudError(f"Black Forest Labs: {status}. {_redact(str(result.get('details') or ''), key)}".strip())
    if urlparse(sample).scheme != "https":
        raise CloudError("Black Forest Labs returned an unexpected image address.")

    def download():
        # A signed URL: the key is not sent.
        with _client(TIMEOUT, follow_redirects=True) as delivery:
            return delivery.get(sample)

    try:
        image = _interruptible(download, should_stop)
    except httpx.HTTPError as error:
        raise _transport_error("bfl", error, key)
    _raise_for_error("bfl", image, key)
    return image.content, meta


_GENERATORS = {"google": _google, "openai": _openai, "bfl": _bfl}


def generate(provider: str, job: Job, should_stop=None) -> tuple[bytes, dict]:
    """Returns the image file's bytes (still in the provider's format) and details for the sidecar."""
    try:
        return _GENERATORS[provider](job, should_stop)
    except (KeyError, IndexError, TypeError, ValueError) as error:  # a response in a shape we don't know
        raise CloudError(f"{PROVIDERS[provider]} sent an unexpected response ({type(error).__name__}).") from None


def decode(data: bytes):
    """The decoded image, or a clear error if the provider's bytes aren't one."""
    from PIL import Image, UnidentifiedImageError

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise CloudError("The provider's response wasn't a readable image.") from None
    return image if image.mode in ("RGB", "RGBA") else image.convert("RGB")


def file_size(data: bytes, output_format: str) -> tuple[int, int] | None:
    """Checks a GIF or SVG before it's saved as it is, and returns its size (None if an SVG doesn't state one)."""
    if output_format == "gif":
        from PIL import Image, UnidentifiedImageError

        try:
            with Image.open(io.BytesIO(data)) as image:
                if image.format != "GIF":
                    raise CloudError("OpenAI's response wasn't a GIF.")
                return image.size
        except (UnidentifiedImageError, OSError):
            raise CloudError("OpenAI's response wasn't a readable GIF.") from None
    return _svg_size(data)


def _svg_size(data: bytes) -> tuple[int, int] | None:
    """OpenAI documents its SVGs as static. This file may later open in a browser, so that's checked, not assumed."""
    from xml.etree import ElementTree

    try:
        text = data.decode("utf-8")
        if "<!doctype" in text.lower() or "<!entity" in text.lower():
            raise CloudError("OpenAI's SVG contains a document type declaration, so it wasn't saved.")
        root = ElementTree.fromstring(text)
    except (UnicodeDecodeError, ElementTree.ParseError):
        raise CloudError("OpenAI's response wasn't a readable SVG.") from None
    if root.tag.rsplit("}", 1)[-1] != "svg":
        raise CloudError("OpenAI's response wasn't an SVG.")
    for element in root.iter():
        name = element.tag.rsplit("}", 1)[-1].lower()
        if name in ("script", "foreignobject") or any(a.rsplit("}", 1)[-1].lower().startswith("on") for a in element.attrib):
            raise CloudError("OpenAI's SVG contains active content, so it wasn't saved.")
    try:
        if root.get("viewBox"):
            parts = root.get("viewBox").replace(",", " ").split()
            return round(float(parts[2])), round(float(parts[3]))
        return round(float(root.get("width"))), round(float(root.get("height")))
    except (IndexError, TypeError, ValueError):  # e.g. width="100%"
        return None


# --- connection tests ----------------------------------------------------------------------------


_VERIFY_URLS = {"google": f"{GOOGLE_BASE}/models?pageSize=1000", "openai": f"{OPENAI_BASE}/models", "bfl": f"{BFL_BASE}/credits"}
_AUTH_HEADERS = {"google": "x-goog-api-key", "openai": "Authorization", "bfl": "x-key"}


def verify(provider: str, models: list[str]) -> dict:
    """Checks the saved key with a free, read-only call. `models` are the IDs the app offers."""
    try:
        key = _key(provider)
    except CloudError as error:
        return {"ok": False, "message": str(error)}
    name = PROVIDERS[provider]
    value = f"Bearer {key}" if provider == "openai" else key
    try:
        with _client(VERIFY_TIMEOUT) as client:
            response = client.get(_VERIFY_URLS[provider], headers={_AUTH_HEADERS[provider]: value})
    except httpx.HTTPError as error:
        return {"ok": False, "message": str(_transport_error(provider, error, key))}
    if _rejected_key(provider, response):
        return {"ok": False, "message": f"{name} rejected this key ({response.status_code}). Check that it was copied completely."}
    if not response.is_success:
        return {"ok": False, "message": f"{name} error {response.status_code}: {_redact(_detail(response), key)}"}
    try:
        body = response.json()
        if provider == "bfl":
            credits = float(body["credits"])
            return {"ok": True, "message": f"Key works. Balance: {credits:g} credits (1 credit = $0.01).", "credits": credits}
        listed = ({m["name"].removeprefix("models/") for m in body.get("models", [])} if provider == "google"
                  else {m["id"] for m in body.get("data", [])})
    except (KeyError, TypeError, ValueError):
        return {"ok": True, "message": "Key accepted, but the response had an unexpected shape."}
    missing = [m for m in models if m not in listed]
    if not missing:
        return {"ok": True, "message": f"Key works. All {len(models)} models are available."}
    return {"ok": True, "message": f"Key works. {len(models) - len(missing)} of {len(models)} models are listed for it; "
                                   f"not listed: {', '.join(missing)}."}


def test_image(provider: str) -> dict:
    """Generates one small image with the provider's cheapest model: this is billed, and it's the only test
    that catches missing billing, quota or (OpenAI) organization verification. Nothing is saved, so it isn't in
    the cost ledger either: the result reports what it cost."""
    model = TEST_MODELS[provider]
    job = Job(model=model, prompt=TEST_PROMPT, width=1024, height=1024, image_paths=[],
              quality="low" if provider == "openai" else None)
    started = time.time()
    try:
        data, meta = generate(provider, job)
        image = decode(data)
    except CloudError as error:
        return {"ok": False, "message": _redact(str(error))}
    seconds = time.time() - started
    size = image.size
    thumb = io.BytesIO()
    preview = image.convert("RGB")
    preview.thumbnail((128, 128))
    preview.save(thumb, format="JPEG", quality=80)
    message = f"{model} made a {size[0]}×{size[1]} image in {seconds:.1f} s."
    cost = meta.get("cost_usd")
    if cost is not None:
        message += f" It cost about ${cost:.4f}."
    return {"ok": True, "message": message, "thumbnail": _b64(thumb.getvalue())}
