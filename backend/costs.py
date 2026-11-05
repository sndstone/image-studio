"""What cloud images cost: published prices, the cost of each response, and a local ledger of the spending.

The ledger describes the images still in the library: an entry goes when its image file does, whether it was
trashed here or removed in Finder. So the totals are what the present images cost, not the account's spending:
only the provider's bill shows that.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

from catalog import APP_SUPPORT

LEDGER = APP_SUPPORT / "cloud-costs.jsonl"
CREDIT_USD = 0.01  # BFL: "1 credit = $0.01 USD"
PRICES_CHECKED = "2026-11-05"

# USD per 1M tokens and per image, from aistudio.google.com/docs/pricing (standard, paid tier).
GOOGLE_PRICES = {
    "gemini-nano-banana-2.1": {"input": 1.50, "text_output": 7.50, "image_output": 30.0, "per_image": {"1K": 0.0336, "2K": 0.0504}},
    "gemini-3.1-flash-image": {"input": 0.50, "text_output": 3.00, "image_output": 60.0, "per_image": {"1K": 0.067, "2K": 0.101}},
    "gemini-3-pro-image": {"input": 2.00, "text_output": 12.00, "image_output": 120.0, "per_image": {"1K": 0.134, "2K": 0.134}},
    "gemini-3.1-flash-lite-image": {"input": 0.25, "text_output": 1.50, "image_output": 30.0, "per_image": {"1K": 0.0336}},
}
# USD per 1M tokens, the same for every model offered, from developers.openai.com/api/docs/pricing. Text output isn't billed.
OPENAI_PRICES = {"text_input": 5.0, "image_input": 8.0, "image_output": 30.0}

_lock = threading.Lock()


def _per_million(tokens, rate) -> float:
    return (tokens or 0) * rate / 1_000_000


def google_cost(model: str, usage: dict | None, image_size: str | None) -> tuple[float | None, str]:
    """Tokens from the response at the published rates; failing that, the list price of one image."""
    prices = GOOGLE_PRICES.get(model)
    if prices is None:
        return None, "unknown"
    output = (usage or {}).get("candidatesTokenCount")
    details = (usage or {}).get("candidatesTokensDetails")
    if output is None or not details:  # without the split, image and text tokens can't be priced apart
        price = prices["per_image"].get(image_size or "1K")
        return (price, "list") if price is not None else (None, "unknown")
    image = sum(item.get("tokenCount", 0) for item in details if item.get("modality") == "IMAGE")
    text = max(0, output - image) + (usage.get("thoughtsTokenCount") or 0)
    return (_per_million(usage.get("promptTokenCount"), prices["input"]) + _per_million(text, prices["text_output"])
            + _per_million(image, prices["image_output"]), "computed")


def openai_cost(usage: dict | None) -> tuple[float | None, str]:
    """The response's tokens at the published rates. OpenAI publishes no per-image price for these models."""
    if not usage or usage.get("output_tokens") is None:
        return None, "unknown"
    details = usage.get("input_tokens_details") or {}
    text = details.get("text_tokens", usage.get("input_tokens", 0) if not details else 0)
    output = (usage.get("output_tokens_details") or {}).get("image_tokens", usage["output_tokens"])
    return (_per_million(text, OPENAI_PRICES["text_input"])
            + _per_million(details.get("image_tokens"), OPENAI_PRICES["image_input"])
            + _per_million(output, OPENAI_PRICES["image_output"]), "computed")


def bfl_cost(credits) -> tuple[float | None, str]:
    """What BFL's response says the request costs. BFL doesn't publish prices in a form an app can read."""
    return (credits * CREDIT_USD, "reported") if isinstance(credits, (int, float)) else (None, "unknown")


def _image_exists(entry: dict) -> bool:
    path = entry.get("path")
    return isinstance(path, str) and Path(path).is_file()


def record(**entry) -> None:
    line = json.dumps({"time": time.time(), **entry})
    with _lock:
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with LEDGER.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def entries() -> list[dict]:
    """Oldest first, for the images that still exist; the file is rewritten without the others. A line damaged
    by a crash mid-write is skipped rather than losing the rest."""
    with _lock:
        try:
            lines = LEDGER.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []
        result = []
        for line in lines:
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if isinstance(entry, dict) and isinstance(entry.get("time"), (int, float)) and _image_exists(entry):
                result.append(entry)
        if len(result) != len(lines):
            _rewrite(result)
    return result


def _rewrite(ledger: list[dict]) -> None:
    temp = LEDGER.with_name(LEDGER.name + ".tmp")
    try:
        temp.write_text("".join(json.dumps(entry) + "\n" for entry in ledger), encoding="utf-8")
        os.replace(temp, LEDGER)
    except OSError:  # a full disk: the summary is still right, and the next read tries again
        temp.unlink(missing_ok=True)


def estimate(provider: str, model: str, width: int, height: int, quality: str | None = None,
             ledger: list[dict] | None = None, **options) -> tuple[float | None, str]:
    """The cost of one image before running it: the list price, else what recent similar images cost here.
    `options` are other ledger fields that change the price: a 16-frame GIF says nothing about a still."""
    if provider == "google":
        prices = GOOGLE_PRICES.get(model, {}).get("per_image", {})
        price = prices.get("2K" if max(width, height) > 1400 else "1K", prices.get("1K"))
        if price is not None:
            return price, "list"
    megapixels = width * height / 1e6
    similar = [e["cost_usd"] for e in (entries() if ledger is None else ledger)
               if e.get("model") == model and isinstance(e.get("cost_usd"), (int, float))
               and e.get("quality") == quality and e.get("megapixels")
               and all(e.get(name) == value for name, value in options.items())
               and 0.8 <= e["megapixels"] / megapixels <= 1.25]
    return (sum(similar[-5:]) / len(similar[-5:]), "history") if similar else (None, "none")


def price_list() -> list[dict]:
    rows = []
    for model, prices in GOOGLE_PRICES.items():
        per_image = " · ".join(f"${price:g} at {size}" for size, price in prices["per_image"].items())
        rows.append({"provider": "google", "model": model, "price": per_image})
    rows.append({"provider": "openai", "model": "All GPT Image models",
                 "price": f"${OPENAI_PRICES['image_output']:g} per 1M image output tokens · "
                          f"${OPENAI_PRICES['image_input']:g} image input · ${OPENAI_PRICES['text_input']:g} text input"})
    rows.append({"provider": "bfl", "model": "All FLUX API models",
                 "price": "Varies by model and size; each job reports its cost (1 credit = $0.01)"})
    return rows


def summary(now: datetime | None = None, days: int = 30, recent: int = 300) -> dict:
    now = now or datetime.now()
    ledger = entries()
    today = now.date()
    day_keys = [(today - timedelta(days=offset)).isoformat() for offset in range(days - 1, -1, -1)]
    totals = {"today": 0.0, "month": 0.0, "all": 0.0}
    by_provider: dict[str, float] = {}
    by_model: dict[tuple[str, str], dict] = {}
    by_day = {day: {} for day in day_keys}
    unknown = 0
    for entry in ledger:
        when = datetime.fromtimestamp(entry["time"])
        provider, model = entry.get("provider", ""), entry.get("model", "")
        row = by_model.setdefault((provider, model), {"provider": provider, "model": model, "images": 0, "cost_usd": 0.0})
        row["images"] += 1
        cost = entry.get("cost_usd")
        if not isinstance(cost, (int, float)):
            unknown += 1
            continue
        row["cost_usd"] += cost
        totals["all"] += cost
        by_provider[provider] = by_provider.get(provider, 0.0) + cost
        if (when.year, when.month) == (now.year, now.month):
            totals["month"] += cost
        if when.date() == today:
            totals["today"] += cost
        day = when.date().isoformat()
        if day in by_day:
            by_day[day][provider] = by_day[day].get(provider, 0.0) + cost
    return {
        **{f"{name}_usd": value for name, value in totals.items()},
        "images": len(ledger),
        "unknown_cost_images": unknown,
        "by_provider": by_provider,
        "by_model": sorted(by_model.values(), key=lambda row: -row["cost_usd"]),
        "by_day": [{"day": day, "provider": provider, "cost_usd": cost}
                   for day in day_keys for provider, cost in sorted(by_day[day].items())],
        "days": day_keys,
        "entries": ledger[-recent:][::-1],
        "prices": price_list(),
        "prices_checked": PRICES_CHECKED,
    }
