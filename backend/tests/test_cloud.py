"""Cloud provider tests. All HTTP goes to a fake transport; nothing reaches the network."""

import base64
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
from PIL import Image

import test_core  # noqa: F401  (replaces MLX before the server is imported)
import catalog
import cloud
import costs
import server


def png_bytes(size=(32, 32), color="red"):
    out = io.BytesIO()
    Image.new("RGB", size, color).save(out, format="PNG")
    return out.getvalue()


class CloudCase(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.handler = None
        transport = httpx.MockTransport(self._handle)
        self.patches = [patch.object(cloud, "_transport", transport), patch.object(server, "OUTPUT_ROOT", self.root),
                        patch.object(costs, "LEDGER", self.root / "ledger.jsonl"),
                        patch.object(cloud.time, "sleep", lambda _: None)]
        for p in self.patches:
            p.start()
        cloud._keys.clear()
        self.addCleanup(cloud._keys.clear)
        server.state.offline = False
        server.state.cancel.clear()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.temp.cleanup()

    def _handle(self, request):
        self.requests.append(request)
        return self.handler(request)

    def body(self, index=0):
        return json.loads(self.requests[index].content)

    def run_model(self, model_id, **fields):
        request = server.GenerateRequest(prompt="a fox", model=model_id, **fields)
        return server.generate(request)

    def reference(self, name="ref.png"):
        path = self.root / name
        Image.new("RGB", (64, 64), "blue").save(path)
        return str(path)


class CatalogTests(unittest.TestCase):
    def test_every_provider_has_models_with_unique_ids(self):
        ids = [m.id for m in catalog.CLOUD_MODELS]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual({m.cloud_provider for m in catalog.CLOUD_MODELS}, set(cloud.PROVIDERS))
        for provider, model in cloud.TEST_MODELS.items():
            self.assertTrue(any(m.cloud_model == model and m.cloud_provider == provider for m in catalog.CLOUD_MODELS))

    def test_cloud_models_are_reachable_but_not_in_the_local_catalog(self):
        self.assertEqual(catalog.get("cloud-gpt-image-2.6-flare").cloud_provider, "openai")
        self.assertFalse(any(m.cloud_provider for m in catalog.CATALOG))
        self.assertFalse(catalog.get("cloud-flux-2-pro").optimizable)

    def test_model_json_is_ready_only_with_a_key(self):
        spec = catalog.get("cloud-flux-2-pro")
        cloud._keys.clear()
        self.assertFalse(server._model_json(spec, {})["status"]["downloaded"])
        cloud.set_key("bfl", "k")
        try:
            data = server._model_json(spec, {})
            self.assertTrue(data["status"]["downloaded"])
            self.assertEqual((data["disk_bytes"], data["memory_gb"], data["double_pass_when"]), (0, {}, "never"))
        finally:
            cloud._keys.clear()

    def test_flex_guidance_does_not_claim_a_second_local_pass(self):
        spec = catalog.get("cloud-flux-2-flex")
        self.assertEqual(spec.passes_per_step(5.0), 1)
        self.assertEqual(spec.to_json()["double_pass_when"], "never")


class SizeTests(unittest.TestCase):
    def test_openai_sizes_meet_the_documented_rules(self):
        for width, height in [(256, 256), (1024, 1024), (2048, 256), (256, 2048), (1920, 1080), (1000, 1336)]:
            w, h = cloud.openai_size(width, height)
            self.assertEqual((w % 16, h % 16), (0, 0))
            self.assertGreaterEqual(w * h, cloud.OPENAI_MIN_PIXELS)
            self.assertLessEqual(max(w, h) / min(w, h), 3.05)

    def test_gemini_aspect_uses_only_what_the_model_offers(self):
        self.assertEqual(cloud.gemini_aspect("gemini-3-pro-image", 1920, 1080), "16:9")
        self.assertEqual(cloud.gemini_aspect("gemini-nano-banana-2.1", 2048, 512), "4:1")
        self.assertEqual(cloud.gemini_aspect("gemini-3-pro-image", 2048, 512), "21:9")  # Pro has no 4:1
        self.assertEqual(cloud.gemini_aspect("gemini-3-pro-image", 1024, 1024), "1:1")


class GoogleTests(CloudCase):
    def test_text_to_image_request_and_saved_result(self):
        cloud.set_key("google", "g-key")
        data = base64.b64encode(png_bytes((48, 32))).decode()
        thought = base64.b64encode(png_bytes((8, 8), "green")).decode()
        self.handler = lambda r: httpx.Response(200, json={"candidates": [{"content": {"parts": [
            {"thought": True, "inlineData": {"mimeType": "image/png", "data": thought}},
            {"text": "here"}, {"inlineData": {"mimeType": "image/png", "data": data}}]}}]})
        result = self.run_model("cloud-gemini-3-pro-image", width=1920, height=1088)
        self.assertIsNone(result["error"])
        request = self.requests[0]
        self.assertTrue(str(request.url).endswith("/models/gemini-3-pro-image:generateContent"))
        self.assertEqual(request.headers["x-goog-api-key"], "g-key")
        self.assertNotIn("g-key", str(request.url))
        self.assertEqual(self.body()["generationConfig"]["imageConfig"], {"aspectRatio": "16:9", "imageSize": "2K"})
        path = Path(result["images"][0]["path"])
        with Image.open(path) as image:
            self.assertEqual(image.size, (48, 32))  # the finished image, not the thought
        meta = json.loads(path.with_suffix(".json").read_text())
        self.assertEqual((meta["width"], meta["effective"]["cloud_model"], meta["effective"]["seed_used"]),
                         (48, "gemini-3-pro-image", False))
        self.assertNotIn("g-key", path.with_suffix(".json").read_text())

    def test_references_are_inline_jpegs_without_exif(self):
        cloud.set_key("google", "g-key")
        source = self.root / "photo.jpg"
        exif = Image.Exif()
        exif[0x010F] = "SecretCamera"
        Image.new("RGB", (64, 64), "blue").save(source, exif=exif)
        data = base64.b64encode(png_bytes()).decode()
        self.handler = lambda r: httpx.Response(200, json={"candidates": [{"content": {"parts": [{"inlineData": {"data": data}}]}}]})
        self.assertIsNone(self.run_model("cloud-gemini-nano-banana-2.1", image_paths=[str(source)])["error"])
        part = self.body()["contents"][0]["parts"][1]["inlineData"]
        self.assertEqual(part["mimeType"], "image/jpeg")
        self.assertNotIn(b"SecretCamera", base64.b64decode(part["data"]))

    def test_lite_sends_no_image_size(self):
        cloud.set_key("google", "g-key")
        data = base64.b64encode(png_bytes()).decode()
        self.handler = lambda r: httpx.Response(200, json={"candidates": [{"content": {"parts": [{"inlineData": {"data": data}}]}}]})
        self.run_model("cloud-gemini-3.1-flash-lite-image")
        self.assertNotIn("imageSize", self.body()["generationConfig"]["imageConfig"])

    def test_blocked_prompt_explains_why(self):
        cloud.set_key("google", "g-key")
        self.handler = lambda r: httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}})
        result = self.run_model("cloud-gemini-3-pro-image")
        self.assertEqual(result["images"], [])
        self.assertIn("SAFETY", result["error"])


class OpenAITests(CloudCase):
    def setUp(self):
        super().setUp()
        cloud.set_key("openai", "sk-test")
        self.handler = lambda r: httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(png_bytes()).decode()}]})

    def test_generation_request(self):
        self.assertIsNone(self.run_model("cloud-gpt-image-2.6-flare", width=1024, height=768, quality="low")["error"])
        self.assertTrue(str(self.requests[0].url).endswith("/images/generations"))
        self.assertEqual(self.requests[0].headers["authorization"], "Bearer sk-test")
        body = self.body()
        self.assertEqual((body["model"], body["size"], body["quality"], body["n"]), ("gpt-image-2.6-flare", "1024x768", "low", 1))

    def test_auto_quality_is_left_to_openai(self):
        self.run_model("cloud-gpt-image-2.6-flare")
        self.assertNotIn("quality", self.body())

    def test_edit_sends_data_urls(self):
        self.assertIsNone(self.run_model("cloud-gpt-image-2.5-sunburst", image_paths=[self.reference()])["error"])
        self.assertTrue(str(self.requests[0].url).endswith("/images/edits"))
        self.assertTrue(self.body()["images"][0]["image_url"].startswith("data:image/jpeg;base64,"))

    def test_error_is_the_providers_and_never_contains_the_key(self):
        self.handler = lambda r: httpx.Response(403, json={"error": {"message": "Your organization must be verified. key sk-test"}})
        result = self.run_model("cloud-gpt-image-2.6-flare")
        self.assertIn("organization must be verified", result["error"])
        self.assertNotIn("sk-test", result["error"])

    def test_count_makes_one_request_per_image_and_stops_at_a_failure(self):
        good = self.handler
        self.handler = lambda r: good(r) if len(self.requests) < 2 else httpx.Response(429, json={"error": {"message": "slow"}})
        result = self.run_model("cloud-gpt-image-2.6-flare", count=5)
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(len(result["images"]), 1)
        self.assertIn("rate limit", result["error"])

    def test_cancel_while_encoding_sends_nothing(self):
        cancelled = lambda path: (server.state.cancel.set(), b"jpeg")[1]
        job = cloud.Job(model="gpt-image-2.6-flare", prompt="x", width=1024, height=1024, image_paths=["a.png"])
        with patch.object(cloud, "encode_reference", cancelled), self.assertRaises(cloud.Cancelled):
            cloud.generate("openai", job, server.state.cancel.is_set)
        self.assertEqual(self.requests, [])

    def test_cancel_between_images_stops_billing(self):
        good = self.handler

        def cancel_after_first(request):
            server.state.cancel.set()
            return good(request)
        self.handler = cancel_after_first
        result = self.run_model("cloud-gpt-image-2.6-flare", count=3)
        self.assertTrue(result["cancelled"])
        self.assertEqual(len(self.requests), 1)


class BFLTests(CloudCase):
    def setUp(self):
        super().setUp()
        cloud.set_key("bfl", "bfl-key")
        self.polls = 0

    def happy(self, request):
        url = str(request.url)
        if request.method == "POST":
            return httpx.Response(200, json={"id": "1", "polling_url": "https://api.us.bfl.ai/v1/get_result?id=1", "cost": 3})
        if "get_result" in url:
            self.polls += 1
            if self.polls == 1:
                return httpx.Response(200, json={"status": "Pending"})
            return httpx.Response(200, json={"status": "Ready", "result": {"sample": "https://delivery.us.bfl.ai/x.png"}})
        return httpx.Response(200, content=png_bytes())

    def test_submit_poll_download(self):
        self.handler = self.happy
        result = self.run_model("cloud-flux-2-pro", seed=7, image_paths=[self.reference()])
        self.assertIsNone(result["error"])
        body = self.body()
        self.assertEqual((body["seed"], body["output_format"]), (7, "png"))
        self.assertIn("input_image", body)
        self.assertEqual(self.requests[1].headers["x-key"], "bfl-key")
        self.assertNotIn("x-key", self.requests[-1].headers)  # the image URL is signed; the key stays home
        meta = json.loads(Path(result["images"][0]["path"]).with_suffix(".json").read_text())
        self.assertEqual(meta["effective"]["cost_credits"], 3)
        self.assertAlmostEqual(meta["cost_usd"], 0.03)
        self.assertEqual(costs.entries()[0]["cost_basis"], "reported")

    def test_flux3_image_only_sends_fields_the_endpoint_allows(self):
        self.handler = self.happy
        self.run_model("cloud-flux-3-image", seed=5, image_paths=[self.reference()])
        self.assertEqual(set(self.body()), {"prompt", "width", "height", "images"})

    def test_flex_sends_steps_and_guidance(self):
        self.handler = self.happy
        self.run_model("cloud-flux-2-flex", steps=30, guidance=4.0)
        self.assertEqual((self.body()["steps"], self.body()["guidance"]), (30, 4.0))

    def test_key_is_never_sent_to_a_foreign_polling_url(self):
        self.handler = lambda r: httpx.Response(200, json={"id": "1", "polling_url": "https://evil.example/poll"})
        result = self.run_model("cloud-flux-2-pro")
        self.assertIn("unexpected polling", result["error"])
        self.assertEqual(len(self.requests), 1)
        for bad in ("http://api.bfl.ai/x", "https://bfl.ai.evil.example/x", "https://notbfl.ai/x"):
            self.assertFalse(cloud._is_bfl_url(bad))

    def test_moderated_and_no_credit_errors(self):
        def moderated(request):
            if request.method == "POST":
                return httpx.Response(200, json={"id": "1", "polling_url": "https://api.bfl.ai/v1/get_result?id=1"})
            return httpx.Response(200, json={"status": "Content Moderated", "details": {"Moderation Reasons": ["x"]}})
        self.handler = moderated
        self.assertIn("Content Moderated", self.run_model("cloud-flux-2-pro")["error"])
        self.handler = lambda r: httpx.Response(402, json={"detail": "no money"})
        self.assertIn("credits", self.run_model("cloud-flux-2-pro")["error"])


class GuardTests(CloudCase):
    def test_missing_key_and_offline_fail_before_any_request(self):
        from fastapi import HTTPException
        self.handler = lambda r: httpx.Response(500)
        with self.assertRaisesRegex(HTTPException, ""):
            self.run_model("cloud-flux-2-pro")
        cloud.set_key("bfl", "k")
        server.state.offline = True
        self.addCleanup(setattr, server.state, "offline", False)
        with self.assertRaisesRegex(HTTPException, ""):
            self.run_model("cloud-flux-2-pro")
        self.assertEqual(self.requests, [])

    def test_loras_are_dropped_and_preflight_ignores_memory(self):
        request, spec, _ = server.normalize_request(server.GenerateRequest(
            prompt="x", model="cloud-flux-2-pro", loras=[server.LoraRequest(path="/a", scale=1.0)]))
        self.assertEqual(request.loras, [])
        self.assertEqual(server.preflight(request)["level"], "ok")
        estimate = server.estimate(server.EstimateRequest(model="cloud-flux-2-pro"))
        self.assertIsNone(estimate["seconds"])
        self.assertEqual(estimate["load_seconds"], 0)

    def test_local_only_actions_are_refused(self):
        from fastapi import HTTPException
        for call in (lambda: server.download_model("cloud-flux-2-pro"),
                     lambda: server.delete_model("cloud-flux-2-pro"),
                     lambda: server.load_model("cloud-flux-2-pro", server.LoadRequest())):
            with self.assertRaises(HTTPException):
                call()

    def test_quality_is_checked_against_the_model(self):
        from fastapi import HTTPException
        for model, quality in (("cloud-gpt-image-2.6-flare", "ultra"), ("cloud-gpt-image-2", "max")):
            with self.assertRaises(HTTPException):
                server.normalize_request(server.GenerateRequest(prompt="p", model=model, quality=quality))
        req, _, _ = server.normalize_request(server.GenerateRequest(prompt="p", model="cloud-gpt-image-2.5-flare", quality="max"))
        self.assertEqual(req.quality, "max")


class ConnectionTestTests(CloudCase):
    def test_without_a_key(self):
        self.assertFalse(cloud.verify("openai", [])["ok"])

    def test_google_key_check_reports_unlisted_models(self):
        cloud.set_key("google", "g")
        self.handler = lambda r: httpx.Response(200, json={"models": [{"name": "models/a"}]})
        result = cloud.verify("google", ["a", "b"])
        self.assertTrue(result["ok"])
        self.assertIn("not listed: b", result["message"])
        self.assertEqual(self.requests[0].headers["x-goog-api-key"], "g")

    def test_google_bad_key_is_400(self):
        cloud.set_key("google", "g")
        self.handler = lambda r: httpx.Response(400, json={"error": {"message": "bad", "details": [{"reason": "API_KEY_INVALID"}]}})
        result = cloud.verify("google", [])
        self.assertFalse(result["ok"])
        self.assertIn("rejected", result["message"])

    def test_openai_all_models(self):
        cloud.set_key("openai", "k")
        self.handler = lambda r: httpx.Response(200, json={"data": [{"id": "a"}]})
        self.assertIn("All 1", cloud.verify("openai", ["a"])["message"])

    def test_openai_rejected(self):
        cloud.set_key("openai", "k")
        self.handler = lambda r: httpx.Response(401, json={"error": {"message": "no"}})
        self.assertFalse(cloud.verify("openai", [])["ok"])

    def test_bfl_balance(self):
        cloud.set_key("bfl", "k")
        self.handler = lambda r: httpx.Response(200, json={"credits": 12.5})
        self.assertIn("12.5 credits", cloud.verify("bfl", [])["message"])

    def test_network_failure_is_not_called_a_bad_key(self):
        cloud.set_key("bfl", "k")

        def fail(request):
            raise httpx.ConnectError("no route")
        self.handler = fail
        result = cloud.verify("bfl", [])
        self.assertFalse(result["ok"])
        self.assertIn("could not be reached", result["message"])

    def test_test_image_returns_a_thumbnail_and_saves_nothing(self):
        cloud.set_key("openai", "k")
        self.handler = lambda r: httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(png_bytes((512, 512))).decode()}]})
        result = cloud.test_image("openai")
        self.assertTrue(result["ok"])
        self.assertEqual(self.body()["quality"], "low")
        with Image.open(io.BytesIO(base64.b64decode(result["thumbnail"]))) as thumb:
            self.assertLessEqual(max(thumb.size), 128)
        self.assertEqual(list(self.root.rglob("*.png")), [])  # it's in the ledger, but not in the history

    def test_routes_refuse_unknown_providers_and_respect_offline(self):
        from fastapi import HTTPException
        with self.assertRaises(HTTPException):
            server.set_cloud_key(server.CloudKeyRequest(provider="nope", key="x"))
        server.state.offline = True
        self.addCleanup(setattr, server.state, "offline", False)
        self.assertIn("Offline", server.test_cloud_key(server.CloudTestRequest(provider="bfl"))["message"])
        self.assertIn("Offline", server.test_cloud_image(server.CloudTestRequest(provider="bfl"))["message"])
        self.assertEqual(self.requests, [])


class CostTests(CloudCase):
    def test_google_uses_tokens_and_prices_thinking_as_text(self):
        usage = {"promptTokenCount": 1000, "candidatesTokenCount": 1120, "thoughtsTokenCount": 100,
                 "candidatesTokensDetails": [{"modality": "IMAGE", "tokenCount": 1120}]}
        cost, basis = costs.google_cost("gemini-nano-banana-2.1", usage, "1K")
        self.assertEqual(basis, "computed")
        self.assertAlmostEqual(cost, 1000 * 1.5e-6 + 100 * 7.5e-6 + 1120 * 30e-6)

    def test_google_falls_back_to_the_list_price(self):
        self.assertEqual(costs.google_cost("gemini-3-pro-image", None, "2K"), (0.134, "list"))
        # Tokens without the image/text split can't be priced, so they aren't guessed at.
        self.assertEqual(costs.google_cost("gemini-3-pro-image", {"candidatesTokenCount": 1220}, "1K"), (0.134, "list"))
        self.assertEqual(costs.google_cost("gemini-unknown", None, "1K"), (None, "unknown"))

    def test_published_per_image_prices_match_token_rates(self):
        self.assertAlmostEqual(1120 * costs.GOOGLE_PRICES["gemini-nano-banana-2.1"]["image_output"] / 1e6, 0.0336)
        self.assertAlmostEqual(1680 * costs.GOOGLE_PRICES["gemini-3.1-flash-image"]["image_output"] / 1e6, 0.1008)

    def test_openai_uses_usage_and_admits_when_it_cannot_know(self):
        usage = {"input_tokens": 300, "output_tokens": 1000,
                 "input_tokens_details": {"text_tokens": 100, "image_tokens": 200}}
        cost, basis = costs.openai_cost(usage)
        self.assertEqual(basis, "computed")
        self.assertAlmostEqual(cost, 100 * 5e-6 + 200 * 8e-6 + 1000 * 30e-6)
        self.assertEqual(costs.openai_cost(None), (None, "unknown"))

    def test_bfl_credits(self):
        self.assertAlmostEqual(costs.bfl_cost(1.4)[0], 0.014)
        self.assertEqual(costs.bfl_cost(None), (None, "unknown"))

    def test_estimate_is_list_price_or_recent_runs_never_a_guess(self):
        self.assertEqual(costs.estimate("google", "gemini-3-pro-image", 2048, 1536), (0.134, "list"))
        self.assertEqual(costs.estimate("openai", "m", 1024, 1024, "low", ledger=[]), (None, "none"))
        ledger = [{"model": "m", "cost_usd": 0.02, "quality": "low", "megapixels": 1.05},
                  {"model": "m", "cost_usd": 0.04, "quality": "low", "megapixels": 1.05},
                  {"model": "m", "cost_usd": 9.0, "quality": "high", "megapixels": 1.05},
                  {"model": "m", "cost_usd": 9.0, "quality": "low", "megapixels": 4.0}]
        cost, basis = costs.estimate("openai", "m", 1024, 1024, "low", ledger=ledger)
        self.assertEqual(basis, "history")
        self.assertAlmostEqual(cost, 0.03)

    def test_summary_totals_by_day_and_damaged_lines(self):
        from datetime import datetime
        now = datetime(2026, 11, 5, 12)
        for when, provider, cost in [(datetime(2026, 11, 5, 9), "google", 0.5), (datetime(2026, 11, 2, 9), "bfl", 0.25),
                                     (datetime(2026, 9, 1, 9), "openai", 1.0), (datetime(2026, 11, 5, 10), "openai", None)]:
            with patch.object(costs.time, "time", return_value=when.timestamp()):
                image = self.root / f"{provider}-{when.day}.png"
                image.touch()
                costs.record(provider=provider, model=provider + "-m", cost_usd=cost, path=str(image))
        with costs.LEDGER.open("a") as stream:
            stream.write('{"broken\n')
        result = costs.summary(now)
        self.assertEqual((result["today_usd"], result["month_usd"], result["all_usd"]), (0.5, 0.75, 1.75))
        self.assertEqual((result["images"], result["unknown_cost_images"]), (4, 1))
        self.assertEqual(result["by_model"][0]["model"], "openai-m")
        self.assertEqual(len(result["days"]), 30)
        self.assertEqual({row["day"] for row in result["by_day"]}, {"2026-11-05", "2026-11-02"})
        self.assertEqual(result["entries"][0]["provider"], "openai")  # newest first

    def test_a_generated_image_reaches_the_ledger_and_a_test_image_does_not(self):
        cloud.set_key("openai", "sk-test")
        self.handler = lambda r: httpx.Response(200, json={
            "data": [{"b64_json": base64.b64encode(png_bytes()).decode()}],
            "usage": {"input_tokens": 10, "output_tokens": 1000, "input_tokens_details": {"text_tokens": 10}}})
        result = self.run_model("cloud-gpt-image-2.6-flare")
        meta = json.loads(Path(result["images"][0]["path"]).with_suffix(".json").read_text())
        self.assertAlmostEqual(meta["cost_usd"], 10 * 5e-6 + 1000 * 30e-6)
        self.assertIn("It cost about", cloud.test_image("openai")["message"])
        self.assertEqual([e["kind"] for e in costs.entries()], ["generate"])
        self.assertNotIn("sk-test", costs.LEDGER.read_text())

    def test_an_image_paid_for_before_cancel_is_kept_and_counted(self):
        cloud.set_key("openai", "sk-test")

        def cancel_then_reply(request):
            server.state.cancel.set()
            return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(png_bytes()).decode()}]})
        self.handler = cancel_then_reply
        with patch.object(cloud, "_interruptible", lambda work, stop: work()):
            result = self.run_model("cloud-gpt-image-2.6-flare", count=2)
        self.assertTrue(result["cancelled"])
        self.assertEqual(len(result["images"]), 1)
        self.assertEqual(len(costs.entries()), 1)


class CloudBatchTests(CloudCase):
    def request(self, model):
        return server.GenerateRequest(prompt="x", model=model, seed=1)

    def test_runs_at_the_same_time_with_separate_files_and_errors(self):
        cloud.set_key("openai", "openai-test-key")
        cloud.set_key("google", "google-test-key")
        barrier = threading.Barrier(3, timeout=5)  # only passes if all three requests are in flight together
        image = base64.b64encode(png_bytes()).decode()

        def handler(request):
            barrier.wait()
            body = json.loads(request.content)
            if "generateContent" in str(request.url):
                return httpx.Response(400, json={"error": {"message": "nope"}})
            return httpx.Response(200, json={"data": [{"b64_json": image}], "usage": {"output_tokens": 100}, "m": body["model"]})
        self.handler = handler
        result = server.generate_cloud_batch(server.CloudBatchRequest(requests=[
            self.request("cloud-gpt-image-2.6-flare"), self.request("cloud-gpt-image-2.6-sunburst"),
            self.request("cloud-gemini-3-pro-image")]))
        by_model = {row["model"]: row for row in result["results"]}
        self.assertEqual(len(by_model["cloud-gpt-image-2.6-flare"]["images"]), 1)
        self.assertEqual(len(by_model["cloud-gpt-image-2.6-sunburst"]["images"]), 1)
        self.assertIn("nope", by_model["cloud-gemini-3-pro-image"]["error"])
        paths = [row["images"][0]["path"] for row in result["results"] if row["images"]]
        self.assertEqual(len(set(paths)), 2)
        self.assertEqual(len(server.state.images_done), 2)
        self.assertFalse(server.state.lock.locked())

    def test_one_models_early_failure_keeps_the_others_finished_images(self):
        cloud.set_key("openai", "openai-test-key")
        real_save = server._save_cloud_image
        self.handler = lambda r: httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(png_bytes()).decode()}]})

        def fail_for_one(image, req, spec, *rest):
            if req.model == "cloud-gpt-image-2.6-flare":
                raise OSError("disk full")
            return real_save(image, req, spec, *rest)
        with patch.object(server, "_save_cloud_image", fail_for_one):
            result = server.generate_cloud_batch(server.CloudBatchRequest(requests=[
                self.request("cloud-gpt-image-2.6-flare"), self.request("cloud-gpt-image-2.6-sunburst")]))
        by_model = {row["model"]: row for row in result["results"]}
        self.assertTrue(by_model["cloud-gpt-image-2.6-flare"]["error"])
        self.assertEqual(len(by_model["cloud-gpt-image-2.6-sunburst"]["images"]), 1)
        self.assertEqual(len(result["images"]), 1)
        # The ledger lists images in the library, so the one that never got a file isn't in it.
        self.assertEqual([e["model"] for e in costs.entries()], ["gpt-image-2.6-sunburst"])

    def test_output_folder_failure_is_that_models_error_not_the_whole_batch(self):
        cloud.set_key("openai", "openai-test-key")
        self.handler = lambda r: httpx.Response(500)
        blocked = self.root / "blocked"
        blocked.write_text("a file, not a folder")
        with patch.object(server, "OUTPUT_ROOT", blocked):
            result = server.generate_cloud_batch(server.CloudBatchRequest(requests=[self.request("cloud-gpt-image-2.6-flare")]))
        self.assertTrue(result["results"][0]["error"])
        self.assertFalse(server.state.lock.locked())

    def test_a_ledger_failure_does_not_fail_a_saved_image(self):
        cloud.set_key("openai", "openai-test-key")
        self.handler = lambda r: httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(png_bytes()).decode()}]})
        with patch.object(costs, "record", side_effect=OSError("disk full")):
            result = self.run_model("cloud-gpt-image-2.6-flare")
        self.assertIsNone(result["error"])
        self.assertEqual(len(result["images"]), 1)

    def test_quality_is_the_one_the_request_ran_with(self):
        cloud.set_key("openai", "openai-test-key")
        self.handler = lambda r: httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(png_bytes()).decode()}]})
        self.run_model("cloud-gpt-image-2.6-flare", quality="low")
        self.assertEqual(costs.entries()[0]["quality"], "low")

    def test_refuses_local_models_and_missing_keys_before_any_request(self):
        from fastapi import HTTPException
        self.handler = lambda r: httpx.Response(500)
        with self.assertRaises(HTTPException):
            server.generate_cloud_batch(server.CloudBatchRequest(requests=[self.request("flux2-klein-4b")]))
        with self.assertRaises(HTTPException):
            server.generate_cloud_batch(server.CloudBatchRequest(requests=[self.request("cloud-flux-2-pro")]))
        self.assertEqual(self.requests, [])


class KeepLoadedTests(CloudCase):
    """Local models in a comparison: both stay in memory only when the next one fits."""

    def setUp(self):
        super().setUp()
        server.on_mlx(server._unload)
        self.addCleanup(lambda: server.on_mlx(server._unload))
        self.built = []

        def pipeline(spec, *args, **kwargs):
            built = MagicMock(name=spec.id)
            self.built.append(spec.id)
            return built
        self.more = [patch.object(server.catalog, "Pipeline", pipeline), patch.object(server, "materialize"),
                     patch.object(server.downloads, "status", return_value={"downloaded": True}),
                     patch.object(server.optimized, "usable", return_value=None)]
        for p in self.more:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in self.more])
        self.a, self.b = catalog.get("flux3-klein-int4"), catalog.get("flux3-klein-int8")

    def load(self, spec, keep=True):
        return server.on_mlx(server._load, spec, None, None, 1.0, None, keep)

    def test_second_model_stays_loaded_when_it_fits_and_returns_without_a_reload(self):
        with patch.object(server, "_fits_alongside", return_value=True):
            first = self.load(self.a)
            second = self.load(self.b)
            self.assertIn(first, server.state.parked.values())
            self.assertIs(self.load(self.a), first)
            self.assertIn(second, server.state.parked.values())
        self.assertEqual(self.built, [self.a.id, self.b.id])

    def test_first_model_unloads_when_the_next_would_not_fit(self):
        with patch.object(server, "_fits_alongside", return_value=False):
            self.load(self.a)
            self.load(self.b)
        self.assertEqual(server.state.parked, {})
        self.assertEqual(server.state.model_key[0], self.b.id)

    def test_normal_generation_and_unload_free_parked_models(self):
        with patch.object(server, "_fits_alongside", return_value=True):
            self.load(self.a)
            self.load(self.b)
        self.load(catalog.get("flux2-klein-4b"), keep=False)
        self.assertEqual(server.state.parked, {})
        with patch.object(server, "_fits_alongside", return_value=True):
            self.load(self.a)
            self.load(self.b)
        server.on_mlx(server._unload)
        self.assertEqual(server.state.parked, {})
        self.assertIsNone(server.state.pipeline)

    def test_a_model_that_freed_its_encoder_is_not_kept(self):
        with patch.object(server, "_fits_alongside", return_value=True):
            self.load(self.a)
            server.state.encoder_evicted = True
            self.load(self.b)
        self.assertEqual(server.state.parked, {})

    def test_fit_check_needs_room_for_the_model_and_other_apps(self):
        gb = 1_000_000_000
        spec = catalog.get("flux3-klein-int4")
        needed = int(server._memory_gb(spec)["4"] * gb)
        snapshot = lambda free: {"free_bytes": free, "pressure": "ok", "swap_bytes": 0}
        with patch.object(server, "memory_snapshot", return_value=snapshot(needed + server.COMPARE_HEADROOM_BYTES)), \
             patch.object(server, "_pressure_level", return_value="normal"):
            self.assertTrue(server._fits_alongside(spec, 4))
            self.assertFalse(server._fits_alongside(spec, 4, loaded_bytes=gb))
        with patch.object(server, "memory_snapshot", return_value=snapshot(needed)), \
             patch.object(server, "_pressure_level", return_value="normal"):
            self.assertFalse(server._fits_alongside(spec, 4))
        with patch.object(server, "memory_snapshot", return_value={**snapshot(10**12), "pressure": "warning"}), \
             patch.object(server, "_pressure_level", return_value="warning"):
            self.assertFalse(server._fits_alongside(spec, 4))

    def test_plan_reports_order_and_cost(self):
        with patch.object(server, "_fits_alongside", return_value=False):
            plan = server.compare_plan(server.ComparePlanRequest(models=[
                self.a.id, self.b.id, "cloud-gemini-3-pro-image", "cloud-gpt-image-2.6-flare"]))
        self.assertEqual((plan["local_models"], plan["cloud_models"], plan["keep_loaded"]), (2, 2, False))
        self.assertAlmostEqual(plan["cost_usd"], 0.134)
        self.assertEqual(plan["costs_unknown"], 1)


class OpenAIOptionTests(CloudCase):
    MODEL = "cloud-gpt-image-2.6-flare"

    def setUp(self):
        super().setUp()
        cloud.set_key("openai", "sk-test")

    @staticmethod
    def reply(data, **extra):
        return lambda r: httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(data).decode()}], **extra})

    def run_format(self, data, model=MODEL, **options):
        self.handler = self.reply(data)
        return self.run_model(model, **options)

    def request_body(self):
        return json.loads(self.requests[-1].content)

    @staticmethod
    def gif_bytes():
        out = io.BytesIO()
        frames = [Image.new("P", (32, 32), i) for i in range(3)]
        frames[0].save(out, format="GIF", save_all=True, append_images=frames[1:], duration=100, loop=0)
        return out.getvalue()

    SVG = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 200"><rect width="300" height="200"/></svg>'

    def test_gif_is_requested_and_saved_byte_for_byte(self):
        data = self.gif_bytes()
        result = self.run_format(data, output_format="gif", frame_count=12, frame_duration_ms=80, quality="low", complexity="medium")
        self.assertIsNone(result["error"])
        body = self.request_body()
        self.assertEqual((body["output_format"], body["frame_count"], body["frame_duration_ms"], body["quality"], body["complexity"]),
                         ("gif", 12, 80, "low", "medium"))
        self.assertNotIn("size", body)  # OpenAI picks a frame size that fits its budget
        path = Path(result["images"][0]["path"])
        self.assertEqual((path.suffix, path.read_bytes()), (".gif", data))
        meta = json.loads(path.with_suffix(".json").read_text())
        self.assertEqual((meta["output_format"], meta["frame_count"], meta["width"]), ("gif", 12, 32))

    def test_gif_with_the_chosen_frame_size_has_no_minimum(self):
        self.run_format(self.gif_bytes(), output_format="gif", gif_auto_size=False, width=500, height=256)
        self.assertEqual(self.request_body()["size"], "496x256")

    def test_svg_is_saved_with_unknown_cost_and_its_token_counts(self):
        self.handler = self.reply(self.SVG, usage={"output_tokens": 10}, svg_usage={"model": "x", "total_tokens": 5})
        result = self.run_model(self.MODEL, output_format="svg", complexity="low")
        path = Path(result["images"][0]["path"])
        self.assertEqual((path.suffix, path.read_bytes()), (".svg", self.SVG))
        self.assertNotIn("size", self.request_body())
        meta = json.loads(path.with_suffix(".json").read_text())
        self.assertEqual((meta["width"], meta["height"]), (300, 200))
        self.assertEqual((meta["cost_usd"], meta["cost_basis"]), (None, "unknown"))
        self.assertEqual(meta["effective"]["svg_usage"]["total_tokens"], 5)
        self.assertEqual(costs.entries()[0]["output_format"], "svg")

    def test_unsafe_or_wrong_files_are_not_saved(self):
        evil = [b'<svg xmlns="http://www.w3.org/2000/svg"><script>x</script></svg>',
                b'<svg xmlns="http://www.w3.org/2000/svg" onload="x"/>',
                b'<!DOCTYPE svg [<!ENTITY a "b">]><svg/>', b"<html/>", b"not xml"]
        for data in evil:
            result = self.run_format(data, output_format="svg")
            self.assertTrue(result["error"], data)
            self.assertEqual(result["images"], [])
        self.assertTrue(self.run_format(png_bytes(), output_format="gif")["error"])
        self.assertEqual(costs.entries(), [])
        self.assertEqual([p for p in self.root.rglob("*") if p.suffix in (".svg", ".gif")], [])

    def test_only_the_models_with_a_feature_accept_it(self):
        from fastapi import HTTPException
        refused = [dict(model="cloud-gpt-image-2.5-flare", output_format="gif"),
                   dict(model="cloud-gpt-image-2", output_format="svg"),
                   dict(model=self.MODEL, output_format="webp"),
                   dict(model=self.MODEL, output_format="gif", frame_count=17),
                   dict(model=self.MODEL, output_format="gif", frame_count=1),
                   dict(model=self.MODEL, output_format="gif", frame_duration_ms=10),
                   dict(model=self.MODEL, complexity="extreme"),
                   dict(model="cloud-gpt-image-2", markup_path="/x.png", image_paths=["/y.png"]),
                   dict(model=self.MODEL, markup_path="/x.png"),
                   dict(model=self.MODEL, output_format="gif", image_paths=["/y.png"])]
        for options in refused:
            with self.assertRaises(HTTPException, msg=options):
                server.normalize_request(server.GenerateRequest(prompt="p", **options))
        req, _, _ = server.normalize_request(server.GenerateRequest(prompt="p", model="cloud-gpt-image-2.5-flare", complexity="low"))
        self.assertIsNone(req.complexity)  # only stated for 2.6, so it isn't sent

    def test_other_providers_never_see_these_fields(self):
        req, _, _ = server.normalize_request(server.GenerateRequest(
            prompt="p", model="cloud-gemini-3-pro-image", quality="max", output_format="gif", frame_count=5, markup_path="/x"))
        self.assertEqual((req.quality, req.output_format, req.frame_count, req.markup_path), (None, None, None, None))

    def test_markup_matches_the_first_image_as_sent(self):
        reference, layer = self.root / "ref.png", self.root / "markup.png"
        Image.new("RGB", (400, 300), "white").save(reference)
        overlay = Image.new("RGBA", (800, 600), (0, 0, 0, 0))
        overlay.paste((255, 0, 0, 255), (0, 0, 100, 100))
        overlay.save(layer)
        result = self.run_format(png_bytes(), image_paths=[str(reference)], markup_path=str(layer))
        self.assertIsNone(result["error"])
        self.assertTrue(str(self.requests[-1].url).endswith("/images/edits"))
        sent = self.request_body()["markup"]["image_url"]
        self.assertTrue(sent.startswith("data:image/png;base64,"))
        with Image.open(io.BytesIO(base64.b64decode(sent.split(",", 1)[1]))) as image:
            self.assertEqual((image.format, image.size, image.mode), ("PNG", (400, 300), "RGBA"))
            self.assertEqual(image.getpixel((399, 299))[3], 0)
            self.assertEqual(image.getpixel((10, 10))[3], 255)

    def test_opaque_markup_is_refused_before_any_request(self):
        reference, layer = self.root / "ref.png", self.root / "markup.png"
        Image.new("RGB", (40, 30)).save(reference)
        Image.new("RGB", (40, 30)).save(layer)
        result = self.run_format(png_bytes(), image_paths=[str(reference)], markup_path=str(layer))
        self.assertIn("transparent", result["error"])
        self.assertEqual(self.requests, [])

    def test_estimate_separates_gifs_from_stills(self):
        ledger = [{"model": "m", "cost_usd": 0.02, "quality": "low", "megapixels": 1.0, "output_format": None, "frame_count": None},
                  {"model": "m", "cost_usd": 0.50, "quality": "low", "megapixels": 1.0, "output_format": "gif", "frame_count": 9}]
        still = costs.estimate("openai", "m", 1000, 1000, "low", ledger, output_format=None, frame_count=None)
        gif = costs.estimate("openai", "m", 1000, 1000, "low", ledger, output_format="gif", frame_count=9)
        other = costs.estimate("openai", "m", 1000, 1000, "low", ledger, output_format="gif", frame_count=16)
        self.assertEqual((still[0], gif[0], other[0]), (0.02, 0.50, None))

    def test_a_failed_settings_file_keeps_the_saved_paid_image(self):
        with patch.object(server, "atomic_json", side_effect=OSError("disk full")):
            result = self.run_format(self.SVG, output_format="svg")
        self.assertIsNone(result["error"])
        path = Path(result["images"][0]["path"])
        self.assertTrue(path.is_file())
        self.assertEqual([e["path"] for e in costs.entries()], [str(path)])

    def test_gif_estimates_keep_chosen_and_automatic_sizes_apart(self):
        self.run_format(self.gif_bytes(), output_format="gif", width=256, height=256, frame_count=4)  # automatic
        self.handler = self.reply(self.gif_bytes(), usage={"output_tokens": 1000})
        self.run_model(self.MODEL, output_format="gif", width=256, height=256, frame_count=4)
        def estimate(**options):
            return server.estimate(server.EstimateRequest(prompt="", model=self.MODEL, output_format="gif", frame_count=4, **options))
        self.assertAlmostEqual(estimate(width=2048, height=1024)["cost_usd"], 1000 * 30e-6)  # the editor's size is irrelevant
        self.assertEqual(estimate(width=256, height=256, gif_auto_size=False)["cost_basis"], "none")


class LedgerFollowsLibraryTests(CloudCase):
    def add(self, name, cost=0.1):
        image = self.root / name
        image.touch()
        costs.record(provider="openai", model="m", kind="generate", cost_usd=cost, prompt=f"secret {name}", path=str(image))
        return image

    def test_a_removed_image_leaves_every_figure_and_the_file(self):
        keep, gone = self.add("keep.png"), self.add("gone.gif", 0.5)
        self.assertEqual(len(costs.entries()), 2)
        gone.unlink()
        summary = costs.summary()
        self.assertEqual((summary["images"], summary["all_usd"]), (1, 0.1))
        self.assertEqual([e["path"] for e in summary["entries"]], [str(keep)])
        self.assertEqual(summary["by_model"][0]["images"], 1)
        self.assertNotIn("secret gone", costs.LEDGER.read_text())

    def test_entries_without_an_image_are_dropped(self):
        costs.record(provider="bfl", model="m", kind="test", cost_usd=1.0, path=None)
        self.assertEqual(costs.entries(), [])
        self.assertEqual(costs.LEDGER.read_text(), "")

    def test_estimates_forget_removed_images(self):
        image = self.add("a.png")
        with costs.LEDGER.open("w") as stream:
            stream.write(json.dumps({"time": 1.0, "model": "m", "cost_usd": 0.1, "quality": "low", "megapixels": 1.0, "path": str(image)}) + "\n")
        self.assertEqual(costs.estimate("openai", "m", 1000, 1000, "low")[1], "history")
        image.unlink()
        self.assertEqual(costs.estimate("openai", "m", 1000, 1000, "low"), (None, "none"))

    def test_a_failed_rewrite_still_reports_correctly(self):
        self.add("keep.png")
        self.add("gone.png").unlink()
        with patch.object(costs.os, "replace", side_effect=OSError):
            self.assertEqual(len(costs.entries()), 1)
        self.assertEqual(list(self.root.glob("*.tmp")), [])
        self.assertEqual(len(costs.entries()), 1)

    def test_forget_clears_the_reference_cache_unless_a_job_runs(self):
        cache = self.root / "reference-cache"
        cache.mkdir()
        (cache / "copy.png").touch()
        with patch.object(server, "REF_CACHE", cache):
            self.assertTrue(server.state.lock.acquire(blocking=False))
            try:
                server.forget_removed_images()
                self.assertTrue(cache.exists())
            finally:
                server.state.lock.release()
            server.forget_removed_images()
            self.assertFalse(cache.exists())
        self.assertIsNone(server.state.busy)

    def test_forget_during_a_job_clears_the_cache_when_it_ends(self):
        cache = self.root / "reference-cache"
        cache.mkdir()
        with patch.object(server, "REF_CACHE", cache):
            self.assertTrue(server.state.lock.acquire(blocking=False))
            server.forget_removed_images()
            self.assertTrue(cache.exists())
            server._release()
            self.assertFalse(cache.exists())
        self.assertFalse(server.state.reference_cache_stale)


if __name__ == "__main__":
    unittest.main()
