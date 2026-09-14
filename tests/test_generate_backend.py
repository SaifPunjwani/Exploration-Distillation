import os
import threading
import time
import unittest
from unittest import mock

from tmx_jax import generate


class _OKResponse:
    def raise_for_status(self):
        return None


class GenerateBackendTests(unittest.TestCase):
    def tearDown(self):
        generate._HEALTH_CACHE.clear()
        generate._PRINTED_BACKEND_CONFIG = False
        for key in (
            "TMX_VLLM_HEALTH_FILTER_TTL_SECONDS",
            "TMX_VLLM_HEALTH_FILTER_CONCURRENCY",
            "TMX_VLLM_ENDPOINT_MAX_INFLIGHT",
            "TMX_VLLM_FAILOVER_ON_ERROR",
        ):
            os.environ.pop(key, None)

    def test_health_filter_keeps_order_and_caches_recent_results(self):
        calls = []

        def fake_get(url, timeout):
            calls.append(url)
            if "bad" in url:
                raise RuntimeError("down")
            return _OKResponse()

        os.environ["TMX_VLLM_HEALTH_FILTER_TTL_SECONDS"] = "60"
        os.environ["TMX_VLLM_HEALTH_FILTER_CONCURRENCY"] = "8"
        urls = ["http://good-a/v1", "http://bad/v1", "http://good-b/v1"]
        with mock.patch.object(generate._SESSION, "get", side_effect=fake_get):
            self.assertEqual(
                generate._filter_live_servers(urls),
                ["http://good-a/v1", "http://good-b/v1"],
            )
            self.assertEqual(
                generate._filter_live_servers(urls),
                ["http://good-a/v1", "http://good-b/v1"],
            )

        self.assertEqual(len(calls), 3)

    def test_batched_generate_limits_endpoint_inflight_and_fails_over(self):
        active = {"u0": 0, "u1": 0}
        max_active = {"u0": 0, "u1": 0}
        lock = threading.Lock()
        failures_left = {"u0": 1}

        def fake_complete(url, prompt, count, *_args, **_kwargs):
            with lock:
                active[url] += 1
                max_active[url] = max(max_active[url], active[url])
            try:
                time.sleep(0.01)
                if url == "u0" and failures_left["u0"] > 0:
                    failures_left["u0"] -= 1
                    raise RuntimeError("transient")
                return [
                    generate.Completion(
                        text=f"{url}:{prompt}:{i}",
                        token_ids=[],
                        token_logprobs=[],
                        finish_reason="stop",
                    )
                    for i in range(count)
                ]
            finally:
                with lock:
                    active[url] -= 1

        os.environ["TMX_VLLM_ENDPOINT_MAX_INFLIGHT"] = "1"
        os.environ["TMX_VLLM_FAILOVER_ON_ERROR"] = "1"
        with mock.patch.object(generate, "_filter_live_servers", side_effect=lambda urls: urls):
            with mock.patch.object(generate, "_vllm_complete", side_effect=fake_complete):
                out = generate.batched_generate(
                    ["p0", "p1"],
                    ["u0", "u1"],
                    n_per_prompt=4,
                    max_tokens=8,
                    concurrency=8,
                    fanout_per_prompt=4,
                )

        self.assertEqual([len(x) for x in out], [4, 4])
        self.assertLessEqual(max_active["u0"], 1)
        self.assertLessEqual(max_active["u1"], 1)
        self.assertEqual(failures_left["u0"], 0)


if __name__ == "__main__":
    unittest.main()
