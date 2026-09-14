"""Reject stale batches when peers arrive before the next publication."""
import http.server
import threading
import urllib.error
import urllib.request

import pytest

from expdis_jax import train


def test_http_handoff_requires_the_requested_generation(monkeypatch):
    monkeypatch.setattr(train, "_BATCH_SHARD_PUBLISHED_GENERATION", 4)
    monkeypatch.setattr(train, "_BATCH_SHARD_PAYLOADS", {1: b"old batch"})
    monkeypatch.setattr(train, "_BATCH_SHARD_SERVED_RANKS", set())
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), train._BatchShardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}/batch"
    try:
        for suffix in ["5/1", "3/1", "1"]:
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(f"{base}/{suffix}")
            assert error.value.code == 404
            assert train._BATCH_SHARD_SERVED_RANKS == set()
        with train._BATCH_SHARD_LOCK:
            train._BATCH_SHARD_PAYLOADS[1] = b"new batch"
            train._BATCH_SHARD_PUBLISHED_GENERATION = 5
        with urllib.request.urlopen(f"{base}/5/1") as response:
            assert response.read() == b"new batch"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
