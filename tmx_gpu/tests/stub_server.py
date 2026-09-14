#!/usr/bin/env python3
"""Stub OpenAI-completions server for CPU end-to-end tests (stdlib only).

Emulates the vLLM /v1/completions + /health surface the pool talks to:
- Prompts embed their ground truth as "[gt=<answer>]"; the stub parses it and
  answers with canned math completions long enough (>= MIN gate of 128 tokens)
  and varied enough (no 40-char loop) to pass the QualityPool gates.
- The request seed deterministically selects a variant so every group of 4
  mixes correct / wrong / length-clipped rows (dynamic sampling stays
  informative): seed % 4 in {1, 2} -> correct, 0 -> wrong, 3 -> clipped.
- A configurable set of request ordinals returns HTTP 500 once each, to
  exercise the pool's retry path without producing error completions.

Run standalone: python stub_server.py [port]. In tests use start_stub_server().
"""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

GT_MARKER = re.compile(r"\[gt=([^\]]+)\]")


def _filler(seed: int, sentences: int = 60) -> str:
    """Non-repeating reasoning filler (~200+ tokens, defeats the loop gate)."""
    return " ".join(
        f"Step {i} considers quantity {seed + i * 7} and refines the estimate."
        for i in range(1, sentences + 1)
    )


class StubState:
    def __init__(self, fail_ordinals=(7,)):
        self.lock = threading.Lock()
        self.request_count = 0
        self.failed_once: set = set()
        self.fail_ordinals = set(fail_ordinals)


def make_handler(state: StubState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # silence request logging
            pass

        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"OK")
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if not self.path.endswith("/completions"):
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            with state.lock:
                state.request_count += 1
                ordinal = state.request_count
                inject_500 = (ordinal in state.fail_ordinals
                              and ordinal not in state.failed_once)
                if inject_500:
                    state.failed_once.add(ordinal)
            if inject_500:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b"injected failure")
                return

            prompt = str(payload.get("prompt", ""))
            seed = int(payload.get("seed", 0) or 0)
            m = GT_MARKER.search(prompt)
            gt = m.group(1) if m else "0"
            variant = seed % 4
            if variant in (1, 2):  # correct, terminated
                text = f"{_filler(seed)} The answer is \\boxed{{{gt}}}."
                finish = "stop"
            elif variant == 0:  # wrong, terminated
                text = f"{_filler(seed)} The answer is \\boxed{{{gt}999}}."
                finish = "stop"
            else:  # clipped: ran out of budget, no final answer
                text = _filler(seed, sentences=80)
                finish = "length"
            body = json.dumps({
                "choices": [{"text": text, "finish_reason": finish}],
                "usage": {"completion_tokens": max(1, len(text) // 4)},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def start_stub_server(port: int = 0, fail_ordinals=(7,)):
    """Start on 127.0.0.1:<port> (0 = ephemeral); returns (server, port, state)."""
    state = StubState(fail_ordinals=fail_ordinals)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_address[1], state


if __name__ == "__main__":
    import sys

    _server, bound_port, _state = start_stub_server(
        int(sys.argv[1]) if len(sys.argv) > 1 else 8399)
    print(f"stub server on 127.0.0.1:{bound_port}", flush=True)
    threading.Event().wait()
