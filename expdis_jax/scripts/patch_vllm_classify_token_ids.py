#!/usr/bin/env python3
"""Extend only vLLM 0.11's /classify request schema to accept token IDs.

The v0.11 renderer already handles ``list[int]`` and ``list[list[int]]``,
but ``ClassificationRequest`` narrows the HTTP schema to strings, so the
semantic-diversity classifier inputs (token IDs) are rejected by the stock
endpoint. This changes only that Pydantic annotation; tokenization, inference,
pooling, and probabilities are unchanged. Run it once in the classifier's
serving environment before starting ``vllm serve``.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
from pathlib import Path


EXPECTED_VERSION = "0.11.0"
OLD = "    input: Union[list[str], str]\n"
NEW = "    input: Union[list[int], list[list[int]], list[str], str]\n"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> int:
    observed_version = importlib.metadata.version("vllm")
    if observed_version != EXPECTED_VERSION:
        raise SystemExit(
            f"refusing to patch vLLM {observed_version!r}; expected {EXPECTED_VERSION!r}"
        )
    spec = importlib.util.find_spec("vllm.entrypoints.openai.protocol")
    if spec is None or spec.origin is None:
        raise SystemExit("could not locate vLLM's OpenAI protocol module")
    path = Path(spec.origin).resolve()
    original = path.read_bytes()
    text = original.decode("utf-8")
    if text.count(OLD) != 1:
        raise SystemExit(
            "refusing non-exact vLLM patch: expected one ClassificationRequest input annotation"
        )
    patched = text.replace(OLD, NEW, 1).encode("utf-8")
    path.write_bytes(patched)

    # Import only after the on-disk patch and prove the Pydantic request keeps
    # nested integer IDs as integers.
    from vllm.entrypoints.openai.protocol import ClassificationRequest

    probe = ClassificationRequest(
        model="classifier",
        input=[[151644, 42, 151645, 43, 151645]],
    )
    if probe.input != [[151644, 42, 151645, 43, 151645]]:
        raise SystemExit(f"patched request schema altered token IDs: {probe.input!r}")
    print(
        "[patch] vLLM /classify accepts token IDs: "
        f"path={path} original_sha256={sha256_bytes(original)} "
        f"patched_sha256={sha256_bytes(patched)} probe_ok=true",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
