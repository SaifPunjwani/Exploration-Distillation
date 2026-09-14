"""vLLM HTTP client for generation.

Ported from `expdis_torch/vllm_generate.py`; this version is pure-Python (no torch).
Generation returns raw text; per-token logprobs are requested only when
``EXPDIS_VLLM_RETURN_LOGPROBS=1`` (debugging). Training never uses vLLM logprobs and
recomputes them on the training policy.
"""

from __future__ import annotations

import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from threading import Semaphore
from typing import Callable, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter


# Module-level Session with connection pooling. Cross-region vLLM RTT can be
# ~80ms; without keep-alive every
# request pays a fresh TCP+TLS handshake, ~10-15s/step wasted at 128 calls.
_SESSION = requests.Session()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)) or str(default))
    except ValueError:
        return default


_HTTP_POOL_MAXSIZE = max(128, _env_int("EXPDIS_HTTP_POOL_MAXSIZE", 512))
_HTTP_ADAPTER = HTTPAdapter(
    pool_connections=_HTTP_POOL_MAXSIZE,
    pool_maxsize=_HTTP_POOL_MAXSIZE,
    max_retries=0,
)
_SESSION.mount("http://", _HTTP_ADAPTER)
_SESSION.mount("https://", _HTTP_ADAPTER)
_HEALTH_CACHE: Dict[str, Tuple[float, bool]] = {}
_PRINTED_BACKEND_CONFIG = False


@dataclass
class Completion:
    text: str
    token_ids: List[str]  # vLLM logprobs.tokens returns token STRINGS; only len() is consumed downstream
    token_logprobs: List[Optional[float]]
    finish_reason: str
    generated_token_ids: Optional[List[int]] = None


def _filter_live_servers(server_urls: List[str]) -> List[str]:
    """Drop endpoints that are already unhealthy before scheduling a rollout."""
    enabled = os.environ.get("EXPDIS_VLLM_FILTER_DEAD_SERVERS", "1").strip().lower()
    if enabled in ("0", "false", "no", "off") or len(server_urls) <= 1:
        return server_urls
    timeout = float(os.environ.get("EXPDIS_VLLM_HEALTH_FILTER_TIMEOUT", "2") or "2")
    ttl = float(os.environ.get("EXPDIS_VLLM_HEALTH_FILTER_TTL_SECONDS", "30") or "30")
    concurrency = max(1, min(len(server_urls), _env_int("EXPDIS_VLLM_HEALTH_FILTER_CONCURRENCY", 64)))

    def _is_live(server_url: str) -> bool:
        now = time.monotonic()
        cached = _HEALTH_CACHE.get(server_url)
        if cached is not None and ttl > 0 and now - cached[0] <= ttl:
            return bool(cached[1])
        base = server_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        try:
            _SESSION.get(base + "/health", timeout=timeout).raise_for_status()
            _HEALTH_CACHE[server_url] = (now, True)
            return True
        except Exception:
            _HEALTH_CACHE[server_url] = (now, False)
            return False

    live_flags: List[bool] = [False] * len(server_urls)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_is_live, url): i for i, url in enumerate(server_urls)}
        for fut in as_completed(futures):
            live_flags[futures[fut]] = bool(fut.result())
    live = [url for url, ok in zip(server_urls, live_flags) if ok]
    if not live:
        raise RuntimeError("No healthy vLLM server URLs passed the pre-rollout health filter")
    if len(live) != len(server_urls):
        print(
            f"[vllm-gen] health-filter kept {len(live)}/{len(server_urls)} server URLs",
            flush=True,
        )
    return live


def _vllm_complete(
    server_url: str,
    prompt: str | List[int],
    n_per_prompt: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    timeout: int,
    api_key: str,
    max_retries: int,
    enable_thinking: str,
    model: str,
    seed: Optional[int] = None,
) -> List[Completion]:
    seed_mode = os.environ.get("EXPDIS_VLLM_SEED_MODE", "api").strip().lower()
    request_prompt = prompt
    if seed is not None and seed_mode in ("prompt_nonce", "whitespace_nonce"):
        raise ValueError("prompt nonces change the sampled policy context; use seed mode api")
    payload = {
        "model": model,
        "prompt": request_prompt,
        "n": n_per_prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": 0.0,
        "return_token_ids": True,
        "stream": False,
    }
    if seed is not None and seed_mode in ("api", "vllm"):
        # TPU vLLM can otherwise initialize concurrent fanout requests with
        # identical RNG state, collapsing a G-way group into duplicate traces.
        payload["seed"] = int(seed)
    if (os.environ.get("EXPDIS_VLLM_RETURN_LOGPROBS", "0") or "").strip().lower() not in ("0", "false", "no", "off"):
        payload["logprobs"] = 1
    # Thinking mode is fixed by the pre-rendered prompt (data.render_chat_prompt);
    # /v1/completions has no chat_template_kwargs, so nothing is sent for it here.
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = server_url.rstrip("/") + "/completions"
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            r = _SESSION.post(url, json=payload, timeout=timeout, headers=headers)
            if (
                r.status_code == 400
                and seed is not None
                and "seed" in payload
                and os.environ.get("EXPDIS_VLLM_SEED_API_FALLBACK", "1").strip().lower()
                not in ("0", "false", "no", "off")
            ):
                payload.pop("seed", None)
                # Keep the exact prompt when a serving build rejects API seeds.
                r = _SESSION.post(url, json=payload, timeout=timeout, headers=headers)
            r.raise_for_status()
            data = r.json()
            completions: List[Completion] = []
            for c in data["choices"]:
                generated_ids = c.get("token_ids")
                if generated_ids is not None and (
                    not isinstance(generated_ids, list)
                    or any(type(token) is not int or token < 0 for token in generated_ids)
                ):
                    raise ValueError("vLLM returned invalid generated token IDs")
                # Token-list prompts identify training calls. Retokenizing text
                # can change the actions whose likelihood is optimized.
                if isinstance(prompt, list) and generated_ids is None and os.environ.get(
                    "EXPDIS_DAPO_DRGRPO_FULL_CONTRACT", "1"
                ) == "1":
                    raise ValueError("paper training requires vLLM return_token_ids support")
                returned_prompt = c.get("prompt_token_ids", data.get("prompt_token_ids"))
                if isinstance(prompt, list) and returned_prompt is not None and returned_prompt != prompt:
                    raise ValueError("vLLM used different prompt tokens from the trainer")
                logps = []
                tok_ids = []
                if c.get("logprobs") and c["logprobs"].get("token_logprobs") is not None:
                    logps = list(c["logprobs"]["token_logprobs"])
                    tok_ids = list(c["logprobs"].get("tokens", []))
                completions.append(Completion(
                    text=c.get("text", ""),
                    token_ids=tok_ids,
                    token_logprobs=logps,
                    finish_reason=c.get("finish_reason", "unknown"),
                    generated_token_ids=generated_ids,
                ))
            if len(completions) != n_per_prompt:
                raise ValueError("vLLM returned a different number of completions than requested")
            return completions
        except Exception as e:
            last_exc = e
            if attempt >= max_retries:
                break
            time.sleep(min(2 ** attempt, 10) + random.random())
    raise RuntimeError(f"vLLM call failed after {max_retries+1} attempts: {last_exc}")


def batched_generate(
    prompts: List[str | List[int]],
    server_urls: List[str],
    n_per_prompt: int,
    max_tokens: int,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 20,
    timeout: int = 600,
    api_key: str = "",
    max_retries: int = 3,
    enable_thinking: str = "auto",
    model: str = "Qwen/Qwen3-1.7B",
    concurrency: int = 16,
    fanout_per_prompt: int = 1,
    allow_error_completions: Optional[bool] = None,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
    seed_base: Optional[int] = None,
) -> List[List[Completion]]:
    """Return per-prompt list of `n_per_prompt` completions.

    Prompts are round-robin distributed across server_urls.
    """
    if not server_urls:
        raise ValueError("Need at least one vLLM server URL")
    server_urls = _filter_live_servers(server_urls)
    if allow_error_completions is None:
        allow_error_completions = (
            os.environ.get("EXPDIS_VLLM_ALLOW_ERROR_COMPLETIONS", "0").strip().lower()
            in ("1", "true", "yes", "on")
        )
    backend = os.environ.get("EXPDIS_ROLLOUT_BACKEND", "adaptive_vllm_http").strip().lower()
    if backend not in ("vllm_http", "adaptive_vllm_http"):
        raise ValueError(
            f"Unsupported EXPDIS_ROLLOUT_BACKEND={backend!r}. "
            "Only adaptive_vllm_http is production-ready in this trainer; "
            "Tunix/SGLang needs separate validated trainer-state integration."
        )
    endpoint_max_inflight = max(1, _env_int("EXPDIS_VLLM_ENDPOINT_MAX_INFLIGHT", 1))
    failover_rounds = max(1, _env_int("EXPDIS_VLLM_FAILOVER_ROUNDS", 2))
    failover_round_sleep = max(0.0, float(os.environ.get("EXPDIS_VLLM_FAILOVER_ROUND_SLEEP", "10") or "10"))
    failover_on_error = (
        os.environ.get("EXPDIS_VLLM_FAILOVER_ON_ERROR", "1").strip().lower()
        not in ("0", "false", "no", "off")
    )
    explicit_seed = seed_base is not None
    randomize_seeds = explicit_seed or (
        os.environ.get("EXPDIS_VLLM_RANDOMIZE_SEEDS", "0").strip().lower()
        not in ("0", "false", "no", "off")
    )
    seed_base_env = os.environ.get("EXPDIS_VLLM_SEED_BASE", "").strip()
    if explicit_seed:
        seed_base = int(seed_base)
    elif seed_base_env:
        try:
            seed_base = int(seed_base_env)
        except ValueError:
            seed_base = abs(hash(seed_base_env))
    else:
        seed_base = (time.time_ns() ^ (os.getpid() << 16)) & 0x7FFFFFFF
    endpoint_limits = {url: Semaphore(endpoint_max_inflight) for url in server_urls}
    global _PRINTED_BACKEND_CONFIG
    if not _PRINTED_BACKEND_CONFIG:
        seed_mode = os.environ.get("EXPDIS_VLLM_SEED_MODE", "api").strip().lower()
        print(
            "[vllm-gen] backend="
            f"{backend} endpoints={len(server_urls)} concurrency={concurrency} "
            f"fanout_per_prompt={fanout_per_prompt} endpoint_max_inflight={endpoint_max_inflight} "
            f"health_ttl={os.environ.get('EXPDIS_VLLM_HEALTH_FILTER_TTL_SECONDS', '30')} "
            f"failover={int(failover_on_error)} failover_rounds={failover_rounds} "
            f"randomize_seeds={int(randomize_seeds)} "
            f"seed_mode={seed_mode if randomize_seeds else 'off'} "
            f"seed_base={seed_base if randomize_seeds else 'off'}",
            flush=True,
        )
        _PRINTED_BACKEND_CONFIG = True

    def _seed_for(prompt_idx: int, split_idx: int, count: int) -> Optional[int]:
        if not randomize_seeds:
            return None
        # Stable if EXPDIS_VLLM_SEED_BASE is provided; distinct across prompts and
        # fanout splits. Keep inside signed int32 because vLLM validates it.
        return int((seed_base + 1_000_003 * prompt_idx + 9_176 * split_idx + 131 * count) % 2_147_483_647)

    def _complete_with_limit(url: str, prompt: str, count: int, seed: Optional[int]) -> List[Completion]:
        candidate_urls = [url]
        if failover_on_error and len(server_urls) > 1:
            start = server_urls.index(url)
            candidate_urls.extend(
                server_urls[(start + offset) % len(server_urls)]
                for offset in range(1, len(server_urls))
            )
        last_exc: Optional[Exception] = None
        for failover_round in range(failover_rounds):
            if failover_round > 0 and failover_round_sleep > 0:
                time.sleep(failover_round_sleep + random.random())
            for candidate in candidate_urls:
                sem = endpoint_limits[candidate]
                with sem:
                    try:
                        return _vllm_complete(
                            candidate,
                            prompt,
                            count,
                            max_tokens,
                            temperature,
                            top_p,
                            top_k,
                            timeout,
                            api_key,
                            max_retries,
                            enable_thinking,
                            model,
                            seed,
                        )
                    except Exception as exc:
                        last_exc = exc
                        _HEALTH_CACHE[candidate] = (time.monotonic(), False)
                        if not failover_on_error:
                            break
            if not failover_on_error:
                break
        raise RuntimeError(f"all candidate vLLM endpoints failed, initial={url}: {last_exc}")

    results: List[Optional[List[Completion]]] = [None] * len(prompts)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {}
        prompt_parts: List[List[List[Completion]]] = [[] for _ in prompts]
        prompt_done_parts = [0] * len(prompts)
        prompt_expected_parts = [0] * len(prompts)
        for i, prompt in enumerate(prompts):
            # Allow repeated use of the same endpoint when serving is serial
            # (for example TPU vLLM with max-num-seqs=1). This keeps the
            # sampling distribution unchanged while avoiding slow multi-sample
            # requests such as n=4 on a single endpoint.
            splits = max(1, min(int(fanout_per_prompt), n_per_prompt))
            counts = [n_per_prompt // splits] * splits
            for j in range(n_per_prompt % splits):
                counts[j] += 1
            counts = [c for c in counts if c > 0]
            prompt_expected_parts[i] = len(counts)
            base = (i * max(1, splits)) % len(server_urls)
            for split_idx, count in enumerate(counts):
                url = server_urls[(base + split_idx) % len(server_urls)]
                fut = pool.submit(
                    _complete_with_limit,
                    url,
                    prompt,
                    count,
                    _seed_for(i, split_idx, count),
                )
                futures[fut] = (i, count)
        done = 0
        for fut in as_completed(futures):
            idx, count = futures[fut]
            try:
                prompt_parts[idx].append(fut.result())
            except Exception as e:
                msg = f"[vllm-gen] ERROR prompt {idx}: {e}"
                if not allow_error_completions:
                    raise RuntimeError(msg) from e
                print(msg, flush=True)
                prompt_parts[idx].append([Completion("", [], [], "error")] * count)
            prompt_done_parts[idx] += 1
            if prompt_done_parts[idx] == prompt_expected_parts[idx]:
                merged: List[Completion] = []
                for part in prompt_parts[idx]:
                    merged.extend(part)
                results[idx] = merged[:n_per_prompt]
            done += 1
            progress_every = max(1, min(20, len(futures) // 4 or 1))
            if done % progress_every == 0 or done == len(futures):
                elapsed = time.perf_counter() - t0
                print(f"[vllm-gen] {done}/{len(futures)} subrequests done ({elapsed:.1f}s elapsed)", flush=True)
                if progress_callback is not None:
                    progress_callback(done, len(futures), elapsed)
    elapsed = time.perf_counter() - t0
    print(f"[vllm-gen] Done: {len(prompts)} prompts x {n_per_prompt} completions in {elapsed:.1f}s", flush=True)
    missing = [i for i, value in enumerate(results) if value is None or len(value) < n_per_prompt]
    if missing:
        raise RuntimeError(f"vLLM generation returned incomplete completions for prompts={missing[:10]}")
    return results  # type: ignore[return-value]


def parse_server_urls(s: str) -> List[str]:
    return [u.strip() for u in s.split(",") if u.strip()]
