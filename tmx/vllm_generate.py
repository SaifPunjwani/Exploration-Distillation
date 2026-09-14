"""
vLLM-accelerated generation client for trajectory collection.

Replaces HF model.generate() with HTTP calls to vLLM servers distributed
across TPU hosts, enabling 16x parallel generation on a v6e-64 slice.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import requests


def _post_vllm_completion(
    server_url: str,
    prompt: str,
    model_name: str,
    n: int = 1,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 0,
    max_tokens: int = 1024,
    timeout: int = 600,
    api_key: str = "",
    enable_thinking: Optional[bool] = None,
) -> List[Dict[str, str]]:
    """Send a completion request to a vLLM server and return completions with finish_reason."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model_name,
        "prompt": prompt,
        "n": n,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": False,
        "logprobs": 1,
    }
    if int(top_k) > 0:
        payload["top_k"] = int(top_k)
    if enable_thinking is not None:
        payload["enable_thinking"] = enable_thinking

    resp = requests.post(
        f"{server_url}/completions",
        json=payload,
        headers=headers,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    results = []
    for choice in data["choices"]:
        entry = {
            "text": choice["text"],
            "finish_reason": choice.get("finish_reason", "length"),
        }
        lp = choice.get("logprobs")
        if lp and lp.get("token_logprobs"):
            entry["token_logprobs"] = lp["token_logprobs"]
        results.append(entry)
    return results


def _post_vllm_completion_with_retry(
    server_urls: List[str],
    primary_idx: int,
    prompt: str,
    model_name: str,
    n: int = 1,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 0,
    max_tokens: int = 1024,
    timeout: int = 600,
    api_key: str = "",
    max_retries: Optional[int] = None,
    enable_thinking: Optional[bool] = None,
) -> List[Dict[str, str]]:
    if not server_urls:
        raise RuntimeError("No vLLM servers configured.")

    n_servers = len(server_urls)
    retries = n_servers - 1 if max_retries is None else max(0, int(max_retries))
    order = [primary_idx % n_servers] + [idx for idx in range(n_servers) if idx != (primary_idx % n_servers)]
    errors = []
    for attempt_idx, server_idx in enumerate(order[: retries + 1]):
        server_url = server_urls[server_idx]
        try:
            return _post_vllm_completion(
                server_url=server_url,
                prompt=prompt,
                model_name=model_name,
                n=n,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                timeout=timeout,
                api_key=api_key,
                enable_thinking=enable_thinking,
            )
        except Exception as exc:
            errors.append(f"{server_url}: {exc}")
            if attempt_idx < min(retries, len(order) - 1):
                print(f"[vllm-gen] WARN retrying on alternate server after failure from {server_url}: {exc}")
    raise RuntimeError("; ".join(errors))


def _distribute_prompts(
    prompts: List[dict],
    server_urls: List[str],
) -> List[Tuple[int, str, dict]]:
    """Assign prompts round-robin across servers.
    Returns list of (server_index, server_url, prompt_dict)."""
    assignments = []
    for i, prompt in enumerate(prompts):
        server_idx = i % len(server_urls)
        assignments.append((server_idx, server_urls[server_idx], prompt))
    return assignments


def generate_completions_vllm(
    prompts: List[str],
    server_urls: List[str],
    model_name: str,
    n_per_prompt: int = 4,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 0,
    max_tokens: int = 1024,
    concurrency: int = 64,
    timeout: int = 600,
    api_key: str = "",
    enable_thinking: Optional[bool] = None,
) -> List[List[Dict[str, str]]]:
    """
    Generate completions for all prompts using distributed vLLM servers.

    Args:
        prompts: List of formatted prompt strings
        server_urls: List of vLLM server URLs (e.g. ["http://<vllm-host>:8000/v1", ...])
        model_name: Model name as registered with vLLM
        n_per_prompt: Number of completions per prompt
        temperature: Sampling temperature
        top_p: Top-p sampling
        top_k: Top-k sampling. `0` disables top-k filtering.
        max_tokens: Max completion tokens
        concurrency: Max concurrent requests across all servers
        timeout: Request timeout in seconds

    Returns:
        List of lists of dicts with "text" and "finish_reason", one inner list per prompt
    """
    results = [None] * len(prompts)
    t0 = time.perf_counter()
    # Default to trying all servers (None → n_servers-1) so a single dead server
    # doesn't cause empty completions. Set TMX_VLLM_MAX_RETRIES to limit.
    max_retries_raw = (os.environ.get("TMX_VLLM_MAX_RETRIES") or "").strip()
    max_retries = int(max_retries_raw) if max_retries_raw else None

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {}
        for i, prompt in enumerate(prompts):
            fut = pool.submit(
                _post_vllm_completion_with_retry,
                server_urls=server_urls,
                primary_idx=i % len(server_urls),
                prompt=prompt,
                model_name=model_name,
                n=n_per_prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                timeout=timeout,
                api_key=api_key,
                max_retries=max_retries,
                enable_thinking=enable_thinking,
            )
            futures[fut] = i

        completed = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                completions = fut.result()
                results[idx] = completions
                completed += 1
                if completed % 20 == 0 or completed == len(prompts):
                    elapsed = time.perf_counter() - t0
                    print(
                        f"[vllm-gen] {completed}/{len(prompts)} prompts done "
                        f"({elapsed:.1f}s elapsed, {completed/elapsed:.1f} prompts/s)"
                    )
            except Exception as e:
                print(f"[vllm-gen] ERROR prompt {idx}: {e}")
                results[idx] = [{"text": "", "finish_reason": "error"}] * n_per_prompt

    elapsed = time.perf_counter() - t0
    print(
        f"[vllm-gen] Done: {len(prompts)} prompts × {n_per_prompt} completions "
        f"in {elapsed:.1f}s ({len(prompts) * n_per_prompt / elapsed:.1f} completions/s)"
    )
    return results


def discover_vllm_servers(
    worker_ips: Optional[List[str]] = None,
    port: int = 8000,
    exclude_workers: Optional[List[int]] = None,
    health_check: bool = True,
    timeout: int = 5,
) -> List[str]:
    """
    Discover available vLLM servers from worker IPs.

    If worker_ips is None, reads from TPU metadata.
    Returns list of healthy server URLs.
    """
    if worker_ips is None:
        endpoints_raw = os.environ.get("TMX_VLLM_WORKER_IPS", "").strip()
        if endpoints_raw:
            worker_ips = [ip.strip() for ip in endpoints_raw.split(",") if ip.strip()]
        else:
            try:
                resp = requests.get(
                    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/worker-network-endpoints",
                    headers={"Metadata-Flavor": "Google"},
                    timeout=5,
                )
                raw = resp.text.strip()
                worker_ips = []
                for entry in raw.split(","):
                    parts = entry.strip().split(":")
                    ip = parts[-1] if parts else ""
                    if ip:
                        worker_ips.append(ip)
            except Exception as e:
                print(f"[vllm-discover] Failed to get worker IPs from metadata: {e}")
                return []

    exclude_set = set(exclude_workers or [])
    urls = []
    for i, ip in enumerate(worker_ips):
        if i in exclude_set:
            continue
        url = f"http://{ip}:{port}/v1"
        if health_check:
            try:
                r = requests.get(f"http://{ip}:{port}/health", timeout=timeout)
                if r.status_code == 200:
                    urls.append(url)
                else:
                    print(f"[vllm-discover] Worker {i} ({ip}) unhealthy: {r.status_code}")
            except Exception:
                print(f"[vllm-discover] Worker {i} ({ip}) unreachable")
        else:
            urls.append(url)

    print(f"[vllm-discover] Found {len(urls)}/{len(worker_ips)} healthy vLLM servers")
    return urls


