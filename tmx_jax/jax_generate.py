"""JAX-native generation using the Flax Qwen3 model.

Slower than vLLM (no paged attention / continuous batching) but fully
self-contained — training can run end-to-end on one TPU worker.

Returns an API compatible with `tmx_jax.generate.Completion` so the
training loop can be swapped without code changes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import partial
from typing import List

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class Completion:
    text: str
    token_ids: List[int]
    token_logprobs: List[float]
    finish_reason: str


def _categorical(rng, logits, temperature: float, top_p: float):
    """Top-p (nucleus) sampling with temperature. Returns sampled token id (int32)."""
    if temperature <= 0:
        return jnp.argmax(logits, axis=-1).astype(jnp.int32)
    scaled = logits / temperature
    # nucleus via sort-prefix mask
    probs = jax.nn.softmax(scaled, axis=-1)
    sorted_probs, sorted_idx = jax.lax.top_k(probs, k=probs.shape[-1])
    cum = jnp.cumsum(sorted_probs, axis=-1)
    # Keep tokens whose cumulative prob <= top_p; always keep first.
    mask = cum - sorted_probs <= top_p
    mask = mask.at[..., 0].set(True)
    masked_probs = jnp.where(mask, sorted_probs, 0.0)
    masked_probs = masked_probs / masked_probs.sum(axis=-1, keepdims=True)
    # sample
    draw = jax.random.choice(rng, sorted_probs.shape[-1], p=masked_probs)
    return sorted_idx[..., draw].astype(jnp.int32)


def generate(
    model,
    params,
    tokenizer,
    prompt_ids,  # (P,) int32 np array
    max_new_tokens: int = 8192,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 20,
    eos_ids: List[int] = (151645, 151643),
    rng_seed: int = 0,
) -> Completion:
    """Simple greedy-decoding style generate (one sample, autoregressive).

    For batched N-per-prompt sampling call this N times with different seeds.
    No KV cache for now; runs full forward pass each step (slow but correct).
    """
    rng = jax.random.PRNGKey(rng_seed)
    tokens = list(prompt_ids.tolist())
    logps: List[float] = []
    P = len(tokens)

    def logits_fn(toks):
        ids = jnp.asarray([toks], dtype=jnp.int32)
        mask = jnp.ones_like(ids, dtype=jnp.int32)
        return model.apply({"params": params}, ids, mask)

    # JIT the single-step forward (compiles once on first call, then fast).
    logits_jit = jax.jit(logits_fn)

    finish_reason = "length"
    for step in range(max_new_tokens):
        logits = logits_jit(tokens)
        next_logits = logits[0, -1]   # (V,)
        rng, sub = jax.random.split(rng)
        next_tok = int(_categorical(sub, next_logits, temperature, top_p))
        # logp of chosen token
        lp = float(jax.nn.log_softmax(next_logits.astype(jnp.float32))[next_tok])
        tokens.append(next_tok)
        logps.append(lp)
        if next_tok in eos_ids:
            finish_reason = "stop"
            break

    comp_ids = tokens[P:]
    text = tokenizer.decode(comp_ids, skip_special_tokens=True)
    return Completion(text=text, token_ids=comp_ids, token_logprobs=logps, finish_reason=finish_reason)


def batched_generate_jax(
    model,
    params,
    tokenizer,
    prompts: List[str],
    n_per_prompt: int,
    max_new_tokens: int = 8192,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 20,
    rng_seed: int = 0,
) -> List[List[Completion]]:
    """Host-side loop over prompts × n_per_prompt.

    Each individual generation is JIT'd per-token forward. Slow but works on
    any TPU worker without vLLM infra.
    """
    results: List[List[Completion]] = []
    t0 = time.perf_counter()
    for i, prompt in enumerate(prompts):
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        per_prompt: List[Completion] = []
        for k in range(n_per_prompt):
            c = generate(
                model, params, tokenizer,
                np.asarray(prompt_ids, dtype=np.int32),
                max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p, top_k=top_k,
                rng_seed=rng_seed + i * n_per_prompt + k,
            )
            per_prompt.append(c)
        results.append(per_prompt)
        elapsed = time.perf_counter() - t0
        print(f"[jax-gen] {i+1}/{len(prompts)} prompts done ({elapsed:.1f}s)", flush=True)
    return results
