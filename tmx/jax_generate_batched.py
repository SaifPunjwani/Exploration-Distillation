"""Batched JAX-native generation for Qwen3 — drop-in replacement for HTTP vLLM.

Architecture choice: pad-to-max-seq-len approach (no KV cache).
  - JIT-compiles model.apply ONCE for the fixed shape (B, max_seq_len)
  - Each decode step is one full forward pass on the padded sequence
  - Attention mask grows by one position per step (1s up to current len, 0s after)
  - Mathematically identical to autoregressive generation; just slower than
    KV-cached decode

Why this design:
  - Reuses tmx_jax.model.Qwen3Model unchanged → no new trainable params,
    no risk of breaking the trained checkpoint
  - Works inside multi-host JAX trainer process (no second libtpu user)
  - Single JIT compile, then fast per-step calls on-device
  - input_ids/attn_mask never round-trip to host — updated in-place via
    dynamic_update_slice; only the new token id (small) is pulled to host
    for EOS detection

Estimated speed (1.7B model, 256 chips, B=256, max_seq_len=18432):
  - Single forward: ~100-200ms (memory-bandwidth bound)
  - 16384 decode tokens × 150ms ≈ 40 min per rollout
  - Acceptable for the contract (codex's vLLM-served pattern is ~5 min/rollout
    but uses a separate 64-chip pod; we use ALL 256 chips for both paths)

API: `batched_generate_jax_batched(...)` returns List[List[Completion]]
matching tmx_jax.generate.Completion shape.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import jax
import jax.numpy as jnp
import numpy as np

@dataclass
class Completion:
    text: str
    token_ids: List[int]
    token_logprobs: List[float]
    finish_reason: str


# Default Qwen3 EOS token IDs (from Qwen3 chat template).
DEFAULT_EOS_IDS = (151645, 151643)  # <|im_end|>, <|endoftext|>


def _sample_top_p(rng, logits, temperature: float, top_p: float, top_k: int):
    """Top-p (nucleus) sampling with temperature and top-k cap. Returns (B,)."""
    if temperature <= 0:
        return jnp.argmax(logits, axis=-1).astype(jnp.int32)
    scaled = logits.astype(jnp.float32) / temperature
    if top_k and top_k > 0:
        topk_vals, _ = jax.lax.top_k(scaled, k=top_k)
        cutoff = topk_vals[..., -1:]
        scaled = jnp.where(scaled >= cutoff, scaled, jnp.full_like(scaled, -1e9))
    probs = jax.nn.softmax(scaled, axis=-1)
    sorted_probs, sorted_idx = jax.lax.top_k(probs, k=probs.shape[-1])
    cum = jnp.cumsum(sorted_probs, axis=-1)
    keep = cum - sorted_probs <= top_p
    keep = keep.at[..., 0].set(True)
    sorted_probs = jnp.where(keep, sorted_probs, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(axis=-1, keepdims=True).clip(min=1e-12)
    sub_logits = jnp.log(sorted_probs.clip(min=1e-12))
    sampled_pos = jax.random.categorical(rng, sub_logits, axis=-1)
    sampled_tok = jnp.take_along_axis(sorted_idx, sampled_pos[..., None], axis=-1).squeeze(-1)
    return sampled_tok.astype(jnp.int32)


def _make_decode_step(model_apply):
    """Build a JIT'd decode step.

    CRITICAL: avoids materializing full (B, T, V) logits tensor (which would be
    256 × 18432 × 151936 × 4 = 2.86TB and overflow XLA's int32 buffer limit).
    Instead: use return_hidden=True to get hidden states (B, T, hidden), slice
    to last position only (B, 1, hidden), then matmul with embed.T to get
    just (B, 1, V) = 156MB. 18432x cheaper memory footprint at lm_head.

    Inputs:
      params         : Flax params pytree (must contain embed_tokens for tied lm_head)
      input_ids      : (B, max_seq_len) int32 — full padded sequence on device
      attn_mask      : (B, max_seq_len) int32 — 1s up to current pos
      cur_len        : scalar int — position cur_len-1 holds the prediction; write at cur_len
      rng            : PRNGKey
      temperature, top_p, top_k : sampling params (static for compile)
      finished_mask  : (B,) bool — sequences already done

    Returns:
      next_tok, next_logp, new_input_ids, new_attn_mask
    """
    def step(params, input_ids, attn_mask, cur_len, rng,
             temperature, top_p, top_k, finished_mask):
        # return_hidden=True yields (B, T, hidden) instead of (B, T, V) — saves
        # the full-sequence lm_head matmul.
        hidden = model_apply({"params": params}, input_ids, attn_mask, return_hidden=True)
        # Slice the LAST hidden state across all sequences: (B, 1, hidden)
        last_h = jax.lax.dynamic_slice_in_dim(hidden, cur_len - 1, 1, axis=1)
        # Apply tied lm_head: logits = last_h @ embed.T → (B, 1, V)
        embed_w = params["embed_tokens"]["embedding"]  # (V, hidden)
        next_logits = (last_h @ embed_w.T).squeeze(1)  # (B, V)

        next_tok = _sample_top_p(rng, next_logits, temperature, top_p, top_k)
        logp_full = jax.nn.log_softmax(next_logits.astype(jnp.float32), axis=-1)
        next_logp = jnp.take_along_axis(logp_full, next_tok[:, None], axis=-1).squeeze(-1)

        # Mask finished sequences (write 0 at cur_len, no attn bump)
        next_tok_write = jnp.where(finished_mask, jnp.int32(0), next_tok)
        ones = jnp.ones((input_ids.shape[0], 1), dtype=attn_mask.dtype)
        attn_inc = jnp.where(finished_mask[:, None], jnp.int32(0), ones).astype(attn_mask.dtype)
        new_input_ids = jax.lax.dynamic_update_slice(input_ids, next_tok_write[:, None], (0, cur_len))
        new_attn_mask = jax.lax.dynamic_update_slice(attn_mask, attn_inc, (0, cur_len))
        return next_tok, next_logp, new_input_ids, new_attn_mask

    return jax.jit(step, static_argnames=("temperature", "top_p", "top_k"))


def batched_generate_jax_batched(
    model,
    params,
    tokenizer,
    prompts: List[str],
    n_per_prompt: int,
    max_new_tokens: int = 16384,
    max_prompt_len: int = 2048,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 20,
    eos_ids=DEFAULT_EOS_IDS,
    rng_seed: int = 0,
    log_every: int = 256,
    mesh=None,
) -> List[List[Completion]]:
    """Batched JAX generation. Pads prompts to max_prompt_len, decodes up to
    max_new_tokens (or EOS) on a fixed (B, max_seq_len) shape.

    Returns nested list shape (n_prompts, n_per_prompt) of Completion.
    """
    n_prompts = len(prompts)
    B = n_prompts * n_per_prompt
    max_seq_len = max_prompt_len + max_new_tokens

    encoded = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=False,
        truncation=True,
        max_length=max_prompt_len,
        return_tensors=None,
    )
    raw_ids = encoded["input_ids"]
    if not isinstance(raw_ids, list):
        raw_ids = raw_ids.tolist()

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    input_ids = np.full((B, max_seq_len), pad_id, dtype=np.int32)
    attn_mask = np.zeros((B, max_seq_len), dtype=np.int32)
    prompt_lens = np.zeros(B, dtype=np.int32)
    for i, ids in enumerate(raw_ids):
        ids = ids[:max_prompt_len]
        for k in range(n_per_prompt):
            row = i * n_per_prompt + k
            L = len(ids)
            input_ids[row, :L] = np.asarray(ids, dtype=np.int32)
            attn_mask[row, :L] = 1
            prompt_lens[row] = L

    # Initial transfer to device — shard along batch dim across the FSDP mesh
    # so per-chip hidden state stays within HBM. Without this, B=256, T=18432,
    # hidden=2048 in bf16 is 19.3GB per chip → OOM.
    if mesh is not None:
        from jax.sharding import NamedSharding, PartitionSpec as _P
        # Use the FSDP axis name from the mesh (single-axis mesh expected).
        fsdp_axis = mesh.axis_names[0] if mesh.axis_names else None
        if fsdp_axis is not None:
            shard_spec = NamedSharding(mesh, _P(fsdp_axis, None))
            input_ids_jax = jax.device_put(input_ids, shard_spec)
            attn_mask_jax = jax.device_put(attn_mask, shard_spec)
        else:
            input_ids_jax = jnp.asarray(input_ids)
            attn_mask_jax = jnp.asarray(attn_mask)
    else:
        input_ids_jax = jnp.asarray(input_ids)
        attn_mask_jax = jnp.asarray(attn_mask)

    decode_step = _make_decode_step(model.apply)

    finished = np.zeros(B, dtype=bool)
    finish_reason = ["length"] * B
    completion_ids: List[List[int]] = [[] for _ in range(B)]
    completion_logps: List[List[float]] = [[] for _ in range(B)]

    rng_master = jax.random.PRNGKey(rng_seed)
    eos_set = set(int(e) for e in eos_ids)

    t0 = time.perf_counter()
    cur_len = int(prompt_lens.max())  # all sequences advance from the longest prompt

    for step in range(max_new_tokens):
        if finished.all():
            break
        if cur_len >= max_seq_len:
            for b in range(B):
                if not finished[b]:
                    finish_reason[b] = "length"
                    finished[b] = True
            break

        rng_master, sub = jax.random.split(rng_master)
        finished_mask_jax = jnp.asarray(finished)
        next_tok, next_logp, input_ids_jax, attn_mask_jax = decode_step(
            params, input_ids_jax, attn_mask_jax,
            cur_len, sub,
            float(temperature), float(top_p), int(top_k),
            finished_mask_jax,
        )
        # Pull only the (B,) sized arrays back to host
        next_tok_np = np.asarray(next_tok)
        next_logp_np = np.asarray(next_logp)

        for b in range(B):
            if finished[b]:
                continue
            tok = int(next_tok_np[b])
            completion_ids[b].append(tok)
            completion_logps[b].append(float(next_logp_np[b]))
            if tok in eos_set:
                finished[b] = True
                finish_reason[b] = "stop"
        cur_len += 1

        if log_every and (step + 1) % log_every == 0:
            elapsed = time.perf_counter() - t0
            done = int(finished.sum())
            print(f"[jax-gen] step {step+1}/{max_new_tokens} done={done}/{B} cur_len={cur_len} elapsed={elapsed:.1f}s", flush=True)

    # Decode token ids to text per (prompt, sample)
    results: List[List[Completion]] = []
    for i in range(n_prompts):
        per_prompt: List[Completion] = []
        for k in range(n_per_prompt):
            row = i * n_per_prompt + k
            tids = completion_ids[row]
            text = tokenizer.decode(tids, skip_special_tokens=True)
            per_prompt.append(Completion(
                text=text,
                token_ids=tids,
                token_logprobs=completion_logps[row],
                finish_reason=finish_reason[row],
            ))
        results.append(per_prompt)
    elapsed = time.perf_counter() - t0
    print(f"[jax-gen] DONE: {n_prompts} prompts × {n_per_prompt} samples in {elapsed:.1f}s", flush=True)
    return results


__all__ = ["Completion", "batched_generate_jax_batched"]
