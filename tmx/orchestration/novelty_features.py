"""Hidden-state extraction from the actor model for RND novelty scoring.

The Flax Qwen3 model (``tmx_jax/model.py:Qwen3Model``) already exposes a
``return_pooled_layers=(7,14,21), layer_pool='mean'`` mode that returns
mean-pooled hidden states at the requested layers as ``(B, num_layers, H)``.
This module wraps that into a Tunix-rollout-friendly call:

  features = extract_features(actor_apply, actor_params, token_ids, mask)
  -> (B, hidden_size)  float32  (averaged across layers per the contract)

The contract here matches the existing tmx_jax pipeline:
  - layers = (7, 14, 21)
  - pool   = mean over non-pad tokens within each layer
  - agg    = mean across layers → 1 vector per sample

Pure JAX. The actor's `apply` callable + its params are provided by the
custom rollout (``novelty_rollout.py``); we don't import the model here so
this module stays light and unit-testable with a mock apply.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp

# Flax apply() signature for Qwen3Model with return_pooled_layers:
#   model.apply({"params": params}, input_ids, attention_mask,
#                return_pooled_layers=layers, layer_pool="mean")
#   -> (B, num_layers, hidden_size)
ActorApply = Callable[..., jnp.ndarray]


def extract_features(
    actor_apply: ActorApply,
    actor_params: dict[str, Any],
    token_ids: jnp.ndarray,
    attention_mask: jnp.ndarray,
    *,
    layers: tuple[int, ...] = (7, 14, 21),
    layer_pool: str = "mean",
    layer_agg: str = "mean",
) -> jnp.ndarray:
    """Return per-sample feature vector for RND scoring.

    token_ids:        (B, T) int32
    attention_mask:   (B, T) int32  (1 for real tokens, 0 for pad)
    returns:          (B, hidden_size) float32

    The actor model must be the SAME model the GRPO loss is computed on —
    novelty must reflect the current policy, not a stale copy. The Tunix
    custom rollout pulls actor params via ``rl_cluster.actor`` and threads
    them through here.
    """
    if token_ids.ndim != 2:
        raise ValueError(f"token_ids must be (B, T); got shape {token_ids.shape}")
    if attention_mask.shape != token_ids.shape:
        raise ValueError(
            f"attention_mask shape {attention_mask.shape} != token_ids {token_ids.shape}"
        )

    # (B, num_layers, H) — already pooled within each layer by Qwen3Model.
    pooled = actor_apply(
        {"params": actor_params},
        token_ids,
        attention_mask,
        return_pooled_layers=layers,
        layer_pool=layer_pool,
    )

    if pooled.ndim != 3:
        raise ValueError(
            f"actor_apply with return_pooled_layers must return rank-3 "
            f"(B, num_layers, H); got shape {pooled.shape}"
        )

    # Aggregate across layers.
    agg = (layer_agg or "mean").lower()
    if agg == "mean":
        return pooled.mean(axis=1).astype(jnp.float32)
    if agg == "sum":
        return pooled.sum(axis=1).astype(jnp.float32)
    raise ValueError(f"unknown layer_agg {agg!r}")


def pad_token_ids(
    sequences: list[list[int]], pad_id: int = 0, max_len: int | None = None
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Pad a list of variable-length token id lists to a (B, T) array.

    Returns ``(token_ids, attention_mask)`` both as jnp int32 arrays.
    Used by the rollout to assemble prompt+completion token IDs into a
    static-shape batch before forwarding through the actor.
    """
    if not sequences:
        raise ValueError("sequences must be non-empty")
    actual_max = max(len(s) for s in sequences)
    target_len = max_len if max_len is not None else actual_max
    if max_len is not None and actual_max > max_len:
        raise ValueError(
            f"sequence length {actual_max} exceeds max_len={max_len}; "
            "truncate at the rollout layer if needed"
        )

    batch = len(sequences)
    ids = jnp.full((batch, target_len), pad_id, dtype=jnp.int32)
    mask = jnp.zeros((batch, target_len), dtype=jnp.int32)
    for i, seq in enumerate(sequences):
        n = len(seq)
        if n > 0:
            ids = ids.at[i, :n].set(jnp.asarray(seq, dtype=jnp.int32))
            mask = mask.at[i, :n].set(1)
    return ids, mask


__all__ = ["extract_features", "pad_token_ids"]
