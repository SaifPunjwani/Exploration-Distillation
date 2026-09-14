"""DR-GRPO loss in JAX.

Reference: `tmx/explorer.py::_train_explorer_from_rollout_groups` and the
DR-GRPO paper. Ported to pure JAX so the gradient path stays inside jit.

Shape conventions (all JAX arrays):
  B:  batch size * num_generations (= "rows")
  T:  completion length (masked where completion_mask==0)
  V:  vocab

Required inputs (per optimizer step):
  logits           : (B, T, V)    current-policy logits
  completion_ids   : (B, T)       tokens
  completion_mask  : (B, T)       1 where token is real
  old_per_token_logps: (B, T)     logp under rollout policy (ideally rescored)
  advantages       : (B,)         per-sample advantage (reward - group_mean)
  clip_range       : float        PPO clip epsilon (default 0.2)

DR-GRPO normalization:
  loss = -sum_over_B_T(mask * min(ratio*adv, clip(ratio,1-eps,1+eps)*adv))
         / (max_completion_len)     <-- DR-GRPO specifically divides by the
                                         fixed max completion length (not the
                                         actual non-pad token count).

Notes on correctness:
  * `selective_log_softmax` avoids materializing full V-wide softmax; important
    for V=151936 (Qwen3) at T=8192.
  * `ratio = exp(new_lp - old_lp)`; stale `old_lp` blows ratio up. We recompute
    old_lp on current training model each round to avoid this (TMX_SKIP_VLLM_LOGPROBS=1 equivalent).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array


@dataclass(frozen=True)
class GrpoConfig:
    clip_epsilon: float = 0.2
    clip_epsilon_high: float | None = None
    max_completion_len: int = 8192   # DR-GRPO divides by this fixed scalar
    kl_beta: float = 0.0             # KL to reference model; 0 = off
    entropy_coef: float = 0.0        # Small entropy bonus; 0 = off
    # Loss aggregation:
    #   "dr_grpo"                  -> sum / (B * max_completion_len)  [Liu 2025]
    #   "sequence-mean-token-mean" -> mean_b(mean_t(loss[mask]))      [vanilla GRPO; matches tunix/Shao 2024]
    #   "token-mean"               -> sum / mask.sum()                [DAPO Eq. 12 token-level]
    loss_aggregation: str = "dr_grpo"


def selective_log_softmax(logits: Array, targets: Array) -> Array:
    """Compute log-prob of `targets` under softmax(logits) without materializing vocab.

    logits:  (..., V)
    targets: (...)
    returns: (...)
    """
    # stabilized: log_softmax then gather
    lse = jax.nn.logsumexp(logits, axis=-1)
    target_logits = jnp.take_along_axis(logits, targets[..., None], axis=-1).squeeze(-1)
    return target_logits - lse


def compute_per_token_logps(logits: Array, completion_ids: Array) -> Array:
    """Per-token logp of the generated completion.

    logits:          (B, T, V) — shifted to align with completion tokens, so
                                  logits[:, t, :] is the distribution over token
                                  completion_ids[:, t].
    completion_ids:  (B, T)
    returns:         (B, T)
    """
    return selective_log_softmax(logits, completion_ids)


def chunked_per_token_logps_from_hidden(
    hidden: Array,
    target_ids: Array,
    lm_head_kernel: Array,
    chunk_tokens: int = 256,
) -> Array:
    """Compute selected-token logps from hidden states without materializing full-sequence logits.

    hidden:         (B, T, H)
    target_ids:     (B, T)
    lm_head_kernel: (H, V)
    returns:        (B, T)
    """
    if lm_head_kernel.ndim != 2:
        raise ValueError(f"lm_head_kernel must be rank-2, got shape {lm_head_kernel.shape}")
    if lm_head_kernel.shape[0] != hidden.shape[-1]:
        if lm_head_kernel.shape[1] == hidden.shape[-1]:
            lm_head_kernel = lm_head_kernel.T
        else:
            raise ValueError(
                f"Hidden size {hidden.shape[-1]} incompatible with lm_head kernel shape {lm_head_kernel.shape}"
            )

    step = max(1, int(chunk_tokens))
    orig_t = hidden.shape[1]
    pad_t = (-orig_t) % step
    if pad_t:
        hidden = jnp.pad(hidden, ((0, 0), (0, pad_t), (0, 0)))
        target_ids = jnp.pad(target_ids, ((0, 0), (0, pad_t)))

    kernel = lm_head_kernel.astype(jnp.float32)
    bsz, total_t, hid = hidden.shape
    num_chunks = total_t // step
    hidden_chunks = hidden.reshape(bsz, num_chunks, step, hid).transpose(1, 0, 2, 3)
    target_chunks = target_ids.reshape(bsz, num_chunks, step).transpose(1, 0, 2)

    def body(_, xs):
        h, y = xs
        logits = jnp.einsum("bth,hv->btv", h.astype(jnp.float32), kernel)
        return None, selective_log_softmax(logits, y)

    _, logp_chunks = jax.lax.scan(body, None, (hidden_chunks, target_chunks))
    out = logp_chunks.transpose(1, 0, 2).reshape(bsz, total_t)
    return out[:, :orig_t]


def dr_grpo_loss_from_logps(
    new_per_token_logps: Array,
    completion_mask: Array,
    old_per_token_logps: Array,
    advantages: Array,
    cfg: GrpoConfig,
    ref_per_token_logps: Array | None = None,
) -> Tuple[Array, dict]:
    """DR-GRPO loss + metrics from already-computed selected token logprobs."""
    eps_low = cfg.clip_epsilon
    eps_high = cfg.clip_epsilon if cfg.clip_epsilon_high is None else cfg.clip_epsilon_high
    new_lp = new_per_token_logps

    log_ratio = new_lp - old_per_token_logps
    ratio = jnp.exp(log_ratio)

    adv = advantages[:, None]
    unclipped = ratio * adv
    clipped = jnp.clip(ratio, 1.0 - eps_low, 1.0 + eps_high) * adv
    token_obj = jnp.minimum(unclipped, clipped)

    mask = completion_mask.astype(token_obj.dtype)
    agg = (cfg.loss_aggregation or "dr_grpo").lower()
    if agg in ("sequence-mean-token-mean", "seq-mean-token-mean", "vanilla"):
        # Vanilla GRPO (Shao 2024 / tunix reference): mean_b(mean_t(loss[mask]))
        non_zero_rows = jnp.maximum((mask.sum(axis=1) > 0).sum(), 1)
        per_seq_tokens = jnp.maximum(mask.sum(axis=1), 1.0)
        per_seq_loss = -(token_obj * mask).sum(axis=1) / per_seq_tokens
        loss = per_seq_loss.sum() / non_zero_rows
    elif agg in ("token-mean", "dapo"):
        # DAPO Eq. 12 token-level: sum over all valid tokens / total valid tokens
        loss = -(token_obj * mask).sum() / jnp.maximum(mask.sum(), 1.0)
    else:
        # dr_grpo (Liu 2025): fixed denominator B * max_completion_len
        loss = -(token_obj * mask).sum() / (mask.shape[0] * cfg.max_completion_len)

    kl_term = jnp.zeros((), dtype=loss.dtype)
    ref_kl_mean = jnp.zeros((), dtype=loss.dtype)
    if ref_per_token_logps is not None and cfg.kl_beta > 0:
        ref_delta = ref_per_token_logps - new_lp
        kl_per_token = (jnp.exp(ref_delta) - ref_delta - 1.0) * mask
        ref_kl_mean = kl_per_token.sum() / jnp.maximum(mask.sum(), 1.0)
        # KL aggregation matches the loss aggregation so they're on the same scale.
        if agg in ("sequence-mean-token-mean", "seq-mean-token-mean", "vanilla"):
            non_zero_rows = jnp.maximum((mask.sum(axis=1) > 0).sum(), 1)
            per_seq_tokens = jnp.maximum(mask.sum(axis=1), 1.0)
            kl_term = cfg.kl_beta * (kl_per_token.sum(axis=1) / per_seq_tokens).sum() / non_zero_rows
        elif agg in ("token-mean", "dapo"):
            kl_term = cfg.kl_beta * kl_per_token.sum() / jnp.maximum(mask.sum(), 1.0)
        else:
            kl_term = cfg.kl_beta * kl_per_token.sum() / (mask.shape[0] * cfg.max_completion_len)
        loss = loss + kl_term

    low_clip = ratio < (1.0 - eps_low)
    high_clip = ratio > (1.0 + eps_high)
    clip_frac = ((low_clip | high_clip) * mask).sum() / jnp.maximum(mask.sum(), 1.0)
    up_clip_frac = (high_clip * mask).sum() / jnp.maximum(mask.sum(), 1.0)
    up_clip_denom = jnp.maximum((high_clip * mask).sum(), 1.0)
    up_clip_prob = (jnp.exp(new_lp) * high_clip * mask).sum() / up_clip_denom
    approx_kl = ((jnp.exp(log_ratio) - 1.0 - log_ratio) * mask).sum() / jnp.maximum(mask.sum(), 1.0)
    mean_ratio = (ratio * mask).sum() / jnp.maximum(mask.sum(), 1.0)
    abs_gap = jnp.abs(new_lp - old_per_token_logps) * mask
    gap_denom = jnp.maximum(mask.sum(), 1.0)
    mean_logprob = (new_lp * mask).sum() / gap_denom
    mean_probability = (jnp.exp(new_lp) * mask).sum() / gap_denom

    metrics = {
        "loss": loss,
        "kl_term": kl_term,
        "ref_kl": ref_kl_mean,
        "clip_fraction": clip_frac,
        "up_clip_fraction": up_clip_frac,
        "up_clipped_probability_mean": up_clip_prob,
        "approx_kl": approx_kl,
        "mean_ratio": mean_ratio,
        "old_new_logprob_abs_mean": abs_gap.sum() / gap_denom,
        "old_new_logprob_abs_max": abs_gap.max(),
        "selected_token_logprob_mean": mean_logprob,
        "selected_token_probability_mean": mean_probability,
        "adv_mean": advantages.mean(),
        "adv_std": advantages.std(),
    }
    return loss, metrics


def dr_grpo_loss(
    logits: Array,
    completion_ids: Array,
    completion_mask: Array,
    old_per_token_logps: Array,
    advantages: Array,
    cfg: GrpoConfig,
    ref_per_token_logps: Array | None = None,
) -> Tuple[Array, dict]:
    """DR-GRPO loss + metrics.

    Returns (scalar_loss, metrics_dict).
    """
    new_lp = compute_per_token_logps(logits, completion_ids)  # (B, T)
    return dr_grpo_loss_from_logps(
        new_lp, completion_mask, old_per_token_logps, advantages, cfg, ref_per_token_logps=ref_per_token_logps
    )


def compute_group_advantages(
    rewards: Array,
    num_generations: int,
    normalization: str = "none",
    eps: float = 1e-4,
) -> Array:
    """Group-relative advantages for GRPO/DAPO.

    rewards: (B,)  where B = num_prompts * num_generations
    returns: (B,)  advantages = reward - group_mean, optionally divided by group std.
    """
    assert rewards.shape[0] % num_generations == 0, "rewards length must be divisible by num_generations"
    grouped = rewards.reshape(-1, num_generations)  # (G, N)
    group_mean = grouped.mean(axis=1, keepdims=True)
    adv = grouped - group_mean
    if str(normalization or "none").lower() in ("std", "zscore", "group_std", "grpo"):
        # Match Tunix/vanilla GRPO: sample std (ddof=1) and additive epsilon.
        group_std = grouped.std(axis=1, keepdims=True, ddof=1)
        adv = adv / (group_std + eps)
    return adv.reshape(-1)
