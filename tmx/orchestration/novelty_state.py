"""RND (Random Network Distillation) novelty state for Explorer training.

Tunix's reward callback receives decoded strings only — no hidden states. To
compute RND novelty per trajectory we need (a) hidden states from the actor
model and (b) a small online-learned predictor MLP that's compared against a
fixed target MLP. This module owns (b): the RND state lifecycle and the
JIT'd novelty score + predictor update step. Hidden-state extraction lives
in ``novelty_features.py``; the Tunix integration in ``novelty_rollout.py``.

Contract this module guarantees:
- novelty per sample = ||target(features) - predictor(features)||  (sqrt MSE)
- predictor learns to match target online (Adam, lr=1e-4)
- target weights are frozen (initialized once, never updated)
- the same features used for the reward are used for the predictor update,
  so the reward reflects the prediction error BEFORE the gradient step
  (matches the original RND paper and our existing tmx_jax/novelty.py path)

Pure JAX, fully testable on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax


@dataclass(frozen=True)
class NoveltyConfig:
    """Static config for the RND scorer."""

    hidden_dim: int = 512
    layers: tuple[int, ...] = (7, 14, 21)
    metric: str = "sqrt_mse"
    rnd_lr: float = 1e-4
    feature_dim: int = 2048  # Qwen3-1.7B hidden_size; aggregate-over-layers vector size

    def __post_init__(self) -> None:
        if self.metric not in {"sqrt_mse", "mse", "l2", "l1", "cosine"}:
            raise ValueError(f"unknown metric {self.metric!r}")
        if self.feature_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("feature_dim and hidden_dim must be positive")


class _RNDBlock(nn.Module):
    """Three-layer MLP used as both target and predictor."""

    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim, name="fc1")(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, name="fc2")(x)
        x = nn.relu(x)
        return nn.Dense(self.out_dim, name="fc3")(x)


@dataclass
class NoveltyState:
    """Carry state for online RND. Stored across rollout batches."""

    target_params: dict[str, Any] = field(default_factory=dict)
    predictor_params: dict[str, Any] = field(default_factory=dict)
    opt_state: Any = None
    config: NoveltyConfig = field(default_factory=NoveltyConfig)


def _block(cfg: NoveltyConfig) -> _RNDBlock:
    return _RNDBlock(hidden_dim=cfg.hidden_dim, out_dim=cfg.hidden_dim)


def create_novelty_state(rng: jax.Array, cfg: NoveltyConfig) -> NoveltyState:
    """Initialize a fresh NoveltyState. Run once at training start."""
    block = _block(cfg)
    dummy = jnp.zeros((1, cfg.feature_dim), dtype=jnp.float32)
    rng, t_rng, p_rng = jax.random.split(rng, 3)
    target_params = block.init(t_rng, dummy)["params"]
    predictor_params = block.init(p_rng, dummy)["params"]
    tx = optax.adam(cfg.rnd_lr)
    return NoveltyState(
        target_params=target_params,
        predictor_params=predictor_params,
        opt_state=tx.init(predictor_params),
        config=cfg,
    )


def _scoring_metric(diff: jnp.ndarray, metric: str) -> jnp.ndarray:
    if metric == "sqrt_mse":
        return jnp.sqrt(jnp.mean(diff**2, axis=-1) + 1e-8)
    if metric == "mse":
        return jnp.mean(diff**2, axis=-1)
    if metric == "l2":
        return jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-8)
    if metric == "l1":
        return jnp.mean(jnp.abs(diff), axis=-1)
    if metric == "cosine":
        # 1 - cos(p, t)
        denom = jnp.linalg.norm(diff, axis=-1) + 1e-8
        return jnp.linalg.norm(diff, axis=-1) / denom
    raise ValueError(f"unknown metric {metric!r}")


def score(state: NoveltyState, features: jnp.ndarray) -> jnp.ndarray:
    """Compute novelty per sample. Does NOT update the predictor.

    Useful for inspection or eval. For training, prefer ``score_and_update``
    so the reward and the predictor step share a single forward pass.

    features: (B, feature_dim) float32
    returns:  (B,) float32 novelty per sample
    """
    block = _block(state.config)
    t = block.apply({"params": state.target_params}, features)
    p = block.apply({"params": state.predictor_params}, features)
    return _scoring_metric(p - t, state.config.metric)


def score_and_update(
    state: NoveltyState, features: jnp.ndarray
) -> tuple[NoveltyState, jnp.ndarray]:
    """Compute novelty + take an Adam step on the predictor.

    Returns ``(new_state, novelty_per_sample)``. The novelty value is
    computed BEFORE the update so the reward reflects the current model's
    prediction error. The predictor then learns from this batch (online).

    features: (B, feature_dim) float32
    returns:  (NoveltyState, (B,) novelty)
    """
    cfg = state.config
    block = _block(cfg)
    target_params = state.target_params
    tx = optax.adam(cfg.rnd_lr)

    # Score first (no grad through this).
    t = block.apply({"params": target_params}, features)
    p_pre = block.apply({"params": state.predictor_params}, features)
    novelty = _scoring_metric(p_pre - t, cfg.metric)

    # Then gradient step on predictor params to reduce ||p - t||^2.
    def loss_fn(predictor_params: dict[str, Any]) -> jnp.ndarray:
        p = block.apply({"params": predictor_params}, features)
        return jnp.mean((p - t) ** 2)

    grads = jax.grad(loss_fn)(state.predictor_params)
    updates, new_opt = tx.update(grads, state.opt_state, state.predictor_params)
    new_predictor = optax.apply_updates(state.predictor_params, updates)

    new_state = NoveltyState(
        target_params=target_params,
        predictor_params=new_predictor,
        opt_state=new_opt,
        config=cfg,
    )
    return new_state, novelty


def blend_reward(
    correctness: jnp.ndarray,
    novelty: jnp.ndarray,
    *,
    lambda_novelty: float,
    incorrect_novelty_scale: float = 1.0,
) -> jnp.ndarray:
    """Final reward: r = correctness + λ · novelty_used.

    Where ``novelty_used = novelty if correct else incorrect_novelty_scale * novelty``.
    Matches ``tmx_jax/rnd.blend_rewards`` semantics with the
    ``incorrect_novelty_scale`` extension from our existing pipeline.
    """
    correct_mask = correctness > 0.0
    novelty_used = jnp.where(correct_mask, novelty, incorrect_novelty_scale * novelty)
    return correctness + lambda_novelty * novelty_used


def state_to_numpy(state: NoveltyState) -> dict[str, np.ndarray]:
    """Flatten state to a dict of np arrays for HF-mirror persistence."""
    flat: dict[str, np.ndarray] = {}

    def _walk(prefix: str, tree: Any) -> None:
        if hasattr(tree, "items"):
            for k, v in tree.items():
                _walk(f"{prefix}/{k}", v)
        elif isinstance(tree, jnp.ndarray | np.ndarray):
            flat[prefix] = np.asarray(tree)

    _walk("target", state.target_params)
    _walk("predictor", state.predictor_params)
    return flat


__all__ = [
    "NoveltyConfig",
    "NoveltyState",
    "blend_reward",
    "create_novelty_state",
    "score",
    "score_and_update",
    "state_to_numpy",
]
