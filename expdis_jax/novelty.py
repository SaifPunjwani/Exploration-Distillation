"""Multilayer RND novelty scorer in JAX.

For each completion, the Explorer model is run once and hidden states are
collected at the configured layers (``novelty_layers``; {7, 14, 21} is the
quarter-depth set for the 28-layer Qwen3-1.7B). Each layer's states are
mean-pooled over non-pad tokens into one feature vector, and every layer has
its own frozen target MLP and trained predictor MLP. The per-layer novelty is
``sqrt(mean((target(f) - predictor(f))**2) + 1e-8)`` (``novelty_metric``
"sqrt_mse"), computed before the predictor update; ``train.py`` averages the
per-layer scores across layers and credits the result only on verifier-correct
completions:
    r = r_correct + lambda * r_novel * 1[correct] + r_overlong.
"""

from __future__ import annotations

from functools import partial as _partial
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import optax


def _init_dense_numpy(rng: np.random.Generator, in_dim: int, out_dim: int) -> Dict[str, np.ndarray]:
    limit = np.sqrt(6.0 / float(in_dim + out_dim))
    return {
        "kernel": rng.uniform(-limit, limit, size=(int(in_dim), int(out_dim))).astype(np.float32),
        "bias": np.zeros((int(out_dim),), dtype=np.float32),
    }


def _rnd_block_params_numpy(rng: np.random.Generator, input_dim: int, hidden_dim: int) -> Dict[str, Dict[str, np.ndarray]]:
    return {
        "fc1": _init_dense_numpy(rng, input_dim, hidden_dim),
        "fc2": _init_dense_numpy(rng, hidden_dim, hidden_dim),
        "fc3": _init_dense_numpy(rng, hidden_dim, hidden_dim),
    }


def _zero_like_tree(tree):
    return jax.tree_util.tree_map(lambda x: np.zeros_like(np.asarray(x), dtype=np.float32), tree)



class RNDBlock(nn.Module):
    hidden_dim: int
    out_dim: int | None = None

    @nn.compact
    def __call__(self, x):
        h = nn.Dense(self.hidden_dim, name="fc1")(x)
        h = nn.relu(h)
        h = nn.Dense(self.hidden_dim, name="fc2")(h)
        h = nn.relu(h)
        return nn.Dense(self.out_dim or self.hidden_dim, name="fc3")(h)




def novelty_reward(target_params, predictor_params, target_apply, predictor_apply, features, metric: str):
    t = target_apply({"params": target_params}, features)
    p = predictor_apply({"params": predictor_params}, features)
    diff = p - t
    mse = jnp.mean(diff ** 2, axis=-1)
    metric = (metric or "mse").lower()
    if metric == "l2":
        se = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-8)
    elif metric == "sqrt_mse":
        se = jnp.sqrt(mse + 1e-8)
    elif metric == "l1":
        se = jnp.mean(jnp.abs(diff), axis=-1)
    elif metric == "cosine":
        denom = jnp.linalg.norm(p, axis=-1) * jnp.linalg.norm(t, axis=-1)
        se = 1.0 - (jnp.sum(p * t, axis=-1) / jnp.maximum(denom, 1e-8))
    else:
        se = mse
    return se


def _rnd_loss(target_params, predictor_params, block: RNDBlock, features):
    t = block.apply({"params": target_params}, features)
    p = block.apply({"params": predictor_params}, features)
    return jnp.mean((p - t) ** 2)


@_partial(jax.jit, static_argnames=("hidden_dim", "metric", "lr", "clip_value", "train"))
def _rnd_jitted_step(
    target_params,
    predictor_params,
    opt_state,
    features,
    *,
    hidden_dim: int,
    lr: float,
    metric: str,
    clip_value: float = 0.0,
    train: bool = True,
):
    """JIT-compiled novelty score + (optional) predictor update.

    Hoisted out of `score_and_update_rnd_map` so JAX can cache the trace across
    steps. Previously we re-traced `value_and_grad` over a Python lambda each
    call (one per layer per step) which dominated novelty wall clock.
    """
    block = RNDBlock(hidden_dim=hidden_dim, out_dim=hidden_dim)
    feats_f = features.astype(jnp.float32)
    values = novelty_reward(
        target_params, predictor_params, block.apply, block.apply, feats_f, metric,
    )
    if float(clip_value or 0.0) > 0.0:
        values = jnp.minimum(values, jnp.float32(clip_value))
    if train:
        _, grads = jax.value_and_grad(
            lambda p: _rnd_loss(target_params, p, block, feats_f)
        )(predictor_params)
        tx = optax.adam(float(lr))
        updates, new_opt_state = tx.update(grads, opt_state, predictor_params)
        new_predictor = optax.apply_updates(predictor_params, updates)
        return new_predictor, new_opt_state, values
    return predictor_params, opt_state, values


def init_rnd_map(
    *,
    rng: jax.Array,
    layers: List[int] | Tuple[int, ...],
    input_dim: int,
    hidden_dim: int,
) -> Dict[str, Dict[str, Any]]:
    """Initialize one fixed-target/trainable-predictor RND block per layer.

    The predictor learning rate is supplied per update to
    :func:`score_and_update_rnd_map`, not stored here.
    """
    if isinstance(rng, (int, np.integer)):
        seed = int(rng)
    else:
        # Both legacy uint32 PRNGKey and typed jax.random.key are supported.
        # Converting a typed key directly to NumPy raises; never silently seed 0.
        key_words = np.asarray(jax.random.key_data(rng), dtype=np.uint32).reshape(-1)
        seed = 0
        for word in key_words:
            seed = (seed << 32) | int(word)
    np_rng = np.random.default_rng(seed)
    out: Dict[str, Dict[str, Any]] = {}
    for layer in layers:
        target_params = _rnd_block_params_numpy(np_rng, int(input_dim), int(hidden_dim))
        predictor_params = _rnd_block_params_numpy(np_rng, int(input_dim), int(hidden_dim))
        zeros = _zero_like_tree(predictor_params)
        out[f"layer_{int(layer)}"] = {
            "target": target_params,
            "predictor": predictor_params,
            "opt_state": (
                optax.ScaleByAdamState(
                    count=np.asarray(0, dtype=np.int32),
                    mu=zeros,
                    nu=zeros,
                ),
                optax.EmptyState(),
            ),
        }
    return out



def is_adam_opt_state(opt_state: Any) -> bool:
    """Return whether ``opt_state`` is a well-formed optax Adam state.

    The canonical ``optax.adam`` state is the tuple ``(ScaleByAdamState,
    EmptyState)``; a bare ``ScaleByAdamState`` is also accepted.  Anything else
    (``None``, an empty container, a state without ``count``/``mu``/``nu``) is
    treated as missing and bootstrapped by :func:`score_and_update_rnd_map`.
    """
    head = opt_state
    if isinstance(opt_state, (tuple, list)):
        if not opt_state:
            return False
        head = opt_state[0]
    return all(hasattr(head, attr) for attr in ("count", "mu", "nu"))


def restore_rnd_map(
    rnd_map: Dict[str, Dict[str, Any]],
    restored: Dict[str, Dict[str, Any]] | None,
) -> Dict[str, Dict[str, Any]]:
    if not restored:
        return rnd_map
    out = dict(rnd_map)
    for key, value in restored.items():
        if key in out and isinstance(value, dict):
            current = dict(out[key])
            for leaf in ("target", "predictor", "opt_state"):
                if leaf in value:
                    current[leaf] = value[leaf]
            out[key] = current
    return out


def score_and_update_rnd_map(
    rnd_map: Dict[str, Dict[str, Any]],
    features_by_layer: Dict[str, jnp.ndarray],
    *,
    hidden_dim: int,
    lr: float,
    metric: str,
    clip_value: float = 0.0,
    train: bool = True,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, np.ndarray]]:
    """Return updated RND map and per-layer novelty vectors.

    The novelty vector is computed before the predictor update, matching the
    Torch path: reward comes from the current prediction error, then the
    predictor learns from this batch.
    """
    updated: Dict[str, Dict[str, Any]] = {}
    per_layer: Dict[str, np.ndarray] = {}

    # Bootstrap a missing or malformed opt_state outside the JIT so the JIT only
    # ever sees the canonical optax.adam state (re-init inside JIT would
    # invalidate the trace cache).  A well-formed state is passed through, so
    # the predictor's Adam moments persist across updates within one Explorer;
    # :func:`init_rnd_map` (a new Explorer or round) is what resets them.
    _bootstrap_tx = optax.adam(float(lr))

    for key, item in rnd_map.items():
        feats = features_by_layer.get(key)
        if feats is None:
            updated[key] = item
            continue
        target_params = item["target"]
        predictor_params = item["predictor"]
        opt_state = item["opt_state"]
        if not is_adam_opt_state(opt_state):
            opt_state = _bootstrap_tx.init(predictor_params)
        new_predictor, new_opt_state, values = _rnd_jitted_step(
            target_params,
            predictor_params,
            opt_state,
            feats,
            hidden_dim=int(hidden_dim),
            lr=float(lr),
            metric=str(metric or "mse"),
            clip_value=float(clip_value or 0.0),
            train=bool(train),
        )
        per_layer[key] = np.asarray(values, dtype=np.float32)
        updated[key] = {
            "target": target_params,
            "predictor": new_predictor,
            "opt_state": new_opt_state,
        }
    return updated, per_layer
