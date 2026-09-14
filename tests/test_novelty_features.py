from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tmx.orchestration.novelty_features import extract_features, pad_token_ids


class _FakeActor:
    """Minimal stand-in for Qwen3Model.apply with return_pooled_layers."""

    def __init__(self, hidden_size: int = 4) -> None:
        self.hidden_size = hidden_size
        self.last_call_kwargs: dict[str, object] = {}

    def __call__(
        self,
        variables,
        token_ids,
        attention_mask,
        *,
        return_pooled_layers,
        layer_pool,
    ):
        self.last_call_kwargs = {
            "return_pooled_layers": tuple(return_pooled_layers),
            "layer_pool": layer_pool,
        }
        b = token_ids.shape[0]
        n_layers = len(return_pooled_layers)
        # Make output deterministic from token_ids so tests can assert content.
        x = jnp.asarray(token_ids, dtype=jnp.float32).mean(axis=1, keepdims=True)
        return jnp.broadcast_to(x[..., None], (b, n_layers, self.hidden_size))


def test_extract_features_returns_one_vector_per_sample() -> None:
    actor = _FakeActor(hidden_size=4)
    token_ids = jnp.asarray([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=jnp.int32)
    mask = jnp.asarray([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=jnp.int32)
    feats = extract_features(actor, {}, token_ids, mask, layers=(0, 1, 2))
    assert feats.shape == (2, 4)
    assert feats.dtype == jnp.float32
    assert actor.last_call_kwargs["return_pooled_layers"] == (0, 1, 2)
    assert actor.last_call_kwargs["layer_pool"] == "mean"


def test_extract_features_aggregates_layers_by_mean() -> None:
    actor = _FakeActor(hidden_size=2)
    token_ids = jnp.asarray([[3, 6]], dtype=jnp.int32)  # mean = 4.5
    mask = jnp.ones_like(token_ids)
    feats = extract_features(actor, {}, token_ids, mask, layers=(0, 1, 2, 3))
    np.testing.assert_allclose(np.asarray(feats), [[4.5, 4.5]], rtol=1e-6)


def test_extract_features_rejects_shape_mismatch() -> None:
    actor = _FakeActor()
    token_ids = jnp.zeros((2, 4), dtype=jnp.int32)
    bad_mask = jnp.zeros((2, 3), dtype=jnp.int32)
    with pytest.raises(ValueError, match="attention_mask"):
        extract_features(actor, {}, token_ids, bad_mask)


def test_pad_token_ids_pads_and_masks() -> None:
    seqs = [[1, 2, 3], [4, 5], [6]]
    ids, mask = pad_token_ids(seqs, pad_id=0)
    np.testing.assert_array_equal(
        np.asarray(ids),
        [[1, 2, 3], [4, 5, 0], [6, 0, 0]],
    )
    np.testing.assert_array_equal(
        np.asarray(mask),
        [[1, 1, 1], [1, 1, 0], [1, 0, 0]],
    )


def test_pad_token_ids_uses_explicit_max_len() -> None:
    seqs = [[1, 2], [3]]
    ids, mask = pad_token_ids(seqs, pad_id=99, max_len=5)
    assert ids.shape == (2, 5)
    assert int(ids[0, 4]) == 99
    assert int(mask[0, 4]) == 0


def test_pad_token_ids_rejects_truncation_required() -> None:
    with pytest.raises(ValueError, match="exceeds max_len"):
        pad_token_ids([[1, 2, 3, 4]], max_len=2)


def test_pad_token_ids_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        pad_token_ids([])
