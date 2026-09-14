from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tmx.orchestration.novelty_state import (
    NoveltyConfig,
    blend_reward,
    create_novelty_state,
    score,
    score_and_update,
    state_to_numpy,
)


def _tiny_cfg(feature_dim: int = 16, hidden_dim: int = 8) -> NoveltyConfig:
    return NoveltyConfig(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        layers=(0,),
        metric="sqrt_mse",
        rnd_lr=1e-3,
    )


def test_create_state_shapes_match_config() -> None:
    cfg = _tiny_cfg(feature_dim=12, hidden_dim=4)
    state = create_novelty_state(jax.random.PRNGKey(0), cfg)
    # fc1: (in=12, out=4); fc2: (in=4, out=4); fc3: (in=4, out=4)
    assert state.target_params["fc1"]["kernel"].shape == (12, 4)
    assert state.target_params["fc3"]["kernel"].shape == (4, 4)
    assert state.predictor_params["fc1"]["kernel"].shape == (12, 4)
    assert state.config.feature_dim == 12


def test_target_and_predictor_start_different() -> None:
    """RND requires target ≠ predictor at init or novelty is identically 0."""
    cfg = _tiny_cfg()
    state = create_novelty_state(jax.random.PRNGKey(7), cfg)
    diff = (
        state.target_params["fc1"]["kernel"] - state.predictor_params["fc1"]["kernel"]
    )
    assert float(jnp.abs(diff).max()) > 0.0


def test_score_returns_one_per_sample() -> None:
    cfg = _tiny_cfg()
    state = create_novelty_state(jax.random.PRNGKey(3), cfg)
    feats = jax.random.normal(jax.random.PRNGKey(4), (5, cfg.feature_dim))
    nov = score(state, feats)
    assert nov.shape == (5,)
    assert jnp.all(nov >= 0.0)


def test_score_and_update_returns_state_and_novelty() -> None:
    cfg = _tiny_cfg()
    state = create_novelty_state(jax.random.PRNGKey(11), cfg)
    feats = jax.random.normal(jax.random.PRNGKey(12), (4, cfg.feature_dim))
    new_state, nov = score_and_update(state, feats)
    assert nov.shape == (4,)
    # opt_state has changed (predictor was updated).
    same_predictor = jnp.allclose(
        state.predictor_params["fc1"]["kernel"],
        new_state.predictor_params["fc1"]["kernel"],
    )
    assert not same_predictor


def test_predictor_update_reduces_loss_on_repeated_features() -> None:
    """Hammering the same features should drive predictor → target."""
    cfg = NoveltyConfig(feature_dim=8, hidden_dim=8, layers=(0,), rnd_lr=5e-3)
    state = create_novelty_state(jax.random.PRNGKey(0), cfg)
    feats = jax.random.normal(jax.random.PRNGKey(1), (16, cfg.feature_dim))
    novelties = []
    for _ in range(60):
        state, nov = score_and_update(state, feats)
        novelties.append(float(nov.mean()))
    # The mean novelty should drop substantially after repeated training.
    assert novelties[-1] < novelties[0] * 0.5, novelties


def test_target_is_frozen_through_updates() -> None:
    cfg = _tiny_cfg()
    state = create_novelty_state(jax.random.PRNGKey(5), cfg)
    target_before = jnp.asarray(state.target_params["fc1"]["kernel"])
    feats = jax.random.normal(jax.random.PRNGKey(6), (4, cfg.feature_dim))
    state, _ = score_and_update(state, feats)
    state, _ = score_and_update(state, feats)
    target_after = jnp.asarray(state.target_params["fc1"]["kernel"])
    assert jnp.array_equal(target_before, target_after)


def test_blend_reward_matches_contract() -> None:
    """r = correctness + λ · novelty_used, where novelty_used scales for incorrect."""
    correctness = jnp.asarray([1.0, -1.0, 1.0, -1.0])
    novelty = jnp.asarray([0.3, 0.4, 0.5, 0.6])
    blended = blend_reward(
        correctness, novelty, lambda_novelty=0.5, incorrect_novelty_scale=1.0
    )
    # With scale=1, novelty_used == novelty. Expected: c + 0.5 * n.
    np.testing.assert_allclose(
        np.asarray(blended), [1.15, -0.80, 1.25, -0.70], rtol=1e-6
    )


def test_blend_reward_with_incorrect_novelty_scale_zero() -> None:
    """incorrect_novelty_scale=0 should zero novelty for incorrect samples."""
    correctness = jnp.asarray([1.0, -1.0])
    novelty = jnp.asarray([0.4, 0.4])
    blended = blend_reward(
        correctness, novelty, lambda_novelty=1.0, incorrect_novelty_scale=0.0
    )
    np.testing.assert_allclose(np.asarray(blended), [1.4, -1.0], rtol=1e-6)


def test_blend_reward_lambda_zero_returns_correctness() -> None:
    correctness = jnp.asarray([1.0, -1.0, 1.0])
    novelty = jnp.asarray([99.0, 99.0, 99.0])
    blended = blend_reward(correctness, novelty, lambda_novelty=0.0)
    np.testing.assert_array_equal(np.asarray(blended), [1.0, -1.0, 1.0])


def test_state_to_numpy_includes_target_and_predictor() -> None:
    cfg = _tiny_cfg()
    state = create_novelty_state(jax.random.PRNGKey(9), cfg)
    flat = state_to_numpy(state)
    assert any(k.startswith("target/fc1") for k in flat)
    assert any(k.startswith("predictor/fc1") for k in flat)
    # The exported params should be plain numpy.
    for v in flat.values():
        assert isinstance(v, np.ndarray)
