"""kNN and elliptical novelty bonuses (paper Appendix C.1)."""
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from expdis_jax import model, train
from expdis_jax.config import TrainConfig, validate_contract
from expdis_jax.generate import Completion
from expdis_jax.novelty import (
    elliptical_novelty,
    elliptical_update,
    init_elliptical_state,
    init_feature_novelty_state,
    init_knn_state,
    knn_append,
    knn_buffer_contents,
    knn_novelty,
    l2_normalize,
)


def test_knn_hand_computed_scores():
    buffer = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    query = np.array([[1.0, 0.0], [0.0, -1.0]])
    # Cosine distances: row 0 -> (0, 1, 2); row 1 -> (1, 2, 1).
    np.testing.assert_allclose(knn_novelty(query, buffer, k=1), [0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(knn_novelty(query, buffer, k=2), [0.5, 1.0], atol=1e-6)
    # Fewer rows than k: use every available neighbor.
    np.testing.assert_allclose(knn_novelty(query, buffer, k=16), [1.0, 4.0 / 3.0], atol=1e-6)


def test_knn_empty_buffer_gives_zero_bonus():
    state = init_knn_state(input_dim=3, buffer_size=4)
    assert knn_buffer_contents(state).shape == (0, 3)
    np.testing.assert_array_equal(knn_novelty(np.ones((2, 3)), knn_buffer_contents(state), k=16), [0.0, 0.0])


def test_knn_uses_cosine_distance_on_normalized_features():
    rng = np.random.default_rng(0)
    buffer, query = rng.normal(size=(6, 5)), rng.normal(size=(3, 5))
    scaled = knn_novelty(query * 7.0, buffer * np.arange(1, 7)[:, None], k=2)
    np.testing.assert_allclose(scaled, knn_novelty(query, buffer, k=2), atol=1e-6)
    q, b = l2_normalize(query), l2_normalize(buffer)
    expected = np.sort(1.0 - q @ b.T, axis=1)[:, :2].mean(axis=1)
    np.testing.assert_allclose(knn_novelty(query, buffer, k=2), expected, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(l2_normalize(query), axis=1), 1.0)


def _rows_as_tuples(state):
    return sorted(tuple(np.round(row, 6)) for row in knn_buffer_contents(state))


def test_knn_buffer_is_fifo_with_cap():
    basis = np.eye(5, dtype=np.float32)
    state = init_knn_state(input_dim=5, buffer_size=3)
    state = knn_append(state, basis[:2])
    assert int(state["size"]) == 2
    state = knn_append(state, basis[2:4])
    assert int(state["size"]) == 3
    assert _rows_as_tuples(state) == sorted(tuple(r) for r in basis[1:4])
    state = knn_append(state, basis[4:5])
    assert _rows_as_tuples(state) == sorted(tuple(r) for r in basis[2:5])
    # A batch larger than the buffer keeps only its most recent rows.
    big = init_knn_state(input_dim=5, buffer_size=3)
    big = knn_append(knn_append(big, basis[:1]), basis)
    assert int(big["size"]) == 3
    assert _rows_as_tuples(big) == sorted(tuple(r) for r in basis[2:5])
    # Stored rows are normalized.
    scaled = knn_append(init_knn_state(input_dim=5, buffer_size=3), 4.0 * basis[:1])
    np.testing.assert_allclose(knn_buffer_contents(scaled), basis[:1])


def test_elliptical_bonus_matches_direct_inverse():
    rng = np.random.default_rng(1)
    ridge = 0.5
    state = init_elliptical_state(input_dim=4, ridge=ridge)
    query = rng.normal(size=(3, 4))
    # Empty history: sqrt(phi^T phi / ridge) = 1 / sqrt(ridge) for unit phi.
    np.testing.assert_allclose(elliptical_novelty(query, state["covariance"]), 1.0 / np.sqrt(ridge), rtol=1e-6)
    past = rng.normal(size=(10, 4))
    state = elliptical_update(state, past)
    assert int(state["count"]) == 10
    phi_past, phi = l2_normalize(past), l2_normalize(query)
    sigma = ridge * np.eye(4) + phi_past.T @ phi_past
    expected = np.sqrt(np.einsum("nd,de,ne->n", phi, np.linalg.inv(sigma), phi))
    np.testing.assert_allclose(elliptical_novelty(query, state["covariance"]), expected, rtol=1e-5)


def test_elliptical_rank_one_update_matches_sherman_morrison():
    rng = np.random.default_rng(2)
    state = elliptical_update(init_elliptical_state(input_dim=3, ridge=1.0), rng.normal(size=(4, 3)))
    inv_before = np.linalg.inv(state["covariance"])
    new = rng.normal(size=(1, 3))
    u = l2_normalize(new)[0]
    inv_after = inv_before - np.outer(inv_before @ u, u @ inv_before) / (1.0 + u @ inv_before @ u)
    query = rng.normal(size=(5, 3))
    phi = l2_normalize(query)
    expected = np.sqrt(np.einsum("nd,de,ne->n", phi, inv_after, phi))
    updated = elliptical_update(state, new)
    np.testing.assert_allclose(elliptical_novelty(query, updated["covariance"]), expected, rtol=1e-6)
    # The bonus of a direction shrinks once it has been seen.
    assert elliptical_novelty(new, updated["covariance"])[0] < elliptical_novelty(new, state["covariance"])[0]


def test_elliptical_ridge_must_be_positive():
    with pytest.raises(ValueError):
        init_elliptical_state(input_dim=2, ridge=0.0)


def test_final_pooled_layer_is_masked_mean_of_last_hidden_state():
    cfg = model.Qwen3Config(hidden_size=8, intermediate_size=16, vocab_size=16,
                            num_hidden_layers=2, num_attention_heads=2,
                            num_key_value_heads=1, head_dim=4, dtype=jnp.float32)
    policy = model.Qwen3Model(cfg)
    ids = jnp.array([[3, 4, 5, 0], [6, 7, 0, 0]], dtype=jnp.int32)
    mask = (ids > 0).astype(jnp.int32)
    params = policy.init(jax.random.key(0), ids, mask)["params"]
    pooled = np.asarray(policy.apply({"params": params}, ids, mask, return_pooled_layers=(2,)))[:, 0]
    hidden = np.asarray(policy.apply({"params": params}, ids, mask, return_hidden=True))
    m = np.asarray(mask, dtype=np.float32)[..., None]
    np.testing.assert_allclose(pooled, (hidden * m).sum(1) / m.sum(1), atol=1e-5)
    with pytest.raises(ValueError):
        policy.apply({"params": params}, ids, mask, return_pooled_layers=(3,))


class Tokenizer:
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        return {"input_ids": [1] * 8}


def _example():
    return SimpleNamespace(problem_id="p", prompt_text="prompt", ground_truth="1")


@pytest.mark.parametrize("method", ["knn", "elliptical"])
def test_bonus_scored_before_update_and_credited_only_on_correct(monkeypatch, method):
    cfg = replace(TrainConfig(), lambda_novelty=0.5, novelty_method=method, knn_k=2,
                  novelty_layers=[4], soft_overlong_expected_len=0)
    features = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    monkeypatch.setattr(train, "_extract_novelty_features", lambda **kwargs: (
        {"layer_4": features},
        dict(count=3, truncated_count=0, token_length_sum=24, max_input_tokens=8, max_length=8)))
    state = init_feature_novelty_state(method, input_dim=3, knn_buffer_size=8, elliptical_ridge=1.0)
    if method == "knn":
        state = knn_append(state, np.array([[1.0, 1.0, 0.0]]))
    completions = [[Completion(r"\boxed{1}", [], [], "stop"),
                    Completion(r"\boxed{2}", [], [], "stop"),
                    Completion(r"\boxed{1}", [], [], "stop")]]
    pending = []
    rows, same_state, _ = train._score_rollouts(
        tokenizer=Tokenizer(), examples=[_example()], completions=completions, cfg=cfg,
        params={}, model=object(), feature_step=object(), rnd_map=state,
        rnd_feature_batches=pending)
    assert same_state is state
    if method == "knn":
        # One neighbor available, (1, 1, 0) / sqrt(2): cos = 1/sqrt(2), 1/sqrt(2), 0.
        raw = [1.0 - 1.0 / np.sqrt(2.0), 1.0 - 1.0 / np.sqrt(2.0), 1.0]
    else:
        raw = [1.0, 1.0, 1.0]  # Sigma = I before any update
    expected = raw[0]
    assert [row["novelty_raw"] for row in rows] == pytest.approx(raw, abs=1e-6)
    assert [row["is_correct"] for row in rows] == [True, False, True]
    assert rows[1]["novelty_reward"] == 0.0
    assert rows[1]["blended_reward"] == pytest.approx(-1.0)
    assert rows[0]["blended_reward"] == pytest.approx(1.0 + 0.5 * expected, abs=1e-6)
    # Only correct rows that were used for training enter the state.
    rows[2]["used_for_training"] = False
    updated = train._update_rnd_after_policy(state, pending, cfg)
    if method == "knn":
        assert int(updated["size"]) == 2
        np.testing.assert_allclose(knn_buffer_contents(updated)[1], features[0])
        assert int(state["size"]) == 1
    else:
        assert int(updated["count"]) == 1
        np.testing.assert_allclose(updated["covariance"], np.eye(3) + np.outer(features[0], features[0]))


@pytest.mark.parametrize("method", ["knn", "elliptical"])
def test_novelty_state_checkpoint_roundtrip(tmp_path, method):
    import orbax.checkpoint as ocp

    state = init_feature_novelty_state(method, input_dim=3, knn_buffer_size=4, elliptical_ridge=1.0)
    state = (knn_append if method == "knn" else elliptical_update)(state, np.eye(3)[:2])
    path = str(tmp_path / "step_000001")
    cfg = replace(TrainConfig(), novelty_method=method, resume_checkpoint=path)
    assert train._novelty_checkpoint_key(cfg) == "novelty_state"
    ocp.PyTreeCheckpointer().save(path, {"novelty_state": train._checkpoint_safe_host_tree(state)})
    fresh = init_feature_novelty_state(method, input_dim=3, knn_buffer_size=4, elliptical_ridge=1.0)
    restored = train._maybe_restore_rnd_from_checkpoint(fresh, cfg)
    for key, value in state.items():
        np.testing.assert_array_equal(np.asarray(restored[key]), np.asarray(value))
    # A cross-stage handoff starts from fresh state.
    handoff = replace(cfg, resume_checkpoint="", init_weights_checkpoint=path)
    assert train._maybe_restore_rnd_from_checkpoint(fresh, handoff) is fresh


def test_contract_accepts_alternative_methods_and_pins_knn_geometry():
    cfg = replace(TrainConfig(), lambda_novelty=0.5)
    validate_contract(cfg)
    validate_contract(replace(cfg, novelty_method="knn"))
    validate_contract(replace(cfg, novelty_method="elliptical", elliptical_ridge=0.1))
    for bad in (dict(novelty_method="knn", knn_k=8),
                dict(novelty_method="knn", knn_buffer_size=1024),
                dict(novelty_method="elliptical", elliptical_ridge=0.0),
                dict(novelty_method="cosine"),
                dict(rnd_lr=1e-3)):
        with pytest.raises(ValueError, match="Contract violation"):
            validate_contract(replace(cfg, **bad))
