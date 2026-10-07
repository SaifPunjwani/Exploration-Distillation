"""Padding must not change active-token features, objectives or derivatives."""
from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

from expdis_jax import distill, model, train
from expdis_jax.batching import sequence_bucket
from expdis_jax.config import TrainConfig, validate_contract


def test_bucket_limits_and_removed_hash_fallback(monkeypatch):
    assert sequence_bucket(2100, 34816) == 4096
    assert sequence_bucket(4500 + 2048, 6548) == 6548
    assert sequence_bucket(12, 20) == 20
    monkeypatch.setenv("EXPDIS_NOVELTY_TEXT_HASH_FALLBACK", "1")
    monkeypatch.setenv("EXPDIS_DAPO_DRGRPO_FULL_CONTRACT", "0")
    with pytest.raises(ValueError, match="text-hash novelty was removed"):
        validate_contract(TrainConfig())


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_sft_padding_preserves_completion_loss_and_parameter_gradients(dtype):
    examples = [dict(p_ids=[1, 2, 3], c_ids=[4, 5, 6, 7, 8, 9]),
                dict(p_ids=[1, 3], c_ids=[2, 4, 9])]
    batch = distill.build_sft_batch_from_pretokenized(examples, 513, 0)
    assert batch["input_ids"].shape == (2, 128)
    assert batch["label_mask"].sum() == 9
    # Independent full-width reference with explicit causal labels.
    full = {key: np.pad(value, ((0, 0), (0, 513 - 128))) for key, value in batch.items()}
    np.testing.assert_array_equal(batch["labels"][0, 2:8], examples[0]["c_ids"])
    cfg = model.Qwen3Config(hidden_size=16, intermediate_size=24, vocab_size=16,
                           num_hidden_layers=2, num_attention_heads=2,
                           num_key_value_heads=1, head_dim=8, dtype=dtype)
    policy = model.Qwen3Model(cfg)
    params = policy.init(jax.random.key(1), jnp.array(batch["input_ids"]),
                         jnp.array(batch["attention_mask"]))["params"]
    def loss(params, data):
        logits = policy.apply({"params": params}, jnp.array(data["input_ids"]),
                              jnp.array(data["attention_mask"]))
        logps = jax.nn.log_softmax(logits.astype(jnp.float32))
        chosen = jnp.take_along_axis(logps, jnp.array(data["labels"])[..., None], axis=-1)[..., 0]
        return -(chosen * data["label_mask"]).sum() / data["label_mask"].sum()
    short_loss, short_grads = jax.value_and_grad(loss)(params, batch)
    full_loss, full_grads = jax.value_and_grad(loss)(params, full)
    tolerance = 1e-5 if dtype == jnp.float32 else .003
    np.testing.assert_allclose(short_loss, full_loss, rtol=tolerance, atol=tolerance)
    for short, reference in zip(jax.tree.leaves(short_grads), jax.tree.leaves(full_grads)):
        np.testing.assert_allclose(short, reference, rtol=tolerance, atol=tolerance)


def test_novelty_trims_each_microbatch_preserving_left_padding_and_features(monkeypatch):
    monkeypatch.setenv("EXPDIS_NOVELTY_BATCH_SIZE", "1")
    cfg = model.Qwen3Config(hidden_size=8, intermediate_size=16, vocab_size=16,
                           num_hidden_layers=2, num_attention_heads=2,
                           num_key_value_heads=1, head_dim=4, dtype=jnp.float32)
    policy = model.Qwen3Model(cfg)
    ids, mask = np.zeros((2, 513), np.int32), np.zeros((2, 513), np.int32)
    ids[0, :9], mask[0, :9] = np.arange(1, 10), 1
    ids[1, 7:12], mask[1, 7:12] = np.arange(1, 6), 1
    params = policy.init(jax.random.key(3), jnp.array(ids), jnp.array(mask))["params"]
    widths = []
    def features(p, i, m):
        widths.append(i.shape[1])
        return policy.apply({"params": p}, i, m, return_pooled_layers=(0, 1))
    reference = np.asarray(features(params, jnp.array(ids), jnp.array(mask)))
    widths.clear()
    actual = train._extract_novelty_features_from_encoded(params=params,
        input_ids=ids, attention_mask=mask,
        cfg=replace(TrainConfig(), novelty_layers=(0, 1)), feature_step=features)
    assert widths == [128, 128]
    for i in (0, 1):
        np.testing.assert_allclose(actual[f"layer_{i}"], reference[:, i], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("length", [128, 129])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_real_pallas_kernel_padding_forward_and_backward_on_cpu(monkeypatch, length, dtype):
    # Interpret the actual tiled TPU kernel, including its custom backward.
    # This checks kernel arithmetic and shape constraints, not TPU compilation.
    monkeypatch.setattr(pl, "pallas_call", partial(pl.pallas_call, interpret=True))
    keys = jax.random.split(jax.random.key(8), 3)
    q, k, v = [jax.random.normal(key, (1, 1, length, 128)).astype(dtype) for key in keys]
    mask = jnp.ones((1, length), jnp.int32).at[:, -3:].set(0)
    scale = 128 ** -.5
    def reference(q, k, v):
        logits = jnp.einsum("bhtd,bhsd->bhts", q.astype(jnp.float32), k.astype(jnp.float32)) * scale
        valid = jnp.tril(jnp.ones((length, length), bool))[None, None]
        valid = valid & (mask[:, None, :, None] == mask[:, None, None, :])
        probabilities = jax.nn.softmax(jnp.where(valid, logits, jnp.finfo(jnp.float32).min), axis=-1)
        return jnp.einsum("bhts,bhsd->bhtd", probabilities.astype(dtype), v)
    def kernel(q, k, v):
        return model._flash_attention_padded(q, k, v, mask, sm_scale=scale)
    tolerance = 2e-5 if dtype == jnp.float32 else .004
    np.testing.assert_allclose(np.asarray(kernel(q, k, v), dtype=np.float32),
                               np.asarray(reference(q, k, v), dtype=np.float32),
                               atol=tolerance, rtol=tolerance)
    def objective(fn, q, k, v):
        return (fn(q, k, v).astype(jnp.float32) ** 2 * mask[:, None, :, None]).sum()
    actual = jax.grad(partial(objective, kernel), argnums=(0, 1, 2))(q, k, v)
    expected = jax.grad(partial(objective, reference), argnums=(0, 1, 2))(q, k, v)
    for got, ref in zip(actual, expected):
        got, ref = np.asarray(got, dtype=np.float32), np.asarray(ref, dtype=np.float32)
        if dtype == jnp.float32:
            np.testing.assert_allclose(got, ref, atol=4e-5, rtol=4e-5)
        else:
            # Tiled backward recomputes softmax and rounds its intermediate
            # products differently from dense autodiff. Compare vector error:
            # coordinate-relative error is ill-conditioned near zero.
            relative_error = np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-12)
            assert relative_error < 2 * float(jnp.finfo(dtype).eps)
        assert np.isfinite(got).all()
        np.testing.assert_array_equal(got[:, :, -3:], 0)
