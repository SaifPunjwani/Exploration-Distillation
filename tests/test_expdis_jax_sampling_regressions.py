"""Sampling boundaries and independent objective derivatives, all offline."""
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from expdis_jax import generate, train, distill
from expdis_jax.config import TrainConfig, validate_contract
from expdis_jax.grpo import GrpoConfig, dr_grpo_loss_from_logps, chunked_per_token_logps_from_hidden
from test_expdis_jax_rollout_contract import Tokenizer, example


def test_sampled_ids_survive_reward_features_loss_and_sft():
    ids = [2, 3, 4, 5]
    completion = generate.Completion(r"\boxed{1}", [], [], "stop", ids)
    tokenizer = Tokenizer(1000)  # intentionally different retokenization
    cfg = replace(TrainConfig(), max_prompt_len=2, max_completion_len=8,
                  novelty_feature_max_length=8, grpo_num_generations=1)
    rows, _, _ = train._score_rollouts(tokenizer=tokenizer, examples=[example()],
                                     completions=[[completion]], cfg=cfg)
    assert rows[0]["completion_token_length"] == 4
    batch = train._build_rollout_batch(tokenizer, [example()], [[completion]], cfg, rows)
    np.testing.assert_array_equal(batch["full_input_ids"][0, 2:6], ids)
    features, mask, _ = train._encode_novelty_texts(tokenizer, [completion.text], cfg, token_ids=[ids])
    np.testing.assert_array_equal(features[0, :4], ids)
    assert mask.sum() == 4
    sft = distill.pretokenize_sft_examples(tokenizer, rows, 10, max_prompt_len=2)
    assert sft[0]["c_ids"] == ids
    assert tokenizer.calls == 2  # prompts only; completion is never retokenized


@pytest.mark.parametrize("missing_ids", [False, True])
def test_http_preserves_action_ids_and_fails_closed(monkeypatch, missing_ids):
    monkeypatch.setenv("EXPDIS_DAPO_DRGRPO_FULL_CONTRACT", "1")
    requests = []
    def post(url, *, json, **kwargs):
        requests.append(json)
        choice = dict(text="answer", finish_reason="stop", prompt_token_ids=[1, 2])
        if not missing_ids:
            choice["token_ids"] = [3, 4]
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None,
                               json=lambda: {"choices": [choice]})
    monkeypatch.setattr(generate._SESSION, "post", post)
    def call():
        return generate._vllm_complete("http://local/v1", [1, 2], 1, 8, 1., .95, 20,
                                       1, "", 0, "auto", "model", seed=42)
    if missing_ids:
        with pytest.raises(RuntimeError, match="return_token_ids"):
            call()
    else:
        assert call()[0].generated_token_ids == [3, 4]
    assert requests[0]["prompt"] == [1, 2]
    assert requests[0]["return_token_ids"] is True
    assert requests[0]["min_p"] == 0


def test_dynamic_sampling_eight_attempts_preserves_selected_budget(monkeypatch):
    monkeypatch.setenv("EXPDIS_DISABLE_ROLLOUT_DIVERSITY", "1")
    monkeypatch.setenv("EXPDIS_DYNAMIC_SAMPLING_GROUPS_PER_ATTEMPT", "1")
    monkeypatch.setenv("EXPDIS_DYNAMIC_SAMPLING_ELIGIBILITY", "blended_reward")
    calls = []
    def sample(prompts, *args, **kwargs):
        calls.append(prompts)
        return [[generate.Completion(r"\boxed{2}", [], [], "stop") for _ in range(16)] for _ in prompts]
    monkeypatch.setattr(train, "batched_generate", sample)
    out = train._dynamic_sample_rollouts(rng=np.random.default_rng(1), tokenizer=Tokenizer(8),
        examples=[example() for _ in range(8)], server_urls=["http://local/v1"],
        cfg=TrainConfig(), state=SimpleNamespace(params={}), model=None,
        feature_step=None, rnd_map=None, norm_state={}, prompts_per_step=4)
    selected, all_rows = out[2], out[3]
    assert len(calls) == 8
    assert len(selected) == 64
    assert len(all_rows) == 8 * 64
    assert sum(row["used_for_training"] for row in all_rows) == 64
    assert all(row["dynamic_sampling_fallback"] for row in selected)


def test_asymmetric_clipping_gradient_and_fixed_denominator():
    # Positive advantages clip above 1.28; negative ones below 0.8. Opposite
    # sides remain active. This tests derivatives, not just a scalar loss.
    ratio = jnp.array([[1.5], [.6], [1.5], [.6]])
    advantages = jnp.array([2., -2., -2., 2.])
    mask = jnp.ones_like(ratio)
    cfg = GrpoConfig(clip_epsilon=.2, clip_epsilon_high=.28, loss_normalizer=32768)
    def loss(lp):
        return dr_grpo_loss_from_logps(lp, mask, jnp.zeros_like(lp), advantages, cfg)[0]
    grad = jax.grad(loss)(jnp.log(ratio))
    expected = np.array([[0.], [0.], [3.], [-1.2]]) / (4 * 32768)
    np.testing.assert_allclose(grad, expected, atol=1e-10, rtol=1e-6)


def test_chunked_head_uses_same_bf16_forward_as_dense():
    hidden = jax.random.normal(jax.random.key(1), (2, 5, 8)).astype(jnp.bfloat16)
    kernel = jax.random.normal(jax.random.key(2), (8, 17))
    targets = jnp.full((2, 5), 3)
    logits = (hidden @ kernel.astype(hidden.dtype)).astype(jnp.float32)
    reference = jax.nn.log_softmax(logits)[..., 3]
    got = chunked_per_token_logps_from_hidden(hidden, targets, kernel, chunk_tokens=3)
    np.testing.assert_allclose(got, reference, atol=1e-6)


@pytest.mark.parametrize("overrides", [
    {"rnd_hidden": 128}, {"novelty_metric": "mse"}, {"grpo_updates_per_rollout": 2},
    {"grpo_lr_schedule": "cosine"}, {"adam_beta2": .999},
    {"round_budget_mode": "full_per_round"}, {"central_init_policy": "base_reset"},
])
def test_paper_contract_rejects_algorithm_drift(monkeypatch, overrides):
    monkeypatch.delenv("EXPDIS_DAPO_DRGRPO_FULL_CONTRACT", raising=False)
    with pytest.raises(ValueError, match="Contract violation"):
        validate_contract(replace(TrainConfig(), **overrides))
