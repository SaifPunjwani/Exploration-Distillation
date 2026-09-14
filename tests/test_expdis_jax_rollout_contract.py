"""CPU regressions for reward, sampling, and predictor integration."""
from dataclasses import replace
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from expdis_jax import train
from expdis_jax.config import TrainConfig
from expdis_jax.generate import Completion
from expdis_jax.novelty import init_rnd_map


class Tokenizer:
    pad_token_id = 0

    def __init__(self, completion_length=30000):
        self.completion_length = completion_length
        self.calls = 0

    def __call__(self, text, **kwargs):
        self.calls += 1
        return {"input_ids": [1] * (2 if text == "prompt" else self.completion_length)}


def example():
    return SimpleNamespace(problem_id="p", prompt_text="prompt", ground_truth="1")


def test_exact_length_drives_reward_and_batch_without_duplicate_tokenization():
    tokenizer = Tokenizer()
    cfg = replace(TrainConfig(), grpo_num_generations=1)
    completion = Completion(r"solution \boxed{1}", [], [], "stop")
    rows, _, _ = train._score_rollouts(
        tokenizer=tokenizer, examples=[example()], completions=[[completion]], cfg=cfg)
    assert rows[0]["completion_token_length"] == 30000
    assert rows[0]["overlong_penalty"] == pytest.approx(-(30000 - 26214) / 6554)
    batch = train._build_rollout_batch(tokenizer, [example()], [[completion]], cfg, rows)
    assert tokenizer.calls == 2  # one completion and one prompt
    assert batch["completion_mask"].sum() == 30000
    assert batch["full_attention_mask"].sum() == 30002
    assert batch["rewards"][0] == pytest.approx(1 - (30000 - 26214) / 6554)


def test_exact_budget_clipping_masks_before_building_loss():
    cfg = replace(TrainConfig(), grpo_num_generations=1)
    completion = Completion(r"\boxed{1}", [], [], "stop")
    tokenizer = Tokenizer(32768)
    rows, _, _ = train._score_rollouts(
        tokenizer=tokenizer, examples=[example()], completions=[[completion]], cfg=cfg)
    batch = train._build_rollout_batch(tokenizer, [example()], [[completion]], cfg, rows)
    assert rows[0]["clipped"] is True
    assert rows[0]["terminated"] is False
    assert batch["completion_mask"].sum() == 0
    assert batch["clipped"][0] == 1


def test_reward_verifies_full_completion_instead_of_only_tail():
    completion = Completion(r"\boxed{2}" + " discussion" * 600 + "\nanswer: 1", [], [], "stop")
    rows, _, _ = train._score_rollouts(
        tokenizer=Tokenizer(2000), examples=[example()], completions=[[completion]], cfg=TrainConfig())
    assert rows[0]["is_correct"] is False


def test_different_typed_jax_keys_produce_independent_rnd_targets():
    a = init_rnd_map(rng=jax.random.key(1), layers=[7], input_dim=4, hidden_dim=4)
    b = init_rnd_map(rng=jax.random.key(2), layers=[7], input_dim=4, hidden_dim=4)
    assert not np.array_equal(a["layer_7"]["target"]["fc1"]["kernel"],
                              b["layer_7"]["target"]["fc1"]["kernel"])


def test_candidate_scoring_freezes_predictor_and_updates_selected_features_once(monkeypatch):
    cfg = replace(TrainConfig(), lambda_novelty=0.5, novelty_layers=[7], rnd_hidden=4)
    rnd = init_rnd_map(rng=1, layers=[7], input_dim=4, hidden_dim=4)
    monkeypatch.setattr(train, "_extract_novelty_features", lambda **kwargs: (
        {"layer_7": np.ones((1, 4), dtype=np.float32)},
        dict(count=1, truncated_count=0, token_length_sum=8, max_input_tokens=8, max_length=8)))
    pending = []
    scores = []
    for _ in range(2):
        rows, rnd, _ = train._score_rollouts(
            tokenizer=Tokenizer(8), examples=[example()],
            completions=[[Completion(r"\boxed{1}", [], [], "stop")]],
            cfg=cfg, params={}, model=object(), feature_step=object(),
            rnd_map=rnd, rnd_feature_batches=pending)
        scores.append(rows[0]["novelty_raw"])
        assert int(rnd["layer_7"]["opt_state"][0].count) == 0
    assert scores[0] == scores[1]
    pending[0][0][0]["used_for_training"] = False
    updated = train._update_rnd_after_policy(rnd, pending, cfg)
    assert int(updated["layer_7"]["opt_state"][0].count) == 1
    np.testing.assert_array_equal(updated["layer_7"]["target"]["fc1"]["kernel"],
                                  rnd["layer_7"]["target"]["fc1"]["kernel"])


def test_failed_reload_upload_cannot_silently_continue(monkeypatch, tmp_path):
    monkeypatch.setenv("EXPDIS_HF_MIRROR_VLLM_EXPORTS", "1")
    monkeypatch.setenv("EXPDIS_EXTERNAL_VLLM_RELOAD_REQUIRED", "1")
    monkeypatch.delenv("EXPDIS_GCS_CHECKPOINT_BASE", raising=False)
    monkeypatch.setattr(train, "_hf_artifact_repo", lambda: "test/repo")
    monkeypatch.setattr(train, "_ensure_space_for_vllm_export", lambda *a: None)
    monkeypatch.setattr(train, "export_flax_params_to_hf_dir", lambda *a, **kw: None)
    monkeypatch.setattr(train, "_start_internal_reload_tar_server", lambda *a: ("", None))
    monkeypatch.setattr(train, "_maybe_upload_path_to_hf", lambda *a, **kw: False)
    cfg = replace(TrainConfig(), output_dir=str(tmp_path))
    model_cfg = SimpleNamespace(num_hidden_layers=1, tie_word_embeddings=True,
                                use_qk_norm=True, hf_weight_prefix="model")
    with pytest.raises(RuntimeError, match="no reload bundle"):
        train._maybe_export_and_reload_vllm(
            SimpleNamespace(params={}), cfg, model_cfg, 0, reason="stage_start")
