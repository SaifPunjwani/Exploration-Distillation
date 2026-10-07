"""Regressions for serialized SFT actions and campaign-to-stage budgets."""

from dataclasses import replace

import pytest

from expdis_jax import distill, pipeline, train
from expdis_jax.config import TrainConfig, validate_contract
from expdis_jax.generate import Completion
import test_expdis_jax_pipeline_cpu as tiny_pipeline


def test_sampled_actions_survive_trajectory_jsonl_into_sft(monkeypatch, tmp_path):
    # The server includes a terminal action that is absent from decoded text.
    # Re-tokenizing the completion would silently lose this sampled token.
    terminal_id = 31  # Inside the tiny model's 32-token vocabulary.

    def completion_with_terminal(text, logprobs, tokens, finish_reason, token_ids):
        return Completion(text, logprobs, tokens, finish_reason, [*token_ids, terminal_id])

    monkeypatch.setattr(tiny_pipeline, "Completion", completion_with_terminal)
    pretokenize = distill.pretokenize_sft_examples
    supervised = []

    def capture_sft(tokenizer, examples, max_total_len, max_prompt_len=2048):
        encoded = pretokenize(tokenizer, examples, max_total_len, max_prompt_len)
        for example, item in zip(examples, encoded):
            assert example["completion_token_ids"] is not None, "trajectory JSONL lost sampled actions"
            assert item["c_ids"] == example["completion_token_ids"]
            assert item["c_ids"][-1] == terminal_id
        supervised.extend(encoded)
        return encoded

    monkeypatch.setattr(distill, "pretokenize_sft_examples", capture_sft)
    tiny_pipeline.run_tiny_pipeline(monkeypatch, tmp_path)
    assert len(supervised) == 2


@pytest.mark.parametrize("rounds,explorers,per_round", [
    (1, 1, [200]),
    (1, 2, [100, 100]),
    (1, 3, [67, 67, 66]),
    (1, 5, [40] * 5),
    (1, 7, [29, 29, 29, 29, 28, 28, 28]),
    (4, 1, [50]),
    (5, 1, [40]),
    (4, 3, [17, 17, 16]),
    (5, 3, [14, 13, 13]),
    (4, 5, [10] * 5),
])
def test_every_paper_budget_reaches_each_native_explorer(monkeypatch, tmp_path, rounds, explorers, per_round):
    # Exercise the actual driver's stage construction and the actual trainer's
    # validation. Stop immediately before device/model initialization.
    class ValidatedStage(Exception):
        pass

    def stop_before_devices():
        raise ValidatedStage

    observed = []
    run_training = train.run_training

    def validate_stage(cfg):
        with pytest.raises(ValidatedStage):
            run_training(cfg)
        observed.append(cfg.grpo_max_steps)

    monkeypatch.setenv("EXPDIS_DAPO_DRGRPO_FULL_CONTRACT", "1")
    monkeypatch.setenv("EXPDIS_PIPELINE_STOP_AFTER_EXPLORER", "1")
    monkeypatch.setattr(train, "init_distributed", stop_before_devices)
    monkeypatch.setattr(train, "run_training", validate_stage)
    monkeypatch.setattr(pipeline, "_wandb_log_pipeline", lambda *a, **kw: None)
    monkeypatch.setattr(pipeline, "_write_run_summary", lambda *a, **kw: None)
    monkeypatch.setattr(pipeline, "_maybe_export_state_to_hf_dir", lambda *a, **kw: None)
    cfg = replace(TrainConfig(), num_rounds=rounds, scouts_per_round=explorers,
                  pipeline_mode="multi_round" if rounds > 1 else "two_model")
    validate_contract(cfg)
    for round_index in range(1, rounds + 1):
        round_cfg = pipeline._build_round_cfg(
            cfg, round_index, rounds, str(tmp_path / "previous-main"), str(tmp_path)
        )
        pipeline.main(round_cfg)
    assert observed == per_round * rounds
    assert sum(observed) == 200


def test_campaign_budget_still_rejects_an_empty_explorer():
    with pytest.raises(ValueError, match="zero updates"):
        validate_contract(replace(TrainConfig(), num_rounds=4, scouts_per_round=5,
                                  grpo_max_steps=19))
