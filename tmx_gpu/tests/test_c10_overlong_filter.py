"""C10 survivor-consistent eligibility, baseline, and provenance tests."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tmx_gpu import c10_evidence, common, grpo_gpu, pipeline_gpu
from tmx_gpu.vllm_pool import Completion


def _grpo_args(*extra: str):
    return grpo_gpu.build_arg_parser().parse_args([
        "--output-dir", "/tmp/c10-test",
        "--rl-validity-mode", "c8",
        "--mask-truncated",
        "--truncation-baseline-mode", "unmasked_survivors",
        "--scientific-variant", grpo_gpu.C10_SCIENTIFIC_VARIANT,
        "--c10-execution-profile", "canary",
        "--expdis-stabilization-mode", grpo_gpu.C9_RND_LIFECYCLE,
        "--novelty-normalization", "selected_batch_zscore",
        *extra,
    ])


class TestFailClosedIdentity:
    def test_exact_c10_filter_contract(self):
        args = _grpo_args()
        grpo_gpu.validate_overlong_filter_contract(args)
        assert grpo_gpu.survivor_consistent_overlong_filter(args)

    @pytest.mark.parametrize("extra,pattern", [
        ([], "permitted only"),
        (["--no-mask-truncated", "--truncation-baseline-mode",
          "unmasked_survivors"],
         "requires the exact C10"),
    ])
    def test_c8_mask_or_survivor_mode_without_c10_is_rejected(
            self, extra, pattern):
        args = grpo_gpu.build_arg_parser().parse_args([
            "--output-dir", "/tmp/c10-negative",
            "--rl-validity-mode", "c8", "--mask-truncated", *extra,
        ])
        with pytest.raises(ValueError, match=pattern):
            grpo_gpu.validate_overlong_filter_contract(args)

    def test_partial_c10_label_is_rejected(self):
        args = grpo_gpu.build_arg_parser().parse_args([
            "--output-dir", "/tmp/c10-partial",
            "--rl-validity-mode", "c8",
            "--scientific-variant", grpo_gpu.C10_SCIENTIFIC_VARIANT,
        ])
        with pytest.raises(ValueError, match="C10 Overlong Filtering contract drift"):
            grpo_gpu.validate_overlong_filter_contract(args)

    def test_c9_c5_mask_off_contract_remains_unchanged(self):
        args = grpo_gpu.build_arg_parser().parse_args([
            "--output-dir", "/tmp/c9-unchanged",
            "--rl-validity-mode", "c8",
            "--c9-execution-profile", "canary",
            "--stabilization-cell", "C5",
            "--scientific-variant", grpo_gpu.C9_SCIENTIFIC_VARIANT,
            "--expdis-stabilization-mode", grpo_gpu.C9_RND_LIFECYCLE,
            "--novelty-normalization", "selected_batch_zscore",
        ])
        grpo_gpu.validate_overlong_filter_contract(args)
        assert not args.mask_truncated
        assert args.truncation_baseline_mode == "all_rows"

    def test_legacy_c7_mask_replay_remains_available(self):
        args = grpo_gpu.build_arg_parser().parse_args([
            "--output-dir", "/tmp/c7-replay", "--mask-truncated",
        ])
        grpo_gpu.validate_overlong_filter_contract(args)


class TestSurvivorEligibility:
    @staticmethod
    def _row(correct: bool, clipped: bool, reward: float, novelty: float = 0.0):
        return {
            "is_correct": correct,
            "clipped": clipped,
            "blended_reward": reward,
            "novelty_used": novelty,
        }

    def test_four_correct_survivors_and_clipped_wrong_rows_are_not_mixed(self):
        rows = [self._row(True, False, 1.0, novelty=float(i))
                for i in range(4)] + [
            self._row(False, True, -2.0) for _ in range(12)]
        reason, eligible, facts = grpo_gpu.group_eligibility_for_rows(
            rows, lambda_novelty=0.5, validity_mode="c8",
            survivor_only=True)
        assert reason == "reward_informative_all_correct"
        assert eligible
        assert facts["eligibility_correct_count"] == 4
        assert facts["eligibility_population_size"] == 4
        assert facts["raw_correct_count"] == 4
        assert facts["masked_count"] == 12

        for row in rows[:4]:
            row["novelty_used"] = 1.0
        reason, eligible, _ = grpo_gpu.group_eligibility_for_rows(
            rows, lambda_novelty=0.5, validity_mode="c8",
            survivor_only=True)
        assert reason == "all_correct"
        assert not eligible

    def test_mixed_survivors_are_eligible(self):
        rows = [
            self._row(True, False, 1.0),
            self._row(False, False, -1.0),
            *[self._row(False, True, -2.0) for _ in range(14)],
        ]
        reason, eligible, facts = grpo_gpu.group_eligibility_for_rows(
            rows, lambda_novelty=0.5, validity_mode="c8",
            survivor_only=True)
        assert (reason, eligible) == ("informative", True)
        assert facts["survivor_count"] == 2
        assert facts["survivor_correct_count"] == 1

    @pytest.mark.parametrize("survivors", [0, 1])
    def test_low_survivor_groups_rejected_upstream(self, survivors):
        rows = [self._row(i == 0, i >= survivors, 1.0 if i == 0 else -2.0)
                for i in range(16)]
        reason, eligible, facts = grpo_gpu.group_eligibility_for_rows(
            rows, lambda_novelty=0.5, validity_mode="c8",
            survivor_only=True)
        assert reason == "insufficient_unmasked_survivors"
        assert not eligible
        assert facts["survivor_count"] == survivors

    def test_dynamic_sampling_retries_low_survivor_candidate(
            self, monkeypatch):
        candidates = [
            [self._row(True, False, 1.0),
             self._row(False, True, -2.0),
             self._row(False, True, -2.0),
             self._row(False, True, -2.0)],
            [self._row(True, False, 1.0),
             self._row(False, False, -1.0),
             self._row(False, True, -2.0),
             self._row(False, True, -2.0)],
        ]
        calls = 0

        def fake_score(*_args, **_kwargs):
            nonlocal calls
            rows = [dict(row) for row in candidates[calls]]
            calls += 1
            return rows

        class Pool:
            def generate(self, prompts, n, **_kwargs):
                return [[Completion(text="x", finish_reason="stop")
                         for _ in range(n)] for _ in prompts]

        monkeypatch.setattr(grpo_gpu, "score_completions", fake_score)
        args = _grpo_args(
            "--phase", "central", "--lambda-novelty", "0",
            "--prompts-per-step", "1", "--num-generations", "4",
            "--dynamic-max-attempts", "2", "--probe-every", "0",
        )
        example = SimpleNamespace(
            problem_id="p0", prompt_text="p", ground_truth="1")
        selected, all_rows, stats, _ = grpo_gpu._dynamic_sample_step(
            rng=np.random.default_rng(0), pool=Pool(), tokenizer=None,
            model=None, rnd=None, examples=[example], args=args, lam=0.0,
            step=1)
        assert len(selected) == 4
        assert len(all_rows) == 8
        assert stats["candidate_group_count"] == 2.0
        assert stats["discarded_group_count"] == 1.0
        assert stats["insufficient_survivor_group_count"] == 1.0
        assert stats["selected_survivor_row_count"] == 2.0
        assert stats["selected_masked_row_count"] == 2.0
        assert stats["fallback_group_count"] == 0.0
        assert all(row["dynamic_sampling_eligibility_population"]
                   == "unmasked_survivors" for row in selected)


class TestSurvivorAdvantagesAndLoss:
    def test_clipped_advantages_zero_and_wrong_survivor_stays_negative(self):
        advantages, stats = grpo_gpu.group_mean_advantages_unmasked(
            np.asarray([1.0, -1.0, -2.0, -2.0], dtype=np.float32),
            np.asarray([True, True, False, False]), 4)
        np.testing.assert_allclose(advantages, [1.0, -1.0, 0.0, 0.0])
        assert advantages[1] < 0.0
        assert stats["survivor_row_count"] == 2.0
        assert stats["max_abs_survivor_advantage_sum"] <= 1e-7

    def test_selected_low_survivor_group_fails_closed(self):
        with pytest.raises(ValueError, match="at least two survivors"):
            grpo_gpu.group_mean_advantages_unmasked(
                np.asarray([1.0, -2.0, -2.0, -2.0]),
                np.asarray([True, False, False, False]), 4)

    def test_fixed_full_batch_denominator_and_mask_accounting(
            self, monkeypatch):
        args = _grpo_args(
            "--device", "cpu", "--max-completion-len", "10",
            "--loss-denominator-len", "10")
        rows = [
            {"prompt_text": "p", "_comp_ids": [2, 3], "clipped": False},
            {"prompt_text": "p", "_comp_ids": [2], "clipped": False},
            {"prompt_text": "p", "_comp_ids": [2], "clipped": True},
            {"prompt_text": "p", "_comp_ids": [2], "clipped": True},
        ]
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        class Tokenizer:
            def __call__(self, _text, add_special_tokens=False):
                return {"input_ids": [1]}

        def fake_logps(model, _pids, comp_ids, _args, *, requires_grad):
            return model.weight.reshape(1).repeat(len(comp_ids))

        monkeypatch.setattr(grpo_gpu, "_row_logps", fake_logps)
        update = grpo_gpu._train_update(
            model, optimizer, Tokenizer(), rows,
            np.asarray([1.0, -1.0, 0.0, 0.0]), args)
        assert update["loss"] == pytest.approx(-0.025)
        assert update["fixed_loss_denominator"] == 40.0
        assert update["raw_selected_rows"] == 4.0
        assert update["survivor_rows"] == 2.0
        assert update["rows_backward"] == 2.0
        assert update["masked_truncated_rows"] == 2.0


def test_runtime_contract_adds_c10_nested_identity_only(monkeypatch):
    args = _grpo_args(
        "--phase", "scout", "--steps", "50", "--lr", "1e-6",
        "--lambda-novelty", "0.5", "--probe-every", "0")
    grpo_gpu.apply_completion_budget(args)
    monkeypatch.setattr(
        grpo_gpu, "requires_canonical_training_execution_environment",
        lambda _args: False)
    contract = grpo_gpu.build_runtime_contract(
        args, resolved_lr=1e-6, resolved_steps=50)
    nested = contract["survivor_overlong_filter_contract"]
    assert nested["dynamic_sampling_eligibility_population"] == \
        "unmasked_survivors"
    assert nested["minimum_dynamic_sampling_survivors"] == 2
    assert nested["loss_denominator_population"] == "full_sampled_rows"
    assert nested["rnd_update_population"] == "all_selected_raw_rows"
    assert contract["version"] == "grpo_runtime_contract_v3"


def test_pipeline_forwards_c10_without_mutating_c9_identity():
    args = pipeline_gpu.build_arg_parser().parse_args([
        "--output-dir", "/tmp/c10-pipeline", "--run-name", "c10-test",
        "--c10-execution-profile", "canary",
        "--expdis-stabilization-mode", grpo_gpu.C9_RND_LIFECYCLE,
        "--novelty-normalization", "selected_batch_zscore",
        "--scientific-variant", grpo_gpu.C10_SCIENTIFIC_VARIANT,
        "--mask-truncated",
        "--truncation-baseline-mode", "unmasked_survivors",
    ])
    stage = pipeline_gpu._grpo_args(
        args, phase="scout", model=args.model, output_dir="/tmp/c10-scout",
        lam=0.5, steps=50, round_idx=1, global_step_offset=0)
    assert stage.c10_execution_profile == "canary"
    assert stage.c9_execution_profile == ""
    assert stage.stabilization_cell == ""
    assert grpo_gpu.survivor_consistent_overlong_filter(stage)


def _synthetic_endpoint_rows_and_metrics():
    rows = []
    metrics_rows = []
    for step in range(1, 51):
        step_rows = []
        for group_index in range(4):
            group = []
            for row_index in range(16):
                correct = row_index < 8
                clipped = row_index >= 14
                true_len = 16384 if clipped else 100 + row_index
                raw_novelty = np.float32(
                    0.5 + step * 0.001 + group_index * 0.03
                    + row_index * 0.007)
                eligibility_novelty = float(raw_novelty) if correct else 0.0
                correctness = 1.0 if correct else -1.0
                overlong = common.soft_overlong_penalty(
                    true_len, 13107, 3277)
                group.append({
                    "trajectory_schema_version":
                        grpo_gpu.TRAJECTORY_RECORD_SCHEMA_VERSION,
                    "scientific_variant":
                        c10_evidence.C10_SCIENTIFIC_VARIANT,
                    "c10_execution_profile": "canary",
                    "c9_execution_profile": "",
                    "stabilization_cell": "",
                    "expdis_stabilization_mode": grpo_gpu.C9_RND_LIFECYCLE,
                    "novelty_normalization": "selected_batch_zscore",
                    "rnd_update_scope": "final_selected_learner_rows",
                    "step": step,
                    "problem_id": f"p-{step}-{group_index}",
                    "prompt_text": f"prompt-{step}-{group_index}",
                    "completion_text": (
                        "\\boxed{4}" if correct else "not an answer"),
                    "ground_truth": "4",
                    "finish_reason": "length" if clipped else "stop",
                    "completion_token_length_true": true_len,
                    "completion_token_length": min(true_len, 16384),
                    "completion_token_length_retokenized": 10,
                    "clipped": clipped,
                    "terminated": not clipped,
                    "valid_answer": correct,
                    "is_correct": correct,
                    "correctness_reward": correctness,
                    "dynamic_sampling_eligibility_novelty_used":
                        eligibility_novelty,
                    "dynamic_sampling_eligibility_blended_reward":
                        correctness + 0.5 * eligibility_novelty + overlong,
                    "novelty_used": eligibility_novelty,
                    "blended_reward":
                        correctness + 0.5 * eligibility_novelty + overlong,
                    "c10_rnd_raw_novelty_score": float(raw_novelty),
                })
            reason, eligible, facts = grpo_gpu.group_eligibility_for_rows(
                group, lambda_novelty=0.5, validity_mode="c8",
                survivor_only=True)
            assert eligible and reason == "informative"
            for row in group:
                row.update({
                    "dynamic_sampling_reason": reason,
                    "dynamic_sampling_group_selected": True,
                    "dynamic_sampling_policy_loss_eligible":
                        not row["clipped"],
                    "dynamic_sampling_raw_group_size":
                        facts["raw_group_size"],
                    "dynamic_sampling_survivor_count":
                        facts["survivor_count"],
                    "dynamic_sampling_masked_count": facts["masked_count"],
                    "dynamic_sampling_raw_correct_count":
                        facts["raw_correct_count"],
                    "dynamic_sampling_survivor_correct_count":
                        facts["survivor_correct_count"],
                    "dynamic_sampling_eligibility_population":
                        "unmasked_survivors",
                    "dynamic_sampling_eligibility_population_size":
                        facts["eligibility_population_size"],
                    "dynamic_sampling_reward_std":
                        facts["eligibility_reward_std"],
                    "dynamic_sampling_novelty_contribution_std":
                        facts["eligibility_novelty_contribution_std"],
                })
            step_rows.extend(group)

        raw = np.asarray([
            row["c10_rnd_raw_novelty_score"] for row in step_rows],
            dtype=np.float32)
        center = float(np.mean(raw, dtype=np.float64))
        scale = float(np.std(raw, dtype=np.float64))
        normalized = ((raw.astype(np.float64) - center) / scale).astype(
            np.float32)
        final_rewards = []
        for index, row in enumerate(step_rows):
            pre_gate = float(normalized[index])
            novelty = pre_gate if row["is_correct"] else 0.0
            reward = ((1.0 if row["is_correct"] else -1.0)
                      + 0.5 * novelty
                      + common.soft_overlong_penalty(
                          row["completion_token_length"], 13107, 3277))
            row.update({
                "c10_normalized_novelty_pre_gate": pre_gate,
                "novelty_reward": novelty,
                "blended_reward": reward,
            })
            final_rewards.append(reward)
        final = np.asarray(final_rewards, dtype=np.float32)
        survivor_mask = np.asarray(
            [not row["clipped"] for row in step_rows], dtype=bool)
        advantages, stats = grpo_gpu.group_mean_advantages_unmasked(
            final, survivor_mask, 16)
        survivors = int(survivor_mask.sum())
        masked = 64 - survivors
        rows_backward = sum(
            survivor_mask[index] and abs(float(advantages[index])) >= 1e-12
            for index in range(64))
        metrics_rows.append({
            "global_step": step,
            "grpo/step": step,
            "phase": common.PARITY_PHASES["scout"],
            "c10_execution_profile": "canary",
            "scientific_variant": c10_evidence.C10_SCIENTIFIC_VARIANT,
            "round": 1,
            "lambda": 0.5,
            "lambda_novelty": 0.5,
            "num_scouts": 1,
            "rl_validity_mode": "c8",
            "c9_execution_profile": "",
            "stabilization_cell": "",
            "expdis_stabilization_mode": grpo_gpu.C9_RND_LIFECYCLE,
            "novelty_normalization": "selected_batch_zscore",
            "survivor_overlong_filter_version":
                grpo_gpu.C10_OVERLONG_FILTER_CONTRACT_VERSION,
            "train/raw_selected_rows": 64,
            "train/survivor_rows": survivors,
            "train/masked_truncated_rows": masked,
            "train/rows_backward": rows_backward,
            "train/zero_advantage_survivor_rows": survivors - rows_backward,
            "train/fixed_loss_denominator": 64 * 16384,
            "grpo/dynamic/selected_raw_row_count": 64,
            "grpo/dynamic/selected_survivor_row_count": survivors,
            "grpo/dynamic/selected_masked_row_count": masked,
            "train/overlong_filter/raw_selected_row_count": 64,
            "train/overlong_filter/survivor_row_count": survivors,
            "train/overlong_filter/masked_row_count": masked,
            "train/overlong_filter/group_count": 4,
            "train/overlong_filter/min_survivors_per_group":
                stats["min_survivors_per_group"],
            "train/overlong_filter/max_survivors_per_group":
                stats["max_survivors_per_group"],
            "train/overlong_filter/mean_survivors_per_group":
                stats["mean_survivors_per_group"],
            "train/overlong_filter/max_abs_survivor_advantage_sum":
                stats["max_abs_survivor_advantage_sum"],
            "quality/clipped_rate": masked / 64,
            "quality/valid_answer_rate": sum(
                row["valid_answer"] for row in step_rows) / 64,
            "train/correctness_rate": sum(
                row["is_correct"] for row in step_rows) / 64,
            "train/reward_total_mean": float(final.mean()),
            "train/reward_correctness_mean": float(np.mean([
                row["correctness_reward"] for row in step_rows])),
            "train/reward_novelty_mean": float(np.mean([
                row["novelty_reward"] for row in step_rows])),
            "grpo/reward_std": float(final.std()),
            "grpo/dynamic/kept_group_count": 4,
            "grpo/dynamic/fallback_group_count": 0,
            "grpo/dynamic/generation_error_rate": 0,
            "grpo/dynamic/generation_error_row_count": 0,
            "grpo/dynamic/generation_requested_row_count": 64,
            "train/rollout_staleness_updates": 0,
            "train/rollout_staleness_known": 1,
            "train/on_policy_rollout": 1,
            "train/updates_per_rollout": 1,
            "train/policy_kl": 0,
            "train/clip_fraction": 0,
            "train/grpo_loss": 0.1,
            "train/grad_norm": 0.2,
            "train/selected_token_logprob_mean": -0.3,
            "grpo/dynamic/rnd_selected_batch_loss": 0.4,
            "grpo/rnd/candidate_rows_scored": 64,
            "grpo/rnd/updates_this_step": 1,
            "grpo/rnd/update_rows": 64,
            "grpo/rnd/selected_batch_update_count": step,
            "grpo/reward/novelty_normalization_center": center,
            "grpo/reward/novelty_normalization_scale": scale,
            "grpo/reward/novelty_normalization_effective_scale": scale,
            "grpo/reward/novelty_normalized_selected_mean": float(
                np.mean(normalized, dtype=np.float64)),
            "grpo/reward/novelty_normalized_selected_population_std": float(
                np.std(normalized, dtype=np.float64)),
        })
        rows.extend(step_rows)
    return rows, metrics_rows


def test_endpoint_replay_recomputes_survivor_contract_and_rejects_tamper():
    rows, metrics = _synthetic_endpoint_rows_and_metrics()
    replay = pipeline_gpu._replay_c10_survivor_contract(rows, metrics)
    assert replay["raw_selected_rows"] == 3200
    assert replay["survivor_rows"] == 2800
    assert replay["masked_rows"] == 400

    prefix = pipeline_gpu._replay_c10_survivor_contract(
        rows[:64], metrics[:1], expected_completed_steps=1)
    assert prefix["raw_selected_rows"] == 64
    assert prefix["group_count"] == 4
    assert prefix["rolling_health"]["completed_step"] == 1

    bad_prefix = [dict(row) for row in rows[:64]]
    bad_prefix[0]["dynamic_sampling_survivor_count"] -= 1
    with pytest.raises(RuntimeError, match="group counts do not replay"):
        pipeline_gpu._replay_c10_survivor_contract(
            bad_prefix, metrics[:1], expected_completed_steps=1)

    transport_metrics = [dict(metrics[0])]
    transport_metrics[0].update({
        "grpo/dynamic/generation_error_row_count": 8,
        "grpo/dynamic/generation_requested_row_count": 64,
        "grpo/dynamic/generation_error_rate": 0.125,
    })
    with pytest.raises(RuntimeError, match="invalid generation transport"):
        pipeline_gpu._replay_c10_survivor_contract(
            rows[:64], transport_metrics, expected_completed_steps=1)
    pipeline_gpu._replay_c10_survivor_contract(
        rows[:64], transport_metrics, expected_completed_steps=1,
        transport_invalid_step=1)

    fractional_transport = [dict(metrics[0])]
    fractional_transport[0].update({
        "grpo/dynamic/generation_error_row_count": 0.5,
        "grpo/dynamic/generation_requested_row_count": 100.5,
        "grpo/dynamic/generation_error_rate": 0.5 / 100.5,
    })
    with pytest.raises(RuntimeError, match="generation accounting"):
        pipeline_gpu._replay_c10_survivor_contract(
            rows[:64], fractional_transport, expected_completed_steps=1)

    fractional_counter = [dict(metrics[0])]
    fractional_counter[0]["grpo/rnd/update_rows"] = 64.000001
    with pytest.raises(RuntimeError, match="expected exact"):
        pipeline_gpu._replay_c10_survivor_contract(
            rows[:64], fractional_counter, expected_completed_steps=1)

    tampered = [dict(row) for row in rows]
    tampered[14]["clipped"] = False
    with pytest.raises(RuntimeError, match="clipped/survivor"):
        pipeline_gpu._replay_c10_survivor_contract(tampered, metrics)

    tampered = [dict(row) for row in rows]
    tampered[0]["completion_text"] = "\\boxed{5}"
    with pytest.raises(RuntimeError, match="correctness/validity flags"):
        pipeline_gpu._replay_c10_survivor_contract(tampered, metrics)

    bad_metrics = [dict(row) for row in metrics]
    bad_metrics[0]["grpo/dynamic/rnd_selected_batch_loss"] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite"):
        pipeline_gpu._replay_c10_survivor_contract(rows, bad_metrics)

    bad_identity_rows = [dict(row) for row in rows]
    bad_identity_rows[0]["scientific_variant"] = "tampered"
    with pytest.raises(RuntimeError, match="contract identity"):
        pipeline_gpu._replay_c10_survivor_contract(
            bad_identity_rows, metrics)
    bad_identity_metrics = [dict(row) for row in metrics]
    bad_identity_metrics[0]["round"] = 999
    with pytest.raises(RuntimeError, match="metric identity"):
        pipeline_gpu._replay_c10_survivor_contract(
            rows, bad_identity_metrics)

    boolean_identity_metrics = [dict(row) for row in metrics]
    boolean_identity_metrics[0]["round"] = True
    boolean_identity_metrics[0]["num_scouts"] = True
    with pytest.raises(RuntimeError, match="numeric identity"):
        pipeline_gpu._replay_c10_survivor_contract(
            rows, boolean_identity_metrics)


@pytest.mark.parametrize("failure_kind,mutate", [
    (
        "learner_health_nonfinite",
        lambda rows, metric: (
            metric.__setitem__("train/grpo_loss", float("nan")),
            {"nonfinite_fields": ["train/grpo_loss"]},
        )[1],
    ),
    (
        "rnd_update_contract_failure",
        lambda rows, metric: (
            metric.__setitem__("grpo/rnd/updates_this_step", 0),
            {
                "rnd_updates_this_step": 0,
                "rnd_update_rows": 64,
                "rnd_selected_batch_update_count": 1,
            },
        )[1],
    ),
    (
        "incorrect_novelty_credit_failure",
        lambda rows, metric: _mutate_incorrect_novelty_failure(rows, metric),
    ),
    (
        "novelty_normalization_failure",
        lambda rows, metric: _mutate_normalization_failure(rows, metric),
    ),
])
def test_terminal_step_health_failure_replay_binds_exact_violation(
        failure_kind, mutate):
    rows, metrics = _synthetic_endpoint_rows_and_metrics()
    terminal_rows = [dict(row) for row in rows[:64]]
    terminal_metric = dict(metrics[0])
    observed = mutate(terminal_rows, terminal_metric)
    replay = pipeline_gpu._replay_c10_terminal_health_failure(
        terminal_rows, [terminal_metric], completed_step=1,
        failure_kind=failure_kind, observed=observed)
    assert replay["raw_selected_rows"] == 64

    tampered = [dict(row) for row in terminal_rows]
    tampered[1]["completion_text"] = "\\boxed{987654321}"
    with pytest.raises(RuntimeError):
        pipeline_gpu._replay_c10_terminal_health_failure(
            tampered, [terminal_metric], completed_step=1,
            failure_kind=failure_kind, observed=observed)


def test_terminal_nonfinite_replay_rejects_unrelated_novelty_and_rnd_drift():
    rows, metrics = _synthetic_endpoint_rows_and_metrics()
    terminal_rows = [dict(row) for row in rows[:64]]
    terminal_metric = dict(metrics[0])
    terminal_metric["train/grpo_loss"] = float("nan")
    terminal_metric["grpo/rnd/selected_batch_update_count"] = 999
    bad_index = next(
        index for index, row in enumerate(terminal_rows)
        if not row["is_correct"])
    terminal_rows[bad_index]["novelty_reward"] = 0.25
    with pytest.raises(RuntimeError):
        pipeline_gpu._replay_c10_terminal_health_failure(
            terminal_rows, [terminal_metric], completed_step=1,
            failure_kind="learner_health_nonfinite",
            observed={"nonfinite_fields": ["train/grpo_loss"]})


def test_terminal_rnd_nonfinite_counter_replays_portably():
    rows, metrics = _synthetic_endpoint_rows_and_metrics()
    terminal_rows = [dict(row) for row in rows[:64]]
    terminal_metric = dict(metrics[0])
    terminal_metric["grpo/rnd/updates_this_step"] = float("nan")
    replay = pipeline_gpu._replay_c10_terminal_health_failure(
        terminal_rows, [terminal_metric], completed_step=1,
        failure_kind="rnd_update_contract_failure",
        observed={
            "rnd_updates_this_step": float("nan"),
            "rnd_update_rows": 64,
            "rnd_selected_batch_update_count": 1,
        })
    assert replay["raw_selected_rows"] == 64


def test_terminal_incorrect_credit_replay_binds_all_bad_rows():
    rows, metrics = _synthetic_endpoint_rows_and_metrics()
    terminal_rows = [dict(row) for row in rows[:64]]
    terminal_metric = dict(metrics[0])
    indexes = [
        index for index, row in enumerate(terminal_rows)
        if not row["is_correct"]][:2]
    for index in indexes:
        terminal_rows[index]["novelty_reward"] = 0.25
        terminal_rows[index]["blended_reward"] = (
            -1.0 + 0.5 * 0.25 + common.soft_overlong_penalty(
                terminal_rows[index]["completion_token_length"],
                13107, 3277))
    pipeline_gpu._repair_c10_novelty_derived_evidence(
        terminal_rows, terminal_metric)
    observed = {"violations": [
        {"row_index": index, "novelty_reward": 0.25}
        for index in indexes]}
    pipeline_gpu._replay_c10_terminal_health_failure(
        terminal_rows, [terminal_metric], completed_step=1,
        failure_kind="incorrect_novelty_credit_failure",
        observed=observed)


def test_terminal_normalization_replay_rejects_unrelated_reward_corruption():
    rows, metrics = _synthetic_endpoint_rows_and_metrics()
    terminal_rows = [dict(row) for row in rows[:64]]
    terminal_metric = dict(metrics[0])
    observed = _mutate_normalization_failure(
        terminal_rows, terminal_metric)
    terminal_rows[0]["blended_reward"] = -777.0
    with pytest.raises(RuntimeError, match="reward consequence"):
        pipeline_gpu._replay_c10_terminal_health_failure(
            terminal_rows, [terminal_metric], completed_step=1,
            failure_kind="novelty_normalization_failure",
            observed=observed)


def test_failed_selected_batch_replay_binds_finalization_manifest():
    rows, _metrics = _synthetic_endpoint_rows_and_metrics()
    failed_rows = [dict(row) for row in rows[:64]]
    for index, row in enumerate(failed_rows):
        row["c10_failed_step"] = 1
        row["c10_failed_selected_row_index"] = index
    manifest = {
        "version": "c10_failed_selected_batch_manifest_v1",
        "failed_step": 1,
        "dynamic_finalization": {
            "kept_group_count": 4,
            "fallback_group_count": 0,
            "selected_raw_row_count": 63,
        },
        "selected_row_count": 64,
        "selected_rows_sha256": common.canonical_json_sha256(failed_rows),
    }
    manifest["evidence_sha256"] = common.canonical_json_sha256(manifest)
    failure = {
        "failed_step": 1,
        "failure_kind": "survivor_filter_contract_failure",
        "failure_details": {"health_gate": {"observed": {
            "selected_raw_row_count": 63,
            "materialized_selected_row_count": 64,
        }}},
    }
    c10_evidence._replay_failed_selected_batch(
        failed_rows, failure, manifest)
    tampered = dict(manifest)
    tampered["dynamic_finalization"] = dict(
        manifest["dynamic_finalization"], selected_raw_row_count=64)
    tampered["evidence_sha256"] = common.canonical_json_sha256({
        key: value for key, value in tampered.items()
        if key != "evidence_sha256"})
    with pytest.raises(ValueError, match="row count does not replay"):
        c10_evidence._replay_failed_selected_batch(
            failed_rows, failure, tampered)

    identity_tampered_rows = [
        dict(row, scientific_variant="tampered") for row in failed_rows]
    identity_manifest = dict(manifest)
    identity_manifest["selected_rows_sha256"] = \
        common.canonical_json_sha256(identity_tampered_rows)
    identity_manifest["evidence_sha256"] = common.canonical_json_sha256({
        key: value for key, value in identity_manifest.items()
        if key != "evidence_sha256"})
    with pytest.raises(ValueError, match="contract identity"):
        c10_evidence._replay_failed_selected_batch(
            identity_tampered_rows, failure, identity_manifest)


def test_failed_selected_batch_replays_nonfinite_group_counter():
    rows, _metrics = _synthetic_endpoint_rows_and_metrics()
    failed_rows = [dict(row) for row in rows[:64]]
    for index, row in enumerate(failed_rows):
        row["c10_failed_step"] = 1
        row["c10_failed_selected_row_index"] = index
    manifest = {
        "version": "c10_failed_selected_batch_manifest_v1",
        "failed_step": 1,
        "dynamic_finalization": {
            "kept_group_count": float("nan"),
            "fallback_group_count": 0,
            "selected_raw_row_count": 64,
        },
        "selected_row_count": 64,
        "selected_rows_sha256": common.canonical_json_sha256(failed_rows),
    }
    manifest["evidence_sha256"] = common.canonical_json_sha256(manifest)
    failure = {
        "failed_step": 1,
        "failure_kind": "selected_group_geometry_failure",
        "failure_details": {"health_gate": {"observed": {
            "kept_group_count": float("nan"),
            "fallback_group_count": 0,
        }}},
    }
    c10_evidence._replay_failed_selected_batch(
        failed_rows, failure, manifest)


def test_c10_structural_failure_is_detected_before_policy_update():
    args = SimpleNamespace(
        c10_execution_profile="canary", prompts_per_step=4,
        num_generations=16)
    rows, _metrics = _synthetic_endpoint_rows_and_metrics()
    selected = [dict(row) for row in rows[:64]]
    for row in selected:
        row["novelty_raw"] = row.pop("c10_rnd_raw_novelty_score")
        row.pop("c10_normalized_novelty_pre_gate")
        row.pop("expdis_stabilization_mode")
    stats = {
        "kept_group_count": 4,
        "fallback_group_count": 0,
        "selected_raw_row_count": 64,
    }
    grpo_gpu.validate_c10_selected_batch_before_update(args, selected, stats)

    bad_stats = dict(stats, selected_raw_row_count=63)
    with pytest.raises(
            grpo_gpu.C10TrainingHealthFailure,
            match="raw-row geometry") as captured:
        grpo_gpu.validate_c10_selected_batch_before_update(
            args, selected, bad_stats)
    assert captured.value.failure_kind == \
        "survivor_filter_contract_failure"

    for field, value in (
            ("kept_group_count", 4.9),
            ("fallback_group_count", 0.9),
            ("selected_raw_row_count", 64.9)):
        fractional_stats = dict(stats, **{field: value})
        with pytest.raises(grpo_gpu.C10TrainingHealthFailure):
            grpo_gpu.validate_c10_selected_batch_before_update(
                args, selected, fractional_stats)

    bad_rows = [dict(row) for row in selected]
    for row in bad_rows[:16]:
        row["clipped"] = True
    with pytest.raises(
            grpo_gpu.C10TrainingHealthFailure,
            match="survivor-consistent"):
        grpo_gpu.validate_c10_selected_batch_before_update(
            args, bad_rows, stats)

    all_wrong = [dict(row) for row in selected]
    first_group = all_wrong[:16]
    for row in first_group:
        row["completion_text"] = "The answer is \\boxed{987654321}."
        row["is_correct"] = False
        row["valid_answer"] = True
        row["correctness_reward"] = -1.0
        row["dynamic_sampling_eligibility_novelty_used"] = 0.0
        row["dynamic_sampling_eligibility_blended_reward"] = (
            -1.0 + common.soft_overlong_penalty(
                row["completion_token_length"], 13107, 3277))
    replay_rows = [{
        "is_correct": row["is_correct"],
        "clipped": row["clipped"],
        "blended_reward": row[
            "dynamic_sampling_eligibility_blended_reward"],
        "novelty_used": row[
            "dynamic_sampling_eligibility_novelty_used"],
    } for row in first_group]
    reason, eligible, facts = grpo_gpu.group_eligibility_for_rows(
        replay_rows, lambda_novelty=0.5, validity_mode="c8",
        survivor_only=True)
    assert reason == "all_wrong" and not eligible
    for row in first_group:
        row.update({
            "dynamic_sampling_reason": reason,
            "dynamic_sampling_raw_correct_count": facts[
                "raw_correct_count"],
            "dynamic_sampling_survivor_correct_count": facts[
                "survivor_correct_count"],
            "dynamic_sampling_reward_std": facts[
                "eligibility_reward_std"],
            "dynamic_sampling_novelty_contribution_std": facts[
                "eligibility_novelty_contribution_std"],
        })
    with pytest.raises(
            grpo_gpu.C10TrainingHealthFailure,
            match="ineligible group") as captured:
        grpo_gpu.validate_c10_selected_batch_before_update(
            args, all_wrong, stats)
    assert captured.value.failure_kind == "selected_batch_integrity_failure"


def _mutate_incorrect_novelty_failure(rows, metric):
    index = next(
        index for index, row in enumerate(rows)
        if not row["is_correct"])
    rows[index]["novelty_reward"] = 0.25
    rows[index]["blended_reward"] = (
        -1.0 + 0.5 * 0.25 + common.soft_overlong_penalty(
            rows[index]["completion_token_length"], 13107, 3277))
    pipeline_gpu._repair_c10_novelty_derived_evidence(rows, metric)
    return {"violations": [{"row_index": index, "novelty_reward": 0.25}]}


def _mutate_normalization_failure(rows, metric):
    serialized = []
    for row in rows:
        row["c10_normalized_novelty_pre_gate"] += 0.25
        pre_gate = row["c10_normalized_novelty_pre_gate"]
        novelty = pre_gate if row["is_correct"] else 0.0
        row["novelty_reward"] = novelty
        row["blended_reward"] = (
            (1.0 if row["is_correct"] else -1.0)
            + 0.5 * novelty
            + common.soft_overlong_penalty(
                row["completion_token_length"], 13107, 3277))
        serialized.append(pre_gate)
    mean = float(np.mean(serialized, dtype=np.float64))
    std = float(np.std(serialized, dtype=np.float64))
    metric["grpo/reward/novelty_normalized_selected_mean"] = mean
    metric["grpo/reward/novelty_normalized_selected_population_std"] = std
    pipeline_gpu._repair_c10_novelty_derived_evidence(rows, metric)
    return {
        "mean": mean,
        "population_std": std,
        "normalized_pre_gate_sha256": common.canonical_json_sha256(
            serialized),
    }


def test_terminal_normalization_accepts_exact_metric_only_failure():
    rows, metrics = _synthetic_endpoint_rows_and_metrics()
    terminal_rows = [dict(row) for row in rows[:64]]
    terminal_metric = dict(metrics[0])
    terminal_metric[
        "grpo/reward/novelty_normalized_selected_mean"] = 0.25
    serialized = [
        row["c10_normalized_novelty_pre_gate"] for row in terminal_rows]
    observed = {
        "mean": 0.25,
        "population_std": terminal_metric[
            "grpo/reward/novelty_normalized_selected_population_std"],
        "normalized_pre_gate_sha256": common.canonical_json_sha256(
            serialized),
    }
    pipeline_gpu._replay_c10_terminal_health_failure(
        terminal_rows, [terminal_metric], completed_step=1,
        failure_kind="novelty_normalization_failure", observed=observed)


def _failed_dynamic_rows_and_details():
    rows = []
    summaries = []
    attempts = []
    group_ordinal = 0
    for attempt in range(1, 65):
        for group_index in range(4):
            problem_id = f"p-{attempt}-{group_index}"
            group = [{
                "candidate_group_ordinal": group_ordinal,
                "row_index": row_index,
                "problem_id": problem_id,
                "dynamic_sampling_attempt": attempt,
                "dynamic_sampling_group_index": group_index,
                "clipped": False,
                "is_correct": False,
                "blended_reward": -1.0,
                "novelty_used": 0.0,
            } for row_index in range(16)]
            reason, eligible, facts = grpo_gpu.group_eligibility_for_rows(
                group, lambda_novelty=0.5, validity_mode="c8",
                survivor_only=True)
            assert not eligible
            for row in group:
                row.update({
                    "dynamic_sampling_reason": reason,
                    "dynamic_sampling_group_selected": False,
                    "dynamic_sampling_survivor_count":
                        facts["survivor_count"],
                    "dynamic_sampling_masked_count": facts["masked_count"],
                    "dynamic_sampling_raw_correct_count":
                        facts["raw_correct_count"],
                    "dynamic_sampling_survivor_correct_count":
                        facts["survivor_correct_count"],
                    "dynamic_sampling_eligibility_population":
                        facts["eligibility_population"],
                    "dynamic_sampling_eligibility_population_size":
                        facts["eligibility_population_size"],
                    "dynamic_sampling_reward_std":
                        facts["eligibility_reward_std"],
                    "dynamic_sampling_novelty_contribution_std":
                        facts["eligibility_novelty_contribution_std"],
                })
            rows.extend(group)
            summaries.append({
                "problem_id": problem_id,
                "attempt": attempt,
                "group_index": group_index,
                "reason": reason,
                "survivor_count": facts["survivor_count"],
                "masked_count": facts["masked_count"],
                "survivor_correct_count": facts["survivor_correct_count"],
                "eligibility_reward_std": facts["eligibility_reward_std"],
                "eligibility_novelty_contribution_std":
                    facts["eligibility_novelty_contribution_std"],
                "selected": False,
            })
            group_ordinal += 1
        attempts.append({
            "attempt": attempt,
            "requested_group_count": 4,
            "candidate_group_count": 4,
            "generation_error_group_count": 0,
            "generation_error_row_count": 0,
            "selected_group_count_after_attempt": 0,
        })
    details = {
        "dynamic_max_attempts": 64,
        "selected_group_count": 0,
        "needed_group_count": 4,
        "candidate_group_count": 64 * 4,
        "discarded_group_count": 64 * 4,
        "generation_error_group_count": 0,
        "generation_error_row_count": 0,
        "generation_requested_row_count": 64 * 4 * 16,
        "generation_attempt_summaries_sha256":
            common.canonical_json_sha256(attempts),
        "generation_attempt_summaries": attempts,
        "candidate_group_summaries_sha256":
            common.canonical_json_sha256(summaries),
        "candidate_group_summaries": summaries,
    }
    return rows, details


def test_dynamic_exhaustion_replays_exact_64_attempt_boundary():
    rows, details = _failed_dynamic_rows_and_details()
    c10_evidence._replay_failed_dynamic_candidates(rows, details)

    bad = dict(details)
    bad_attempts = [dict(item) for item in details[
        "generation_attempt_summaries"][:-1]]
    bad["generation_attempt_summaries"] = bad_attempts
    bad["generation_attempt_summaries_sha256"] = \
        common.canonical_json_sha256(bad_attempts)
    with pytest.raises(ValueError, match="retry boundary"):
        c10_evidence._replay_failed_dynamic_candidates(rows, bad)


def test_dynamic_transport_replays_all_4096_requested_rows_as_errors():
    attempts = [{
        "attempt": attempt,
        "requested_group_count": 4,
        "candidate_group_count": 0,
        "generation_error_group_count": 4,
        "generation_error_row_count": 64,
        "selected_group_count_after_attempt": 0,
    } for attempt in range(1, 65)]
    details = {
        "dynamic_max_attempts": 64,
        "selected_group_count": 0,
        "needed_group_count": 4,
        "candidate_group_count": 0,
        "discarded_group_count": 0,
        "generation_error_group_count": 256,
        "generation_error_row_count": 4096,
        "generation_requested_row_count": 4096,
        "generation_attempt_summaries_sha256":
            common.canonical_json_sha256(attempts),
        "generation_attempt_summaries": attempts,
        "candidate_group_summaries_sha256":
            common.canonical_json_sha256([]),
        "candidate_group_summaries": [],
    }
    c10_evidence._replay_failed_dynamic_candidates([], details)

    args = SimpleNamespace(
        orchestrator_run_name=c10_evidence.C10_CANONICAL_RUN_NAME,
        run_name=c10_evidence.C10_CANONICAL_RUN_NAME,
        num_generations=16,
        steps=50,
    )
    runtime = {"runtime": "bound"}
    code = {"file_manifest_sha256": "a" * 64}
    journal = grpo_gpu._signed_c10_dynamic_failure_journal(
        args, completed_step=0, failed_step=1,
        runtime_contract=runtime, code_artifact=code,
        details=details, candidate_rows=[])
    assert grpo_gpu._validate_c10_dynamic_failure_journal(
        journal, args, runtime_contract=runtime,
        code_artifact=code) is journal

    bad = dict(details)
    bad_attempts = [dict(item) for item in attempts]
    bad_attempts[-1]["generation_error_row_count"] = 63
    bad["generation_attempt_summaries"] = bad_attempts
    bad["generation_attempt_summaries_sha256"] = \
        common.canonical_json_sha256(bad_attempts)
    with pytest.raises(ValueError, match="aggregate does not replay"):
        c10_evidence._replay_failed_dynamic_candidates([], bad)


def test_c10_behavior_rollout_journal_reuses_only_complete_mode_prefixes():
    examples = [SimpleNamespace(problem_id=f"p-{index}")
                for index in range(c10_evidence.GATE_NUM_PROMPTS)]
    rows = []
    for mode in pipeline_gpu.sft_gate_gpu.PROMPT_MODES[:1]:
        for prompt_idx, example in enumerate(examples):
            for sample_idx in range(c10_evidence.GATE_SAMPLES_PER_PROMPT):
                rows.append({
                    "metric_contract_version":
                        pipeline_gpu.sft_gate_gpu.METRIC_CONTRACT_VERSION,
                    "problem_id": example.problem_id,
                    "prompt_mode": mode,
                    "sample_idx": sample_idx,
                    "sample_seed": (
                        c10_evidence.GATE_GENERATION_SEED
                        + prompt_idx * 100003 + sample_idx),
                    "finish_reason": "stop",
                    "generation_error_detail": None,
                    "completion_tokens": 2,
                    "completion_text": "\\boxed{1}",
                    "correctness_diagnostic_only": False,
                })
    assert pipeline_gpu._c10_completed_behavior_modes(rows, examples) == (
        pipeline_gpu.sft_gate_gpu.PROMPT_MODES[0],)
    rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(RuntimeError, match="fixed-panel prefix"):
        pipeline_gpu._c10_completed_behavior_modes(rows, examples)


def test_c10_panel_identity_is_checked_before_first_generation():
    source = inspect.getsource(pipeline_gpu._run_c9_behavior_gate)
    assert source.index("GATE_HELDOUT_SELECTION_SHA256") < source.index(
        "pool.generate(")
    assert "reusing persisted fixed-panel mode" in source
    assert "enforce_consecutive_failure_limit=not c10_mode" in source
    dynamic_source = inspect.getsource(grpo_gpu._dynamic_sample_step)
    assert "enforce_consecutive_failure_limit" in dynamic_source
    assert '!= "canary"' in dynamic_source


def test_dynamic_failure_journal_is_signed_and_recoverable():
    args = SimpleNamespace(
        orchestrator_run_name=c10_evidence.C10_CANONICAL_RUN_NAME,
        run_name=c10_evidence.C10_CANONICAL_RUN_NAME,
        num_generations=16,
        steps=50,
    )
    rows, details = _failed_dynamic_rows_and_details()
    runtime = {"runtime": "bound"}
    code = {"file_manifest_sha256": "a" * 64}
    journal = grpo_gpu._signed_c10_dynamic_failure_journal(
        args, completed_step=0, failed_step=1,
        runtime_contract=runtime, code_artifact=code,
        details=details, candidate_rows=rows)
    assert grpo_gpu._validate_c10_dynamic_failure_journal(
        journal, args, runtime_contract=runtime,
        code_artifact=code) is journal
    journal["candidate_rows"][0]["is_correct"] = True
    with pytest.raises(RuntimeError, match="signature mismatch"):
        grpo_gpu._validate_c10_dynamic_failure_journal(
            journal, args, runtime_contract=runtime,
            code_artifact=code)
