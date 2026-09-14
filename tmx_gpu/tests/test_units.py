"""Unit tests: filter policies, reward math, metric math, failure handling."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os

import numpy as np
import pytest

from tmx_gpu import common, eval_gpu, filter_pool, grpo_gpu
from tmx_gpu.vllm_pool import Completion, VLLMPool, completion_failed


def make_row(*, correct=True, valid=True, terminated=True, clipped=False,
             tokens=500, reward=1.0, text=None, source="s1.jsonl"):
    if text is None:
        text = " ".join(f"step {i} explores a fresh idea" for i in range(40))
    return {
        "problem_id": "p0", "prompt_text": "prompt", "completion_text": text,
        "ground_truth": "4", "is_correct": correct, "valid_answer": valid,
        "terminated": terminated, "clipped": clipped,
        "completion_token_length": tokens, "blended_reward": reward,
        "reward": reward, "source_file": source, "step": 1,
    }


# ------------------------------------------------------------ filter_pool ----

class TestQualityGates:
    def test_clean_row_passes(self):
        assert filter_pool.passes_quality_gates(make_row())

    @pytest.mark.parametrize("field,value", [
        ("correct", False), ("valid", False), ("terminated", False), ("clipped", True),
    ])
    def test_flag_gates(self, field, value):
        assert not filter_pool.passes_quality_gates(make_row(**{field: value}))

    @pytest.mark.parametrize("tokens", [127, 16001])
    def test_token_bounds(self, tokens):
        assert not filter_pool.passes_quality_gates(make_row(tokens=tokens))
        assert filter_pool.passes_quality_gates(make_row(tokens=128))
        assert filter_pool.passes_quality_gates(make_row(tokens=16000))

    def test_looping_gate(self):
        chunk = "the same forty characters repeat again!!"  # not 40 yet
        chunk = chunk[:40].ljust(40, "x")
        assert not filter_pool.passes_quality_gates(make_row(text=chunk * 4))
        assert filter_pool.is_looping(chunk * 4)
        assert not filter_pool.is_looping(chunk * 3)


def _write_traj(tmp_path, name, rows):
    path = tmp_path / name
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != "source_file"}) + "\n")
    return str(path)


class TestPolicies:
    def test_naive_pool_is_tpu_collect_accepted(self, tmp_path):
        """correct-only + quality_first sort + cap (incorrect rows dropped,
        clipped/invalid correct rows KEPT but ranked below clean ones)."""
        rows = [
            make_row(correct=True, clipped=True, reward=5.0),   # correct but clipped
            make_row(correct=False),                            # dropped
            make_row(correct=True, reward=0.5),                 # clean, low reward
            make_row(correct=True, reward=2.0),                 # clean, high reward
        ]
        path = _write_traj(tmp_path, "t.jsonl", rows)
        out, stats = filter_pool.pool_trajectories(
            [path], policy="naive_pool", max_examples=3,
            output=str(tmp_path / "acc.jsonl"))
        accepted = [json.loads(l) for l in open(out)]
        assert stats["correct"] == 3 and stats["accepted"] == 3
        rewards = [r["blended_reward"] for r in accepted]
        # quality_first: unclipped rows first (desc reward), clipped last.
        assert rewards == [2.0, 0.5, 5.0]

    def test_quality_pool_gates_and_caps(self, tmp_path):
        rows = [make_row(reward=float(i)) for i in range(5)] + [make_row(clipped=True)]
        path = _write_traj(tmp_path, "t.jsonl", rows)
        out, stats = filter_pool.pool_trajectories(
            [path], policy="quality_pool", max_examples=3,
            output=str(tmp_path / "acc.jsonl"))
        accepted = [json.loads(l) for l in open(out)]
        assert stats["raw"] == 6 and stats["correct"] == 6
        assert stats["clean_correct"] == 5 and stats["accepted"] == 3
        assert [r["blended_reward"] for r in accepted] == [4.0, 3.0, 2.0]

    def test_best_scout_picks_best_single_file(self, tmp_path):
        p1 = _write_traj(tmp_path, "s1.jsonl", [make_row()] * 2)
        p2 = _write_traj(tmp_path, "s2.jsonl", [make_row()] * 5)
        out, stats = filter_pool.pool_trajectories(
            [p1, p2], policy="best_scout", max_examples=10,
            output=str(tmp_path / "acc.jsonl"))
        accepted = [json.loads(l) for l in open(out)]
        assert stats["accepted"] == 5
        assert {r["source_file"] for r in accepted} == {p2}


class TestPolicySurvivors:
    def test_best_scout_counts_max_per_source_not_sum(self, tmp_path):
        from tmx_gpu import pipeline_gpu

        p1 = _write_traj(tmp_path, "s1.jsonl", [make_row()] * 3)
        p2 = _write_traj(tmp_path, "s2.jsonl", [make_row()] * 2)
        assert pipeline_gpu._policy_survivors([p1, p2], "best_scout", 16000) == 3
        assert pipeline_gpu._policy_survivors([p1, p2], "quality_pool", 16000) == 5


# ------------------------------------------------------------- reward math ----

class TestSoftOverlong:
    def test_contract_geometry(self):
        assert common.soft_overlong_penalty(13107, 13107, 3277) == 0.0
        assert common.soft_overlong_penalty(100, 13107, 3277) == 0.0
        mid = common.soft_overlong_penalty(13107 + 1638, 13107, 3277)
        assert math.isclose(mid, -1638 / 3277, rel_tol=1e-9)
        assert common.soft_overlong_penalty(16384, 13107, 3277) == -1.0
        assert common.soft_overlong_penalty(50000, 13107, 3277) == -1.0  # clamped

    def test_disabled_when_unconfigured(self):
        assert common.soft_overlong_penalty(99999, 0, 3277) == 0.0
        assert common.soft_overlong_penalty(99999, 13107, 0) == 0.0


class TestCompletionLengthAccounting:
    class Tokenizer:
        def __call__(self, text, add_special_tokens=False):
            del add_special_tokens
            return {"input_ids": list(range(len(text.split())))}

    class Example:
        problem_id = "p0"
        prompt_text = "solve"
        ground_truth = "4"

    def test_vllm_usage_is_authoritative_for_overlong_shaping(self):
        row = grpo_gpu.score_completions(
            self.Tokenizer(), self.Example(),
            [Completion("brief answer \\boxed{4}", "stop", completion_tokens=14000)],
        )[0]
        assert row["completion_token_length"] == 14000
        assert row["completion_token_length_true"] == 14000
        assert row["completion_token_length_retokenized"] == 3
        assert row["terminated"] and not row["clipped"]
        assert row["overlong_penalty"] < 0.0

    def test_length_finish_forces_cap_and_missing_finish_fails_closed(self):
        length_row, missing_row = grpo_gpu.score_completions(
            self.Tokenizer(), self.Example(),
            [
                Completion("cut \\boxed{4}", "length", completion_tokens=15000),
                Completion("answer \\boxed{4}", "", completion_tokens=20),
            ],
        )
        assert length_row["completion_token_length_true"] == 16384
        assert length_row["completion_token_length"] == 16384
        assert length_row["clipped"] and not length_row["terminated"]
        assert length_row["overlong_penalty"] == -1.0
        assert not missing_row["terminated"]

    def test_full_completion_is_scored_and_tail_override_is_rejected(self):
        # The valid answer is intentionally more than 4,096 characters from
        # the end.  The retired tail-only scorer incorrectly marked this row
        # wrong even though the exact persisted completion and later C8 replay
        # both contain the boxed answer.
        text = "reasoning concludes \\boxed{4}\n" + ("harmless prose " * 400)
        row = grpo_gpu.score_completions(
            self.Tokenizer(), self.Example(), [Completion(text, "stop")],
        )[0]
        assert row["is_correct"] is True
        assert row["valid_answer"] is True
        with pytest.raises(ValueError, match="tail-only reward scoring is retired"):
            grpo_gpu.score_completions(
                self.Tokenizer(), self.Example(), [Completion(text, "stop")],
                answer_tail_chars=4096,
            )

    def test_trajectory_record_serializes_both_length_observations(self):
        row = grpo_gpu.score_completions(
            self.Tokenizer(), self.Example(),
            [Completion("brief answer \\boxed{4}", "stop",
                        completion_tokens=14000)],
        )[0]
        record = grpo_gpu.trajectory_record(row, 7)
        assert record["trajectory_schema_version"] == \
            grpo_gpu.TRAJECTORY_RECORD_SCHEMA_VERSION
        assert record["completion_token_length"] == 14000
        assert record["completion_token_length_true"] == 14000
        assert record["completion_token_length_retokenized"] == 3


class TestAdvantages:
    def test_group_mean_no_std_division(self):
        rewards = np.array([1.0, -1.0, 1.0, 1.0, 10.0, -10.0, 10.0, -10.0])
        adv = grpo_gpu.group_mean_advantages(rewards, 4)
        np.testing.assert_allclose(adv[:4], [0.5, -1.5, 0.5, 0.5])
        # Second group has 20x the std; values must NOT be normalized by it.
        np.testing.assert_allclose(adv[4:], [10.0, -10.0, 10.0, -10.0])

    def test_zero_for_uniform_group(self):
        np.testing.assert_allclose(grpo_gpu.group_mean_advantages(np.ones(4), 4), 0.0)


class TestDynamicSamplingRules:
    def test_informative_iff_mixed_correctness(self):
        for c in range(17):
            reason, eligible = grpo_gpu.group_reason(c, 16, 0.0, 0.0)
            assert eligible == (0 < c < 16)
        assert grpo_gpu.group_reason(0, 16, 0.0, 0.0) == ("all_wrong", False)
        assert grpo_gpu.group_reason(16, 16, 0.0, 0.0) == ("all_correct", False)

    def test_lambda_reward_std_escape(self):
        # lambda>0 + blended-reward std > 1e-6: all-wrong/all-correct accepted.
        assert grpo_gpu.group_reason(0, 16, 0.5, 0.5) == ("reward_informative_all_wrong", True)
        assert grpo_gpu.group_reason(16, 16, 0.5, 0.5) == ("reward_informative_all_correct", True)
        # std below threshold or lambda == 0: not accepted.
        assert not grpo_gpu.group_reason(0, 16, 1e-7, 0.5)[1]
        assert not grpo_gpu.group_reason(0, 16, 0.5, 0.0)[1]


# ------------------------------------------------------------- metric math ----

def brute_force_pass_at_k(n: int, c: int, k: int) -> float:
    """P(at least one correct in a uniformly random k-subset of n samples)."""
    flags = [True] * c + [False] * (n - c)
    subsets = list(itertools.combinations(range(n), k))
    return sum(any(flags[i] for i in sub) for sub in subsets) / len(subsets)


class TestEvalMathParity:
    def test_pass_at_k_unbiased(self):
        for n in (4, 6):
            for c in range(n + 1):
                for k in range(1, n + 1):
                    assert math.isclose(eval_gpu.pass_at_k(n, c, k),
                                        brute_force_pass_at_k(n, c, k), rel_tol=1e-12)

    def test_probe_average_is_avg_at_k(self):
        # avg@k = mean accuracy over the first k samples/problem; with a fixed
        # k for every problem it equals the flat mean over all flags.
        per_problem = [[True, False, True, True], [False, False, False, True]]
        expected = np.mean([np.mean(p) for p in per_problem])
        flat = [f for p in per_problem for f in p]
        assert math.isclose(grpo_gpu.probe_average(flat), expected)
        # No graded flags -> NaN, never a fake 0.0 paper-curve datapoint.
        assert math.isnan(grpo_gpu.probe_average([]))


# ------------------------------------------------------- failure handling ----

class TestErrorCompletions:
    def test_completion_failed(self):
        assert completion_failed(Completion(text="", finish_reason="error", error="boom"))
        assert not completion_failed(Completion(text="x", finish_reason="stop"))

    def test_pool_forwards_explicit_min_p_zero(self, monkeypatch):
        pool = VLLMPool.attach(["http://127.0.0.1:1"], "m")
        seen = []

        def fake_post(_prompt, _rid, sampling, _timeout):
            seen.append(dict(sampling))
            return Completion(text="ok", finish_reason="stop")

        monkeypatch.setattr(pool, "_post_one", fake_post)
        pool.generate(["p"], 1, min_p=0.0)
        assert seen == [{
            "max_tokens": 16384,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
        }]

    def test_consecutive_failure_threshold_raises(self, monkeypatch):
        pool = VLLMPool.attach(["http://127.0.0.1:1"], "m")
        pool.max_consecutive_failures = 3
        monkeypatch.setattr(
            pool, "_post_one",
            lambda *a, **k: Completion(text="", finish_reason="error", error="down"))
        for _ in range(2):
            pool.generate(["p"], 1)  # warns, counts
        with pytest.raises(RuntimeError, match="failing persistently"):
            pool.generate(["p"], 1)
        assert pool.consecutive_failures == 3

    def test_clean_call_resets_counter(self, monkeypatch):
        pool = VLLMPool.attach(["http://127.0.0.1:1"], "m")
        pool.max_consecutive_failures = 2
        bad = Completion(text="", finish_reason="error", error="down")
        good = Completion(text="ok", finish_reason="stop")
        replies = iter([bad, good, bad, good])
        monkeypatch.setattr(pool, "_post_one", lambda *a, **k: next(replies))
        for _ in range(4):
            pool.generate(["p"], 1)  # never two failures in a row
        assert pool.consecutive_failures == 0

    def test_caller_owned_error_rate_bypasses_legacy_call_counter(
            self, monkeypatch):
        pool = VLLMPool.attach(["http://127.0.0.1:1"], "m")
        pool.max_consecutive_failures = 3

        def one_low_rate_error(prompt, *_args, **_kwargs):
            if prompt == "bad":
                return Completion(
                    text="", finish_reason="error", error="transient")
            return Completion(text="ok", finish_reason="stop")

        monkeypatch.setattr(pool, "_post_one", one_low_rate_error)
        for _ in range(3):
            grouped = pool.generate(
                ["bad", *[f"good-{index}" for index in range(31)]], 1,
                enforce_consecutive_failure_limit=False)
            assert sum(completion_failed(item[0]) for item in grouped) == 1
        assert pool.consecutive_failures == 0

    def test_error_groups_never_scored_or_persisted(self, tiny_model_dir):
        """The NEVER-enter guarantee: an error-bearing group is dropped before
        scoring; a later attempt's clean group trains instead."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tiny_model_dir)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        class Ex:
            problem_id, prompt_text, ground_truth = "p0", "Compute 2 + 2. [gt=4]", "4"

        good_text = " ".join(f"step {i} reasons anew" for i in range(50)) + " \\boxed{4}."
        calls = {"n": 0}

        class FakePool:
            def generate(self, prompts, n, **kwargs):
                calls["n"] += 1
                if calls["n"] == 1:  # one failed request poisons the group
                    return [[Completion(text="", finish_reason="error", error="x")]
                            + [Completion(text=good_text, finish_reason="stop")] * (n - 1)
                            for _ in prompts]
                return [[Completion(text=good_text, finish_reason="stop"),
                         Completion(text=good_text, finish_reason="stop"),
                         Completion(text="wrong \\boxed{5}.", finish_reason="stop"),
                         Completion(text=good_text, finish_reason="stop")]
                        for _ in prompts]

        args = grpo_gpu.build_arg_parser().parse_args([
            "--output-dir", "/tmp/unused", "--phase", "scout",
            "--prompts-per-step", "1", "--num-generations", "4",
            "--device", "cpu",
        ])
        selected, all_rows, stats, _dt = grpo_gpu._dynamic_sample_step(
            rng=np.random.default_rng(0), pool=FakePool(), tokenizer=tokenizer,
            model=None, rnd=None, examples=[Ex()], args=args, lam=0.0, step=1)
        assert stats["generation_error_group_count"] == 1.0
        assert stats["generation_error_row_count"] == 1.0
        assert len(selected) == 4
        assert all(r["finish_reason"] != "error" for r in all_rows)
        assert calls["n"] == 2  # dropped group was regenerated on attempt 2


# ---------------------------------------------------------- crash recovery ----

class TestJsonlTrims:
    def test_trim_to_step(self, tmp_path):
        path = str(tmp_path / "t.jsonl")
        with open(path, "w") as f:
            for s in (1, 2, 3, 4):
                f.write(json.dumps({"step": s}) + "\n")
        common.trim_jsonl_to_step(path, 2)
        assert [json.loads(l)["step"] for l in open(path)] == [1, 2]

    def test_fresh_start_rotates(self, tmp_path):
        path = str(tmp_path / "t.jsonl")
        with open(path, "w") as f:
            f.write(json.dumps({"step": 9}) + "\n")
        common.trim_jsonl_to_step(path, 0)
        assert not os.path.exists(path)
        assert any(name.startswith("t.jsonl.stale.") for name in os.listdir(tmp_path))

    def test_trim_metrics_stage_scoped(self, tmp_path):
        path = str(tmp_path / "m.jsonl")
        logger = common.MetricsLogger(path)
        logger.log({"phase": "scout_grpo", "round": 1, "global_step": 1})
        logger.log({"phase": "scout_grpo", "round": 1, "global_step": 5})
        logger.log({"phase": "scout_grpo", "round": 2, "global_step": 9})
        logger.log({"phase": "central_sft", "round": 1, "global_step": 5})
        common.trim_metrics_stage(path, phase="scout_grpo", round_idx=1,
                                  keep_upto_global_step=1)
        rows = [json.loads(l) for l in open(path)]
        assert len(rows) == 3
        assert {(r["phase"], r["global_step"]) for r in rows} == {
            ("scout_grpo", 1), ("scout_grpo", 9), ("central_sft", 5)}


# ------------------------------------------------------------------- SFT ----

class _ToyTokenizer:
    eos_token_id = 99
    eos_token = "<eos>"
    truncation_side = "right"

    def __call__(self, text, *, truncation=False, max_length=None,
                 add_special_tokens=False):
        del add_special_tokens
        ids = [self.eos_token_id if x == self.eos_token else int(x)
               for x in text.split()]
        if truncation and max_length is not None and len(ids) > max_length:
            if self.truncation_side == "left":
                ids = ids[-max_length:]
            else:
                ids = ids[:max_length]
        return {"input_ids": ids}


class TestSftTerminationTokenization:
    def test_sft_save_step_parser_is_opt_in_sorted_and_strict(self):
        from tmx_gpu import sft_gpu

        assert sft_gpu.parse_sft_save_steps("") == ()
        assert sft_gpu.parse_sft_save_steps("250,125,250,1000") == (125, 250, 1000)
        with pytest.raises(argparse.ArgumentTypeError, match="positive"):
            sft_gpu.parse_sft_save_steps("0,125")
        with pytest.raises(argparse.ArgumentTypeError, match="comma-separated"):
            sft_gpu.parse_sft_save_steps("125,,250")

    def test_c8_reserves_and_supervises_exactly_one_eos(self):
        from tmx_gpu import sft_gpu

        tokenizer = _ToyTokenizer()
        rows = sft_gpu.pretokenize_sft_examples(
            tokenizer,
            [{"prompt_text": "1 2 3 4", "completion_text": "5 6 7"}],
            6,
        )
        row = rows[0]
        assert row["p_ids"] == [1, 2, 3, 4]
        assert row["c_ids"] == [5, tokenizer.eos_token_id]
        assert row["c_ids"].count(tokenizer.eos_token_id) == 1
        assert row["completion_truncated"]
        assert row["eos_supervised"]

    def test_c8_normalizes_existing_trailing_eos(self):
        from tmx_gpu import sft_gpu

        tokenizer = _ToyTokenizer()
        row = sft_gpu.pretokenize_sft_examples(
            tokenizer,
            [{"prompt_text": "1 2", "completion_text": "3 <eos> <eos>"}],
            8,
        )[0]
        assert row["c_ids"] == [3, tokenizer.eos_token_id]
        assert row["eos_already_present"]
        assert row["trailing_eos_removed"] == 2

    def test_legacy_mode_preserves_no_eos_c7_behavior(self):
        from tmx_gpu import sft_gpu

        tokenizer = _ToyTokenizer()
        row = sft_gpu.pretokenize_sft_examples(
            tokenizer,
            [{"prompt_text": "1 2 3 4", "completion_text": "5 6"}],
            5,
            termination_mode="legacy",
        )[0]
        assert row["p_ids"] == [1, 2, 3, 4]
        assert row["c_ids"] == [5]
        assert tokenizer.eos_token_id not in row["c_ids"]
        assert not row["eos_supervised"]

    def test_pipeline_wires_c8_default_and_sft_only_gate(self):
        from tmx_gpu import pipeline_gpu

        args = pipeline_gpu.build_arg_parser().parse_args([
            "--output-dir", "/tmp/out", "--run-name", "c8-test",
        ])
        assert args.policy == "coverage_pool_c8"
        assert args.rl_validity_mode == "c8"
        assert args.sft_termination_mode == "append_eos"
        assert args.sft_grad_accum == 1
        assert args.keep_checkpoints == 1
        assert args.mask_truncated is False
        assert args.harvest_max_passes == 0
        assert args.dynamic_max_attempts == 0
        assert args.sft_save_steps == ()
        assert args.sft_model is None
        assert args.sft_evidence_index == ""
        assert args.frozen_sft_library_evidence_index == ""
        assert args.scientific_variant == ""
        sargs = pipeline_gpu._sft_args(
            args, accepted="/tmp/accepted.jsonl", output_dir="/tmp/sft",
            round_idx=1, lam=0.5, global_step_offset=200,
        )
        assert sargs.sft_termination_mode == "append_eos"
        assert sargs.grad_accum == 1

        args.sft_save_steps = (125, 250, 500, 1000)
        checkpoint_sargs = pipeline_gpu._sft_args(
            args, accepted="/tmp/accepted.jsonl", output_dir="/tmp/sft",
            round_idx=1, lam=0.5, global_step_offset=200,
        )
        assert checkpoint_sargs.sft_save_steps == (125, 250, 500, 1000)
        args.sft_save_steps = ()

        gargs = pipeline_gpu._grpo_args(
            args, phase="scout", model=args.model, output_dir="/tmp/scout",
            lam=0.5, steps=200, round_idx=1, global_step_offset=0,
        )
        assert gargs.rl_validity_mode == "c8"
        assert gargs.keep_checkpoints == 1
        assert gargs.mask_truncated is False
        assert gargs.dynamic_max_attempts == 64

        args.rl_validity_mode = "legacy"
        legacy_gargs = pipeline_gpu._grpo_args(
            args, phase="scout", model=args.model, output_dir="/tmp/scout-legacy",
            lam=0.5, steps=200, round_idx=1, global_step_offset=0,
        )
        assert legacy_gargs.dynamic_max_attempts == 8
        args.rl_validity_mode = "c8"

        args.dynamic_max_attempts = 17
        explicit_gargs = pipeline_gpu._grpo_args(
            args, phase="scout", model=args.model, output_dir="/tmp/scout-explicit",
            lam=0.5, steps=200, round_idx=1, global_step_offset=0,
        )
        assert explicit_gargs.dynamic_max_attempts == 17
        args.dynamic_max_attempts = 0

        args.mask_truncated = True
        masked_gargs = pipeline_gpu._grpo_args(
            args, phase="scout", model=args.model, output_dir="/tmp/scout-masked",
            lam=0.5, steps=200, round_idx=1, global_step_offset=0,
        )
        assert masked_gargs.mask_truncated is True

        args.stop_after_sft = True
        args.rounds = 2
        with pytest.raises(ValueError, match="requires --rounds 1"):
            pipeline_gpu.validate_args(args)

class TestSftAccumulation:
    def test_partial_tail_is_flushed_and_counted(self, tiny_model_dir, tmp_path):
        """Six c8 rows at grad-accum 2 yield three optimizer steps.

        Even the prompt-heavy row remains trainable because EOS-aware
        tokenization reserves one completion-content token plus EOS.
        """
        from tmx_gpu import sft_gpu

        accepted = tmp_path / "accepted.jsonl"
        with open(accepted, "w") as f:
            for i in range(5):
                f.write(json.dumps({"prompt_text": f"question {i}?",
                                    "completion_text": f"answer {i} is fine."}) + "\n")
            # Degenerate: the prompt fills the whole budget -> no completion ids.
            f.write(json.dumps({"prompt_text": "word " * 200,
                                "completion_text": "never trained"}) + "\n")
        metrics_path = str(tmp_path / "metrics.jsonl")
        source_manifest = {
            "version": common.TRAJECTORY_SOURCE_DIGEST_VERSION,
            "artifacts": [{
                "artifact_ref": "entity/project/scout:v0",
                "artifact_digest": "artifact-digest",
                "files": [{"relative_path": "scout.jsonl", "sha256": "a" * 64}],
            }],
        }
        source_manifest_path = tmp_path / "trajectory_sources.json"
        source_manifest_path.write_text(json.dumps(source_manifest))
        args = sft_gpu.build_arg_parser().parse_args([
            "--model", tiny_model_dir, "--accepted", str(accepted),
            "--output-dir", str(tmp_path / "out"), "--epochs", "1",
            "--grad-accum", "2", "--max-total-len", "24",
            "--sft-save-steps", "1,3",
            "--trajectory-source-manifest", str(source_manifest_path),
            "--device", "cpu", "--metrics-jsonl", metrics_path,
        ])
        out_dir = sft_gpu.run_sft(args)
        assert os.path.isdir(out_dir)
        rows = [json.loads(l) for l in open(metrics_path)]
        steps = [r for r in rows if "sft/optimizer_steps" in r]
        assert steps and steps[-1]["sft/optimizer_steps"] == 3.0
        micro = [r for r in rows if "sft/loss" in r]
        assert len(micro) == 6
        setup = [r for r in rows if "sft/eos_supervised_examples" in r]
        assert setup[-1]["sft/eos_supervised_examples"] == 6.0
        assert setup[-1]["sft/planned_optimizer_steps"] == 3.0
        provenance = json.load(open(os.path.join(out_dir, "sft_provenance.json")))
        assert provenance["termination_mode"] == "append_eos"
        assert provenance["eos_supervised_examples"] == 6
        assert provenance["optimizer_steps"] == 3
        expected_digest = common.semantic_sft_jsonl_digest(str(accepted))
        assert provenance["accepted_library_digest"] == expected_digest
        assert provenance["accepted_file_sha256"] == common.file_sha256(
            str(accepted))
        expected_sequence = common.sft_training_sequence_digest(
            [json.loads(line) for line in accepted.read_text().splitlines()],
            epochs=1, seed=0)
        assert provenance["training_sequence_digest"] == expected_sequence
        assert provenance["trajectory_source"]["digest"] == \
            common.trajectory_source_manifest_digest(source_manifest)
        assert provenance["requested_save_optimizer_steps"] == [1, 3]
        assert provenance["saved_optimizer_steps"] == [1, 3]
        completed = provenance["completed_run_evidence"]
        unsigned_completed = dict(completed)
        unsigned_completed.pop("evidence_sha256")
        assert completed["evidence_sha256"] == \
            common.canonical_json_sha256(unsigned_completed)
        assert completed == json.load(open(tmp_path / "out" /
                                           "sft_completed_run.json"))
        runtime = provenance["sft_runtime_contract"]
        assert runtime["resolved_max_total_len"] == 24
        assert runtime["completion_budget_flag"] == 0
        assert runtime["seed"] == 0
        assert runtime["adam_beta2"] == 0.95
        for step in (1, 3):
            checkpoint_dir = tmp_path / "out" / f"optimizer_step_{step:06d}"
            assert (checkpoint_dir / "config.json").is_file()
            checkpoint_provenance = json.load(open(
                checkpoint_dir / "sft_checkpoint.json"))
            assert checkpoint_provenance["optimizer_step"] == step
            assert checkpoint_provenance["checkpoint_kind"] == \
                "post_optimizer_step_hf_model"
            assert checkpoint_provenance["accepted_library_digest"] == expected_digest
            assert checkpoint_provenance["accepted_file_sha256"] == \
                common.file_sha256(str(accepted))
            assert checkpoint_provenance["training_sequence_digest"] == \
                expected_sequence
            assert checkpoint_provenance["trajectory_source"]["digest"] == \
                common.trajectory_source_manifest_digest(source_manifest)
            assert checkpoint_provenance["micro_steps_seen"] == step * 2
            assert checkpoint_provenance["training_completed"] is True
            assert checkpoint_provenance["optimizer_steps_executed"] == 3
            assert checkpoint_provenance["micro_steps_executed"] == 6
            assert checkpoint_provenance["requested_save_optimizer_steps"] == [1, 3]
            assert checkpoint_provenance["saved_optimizer_steps"] == [1, 3]
            assert checkpoint_provenance["sft_runtime_contract"] == runtime
            assert checkpoint_provenance["completed_run_evidence"] == completed


class TestSftSemanticDigest:
    def test_digest_uses_prompt_completion_multiset_only(self):
        rows_a = [
            {"problem_id": "a", "prompt_text": "p1", "completion_text": "c1"},
            {"problem_id": "b", "prompt_text": "p2", "completion_text": "c2"},
        ]
        rows_b = [
            {"problem_id": "changed", "prompt_text": "p2", "completion_text": "c2"},
            {"irrelevant": 9, "prompt_text": "p1", "completion_text": "c1"},
        ]
        assert common.semantic_sft_dataset_digest(rows_a) == \
            common.semantic_sft_dataset_digest(rows_b)
        rows_b[0]["completion_text"] = "different"
        assert common.semantic_sft_dataset_digest(rows_a) != \
            common.semantic_sft_dataset_digest(rows_b)
