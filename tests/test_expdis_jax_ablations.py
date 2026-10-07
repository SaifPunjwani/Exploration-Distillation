"""CPU tests for the Appendix E ablation options."""

from __future__ import annotations

from dataclasses import replace
import json
import random
import sys

import pytest

from expdis_jax import data, pipeline
from expdis_jax.config import TrainConfig, parse_args, validate_contract
from expdis_jax.filtering import (
    MAX_GLOBAL_CAP,
    NAIVE_POOL_POLICY,
    QUALITY_POOL_POLICY,
    UNFILTERED_POLICY,
    pool_trajectory_files,
    resolve_selection_policy,
    select_naive_pool,
    select_unfiltered,
)
from expdis_jax.lineage import novelty_weight_for_round


def _row(problem="p0", answer="42", **overrides):
    row = {
        "problem_id": problem,
        "prompt_text": f"Problem {problem}",
        "completion_text": f"Work. \\boxed{{{answer}}}",
        "ground_truth": "42",
        "is_correct": answer == "42",
        "terminated": True,
        "valid_answer": True,
        "clipped": False,
        "completion_token_length": 100,
        "blended_reward": 1.0,
        "explorer_step": 1,
    }
    row.update(overrides)
    return row


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return str(path)


# Selection policies.

def test_policy_names_resolve_and_default_is_quality_pool():
    assert resolve_selection_policy(TrainConfig().accepted_selection_policy) == QUALITY_POOL_POLICY
    assert resolve_selection_policy(QUALITY_POOL_POLICY) == QUALITY_POOL_POLICY
    assert resolve_selection_policy(" Naive_Pool ") == NAIVE_POOL_POLICY
    assert resolve_selection_policy("unfiltered") == UNFILTERED_POLICY
    for bad in ("", "quality_first", "chronological"):
        with pytest.raises(ValueError):
            resolve_selection_policy(bad)


def test_naive_pool_keeps_every_correct_row_without_quality_gates():
    rows = [
        _row("p0", completion_token_length=300),
        _row("p0", completion_token_length=200),          # same problem: kept
        _row("p1", terminated=False),                     # nonterminated: kept
        _row("p2", clipped=True),                         # clipped: kept
        _row("p3", valid_answer=False),                   # invalid answer: kept
        _row("p4", completion_text="Work. 42 " * 30),     # no boxed answer: kept
        _row("p5", completion_text="ab" * 400),           # repetition loop: kept
        _row("p6", answer="7"),                           # incorrect: dropped
        _row("p7", completion_text=None),                 # cannot be distilled
    ]
    selected, stats = select_naive_pool(rows)
    assert len(selected) == 7
    assert all(row["is_correct"] for row in selected)
    assert stats["policy"] == NAIVE_POOL_POLICY
    assert stats["rejected"] == {"incorrect": 1, "missing_metadata": 1}
    assert stats["accepted_unique_problems"] == 6


def test_naive_pool_cap_order_matches_original_ranking():
    rows = [
        _row("a", blended_reward=5.0, clipped=True),
        _row("b", blended_reward=0.5),
        _row("c", blended_reward=2.0),
        _row("d", blended_reward=2.0, completion_token_length=50),
        _row("e", blended_reward=9.0, terminated=False),
        _row("f", answer="1"),
    ]
    selected, stats = select_naive_pool(rows, max_examples=4)
    # Terminated, valid, unclipped rows first by reward then length; clipped
    # and nonterminated rows only fill the remainder.
    assert [row["problem_id"] for row in selected] == ["d", "c", "b", "a"]
    assert stats["eligible"] == 5 and stats["capped_out"] == 1
    shuffled = rows[:]
    random.Random(3).shuffle(shuffled)
    assert select_naive_pool(shuffled, max_examples=4)[0] == selected


def test_unfiltered_keeps_rows_regardless_of_correctness():
    rows = [_row("p0"), _row("p0", answer="1"), _row("p1", answer="2", terminated=False, clipped=True),
            _row("p2", prompt_text=None)]
    selected, stats = select_unfiltered(rows)
    assert len(selected) == 3
    assert stats["accepted_correct"] == 1
    assert stats["rejected"] == {"missing_metadata": 1}
    assert stats["policy"] == UNFILTERED_POLICY


def test_unfiltered_cap_is_a_fixed_subset_independent_of_input_order():
    rows = [_row(f"p{i}", answer=str(41 + i % 2), explorer_step=i) for i in range(200)]
    selected, stats = select_unfiltered(rows, max_examples=50)
    assert len(selected) == 50 and stats["capped_out"] == 150
    reversed_selected, _ = select_unfiltered(list(reversed(rows)), max_examples=50)
    assert reversed_selected == selected
    # Not simply the leading rows, and not ranked by correctness.
    assert selected != rows[:50]
    assert 0 < stats["accepted_correct"] < 50


@pytest.mark.parametrize("select", [select_naive_pool, select_unfiltered])
def test_ablation_policies_keep_the_500_row_cap(select):
    rows = [_row(f"p{i}") for i in range(MAX_GLOBAL_CAP + 20)]
    selected, stats = select(rows)
    assert len(selected) == MAX_GLOBAL_CAP
    with pytest.raises(ValueError):
        select(rows, max_examples=MAX_GLOBAL_CAP + 1)


def test_pool_files_dispatches_on_policy(tmp_path):
    first = _write(tmp_path / "a.jsonl", [_row("p0"), _row("p0", completion_token_length=50)])
    second = _write(tmp_path / "b.jsonl", [_row("p1", answer="3"), _row("p2", clipped=True)])
    counts = {}
    for policy in (QUALITY_POOL_POLICY, NAIVE_POOL_POLICY, UNFILTERED_POLICY):
        selected, stats = pool_trajectory_files([second, first], policy=policy)
        counts[policy] = len(selected)
        assert stats["policy"] == policy
    assert counts == {QUALITY_POOL_POLICY: 1, NAIVE_POOL_POLICY: 3, UNFILTERED_POLICY: 4}


@pytest.mark.parametrize("policy,expected", [
    ("coverage_pool_c8", 1), ("naive_pool", 3), ("unfiltered", 4),
])
def test_collect_accepted_honors_configured_policy(tmp_path, policy, expected):
    paths = [
        _write(tmp_path / "a.jsonl", [_row("p0"), _row("p0", completion_token_length=50)]),
        _write(tmp_path / "b.jsonl", [_row("p1", answer="3"), _row("p2", clipped=True)]),
    ]
    cfg = replace(TrainConfig(), accepted_selection_policy=policy)
    out = pipeline.collect_accepted(cfg, paths, output_path=str(tmp_path / "accepted.jsonl"))
    assert len(open(out).read().splitlines()) == expected
    funnel = json.loads(open(out + ".funnel.json").read())
    assert funnel["policy"] == resolve_selection_policy(policy)


# Unfiltered and NaivePool through the real SFT stage.

@pytest.mark.parametrize("policy,accepted,incorrect", [
    ("naive_pool", 4, 0), ("unfiltered", 8, 4),
])
def test_ablation_pool_reaches_student_sft(monkeypatch, tmp_path, policy, accepted, incorrect):
    import test_expdis_jax_pipeline_cpu as tiny_pipeline

    run = tiny_pipeline.run_tiny_pipeline(
        monkeypatch, tmp_path, selection_policy=policy, stop_after_sft=True,
    )
    rows = [json.loads(line) for line in (run / "trajectory_library.accepted.jsonl").read_text().splitlines()]
    assert len(rows) == accepted
    assert sum(not row["is_correct"] for row in rows) == incorrect
    assert (run / "actual/actual_sft_final").is_dir()
    assert not (run / "actual/grpo").exists()


# Filtered SFT, no RL.

def test_stop_after_sft_evaluates_the_served_sft_student(monkeypatch, tmp_path):
    import test_expdis_jax_pipeline_cpu as tiny_pipeline
    from expdis_jax import eval as evaluation

    reloads = []
    evals = []
    monkeypatch.setattr(pipeline, "_hf_artifact_repo", lambda: "owner/repo")
    monkeypatch.setattr(pipeline, "_maybe_upload_path_to_hf", lambda *a, **kw: True)

    def reload(gcs_path, cfg, **kw):
        reloads.append(kw["stage"])
        return "http://sft-server/v1"

    def run_eval(tok, server_urls, **kw):
        evals.append(server_urls)
        return {"avg_at_n": 0.25, "num_problems": 30, "num_rollouts": kw["num_rollouts"]}

    monkeypatch.setattr(pipeline, "_maybe_reload_vllm_slice", reload)
    monkeypatch.setattr(evaluation, "run_eval", run_eval)
    run = tiny_pipeline.run_tiny_pipeline(
        monkeypatch, tmp_path, stop_after_sft=True,
        env={"EXPDIS_PIPELINE_EVAL_AFTER_SFT": "1", "EXPDIS_HF_MIRROR_FINAL_EXPORTS": "1",
             "EXPDIS_HF_MIRROR_FINAL_EVAL": "0"},
    )
    assert reloads == ["actual_sft"]
    assert evals == [["http://sft-server/v1"]]
    result = json.loads((run / "final_eval_aime24.json").read_text())
    assert result["avg_at_n"] == 0.25
    assert result["protocol"]["evaluated_stage"] == "actual_sft"
    assert result["protocol"]["source_checkpoint"].endswith("/actual_sft_hf")
    assert (run / "actual/actual_sft_hf/model.safetensors").is_file()
    assert not (run / "actual/grpo").exists()


def test_eval_after_sft_refuses_unreloaded_servers(monkeypatch, tmp_path):
    import test_expdis_jax_pipeline_cpu as tiny_pipeline

    with pytest.raises(RuntimeError, match="EXPDIS_PIPELINE_EVAL_AFTER_SFT"):
        tiny_pipeline.run_tiny_pipeline(
            monkeypatch, tmp_path, stop_after_sft=True,
            env={"EXPDIS_PIPELINE_EVAL_AFTER_SFT": "1"},
        )


# Training-set subsets (Table 9).

def test_first_subset_is_the_existing_behavior():
    assert data.subset_indices(10, 4) == [0, 1, 2, 3]
    assert data.subset_indices(10, None) == list(range(10))
    assert data.subset_indices(3, 10, "random", 5) == [0, 1, 2]


def test_random_subset_is_seeded_uniform_and_in_source_order():
    picked = data.subset_indices(40_315, 17_000, "random", 0)
    assert len(picked) == len(set(picked)) == 17_000
    assert picked == sorted(picked)
    assert picked == data.subset_indices(40_315, 17_000, "random", 0)
    assert picked != data.subset_indices(40_315, 17_000, "random", 1)
    assert picked[-1] > 17_000  # drawn from the whole pool
    with pytest.raises(ValueError):
        data.subset_indices(10, 4, "head")


class _Tokenizer:
    def apply_chat_template(self, messages, **kw):
        return messages[-1]["content"]


def test_deepscaler_loader_applies_random_subset(monkeypatch):
    from datasets import Dataset

    raw = Dataset.from_list([
        {"problem": f"q{i}", "answer": str(i), "solution": ""} for i in range(50)
    ])
    monkeypatch.setattr(data, "_load_raw_deepscaler", lambda: raw)
    for name in ("EXPDIS_TRAIN_DATASET_JSONL", "EXPDIS_DATASET_SHARD_COUNT"):
        monkeypatch.delenv(name, raising=False)
    first = data.load_examples("deepscaler", _Tokenizer(), max_examples=17)
    assert [ex.ground_truth for ex in first] == [str(i) for i in range(17)]
    sampled = data.load_examples(
        "deepscaler", _Tokenizer(), max_examples=17, subset_policy="random", subset_seed=0,
    )
    expected = data.subset_indices(50, 17, "random", 0)
    assert [ex.ground_truth for ex in sampled] == [str(i) for i in expected]


def test_jsonl_loader_applies_random_subset(monkeypatch, tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text("".join(
        json.dumps({"problem": f"q{i}", "answer": str(i)}) + "\n" for i in range(30)
    ))
    monkeypatch.delenv("EXPDIS_DATASET_SHARD_COUNT", raising=False)
    examples = data.load_jsonl_examples(
        str(path), _Tokenizer(), max_examples=10, subset_policy="random", subset_seed=2,
    )
    assert [ex.ground_truth for ex in examples] == [
        str(i) for i in data.subset_indices(30, 10, "random", 2)
    ]


def test_subset_options_are_cli_flags_and_validated(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "prog", "--train-subset-policy", "random", "--train-subset-seed", "7",
        "--dataset-name", "deepscaler", "--max-train-examples", "17000",
        "--accepted-selection-policy", "naive_pool",
    ])
    cfg = parse_args()
    assert (cfg.train_subset_policy, cfg.train_subset_seed) == ("random", 7)
    assert cfg.accepted_selection_policy == "naive_pool"
    validate_contract(cfg, require_eval_contract=True)
    with pytest.raises(ValueError, match="train_subset_policy"):
        validate_contract(replace(cfg, train_subset_policy="head"))


# Fixed versus annealed lambda (Appendix E).

def test_fixed_and_annealed_lambda_across_rounds():
    fixed = [novelty_weight_for_round(scalar_weight=0.5, schedule="", rounds=4, round_index=r)
             for r in range(1, 5)]
    annealed = [novelty_weight_for_round(scalar_weight=0.5, schedule="0.75,0.50,0.35,0.25",
                                         rounds=4, round_index=r) for r in range(1, 5)]
    assert fixed == [0.5] * 4
    assert annealed == [0.75, 0.5, 0.35, 0.25]
