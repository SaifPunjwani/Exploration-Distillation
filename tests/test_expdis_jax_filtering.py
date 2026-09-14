"""Contract tests for the dependency-light JAX MR-ME QualityPool."""

from __future__ import annotations

import json
import random
import unittest

try:
    import pytest
except ImportError:  # unittest-discover without pytest: skip cleanly
    raise unittest.SkipTest("pytest not installed") from None

from expdis_jax.filtering import (
    DEFAULT_MAX_COMPLETION_TOKENS,
    DEFAULT_MIN_COMPLETION_TOKENS,
    MAX_GLOBAL_CAP,
    QUALITY_POOL_POLICY,
    REPETITION_SCREEN_VERSION,
    exact_problem_key,
    is_looping,
    passes_quality_gates,
    pool_trajectory_files,
    rejection_reason,
    select_quality_pool,
)


def _row(**overrides):
    row = {
        "problem_id": "problem-0",
        "prompt_text": "Compute one particular exact answer.",
        "completion_text": (
            "We simplify the expression carefully and verify the result. "
            "Therefore the final answer is \\boxed{42}."
        ),
        "ground_truth": "42",
        "is_correct": True,
        "terminated": True,
        "valid_answer": True,
        "clipped": False,
        "completion_token_length": 500,
        "blended_reward": 1.0,
        "novelty_reward": 0.25,
        "explorer_step": 7,
    }
    row.update(overrides)
    return row


def test_clean_row_passes_complete_contract():
    assert rejection_reason(_row()) is None
    assert passes_quality_gates(_row())


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"is_correct": False}, "incorrect"),
        ({"terminated": False}, "nonterminated"),
        ({"valid_answer": False}, "invalid_answer"),
        ({"completion_text": None}, "missing_metadata"),
        ({"ground_truth": None}, "missing_metadata"),
        ({"prompt_text": "", "problem_id": ""}, "missing_metadata"),
        ({"completion_text": "The final answer is 42."}, "no_boxed_answer"),
        ({"completion_text": "The final answer is \\boxed{41}."}, "boxed_answer_mismatch"),
        ({"completion_text": "The final answer is \\boxed{42."}, "no_boxed_answer"),
        ({"clipped": True}, "clipped"),
        ({"completion_token_length": None}, "missing_metadata"),
        ({"completion_token_length": -1}, "missing_metadata"),
        ({"completion_token_length": 127}, "too_short"),
        ({"completion_token_length": 4_501}, "too_long"),
    ],
)
def test_each_hard_gate_fails_closed(overrides, reason):
    assert rejection_reason(_row(**overrides)) == reason


def test_final_balanced_box_is_the_answer_that_must_match():
    assert rejection_reason(_row(
        completion_text="A discarded guess was \\boxed{41}; the result is \\boxed{42}."
    )) is None
    assert rejection_reason(_row(
        completion_text="A discarded guess was \\boxed{42}; the result is \\boxed{41}."
    )) == "boxed_answer_mismatch"


def test_boxed_match_reuses_exact_numeric_verifier():
    row = _row(
        completion_text=r"A reduction gives the result \boxed{\frac{1}{2}}.",
        ground_truth="0.5",
    )
    assert rejection_reason(row) is None


def test_length_window_is_inclusive_and_requires_explicit_tokens():
    assert DEFAULT_MIN_COMPLETION_TOKENS == 128
    assert DEFAULT_MAX_COMPLETION_TOKENS == 4_500
    assert passes_quality_gates(_row(
        completion_token_length=DEFAULT_MIN_COMPLETION_TOKENS
    ))
    assert passes_quality_gates(_row(
        completion_token_length=DEFAULT_MAX_COMPLETION_TOKENS
    ))
    missing = _row()
    del missing["completion_token_length"]
    assert rejection_reason(missing) == "missing_metadata"


def test_frozen_repetition_screen_boundary():
    chunk = "abcdefghij0123456789ABCDEFGHIJ9876543210"
    assert len(chunk) == 40
    assert not is_looping(chunk * 3)
    assert is_looping(chunk * 4)
    loop = f"{chunk * 4} The final answer is \\boxed{{42}}."
    assert rejection_reason(_row(completion_text=loop)) == "looping"
    assert REPETITION_SCREEN_VERSION == "consecutive_chars_c40_r4_v1"


def test_shortest_clean_winner_ignores_every_reward_field():
    prompt = "The same exact problem."
    short = _row(
        prompt_text=prompt,
        problem_id="short-id",
        completion_token_length=200,
        blended_reward=-999.0,
        novelty_reward=-999.0,
        completion_text="A concise derivation concludes \\boxed{42}.",
    )
    long = _row(
        prompt_text=prompt,
        problem_id="long-id",
        completion_token_length=800,
        blended_reward=999.0,
        novelty_reward=999.0,
        completion_text=("A longer but still clean derivation. " * 3) + "\\boxed{42}.",
    )
    selected, stats = select_quality_pool([long, short])
    assert selected == [short]
    assert stats["duplicate_clean_rows_removed"] == 1
    # Reward metadata is not even required for selection.
    rewardless = _row(prompt_text="A second problem.")
    rewardless.pop("blended_reward")
    rewardless.pop("novelty_reward")
    assert passes_quality_gates(rewardless)


def test_exact_prompt_identity_precedes_dataset_id():
    first = _row(prompt_text="duplicate prompt", problem_id="dataset-row-a")
    second = _row(prompt_text="duplicate prompt", problem_id="dataset-row-b")
    third = _row(prompt_text="different prompt", problem_id="dataset-row-a")
    assert exact_problem_key(first) == exact_problem_key(second)
    assert exact_problem_key(first) != exact_problem_key(third)
    selected, _ = select_quality_pool([first, second, third])
    assert len(selected) == 2


def test_selection_is_independent_of_input_order_and_globally_capped():
    rows = [
        _row(
            problem_id=f"p-{index}",
            prompt_text=f"Exact problem {index}",
            completion_text=f"A clean solution for item {index} gives \\boxed{{42}}.",
            completion_token_length=200 + (index % 11),
        )
        for index in range(530)
    ]
    shuffled = rows[:]
    random.Random(17).shuffle(shuffled)
    selected_a, stats_a = select_quality_pool(rows)
    selected_b, stats_b = select_quality_pool(shuffled)
    assert len(selected_a) == MAX_GLOBAL_CAP
    assert [row["prompt_text"] for row in selected_a] == [
        row["prompt_text"] for row in selected_b
    ]
    assert stats_a == stats_b
    assert stats_a["unique_clean_problems"] == 530
    assert stats_a["capped_out"] == 30


def test_cap_cannot_silently_exceed_frozen_global_ceiling():
    with pytest.raises(ValueError, match=r"\[0, 500\]"):
        select_quality_pool([_row()], max_examples=501)
    selected, stats = select_quality_pool([_row()], max_examples=0)
    assert selected == []
    assert stats["accepted"] == 0 and stats["capped_out"] == 1


def test_merged_k_inputs_have_deterministic_output_funnel_and_source_stats(tmp_path):
    source_a = tmp_path / "scout-a.jsonl"
    source_b = tmp_path / "scout-b.jsonl"
    output_a = tmp_path / "accepted-a.jsonl"
    output_b = tmp_path / "accepted-b.jsonl"

    shared_long = _row(
        prompt_text="shared exact problem",
        problem_id="shared-a",
        completion_token_length=700,
        completion_text="A long clean route eventually reaches \\boxed{42}.",
    )
    unique_a = _row(
        prompt_text="unique to scout a",
        problem_id="a",
        completion_text="Scout A derives \\boxed{42}.",
    )
    shared_short = _row(
        prompt_text="shared exact problem",
        problem_id="shared-b",
        completion_token_length=200,
        completion_text="A short clean route reaches \\boxed{42}.",
    )
    rejected_b = _row(
        prompt_text="bad row",
        problem_id="bad",
        is_correct=False,
    )
    source_a.write_text(
        "".join(json.dumps(row) + "\n" for row in [shared_long, unique_a]),
        encoding="utf-8",
    )
    source_b.write_text(
        "".join(json.dumps(row) + "\n" for row in [rejected_b, shared_short]),
        encoding="utf-8",
    )

    selected_a, stats_a = pool_trajectory_files(
        [source_b, source_a], output_path=output_a
    )
    selected_b, stats_b = pool_trajectory_files(
        [source_a, source_b], output_path=output_b
    )
    assert [row["completion_text"] for row in selected_a] == [
        row["completion_text"] for row in selected_b
    ]
    assert shared_short["completion_text"] in {
        row["completion_text"] for row in selected_a
    }
    assert len(selected_a) == 2
    assert output_a.read_bytes() == output_b.read_bytes()
    assert stats_a == stats_b
    assert stats_a["policy"] == QUALITY_POOL_POLICY
    assert stats_a["raw"] == 4
    assert stats_a["clean_correct"] == 3
    assert stats_a["unique_clean_problems"] == 2
    assert stats_a["rejected"]["incorrect"] == 1
    assert stats_a["source_counts"]["raw"] == {
        str(source_a.resolve()): 2,
        str(source_b.resolve()): 2,
    }
    assert stats_a["source_counts"]["accepted"] == {
        str(source_a.resolve()): 1,
        str(source_b.resolve()): 1,
    }


def test_merged_loader_fails_closed_on_malformed_json(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid trajectory JSON"):
        pool_trajectory_files([bad])
