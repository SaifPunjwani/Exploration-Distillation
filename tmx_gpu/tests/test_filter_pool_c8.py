"""Focused tests for the reward-independent c8 trajectory selector."""

from __future__ import annotations

import json
from pathlib import Path

from tmx_gpu import filter_pool


def make_row(
    *,
    problem_id: str,
    tokens: int = 500,
    reward: float = 1.0,
    text: str | None = None,
    clipped: bool = False,
    step: int = 1,
) -> dict:
    if text is None:
        text = " ".join(f"step {i} explores a fresh idea" for i in range(40))
    if "\\boxed{" not in text:
        text = f"{text} \\boxed{{4}}"
    return {
        "problem_id": problem_id,
        "prompt_text": f"prompt {problem_id}",
        "completion_text": text,
        "ground_truth": "4",
        "is_correct": True,
        "valid_answer": True,
        "terminated": True,
        "clipped": clipped,
        "completion_token_length": tokens,
        "blended_reward": reward,
        "reward": reward,
        "step": step,
    }


def write_trajectories(path: Path, rows: list[dict]) -> str:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return str(path)


def read_jsonl(path: str) -> list[dict]:
    with open(path) as handle:
        return [json.loads(line) for line in handle]


def test_one_shortest_clean_trajectory_per_problem(tmp_path):
    rows = [
        make_row(
            problem_id="p0",
            tokens=900,
            reward=1_000_000.0,
            text="long but enormous raw RND reward " * 80,
        ),
        make_row(
            problem_id="p0",
            tokens=300,
            reward=-1_000_000.0,
            text="short clean solution " * 40,
        ),
        make_row(
            problem_id="p0",
            tokens=200,
            reward=2_000_000.0,
            text="clipped candidate " * 40,
            clipped=True,
        ),
        make_row(problem_id="p1", tokens=400, reward=0.0),
        make_row(problem_id="p2", tokens=500, reward=0.0),
    ]
    path = write_trajectories(tmp_path / "t.jsonl", rows)
    output, stats = filter_pool.pool_trajectories(
        [path],
        policy=filter_pool.C8_POLICY,
        max_examples=10,
        output=str(tmp_path / "accepted.jsonl"),
    )
    accepted = read_jsonl(output)

    assert stats["clean_correct"] == 4
    assert stats["unique_clean_problems"] == 3
    assert stats["accepted_unique_problems"] == 3
    assert len({row["problem_id"] for row in accepted}) == len(accepted) == 3
    p0 = next(row for row in accepted if row["problem_id"] == "p0")
    assert p0["completion_token_length"] == 300
    assert p0["blended_reward"] == -1_000_000.0


def test_deterministic_under_input_reordering(tmp_path):
    rows = [
        make_row(
            problem_id=f"p{i // 2}",
            tokens=300 + i,
            reward=float((-1) ** i * i),
            text=f"candidate {i} " * 50,
            step=i,
        )
        for i in range(12)
    ]
    path = write_trajectories(tmp_path / "same.jsonl", rows)
    first_path, _ = filter_pool.pool_trajectories(
        [path],
        policy=filter_pool.C8_POLICY,
        max_examples=4,
        output=str(tmp_path / "first.jsonl"),
    )
    first = Path(first_path).read_text()

    write_trajectories(tmp_path / "same.jsonl", list(reversed(rows)))
    second_path, _ = filter_pool.pool_trajectories(
        [path],
        policy=filter_pool.C8_POLICY,
        max_examples=4,
        output=str(tmp_path / "second.jsonl"),
    )
    assert Path(second_path).read_text() == first


def test_global_cap_is_independent_of_rnd_reward_scale(tmp_path):
    rows = [
        make_row(problem_id=f"p{i}", tokens=400 + i, reward=float(i))
        for i in range(12)
    ]
    first_input = write_trajectories(tmp_path / "rewards_a.jsonl", rows)
    first_output, _ = filter_pool.pool_trajectories(
        [first_input],
        policy=filter_pool.C8_POLICY,
        max_examples=5,
        output=str(tmp_path / "a.jsonl"),
    )
    selected_a = [row["problem_id"] for row in read_jsonl(first_output)]

    # Reverse and magnify reward ordering while leaving trajectory quality and
    # problem identities unchanged. The selected problems must not move.
    for i, row in enumerate(rows):
        row["reward"] = row["blended_reward"] = float((12 - i) * 1_000_000)
    second_input = write_trajectories(tmp_path / "rewards_b.jsonl", list(reversed(rows)))
    second_output, _ = filter_pool.pool_trajectories(
        [second_input],
        policy=filter_pool.C8_POLICY,
        max_examples=5,
        output=str(tmp_path / "b.jsonl"),
    )
    selected_b = [row["problem_id"] for row in read_jsonl(second_output)]
    assert selected_b == selected_a


def test_optional_source_balance_round_robins_without_duplicates(tmp_path):
    scout1 = [make_row(problem_id=f"s1-{i}") for i in range(8)]
    scout2 = [make_row(problem_id=f"s2-{i}") for i in range(3)]
    path1 = write_trajectories(tmp_path / "scout1.jsonl", scout1)
    path2 = write_trajectories(tmp_path / "scout2.jsonl", scout2)
    output, stats = filter_pool.pool_trajectories(
        [path2, path1],
        policy=filter_pool.C8_POLICY,
        max_examples=6,
        source_balance=True,
        output=str(tmp_path / "balanced.jsonl"),
    )
    accepted = read_jsonl(output)

    assert len({row["problem_id"] for row in accepted}) == 6
    assert stats["accepted_per_source"] == {path1: 3, path2: 3}
    assert [row["source_file"] for row in accepted] == [path1, path2] * 3


def test_source_balance_does_not_override_per_problem_quality(tmp_path):
    path1 = write_trajectories(
        tmp_path / "scout1.jsonl",
        [
            make_row(
                problem_id="shared",
                tokens=250,
                text=" ".join(f"short reasoning step {i}" for i in range(80)),
            ),
            make_row(problem_id="only-1"),
        ],
    )
    path2 = write_trajectories(
        tmp_path / "scout2.jsonl",
        [
            make_row(
                problem_id="shared",
                tokens=900,
                text=" ".join(f"long wandering reasoning step {i}" for i in range(200)),
            ),
            make_row(problem_id="only-2"),
        ],
    )
    output, _ = filter_pool.pool_trajectories(
        [path1, path2],
        policy=filter_pool.C8_POLICY,
        max_examples=10,
        source_balance=True,
        output=str(tmp_path / "balanced.jsonl"),
    )
    shared = [row for row in read_jsonl(output) if row["problem_id"] == "shared"]
    assert len(shared) == 1
    assert shared[0]["completion_token_length"] == 250
    assert shared[0]["source_file"] == path1


def test_c8_uses_frozen_boxed_gates_not_qualitypool_length_or_loop_gates(tmp_path):
    loop = "x" * 40 * 4
    rows = [
        make_row(problem_id="short", tokens=12, text="brief \\boxed{4}"),
        make_row(problem_id="long", tokens=16383, text=f"{loop} \\boxed{{4}}"),
        {
            **make_row(problem_id="unboxed", tokens=200, text="the answer is 4"),
            "completion_text": "the answer is 4",
        },
        make_row(problem_id="wrong-box", tokens=200, text="reason \\boxed{5}"),
    ]
    path = write_trajectories(tmp_path / "gates.jsonl", rows)
    output, stats = filter_pool.pool_trajectories(
        [path], policy=filter_pool.C8_POLICY, max_examples=10,
        output=str(tmp_path / "accepted.jsonl"),
    )
    accepted_ids = {row["problem_id"] for row in read_jsonl(output)}
    assert accepted_ids == {"short", "long"}
    assert stats["clean_correct"] == 2


def test_exact_duplicate_prompts_deduped_across_dataset_ids(tmp_path):
    first = make_row(problem_id="id-a", tokens=300)
    second = make_row(problem_id="id-b", tokens=200)
    second["prompt_text"] = first["prompt_text"]
    path = write_trajectories(tmp_path / "dups.jsonl", [first, second])
    output, stats = filter_pool.pool_trajectories(
        [path], policy=filter_pool.C8_POLICY, max_examples=10,
        output=str(tmp_path / "accepted.jsonl"),
    )
    accepted = read_jsonl(output)
    assert len(accepted) == 1
    assert accepted[0]["problem_id"] == "id-b"
    assert stats["unique_clean_problems"] == 1


def test_original_blended_control_is_per_problem_not_global(tmp_path):
    rows = [
        make_row(problem_id="p0", tokens=200, reward=1.0),
        make_row(problem_id="p0", tokens=300, reward=9.0),
        make_row(problem_id="p1", tokens=250, reward=-1000.0),
    ]
    path = write_trajectories(tmp_path / "control.jsonl", rows)
    output, stats = filter_pool.pool_trajectories(
        [path], policy=filter_pool.ORIGINAL_BLENDED_POLICY, max_examples=10,
        output=str(tmp_path / "accepted.jsonl"),
    )
    accepted = read_jsonl(output)
    assert len(accepted) == 2
    p0 = next(row for row in accepted if row["problem_id"] == "p0")
    assert p0["blended_reward"] == 9.0
    assert {row["problem_id"] for row in accepted} == {"p0", "p1"}
    assert stats["accepted_unique_problems"] == 2
