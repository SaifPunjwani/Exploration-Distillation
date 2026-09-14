"""Focused tests for the benchmark-independent SFT behavioral gate."""

from __future__ import annotations

import json

from tmx_gpu import sft_gate_gpu
from tmx_gpu.vllm_pool import Completion


class RecordingTokenizer:
    def __init__(self):
        self.calls: list[tuple[list[dict[str, str]], dict]] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return "\n".join(f"{m['role'].upper()}={m['content']}" for m in messages) + "\nGEN"

    def __call__(self, text, **_kwargs):
        return {"input_ids": text.split()}


def _accepted(tmp_path, rows):
    path = tmp_path / "accepted.jsonl"
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return str(path)


def test_heldout_selection_excludes_ids_and_exact_prompts_and_is_deterministic(tmp_path):
    tokenizer = RecordingTokenizer()
    raw = [
        {"id": f"p{i}", "prompt": f"problem {i}", "solution": str(i)}
        for i in range(8)
    ]
    from tmx_jax.data import render_chat_prompt

    prompt_p1 = render_chat_prompt("problem 1", tokenizer, enable_thinking=True)
    path = _accepted(
        tmp_path,
        [
            {"problem_id": "p0", "prompt_text": "unrelated"},
            {"problem_id": "legacy-id", "prompt_text": prompt_p1},
        ],
    )
    exclusions = sft_gate_gpu.load_accepted_exclusions([path])
    first, stats = sft_gate_gpu.select_heldout_examples(
        raw, tokenizer, exclusions, num_prompts=4, selection_seed=17
    )
    second, _ = sft_gate_gpu.select_heldout_examples(
        reversed(raw), tokenizer, exclusions, num_prompts=4, selection_seed=17
    )
    assert [row.problem_id for row in first] == [row.problem_id for row in second]
    assert not ({"p0", "p1"} & {row.problem_id for row in first})
    assert stats["excluded_by_problem_id"] == 1
    assert stats["excluded_by_prompt_hash"] == 1


def test_selection_fails_closed_without_accepted_identity(tmp_path):
    path = _accepted(tmp_path, [{"completion_text": "x"}])
    try:
        sft_gate_gpu.load_accepted_exclusions([path])
    except ValueError as exc:
        assert "cannot prove held-out disjointness" in str(exc)
    else:
        raise AssertionError("identity-free accepted row must fail closed")


def test_training_and_evaluation_share_the_qwen_user_prompt():
    tokenizer = RecordingTokenizer()
    from tmx_jax.data import SYSTEM_PROMPT as TRAINING_SYSTEM_PROMPT
    from tmx_jax.data import render_chat_prompt

    training_prompt = render_chat_prompt("solve me", tokenizer, enable_thinking=True)
    example = sft_gate_gpu.GateExample("p", "solve me", "4", training_prompt, "d")
    variants = sft_gate_gpu.render_prompt_variants(example, tokenizer)
    from tmx_gpu.eval_gpu import SYSTEM_PROMPT as PAPER_SYSTEM_PROMPT

    assert TRAINING_SYSTEM_PROMPT == PAPER_SYSTEM_PROMPT
    assert f"USER=solve me\n\n{PAPER_SYSTEM_PROMPT}" in variants["training"]
    assert variants["training"] == variants["paper_eval"]
    assert "SYSTEM=" not in variants["training"]


def test_completion_health_diagnostics_cover_clipping_repetition_and_think_closure():
    tokenizer = RecordingTokenizer()
    example = sft_gate_gpu.GateExample("p", "problem", "4", "prompt", "digest")
    repeated = "A sufficiently long repeated diagnostic line.\n" * 4
    bad = sft_gate_gpu.completion_record(
        example=example,
        prompt_mode="training",
        sample_idx=0,
        sample_seed=1,
        completion=Completion(
            text=f"<think>still reasoning\n{repeated}",
            finish_reason="length",
            completion_tokens=100,
        ),
        tokenizer=tokenizer,
        max_completion_tokens=100,
    )
    good_wrong = sft_gate_gpu.completion_record(
        example=example,
        prompt_mode="training",
        sample_idx=1,
        sample_seed=2,
        completion=Completion(
            text="<think>reason</think> Therefore \\boxed{5}.",
            finish_reason="stop",
            completion_tokens=12,
        ),
        tokenizer=tokenizer,
        max_completion_tokens=100,
    )
    assert bad["clipped"] and not bad["terminated"]
    assert bad["tail_repeated_line"] and bad["repetition_detected"]
    assert bad["unclosed_think"]
    assert good_wrong["terminated"] and good_wrong["valid_boxed_answer"]
    assert good_wrong["boxed_after_closed_think"]
    assert not good_wrong["correctness_diagnostic_only"]


def test_missing_finish_reason_does_not_prove_natural_termination():
    tokenizer = RecordingTokenizer()
    example = sft_gate_gpu.GateExample("p", "problem", "4", "prompt", "digest")
    row = sft_gate_gpu.completion_record(
        example=example,
        prompt_mode="paper_eval",
        sample_idx=0,
        sample_seed=1,
        completion=Completion(
            text="<think>reason</think> Therefore \\boxed{4}.",
            finish_reason="",
            completion_tokens=12,
        ),
        tokenizer=tokenizer,
        max_completion_tokens=100,
    )
    assert not row["clipped"]
    assert not row["terminated"]
    assert row["valid_boxed_answer"]


def _healthy_metrics(correctness):
    return {
        "termination_rate": 0.95,
        "clipped_rate": 0.05,
        "valid_boxed_answer_rate": 0.95,
        "median_completion_tokens": 100,
        "repetition_rate": 0.0,
        "unclosed_think_rate": 0.0,
        "generation_error_rate": 0.0,
        "correctness_rate_diagnostic_only": correctness,
    }


def test_gate_verdict_cannot_depend_on_correctness():
    low = {
        mode: _healthy_metrics(0.0)
        for mode in sft_gate_gpu.PROMPT_MODES
    }
    high = {
        mode: _healthy_metrics(1.0)
        for mode in sft_gate_gpu.PROMPT_MODES
    }
    low_gate = sft_gate_gpu.health_gate(low, max_completion_tokens=1000)
    high_gate = sft_gate_gpu.health_gate(high, max_completion_tokens=1000)
    assert low_gate == high_gate
    assert low_gate["passed"]
    assert low_gate["selection_uses_correctness"] is False
    assert all(
        "correct" not in criterion
        for result in low_gate["by_prompt_mode"].values()
        for criterion in result["criteria"]
    )


def test_both_prompt_modes_must_pass():
    metrics = {
        "training": _healthy_metrics(0.5),
        "paper_eval": {**_healthy_metrics(0.5), "clipped_rate": 0.9},
    }
    gate = sft_gate_gpu.health_gate(metrics, max_completion_tokens=1000)
    assert gate["by_prompt_mode"]["training"]["passed"]
    assert not gate["by_prompt_mode"]["paper_eval"]["passed"]
    assert not gate["passed"]
