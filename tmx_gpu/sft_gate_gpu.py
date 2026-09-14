#!/usr/bin/env python3
"""Benchmark-independent behavioral gate for an SFT checkpoint served by vLLM.

This gate deliberately does *not* select checkpoints by benchmark accuracy.  It
uses a deterministic set of DAPO-Math-17k problems that is disjoint from the
accepted SFT JSONL and measures whether the model still behaves like a usable
reasoner after distillation:

* termination / clipping;
* valid ``\\boxed{}`` answer presentation;
* completion length;
* repeated-output pathologies; and
* Qwen-style ``<think>`` / ``</think>`` closure.

Every held-out problem is generated twice with the same sampling seeds:

``training``
    The exact training system prompt and renderer from :mod:`tmx_jax.data`.

``paper_eval``
    The exact paper-evaluation system prompt and renderer from
    :mod:`tmx_gpu.eval_gpu`.

Correctness is reported as a diagnostic, but it is intentionally absent from
``health_gate.criteria`` and from held-out problem selection.  The script only
attaches to already-running vLLM OpenAI servers; it never launches, reloads, or
changes a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tmx_gpu import common  # noqa: E402
from tmx_gpu.filter_pool import is_looping  # noqa: E402
from tmx_gpu.vllm_pool import Completion, VLLMPool, completion_failed  # noqa: E402
from tmx_jax.rewarding import (  # noqa: E402
    _extract_boxed_candidates,
    is_correct_completion,
)

PROMPT_MODES = ("training", "paper_eval")
METRIC_CONTRACT_VERSION = "sft_behavior_gate_v1"

# Frozen, benchmark-independent health thresholds.  These are deliberately
# about behavior, not answer accuracy.  Values are CLI-visible/overridable so
# an experiment can pre-register a different contract without editing code.
DEFAULT_MIN_TERMINATION_RATE = 0.80
DEFAULT_MAX_CLIPPED_RATE = 0.20
DEFAULT_MIN_VALID_BOXED_RATE = 0.90
DEFAULT_MAX_MEDIAN_TOKEN_FRACTION = 0.75
# Frozen Base-calibrated value (7 repeated generations out of 64 paper-prompt
# samples), set before inspecting any SFT checkpoint gate.
DEFAULT_MAX_REPETITION_RATE = 7 / 64
DEFAULT_MAX_UNCLOSED_THINK_RATE = 0.20
DEFAULT_MAX_GENERATION_ERROR_RATE = 0.05

TERMINATED_FINISH_REASONS = {"stop", "eos", "eos_token", "stop_sequence"}
TAIL_CHARS_FOR_LINE_REPETITION = 16_384
MIN_REPEATED_LINE_CHARS = 20
MIN_REPEATED_LINE_COUNT = 4


@dataclass(frozen=True)
class GateExample:
    """One deterministically selected DAPO prompt."""

    problem_id: str
    problem: str
    ground_truth: str
    training_prompt: str
    selection_digest: str


@dataclass(frozen=True)
class AcceptedExclusions:
    """Union of identities observed in one or more accepted SFT JSONLs."""

    problem_ids: frozenset[str]
    prompt_hashes: frozenset[str]
    row_count: int


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.expanduser().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object in {path}:{line_number}")
            yield row


def load_accepted_exclusions(paths: Sequence[str]) -> AcceptedExclusions:
    """Load the fail-closed union of accepted problem and prompt identities."""

    if not paths:
        raise ValueError("at least one --accepted SFT JSONL is required")
    problem_ids: set[str] = set()
    prompt_hashes: set[str] = set()
    row_count = 0
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"accepted SFT JSONL not found: {path}")
        for row in _read_jsonl(path):
            row_count += 1
            problem_id = str(row.get("problem_id", "") or "").strip()
            prompt_text = str(row.get("prompt_text", "") or "")
            if not problem_id and not prompt_text.strip():
                raise ValueError(
                    f"accepted row {row_count} in {path} has neither problem_id nor prompt_text; "
                    "cannot prove held-out disjointness"
                )
            if problem_id:
                problem_ids.add(problem_id)
            if prompt_text.strip():
                prompt_hashes.add(_sha256(prompt_text))
    if row_count == 0:
        raise ValueError("accepted SFT JSONL(s) contain no rows")
    return AcceptedExclusions(
        problem_ids=frozenset(problem_ids),
        prompt_hashes=frozenset(prompt_hashes),
        row_count=row_count,
    )


def _dapo_fields(row: dict[str, Any], index: int) -> tuple[str, str, str]:
    """Mirror ``tmx_jax.data.load_dapo`` field precedence exactly."""

    problem = str(row.get("prompt") or row.get("problem") or row.get("question") or "")
    answer = str(row.get("solution") or row.get("answer") or row.get("final_answer") or "")
    problem_id = str(row.get("id", index))
    return problem_id, problem, answer


def select_heldout_examples(
    raw_rows: Iterable[dict[str, Any]],
    tokenizer: Any,
    exclusions: AcceptedExclusions,
    *,
    num_prompts: int,
    selection_seed: int,
) -> tuple[list[GateExample], dict[str, int]]:
    """Select stable hash-ranked DAPO rows after accepted-set exclusion.

    Selection depends only on the seed and problem identity/text.  It never
    inspects ground-truth correctness, model outputs, or benchmark results.
    Exclusion checks both the canonical DAPO ``problem_id`` and the exact
    training-prompt hash, so a legacy accepted file missing one identity still
    cannot leak into the gate set.
    """

    if num_prompts <= 0:
        raise ValueError("num_prompts must be positive")
    from tmx_jax.data import render_chat_prompt

    candidates: list[GateExample] = []
    seen_problem_ids: set[str] = set()
    excluded_by_id = 0
    excluded_by_prompt = 0
    missing = 0
    duplicate_ids = 0
    for index, row in enumerate(raw_rows):
        problem_id, problem, answer = _dapo_fields(row, index)
        if not problem or not answer:
            missing += 1
            continue
        if problem_id in seen_problem_ids:
            duplicate_ids += 1
            continue
        seen_problem_ids.add(problem_id)
        # This is the exact renderer used by load_dapo for thinking=auto.
        training_prompt = render_chat_prompt(problem, tokenizer, enable_thinking=True)
        by_id = problem_id in exclusions.problem_ids
        by_prompt = _sha256(training_prompt) in exclusions.prompt_hashes
        if by_id or by_prompt:
            excluded_by_id += int(by_id)
            excluded_by_prompt += int(by_prompt)
            continue
        digest = _sha256(f"{int(selection_seed)}\0{problem_id}\0{problem}")
        candidates.append(
            GateExample(
                problem_id=problem_id,
                problem=problem,
                ground_truth=answer,
                training_prompt=training_prompt,
                selection_digest=digest,
            )
        )
    candidates.sort(key=lambda ex: (ex.selection_digest, ex.problem_id, ex.problem))
    if len(candidates) < num_prompts:
        raise RuntimeError(
            f"only {len(candidates)} eligible held-out DAPO rows remain after excluding "
            f"{exclusions.row_count} accepted rows; requested {num_prompts}"
        )
    selected = candidates[:num_prompts]
    selected_ids = {ex.problem_id for ex in selected}
    if selected_ids & exclusions.problem_ids:
        raise AssertionError("internal error: selected held-out IDs overlap accepted SFT IDs")
    if any(_sha256(ex.training_prompt) in exclusions.prompt_hashes for ex in selected):
        raise AssertionError("internal error: selected held-out prompts overlap accepted SFT prompts")
    stats = {
        "raw_rows": len(seen_problem_ids) + duplicate_ids + missing,
        "accepted_rows": exclusions.row_count,
        "accepted_unique_problem_ids": len(exclusions.problem_ids),
        "accepted_unique_prompt_hashes": len(exclusions.prompt_hashes),
        "excluded_by_problem_id": excluded_by_id,
        "excluded_by_prompt_hash": excluded_by_prompt,
        "missing_problem_or_answer": missing,
        "duplicate_problem_ids": duplicate_ids,
        "eligible_rows": len(candidates),
        "selected_rows": len(selected),
    }
    return selected, stats


def load_heldout_dapo(
    tokenizer: Any,
    accepted_paths: Sequence[str],
    *,
    num_prompts: int,
    selection_seed: int,
) -> tuple[list[GateExample], dict[str, int]]:
    """Load DAPO-Math-17k and return accepted-disjoint gate examples."""

    from tmx_jax.data import _load_raw

    exclusions = load_accepted_exclusions(accepted_paths)
    return select_heldout_examples(
        _load_raw(),
        tokenizer,
        exclusions,
        num_prompts=num_prompts,
        selection_seed=selection_seed,
    )


def render_prompt_variants(example: GateExample, tokenizer: Any) -> dict[str, str]:
    """Return the two canonical prompts for the same raw problem."""

    from tmx_gpu.eval_gpu import render_prompt as render_paper_eval_prompt

    return {
        "training": example.training_prompt,
        # auto intentionally leaves Qwen3's default thinking mode untouched,
        # exactly as the paper benchmark harness does.
        "paper_eval": render_paper_eval_prompt(tokenizer, example.problem, "auto"),
    }


def has_tail_repeated_line(text: str) -> bool:
    """Detect an exact nontrivial line repeated >=4 times near the output tail."""

    tail = (text or "")[-TAIL_CHARS_FOR_LINE_REPETITION:]
    lines = [" ".join(line.split()) for line in tail.splitlines()]
    counts = Counter(line for line in lines if len(line) >= MIN_REPEATED_LINE_CHARS)
    return bool(counts and max(counts.values()) >= MIN_REPEATED_LINE_COUNT)


def completion_record(
    *,
    example: GateExample,
    prompt_mode: str,
    sample_idx: int,
    sample_seed: int,
    completion: Completion,
    tokenizer: Any,
    max_completion_tokens: int,
) -> dict[str, Any]:
    """Convert one vLLM result into frozen health diagnostics."""

    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"unknown prompt_mode={prompt_mode!r}")
    text = completion.text or ""
    token_count = int(completion.completion_tokens or 0)
    if token_count <= 0 and text:
        token_count = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    finish_reason = str(completion.finish_reason or "")
    failed = completion_failed(completion)
    clipped = bool(
        not failed
        and (finish_reason == "length" or token_count >= int(max_completion_tokens))
    )
    # Fail closed on missing/unknown finish reasons.  The behavioral gate is
    # specifically meant to prove natural termination; absence of a serving
    # signal cannot count as that proof even when the text is below the cap.
    terminated = bool(
        not failed and finish_reason in TERMINATED_FINISH_REASONS
    )
    boxed = _extract_boxed_candidates(text)
    has_open_think = "<think>" in text
    has_closed_think = "</think>" in text
    last_close = text.rfind("</think>")
    last_box = text.rfind("\\boxed")
    chunk_loop = is_looping(text)
    repeated_line = has_tail_repeated_line(text)
    return {
        "metric_contract_version": METRIC_CONTRACT_VERSION,
        "problem_id": example.problem_id,
        "prompt_mode": prompt_mode,
        "sample_idx": int(sample_idx),
        "sample_seed": int(sample_seed),
        "finish_reason": finish_reason,
        "generation_error": failed,
        "generation_error_detail": completion.error,
        "completion_tokens": token_count,
        "terminated": terminated,
        "clipped": clipped,
        "valid_boxed_answer": bool(boxed and boxed[-1][1].strip()),
        "has_open_think": has_open_think,
        "has_closed_think": has_closed_think,
        "unclosed_think": bool(has_open_think and not has_closed_think),
        "boxed_after_closed_think": bool(last_close >= 0 and last_box > last_close),
        "chunk_loop": chunk_loop,
        "tail_repeated_line": repeated_line,
        "repetition_detected": bool(chunk_loop or repeated_line),
        # Diagnostic only.  This field is never consumed by health_gate().
        "correctness_diagnostic_only": bool(
            not failed and is_correct_completion(text, example.ground_truth)
        ),
        "completion_text": text,
    }


def recompute_health_record(
    row: dict[str, Any], *, max_completion_tokens: int,
) -> dict[str, Any]:
    """Re-derive every health field from the persisted primitive rollout.

    This intentionally ignores serialized ``terminated``, ``clipped``, boxed,
    repetition, and think-tag booleans.  Selection therefore cannot be changed
    by editing summarized rates or derived row fields while leaving the raw
    completion text/finish reason/token count/error unchanged.
    """
    required = {
        "metric_contract_version", "problem_id", "prompt_mode", "sample_idx",
        "sample_seed", "finish_reason", "generation_error_detail",
        "completion_tokens", "completion_text",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"raw gate rollout lacks primitive fields: {missing}")
    if row.get("metric_contract_version") != METRIC_CONTRACT_VERSION:
        raise ValueError("raw gate rollout metric contract mismatch")
    prompt_mode = str(row["prompt_mode"])
    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"raw gate rollout has unknown prompt mode {prompt_mode!r}")
    text = str(row.get("completion_text") or "")
    finish_reason = str(row.get("finish_reason") or "")
    token_count = int(row.get("completion_tokens", 0))
    if token_count < 0:
        raise ValueError("raw gate rollout completion_tokens must be nonnegative")
    error_detail = row.get("generation_error_detail")
    failed = finish_reason == "error" or error_detail is not None
    clipped = bool(
        not failed
        and (finish_reason == "length" or token_count >= int(max_completion_tokens))
    )
    terminated = bool(not failed and finish_reason in TERMINATED_FINISH_REASONS)
    boxed = _extract_boxed_candidates(text)
    has_open_think = "<think>" in text
    has_closed_think = "</think>" in text
    last_close = text.rfind("</think>")
    last_box = text.rfind("\\boxed")
    chunk_loop = is_looping(text)
    repeated_line = has_tail_repeated_line(text)
    return {
        "metric_contract_version": METRIC_CONTRACT_VERSION,
        "problem_id": str(row["problem_id"]),
        "prompt_mode": prompt_mode,
        "sample_idx": int(row["sample_idx"]),
        "sample_seed": int(row["sample_seed"]),
        "finish_reason": finish_reason,
        "generation_error": failed,
        "generation_error_detail": error_detail,
        "completion_tokens": token_count,
        "terminated": terminated,
        "clipped": clipped,
        "valid_boxed_answer": bool(boxed and boxed[-1][1].strip()),
        "has_open_think": has_open_think,
        "has_closed_think": has_closed_think,
        "unclosed_think": bool(has_open_think and not has_closed_think),
        "boxed_after_closed_think": bool(last_close >= 0 and last_box > last_close),
        "chunk_loop": chunk_loop,
        "tail_repeated_line": repeated_line,
        "repetition_detected": bool(chunk_loop or repeated_line),
        # Persisted only as a diagnostic and excluded from every criterion.
        "correctness_diagnostic_only": bool(
            row.get("correctness_diagnostic_only", False)),
    }


def recompute_gate_from_rollouts(
    rows: Sequence[dict[str, Any]], *, num_prompts: int, samples_per_prompt: int,
    generation_seed: int, max_completion_tokens: int,
    ordered_problem_ids: Sequence[str], thresholds: dict[str, float],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Validate paired raw-rollout geometry and recompute metrics + verdict."""
    expected_ids = [str(item) for item in ordered_problem_ids]
    if len(expected_ids) != int(num_prompts) or len(set(expected_ids)) != len(expected_ids):
        raise ValueError("held-out selection problem IDs are missing or duplicated")
    recomputed = [
        recompute_health_record(row, max_completion_tokens=max_completion_tokens)
        for row in rows
    ]
    expected_per_mode = int(num_prompts) * int(samples_per_prompt)
    expected_slots = {
        (mode, problem_id, sample_idx): int(generation_seed) + prompt_idx * 100003 + sample_idx
        for mode in PROMPT_MODES
        for prompt_idx, problem_id in enumerate(expected_ids)
        for sample_idx in range(int(samples_per_prompt))
    }
    observed: dict[tuple[str, str, int], int] = {}
    for row in recomputed:
        key = (row["prompt_mode"], row["problem_id"], int(row["sample_idx"]))
        if key in observed:
            raise ValueError(f"duplicate raw gate rollout slot: {key!r}")
        observed[key] = int(row["sample_seed"])
    if set(observed) != set(expected_slots):
        missing = sorted(set(expected_slots) - set(observed))
        extra = sorted(set(observed) - set(expected_slots))
        raise ValueError(
            f"raw gate rollout slots are incomplete (missing={missing}, extra={extra})")
    bad_seeds = [key for key in expected_slots if observed[key] != expected_slots[key]]
    if bad_seeds:
        raise ValueError(f"raw gate rollout sample seeds mismatch: {bad_seeds[:3]}")
    metrics = {
        mode: aggregate_mode(
            [row for row in recomputed if row["prompt_mode"] == mode],
            expected_total=expected_per_mode,
        )
        for mode in PROMPT_MODES
    }
    gate = health_gate(
        metrics,
        max_completion_tokens=max_completion_tokens,
        min_termination_rate=float(thresholds["min_termination_rate"]),
        max_clipped_rate=float(thresholds["max_clipped_rate"]),
        min_valid_boxed_rate=float(thresholds["min_valid_boxed_answer_rate"]),
        max_median_token_fraction=(
            float(thresholds["max_median_completion_tokens"])
            / float(max_completion_tokens)
        ),
        max_repetition_rate=float(thresholds["max_repetition_rate"]),
        max_unclosed_think_rate=float(thresholds["max_unclosed_think_rate"]),
        max_generation_error_rate=float(thresholds["max_generation_error_rate"]),
    )
    return metrics, gate


def _rate(rows: Sequence[dict[str, Any]], key: str) -> float:
    return sum(bool(row.get(key)) for row in rows) / len(rows) if rows else 0.0


def aggregate_mode(
    rows: Sequence[dict[str, Any]],
    *,
    expected_total: int,
) -> dict[str, Any]:
    """Aggregate one prompt mode, excluding transport errors from health rates."""

    usable = [row for row in rows if not row.get("generation_error")]
    errors = len(rows) - len(usable)
    token_counts = [int(row["completion_tokens"]) for row in usable]
    open_think = [row for row in usable if row.get("has_open_think")]
    return {
        "metric_contract_version": METRIC_CONTRACT_VERSION,
        "expected_generations": int(expected_total),
        "observed_generations": len(rows),
        "usable_generations": len(usable),
        "generation_error_count": errors,
        "generation_error_rate": errors / max(1, expected_total),
        "termination_rate": _rate(usable, "terminated"),
        "nontermination_rate": 1.0 - _rate(usable, "terminated"),
        "clipped_rate": _rate(usable, "clipped"),
        "valid_boxed_answer_rate": _rate(usable, "valid_boxed_answer"),
        "mean_completion_tokens": statistics.fmean(token_counts) if token_counts else 0.0,
        "median_completion_tokens": statistics.median(token_counts) if token_counts else 0.0,
        "p90_completion_tokens": (
            sorted(token_counts)[math.ceil(0.9 * len(token_counts)) - 1]
            if token_counts
            else 0
        ),
        "chunk_loop_rate": _rate(usable, "chunk_loop"),
        "tail_repeated_line_rate": _rate(usable, "tail_repeated_line"),
        "repetition_rate": _rate(usable, "repetition_detected"),
        "think_open_rate": _rate(usable, "has_open_think"),
        "think_close_rate": _rate(usable, "has_closed_think"),
        "unclosed_think_rate": _rate(usable, "unclosed_think"),
        "unclosed_think_given_open_rate": _rate(open_think, "unclosed_think"),
        "boxed_after_closed_think_rate": _rate(usable, "boxed_after_closed_think"),
        # Explicitly named diagnostic: it cannot be mistaken for a gate input.
        "correctness_rate_diagnostic_only": _rate(
            usable, "correctness_diagnostic_only"
        ),
    }


def health_gate(
    metrics_by_mode: dict[str, dict[str, Any]],
    *,
    max_completion_tokens: int,
    min_termination_rate: float = DEFAULT_MIN_TERMINATION_RATE,
    max_clipped_rate: float = DEFAULT_MAX_CLIPPED_RATE,
    min_valid_boxed_rate: float = DEFAULT_MIN_VALID_BOXED_RATE,
    max_median_token_fraction: float = DEFAULT_MAX_MEDIAN_TOKEN_FRACTION,
    max_repetition_rate: float = DEFAULT_MAX_REPETITION_RATE,
    max_unclosed_think_rate: float = DEFAULT_MAX_UNCLOSED_THINK_RATE,
    max_generation_error_rate: float = DEFAULT_MAX_GENERATION_ERROR_RATE,
) -> dict[str, Any]:
    """Apply health-only criteria to both prompt modes.

    There is intentionally no correctness argument or criterion here.
    """

    max_median_tokens = float(max_completion_tokens) * float(max_median_token_fraction)
    thresholds = {
        "min_termination_rate": float(min_termination_rate),
        "max_clipped_rate": float(max_clipped_rate),
        "min_valid_boxed_answer_rate": float(min_valid_boxed_rate),
        "max_median_completion_tokens": max_median_tokens,
        "max_repetition_rate": float(max_repetition_rate),
        "max_unclosed_think_rate": float(max_unclosed_think_rate),
        "max_generation_error_rate": float(max_generation_error_rate),
    }
    by_mode: dict[str, Any] = {}
    for mode in PROMPT_MODES:
        metrics = metrics_by_mode.get(mode, {})
        criteria = {
            "termination_rate": float(metrics.get("termination_rate", 0.0))
            >= thresholds["min_termination_rate"],
            "clipped_rate": float(metrics.get("clipped_rate", 1.0))
            <= thresholds["max_clipped_rate"],
            "valid_boxed_answer_rate": float(metrics.get("valid_boxed_answer_rate", 0.0))
            >= thresholds["min_valid_boxed_answer_rate"],
            "median_completion_tokens": float(metrics.get("median_completion_tokens", math.inf))
            <= thresholds["max_median_completion_tokens"],
            "repetition_rate": float(metrics.get("repetition_rate", 1.0))
            <= thresholds["max_repetition_rate"],
            "unclosed_think_rate": float(metrics.get("unclosed_think_rate", 1.0))
            <= thresholds["max_unclosed_think_rate"],
            "generation_error_rate": float(metrics.get("generation_error_rate", 1.0))
            <= thresholds["max_generation_error_rate"],
        }
        by_mode[mode] = {
            "passed": bool(criteria and all(criteria.values())),
            "criteria": criteria,
        }
    return {
        "contract_version": METRIC_CONTRACT_VERSION,
        "selection_uses_correctness": False,
        "thresholds": thresholds,
        "by_prompt_mode": by_mode,
        "passed": all(result["passed"] for result in by_mode.values()),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoints", required=True, help="comma-separated vLLM endpoints")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-1.7B",
        help="local/HF model path used for tokenizer provenance",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="optional local/HF tokenizer path (defaults to --model)",
    )
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--model-source-ref", default="")
    parser.add_argument("--model-source-digest", default="")
    parser.add_argument(
        "--model-files-manifest", default="",
        help="download manifest binding every local model artifact member",
    )
    parser.add_argument(
        "--accepted-source-manifest", default="",
        help="optional JSON binding accepted.jsonl inputs to exact W&B artifacts",
    )
    parser.add_argument(
        "--accepted",
        action="append",
        required=True,
        help="accepted SFT JSONL to exclude; repeat for multiple libraries",
    )
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--samples-per-prompt", type=int, default=2)
    parser.add_argument("--selection-seed", type=int, default=20260713)
    parser.add_argument("--generation-seed", type=int, default=20260713)
    parser.add_argument("--max-completion-tokens", type=int, default=32768)
    parser.add_argument("--server-max-model-len", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--concurrency-per-server", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--out-dir", default="baseline_runs_gpu/sft_gate")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="exit 2 after writing artifacts when the health gate fails",
    )
    parser.add_argument("--min-termination-rate", type=float, default=DEFAULT_MIN_TERMINATION_RATE)
    parser.add_argument("--max-clipped-rate", type=float, default=DEFAULT_MAX_CLIPPED_RATE)
    parser.add_argument("--min-valid-boxed-rate", type=float, default=DEFAULT_MIN_VALID_BOXED_RATE)
    parser.add_argument(
        "--max-median-token-fraction",
        type=float,
        default=DEFAULT_MAX_MEDIAN_TOKEN_FRACTION,
    )
    parser.add_argument("--max-repetition-rate", type=float, default=DEFAULT_MAX_REPETITION_RATE)
    parser.add_argument(
        "--max-unclosed-think-rate",
        type=float,
        default=DEFAULT_MAX_UNCLOSED_THINK_RATE,
    )
    parser.add_argument(
        "--max-generation-error-rate",
        type=float,
        default=DEFAULT_MAX_GENERATION_ERROR_RATE,
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.samples_per_prompt <= 0:
        raise SystemExit("--samples-per-prompt must be positive")
    if args.max_completion_tokens <= 0:
        raise SystemExit("--max-completion-tokens must be positive")
    unit_interval = (
        "min_termination_rate",
        "max_clipped_rate",
        "min_valid_boxed_rate",
        "max_median_token_fraction",
        "max_repetition_rate",
        "max_unclosed_think_rate",
        "max_generation_error_rate",
    )
    for name in unit_interval:
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be in [0, 1], got {value}")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _model_checkpoint_metadata(model_dir: str) -> dict[str, Any]:
    for name in ("sft_checkpoint.json", "sft_provenance.json"):
        path = Path(model_dir).expanduser() / name
        if path.is_file():
            return {"metadata_file": name, "metadata": json.loads(path.read_text())}
    return {"metadata_file": None, "metadata": {}}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    _validate_args(args)

    from transformers import AutoTokenizer

    tokenizer_name = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=bool(args.trust_remote_code),
    )
    examples, selection_stats = load_heldout_dapo(
        tokenizer,
        args.accepted,
        num_prompts=args.num_prompts,
        selection_seed=args.selection_seed,
    )
    accepted_rows = [
        row
        for accepted_path in args.accepted
        for row in _read_jsonl(Path(accepted_path).expanduser())
    ]
    accepted_library_digest = common.semantic_sft_dataset_digest(accepted_rows)
    accepted_source_manifest = None
    if args.accepted_source_manifest:
        raw_accepted_manifest = json.loads(
            Path(args.accepted_source_manifest).read_text(encoding="utf-8"))
        if raw_accepted_manifest.get("version") != "wandb_accepted_sources_v2":
            raise SystemExit("accepted source manifest version mismatch")
        raw_downloads = raw_accepted_manifest.get("download_manifests")
        if not isinstance(raw_downloads, list) or not raw_downloads:
            raise SystemExit("accepted source manifest has no artifact downloads")
        validated_downloads = [
            common.validate_artifact_download_manifest(item)
            for item in raw_downloads
        ]
        attributed_accepted: set[str] = set()
        sources = []
        for item in validated_downloads:
            root = item["download_root"]
            accepted_files = [
                file_info for file_info in item["files"]
                if file_info["relative_path"].endswith("accepted.jsonl")
            ]
            if not accepted_files:
                raise SystemExit("accepted artifact has no accepted.jsonl member")
            for file_info in accepted_files:
                attributed_accepted.add(str(Path(root) / file_info["relative_path"]))
            sources.append({
                **item["source"],
                "file_manifest_sha256": item["file_manifest_sha256"],
            })
        actual_accepted = {str(Path(path).expanduser().resolve()) for path in args.accepted}
        attributed_accepted = {
            str(Path(path).expanduser().resolve()) for path in attributed_accepted
        }
        if actual_accepted != attributed_accepted:
            raise SystemExit(
                "--accepted paths do not exactly match accepted artifact membership")
        accepted_source_manifest = {
            "version": "wandb_accepted_sources_v2",
            "sources": sources,
        }
        accepted_source_manifest["sha256"] = common.canonical_json_sha256(
            accepted_source_manifest)
    model_file_manifest = None
    if args.model_source_ref or args.model_source_digest:
        if not (args.model_source_ref and args.model_source_digest
                and args.model_files_manifest):
            raise SystemExit(
                "artifact-backed gate requires model source ref, digest, and "
                "--model-files-manifest together")
        raw_model_manifest = json.loads(
            Path(args.model_files_manifest).read_text(encoding="utf-8"))
        model_file_manifest = common.validate_artifact_download_manifest(
            raw_model_manifest,
            expected_source={
                "artifact_ref": args.model_source_ref,
                "artifact_digest": args.model_source_digest,
            },
        )
    prompt_variants = [render_prompt_variants(example, tokenizer) for example in examples]
    max_prompt_tokens = max(
        len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        for variants in prompt_variants
        for prompt in variants.values()
    )
    if (
        args.server_max_model_len is not None
        and max_prompt_tokens + args.max_completion_tokens > args.server_max_model_len
    ):
        raise SystemExit(
            "vLLM context too short for gate: longest rendered prompt "
            f"{max_prompt_tokens} + completion budget {args.max_completion_tokens} > "
            f"server max model len {args.server_max_model_len}"
        )

    endpoints = [item.strip() for item in args.endpoints.split(",") if item.strip()]
    if not endpoints:
        raise SystemExit("--endpoints contains no endpoints")
    pool = VLLMPool.attach(endpoints, args.served_model_name or args.model)

    out_dir = Path(args.out_dir).expanduser() / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    selection_path = out_dir / "heldout_selection.jsonl"
    rollout_path = out_dir / "rollouts.jsonl"
    aggregate_path = out_dir / "aggregate_metrics.json"
    _write_jsonl(
        selection_path,
        (
            {
                **asdict(example),
                "training_prompt_sha256": _sha256(example.training_prompt),
            }
            for example in examples
        ),
    )
    heldout_selection_digest = common.canonical_json_sha256([
        {
            "problem_id": example.problem_id,
            "selection_digest": example.selection_digest,
            "training_prompt_sha256": _sha256(example.training_prompt),
        }
        for example in examples
    ])

    all_records: list[dict[str, Any]] = []
    t0 = time.time()
    for mode in PROMPT_MODES:
        prompts = [variants[mode] for variants in prompt_variants]
        grouped = pool.generate(
            prompts,
            args.samples_per_prompt,
            max_tokens=args.max_completion_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed_base=args.generation_seed,
            timeout=args.timeout,
            concurrency_per_server=args.concurrency_per_server,
        )
        for prompt_idx, (example, completions) in enumerate(zip(examples, grouped, strict=True)):
            for sample_idx, completion in enumerate(completions):
                sample_seed = args.generation_seed + prompt_idx * 100003 + sample_idx
                all_records.append(
                    completion_record(
                        example=example,
                        prompt_mode=mode,
                        sample_idx=sample_idx,
                        sample_seed=sample_seed,
                        completion=completion,
                        tokenizer=tokenizer,
                        max_completion_tokens=args.max_completion_tokens,
                    )
                )
        # Persist after each mode so a later serving failure cannot erase the
        # completed half of the paired diagnostic.
        _write_jsonl(rollout_path, all_records)

    expected_per_mode = len(examples) * args.samples_per_prompt
    metrics_by_mode = {
        mode: aggregate_mode(
            [row for row in all_records if row["prompt_mode"] == mode],
            expected_total=expected_per_mode,
        )
        for mode in PROMPT_MODES
    }
    gate = health_gate(
        metrics_by_mode,
        max_completion_tokens=args.max_completion_tokens,
        min_termination_rate=args.min_termination_rate,
        max_clipped_rate=args.max_clipped_rate,
        min_valid_boxed_rate=args.min_valid_boxed_rate,
        max_median_token_fraction=args.max_median_token_fraction,
        max_repetition_rate=args.max_repetition_rate,
        max_unclosed_think_rate=args.max_unclosed_think_rate,
        max_generation_error_rate=args.max_generation_error_rate,
    )
    aggregate = {
        "metric_contract_version": METRIC_CONTRACT_VERSION,
        "model": args.model,
        "tokenizer": tokenizer_name,
        "served_model_name": args.served_model_name or args.model,
        "model_source": {
            "artifact_ref": args.model_source_ref,
            "artifact_digest": args.model_source_digest,
        },
        "model_file_manifest": model_file_manifest,
        "model_checkpoint": _model_checkpoint_metadata(args.model),
        "dataset": "open-r1/DAPO-Math-17k-Processed:all/train",
        "accepted_jsonls": [str(Path(path).expanduser()) for path in args.accepted],
        "accepted_library_digest": accepted_library_digest,
        "accepted_source_manifest": accepted_source_manifest,
        "selection": {
            "policy": "sha256(selection_seed, problem_id, problem)",
            "uses_model_outputs": False,
            "uses_correctness": False,
            "selection_seed": args.selection_seed,
            "heldout_selection_sha256": heldout_selection_digest,
            **selection_stats,
        },
        "sampling": {
            "paired_seeds_across_prompt_modes": True,
            "generation_seed": args.generation_seed,
            "num_prompts": len(examples),
            "samples_per_prompt": args.samples_per_prompt,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_completion_tokens": args.max_completion_tokens,
            "max_rendered_prompt_tokens": max_prompt_tokens,
        },
        "prompt_modes": {
            "training": "qwen3_math_user_suffix_v1; render_chat_prompt(enable_thinking=True)",
            "paper_eval": "qwen3_math_user_suffix_v1; render_prompt(thinking_mode=auto)",
        },
        "metrics_by_prompt_mode": metrics_by_mode,
        "health_gate": gate,
        "correctness_policy": (
            "reported for diagnosis only; excluded from held-out selection and health gate"
        ),
        "elapsed_s": time.time() - t0,
        "completed_at_unix": time.time(),
        "artifacts": {
            "heldout_selection": str(selection_path),
            "rollouts": str(rollout_path),
            "aggregate": str(aggregate_path),
        },
    }
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2, sort_keys=True), flush=True)
    if args.require_pass and not gate["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
