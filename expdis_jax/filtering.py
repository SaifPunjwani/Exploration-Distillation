"""Dependency-light MR-ME trajectory pooling for the JAX pipeline.

This module contains the portable part of the measured GPU QualityPool repair:
all Scout files are merged, hard quality gates are applied, and at most one
trajectory is selected for each exact problem before the global cap is applied.
Selection never ranks candidates by blended, correctness, novelty, or RND
reward.  The shortest clean completion wins within a problem; stable content
keys and a stable problem hash make the result independent of input order.

The implementation is deliberately plain Python.  It depends only on the
production answer verifier in :mod:`expdis_jax.rewarding`, so its behavior can be
tested without importing JAX, a tokenizer, or a model runtime.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

from .rewarding import _extract_boxed_candidates, is_correct_completion


QUALITY_POOL_POLICY = "quality_pool_mrme_v2"
QUALITY_POOL_CONTRACT_VERSION = "quality_pool_mrme_v2_exact_problem_v1"

# Frozen GPU repetition screen: any 40-character span repeated four times
# consecutively is considered a generation loop.
REPETITION_SCREEN_VERSION = "consecutive_chars_c40_r4_v1"
LOOP_CHUNK_CHARS = 40
LOOP_MIN_REPEATS = 4

# The measured R2 GPU transfer contract accepted only completions in the
# inclusive 128..4,500-token window.  This curation bound is intentionally
# separate from the model's larger generation budget and is recorded in every
# funnel.  No character-count approximation is allowed at filtering time.
DEFAULT_MIN_COMPLETION_TOKENS = 128
DEFAULT_MAX_COMPLETION_TOKENS = 4_500
MAX_GLOBAL_CAP = 500

_REJECTION_REASONS = (
    "incorrect",
    "nonterminated",
    "invalid_answer",
    "missing_metadata",
    "no_boxed_answer",
    "boxed_answer_mismatch",
    "clipped",
    "looping",
    "too_short",
    "too_long",
)


def is_looping(
    text: str,
    *,
    chunk_chars: int = LOOP_CHUNK_CHARS,
    min_repeats: int = LOOP_MIN_REPEATS,
) -> bool:
    """Return whether an exact character span repeats consecutively.

    Defaults are frozen contract constants.  Optional arguments exist only for
    focused unit tests and diagnostics; production callers should use defaults.
    """
    if int(chunk_chars) <= 0:
        raise ValueError("chunk_chars must be positive")
    if int(min_repeats) < 2:
        raise ValueError("min_repeats must be at least 2")
    value = str(text or "")
    span = int(chunk_chars) * int(min_repeats)
    if len(value) < span:
        return False
    for start in range(len(value) - span + 1):
        if value[start] != value[start + int(chunk_chars)]:
            continue
        chunk = value[start : start + int(chunk_chars)]
        if all(
            value[
                start + repeat * int(chunk_chars) :
                start + (repeat + 1) * int(chunk_chars)
            ] == chunk
            for repeat in range(1, int(min_repeats))
        ):
            return True
    return False


def completion_token_length(row: dict[str, Any]) -> int | None:
    """Return an explicit non-negative completion length, or ``None``.

    Filtering fails closed when token-length metadata is absent.  Falling back
    to a character heuristic would make the accepted set tokenizer-dependent
    and would not enforce the stated token window.
    """
    for key in (
        "completion_token_length",
        "completion_token_length_true",
        "completion_token_length_retokenized",
        "completion_len",
        "num_completion_tokens",
    ):
        if key not in row or row.get(key) is None:
            continue
        value = row[key]
        if isinstance(value, bool):
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(numeric) or not numeric.is_integer():
            return None
        length = int(numeric)
        return length if length >= 0 else None
    return None


def exact_problem_key(row: dict[str, Any]) -> tuple[str, str]:
    """Return the exact identity used for one-row-per-problem selection.

    Exact prompt text takes precedence because duplicated dataset records can
    have different IDs for the same problem.  ``problem_id`` is a fail-safe for
    legacy rows that genuinely lack prompt text; a row with neither is invalid.
    """
    prompt = row.get("prompt_text")
    if prompt is not None and str(prompt).strip():
        return "prompt_text", str(prompt)
    problem_id = row.get("problem_id")
    if problem_id is not None and str(problem_id).strip():
        return "problem_id", str(problem_id)
    raise ValueError("trajectory has neither non-empty prompt_text nor problem_id")


def _source_name(row: dict[str, Any]) -> str:
    value = str(row.get("source_file", "") or "").strip()
    return value or "<direct>"


def _canonical_row_json(row: dict[str, Any]) -> str:
    return json.dumps(
        row,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def _within_problem_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Reward-independent order for clean candidates of one problem."""
    length = completion_token_length(row)
    if length is None:  # Rejected rows must never reach this helper.
        raise ValueError("candidate is missing completion token length")
    return (
        length,
        str(row.get("completion_text", "") or ""),
        _source_name(row),
        str(row.get("explorer_step", row.get("step", "")) or ""),
        # This last-resort key only distinguishes otherwise identical records.
        # Rewards never precede or displace the shortest-completion decision.
        _canonical_row_json(row),
    )


def _problem_order(problem_key: tuple[str, str]) -> tuple[str, str, str]:
    namespace, value = problem_key
    digest = hashlib.sha256(
        f"{namespace}\0{value}".encode("utf-8", errors="replace")
    ).hexdigest()
    return digest, namespace, value


def rejection_reason(
    row: dict[str, Any],
    *,
    min_completion_tokens: int = DEFAULT_MIN_COMPLETION_TOKENS,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
) -> str | None:
    """Return the first failed QualityPool gate, or ``None`` when clean."""
    if int(min_completion_tokens) < 0:
        raise ValueError("min_completion_tokens must be non-negative")
    if int(max_completion_tokens) < int(min_completion_tokens):
        raise ValueError(
            "max_completion_tokens must be at least min_completion_tokens"
        )
    if not bool(row.get("is_correct", False)):
        return "incorrect"
    if not bool(row.get("terminated", False)):
        return "nonterminated"
    if not bool(row.get("valid_answer", False)):
        return "invalid_answer"

    completion = row.get("completion_text")
    ground_truth = row.get("ground_truth")
    if completion is None or ground_truth is None or not str(ground_truth).strip():
        return "missing_metadata"
    try:
        exact_problem_key(row)
    except ValueError:
        return "missing_metadata"

    boxes = _extract_boxed_candidates(str(completion))
    if not boxes:
        return "no_boxed_answer"
    final_box = boxes[-1][1]
    if not is_correct_completion(f"\\boxed{{{final_box}}}", str(ground_truth)):
        return "boxed_answer_mismatch"
    if bool(row.get("clipped", False)):
        return "clipped"
    if is_looping(str(completion)):
        return "looping"

    length = completion_token_length(row)
    if length is None:
        return "missing_metadata"
    if length < int(min_completion_tokens):
        return "too_short"
    if length > int(max_completion_tokens):
        return "too_long"
    return None


def passes_quality_gates(row: dict[str, Any], **kwargs: Any) -> bool:
    """Return whether ``row`` clears the complete frozen filter contract."""
    return rejection_reason(row, **kwargs) is None


def _sorted_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(_source_name(row) for row in rows).items()))


def select_quality_pool(
    rows: Sequence[dict[str, Any]],
    *,
    max_examples: int = MAX_GLOBAL_CAP,
    min_completion_tokens: int = DEFAULT_MIN_COMPLETION_TOKENS,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Filter a merged K-Scout pool and select one clean row per problem.

    ``max_examples`` is a global cap after merging all Scout inputs.  It may be
    lowered for an ablation or smoke test, but cannot exceed the frozen 500-row
    campaign ceiling.
    """
    max_examples = int(max_examples)
    if max_examples < 0 or max_examples > MAX_GLOBAL_CAP:
        raise ValueError(
            f"max_examples must be in [0, {MAX_GLOBAL_CAP}], got {max_examples}"
        )
    # Validate the window even for an empty input.
    if int(min_completion_tokens) < 0:
        raise ValueError("min_completion_tokens must be non-negative")
    if int(max_completion_tokens) < int(min_completion_tokens):
        raise ValueError(
            "max_completion_tokens must be at least min_completion_tokens"
        )

    raw_rows = list(rows)
    rejected = {reason: 0 for reason in _REJECTION_REASONS}
    clean: list[dict[str, Any]] = []
    for row in raw_rows:
        reason = rejection_reason(
            row,
            min_completion_tokens=min_completion_tokens,
            max_completion_tokens=max_completion_tokens,
        )
        if reason is None:
            clean.append(row)
        else:
            rejected[reason] += 1

    by_problem: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in clean:
        by_problem[exact_problem_key(row)].append(row)
    winners = [
        min(candidates, key=_within_problem_key)
        for candidates in by_problem.values()
    ]
    winners.sort(key=lambda row: _problem_order(exact_problem_key(row)))
    selected = winners[:max_examples]

    stats: dict[str, Any] = {
        "policy": QUALITY_POOL_POLICY,
        "contract_version": QUALITY_POOL_CONTRACT_VERSION,
        "repetition_screen_version": REPETITION_SCREEN_VERSION,
        "raw": len(raw_rows),
        "stored_correct": sum(bool(row.get("is_correct", False)) for row in raw_rows),
        "clean_correct": len(clean),
        "unique_clean_problems": len(winners),
        "duplicate_clean_rows_removed": len(clean) - len(winners),
        "accepted": len(selected),
        "capped_out": len(winners) - len(selected),
        "max_examples": max_examples,
        "min_completion_tokens": int(min_completion_tokens),
        "max_completion_tokens": int(max_completion_tokens),
        "rejected": rejected,
        "source_counts": {
            "raw": _sorted_counts(raw_rows),
            "clean": _sorted_counts(clean),
            "accepted": _sorted_counts(selected),
        },
    }
    stats["selection_sha256"] = hashlib.sha256(
        "\n".join(_canonical_row_json(row) for row in selected).encode("utf-8")
    ).hexdigest()
    return selected, stats


def load_trajectory_files(paths: Sequence[str | os.PathLike[str]]) -> list[dict[str, Any]]:
    """Load and merge K trajectory JSONL files in deterministic path order."""
    rows: list[dict[str, Any]] = []
    for raw_path in sorted(os.fspath(path) for path in paths):
        path = os.path.realpath(os.path.expanduser(raw_path))
        with open(path, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid trajectory JSON at {path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError(
                        f"trajectory row at {path}:{line_number} is not an object"
                    )
                row = dict(row)
                row["source_file"] = path
                rows.append(row)
    return rows


def write_selected_jsonl(
    path: str | os.PathLike[str], rows: Sequence[dict[str, Any]]
) -> str:
    """Write selected rows with canonical key order and return their SHA256."""
    output_path = os.path.realpath(os.path.expanduser(os.fspath(path)))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    digest = hashlib.sha256()
    with open(output_path, "w", encoding="utf-8") as handle:
        for row in rows:
            payload = _canonical_row_json(row) + "\n"
            handle.write(payload)
            digest.update(payload.encode("utf-8"))
    return digest.hexdigest()


def pool_trajectory_files(
    paths: Sequence[str | os.PathLike[str]],
    *,
    output_path: str | os.PathLike[str] | None = None,
    max_examples: int = MAX_GLOBAL_CAP,
    min_completion_tokens: int = DEFAULT_MIN_COMPLETION_TOKENS,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Merge K files, run QualityPool once, and optionally write the result."""
    rows = load_trajectory_files(paths)
    selected, stats = select_quality_pool(
        rows,
        max_examples=max_examples,
        min_completion_tokens=min_completion_tokens,
        max_completion_tokens=max_completion_tokens,
    )
    if output_path is not None:
        output_sha256 = write_selected_jsonl(output_path, selected)
        # Keep the manifest path-independent while still verifying bytes.
        stats["output_sha256"] = output_sha256
    return selected, stats
