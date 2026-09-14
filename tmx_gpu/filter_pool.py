#!/usr/bin/env python3
"""Trajectory pooling + QualityPool filter for (multi-)scout GRPO outputs.

Input: one or more explorer_trajectories.jsonl files (from K scouts).
Output: accepted.jsonl of {prompt_text, completion_text} + metadata rows for
Central SFT, plus funnel stats (raw -> correct -> clean_correct -> accepted;
these feed the pipeline's funnel/* metrics, matching the lambda-annealing
funnel-table semantics).

Policies:
- quality_pool (default): keep rows with is_correct AND valid_answer AND
  terminated AND not clipped AND not looping AND
  128 <= completion_tokens <= 16000; sort quality_first
  (terminated, valid_answer, not clipped, blended_reward desc, shorter first
  — the tmx_jax.pipeline.collect_accepted key) and cap at --max-examples.
- naive_pool: keep is_correct only, sort quality_first, cap — exactly the TPU
  production collect_accepted filter (no quality gates).
- best_scout: count quality-gate survivors per source file, keep only the
  best file's gated rows (sorted quality_first, capped). Harvest top-up files
  must not compete (the pipeline excludes them: harvest rows come from the
  served policy, not from any scout).
- coverage_pool_c8: the corrected c8 selector. Apply exactly the frozen c8
  gates (correct, naturally terminated, matching boxed answer, unclipped,
  exact-problem deduplication), choose at most one trajectory per problem, and
  cap problems in a deterministic reward-independent order. It deliberately
  does *not* inherit QualityPool's experimental 128/16000-token or looping
  heuristics. Within a problem, prefer the shorter completion, then a stable
  content key. Raw blended/RND rewards are never compared.
- original_blended_per_problem: a labeled selector control using the same c8
  hard gates and coverage cap, but choosing the highest blended-reward row
  within each problem before preferring shorter traces. It never compares raw
  RND rewards across different problems.

The cap default (500) is the TPU production max_accepted_trajectories; the
pipeline treats a raised cap as a GPU-MAX tier change.

DEVIATIONS (vs tmx_jax.pipeline.collect_accepted, which only filtered
is_correct then sorted/capped):
- QualityPool adds hard gates on valid_answer/terminated/clipped, a looping
  heuristic (any 40-char chunk repeated >= 4 times consecutively), and token
  length bounds [128, 16000] — these were paper-planned but never implemented
  on TPU (see the GPU reproduction plan's "not implemented anywhere" list).
- completion token length uses the row's completion_token_length field
  (always populated by grpo_gpu); if missing/zero we fall back to a
  len(text)/3.5 char heuristic rather than requiring a tokenizer here.
- the [128, 16000] token gate matches the contract 16384 completion budget;
  --max-completion-tokens lets GPU-MAX runs scale the cap with their budget
  (the pipeline passes budget - 384, mirroring 16384 -> 16000).

Pure Python (json/argparse only) — importable by the pipeline driver without
torch/datasets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict, deque
from typing import Any, Dict, List, Sequence, Tuple

from tmx_jax.rewarding import _extract_boxed_candidates, is_correct_completion

LOOP_CHUNK_CHARS = 40
LOOP_MIN_REPEATS = 4
MIN_COMPLETION_TOKENS = 128
MAX_COMPLETION_TOKENS = 16000
C8_POLICY = "coverage_pool_c8"
ORIGINAL_BLENDED_POLICY = "original_blended_per_problem"

METADATA_KEYS = (
    "trajectory_schema_version",
    "problem_id", "ground_truth", "is_correct", "blended_reward", "reward",
    "correctness_reward", "novelty_reward", "completion_token_length",
    "completion_token_length_true", "completion_token_length_retokenized",
    "terminated", "clipped", "valid_answer", "finish_reason",
    "explorer_step", "step",
)


def is_looping(text: str, *, chunk: int = LOOP_CHUNK_CHARS, repeats: int = LOOP_MIN_REPEATS) -> bool:
    """True if any `chunk`-char substring repeats >= `repeats` times back-to-back."""
    text = text or ""
    span = chunk * repeats
    n = len(text)
    if n < span:
        return False
    for i in range(0, n - span + 1):
        # Cheap first-char filter before building substrings.
        if text[i] != text[i + chunk]:
            continue
        piece = text[i:i + chunk]
        ok = True
        for r in range(1, repeats):
            if text[i + r * chunk: i + (r + 1) * chunk] != piece:
                ok = False
                break
        if ok:
            return True
    return False


def completion_tokens(row: Dict[str, Any]) -> int:
    tok = int(row.get("completion_token_length", 0) or 0)
    if tok > 0:
        return tok
    return max(1, int(len(str(row.get("completion_text", ""))) / 3.5))


def passes_quality_gates(
    row: Dict[str, Any],
    *,
    min_tokens: int = MIN_COMPLETION_TOKENS,
    max_tokens: int = MAX_COMPLETION_TOKENS,
) -> bool:
    """QualityPool hard gates (looping check last: it is the expensive one)."""
    if not bool(row.get("is_correct", False)):
        return False
    if not bool(row.get("valid_answer", False)):
        return False
    if not bool(row.get("terminated", False)):
        return False
    if bool(row.get("clipped", False)):
        return False
    tok = completion_tokens(row)
    if tok < min_tokens or tok > max_tokens:
        return False
    if is_looping(str(row.get("completion_text", ""))):
        return False
    return True


def passes_c8_contract_gates(row: Dict[str, Any]) -> bool:
    """Frozen c8/pasted-method gates, independent of QualityPool heuristics.

    The stored ``is_correct`` flag is necessary but not sufficient: the pasted
    method requires a valid ``\\boxed{}`` answer that itself matches the gold
    answer. Rechecking the final boxed payload also makes imported/legacy rows
    fail closed if their broad ``valid_answer`` parser accepted an unboxed
    numeric final line.
    """
    if not bool(row.get("is_correct", False)):
        return False
    if not bool(row.get("valid_answer", False)):
        return False
    if not bool(row.get("terminated", False)):
        return False
    if bool(row.get("clipped", False)):
        return False
    text = str(row.get("completion_text", "") or "")
    boxes = _extract_boxed_candidates(text)
    ground_truth = str(row.get("ground_truth", "") or "").strip()
    if not boxes or not ground_truth:
        return False
    final_box = boxes[-1][1]
    return bool(is_correct_completion(f"\\boxed{{{final_box}}}", ground_truth))


def quality_first_key(row: Dict[str, Any]):
    """tmx_jax.pipeline.collect_accepted sort key (used with reverse=True)."""
    return (
        1 if row.get("terminated") else 0,
        1 if row.get("valid_answer") else 0,
        0 if row.get("clipped") else 1,
        float(row.get("blended_reward", row.get("reward", 0.0)) or 0.0),
        -float(completion_tokens(row)),
    )


def c8_problem_key(row: Dict[str, Any]) -> Tuple[str, str]:
    """Return a stable identity used to enforce c8's one-row-per-problem rule.

    Exact prompt identity is preferred so duplicate DAPO rows with distinct
    dataset IDs cannot occupy two SFT slots. ``problem_id`` is the fallback for
    imported records lacking prompt text. Silently treating rows with neither
    field as distinct would defeat the coverage guarantee, so c8 fails closed.
    """
    prompt = str(row.get("prompt_text", "") or "")
    if prompt.strip():
        return "prompt_sha256", hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    problem_id = row.get("problem_id")
    if problem_id is not None and str(problem_id).strip():
        return "problem_id", str(problem_id)
    raise ValueError(
        f"{C8_POLICY} requires a non-empty problem_id or prompt_text on every row"
    )


def _canonical_row_json(row: Dict[str, Any]) -> str:
    """Canonical last-resort tie-breaker, independent of input file order."""
    return json.dumps(row, sort_keys=True, separators=(",", ":"), default=str)


def c8_within_problem_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    """Reward-independent quality order for candidates of one problem.

    The hard gates normally make the first component equal for every candidate,
    but keeping termination explicit documents the intended quality order and
    makes this helper safe to inspect in isolation. The full canonical row is
    consulted only after semantic fields are identical; it makes exact ties
    deterministic without giving raw RND/blended reward an ordering role.
    """
    return (
        0 if bool(row.get("terminated", False)) else 1,
        completion_tokens(row),
        str(row.get("completion_text", "") or ""),
        str(row.get("source_file", "") or ""),
        str(row.get("explorer_step", row.get("step", "")) or ""),
        _canonical_row_json(row),
    )


def original_blended_within_problem_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    """Pasted-paper selector control: blended reward, then shorter, per problem."""
    return (
        -float(row.get("blended_reward", row.get("reward", 0.0)) or 0.0),
        completion_tokens(row),
        str(row.get("completion_text", "") or ""),
        str(row.get("source_file", "") or ""),
        str(row.get("explorer_step", row.get("step", "")) or ""),
        _canonical_row_json(row),
    )


def _c8_problem_order(problem_key: Tuple[str, str]) -> Tuple[str, str, str]:
    """Stable pseudo-random problem order for reward-independent global caps."""
    namespace, value = problem_key
    digest = hashlib.sha256(f"{namespace}\0{value}".encode("utf-8")).hexdigest()
    return digest, namespace, value


def select_coverage_pool_c8(
    clean_rows: Sequence[Dict[str, Any]],
    *,
    max_examples: int,
    source_balance: bool = False,
) -> List[Dict[str, Any]]:
    """Select deterministic, coverage-balanced c8 SFT rows.

    Selection has two deliberately separate phases:

    1. Choose the reward-independent quality winner for each problem.
    2. Apply the global cap in a stable problem order, optionally round-robin
       across the winners' source files for MultiScout representation.

    Because source balancing happens only after phase 1, it never changes which
    trajectory represents a problem and can never admit two rows for one
    problem.
    """
    if max_examples < 0:
        raise ValueError("max_examples must be non-negative")

    by_problem: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in clean_rows:
        by_problem[c8_problem_key(row)].append(row)

    winners = [
        min(candidates, key=c8_within_problem_key)
        for candidates in by_problem.values()
    ]
    winners.sort(key=lambda row: _c8_problem_order(c8_problem_key(row)))

    if not source_balance:
        return winners[:max_examples]

    by_source: Dict[str, deque[Dict[str, Any]]] = defaultdict(deque)
    for row in winners:
        by_source[str(row.get("source_file", "") or "")].append(row)

    accepted: List[Dict[str, Any]] = []
    source_order = sorted(by_source)
    while len(accepted) < max_examples:
        made_progress = False
        for source in source_order:
            queue = by_source[source]
            if not queue:
                continue
            accepted.append(queue.popleft())
            made_progress = True
            if len(accepted) >= max_examples:
                break
        if not made_progress:
            break
    return accepted


def select_original_blended_per_problem(
    clean_rows: Sequence[Dict[str, Any]], *, max_examples: int
) -> List[Dict[str, Any]]:
    """Coverage-preserving control for the paper's blended-reward tie-break."""
    if max_examples < 0:
        raise ValueError("max_examples must be non-negative")
    by_problem: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in clean_rows:
        by_problem[c8_problem_key(row)].append(row)
    winners = [
        min(candidates, key=original_blended_within_problem_key)
        for candidates in by_problem.values()
    ]
    winners.sort(key=lambda row: _c8_problem_order(c8_problem_key(row)))
    return winners[:max_examples]


def load_trajectories(paths: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        n_before = len(rows)
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                row["source_file"] = path
                rows.append(row)
        print(f"[filter-pool] {path}: {len(rows) - n_before} rows", flush=True)
    return rows


def pool_trajectories(
    paths: Sequence[str],
    *,
    policy: str = "quality_pool",
    max_examples: int = 500,
    output: str = "accepted.jsonl",
    max_completion_tokens: int = MAX_COMPLETION_TOKENS,
    source_balance: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """Pool + filter trajectories; write accepted.jsonl; return (path, stats)."""
    policy = (policy or "quality_pool").strip().lower()
    if source_balance and policy != C8_POLICY:
        raise ValueError(f"source_balance is only supported for policy={C8_POLICY!r}")
    rows = load_trajectories(paths)
    correct = [r for r in rows if bool(r.get("is_correct", False))]

    if policy == "quality_pool":
        clean = [r for r in correct if passes_quality_gates(r, max_tokens=max_completion_tokens)]
        clean.sort(key=quality_first_key, reverse=True)
        accepted = clean[:max_examples]
    elif policy == "naive_pool":
        # TPU collect_accepted: correct-only, quality_first sort, cap.
        clean = list(correct)
        clean.sort(key=quality_first_key, reverse=True)
        accepted = clean[:max_examples]
    elif policy == "best_scout":
        by_source: Dict[str, List[Dict[str, Any]]] = {}
        for r in correct:
            if passes_quality_gates(r, max_tokens=max_completion_tokens):
                by_source.setdefault(r["source_file"], []).append(r)
        per_source = {src: len(rs) for src, rs in by_source.items()}
        for src in paths:
            per_source.setdefault(src, 0)
        best_src = max(per_source, key=lambda s: per_source[s])
        print(f"[filter-pool] best_scout accepted-per-source: "
              f"{json.dumps(per_source, indent=None)} -> best={best_src}", flush=True)
        clean = by_source.get(best_src, [])
        clean.sort(key=quality_first_key, reverse=True)
        accepted = clean[:max_examples]
    elif policy == C8_POLICY:
        clean = [r for r in correct if passes_c8_contract_gates(r)]
        accepted = select_coverage_pool_c8(
            clean,
            max_examples=max_examples,
            source_balance=source_balance,
        )
    elif policy == ORIGINAL_BLENDED_POLICY:
        clean = [r for r in correct if passes_c8_contract_gates(r)]
        accepted = select_original_blended_per_problem(
            clean, max_examples=max_examples)
    else:
        raise ValueError(f"unsupported policy={policy!r}; "
                         "expected quality_pool | naive_pool | best_scout | "
                         f"{C8_POLICY} | {ORIGINAL_BLENDED_POLICY}")

    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
    with open(output, "w") as f:
        for r in accepted:
            out_row: Dict[str, Any] = {
                "prompt_text": r.get("prompt_text", ""),
                "completion_text": r.get("completion_text", ""),
                **{k: r[k] for k in METADATA_KEYS if k in r},
                "source_file": r.get("source_file", ""),
                "selection_policy": policy,
            }
            if policy in (C8_POLICY, ORIGINAL_BLENDED_POLICY):
                out_row["selection_source_balance"] = bool(source_balance)
            f.write(json.dumps(out_row) + "\n")

    stats = {
        "policy": policy,
        "num_sources": len(set(paths)),
        "raw": len(rows),
        "correct": len(correct),
        "clean_correct": len(clean),
        "accepted": len(accepted),
        "max_examples": int(max_examples),
        "max_completion_tokens": int(max_completion_tokens),
        "output": output,
    }
    if policy in (C8_POLICY, ORIGINAL_BLENDED_POLICY):
        stats.update({
            "unique_clean_problems": len({c8_problem_key(r) for r in clean}),
            "accepted_unique_problems": len({c8_problem_key(r) for r in accepted}),
            "accepted_per_source": dict(sorted(Counter(
                str(r.get("source_file", "") or "") for r in accepted
            ).items())),
            "source_balance": bool(source_balance),
        })
    print(f"[filter-pool] funnel ({policy}): raw={stats['raw']} -> "
          f"correct={stats['correct']} -> clean_correct={stats['clean_correct']} -> "
          f"accepted={stats['accepted']} (cap {max_examples}) -> {output}", flush=True)
    return output, stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--trajectories", nargs="+", required=True,
                   help="one or more explorer_trajectories.jsonl paths")
    p.add_argument("--policy", default="quality_pool",
                   choices=["quality_pool", "naive_pool", "best_scout", C8_POLICY,
                            ORIGINAL_BLENDED_POLICY])
    p.add_argument("--c8-source-balance", action="store_true",
                   help="round-robin c8 per-problem winners across source files")
    p.add_argument("--max-examples", type=int, default=500,
                   help="accepted cap; contract 500 (TPU max_accepted_trajectories)")
    p.add_argument("--max-completion-tokens", type=int, default=MAX_COMPLETION_TOKENS,
                   help="upper token gate; GPU-MAX runs pass budget - 384")
    p.add_argument("--min-accepted", type=int, default=0,
                   help="fail if fewer rows survive (0 = no floor)")
    p.add_argument("--output", default="accepted.jsonl")
    args = p.parse_args()
    _, stats = pool_trajectories(
        args.trajectories,
        policy=args.policy,
        max_examples=args.max_examples,
        output=args.output,
        max_completion_tokens=args.max_completion_tokens,
        source_balance=args.c8_source_balance,
    )
    if args.min_accepted and stats["accepted"] < args.min_accepted:
        raise SystemExit(
            f"[filter-pool] FATAL: accepted={stats['accepted']} below "
            f"--min-accepted={args.min_accepted}; collect more trajectories."
        )


if __name__ == "__main__":
    main()
