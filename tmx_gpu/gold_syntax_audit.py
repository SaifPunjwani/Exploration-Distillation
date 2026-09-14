"""Audit the exact-answer syntax covered by the publication verifier.

The verifier intentionally evaluates only a small numeric grammar.  All other
answers remain eligible for normalized exact-string comparison, but are
classified here so that symbolic golds cannot be mistaken for numeric-
equivalence coverage.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from typing import Any, Iterable

from tmx_jax.rewarding import _parse_simple_numeric
from tmx_gpu.eval_gpu import BENCHMARKS, load_benchmark, validate_loaded_benchmark


AUDITED_BENCHMARKS = ("MATH500", "Minerva-Math")


def classify_gold_syntax(answer: str) -> str:
    """Classify numeric-equivalence support or the exact-string fallback."""
    if _parse_simple_numeric(answer) is not None:
        return "numeric_equivalence_supported"
    text = str(answer or "").strip()
    if re.search(r"\\(?:frac|dfrac|tfrac)", text):
        return "exact_string_symbolic_fraction"
    if re.search(r"\\sqrt", text):
        return "exact_string_radical"
    if re.search(r"\\pi|π", text):
        return "exact_string_pi_expression"
    if re.search(r"\\(?:text|mathrm|operatorname|mbox)", text):
        return "exact_string_text_or_units"
    if any(token in text for token in ("(", ")", "[", "]")):
        return "exact_string_tuple_interval_or_function"
    if any(token in text for token in ("{", "}")):
        return "exact_string_set_matrix_or_braced"
    if re.search(r"[A-Za-z]", text):
        return "exact_string_algebraic_or_text"
    if any(token in text for token in ("^", "\\", "±")):
        return "exact_string_other_math"
    return "exact_string_other"


def audit_gold_answers(
    benchmark: str, answers: Iterable[str], *, rows_sha256: str,
) -> dict[str, Any]:
    golds = [str(answer) for answer in answers]
    counts = collections.Counter(classify_gold_syntax(answer) for answer in golds)
    numeric_count = int(counts.get("numeric_equivalence_supported", 0))
    return {
        "benchmark": benchmark,
        "dataset_revision": BENCHMARKS[benchmark]["revision"],
        "canonical_rows_sha256": rows_sha256,
        "problem_count": len(golds),
        "gold_answers_sha256": hashlib.sha256(json.dumps(
            golds, ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8")).hexdigest(),
        "numeric_equivalence_supported_count": numeric_count,
        "exact_string_only_count": len(golds) - numeric_count,
        "syntax_counts": dict(sorted(counts.items())),
    }


def audit_pinned_benchmark(benchmark: str) -> dict[str, Any]:
    if benchmark not in AUDITED_BENCHMARKS:
        raise ValueError(f"unsupported gold-syntax audit benchmark: {benchmark}")
    rows = load_benchmark(benchmark)
    rows_sha256 = validate_loaded_benchmark(
        benchmark, rows, publication=True)
    return audit_gold_answers(
        benchmark,
        (str(row["ground_truth"]) for row in rows),
        rows_sha256=rows_sha256,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "benchmarks", nargs="*", choices=AUDITED_BENCHMARKS,
        default=list(AUDITED_BENCHMARKS),
    )
    args = parser.parse_args()
    report = {
        "schema_version": "tmx_gpu.gold_syntax_audit_v1",
        "verifier_scope": (
            "bounded numeric equivalence plus normalized exact-string fallback"),
        "benchmarks": {
            benchmark: audit_pinned_benchmark(benchmark)
            for benchmark in args.benchmarks
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
