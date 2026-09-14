"""Coverage guard for pinned MATH500 and Minerva exact-answer syntax."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tmx_gpu import eval_gpu, gold_syntax_audit


REPORT_PATH = (
    Path(__file__).resolve().parents[2]
    / "baseline_runs_gpu"
    / "GOLD_SYNTAX_AUDIT.json"
)


@pytest.mark.parametrize(
    ("answer", "category"),
    [
        (r"\frac{-1}{2}", "numeric_equivalence_supported"),
        ("4.5e33", "numeric_equivalence_supported"),
        (r"\frac{3\sqrt{3}}{4}", "exact_string_symbolic_fraction"),
        (r"3\sqrt{13}", "exact_string_radical"),
        (r"18+2\pi", "exact_string_pi_expression"),
        (r"\text{east}", "exact_string_text_or_units"),
        ("(6,31,-1)", "exact_string_tuple_interval_or_function"),
        ("x^3+3x-6", "exact_string_algebraic_or_text"),
    ],
)
def test_gold_syntax_classifier_is_explicit(answer: str, category: str) -> None:
    assert gold_syntax_audit.classify_gold_syntax(answer) == category


def test_checked_in_gold_audit_is_bound_to_pinned_canonical_rows() -> None:
    report = json.loads(REPORT_PATH.read_text())
    assert report["schema_version"] == "tmx_gpu.gold_syntax_audit_v1"
    assert set(report["benchmarks"]) == set(
        gold_syntax_audit.AUDITED_BENCHMARKS)
    for benchmark, item in report["benchmarks"].items():
        spec = eval_gpu.BENCHMARKS[benchmark]
        assert item["dataset_revision"] == spec["revision"]
        assert item["canonical_rows_sha256"] == spec["canonical_rows_sha256"]
        assert item["problem_count"] == spec["canonical_count"]
        assert sum(item["syntax_counts"].values()) == item["problem_count"]
        assert item["numeric_equivalence_supported_count"] == (
            item["syntax_counts"]["numeric_equivalence_supported"])
        assert item["exact_string_only_count"] == (
            item["problem_count"] - item["numeric_equivalence_supported_count"])
        # Unsupported symbolic categories are visible by design; do not let a
        # verifier change silently reclassify them as numeric equivalence.
        assert item["exact_string_only_count"] > 0


@pytest.mark.skipif(
    os.environ.get("TMX_VERIFY_PINNED_DATASETS") != "1",
    reason="set TMX_VERIFY_PINNED_DATASETS=1 to re-download/replay pinned rows",
)
def test_live_pinned_gold_audit_matches_checked_in_report() -> None:
    expected = json.loads(REPORT_PATH.read_text())
    observed = {
        "schema_version": "tmx_gpu.gold_syntax_audit_v1",
        "verifier_scope": (
            "bounded numeric equivalence plus normalized exact-string fallback"),
        "benchmarks": {
            benchmark: gold_syntax_audit.audit_pinned_benchmark(benchmark)
            for benchmark in gold_syntax_audit.AUDITED_BENCHMARKS
        },
    }
    assert observed == expected
