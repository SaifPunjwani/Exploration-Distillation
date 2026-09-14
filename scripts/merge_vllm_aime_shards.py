#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from expdis_torch.eval_plot import _pass_at_k_estimate


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {"count": 0}
    correct = [1.0 if bool(row.get("is_correct", False)) else 0.0 for row in samples]
    prompt_lengths = [float(row.get("prompt_token_length", 0) or 0.0) for row in samples]
    completion_lengths = [float(row.get("completion_token_length", 0) or 0.0) for row in samples]
    latencies = [float(row.get("latency_seconds", 0.0) or 0.0) for row in samples]
    valid_answer = [1.0 if bool(row.get("valid_answer_presentation", False)) else 0.0 for row in samples]
    return {
        "count": int(len(samples)),
        "correct_fraction": _mean(correct),
        "mean_prompt_token_length": _mean(prompt_lengths),
        "mean_completion_token_length": _mean(completion_lengths),
        "valid_answer_fraction": _mean(valid_answer),
        "mean_latency_seconds": _mean(latencies),
        "examples": {
            "correct": [row for row in samples if bool(row.get("is_correct", False))][:3],
            "incorrect": [row for row in samples if not bool(row.get("is_correct", False))][:3],
        },
    }


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge sharded vLLM AIME benchmark outputs.")
    parser.add_argument("--run-dir", required=True, help="Output run dir for the merged benchmark.")
    parser.add_argument("--role-name", default="base")
    parser.add_argument("--shard-run-dirs", nargs="+", required=True)
    args = parser.parse_args()

    role = args.role_name
    shard_dirs = [Path(p).resolve() for p in args.shard_run_dirs]
    run_dir = Path(args.run_dir).resolve()
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    merged_samples: list[dict[str, Any]] = []
    merged_problem_rollouts: dict[str, list[dict[str, Any]]] = {}
    merged_problem_summaries: list[dict[str, Any]] = []
    protocol: dict[str, Any] | None = None
    manifest: dict[str, Any] | None = None
    dataset_name = ""
    dataset_config_name = ""

    for shard_dir in shard_dirs:
        summary = _load_json(shard_dir / "analysis" / "benchmark_summary.json")
        eval_samples = _load_json(shard_dir / "analysis" / "benchmark_eval_samples.json")
        rollouts = _load_json(shard_dir / "analysis" / f"benchmark_rollouts_{role}.json")

        dataset_name = str(summary.get("dataset_name") or dataset_name)
        dataset_config_name = str(summary.get("dataset_config_name") or dataset_config_name)
        if protocol is None:
            protocol = dict(summary.get("protocol") or {})
        if manifest is None:
            manifest = dict(summary.get("manifest") or {})

        merged_samples.extend(list((eval_samples.get(role) or [])))
        for problem_idx, rows in dict(rollouts).items():
            merged_problem_rollouts[str(problem_idx)] = list(rows)
        merged_problem_summaries.extend(list((summary.get("problem_summaries") or {}).get(role) or []))

    merged_problem_summaries.sort(key=lambda row: int(row.get("problem_idx", 0)))
    merged_samples.sort(key=lambda row: (int(row.get("problem_idx", 0)), int(row.get("rollout_idx", 0))))

    num_rollouts = int((protocol or {}).get("num_rollouts") or 1)
    pass_k_values = list((protocol or {}).get("pass_k_values") or [1])
    total_correct = int(sum(1 for row in merged_samples if bool(row.get("is_correct", False))))
    total_rollouts = int(len(merged_samples))
    accuracy = float(total_correct / max(1, total_rollouts))
    rollout_latencies = [float(row.get("latency_seconds", 0.0) or 0.0) for row in merged_samples]
    pass_at_k = {
        str(k): float(
            _mean([_pass_at_k_estimate(num_rollouts, row["num_correct"], int(k)) for row in merged_problem_summaries])
        )
        for k in pass_k_values
    }
    primary_metric_name = f"avg@{num_rollouts}"

    benchmark_rollout_path = analysis_dir / f"benchmark_rollouts_{role}.json"
    benchmark_eval_samples_path = analysis_dir / "benchmark_eval_samples.json"
    benchmark_summary_path = analysis_dir / "benchmark_summary.json"

    benchmark_rollout_path.write_text(json.dumps(merged_problem_rollouts, indent=2), encoding="utf-8")
    benchmark_eval_samples_path.write_text(json.dumps({role: merged_samples}, indent=2), encoding="utf-8")

    merged_manifest = dict(manifest or {})
    merged_manifest["shard_merge"] = {
        "source_run_dirs": [str(path) for path in shard_dirs],
        "num_shards_merged": len(shard_dirs),
    }
    merged_manifest["eval_count_selected"] = int(len(merged_problem_summaries))

    summary_payload = {
        "dataset_name": dataset_name,
        "dataset_config_name": dataset_config_name,
        "manifest": merged_manifest,
        "protocol": protocol or {},
        "base": _summarize_samples(merged_samples) if role == "base" else {},
        "explorer": _summarize_samples(merged_samples) if role == "explorer" else {},
        "actual": _summarize_samples(merged_samples) if role == "actual" else {},
        "accuracies": {
            "base": accuracy if role == "base" else None,
            "explorer": accuracy if role == "explorer" else None,
            "actual": accuracy if role == "actual" else None,
        },
        "primary_metrics": {
            "base": primary_metric_name if role == "base" else None,
            "explorer": primary_metric_name if role == "explorer" else None,
            "actual": primary_metric_name if role == "actual" else None,
        },
        "pass_at_k": {
            "base": pass_at_k if role == "base" else {},
            "explorer": pass_at_k if role == "explorer" else {},
            "actual": pass_at_k if role == "actual" else {},
        },
        "problem_summaries": {
            "base": merged_problem_summaries if role == "base" else [],
            "explorer": merged_problem_summaries if role == "explorer" else [],
            "actual": merged_problem_summaries if role == "actual" else [],
        },
        "rollout_json_paths": {
            role: str(benchmark_rollout_path),
        },
        "mean_latency_seconds": _mean(rollout_latencies),
    }
    benchmark_summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(f"merged summary: {benchmark_summary_path}")
    print(f"merged rollouts: {benchmark_rollout_path}")


if __name__ == "__main__":
    main()
