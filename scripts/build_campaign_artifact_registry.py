#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def _first_existing(paths: list[Path]) -> str:
    for path in paths:
        if path.exists():
            return str(path.resolve())
    return ""


def _rows_from_campaign(campaign_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for novelty_path in sorted(campaign_root.glob("**/analysis/novelty_summary.json")):
        run_dir = novelty_path.parent.parent
        analysis_dir = novelty_path.parent
        novelty = _load_json(novelty_path)
        config = novelty.get("config") or {}
        base_artifact = novelty.get("base_model_artifact") or {}
        actual_artifact = novelty.get("actual_model_artifact") or {}
        benchmark = novelty.get("benchmark_summary") or {}
        benchmark_rollouts = benchmark.get("rollout_json_paths") or {}

        trajectory_audit = analysis_dir / "trajectory_audit.json"
        eval_samples = analysis_dir / "eval_samples.json"
        benchmark_summary = analysis_dir / "benchmark_summary.json"
        benchmark_eval_samples = analysis_dir / "benchmark_eval_samples.json"
        accuracy_over_time = analysis_dir / "accuracy_over_time.json"
        multi_round_summary = analysis_dir / "multi_round_summary.json"
        explorer_traj = _first_existing([
            run_dir / "explorer_trajectories.jsonl",
            run_dir / "explorer_trajectories_retry1.jsonl",
        ])

        rows.append(
            {
                "run_dir": str(run_dir.resolve()),
                "analysis_dir": str(analysis_dir.resolve()),
                "run_name": config.get("wandb_run_name") or run_dir.name,
                "model_name": config.get("model_name", ""),
                "dataset_name": config.get("dataset_name", ""),
                "pipeline_mode": config.get("pipeline_mode", ""),
                "actual_train_mode": config.get("actual_train_mode", ""),
                "actual_target_contract": config.get("actual_target_contract", ""),
                "saved_model_root": config.get("saved_model_root", ""),
                "trajectory_audit_path": str(trajectory_audit.resolve()) if trajectory_audit.exists() else "",
                "explorer_trajectories_path": explorer_traj,
                "eval_samples_path": str(eval_samples.resolve()) if eval_samples.exists() else "",
                "accuracy_over_time_path": str(accuracy_over_time.resolve()) if accuracy_over_time.exists() else "",
                "multi_round_summary_path": str(multi_round_summary.resolve()) if multi_round_summary.exists() else "",
                "benchmark_summary_path": str(benchmark_summary.resolve()) if benchmark_summary.exists() else "",
                "benchmark_eval_samples_path": str(benchmark_eval_samples.resolve()) if benchmark_eval_samples.exists() else "",
                "benchmark_rollout_base_path": benchmark_rollouts.get("base", ""),
                "benchmark_rollout_explorer_path": benchmark_rollouts.get("explorer", ""),
                "benchmark_rollout_actual_path": benchmark_rollouts.get("actual", ""),
                "base_model_status": (base_artifact or {}).get("status", ""),
                "base_model_dir": (base_artifact or {}).get("model_dir", ""),
                "actual_model_status": (actual_artifact or {}).get("status", ""),
                "actual_model_dir": (actual_artifact or {}).get("model_dir", ""),
                "base_model_artifact": base_artifact,
                "actual_model_artifact": actual_artifact,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a registry of campaign run artifacts.")
    parser.add_argument("--campaign-root", required=True, help="Root directory for a campaign, e.g. runs/<tag>.")
    parser.add_argument("--out-dir", default="", help="Directory to write registry outputs into.")
    args = parser.parse_args()

    campaign_root = Path(args.campaign_root).expanduser().resolve()
    if not campaign_root.exists():
        raise SystemExit(f"campaign root not found: {campaign_root}")

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else campaign_root / "registry"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _rows_from_campaign(campaign_root)
    rows = sorted(rows, key=lambda row: (row.get("model_name", ""), row.get("run_name", "")))

    json_path = out_dir / "campaign_registry.json"
    csv_path = out_dir / "campaign_registry.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    _write_csv(csv_path, rows)

    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
