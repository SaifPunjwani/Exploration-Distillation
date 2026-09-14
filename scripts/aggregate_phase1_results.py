#!/usr/bin/env python3
import argparse
import csv
import json
import os
from collections import defaultdict


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def _rel_condition(root: str, analysis_path: str) -> tuple[str, str]:
    rel = os.path.relpath(analysis_path, root)
    parts = rel.split(os.sep)
    if len(parts) >= 4 and parts[-2] == "analysis":
        dataset = parts[0]
        condition = parts[1]
        return dataset, condition
    if len(parts) == 2 and parts[0] == "analysis":
        return "unknown", os.path.basename(os.path.abspath(root))
    return "unknown", "unknown"


def load_records(root: str):
    records = []
    paths = []
    for dirpath, _, filenames in os.walk(root):
        if os.path.basename(dirpath) != "analysis":
            continue
        if "novelty_summary.json" not in filenames:
            continue
        paths.append(os.path.join(dirpath, "novelty_summary.json"))
    for path in sorted(paths):
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue
        cfg = payload.get("config", {}) or {}
        res = payload.get("results", {}) or {}
        manifest = payload.get("dataset_manifest", {}) or {}
        traj = payload.get("trajectory_summary", {}) or {}
        runtime = (payload.get("runtime_summary", {}) or {}).get("stage_timings", {}) or {}
        dataset_from_path, condition = _rel_condition(root, path)
        dataset_name = cfg.get("dataset_name") or dataset_from_path

        base_acc = _safe_float(res.get("base_accuracy"))
        explorer_acc = _safe_float(res.get("explorer_accuracy"))
        actual_acc = _safe_float(res.get("actual_accuracy"))
        base_novelty = _safe_float(res.get("base_novelty_mean"))
        explorer_novelty = _safe_float(res.get("explorer_novelty_mean"))
        actual_novelty = _safe_float(res.get("actual_novelty_mean"))

        record = {
            "path": path,
            "dataset_name": dataset_name,
            "dataset_config_name": cfg.get("dataset_config_name") or "",
            "condition": condition,
            "pipeline_mode": cfg.get("pipeline_mode", "unknown"),
            "actual_train_mode": cfg.get("actual_train_mode", "unknown"),
            "actual_reward_source": cfg.get("actual_reward_source", "unknown"),
            "actual_target_correct_fraction": _safe_float(cfg.get("actual_target_correct_fraction")),
            "anchor_metric": cfg.get("anchor_metric", "none"),
            "anchor_weight": _safe_float(cfg.get("anchor_weight")),
            "novelty_metric": cfg.get("novelty_metric", "unknown"),
            "novelty_feature_source": cfg.get("novelty_feature_source", "unknown"),
            "lambda_novelty": _safe_float(cfg.get("lambda_novelty")),
            "model_name": cfg.get("model_name", ""),
            "wandb_run_name": cfg.get("wandb_run_name", ""),
            "device_preference": cfg.get("device_preference", ""),
            "train_count_selected": _safe_int(manifest.get("train_count_selected")),
            "eval_count_selected": _safe_int(manifest.get("eval_count_selected")),
            "resolved_holdout_topics": ",".join(manifest.get("resolved_holdout_topics", []) or []),
            "trajectory_count": _safe_int(traj.get("traj/count")),
            "trajectory_correct_fraction": _safe_float(traj.get("traj/correct_fraction")),
            "base_accuracy": base_acc,
            "explorer_accuracy": explorer_acc,
            "actual_accuracy": actual_acc,
            "explorer_minus_base": explorer_acc - base_acc,
            "actual_minus_base": actual_acc - base_acc,
            "actual_minus_explorer": actual_acc - explorer_acc,
            "base_novelty_mean": base_novelty,
            "explorer_novelty_mean": explorer_novelty,
            "actual_novelty_mean": actual_novelty,
            "explorer_novelty_minus_base": explorer_novelty - base_novelty,
            "actual_novelty_minus_base": actual_novelty - base_novelty,
            "base_id_accuracy": _safe_float(res.get("base_id_accuracy")),
            "base_ood_accuracy": _safe_float(res.get("base_ood_accuracy")),
            "explorer_id_accuracy": _safe_float(res.get("explorer_id_accuracy")),
            "explorer_ood_accuracy": _safe_float(res.get("explorer_ood_accuracy")),
            "actual_id_accuracy": _safe_float(res.get("actual_id_accuracy")),
            "actual_ood_accuracy": _safe_float(res.get("actual_ood_accuracy")),
            "total_wall_clock_seconds": _safe_float(runtime.get("total_wall_clock_seconds")),
            "baseline_train_seconds": _safe_float(runtime.get("baseline_train_seconds")),
            "explorer_train_seconds": _safe_float(runtime.get("explorer_train_seconds")),
            "trajectory_collection_seconds": _safe_float(runtime.get("trajectory_collection_seconds")),
            "actual_train_seconds": _safe_float(runtime.get("actual_train_seconds")),
            "eval_seconds": _safe_float(runtime.get("eval_seconds")),
        }
        records.append(record)
    return records


def _sort_key(row):
    return (
        row["dataset_name"],
        -row["actual_minus_base"],
        -row["explorer_minus_base"],
        row["total_wall_clock_seconds"],
        row["condition"],
    )


def build_best_by_dataset(records):
    grouped = defaultdict(list)
    for row in records:
        grouped[row["dataset_name"]].append(row)

    best_rows = []
    for dataset_name, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: (-row["actual_minus_base"], -row["explorer_minus_base"], row["total_wall_clock_seconds"]))
        best = dict(rows[0])
        best["num_runs"] = len(rows)
        best_rows.append(best)
    return best_rows


def _write_csv(path: str, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: str, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _print_ranked_summary(records, best_rows):
    if not records:
        print("No phase-1 records found.")
        return

    print("Top runs by actual-minus-base:")
    for row in sorted(records, key=_sort_key)[:10]:
        print(
            f"  dataset={row['dataset_name']} condition={row['condition']} "
            f"actual-base={row['actual_minus_base']:.4f} "
            f"explorer-base={row['explorer_minus_base']:.4f} "
            f"actual={row['actual_accuracy']:.4f} base={row['base_accuracy']:.4f} "
            f"wall_clock_s={row['total_wall_clock_seconds']:.1f} "
            f"run={row['wandb_run_name'] or os.path.basename(os.path.dirname(os.path.dirname(row['path'])))}"
        )

    print("\nBest run per dataset:")
    for row in best_rows:
        print(
            f"  dataset={row['dataset_name']} best_condition={row['condition']} "
            f"actual-base={row['actual_minus_base']:.4f} "
            f"explorer-base={row['explorer_minus_base']:.4f} "
            f"actual_train_mode={row['actual_train_mode']} "
            f"actual_reward_source={row['actual_reward_source']} "
            f"target_correct_fraction={row['actual_target_correct_fraction']:.2f} "
            f"anchor_metric={row['anchor_metric']} "
            f"wall_clock_s={row['total_wall_clock_seconds']:.1f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Aggregate phase-1 run outputs into comparable tables.")
    parser.add_argument("--root", type=str, default="runs", help="Root directory to scan for analysis outputs.")
    parser.add_argument("--out-dir", type=str, default="runs/phase1_aggregate", help="Directory to write summary files.")
    args = parser.parse_args()

    records = load_records(args.root)
    records = sorted(records, key=_sort_key)
    best_rows = build_best_by_dataset(records)

    os.makedirs(args.out_dir, exist_ok=True)
    _write_json(os.path.join(args.out_dir, "phase1_run_table.json"), records)
    _write_csv(os.path.join(args.out_dir, "phase1_run_table.csv"), records)
    _write_json(os.path.join(args.out_dir, "phase1_best_by_dataset.json"), best_rows)
    _write_csv(os.path.join(args.out_dir, "phase1_best_by_dataset.csv"), best_rows)
    _print_ranked_summary(records, best_rows)


if __name__ == "__main__":
    main()
