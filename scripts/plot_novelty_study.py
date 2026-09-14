#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def _mean_std(vals):
    if not vals:
        return 0.0, 0.0
    m = sum(vals) / len(vals)
    var = sum((v - m) ** 2 for v in vals) / max(1, len(vals))
    return float(m), float(math.sqrt(max(var, 0.0)))


def load_records(root: str, tag_filter: str):
    paths = glob.glob(os.path.join(root, "**", "analysis", "novelty_summary.json"), recursive=True)
    records = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue
        cfg = payload.get("config", {})
        if tag_filter and cfg.get("novelty_study_tag") != tag_filter:
            continue
        res = payload.get("results", {})
        analysis_dir = os.path.dirname(path)
        emb_path = os.path.join(analysis_dir, "embedding_novelty_summary.json")
        topic_path = os.path.join(analysis_dir, "topic_id_ood_summary.json")

        emb_payload = {}
        if os.path.exists(emb_path):
            try:
                with open(emb_path, "r", encoding="utf-8") as ef:
                    emb_payload = json.load(ef)
            except Exception:
                emb_payload = {}

        topic_payload = {}
        if os.path.exists(topic_path):
            try:
                with open(topic_path, "r", encoding="utf-8") as tf:
                    topic_payload = json.load(tf)
            except Exception:
                topic_payload = {}

        emb_summary = emb_payload.get("summary_by_model", {}) if isinstance(emb_payload, dict) else {}
        emb_ptests = emb_payload.get("permutation_tests", {}) if isinstance(emb_payload, dict) else {}

        rec = {
            "path": path,
            "dataset_name": cfg.get("dataset_name", "unknown"),
            "reward_shape": cfg.get("reward_shape", "unknown"),
            "novelty_feature_source": cfg.get("novelty_feature_source", "unknown"),
            "novelty_metric": cfg.get("novelty_metric", "unknown"),
            "lambda_novelty": _safe_float(cfg.get("lambda_novelty", 0.0)),
            "base_accuracy": _safe_float(res.get("base_accuracy", 0.0)),
            "explorer_accuracy": _safe_float(res.get("explorer_accuracy", 0.0)),
            "actual_accuracy": _safe_float(res.get("actual_accuracy", 0.0)),
            "base_novelty_mean": _safe_float(res.get("base_novelty_mean", 0.0)),
            "explorer_novelty_mean": _safe_float(res.get("explorer_novelty_mean", 0.0)),
            "actual_novelty_mean": _safe_float(res.get("actual_novelty_mean", 0.0)),
            "base_id_accuracy": _safe_float(res.get("base_id_accuracy", 0.0)),
            "base_ood_accuracy": _safe_float(res.get("base_ood_accuracy", 0.0)),
            "explorer_id_accuracy": _safe_float(res.get("explorer_id_accuracy", 0.0)),
            "explorer_ood_accuracy": _safe_float(res.get("explorer_ood_accuracy", 0.0)),
            "actual_id_accuracy": _safe_float(res.get("actual_id_accuracy", 0.0)),
            "actual_ood_accuracy": _safe_float(res.get("actual_ood_accuracy", 0.0)),
            "base_dispersion_mean": _safe_float(emb_summary.get("Base", {}).get("dispersion_mean", 0.0)),
            "explorer_dispersion_mean": _safe_float(emb_summary.get("Explorer", {}).get("dispersion_mean", 0.0)),
            "actual_dispersion_mean": _safe_float(emb_summary.get("Actual", {}).get("dispersion_mean", 0.0)),
            "base_answer_diversity": _safe_float(emb_summary.get("Base", {}).get("answer_diversity_ratio", 0.0)),
            "explorer_answer_diversity": _safe_float(emb_summary.get("Explorer", {}).get("answer_diversity_ratio", 0.0)),
            "actual_answer_diversity": _safe_float(emb_summary.get("Actual", {}).get("answer_diversity_ratio", 0.0)),
            "p_dispersion_explorer_vs_base": _safe_float(
                emb_ptests.get("dispersion_explorer_vs_base", {}).get("p_value_two_sided", 1.0),
                default=1.0,
            ),
            "p_dispersion_actual_vs_base": _safe_float(
                emb_ptests.get("dispersion_actual_vs_base", {}).get("p_value_two_sided", 1.0),
                default=1.0,
            ),
            "p_dispersion_actual_vs_explorer": _safe_float(
                emb_ptests.get("dispersion_actual_vs_explorer", {}).get("p_value_two_sided", 1.0),
                default=1.0,
            ),
            "holdout_topics": ",".join(topic_payload.get("holdout_topics", []) or []),
        }
        records.append(rec)
    return records


def aggregate(records):
    groups = defaultdict(list)
    for r in records:
        key = (
            r["dataset_name"],
            r["reward_shape"],
            r["novelty_feature_source"],
            r["novelty_metric"],
            r["lambda_novelty"],
        )
        groups[key].append(r)

    rows = []
    for key, recs in sorted(groups.items()):
        base_acc = [r["base_accuracy"] for r in recs]
        explorer_acc = [r["explorer_accuracy"] for r in recs]
        actual_acc = [r["actual_accuracy"] for r in recs]
        base_nov = [r["base_novelty_mean"] for r in recs]
        explorer_nov = [r["explorer_novelty_mean"] for r in recs]
        actual_nov = [r["actual_novelty_mean"] for r in recs]
        base_gap = [r["base_id_accuracy"] - r["base_ood_accuracy"] for r in recs]
        explorer_gap = [r["explorer_id_accuracy"] - r["explorer_ood_accuracy"] for r in recs]
        actual_gap = [r["actual_id_accuracy"] - r["actual_ood_accuracy"] for r in recs]
        base_disp = [r["base_dispersion_mean"] for r in recs]
        explorer_disp = [r["explorer_dispersion_mean"] for r in recs]
        actual_disp = [r["actual_dispersion_mean"] for r in recs]
        p_exp_base = [r["p_dispersion_explorer_vs_base"] for r in recs]
        p_act_base = [r["p_dispersion_actual_vs_base"] for r in recs]

        row = {
            "dataset_name": key[0],
            "reward_shape": key[1],
            "novelty_feature_source": key[2],
            "novelty_metric": key[3],
            "lambda_novelty": key[4],
            "n_runs": len(recs),
            "base_acc_mean": _mean_std(base_acc)[0],
            "base_acc_std": _mean_std(base_acc)[1],
            "explorer_acc_mean": _mean_std(explorer_acc)[0],
            "explorer_acc_std": _mean_std(explorer_acc)[1],
            "actual_acc_mean": _mean_std(actual_acc)[0],
            "actual_acc_std": _mean_std(actual_acc)[1],
            "base_nov_mean": _mean_std(base_nov)[0],
            "base_nov_std": _mean_std(base_nov)[1],
            "explorer_nov_mean": _mean_std(explorer_nov)[0],
            "explorer_nov_std": _mean_std(explorer_nov)[1],
            "actual_nov_mean": _mean_std(actual_nov)[0],
            "actual_nov_std": _mean_std(actual_nov)[1],
            "base_id_ood_acc_gap_mean": _mean_std(base_gap)[0],
            "base_id_ood_acc_gap_std": _mean_std(base_gap)[1],
            "explorer_id_ood_acc_gap_mean": _mean_std(explorer_gap)[0],
            "explorer_id_ood_acc_gap_std": _mean_std(explorer_gap)[1],
            "actual_id_ood_acc_gap_mean": _mean_std(actual_gap)[0],
            "actual_id_ood_acc_gap_std": _mean_std(actual_gap)[1],
            "base_dispersion_mean": _mean_std(base_disp)[0],
            "explorer_dispersion_mean": _mean_std(explorer_disp)[0],
            "actual_dispersion_mean": _mean_std(actual_disp)[0],
            "p_dispersion_explorer_vs_base_mean": _mean_std(p_exp_base)[0],
            "p_dispersion_actual_vs_base_mean": _mean_std(p_act_base)[0],
        }
        rows.append(row)

    return rows


def write_outputs(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    summary_json = os.path.join(out_dir, "novelty_study_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    summary_csv = os.path.join(out_dir, "novelty_study_summary.csv")
    if rows:
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] matplotlib unavailable: {e}")
        return summary_json, summary_csv

    if rows:
        labels = [
            f"{r['dataset_name']}|{r['reward_shape']}|{r['novelty_feature_source']}"
            for r in rows
        ]

        fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.9), 4.8))
        ax.bar(labels, [100.0 * r["actual_acc_mean"] for r in rows], color="#54A24B", alpha=0.85)
        ax.set_ylabel("Actual accuracy (%)")
        ax.set_title("Novelty Study: Actual Accuracy by Condition")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "actual_accuracy_by_condition.png"), dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.9), 4.8))
        ax.bar(labels, [r["explorer_nov_mean"] for r in rows], color="#F58518", alpha=0.85)
        ax.set_ylabel("Explorer novelty mean")
        ax.set_title("Novelty Study: Explorer Novelty by Condition")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "explorer_novelty_by_condition.png"), dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.9), 4.8))
        ax.bar(labels, [r["actual_id_ood_acc_gap_mean"] for r in rows], color="#E45756", alpha=0.85)
        ax.set_ylabel("ID-OOD accuracy gap (Actual)")
        ax.set_title("Novelty Study: Actual ID-OOD Gap by Condition")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "actual_id_ood_gap_by_condition.png"), dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.9), 4.8))
        ax.bar(labels, [r["explorer_dispersion_mean"] for r in rows], color="#72B7B2", alpha=0.85, label="Explorer")
        ax.plot(labels, [r["actual_dispersion_mean"] for r in rows], color="#54A24B", marker="o", linewidth=2, label="Actual")
        ax.set_ylabel("Embedding dispersion")
        ax.set_title("Novelty Study: Embedding Dispersion by Condition")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "embedding_dispersion_by_condition.png"), dpi=160)
        plt.close(fig)

    return summary_json, summary_csv


def main():
    ap = argparse.ArgumentParser(description="Aggregate novelty study outputs from analysis/novelty_summary.json files.")
    ap.add_argument("--root", type=str, default="runs", help="Root directory to scan.")
    ap.add_argument("--tag", type=str, default="", help="Optional novelty_study_tag filter.")
    ap.add_argument("--out-dir", type=str, default="runs/novelty_aggregate", help="Output directory for aggregate files.")
    args = ap.parse_args()

    records = load_records(args.root, args.tag)
    if not records:
        print("No novelty study records found.")
        return

    rows = aggregate(records)
    summary_json, summary_csv = write_outputs(rows, args.out_dir)
    print(f"Aggregated {len(records)} runs into {len(rows)} grouped conditions.")
    print(f"JSON: {summary_json}")
    print(f"CSV:  {summary_csv}")


if __name__ == "__main__":
    main()
