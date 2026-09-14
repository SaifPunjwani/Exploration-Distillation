#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Tuple


def _parse_args_list(args: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    i = 0
    while i < len(args):
        tok = args[i]
        if not tok.startswith("--"):
            i += 1
            continue
        if "=" in tok:
            k, v = tok[2:].split("=", 1)
            out[k] = v
            i += 1
            continue
        key = tok[2:]
        if i + 1 < len(args) and not args[i + 1].startswith("--"):
            out[key] = args[i + 1]
            i += 2
        else:
            out[key] = "true"
            i += 1
    return out


def _status_from_output_log(path: str) -> str:
    if not os.path.exists(path):
        return "unknown"
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
    except Exception:
        return "unknown"

    if "Traceback (most recent call last)" in txt:
        return "failed"
    if "Done. Check '" in txt:
        return "completed"

    # Recently updated logs are treated as actively running.
    try:
        mtime = os.path.getmtime(path)
        now = datetime.now(timezone.utc).timestamp()
        if now - mtime < 300:
            return "running"
    except Exception:
        pass
    return "incomplete"


def _coverage_tags(argm: Dict[str, str]) -> Tuple[bool, bool, bool]:
    has_id_ood = (
        argm.get("topic-split-mode", "") == "heuristic_math_topics"
        and bool(argm.get("id-ood-holdout-topics", "").strip())
    )
    has_embedding = argm.get("embedding-novelty-eval", "") == "true"
    feat_sweep = argm.get("novelty-feature-source-sweep", "")
    has_multilayer = ("multilayer" in feat_sweep.split(",")) or bool(argm.get("novelty-layers", "").strip())
    return has_id_ood, has_embedding, has_multilayer


def _intent_label(argm: Dict[str, str]) -> str:
    if argm.get("novelty-study-mode") == "full":
        return "Full novelty study"
    if argm.get("reward-shape-sweep"):
        return "Reward-shape ablation sweep"
    if argm.get("novelty-feature-source-sweep"):
        return "Feature-source ablation sweep"
    if argm.get("reward-incorrect") == "-1.0":
        return "Reward semantics check (incorrect=-1)"
    if argm.get("reward-incorrect") == "0.0":
        return "Reward semantics check (incorrect=0)"
    return "General decoupled run"


def _canonical_id(idx: int, argm: Dict[str, str]) -> str:
    model = argm.get("model-name", "model")
    if "Qwen2.5-0.5B" in model:
        mtag = "QWEN05B"
    elif "SmolLM2-135M" in model:
        mtag = "SMOL135M"
    else:
        mtag = "OTHER"
    device = (argm.get("device") or "auto").upper()
    mode = "FULL" if argm.get("novelty-study-mode") == "full" else "ABL"
    return f"EXP-{idx:02d}-{mode}-{device}-{mtag}"


def _collect_wandb_runs(wandb_root: str) -> List[Dict[str, str]]:
    metas = sorted(glob.glob(os.path.join(wandb_root, "run-*", "files", "wandb-metadata.json")))
    rows: List[Dict[str, str]] = []
    for meta in metas:
        run_dir = os.path.dirname(os.path.dirname(meta))
        run_slug = os.path.basename(run_dir)
        run_id = run_slug.split("-")[-1]

        try:
            payload = json.load(open(meta, "r", encoding="utf-8"))
        except Exception:
            continue

        args = payload.get("args", []) or []
        argm = _parse_args_list(args)

        if not argm.get("decoupling-mode"):
            continue

        has_id_ood, has_embedding, has_multilayer = _coverage_tags(argm)
        output_log = os.path.join(run_dir, "files", "output.log")
        status = _status_from_output_log(output_log)

        plots_dir = argm.get("plots-dir", "")
        local_run_dir = os.path.dirname(plots_dir) if plots_dir.endswith("/plots") else plots_dir
        wandb_entity = os.environ.get("WANDB_ENTITY", "")
        wandb_url = f"https://wandb.ai/{wandb_entity}/two-model-explore/runs/{run_id}" if wandb_entity else ""

        rows.append(
            {
                "wandb_run_id": run_id,
                "wandb_local_dir": run_dir,
                "wandb_url": wandb_url,
                "status": status,
                "intent": _intent_label(argm),
                "device": argm.get("device", ""),
                "model": argm.get("model-name", ""),
                "dataset": argm.get("dataset-name", ""),
                "decoupling_mode": argm.get("decoupling-mode", ""),
                "actual_train_mode": argm.get("actual-train-mode", ""),
                "base_eval_mode": argm.get("base-eval-mode", ""),
                "novelty_study_mode": argm.get("novelty-study-mode", "off"),
                "novelty_feature_source_sweep": argm.get("novelty-feature-source-sweep", argm.get("novelty-feature-source", "")),
                "reward_shape": argm.get("reward-shape", ""),
                "reward_shape_sweep": argm.get("reward-shape-sweep", ""),
                "lambda_novelty": argm.get("lambda-novelty", ""),
                "novelty_metric": argm.get("novelty-metric", ""),
                "id_ood_enabled": str(has_id_ood).lower(),
                "embedding_eval_enabled": str(has_embedding).lower(),
                "multilayer_eval_enabled": str(has_multilayer).lower(),
                "id_ood_holdout_topics": argm.get("id-ood-holdout-topics", ""),
                "topic_split_mode": argm.get("topic-split-mode", ""),
                "embedding_model": argm.get("embedding-model-name", ""),
                "embedding_cluster_k": argm.get("embedding-cluster-k", ""),
                "plots_dir": plots_dir,
                "local_run_dir": local_run_dir,
                "trajectory_path": argm.get("explorer-traj-path", ""),
                "wandb_run_name": argm.get("wandb-run-name", ""),
                "novelty_study_tag": argm.get("novelty-study-tag", ""),
            }
        )
    return rows


def _write_csv(path: str, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _write_markdown(path: str, rows: List[Dict[str, str]]) -> None:
    lines: List[str] = []
    lines.append("# Experiment Registry")
    lines.append("")
    lines.append("| exp_id | intent | status | model | device | novelty study | id/ood | embedding | multilayer | wandb |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            "| {exp_id} | {intent} | {status} | {model} | {device} | {novelty_study_mode} | {id_ood_enabled} | {embedding_eval_enabled} | {multilayer_eval_enabled} | {wandb_run_id} |".format(
                **r
            )
        )
    lines.append("")
    lines.append("## Legend")
    lines.append("")
    lines.append("- `id/ood=true`: run has heuristic topic split + holdout topics configured.")
    lines.append("- `embedding=true`: run computes external embedding novelty/diversity analysis.")
    lines.append("- `multilayer=true`: run evaluates/uses multi-layer novelty features.")
    lines.append("")
    lines.append("## Paths")
    lines.append("")
    for r in rows:
        lines.append(f"- `{r['exp_id']}`: `{r['local_run_dir']}` | `{r['plots_dir']}` | `{r['trajectory_path']}`")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _write_focus_markdown(path: str, rows: List[Dict[str, str]]) -> None:
    focus = []
    for r in rows:
        run_path = r.get("wandb_local_dir", "")
        is_recent = ("20260216" in run_path) or ("20260217" in run_path)
        if not is_recent:
            continue
        if r.get("novelty_study_mode") == "full":
            focus.append(r)
            continue
        if r.get("intent", "").startswith("Reward semantics"):
            focus.append(r)
            continue
    lines: List[str] = []
    lines.append("# Meeting Focus Matrix")
    lines.append("")
    lines.append("| exp_id | question answered | status | id/ood | embedding clustering | key path |")
    lines.append("|---|---|---|---|---|---|")
    for r in focus:
        if r.get("novelty_study_mode") == "full":
            question = "Decoupled novelty study (base vs explorer vs actual) with advanced eval"
        elif "incorrect=-1" in r.get("intent", ""):
            question = "Reward semantics branch A (incorrect=-1)"
        elif "incorrect=0" in r.get("intent", ""):
            question = "Reward semantics branch B (incorrect=0)"
        else:
            question = r.get("intent", "")
        key_path = r.get("local_run_dir") or r.get("wandb_local_dir")
        lines.append(
            f"| {r['exp_id']} | {question} | {r['status']} | {r['id_ood_enabled']} | {r['embedding_eval_enabled']} | `{key_path}` |"
        )

    lines.append("")
    lines.append("## Current Primary Run")
    lines.append("")
    primary = None
    for r in reversed(rows):
        if r.get("status") == "running":
            primary = r
            break
    if primary:
        lines.append(f"- `exp_id`: `{primary['exp_id']}`")
        lines.append(f"- `wandb`: {primary['wandb_url']}")
        lines.append(f"- `intent`: {primary['intent']}")
        lines.append(f"- `coverage`: id/ood={primary['id_ood_enabled']}, embedding={primary['embedding_eval_enabled']}, multilayer={primary['multilayer_eval_enabled']}")
        lines.append(f"- `local`: `{primary['local_run_dir']}`")
    else:
        lines.append("- No currently running run detected from local W&B metadata.")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb-root", default="wandb")
    ap.add_argument("--out-dir", default="runs/experiment_registry")
    args = ap.parse_args()

    rows = _collect_wandb_runs(args.wandb_root)
    rows = sorted(rows, key=lambda r: r["wandb_local_dir"])
    for i, r in enumerate(rows, start=1):
        r["exp_id"] = _canonical_id(i, {"model-name": r["model"], "device": r["device"], "novelty-study-mode": r["novelty_study_mode"]})

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "experiment_registry.csv")
    md_path = os.path.join(args.out_dir, "experiment_registry.md")
    focus_md_path = os.path.join(args.out_dir, "meeting_focus_matrix.md")
    json_path = os.path.join(args.out_dir, "experiment_registry.json")

    _write_csv(csv_path, rows)
    _write_markdown(md_path, rows)
    _write_focus_markdown(focus_md_path, rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"rows={len(rows)}")
    print(f"csv={csv_path}")
    print(f"md={md_path}")
    print(f"focus_md={focus_md_path}")
    print(f"json={json_path}")


if __name__ == "__main__":
    main()
