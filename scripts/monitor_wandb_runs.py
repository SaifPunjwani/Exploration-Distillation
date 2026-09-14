#!/usr/bin/env python3
import argparse
import json
import sys
import time
from typing import Any, Dict, List

import wandb


DEFAULT_KEYS = [
    "grpo/step",
    "grpo/rewards/GRPOReward/mean",
    "grpo/completions/clipped_ratio",
    "grpo/completions/mean_length",
    "train/wall_time_seconds",
    "eval/Base_accuracy",
    "eval/Explorer_accuracy",
    "eval/Actual_accuracy",
    "benchmark/Base_accuracy",
    "benchmark/Explorer_accuracy",
    "benchmark/Actual_accuracy",
    "benchmark/Base_avg_at_n",
    "benchmark/Explorer_avg_at_n",
    "benchmark/Actual_avg_at_n",
]


def _norm(v: Any) -> Any:
    if isinstance(v, float):
        return round(v, 6)
    return v


def _collect(api: wandb.Api, entity: str, project: str, prefixes: List[str], keys: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for run in api.runs(f"{entity}/{project}"):
        name = run.name or ""
        if prefixes and not any(name.startswith(prefix) for prefix in prefixes):
            continue
        summary = dict(run.summary)
        row: Dict[str, Any] = {
            "name": name,
            "id": run.id,
            "state": run.state,
            "url": run.url,
            "created_at": run.created_at,
            "_runtime": _norm(summary.get("_runtime")),
            "_step": _norm(summary.get("_step")),
        }
        for key in keys:
            row[key] = _norm(summary.get(key))
        out.append(row)
    out.sort(key=lambda r: (r["name"], r["id"]))
    return out


def _print(rows: List[Dict[str, Any]], keys: List[str]) -> None:
    if not rows:
        print("No matching runs.")
        return
    for row in rows:
        print(f"{row['name']} [{row['state']}] id={row['id']}")
        print(f"  url: {row['url']}")
        print(f"  created_at: {row['created_at']}  runtime_s: {row['_runtime']}  summary_step: {row['_step']}")
        for key in keys:
            value = row.get(key)
            if value is not None:
                print(f"  {key}: {value}")
        print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY") or None)
    ap.add_argument("--project", default="two-model-explore")
    ap.add_argument("--prefix", action="append", dest="prefixes", default=[])
    ap.add_argument("--key", action="append", dest="keys", default=[])
    ap.add_argument("--json-out", default="")
    ap.add_argument("--poll-seconds", type=int, default=0)
    ap.add_argument("--iterations", type=int, default=1)
    args = ap.parse_args()

    keys = args.keys or DEFAULT_KEYS
    api = wandb.Api(timeout=40)

    remaining = max(1, args.iterations)
    while True:
        rows = _collect(api, args.entity, args.project, args.prefixes, keys)
        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump({"rows": rows, "keys": keys}, f, indent=2)
        _print(rows, keys)
        remaining -= 1
        if args.poll_seconds <= 0 or remaining <= 0:
            break
        time.sleep(args.poll_seconds)
        print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
