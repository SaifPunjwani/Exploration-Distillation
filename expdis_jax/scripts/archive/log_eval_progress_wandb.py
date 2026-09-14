#!/usr/bin/env python3
"""Log live eval progress from eval.log to W&B.

This sidecar is intentionally read-only with respect to the eval run. Final
accuracy metrics still come from final_eval_aime24.json; this script only logs
progress/ETA while generation is still running.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import wandb

from expdis_jax.eval import aime24_wandb_metrics


PROGRESS_RE = re.compile(
    r"\[vllm-gen\]\s+(\d+)/(\d+)\s+subrequests done\s+\(([^s]+)s elapsed\)"
)


def parse_progress(log_path: Path) -> tuple[int, int, float] | None:
    if not log_path.exists():
        return None
    latest: tuple[int, int, float] | None = None
    with log_path.open(errors="ignore") as f:
        for line in f:
            match = PROGRESS_RE.search(line)
            if match:
                latest = (int(match.group(1)), int(match.group(2)), float(match.group(3)))
    return latest


def final_metrics(final_json: Path) -> dict[str, float | int | str]:
    if not final_json.exists() or final_json.stat().st_size == 0:
        return {}
    with final_json.open() as f:
        data = json.load(f)
    return aime24_wandb_metrics(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY") or None)
    parser.add_argument("--project", default="expdis")
    parser.add_argument("--run-id", default=os.environ.get("EXPDIS_WANDB_RUN_ID", ""))
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    log_path = root / "eval.log"
    final_json = root / "analysis" / "final_eval_aime24.json"
    if not final_json.exists():
        final_json = root / "final_eval_aime24.json"
    run_id = args.run_id or f"{args.run_name}_eval"

    run = wandb.init(
        entity=args.entity,
        project=args.project,
        name=args.run_name,
        id=run_id,
        resume="allow",
        job_type="eval",
    )
    assert run is not None

    last_done = -1
    while True:
        progress = parse_progress(log_path)
        if progress is not None:
            done, total, elapsed = progress
            if done != last_done:
                eta = (elapsed / done) * (total - done) if done else 0.0
                wandb.log(
                    {
                        "eval/aime24_32k/subrequests_done": done,
                        "eval/aime24_32k/subrequests_total": total,
                        "eval/aime24_32k/progress_fraction": done / total if total else 0.0,
                        "eval/aime24_32k/elapsed_seconds": elapsed,
                        "eval/aime24_32k/eta_seconds": eta,
                    }
                )
                last_done = done

        metrics = final_metrics(final_json)
        if metrics:
            metrics["eval/aime24_32k/max_tokens"] = 32768
            metrics["eval/aime24_32k/num_rollouts"] = 32
            wandb.log(metrics)
            artifact = wandb.Artifact(f"{args.run_name}_aime24_eval_json", type="eval_result")
            artifact.add_file(str(final_json))
            wandb.log_artifact(artifact)
            break

        time.sleep(args.poll_seconds)

    wandb.finish()


if __name__ == "__main__":
    main()
