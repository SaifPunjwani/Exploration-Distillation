#!/usr/bin/env python3
"""Mirror JAX training log metrics into W&B.

This is an operational recovery tool for runs that were launched without
WANDB_API_KEY in the process environment. It parses `[train] step ...` log
lines and resumes the existing W&B run id when available.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path


STEP_RE = re.compile(
    r"^\[train\] step (?P<step>\d+)/(?P<target>\d+)\s+"
    r"loss=(?P<loss>[-+0-9.eE]+)\s+"
    r"reward=(?P<reward>[-+0-9.eE]+)\s+"
    r"(?:reward25=(?P<reward25>[-+0-9.eE]+)\s+)?"
    r"(?:reward_trend25=(?P<reward_trend25>[-+0-9.eE]+)\s+)?"
    r"(?:eff_groups=(?P<eff_groups>[-+0-9.eE]+)\s+)?"
    r"clip=(?P<clip>[-+0-9.eE]+)\s+"
    r"kl=(?P<kl>[-+0-9.eE]+)\s+"
    r"logp_gap_mean=(?P<gap_mean>[-+0-9.eE]+)\s+"
    r"logp_gap_max=(?P<gap_max>[-+0-9.eE]+)\s+"
    r"(?:external_gap_mean=(?P<external_gap_mean>[-+0-9.eE]+)\s+)?"
    r"(?:external_gap_max=(?P<external_gap_max>[-+0-9.eE]+)\s+)?"
    r"grad=(?P<grad>[-+0-9.eE]+)\s+"
    r"rollout=(?P<rollout>[-+0-9.eE]+)s\s+"
    r"rescore=(?P<rescore>[-+0-9.eE]+)s\s+"
    r"(?:ref_rescore=(?P<ref_rescore>[-+0-9.eE]+)s\s+)?"
    r"train=(?P<train>[-+0-9.eE]+)s"
)
URL_RE = re.compile(r"https://wandb\.ai/[^/\s]+/[^/\s]+/runs/([^/\s]+)")


def _read_key() -> None:
    if os.environ.get("WANDB_API_KEY"):
        return
    key_path = Path.home() / ".wandb_key"
    if key_path.exists():
        os.environ["WANDB_API_KEY"] = key_path.read_text().strip()


def _infer_run_id(log_text: str, fallback: str) -> str:
    ids = URL_RE.findall(log_text)
    non_pipeline = [x for x in ids if not x.startswith("jaxpipe-")]
    if non_pipeline:
        return non_pipeline[-1]
    if ids:
        return ids[-1]
    return fallback


def _payload(match: re.Match[str]) -> tuple[int, dict]:
    vals = {
        k: float(v)
        for k, v in match.groupdict().items()
        if k not in ("step", "target") and v is not None
    }
    step = int(match.group("step"))
    total = vals["rollout"] + vals["rescore"] + vals.get("ref_rescore", 0.0) + vals["train"]
    payload = {
        "explorer_grpo/step": step,
        "explorer_grpo/loss": vals["loss"],
        "explorer_grpo/reward": vals["reward"],
        "explorer_grpo/clip_fraction": vals["clip"],
        "explorer_grpo/kl": vals["kl"],
        "explorer_grpo/grad_norm": vals["grad"],
        "explorer_grpo/rollout_time_s": vals["rollout"],
        "explorer_grpo/rescore_time_s": vals["rescore"],
        "explorer_grpo/ref_rescore_time_s": vals.get("ref_rescore", 0.0),
        "explorer_grpo/train_time_s": vals["train"],
        "explorer_grpo/total_time_s": total,
    }
    if "reward25" in vals:
        payload["explorer_grpo/reward_rolling_25"] = vals["reward25"]
    if "reward_trend25" in vals:
        payload["explorer_grpo/reward_trend_25"] = vals["reward_trend25"]
    if "eff_groups" in vals:
        payload["explorer_grpo/effective_group_fraction"] = vals["eff_groups"]
    return step, payload


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / max(1, len(xs)))


def _trajectory_summary(traj_path: Path, *, num_generations: int = 8) -> tuple[int | None, dict]:
    if not traj_path.exists():
        return None, {}

    rows: list[dict] = []
    with traj_path.open(errors="ignore") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    if not rows:
        return None, {}

    def values(key: str) -> list[float]:
        out = []
        for row in rows:
            try:
                out.append(float(row.get(key, 0.0) or 0.0))
            except Exception:
                out.append(0.0)
        return out

    rewards = values("blended_reward")
    correctness_rewards = values("correctness_reward")
    novelty_rewards = values("novelty_reward")
    lengths = values("completion_token_length")
    correct = [1.0 if bool(row.get("is_correct", False)) else 0.0 for row in rows]
    clipped = [1.0 if bool(row.get("clipped", False)) else 0.0 for row in rows]
    terminated = [1.0 if bool(row.get("terminated", False)) else 0.0 for row in rows]
    valid_answer = [1.0 if bool(row.get("valid_answer", False)) else 0.0 for row in rows]
    steps = []
    for row in rows:
        try:
            steps.append(int(row.get("explorer_step", -1)))
        except Exception:
            pass
    if not steps:
        return None, {}
    # `explorer_step` is written before incrementing the optimizer step.
    wandb_step = max(steps) + 1

    prompt_any = []
    for i in range(0, len(correct), num_generations):
        group = correct[i:i + num_generations]
        if group:
            prompt_any.append(1.0 if max(group) > 0.5 else 0.0)

    terminated_idx = [i for i, v in enumerate(terminated) if v > 0.5]
    if terminated_idx:
        terminated_final = _mean([rewards[i] for i in terminated_idx])
        terminated_correctness = _mean([correctness_rewards[i] for i in terminated_idx])
        terminated_novelty = _mean([novelty_rewards[i] for i in terminated_idx])
    else:
        terminated_final = 0.0
        terminated_correctness = 0.0
        terminated_novelty = 0.0

    payload = {
        "explorer_grpo/correctness": _mean(correct),
        "explorer_grpo/prompt_any_correct_fraction": _mean(prompt_any),
        "explorer_grpo/valid_answer_fraction": _mean(valid_answer),
        "explorer_grpo/terminated_fraction": _mean(terminated),
        "explorer_grpo/clipped_fraction": _mean(clipped),
        "explorer_grpo/reward": _mean(rewards),
        "explorer_grpo/correctness_used_mean": _mean(correctness_rewards),
        "explorer_grpo/novelty_reward_mean": _mean(novelty_rewards),
        "explorer_grpo/terminated_correctness_mean": terminated_correctness,
        "explorer_grpo/terminated_novelty_mean": terminated_novelty,
        "traj/count": float(len(rows)),
        "traj/reward_mean": _mean(rewards),
        "traj/correct_fraction": _mean(correct),
        "traj/prompt_any_correct_fraction": _mean(prompt_any),
        "traj/valid_answer_fraction": _mean(valid_answer),
        "traj/terminated_fraction": _mean(terminated),
        "traj/clipped_fraction": _mean(clipped),
    }
    return wandb_step, payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "two-model-explore"))
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY") or None)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--state-path", default="")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    args = parser.parse_args()

    _read_key()
    import wandb

    log_path = Path(args.log_path)
    state_path = Path(args.state_path) if args.state_path else log_path.with_suffix(".wandb_mirror_state.json")
    logged: set[int] = set()
    last_traj_logged_step = 0
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
            logged = {int(x) for x in state.get("logged_steps", [])}
            last_traj_logged_step = int(state.get("last_traj_logged_step", 0) or 0)
        except Exception:
            logged = set()
            last_traj_logged_step = 0

    text = log_path.read_text(errors="ignore") if log_path.exists() else ""
    run_id = args.run_id or _infer_run_id(text, "mirror-" + re.sub(r"[^a-zA-Z0-9_-]", "-", args.run_name)[:64])
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name + "_explorer",
        id=run_id,
        resume="allow",
        config={"source": "tmx_jax/scripts/mirror_wandb_from_log.py", "log_path": str(log_path)},
    )
    try:
        wandb.define_metric("explorer_grpo/step")
        wandb.define_metric("explorer_grpo/*", step_metric="explorer_grpo/step")
        wandb.define_metric("traj/step")
        wandb.define_metric("traj/*", step_metric="traj/step")
    except Exception:
        pass
    explicit_step = (os.environ.get("TMX_WANDB_MIRROR_EXPLICIT_STEP", "0") or "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    try:
        while True:
            if log_path.exists():
                text = log_path.read_text(errors="ignore")
                for line in text.splitlines():
                    match = STEP_RE.match(line.strip())
                    if not match:
                        continue
                    step, payload = _payload(match)
                    if step in logged:
                        continue
                    if explicit_step:
                        run.log(payload, step=step)
                    else:
                        run.log(payload)
                    logged.add(step)
                traj_step, traj_payload = _trajectory_summary(
                    log_path.parent / "explorer" / "explorer_trajectories.jsonl"
                )
                if traj_step is not None and traj_payload and traj_step >= last_traj_logged_step:
                    if explicit_step:
                        run.log(traj_payload, step=traj_step)
                    else:
                        run.log(traj_payload)
                    last_traj_logged_step = int(traj_step)
                state_path.write_text(
                    json.dumps(
                        {
                            "logged_steps": sorted(logged),
                            "last_traj_logged_step": last_traj_logged_step,
                        },
                        indent=2,
                    )
                )
            time.sleep(max(5.0, args.poll_seconds))
    finally:
        run.finish()


if __name__ == "__main__":
    main()
