#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TMX_SUBMIT = ROOT / "scripts" / "tmx_submit.py"

STAGE_SPECS = {
    "explorer-correctness": ROOT / "configs" / "production_trc2_v5lite256_explorer_correctness.json",
    "explorer-novelty": ROOT / "configs" / "production_trc2_v5lite256_explorer_novelty075.json",
    "explorer-novelty05-extend": ROOT / "configs" / "production_trc2_v5lite256_explorer_novelty05_extend.json",
    "actual-sft": ROOT / "configs" / "production_trc2_v5lite256_actual_sft.json",
    "actual-grpo": ROOT / "configs" / "production_trc2_v5lite256_actual_grpo.json",
    "eval-aime24": ROOT / "configs" / "production_trc2_v5lite256_eval_aime24.json",
}


def _run(args: list[str]) -> int:
    return subprocess.run([sys.executable, str(TMX_SUBMIT), *args], cwd=ROOT).returncode


def _stage_names(stage: str) -> list[str]:
    if stage == "all":
        return list(STAGE_SPECS)
    return [stage]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="One-command paper pipeline wrapper for production TMX specs."
    )
    parser.add_argument(
        "action",
        choices=("validate", "plan", "submit", "launch", "status", "wait-ready", "preflight", "doctor"),
        help="Operation to run.",
    )
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=(*STAGE_SPECS.keys(), "all"),
        help="Pipeline stage. Defaults to all for validate/doctor.",
    )
    parser.add_argument(
        "--backend",
        default="raw-tpu-tmx-jax",
        help="Backend passed to tmx_submit.py for plan/submit.",
    )
    parser.add_argument(
        "--allow-blocked",
        action="store_true",
        help="For plan only, return success even when a target backend still has blockers.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually launch submit commands. Omit for dry-run.",
    )
    parser.add_argument(
        "--wait-timeout-seconds",
        type=int,
        default=7200,
        help="How long launch/wait-ready should wait for TPU readiness.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=60,
        help="TPU readiness polling interval.",
    )
    args = parser.parse_args(argv)

    if args.action == "doctor":
        return _run(["doctor"])

    rc = 0
    for stage in _stage_names(args.stage):
        spec = STAGE_SPECS[stage]
        print(f"== {args.action}: {stage} ({spec.relative_to(ROOT)}) ==", flush=True)
        if args.action == "validate":
            step_rc = _run(["validate", "--spec", str(spec)])
        elif args.action == "plan":
            plan_args = ["plan", "--spec", str(spec), "--backend", args.backend]
            if args.allow_blocked:
                plan_args.append("--allow-blocked")
            step_rc = _run(plan_args)
        elif args.action == "preflight":
            step_rc = _run(["preflight", "--spec", str(spec), "--allow-not-ready"])
        elif args.action == "submit":
            submit_args = ["submit", "--spec", str(spec), "--backend", args.backend]
            submit_args.append("--execute" if args.execute else "--dry-run")
            step_rc = _run(submit_args)
        elif args.action == "launch":
            step_rc = _run([
                "submit",
                "--spec",
                str(spec),
                "--backend",
                args.backend,
                "--execute",
                "--wait-ready",
                "--wait-timeout-seconds",
                str(args.wait_timeout_seconds),
                "--poll-seconds",
                str(args.poll_seconds),
                "--quiet-plan",
            ])
        elif args.action == "status":
            step_rc = _run(["status", "--spec", str(spec), "--allow-not-ready"])
        else:
            step_rc = _run([
                "wait-ready",
                "--spec",
                str(spec),
                "--timeout-seconds",
                str(args.wait_timeout_seconds),
                "--poll-seconds",
                str(args.poll_seconds),
            ])
        rc = rc or step_rc
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
