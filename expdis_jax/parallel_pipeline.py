"""Run independent Explorer process groups concurrently, then filter/SFT/Main.

The manifest declares one fixed allocation and explicit, disjoint Explorer
slots. Each command launches ``python -m expdis_jax.pipeline`` (possibly through
a scheduler) and must wait for all its trainer ranks. Output and checkpoint
directories must be mounted at the same paths in every slot. This driver does
not provision devices or infer equal elapsed time from equal update counts.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from .config import config_from_dict, public_config_name, validate_contract
from .lineage import novelty_weight_for_round, split_budget
from .checkpointing import checkpoint_path


def _arguments(cfg):
    result = []
    for name, value in asdict(cfg).items():
        if isinstance(value, list):
            value = ",".join(map(str, value)) or "auto"
        elif isinstance(value, bool):
            value = str(value).lower()
        result.extend(["--" + public_config_name(name).replace("_", "-"), str(value)])
    return result


def build_plan(manifest):
    cfg = config_from_dict(manifest["config"])
    validate_contract(cfg)
    root = Path(cfg.output_dir)
    if not root.is_absolute() or not cfg.checkpoint_root.startswith("/"):
        raise ValueError("parallel execution requires absolute shared output_dir and checkpoint_root")
    slots = manifest["explorers"]
    main = manifest["main"]
    allocation = set(manifest["allocation"])
    if not allocation or len(slots) != cfg.scouts_per_round:
        raise ValueError("allocation must be nonempty and Explorer slots must equal scouts_per_round")
    occupied, endpoints = set(), set()
    for slot in [*slots, main]:
        if not slot.get("command") or not all(isinstance(s, str) for s in slot["command"]):
            raise ValueError("each slot needs an argv command that waits for all trainer ranks")
        devices = set(slot["devices"])
        urls = set(slot["serving_urls"])
        if not devices or not devices <= allocation or not urls:
            raise ValueError("every slot needs devices inside the fixed allocation and serving_urls")
        if slot is not main:
            if occupied & devices or endpoints & urls:
                raise ValueError("parallel Explorers must have disjoint devices and serving endpoints")
            occupied |= devices
            endpoints |= urls
    if set(main["devices"]) != allocation or occupied != allocation:
        raise ValueError("Explorer slots partition the same allocation used by Main")
    rounds = int(cfg.num_rounds)
    explorer_budgets = split_budget(cfg.grpo_max_steps, rounds)
    main_budgets = split_budget(cfg.actual_online_grpo_max_steps, rounds)
    parent = cfg.init_weights_checkpoint or cfg.central_parent_checkpoint
    plan = []
    for r in range(1, rounds + 1):
        output = root / f"round_{r:02d}"
        rcfg = replace(cfg, output_dir=str(output), pipeline_root_dir=str(root),
            num_rounds=1, pipeline_mode="two_model", round_novelty_schedule="",
            grpo_max_steps=explorer_budgets[r - 1],
            actual_online_grpo_max_steps=main_budgets[r - 1],
            lambda_novelty=novelty_weight_for_round(scalar_weight=cfg.lambda_novelty,
                schedule=cfg.round_novelty_schedule, round_index=r, rounds=rounds),
            seed=cfg.seed + (r - 1) * 100_003,
            init_weights_checkpoint=parent, central_parent_checkpoint=parent,
            resume_checkpoint="", init_checkpoint="")
        shares = split_budget(rcfg.grpo_max_steps, len(slots))
        if min([*shares, rcfg.actual_online_grpo_max_steps]) < 1:
            raise ValueError("each Explorer and Main stage must receive at least one update")
        common = {
            "EXPDIS_DATASET_SHARD_INDEX": str(r - 1),
            "EXPDIS_DATASET_SHARD_COUNT": str(rounds),
            "EXPDIS_DATASET_SHARD_POLICY": "uniform",
            "EXPDIS_DATASET_SHARD_SEED": str(cfg.seed),
            "EXPDIS_MULTIROUND_RESET_RND": "1",
            "EXPDIS_FORCE_FINAL_PHASE_CHECKPOINT": "1",
            "EXPDIS_TRAIN_VLLM_RELOAD_EVERY_STEPS": "1",
            "EXPDIS_PIPELINE_STOP_AFTER_SFT": "0",
            "EXPDIS_SKIP_FINAL_EVAL": "1",
        }
        jobs, pools = [], []
        for j, (slot, steps) in enumerate(zip(slots, shares), 1):
            ecfg = replace(rcfg, output_dir=str(output / "components" / f"explorer_{j:02d}"),
                scouts_per_round=1, scout_index=j, grpo_max_steps=steps,
                seed=rcfg.seed + (j - 1) * 10_003)
            pool = str(Path(ecfg.output_dir) / "explorer" / "explorer_trajectories.jsonl")
            pools.append(pool)
            jobs.append(_job(slot, ecfg, common | {
                "EXPDIS_PIPELINE_STAGE": "explorer_novelty",
                "EXPDIS_PIPELINE_STOP_AFTER_EXPLORER": "1",
                "EXPDIS_PIPELINE_SKIP_EXPLORER": "0",
                "EXPDIS_EXISTING_TRAJECTORIES": "",
            }, pool, steps * cfg.grpo_batch_size * cfg.grpo_grad_accum * cfg.grpo_num_generations))
        parent = checkpoint_path(replace(rcfg, output_dir=str(output / "actual" / "grpo")),
                                 f"step_{rcfg.actual_online_grpo_max_steps:06d}")
        main_job = _job(main, rcfg, common | {
            "EXPDIS_PIPELINE_STAGE": "explorer_novelty",
            "EXPDIS_PIPELINE_STOP_AFTER_EXPLORER": "0",
            "EXPDIS_PIPELINE_SKIP_EXPLORER": "1",
            "EXPDIS_EXISTING_TRAJECTORIES": ",".join(pools),
        }, parent, None)
        plan.append({"round": r, "explorers": jobs, "main": main_job})
    return plan


def _job(slot, cfg, controls, artifact, rows):
    # Explicit CLI values take precedence over environment variables in the
    # trainer; a manifest-wide URL must not override this slot's isolation.
    cfg = replace(cfg, vllm_server_urls=",".join(slot["serving_urls"]))
    return {
        "command": slot["command"] + _arguments(cfg),
        "env": {**slot.get("env", {}), **controls,
            "EXPDIS_VLLM_SERVER_URLS": ",".join(slot["serving_urls"]),
            "EXPDIS_VLLM_RELOAD_SERVER_URLS": ",".join(slot["serving_urls"]),
            "EXPDIS_HF_ARTIFACT_PREFIX": Path(cfg.output_dir).as_posix().strip("/"),
            "EXPDIS_WANDB_RUN_ID": "", "EXPDIS_WANDB_EXPLORER_RUN_ID": "",
            "EXPDIS_WANDB_DISTILL_RUN_ID": "", "EXPDIS_WANDB_ACTUAL_RUN_ID": "",
            "EXPDIS_WANDB_ACTUAL_GRPO_RUN_ID": ""},
        "output_dir": cfg.output_dir, "artifact": artifact, "expected_rows": rows,
        "steps": cfg.grpo_max_steps if rows is not None else cfg.actual_online_grpo_max_steps,
    }


def _run_jobs(jobs, timeout_seconds):
    running = []
    try:
        for job in jobs:
            out = Path(job["output_dir"])
            out.mkdir(parents=True, exist_ok=True)
            log = (out / "launcher.log").open("w")
            process = subprocess.Popen(job["command"], env=os.environ | job["env"],
                                       stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            running.append((process, log, job))
        deadline = time.monotonic() + timeout_seconds
        while any(p.poll() is None for p, _, _ in running):
            for process, _, job in running:
                if process.poll() not in (None, 0):
                    raise RuntimeError(f"stage failed: {job['output_dir']}/launcher.log")
            if time.monotonic() >= deadline:
                raise TimeoutError("stage deadline exceeded; inspect launcher.log")
            time.sleep(0.2)
        for process, _, job in running:
            if process.returncode:
                raise RuntimeError(f"stage failed: {job['output_dir']}/launcher.log")
            artifact = Path(job["artifact"])
            if not artifact.exists():
                raise RuntimeError(f"stage did not produce its required artifact: {artifact}")
            if job["expected_rows"] is not None:
                with artifact.open() as stream:
                    count = sum(1 for line in stream if line.strip())
                if count != job["expected_rows"]:
                    raise RuntimeError(f"selected rollout budget mismatch: {count} != {job['expected_rows']}")
    finally:
        for process, log, _ in running:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            log.close()


def run_plan(plan, timeout_seconds=86400):
    # No implicit reuse of an old pool or stage checkpoint on a failed rerun.
    for round_plan in plan:
        for job in [*round_plan["explorers"], round_plan["main"]]:
            if Path(job["artifact"]).exists():
                raise FileExistsError(f"choose a fresh run directory: {job['artifact']}")
    for round_plan in plan:
        start = time.monotonic()
        _run_jobs(round_plan["explorers"], timeout_seconds)
        explorer_seconds = time.monotonic() - start
        start = time.monotonic()
        _run_jobs([round_plan["main"]], timeout_seconds)
        receipt = {"round": round_plan["round"], "explorer_seconds": explorer_seconds,
                   "main_seconds": time.monotonic() - start,
                   "explorer_updates": sum(j["steps"] for j in round_plan["explorers"]),
                   "main_updates": round_plan["main"]["steps"],
                   "checkpoint": round_plan["main"]["artifact"]}
        (Path(round_plan["main"]["output_dir"]) / "parallel_receipt.json").write_text(json.dumps(receipt, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--run", action="store_true", help="execute; default prints the planned commands")
    parser.add_argument("--timeout-seconds", type=float, default=86400)
    args = parser.parse_args()
    plan = build_plan(json.loads(args.manifest.read_text()))
    if args.run:
        run_plan(plan, args.timeout_seconds)
    else:
        print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
