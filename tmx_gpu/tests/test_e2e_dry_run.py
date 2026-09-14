"""End-to-end CPU dry run of the full ExpDis pipeline against a stub vLLM server.

Drives the REAL pipeline_gpu flow (scout GRPO -> harvest -> filter_pool ->
central SFT -> central GRPO, 2 rounds) with a tiny random Llama and the
stub_server OpenAI-completions emulator, then asserts the contract surface:
stage order, round handoff, RND reset, error-row exclusion, metric parity key
names, monotonic global steps, and crash-resume via pipeline_state.json.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

PARITY_KEYS = {
    "train/reward_total_mean", "train/reward_correctness_mean",
    "train/reward_novelty_mean", "train/clip_fraction", "train/grpo_loss",
    "train/policy_kl", "quality/valid_answer_rate", "quality/clipped_rate",
    "quality/nontermination_rate", "quality/mean_completion_tokens",
    "sft/loss", "sft/accepted_trajectories", "sft/acceptance_rate",
    "eval/AIME24_avg_at_4",
    "funnel/raw", "funnel/correct", "funnel/clean_correct", "funnel/accepted",
}
TRAJECTORY_FIELDS = {
    "problem_id", "prompt_text", "completion_text", "ground_truth",
    "is_correct", "blended_reward", "correctness_reward", "novelty_reward",
    "completion_token_length", "clipped", "terminated", "valid_answer",
    "finish_reason", "step",
}


def pipeline_argv(out_dir: str, model_dir: str, endpoint: str, train_jsonl: str,
                  probe_jsonl: str, novelty_layers: str, *, wandb: bool,
                  probe: bool) -> list:
    argv = [
        "--model", model_dir,
        "--output-dir", out_dir,
        "--run-name", "e2e-dry",
        "--rounds", "2",
        "--explorer-steps", "2",
        "--main-steps", "1",
        "--lambda-schedule", "0.75,0.5",
        "--policy", "quality_pool",
        "--min-accepted", "10",
        "--harvest-max-passes", "4",
        "--harvest-prompts-per-pass", "2",
        "--prompts-per-step", "2",
        "--num-generations", "4",
        "--dataset", train_jsonl,
        "--max-train-examples", "6",
        "--sync-dir", os.path.join(out_dir, "sync"),
        "--endpoints", endpoint,
        # The stub is attach-only and cannot reload weights. These tests cover
        # orchestration/resume rather than c8's managed-pool freshness guard,
        # so request the legacy serving contract explicitly.
        "--rl-validity-mode", "legacy",
        # Attach-mode + probes: declare the (stub) servers large enough for
        # the frozen probe protocol (2048 + PROBE_MAX_TOKENS) — the pipeline
        # fails fast otherwise.
        "--vllm-max-model-len", "36864",
        "--novelty-layers", novelty_layers,
        "--probe-every", "1" if probe else "0",
        "--probe-dataset", probe_jsonl,
        "--ckpt-artifact-every", "0",
        "--save-every", "1",
        "--sft-epochs", "1",
        "--sft-grad-accum", "4",
        "--device", "cpu",
        "--param-dtype", "float32",
        "--seed", "0",
        "--resume",
    ]
    if wandb:
        argv.append("--wandb")
    return argv


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def test_full_pipeline_dry_run(tmp_path, monkeypatch, stub_pool, tiny_model_dir,
                               train_jsonl, probe_jsonl, novelty_layers):
    endpoint, stub_state = stub_pool
    out_dir = str(tmp_path / "run")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path / "wandb"))
    monkeypatch.setattr(sys, "argv", ["pipeline_gpu.py"] + pipeline_argv(
        out_dir, tiny_model_dir, endpoint, train_jsonl, probe_jsonl,
        novelty_layers, wandb=True, probe=True))

    from tmx_gpu import pipeline_gpu

    pipeline_gpu.main()

    # ---- pipeline completed: 2 rounds, correct handoff --------------------
    summary = json.load(open(os.path.join(out_dir, "run_summary.json")))
    assert len(summary["rounds"]) == 2
    r1, r2 = summary["rounds"]
    assert r2["scout_init"] == r1["central_final"], \
        "round-2 scout must initialize from round-1 central"
    assert summary["final_model"] == r2["central_final"]
    # The test's tiny geometry (2 prompts/step vs contract 4) must be
    # self-describing as a tier deviation.
    assert summary["config_tier"] == "gpu-max:harvest_max_passes,prompts_per_step"

    # Phase-final directories are deployment/model-handoff artifacts, not
    # crash-resume checkpoints. Optimizer moments must stay in step_* only.
    from tmx_gpu import grpo_gpu

    for round_name in ("round_01", "round_02"):
        for phase in ("scout", "central"):
            final_dir = os.path.join(out_dir, round_name, phase, "final")
            final_state = json.load(open(os.path.join(final_dir, "trainer_state.json")))
            assert final_state["checkpoint_kind"] == "deployable_model"
            assert final_state["optimizer_state_saved"] is False
            assert not os.path.exists(
                os.path.join(final_dir, grpo_gpu.OPTIMIZER_STATE_FILENAME))

    # ---- stage order (pipeline_state.json completion timestamps) ----------
    state = json.load(open(os.path.join(out_dir, "pipeline_state.json")))
    expected_order = [f"r{r:02d}/{s}" for r in (1, 2)
                      for s in ("scout", "harvest", "filter", "sft", "central")]
    assert list(state["stages"]) == expected_order
    times = [state["stages"][k]["completed_at"] for k in expected_order]
    assert times == sorted(times)

    # ---- RND reset between rounds (fresh targets, per-round seed) ---------
    import torch

    layer = int(novelty_layers)
    rnd1 = torch.load(os.path.join(out_dir, "round_01", "scout", "final", "rnd_state.pt"),
                      map_location="cpu", weights_only=False)
    rnd2 = torch.load(os.path.join(out_dir, "round_02", "scout", "final", "rnd_state.pt"),
                      map_location="cpu", weights_only=False)
    t1 = rnd1["targets"][str(layer)]["fc1.weight"]
    t2 = rnd2["targets"][str(layer)]["fc1.weight"]
    assert not torch.equal(t1, t2), "RND must be rebuilt each round (new seed)"

    # ---- trajectory + accepted files: exist, complete rows, no error rows -
    for r in ("round_01", "round_02"):
        traj = load_jsonl(os.path.join(out_dir, r, "scout", "explorer_trajectories.jsonl"))
        assert traj, f"{r}: empty scout trajectories"
        assert TRAJECTORY_FIELDS <= set(traj[0])
        accepted = load_jsonl(os.path.join(out_dir, r, "accepted.jsonl"))
        assert len(accepted) >= 10
        assert all(row["is_correct"] for row in accepted)
        rollouts = load_jsonl(os.path.join(out_dir, r, "scout", "rollouts_all.jsonl"))
        for row in traj + accepted + rollouts:
            assert row.get("finish_reason") != "error"
            assert "novelty_text" not in row and "_novelty_text" not in row

    # ---- metric parity keys + columns --------------------------------------
    metrics = load_jsonl(os.path.join(out_dir, "metrics.jsonl"))
    seen_keys = {k for row in metrics for k in row}
    missing = PARITY_KEYS - seen_keys
    assert not missing, f"missing parity metric keys: {sorted(missing)}"
    phases = {row["phase"] for row in metrics}
    assert {"scout_grpo", "central_grpo", "central_sft", "filter_pool"} <= phases
    for row in metrics:
        assert {"global_step", "phase", "round", "lambda", "num_scouts"} <= set(row)

    # ---- REINFORCE contract: clip inert, KL zero ---------------------------
    grpo_rows = [r for r in metrics if "train/grpo_loss" in r]
    assert grpo_rows
    assert all(r["train/clip_fraction"] == 0.0 for r in grpo_rows)
    assert all(r["train/policy_kl"] == 0.0 for r in grpo_rows)
    # lambda column follows the schedule (0.75 round 1, 0.5 round 2; central 0).
    assert {r["lambda"] for r in grpo_rows if r["phase"] == "scout_grpo"} == {0.75, 0.5}
    assert {r["lambda"] for r in grpo_rows if r["phase"] == "central_grpo"} == {0.0}

    # ---- global steps monotonic across rounds/stages -----------------------
    # rounds=2 x (scout 2 + central 1): scout r1 {1,2}, central r1 {3},
    # scout r2 {4,5}, central r2 {6}.
    by_stage = {}
    for r in grpo_rows:
        by_stage.setdefault((r["round"], r["phase"]), set()).add(r["global_step"])
    assert by_stage[(1, "scout_grpo")] == {1, 2}
    assert by_stage[(1, "central_grpo")] == {3}
    assert by_stage[(2, "scout_grpo")] == {4, 5}
    assert by_stage[(2, "central_grpo")] == {6}

    # ---- probe fired on every GRPO step ------------------------------------
    probe_rows = [r for r in metrics if "eval/AIME24_avg_at_4" in r]
    assert {r["global_step"] for r in probe_rows} == {1, 2, 3, 4, 5, 6}
    assert all(0.0 <= r["eval/AIME24_avg_at_4"] <= 1.0 for r in probe_rows)
    assert all("eval/AIME24_avg" not in r for r in probe_rows)

    # ---- stub retry path was exercised (one injected 500, then success) ---
    assert stub_state.failed_once, "stub should have injected a 500"


def test_resume_after_crash(tmp_path, stub_pool, tiny_model_dir, train_jsonl,
                            probe_jsonl, novelty_layers):
    """Kill the pipeline right after round 1 completes; rerun with --resume;
    round 1 must be skipped (outputs untouched) and the run must finish."""
    endpoint, _state = stub_pool
    out_dir = str(tmp_path / "run")
    argv = pipeline_argv(out_dir, tiny_model_dir, endpoint, train_jsonl,
                         probe_jsonl, novelty_layers, wandb=False, probe=False)
    cmd = [sys.executable, str(REPO_ROOT / "tmx_gpu" / "pipeline_gpu.py")] + argv
    env = {**os.environ, "WANDB_MODE": "offline", "PYTHONUNBUFFERED": "1"}

    state_path = os.path.join(out_dir, "pipeline_state.json")
    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            start_new_session=True)
    try:
        deadline = time.time() + 540
        while time.time() < deadline:
            if proc.poll() is not None:
                out = proc.stdout.read().decode(errors="replace")
                pytest.fail(f"pipeline exited before round 1 completed:\n{out[-4000:]}")
            if os.path.exists(state_path):
                stages = json.load(open(state_path)).get("stages", {})
                if "r01/central" in stages:
                    break
            time.sleep(0.2)
        else:
            pytest.fail("timed out waiting for round 1 to complete")
    finally:
        if proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()

    marker = os.path.join(out_dir, "round_01", "central", "final", "trainer_state.json")
    assert os.path.exists(marker)
    mtime_before = os.path.getmtime(marker)

    rerun = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=540)
    out = rerun.stdout.decode(errors="replace")
    assert rerun.returncode == 0, f"resume run failed:\n{out[-4000:]}"
    assert "scout already complete" in out, "resume did not skip round-1 scout"

    # Round 1 outputs untouched; both rounds present in the final summary.
    assert os.path.getmtime(marker) == mtime_before, "round 1 was re-run"
    summary = json.load(open(os.path.join(out_dir, "run_summary.json")))
    assert len(summary["rounds"]) == 2
    assert summary["rounds"][1]["scout_init"] == summary["rounds"][0]["central_final"]

    # Round-2 scout trajectories contain each step exactly once (trim +
    # resume must not leave duplicate step rows after the crash).
    traj = load_jsonl(os.path.join(out_dir, "round_02", "scout",
                                   "explorer_trajectories.jsonl"))
    rows_per_step = {}
    for row in traj:
        rows_per_step[row["step"]] = rows_per_step.get(row["step"], 0) + 1
    assert set(rows_per_step) == {1, 2}
    assert all(n == 8 for n in rows_per_step.values()), rows_per_step  # 2 prompts x 4 gens
