#!/usr/bin/env python3
"""One-round (and multi-round) ExpDis pipeline driver on GPU:

  scout GRPO -> harvest trajectories -> filter_pool -> Central SFT ->
  central GRPO (correctness-only) -> final model dir

Mirrors tmx_jax/pipeline.py (main + multi_round_main) with the GPU stack:
one training GPU (grpo_gpu / sft_gpu) + a restart-reload vLLM pool
(vllm_pool) on the remaining GPUs.

Stage flow per round r (round dirs under --output-dir/round_XX):
  1. Scout GRPO (--scout-steps, lambda from --lambda-schedule / --lambda-novelty);
     round r scout initializes from round r-1 central output (round 1: --model).
     RND is reset each round (run_grpo builds a fresh MultilayerRND per phase,
     matching multi_round_reset_rnd=True).
  2. Harvest: trajectories accumulate in explorer_trajectories.jsonl during
     training; if fewer than --min-accepted rows survive the pooling policy's
     gates, extra harvest passes generate more groups at the FINAL scout
     policy via vLLM at T=1.0 (run_grpo leaves the pool serving final weights)
     and append to harvest_trajectories.jsonl.
  3. filter_pool.pool_trajectories (--policy coverage_pool_c8|quality_pool|
     naive_pool|best_scout,
     cap --max-accepted) -> accepted.jsonl.
  4. Central SFT (sft_gpu) initialized from the BASE --model — faithful to
     tmx_jax.distill.run_sft, which always distills into cfg.model_name, even
     in multi-round (only the Explorer inherits the previous Central).  A
     single-round --skip-scout continuation may instead supply --sft-model:
     the exact evidence-bound frozen c7 library is copied (never re-filtered
     under c8), while the original trajectory artifacts remain provenance-
     bound; the health-gate-selected SFT checkpoint is recorded and SFT is
     not retrained. A separate --frozen-sft-library-evidence-index route copies
     the same library but actually retrains ga8 SFT and stops for gating.
  5. Central GRPO from the SFT output: lambda forced 0, lr 1e-6,
     --central-steps.

--skip-scout with --trajectories runs stages 2-5 from pooled multi-scout
input (round 1 only); harvest passes then sample at the currently served
policy (the base model) — prefer providing enough trajectories.

MULTI-ROUND DATA SHARDS: when --rounds > 1 the pipeline sets
TMX_DATASET_SHARD_INDEX=round-1 / _COUNT=rounds / _POLICY=uniform /
_SEED=--seed for the duration of each round, exactly as
tmx_jax/pipeline.py multi_round_main does. tmx_jax.data._apply_round_shard
then deterministically shuffles by the shard seed and takes the round's
contiguous slice, so every round trains (and harvests) on a disjoint 1/N
shard. The env vars cover both the harvest example pool (reloaded per round)
and the stage trainers' own dataset loads.

CRASH RECOVERY: pipeline_state.json (in --output-dir, written atomically)
records each completed stage's outputs. With --resume, completed
rounds/stages are skipped, an interrupted GRPO stage restarts from its newest
step_* checkpoint (--resume-from is passed through, which also trims stale
JSONL/metric rows past the resume step), and a crashed harvest's partial file
is rotated aside. A resumed harvest first reloads the pool to the RECORDED
scout-final checkpoint (hard error if it cannot — e.g. attach-mode, or the
checkpoint is gone), and a harvest that ends below --min-accepted is left
UNMARKED so the next --resume retries top-up instead of deadlocking at the
filter-stage floor.

METRICS: one shared metrics.jsonl (in --output-dir) receives every stage's
parity metrics plus the pipeline's per-round trajectory-funnel counters
(funnel/raw, funnel/correct, funnel/clean_correct, funnel/accepted —
lambda-annealing funnel-table semantics) and sft/accepted_trajectories /
sft/acceptance_rate (= accepted/raw). Stage global steps are offset by the
sum of prior GRPO stages' steps (scout r + central r accumulate), so
multi-round W&B curves remain monotonic across round boundaries.

DURABILITY: after every stage the pipeline pushes a versioned
"{run-name}-results" W&B artifact (metrics.jsonl, pipeline_state.json,
run_summary.json, accepted/trajectory JSONLs) — the Lepton workspace has no durable
disk, so mid-run state must reach W&B before a pod loss, not only at job end.

CONFIG TIERS: the pipeline passes the GPU-MAX flags straight through to the
stage trainers (--completion-budget, --prompts-per-step, --ppo-mode,
--updates-per-rollout, --optimizer for GRPO stages, --sft-optimizer for SFT)
and sizes the shared vLLM pool's max-model-len from the completion budget.
All defaults are the CONTRACT tier (including --max-accepted 500, the TPU
production max_accepted_trajectories; raising it — e.g. 2000 — is a GPU-MAX
change and flips the logged config_tier). The pipeline W&B run logs
config_tier alongside every stage run's own config_tier.

DEVIATIONS (vs tmx_jax/pipeline.py):
- collect_accepted is replaced by filter_pool: --policy naive_pool IS the TPU
  production filter (correct-only + quality_first sort + cap); quality_pool
  (default) adds the paper-planned QualityPool hard gates (see filter_pool
  docstring).
- Harvest top-up passes are new (TPU raised if min_accepted was not reached);
  they generate at the final scout policy with the training sampling contract
  (T=1.0 / top_p 0.95 / top_k 20 / 16384 max tokens). best_scout skips
  top-up: harvest rows come from the served policy, not from any scout, so
  they are excluded from the best_scout competition and cannot raise it.
- Multi-round budgets: each round runs the FULL --scout-steps/--central-steps
  (interpreted as per-round budgets). The TPU default round_budget_mode=split
  divided the totals across rounds; pass scout-steps = total/rounds for
  TPU-split parity.
- W&B: one pipeline run logs round/stage transitions + funnel stats; each
  stage additionally opens its own run (reinit="create_new") in the same group.
"""

from __future__ import annotations

import argparse
import copy
import contextlib
import glob
import json
import math
import numbers
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tmx_gpu import (
    c9_evidence,  # noqa: E402
    c10_evidence,  # noqa: E402
    common,  # noqa: E402
    fresh_c8_sft,  # noqa: E402
    grpo_gpu,  # noqa: E402
    mrme_contract,  # noqa: E402
    sft_gpu,  # noqa: E402
)  # noqa: E402
from tmx_gpu import filter_pool as filter_pool_mod  # noqa: E402
from tmx_gpu import select_sft_checkpoint as sft_selector  # noqa: E402
from tmx_gpu import sft_gate_gpu  # noqa: E402
from tmx_gpu.vllm_pool import VLLMPool, completion_failed  # noqa: E402
from tmx_jax.rewarding import (  # noqa: E402
    analyze_completion_text,
    is_correct_completion,
)

# Pipeline-level GPU-MAX tier detection (see grpo_gpu.compute_config_tier).
# max_accepted is included: the contract cap is the TPU production 500;
# raising it (e.g. 2000) must flip the logged config_tier.
PIPELINE_GPU_MAX_FLAG_DEFAULTS = {
    "completion_budget": 0,
    "prompts_per_step": 4,
    "ppo_mode": False,
    "updates_per_rollout": 1,
    "optimizer": "adamw",
    "sft_optimizer": "adamw",
    "max_accepted": 500,
    "mask_truncated": False,
    "truncation_baseline_mode": "all_rows",
    "harvest_max_passes": 0,
    "scout_lr": None,
    "central_lr": None,
    "expdis_stabilization_mode": grpo_gpu.C8_RND_LIFECYCLE,
    "novelty_normalization": "raw",
}

RESULT_ARTIFACT_PATTERNS = (
    "metrics.jsonl", "pipeline_state.json", "run_summary.json",
    "c10_exclusive_lease_binding.json",
    "round_*/accepted.jsonl", "round_*/harvest_trajectories.jsonl",
    "round_*/*/explorer_trajectories.jsonl",
    "round_*/scout/fresh_c8_scout_completion_evidence.json",
    "round_*/scout/fresh_c8_code_artifact_binding.json",
    "round_*/scout/final/trainer_state.json",
    "round_*/scout/final_artifact_binding.json",
    "round_*/sft/actual_sft_final/sft_provenance.json",
    "round_*/sft_model_artifact.json",
    "round_*/sft_selection_provenance.json",
    "round_*/external_central_lineage.json",
    "round_*/fresh_c8_sft_training_evidence.json",
    "round_*/sft/sft_completed_run.json",
    "round_*/scout/c9_scout_completion_evidence.json",
    "round_*/scout/c10_training_terminal_outcome.json",
    "round_*/scout/c10_training_transport_invalid.json",
    "round_*/scout/c10_dynamic_sampling_failure_journal.json",
    "round_*/scout/c10_failed_step_candidates.jsonl",
    "round_*/scout/c10_failed_selected_batch.jsonl",
    "round_*/scout/c10_failed_selected_batch_manifest.json",
    "round_*/c9_sft_training_evidence.json",
    "round_*/c9_central_lineage.json",
    "round_*/c9_multi_explorer_central_lineage.json",
    "round_*/c9_behavior_gates/*/heldout_selection.jsonl",
    "round_*/c9_behavior_gates/*/rollouts.jsonl",
    "round_*/c9_behavior_gates/*/aggregate_metrics.json",
    "round_*/c9_behavior_gates/*/evidence.json",
    "round_*/c10_behavior_gates/*/heldout_selection.jsonl",
    "round_*/c10_behavior_gates/*/rollouts.jsonl",
    "round_*/c10_behavior_gates/*/aggregate_metrics.json",
    "round_*/c10_behavior_gates/*/evidence.json",
    "round_*/c10_endpoint_replay.json",
    "mrme_contract.json",
    "mrme_shard_partition.json",
    "mrme_completed_campaign.json",
    "mrme_portable_campaign.json",
    "round_*/dataset_shard_manifest.json",
    "round_*/mrme_scout_sources.json",
    "round_*/mrme_trajectory_source_manifest.json",
    "round_*/scouts/*/mrme_scout_source.json",
    "round_*/scouts/*/explorer_trajectories.jsonl",
    "round_*/scouts/*/final/trainer_state.json",
    "round_*/scouts/*/final_artifact_binding.json",
    "round_*/central/final_artifact_binding.json",
    "round_*/scouts/*/c9_behavior_gates/*/heldout_selection.jsonl",
    "round_*/scouts/*/c9_behavior_gates/*/rollouts.jsonl",
    "round_*/scouts/*/c9_behavior_gates/*/aggregate_metrics.json",
    "round_*/scouts/*/c9_behavior_gates/*/evidence.json",
)

SFT_MODEL_ARTIFACT_BINDING_VERSION = "pipeline_sft_model_artifact_v1"
SFT_MODEL_ARTIFACT_SUBDIR = "model"


# ---------------------------------------------------------------- helpers ----

def _token_gate_cap(args: argparse.Namespace) -> int:
    """QualityPool upper token gate: contract 16000; GPU-MAX budget - 384."""
    budget = int(args.completion_budget) or 0
    if budget > 0:
        return budget - 384
    return filter_pool_mod.MAX_COMPLETION_TOKENS


def _policy_survivors(paths: List[str], policy: str, max_tokens: int) -> int:
    """Rows that would survive the pooling policy (harvest stopping rule).

    Must mirror filter_pool.pool_trajectories per policy: naive_pool counts
    correct rows, best_scout counts the BEST single source file's gated rows
    (summing across files would stop the harvest before the best file alone
    reaches --min-accepted), quality_pool counts gated rows across files, and
    coverage_pool_c8 counts distinct gated problems.
    """
    rows = filter_pool_mod.load_trajectories(paths)
    if policy == "naive_pool":
        return sum(1 for r in rows if bool(r.get("is_correct", False)))
    if policy == "best_scout":
        per_source: Dict[str, int] = {}
        for r in rows:
            if filter_pool_mod.passes_quality_gates(r, max_tokens=max_tokens):
                per_source[r["source_file"]] = per_source.get(r["source_file"], 0) + 1
        return max(per_source.values(), default=0)
    if policy in (
        filter_pool_mod.C8_POLICY,
        filter_pool_mod.ORIGINAL_BLENDED_POLICY,
    ):
        clean = [
            r for r in rows
            if bool(r.get("is_correct", False))
            and filter_pool_mod.passes_c8_contract_gates(r)
        ]
        return len({filter_pool_mod.c8_problem_key(r) for r in clean})
    return sum(1 for r in rows
               if filter_pool_mod.passes_quality_gates(r, max_tokens=max_tokens))


def _sft_checkpoint_metadata(model_dir: str) -> Dict[str, Any]:
    """Read the local SFT checkpoint's own training metadata when present.

    Exact optimizer-step snapshots carry ``sft_checkpoint.json`` while the
    ordinary final SFT directory carries ``sft_provenance.json``.  External
    Hugging Face checkpoints without either file remain usable, but the
    absence is recorded rather than silently inventing training provenance.
    """
    for name in ("sft_checkpoint.json", "sft_provenance.json"):
        path = os.path.join(model_dir, name)
        if os.path.isfile(path):
            with open(path) as f:
                metadata = json.load(f)
            return {"metadata_file": name, "metadata": metadata}
    return {"metadata_file": None, "metadata": {}}


def _load_json_object(path: str, field: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise ValueError(f"{field} is not a file: {path!r}")
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{field} must contain one JSON object")
    return value


EXTERNAL_EVIDENCE_INDEX_VERSION = "external_sft_evidence_bundle_v2"
FROZEN_SFT_LIBRARY_EVIDENCE_INDEX_VERSION = "frozen_sft_library_evidence_v1"
FROZEN_SFT_LIBRARY_VARIANT = "frozen_c7_library_ga8_sft_retrain"
C7_REUSE_ACCEPTED_ARTIFACT_NAME = (
    "gpu-jrl-c8-c7reuse-sft-ga8-checkpoints-20260713-results:v1")
C7_REUSE_ACCEPTED_ARTIFACT_DIGEST = "64b9241d1ba4509395ff6f4473c794ee"
C7_REUSE_ACCEPTED_DIGEST = {
    "version": common.SFT_DATASET_DIGEST_VERSION,
    "row_count": 500,
    "sha256": "337ab6ebd154c9ef6a3616cc72aa7a4c8ae15e89a732ad929464218945f8712f",
}


def _external_evidence_profile(args: argparse.Namespace) -> str:
    index = _load_json_object(args.sft_evidence_index, "--sft-evidence-index")
    version = index.get("version")
    if version == EXTERNAL_EVIDENCE_INDEX_VERSION:
        return sft_selector.DIAGNOSTIC_PROFILE
    if version == fresh_c8_sft.EXTERNAL_EVIDENCE_INDEX_VERSION:
        return fresh_c8_sft.PROFILE
    raise ValueError("external SFT evidence index version mismatch")


def _fresh_c8_scout_evidence_profile(
    args: argparse.Namespace,
) -> Optional[str]:
    """Return the one explicitly requested fresh-c8 Scout evidence profile."""
    canonical = bool(getattr(args, "fresh_c8_sft_gate", False))
    component = bool(getattr(args, "fresh_c8_multiscout_component", False))
    if canonical and component:
        raise ValueError(
            "fresh-c8 SingleScout SFT and MultiScout-component profiles are "
            "mutually exclusive")
    if canonical:
        return fresh_c8_sft.PROFILE
    if component:
        return fresh_c8_sft.MULTISCOUT_COMPONENT_PROFILE
    return None


def _validate_fresh_c8_scout_completion_for_profile(
    value: Any, profile: str,
) -> Dict[str, Any]:
    if profile == fresh_c8_sft.PROFILE:
        return fresh_c8_sft.validate_scout_completion_evidence(value)
    if profile == fresh_c8_sft.MULTISCOUT_COMPONENT_PROFILE:
        return fresh_c8_sft.validate_multiscout_component_completion_evidence(
            value)
    raise ValueError(f"unknown fresh-c8 Scout evidence profile {profile!r}")


def _validated_executing_code_source_manifest(
    args: argparse.Namespace, *, profile_label: str,
) -> Dict[str, Any]:
    """Prove a manifest describes the immutable tree executing this module."""
    raw = _load_json_object(args.code_source_manifest, "--code-source-manifest")
    validated = common.validate_artifact_download_manifest(raw)
    common.require_immutable_wandb_artifact_ref(
        validated["source"]["artifact_ref"], field="fresh-c8 CODE_ARTIFACT")
    root = os.path.realpath(str(validated["download_root"]))
    executing_pipeline = os.path.realpath(__file__)
    try:
        contained = os.path.commonpath([root, executing_pipeline]) == root
    except ValueError:
        contained = False
    if not contained:
        raise ValueError(
            f"{profile_label} code manifest does not describe the actual "
            "executing immutable code root")
    relative = os.path.relpath(executing_pipeline, root).replace(os.sep, "/")
    members = {
        str(item["relative_path"]): item for item in validated["files"]
    }
    if relative not in members:
        raise ValueError(
            f"{profile_label} code manifest does not contain the actual "
            "executing pipeline module")
    return _stable_artifact_identity(validated)


def _validated_code_source_manifest(args: argparse.Namespace) -> Dict[str, Any]:
    return _validated_executing_code_source_manifest(
        args, profile_label="fresh-c8 canonical SingleScout")


def _validated_multiscout_component_code_source_manifest(
    args: argparse.Namespace,
) -> Dict[str, Any]:
    return _validated_executing_code_source_manifest(
        args, profile_label="fresh-c8 MultiScout component")


def _validated_code_source_manifest_for_profile(
    args: argparse.Namespace, profile: str,
) -> Dict[str, Any]:
    if profile == fresh_c8_sft.PROFILE:
        # Preserve the frozen canonical seed-0 validation path exactly.
        return _validated_code_source_manifest(args)
    if profile == fresh_c8_sft.MULTISCOUT_COMPONENT_PROFILE:
        return _validated_multiscout_component_code_source_manifest(args)
    raise ValueError(f"unknown fresh-c8 Scout evidence profile {profile!r}")


def _prepare_fresh_c8_scout_code_binding(
    args: argparse.Namespace, gargs: argparse.Namespace,
) -> Dict[str, Any]:
    """Bind immutable code before Scout update 1 and validate every resume.

    A local sidecar establishes the binding before ``run_grpo`` is entered;
    every durable ``step_*`` checkpoint independently embeds the same object.
    Cross-pod recovery can therefore recreate the sidecar from a restored
    checkpoint, but can never resume an old/unbound or differently bound one.
    """
    profile = _fresh_c8_scout_evidence_profile(args)
    if profile is None:
        raise RuntimeError("fresh-c8 Scout code binding lacks an evidence profile")
    lr = float(gargs.lr) if gargs.lr is not None else 5e-6
    steps = int(gargs.steps) if gargs.steps is not None else 200
    grpo_gpu.apply_completion_budget(gargs)
    grpo_gpu.enforce_probe_serving_len(gargs, pool=None)
    runtime_contract = grpo_gpu.build_runtime_contract(
        gargs, resolved_lr=lr, resolved_steps=steps)
    if profile == fresh_c8_sft.PROFILE:
        fresh_c8_sft.validate_scout_runtime_contract(runtime_contract)
    else:
        fresh_c8_sft.validate_multiscout_component_runtime_contract(
            runtime_contract, expected_seed=int(args.seed))
    current = _validated_code_source_manifest_for_profile(args, profile)
    if os.path.realpath(str(getattr(gargs, "code_source_manifest", ""))) != \
            os.path.realpath(args.code_source_manifest):
        raise RuntimeError("fresh-c8 Scout GRPO lacks the current code manifest")
    os.makedirs(gargs.output_dir, exist_ok=True)
    sidecar = os.path.join(
        gargs.output_dir, "fresh_c8_code_artifact_binding.json")
    if os.path.exists(sidecar):
        recorded_sidecar = _load_json_object(
            sidecar, "fresh-c8 Scout code binding sidecar")
        if recorded_sidecar != current:
            raise RuntimeError(
                "fresh-c8 Scout code artifact binding mismatch before run_grpo")

    checkpoints = sorted(glob.glob(os.path.join(
        gargs.output_dir, "step_[0-9]*")))
    for checkpoint in checkpoints:
        state_path = os.path.join(checkpoint, "trainer_state.json")
        if not os.path.isfile(state_path):
            raise RuntimeError(
                "fresh-c8 Scout checkpoint exists but trainer_state binding is missing")
        recorded = _load_json_object(
            state_path, "fresh-c8 Scout resume trainer_state")
        if recorded.get("code_artifact_binding") is None:
            raise RuntimeError(
                "fresh-c8 Scout checkpoint exists but code artifact binding is missing")
        if recorded["code_artifact_binding"] != current:
            raise RuntimeError(
                "fresh-c8 Scout code artifact binding mismatch before run_grpo")
    if checkpoints and not str(getattr(gargs, "resume_from", "")):
        raise RuntimeError(
            "fresh-c8 Scout checkpoints exist but no exact resume checkpoint was selected")

    # Atomic and deliberately written only after all old state agrees.  This
    # happens before run_grpo can perform the first learner update.
    common.atomic_write_json(sidecar, current)
    return current


def _read_jsonl_objects(path: str, field: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{field}:{line_number} is not a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{field} contains no rows")
    return rows


def _fresh_c8_trajectory_completion_facts(
    path: str, *, profile: str = fresh_c8_sft.PROFILE,
    component_seed: Optional[int] = None,
) -> Dict[str, Any]:
    rows = _read_jsonl_objects(path, "fresh-c8 Scout trajectories")
    if profile == fresh_c8_sft.MULTISCOUT_COMPONENT_PROFILE:
        if type(component_seed) is not int or component_seed not in \
                fresh_c8_sft.MULTISCOUT_COMPONENT_SEEDS:
            raise ValueError(
                "fresh-c8 component trajectory seed must be an exact integer "
                "1 or 2")
    elif profile != fresh_c8_sft.PROFILE:
        raise ValueError(f"unknown fresh-c8 Scout evidence profile {profile!r}")
    per_step: Dict[int, int] = {}
    for row_number, row in enumerate(rows, 1):
        step_value = row.get("step", -1)
        if type(step_value) is not int:
            raise ValueError(
                "fresh-c8 trajectory step must be an exact integer at row "
                f"{row_number}")
        step = step_value
        if profile == fresh_c8_sft.MULTISCOUT_COMPONENT_PROFILE:
            row_seed = row.get("component_seed")
            if row.get("fresh_c8_evidence_profile") != profile or \
                    type(row_seed) is not int or row_seed != component_seed:
                raise ValueError(
                    "fresh-c8 component trajectory row provenance mismatch "
                    f"at row {row_number}")
        else:
            row_profile = row.get("fresh_c8_evidence_profile")
            if "component_seed" in row or row_profile == \
                    fresh_c8_sft.MULTISCOUT_COMPONENT_PROFILE:
                raise ValueError(
                    "canonical fresh-c8 trajectories cannot carry component "
                    f"provenance at row {row_number}")
            if row_profile not in (None, fresh_c8_sft.PROFILE):
                raise ValueError(
                    "canonical fresh-c8 trajectory profile mismatch at row "
                    f"{row_number}")
            if row_profile == fresh_c8_sft.PROFILE and (
                    type(row.get("seed")) is not int
                    or row.get("seed") != 0):
                raise ValueError(
                    "canonical fresh-c8 trajectory seed mismatch at row "
                    f"{row_number}")
        per_step[step] = per_step.get(step, 0) + 1
    if set(per_step) != set(range(1, 201)) or \
            any(count != 64 for count in per_step.values()):
        raise ValueError(
            "fresh-c8 trajectories must contain exactly 64 rows for every "
            "step 1..200")
    return {
        "row_count": len(rows),
        "steps": len(per_step),
        "rows_per_step": 64,
        "file_sha256": common.file_sha256(path),
        "semantic_digest": fresh_c8_sft.trajectory_semantic_digest(rows),
    }


def _record_fresh_c8_scout_completion(
    args: argparse.Namespace, *, scout_final: str, trajectory_path: str,
    profile: str = fresh_c8_sft.PROFILE,
) -> tuple[Dict[str, Any], str]:
    """Persist provenance at Scout completion, before filtering or SFT."""
    scout_final = os.path.realpath(scout_final)
    state_path = os.path.join(scout_final, "trainer_state.json")
    scout_state = _load_json_object(state_path, "fresh-c8 Scout trainer_state")
    model_manifest = scout_state.get("checkpoint_file_manifest")
    if not isinstance(model_manifest, dict):
        raise ValueError("fresh-c8 Scout final lacks model-file manifest")
    common.verify_directory_file_manifest(
        scout_final, model_manifest, exclude_relative_paths=("trainer_state.json",))
    code_artifact = _validated_code_source_manifest_for_profile(args, profile)
    if scout_state.get("code_artifact_binding") != code_artifact:
        raise ValueError(
            "fresh-c8 Scout final code binding differs from current artifact")
    if profile == fresh_c8_sft.PROFILE:
        version = fresh_c8_sft.SCOUT_COMPLETION_EVIDENCE_VERSION
        profile_fields: Dict[str, Any] = {}
        component_seed = None
    elif profile == fresh_c8_sft.MULTISCOUT_COMPONENT_PROFILE:
        if type(args.seed) is not int or args.seed not in \
                fresh_c8_sft.MULTISCOUT_COMPONENT_SEEDS:
            raise ValueError(
                "fresh-c8 component completion seed must be an exact integer "
                "1 or 2")
        version = fresh_c8_sft.MULTISCOUT_COMPONENT_EVIDENCE_VERSION
        profile_fields = {"component_seed": int(args.seed)}
        component_seed = args.seed
    else:
        raise ValueError(f"unknown fresh-c8 Scout evidence profile {profile!r}")
    trajectory = _fresh_c8_trajectory_completion_facts(
        trajectory_path, profile=profile, component_seed=component_seed)
    evidence = fresh_c8_sft.signed_payload(
        version,
        {
            "profile": profile,
            **profile_fields,
            "code_artifact": code_artifact,
            "trajectory": trajectory,
            "scout": {
                "phase": scout_state.get("phase"),
                "checkpoint_kind": scout_state.get("checkpoint_kind"),
                "step": scout_state.get("step"),
                "model": scout_state.get("model"),
                "config_tier": scout_state.get("config_tier"),
                "rl_validity_mode": scout_state.get("rl_validity_mode"),
                "lambda_novelty": scout_state.get("lambda"),
                "runtime_contract": scout_state.get("runtime_contract"),
                "checkpoint_file_manifest": model_manifest,
            },
        },
    )
    _validate_fresh_c8_scout_completion_for_profile(evidence, profile)
    evidence_path = os.path.join(
        os.path.dirname(scout_final), "fresh_c8_scout_completion_evidence.json")
    common.atomic_write_json(evidence_path, evidence)
    scout_state["fresh_c8_scout_completion_evidence"] = evidence
    common.atomic_write_json(state_path, scout_state)
    return evidence, evidence_path


def _validate_persisted_fresh_c8_scout_completion(
    args: argparse.Namespace, scout_stage: Dict[str, Any], *,
    profile: str = fresh_c8_sft.PROFILE,
) -> Dict[str, Any]:
    """Revalidate completed-Scout bytes; never attribute live bytes ex post."""
    scout_final = os.path.realpath(str(scout_stage.get("final", "")))
    expected_path = os.path.realpath(os.path.join(
        os.path.dirname(scout_final), "fresh_c8_scout_completion_evidence.json"))
    evidence_path = os.path.realpath(str(
        scout_stage.get("scout_completion_evidence_path", "")))
    if evidence_path != expected_path or not os.path.isfile(evidence_path):
        raise RuntimeError(
            "completed fresh-c8 Scout lacks its completion-time evidence file")
    evidence = _validate_fresh_c8_scout_completion_for_profile(
        _load_json_object(evidence_path, "fresh-c8 Scout completion evidence"),
        profile,
    )
    expected_seed = 0 if profile == fresh_c8_sft.PROFILE else args.seed
    recorded_seed = evidence["scout"]["runtime_contract"].get("seed", -1)
    if type(expected_seed) is not int or type(recorded_seed) is not int or \
            recorded_seed != expected_seed:
        raise RuntimeError(
            "completed fresh-c8 Scout evidence seed differs from requested "
            f"profile run: recorded={recorded_seed}, requested={expected_seed}")
    if scout_stage.get("scout_completion_evidence_sha256") != \
            evidence["evidence_sha256"]:
        raise RuntimeError("completed fresh-c8 Scout evidence digest mismatch")
    state_path = os.path.join(scout_final, "trainer_state.json")
    scout_state = _load_json_object(state_path, "fresh-c8 Scout trainer_state")
    if scout_state.get("fresh_c8_scout_completion_evidence") != evidence:
        raise RuntimeError(
            "completed fresh-c8 Scout trainer_state evidence mismatch")
    if scout_state.get("code_artifact_binding") != evidence["code_artifact"]:
        raise RuntimeError(
            "completed fresh-c8 Scout trainer_state code binding mismatch")
    model_manifest = scout_state.get("checkpoint_file_manifest")
    common.verify_directory_file_manifest(
        scout_final, model_manifest, exclude_relative_paths=("trainer_state.json",))
    current_scout = {
        "phase": scout_state.get("phase"),
        "checkpoint_kind": scout_state.get("checkpoint_kind"),
        "step": scout_state.get("step"),
        "model": scout_state.get("model"),
        "config_tier": scout_state.get("config_tier"),
        "rl_validity_mode": scout_state.get("rl_validity_mode"),
        "lambda_novelty": scout_state.get("lambda"),
        "runtime_contract": scout_state.get("runtime_contract"),
        "checkpoint_file_manifest": model_manifest,
    }
    if current_scout != evidence["scout"]:
        raise RuntimeError("completed fresh-c8 Scout model/runtime drifted")
    if _validated_code_source_manifest_for_profile(args, profile) != \
            evidence["code_artifact"]:
        raise RuntimeError(
            "completed fresh-c8 Scout code artifact differs from completion time")
    traj_paths = list(scout_stage.get("traj_paths", []))
    if len(traj_paths) != 1:
        raise RuntimeError("completed fresh-c8 Scout must record one trajectory file")
    current_trajectory = _fresh_c8_trajectory_completion_facts(
        traj_paths[0], profile=profile,
        component_seed=(expected_seed if profile ==
                        fresh_c8_sft.MULTISCOUT_COMPONENT_PROFILE else None))
    if current_trajectory != evidence["trajectory"]:
        raise RuntimeError(
            "completed fresh-c8 Scout trajectory bytes/semantics drifted")
    return evidence


def _build_fresh_c8_sft_training_evidence(
    args: argparse.Namespace, state: Dict[str, Any], *, round_idx: int,
    traj_paths: List[str], accepted_path: str, funnel: Dict[str, Any],
) -> Dict[str, Any]:
    """Prove every canonical fact before fresh-c8 SFT can start."""
    if len(traj_paths) != 1:
        raise ValueError("fresh-c8 SingleScout requires exactly one trajectory file")
    scout_stage = _stage_done(state, round_idx, "scout")
    if not isinstance(scout_stage, dict):
        raise ValueError("fresh-c8 SFT requires a completed Scout stage")
    completion_evidence = _validate_persisted_fresh_c8_scout_completion(
        args, scout_stage, profile=fresh_c8_sft.PROFILE)
    scout_final = os.path.realpath(str(scout_stage.get("final", "")))
    state_path = os.path.join(scout_final, "trainer_state.json")
    scout_state = _load_json_object(state_path, "fresh-c8 Scout trainer_state")
    manifest = scout_state.get("checkpoint_file_manifest")
    if not isinstance(manifest, dict):
        raise ValueError("fresh-c8 Scout final lacks model-file manifest")
    common.verify_directory_file_manifest(
        scout_final, manifest, exclude_relative_paths=("trainer_state.json",))

    trajectory_path = os.path.realpath(traj_paths[0])
    if trajectory_path != os.path.realpath(str(scout_stage.get("traj_paths", [""])[0])):
        raise ValueError("fresh-c8 trajectory path differs from completed Scout stage")
    accepted_rows = _read_jsonl_objects(accepted_path, "fresh-c8 accepted library")
    unique_problems = {filter_pool_mod.c8_problem_key(row) for row in accepted_rows}
    if len(accepted_rows) != 500 or len(unique_problems) != 500:
        raise ValueError(
            "fresh-c8 coverage_pool_c8 must select exactly 500 unique problems")
    code_identity = completion_evidence["code_artifact"]
    code_source = {
        **code_identity["source"],
        "file_manifest_sha256": code_identity["file_manifest_sha256"],
    }
    runtime = scout_state.get("runtime_contract")
    evidence = fresh_c8_sft.signed_payload(
        fresh_c8_sft.TRAINING_EVIDENCE_VERSION,
        {
            "profile": fresh_c8_sft.PROFILE,
            "scout_completion_evidence": completion_evidence,
            "code_source": code_source,
            "scout": {
                "phase": scout_state.get("phase"),
                "checkpoint_kind": scout_state.get("checkpoint_kind"),
                "step": scout_state.get("step"),
                "model": scout_state.get("model"),
                "config_tier": scout_state.get("config_tier"),
                "rl_validity_mode": scout_state.get("rl_validity_mode"),
                "lambda_novelty": scout_state.get("lambda"),
                "runtime_contract": runtime,
                "checkpoint_file_manifest": manifest,
            },
            "trajectories": {
                **completion_evidence["trajectory"],
            },
            "accepted_library": {
                "policy": str(args.policy),
                "row_count": len(accepted_rows),
                "unique_problem_count": len(unique_problems),
                "file_sha256": common.file_sha256(accepted_path),
                "semantic_digest": common.semantic_sft_dataset_digest(accepted_rows),
                "training_sequence_digest": common.sft_training_sequence_digest(
                    accepted_rows, epochs=int(args.sft_epochs), seed=int(args.seed)),
            },
            "filter_funnel": funnel,
            "sft_runtime_contract": fresh_c8_sft.SFT_RUNTIME_CONTRACT,
        },
    )
    fresh_c8_sft.validate_training_evidence(evidence)
    return evidence


def _source_label(source: Dict[str, Any], field: str) -> str:
    if not isinstance(source, dict) or not source.get("artifact_ref") \
            or not source.get("artifact_digest"):
        raise ValueError(f"{field} lacks exact artifact ref/digest")
    return f"{source['artifact_ref']}#digest={source['artifact_digest']}"


def _validate_artifact_entry(entry: Any, field: str) -> Dict[str, Any]:
    if not isinstance(entry, dict):
        raise ValueError(f"{field} must be an object")
    source = entry.get("source")
    _source_label(source, f"{field}.source")
    manifest_path = str(entry.get("download_manifest", ""))
    raw_manifest = _load_json_object(manifest_path, f"{field}.download_manifest")
    validated = common.validate_artifact_download_manifest(
        raw_manifest, expected_source=source)
    return {"entry": entry, "source": source, "download": validated}


def _require_attributed_path(path: str, download: Dict[str, Any], field: str) -> str:
    root = os.path.realpath(download["download_root"])
    target = os.path.realpath(path)
    if os.path.commonpath([root, target]) != root:
        raise ValueError(f"{field} escapes its artifact download root")
    if not os.path.exists(target):
        raise ValueError(f"{field} does not exist: {target!r}")
    relative = os.path.relpath(target, root).replace(os.sep, "/")
    members = {item["relative_path"] for item in download["files"]}
    if os.path.isfile(target) and relative not in members:
        raise ValueError(f"{field} is not an attributed artifact member")
    if os.path.isdir(target):
        prefix = "" if relative == "." else relative.rstrip("/") + "/"
        if not any(member.startswith(prefix) for member in members):
            raise ValueError(f"{field} contains no attributed artifact members")
    return target


def _stable_artifact_identity(download: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "version": download["version"],
        "source": download["source"],
        "files": download["files"],
        "file_manifest_sha256": download["file_manifest_sha256"],
    }


def _is_c9_full(args: argparse.Namespace) -> bool:
    return (
        str(getattr(args, "c9_execution_profile", "") or "") ==
        c9_evidence.C9_PROFILE
        and str(getattr(args, "stabilization_cell", "") or "") ==
        c9_evidence.C9_CELL
    )


def _c9_code_identity(args: argparse.Namespace) -> Dict[str, Any]:
    return _validated_executing_code_source_manifest(
        args, profile_label="C9 full/C5")


def _c9_scout_runtime_contract(
    args: argparse.Namespace, *, scout_init: str, round_idx: int, lam: float,
) -> Dict[str, Any]:
    gargs = _grpo_args(
        args, phase="scout", model=scout_init,
        output_dir=os.path.join(
            args.output_dir, f"round_{round_idx:02d}", "scout"),
        lam=lam, steps=int(args.scout_steps), round_idx=round_idx,
        global_step_offset=(round_idx - 1) * (
            int(args.scout_steps) + int(args.central_steps)),
    )
    grpo_gpu.apply_completion_budget(gargs)
    grpo_gpu.enforce_probe_serving_len(gargs, pool=None)
    return grpo_gpu.build_runtime_contract(
        gargs, resolved_lr=float(gargs.lr),
        resolved_steps=int(gargs.steps))


def _c9_trajectory_facts(path: str) -> Dict[str, Any]:
    rows = _read_jsonl_objects(path, "C9 Scout trajectories")
    per_step: Dict[int, int] = {}
    for row_number, row in enumerate(rows, 1):
        step = row.get("step")
        if type(step) is not int or not 1 <= step <= c9_evidence.SCOUT_STEPS:
            raise ValueError(
                f"C9 trajectory row {row_number} has invalid step {step!r}")
        per_step[step] = per_step.get(step, 0) + 1
        expected = {
            "trajectory_schema_version": (
                grpo_gpu.TRAJECTORY_RECORD_SCHEMA_VERSION),
            "expdis_stabilization_mode": c9_evidence.C9_RND_LIFECYCLE,
            "scientific_variant": c9_evidence.C9_SCIENTIFIC_VARIANT,
            "c9_execution_profile": c9_evidence.C9_PROFILE,
            "stabilization_cell": c9_evidence.C9_CELL,
            "novelty_normalization": c9_evidence.C9_NOVELTY_NORMALIZATION,
            "rnd_update_scope": "final_selected_learner_rows",
        }
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError(
                f"C9 trajectory row {row_number} has wrong lifecycle/profile")
        lengths = {
            key: row.get(key) for key in (
                "completion_token_length",
                "completion_token_length_true",
                "completion_token_length_retokenized",
            )
        }
        if any(type(value) is not int or value < 0
               for value in lengths.values()):
            raise ValueError(
                f"C9 trajectory row {row_number} lacks exact completion "
                "length accounting")
        capped = lengths["completion_token_length"]
        true = lengths["completion_token_length_true"]
        if capped != min(true, 16384):
            raise ValueError(
                f"C9 trajectory row {row_number} has inconsistent true/capped "
                "completion lengths")
        for key in (
            "novelty_reward", "blended_reward", "correctness_reward",
            "novelty_normalization_center", "novelty_normalization_scale",
            "novelty_normalization_effective_scale",
        ):
            if not math.isfinite(float(row.get(key, math.nan))):
                raise ValueError(
                    f"C9 trajectory row {row_number} has nonfinite {key}")
        if not bool(row.get("is_correct", False)) and not math.isclose(
                float(row["novelty_reward"]), 0.0, abs_tol=1e-7):
            raise ValueError(
                "C9 incorrect trajectory received nonzero novelty credit")
    if len(rows) != c9_evidence.SCOUT_TRAJECTORY_ROWS:
        raise ValueError(
            f"C9 Scout produced {len(rows)} trajectories; expected 12,800")
    if set(per_step) != set(range(1, c9_evidence.SCOUT_STEPS + 1)) or \
            set(per_step.values()) != {c9_evidence.SCOUT_ROWS_PER_STEP}:
        raise ValueError("C9 Scout trajectory steps are not exactly 200 x 64")
    return {
        "row_count": len(rows),
        "step_count": len(per_step),
        "rows_per_step": c9_evidence.SCOUT_ROWS_PER_STEP,
        "file_sha256": common.file_sha256(path),
        "canonical_rows_sha256": common.canonical_json_sha256(rows),
    }


def _replay_c9_single_scout_filter(
    args: argparse.Namespace, *, trajectory_paths: List[str],
    accepted_path: str, funnel: Dict[str, Any],
) -> Dict[str, Any]:
    """Reconstruct the exact C9 SingleScout selector output from raw rows.

    The MR-ME validator already owns a strict, independent replay of
    ``coverage_pool_c8``.  Reusing it here keeps SingleScout and MultiScout on
    one selector implementation while adding a signed certificate that binds
    the replayed ordered rows to the raw Scout and serialized SFT library.
    """
    if len(trajectory_paths) != 1:
        raise RuntimeError(
            "C9 SingleScout filter replay requires exactly one raw trajectory file")
    source = os.path.realpath(str(trajectory_paths[0]))
    output = os.path.realpath(str(accepted_path))
    replay = mrme_contract.replay_coverage_pool_c8(
        [source], accepted_path=output, recorded_funnel=funnel,
        max_examples=c9_evidence.ACCEPTED_ROWS,
        max_completion_tokens=_token_gate_cap(args),
    )
    certificate = c9_evidence.signed_payload(
        c9_evidence.SINGLESCOUT_FILTER_REPLAY_VERSION,
        {
            "policy": filter_pool_mod.C8_POLICY,
            "num_sources": 1,
            "source_row_count": int(funnel["raw"]),
            "accepted_row_count": len(replay["accepted_rows"]),
            "accepted_unique_problems": int(
                funnel["accepted_unique_problems"]),
            "source_balance": False,
            "selector_replay": "raw_to_serialized_accepted_exact_order",
            "source_file_sha256": common.file_sha256(source),
            "accepted_file_sha256": common.file_sha256(output),
            "accepted_rows_sha256": common.canonical_json_sha256(
                replay["accepted_rows"]),
            "funnel_sha256": common.canonical_json_sha256(replay["funnel"]),
        },
    )
    return c9_evidence.validate_single_scout_filter_replay(certificate)


def _c9_probe_history_facts(
    args: argparse.Namespace, *, phase: str, local_steps: int,
    global_step_offset: int,
) -> Dict[str, Any]:
    """Recompute the exact C9 diagnostic-probe history from durable metrics.

    Probes are cheap avg@4 diagnostics, not headline avg@64 estimates.  They
    nevertheless use the publication evaluator's AIME24 rows, prompt renderer,
    grader, P1 decoding settings, and completion budget.  Every scheduled C9
    boundary must be present, fresh, and complete before a stage can hand off.
    """
    if int(args.probe_every) <= 0:
        raise RuntimeError("C9 full stages require diagnostic probes")
    if grpo_gpu.PROBE_PROTOCOL_ID != c9_evidence.C9_PROBE_PROTOCOL_ID:
        raise RuntimeError("C9 signed probe protocol and GRPO probe drifted")
    parity_phase = common.PARITY_PHASES[phase]
    expected_steps = [
        global_step_offset + step
        for step in range(int(args.probe_every), local_steps + 1,
                          int(args.probe_every))
    ]
    metrics_path = str(getattr(args, "metrics_jsonl", "") or os.path.join(
        args.output_dir, "metrics.jsonl"))
    rows = _read_jsonl_objects(metrics_path, "C9 durable metrics")
    probe_rows = [
        row for row in rows
        if row.get("phase") == parity_phase
        and "eval/probe_protocol" in row
    ]
    by_step: Dict[int, List[Dict[str, Any]]] = {}
    for row in probe_rows:
        by_step.setdefault(int(row.get("global_step", -1)), []).append(row)
    if set(by_step) != set(expected_steps) or any(
            len(by_step[step]) != 1 for step in expected_steps):
        raise RuntimeError(
            f"C9 {phase} probe history does not contain exactly one row at "
            f"each boundary {expected_steps}; observed={sorted(by_step)}")
    exact_rows: List[Dict[str, Any]] = []
    for step in expected_steps:
        row = by_step[step][0]
        exact = {
            "global_step": step,
            "phase": parity_phase,
            "protocol": row.get("eval/probe_protocol"),
            "headline_eligible": row.get("eval/probe_headline_eligible"),
            "metric_scale": row.get("eval/probe_metric_scale"),
            "sample_seed_rule": row.get("eval/probe_sample_seed_rule"),
            "benchmark_revision": row.get("eval/probe_benchmark_revision"),
            "benchmark_rows_sha256": row.get(
                "eval/probe_benchmark_rows_sha256"),
            "prompt_matrix_sha256": row.get(
                "eval/probe_prompt_matrix_sha256"),
            "protocol_complete": row.get("eval/probe_protocol_complete"),
            "graded_rows": row.get("eval/probe_graded_rows"),
            "expected_rows": row.get("eval/probe_expected_rows"),
            "complete_problem_count": row.get(
                "eval/probe_complete_problem_count"),
            "expected_problem_count": row.get(
                "eval/probe_expected_problem_count"),
            "samples_per_problem": row.get("eval/probe_samples_per_problem"),
            "max_completion_tokens": row.get(
                "eval/probe_max_completion_tokens"),
            "temperature": row.get("eval/probe_temperature"),
            "top_p": row.get("eval/probe_top_p"),
            "top_k": row.get("eval/probe_top_k"),
            "min_p": row.get("eval/probe_min_p"),
            "seed_base": row.get("eval/probe_seed_base"),
            "policy_step": row.get("eval/probe_policy_step"),
            "staleness_updates": row.get("eval/probe_staleness_updates"),
            "fresh_weights": row.get("eval/probe_fresh_weights"),
            "avg_at_4": row.get("eval/AIME24_avg_at_4"),
        }
        expected = {
            "protocol": grpo_gpu.PROBE_PROTOCOL_ID,
            "headline_eligible": 0.0,
            "metric_scale": "fraction",
            "sample_seed_rule": grpo_gpu.PROBE_SAMPLE_SEED_RULE,
            "benchmark_revision": grpo_gpu.PROBE_BENCHMARK_REVISION,
            "benchmark_rows_sha256": grpo_gpu.PROBE_BENCHMARK_ROWS_SHA256,
            "protocol_complete": 1.0,
            "graded_rows": 30.0 * grpo_gpu.PROBE_NUM_SAMPLES,
            "expected_rows": 30.0 * grpo_gpu.PROBE_NUM_SAMPLES,
            "complete_problem_count": 30.0,
            "expected_problem_count": 30.0,
            "samples_per_problem": float(grpo_gpu.PROBE_NUM_SAMPLES),
            "max_completion_tokens": float(grpo_gpu.PROBE_MAX_TOKENS),
            "temperature": float(grpo_gpu.PROBE_TEMPERATURE),
            "top_p": float(grpo_gpu.PROBE_TOP_P),
            "top_k": float(grpo_gpu.PROBE_TOP_K),
            "min_p": float(grpo_gpu.PROBE_MIN_P),
            "seed_base": float(grpo_gpu.PROBE_SEED_BASE),
            "policy_step": step - global_step_offset,
            "staleness_updates": 0,
            "fresh_weights": 1.0,
        }
        drift = [
            f"{key}={exact.get(key)!r} (expected {value!r})"
            for key, value in expected.items() if exact.get(key) != value
        ]
        prompt_sha = exact.get("prompt_matrix_sha256")
        avg_at_4 = float(exact.get("avg_at_4", math.nan))
        if drift or not isinstance(prompt_sha, str) or len(prompt_sha) != 64 or \
                not math.isfinite(avg_at_4) or not 0.0 <= avg_at_4 <= 1.0:
            raise RuntimeError(
                f"C9 {phase} probe at global step {step} is not exact/fresh: "
                + "; ".join(drift or [
                    "prompt hash or fractional avg_at_4 is invalid"]))
        exact_rows.append(exact)
    if len({row["prompt_matrix_sha256"] for row in exact_rows}) != 1:
        raise RuntimeError("C9 diagnostic probe prompt matrix changed across steps")
    return {
        "probe_protocol_id": grpo_gpu.PROBE_PROTOCOL_ID,
        "phase": parity_phase,
        "expected_global_steps": expected_steps,
        "row_count": len(exact_rows),
        "prompt_matrix_sha256": exact_rows[0]["prompt_matrix_sha256"],
        "canonical_rows_sha256": common.canonical_json_sha256(exact_rows),
    }


def _record_c9_scout_completion(
    args: argparse.Namespace, *, scout_final: str, trajectory_path: str,
    scout_init: str, round_idx: int, lam: float,
) -> tuple[Dict[str, Any], str]:
    if not _is_c9_full(args):
        raise RuntimeError("C9 completion evidence is only for full/C5")
    state = _load_json_object(
        os.path.join(scout_final, "trainer_state.json"),
        "C9 Scout final trainer_state")
    expected_runtime = _c9_scout_runtime_contract(
        args, scout_init=scout_init, round_idx=round_idx, lam=lam)
    expected_code = _c9_code_identity(args)
    if state.get("phase") != "scout" or state.get("checkpoint_kind") != \
            "deployable_model" or state.get("step") != c9_evidence.SCOUT_STEPS:
        raise RuntimeError("C9 Scout final is not the fixed 200-step endpoint")
    if state.get("runtime_contract") != expected_runtime:
        raise RuntimeError("C9 Scout final runtime contract mismatch")
    if state.get("code_artifact_binding") != expected_code:
        raise RuntimeError("C9 Scout final code binding mismatch")
    if state.get("rnd_selected_batch_update_count") != \
            c9_evidence.SCOUT_STEPS:
        raise RuntimeError("C9 Scout final does not certify one RND update per step")
    model_manifest = state.get("checkpoint_file_manifest")
    common.verify_directory_file_manifest(
        scout_final, model_manifest, exclude_relative_paths=("trainer_state.json",))
    evidence = c9_evidence.signed_payload(
        c9_evidence.SCOUT_COMPLETION_EVIDENCE_VERSION,
        {
            "c9_execution_profile": c9_evidence.C9_PROFILE,
            "stabilization_cell": c9_evidence.C9_CELL,
            "scientific_variant": c9_evidence.C9_SCIENTIFIC_VARIANT,
            "seed": 0,
            "round": 1,
            "scout_steps": c9_evidence.SCOUT_STEPS,
            "rnd_selected_batch_update_count": state[
                "rnd_selected_batch_update_count"],
            "scout_runtime_contract": expected_runtime,
            "code_artifact": expected_code,
            "model_file_manifest": model_manifest,
            "trajectory": _c9_trajectory_facts(trajectory_path),
            "probe_history": _c9_probe_history_facts(
                args, phase="scout", local_steps=c9_evidence.SCOUT_STEPS,
                global_step_offset=0),
        },
    )
    c9_evidence.validate_scout_completion_evidence(
        evidence, expected_runtime_contract=expected_runtime,
        expected_code_artifact=expected_code)
    path = os.path.join(
        os.path.dirname(scout_final), "c9_scout_completion_evidence.json")
    common.atomic_write_json(path, evidence)
    return evidence, path


def _validate_persisted_c9_scout_completion(
    args: argparse.Namespace, scout_stage: Dict[str, Any], *, scout_init: str,
    round_idx: int, lam: float,
) -> Dict[str, Any]:
    path = str(scout_stage.get("c9_scout_completion_evidence_path", ""))
    evidence = _load_json_object(path, "C9 Scout completion evidence")
    expected_runtime = _c9_scout_runtime_contract(
        args, scout_init=scout_init, round_idx=round_idx, lam=lam)
    expected_code = _c9_code_identity(args)
    c9_evidence.validate_scout_completion_evidence(
        evidence, expected_runtime_contract=expected_runtime,
        expected_code_artifact=expected_code)
    if scout_stage.get("c9_scout_completion_evidence_sha256") != \
            evidence["evidence_sha256"]:
        raise RuntimeError("C9 pipeline-state Scout evidence SHA mismatch")
    final_dir = str(scout_stage.get("final", ""))
    common.verify_directory_file_manifest(
        final_dir, evidence["model_file_manifest"],
        exclude_relative_paths=("trainer_state.json",))
    trainer_state = _load_json_object(
        os.path.join(final_dir, "trainer_state.json"),
        "persisted C9 Scout trainer_state")
    if trainer_state.get("runtime_contract") != expected_runtime or \
            trainer_state.get("code_artifact_binding") != expected_code or \
            trainer_state.get("rnd_selected_batch_update_count") != \
            c9_evidence.SCOUT_STEPS:
        raise RuntimeError("persisted C9 Scout trainer state drifted")
    paths = list(scout_stage.get("traj_paths", []))
    if len(paths) != 1 or _c9_trajectory_facts(paths[0]) != evidence["trajectory"]:
        raise RuntimeError("persisted C9 Scout trajectory bytes drifted")
    if _c9_probe_history_facts(
            args, phase="scout", local_steps=c9_evidence.SCOUT_STEPS,
            global_step_offset=0) != evidence.get("probe_history"):
        raise RuntimeError("persisted C9 Scout probe history drifted")
    return evidence


def _validate_c9_canary_scout_endpoint(
    args: argparse.Namespace, scout_stage: Dict[str, Any], *, scout_init: str,
    round_idx: int, lam: float,
) -> Dict[str, Any]:
    """Prove the step-50 mechanism-screen endpoint before its panel gate."""
    if str(args.c9_execution_profile) != "canary":
        raise RuntimeError("C9 canary endpoint validator used outside canary")
    final_dir = str(scout_stage.get("final", ""))
    state = _load_json_object(
        os.path.join(final_dir, "trainer_state.json"),
        "C9 canary Scout trainer_state")
    runtime = _c9_scout_runtime_contract(
        args, scout_init=scout_init, round_idx=round_idx, lam=lam)
    code = _c9_code_identity(args)
    if state.get("phase") != "scout" or state.get("checkpoint_kind") != \
            "deployable_model" or state.get("step") != 50 or \
            state.get("runtime_contract") != runtime or \
            state.get("code_artifact_binding") != code:
        raise RuntimeError("C9 canary is not the exact code-bound step-50 endpoint")
    health = state.get("c9_canary_health_state")
    if not isinstance(health, dict) or health.get("completed_step") != 50 or \
            int(health.get("consecutive_bad_windows", 3)) >= 3:
        raise RuntimeError("C9 canary lacks passing resume-stable rolling health")
    recent = health.get("recent_clip_valid")
    if not isinstance(recent, list) or len(recent) != 8 or any(
            not isinstance(item, list) or len(item) != 2
            or not all(math.isfinite(float(value)) for value in item)
            for item in recent):
        raise RuntimeError("C9 canary rolling-health history is invalid")
    expected_rnd_updates = (
        50 if args.expdis_stabilization_mode == c9_evidence.C9_RND_LIFECYCLE
        else None)
    if state.get("rnd_selected_batch_update_count") != expected_rnd_updates:
        raise RuntimeError("C9 canary RND lifecycle/update count mismatch")
    manifest = state.get("checkpoint_file_manifest")
    common.verify_directory_file_manifest(
        final_dir, manifest, exclude_relative_paths=("trainer_state.json",))
    paths = list(scout_stage.get("traj_paths", []))
    if len(paths) != 1:
        raise RuntimeError("C9 canary must have one trajectory file")
    rows = _read_jsonl_objects(paths[0], "C9 canary trajectories")
    per_step: Dict[int, int] = {}
    expected_variant = (
        c9_evidence.C9_SCIENTIFIC_VARIANT
        if str(args.stabilization_cell) == "C5"
        else c9_evidence.C9_SCREEN_VARIANT)
    for row_number, row in enumerate(rows, 1):
        step = row.get("step")
        if type(step) is not int or not 1 <= step <= 50:
            raise RuntimeError(f"C9 canary trajectory row {row_number} step mismatch")
        per_step[step] = per_step.get(step, 0) + 1
        if row.get("c9_execution_profile") != "canary" or \
                row.get("stabilization_cell") != str(args.stabilization_cell) or \
                row.get("scientific_variant") != expected_variant:
            raise RuntimeError("C9 canary trajectory identity mismatch")
        numeric_keys = (
            "correctness_reward", "novelty_reward", "blended_reward")
        if any(not math.isfinite(float(row.get(key, math.nan)))
               for key in numeric_keys):
            raise RuntimeError("C9 canary trajectory contains non-finite rewards")
        if not bool(row.get("is_correct", False)) and not math.isclose(
                float(row.get("novelty_reward", math.nan)), 0.0, abs_tol=1e-7):
            raise RuntimeError("C9 canary incorrect row has nonzero novelty credit")
        if args.expdis_stabilization_mode == c9_evidence.C9_RND_LIFECYCLE:
            if row.get("expdis_stabilization_mode") != \
                    c9_evidence.C9_RND_LIFECYCLE or \
                    row.get("rnd_update_scope") != \
                    "final_selected_learner_rows" or \
                    row.get("novelty_normalization") != \
                    str(args.novelty_normalization):
                raise RuntimeError("C9 canary RND trajectory contract mismatch")
            for key in (
                    "novelty_normalization_center",
                    "novelty_normalization_scale",
                    "novelty_normalization_effective_scale"):
                if not math.isfinite(float(row.get(key, math.nan))):
                    raise RuntimeError(
                        "C9 canary trajectory has non-finite normalization")
    if len(rows) != 50 * c9_evidence.SCOUT_ROWS_PER_STEP or \
            set(per_step) != set(range(1, 51)) or \
            set(per_step.values()) != {c9_evidence.SCOUT_ROWS_PER_STEP}:
        raise RuntimeError("C9 canary trajectories are not exactly 50 x 64")
    return {
        "runtime_contract": runtime,
        "code_artifact": code,
        "model_file_manifest": manifest,
        "trajectory_file_sha256": common.file_sha256(paths[0]),
        "trajectory_rows": len(rows),
        "rolling_health": health,
    }


def _metric_close(actual: Any, expected: float, *, field: str) -> None:
    if isinstance(actual, bool) or isinstance(actual, np.bool_):
        raise RuntimeError(f"C10 metric {field} is Boolean, not numeric")
    try:
        value = float(actual)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"C10 metric {field} is not numeric") from exc
    if not math.isfinite(value) or not math.isclose(
            value, float(expected), rel_tol=1e-7, abs_tol=1e-7):
        raise RuntimeError(
            f"C10 metric {field}={value!r}; expected {expected!r}")


def _c10_observed_equal(left: Any, right: Any) -> bool:
    """Portable equality for a persisted C10 failure observation.

    JSON round-trips preserve non-finite Python floats but ordinary equality
    does not consider NaN equal to itself.  Failure evidence must still bind
    the exact *kind* of non-finite value observed, while accepting harmless
    Python/NumPy scalar representation changes.
    """
    if isinstance(left, dict) or isinstance(right, dict):
        return isinstance(left, dict) and isinstance(right, dict) and \
            set(left) == set(right) and all(
                _c10_observed_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return isinstance(left, (list, tuple)) and \
            isinstance(right, (list, tuple)) and len(left) == len(right) and \
            all(_c10_observed_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, bool) or isinstance(right, bool) or \
            isinstance(left, np.bool_) or isinstance(right, np.bool_):
        return type(left) is type(right) and bool(left) == bool(right)
    if isinstance(left, numbers.Real) and isinstance(right, numbers.Real):
        left_float, right_float = float(left), float(right)
        if math.isnan(left_float) or math.isnan(right_float):
            return math.isnan(left_float) and math.isnan(right_float)
        return left_float == right_float
    return type(left) is type(right) and left == right


def _replay_c10_survivor_contract(
    rows: List[Dict[str, Any]], metrics_rows: List[Dict[str, Any]],
    *, expected_completed_steps: int = 50,
    transport_invalid_step: Optional[int] = None,
) -> Dict[str, Any]:
    """Recompute every selected group, policy baseline, and health window."""
    if type(expected_completed_steps) is not int or not \
            0 <= expected_completed_steps <= 50:
        raise RuntimeError("C10 replay step boundary is invalid")
    if transport_invalid_step is not None and (
            type(transport_invalid_step) is not int or
            transport_invalid_step != expected_completed_steps or
            transport_invalid_step < 1):
        raise RuntimeError("C10 transport replay boundary is invalid")
    rows_by_step: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        exact_row_identity = {
            "trajectory_schema_version":
                grpo_gpu.TRAJECTORY_RECORD_SCHEMA_VERSION,
            "scientific_variant": c10_evidence.C10_SCIENTIFIC_VARIANT,
            "c10_execution_profile": "canary",
            "c9_execution_profile": "",
            "stabilization_cell": "",
            "expdis_stabilization_mode": grpo_gpu.C9_RND_LIFECYCLE,
            "novelty_normalization": "selected_batch_zscore",
            "rnd_update_scope": "final_selected_learner_rows",
        }
        if any(row.get(key) != wanted
               for key, wanted in exact_row_identity.items()):
            raise RuntimeError("C10 trajectory contract identity mismatch")
        true_len_raw = row.get("completion_token_length_true")
        scored_len_raw = row.get("completion_token_length")
        retokenized_len_raw = row.get("completion_token_length_retokenized")
        if not grpo_gpu.c10_integral_numeric(true_len_raw) or not \
                grpo_gpu.c10_integral_numeric(scored_len_raw) or not \
                grpo_gpu.c10_integral_numeric(retokenized_len_raw):
            raise RuntimeError("C10 trajectory length evidence is invalid")
        true_len = int(float(true_len_raw))
        scored_len = int(float(scored_len_raw))
        expected_clipped = (
            str(row.get("finish_reason", "")).lower() == "length"
            or true_len >= 16384)
        if true_len < 0 or scored_len != min(true_len, 16384) or \
                type(row.get("clipped")) is not bool or \
                bool(row.get("clipped")) != expected_clipped:
            raise RuntimeError(
                "C10 trajectory clipped/survivor status does not replay")
        if type(row.get("step")) is not int:
            raise RuntimeError("C10 trajectory step identity is invalid")
        rows_by_step.setdefault(row["step"], []).append(row)
    metrics_by_step: Dict[int, Dict[str, Any]] = {}
    for metric in metrics_rows:
        step = metric.get("global_step")
        if type(step) is not int or metric.get("grpo/step") != step or \
                step in metrics_by_step:
            raise RuntimeError("C10 metrics have invalid or duplicate steps")
        metrics_by_step[step] = metric
    expected_steps = set(range(1, expected_completed_steps + 1))
    if set(rows_by_step) != expected_steps or \
            set(metrics_by_step) != expected_steps:
        raise RuntimeError(
            "C10 replay requires the exact completed-step prefix")

    total_survivors = 0
    total_masked = 0
    min_survivors = 16
    max_survivors = 0
    max_advantage_sum = 0.0
    health_points: List[List[float]] = []
    consecutive_bad_windows = 0
    per_step_replay: List[Dict[str, Any]] = []
    for step in range(1, expected_completed_steps + 1):
        step_rows = rows_by_step[step]
        if len(step_rows) != 64:
            raise RuntimeError(f"C10 step {step} does not contain 64 rows")
        group_summaries = []
        for start in range(0, 64, 16):
            group = step_rows[start:start + 16]
            problem_ids = {row.get("problem_id") for row in group}
            prompts = {row.get("prompt_text") for row in group}
            if any(not isinstance(value, str) or not value
                   for value in problem_ids) or len(problem_ids) != 1 or \
                    any(not isinstance(value, str) or not value
                        for value in prompts) or len(prompts) != 1:
                raise RuntimeError(
                    f"C10 step {step} group does not contain one prompt")
            for row in group:
                completion_text = row.get("completion_text")
                ground_truth = row.get("ground_truth")
                if not isinstance(completion_text, str) or not isinstance(
                        ground_truth, str):
                    raise RuntimeError(
                        "C10 trajectory lacks raw completion/ground truth")
                replay_correct = is_correct_completion(
                    completion_text, ground_truth)
                replay_valid = bool(analyze_completion_text(
                    completion_text).get("valid_answer_presentation", False))
                if type(row.get("is_correct")) is not bool or type(row.get(
                        "valid_answer")) is not bool or bool(row.get(
                            "is_correct")) is not replay_correct or bool(
                                row.get("valid_answer")) is not replay_valid:
                    raise RuntimeError(
                        "C10 trajectory correctness/validity flags do not "
                        "replay from raw completion and ground truth")
                expected_correctness = 1.0 if replay_correct else -1.0
                expected_eligibility_novelty = (
                    float(row["c10_rnd_raw_novelty_score"])
                    if replay_correct else 0.0)
                _metric_close(
                    row.get("dynamic_sampling_eligibility_novelty_used"),
                    expected_eligibility_novelty,
                    field="trajectory raw correct-gated eligibility novelty")
                _metric_close(
                    row.get("correctness_reward"), expected_correctness,
                    field="trajectory correctness reward")
                expected_eligibility_reward = (
                    expected_correctness
                    + 0.5 * float(row[
                        "dynamic_sampling_eligibility_novelty_used"])
                    + common.soft_overlong_penalty(
                        int(row["completion_token_length"]), 13107, 3277)
                )
                _metric_close(
                    row.get("dynamic_sampling_eligibility_blended_reward"),
                    expected_eligibility_reward,
                    field="trajectory eligibility blended reward")
            replay_rows = [{
                "is_correct": bool(row["is_correct"]),
                "clipped": bool(row["clipped"]),
                "blended_reward": float(row[
                    "dynamic_sampling_eligibility_blended_reward"]),
                "novelty_used": float(row[
                    "dynamic_sampling_eligibility_novelty_used"]),
            } for row in group]
            reason, eligible, facts = grpo_gpu.group_eligibility_for_rows(
                replay_rows, lambda_novelty=0.5, validity_mode="c8",
                survivor_only=True)
            if not eligible or any(
                    row.get("dynamic_sampling_reason") != reason
                    or row.get("dynamic_sampling_group_selected") is not True
                    for row in group):
                raise RuntimeError(
                    f"C10 step {step} selected an ineligible group")
            exact_fields = {
                "dynamic_sampling_raw_group_size": facts["raw_group_size"],
                "dynamic_sampling_survivor_count": facts["survivor_count"],
                "dynamic_sampling_masked_count": facts["masked_count"],
                "dynamic_sampling_raw_correct_count": facts[
                    "raw_correct_count"],
                "dynamic_sampling_survivor_correct_count": facts[
                    "survivor_correct_count"],
                "dynamic_sampling_eligibility_population_size": facts[
                    "eligibility_population_size"],
            }
            for row in group:
                if any(not grpo_gpu.c10_exact_numeric(
                        row.get(key), expected)
                       for key, expected in exact_fields.items()):
                    raise RuntimeError(
                        f"C10 step {step} declared group counts do not replay")
                if row.get("dynamic_sampling_policy_loss_eligible") is not \
                        (not bool(row["clipped"])):
                    raise RuntimeError(
                        f"C10 step {step} row policy eligibility is false")
                _metric_close(
                    row.get("dynamic_sampling_reward_std"),
                    facts["eligibility_reward_std"],
                    field="trajectory eligibility reward std")
                _metric_close(
                    row.get("dynamic_sampling_novelty_contribution_std"),
                    facts["eligibility_novelty_contribution_std"],
                    field="trajectory eligibility novelty std")
            group_summaries.append({
                "problem_id": next(iter(problem_ids)),
                "reason": reason,
                "survivors": int(facts["survivor_count"]),
                "masked": int(facts["masked_count"]),
                "survivor_correct": int(facts["survivor_correct_count"]),
                "eligibility_reward_std": float(
                    facts["eligibility_reward_std"]),
                "eligibility_novelty_contribution_std": float(
                    facts["eligibility_novelty_contribution_std"]),
            })

        raw_novelty = np.asarray([
            float(row["c10_rnd_raw_novelty_score"])
            for row in step_rows], dtype=np.float32)
        if not np.all(np.isfinite(raw_novelty)):
            raise RuntimeError("C10 trajectory raw RND scores are non-finite")
        normalization_center = float(np.mean(raw_novelty, dtype=np.float64))
        normalization_scale = float(np.std(raw_novelty, dtype=np.float64))
        if normalization_scale <= grpo_gpu.C9_NORMALIZATION_EPSILON:
            raise RuntimeError("C10 endpoint has degenerate RND normalization")
        normalized = ((raw_novelty.astype(np.float64)
                       - normalization_center) / normalization_scale).astype(
                           np.float32)
        final_rewards_list: List[float] = []
        for index, row in enumerate(step_rows):
            pre_gate = float(row["c10_normalized_novelty_pre_gate"])
            _metric_close(
                pre_gate, float(normalized[index]),
                field="trajectory normalized novelty")
            expected_novelty = pre_gate if bool(row["is_correct"]) else 0.0
            _metric_close(
                row.get("novelty_reward"), expected_novelty,
                field="trajectory correct-gated novelty")
            expected_reward = (
                (1.0 if bool(row["is_correct"]) else -1.0)
                + 0.5 * expected_novelty
                + common.soft_overlong_penalty(
                    int(row["completion_token_length"]), 13107, 3277)
            )
            _metric_close(
                row.get("blended_reward"), expected_reward,
                field="trajectory final blended reward")
            final_rewards_list.append(expected_reward)
        final_rewards = np.asarray(final_rewards_list, dtype=np.float32)
        survivor_mask = np.asarray(
            [not bool(row["clipped"]) for row in step_rows], dtype=bool)
        advantages, advantage_stats = \
            grpo_gpu.group_mean_advantages_unmasked(
                final_rewards, survivor_mask, 16)
        survivors = int(survivor_mask.sum())
        masked = 64 - survivors
        rows_backward = sum(
            bool(survivor_mask[index])
            and int(step_rows[index][
                "completion_token_length_retokenized"]) > 0
            and abs(float(advantages[index])) >= 1e-12
            for index in range(64)
        )
        zero_advantage_survivors = survivors - rows_backward
        metric = metrics_by_step[step]
        exact_metric_identity = {
            "phase": common.PARITY_PHASES["scout"],
            "rl_validity_mode": "c8",
            "c10_execution_profile": "canary",
            "c9_execution_profile": "",
            "stabilization_cell": "",
            "expdis_stabilization_mode": grpo_gpu.C9_RND_LIFECYCLE,
            "novelty_normalization": "selected_batch_zscore",
            "survivor_overlong_filter_version":
                grpo_gpu.C10_OVERLONG_FILTER_CONTRACT_VERSION,
            "scientific_variant": c10_evidence.C10_SCIENTIFIC_VARIANT,
        }
        if any(metric.get(key) != wanted
               for key, wanted in exact_metric_identity.items()):
            raise RuntimeError("C10 metric identity mismatch")
        exact_numeric_identity = {
            "round": 1,
            "lambda": 0.5,
            "lambda_novelty": 0.5,
            "num_scouts": 1,
        }
        if any(not grpo_gpu.c10_exact_numeric(
                metric.get(key), wanted)
               for key, wanted in exact_numeric_identity.items()):
            raise RuntimeError(
                "C10 metric identity mismatch: numeric identity drift")
        expected_metrics = {
            "train/raw_selected_rows": 64,
            "train/survivor_rows": survivors,
            "train/masked_truncated_rows": masked,
            "train/rows_backward": rows_backward,
            "train/zero_advantage_survivor_rows":
                zero_advantage_survivors,
            "train/fixed_loss_denominator": 64 * 16384,
            "grpo/dynamic/selected_raw_row_count": 64,
            "grpo/dynamic/selected_survivor_row_count": survivors,
            "grpo/dynamic/selected_masked_row_count": masked,
            "train/overlong_filter/raw_selected_row_count": 64,
            "train/overlong_filter/survivor_row_count": survivors,
            "train/overlong_filter/masked_row_count": masked,
            "train/overlong_filter/group_count": 4,
            "train/overlong_filter/min_survivors_per_group":
                advantage_stats["min_survivors_per_group"],
            "train/overlong_filter/max_survivors_per_group":
                advantage_stats["max_survivors_per_group"],
            "train/overlong_filter/mean_survivors_per_group":
                advantage_stats["mean_survivors_per_group"],
            "train/overlong_filter/max_abs_survivor_advantage_sum":
                advantage_stats["max_abs_survivor_advantage_sum"],
            "quality/clipped_rate": masked / 64,
            "quality/valid_answer_rate": sum(
                bool(row["valid_answer"]) for row in step_rows) / 64,
            "train/correctness_rate": sum(
                bool(row["is_correct"]) for row in step_rows) / 64,
            "train/reward_total_mean": float(final_rewards.mean()),
            "train/reward_correctness_mean": float(np.mean([
                float(row["correctness_reward"]) for row in step_rows])),
            "train/reward_novelty_mean": float(np.mean([
                float(row["novelty_reward"]) for row in step_rows])),
            "grpo/reward_std": float(final_rewards.std()),
            "grpo/rnd/updates_this_step": 1,
            "grpo/rnd/update_rows": 64,
            "grpo/rnd/selected_batch_update_count": step,
            "grpo/dynamic/kept_group_count": 4,
            "grpo/dynamic/fallback_group_count": 0,
            "train/rollout_staleness_updates": 0,
            "train/rollout_staleness_known": 1,
            "train/on_policy_rollout": 1,
            "train/updates_per_rollout": 1,
            "train/policy_kl": 0,
            "train/clip_fraction": 0,
            "grpo/rnd/updates_this_step": 1,
            "grpo/rnd/update_rows": 64,
            "grpo/rnd/selected_batch_update_count": step,
            "grpo/reward/novelty_normalization_center":
                normalization_center,
            "grpo/reward/novelty_normalization_scale":
                normalization_scale,
            "grpo/reward/novelty_normalization_effective_scale":
                normalization_scale,
            "grpo/reward/novelty_normalized_selected_mean":
                float(np.mean(normalized, dtype=np.float64)),
            "grpo/reward/novelty_normalized_selected_population_std":
                float(np.std(normalized, dtype=np.float64)),
        }
        exact_metric_fields = {
            "train/raw_selected_rows", "train/survivor_rows",
            "train/masked_truncated_rows", "train/rows_backward",
            "train/zero_advantage_survivor_rows",
            "train/fixed_loss_denominator",
            "grpo/dynamic/selected_raw_row_count",
            "grpo/dynamic/selected_survivor_row_count",
            "grpo/dynamic/selected_masked_row_count",
            "train/overlong_filter/raw_selected_row_count",
            "train/overlong_filter/survivor_row_count",
            "train/overlong_filter/masked_row_count",
            "train/overlong_filter/group_count",
            "train/overlong_filter/min_survivors_per_group",
            "train/overlong_filter/max_survivors_per_group",
            "grpo/rnd/updates_this_step", "grpo/rnd/update_rows",
            "grpo/rnd/selected_batch_update_count",
            "grpo/dynamic/kept_group_count",
            "grpo/dynamic/fallback_group_count",
            "train/rollout_staleness_updates",
            "train/rollout_staleness_known", "train/on_policy_rollout",
            "train/updates_per_rollout", "train/policy_kl",
            "train/clip_fraction",
        }
        error_rate = float(metric.get(
            "grpo/dynamic/generation_error_rate", math.inf))
        error_rows_raw = metric.get(
            "grpo/dynamic/generation_error_row_count", math.nan)
        requested_rows_raw = metric.get(
            "grpo/dynamic/generation_requested_row_count", math.nan)
        error_rows = float(error_rows_raw)
        requested_rows = float(requested_rows_raw)
        terminal_transport = transport_invalid_step == step
        if not math.isfinite(error_rate) or (
                error_rate <= 0.05 if terminal_transport
                else error_rate > 0.05):
            raise RuntimeError(
                f"C10 step {step} has invalid generation transport")
        if not grpo_gpu.c10_integral_numeric(error_rows_raw) or not \
                grpo_gpu.c10_integral_numeric(
                    requested_rows_raw, minimum=64, multiple_of=16) or \
                error_rows > requested_rows or not math.isclose(
                    error_rate, error_rows / requested_rows,
                    rel_tol=1e-9, abs_tol=1e-9):
            raise RuntimeError(
                f"C10 step {step} generation accounting does not replay")
        candidate_rows = float(metric.get(
            "grpo/rnd/candidate_rows_scored", math.nan))
        if not math.isfinite(candidate_rows) or candidate_rows < 64 or \
                candidate_rows % 16 != 0:
            raise RuntimeError(
                f"C10 step {step} has invalid candidate RND geometry")
        for field in (
                "train/grpo_loss", "train/grad_norm",
                "train/selected_token_logprob_mean",
                "grpo/dynamic/rnd_selected_batch_loss"):
            value = float(metric.get(field, math.nan))
            if not math.isfinite(value):
                raise RuntimeError(
                    f"C10 step {step} has non-finite {field}")
        for field, expected in expected_metrics.items():
            if field in exact_metric_fields:
                if not grpo_gpu.c10_exact_numeric(
                        metric.get(field), expected):
                    raise RuntimeError(
                        f"C10 metric {field}={metric.get(field)!r}; "
                        f"expected exact {expected!r}")
            else:
                _metric_close(metric.get(field), expected, field=field)
        total_survivors += survivors
        total_masked += masked
        min_survivors = min(
            min_survivors, int(advantage_stats["min_survivors_per_group"]))
        max_survivors = max(
            max_survivors, int(advantage_stats["max_survivors_per_group"]))
        max_advantage_sum = max(
            max_advantage_sum,
            float(advantage_stats["max_abs_survivor_advantage_sum"]))
        point = [
            float(metric["quality/clipped_rate"]),
            float(metric["quality/valid_answer_rate"]),
        ]
        health_points.append(point)
        if step >= 10:
            recent = health_points[-5:]
            bad_window = (
                sum(item[0] for item in recent) / 5.0 >= 0.50
                and sum(item[1] for item in recent) / 5.0 <= 0.65)
            consecutive_bad_windows = (
                consecutive_bad_windows + 1 if bad_window else 0)
        per_step_replay.append({
            "step": step,
            "groups": group_summaries,
            "survivors": survivors,
            "masked": masked,
            "rows_backward": rows_backward,
            "max_abs_survivor_advantage_sum": float(
                advantage_stats["max_abs_survivor_advantage_sum"]),
        })

    return {
        "raw_selected_rows": expected_completed_steps * 64,
        "survivor_rows": total_survivors,
        "masked_rows": total_masked,
        "group_count": expected_completed_steps * 4,
        "minimum_survivors_per_group": (
            min_survivors if expected_completed_steps else 0),
        "maximum_survivors_per_group": max_survivors,
        "max_abs_survivor_advantage_sum": max_advantage_sum,
        "per_step_sha256": common.canonical_json_sha256(per_step_replay),
        "rolling_health": {
            "recent_clip_valid": health_points[-8:],
            "consecutive_bad_windows": consecutive_bad_windows,
            "completed_step": expected_completed_steps,
        },
    }


_C10_STEP_HEALTH_FAILURE_KINDS = {
    "learner_health_nonfinite",
    "rnd_update_contract_failure",
    "incorrect_novelty_credit_failure",
    "novelty_normalization_failure",
}


def _repair_c10_novelty_derived_evidence(
    step_rows: List[Dict[str, Any]], metric: Dict[str, Any], *,
    repair_novelty_indices: Optional[set[int]] = None,
    repair_normalization_rows: bool = False,
    repair_normalization_metrics: bool = False,
) -> None:
    """Repair only a claimed novelty violation and its derived aggregates."""
    raw = np.asarray([
        float(row["c10_rnd_raw_novelty_score"])
        for row in step_rows], dtype=np.float32)
    center = float(np.mean(raw, dtype=np.float64))
    scale = float(np.std(raw, dtype=np.float64))
    if not np.all(np.isfinite(raw)) or \
            scale <= grpo_gpu.C9_NORMALIZATION_EPSILON:
        raise RuntimeError("C10 terminal failure has invalid raw RND evidence")
    normalized = ((raw.astype(np.float64) - center) / scale).astype(
        np.float32)
    repair_indices = set(repair_novelty_indices or set())
    affected = set(range(64)) if repair_normalization_rows else repair_indices
    if repair_normalization_rows:
        # Incorrect-row credit is checked earlier than normalization in the
        # live gate, so it is an unrelated violation here, never something a
        # normalization failure may erase.
        if any(
                not bool(row["is_correct"]) and not math.isclose(
                    float(row.get("novelty_reward", math.nan)), 0.0,
                    abs_tol=1e-7)
                for row in step_rows):
            raise RuntimeError(
                "C10 normalization failure also has incorrect novelty credit")
    final = []
    for index, row in enumerate(step_rows):
        if repair_normalization_rows:
            row["c10_normalized_novelty_pre_gate"] = float(normalized[index])
        pre_gate = float(row["c10_normalized_novelty_pre_gate"])
        canonical_novelty = pre_gate if bool(row["is_correct"]) else 0.0
        if index in affected:
            row["novelty_reward"] = canonical_novelty
        novelty = float(row["novelty_reward"])
        reward = (
            (1.0 if bool(row["is_correct"]) else -1.0)
            + 0.5 * novelty
            + common.soft_overlong_penalty(
                int(row["completion_token_length"]), 13107, 3277)
        )
        if index in affected:
            row["blended_reward"] = reward
        final.append(reward)
    final_rewards = np.asarray(final, dtype=np.float32)
    survivor_mask = np.asarray(
        [not bool(row["clipped"]) for row in step_rows], dtype=bool)
    advantages, stats = grpo_gpu.group_mean_advantages_unmasked(
        final_rewards, survivor_mask, 16)
    survivors = int(survivor_mask.sum())
    rows_backward = sum(
        bool(survivor_mask[index])
        and int(step_rows[index]["completion_token_length_retokenized"]) > 0
        and abs(float(advantages[index])) >= 1e-12
        for index in range(64)
    )
    derived = {
        "train/reward_total_mean": float(final_rewards.mean()),
        "train/reward_novelty_mean": float(np.mean([
            float(row["novelty_reward"]) for row in step_rows])),
        "grpo/reward_std": float(final_rewards.std()),
        "train/rows_backward": rows_backward,
        "train/zero_advantage_survivor_rows": survivors - rows_backward,
        "train/overlong_filter/min_survivors_per_group":
            stats["min_survivors_per_group"],
        "train/overlong_filter/max_survivors_per_group":
            stats["max_survivors_per_group"],
        "train/overlong_filter/mean_survivors_per_group":
            stats["mean_survivors_per_group"],
        "train/overlong_filter/max_abs_survivor_advantage_sum":
            stats["max_abs_survivor_advantage_sum"],
    }
    metric.update(derived)
    if repair_normalization_metrics:
        metric.update({
            "grpo/reward/novelty_normalized_selected_mean": float(
                np.mean(normalized, dtype=np.float64)),
            "grpo/reward/novelty_normalized_selected_population_std": float(
                np.std(normalized, dtype=np.float64)),
        })


def _validate_c10_failure_state_reward_derivations(
    step_rows: List[Dict[str, Any]], metric: Dict[str, Any],
) -> None:
    """Prove every field a targeted novelty repair would overwrite."""
    final = []
    for row in step_rows:
        novelty = float(row["novelty_reward"])
        reward = (
            (1.0 if bool(row["is_correct"]) else -1.0)
            + 0.5 * novelty
            + common.soft_overlong_penalty(
                int(row["completion_token_length"]), 13107, 3277))
        _metric_close(
            row.get("blended_reward"), reward,
            field="terminal failure-state blended reward")
        final.append(reward)
    final_rewards = np.asarray(final, dtype=np.float32)
    survivor_mask = np.asarray(
        [not bool(row["clipped"]) for row in step_rows], dtype=bool)
    advantages, stats = grpo_gpu.group_mean_advantages_unmasked(
        final_rewards, survivor_mask, 16)
    survivors = int(survivor_mask.sum())
    rows_backward = sum(
        bool(survivor_mask[index])
        and int(step_rows[index]["completion_token_length_retokenized"]) > 0
        and abs(float(advantages[index])) >= 1e-12
        for index in range(64))
    expected = {
        "train/reward_total_mean": float(final_rewards.mean()),
        "train/reward_novelty_mean": float(np.mean([
            float(row["novelty_reward"]) for row in step_rows])),
        "grpo/reward_std": float(final_rewards.std()),
        "train/rows_backward": rows_backward,
        "train/zero_advantage_survivor_rows": survivors - rows_backward,
        "train/overlong_filter/min_survivors_per_group":
            stats["min_survivors_per_group"],
        "train/overlong_filter/max_survivors_per_group":
            stats["max_survivors_per_group"],
        "train/overlong_filter/mean_survivors_per_group":
            stats["mean_survivors_per_group"],
        "train/overlong_filter/max_abs_survivor_advantage_sum":
            stats["max_abs_survivor_advantage_sum"],
    }
    raw = np.asarray([
        float(row["c10_rnd_raw_novelty_score"])
        for row in step_rows], dtype=np.float32)
    center = float(np.mean(raw, dtype=np.float64))
    scale = float(np.std(raw, dtype=np.float64))
    expected.update({
        "grpo/reward/novelty_normalization_center": center,
        "grpo/reward/novelty_normalization_scale": scale,
        "grpo/reward/novelty_normalization_effective_scale": scale,
    })
    for field, wanted in expected.items():
        _metric_close(metric.get(field), wanted, field=field)


def _c10_canonical_novelty_metric_values(
    step_rows: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Compute canonical values for individually claimed nonfinite metrics."""
    raw = np.asarray([
        float(row["c10_rnd_raw_novelty_score"])
        for row in step_rows], dtype=np.float32)
    center = float(np.mean(raw, dtype=np.float64))
    scale = float(np.std(raw, dtype=np.float64))
    normalized = ((raw.astype(np.float64) - center) / scale).astype(
        np.float32)
    novelty = np.asarray([
        float(normalized[index]) if bool(row["is_correct"]) else 0.0
        for index, row in enumerate(step_rows)], dtype=np.float32)
    final = np.asarray([
        (1.0 if bool(row["is_correct"]) else -1.0)
        + 0.5 * float(novelty[index])
        + common.soft_overlong_penalty(
            int(row["completion_token_length"]), 13107, 3277)
        for index, row in enumerate(step_rows)], dtype=np.float32)
    return {
        "train/reward_total_mean": float(final.mean()),
        "train/reward_correctness_mean": float(np.mean([
            1.0 if bool(row["is_correct"]) else -1.0
            for row in step_rows])),
        "train/reward_novelty_mean": float(novelty.mean()),
        "grpo/reward/novelty_normalization_center": center,
        "grpo/reward/novelty_normalization_scale": scale,
        "grpo/reward/novelty_normalization_effective_scale": scale,
        "grpo/reward/novelty_normalized_selected_mean": float(
            np.mean(normalized, dtype=np.float64)),
        "grpo/reward/novelty_normalized_selected_population_std": float(
            np.std(normalized, dtype=np.float64)),
    }


def _replay_c10_terminal_health_failure(
    rows: List[Dict[str, Any]], metrics_rows: List[Dict[str, Any]], *,
    completed_step: int, failure_kind: str, observed: Dict[str, Any],
) -> Dict[str, Any]:
    """Bind one post-update health violation and replay everything else.

    The violating step is persisted before the trainer stops.  We first bind
    the claimed violation to its raw terminal rows/metric, then repair only
    that invariant (and fields causally downstream of it) in memory so the
    ordinary strict replay can validate every unrelated invariant.
    """
    if failure_kind not in _C10_STEP_HEALTH_FAILURE_KINDS or \
            type(completed_step) is not int or completed_step < 1 or \
            not isinstance(observed, dict):
        raise RuntimeError("C10 terminal health-failure request is invalid")
    if completed_step > 1:
        _replay_c10_survivor_contract(
            [row for row in rows if int(row.get("step", -1)) < completed_step],
            [metric for metric in metrics_rows if int(metric.get(
                "global_step", -1)) < completed_step],
            expected_completed_steps=completed_step - 1)
    repaired_rows = copy.deepcopy(rows)
    repaired_metrics = copy.deepcopy(metrics_rows)
    step_rows = [
        row for row in repaired_rows
        if int(row.get("step", -1)) == completed_step]
    metric_matches = [
        metric for metric in repaired_metrics
        if int(metric.get("global_step", -1)) == completed_step]
    if len(step_rows) != 64 or len(metric_matches) != 1:
        raise RuntimeError("C10 terminal health step geometry is invalid")
    metric = metric_matches[0]

    if failure_kind == "selected_group_geometry_failure":
        actual = {
            "kept_group_count": metric.get(
                "grpo/dynamic/kept_group_count"),
            "fallback_group_count": metric.get(
                "grpo/dynamic/fallback_group_count"),
        }
        if not _c10_observed_equal(actual, observed) or (
                int(actual["kept_group_count"]) == 4
                and int(actual["fallback_group_count"]) == 0):
            raise RuntimeError("C10 selected-group failure does not replay")
        metric["grpo/dynamic/kept_group_count"] = 4
        metric["grpo/dynamic/fallback_group_count"] = 0

    elif failure_kind == "survivor_filter_contract_failure":
        if "selected_raw_row_count" in observed:
            actual = metric.get("grpo/dynamic/selected_raw_row_count")
            if not _c10_observed_equal(
                    observed, {"selected_raw_row_count": actual}) or \
                    int(actual) == 64:
                raise RuntimeError("C10 raw-row failure does not replay")
            metric["grpo/dynamic/selected_raw_row_count"] = 64
        else:
            try:
                start = int(observed["group_start"])
                group = step_rows[start:start + 16]
                survivors = sum(not bool(row["clipped"]) for row in group)
                populations = sorted(set(str(row.get(
                    "dynamic_sampling_eligibility_population"))
                    for row in group))
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError("C10 survivor failure is malformed") from exc
            actual = {
                "group_start": start,
                "survivors": survivors,
                "populations": populations,
            }
            if not _c10_observed_equal(actual, observed) or (
                    survivors >= 2 and populations == [
                    "unmasked_survivors"]):
                raise RuntimeError("C10 survivor failure does not replay")
            if start not in range(0, 64, 16) or len(group) != 16 or \
                    survivors < 2:
                raise RuntimeError(
                    "C10 completed an update with an ineligible survivor "
                    "group; terminal-step replay is unsafe")
            replay_rows = [{
                "is_correct": bool(row["is_correct"]),
                "clipped": bool(row["clipped"]),
                "blended_reward": float(row[
                    "dynamic_sampling_eligibility_blended_reward"]),
                "novelty_used": float(row[
                    "dynamic_sampling_eligibility_novelty_used"]),
            } for row in group]
            reason, eligible, facts = grpo_gpu.group_eligibility_for_rows(
                replay_rows, lambda_novelty=0.5, validity_mode="c8",
                survivor_only=True)
            if not eligible:
                raise RuntimeError("C10 survivor repair group is ineligible")
            for row in group:
                row.update({
                    "dynamic_sampling_reason": reason,
                    "dynamic_sampling_group_selected": True,
                    "dynamic_sampling_eligibility_population":
                        "unmasked_survivors",
                    "dynamic_sampling_raw_group_size": facts["raw_group_size"],
                    "dynamic_sampling_survivor_count": facts["survivor_count"],
                    "dynamic_sampling_masked_count": facts["masked_count"],
                    "dynamic_sampling_raw_correct_count":
                        facts["raw_correct_count"],
                    "dynamic_sampling_survivor_correct_count":
                        facts["survivor_correct_count"],
                    "dynamic_sampling_eligibility_population_size":
                        facts["eligibility_population_size"],
                    "dynamic_sampling_policy_loss_eligible":
                        not bool(row["clipped"]),
                    "dynamic_sampling_reward_std":
                        facts["eligibility_reward_std"],
                    "dynamic_sampling_novelty_contribution_std":
                        facts["eligibility_novelty_contribution_std"],
                })

    elif failure_kind == "learner_health_nonfinite":
        finite_keys = (
            "train/grpo_loss", "train/reward_total_mean",
            "train/reward_correctness_mean", "train/reward_novelty_mean",
            "train/grad_norm", "train/selected_token_logprob_mean",
            "grpo/dynamic/rnd_selected_batch_loss",
            "grpo/reward/novelty_normalization_center",
            "grpo/reward/novelty_normalization_scale",
            "grpo/reward/novelty_normalization_effective_scale",
            "grpo/reward/novelty_normalized_selected_mean",
            "grpo/reward/novelty_normalized_selected_population_std",
        )
        nonfinite = [
            key for key in finite_keys
            if not math.isfinite(float(metric.get(key, math.nan)))]
        if not _c10_observed_equal(
                observed, {"nonfinite_fields": nonfinite}) or not nonfinite:
            raise RuntimeError("C10 nonfinite failure does not replay")
        canonical = _c10_canonical_novelty_metric_values(step_rows)
        for key in nonfinite:
            metric[key] = canonical.get(key, 0.0)

    elif failure_kind == "rnd_update_contract_failure":
        actual = {
            "rnd_updates_this_step": metric.get("grpo/rnd/updates_this_step"),
            "rnd_update_rows": metric.get("grpo/rnd/update_rows"),
            "rnd_selected_batch_update_count": metric.get(
                "grpo/rnd/selected_batch_update_count"),
        }
        if not _c10_observed_equal(actual, observed) or (
                grpo_gpu.c10_exact_numeric(
                    actual["rnd_updates_this_step"], 1)
                and grpo_gpu.c10_exact_numeric(
                    actual["rnd_update_rows"], 64)
                and grpo_gpu.c10_exact_numeric(
                    actual["rnd_selected_batch_update_count"],
                    completed_step)):
            raise RuntimeError("C10 RND-update failure does not replay")
        metric.update({
            "grpo/rnd/updates_this_step": 1,
            "grpo/rnd/update_rows": 64,
            "grpo/rnd/selected_batch_update_count": completed_step,
        })

    elif failure_kind == "incorrect_novelty_credit_failure":
        failing = [
            {"row_index": index,
             "novelty_reward": row.get("novelty_reward")}
            for index, row in enumerate(step_rows)
            if not bool(row.get("is_correct")) and not math.isclose(
                float(row.get("novelty_reward", math.nan)), 0.0,
                abs_tol=1e-7)]
        if not failing or not _c10_observed_equal(
                observed, {"violations": failing}):
            raise RuntimeError("C10 incorrect-novelty failure does not replay")
        _validate_c10_failure_state_reward_derivations(step_rows, metric)
        _repair_c10_novelty_derived_evidence(
            step_rows, metric, repair_novelty_indices={
                int(item["row_index"]) for item in failing})

    else:
        mean = metric.get(
            "grpo/reward/novelty_normalized_selected_mean")
        std = metric.get(
            "grpo/reward/novelty_normalized_selected_population_std")
        serialized = [
            float(row.get("c10_normalized_novelty_pre_gate", math.nan))
            for row in step_rows]
        serialized_mean = float(np.mean(serialized, dtype=np.float64))
        serialized_std = float(np.std(serialized, dtype=np.float64))
        expected_observed = {
            "mean": mean,
            "population_std": std,
            "normalized_pre_gate_sha256":
                common.canonical_json_sha256(serialized),
        }
        raw = np.asarray([
            float(row["c10_rnd_raw_novelty_score"])
            for row in step_rows], dtype=np.float32)
        raw_center = float(np.mean(raw, dtype=np.float64))
        raw_scale = float(np.std(raw, dtype=np.float64))
        canonical = ((raw.astype(np.float64) - raw_center) / raw_scale).astype(
            np.float32)
        rows_canonical = bool(np.allclose(
            np.asarray(serialized, dtype=np.float32), canonical,
            rtol=1e-7, atol=1e-7))
        metric_matches_rows = (
            math.isclose(float(mean), serialized_mean, abs_tol=1e-7)
            and math.isclose(float(std), serialized_std, abs_tol=1e-7))
        if not _c10_observed_equal(observed, expected_observed) or (
                math.isclose(float(mean), 0.0, abs_tol=1e-5)
                and math.isclose(float(std), 1.0, abs_tol=1e-5)) or \
                (not rows_canonical and not metric_matches_rows):
            raise RuntimeError("C10 normalization failure does not replay")
        # Before repairing the normalization vector, prove every serialized
        # reward is exactly its causal consequence.  This prevents a claimed
        # aggregate-normalization failure from laundering unrelated row
        # corruption.
        for row, pre_gate in zip(step_rows, serialized):
            expected_novelty = pre_gate if bool(row["is_correct"]) else 0.0
            expected_reward = (
                (1.0 if bool(row["is_correct"]) else -1.0)
                + 0.5 * expected_novelty
                + common.soft_overlong_penalty(
                    int(row["completion_token_length"]), 13107, 3277))
            _metric_close(
                row.get("novelty_reward"), expected_novelty,
                field="terminal normalization novelty consequence")
            _metric_close(
                row.get("blended_reward"), expected_reward,
                field="terminal normalization reward consequence")
        _validate_c10_failure_state_reward_derivations(step_rows, metric)
        if rows_canonical:
            # Metric-only reporting failure: raw rows and all downstream
            # rewards are already canonical, so repair only the two aggregates
            # that the live gate declared invalid.
            metric.update({
                "grpo/reward/novelty_normalized_selected_mean": float(
                    np.mean(canonical, dtype=np.float64)),
                "grpo/reward/novelty_normalized_selected_population_std":
                    float(np.std(canonical, dtype=np.float64)),
            })
        else:
            _repair_c10_novelty_derived_evidence(
                step_rows, metric, repair_normalization_rows=True,
                repair_normalization_metrics=True)

    return _replay_c10_survivor_contract(
        repaired_rows, repaired_metrics,
        expected_completed_steps=completed_step)


def _validate_c10_canary_scout_endpoint(
    args: argparse.Namespace, scout_stage: Dict[str, Any], *, scout_init: str,
    round_idx: int, lam: float,
) -> Dict[str, Any]:
    """Replay the exact C10 step-50 endpoint before its fixed-panel gate."""
    if str(args.c10_execution_profile) != "canary" or \
            str(args.c9_execution_profile or "") or \
            str(args.stabilization_cell or ""):
        raise RuntimeError("C10 endpoint validator used outside isolated C10")
    final_dir = str(scout_stage.get("final", ""))
    state = _load_json_object(
        os.path.join(final_dir, "trainer_state.json"),
        "C10 canary Scout trainer_state")
    runtime = _c9_scout_runtime_contract(
        args, scout_init=scout_init, round_idx=round_idx, lam=lam)
    c10_evidence.validate_runtime_contract(runtime)
    code = _c9_code_identity(args)
    if state.get("phase") != "scout" or state.get("checkpoint_kind") != \
            "deployable_model" or state.get("step") != 50 or \
            state.get("runtime_contract") != runtime or \
            state.get("code_artifact_binding") != code:
        raise RuntimeError("C10 canary is not the exact code-bound endpoint")
    health = state.get("c10_canary_health_state")
    if not isinstance(health, dict) or health.get("completed_step") != 50 or \
            int(health.get("consecutive_bad_windows", 3)) >= 3:
        raise RuntimeError("C10 canary lacks passing resume-stable health")
    recent = health.get("recent_clip_valid")
    if not isinstance(recent, list) or len(recent) != 8 or any(
            not isinstance(item, list) or len(item) != 2 or
            not all(math.isfinite(float(value)) for value in item)
            for item in recent):
        raise RuntimeError("C10 rolling-health history is invalid")
    if state.get("rnd_selected_batch_update_count") != 50:
        raise RuntimeError("C10 RND update count mismatch")
    manifest = state.get("checkpoint_file_manifest")
    common.verify_directory_file_manifest(
        final_dir, manifest, exclude_relative_paths=("trainer_state.json",))
    paths = list(scout_stage.get("traj_paths", []))
    if len(paths) != 1:
        raise RuntimeError("C10 canary must have one trajectory file")
    rows = _read_jsonl_objects(paths[0], "C10 canary trajectories")
    per_step: Dict[int, int] = {}
    for row_number, row in enumerate(rows, 1):
        step = row.get("step")
        if type(step) is not int or not 1 <= step <= 50:
            raise RuntimeError(
                f"C10 trajectory row {row_number} step mismatch")
        per_step[step] = per_step.get(step, 0) + 1
        if row.get("scientific_variant") != \
                c10_evidence.C10_SCIENTIFIC_VARIANT or \
                row.get("c10_execution_profile") != "canary" or \
                row.get("c9_execution_profile") not in (None, "") or \
                row.get("stabilization_cell") not in (None, ""):
            raise RuntimeError("C10 trajectory identity mismatch")
        clipped = bool(row.get("clipped", False))
        if row.get("dynamic_sampling_eligibility_population") != \
                "unmasked_survivors" or int(row.get(
                    "dynamic_sampling_survivor_count", -1)) < 2 or \
                int(row.get("dynamic_sampling_raw_group_size", -1)) != 16 or \
                int(row.get("dynamic_sampling_survivor_count", -1)) + int(
                    row.get("dynamic_sampling_masked_count", -1)) != 16 or \
                row.get("dynamic_sampling_group_selected") is not True or \
                row.get("dynamic_sampling_policy_loss_eligible") is not \
                (not clipped):
            raise RuntimeError("C10 trajectory survivor eligibility mismatch")
        if row.get("expdis_stabilization_mode") != \
                c9_evidence.C9_RND_LIFECYCLE or \
                row.get("rnd_update_scope") != \
                "final_selected_learner_rows" or \
                row.get("novelty_normalization") != \
                "selected_batch_zscore":
            raise RuntimeError("C10 held-fixed RND contract mismatch")
        if not bool(row.get("is_correct", False)) and not math.isclose(
                float(row.get("novelty_reward", math.nan)), 0.0,
                abs_tol=1e-7):
            raise RuntimeError("C10 incorrect row has nonzero novelty credit")
    if len(rows) != 50 * c9_evidence.SCOUT_ROWS_PER_STEP or \
            set(per_step) != set(range(1, 51)) or \
            set(per_step.values()) != {c9_evidence.SCOUT_ROWS_PER_STEP}:
        raise RuntimeError("C10 trajectories are not exactly 50 x 64")
    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    metrics_rows = [
        row for row in _read_jsonl_objects(
            metrics_path, "C10 canary metrics")
        if row.get("phase") == common.PARITY_PHASES["scout"]
        and type(row.get("grpo/step")) is int
        and "train/grpo_loss" in row
    ]
    replay = _replay_c10_survivor_contract(rows, metrics_rows)
    expected_health = replay.pop("rolling_health")
    if health != expected_health:
        raise RuntimeError("C10 rolling-health state does not replay")
    payload = {
        "c10_execution_profile": "canary",
        "scientific_variant": c10_evidence.C10_SCIENTIFIC_VARIANT,
        "round": 1,
        "model_step": 50,
        "preregistration_file": c10_evidence.C10_PREREGISTRATION_FILE,
        "preregistration_sha256":
            c10_evidence.C10_PREREGISTRATION_SHA256,
        "runtime_contract": runtime,
        "runtime_contract_sha256": common.canonical_json_sha256(runtime),
        "code_artifact": code,
        "model_file_manifest": manifest,
        "trajectory": {
            "file_sha256": common.file_sha256(paths[0]),
            "canonical_rows_sha256": common.canonical_json_sha256(rows),
            "row_count": len(rows),
            "step_count": 50,
            "rows_per_step": c9_evidence.SCOUT_ROWS_PER_STEP,
            "group_count": 50 * int(args.prompts_per_step),
        },
        "metrics": {
            "file_sha256": common.file_sha256(metrics_path),
            "canonical_rows_sha256":
                common.canonical_json_sha256(metrics_rows),
            "row_count": len(metrics_rows),
        },
        "survivor_replay": replay,
        "rolling_health": health,
    }
    evidence = c10_evidence.signed_endpoint_replay(payload)
    c10_evidence.validate_endpoint_replay(
        evidence, expected_runtime_contract=runtime,
        expected_model_manifest=manifest, expected_code_artifact=code)
    return evidence


def _write_jsonl_atomic(path: str, rows: List[Dict[str, Any]]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _c10_completed_behavior_modes(
    rows: List[Dict[str, Any]], examples: List[Any],
) -> tuple[str, ...]:
    """Validate the persisted fixed-panel draw as complete mode prefixes.

    C10 writes one full prompt mode atomically.  A process restart may
    continue only after those exact persisted modes; it can never overwrite
    and resample them.
    """
    per_mode = len(examples) * c9_evidence.GATE_SAMPLES_PER_PROMPT
    if not rows:
        return ()
    if len(rows) % per_mode or len(rows) > \
            per_mode * len(sft_gate_gpu.PROMPT_MODES):
        raise RuntimeError("C10 behavior rollout journal has partial geometry")
    completed = len(rows) // per_mode
    expected_modes = sft_gate_gpu.PROMPT_MODES[:completed]
    expected_slots = []
    for mode in expected_modes:
        for prompt_idx, example in enumerate(examples):
            for sample_idx in range(c9_evidence.GATE_SAMPLES_PER_PROMPT):
                expected_slots.append((
                    mode, str(example.problem_id), sample_idx,
                    c9_evidence.GATE_GENERATION_SEED
                    + prompt_idx * 100003 + sample_idx,
                ))
    observed_slots = []
    for row in rows:
        try:
            recomputed = sft_gate_gpu.recompute_health_record(
                row,
                max_completion_tokens=
                    c9_evidence.GATE_MAX_COMPLETION_TOKENS,
            )
            observed_slots.append((
                recomputed["prompt_mode"], recomputed["problem_id"],
                int(recomputed["sample_idx"]),
                int(recomputed["sample_seed"]),
            ))
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "C10 behavior rollout journal is malformed") from exc
    if observed_slots != expected_slots:
        raise RuntimeError(
            "C10 behavior rollout journal is not the fixed-panel prefix")
    return tuple(expected_modes)


def _c9_fixed_gate_examples(tokenizer) -> tuple[List[Any], Dict[str, int]]:
    """The preregistered fixed hash-ranked non-benchmark DAPO panel.

    Unlike the generic SFT checkpoint-selection gate, C9's panel identity is
    fixed before filtering and therefore cannot depend on the accepted set.
    Correctness and model outputs do not participate in selection.  The panel
    is a behavior diagnostic, not a Scout holdout: its prompts come from the
    same DAPO snapshot used for Scout training.
    """
    from tmx_jax.data import _load_raw

    exclusions = sft_gate_gpu.AcceptedExclusions(
        problem_ids=frozenset(), prompt_hashes=frozenset(), row_count=0)
    return sft_gate_gpu.select_heldout_examples(
        _load_raw(), tokenizer, exclusions,
        num_prompts=c9_evidence.GATE_NUM_PROMPTS,
        selection_seed=c9_evidence.GATE_SELECTION_SEED,
    )


def _c9_gate_thresholds() -> Dict[str, float]:
    resolved = {
        "min_termination_rate": sft_gate_gpu.DEFAULT_MIN_TERMINATION_RATE,
        "max_clipped_rate": sft_gate_gpu.DEFAULT_MAX_CLIPPED_RATE,
        "min_valid_boxed_answer_rate": sft_gate_gpu.DEFAULT_MIN_VALID_BOXED_RATE,
        "max_median_completion_tokens": (
            c9_evidence.GATE_MAX_COMPLETION_TOKENS
            * sft_gate_gpu.DEFAULT_MAX_MEDIAN_TOKEN_FRACTION),
        "max_repetition_rate": sft_gate_gpu.DEFAULT_MAX_REPETITION_RATE,
        "max_unclosed_think_rate": sft_gate_gpu.DEFAULT_MAX_UNCLOSED_THINK_RATE,
        "max_generation_error_rate": sft_gate_gpu.DEFAULT_MAX_GENERATION_ERROR_RATE,
    }
    if resolved != c9_evidence.GATE_THRESHOLDS:
        raise RuntimeError("C9 behavior-gate threshold constants drifted")
    return dict(c9_evidence.GATE_THRESHOLDS)


def _c9_model_manifest(model_dir: str, *, stage: str) -> Dict[str, Any]:
    if stage in ("scout_step50", "scout_final"):
        state = _load_json_object(
            os.path.join(model_dir, "trainer_state.json"),
            "C9 Scout trainer_state")
        manifest = state.get("checkpoint_file_manifest")
        common.verify_directory_file_manifest(
            model_dir, manifest, exclude_relative_paths=("trainer_state.json",))
        return manifest
    provenance = _load_json_object(
        os.path.join(model_dir, "sft_provenance.json"),
        "C9 SFT provenance")
    manifest = provenance.get("checkpoint_file_manifest")
    common.verify_directory_file_manifest(
        model_dir, manifest, exclude_relative_paths=("sft_provenance.json",))
    return manifest


def _run_c9_behavior_gate(
    args: argparse.Namespace, pool: VLLMPool, tokenizer, *, stage: str,
    model_dir: str, accepted_path: Optional[str], round_dir: str,
    expected_selection_sha256: Optional[str] = None,
    expected_endpoint_replay_sha256: Optional[str] = None,
    multi_identity: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], str]:
    c10_mode = str(getattr(args, "c10_execution_profile", "") or "") \
        == "canary"
    if c10_mode and multi_identity is not None:
        raise RuntimeError("C10 behavior gates cannot use C9 multi identity")
    if c10_mode and (not isinstance(expected_endpoint_replay_sha256, str)
                     or len(expected_endpoint_replay_sha256) != 64):
        raise RuntimeError("C10 behavior gate lacks its endpoint replay binding")
    if not pool.managed:
        raise RuntimeError("C9 behavior gates require a managed reloadable pool")
    if os.path.realpath(str(pool.model_dir)) != os.path.realpath(model_dir):
        pool.reload(model_dir)
    examples, selection_stats = _c9_fixed_gate_examples(tokenizer)
    variants = [
        sft_gate_gpu.render_prompt_variants(example, tokenizer)
        for example in examples
    ]
    gate_dir = os.path.join(
        round_dir,
        "c10_behavior_gates" if c10_mode else "c9_behavior_gates",
        stage,
    )
    os.makedirs(gate_dir, exist_ok=True)
    selection_path = os.path.join(gate_dir, "heldout_selection.jsonl")
    rollout_path = os.path.join(gate_dir, "rollouts.jsonl")
    aggregate_path = os.path.join(gate_dir, "aggregate_metrics.json")
    evidence_path = os.path.join(gate_dir, "evidence.json")
    selection_rows = [{
        "problem_id": example.problem_id,
        "selection_digest": example.selection_digest,
        "training_prompt_sha256": sft_gate_gpu._sha256(example.training_prompt),
    } for example in examples]
    if c10_mode and os.path.isfile(selection_path):
        persisted_selection = _read_jsonl_objects(
            selection_path, "C10 held-out selection")
        if persisted_selection != selection_rows:
            raise RuntimeError(
                "C10 held-out selection journal differs on resume")
    else:
        _write_jsonl_atomic(selection_path, selection_rows)
    selection_sha = common.canonical_json_sha256(selection_rows)
    selection_file_sha = common.file_sha256(selection_path)
    if c10_mode and (
            selection_sha != c10_evidence.GATE_HELDOUT_SELECTION_SHA256 or
            selection_file_sha !=
            c10_evidence.GATE_HELDOUT_SELECTION_FILE_SHA256):
        # C10 is one preregistered draw on the exact C5 panel.  Check both
        # semantic rows and serialized bytes before the first generation so a
        # drifted panel is never sampled or observed.
        raise RuntimeError(
            "C10 fixed behavior panel differs before generation")
    if expected_selection_sha256 is not None and selection_sha != \
            expected_selection_sha256:
        raise RuntimeError("C9 gates did not resolve to the same fixed panel")

    records: List[Dict[str, Any]] = (
        _read_jsonl_objects(rollout_path, "C10 behavior rollouts")
        if c10_mode and os.path.isfile(rollout_path) else [])
    completed_modes = (
        _c10_completed_behavior_modes(records, examples)
        if c10_mode else ())
    if c10_mode and os.path.isfile(aggregate_path) and len(
            completed_modes) != len(sft_gate_gpu.PROMPT_MODES):
        raise RuntimeError(
            "C10 aggregate exists before the raw paired draw completed")
    for mode in sft_gate_gpu.PROMPT_MODES:
        if mode in completed_modes:
            print(
                f"[c10] reusing persisted fixed-panel mode={mode}",
                flush=True,
            )
            continue
        grouped = pool.generate(
            [item[mode] for item in variants],
            c9_evidence.GATE_SAMPLES_PER_PROMPT,
            max_tokens=c9_evidence.GATE_MAX_COMPLETION_TOKENS,
            temperature=c9_evidence.GATE_TEMPERATURE,
            top_p=c9_evidence.GATE_TOP_P,
            top_k=c9_evidence.GATE_TOP_K,
            min_p=c9_evidence.GATE_MIN_P,
            seed_base=c9_evidence.GATE_GENERATION_SEED,
            timeout=3600.0,
            concurrency_per_server=int(args.gen_concurrency),
            enforce_consecutive_failure_limit=not c10_mode,
        )
        for prompt_idx, (example, completions) in enumerate(
                zip(examples, grouped, strict=True)):
            if len(completions) != c9_evidence.GATE_SAMPLES_PER_PROMPT:
                raise RuntimeError("C9 behavior gate returned an incomplete sample group")
            for sample_idx, completion in enumerate(completions):
                records.append(sft_gate_gpu.completion_record(
                    example=example, prompt_mode=mode, sample_idx=sample_idx,
                    sample_seed=(c9_evidence.GATE_GENERATION_SEED
                                 + prompt_idx * 100003 + sample_idx),
                    completion=completion, tokenizer=tokenizer,
                    max_completion_tokens=c9_evidence.GATE_MAX_COMPLETION_TOKENS,
                ))
        _write_jsonl_atomic(rollout_path, records)
    thresholds = _c9_gate_thresholds()
    metrics, gate = sft_gate_gpu.recompute_gate_from_rollouts(
        records, num_prompts=c9_evidence.GATE_NUM_PROMPTS,
        samples_per_prompt=c9_evidence.GATE_SAMPLES_PER_PROMPT,
        generation_seed=c9_evidence.GATE_GENERATION_SEED,
        max_completion_tokens=c9_evidence.GATE_MAX_COMPLETION_TOKENS,
        ordered_problem_ids=[example.problem_id for example in examples],
        thresholds=thresholds,
    )
    aggregate = {
        "metric_contract_version": sft_gate_gpu.METRIC_CONTRACT_VERSION,
        "gate_stage": stage,
        "selection": {
            "policy": "sha256(selection_seed, problem_id, problem)",
            "uses_model_outputs": False,
            "uses_correctness": False,
            "panel_role": c9_evidence.GATE_PANEL_ROLE,
            "held_out_from_scout_training":
                c9_evidence.GATE_PANEL_HELD_OUT_FROM_SCOUT_TRAINING,
            "heldout_selection_sha256": selection_sha,
            **selection_stats,
        },
        "sampling": {
            "num_prompts": c9_evidence.GATE_NUM_PROMPTS,
            "samples_per_prompt": c9_evidence.GATE_SAMPLES_PER_PROMPT,
            "selection_seed": c9_evidence.GATE_SELECTION_SEED,
            "generation_seed": c9_evidence.GATE_GENERATION_SEED,
            "temperature": c9_evidence.GATE_TEMPERATURE,
            "top_p": c9_evidence.GATE_TOP_P,
            "top_k": c9_evidence.GATE_TOP_K,
            "min_p": c9_evidence.GATE_MIN_P,
            "max_completion_tokens": c9_evidence.GATE_MAX_COMPLETION_TOKENS,
            "paired_seeds_across_prompt_modes": True,
        },
        "metrics_by_prompt_mode": metrics,
        "health_gate": gate,
        "correctness_policy": "diagnostic only; excluded from selection and gate",
    }
    if c10_mode and os.path.isfile(aggregate_path):
        persisted_aggregate = _load_json_object(
            aggregate_path, "C10 aggregate journal")
        if persisted_aggregate != aggregate:
            raise RuntimeError(
                "C10 persisted aggregate differs from raw-rollout replay")
    else:
        common.atomic_write_json(aggregate_path, aggregate)
    no_library_version = (
        "c10_no_sft_library_v1" if c10_mode else "c9_no_sft_library_v1")
    accepted_digest = (
        common.semantic_sft_jsonl_digest(accepted_path)
        if accepted_path else {
            "version": no_library_version, "row_count": 0,
            "sha256": common.canonical_json_sha256(
                {"version": no_library_version, "rows": []}),
        })
    model_manifest = _c9_model_manifest(model_dir, stage=stage)
    if stage in ("scout_step50", "scout_final"):
        model_metadata = _load_json_object(
            os.path.join(model_dir, "trainer_state.json"),
            "C9 gated Scout trainer_state")
        model_runtime = model_metadata.get("runtime_contract")
        model_step = int(model_metadata.get("step", -1))
    else:
        model_metadata = _load_json_object(
            os.path.join(model_dir, "sft_provenance.json"),
            "C9 gated SFT provenance")
        model_runtime = model_metadata.get("sft_runtime_contract")
        model_step = int(model_metadata.get("optimizer_steps", -1))
    evidence_payload = {
            "c9_execution_profile": str(args.c9_execution_profile),
            "stabilization_cell": str(args.stabilization_cell),
            "scientific_variant": str(args.scientific_variant),
            "gate_stage": stage,
            "selection_uses_correctness": False,
            "panel_role": c9_evidence.GATE_PANEL_ROLE,
            "panel_held_out_from_scout_training":
                c9_evidence.GATE_PANEL_HELD_OUT_FROM_SCOUT_TRAINING,
            "passed": bool(gate["passed"]),
            "sampling": aggregate["sampling"],
            "thresholds": thresholds,
            "heldout_selection_sha256": selection_sha,
            "heldout_selection_file_sha256": common.file_sha256(selection_path),
            "raw_rollout_count": len(records),
            "raw_rollouts_file_sha256": common.file_sha256(rollout_path),
            "aggregate_file_sha256": common.file_sha256(aggregate_path),
            "model_file_manifest": model_manifest,
            "model_runtime_contract": model_runtime,
            "model_step": model_step,
            "accepted_library_digest": accepted_digest,
            "accepted_file_sha256": (
                common.file_sha256(accepted_path) if accepted_path else None),
            "code_artifact": _c9_code_identity(args),
        }
    if c10_mode:
        evidence_payload.pop("c9_execution_profile")
        evidence_payload.pop("stabilization_cell")
        evidence_payload.update({
            "c10_execution_profile": "canary",
            "scientific_variant":
                c10_evidence.C10_SCIENTIFIC_VARIANT,
            "preregistration_file":
                c10_evidence.C10_PREREGISTRATION_FILE,
            "preregistration_sha256":
                c10_evidence.C10_PREREGISTRATION_SHA256,
            "model_runtime_contract_sha256":
                common.canonical_json_sha256(model_runtime),
            "endpoint_replay_evidence_sha256":
                expected_endpoint_replay_sha256,
            "gate_outcome": c10_evidence.classify_gate_outcome(
                metrics, gate),
        })
    if multi_identity is None:
        evidence = (
            c10_evidence.signed_payload(evidence_payload)
            if c10_mode else c9_evidence.signed_payload(
                c9_evidence.BEHAVIOR_GATE_EVIDENCE_VERSION,
                evidence_payload)
        )
    else:
        contract = multi_identity.get("contract")
        if not isinstance(contract, dict) or not \
                mrme_contract.is_c9_multi_explorer(contract):
            raise RuntimeError(
                "multi-explorer behavior gate lacks its stabilized contract")
        evidence_payload.update({
            "c9_execution_profile": "full",
            "stabilization_cell": "C5",
            "scientific_variant":
                mrme_contract.C9_MULTIEXPLORER_SCIENTIFIC_VARIANT,
            "contract_sha256": contract["contract_sha256"],
            "method": contract["method"],
            "stabilization_profile": contract["stabilization_profile"],
            "round_index": int(multi_identity["round_index"]),
            "scout_index": int(multi_identity["scout_index"]),
            "seed": int(multi_identity["seed"]),
            "code_artifact": contract["code_identity"],
        })
        if str(multi_identity.get("gate_role", "scout")) == "pooled_sft":
            evidence = \
                mrme_contract.sign_c9_multi_explorer_sft_gate_evidence(
                    evidence_payload)
        else:
            evidence = mrme_contract.sign_c9_multi_explorer_gate_evidence(
                evidence_payload)
    common.atomic_write_json(evidence_path, evidence)
    # A failed gate is durable negative evidence, but it can never authorize
    # the next training stage.
    if multi_identity is None:
        if c10_mode:
            c10_evidence.validate_behavior_gate_evidence(
                evidence, expected_model_manifest=model_manifest,
                expected_code_artifact=_c9_code_identity(args),
                expected_selection_sha256=expected_selection_sha256,
                expected_endpoint_replay_sha256=
                    expected_endpoint_replay_sha256,
                require_passed=None)
        else:
            c9_evidence.validate_behavior_gate_evidence(
                evidence, expected_stage=stage,
                expected_model_manifest=model_manifest,
                expected_accepted_digest=accepted_digest,
                expected_code_artifact=_c9_code_identity(args),
                expected_selection_sha256=expected_selection_sha256,
                expected_profile=str(args.c9_execution_profile),
                expected_cell=str(args.stabilization_cell))
    else:
        if str(multi_identity.get("gate_role", "scout")) == "pooled_sft":
            mrme_contract.validate_c9_multi_explorer_sft_gate_evidence(
                evidence, contract=multi_identity["contract"],
                round_index=int(multi_identity["round_index"]),
                expected_runtime_contract=model_runtime,
                expected_code_identity=multi_identity["contract"][
                    "code_identity"],
                expected_model_manifest=model_manifest,
                expected_accepted_digest=accepted_digest,
                expected_selection_sha256=expected_selection_sha256)
        else:
            mrme_contract.validate_c9_multi_explorer_gate_evidence(
                evidence, contract=multi_identity["contract"],
                round_index=int(multi_identity["round_index"]),
                scout_index=int(multi_identity["scout_index"]),
                expected_runtime_contract=model_runtime,
                expected_code_identity=multi_identity["contract"][
                    "code_identity"],
                expected_model_manifest=model_manifest)
    return evidence, evidence_path


def _validate_persisted_c9_behavior_gate(
    args: argparse.Namespace, *, evidence_path: str, stage: str,
    model_dir: str, accepted_path: Optional[str],
    expected_selection_sha256: Optional[str] = None,
    expected_endpoint_replay_sha256: Optional[str] = None,
    multi_identity: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    c10_mode = str(getattr(args, "c10_execution_profile", "") or "") \
        == "canary"
    evidence = _load_json_object(evidence_path, "persisted C9 behavior gate")
    gate_dir = os.path.dirname(evidence_path)
    selection_path = os.path.join(gate_dir, "heldout_selection.jsonl")
    rollout_path = os.path.join(gate_dir, "rollouts.jsonl")
    aggregate_path = os.path.join(gate_dir, "aggregate_metrics.json")
    if common.file_sha256(selection_path) != \
            evidence.get("heldout_selection_file_sha256") or \
            common.file_sha256(rollout_path) != \
            evidence.get("raw_rollouts_file_sha256") or \
            common.file_sha256(aggregate_path) != \
            evidence.get("aggregate_file_sha256"):
        raise RuntimeError("persisted C9 behavior-gate files drifted")
    selection = _read_jsonl_objects(
        selection_path, "C9 fixed diagnostic selection")
    if common.canonical_json_sha256(selection) != \
            evidence.get("heldout_selection_sha256"):
        raise RuntimeError("persisted C9 behavior-gate panel drifted")
    rows = _read_jsonl_objects(rollout_path, "C9 behavior-gate rollouts")
    aggregate = _load_json_object(aggregate_path, "C9 behavior-gate aggregate")
    metrics, gate = sft_gate_gpu.recompute_gate_from_rollouts(
        rows, num_prompts=c9_evidence.GATE_NUM_PROMPTS,
        samples_per_prompt=c9_evidence.GATE_SAMPLES_PER_PROMPT,
        generation_seed=c9_evidence.GATE_GENERATION_SEED,
        max_completion_tokens=c9_evidence.GATE_MAX_COMPLETION_TOKENS,
        ordered_problem_ids=[str(row["problem_id"]) for row in selection],
        thresholds=_c9_gate_thresholds(),
    )
    if metrics != aggregate.get("metrics_by_prompt_mode") or \
            gate != aggregate.get("health_gate"):
        raise RuntimeError("persisted C9 behavior-gate verdict is invalid")
    if not c10_mode and gate.get("passed") is not True:
        raise RuntimeError("persisted C9 behavior-gate verdict is invalid")
    if c10_mode and c10_evidence.classify_gate_outcome(
            metrics, gate) != evidence.get("gate_outcome"):
        raise RuntimeError("persisted C10 behavior-gate outcome drifted")
    manifest = _c9_model_manifest(model_dir, stage=stage)
    if stage in ("scout_step50", "scout_final"):
        current_metadata = _load_json_object(
            os.path.join(model_dir, "trainer_state.json"),
            "persisted C9 gated Scout trainer_state")
        current_runtime = current_metadata.get("runtime_contract")
        current_step = int(current_metadata.get("step", -1))
    else:
        current_metadata = _load_json_object(
            os.path.join(model_dir, "sft_provenance.json"),
            "persisted C9 gated SFT provenance")
        current_runtime = current_metadata.get("sft_runtime_contract")
        current_step = int(current_metadata.get("optimizer_steps", -1))
    if current_runtime != evidence.get("model_runtime_contract") or \
            current_step != evidence.get("model_step"):
        raise RuntimeError("persisted C9 gated model runtime/step drifted")
    no_library_version = (
        "c10_no_sft_library_v1" if c10_mode else "c9_no_sft_library_v1")
    accepted_digest = (
        common.semantic_sft_jsonl_digest(accepted_path)
        if accepted_path else {
            "version": no_library_version, "row_count": 0,
            "sha256": common.canonical_json_sha256(
                {"version": no_library_version, "rows": []}),
        })
    if multi_identity is None:
        if c10_mode:
            c10_evidence.validate_behavior_gate_evidence(
                evidence, expected_model_manifest=manifest,
                expected_code_artifact=_c9_code_identity(args),
                expected_selection_sha256=expected_selection_sha256,
                expected_endpoint_replay_sha256=
                    expected_endpoint_replay_sha256,
                require_passed=None)
        else:
            c9_evidence.validate_behavior_gate_evidence(
                evidence, expected_stage=stage,
                expected_model_manifest=manifest,
                expected_accepted_digest=accepted_digest,
                expected_code_artifact=_c9_code_identity(args),
                expected_selection_sha256=expected_selection_sha256,
                expected_profile=str(args.c9_execution_profile),
                expected_cell=str(args.stabilization_cell))
    else:
        if str(multi_identity.get("gate_role", "scout")) == "pooled_sft":
            mrme_contract.validate_c9_multi_explorer_sft_gate_evidence(
                evidence, contract=multi_identity["contract"],
                round_index=int(multi_identity["round_index"]),
                expected_runtime_contract=current_runtime,
                expected_code_identity=multi_identity["contract"][
                    "code_identity"],
                expected_model_manifest=manifest,
                expected_accepted_digest=accepted_digest,
                expected_selection_sha256=expected_selection_sha256)
        else:
            mrme_contract.validate_c9_multi_explorer_gate_evidence(
                evidence, contract=multi_identity["contract"],
                round_index=int(multi_identity["round_index"]),
                scout_index=int(multi_identity["scout_index"]),
                expected_runtime_contract=current_runtime,
                expected_code_identity=multi_identity["contract"][
                    "code_identity"],
                expected_model_manifest=manifest)
    current_accepted_sha = (
        common.file_sha256(accepted_path) if accepted_path else None)
    if current_accepted_sha != evidence.get("accepted_file_sha256"):
        raise RuntimeError("C9 behavior-gate accepted library bytes drifted")
    return evidence


def _validate_c9_sft_endpoint(
    model_dir: str, training_evidence: Dict[str, Any],
) -> Dict[str, Any]:
    c9_evidence.validate_sft_training_evidence(training_evidence)
    provenance = _load_json_object(
        os.path.join(model_dir, "sft_provenance.json"),
        "C9 fixed SFT endpoint provenance")
    exact = {
        "sft_contract_version": "c8-eos-aware",
        "termination_mode": "append_eos",
        "accepted_examples": c9_evidence.ACCEPTED_ROWS,
        "trainable_examples": c9_evidence.ACCEPTED_ROWS,
        "eos_supervised_examples": c9_evidence.ACCEPTED_ROWS,
        "epochs": c9_evidence.SFT_EPOCHS,
        "model_init": "Qwen/Qwen3-1.7B",
        "optimizer": "adamw",
        "grad_accum": 1,
        "micro_steps": c9_evidence.SFT_OPTIMIZER_STEPS,
        "optimizer_steps": c9_evidence.SFT_OPTIMIZER_STEPS,
        "planned_optimizer_steps": c9_evidence.SFT_OPTIMIZER_STEPS,
        "requested_save_optimizer_steps": [],
        "saved_optimizer_steps": [],
    }
    drift = [
        f"{key}={provenance.get(key)!r} (expected {value!r})"
        for key, value in exact.items() if provenance.get(key) != value
    ]
    if not math.isclose(
            float(provenance.get("learning_rate", -1.0)), 5e-6,
            rel_tol=0.0, abs_tol=1e-15):
        drift.append(
            f"learning_rate={provenance.get('learning_rate')!r} (expected 5e-6)")
    if drift:
        raise RuntimeError("C9 fixed SFT endpoint contract drift: " + "; ".join(drift))
    if provenance.get("c9_training_evidence") != training_evidence or \
            provenance.get("fresh_c8_training_evidence") is not None:
        raise RuntimeError("C9 SFT provenance training-evidence mismatch")
    if provenance.get("accepted_library_digest") != \
            training_evidence["accepted_library"]["semantic_digest"]:
        raise RuntimeError("C9 SFT accepted-library digest mismatch")
    if provenance.get("accepted_file_sha256") != \
            training_evidence["accepted_library"]["file_sha256"]:
        raise RuntimeError("C9 SFT accepted-library file SHA mismatch")
    if provenance.get("training_sequence_digest") != \
            training_evidence["accepted_library"]["training_sequence_digest"]:
        raise RuntimeError("C9 SFT training-sequence digest mismatch")
    if provenance.get("sft_runtime_contract") != \
            training_evidence["sft_runtime_contract"]:
        raise RuntimeError("C9 SFT runtime contract mismatch")
    completed = provenance.get("completed_run_evidence")
    if not isinstance(completed, dict):
        raise RuntimeError("C9 SFT endpoint lacks completed-run evidence")
    unsigned = dict(completed)
    recorded_sha = unsigned.pop("evidence_sha256", None)
    if recorded_sha != common.canonical_json_sha256(unsigned) or \
            completed.get("training_completed") is not True or \
            completed.get("optimizer_steps_executed") != \
            c9_evidence.SFT_OPTIMIZER_STEPS or \
            completed.get("micro_steps_executed") != \
            c9_evidence.SFT_OPTIMIZER_STEPS or \
            completed.get("training_evidence_sha256") != \
            training_evidence["evidence_sha256"]:
        raise RuntimeError("C9 SFT completed-run evidence is invalid")
    manifest = provenance.get("checkpoint_file_manifest")
    common.verify_directory_file_manifest(
        model_dir, manifest, exclude_relative_paths=("sft_provenance.json",))
    return provenance


def _validate_c9_multi_pooled_sft_endpoint(
    model_dir: str, *, contract: Dict[str, Any], round_idx: int,
    accepted_path: str, trajectory_source_manifest_path: str,
) -> Dict[str, Any]:
    """Require the exact Base-init 500x2 pooled endpoint before its gate.

    The behavior panel is a safety check, not a substitute for training
    provenance.  This preflight therefore proves the fixed SFT arithmetic and
    the union-of-K source binding before any gate rollout or Central update.
    """
    if not mrme_contract.is_c9_multi_explorer(contract):
        raise RuntimeError("pooled C9 SFT validation requires a C9 campaign")
    provenance = _load_json_object(
        os.path.join(model_dir, "sft_provenance.json"),
        "C9 multi-explorer pooled SFT provenance")
    source_manifest = _load_json_object(
        trajectory_source_manifest_path,
        "C9 multi-explorer trajectory source manifest")
    common.verify_trajectory_source_manifest(source_manifest)
    source_digest = common.trajectory_source_manifest_digest(source_manifest)
    accepted_digest = common.semantic_sft_jsonl_digest(accepted_path)
    runtime = provenance.get("sft_runtime_contract")
    expected_runtime = dict(contract["training_contract"]["central_sft"])
    expected_runtime["seed"] = int(contract["base_seed"]) + int(round_idx) - 1
    exact = {
        "model_init": contract["model"],
        "accepted_examples": 500,
        "trainable_examples": 500,
        "eos_supervised_examples": 500,
        "epochs": 2,
        "optimizer": "adamw",
        "grad_accum": 1,
        "micro_steps": 1000,
        "optimizer_steps": 1000,
        "planned_optimizer_steps": 1000,
    }
    drift = [
        f"{key}={provenance.get(key)!r} (expected {expected!r})"
        for key, expected in exact.items()
        if provenance.get(key) != expected
    ]
    if runtime != expected_runtime:
        drift.append("sft_runtime_contract differs from campaign template")
    if provenance.get("accepted_library_digest") != accepted_digest:
        drift.append("accepted-library digest")
    if provenance.get("accepted_file_sha256") != \
            common.file_sha256(accepted_path):
        drift.append("accepted-library bytes")
    if provenance.get("trajectory_source", {}).get("manifest") != \
            source_manifest or provenance.get("trajectory_source", {}).get(
                "digest") != source_digest:
        drift.append("pooled trajectory-source binding")
    completed = provenance.get("completed_run_evidence")
    if not isinstance(completed, dict):
        drift.append("completed-run evidence")
    else:
        unsigned = dict(completed)
        recorded_sha = unsigned.pop("evidence_sha256", None)
        if recorded_sha != common.canonical_json_sha256(unsigned) or \
                completed.get("training_completed") is not True or \
                completed.get("optimizer_steps_executed") != 1000 or \
                completed.get("micro_steps_executed") != 1000:
            drift.append("completed-run evidence")
    manifest = provenance.get("checkpoint_file_manifest")
    try:
        common.verify_directory_file_manifest(
            model_dir, manifest,
            exclude_relative_paths=("sft_provenance.json",))
    except (TypeError, ValueError) as exc:
        drift.append(f"SFT model manifest: {exc}")
    if drift:
        raise RuntimeError(
            "C9 multi-explorer pooled SFT endpoint drift: "
            + "; ".join(drift))
    return provenance


def _ensure_c9_multi_pooled_sft_gate(
    args: argparse.Namespace, pool: VLLMPool, tokenizer, *,
    contract: Dict[str, Any], round_idx: int, round_dir: str,
    sft_dir: str, accepted_path: str,
    sft_model_binding: Dict[str, Any], central_runtime: Dict[str, Any],
    state: Dict[str, Any], wandb_run,
) -> tuple[Dict[str, Any], Dict[str, Any], str]:
    """Authorize Central only after the pooled fixed SFT endpoint passes C5.

    The signed gate, its three raw support files, the round source union, the
    SFT bytes, and the Central runtime are all revalidated on resume.  A failed
    gate is left on disk as durable negative evidence and cannot be overwritten
    or converted into a completed pipeline stage by ``--resume``.
    """
    if not mrme_contract.is_c9_multi_explorer(contract):
        raise RuntimeError("pooled C9 gate requested for a non-C9 campaign")
    source_manifest_path = str(args.trajectory_source_manifest or "")
    if not source_manifest_path:
        raise RuntimeError("pooled C9 gate lacks its trajectory source manifest")
    _validate_c9_multi_pooled_sft_endpoint(
        sft_dir, contract=contract, round_idx=round_idx,
        accepted_path=accepted_path,
        trajectory_source_manifest_path=source_manifest_path)
    _validate_sft_model_artifact_binding(sft_model_binding)
    if sft_model_binding.get("durable_wandb_upload") is not True:
        raise RuntimeError(
            "canonical C9 multi-explorer requires a durable uploaded SFT model")
    if sft_model_binding.get("artifact_subdir") != SFT_MODEL_ARTIFACT_SUBDIR:
        raise RuntimeError("C9 pooled SFT artifact subdirectory drifted")
    source = sft_model_binding.get("source")
    if not isinstance(source, dict) or not source.get("artifact_digest"):
        raise RuntimeError("C9 pooled SFT artifact lacks its durable source")
    common.require_immutable_wandb_artifact_ref(
        str(source.get("artifact_ref", "")),
        field="C9 multi-explorer pooled SFT model artifact")
    policy_manifest = _c9_model_manifest(
        sft_dir, stage="sft_fixed_endpoint")
    if sft_model_binding.get("policy_file_manifest") != policy_manifest:
        raise RuntimeError(
            "C9 pooled SFT policy bytes differ from the durable artifact")

    source_index_path = os.path.join(round_dir, "mrme_scout_sources.json")
    source_index = _load_json_object(
        source_index_path, "C9 multi-explorer round source index")
    if source_index.get("contract_sha256") != contract["contract_sha256"] or \
            int(source_index.get("round_index", 0)) != int(round_idx):
        raise RuntimeError("pooled C9 gate source index is not round-bound")
    source_panels = {
        str(source.get("c9_multi_explorer_scout_evidence", {}).get(
            "behavior_gate", {}).get("heldout_selection_sha256", ""))
        for source in source_index.get("sources", [])
    }
    if len(source_panels) != 1 or "" in source_panels:
        raise RuntimeError(
            "C9 multi-explorer Scouts did not use one fixed diagnostic panel")
    expected_panel = next(iter(source_panels))
    stage_seed = int(contract["base_seed"]) + int(round_idx) - 1
    multi_identity = {
        "contract": contract,
        "round_index": int(round_idx),
        "scout_index": 0,
        "seed": stage_seed,
        "gate_role": "pooled_sft",
    }
    gate_path = os.path.join(
        round_dir, "c9_behavior_gates", "sft_fixed_endpoint",
        "evidence.json")
    gate_done = _stage_done(state, round_idx, "sft_behavior_gate")
    if gate_done:
        if os.path.realpath(str(gate_done.get("evidence_path", ""))) != \
                os.path.realpath(gate_path):
            raise RuntimeError("C9 pooled-SFT gate evidence path drifted")
        gate = _validate_persisted_c9_behavior_gate(
            args, evidence_path=gate_path, stage="sft_fixed_endpoint",
            model_dir=sft_dir, accepted_path=accepted_path,
            expected_selection_sha256=expected_panel,
            multi_identity=multi_identity)
        if gate_done.get("evidence_sha256") != gate["evidence_sha256"]:
            raise RuntimeError("C9 pooled-SFT gate pipeline-state SHA mismatch")
    elif os.path.isfile(gate_path):
        # Recover a passed gate after a crash before the atomic stage mark.  A
        # failed or tampered gate raises here and is never regenerated in-place.
        gate = _validate_persisted_c9_behavior_gate(
            args, evidence_path=gate_path, stage="sft_fixed_endpoint",
            model_dir=sft_dir, accepted_path=accepted_path,
            expected_selection_sha256=expected_panel,
            multi_identity=multi_identity)
    else:
        gate, _ = _run_c9_behavior_gate(
            args, pool, tokenizer, stage="sft_fixed_endpoint",
            model_dir=sft_dir, accepted_path=accepted_path,
            round_dir=round_dir, expected_selection_sha256=expected_panel,
            multi_identity=multi_identity)

    trajectory_manifest = _load_json_object(
        source_manifest_path, "C9 pooled trajectory source manifest")
    common.verify_trajectory_source_manifest(trajectory_manifest)
    trajectory_digest = common.trajectory_source_manifest_digest(
        trajectory_manifest)
    accepted_digest = common.semantic_sft_jsonl_digest(accepted_path)
    model_manifest = policy_manifest
    lineage = mrme_contract.build_c9_multi_explorer_central_lineage(
        contract=contract, round_index=round_idx,
        shard_manifest_sha256=str(args._mrme_shard_manifest_sha256),
        trajectory_source_digest=trajectory_digest,
        accepted_library_digest=accepted_digest,
        sft_behavior_gate=gate,
        sft_model_file_manifest=model_manifest,
        sft_model_artifact=sft_model_binding,
        central_runtime_contract=central_runtime)
    lineage_path = os.path.join(
        round_dir, "c9_multi_explorer_central_lineage.json")
    if gate_done and not os.path.isfile(lineage_path):
        raise RuntimeError(
            "completed C9 pooled-SFT gate is missing its Central lineage")
    _write_or_verify_json(
        lineage_path, lineage,
        label="C9 multi-explorer pooled-SFT Central lineage")

    stage_outputs = {
        "evidence_path": gate_path,
        "evidence_sha256": gate["evidence_sha256"],
        "lineage_path": lineage_path,
        "lineage_sha256": lineage["lineage_sha256"],
        "contract_sha256": contract["contract_sha256"],
        "round_index": int(round_idx),
        "model_file_manifest": model_manifest,
        "accepted_library_digest": accepted_digest,
        "trajectory_source_digest": trajectory_digest,
    }
    if gate_done:
        for key, expected in stage_outputs.items():
            if gate_done.get(key) != expected:
                raise RuntimeError(
                    f"C9 pooled-SFT gate stage binding drifted: {key}")
    else:
        _mark_stage(
            args, state, round_idx, "sft_behavior_gate", **stage_outputs)
        _push_results_artifact(
            wandb_run, args, f"r{round_idx}-pooled-sft-behavior-gate")
    return gate, lineage, lineage_path


def _validated_frozen_accepted_entry(
    entry: Any, *, field: str = "evidence.accepted",
) -> Dict[str, Any]:
    accepted_entry = _validate_artifact_entry(
        entry, field)
    accepted_path = _require_attributed_path(
        str(accepted_entry["entry"].get("accepted_path", "")),
        accepted_entry["download"], f"{field}.accepted_path")
    artifact_ref = str(accepted_entry["source"]["artifact_ref"])
    if artifact_ref.split("/")[-1] != C7_REUSE_ACCEPTED_ARTIFACT_NAME:
        raise ValueError(
            "external ga8 diagnostic must use the frozen c7-reuse accepted artifact")
    if str(accepted_entry["source"]["artifact_digest"]) != \
            C7_REUSE_ACCEPTED_ARTIFACT_DIGEST:
        raise ValueError(
            "frozen c7-reuse accepted artifact digest does not match results:v1")
    digest = common.semantic_sft_jsonl_digest(accepted_path)
    if digest != C7_REUSE_ACCEPTED_DIGEST:
        raise ValueError(
            "external ga8 diagnostic accepted library differs from the frozen 500 rows")
    return {
        "path": accepted_path,
        "digest": digest,
        "source": accepted_entry["source"],
        "artifact": _stable_artifact_identity(accepted_entry["download"]),
        "file_sha256": common.file_sha256(accepted_path),
    }


def _validated_external_accepted(args: argparse.Namespace) -> Dict[str, Any]:
    index = _load_json_object(args.sft_evidence_index, "--sft-evidence-index")
    if index.get("version") == EXTERNAL_EVIDENCE_INDEX_VERSION:
        return _validated_frozen_accepted_entry(index.get("accepted"))
    if index.get("version") != fresh_c8_sft.EXTERNAL_EVIDENCE_INDEX_VERSION:
        raise ValueError("external SFT evidence index version mismatch")
    accepted_entry = _validate_artifact_entry(
        index.get("accepted"), "fresh_c8_evidence.accepted")
    accepted_path = _require_attributed_path(
        str(accepted_entry["entry"].get("accepted_path", "")),
        accepted_entry["download"], "fresh_c8_evidence.accepted.accepted_path")
    rows = _read_jsonl_objects(accepted_path, "fresh-c8 accepted artifact")
    if len(rows) != 500 or len({filter_pool_mod.c8_problem_key(row) for row in rows}) != 500:
        raise ValueError(
            "fresh-c8 accepted artifact must contain exactly 500 unique problems")
    return {
        "path": accepted_path,
        "digest": common.semantic_sft_dataset_digest(rows),
        "source": accepted_entry["source"],
        "artifact": _stable_artifact_identity(accepted_entry["download"]),
        "file_sha256": common.file_sha256(accepted_path),
    }


def _validated_frozen_sft_library(args: argparse.Namespace) -> Dict[str, Any]:
    index = _load_json_object(
        args.frozen_sft_library_evidence_index,
        "--frozen-sft-library-evidence-index")
    if index.get("version") != FROZEN_SFT_LIBRARY_EVIDENCE_INDEX_VERSION:
        raise ValueError("frozen SFT library evidence index version mismatch")
    return _validated_frozen_accepted_entry(
        index.get("accepted"), field="frozen_sft_library.accepted")


def _validated_trajectory_provenance(args: argparse.Namespace) -> Dict[str, Any]:
    manifest = _load_json_object(
        args.trajectory_source_manifest, "--trajectory-source-manifest")
    digest = common.verify_trajectory_source_manifest(manifest)
    manifest_paths = [
        os.path.realpath(str(file_info["local_path"]))
        for artifact in manifest.get("artifacts", [])
        for file_info in artifact.get("files", [])
    ]
    if sorted(manifest_paths) != sorted(
            os.path.realpath(path) for path in (args.trajectories or [])):
        raise ValueError(
            "--trajectories do not exactly match the attributed source manifest")
    sources = {
        "version": manifest["version"],
        "artifacts": sorted([
            {
                "artifact_ref": str(artifact["artifact_ref"]),
                "artifact_digest": str(artifact["artifact_digest"]),
                "files": sorted([
                    {
                        "relative_path": str(item["relative_path"]),
                        "sha256": str(item["sha256"]),
                    }
                    for item in artifact["files"]
                ], key=lambda item: item["relative_path"]),
            }
            for artifact in manifest["artifacts"]
        ], key=lambda item: (item["artifact_ref"], item["artifact_digest"])),
    }
    return {"manifest": manifest, "digest": digest, "sources": sources}


def _validated_frozen_sft_training_evidence(
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Byte/source evidence for the noncanonical frozen-library SFT rerun."""
    accepted = _validated_frozen_sft_library(args)
    trajectory = _validated_trajectory_provenance(args)
    evidence = {
        "version": "frozen_c7_library_ga8_training_evidence_v1",
        "scientific_variant": FROZEN_SFT_LIBRARY_VARIANT,
        "accepted_library_digest": accepted["digest"],
        "accepted_file_sha256": accepted["file_sha256"],
        "accepted_source": accepted["source"],
        "accepted_artifact": accepted["artifact"],
        "trajectory_source_digest": trajectory["digest"],
        "trajectory_sources": trajectory["sources"],
    }
    evidence["evidence_sha256"] = common.canonical_json_sha256(evidence)
    return evidence


def _validate_c7_external_sft_evidence(
    args: argparse.Namespace, accepted_path: str,
) -> Dict[str, Any]:
    """Rebuild the all-four raw-gate selection and bind every artifact byte."""
    index = _load_json_object(args.sft_evidence_index, "--sft-evidence-index")
    if index.get("version") != EXTERNAL_EVIDENCE_INDEX_VERSION:
        raise ValueError("external SFT evidence index version mismatch")
    trajectory_evidence = _validated_trajectory_provenance(args)
    accepted_evidence = _validated_external_accepted(args)
    if os.path.realpath(accepted_path) == os.path.realpath(accepted_evidence["path"]):
        raise ValueError(
            "pipeline accepted handoff must be a copied durable file, not the "
            "artifact download path itself")

    model_entry = _validate_artifact_entry(index.get("model"), "evidence.model")
    model_dir = _require_attributed_path(
        str(model_entry["entry"].get("model_dir", "")),
        model_entry["download"], "evidence.model.model_dir")
    if model_dir != os.path.realpath(args.sft_model):
        raise ValueError("--sft-model does not match evidence-index model_dir")
    model_identity = _stable_artifact_identity(model_entry["download"])

    gate_entries = index.get("gates")
    if not isinstance(gate_entries, list) or len(gate_entries) != 4:
        raise ValueError("evidence index must contain all four gate bundles")
    bundle_dirs: Dict[int, str] = {}
    gate_sources: Dict[int, str] = {}
    gate_download_identities: Dict[str, Any] = {}
    selected_aggregate: Optional[Dict[str, Any]] = None
    for raw_entry in gate_entries:
        step = int(raw_entry.get("optimizer_step", -1))
        if step in bundle_dirs:
            raise ValueError(f"duplicate gate bundle for optimizer step {step}")
        validated = _validate_artifact_entry(raw_entry, f"evidence.gate[{step}]")
        bundle = _require_attributed_path(
            str(raw_entry.get("bundle_dir", "")), validated["download"],
            f"evidence.gate[{step}].bundle_dir")
        for filename in ("aggregate_metrics.json", "rollouts.jsonl",
                         "heldout_selection.jsonl"):
            _require_attributed_path(
                os.path.join(bundle, filename), validated["download"],
                f"evidence.gate[{step}]/{filename}")
        bundle_dirs[step] = bundle
        gate_sources[step] = _source_label(
            validated["source"], f"evidence.gate[{step}].source")
        gate_download_identities[str(step)] = _stable_artifact_identity(
            validated["download"])
    if tuple(sorted(bundle_dirs)) != sft_selector.PREREGISTERED_STEPS:
        raise ValueError("gate evidence does not cover exact steps 16/32/64/126")

    rebuilt_manifest = sft_selector.build_selection_manifest_from_bundles(
        bundle_dirs, gate_sources)
    # Do not treat the selector's ``REF#digest=SHA`` string as standalone
    # provenance.  Bind every rebuilt candidate back to the exact source in
    # the already-validated artifact download manifest from the evidence
    # index.  This is deliberately redundant with ``gate_sources`` above: it
    # makes the trust boundary explicit and prevents a future selector change
    # from silently turning a parsed label into the source of truth.
    for candidate in rebuilt_manifest["candidates"]:
        step_key = str(int(candidate["optimizer_step"]))
        attributed_source = gate_download_identities[step_key]["source"]
        attributed_label = _source_label(
            attributed_source, f"evidence.gate[{step_key}].validated_source")
        if candidate.get("gate_artifact_identity") != attributed_source \
                or candidate.get("gate_source") != attributed_label:
            raise ValueError(
                f"gate candidate {step_key} source is not bound to its "
                "validated artifact download manifest")
    selection_entry = _validate_artifact_entry(
        index.get("selection"), "evidence.selection")
    selection_path = _require_attributed_path(
        str(selection_entry["entry"].get("manifest_path", "")),
        selection_entry["download"], "evidence.selection.manifest_path")
    supplied_manifest = _load_json_object(selection_path, "selection manifest")
    if supplied_manifest != rebuilt_manifest:
        raise ValueError(
            "selection manifest differs from local all-four raw-gate rebuild")
    selected_step = int(rebuilt_manifest["selected_optimizer_step"])
    if rebuilt_manifest.get("selected_model_source") != model_entry["source"]:
        raise ValueError("selected gate points to a different model artifact")
    if rebuilt_manifest.get("selected_model_file_manifest") != model_identity:
        raise ValueError("selected gate model-file manifest differs from downloaded model")

    selected_bundle = bundle_dirs[selected_step]
    selected_aggregate = _load_json_object(
        os.path.join(selected_bundle, "aggregate_metrics.json"),
        "selected gate aggregate")
    local_checkpoint = _sft_checkpoint_metadata(model_dir)
    if local_checkpoint != selected_aggregate.get("model_checkpoint"):
        raise ValueError("downloaded SFT checkpoint metadata differs from selected gate")
    checkpoint_metadata = local_checkpoint.get("metadata", {})
    sft_selector._validate_checkpoint_contract(checkpoint_metadata, selected_step)

    fresh_accepted_digest = common.semantic_sft_jsonl_digest(accepted_path)
    if fresh_accepted_digest != accepted_evidence["digest"]:
        raise ValueError("pipeline accepted handoff differs from frozen c7-reuse library")
    selected_candidate = next(
        item for item in rebuilt_manifest["candidates"]
        if int(item["optimizer_step"]) == selected_step)
    bound_accepted_digests = (
        checkpoint_metadata.get("accepted_library_digest"),
        selected_aggregate.get("accepted_library_digest"),
        rebuilt_manifest.get("accepted_library_digest"),
        selected_candidate.get("accepted_library_digest"),
    )
    if any(item != fresh_accepted_digest for item in bound_accepted_digests):
        raise ValueError("frozen accepted.jsonl semantic digest does not match SFT/gate/selection")
    accepted_sources = rebuilt_manifest.get("accepted_source_manifest", {}).get(
        "sources", [])
    if not any(
        item.get("artifact_ref") == accepted_evidence["source"]["artifact_ref"]
        and item.get("artifact_digest") == accepted_evidence["source"]["artifact_digest"]
        and item.get("file_manifest_sha256")
        == accepted_evidence["artifact"]["file_manifest_sha256"]
        for item in accepted_sources
    ):
        raise ValueError(
            "gate accepted-library source does not match frozen accepted artifact")

    trajectory_digest = trajectory_evidence["digest"]
    bound_trajectory_digests = (
        checkpoint_metadata.get("trajectory_source", {}).get("digest"),
        rebuilt_manifest.get("trajectory_source_digest"),
        selected_candidate.get("trajectory_source_digest"),
    )
    if any(item != trajectory_digest for item in bound_trajectory_digests):
        raise ValueError("trajectory artifact/source provenance does not match SFT/gate/selection")
    trajectory_sources = trajectory_evidence["sources"]

    lineage = {
        "lineage_contract_version": grpo_gpu.EXTERNAL_LINEAGE_CONTRACT_VERSION,
        "scientific_variant": "external_sft_exposure_gated",
        "model_source": model_entry["source"],
        "model_file_manifest": model_identity,
        "gate_artifacts": gate_download_identities,
        "selection_source": selection_entry["source"],
        "selection_artifact": _stable_artifact_identity(selection_entry["download"]),
        "selected_optimizer_step": selected_step,
        "accepted_library_digest": fresh_accepted_digest,
        "accepted_source": accepted_evidence["source"],
        "accepted_artifact": accepted_evidence["artifact"],
        "trajectory_source_digest": trajectory_digest,
        "trajectory_sources": trajectory_sources,
        "selection_manifest_sha256": rebuilt_manifest["manifest_sha256"],
        "candidate_raw_evidence": [
            {
                key: candidate[key]
                for key in (
                    "optimizer_step", "passed", "gate_source", "aggregate_sha256",
                    "raw_rollouts_sha256", "heldout_selection_file_sha256",
                )
            }
            for candidate in rebuilt_manifest["candidates"]
        ],
        "evidence_index_sha256": common.canonical_json_sha256(index),
        "central_runtime_contract": _external_central_runtime_contract(args),
    }
    lineage["lineage_sha256"] = common.canonical_json_sha256(lineage)
    return lineage


def _validate_fresh_c8_external_sft_evidence(
    args: argparse.Namespace, accepted_path: str,
) -> Dict[str, Any]:
    """Bind a fresh-c8 selected SFT checkpoint to every durable artifact.

    Unlike the frozen c7-reuse path, selecting the full 1,000-update endpoint
    preserves canonical c8 *only* when this validator can prove every fixed
    Scout/filter/SFT fact.  Earlier passing snapshots are automatically and
    immutably labeled as the exposure-gated scientific variant.
    """
    index = _load_json_object(args.sft_evidence_index, "--sft-evidence-index")
    if index.get("version") != fresh_c8_sft.EXTERNAL_EVIDENCE_INDEX_VERSION:
        raise ValueError("fresh-c8 external SFT evidence index version mismatch")
    accepted_evidence = _validated_external_accepted(args)
    if os.path.realpath(accepted_path) == os.path.realpath(accepted_evidence["path"]):
        raise ValueError(
            "pipeline accepted handoff must be a copied durable file, not the "
            "artifact download path itself")
    if common.semantic_sft_jsonl_digest(accepted_path) != accepted_evidence["digest"] \
            or common.file_sha256(accepted_path) != accepted_evidence["file_sha256"]:
        raise ValueError("pipeline accepted handoff differs from fresh-c8 artifact")

    trajectory = _validated_trajectory_provenance(args)
    if int(trajectory["digest"].get("artifact_count", -1)) != 1:
        raise ValueError("fresh-c8 SingleScout continuation requires one trajectory artifact")
    trajectory_files = [
        item
        for artifact in trajectory["sources"]["artifacts"]
        for item in artifact["files"]
    ]
    if len(trajectory_files) != 1:
        raise ValueError("fresh-c8 SingleScout continuation requires one trajectory file")

    code_entry = _validate_artifact_entry(index.get("code"), "fresh_c8_evidence.code")
    code_identity = _stable_artifact_identity(code_entry["download"])
    model_entry = _validate_artifact_entry(index.get("model"), "fresh_c8_evidence.model")
    model_dir = _require_attributed_path(
        str(model_entry["entry"].get("model_dir", "")),
        model_entry["download"], "fresh_c8_evidence.model.model_dir")
    if model_dir != os.path.realpath(args.sft_model):
        raise ValueError("--sft-model does not match fresh-c8 evidence model_dir")
    model_identity = _stable_artifact_identity(model_entry["download"])

    gate_entries = index.get("gates")
    if not isinstance(gate_entries, list) or len(gate_entries) != 4:
        raise ValueError("fresh-c8 evidence index must contain all four gate bundles")
    bundle_dirs: Dict[int, str] = {}
    gate_sources: Dict[int, str] = {}
    gate_download_identities: Dict[str, Any] = {}
    for raw_entry in gate_entries:
        step = int(raw_entry.get("optimizer_step", -1))
        if step in bundle_dirs:
            raise ValueError(f"duplicate fresh-c8 gate bundle for optimizer step {step}")
        validated = _validate_artifact_entry(
            raw_entry, f"fresh_c8_evidence.gate[{step}]")
        bundle = _require_attributed_path(
            str(raw_entry.get("bundle_dir", "")), validated["download"],
            f"fresh_c8_evidence.gate[{step}].bundle_dir")
        for filename in ("aggregate_metrics.json", "rollouts.jsonl",
                         "heldout_selection.jsonl"):
            _require_attributed_path(
                os.path.join(bundle, filename), validated["download"],
                f"fresh_c8_evidence.gate[{step}]/{filename}")
        bundle_dirs[step] = bundle
        gate_sources[step] = _source_label(
            validated["source"], f"fresh_c8_evidence.gate[{step}].source")
        gate_download_identities[str(step)] = _stable_artifact_identity(
            validated["download"])
    if tuple(sorted(bundle_dirs)) != fresh_c8_sft.PREREGISTERED_STEPS:
        raise ValueError("fresh-c8 gate evidence must cover steps 125/250/500/1000")

    rebuilt = sft_selector.build_selection_manifest_from_bundles(
        bundle_dirs, gate_sources, fresh_c8_sft.PROFILE,
        accepted_jsonl=accepted_evidence["path"],
        accepted_artifact_identity=accepted_evidence["artifact"])
    for candidate in rebuilt["candidates"]:
        step_key = str(int(candidate["optimizer_step"]))
        attributed = gate_download_identities[step_key]["source"]
        if candidate.get("gate_artifact_identity") != attributed or \
                candidate.get("gate_source") != _source_label(
                    attributed, f"fresh_c8_evidence.gate[{step_key}].validated_source"):
            raise ValueError(
                f"fresh-c8 gate candidate {step_key} is not artifact-bound")

    selection_entry = _validate_artifact_entry(
        index.get("selection"), "fresh_c8_evidence.selection")
    selection_path = _require_attributed_path(
        str(selection_entry["entry"].get("manifest_path", "")),
        selection_entry["download"], "fresh_c8_evidence.selection.manifest_path")
    supplied = _load_json_object(selection_path, "fresh-c8 selection manifest")
    if supplied != rebuilt:
        raise ValueError("fresh-c8 selection differs from local raw-gate rebuild")
    selected_step = int(rebuilt["selected_optimizer_step"])
    if rebuilt.get("selected_model_source") != model_entry["source"] or \
            rebuilt.get("selected_model_file_manifest") != model_identity:
        raise ValueError("fresh-c8 selected model artifact differs from gate evidence")

    selected_aggregate = _load_json_object(
        os.path.join(bundle_dirs[selected_step], "aggregate_metrics.json"),
        "fresh-c8 selected gate aggregate")
    local_checkpoint = _sft_checkpoint_metadata(model_dir)
    if local_checkpoint != selected_aggregate.get("model_checkpoint"):
        raise ValueError("downloaded fresh-c8 checkpoint differs from selected gate")
    metadata = local_checkpoint.get("metadata", {})
    sft_selector._validate_checkpoint_contract(
        metadata, selected_step, fresh_c8_sft.PROFILE)
    training = fresh_c8_sft.validate_training_evidence(
        metadata.get("fresh_c8_training_evidence"))
    if training != rebuilt.get("fresh_c8_training_evidence"):
        raise ValueError("fresh-c8 training evidence differs across selection/model")
    if training["accepted_library"]["semantic_digest"] != accepted_evidence["digest"] \
            or training["accepted_library"]["file_sha256"] != \
            accepted_evidence["file_sha256"]:
        raise ValueError("fresh-c8 accepted artifact differs from training bytes")
    if training["trajectories"]["file_sha256"] != trajectory_files[0]["sha256"]:
        raise ValueError("fresh-c8 trajectory artifact differs from training bytes")
    training_code = training["code_source"]
    if code_identity["source"] != {
        "artifact_ref": training_code["artifact_ref"],
        "artifact_digest": training_code["artifact_digest"],
    } or code_identity["file_manifest_sha256"] != training_code["file_manifest_sha256"]:
        raise ValueError("fresh-c8 continuation code differs from SFT training code")
    accepted_sources = rebuilt.get("accepted_source_manifest", {}).get("sources", [])
    if not any(
        source.get("artifact_ref") == accepted_evidence["source"]["artifact_ref"]
        and source.get("artifact_digest") == accepted_evidence["source"]["artifact_digest"]
        and source.get("file_manifest_sha256")
        == accepted_evidence["artifact"]["file_manifest_sha256"]
        for source in accepted_sources
    ):
        raise ValueError("fresh-c8 gate accepted source differs from accepted artifact")

    # Resolve the Stage-5 GRPO arguments exactly once, then fail closed before
    # signing lineage if the fresh-c8 resampling ceiling is not the frozen 64.
    # The pipeline sentinel 0 must resolve to 64; an explicit smaller value
    # must never be laundered into a canonical run.
    central_runtime = _external_central_runtime_contract(args)
    if int(central_runtime.get("dynamic_max_attempts", -1)) != 64:
        raise ValueError(
            "fresh-c8 external Central requires dynamic_max_attempts=64")

    canonical = selected_step == 1000
    variant = "" if canonical else fresh_c8_sft.EARLY_SELECTION_VARIANT
    args.scientific_variant = variant
    lineage = {
        "lineage_contract_version": fresh_c8_sft.LINEAGE_CONTRACT_VERSION,
        "canonical": canonical,
        "scientific_variant": variant,
        "code_artifact": code_identity,
        "model_source": model_entry["source"],
        "model_file_manifest": model_identity,
        "gate_artifacts": gate_download_identities,
        "selection_source": selection_entry["source"],
        "selection_artifact": _stable_artifact_identity(selection_entry["download"]),
        "selected_optimizer_step": selected_step,
        "accepted_library_digest": accepted_evidence["digest"],
        "accepted_source": accepted_evidence["source"],
        "accepted_artifact": accepted_evidence["artifact"],
        "trajectory_source_digest": trajectory["digest"],
        "trajectory_sources": trajectory["sources"],
        "fresh_c8_training_evidence": training,
        "selection_manifest_sha256": rebuilt["manifest_sha256"],
        "candidate_raw_evidence": [
            {
                key: candidate[key]
                for key in (
                    "optimizer_step", "passed", "gate_source", "aggregate_sha256",
                    "raw_rollouts_sha256", "heldout_selection_file_sha256",
                )
            }
            for candidate in rebuilt["candidates"]
        ],
        "evidence_index_sha256": common.canonical_json_sha256(index),
        "central_runtime_contract": central_runtime,
    }
    lineage["lineage_sha256"] = common.canonical_json_sha256(lineage)
    return lineage


def _validate_external_sft_evidence(
    args: argparse.Namespace, accepted_path: str,
) -> Dict[str, Any]:
    profile = _external_evidence_profile(args)
    if profile == fresh_c8_sft.PROFILE:
        return _validate_fresh_c8_external_sft_evidence(args, accepted_path)
    return _validate_c7_external_sft_evidence(args, accepted_path)


def _fresh_external_declared_variant(args: argparse.Namespace) -> str:
    """Read only the artifact-bound selection label for pre-run config naming.

    The complete raw-gate rebuild still happens in Stage 4.  This early read
    cannot authorize Central; it solely ensures that W&B/config-tier labeling
    is already correct before the pipeline run is initialized.
    """
    index = _load_json_object(args.sft_evidence_index, "--sft-evidence-index")
    selection_entry = _validate_artifact_entry(
        index.get("selection"), "fresh_c8_evidence.selection")
    path = _require_attributed_path(
        str(selection_entry["entry"].get("manifest_path", "")),
        selection_entry["download"], "fresh_c8_evidence.selection.manifest_path")
    manifest = _load_json_object(path, "fresh-c8 selection manifest")
    if manifest.get("selection_profile") != fresh_c8_sft.PROFILE:
        raise ValueError("fresh-c8 selection manifest profile mismatch")
    step = int(manifest.get("selected_optimizer_step", -1))
    if step not in fresh_c8_sft.PREREGISTERED_STEPS:
        raise ValueError("fresh-c8 selection manifest chose an unregistered step")
    return "" if step == 1000 else fresh_c8_sft.EARLY_SELECTION_VARIANT


def _validate_completed_central(
    model_dir: str, expected_lineage: Optional[Dict[str, Any]],
    *, expected_steps: int,
    expected_runtime_contract: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Require a complete, byte-bound Central model before skipping Stage 5."""
    if not os.path.isdir(model_dir):
        raise RuntimeError(f"completed Central model directory is missing: {model_dir!r}")
    if not os.path.isfile(os.path.join(model_dir, "config.json")):
        raise RuntimeError("completed Central model lacks config.json")
    state_path = os.path.join(model_dir, "trainer_state.json")
    if not os.path.isfile(state_path):
        raise RuntimeError("completed Central model lacks trainer_state.json")
    state = _load_json_object(state_path, "completed Central trainer_state")
    if state.get("phase") != "central" or state.get("checkpoint_kind") != "deployable_model":
        raise RuntimeError("completed Central trainer_state is not a deployable Central model")
    if state.get("external_lineage") != expected_lineage:
        raise RuntimeError("completed Central external lineage mismatch")
    expected_sha = (
        (expected_lineage.get("lineage_sha256")
         or expected_lineage.get("evidence_sha256"))
        if expected_lineage else None)
    if state.get("external_lineage_sha256") != expected_sha:
        raise RuntimeError("completed Central lineage SHA mismatch")
    if int(state.get("step", -1)) != int(expected_steps):
        raise RuntimeError(
            "completed Central step mismatch: "
            f"state={state.get('step')!r}, requested={int(expected_steps)}")
    if expected_runtime_contract is not None and \
            state.get("runtime_contract") != expected_runtime_contract:
        raise RuntimeError("completed Central runtime contract mismatch")
    if expected_lineage is not None and expected_lineage.get(
            "lineage_contract_version") == c9_evidence.CENTRAL_LINEAGE_VERSION:
        c9_evidence.validate_central_lineage(
            expected_lineage,
            expected_runtime_contract=expected_runtime_contract)
        runtime = state.get("runtime_contract", {})
        if not math.isclose(float(state.get("lambda", -1.0)), 0.0) or \
                not math.isclose(float(runtime.get("lambda_novelty", -1.0)), 0.0):
            raise RuntimeError("completed C9 Central novelty is not identically zero")
        if "scout_novelty_contract" in runtime or \
                os.path.exists(os.path.join(model_dir, "rnd_state.pt")):
            raise RuntimeError("completed C9 Central unexpectedly contains RND state")
    manifest = state.get("checkpoint_file_manifest")
    if not isinstance(manifest, dict):
        raise RuntimeError("completed Central lacks a model-file manifest")
    common.verify_directory_file_manifest(
        model_dir, manifest, exclude_relative_paths=("trainer_state.json",))
    return state


@contextlib.contextmanager
def _round_shard_env(round_idx: int, rounds: int, seed: int) -> Iterator[None]:
    """TMX_DATASET_SHARD_* for one round (multi-round disjoint data shards).

    Mirrors tmx_jax/pipeline.py multi_round_main: uniform policy =
    deterministic shuffle by the shard seed, then the round's contiguous
    slice (tmx_jax.data._apply_round_shard). No-op for single-round runs.
    """
    if int(rounds) <= 1:
        yield
        return
    shard_env = {
        "TMX_DATASET_SHARD_INDEX": str(int(round_idx) - 1),
        "TMX_DATASET_SHARD_COUNT": str(int(rounds)),
        "TMX_DATASET_SHARD_POLICY": "uniform",
        "TMX_DATASET_SHARD_SEED": str(int(seed)),
    }
    saved = {k: os.environ.get(k) for k in shard_env}
    os.environ.update(shard_env)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ------------------------------------------------------------ state / W&B ----

def _c9_pipeline_execution_contract(args: argparse.Namespace) -> Dict[str, Any]:
    """Bind a resumable C9 output directory to one exact execution request."""
    sargs = _sft_args(
        args,
        accepted=os.path.join(args.output_dir, "round_01", "accepted.jsonl"),
        output_dir=os.path.join(args.output_dir, "round_01", "sft"),
        round_idx=1, lam=0.5, global_step_offset=c9_evidence.SCOUT_STEPS,
    )
    sft_runtime = sft_gpu.build_sft_runtime_contract(
        sargs, resolved_max_total_len=int(sargs.max_total_len))
    payload = {
        "version": c9_evidence.PIPELINE_EXECUTION_CONTRACT_VERSION,
        "run_name": str(args.run_name),
        "output_dir": os.path.realpath(args.output_dir),
        "c9_execution_profile": str(args.c9_execution_profile),
        "stabilization_cell": str(args.stabilization_cell),
        "scientific_variant": str(args.scientific_variant),
        "model": str(args.model),
        "model_revision": str(args.model_revision or ""),
        "dataset": str(args.dataset),
        "seed": int(args.seed),
        "rounds": int(args.rounds),
        "lambda_schedule": parse_lambda_schedule(args),
        "scout_steps": int(args.scout_steps),
        "central_steps": int(args.central_steps),
        "policy": str(args.policy),
        "min_accepted": int(args.min_accepted),
        "max_accepted": int(args.max_accepted),
        "harvest_max_passes": int(args.harvest_max_passes),
        "scout_runtime_contract": _c9_scout_runtime_contract(
            args, scout_init=str(args.model), round_idx=1, lam=0.5),
        "sft_runtime_contract": sft_runtime,
        "behavior_gate_contract": {
            "metric_contract_version": sft_gate_gpu.METRIC_CONTRACT_VERSION,
            "sampling": {
                "num_prompts": c9_evidence.GATE_NUM_PROMPTS,
                "samples_per_prompt": c9_evidence.GATE_SAMPLES_PER_PROMPT,
                "selection_seed": c9_evidence.GATE_SELECTION_SEED,
                "generation_seed": c9_evidence.GATE_GENERATION_SEED,
                "temperature": c9_evidence.GATE_TEMPERATURE,
                "top_p": c9_evidence.GATE_TOP_P,
                "top_k": c9_evidence.GATE_TOP_K,
                "min_p": c9_evidence.GATE_MIN_P,
                "max_completion_tokens": c9_evidence.GATE_MAX_COMPLETION_TOKENS,
            },
            "thresholds": _c9_gate_thresholds(),
        },
        "artifact_durability_contract": {
            "online_wandb_required": True,
            "grpo_final_artifact_every_steps": int(
                args.ckpt_artifact_every),
            "durable_scout_sft_central_required": True,
        },
        "code_artifact": _c9_code_identity(args),
    }
    payload["contract_sha256"] = common.canonical_json_sha256(payload)
    return payload


def _c9_canary_execution_contract(args: argparse.Namespace) -> Dict[str, Any]:
    """Bind a resumable C9 mechanism screen before Scout update one.

    Canary runs intentionally stop after the fixed step-50 Scout endpoint, so
    their execution contract must not consult SFT/Central arguments.  Keeping
    this separate from the full/C5 contract also makes an old, unbound v31
    canary ineligible for promotion merely because its output directory still
    exists.
    """
    payload = {
        "version": c9_evidence.PIPELINE_EXECUTION_CONTRACT_VERSION,
        "run_name": str(args.run_name),
        "output_dir": os.path.realpath(args.output_dir),
        "c9_execution_profile": str(args.c9_execution_profile),
        "stabilization_cell": str(args.stabilization_cell),
        "scientific_variant": str(args.scientific_variant),
        "model": str(args.model),
        "model_revision": str(args.model_revision or ""),
        "dataset": str(args.dataset),
        "seed": int(args.seed),
        "rounds": int(args.rounds),
        "lambda_schedule": parse_lambda_schedule(args),
        "scout_steps": int(args.scout_steps),
        "scout_only": bool(args.scout_only),
        "scout_runtime_contract": _c9_scout_runtime_contract(
            args, scout_init=str(args.model), round_idx=1, lam=0.5),
        "behavior_gate_contract": {
            "metric_contract_version": sft_gate_gpu.METRIC_CONTRACT_VERSION,
            "gate_stage": "scout_step50",
            "sampling": {
                "num_prompts": c9_evidence.GATE_NUM_PROMPTS,
                "samples_per_prompt": c9_evidence.GATE_SAMPLES_PER_PROMPT,
                "selection_seed": c9_evidence.GATE_SELECTION_SEED,
                "generation_seed": c9_evidence.GATE_GENERATION_SEED,
                "temperature": c9_evidence.GATE_TEMPERATURE,
                "top_p": c9_evidence.GATE_TOP_P,
                "top_k": c9_evidence.GATE_TOP_K,
                "min_p": c9_evidence.GATE_MIN_P,
                "max_completion_tokens":
                    c9_evidence.GATE_MAX_COMPLETION_TOKENS,
            },
            "thresholds": _c9_gate_thresholds(),
        },
        "code_artifact": _c9_code_identity(args),
    }
    payload["contract_sha256"] = common.canonical_json_sha256(payload)
    return payload


def _c10_canary_execution_contract(args: argparse.Namespace) -> Dict[str, Any]:
    """Bind the isolated C10 mechanism screen before Scout update one."""
    runtime = _c9_scout_runtime_contract(
        args, scout_init=str(args.model), round_idx=1, lam=0.5)
    c10_evidence.validate_runtime_contract(runtime)
    lease = c10_evidence.load_exclusive_lease_binding(
        args.c10_exclusive_lease_binding,
        expected_orchestrator_binding=args._orchestrator_receipt_binding)
    payload = {
        "version": c10_evidence.C10_PIPELINE_EXECUTION_CONTRACT_VERSION,
        "run_name": str(args.run_name),
        "output_dir": os.path.realpath(args.output_dir),
        "c10_execution_profile": str(args.c10_execution_profile),
        "scientific_variant": str(args.scientific_variant),
        "preregistration_file": c10_evidence.C10_PREREGISTRATION_FILE,
        "preregistration_sha256": c10_evidence.C10_PREREGISTRATION_SHA256,
        "model": str(args.model),
        "model_revision": str(args.model_revision or ""),
        "dataset": str(args.dataset),
        "seed": int(args.seed),
        "rounds": int(args.rounds),
        "lambda_schedule": parse_lambda_schedule(args),
        "scout_steps": int(args.scout_steps),
        "scout_only": bool(args.scout_only),
        "scout_runtime_contract": runtime,
        "behavior_gate_contract": {
            "metric_contract_version": sft_gate_gpu.METRIC_CONTRACT_VERSION,
            "gate_stage": "scout_step50",
            "sampling": {
                "num_prompts": c10_evidence.GATE_NUM_PROMPTS,
                "samples_per_prompt": c10_evidence.GATE_SAMPLES_PER_PROMPT,
                "selection_seed": c10_evidence.GATE_SELECTION_SEED,
                "generation_seed": c10_evidence.GATE_GENERATION_SEED,
                "temperature": c10_evidence.GATE_TEMPERATURE,
                "top_p": c10_evidence.GATE_TOP_P,
                "top_k": c10_evidence.GATE_TOP_K,
                "min_p": c10_evidence.GATE_MIN_P,
                "max_completion_tokens":
                    c10_evidence.GATE_MAX_COMPLETION_TOKENS,
            },
            "thresholds": dict(c10_evidence.GATE_THRESHOLDS),
        },
        "code_artifact": _c9_code_identity(args),
        "exclusive_lease": lease,
    }
    payload["contract_sha256"] = common.canonical_json_sha256(payload)
    return payload


def _resolved_c9_execution_contract(
    args: argparse.Namespace,
) -> Optional[Dict[str, Any]]:
    profile = str(getattr(args, "c9_execution_profile", "") or "")
    if profile == "full":
        return _c9_pipeline_execution_contract(args)
    if profile == "canary":
        return _c9_canary_execution_contract(args)
    return None


def _load_state(args: argparse.Namespace) -> Dict[str, Any]:
    """pipeline_state.json: {"stages": {"r01/scout": {...outputs}, ...}}."""
    path = os.path.join(args.output_dir, "pipeline_state.json")
    expected_c9 = _resolved_c9_execution_contract(args)
    expected_c10 = (
        _c10_canary_execution_contract(args)
        if str(getattr(args, "c10_execution_profile", "") or "")
        == "canary" else None)
    if args.resume and os.path.exists(path):
        with open(path) as f:
            state = json.load(f)
        if expected_c9 is not None and state.get("c9_execution_contract") != expected_c9:
            raise RuntimeError(
                "C9 resume execution contract differs from pipeline_state; "
                "use the original flags/code/source or a fresh output directory")
        if expected_c10 is not None and state.get(
                "c10_execution_contract") != expected_c10:
            raise RuntimeError(
                "C10 resume execution contract differs from pipeline_state; "
                "use the original flags/code/source or a fresh output directory")
        if expected_c10 is not None:
            terminal_path = os.path.join(
                args.output_dir, "round_01", "scout",
                "c10_training_terminal_outcome.json")
            if os.path.isfile(terminal_path):
                # This guard runs before model loading or sampling.  Validate
                # the local record, then refuse unconditionally: a terminal
                # scientific outcome cannot be replaced merely because its
                # final remote artifact upload was interrupted.
                try:
                    c10_evidence.validate_terminal_training_failure(
                        args.output_dir)
                except ValueError as exc:
                    raise RuntimeError(
                        "C10 resume found malformed/conflicting terminal "
                        "training evidence; refusing to sample") from exc
                raise RuntimeError(
                    "C10 already has a terminal pre-gate scientific failure; "
                    "resume/replacement sampling is forbidden")
            transport_path = os.path.join(
                args.output_dir, "round_01", "scout",
                "c10_training_transport_invalid.json")
            if os.path.isfile(transport_path):
                try:
                    c10_evidence.validate_training_transport_bundle(
                        args.output_dir)
                except ValueError as exc:
                    raise RuntimeError(
                        "C10 resume found malformed/conflicting training "
                        "transport evidence; refusing to sample") from exc
                raise RuntimeError(
                    "C10 transport-invalid attempts are immutable; rerun the "
                    "unchanged canary in a fresh output directory after the "
                    "transport lease is released")
            unstaged_gate_path = os.path.join(
                args.output_dir, "round_01", "c10_behavior_gates",
                "scout_step50", "evidence.json")
            stages = state.get("stages", {})
            if os.path.isfile(unstaged_gate_path) and \
                    "r01/c10_canary_behavior_gate" not in stages:
                try:
                    unstaged = c10_evidence.validate_unstaged_behavior_bundle(
                        args.output_dir)
                except ValueError as exc:
                    raise RuntimeError(
                        "C10 resume found malformed unstaged behavior "
                        "evidence; refusing to sample") from exc
                outcome = unstaged["behavior_gate"]["gate_outcome"]
                if outcome == "transport_invalid":
                    raise RuntimeError(
                        "C10 behavior transport attempts are immutable; rerun "
                        "the unchanged canary in a fresh output directory "
                        "after the transport lease is released")
                raise RuntimeError(
                    "C10 already has an observed terminal fixed-panel outcome "
                    f"({outcome}); replacement sampling is forbidden")
        print(f"[pipeline] resume: {len(state.get('stages', {}))} completed "
              f"stages in {path}", flush=True)
        return state
    if expected_c9 is not None:
        stale = [
            os.path.join(args.output_dir, name)
            for name in ("pipeline_state.json", "run_summary.json", "round_01")
            if os.path.exists(os.path.join(args.output_dir, name))
        ]
        if stale:
            raise RuntimeError(
                "C9 output directory contains prior pipeline state but --resume "
                f"was not requested: {stale}")
        state = {"stages": {}, "c9_execution_contract": expected_c9}
        common.atomic_write_json(path, state)
        return state
    if expected_c10 is not None:
        stale = [
            os.path.join(args.output_dir, name)
            for name in ("pipeline_state.json", "run_summary.json", "round_01")
            if os.path.exists(os.path.join(args.output_dir, name))
        ]
        if stale:
            raise RuntimeError(
                "C10 output directory contains prior pipeline state but "
                f"--resume was not requested: {stale}")
        state = {"stages": {}, "c10_execution_contract": expected_c10}
        common.atomic_write_json(path, state)
        return state
    return {"stages": {}}


def _stage_key(round_idx: int, stage: str) -> str:
    return f"r{round_idx:02d}/{stage}"


def _stage_done(state: Dict[str, Any], round_idx: int, stage: str) -> Optional[Dict[str, Any]]:
    return state["stages"].get(_stage_key(round_idx, stage))


def _mark_stage(args: argparse.Namespace, state: Dict[str, Any], round_idx: int,
                stage: str, **outputs: Any) -> None:
    state["stages"][_stage_key(round_idx, stage)] = {**outputs, "completed_at": time.time()}
    # Atomic (tmp + os.replace): a crash mid-write would leave a torn
    # pipeline_state.json that permanently bricks --resume.
    common.atomic_write_json(
        os.path.join(args.output_dir, "pipeline_state.json"), state)


def _push_results_artifact(wandb_run, args: argparse.Namespace, note: str) -> None:
    """Versioned results artifact after every stage (durable state on W&B)."""
    if not _wandb_artifact_upload_available(wandb_run):
        return
    import wandb

    evidence_profile = _fresh_c8_scout_evidence_profile(args)
    metadata: Dict[str, Any] = {"note": note}
    metadata.update(grpo_gpu._c9_artifact_metadata(args))
    c10_terminal_metadata = getattr(args, "_c10_terminal_metadata", None)
    if isinstance(c10_terminal_metadata, dict):
        metadata.update(c10_terminal_metadata)
    raw_contract = getattr(args, "_mrme_contract", None)
    if isinstance(raw_contract, dict):
        metadata.update(_round_campaign_summary_identity(
            mrme_contract.validate_contract(raw_contract)))
    if evidence_profile == fresh_c8_sft.MULTISCOUT_COMPONENT_PROFILE:
        metadata.update({
            "fresh_c8_evidence_profile": evidence_profile,
            "component_seed": int(args.seed),
        })
    art = wandb.Artifact(common.sanitize_wandb_id(f"{args.run_name}-results"),
                         type="expdis-results", metadata=metadata)
    n = 0
    for pattern in RESULT_ARTIFACT_PATTERNS:
        for path in glob.glob(os.path.join(args.output_dir, pattern)):
            art.add_file(path, name=os.path.relpath(path, args.output_dir))
            n += 1
    if n:
        logged = wandb_run.log_artifact(art)
        # pipeline_state.json is the cross-pod recovery record.  Do not report
        # it durable or advance to the next stage while its upload is merely
        # queued in the SDK's background thread.
        logged.wait()
        print(f"[pipeline] pushed results artifact ({n} files) after {note}", flush=True)


def _durably_publish_c10_training_outcome_if_present(
    args: argparse.Namespace, wandb_run, *, round_idx: int,
) -> Optional[str]:
    """Synchronously persist a validated C10 training outcome before unwind.

    GRPO records its signed semantic/transport evidence before raising.  The
    ordinary launch wrapper later republishes it and, for transport, releases
    the lease.  This nearer persistence boundary closes the long exception-
    unwind window: if the pod dies after this upload, the next bootstrap can
    independently revalidate the durable result and finish the exact release.
    """
    if str(getattr(args, "c10_execution_profile", "") or "") != "canary":
        return None
    scout_dir = os.path.join(
        args.output_dir, f"round_{int(round_idx):02d}", "scout")
    semantic_path = os.path.join(
        scout_dir, "c10_training_terminal_outcome.json")
    transport_path = os.path.join(
        scout_dir, "c10_training_transport_invalid.json")
    present = [
        path for path in (semantic_path, transport_path)
        if os.path.isfile(path)
    ]
    if not present:
        return None
    if len(present) != 1:
        raise RuntimeError(
            "C10 training produced mutually exclusive terminal outcomes")
    if present[0] == semantic_path:
        validated = c10_evidence.validate_terminal_training_failure(
            args.output_dir)
        execution = validated["execution_contract"]
        failure = validated["training_failure"]
        args._c10_terminal_metadata = {
            "c10_outcome": "semantic_failed",
            "c10_terminal_scientific_failure": True,
            "c10_training_failure_kind": failure["failure_kind"],
            "c10_pipeline_execution_contract_sha256": execution[
                "contract_sha256"],
            "c10_training_failure_evidence_sha256": failure[
                "evidence_sha256"],
            "c10_preregistration_sha256":
                c10_evidence.C10_PREREGISTRATION_SHA256,
        }
        outcome = "semantic_failed"
    else:
        validated = c10_evidence.validate_training_transport_bundle(
            args.output_dir)
        execution = validated["execution_contract"]
        transport = validated["training_transport"]
        args._c10_terminal_metadata = {
            "c10_outcome": "transport_invalid",
            "c10_terminal_scientific_failure": False,
            "c10_transport_scope": "training",
            "c10_pipeline_execution_contract_sha256": execution[
                "contract_sha256"],
            "c10_training_transport_evidence_sha256": transport[
                "evidence_sha256"],
            "c10_preregistration_sha256":
                c10_evidence.C10_PREREGISTRATION_SHA256,
        }
        outcome = "transport_invalid"
    _push_results_artifact(
        wandb_run, args,
        f"r{int(round_idx)}-c10-training-{outcome}")
    return outcome


def _sft_input_identity(
    accepted_path: str, *, epochs: int, seed: int,
) -> Dict[str, Any]:
    rows = _read_jsonl_objects(accepted_path, "SFT accepted library")
    return {
        "accepted_library_digest": common.semantic_sft_dataset_digest(rows),
        "accepted_file_sha256": common.file_sha256(accepted_path),
        "training_sequence_digest": common.sft_training_sequence_digest(
            rows, epochs=int(epochs), seed=int(seed)),
    }


def _validate_local_sft_handoff(
    model_dir: str, *, accepted_path: str,
    expected_runtime_contract: Dict[str, Any], epochs: int, seed: int,
) -> Dict[str, Any]:
    """Validate a completed SFT model and every order-sensitive input byte."""
    if not os.path.isdir(model_dir) or not os.path.isfile(
            os.path.join(model_dir, "config.json")):
        raise RuntimeError(f"SFT handoff is not a complete HF model: {model_dir!r}")
    provenance = _load_json_object(
        os.path.join(model_dir, "sft_provenance.json"),
        "completed SFT provenance")
    manifest = provenance.get("checkpoint_file_manifest")
    common.verify_directory_file_manifest(
        model_dir, manifest, exclude_relative_paths=("sft_provenance.json",))
    expected_input = _sft_input_identity(
        accepted_path, epochs=int(epochs), seed=int(seed))
    drift = [
        name for name, expected in expected_input.items()
        if provenance.get(name) != expected
    ]
    if drift:
        raise RuntimeError(
            "SFT handoff input identity mismatch: " + ", ".join(drift))
    if provenance.get("sft_runtime_contract") != expected_runtime_contract:
        raise RuntimeError("SFT handoff runtime contract mismatch")
    completed = provenance.get("completed_run_evidence")
    if not isinstance(completed, dict):
        raise RuntimeError("SFT handoff lacks completed-run evidence")
    unsigned = dict(completed)
    recorded_sha = unsigned.pop("evidence_sha256", None)
    if recorded_sha != common.canonical_json_sha256(unsigned) or \
            completed.get("training_completed") is not True or \
            completed.get("optimizer_steps_executed") != provenance.get(
                "optimizer_steps") or \
            completed.get("micro_steps_executed") != provenance.get("micro_steps"):
        raise RuntimeError("SFT handoff completed-run evidence is invalid")
    return {
        "provenance": provenance,
        "policy_file_manifest": manifest,
        "model_file_manifest": common.directory_file_manifest(model_dir),
        **expected_input,
    }


def _wandb_artifact_upload_available(wandb_run) -> bool:
    if wandb_run is None:
        return False
    mode = str(os.environ.get("WANDB_MODE", "")).strip().lower()
    return mode not in {"offline", "dryrun", "disabled"}


def _artifact_source_from_logged(logged) -> Dict[str, str]:
    ref = str(
        getattr(logged, "qualified_name", "")
        or getattr(logged, "name", "")
        or "")
    common.require_immutable_wandb_artifact_ref(
        ref, field="completed SFT model artifact")
    digest = str(getattr(logged, "digest", "") or "")
    if not digest:
        raise RuntimeError("completed SFT model artifact lacks a remote digest")
    return {"artifact_ref": ref, "artifact_digest": digest}


def _validate_sft_model_artifact_binding(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict) or value.get("version") != \
            SFT_MODEL_ARTIFACT_BINDING_VERSION:
        raise RuntimeError("SFT model artifact binding version mismatch")
    unsigned = dict(value)
    recorded_sha = unsigned.pop("binding_sha256", None)
    if recorded_sha != common.canonical_json_sha256(unsigned):
        raise RuntimeError("SFT model artifact binding SHA256 mismatch")
    manifest = value.get("model_file_manifest")
    if not isinstance(manifest, dict) or manifest.get("version") != \
            common.DIRECTORY_FILE_MANIFEST_VERSION or not manifest.get("files") or \
            not isinstance(manifest.get("sha256"), str):
        raise RuntimeError("SFT model artifact binding lacks a complete manifest")
    durable = value.get("durable_wandb_upload")
    source = value.get("source")
    if durable is True:
        if not isinstance(source, dict) or not source.get("artifact_digest"):
            raise RuntimeError("durable SFT model binding lacks exact source")
        common.require_immutable_wandb_artifact_ref(
            str(source.get("artifact_ref", "")),
            field="SFT model artifact binding source")
    elif durable is False:
        if source is not None:
            raise RuntimeError("local-only SFT model binding has a remote source")
    else:
        raise RuntimeError("SFT model binding lacks durability status")
    if value.get("artifact_subdir") != SFT_MODEL_ARTIFACT_SUBDIR:
        raise RuntimeError("SFT model artifact subdirectory mismatch")
    if value.get("artifact_type") != "sft-model" or not isinstance(
            value.get("artifact_metadata"), dict):
        raise RuntimeError("SFT model artifact identity metadata is incomplete")
    return value


def _validated_grpo_final_artifact_binding(
    stage_output_dir: str, model_dir: str, *, phase: str,
    round_idx: int, steps: int, required: bool,
) -> Optional[Dict[str, Any]]:
    """Re-hash one uploaded GRPO final and its immutable W&B binding."""
    path = os.path.join(stage_output_dir, "final_artifact_binding.json")
    if not os.path.isfile(path):
        if required:
            raise RuntimeError(
                f"{phase} final lacks its durable model-artifact binding")
        return None
    binding = _load_json_object(
        path, f"{phase} final model artifact binding")
    trainer = _load_json_object(
        os.path.join(model_dir, "trainer_state.json"),
        f"{phase} final trainer_state")
    try:
        mrme_contract.validate_grpo_final_artifact_binding(
            binding, expected_phase=phase, expected_round=round_idx,
            expected_step=steps,
            expected_policy_manifest=trainer.get("checkpoint_file_manifest"),
            expected_mrme_provenance=trainer.get("mrme_provenance"))
        common.verify_directory_file_manifest(
            model_dir, binding["model_file_manifest"])
        common.verify_directory_file_manifest(
            model_dir, binding["policy_file_manifest"],
            exclude_relative_paths=("trainer_state.json",))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{phase} final model artifact binding is invalid: {exc}") from exc
    return binding


def _persist_sft_model_artifact(
    wandb_run, args: argparse.Namespace, *, model_dir: str, round_idx: int,
    stage_mode: str, handoff: Dict[str, Any],
) -> Dict[str, Any]:
    """Synchronously persist the complete SFT handoff before Central starts."""
    manifest = handoff["model_file_manifest"]
    source: Optional[Dict[str, str]] = None
    durable = _wandb_artifact_upload_available(wandb_run)
    mrme_metadata: Dict[str, Any] = {}
    raw_mrme_contract = getattr(args, "_mrme_contract", None)
    if isinstance(raw_mrme_contract, dict):
        contract = mrme_contract.validate_contract(raw_mrme_contract)
        shard = _load_json_object(
            os.path.join(
                args.output_dir, f"round_{int(round_idx):02d}",
                "dataset_shard_manifest.json"),
            "MR-ME SFT dataset-shard manifest")
        mrme_metadata = {
            **_round_campaign_summary_identity(contract),
            "method": contract["method"],
            "stabilization_profile": contract["stabilization_profile"],
            "scientific_variant": contract["scientific_variant"],
            "mrme_contract_sha256": contract["contract_sha256"],
            "mrme_shard_manifest_sha256": shard["manifest_sha256"],
            "mrme_scout_index": 0,
            "mrme_scouts_per_round": int(contract["scouts_per_round"]),
            "phase": "sft",
        }
    artifact_metadata = {
        "round": int(round_idx),
        "stage_mode": str(stage_mode),
        "model_file_manifest_sha256": manifest["sha256"],
        "accepted_file_sha256": handoff["accepted_file_sha256"],
        "training_sequence_sha256": handoff[
            "training_sequence_digest"]["sha256"],
        **grpo_gpu._c9_artifact_metadata(args),
        **mrme_metadata,
    }
    if durable:
        import wandb

        artifact = wandb.Artifact(
            common.sanitize_wandb_id(
                f"{args.run_name}-sft-r{int(round_idx)}-final"),
            type="sft-model",
            metadata=artifact_metadata,
        )
        artifact.add_dir(model_dir, name=SFT_MODEL_ARTIFACT_SUBDIR)
        logged = wandb_run.log_artifact(
            artifact, aliases=["latest", f"round-{int(round_idx)}"])
        # Upload failures are fatal and occur before pipeline_state marks SFT
        # complete, so Central can never consume a model that exists only on
        # an evictable worker filesystem.
        logged.wait()
        source = _artifact_source_from_logged(logged)
        if not callable(getattr(logged, "files", None)):
            raise RuntimeError("completed SFT artifact cannot enumerate members")
        remote_names = sorted(str(item.name) for item in logged.files())
        expected_names = sorted(
            f"{SFT_MODEL_ARTIFACT_SUBDIR}/{item['relative_path']}"
            for item in manifest["files"])
        if remote_names != expected_names:
            raise RuntimeError(
                "completed SFT artifact membership mismatch after upload")
        # Detect a local mutation racing the background uploader.
        common.verify_directory_file_manifest(model_dir, manifest)
    payload: Dict[str, Any] = {
        "version": SFT_MODEL_ARTIFACT_BINDING_VERSION,
        "durable_wandb_upload": durable,
        "source": source,
        "artifact_type": "sft-model",
        "artifact_metadata": artifact_metadata,
        "artifact_subdir": SFT_MODEL_ARTIFACT_SUBDIR,
        "model_file_manifest": manifest,
        "policy_file_manifest": handoff["policy_file_manifest"],
        "accepted_library_digest": handoff["accepted_library_digest"],
        "accepted_file_sha256": handoff["accepted_file_sha256"],
        "training_sequence_digest": handoff["training_sequence_digest"],
        "round": int(round_idx),
        "stage_mode": str(stage_mode),
    }
    payload["binding_sha256"] = common.canonical_json_sha256(payload)
    _validate_sft_model_artifact_binding(payload)
    common.atomic_write_json(
        os.path.join(
            args.output_dir, f"round_{int(round_idx):02d}",
            "sft_model_artifact.json"),
        payload,
    )
    return payload


def _restore_or_validate_sft_model(
    args: argparse.Namespace, stage: Dict[str, Any], *, accepted_path: str,
    expected_runtime_contract: Dict[str, Any], round_idx: int,
) -> tuple[str, Dict[str, Any]]:
    """Verify local SFT bytes, or restore the exact uploaded artifact."""
    binding = _validate_sft_model_artifact_binding(
        stage.get("sft_model_artifact"))
    model_dir = str(stage.get("final", ""))
    if not os.path.isdir(model_dir):
        if binding["durable_wandb_upload"] is not True:
            raise RuntimeError(
                "completed SFT model is missing and has no durable W&B source")
        source = binding["source"]
        import wandb

        artifact = wandb.Api().artifact(source["artifact_ref"])
        if str(getattr(artifact, "digest", "") or "") != \
                source["artifact_digest"]:
            raise RuntimeError("restored SFT artifact digest mismatch")
        tmp_root = f"{model_dir}.artifact-download.tmp"
        shutil.rmtree(tmp_root, ignore_errors=True)
        os.makedirs(tmp_root, exist_ok=False)
        try:
            artifact.download(root=tmp_root)
            downloaded = os.path.join(tmp_root, SFT_MODEL_ARTIFACT_SUBDIR)
            downloaded_root_manifest = common.directory_file_manifest(tmp_root)
            expected_downloaded_files = [
                {
                    **item,
                    "relative_path":
                        f"{SFT_MODEL_ARTIFACT_SUBDIR}/{item['relative_path']}",
                }
                for item in binding["model_file_manifest"]["files"]
            ]
            if downloaded_root_manifest["files"] != expected_downloaded_files:
                raise RuntimeError(
                    "restored SFT artifact has missing or extra members")
            common.verify_directory_file_manifest(
                downloaded, binding["model_file_manifest"])
            os.makedirs(os.path.dirname(model_dir), exist_ok=True)
            os.replace(downloaded, model_dir)
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)
    handoff = _validate_local_sft_handoff(
        model_dir, accepted_path=accepted_path,
        expected_runtime_contract=expected_runtime_contract,
        epochs=int(expected_runtime_contract["epochs"]),
        seed=int(expected_runtime_contract["seed"]))
    if handoff["model_file_manifest"] != binding["model_file_manifest"] or \
            handoff["policy_file_manifest"] != binding[
                "policy_file_manifest"] or \
            handoff["accepted_library_digest"] != binding[
                "accepted_library_digest"] or \
            handoff["accepted_file_sha256"] != binding[
                "accepted_file_sha256"] or \
            handoff["training_sequence_digest"] != binding[
                "training_sequence_digest"] or \
            binding.get("round") != int(round_idx):
        raise RuntimeError("completed SFT model differs from pipeline artifact binding")
    return model_dir, binding


def _num_scouts(args: argparse.Namespace) -> int:
    """K Scouts feeding the one filter/SFT/Central handoff."""
    if getattr(args, "_mrme_internal_pool", False):
        return int(args.scouts_per_round)
    if args.skip_scout and args.trajectories:
        return len(args.trajectories)
    return 1


def _reject_multiscout_component_local_trajectory_inputs(
    paths: List[str],
) -> None:
    """Keep official Scout components out of the generic local pool route.

    The W&B launcher rejects component artifact metadata and signed completion
    evidence before download.  ``pipeline_gpu.py --skip-scout`` is also a
    public direct/local entrypoint, however, so it must independently inspect
    both the completion-evidence sibling and provenance embedded in official
    trajectory rows.  A copied but otherwise unmodified official JSONL thus
    remains identifiable even after its artifact directory is discarded.

    Historical/canonical files have neither the component row profile nor
    component completion evidence and remain valid generic inputs.  A caller
    that strips or changes these markers has destroyed strict provenance; the
    resulting bytes are generic/noncanonical and are not attributed to the
    preregistered component profile.
    """
    evidence_name = "fresh_c8_scout_completion_evidence.json"
    for raw_path in paths:
        path = os.path.realpath(str(raw_path))
        if not os.path.isfile(path):
            # Preserve the existing downstream missing-input error.  This
            # guard is specifically an attribution boundary, not a duplicate
            # trajectory-file validator.
            continue

        sibling = os.path.join(os.path.dirname(path), evidence_name)
        if os.path.isfile(sibling):
            evidence = _load_json_object(
                sibling, "local Scout completion evidence")
            fresh_c8_sft.reject_multiscout_component_from_generic_pool(
                artifact_metadata={}, completion_evidence=[evidence])

        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except (TypeError, ValueError):
                    # ``filter_pool.load_trajectories`` skips malformed rows;
                    # mirror that behavior instead of changing legacy input
                    # semantics in this provenance-only preflight.
                    continue
                if not isinstance(row, dict):
                    continue
                fresh_c8_sft.reject_multiscout_component_from_generic_pool(
                    artifact_metadata=row, completion_evidence=[])


def _resolved_dynamic_max_attempts(args: argparse.Namespace) -> int:
    """Resolve the pipeline sentinel exactly as each GRPO stage will."""
    value = int(args.dynamic_max_attempts)
    if value <= 0:
        return 64 if args.rl_validity_mode == "c8" else 8
    return value


def _grpo_args(args: argparse.Namespace, *, phase: str, model: str, output_dir: str,
               lam: float, steps: int, round_idx: int,
               global_step_offset: int,
               external_lineage_json: str = "",
               seed_override: Optional[int] = None,
               run_name_override: Optional[str] = None,
               sync_dir_override: Optional[str] = None,
               num_scouts_override: Optional[int] = None) -> argparse.Namespace:
    # DAPO dynamic sampling resamples until the requested informative-group
    # batch is full. A collapsed policy fails visibly at this ceiling instead
    # of hanging; legacy retains its historical eight-attempt behavior.
    dynamic_max_attempts = _resolved_dynamic_max_attempts(args)
    strict_training_environment = bool(
        str(getattr(args, "c9_execution_profile", "") or "")
        or str(getattr(args, "c10_execution_profile", "") or "")
        or str(getattr(args, "c9_multiexplorer_profile", "") or "")
        or str(getattr(
            args, "single_scout_multiround_profile", "") or "")
        or str(getattr(args, "expdis_stabilization_mode", "") or "")
        == grpo_gpu.C9_RND_LIFECYCLE
    )
    argv = [
        "--model", model,
        "--model-revision", str(args.model_revision or ""),
        "--output-dir", output_dir,
        "--run-name", run_name_override or args.run_name,
        "--orchestrator-run-name", args.run_name,
        "--phase", phase,
        "--container-image-identity", str(
            getattr(args, "container_image_identity", "") or ""),
        "--steps", str(int(steps)),
        "--lambda-novelty", str(float(lam)),
        "--expdis-stabilization-mode", args.expdis_stabilization_mode,
        "--novelty-normalization", args.novelty_normalization,
        "--c9-execution-profile", args.c9_execution_profile,
        "--c10-execution-profile", args.c10_execution_profile,
        "--stabilization-cell", args.stabilization_cell,
        "--prompts-per-step", str(int(args.prompts_per_step)),
        "--num-generations", str(int(args.num_generations)),
        "--rl-validity-mode", args.rl_validity_mode,
        "--reload-every", str(int(args.reload_every)),
        "--save-every", str(int(args.save_every)),
        "--keep-checkpoints", str(int(args.keep_checkpoints)),
        "--ckpt-artifact-every", str(int(args.ckpt_artifact_every)),
        "--probe-every", str(int(args.probe_every)),
        "--probe-dataset", args.probe_dataset,
        "--dataset", args.dataset,
        "--max-train-examples", str(int(args.max_train_examples)),
        "--sync-dir", sync_dir_override or args.sync_dir,
        "--device", args.device,
        "--param-dtype", args.param_dtype,
        "--seed", str(
            int(seed_override) if seed_override is not None
            else int(args.seed) + (round_idx - 1)),
        "--round", str(int(round_idx)),
        "--served-model-name", args.served_model_name or args.model,
        "--gen-concurrency", str(int(args.gen_concurrency)),
        "--novelty-layers", args.novelty_layers,
        "--max-consecutive-gen-failures", str(int(args.max_consecutive_gen_failures)),
        "--dynamic-max-attempts", str(dynamic_max_attempts),
        "--metrics-jsonl", os.path.join(args.output_dir, "metrics.jsonl"),
        "--global-step-offset", str(int(global_step_offset)),
        "--num-scouts", str(
            int(num_scouts_override) if num_scouts_override is not None
            else _num_scouts(args)),
        # The pipeline's resolved (probe/budget-validated) pool sizing, so the
        # stage's own probe-sizing guard sees the real serving window.
        "--vllm-max-model-len", str(int(args.vllm_max_model_len)),
        # GPU-MAX tier pass-through (contract defaults leave these inert).
        "--completion-budget", str(int(args.completion_budget)),
        "--updates-per-rollout", str(int(args.updates_per_rollout)),
        "--optimizer", args.optimizer,
        "--scientific-variant", args.scientific_variant,
        "--mask-truncated" if bool(args.mask_truncated) else "--no-mask-truncated",
        "--truncation-baseline-mode", args.truncation_baseline_mode,
    ]
    if strict_training_environment:
        argv += [
            "--orchestrator-receipt-binding",
            str(getattr(args, "orchestrator_receipt_binding", "") or ""),
            "--orchestrator-command-sha256",
            str(getattr(args, "orchestrator_command_sha256", "") or ""),
            "--orchestrator-receipt-artifact",
            str(getattr(args, "orchestrator_receipt_artifact", "") or ""),
        ]
    phase_lr = args.scout_lr if phase == "scout" else args.central_lr
    if phase_lr is not None:
        argv += ["--lr", str(float(phase_lr))]
    fresh_evidence_profile = _fresh_c8_scout_evidence_profile(args)
    if (fresh_evidence_profile or strict_training_environment or
            args.expdis_stabilization_mode == grpo_gpu.C9_RND_LIFECYCLE or
            getattr(args, "_mrme_contract", None)) \
            and (phase == "scout" or strict_training_environment):
        argv += ["--code-source-manifest", args.code_source_manifest]
    if external_lineage_json:
        argv += ["--external-lineage-json", external_lineage_json]
    if args.resume:
        # Crash recovery: restart an interrupted stage from its newest local
        # step checkpoint (run_grpo trims stale JSONL/metric rows past it).
        ckpts = sorted(glob.glob(os.path.join(output_dir, "step_[0-9]*")))
        ckpts = [c for c in ckpts if os.path.exists(os.path.join(c, "trainer_state.json"))]
        if ckpts:
            argv += ["--resume-from", ckpts[-1]]
    if args.ppo_mode:
        argv.append("--ppo-mode")
    if args.wandb:
        argv.append("--wandb")
    parsed = grpo_gpu.build_arg_parser().parse_args(argv)
    multi_contract = getattr(args, "_mrme_contract", None)
    c9_round_campaign = _is_c9_round_campaign(args)
    if c9_round_campaign:
        # The pipeline-level profile is deliberately separate from
        # SingleScout full/C5.  Each child GRPO stage still opts into the same
        # strict probe mechanics, while the distinct scientific variant and
        # campaign binding prevent relabeling it as preregistered SingleScout.
        parsed.c9_execution_profile = "full"
        parsed.stabilization_cell = "C5"
        parsed.scientific_variant = \
            mrme_contract.C9_MULTIEXPLORER_SCIENTIFIC_VARIANT
    if isinstance(multi_contract, dict):
        parsed.mrme_contract_json = os.path.join(
            args.output_dir, "mrme_contract.json")
        parsed.mrme_contract_sha256 = multi_contract["contract_sha256"]
        parsed.mrme_method = multi_contract["method"]
        parsed.mrme_method_version = str(
            multi_contract.get("method_version", "") or "")
        parsed.single_scout_multiround = \
            mrme_contract.is_single_scout_multiround(multi_contract)
        parsed.mrme_stabilization_profile = multi_contract[
            "stabilization_profile"]
        # The slot-specific fields are overwritten by the Scout fan-out.  A
        # Central has index zero but remains campaign-bound.
        parsed.mrme_scout_index = int(
            getattr(parsed, "mrme_scout_index", 0) or 0)
        parsed.mrme_scouts_per_round = int(args.scouts_per_round)
        parsed.mrme_shard_manifest_sha256 = str(
            getattr(args, "_mrme_shard_manifest_sha256", "") or "")
    if fresh_evidence_profile and phase == "scout":
        # Provenance-only W&B config label; the signed completion evidence is
        # still the fail-closed authority for this profile.
        parsed.fresh_c8_evidence_profile = fresh_evidence_profile
    if strict_training_environment:
        for attr in (
                "_training_execution_environment",
                "_orchestrator_receipt_binding",
                "_code_artifact_binding"):
            if hasattr(args, attr):
                setattr(parsed, attr, getattr(args, attr))
    return parsed


def _external_central_runtime_contract(args: argparse.Namespace) -> Dict[str, Any]:
    """Resolve the exact 100-step external-Central contract for lineage.

    This uses the same argument builder as Stage 5, then applies completion
    and probe-window resolution in the same order as ``run_grpo``.  The
    latter recomputes this object independently and requires exact equality.
    """
    cargs = _grpo_args(
        args, phase="central", model=str(args.sft_model),
        output_dir=os.path.join(args.output_dir, "round_01", "central"),
        lam=0.0, steps=int(args.central_steps), round_idx=1,
        global_step_offset=int(args.scout_steps),
    )
    grpo_gpu.apply_completion_budget(cargs)
    grpo_gpu.enforce_probe_serving_len(cargs, pool=None)
    resolved_lr = float(cargs.lr) if cargs.lr is not None else 1e-6
    resolved_steps = int(cargs.steps) if cargs.steps is not None else 100
    return grpo_gpu.build_runtime_contract(
        cargs, resolved_lr=resolved_lr, resolved_steps=resolved_steps)


def _resolved_central_runtime_contract(
    args: argparse.Namespace, *, model: str, round_idx: int,
    global_step_offset: int,
) -> Dict[str, Any]:
    cargs = _grpo_args(
        args, phase="central", model=model,
        output_dir=os.path.join(
            args.output_dir, f"round_{round_idx:02d}", "central"),
        lam=0.0, steps=int(args.central_steps), round_idx=round_idx,
        global_step_offset=global_step_offset,
    )
    grpo_gpu.apply_completion_budget(cargs)
    grpo_gpu.enforce_probe_serving_len(cargs, pool=None)
    return grpo_gpu.build_runtime_contract(
        cargs,
        resolved_lr=(float(cargs.lr) if cargs.lr is not None else 1e-6),
        resolved_steps=int(cargs.steps),
    )


def _sft_args(args: argparse.Namespace, *, accepted: str, output_dir: str,
              round_idx: int, lam: float, global_step_offset: int) -> argparse.Namespace:
    argv = [
        "--model", args.model,  # Central SFT always initializes from BASE (JAX-faithful)
        "--model-revision", str(args.model_revision or ""),
        "--accepted", accepted,
        "--output-dir", output_dir,
        "--run-name", args.run_name,
        "--orchestrator-run-name", args.run_name,
        "--epochs", str(int(args.sft_epochs)),
        "--lr", str(float(args.sft_lr)),
        "--grad-accum", str(int(args.sft_grad_accum)),
        "--sft-termination-mode", args.sft_termination_mode,
        "--device", args.device,
        "--param-dtype", args.param_dtype,
        "--seed", str(int(args.seed) + (round_idx - 1)),
        "--round", str(int(round_idx)),
        "--round-lambda", str(float(lam)),
        "--metrics-jsonl", os.path.join(args.output_dir, "metrics.jsonl"),
        "--global-step-offset", str(int(global_step_offset)),
        "--num-scouts", str(_num_scouts(args)),
        "--optimizer", args.sft_optimizer,
        "--completion-budget", str(int(args.completion_budget)),
    ]
    strict_training_environment = bool(
        str(getattr(args, "c9_execution_profile", "") or "")
        or str(getattr(args, "c9_multiexplorer_profile", "") or "")
        or str(getattr(
            args, "single_scout_multiround_profile", "") or "")
        or str(getattr(args, "expdis_stabilization_mode", "") or "")
        == grpo_gpu.C9_RND_LIFECYCLE
    )
    if strict_training_environment:
        argv += [
            "--container-image-identity", str(
                getattr(args, "container_image_identity", "") or ""),
            "--code-source-manifest", str(
                getattr(args, "code_source_manifest", "") or ""),
            "--orchestrator-receipt-binding", str(
                getattr(args, "orchestrator_receipt_binding", "") or ""),
            "--orchestrator-command-sha256", str(
                getattr(args, "orchestrator_command_sha256", "") or ""),
            "--orchestrator-receipt-artifact", str(
                getattr(args, "orchestrator_receipt_artifact", "") or ""),
            "--require-canonical-training-environment",
        ]
    if args.sft_save_steps:
        argv += ["--sft-save-steps", ",".join(str(x) for x in args.sft_save_steps)]
    if args.trajectory_source_manifest:
        argv += ["--trajectory-source-manifest", args.trajectory_source_manifest]
    if args.sft_training_evidence_manifest:
        argv += [
            "--sft-training-evidence-manifest",
            args.sft_training_evidence_manifest,
        ]
    if args.wandb:
        argv.append("--wandb")
    parsed = sft_gpu.build_arg_parser().parse_args(argv)
    if strict_training_environment:
        for attr in (
                "_training_execution_environment",
                "_orchestrator_receipt_binding",
                "_code_artifact_binding"):
            if hasattr(args, attr):
                setattr(parsed, attr, getattr(args, attr))
    return parsed


def _ensure_harvest_policy(pool: VLLMPool, expected_policy: Optional[str],
                           *, strict: bool) -> None:
    """Harvest top-up must sample the FINAL scout policy — never whatever the
    pool happens to serve (the BASE model, after a resume that skipped the
    scout stage; its rows would silently poison accepted.jsonl / Central SFT).

    Reloads a managed pool to expected_policy when it serves anything else.
    strict=True (resume path: scout skipped via pipeline_state): refuse to
    generate if the pool cannot be brought to the recorded scout-final
    checkpoint (attach-mode pool, or the checkpoint dir is gone) — a hard
    error, never silent wrong-policy data.
    """
    if expected_policy is None:
        return
    if not pool.managed:
        msg = ("harvest top-up requires the pool to serve the final scout "
               f"policy ({expected_policy}), but an attach-mode pool cannot "
               "be reloaded")
        if strict:
            raise RuntimeError(
                f"{msg}; refusing to harvest at the wrong policy — use a "
                "managed pool (--vllm-gpus), point the attached servers at "
                "the scout-final weights, or provide enough trajectories")
        print(f"[pipeline] WARNING: {msg}; harvest rows come from the "
              "currently served policy", flush=True)
        return
    if os.path.abspath(str(pool.model_dir)) == os.path.abspath(str(expected_policy)):
        return
    if not os.path.isdir(expected_policy):
        raise RuntimeError(
            f"recorded scout-final checkpoint {expected_policy} does not exist "
            "on this workspace; restore it (W&B checkpoint artifact) before "
            "resuming harvest — refusing to top-up at the wrong policy")
    print(f"[pipeline] reloading pool to the recorded scout-final policy: "
          f"{expected_policy}", flush=True)
    pool.reload(expected_policy)  # raises (after one retry) if it cannot


def _harvest(args: argparse.Namespace, pool: VLLMPool, tokenizer, examples,
             traj_paths: List[str], harvest_path: str, round_idx: int,
             *, expected_policy: Optional[str] = None,
             strict_policy: bool = False) -> List[str]:
    """Top-up generation at the FINAL scout policy, T=1.0.

    expected_policy (the recorded scout-final checkpoint) is enforced via
    _ensure_harvest_policy before ANY generation; None only for --skip-scout
    pooled runs, where harvest samples the served (base) policy by design.

    best_scout never tops up: harvest rows are generated at the served policy
    (the base model under --skip-scout), are excluded from the best_scout
    competition in stage 3, and therefore cannot raise its accepted count.
    Failed generations are skipped — error rows never enter trajectory files;
    a pass that produced ZERO rows never registers the harvest file (a
    registered-but-unwritten file would crash _policy_survivors/filter_pool).
    """
    token_cap = _token_gate_cap(args)
    survivors = _policy_survivors(traj_paths, args.policy, token_cap)
    print(f"[pipeline] harvest check: {survivors} policy-survivors "
          f"(need {args.min_accepted})", flush=True)
    if survivors >= int(args.min_accepted):
        return traj_paths
    if args.policy == "best_scout":
        print("[pipeline] WARNING: best_scout below --min-accepted but harvest "
              "top-up cannot help (harvest rows are not scout rows); stage 3 "
              "will raise if the best scout stays short", flush=True)
        return traj_paths
    _ensure_harvest_policy(pool, expected_policy, strict=strict_policy)
    rng = np.random.default_rng(int(args.seed) + 9973 * round_idx)
    out_paths = list(traj_paths)
    # Effective completion budget (contract 16384 -> overlong 13107/3277).
    budget = int(args.completion_budget) or 16384
    expected_len = int(0.8 * budget)
    cache_len = budget - expected_len
    for pass_idx in range(1, int(args.harvest_max_passes) + 1):
        # Resurrect dead/hung vLLM servers between passes (a server that dies
        # mid-harvest would otherwise fail every remaining request).
        pool.health_check()
        idxs = rng.choice(len(examples), size=min(int(args.harvest_prompts_per_pass),
                                                  len(examples)), replace=False).tolist()
        batch = [examples[i] for i in idxs]
        print(f"[pipeline] harvest pass {pass_idx}/{args.harvest_max_passes}: "
              f"{len(batch)} prompts x {args.num_generations}", flush=True)
        completions = pool.generate(
            [e.prompt_text for e in batch],
            n=int(args.num_generations),
            max_tokens=budget, temperature=1.0, top_p=0.95, top_k=20,
            seed_base=(int(args.seed) + 7919 * round_idx + pass_idx * 613) % (2 ** 31),
            concurrency_per_server=int(args.gen_concurrency),
        )
        records = []
        for ex, comps in zip(batch, completions):
            ok_comps = [c for c in comps if not completion_failed(c)]
            rows = grpo_gpu.score_completions(
                tokenizer, ex, ok_comps,
                max_completion_len=budget,
                soft_overlong_expected_len=expected_len,
                soft_overlong_cache_len=cache_len,
            )
            for row in rows:
                records.append(grpo_gpu.trajectory_record(row, -pass_idx))
        if records:
            common.append_jsonl(harvest_path, records)
        else:
            print(f"[pipeline] WARNING: harvest pass {pass_idx} produced 0 "
                  "scored rows (all generations failed?)", flush=True)
        if harvest_path not in out_paths and os.path.exists(harvest_path):
            out_paths.append(harvest_path)
        survivors = _policy_survivors(out_paths, args.policy, token_cap)
        print(f"[pipeline] harvest pass {pass_idx} done: {survivors} policy-survivors", flush=True)
        if survivors >= int(args.min_accepted):
            break
    return out_paths


# ------------------------------------------------------------------ round ----

def run_round(args: argparse.Namespace, pool: VLLMPool, tokenizer, examples,
              round_idx: int, lam: float, scout_init: str,
              wandb_run, metrics_logger: common.MetricsLogger,
              state: Dict[str, Any]) -> Dict[str, Any]:
    """One ExpDis round (stages 1-5); with --resume, stages recorded complete in
    pipeline_state.json are skipped and their outputs reused."""
    round_dir = os.path.join(args.output_dir, f"round_{round_idx:02d}")
    os.makedirs(round_dir, exist_ok=True)
    summary: Dict[str, Any] = {"round": round_idx, "lambda": lam, "scout_init": scout_init}
    # Monotonic global steps. True MR-ME logs all K Scout learners before the
    # one Central learner in each round; ordinary runs retain the historical
    # one-Scout span exactly.
    round_scouts = (
        int(args.scouts_per_round)
        if getattr(args, "_mrme_internal_pool", False) else 1)
    round_span = round_scouts * int(args.scout_steps) + int(args.central_steps)
    scout_offset = (round_idx - 1) * round_span
    central_offset = scout_offset + round_scouts * int(args.scout_steps)
    frozen_sft_training = bool(args.frozen_sft_library_evidence_index)
    fresh_sft_training = bool(args.fresh_c8_sft_gate)
    fresh_scout_evidence_profile = _fresh_c8_scout_evidence_profile(args)
    c9_full = _is_c9_full(args)
    multi_contract = getattr(args, "_mrme_contract", None)
    c9_multi = isinstance(multi_contract, dict) and \
        mrme_contract.is_c9_multi_explorer(multi_contract)
    external_profile = (
        _external_evidence_profile(args) if args.sft_model else None)
    fresh_external = external_profile == fresh_c8_sft.PROFILE

    def log(payload: Dict[str, Any]) -> None:
        if wandb_run is not None:
            wandb_run.log({"round": round_idx, "lambda": lam, **payload})

    # ---- Stage 1: Scout GRPO -------------------------------------------------
    skip_scout = bool(args.skip_scout) and (
        round_idx == 1 or getattr(args, "_mrme_internal_pool", False))
    # scout_resumed: the scout stage was skipped via pipeline_state (resume) —
    # the pool then still serves whatever it served at pipeline start (the
    # BASE model), NOT the scout-final policy; harvest must reload it first.
    scout_resumed = False
    done = _stage_done(state, round_idx, "scout")
    if skip_scout:
        traj_paths = list(args.trajectories or [])
        if not traj_paths:
            raise ValueError("--skip-scout requires --trajectories")
        # Defense in depth: validate_args performs the same preflight, but a
        # long-lived caller could mutate args or replace local bytes before
        # run_round.  Recheck immediately before any harvest/filter/SFT work.
        _reject_multiscout_component_local_trajectory_inputs(traj_paths)
        print(f"[pipeline] round {round_idx}: skipping scout; "
              f"pooling {len(traj_paths)} trajectory files", flush=True)
        log({"pipeline/stage": 1, "pipeline/scout_skipped": 1.0})
    elif done:
        if fresh_scout_evidence_profile:
            _validate_persisted_fresh_c8_scout_completion(
                args, done, profile=fresh_scout_evidence_profile)
        if c9_full:
            _validate_persisted_c9_scout_completion(
                args, done, scout_init=scout_init, round_idx=round_idx,
                lam=lam)
            scout_artifact = _validated_grpo_final_artifact_binding(
                os.path.join(round_dir, "scout"), str(done["final"]),
                phase="scout", round_idx=round_idx,
                steps=int(args.scout_steps), required=True)
            if done.get("model_artifact") != scout_artifact:
                raise RuntimeError(
                    "completed C9 Scout pipeline-state artifact binding mismatch")
        print(f"[pipeline] round {round_idx}: scout already complete "
              f"({done['final']})", flush=True)
        summary["scout_final"] = done["final"]
        traj_paths = list(done["traj_paths"])
        scout_resumed = True
    else:
        log({"pipeline/stage": 1})
        scout_dir = os.path.join(round_dir, "scout")
        gargs = _grpo_args(args, phase="scout", model=scout_init, output_dir=scout_dir,
                           lam=lam, steps=int(args.scout_steps), round_idx=round_idx,
                           global_step_offset=scout_offset)
        if fresh_scout_evidence_profile:
            _prepare_fresh_c8_scout_code_binding(args, gargs)
            _push_results_artifact(
                wandb_run, args, f"r{round_idx}-scout-code-binding")
        try:
            scout_final = grpo_gpu.run_grpo(gargs, pool=pool)
        except BaseException:
            # C10 is a true one-shot lease.  Persist any fully validated
            # training-side terminal/transport bundle synchronously before
            # exception unwinding reaches the outer launcher.  Unknown or
            # partial failures remain incomplete and deliberately keep the
            # lease active.
            _durably_publish_c10_training_outcome_if_present(
                args, wandb_run, round_idx=round_idx)
            raise
        summary["scout_final"] = scout_final
        traj_paths = [os.path.join(scout_dir, "explorer_trajectories.jsonl")]
        scout_stage_outputs: Dict[str, Any] = {
            "final": scout_final, "traj_paths": traj_paths}
        if fresh_scout_evidence_profile:
            scout_completion_evidence, scout_completion_evidence_path = \
                _record_fresh_c8_scout_completion(
                    args, scout_final=scout_final,
                    trajectory_path=traj_paths[0],
                    profile=fresh_scout_evidence_profile)
            scout_stage_outputs.update({
                "scout_completion_evidence_path": scout_completion_evidence_path,
                "scout_completion_evidence_sha256": scout_completion_evidence[
                    "evidence_sha256"],
            })
        if c9_full:
            c9_completion, c9_completion_path = _record_c9_scout_completion(
                args, scout_final=scout_final,
                trajectory_path=traj_paths[0], scout_init=scout_init,
                round_idx=round_idx, lam=lam)
            scout_stage_outputs.update({
                "c9_scout_completion_evidence_path": c9_completion_path,
                "c9_scout_completion_evidence_sha256": c9_completion[
                    "evidence_sha256"],
            })
            scout_stage_outputs["model_artifact"] = \
                _validated_grpo_final_artifact_binding(
                    scout_dir, scout_final, phase="scout",
                    round_idx=round_idx, steps=int(args.scout_steps),
                    required=True)
        _mark_stage(args, state, round_idx, "scout", **scout_stage_outputs)
        _push_results_artifact(wandb_run, args, f"r{round_idx}-scout")
    summary["trajectories"] = list(traj_paths)

    # The preregistered stabilization screen is not authorized by loss/clip
    # telemetry alone.  Before a step-50 cell may be considered for C5, run
    # the same fixed, output-independent diagnostic panel used by the full C9
    # handoffs and bind its raw rollouts to the exact Scout/code/runtime.
    if (str(getattr(args, "c9_execution_profile", "") or "") == "canary"
            or str(getattr(args, "c10_execution_profile", "") or "")
            == "canary"):
        c10_canary = str(args.c10_execution_profile or "") == "canary"
        scout_stage = _stage_done(state, round_idx, "scout")
        if not isinstance(scout_stage, dict):
            raise RuntimeError("stabilization canary lacks a completed Scout stage")
        endpoint = (
            _validate_c10_canary_scout_endpoint(
                args, scout_stage, scout_init=scout_init,
                round_idx=round_idx, lam=lam)
            if c10_canary else _validate_c9_canary_scout_endpoint(
                args, scout_stage, scout_init=scout_init,
                round_idx=round_idx, lam=lam)
        )
        endpoint_path = os.path.join(round_dir, "c10_endpoint_replay.json") \
            if c10_canary else ""
        if c10_canary:
            _write_or_verify_json(
                endpoint_path, endpoint, label="C10 endpoint replay")
            c10_evidence.validate_endpoint_replay(endpoint)
            endpoint_sha = str(endpoint["evidence_sha256"])
            endpoint_file_sha = common.file_sha256(endpoint_path)
        else:
            endpoint_sha = common.canonical_json_sha256(endpoint)
            endpoint_file_sha = ""
        gate_stage_name = (
            "c10_canary_behavior_gate" if c10_canary
            else "canary_behavior_gate")
        gate_done = _stage_done(state, round_idx, gate_stage_name)
        if gate_done:
            canary_gate = _validate_persisted_c9_behavior_gate(
                args,
                evidence_path=str(gate_done.get("evidence_path", "")),
                stage="scout_step50",
                model_dir=str(scout_stage["final"]),
                accepted_path=None,
                expected_endpoint_replay_sha256=(
                    endpoint_sha if c10_canary else None),
            )
            if gate_done.get("evidence_sha256") != \
                    canary_gate["evidence_sha256"] or \
                    gate_done.get("canary_endpoint_sha256") != endpoint_sha:
                raise RuntimeError(
                    "canary behavior-gate pipeline-state binding mismatch")
            if c10_canary and (
                    gate_done.get("gate_outcome") !=
                    canary_gate.get("gate_outcome") or
                    gate_done.get("passed") is not
                    canary_gate.get("passed") or
                    gate_done.get("endpoint_replay_path") != endpoint_path or
                    gate_done.get("endpoint_replay_file_sha256") !=
                    endpoint_file_sha):
                raise RuntimeError(
                    "C10 terminal gate state/outcome binding mismatch")
        else:
            canary_gate, gate_path = _run_c9_behavior_gate(
                args, pool, tokenizer, stage="scout_step50",
                model_dir=str(scout_stage["final"]), accepted_path=None,
                round_dir=round_dir,
                expected_endpoint_replay_sha256=(
                    endpoint_sha if c10_canary else None))
            if c10_canary and canary_gate["gate_outcome"] == \
                    "transport_invalid":
                args._c10_terminal_metadata = {
                    "c10_outcome": "transport_invalid",
                    "c10_terminal_scientific_failure": False,
                    "c10_transport_scope": "behavior_gate",
                    "c10_pipeline_execution_contract_sha256": state[
                        "c10_execution_contract"]["contract_sha256"],
                    "c10_endpoint_replay_evidence_sha256": endpoint_sha,
                    "c10_canary_behavior_gate_evidence_sha256":
                        canary_gate["evidence_sha256"],
                    "c10_preregistration_sha256":
                        c10_evidence.C10_PREREGISTRATION_SHA256,
                }
                _push_results_artifact(
                    wandb_run, args,
                    f"r{round_idx}-c10-canary-transport-invalid")
                raise RuntimeError(
                    "C10_CANARY_TRANSPORT_INVALID: the fixed behavior panel "
                    "may be rerun unchanged")
            stage_outputs = {
                "evidence_path": gate_path,
                "evidence_sha256": canary_gate["evidence_sha256"],
                "canary_endpoint_sha256": endpoint_sha,
                "model_file_manifest": canary_gate["model_file_manifest"],
            }
            if c10_canary:
                stage_outputs.update({
                    "gate_outcome": canary_gate["gate_outcome"],
                    "passed": canary_gate["passed"],
                    "endpoint_replay_path": endpoint_path,
                    "endpoint_replay_file_sha256": endpoint_file_sha,
                })
            _mark_stage(
                args, state, round_idx, gate_stage_name, **stage_outputs)
            if c10_canary:
                args._c10_terminal_metadata = {
                    "c10_outcome": canary_gate["gate_outcome"],
                    "c10_pipeline_execution_contract_sha256": state[
                        "c10_execution_contract"]["contract_sha256"],
                    "c10_endpoint_replay_evidence_sha256": endpoint_sha,
                    "c10_canary_behavior_gate_evidence_sha256":
                        canary_gate["evidence_sha256"],
                    "c10_preregistration_sha256":
                        c10_evidence.C10_PREREGISTRATION_SHA256,
                }
            _push_results_artifact(
                wandb_run, args,
                f"r{round_idx}-{'c10-' if c10_canary else ''}"
                "canary-behavior-gate")
        if c10_canary:
            summary.update({
                "c10_canary_endpoint_sha256": endpoint_sha,
                "c10_canary_endpoint_replay_path": endpoint_path,
                "c10_canary_endpoint_replay_file_sha256": endpoint_file_sha,
                "c10_behavior_gate_passed": canary_gate["passed"],
                "c10_behavior_gate_outcome": canary_gate["gate_outcome"],
                "c10_behavior_gate_evidence_sha256":
                    canary_gate["evidence_sha256"],
            })
            if canary_gate["gate_outcome"] == "semantic_failed":
                raise RuntimeError(
                    "C10_CANARY_SEMANTIC_FAILURE: terminal fixed-panel "
                    "behavior failure; unchanged resume is forbidden")
            if canary_gate["gate_outcome"] != "passed":
                raise RuntimeError("C10 canary has an invalid gate outcome")
        else:
            summary.update({
                "c9_canary_endpoint_sha256": endpoint_sha,
                "c9_behavior_gate_passed": True,
                "c9_behavior_gate_evidence_sha256":
                    canary_gate["evidence_sha256"],
            })

    if getattr(args, "scout_only", False):
        # Single-model RL baseline (e.g. DAPO-style correctness-only): the
        # scout checkpoint IS the final model; no harvest/SFT/central stages.
        summary["central_final"] = summary["scout_final"]
        return summary

    # ---- Stage 2: Harvest ----------------------------------------------------
    harvest_path = os.path.join(round_dir, "harvest_trajectories.jsonl")
    done = _stage_done(state, round_idx, "harvest")
    if args.sft_model or frozen_sft_training:
        trajectory_evidence = _validated_trajectory_provenance(args)
        source_stage_mode = (
            "fresh_c8_external_selected_source_provenance" if fresh_external
            else "external_selected_sft_source_provenance" if args.sft_model
            else "frozen_library_source_provenance")
        if done:
            if done.get("stage_mode") != source_stage_mode \
                    or list(done.get("traj_paths", [])) != list(traj_paths) \
                    or done.get("trajectory_source_digest") != \
                    trajectory_evidence["digest"]:
                raise RuntimeError(
                    "completed frozen-library source stage no longer matches "
                    "the exact trajectory provenance")
        else:
            # No harvesting, survivor counting, or c8 gate consultation is
            # permitted for this frozen c7-library training diagnostic.
            _mark_stage(
                args, state, round_idx, "harvest", traj_paths=list(traj_paths),
                stage_mode=source_stage_mode,
                trajectory_source_digest=trajectory_evidence["digest"],
            )
        print(f"[pipeline] round {round_idx}: external frozen-library route binds "
              "trajectory provenance without harvest/filter consultation",
              flush=True)
    elif done:
        print(f"[pipeline] round {round_idx}: harvest already complete", flush=True)
        traj_paths = list(done["traj_paths"])
    else:
        log({"pipeline/stage": 2})
        # A crashed attempt's partial harvest was generated at a possibly
        # different policy — rotate it aside rather than appending to it.
        common.trim_jsonl_to_step(harvest_path, 0)
        traj_paths = _harvest(args, pool, tokenizer, examples, traj_paths,
                              harvest_path, round_idx,
                              expected_policy=summary.get("scout_final"),
                              strict_policy=scout_resumed)
        survivors = _policy_survivors(traj_paths, args.policy, _token_gate_cap(args))
        if survivors >= int(args.min_accepted) or args.policy == "best_scout":
            # best_scout is marked regardless: top-up cannot raise it, so
            # retrying harvest on resume would never help (the filter-stage
            # floor below is the true signal).
            _mark_stage(args, state, round_idx, "harvest", traj_paths=traj_paths)
        else:
            print(f"[pipeline] WARNING: harvest still below --min-accepted "
                  f"({survivors} < {args.min_accepted}) after "
                  f"{args.harvest_max_passes} passes; leaving the harvest stage "
                  "INCOMPLETE so --resume retries top-up instead of deadlocking "
                  "at the filter floor", flush=True)

    # ---- Stage 3: filter_pool ------------------------------------------------
    c9_filter_replay: Optional[Dict[str, Any]] = None
    done = _stage_done(state, round_idx, "filter")
    if done:
        print(f"[pipeline] round {round_idx}: filter already complete "
              f"({done['accepted']})", flush=True)
        accepted_path = done["accepted"]
        if not os.path.isfile(accepted_path):
            raise RuntimeError(
                f"round {round_idx}: completed filter output is missing: {accepted_path}")
        expected_filter_mode = (
            "fresh_c8_external_preselected" if fresh_external
            else "external_preselected_c7_reuse" if args.sft_model
            else "frozen_c7_library_training_input" if frozen_sft_training
            else "trained_pipeline_filter")
        recorded_filter_mode = done.get("stage_mode", "trained_pipeline_filter")
        if recorded_filter_mode != expected_filter_mode:
            raise RuntimeError(
                f"round {round_idx}: completed filter stage_mode="
                f"{recorded_filter_mode!r}, requested {expected_filter_mode!r}")
        recorded_digest = done.get("accepted_library_digest")
        current_digest = common.semantic_sft_jsonl_digest(accepted_path)
        if recorded_digest is not None and recorded_digest != current_digest:
            raise RuntimeError("completed filter accepted-library digest mismatch")
        if args.sft_model or frozen_sft_training:
            expected_external_digest = _validated_external_accepted(args)["digest"] \
                if args.sft_model else C7_REUSE_ACCEPTED_DIGEST
            if current_digest != expected_external_digest:
                raise RuntimeError("completed external filter accepted-library mismatch")
        if frozen_sft_training:
            resumed_frozen_evidence = _validated_frozen_sft_training_evidence(args)
            if done.get("validated_frozen_sft_evidence") != resumed_frozen_evidence:
                raise RuntimeError(
                    "completed frozen-library input evidence no longer matches")
            if common.file_sha256(accepted_path) != \
                    resumed_frozen_evidence["accepted_file_sha256"]:
                raise RuntimeError(
                    "completed frozen accepted.jsonl is not the byte-exact source")
        funnel = done["funnel"]
        if c9_full:
            if int(funnel.get("raw", -1)) != c9_evidence.SCOUT_TRAJECTORY_ROWS or \
                    int(funnel.get("accepted", -1)) != c9_evidence.ACCEPTED_ROWS or \
                    int(funnel.get("accepted_unique_problems", -1)) != \
                    c9_evidence.ACCEPTED_ROWS:
                raise RuntimeError("completed C9 filter does not certify 12,800 -> 500 unique")
            if done.get("accepted_file_sha256") != common.file_sha256(accepted_path):
                raise RuntimeError("completed C9 accepted-library bytes drifted")
            scout_done = _stage_done(state, round_idx, "scout")
            c9_completion = _validate_persisted_c9_scout_completion(
                args, scout_done, scout_init=scout_init, round_idx=round_idx,
                lam=lam)
            if done.get("scout_completion_evidence_sha256") != \
                    c9_completion["evidence_sha256"]:
                raise RuntimeError("completed C9 filter is not bound to this Scout")
            c9_filter_replay = _replay_c9_single_scout_filter(
                args, trajectory_paths=list(traj_paths),
                accepted_path=accepted_path, funnel=funnel)
            if done.get("single_scout_filter_replay") != c9_filter_replay:
                raise RuntimeError(
                    "completed C9 filter replay certificate drifted")
        summary["funnel"] = funnel
    else:
        log({"pipeline/stage": 3})
        accepted_path = os.path.join(round_dir, "accepted.jsonl")
        if args.sft_model or frozen_sft_training:
            # The preregistered c7-reuse ga8 exposure diagnostic preserves the
            # original 500-example library byte-for-byte.  Re-filtering the
            # 32K c7 trajectory bank with later c8 gates produces a different
            # (>16K-admitting) library and would silently redefine the arm.
            frozen = (_validated_external_accepted(args) if args.sft_model
                      else _validated_frozen_sft_library(args))
            shutil.copyfile(frozen["path"], accepted_path)
            if common.file_sha256(accepted_path) != frozen["file_sha256"]:
                raise RuntimeError("frozen accepted.jsonl byte copy mismatch")
            accepted_rows = sum(
                1 for line in Path(accepted_path).read_text(encoding="utf-8").splitlines()
                if line.strip())
            raw_rows = sum(
                1 for path in traj_paths
                for line in Path(path).read_text(encoding="utf-8").splitlines()
                if line.strip())
            funnel = {
                "raw": raw_rows,
                "correct": accepted_rows,
                "clean_correct": accepted_rows,
                "accepted": accepted_rows,
                ("fresh_c8_external_preselected" if fresh_external
                 else "external_preselected_c7_reuse"): 1,
            }
        else:
            pool_paths = traj_paths
            if args.policy == "best_scout":
                # Harvest rows come from the served policy, not from any scout —
                # they must not compete for (or win) the best_scout selection.
                pool_paths = [p for p in traj_paths if p != harvest_path] or traj_paths
            accepted_path, funnel = filter_pool_mod.pool_trajectories(
                pool_paths, policy=args.policy, max_examples=int(args.max_accepted),
                output=accepted_path, max_completion_tokens=_token_gate_cap(args),
                source_balance=bool(args.c8_source_balance),
            )
        summary["funnel"] = funnel
        if (args.rl_validity_mode == "c8" and not args.skip_scout
                and int(args.harvest_max_passes) == 0):
            expected_raw = (int(args.scout_steps) * int(args.prompts_per_step)
                            * int(args.num_generations))
            if int(funnel["raw"]) != expected_raw:
                raise RuntimeError(
                    f"round {round_idx}: canonical c8 requires exactly "
                    f"{expected_raw} raw Scout trajectories "
                    f"({args.scout_steps} steps x {args.prompts_per_step} prompts "
                    f"x {args.num_generations} generations), but the funnel has "
                    f"{funnel['raw']}. Refusing harvest/top-up or silent budget "
                    "drift; diagnose generation failures and relaunch fresh."
                )
        # Funnel counters + SFT acceptance (lambda-annealing funnel-table
        # semantics). Phase "filter_pool" so an SFT rerun's metric-row trim
        # (phase central_sft) cannot drop them.
        funnel_metrics = {
            "funnel/raw": float(funnel["raw"]),
            "funnel/correct": float(funnel["correct"]),
            "funnel/clean_correct": float(funnel["clean_correct"]),
            "funnel/accepted": float(funnel["accepted"]),
            "sft/accepted_trajectories": float(funnel["accepted"]),
            "sft/acceptance_rate": float(funnel["accepted"]) / max(1.0, float(funnel["raw"])),
        }
        log(funnel_metrics)
        metrics_logger.log({"global_step": central_offset, "phase": "filter_pool",
                            "round": round_idx, "lambda": lam, "lambda_novelty": lam,
                            "num_scouts": _num_scouts(args), **funnel_metrics})
        if funnel["accepted"] < int(args.min_accepted):
            raise RuntimeError(
                f"round {round_idx}: only {funnel['accepted']} accepted trajectories "
                f"(< --min-accepted {args.min_accepted}) after "
                f"{args.harvest_max_passes} harvest passes"
            )
        if c9_full and (
                int(funnel.get("accepted", -1)) != c9_evidence.ACCEPTED_ROWS
                or int(funnel.get("accepted_unique_problems", -1)) !=
                c9_evidence.ACCEPTED_ROWS):
            raise RuntimeError(
                "C9 full requires exactly 500 accepted trajectories from 500 "
                "unique problems")
        c9_filter_replay = (
            _replay_c9_single_scout_filter(
                args, trajectory_paths=list(traj_paths),
                accepted_path=accepted_path, funnel=funnel)
            if c9_full else None)
        frozen_training_evidence = (
            _validated_frozen_sft_training_evidence(args)
            if frozen_sft_training else None)
        c9_completion = (
            _validate_persisted_c9_scout_completion(
                args, _stage_done(state, round_idx, "scout"),
                scout_init=scout_init, round_idx=round_idx, lam=lam)
            if c9_full else None)
        _mark_stage(
            args, state, round_idx, "filter", accepted=accepted_path, funnel=funnel,
            stage_mode=("fresh_c8_external_preselected" if fresh_external
                        else "external_preselected_c7_reuse" if args.sft_model
                        else "frozen_c7_library_training_input"
                        if frozen_sft_training else "trained_pipeline_filter"),
            accepted_library_digest=common.semantic_sft_jsonl_digest(accepted_path),
            accepted_file_sha256=(common.file_sha256(accepted_path)
                                  if c9_full else None),
            scout_completion_evidence_sha256=(
                c9_completion["evidence_sha256"] if c9_completion else None),
            single_scout_filter_replay=c9_filter_replay,
            validated_frozen_sft_evidence=frozen_training_evidence,
        )

    c9_training_evidence: Optional[Dict[str, Any]] = None
    c9_scout_gate: Optional[Dict[str, Any]] = None
    if c9_full:
        scout_done = _stage_done(state, round_idx, "scout")
        c9_completion = _validate_persisted_c9_scout_completion(
            args, scout_done, scout_init=scout_init, round_idx=round_idx,
            lam=lam)
        gate_done = _stage_done(state, round_idx, "scout_behavior_gate")
        if gate_done:
            c9_scout_gate = _validate_persisted_c9_behavior_gate(
                args, evidence_path=str(gate_done.get("evidence_path", "")),
                stage="scout_final", model_dir=str(scout_done["final"]),
                accepted_path=accepted_path)
            if gate_done.get("evidence_sha256") != \
                    c9_scout_gate["evidence_sha256"]:
                raise RuntimeError("C9 Scout-gate pipeline-state SHA mismatch")
        else:
            c9_scout_gate, gate_path = _run_c9_behavior_gate(
                args, pool, tokenizer, stage="scout_final",
                model_dir=str(scout_done["final"]),
                accepted_path=accepted_path, round_dir=round_dir)
            _mark_stage(
                args, state, round_idx, "scout_behavior_gate",
                evidence_path=gate_path,
                evidence_sha256=c9_scout_gate["evidence_sha256"],
                model_file_manifest=c9_scout_gate["model_file_manifest"],
            )
            _push_results_artifact(
                wandb_run, args, f"r{round_idx}-scout-behavior-gate")

        sargs_preview = _sft_args(
            args, accepted=accepted_path,
            output_dir=os.path.join(round_dir, "sft"), round_idx=round_idx,
            lam=lam, global_step_offset=central_offset)
        expected_sft_runtime = sft_gpu.build_sft_runtime_contract(
            sargs_preview, resolved_max_total_len=int(sargs_preview.max_total_len))
        c9_training_evidence = c9_evidence.signed_payload(
            c9_evidence.SFT_TRAINING_EVIDENCE_VERSION,
            {
                "c9_execution_profile": c9_evidence.C9_PROFILE,
                "stabilization_cell": c9_evidence.C9_CELL,
                "scientific_variant": c9_evidence.C9_SCIENTIFIC_VARIANT,
                "code_artifact": _c9_code_identity(args),
                "scout_completion_evidence": c9_completion,
                "scout_behavior_gate": c9_scout_gate,
                "filter_funnel": summary["funnel"],
                "single_scout_filter_replay": c9_filter_replay,
                "accepted_library": {
                    "semantic_digest": common.semantic_sft_jsonl_digest(
                        accepted_path),
                    "file_sha256": common.file_sha256(accepted_path),
                    "training_sequence_digest": common.sft_training_sequence_digest(
                        _read_jsonl_objects(
                            accepted_path, "C9 accepted SFT library"),
                        epochs=int(args.sft_epochs), seed=int(args.seed)),
                },
                "sft_runtime_contract": expected_sft_runtime,
            },
        )
        c9_evidence.validate_sft_training_evidence(c9_training_evidence)
        evidence_path = os.path.join(
            round_dir, "c9_sft_training_evidence.json")
        if os.path.isfile(evidence_path):
            if _load_json_object(
                    evidence_path, "C9 SFT training evidence") != \
                    c9_training_evidence:
                raise RuntimeError("C9 SFT training evidence drifted on resume")
        else:
            common.atomic_write_json(evidence_path, c9_training_evidence)
        args.sft_training_evidence_manifest = evidence_path

    if fresh_sft_training:
        fresh_training_evidence = _build_fresh_c8_sft_training_evidence(
            args, state, round_idx=round_idx, traj_paths=list(traj_paths),
            accepted_path=accepted_path, funnel=summary["funnel"])
        evidence_path = os.path.join(
            round_dir, "fresh_c8_sft_training_evidence.json")
        if os.path.isfile(evidence_path):
            recorded = _load_json_object(
                evidence_path, "completed fresh-c8 SFT training evidence")
            if recorded != fresh_training_evidence:
                raise RuntimeError(
                    "fresh-c8 SFT training evidence drifted on resume")
        else:
            common.atomic_write_json(evidence_path, fresh_training_evidence)
        args.sft_training_evidence_manifest = evidence_path

    # ---- Stage 4: Central SFT / health-gated SFT selection ------------------
    requested_sft_mode = (
        "fresh_c8_external_health_gate_selected" if fresh_external
        else "external_health_gate_selected" if args.sft_model
        else "fresh_c8_preregistered_sft_trained" if fresh_sft_training
        else "c9_fixed_endpoint_trained" if c9_full
        else "c9_multi_explorer_fixed_endpoint_trained" if c9_multi
        else "frozen_c7_library_ga8_trained" if frozen_sft_training
        else "trained_in_pipeline")
    trained_sft_modes = (
        "trained_in_pipeline", "frozen_c7_library_ga8_trained",
        "fresh_c8_preregistered_sft_trained", "c9_fixed_endpoint_trained",
        "c9_multi_explorer_fixed_endpoint_trained",
    )
    sft_output_dir = os.path.join(round_dir, "sft")
    resolved_sft_args = _sft_args(
        args, accepted=accepted_path, output_dir=sft_output_dir,
        round_idx=round_idx, lam=lam,
        global_step_offset=central_offset)
    expected_sft_runtime = sft_gpu.build_sft_runtime_contract(
        resolved_sft_args,
        resolved_max_total_len=int(resolved_sft_args.max_total_len))
    sft_model_binding: Optional[Dict[str, Any]] = None
    done = _stage_done(state, round_idx, "sft")
    if done:
        # Absence of stage_mode is an old in-pipeline SFT state.  Never let a
        # resume silently substitute it for an externally gate-selected model
        # (or vice versa).
        recorded_stage_mode = done.get("stage_mode", "trained_in_pipeline")
        if recorded_stage_mode != requested_sft_mode:
            raise RuntimeError(
                f"round {round_idx}: completed SFT stage_mode="
                f"{recorded_stage_mode!r}, but this run requests "
                f"{requested_sft_mode!r}. Use the same --sft-model selection "
                "or a fresh --output-dir.")
        if requested_sft_mode in trained_sft_modes:
            sft_dir, sft_model_binding = _restore_or_validate_sft_model(
                args, done, accepted_path=accepted_path,
                expected_runtime_contract=expected_sft_runtime,
                round_idx=round_idx)
            # Old c7 pipeline states did not record a termination mode;
            # absence therefore means legacy/no-EOS. Never silently reuse one
            # as a c8 SFT model.
            recorded_mode = done.get("termination_mode", "legacy")
            if recorded_mode != args.sft_termination_mode:
                raise RuntimeError(
                    f"round {round_idx}: completed SFT uses termination_mode="
                    f"{recorded_mode!r}, but this run requests "
                    f"{args.sft_termination_mode!r}. Use a fresh --output-dir "
                    "(recommended) or explicitly resume with the recorded mode.")
            if frozen_sft_training:
                resumed_frozen_evidence = _validated_frozen_sft_training_evidence(args)
                if done.get("validated_frozen_sft_evidence") != resumed_frozen_evidence:
                    raise RuntimeError(
                        "completed frozen-library SFT evidence no longer matches")
                provenance = _load_json_object(
                    os.path.join(done["final"], "sft_provenance.json"),
                    "completed frozen-library SFT provenance")
                if provenance.get("accepted_library_digest") != \
                        resumed_frozen_evidence["accepted_library_digest"]:
                    raise RuntimeError(
                        "completed frozen-library SFT accepted digest mismatch")
                if provenance.get("trajectory_source", {}).get("digest") != \
                        resumed_frozen_evidence["trajectory_source_digest"]:
                    raise RuntimeError(
                        "completed frozen-library SFT trajectory provenance mismatch")
                if provenance.get("sft_runtime_contract") != \
                        sft_selector.DIAGNOSTIC_SFT_RUNTIME_CONTRACT:
                    raise RuntimeError(
                        "completed frozen-library SFT runtime contract mismatch")
            if fresh_sft_training:
                provenance = _load_json_object(
                    os.path.join(done["final"], "sft_provenance.json"),
                    "completed fresh-c8 SFT provenance")
                if provenance.get("fresh_c8_training_evidence") != fresh_training_evidence:
                    raise RuntimeError(
                        "completed fresh-c8 SFT training evidence mismatch")
                if provenance.get("sft_runtime_contract") != \
                        fresh_c8_sft.SFT_RUNTIME_CONTRACT:
                    raise RuntimeError("completed fresh-c8 SFT runtime contract mismatch")
                fresh_c8_sft.validate_completed_run_evidence(
                    provenance.get("completed_run_evidence"),
                    expected_training_evidence_sha256=
                    fresh_training_evidence["evidence_sha256"])
            if c9_full:
                if done.get("c9_training_evidence_sha256") != \
                        c9_training_evidence["evidence_sha256"]:
                    raise RuntimeError("completed C9 SFT pipeline-state evidence mismatch")
                _validate_c9_sft_endpoint(done["final"], c9_training_evidence)
        else:
            recorded_final = os.path.realpath(done["final"])
            requested_final = os.path.realpath(args.sft_model)
            if recorded_final != requested_final:
                raise RuntimeError(
                    f"round {round_idx}: completed external SFT selection is "
                    f"{done['final']!r}, but --sft-model is "
                    f"{args.sft_model!r}. Use the same selected checkpoint or "
                    "a fresh --output-dir.")
            resumed_evidence = _validate_external_sft_evidence(args, accepted_path)
            if done.get("validated_external_evidence") != resumed_evidence:
                raise RuntimeError(
                    "completed external SFT stage evidence no longer matches "
                    "the model/gate/selection/trajectory inputs")
            external_evidence = resumed_evidence
        print(f"[pipeline] round {round_idx}: SFT already complete ({done['final']})",
              flush=True)
        sft_dir = str(done["final"])
    elif args.sft_model:
        external_evidence = _validate_external_sft_evidence(args, accepted_path)
        external_canonical = bool(external_evidence.get("canonical", False))
        external_variant = str(external_evidence.get("scientific_variant", ""))
        log({"pipeline/stage": 4,
             "pipeline/sft_training_skipped": 1.0,
             "pipeline/sft_external_health_gate_selected": 1.0,
             "pipeline/noncanonical_scientific_variant": float(not external_canonical)})
        sft_dir = str(args.sft_model)
        checkpoint_info = _sft_checkpoint_metadata(sft_dir)
        provenance_path = os.path.join(round_dir, "sft_selection_provenance.json")
        source = _source_label(external_evidence["model_source"], "model source")
        selection_provenance = {
            "stage_mode": requested_sft_mode,
            "health_gate_selected": True,
            "sft_training_skipped": True,
            "canonical": external_canonical,
            "scientific_variant": external_variant,
            "model_path": sft_dir,
            "source": source,
            "accepted_trajectories": accepted_path,
            "filter_policy": str(args.policy),
            "filter_funnel": summary.get("funnel", {}),
            "validated_external_evidence": external_evidence,
            **checkpoint_info,
        }
        common.atomic_write_json(provenance_path, selection_provenance)
        metrics_logger.log({
            "global_step": central_offset,
            "phase": "sft_selection",
            "round": round_idx,
            "lambda": lam,
            "lambda_novelty": lam,
            "num_scouts": _num_scouts(args),
            "pipeline/sft_training_skipped": 1.0,
            "pipeline/sft_external_health_gate_selected": 1.0,
            "pipeline/noncanonical_scientific_variant": float(not external_canonical),
        })
        _mark_stage(
            args, state, round_idx, "sft", final=sft_dir,
            stage_mode=requested_sft_mode, source=source,
            health_gate_selected=True, sft_training_skipped=True,
            accepted=accepted_path, selection_provenance=provenance_path,
            canonical=external_canonical, scientific_variant=external_variant,
            validated_external_evidence=external_evidence,
        )
        print(f"[pipeline] round {round_idx}: using externally supplied, "
              f"health-gate-selected SFT model ({sft_dir}); SFT training skipped",
              flush=True)
        _push_results_artifact(wandb_run, args, f"r{round_idx}-sft-selection")
    else:
        log({"pipeline/stage": 4})
        recovered_complete_sft = False
        if os.path.isdir(sft_output_dir) and \
                any(Path(sft_output_dir).iterdir()):
            candidate = os.path.join(sft_output_dir, "actual_sft_final")
            if not args.resume or not os.path.isdir(candidate):
                raise RuntimeError(
                    "incomplete gated SFT output exists but no completed "
                    "pipeline stage is recorded; use --resume only when a "
                    "complete, byte-verifiable actual_sft_final exists")
            _validate_local_sft_handoff(
                candidate, accepted_path=accepted_path,
                expected_runtime_contract=expected_sft_runtime,
                epochs=int(resolved_sft_args.epochs),
                seed=int(resolved_sft_args.seed))
            sft_dir = candidate
            recovered_complete_sft = True
            print(
                f"[pipeline] round {round_idx}: recovered complete local SFT "
                "endpoint; retrying durable artifact handoff", flush=True)
        else:
            sft_dir = sft_gpu.run_sft(resolved_sft_args)
        frozen_training_evidence = (
            _validated_frozen_sft_training_evidence(args)
            if frozen_sft_training else None)
        if frozen_sft_training:
            provenance = _load_json_object(
                os.path.join(sft_dir, "sft_provenance.json"),
                "frozen-library SFT provenance")
            if provenance.get("accepted_library_digest") != \
                    frozen_training_evidence["accepted_library_digest"]:
                raise RuntimeError("frozen-library SFT accepted digest mismatch")
            if provenance.get("trajectory_source", {}).get("digest") != \
                    frozen_training_evidence["trajectory_source_digest"]:
                raise RuntimeError("frozen-library SFT trajectory provenance mismatch")
            if provenance.get("sft_runtime_contract") != \
                    sft_selector.DIAGNOSTIC_SFT_RUNTIME_CONTRACT:
                raise RuntimeError("frozen-library SFT runtime contract mismatch")
        if fresh_sft_training:
            provenance = _load_json_object(
                os.path.join(sft_dir, "sft_provenance.json"),
                "fresh-c8 SFT provenance")
            if provenance.get("fresh_c8_training_evidence") != fresh_training_evidence:
                raise RuntimeError("fresh-c8 SFT training evidence mismatch")
            if provenance.get("sft_runtime_contract") != \
                    fresh_c8_sft.SFT_RUNTIME_CONTRACT:
                raise RuntimeError("fresh-c8 SFT runtime contract mismatch")
            fresh_c8_sft.validate_completed_run_evidence(
                provenance.get("completed_run_evidence"),
                expected_training_evidence_sha256=
                fresh_training_evidence["evidence_sha256"])
        if c9_full:
            _validate_c9_sft_endpoint(sft_dir, c9_training_evidence)
        sft_handoff = _validate_local_sft_handoff(
            sft_dir, accepted_path=accepted_path,
            expected_runtime_contract=expected_sft_runtime,
            epochs=int(resolved_sft_args.epochs),
            seed=int(resolved_sft_args.seed))
        sft_model_binding = _persist_sft_model_artifact(
            wandb_run, args, model_dir=sft_dir, round_idx=round_idx,
            stage_mode=requested_sft_mode, handoff=sft_handoff)
        sft_stage_outputs: Dict[str, Any] = {
            "final": sft_dir,
            "stage_mode": requested_sft_mode,
            "termination_mode": args.sft_termination_mode,
            "validated_frozen_sft_evidence": frozen_training_evidence,
            "validated_fresh_c8_training_evidence": (
                fresh_training_evidence if fresh_sft_training else None),
            "c9_training_evidence_sha256": (
                c9_training_evidence["evidence_sha256"] if c9_full else None),
            "sft_model_artifact": sft_model_binding,
            "recovered_complete_sft": recovered_complete_sft,
        }
        if fresh_sft_training:
            # Passing the independent behavior gate is a necessary part of the
            # canonical claim.  A trained endpoint is only a pending candidate.
            sft_stage_outputs["canonical_pending_gate"] = True
        elif c9_full or c9_multi:
            sft_stage_outputs.update({
                "canonical_pending_gate": True,
                "scientific_variant": (
                    c9_evidence.C9_SCIENTIFIC_VARIANT if c9_full else
                    mrme_contract.C9_MULTIEXPLORER_SCIENTIFIC_VARIANT),
            })
        else:
            sft_stage_outputs.update({
                "canonical": not frozen_sft_training,
                "scientific_variant": (
                    FROZEN_SFT_LIBRARY_VARIANT if frozen_sft_training else ""),
            })
        _mark_stage(args, state, round_idx, "sft", **sft_stage_outputs)
        _push_results_artifact(wandb_run, args, f"r{round_idx}-sft")
    if done and requested_sft_mode in trained_sft_modes:
        # A prior attempt may have crashed after the atomic local stage mark
        # but before the synchronous results-artifact upload completed. Retry
        # that durable state barrier before any gate or Central work.
        _push_results_artifact(
            wandb_run, args, f"r{round_idx}-sft-resume-barrier")
    summary["sft_final"] = sft_dir
    summary["sft_stage_mode"] = requested_sft_mode
    summary["sft_training_skipped"] = bool(args.sft_model)
    summary["sft_health_gate_selected"] = bool(args.sft_model)
    if sft_model_binding is not None:
        summary["sft_model_artifact_binding_sha256"] = sft_model_binding[
            "binding_sha256"]
    if frozen_sft_training:
        summary["canonical"] = False
        summary["scientific_variant"] = FROZEN_SFT_LIBRARY_VARIANT
    if fresh_sft_training:
        summary["canonical_pending_gate"] = True
    c9_lineage: Optional[Dict[str, Any]] = None
    c9_multi_lineage: Optional[Dict[str, Any]] = None
    if c9_full:
        _validate_c9_sft_endpoint(sft_dir, c9_training_evidence)
        gate_done = _stage_done(state, round_idx, "sft_behavior_gate")
        expected_panel = c9_scout_gate["heldout_selection_sha256"]
        if gate_done:
            c9_sft_gate = _validate_persisted_c9_behavior_gate(
                args, evidence_path=str(gate_done.get("evidence_path", "")),
                stage="sft_fixed_endpoint", model_dir=sft_dir,
                accepted_path=accepted_path,
                expected_selection_sha256=expected_panel)
            if gate_done.get("evidence_sha256") != c9_sft_gate["evidence_sha256"]:
                raise RuntimeError("C9 SFT-gate pipeline-state SHA mismatch")
        else:
            c9_sft_gate, gate_path = _run_c9_behavior_gate(
                args, pool, tokenizer, stage="sft_fixed_endpoint",
                model_dir=sft_dir, accepted_path=accepted_path,
                round_dir=round_dir, expected_selection_sha256=expected_panel)
            _mark_stage(
                args, state, round_idx, "sft_behavior_gate",
                evidence_path=gate_path,
                evidence_sha256=c9_sft_gate["evidence_sha256"],
                model_file_manifest=c9_sft_gate["model_file_manifest"],
            )
            _push_results_artifact(
                wandb_run, args, f"r{round_idx}-sft-behavior-gate")
        central_runtime = _resolved_central_runtime_contract(
            args, model=sft_dir, round_idx=round_idx,
            global_step_offset=central_offset)
        c9_lineage = c9_evidence.signed_payload(
            c9_evidence.CENTRAL_LINEAGE_VERSION,
            {
                "lineage_contract_version": c9_evidence.CENTRAL_LINEAGE_VERSION,
                "c9_execution_profile": c9_evidence.C9_PROFILE,
                "stabilization_cell": c9_evidence.C9_CELL,
                "scientific_variant": c9_evidence.C9_SCIENTIFIC_VARIANT,
                "fixed_sft_endpoint": True,
                "sft_optimizer_steps": c9_evidence.SFT_OPTIMIZER_STEPS,
                "sft_training_evidence": c9_training_evidence,
                "sft_behavior_gate": c9_sft_gate,
                "sft_model_file_manifest": _c9_model_manifest(
                    sft_dir, stage="sft_fixed_endpoint"),
                "sft_model_artifact": sft_model_binding,
                "central_runtime_contract": central_runtime,
            },
        )
        c9_evidence.validate_central_lineage(
            c9_lineage, expected_runtime_contract=central_runtime)
        external_lineage_path = os.path.join(
            round_dir, "c9_central_lineage.json")
        if os.path.isfile(external_lineage_path):
            if _load_json_object(
                    external_lineage_path, "C9 Central lineage") != c9_lineage:
                raise RuntimeError("C9 Central lineage drifted on resume")
        else:
            common.atomic_write_json(external_lineage_path, c9_lineage)
        summary.update({
            "canonical_pending_gate": False,
            "sft_health_gate_selected": True,
            "c9_scout_gate_passed": True,
            "c9_sft_gate_passed": True,
            "scientific_variant": c9_evidence.C9_SCIENTIFIC_VARIANT,
        })
    elif c9_multi:
        central_runtime = _resolved_central_runtime_contract(
            args, model=sft_dir, round_idx=round_idx,
            global_step_offset=central_offset)
        c9_sft_gate, c9_multi_lineage, external_lineage_path = \
            _ensure_c9_multi_pooled_sft_gate(
                args, pool, tokenizer, contract=multi_contract,
                round_idx=round_idx, round_dir=round_dir,
                sft_dir=sft_dir, accepted_path=accepted_path,
                sft_model_binding=sft_model_binding,
                central_runtime=central_runtime, state=state,
                wandb_run=wandb_run)
        summary.update({
            "canonical_pending_gate": False,
            "sft_health_gate_selected": True,
            "c9_sft_gate_passed": True,
            "c9_sft_gate_evidence_sha256": c9_sft_gate["evidence_sha256"],
            "c9_central_lineage_sha256": c9_multi_lineage["lineage_sha256"],
            "scientific_variant":
                mrme_contract.C9_MULTIEXPLORER_SCIENTIFIC_VARIANT,
        })
    elif args.sft_model:
        summary["sft_source"] = _source_label(
            external_evidence["model_source"], "model source")
        summary["canonical"] = bool(external_evidence.get("canonical", False))
        summary["scientific_variant"] = str(
            external_evidence.get("scientific_variant", ""))
        external_lineage_path = os.path.join(
            round_dir, "external_central_lineage.json")
        common.atomic_write_json(external_lineage_path, external_evidence)
    else:
        external_lineage_path = ""

    if args.stop_after_sft:
        # Clean SFT-only checkpoint/gate hook: downstream evaluation can test
        # termination/validity before any Central RL update is allowed to
        # compound a bad distillation checkpoint.
        print(f"[pipeline] round {round_idx}: stopping after SFT for evaluation "
              f"gate ({sft_dir})", flush=True)
        summary["stopped_after_sft"] = True
        summary["central_final"] = sft_dir
        log({"pipeline/stage": 4, "pipeline/stopped_after_sft": 1.0})
        return summary

    # ---- Stage 5: Central GRPO (correctness-only) -----------------------------
    central_lineage = (
        c9_lineage if c9_full else
        c9_multi_lineage if c9_multi else
        external_evidence if args.sft_model else None)
    central_runtime = (
        central_lineage.get("central_runtime_contract")
        if central_lineage else _resolved_central_runtime_contract(
            args, model=sft_dir, round_idx=round_idx,
            global_step_offset=central_offset))
    done = _stage_done(state, round_idx, "central")
    central_probe_history: Optional[Dict[str, Any]] = None
    central_artifact: Optional[Dict[str, Any]] = None
    central_output_dir = os.path.join(round_dir, "central")
    if done:
        completed_state = _validate_completed_central(
            done["final"], central_lineage,
            expected_steps=int(args.central_steps),
            expected_runtime_contract=central_runtime)
        if done.get("external_lineage_sha256") != completed_state.get(
                "external_lineage_sha256"):
            raise RuntimeError("completed Central pipeline-state lineage mismatch")
        if (c9_full or c9_multi or
                done.get("runtime_contract_sha256") is not None) and \
                done.get("runtime_contract_sha256") != common.canonical_json_sha256(
                    central_runtime):
            raise RuntimeError("completed Central pipeline-state runtime mismatch")
        if c9_full or c9_multi:
            central_probe_history = _c9_probe_history_facts(
                args, phase="central", local_steps=int(args.central_steps),
                global_step_offset=central_offset)
            if done.get("c9_probe_history") != central_probe_history:
                raise RuntimeError(
                    "completed C9 Central pipeline-state probe history mismatch")
        central_artifact = _validated_grpo_final_artifact_binding(
            central_output_dir, str(done["final"]), phase="central",
            round_idx=round_idx, steps=int(args.central_steps),
            required=c9_multi)
        if done.get("model_artifact") != central_artifact:
            raise RuntimeError(
                "completed Central pipeline-state artifact binding mismatch")
        print(f"[pipeline] round {round_idx}: central already complete "
              f"({done['final']})", flush=True)
        central_final = done["final"]
    else:
        log({"pipeline/stage": 5})
        if not pool.managed:
            print("[pipeline] WARNING: attach-mode pool cannot reload to the SFT "
                  "model; central rollouts use stale weights", flush=True)
        elif pool.model_dir != sft_dir:
            # Make the selected checkpoint the serving policy before Central
            # is even entered. run_grpo independently checks this invariant;
            # doing it here makes the health-gate handoff explicit and testable.
            print(f"[pipeline] reloading managed pool to Central init: {sft_dir}",
                  flush=True)
            pool.reload(sft_dir)
        cargs = _grpo_args(args, phase="central", model=sft_dir,
                           output_dir=central_output_dir,
                           lam=0.0, steps=int(args.central_steps), round_idx=round_idx,
                           global_step_offset=central_offset,
                           external_lineage_json=external_lineage_path)
        central_final = grpo_gpu.run_grpo(cargs, pool=pool)
        completed_state = _validate_completed_central(
            central_final, central_lineage,
            expected_steps=int(args.central_steps),
            expected_runtime_contract=central_runtime,
        )
        central_artifact = _validated_grpo_final_artifact_binding(
            central_output_dir, central_final, phase="central",
            round_idx=round_idx, steps=int(args.central_steps),
            required=c9_multi)
        if c9_full or c9_multi:
            central_probe_history = _c9_probe_history_facts(
                args, phase="central", local_steps=int(args.central_steps),
                global_step_offset=central_offset)
        _mark_stage(
            args, state, round_idx, "central", final=central_final,
            external_lineage_sha256=completed_state.get("external_lineage_sha256"),
            runtime_contract_sha256=common.canonical_json_sha256(central_runtime),
            checkpoint_file_manifest=completed_state.get("checkpoint_file_manifest"),
            model_artifact=central_artifact,
            c9_probe_history=central_probe_history,
        )
        _push_results_artifact(wandb_run, args, f"r{round_idx}-central")
    summary["central_final"] = central_final
    if central_artifact is not None:
        summary["central_model_artifact_binding_sha256"] = \
            central_artifact["binding_sha256"]
    if c9_full or c9_multi:
        if c9_full:
            c9_evidence.require_shared_probe_prompt_matrix(
                [c9_completion["probe_history"], central_probe_history],
                context="C9 SingleScout campaign")
        summary["c9_central_probe_history"] = central_probe_history
    log({"pipeline/stage": 6, "pipeline/round_complete": 1.0})
    return summary


# ------------------------------------------------------------------- main ----

DEFAULT_LAMBDA_NOVELTY = 0.5
DEFAULT_LAMBDA_SCHEDULE = "0.75,0.5,0.35,0.25"


def parse_lambda_schedule(args: argparse.Namespace) -> List[float]:
    """Per-round lambda list with explicit-flag precedence.

    - rounds == 1: --lambda-novelty (explicit or its 0.5 default) always wins;
      the schedule is ignored.
    - rounds > 1: the schedule wins ONLY when the user did not pass
      --lambda-novelty explicitly. An explicit --lambda-novelty without an
      explicit schedule means a constant lambda across rounds. Passing BOTH
      explicitly is an error unless they agree (previously --lambda-novelty
      was silently ignored on every multi-round run).
    Both flags default to None sentinels so "explicitly passed" is detectable.
    """
    rounds = int(args.rounds)
    lam_explicit = args.lambda_novelty is not None
    sched_explicit = args.lambda_schedule is not None
    lam = float(args.lambda_novelty) if lam_explicit else DEFAULT_LAMBDA_NOVELTY
    sched_str = args.lambda_schedule if sched_explicit else DEFAULT_LAMBDA_SCHEDULE
    if rounds <= 1:
        return [lam] * max(1, rounds)
    sched = [float(x) for x in str(sched_str).split(",") if x.strip()]
    if not sched:
        raise ValueError(f"bad --lambda-schedule {sched_str!r}")
    per_round = [sched[min(r, len(sched) - 1)] for r in range(rounds)]
    if lam_explicit and sched_explicit:
        if any(x != lam for x in per_round):
            raise ValueError(
                f"--lambda-novelty {lam} and --lambda-schedule {sched_str!r} were "
                f"both given but disagree for a {rounds}-round run ({per_round}); "
                "drop one (the schedule wins only when --lambda-novelty is not "
                "passed explicitly)")
        return per_round
    if lam_explicit:
        return [lam] * rounds
    return per_round


def _true_mrme(args: argparse.Namespace) -> bool:
    # Historical name retained for call-site compatibility.  The integrated
    # R=1,K>1 MultiScout, R>1,K>1 MR-ME, and explicitly requested K=1
    # multi-round variants share one fan-out/fan-in implementation but carry
    # different immutable method identities.  K=1 never enters this path by
    # inference from --rounds alone: the distinct profile is the authorization
    # boundary that prevents ordinary legacy multi-round runs from being
    # relabelled as the signed publication method.
    return (
        int(args.scouts_per_round) > 1
        or bool(str(getattr(
            args, "single_scout_multiround_profile", "") or ""))
    )


def _is_single_scout_multiround_profile(args: argparse.Namespace) -> bool:
    return bool(str(getattr(
        args, "single_scout_multiround_profile", "") or ""))


def _is_c9_round_campaign(args: argparse.Namespace) -> bool:
    return bool(
        str(getattr(args, "c9_multiexplorer_profile", "") or "")
        or _is_single_scout_multiround_profile(args)
    )


@contextlib.contextmanager
def _temporary_environment(values: Dict[str, str]) -> Iterator[None]:
    saved = {name: os.environ.get(name) for name in values}
    os.environ.update({name: str(value) for name, value in values.items()})
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _runtime_contract_template(
    gargs: argparse.Namespace, *, resolved_lr: float, resolved_steps: int,
) -> Dict[str, Any]:
    """Resolve a GRPO contract and remove fields owned by MR-ME round specs."""
    grpo_gpu.apply_completion_budget(gargs)
    grpo_gpu.enforce_probe_serving_len(gargs, pool=None)
    runtime = grpo_gpu.build_runtime_contract(
        gargs, resolved_lr=resolved_lr, resolved_steps=resolved_steps)
    for field in ("model_init", "seed", "round", "lambda_novelty"):
        runtime.pop(field, None)
    return runtime


def _resolved_mrme_training_contract(
    args: argparse.Namespace, lambdas: List[float],
) -> Dict[str, Any]:
    """Training-affecting template shared by the contract's round identities."""
    k = int(args.scouts_per_round)
    scout_lr = float(args.scout_lr) if args.scout_lr is not None else 5e-6
    central_lr = float(args.central_lr) if args.central_lr is not None else 1e-6
    scout_args = _grpo_args(
        args, phase="scout", model=args.model,
        output_dir=os.path.join(args.output_dir, "contract_template", "scout"),
        lam=float(lambdas[0]), steps=int(args.scout_steps), round_idx=1,
        global_step_offset=0, seed_override=int(args.seed),
        num_scouts_override=k,
    )
    central_args = _grpo_args(
        args, phase="central", model=args.model,
        output_dir=os.path.join(args.output_dir, "contract_template", "central"),
        lam=0.0, steps=int(args.central_steps), round_idx=1,
        global_step_offset=k * int(args.scout_steps),
        seed_override=int(args.seed), num_scouts_override=k,
    )
    sft_args = _sft_args(
        args,
        accepted=os.path.join(args.output_dir, "contract_template", "accepted.jsonl"),
        output_dir=os.path.join(args.output_dir, "contract_template", "sft"),
        round_idx=1, lam=float(lambdas[0]),
        global_step_offset=k * int(args.scout_steps),
    )
    sft_max_len = (
        2048 + int(sft_args.completion_budget)
        if int(sft_args.completion_budget) > 0 else int(sft_args.max_total_len))
    sft_runtime = sft_gpu.build_sft_runtime_contract(
        sft_args, resolved_max_total_len=sft_max_len)
    sft_runtime.pop("seed", None)
    return {
        "version": "gpu_mrme_training_template_v1",
        "stabilization_profile": (
            mrme_contract.C9_MULTIEXPLORER_PROFILE
            if _is_c9_round_campaign(args)
            else mrme_contract.C8_MULTIEXPLORER_PROFILE),
        "scout_grpo": _runtime_contract_template(
            scout_args, resolved_lr=scout_lr,
            resolved_steps=int(args.scout_steps)),
        "central_grpo": _runtime_contract_template(
            central_args, resolved_lr=central_lr,
            resolved_steps=int(args.central_steps)),
        "central_sft": sft_runtime,
        "scout_seed_rule": "contract_rounds_spec",
        "central_and_sft_seed_rule": "base_seed_plus_round_index_minus_one",
        "rnd_reset": "fresh_per_scout",
        "central_sft_init": "base_model_every_round",
        "next_round_scout_init": "previous_round_central_final",
        "pooling_policy": str(args.policy),
        "pooling_source_balance": bool(args.c8_source_balance),
        "accepted_floor": int(args.min_accepted),
        "accepted_cap": int(args.max_accepted),
        "lambda_schedule": [float(value) for value in lambdas],
    }


def _build_mrme_contract(
    args: argparse.Namespace, lambdas: List[float],
) -> Dict[str, Any]:
    code_identity = _validated_executing_code_source_manifest(
        args, profile_label=(
            "C9 stabilized single-Scout multi-round"
            if _is_single_scout_multiround_profile(args)
            else "C9 stabilized multi-explorer"
            if _is_c9_round_campaign(args)
            else "MR-ME"))
    from tmx_jax.data import training_dataset_identity

    return mrme_contract.build_contract(
        model=args.model,
        model_revision=str(args.model_revision or ""),
        dataset=args.dataset,
        dataset_identity=training_dataset_identity(args.dataset),
        max_train_examples=int(args.max_train_examples),
        rounds=int(args.rounds),
        scouts_per_round=int(args.scouts_per_round),
        lambda_schedule=lambdas,
        base_seed=int(args.seed),
        scout_steps=int(args.scout_steps),
        central_steps=int(args.central_steps),
        prompts_per_step=int(args.prompts_per_step),
        num_generations=int(args.num_generations),
        pooling_policy=args.policy,
        max_accepted=int(args.max_accepted),
        harvest_max_passes=int(args.harvest_max_passes),
        stabilization_profile=(
            mrme_contract.C9_MULTIEXPLORER_PROFILE
            if _is_c9_round_campaign(args)
            else mrme_contract.C8_MULTIEXPLORER_PROFILE),
        training_contract=_resolved_mrme_training_contract(args, lambdas),
        code_identity=code_identity,
    )


def _round_campaign_summary_identity(
    contract: Dict[str, Any],
) -> Dict[str, Any]:
    """Publication-facing identity for top-level and per-round summaries."""
    single = mrme_contract.is_single_scout_multiround(contract)
    identity: Dict[str, Any] = {
        "method": contract["method"],
        "mrme": not single,
        "multi_explorer": not single,
        "single_scout_multiround": single,
    }
    if single:
        identity["method_version"] = contract["method_version"]
    return identity


def _write_or_verify_json(path: str, value: Dict[str, Any], *, label: str) -> None:
    if os.path.isfile(path):
        existing = _load_json_object(path, label)
        if existing != value:
            raise RuntimeError(f"{label} differs from existing {path!r}")
        return
    common.atomic_write_json(path, value)


def _prepare_mrme_state(
    args: argparse.Namespace, state: Dict[str, Any], contract: Dict[str, Any],
) -> Dict[str, Any]:
    """Bind output state and contract before tokenizer or GPU startup."""
    state_path = os.path.join(args.output_dir, "pipeline_state.json")
    if args.resume and not os.path.isfile(state_path):
        orphaned = [
            path for path in (
                os.path.join(args.output_dir, "run_summary.json"),
                *glob.glob(os.path.join(args.output_dir, "round_[0-9][0-9]")),
            )
            if os.path.exists(path)
        ]
        if orphaned:
            raise RuntimeError(
                "MR-ME --resume found round outputs without pipeline_state.json; "
                "refusing to reconstruct stage completion from ambiguous files: "
                f"{orphaned}")
    if not args.resume:
        stale = [
            path for path in (
                os.path.join(args.output_dir, "pipeline_state.json"),
                os.path.join(args.output_dir, "run_summary.json"),
                *glob.glob(os.path.join(args.output_dir, "round_[0-9][0-9]")),
            )
            if os.path.exists(path)
        ]
        if stale:
            raise RuntimeError(
                "MR-ME output directory contains prior state but --resume was "
                f"not requested: {stale}")
    bound = mrme_contract.bind_pipeline_state(state, contract)
    _write_or_verify_json(
        os.path.join(args.output_dir, "mrme_contract.json"), contract,
        label="MR-ME contract")
    common.atomic_write_json(state_path, bound)
    return bound


def _load_round_shard_manifest(path: str) -> Dict[str, Any]:
    from tmx_jax.data import validate_round_shard_manifest

    manifest = _load_json_object(path, "MR-ME round shard manifest")
    return validate_round_shard_manifest(manifest)


def _mrme_round_step_base(args: argparse.Namespace, round_idx: int) -> int:
    span = (
        int(args.scouts_per_round) * int(args.scout_steps)
        + int(args.central_steps))
    return (int(round_idx) - 1) * span


def _execute_mrme_round_scouts(
    args: argparse.Namespace, contract: Dict[str, Any], pool: VLLMPool,
    *, round_idx: int, scout_init: str, shard_manifest: Dict[str, Any],
    tokenizer, wandb_run, state: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Train/resume K Scouts and return only fully byte-validated raw pools."""

    def execute_one(plan: Dict[str, Any]) -> Dict[str, Any]:
        scout_index = int(plan["scout_index"])
        stage = f"scout_{scout_index:02d}"
        source_path = os.path.join(plan["output_dir"], "mrme_scout_source.json")
        trajectory_path = os.path.join(
            plan["output_dir"], "explorer_trajectories.jsonl")
        final_path = os.path.join(plan["output_dir"], "final")
        scout_offset = (
            _mrme_round_step_base(args, round_idx)
            + (scout_index - 1) * int(args.scout_steps))
        c9_multi = mrme_contract.is_c9_multi_explorer(contract)
        gate_path = os.path.join(
            plan["output_dir"], "c9_behavior_gates", "scout_final",
            "evidence.json")
        multi_identity = {
            "contract": contract,
            "round_index": int(round_idx),
            "scout_index": scout_index,
            "seed": int(plan["seed"]),
        }

        def current_gate() -> Optional[Dict[str, Any]]:
            if not c9_multi:
                return None
            if not os.path.isfile(gate_path):
                return None
            return _validate_persisted_c9_behavior_gate(
                args, evidence_path=gate_path, stage="scout_final",
                model_dir=final_path, accepted_path=None,
                multi_identity=multi_identity)

        def current_source(
            model_dir: str, gate: Optional[Dict[str, Any]],
            final_artifact: Optional[Dict[str, Any]],
        ) -> Dict[str, Any]:
            return mrme_contract.build_scout_source_record(
                contract=contract,
                round_index=round_idx,
                scout_index=scout_index,
                trajectory_path=trajectory_path,
                shard_manifest=shard_manifest,
                scout_init_identity=plan["scout_init_identity"],
                scout_final=model_dir,
                metrics_path=(
                    os.path.join(args.output_dir, "metrics.jsonl")
                    if c9_multi else None),
                global_step_offset=(scout_offset if c9_multi else None),
                behavior_gate=gate,
                behavior_gate_path=(gate_path if c9_multi else None),
                scout_final_artifact=final_artifact,
            )

        def current_final_artifact(
            model_dir: str,
        ) -> Optional[Dict[str, Any]]:
            return _validated_grpo_final_artifact_binding(
                plan["output_dir"], model_dir, phase="scout",
                round_idx=round_idx, steps=int(args.scout_steps),
                required=c9_multi)

        done = _stage_done(state, round_idx, stage)
        if done:
            if os.path.realpath(str(done.get("final", ""))) != \
                    os.path.realpath(final_path):
                raise RuntimeError(
                    f"round {round_idx} Scout {scout_index} final path drift")
            recorded_path = os.path.realpath(
                str(done.get("source_record_path", "")))
            if recorded_path != os.path.realpath(source_path):
                raise RuntimeError(
                    f"round {round_idx} Scout {scout_index} source-record path drift")
            recorded = _load_json_object(
                recorded_path, f"round {round_idx} Scout {scout_index} source")
            final_artifact = current_final_artifact(
                str(done.get("final", "")))
            gate = current_gate()
            if c9_multi and gate is None:
                raise RuntimeError(
                    f"round {round_idx} Scout {scout_index} lacks its "
                    "required C9 behavior gate")
            rebuilt = current_source(
                str(done.get("final", "")), gate, final_artifact)
            if recorded != rebuilt or done.get("source_record_sha256") != rebuilt[
                    "record_sha256"]:
                raise RuntimeError(
                    f"round {round_idx} Scout {scout_index} source bytes drifted")
            if done.get("model_artifact") != final_artifact:
                raise RuntimeError(
                    f"round {round_idx} Scout {scout_index} artifact binding drifted")
            return rebuilt

        if os.path.isfile(os.path.join(final_path, "trainer_state.json")):
            # Crash after durable final/source creation but before pipeline
            # state: recover only if every raw row and model byte validates.
            final_artifact = current_final_artifact(final_path)
            gate = current_gate()
            if c9_multi and gate is None:
                gate, _ = _run_c9_behavior_gate(
                    args, pool, tokenizer, stage="scout_final",
                    model_dir=final_path, accepted_path=None,
                    round_dir=plan["output_dir"],
                    multi_identity=multi_identity)
            source = current_source(final_path, gate, final_artifact)
        else:
            gargs = _grpo_args(
                args,
                phase="scout",
                model=scout_init,
                output_dir=plan["output_dir"],
                lam=float(plan["lambda_novelty"]),
                steps=int(args.scout_steps),
                round_idx=round_idx,
                global_step_offset=scout_offset,
                seed_override=int(plan["seed"]),
                run_name_override=str(plan["run_name"]),
                sync_dir_override=str(plan["sync_dir"]),
                num_scouts_override=int(args.scouts_per_round),
            )
            gargs.mrme_contract_sha256 = contract["contract_sha256"]
            gargs.mrme_shard_manifest_sha256 = shard_manifest[
                "manifest_sha256"]
            gargs.mrme_scout_index = scout_index
            gargs.mrme_scouts_per_round = int(args.scouts_per_round)
            scout_final = grpo_gpu.run_grpo(gargs, pool=pool)
            if os.path.realpath(scout_final) != os.path.realpath(final_path):
                raise RuntimeError(
                    f"round {round_idx} Scout {scout_index} returned an "
                    "unexpected final-model path")
            final_artifact = current_final_artifact(scout_final)
            gate = None
            if c9_multi:
                gate, _ = _run_c9_behavior_gate(
                    args, pool, tokenizer, stage="scout_final",
                    model_dir=scout_final, accepted_path=None,
                    round_dir=plan["output_dir"],
                    multi_identity=multi_identity)
            source = current_source(scout_final, gate, final_artifact)

        _write_or_verify_json(
            source_path, source,
            label=f"round {round_idx} Scout {scout_index} source record")
        _mark_stage(
            args, state, round_idx, stage,
            final=source["scout_final"],
            traj_paths=[source["trajectory_path"]],
            seed=int(plan["seed"]),
            lambda_novelty=float(plan["lambda_novelty"]),
            shard_manifest_sha256=source["shard_manifest_sha256"],
            mrme_contract_sha256=contract["contract_sha256"],
            source_record_path=source_path,
            source_record_sha256=source["record_sha256"],
            model_artifact=source.get("scout_final_artifact"),
            c9_behavior_gate_path=source.get("c9_behavior_gate_path"),
            c9_behavior_gate_evidence_sha256=(
                source.get("c9_multi_explorer_scout_evidence", {}) or {}
            ).get("behavior_gate", {}).get("evidence_sha256"),
        )
        if wandb_run is not None:
            wandb_run.log({
                "round": int(round_idx),
                "lambda": float(plan["lambda_novelty"]),
                "pipeline/stage": 1,
                "pipeline/mrme_scout_index": scout_index,
                "pipeline/mrme_scout_complete": 1.0,
            })
        _push_results_artifact(
            wandb_run, args, f"r{round_idx}-mrme-scout-{scout_index:02d}")
        return source

    return mrme_contract.execute_round_scouts(
        contract,
        round_idx,
        output_dir=args.output_dir,
        sync_dir=args.sync_dir,
        run_name=args.run_name,
        scout_init=scout_init,
        expected_shard_manifest_sha256=str(
            shard_manifest["manifest_sha256"]),
        execute_one=execute_one,
    )


def run_mrme_round(
    args: argparse.Namespace, contract: Dict[str, Any], pool: VLLMPool,
    tokenizer, examples, round_idx: int, scout_init: str,
    shard_manifest: Dict[str, Any], wandb_run,
    metrics_logger: common.MetricsLogger, state: Dict[str, Any],
) -> Dict[str, Any]:
    """One true MR-ME round: K raw Scouts, one pool, one SFT, one Central."""
    if getattr(args, "_mrme_contract", None) != contract:
        raise RuntimeError("MR-ME runtime args are not bound to this contract")
    mrme_contract.bind_pipeline_state(state, contract)
    spec = mrme_contract.round_spec(contract, round_idx)
    sources = _execute_mrme_round_scouts(
        args, contract, pool, round_idx=round_idx, scout_init=scout_init,
        shard_manifest=shard_manifest, tokenizer=tokenizer,
        wandb_run=wandb_run, state=state)
    round_dir = os.path.join(args.output_dir, f"round_{round_idx:02d}")
    source_index = {
        "version": "gpu_mrme_round_sources_v1",
        "contract_sha256": contract["contract_sha256"],
        "round_index": int(round_idx),
        "shard_manifest_sha256": shard_manifest["manifest_sha256"],
        "sources": sources,
    }
    source_index["source_index_sha256"] = common.canonical_json_sha256(
        source_index)
    source_index_path = os.path.join(round_dir, "mrme_scout_sources.json")
    _write_or_verify_json(
        source_index_path, source_index, label="MR-ME round source index")

    trajectory_manifest = mrme_contract.build_trajectory_source_manifest(
        contract, round_idx, sources)
    trajectory_manifest_path = os.path.join(
        round_dir, "mrme_trajectory_source_manifest.json")
    _write_or_verify_json(
        trajectory_manifest_path, trajectory_manifest,
        label="MR-ME trajectory source manifest")

    downstream = copy.copy(args)
    downstream._mrme_internal_pool = True
    downstream._mrme_contract = contract
    downstream._mrme_shard_manifest_sha256 = shard_manifest[
        "manifest_sha256"]
    downstream.skip_scout = True
    downstream.trajectories = [source["trajectory_path"] for source in sources]
    downstream.trajectory_source_manifest = trajectory_manifest_path
    summary = run_round(
        downstream, pool, tokenizer, examples, round_idx,
        float(spec["lambda_novelty"]), scout_init,
        wandb_run, metrics_logger, state)
    if mrme_contract.is_c9_multi_explorer(contract):
        c9_evidence.require_shared_probe_prompt_matrix(
            [
                *(source["c9_multi_explorer_scout_evidence"]["probe_history"]
                  for source in sources),
                summary["c9_central_probe_history"],
            ],
            context=f"C9 MR-ME round {round_idx}")

    expected_source_digest = common.trajectory_source_manifest_digest(
        trajectory_manifest)
    sft_provenance = _load_json_object(
        os.path.join(summary["sft_final"], "sft_provenance.json"),
        "MR-ME SFT provenance")
    if sft_provenance.get("trajectory_source", {}).get("digest") != \
            expected_source_digest:
        raise RuntimeError(
            f"round {round_idx} SFT is not bound to the K raw Scout sources")
    summary.update({
        **_round_campaign_summary_identity(contract),
        "stabilization_profile": contract["stabilization_profile"],
        "scientific_variant": contract["scientific_variant"],
        "external_p1_required": True,
        "headline_eligible": False,
        "scouts_per_round": int(args.scouts_per_round),
        "scout_finals": [source["scout_final"] for source in sources],
        "mrme_source_index": source_index_path,
        "trajectory_source_manifest": trajectory_manifest_path,
        "trajectory_source_digest": expected_source_digest,
        "dataset_shard_manifest_sha256": shard_manifest["manifest_sha256"],
        "dataset_shard_rows": int(shard_manifest["num_rows"]),
    })
    return summary


def validate_args(args: argparse.Namespace) -> None:
    """Fail-fast argument cross-checks (silent misconfigurations observed in
    the field: ignored trajectories, unreachable min-accepted floors)."""
    scouts_per_round = int(args.scouts_per_round)
    if scouts_per_round < 1:
        raise ValueError("--scouts-per-round must be >= 1")
    single_scout_multiround = _is_single_scout_multiround_profile(args)
    true_mrme = _true_mrme(args)
    c9_multi = bool(str(
        getattr(args, "c9_multiexplorer_profile", "") or ""))
    c9_round_campaign = c9_multi or single_scout_multiround
    strict_training_environment = bool(
        str(getattr(args, "c9_execution_profile", "") or "")
        or c9_round_campaign
        or str(getattr(args, "expdis_stabilization_mode", "") or "")
        == grpo_gpu.C9_RND_LIFECYCLE
    )
    if strict_training_environment:
        from tmx_gpu import orchestrator_receipt

        if str(getattr(args, "container_image_identity", "") or "") != \
                grpo_gpu.CANONICAL_TRAINING_CONTAINER_IMAGE:
            raise ValueError(
                "strict pipeline requires the exact canonical container image")
        code_binding = getattr(args, "_code_artifact_binding", None)
        if code_binding is None:
            code_binding = grpo_gpu.load_code_source_manifest(str(
                getattr(args, "code_source_manifest", "") or ""))
        if code_binding is None:
            raise ValueError(
                "strict pipeline requires --code-source-manifest")
        source = code_binding["source"]
        expected_receipt = {
            "expected_run_name": str(args.run_name),
            "expected_container_image": str(
                getattr(args, "container_image_identity", "") or ""),
            "expected_code_artifact_ref": source["artifact_ref"],
            "expected_code_artifact_digest": source["artifact_digest"],
            "expected_command_sha256": str(getattr(
                args, "orchestrator_command_sha256", "") or ""),
            "expected_polling_artifact_ref": str(getattr(
                args, "orchestrator_receipt_artifact", "") or ""),
        }
        receipt_binding = getattr(
            args, "_orchestrator_receipt_binding", None)
        if receipt_binding is None:
            receipt_binding = orchestrator_receipt.load_binding(
                str(getattr(args, "orchestrator_receipt_binding", "") or ""),
                **expected_receipt,
            )
        else:
            orchestrator_receipt.validate_binding(
                receipt_binding, **expected_receipt)
        args._code_artifact_binding = code_binding
        args._orchestrator_receipt_binding = receipt_binding
    if c9_multi and scouts_per_round <= 1:
        raise ValueError(
            "--c9-multiexplorer-profile requires --scouts-per-round > 1; "
            "SingleScout full/C5 remains a separate preregistered profile")
    if single_scout_multiround:
        if str(args.single_scout_multiround_profile) != \
                mrme_contract.SINGLESCOUT_MULTIROUND_PROFILE:
            raise ValueError(
                "unknown --single-scout-multiround-profile identity")
        if scouts_per_round != 1:
            raise ValueError(
                "--single-scout-multiround-profile requires "
                "--scouts-per-round 1")
        if not 1 <= int(args.rounds) <= 4:
            raise ValueError(
                "--single-scout-multiround-profile requires 1 <= --rounds <= 4")
        if c9_multi or bool(str(args.c9_execution_profile or "")) or \
                bool(str(args.stabilization_cell or "")):
            raise ValueError(
                "--single-scout-multiround-profile is mutually exclusive "
                "with C9 execution and multi-explorer profiles")
        lambdas = parse_lambda_schedule(args)
        if int(args.rounds) > 1 and any(
                later >= earlier
                for earlier, later in zip(lambdas, lambdas[1:])):
            raise ValueError(
                "single-Scout multi-round requires a predeclared strictly "
                "decreasing lambda schedule")
    if true_mrme:
        if scouts_per_round > 1 and int(args.rounds) <= 1 and not c9_multi:
            raise ValueError(
                "R=1 integrated MultiScout requires the explicit stabilized "
                "--c9-multiexplorer-profile")
        incompatible = {
            "skip_scout": bool(args.skip_scout),
            "trajectories": bool(args.trajectories),
            "scout_only": bool(args.scout_only),
            "stop_after_sft": bool(args.stop_after_sft),
            "sft_model": bool(args.sft_model),
            "sft_evidence_index": bool(args.sft_evidence_index),
            "frozen_sft_library_evidence_index": bool(
                args.frozen_sft_library_evidence_index),
            "fresh_c8_sft_gate": bool(args.fresh_c8_sft_gate),
            "fresh_c8_multiscout_component": bool(
                args.fresh_c8_multiscout_component),
            "trajectory_source_manifest": bool(args.trajectory_source_manifest),
            "c9_execution_profile": bool(args.c9_execution_profile),
            "stabilization_cell": bool(args.stabilization_cell),
            "endpoints": bool(args.endpoints),
        }
        enabled = sorted(name for name, value in incompatible.items() if value)
        if enabled:
            raise ValueError(
                "true MR-ME owns its K Scout sources and one downstream "
                "handoff per round; incompatible options: " + ", ".join(enabled))
        exact = {
            "policy": mrme_contract.MRME_POOLING_POLICY,
            "min_accepted": 500,
            "max_accepted": 500,
            "harvest_max_passes": 0,
            "c8_source_balance": False,
            "rl_validity_mode": "c8",
        }
        if c9_round_campaign:
            exact.update({
                "expdis_stabilization_mode": grpo_gpu.C9_RND_LIFECYCLE,
                "novelty_normalization": "selected_batch_zscore",
                "scientific_variant":
                    mrme_contract.C9_MULTIEXPLORER_SCIENTIFIC_VARIANT,
                "scout_steps": 200,
                "central_steps": 100,
                "prompts_per_step": 4,
                "num_generations": 16,
                "reload_every": 1,
                "probe_every": 25,
                "completion_budget": 0,
                "mask_truncated": False,
                "ppo_mode": False,
                "updates_per_rollout": 1,
                "novelty_layers": "7,14,21",
                "model": "Qwen/Qwen3-1.7B",
                "model_revision": common.CANONICAL_TRAINING_MODEL_REVISION,
                "dataset": "dapo_math_17k",
                "max_train_examples": 20000,
                "sft_grad_accum": 1,
                "sft_epochs": 2,
                "sft_termination_mode": "append_eos",
                "sft_save_steps": (),
                "sft_optimizer": "adamw",
                "optimizer": "adamw",
                "param_dtype": "float32",
                "probe_dataset": "aime24",
                "vllm_max_model_len": 36864,
            })
        else:
            exact.update({
                "expdis_stabilization_mode": grpo_gpu.C8_RND_LIFECYCLE,
                "novelty_normalization": "raw",
            })
        drift = [
            f"{name}={getattr(args, name)!r} (expected {expected!r})"
            for name, expected in exact.items()
            if getattr(args, name) != expected
        ]
        if drift:
            raise ValueError(
                "MR-ME experiment-contract drift: " + "; ".join(drift))
        if c9_round_campaign:
            if not bool(args.wandb) or str(os.environ.get(
                    "WANDB_MODE", "")).strip().lower() in {
                        "offline", "dryrun", "disabled"}:
                raise ValueError(
                    "C9 round campaign requires online --wandb so the pooled "
                    "SFT model is durably uploaded before Central")
            if int(args.ckpt_artifact_every) <= 0:
                raise ValueError(
                    "C9 round campaign requires --ckpt-artifact-every > 0 "
                    "so every Scout/Central final is durably uploaded")
            if args.scout_lr is None or not math.isclose(
                    float(args.scout_lr), 1e-6, rel_tol=0.0, abs_tol=1e-15):
                raise ValueError(
                    "C9 round campaign requires explicit --scout-lr 1e-6")
            if args.central_lr is not None and not math.isclose(
                    float(args.central_lr), 1e-6,
                    rel_tol=0.0, abs_tol=1e-15):
                raise ValueError(
                    "C9 round campaign Central LR must be None or 1e-6")
            if not math.isclose(
                    float(args.sft_lr), 5e-6,
                    rel_tol=0.0, abs_tol=1e-15):
                raise ValueError("C9 round campaign requires SFT LR 5e-6")
            if _resolved_dynamic_max_attempts(args) != 64:
                raise ValueError(
                    "C9 round campaign requires dynamic_max_attempts=64")
        if not args.code_source_manifest or not os.path.isfile(
                args.code_source_manifest):
            raise ValueError(
                "true MR-ME requires an immutable --code-source-manifest")
        if not str(args.vllm_gpus).strip():
            raise ValueError(
                "true MR-ME requires a managed --vllm-gpus pool so each "
                "Scout/SFT/Central handoff reloads the exact current policy")
        if int(args.scout_steps) <= 0 or int(args.central_steps) <= 0:
            raise ValueError("MR-ME requires positive Scout and Central steps")
        if int(args.prompts_per_step) <= 0 or int(args.num_generations) <= 0:
            raise ValueError("MR-ME rollout geometry must be positive")
        if int(args.max_train_examples) <= 0:
            raise ValueError("MR-ME requires --max-train-examples > 0")
        if int(args.seed) < 0:
            raise ValueError("MR-ME requires a non-negative base seed")
    if args.trajectories and not args.skip_scout:
        raise ValueError(
            "--trajectories was given without --skip-scout: the pooled "
            "trajectory files would be silently IGNORED and a fresh scout "
            "trained instead (a MultiScout run degenerating to single-scout). "
            "Pass --skip-scout for a pooled run (expdis_job.sh sets SKIP_SCOUT=1 "
            "automatically when TRAJ_ARTIFACTS is set), or drop --trajectories.")
    if args.skip_scout and not args.trajectories:
        raise ValueError("--skip-scout requires --trajectories")
    if args.skip_scout and args.trajectories:
        _reject_multiscout_component_local_trajectory_inputs(
            list(args.trajectories))
    if args.stop_after_sft and int(args.rounds) != 1:
        raise ValueError("--stop-after-sft is an SFT evaluation gate and requires --rounds 1")
    if args.stop_after_sft and args.scout_only:
        raise ValueError("--stop-after-sft cannot be combined with --scout-only")
    if args.sft_evidence_index and not args.sft_model:
        raise ValueError("--sft-evidence-index requires --sft-model")
    frozen_sft_training = bool(args.frozen_sft_library_evidence_index)
    fresh_sft_training = bool(args.fresh_c8_sft_gate)
    fresh_multiscout_component = bool(args.fresh_c8_multiscout_component)
    _fresh_c8_scout_evidence_profile(args)
    protected_c8_profile = fresh_sft_training or fresh_multiscout_component
    if protected_c8_profile and (
            args.expdis_stabilization_mode != grpo_gpu.C8_RND_LIFECYCLE
            or args.novelty_normalization != "raw"
            or args.scout_lr is not None
            or args.central_lr is not None):
        raise ValueError(
            "frozen fresh-c8 evidence profiles require candidate-attempt RND, "
            "raw novelty, and implicit 5e-6/1e-6 GRPO learning rates; C9 is a "
            "separate scientific variant")

    c10_profile = str(args.c10_execution_profile or "")
    if c10_profile:
        c10_evidence.validate_preregistration_file(REPO_ROOT)
        c10_evidence.validate_canonical_dataset_environment(os.environ)
        if args.c9_execution_profile or args.stabilization_cell or \
                args.c9_multiexplorer_profile or \
                args.single_scout_multiround_profile or \
                int(args.scouts_per_round) != 1 or int(args.rounds) != 1:
            raise ValueError(
                "C10 canary is mutually exclusive with C9 cells, "
                "MultiScout, multi-round, and MR-ME profiles")
        if parse_lambda_schedule(args) != [0.5]:
            raise ValueError("C10 canary requires exactly lambda=0.5")
        c10_expected = {
            "c10_execution_profile": c10_evidence.C10_PROFILE,
            "run_name": c10_evidence.C10_CANONICAL_RUN_NAME,
            "seed": 0,
            "model": "Qwen/Qwen3-1.7B",
            "model_revision": common.CANONICAL_TRAINING_MODEL_REVISION,
            "dataset": "dapo_math_17k",
            "max_train_examples": 20000,
            "prompts_per_step": 4,
            "num_generations": 16,
            "rl_validity_mode": "c8",
            "reload_every": 1,
            "save_every": 25,
            "keep_checkpoints": 1,
            "ckpt_artifact_every": 25,
            "completion_budget": 0,
            "mask_truncated": True,
            "truncation_baseline_mode": "unmasked_survivors",
            "ppo_mode": False,
            "updates_per_rollout": 1,
            "novelty_layers": "7,14,21",
            "expdis_stabilization_mode": grpo_gpu.C9_RND_LIFECYCLE,
            "novelty_normalization": "selected_batch_zscore",
            "scientific_variant": grpo_gpu.C10_SCIENTIFIC_VARIANT,
            "scout_lr": 1e-6,
            "policy": filter_pool_mod.C8_POLICY,
            "vllm_max_model_len": 36864,
            "optimizer": "adamw",
            "sft_optimizer": "adamw",
            "param_dtype": "float32",
            "endpoints": "",
            "probe_dataset": "aime24",
            "scout_steps": 50,
            "scout_only": True,
            "skip_scout": False,
            "probe_every": 0,
            "stop_after_sft": False,
        }
        drift = [
            f"{name}={getattr(args, name)!r} (expected {expected!r})"
            for name, expected in c10_expected.items()
            if getattr(args, name) != expected
        ]
        if _resolved_dynamic_max_attempts(args) != 64:
            drift.append(
                f"dynamic_max_attempts={_resolved_dynamic_max_attempts(args)!r} "
                "(expected 64)")
        if not str(args.vllm_gpus or ""):
            drift.append("vllm_gpus is empty (C10 requires a managed pool)")
        if not bool(args.wandb) or str(os.environ.get(
                "WANDB_MODE", "")).strip().lower() in {
                    "offline", "dryrun", "disabled"}:
            drift.append("online wandb is required for durable C10 evidence")
        try:
            c10_evidence.load_exclusive_lease_binding(
                args.c10_exclusive_lease_binding,
                expected_orchestrator_binding=
                    args._orchestrator_receipt_binding)
        except ValueError as exc:
            drift.append(f"exclusive C10 lease is invalid ({exc})")
        if drift:
            raise ValueError(
                "C10 canary contract drift: " + "; ".join(drift))
        _validated_executing_code_source_manifest(
            args, profile_label="C10 canary")

    profile = str(args.c9_execution_profile or "")
    cell = str(args.stabilization_cell or "")
    if bool(profile) != bool(cell):
        raise ValueError(
            "--c9-execution-profile and --stabilization-cell are required together")
    if profile:
        if int(args.rounds) != 1 or parse_lambda_schedule(args) != [0.5]:
            raise ValueError("C9 profiles require one round and lambda=0.5")
        common_expected = {
            "seed": 0,
            "model": "Qwen/Qwen3-1.7B",
            "model_revision": common.CANONICAL_TRAINING_MODEL_REVISION,
            "dataset": "dapo_math_17k",
            "max_train_examples": 20000,
            "prompts_per_step": 4,
            "num_generations": 16,
            "rl_validity_mode": "c8",
            "reload_every": 1,
            "completion_budget": 0,
            "mask_truncated": False,
            "truncation_baseline_mode": "all_rows",
            "ppo_mode": False,
            "updates_per_rollout": 1,
            "novelty_layers": "7,14,21",
            "policy": filter_pool_mod.C8_POLICY,
            "vllm_max_model_len": 36864,
            "optimizer": "adamw",
            "sft_optimizer": "adamw",
            "param_dtype": "float32",
            "endpoints": "",
            "probe_dataset": "aime24",
        }
        drift = [
            f"{name}={getattr(args, name)!r} (expected {expected!r})"
            for name, expected in common_expected.items()
            if getattr(args, name) != expected
        ]
        if _resolved_dynamic_max_attempts(args) != 64:
            drift.append(
                f"dynamic_max_attempts={_resolved_dynamic_max_attempts(args)!r} "
                "(expected 64)")
        cell_contract = {
            "C1": (grpo_gpu.C8_RND_LIFECYCLE, "raw", 1e-6,
                   grpo_gpu.C9_SCREEN_VARIANT),
            "C2": (grpo_gpu.C9_RND_LIFECYCLE, "raw", 5e-6,
                   grpo_gpu.C9_SCREEN_VARIANT),
            "C3": (grpo_gpu.C9_RND_LIFECYCLE, "raw", 1e-6,
                   grpo_gpu.C9_SCREEN_VARIANT),
            "C4": (grpo_gpu.C9_RND_LIFECYCLE, "selected_batch_zscore", 5e-6,
                   grpo_gpu.C9_SCREEN_VARIANT),
            "C5": (grpo_gpu.C9_RND_LIFECYCLE, "selected_batch_zscore", 1e-6,
                   grpo_gpu.C9_SCIENTIFIC_VARIANT),
        }
        lifecycle, normalization, expected_lr, expected_variant = cell_contract[cell]
        if args.expdis_stabilization_mode != lifecycle:
            drift.append(
                f"expdis_stabilization_mode={args.expdis_stabilization_mode!r} "
                f"(expected {lifecycle!r})")
        if args.novelty_normalization != normalization:
            drift.append(
                f"novelty_normalization={args.novelty_normalization!r} "
                f"(expected {normalization!r})")
        if args.scout_lr is None or not math.isclose(
                float(args.scout_lr), expected_lr, rel_tol=0.0, abs_tol=1e-15):
            drift.append(
                f"scout_lr={args.scout_lr!r} (expected {expected_lr!r})")
        if args.scientific_variant != expected_variant:
            drift.append(
                f"scientific_variant={args.scientific_variant!r} "
                f"(expected {expected_variant!r})")
        if profile == "canary":
            canary_expected = {
                "scout_steps": 50,
                "scout_only": True,
                "skip_scout": False,
                "probe_every": 0,
                "stop_after_sft": False,
            }
            drift.extend(
                f"{name}={getattr(args, name)!r} (expected {expected!r})"
                for name, expected in canary_expected.items()
                if getattr(args, name) != expected
            )
        elif profile == "full":
            full_expected = {
                "stabilization_cell": "C5",
                "scout_steps": 200,
                "central_steps": 100,
                "scout_only": False,
                "skip_scout": False,
                "probe_every": 25,
                "min_accepted": 500,
                "max_accepted": 500,
                "harvest_max_passes": 0,
                "sft_grad_accum": 1,
                "sft_epochs": 2,
                "sft_termination_mode": "append_eos",
                "sft_save_steps": (),
                "stop_after_sft": False,
            }
            drift.extend(
                f"{name}={getattr(args, name)!r} (expected {expected!r})"
                for name, expected in full_expected.items()
                if getattr(args, name) != expected
            )
            if args.central_lr is not None and not math.isclose(
                    float(args.central_lr), 1e-6, rel_tol=0.0, abs_tol=1e-15):
                drift.append(
                    f"central_lr={args.central_lr!r} (expected None or 1e-6)")
            if not math.isclose(
                    float(args.sft_lr), 5e-6, rel_tol=0.0, abs_tol=1e-15):
                drift.append(f"sft_lr={args.sft_lr!r} (expected 5e-6)")
            if not str(args.vllm_gpus or ""):
                drift.append("vllm_gpus is empty (C9 gates require a managed pool)")
            if not bool(args.wandb) or str(os.environ.get(
                    "WANDB_MODE", "")).strip().lower() in {
                        "offline", "dryrun", "disabled"}:
                drift.append(
                    "online wandb is required for durable Scout/SFT/Central artifacts")
            if int(args.ckpt_artifact_every) <= 0:
                drift.append(
                    "ckpt_artifact_every must be > 0 for durable GRPO finals")
        if drift:
            raise ValueError(
                f"C9 {profile}/{cell} contract drift: " + "; ".join(drift))
        _validated_executing_code_source_manifest(
            args, profile_label=f"C9 {profile}/{cell}")

    if args.expdis_stabilization_mode == grpo_gpu.C9_RND_LIFECYCLE and not \
            c9_round_campaign:
        if args.skip_scout:
            raise ValueError(
                "C9 pipeline provenance currently requires training its own "
                "Scout; pooled C9 consumers need a separate signed handoff")
        for lam in parse_lambda_schedule(args):
            grpo_gpu.validate_expdis_stabilization(
                args, phase="scout", lambda_novelty=lam)
        if not profile:
            _validated_executing_code_source_manifest(
                args, profile_label="C9 selected-batch RND")
    elif not c9_round_campaign:
        # This also rejects a C9-only z-score or label attached to frozen C8.
        grpo_gpu.validate_expdis_stabilization(
            args, phase="scout",
            lambda_novelty=parse_lambda_schedule(args)[0])
    if frozen_sft_training and args.sft_model:
        raise ValueError(
            "--frozen-sft-library-evidence-index and --sft-model are mutually exclusive")
    if fresh_sft_training and (frozen_sft_training or args.sft_model):
        raise ValueError(
            "--fresh-c8-sft-gate is mutually exclusive with frozen-library "
            "training and external --sft-model continuation")
    if fresh_multiscout_component and (
            frozen_sft_training or fresh_sft_training or args.sft_model):
        raise ValueError(
            "--fresh-c8-multiscout-component is a disjoint Scout-only "
            "profile and cannot use any SFT training/continuation route")
    if args.sft_training_evidence_manifest:
        raise ValueError(
            "--sft-training-evidence-manifest is pipeline-internal and cannot "
            "be supplied by a caller")
    if args.trajectory_source_manifest and not os.path.isfile(args.trajectory_source_manifest):
        raise ValueError(
            "--trajectory-source-manifest is not a file: "
            f"{args.trajectory_source_manifest!r}")
    if args.sft_model:
        if int(args.rounds) != 1:
            raise ValueError("--sft-model requires --rounds 1")
        if not args.skip_scout:
            raise ValueError(
                "--sft-model requires --skip-scout with the original "
                "--trajectories so filtering and handoff provenance still run")
        if args.scout_only:
            raise ValueError("--sft-model cannot be combined with --scout-only")
        if args.stop_after_sft:
            raise ValueError(
                "--sft-model is already a health-gate-selected checkpoint and "
                "must continue to Central; do not combine it with --stop-after-sft")
        if int(args.harvest_max_passes) != 0:
            raise ValueError(
                "--sft-model requires --harvest-max-passes 0: the selected SFT "
                "checkpoint must be continued with its original Scout "
                "trajectories, not newly harvested rows")
        if int(args.central_steps) <= 0:
            raise ValueError("--sft-model requires --central-steps > 0")
        if int(args.central_steps) != 100:
            raise ValueError(
                "external frozen diagnostic requires --central-steps 100")
        if args.model != sft_selector.DIAGNOSTIC_MODEL_INIT:
            raise ValueError(
                "external ga8 diagnostic requires base model "
                f"{sft_selector.DIAGNOSTIC_MODEL_INIT}")
        if int(args.min_accepted) > 500 or int(args.max_accepted) != 500:
            raise ValueError(
                "external ga8 diagnostic requires the frozen 500-example library "
                "(--max-accepted 500 and --min-accepted <= 500)")
        if not os.path.isdir(args.sft_model):
            raise ValueError(f"--sft-model is not a local directory: {args.sft_model!r}")
        if not os.path.isfile(os.path.join(args.sft_model, "config.json")):
            raise ValueError(
                f"--sft-model {args.sft_model!r} is not a Hugging Face model "
                "directory (config.json is missing)")
        checkpoint_info = _sft_checkpoint_metadata(args.sft_model)
        recorded_termination = checkpoint_info["metadata"].get("termination_mode")
        if (recorded_termination is not None
                and recorded_termination != args.sft_termination_mode):
            raise ValueError(
                f"--sft-model records termination_mode={recorded_termination!r}, "
                f"but --sft-termination-mode={args.sft_termination_mode!r}")
        required_evidence = {
            "--sft-evidence-index": args.sft_evidence_index,
            "--trajectory-source-manifest": args.trajectory_source_manifest,
        }
        missing = [name for name, value in required_evidence.items() if not value]
        if missing:
            raise ValueError(
                "--sft-model external continuation requires exact gate/selection/"
                f"source evidence; missing {', '.join(missing)}")
        for flag, path in (
            ("--sft-evidence-index", args.sft_evidence_index),
            ("--trajectory-source-manifest", args.trajectory_source_manifest),
        ):
            if not os.path.isfile(path):
                raise ValueError(f"{flag} is not a file: {path!r}")
        external_profile = _external_evidence_profile(args)
        _validated_external_accepted(args)
        args.scientific_variant = (
            _fresh_external_declared_variant(args)
            if external_profile == fresh_c8_sft.PROFILE
            else "external_sft_exposure_gated")
        if external_profile == fresh_c8_sft.PROFILE:
            fresh_exact = {
                "central_steps": 100,
                "policy": filter_pool_mod.C8_POLICY,
                "min_accepted": 500,
                "max_accepted": 500,
                "harvest_max_passes": 0,
                "prompts_per_step": 4,
                "num_generations": 16,
                "completion_budget": 0,
                "rl_validity_mode": "c8",
                "dataset": "dapo_math_17k",
                "max_train_examples": 20000,
                "seed": 0,
                "optimizer": "adamw",
                "mask_truncated": False,
                "ppo_mode": False,
                "updates_per_rollout": 1,
                "scout_lr": None,
                "central_lr": None,
                "expdis_stabilization_mode": grpo_gpu.C8_RND_LIFECYCLE,
                "novelty_normalization": "raw",
            }
            drift = [
                f"{name}={getattr(args, name)!r} (expected {expected!r})"
                for name, expected in fresh_exact.items()
                if getattr(args, name) != expected
            ]
            if drift:
                raise ValueError(
                    "fresh-c8 Central continuation contract drift: "
                    + "; ".join(drift))
    if frozen_sft_training:
        if int(args.rounds) != 1 or not args.skip_scout or not args.trajectories:
            raise ValueError(
                "frozen-library ga8 SFT requires --rounds 1, --skip-scout, "
                "and the exact source --trajectories")
        if not args.stop_after_sft:
            raise ValueError(
                "frozen-library ga8 route is training-only and requires "
                "--stop-after-sft before any health-gated Central continuation")
        if int(args.harvest_max_passes) != 0:
            raise ValueError(
                "frozen-library ga8 SFT requires --harvest-max-passes 0")
        if args.model != sft_selector.DIAGNOSTIC_MODEL_INIT:
            raise ValueError(
                "frozen-library ga8 SFT requires base model "
                f"{sft_selector.DIAGNOSTIC_MODEL_INIT}")
        if not args.trajectory_source_manifest:
            raise ValueError(
                "frozen-library ga8 SFT requires --trajectory-source-manifest")
        if int(args.max_accepted) != 500 or int(args.min_accepted) > 500:
            raise ValueError(
                "frozen-library ga8 SFT requires the exact 500-row library")
        exact = {
            "sft_epochs": 2,
            "sft_grad_accum": 8,
            "sft_optimizer": "adamw",
            "optimizer": "adamw",
            "sft_termination_mode": "append_eos",
            "completion_budget": 0,
            "seed": 0,
            "param_dtype": "float32",
        }
        drift = [
            f"{name}={getattr(args, name)!r} (expected {expected!r})"
            for name, expected in exact.items()
            if getattr(args, name) != expected
        ]
        if not math.isclose(float(args.sft_lr), 5e-6, rel_tol=0.0, abs_tol=1e-15):
            drift.append(f"sft_lr={args.sft_lr!r} (expected 5e-6)")
        if tuple(args.sft_save_steps) != sft_selector.PREREGISTERED_STEPS:
            drift.append(
                f"sft_save_steps={tuple(args.sft_save_steps)!r} "
                f"(expected {sft_selector.PREREGISTERED_STEPS!r})")
        if drift:
            raise ValueError(
                "frozen-library ga8 SFT contract drift: " + "; ".join(drift))
        _validated_frozen_sft_training_evidence(args)
        args.scientific_variant = FROZEN_SFT_LIBRARY_VARIANT
    if fresh_sft_training:
        exact = {
            "rounds": 1,
            "scout_steps": 200,
            "central_steps": 100,
            "model": fresh_c8_sft.MODEL_INIT,
            "policy": filter_pool_mod.C8_POLICY,
            "min_accepted": 500,
            "max_accepted": 500,
            "harvest_max_passes": 0,
            "prompts_per_step": 4,
            "num_generations": 16,
            "novelty_layers": "7,14,21",
            "completion_budget": 0,
            "rl_validity_mode": "c8",
            "dataset": "dapo_math_17k",
            "max_train_examples": 20000,
            "seed": 0,
            "sft_epochs": 2,
            "sft_grad_accum": 1,
            "sft_optimizer": "adamw",
            "sft_termination_mode": "append_eos",
            "param_dtype": "float32",
            "mask_truncated": False,
            "ppo_mode": False,
            "updates_per_rollout": 1,
            "optimizer": "adamw",
            "scout_lr": None,
            "central_lr": None,
            "expdis_stabilization_mode": grpo_gpu.C8_RND_LIFECYCLE,
            "novelty_normalization": "raw",
        }
        drift = [
            f"{name}={getattr(args, name)!r} (expected {expected!r})"
            for name, expected in exact.items()
            if getattr(args, name) != expected
        ]
        if args.skip_scout or args.trajectories:
            drift.append("fresh-c8 gated SFT must follow its in-pipeline SingleScout")
        if not args.stop_after_sft:
            drift.append("stop_after_sft=False (required True)")
        if tuple(args.sft_save_steps) != fresh_c8_sft.PREREGISTERED_STEPS:
            drift.append(
                f"sft_save_steps={tuple(args.sft_save_steps)!r} "
                f"(expected {fresh_c8_sft.PREREGISTERED_STEPS!r})")
        if not math.isclose(float(args.sft_lr), 5e-6, rel_tol=0.0, abs_tol=1e-15):
            drift.append(f"sft_lr={args.sft_lr!r} (expected 5e-6)")
        if parse_lambda_schedule(args) != [0.5]:
            drift.append("lambda schedule must resolve to exactly [0.5]")
        resolved_attempts = _resolved_dynamic_max_attempts(args)
        if resolved_attempts != 64:
            drift.append(
                "dynamic_max_attempts resolves to "
                f"{resolved_attempts!r} (expected 64)")
        if not args.code_source_manifest or not os.path.isfile(args.code_source_manifest):
            drift.append("exact --code-source-manifest is required")
        if drift:
            raise ValueError("fresh-c8 gated SFT contract drift: " + "; ".join(drift))
        _validated_code_source_manifest(args)
    if fresh_multiscout_component:
        exact = {
            "rounds": 1,
            "scout_steps": 200,
            "model": fresh_c8_sft.MODEL_INIT,
            "scientific_variant": "",
            "scout_only": True,
            "harvest_max_passes": 0,
            "prompts_per_step": 4,
            "num_generations": 16,
            "novelty_layers": "7,14,21",
            "completion_budget": 0,
            "rl_validity_mode": "c8",
            "dataset": "dapo_math_17k",
            "max_train_examples": 20000,
            "param_dtype": "float32",
            "mask_truncated": False,
            "ppo_mode": False,
            "updates_per_rollout": 1,
            "optimizer": "adamw",
            "save_every": 25,
            "keep_checkpoints": 1,
            "ckpt_artifact_every": 25,
            "probe_every": 25,
            "vllm_max_model_len": 36864,
            "max_consecutive_gen_failures": 3,
            "gen_concurrency": 8,
            "scout_lr": None,
            "central_lr": None,
            "expdis_stabilization_mode": grpo_gpu.C8_RND_LIFECYCLE,
            "novelty_normalization": "raw",
        }
        drift = [
            f"{name}={getattr(args, name)!r} (expected {expected!r})"
            for name, expected in exact.items()
            if getattr(args, name) != expected
        ]
        if int(args.seed) not in fresh_c8_sft.MULTISCOUT_COMPONENT_SEEDS:
            drift.append(
                f"seed={args.seed!r} (expected exactly one of "
                f"{fresh_c8_sft.MULTISCOUT_COMPONENT_SEEDS!r})")
        if args.skip_scout or args.trajectories:
            drift.append(
                "MultiScout component must train its own in-pipeline Scout")
        if args.stop_after_sft or tuple(args.sft_save_steps):
            drift.append(
                "MultiScout component is Scout-only, not an SFT gate route")
        if parse_lambda_schedule(args) != [0.5]:
            drift.append("lambda schedule must resolve to exactly [0.5]")
        resolved_attempts = _resolved_dynamic_max_attempts(args)
        if resolved_attempts != 64:
            drift.append(
                "dynamic_max_attempts resolves to "
                f"{resolved_attempts!r} (expected 64)")
        if not args.code_source_manifest \
                or not os.path.isfile(args.code_source_manifest):
            drift.append("exact --code-source-manifest is required")
        if drift:
            raise ValueError(
                "fresh-c8 MultiScout component contract drift: "
                + "; ".join(drift))
        _validated_multiscout_component_code_source_manifest(args)
    if int(args.min_accepted) > int(args.max_accepted):
        raise ValueError(
            f"--min-accepted {args.min_accepted} > --max-accepted "
            f"{args.max_accepted}: the filter stage caps accepted rows at "
            "--max-accepted, so the floor could never be met (unrecoverable "
            "crash loop)")


def build_arg_parser() -> argparse.ArgumentParser:
    # Evidence-profile switches are security/attribution boundaries.  Python's
    # default long-option abbreviation would otherwise accept e.g.
    # ``--fresh-c8-m`` as ``--fresh-c8-multiscout-component`` and bypass
    # launcher's protected-token checks.
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], allow_abbrev=False)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument(
        "--model-revision", default=common.CANONICAL_TRAINING_MODEL_REVISION,
        help="immutable Hub revision for remote Base loads; never applied to "
             "local Scout/SFT/Central handoffs",
    )
    p.add_argument("--scientific-variant", default="", help=argparse.SUPPRESS)
    p.add_argument(
        "--container-image-identity", default="",
        help="immutable OCI image tag@sha256 identity required for strict "
             "C9/MR-ME Scout, SFT, and Central training",
    )
    p.add_argument(
        "--orchestrator-receipt-binding", default="",
        help="resolved immutable W&B/Lepton job-spec binding required for "
             "strict C9/MR-ME Scout, SFT, and Central training",
    )
    p.add_argument(
        "--orchestrator-command-sha256", default="",
        help="controller-preregistered SHA256 of the exact Lepton job command",
    )
    p.add_argument(
        "--orchestrator-receipt-artifact", default="",
        help="RUN_NAME-unique receipt polling alias captured by the job spec",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument(
        "--explorers-per-round", "--scouts-per-round", dest="scouts_per_round", type=int, default=1,
        help="true multi-explorer fan-out K. Under the explicit stabilized "
             "C9 profile, R=1 is integrated full-data MultiScout; R>1 trains "
             "K independent full-budget Scouts on the same exact 1/R shard "
             "per round, then pools their raw trajectories once",
    )
    p.add_argument("--explorer-steps", "--scout-steps", dest="scout_steps", type=int, default=200)
    p.add_argument("--main-steps", "--central-steps", dest="central_steps", type=int, default=100)
    p.add_argument(
        "--explorer-lr", "--scout-lr", dest="scout_lr", type=float, default=None,
        help="explicit Scout GRPO learning rate; default None resolves to the "
             "frozen C8 5e-6 inside grpo_gpu",
    )
    p.add_argument(
        "--main-lr", "--central-lr", dest="central_lr", type=float, default=None,
        help="explicit Central GRPO learning rate; default None resolves to 1e-6",
    )
    p.add_argument("--lambda-novelty", dest="lambda_novelty", type=float, default=None,
                   help=f"scout lambda (default {DEFAULT_LAMBDA_NOVELTY}); when "
                        "passed explicitly it overrides --lambda-schedule "
                        "(multi-round: constant lambda; error if an explicit "
                        "schedule disagrees)")
    p.add_argument("--lambda-schedule", default=None,
                   help="per-round lambda for --rounds > 1 (default "
                        f"{DEFAULT_LAMBDA_SCHEDULE}); ignored for single-round "
                        "runs and whenever --lambda-novelty is passed explicitly")
    p.add_argument("--novelty-layers", default="7,14,21",
                   help="RND feature layers (0-indexed policy blocks); the "
                        "contract 7/14/21 fits Qwen3-1.7B's 28 blocks")
    p.add_argument(
        "--expdis-stabilization-mode", default=grpo_gpu.C8_RND_LIFECYCLE,
        choices=list(grpo_gpu.EXPDIS_STABILIZATION_MODES),
        help="opt-in RND lifecycle; C9 requires the exact scientific label and "
             "immutable code manifest",
    )
    p.add_argument(
        "--novelty-normalization", default="raw",
        choices=list(grpo_gpu.NOVELTY_NORMALIZATION_MODES),
        help="raw (frozen C8) or C9 selected-batch population z-score",
    )
    p.add_argument(
        "--c9-execution-profile", default="",
        choices=list(grpo_gpu.C9_EXECUTION_PROFILES),
        help="preregistered stabilization profile (canary or full)",
    )
    p.add_argument(
        "--c10-execution-profile", default="",
        choices=list(grpo_gpu.C10_EXECUTION_PROFILES),
        help="separate survivor-consistent Overlong Filtering profile; "
             "mutually exclusive with C9 and higher-order profiles",
    )
    p.add_argument(
        "--c10-exclusive-lease-binding", default="",
        help="resolved active W&B lease required by the one-draw C10 canary",
    )
    p.add_argument(
        "--c9-multiexplorer-profile", default="",
        choices=("", mrme_contract.C9_MULTIEXPLORER_PROFILE),
        help="distinct stabilized C9 K-Scout profile. R=1 is integrated "
             "MultiScout over the full dataset; R>1 is the explicitly "
             "disjoint-shard annealed MR-ME method",
    )
    p.add_argument(
        "--single-scout-multiround-profile", default="",
        choices=("", mrme_contract.SINGLESCOUT_MULTIROUND_PROFILE),
        help="explicit signed K=1, R=1..4 campaign profile. Uses the same "
             "C9 Scout/QualityPool/SFT/Central mechanics and deterministic "
             "1/R sharding, but is neither MultiScout nor MR-ME",
    )
    p.add_argument(
        "--stabilization-cell", default="",
        choices=list(grpo_gpu.C9_STABILIZATION_CELLS),
        help="preregistered stabilization cell C1--C5",
    )
    p.add_argument("--policy", default=filter_pool_mod.C8_POLICY,
                   choices=["quality_pool", "naive_pool", "best_scout",
                            filter_pool_mod.C8_POLICY,
                            filter_pool_mod.ORIGINAL_BLENDED_POLICY])
    p.add_argument(
        "--c8-source-balance", action="store_true",
        help="for coverage_pool_c8 MultiScout only: round-robin the already "
             "selected per-problem winners across source files",
    )
    p.add_argument("--min-accepted", type=int, default=500)
    p.add_argument("--max-accepted", type=int, default=500,
                   help="accepted cap for Central SFT; contract 500 (TPU "
                        "max_accepted_trajectories). Raising it (e.g. 2000) is "
                        "a GPU-MAX change and flips the logged config_tier")
    p.add_argument("--explorer-only", "--scout-only", dest="scout_only", action="store_true",
                   help="single-model RL baseline: stop after scout GRPO "
                        "(no harvest/SFT/central); scout ckpt = final model")
    p.add_argument("--skip-explorer", "--skip-scout", dest="skip_scout", action="store_true",
                   help="round 1 only: pool --trajectories instead of training a scout")
    p.add_argument("--trajectories", nargs="*", default=None,
                   help="pooled multi-scout explorer_trajectories.jsonl paths")
    p.add_argument(
        "--harvest-max-passes", type=int, default=0,
        help="post-Scout top-up passes. Canonical c8/TPU-reference runs use 0 "
             "so each 200-step Scout contributes exactly 12,800 trajectories; "
             "positive values are labeled data-budget deviations",
    )
    p.add_argument("--harvest-prompts-per-pass", type=int, default=16)
    p.add_argument("--prompts-per-step", type=int, default=4,
                   help="contract 4; GPU-MAX 8/16")
    p.add_argument("--num-generations", type=int, default=16)
    p.add_argument("--completion-budget", type=int, default=0,
                   help="GPU-MAX: completion budget N for all stages; 0 = contract 16384")
    p.add_argument("--ppo-mode", action="store_true",
                   help="GPU-MAX: real PPO clipping in GRPO stages")
    p.add_argument("--updates-per-rollout", type=int, default=1,
                   help="GPU-MAX (with --ppo-mode): PPO epochs per rollout batch")
    p.add_argument(
        "--mask-truncated", dest="mask_truncated", action="store_true",
        default=False,
        help="optional truncated-completion loss masking. The canonical c8 "
             "replication keeps this off to match the measured TPU reference; "
             "enable only for a labeled Overlong-Filtering ablation",
    )
    p.add_argument("--no-mask-truncated", dest="mask_truncated", action="store_false")
    p.add_argument(
        "--truncation-baseline-mode", default="all_rows",
        choices=list(grpo_gpu.TRUNCATION_BASELINE_MODES),
        help="group-mean and dynamic-eligibility population for masked rows; "
             "unmasked_survivors is C10-only",
    )
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "adafactor"],
                   help="GRPO stage optimizer (contract: adamw)")
    p.add_argument("--sft-optimizer", default="adamw", choices=["adamw", "adafactor"],
                   help="SFT stage optimizer (contract: adamw; TPU used adafactor "
                        "as a memory compromise)")
    p.add_argument("--reload-every", type=int, default=10)
    p.add_argument(
        "--rl-validity-mode", default="c8",
        choices=list(grpo_gpu.RL_VALIDITY_MODES),
        help="c8 enforces current-policy rollouts, novelty-only homogeneous "
             "group eligibility, and optimizer-complete resume; legacy "
             "reproduces c7 behavior",
    )
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument(
        "--keep-checkpoints", type=int, default=1,
        help="local step checkpoints retained per GRPO phase. c8 keeps one; "
             "durable W&B checkpoint versions remain independently available",
    )
    p.add_argument("--ckpt-artifact-every", type=int, default=100,
                   help="stage checkpoint/final-model W&B push cadence "
                        "(steps); 0 = never (see grpo_gpu)")
    p.add_argument("--probe-every", type=int, default=25,
                   help="mid-training AIME24 probe cadence for GRPO stages "
                        "(steps); 0 = off (see grpo_gpu)")
    p.add_argument("--probe-dataset", default="aime24",
                   help='"aime24" or a local {"problem","answer"} JSONL')
    p.add_argument("--max-consecutive-gen-failures", type=int, default=3,
                   help="abort after N consecutive failing generate() calls")
    p.add_argument(
        "--dynamic-max-attempts", type=int, default=0,
        help="dynamic-sampling retry ceiling per learner step; 0 resolves to "
             "64 in fail-closed c8 mode and the historical 8 in legacy mode",
    )
    p.add_argument("--resume", action="store_true",
                   help="resume from pipeline_state.json: skip completed "
                        "stages/rounds, restart an interrupted GRPO stage from "
                        "its newest checkpoint (stale JSONL rows are trimmed)")
    p.add_argument("--sft-epochs", type=int, default=2)
    p.add_argument("--sft-lr", type=float, default=5e-6)
    p.add_argument(
        "--sft-grad-accum", type=int, default=1,
        help="SFT gradient accumulation; c8 contract 1 matches the TPU "
             "per-example optimizer stepping (c7 used the labeled deviation 8)",
    )
    p.add_argument(
        "--sft-termination-mode", default="append_eos",
        choices=list(sft_gpu.SFT_TERMINATION_MODES),
        help="c8 default append_eos supervises one model EOS/chat-stop; "
             "legacy explicitly reproduces c7 SFT tokenization",
    )
    p.add_argument(
        "--sft-save-steps", type=sft_gpu.parse_sft_save_steps, default=(),
        metavar="STEPS",
        help="optional comma-separated post-update SFT optimizer steps to "
             "save as full Hugging Face checkpoints; instrumentation only",
    )
    p.add_argument(
        "--stop-after-sft", action="store_true",
        help="produce an SFT-only final model and stop before Central GRPO so "
             "termination/validity can be evaluated as a required gate",
    )
    p.add_argument(
        "--sft-model", default=None,
        help="local pre-evaluated SFT Hugging Face checkpoint to use as the "
             "Central init. Requires --rounds 1 and --skip-scout with the "
             "original trajectories; Stage 3 copies the exact frozen accepted "
             "library without c8 re-filtering, and Stage 4 "
             "records the health-gate selection and skips SFT training",
    )
    p.add_argument(
        "--sft-evidence-index", default="",
        help="local JSON index binding the selected model, frozen accepted "
             "library, all four raw gate bundles, and selection artifact to "
             "clean digest-scoped W&B downloads",
    )
    p.add_argument(
        "--frozen-sft-library-evidence-index", default="",
        help="training-only, noncanonical c7-reuse ga8 route: exact W&B "
             "results:v1 accepted-library evidence. Copies accepted.jsonl "
             "byte-for-byte, trains SFT, and requires --stop-after-sft.",
    )
    p.add_argument(
        "--fresh-c8-sft-gate", action="store_true",
        help="preregister and train fresh-c8 SFT candidate snapshots at "
             "updates 125/250/500/1000, mark them canonical-pending-gate, "
             "then stop before Central for the independent behavior gates",
    )
    p.add_argument(
        "--fresh-c8-multiscout-component", action="store_true",
        help="produce a provenance-bound Scout-only MultiScout component at "
             "preregistered seed 1 or 2; this profile is disjoint from the "
             "canonical seed-0 SingleScout SFT gate",
    )
    p.add_argument(
        "--code-source-manifest", default="",
        help="exact digest-scoped code-artifact download manifest; mandatory "
             "for either fresh-c8 Scout evidence profile",
    )
    p.add_argument(
        "--sft-training-evidence-manifest", default="", help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--trajectory-source-manifest", default="",
        help="JSON binding --trajectories to exact W&B artifact refs/digests "
             "and file SHA256 values",
    )
    p.add_argument("--dataset", default="dapo_math_17k")
    p.add_argument("--max-train-examples", type=int, default=20000)
    p.add_argument("--sync-dir", default="/workspace")
    p.add_argument("--vllm-gpus", default="", help="comma GPU ids for the managed vLLM pool")
    p.add_argument("--endpoints", default="", help="attach to running servers (no reload!)")
    p.add_argument("--served-model-name", default=None,
                   help="constant across reloads (default: --model)")
    p.add_argument("--vllm-port-base", type=int, default=8000)
    p.add_argument("--vllm-max-model-len", type=int, default=18432)
    p.add_argument("--vllm-gpu-mem-util", type=float, default=0.90)
    p.add_argument("--vllm-max-num-seqs", type=int, default=32)
    p.add_argument("--gen-concurrency", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--param-dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    validate_args(args)
    if bool(
            str(getattr(args, "c9_execution_profile", "") or "")
            or str(getattr(args, "c9_multiexplorer_profile", "") or "")
            or str(getattr(
                args, "single_scout_multiround_profile", "") or "")
            or str(getattr(args, "expdis_stabilization_mode", "") or "")
            == grpo_gpu.C9_RND_LIFECYCLE):
        args._training_execution_environment = \
            grpo_gpu.validate_canonical_training_execution_environment(
                args.container_image_identity,
                args._orchestrator_receipt_binding,
            )
    os.makedirs(args.output_dir, exist_ok=True)
    served = args.served_model_name or args.model
    lambdas = parse_lambda_schedule(args)
    mrme = _true_mrme(args)
    mrme_campaign: Optional[Dict[str, Any]] = None
    if mrme:
        mrme_campaign = _build_mrme_contract(args, lambdas)
        # The marker causes every Scout GRPO invocation to consume the same
        # immutable code-source manifest recorded in the campaign contract.
        args._mrme_contract = mrme_campaign
    config_tier = grpo_gpu.compute_config_tier(args, PIPELINE_GPU_MAX_FLAG_DEFAULTS)
    if args.scientific_variant:
        config_tier = f"{config_tier}+scientific-variant:{args.scientific_variant}"
    if int(args.completion_budget) > 0:
        # Shared pool must fit prompt (2048) + the raised completion budget.
        args.vllm_max_model_len = max(int(args.vllm_max_model_len),
                                      2048 + int(args.completion_budget))
    if int(args.probe_every) > 0:
        # Frozen probe protocol: 2048-prompt + PROBE_MAX_TOKENS requests every
        # --probe-every steps. An undersized pool 400s every probe request and
        # the probe silently degrades to error rows.
        probe_required = 2048 + grpo_gpu.PROBE_MAX_TOKENS
        if args.endpoints:
            if int(args.vllm_max_model_len) < probe_required:
                raise ValueError(
                    f"--probe-every {args.probe_every} needs serving "
                    f"max_model_len >= {probe_required} (2048 prompt + "
                    f"PROBE_MAX_TOKENS {grpo_gpu.PROBE_MAX_TOKENS}) but the "
                    f"attach-mode pool declares --vllm-max-model-len "
                    f"{args.vllm_max_model_len}; relaunch the servers with a "
                    "larger --max-model-len (expdis_job.sh MAX_MODEL_LEN) and pass "
                    "the matching --vllm-max-model-len, or set --probe-every 0")
        elif int(args.vllm_max_model_len) < probe_required:
            print(f"[pipeline] raising --vllm-max-model-len "
                  f"{args.vllm_max_model_len} -> {probe_required} "
                  "(frozen probe protocol)", flush=True)
            args.vllm_max_model_len = probe_required
    print(f"[pipeline] rounds={args.rounds} scouts_per_round={args.scouts_per_round} "
          f"lambdas={lambdas} policy={args.policy} "
          f"scout_steps={args.scout_steps} central_steps={args.central_steps} "
          f"config_tier={config_tier} max_model_len={args.vllm_max_model_len}", flush=True)

    # Bind resumable C9 state before tokenizer/model/server startup.  A stale or
    # differently configured output directory must fail before consuming GPUs.
    state = _load_state(args)
    if mrme_campaign is not None:
        state = _prepare_mrme_state(args, state, mrme_campaign)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True,
        **common.remote_revision_kwargs(args.model, args.model_revision))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.endpoints:
        pool = VLLMPool.attach(args.endpoints.split(","), served)
        print("[pipeline] WARNING: attach mode — weight reloads are disabled; "
              "training will run against stale rollouts", flush=True)
    elif args.vllm_gpus:
        pool = VLLMPool(
            model_dir=args.model,
            model_revision=(
                str(args.model_revision or "")
                if not common.is_local_model_reference(args.model) else ""),
            gpu_ids=[int(x) for x in args.vllm_gpus.split(",")],
            served_model_name=served,
            port_base=int(args.vllm_port_base),
            max_model_len=int(args.vllm_max_model_len),
            gpu_memory_utilization=float(args.vllm_gpu_mem_util),
            max_num_seqs=int(args.vllm_max_num_seqs),
            log_dir=os.path.join(args.output_dir, "vllm_logs"),
        )
        try:
            pool.launch()
            pool.wait_healthy()
        except Exception:
            # A failed launch/health-wait must not orphan the servers that
            # DID start (they would hold GPU memory and trip the stale-port
            # guard on every retry).
            pool.shutdown()
            raise
    else:
        raise ValueError("provide --vllm-gpus (managed pool) or --endpoints")
    pool.max_consecutive_failures = int(args.max_consecutive_gen_failures)

    wandb_run = None
    if args.wandb:
        os.environ.setdefault("WANDB_RUN_GROUP", args.run_name)
        wandb_run = common.init_stage_wandb(
            name=f"{args.run_name}-pipeline",
            group=os.environ.get("WANDB_RUN_GROUP", args.run_name),
            config={**vars(args), "config_tier": config_tier,
                    "lambdas": lambdas, "backend": "gpu-torch"},
        )

    metrics_logger = common.MetricsLogger(os.path.join(args.output_dir, "metrics.jsonl"))
    t0 = time.time()
    summaries: List[Dict[str, Any]] = []
    scout_init = args.model
    try:
        for round_idx in range(1, int(args.rounds) + 1):
            lam = lambdas[round_idx - 1]
            print(f"[pipeline] === round {round_idx}/{args.rounds} lambda={lam} "
                  f"scout_init={scout_init} ===", flush=True)
            if mrme_campaign is not None:
                shard_path = os.path.join(
                    args.output_dir, f"round_{round_idx:02d}",
                    "dataset_shard_manifest.json")
                shard_env = mrme_contract.shard_environment(
                    mrme_campaign, round_idx, shard_path)
                with _temporary_environment(shard_env):
                    # This first load atomically creates/verifies the exact
                    # membership manifest. Every K Scout loader and the
                    # downstream round run under the same fail-closed env.
                    examples = grpo_gpu.load_training_examples(
                        args.dataset, tokenizer, int(args.max_train_examples))
                    shard_manifest = _load_round_shard_manifest(shard_path)
                    summary = run_mrme_round(
                        args, mrme_campaign, pool, tokenizer, examples,
                        round_idx, scout_init, shard_manifest, wandb_run,
                        metrics_logger, state)
            else:
                with _round_shard_env(round_idx, int(args.rounds), int(args.seed)):
                    # Per-round harvest pool: re-loaded under the round's shard
                    # env so multi-round harvests also draw from disjoint shards.
                    examples = grpo_gpu.load_training_examples(
                        args.dataset, tokenizer, int(args.max_train_examples))
                    summary = run_round(
                        args, pool, tokenizer, examples, round_idx,
                        lam, scout_init, wandb_run, metrics_logger, state)
            summaries.append(summary)
            # Handoff: next round's scout starts from this round's central.
            scout_init = summary["central_final"]
            common.atomic_write_json(
                os.path.join(args.output_dir, "run_summary.json"),
                {"rounds": summaries, "final_model": scout_init,
                 "config_tier": config_tier,
                 **(_round_campaign_summary_identity(mrme_campaign)
                    if mrme_campaign is not None else {}),
                 "mrme_contract_sha256": (
                     mrme_campaign["contract_sha256"]
                     if mrme_campaign is not None else None),
                 "elapsed_s": time.time() - t0, "args": vars(args)})
            _push_results_artifact(wandb_run, args, f"round-{round_idx}-complete")
        if mrme_campaign is not None:
            manifests = [
                _load_round_shard_manifest(os.path.join(
                    args.output_dir, f"round_{round_idx:02d}",
                    "dataset_shard_manifest.json"))
                for round_idx in range(1, int(args.rounds) + 1)
            ]
            partition = mrme_contract.assert_round_manifests_disjoint(
                manifests, expected_rounds=int(args.rounds),
                expected_method=str(mrme_campaign["method"]))
            partition_path = os.path.join(
                args.output_dir, "mrme_shard_partition.json")
            _write_or_verify_json(
                partition_path, partition, label="MR-ME shard partition")
            common.atomic_write_json(
                os.path.join(args.output_dir, "run_summary.json"),
                {"rounds": summaries, "final_model": scout_init,
                 "config_tier": config_tier,
                 **_round_campaign_summary_identity(mrme_campaign),
                 "mrme_contract_sha256": mrme_campaign["contract_sha256"],
                 "mrme_shard_partition": partition,
                 "elapsed_s": time.time() - t0, "args": vars(args)})
            _push_results_artifact(wandb_run, args, "mrme-shard-partition")
    finally:
        pool.shutdown()
        if wandb_run is not None:
            wandb_run.log({"pipeline/complete": float(len(summaries) == int(args.rounds))})
            wandb_run.finish()
    print(f"[pipeline] DONE {len(summaries)}/{args.rounds} rounds; "
          f"final model = {scout_init}", flush=True)


if __name__ == "__main__":
    main()
