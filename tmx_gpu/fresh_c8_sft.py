#!/usr/bin/env python3
"""Fail-closed contracts for the fresh-c8 SFT gate and continuation.

This module intentionally does not generalize the frozen c7-reuse GA8
diagnostic.  It defines a second, stricter evidence profile for the canonical
fresh-c8 SingleScout path: a 200-step Scout produces exactly 12,800
training-time trajectories, ``coverage_pool_c8`` selects exactly 500 unique
problems, and Base-initialized EOS-aware SFT performs exactly 1,000 AdamW
updates.  Only the preregistered post-update snapshots may be behavior gated.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from tmx_gpu import common, sft_gpu

PROFILE = "fresh_c8_single_scout"
MULTISCOUT_COMPONENT_PROFILE = "fresh_c8_multiscout_component"
MULTISCOUT_COMPONENT_SEEDS = (1, 2)
PREREGISTERED_STEPS = (125, 250, 500, 1000)
MODEL_INIT = "Qwen/Qwen3-1.7B"
SCOUT_COMPLETION_EVIDENCE_VERSION = "fresh_c8_scout_completion_evidence_v2"
MULTISCOUT_COMPONENT_EVIDENCE_VERSION = (
    "fresh_c8_multiscout_component_completion_evidence_v2"
)
TRAJECTORY_SEMANTIC_DIGEST_VERSION = "fresh_c8_trajectory_rows_v1"
TRAINING_EVIDENCE_VERSION = "fresh_c8_sft_training_evidence_v2"
COMPLETED_RUN_EVIDENCE_VERSION = "sft_completed_run_evidence_v1"
EXTERNAL_EVIDENCE_INDEX_VERSION = "fresh_c8_sft_evidence_bundle_v1"
LINEAGE_CONTRACT_VERSION = "fresh_c8_sft_central_lineage_v1"
EARLY_SELECTION_VARIANT = "fresh_c8_sft_exposure_gated"

SFT_RUNTIME_CONTRACT = {
    "version": sft_gpu.SFT_RUNTIME_CONTRACT_VERSION,
    "config_tier": "contract",
    "model_init": MODEL_INIT,
    "base_model_revision": common.CANONICAL_TRAINING_MODEL_REVISION,
    "optimizer": "adamw",
    "learning_rate": 5e-6,
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
    "adam_eps": 1e-8,
    "weight_decay": 0.0,
    "grad_clip": 1.0,
    "epochs": 2,
    "batch_size": 1,
    "grad_accum": 1,
    "completion_budget_flag": 0,
    "resolved_max_total_len": 18432,
    "termination_mode": "append_eos",
    "seed": 0,
    "shuffle_order": "numpy_default_rng_seed_plus_epoch",
    "parameter_dtype": "float32",
    "chunk_tokens": 256,
    "loss": "completion_only_token_mean_cross_entropy",
    "prompt_truncation": "left",
    "completion_truncation": "right_reserve_exactly_one_eos",
}

CHECKPOINT_CONTRACT = {
    "model_init": MODEL_INIT,
    "optimizer": "adamw",
    "learning_rate": 5e-6,
    "epochs_planned": 2,
    "grad_accum": 1,
    "accepted_examples": 500,
    "trainable_examples": 500,
    "planned_optimizer_steps": 1000,
    "optimizer_steps_executed": 1000,
    "micro_steps_executed": 1000,
    "training_completed": True,
    "termination_mode": "append_eos",
    "sft_contract_version": "c8-eos-aware",
    "requested_save_optimizer_steps": list(PREREGISTERED_STEPS),
    "saved_optimizer_steps": list(PREREGISTERED_STEPS),
    "sft_runtime_contract": SFT_RUNTIME_CONTRACT,
}


def signed_payload(version: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-stable evidence object with a self-authenticating SHA."""
    result = {"version": version, **dict(payload)}
    result["evidence_sha256"] = common.canonical_json_sha256(result)
    return result


def verify_signed_payload(
    value: Any, *, version: str, field: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    if value.get("version") != version:
        raise ValueError(f"{field} version mismatch")
    recorded = value.get("evidence_sha256")
    unsigned = dict(value)
    unsigned.pop("evidence_sha256", None)
    if recorded != common.canonical_json_sha256(unsigned):
        raise ValueError(f"{field} SHA256 is missing or invalid")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def trajectory_semantic_digest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Hash parsed trajectory objects while preserving their row order.

    The byte SHA catches any file mutation; this second digest distinguishes a
    formatting-only rewrite from a semantic row change and is recorded at the
    instant the Scout completes rather than inferred later at SFT time.
    """
    _require(bool(rows), "fresh-c8 trajectory rows are empty")
    envelope = {
        "version": TRAJECTORY_SEMANTIC_DIGEST_VERSION,
        "rows": rows,
    }
    return {
        "version": TRAJECTORY_SEMANTIC_DIGEST_VERSION,
        "row_count": len(rows),
        "sha256": common.canonical_json_sha256(envelope),
    }


def _validate_exact_scout_runtime(
    runtime: Any, *, expected_seed: int = 0,
) -> None:
    from tmx_gpu import grpo_gpu
    from tmx_jax.data import training_dataset_identity
    from tmx_jax.prompting import prompt_metadata

    _require(type(expected_seed) is int,
             "fresh-c8 Scout expected seed must be an exact integer")
    _require(isinstance(runtime, dict), "fresh-c8 Scout runtime contract missing")
    exact = {
        "version": grpo_gpu.GRPO_RUNTIME_CONTRACT_VERSION,
        "phase": "scout",
        "scientific_variant": "",
        "config_tier": "contract",
        "model_init": MODEL_INIT,
        "base_model_revision": common.CANONICAL_TRAINING_MODEL_REVISION,
        "expected_steps": 200,
        "resolved_learning_rate": 5e-6,
        "optimizer": "adamw",
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "adam_eps": 1e-8,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "parameter_dtype": "float32",
        "rl_validity_mode": "c8",
        "completion_budget_flag": 0,
        "resolved_max_prompt_len": 2048,
        "resolved_max_completion_len": 16384,
        "resolved_vllm_max_model_len": 36864,
        "reward_scoring_text_scope": grpo_gpu.REWARD_SCORING_TEXT_SCOPE,
        "trajectory_record_schema_version": (
            grpo_gpu.TRAJECTORY_RECORD_SCHEMA_VERSION),
        "soft_overlong_expected_len": 13107,
        "soft_overlong_cache_len": 3277,
        "loss_denominator_len": 16384,
        "loss_normalization": "dr_grpo_fixed_denominator",
        "advantage_standardization": False,
        "kl_beta": 0.0,
        "prompts_per_step": 4,
        "num_generations": 16,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "training_prompt": prompt_metadata(),
        "dynamic_sampling": True,
        "dynamic_max_attempts": 64,
        "mask_truncated": False,
        "clip_epsilon_low": 0.2,
        "clip_epsilon_high": 0.28,
        "ppo_mode": False,
        "updates_per_rollout": 1,
        "incorrect_novelty_scale": 0.0,
        "lambda_novelty": 0.5,
        "dataset": "dapo_math_17k",
        "training_dataset_identity": training_dataset_identity(
            "dapo_math_17k"),
        "max_train_examples": 20000,
        "seed": expected_seed,
        "round": 1,
        "max_consecutive_generation_failures": 3,
        "generation_timeout_seconds": 3600.0,
        "generation_concurrency_per_server": 8,
        "chunk_tokens": 256,
        "scout_novelty_contract": {
            "novelty_layers": "7,14,21",
            "rnd_learning_rate": 1e-4,
            "rnd_hidden_dim": 512,
            "novelty_feature_max_length": 16384,
            "incorrect_novelty_scale": 0.0,
        },
    }
    actual_keys = set(runtime)
    expected_keys = set(exact)
    _require(actual_keys == expected_keys,
             "fresh-c8 Scout runtime key set mismatch "
             f"(missing={sorted(expected_keys - actual_keys)}, "
             f"extra={sorted(actual_keys - expected_keys)})")
    for key, expected in exact.items():
        actual = runtime.get(key)
        if type(expected) is bool:
            ok = type(actual) is bool and actual is expected
        elif type(expected) is int:
            ok = type(actual) is int and actual == expected
        elif isinstance(expected, float):
            ok = type(actual) in (int, float) and math.isclose(
                float(actual), expected, rel_tol=0.0, abs_tol=1e-15)
        else:
            ok = actual == expected
        _require(ok, f"fresh-c8 Scout runtime {key}={actual!r}, expected {expected!r}")


def validate_scout_runtime_contract(value: Any) -> dict[str, Any]:
    """Validate the exact canonical seed-0 Scout runtime before update 1."""
    _validate_exact_scout_runtime(value, expected_seed=0)
    return dict(value)


def validate_multiscout_component_runtime_contract(
    value: Any, *, expected_seed: int,
) -> dict[str, Any]:
    """Validate a seed-1/2 component's complete resolved GRPO contract."""
    _require(type(expected_seed) is int
             and expected_seed in MULTISCOUT_COMPONENT_SEEDS,
             "fresh-c8 MultiScout component seed must be exactly 1 or 2")
    _validate_exact_scout_runtime(value, expected_seed=expected_seed)
    return dict(value)


def reject_multiscout_component_from_generic_pool(
    *, artifact_metadata: Any, completion_evidence: Any,
) -> None:
    """Fail closed until a signed, strict MultiScout consumer exists."""
    metadata = artifact_metadata if isinstance(artifact_metadata, Mapping) else {}
    _require(metadata.get("fresh_c8_evidence_profile")
             != MULTISCOUT_COMPONENT_PROFILE,
             "fresh-c8 MultiScout component artifacts require the strict "
             "pooled consumer, which is not implemented")
    _require("component_seed" not in metadata,
             "reserved component_seed metadata cannot enter generic "
             "trajectory pooling without the strict pooled consumer")
    evidence_values = (
        completion_evidence if isinstance(completion_evidence, list) else [])
    for evidence in evidence_values:
        if not isinstance(evidence, Mapping):
            continue
        is_component = (
            evidence.get("version") == MULTISCOUT_COMPONENT_EVIDENCE_VERSION
            or evidence.get("profile") == MULTISCOUT_COMPONENT_PROFILE
            or "component_seed" in evidence
        )
        _require(not is_component,
                 "fresh-c8 MultiScout component evidence cannot enter the "
                 "generic trajectory pooling route")


def _validate_scout_completion_evidence(
    value: Any, *, version: str, profile: str, expected_seed: int,
    field: str,
) -> dict[str, Any]:
    """Validate one explicitly named Scout-completion evidence profile."""
    _require(type(expected_seed) is int,
             f"{field} expected seed must be an exact integer")
    evidence = verify_signed_payload(
        value, version=version, field=field)
    _require(evidence.get("profile") == profile,
             f"{field} profile mismatch")
    code = evidence.get("code_artifact")
    _require(isinstance(code, dict)
             and code.get("version") == common.ARTIFACT_DOWNLOAD_MANIFEST_VERSION
             and isinstance(code.get("source"), dict)
             and code["source"].get("artifact_ref")
             and code["source"].get("artifact_digest")
             and code.get("file_manifest_sha256"),
             "fresh-c8 Scout completion lacks exact code artifact evidence")
    common.require_immutable_wandb_artifact_ref(
        code["source"]["artifact_ref"], field="fresh-c8 CODE_ARTIFACT")
    code_normalized = {
        "version": code["version"],
        "source": code["source"],
        "files": code.get("files"),
    }
    _require(isinstance(code_normalized["files"], list)
             and bool(code_normalized["files"])
             and code["file_manifest_sha256"]
             == common.canonical_json_sha256(code_normalized),
             "fresh-c8 Scout completion code file-manifest SHA mismatch")
    trajectory = evidence.get("trajectory")
    _require(isinstance(trajectory, dict),
             "fresh-c8 Scout completion trajectory evidence missing")
    _require(trajectory.get("row_count") == 12800
             and trajectory.get("steps") == 200
             and trajectory.get("rows_per_step") == 64,
             "fresh-c8 Scout completion must prove 12,800 rows at 64 x 200")
    _require(isinstance(trajectory.get("file_sha256"), str)
             and len(trajectory["file_sha256"]) == 64,
             "fresh-c8 Scout completion trajectory byte SHA missing")
    semantic = trajectory.get("semantic_digest")
    _require(isinstance(semantic, dict)
             and semantic.get("version") == TRAJECTORY_SEMANTIC_DIGEST_VERSION
             and semantic.get("row_count") == 12800
             and isinstance(semantic.get("sha256"), str)
             and len(semantic["sha256"]) == 64,
             "fresh-c8 Scout completion trajectory semantic SHA missing")
    scout = evidence.get("scout")
    _require(isinstance(scout, dict)
             and scout.get("phase") == "scout"
             and scout.get("checkpoint_kind") == "deployable_model"
             and scout.get("step") == 200
             and scout.get("model") == MODEL_INIT
             and scout.get("config_tier") == "contract"
             and scout.get("rl_validity_mode") == "c8"
             and scout.get("lambda_novelty") == 0.5
             and isinstance(scout.get("runtime_contract"), dict)
             and isinstance(scout.get("checkpoint_file_manifest"), dict),
             "fresh-c8 Scout completion model/runtime evidence mismatch")
    if profile == PROFILE:
        validate_scout_runtime_contract(scout["runtime_contract"])
    else:
        validate_multiscout_component_runtime_contract(
            scout["runtime_contract"], expected_seed=expected_seed)
    return evidence


def validate_scout_completion_evidence(value: Any) -> dict[str, Any]:
    """Validate the frozen, canonical seed-0 SingleScout completion."""
    return _validate_scout_completion_evidence(
        value,
        version=SCOUT_COMPLETION_EVIDENCE_VERSION,
        profile=PROFILE,
        expected_seed=0,
        field="fresh-c8 Scout completion evidence",
    )


def validate_multiscout_component_completion_evidence(
    value: Any,
) -> dict[str, Any]:
    """Validate a Scout-only seed-1/2 MultiScout component completion.

    This is intentionally a disjoint public validator.  Component evidence is
    never accepted as canonical SingleScout SFT or Central-continuation
    evidence.
    """
    if not isinstance(value, dict):
        raise ValueError("fresh-c8 MultiScout component evidence must be an object")
    component_seed = value.get("component_seed")
    _require(type(component_seed) is int
             and component_seed in MULTISCOUT_COMPONENT_SEEDS,
             "fresh-c8 MultiScout component seed must be exactly 1 or 2")
    evidence = _validate_scout_completion_evidence(
        value,
        version=MULTISCOUT_COMPONENT_EVIDENCE_VERSION,
        profile=MULTISCOUT_COMPONENT_PROFILE,
        expected_seed=component_seed,
        field="fresh-c8 MultiScout component completion evidence",
    )
    runtime_seed = evidence["scout"]["runtime_contract"].get("seed")
    _require(runtime_seed == component_seed,
             "fresh-c8 MultiScout component seed/runtime mismatch")
    return evidence


def validate_training_evidence(value: Any) -> dict[str, Any]:
    """Validate the durable, pre-SFT fresh-c8 Scout/filter evidence."""
    evidence = verify_signed_payload(
        value, version=TRAINING_EVIDENCE_VERSION,
        field="fresh-c8 SFT training evidence")
    _require(evidence.get("profile") == PROFILE,
             "fresh-c8 training evidence profile mismatch")
    completion = validate_scout_completion_evidence(
        evidence.get("scout_completion_evidence"))
    code = evidence.get("code_source")
    _require(isinstance(code, dict) and code.get("artifact_ref")
             and code.get("artifact_digest") and code.get("file_manifest_sha256"),
             "fresh-c8 training evidence lacks exact code artifact identity")
    _require(code == {
        "artifact_ref": completion["code_artifact"]["source"]["artifact_ref"],
        "artifact_digest": completion["code_artifact"]["source"]["artifact_digest"],
        "file_manifest_sha256": completion["code_artifact"][
            "file_manifest_sha256"],
    }, "fresh-c8 training code does not equal Scout-completion code")
    scout = evidence.get("scout")
    _require(isinstance(scout, dict), "fresh-c8 Scout evidence missing")
    exact_scout = {
        "phase": "scout",
        "checkpoint_kind": "deployable_model",
        "step": 200,
        "model": MODEL_INIT,
        "config_tier": "contract",
        "rl_validity_mode": "c8",
        "lambda_novelty": 0.5,
    }
    for key, expected in exact_scout.items():
        actual = scout.get(key)
        if isinstance(expected, float):
            ok = isinstance(actual, (int, float)) and math.isclose(
                float(actual), expected, rel_tol=0.0, abs_tol=1e-15)
        else:
            ok = actual == expected
        _require(ok, f"fresh-c8 Scout {key}={actual!r}, expected {expected!r}")
    runtime = scout.get("runtime_contract")
    _validate_exact_scout_runtime(runtime)
    from tmx_gpu import grpo_gpu

    runtime_exact = {
        "version": grpo_gpu.GRPO_RUNTIME_CONTRACT_VERSION,
        "phase": "scout",
        "scientific_variant": "",
        "config_tier": "contract",
        "model_init": MODEL_INIT,
        "base_model_revision": common.CANONICAL_TRAINING_MODEL_REVISION,
        "expected_steps": 200,
        "resolved_learning_rate": 5e-6,
        "optimizer": "adamw",
        "rl_validity_mode": "c8",
        "completion_budget_flag": 0,
        "resolved_max_completion_len": 16384,
        "reward_scoring_text_scope": grpo_gpu.REWARD_SCORING_TEXT_SCOPE,
        "trajectory_record_schema_version": (
            grpo_gpu.TRAJECTORY_RECORD_SCHEMA_VERSION),
        "loss_denominator_len": 16384,
        "prompts_per_step": 4,
        "num_generations": 16,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "dynamic_sampling": True,
        "dynamic_max_attempts": 64,
        "mask_truncated": False,
        "ppo_mode": False,
        "updates_per_rollout": 1,
        "incorrect_novelty_scale": 0.0,
        "lambda_novelty": 0.5,
        "dataset": "dapo_math_17k",
        "seed": 0,
        "round": 1,
    }
    for key, expected in runtime_exact.items():
        actual = runtime.get(key)
        if isinstance(expected, float):
            ok = isinstance(actual, (int, float)) and math.isclose(
                float(actual), expected, rel_tol=0.0, abs_tol=1e-15)
        else:
            ok = actual == expected
        _require(ok, f"fresh-c8 Scout runtime {key}={actual!r}, expected {expected!r}")
    _require(runtime.get("scout_novelty_contract") == {
        "novelty_layers": "7,14,21",
        "rnd_learning_rate": 1e-4,
        "rnd_hidden_dim": 512,
        "novelty_feature_max_length": 16384,
        "incorrect_novelty_scale": 0.0,
    }, "fresh-c8 Scout novelty/RND contract mismatch")
    model_manifest = scout.get("checkpoint_file_manifest")
    _require(isinstance(model_manifest, dict) and model_manifest.get("sha256"),
             "fresh-c8 Scout model-file manifest missing")

    trajectories = evidence.get("trajectories")
    _require(isinstance(trajectories, dict), "fresh-c8 trajectory evidence missing")
    _require(trajectories.get("row_count") == 12800,
             "fresh-c8 requires exactly 12,800 trajectories")
    _require(trajectories.get("steps") == 200,
             "fresh-c8 trajectories must cover exactly 200 steps")
    _require(trajectories.get("rows_per_step") == 64,
             "fresh-c8 trajectories must contain exactly 64 rows per step")
    _require(isinstance(trajectories.get("file_sha256"), str)
             and len(trajectories["file_sha256"]) == 64,
             "fresh-c8 trajectory file SHA256 missing")
    _require(trajectories == completion["trajectory"],
             "fresh-c8 training trajectories differ from Scout completion")

    accepted = evidence.get("accepted_library")
    _require(isinstance(accepted, dict), "fresh-c8 accepted-library evidence missing")
    _require(accepted.get("policy") == "coverage_pool_c8",
             "fresh-c8 filter policy must be coverage_pool_c8")
    _require(accepted.get("row_count") == 500,
             "fresh-c8 accepted library must contain exactly 500 rows")
    _require(accepted.get("unique_problem_count") == 500,
             "fresh-c8 accepted library must contain 500 unique problems")
    _require(isinstance(accepted.get("file_sha256"), str)
             and len(accepted["file_sha256"]) == 64,
             "fresh-c8 accepted-library file SHA256 missing")
    digest = accepted.get("semantic_digest")
    _require(isinstance(digest, dict)
             and digest.get("version") == common.SFT_DATASET_DIGEST_VERSION
             and digest.get("row_count") == 500
             and isinstance(digest.get("sha256"), str)
             and len(digest["sha256"]) == 64,
             "fresh-c8 accepted-library semantic digest invalid")
    _require(evidence.get("sft_runtime_contract") == SFT_RUNTIME_CONTRACT,
             "fresh-c8 SFT runtime contract mismatch")
    funnel = evidence.get("filter_funnel")
    _require(isinstance(funnel, dict) and funnel.get("raw") == 12800
             and funnel.get("accepted") == 500,
             "fresh-c8 filter funnel must record raw=12,800 and accepted=500")
    return evidence


def build_completed_run_evidence(
    *, optimizer_steps: int, micro_steps: int, saved_steps: list[int],
    training_evidence_sha256: str | None,
) -> dict[str, Any]:
    return signed_payload(COMPLETED_RUN_EVIDENCE_VERSION, {
        "training_completed": True,
        "optimizer_steps_executed": int(optimizer_steps),
        "micro_steps_executed": int(micro_steps),
        "saved_optimizer_steps": list(saved_steps),
        "training_evidence_sha256": training_evidence_sha256,
    })


def validate_completed_run_evidence(
    value: Any, *, expected_training_evidence_sha256: str | None,
) -> dict[str, Any]:
    evidence = verify_signed_payload(
        value, version=COMPLETED_RUN_EVIDENCE_VERSION,
        field="SFT completed-run evidence")
    _require(evidence.get("training_completed") is True,
             "SFT completed-run evidence is not complete")
    _require(evidence.get("optimizer_steps_executed") == 1000,
             "fresh-c8 SFT did not execute exactly 1,000 optimizer updates")
    _require(evidence.get("micro_steps_executed") == 1000,
             "fresh-c8 SFT did not execute exactly 1,000 micro steps")
    _require(evidence.get("saved_optimizer_steps") == list(PREREGISTERED_STEPS),
             "fresh-c8 SFT completed-run snapshot set mismatch")
    _require(evidence.get("training_evidence_sha256")
             == expected_training_evidence_sha256,
             "SFT completed-run evidence points to different training evidence")
    return evidence
