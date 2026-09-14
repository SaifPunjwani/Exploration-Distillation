#!/usr/bin/env python3
"""Fail-closed evidence contracts for the preregistered C9 ExpDis pipeline.

The C9 training implementation deliberately has a different RND lifecycle
from the frozen C8 replication.  This module keeps that scientific variant
from being reduced to a human-readable run label: every handoff is signed,
byte-bound, and validated again by its consumer.

The helpers here are intentionally pure (no model loading and no W&B calls),
so pipeline, SFT, Central GRPO, artifact publication, and tests can all enforce
the same contract without circular imports.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, Optional

from tmx_gpu import common


C9_PROFILE = "full"
C9_CELL = "C5"
C9_SCIENTIFIC_VARIANT = "expdis_c9_selected_batch_rnd"
C9_SCREEN_VARIANT = "expdis_c9_stability_screen"
C9_RND_LIFECYCLE = "c9_selected_batch_update"
C9_NOVELTY_NORMALIZATION = "selected_batch_zscore"

SCOUT_COMPLETION_EVIDENCE_VERSION = "c9_scout_completion_evidence_v2"
BEHAVIOR_GATE_EVIDENCE_VERSION = "c9_behavior_gate_evidence_v2"
SINGLESCOUT_FILTER_REPLAY_VERSION = "c9_single_scout_filter_replay_v1"
SFT_TRAINING_EVIDENCE_VERSION = "c9_sft_training_evidence_v3"
CENTRAL_LINEAGE_VERSION = "c9_sft_central_lineage_v3"
PIPELINE_EXECUTION_CONTRACT_VERSION = "c9_pipeline_execution_contract_v1"
C9_PROBE_PROTOCOL_ID = (
    "qwen3_p1_components_t0.6_p0.95_k20_minp0_32k_avg4_diagnostic_v2")
C9_PROBE_CONTRACT = {
    "cadence_steps": 25,
    "dataset": "aime24",
    "protocol_id": C9_PROBE_PROTOCOL_ID,
    "headline_eligible": False,
    "metric_scale": "fraction",
    "num_samples_per_problem": 4,
    "max_completion_tokens": 32768,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "sample_seed_rule": "problem_idx*100003+sample_idx",
    "seed_base": 0,
    "benchmark_revision": "8d88b2876a82a080e2f172cc9b25d0d9d2cb4792",
    "benchmark_rows_sha256": (
        "33f24dfe5e519b2f90c44fbb57fb8ce4db12f1e0af457fb92b47df027fa90630"),
    "prompt_renderer": "tmx_gpu.eval_gpu.render_prompt:auto",
    "grader": "tmx_gpu.eval_gpu.grade",
}


def require_shared_probe_prompt_matrix(
    histories: Iterable[Mapping[str, Any]], *, context: str,
) -> str:
    """Require every signed probe history to bind one shared prompt matrix.

    Per-phase stability is insufficient for a Scout/Central comparison: an
    entire phase could consistently render a different chat prompt and still
    pass its local replay.  Campaign validators call this helper across every
    phase (and every MR-ME Scout/round) before accepting completion evidence.
    """
    hashes = []
    for index, history in enumerate(histories, 1):
        value = history.get("prompt_matrix_sha256") \
            if isinstance(history, Mapping) else None
        if not isinstance(value, str) or len(value) != 64 or any(
                char not in "0123456789abcdef" for char in value):
            raise ValueError(
                f"{context} probe history {index} lacks a canonical prompt "
                "matrix SHA256")
        hashes.append(value)
    if not hashes:
        raise ValueError(f"{context} has no probe histories")
    if len(set(hashes)) != 1:
        raise ValueError(
            f"{context} Scout/Central probe prompt matrices differ")
    return hashes[0]

SCOUT_STEPS = 200
SCOUT_ROWS_PER_STEP = 64
SCOUT_TRAJECTORY_ROWS = SCOUT_STEPS * SCOUT_ROWS_PER_STEP
ACCEPTED_ROWS = 500
SFT_EPOCHS = 2
SFT_OPTIMIZER_STEPS = ACCEPTED_ROWS * SFT_EPOCHS
CENTRAL_STEPS = 100

GATE_NUM_PROMPTS = 32
GATE_SAMPLES_PER_PROMPT = 2
GATE_SELECTION_SEED = 20260713
GATE_GENERATION_SEED = 20260713
GATE_MAX_COMPLETION_TOKENS = 32768
GATE_TEMPERATURE = 0.6
GATE_TOP_P = 0.95
GATE_TOP_K = 20
GATE_MIN_P = 0.0
GATE_PANEL_ROLE = "fixed_nonbenchmark_dapo_behavior_diagnostic_not_scout_holdout"
GATE_PANEL_HELD_OUT_FROM_SCOUT_TRAINING = False
GATE_THRESHOLDS = {
    "min_termination_rate": 0.80,
    "max_clipped_rate": 0.20,
    "min_valid_boxed_answer_rate": 0.90,
    "max_median_completion_tokens": GATE_MAX_COMPLETION_TOKENS * 0.75,
    "max_repetition_rate": 7 / 64,
    "max_unclosed_think_rate": 0.20,
    "max_generation_error_rate": 0.05,
}


def signed_payload(version: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``payload`` with an exact version and canonical SHA256."""
    if "version" in payload or "evidence_sha256" in payload:
        raise ValueError("signed payload must not pre-populate signature fields")
    value = {"version": str(version), **payload}
    value["evidence_sha256"] = common.canonical_json_sha256(value)
    return value


def validate_signed_payload(value: Any, expected_version: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("C9 evidence must be one JSON object")
    if value.get("version") != expected_version:
        raise ValueError(
            f"C9 evidence version mismatch: {value.get('version')!r} != "
            f"{expected_version!r}")
    unsigned = dict(value)
    recorded = unsigned.pop("evidence_sha256", None)
    if recorded != common.canonical_json_sha256(unsigned):
        raise ValueError("C9 evidence SHA256 is missing or invalid")
    return value


def _require_c9_identity(value: Dict[str, Any]) -> None:
    expected = {
        "c9_execution_profile": C9_PROFILE,
        "stabilization_cell": C9_CELL,
        "scientific_variant": C9_SCIENTIFIC_VARIANT,
    }
    drift = [
        f"{key}={value.get(key)!r} (expected {expected_value!r})"
        for key, expected_value in expected.items()
        if value.get(key) != expected_value
    ]
    if drift:
        raise ValueError("C9 identity mismatch: " + "; ".join(drift))


def validate_scout_completion_evidence(
    value: Any, *, expected_runtime_contract: Optional[Dict[str, Any]] = None,
    expected_code_artifact: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    evidence = validate_signed_payload(
        value, SCOUT_COMPLETION_EVIDENCE_VERSION)
    _require_c9_identity(evidence)
    exact = {
        "seed": 0,
        "round": 1,
        "scout_steps": SCOUT_STEPS,
        "rnd_selected_batch_update_count": SCOUT_STEPS,
    }
    for key, expected in exact.items():
        if evidence.get(key) != expected:
            raise ValueError(
                f"C9 Scout evidence {key}={evidence.get(key)!r}; "
                f"expected {expected!r}")
    trajectory = evidence.get("trajectory")
    if not isinstance(trajectory, dict):
        raise ValueError("C9 Scout evidence lacks trajectory facts")
    if trajectory.get("row_count") != SCOUT_TRAJECTORY_ROWS:
        raise ValueError("C9 Scout evidence must bind exactly 12,800 rows")
    if trajectory.get("rows_per_step") != SCOUT_ROWS_PER_STEP or \
            trajectory.get("step_count") != SCOUT_STEPS:
        raise ValueError("C9 Scout evidence trajectory geometry mismatch")
    if not isinstance(trajectory.get("file_sha256"), str) or \
            len(trajectory["file_sha256"]) != 64:
        raise ValueError("C9 Scout evidence lacks trajectory byte SHA256")
    if not isinstance(trajectory.get("canonical_rows_sha256"), str) or \
            len(trajectory["canonical_rows_sha256"]) != 64:
        raise ValueError("C9 Scout evidence lacks trajectory semantic SHA256")
    probes = evidence.get("probe_history")
    expected_probe_steps = list(range(25, SCOUT_STEPS + 1, 25))
    if not isinstance(probes, dict) or probes.get("probe_protocol_id") != \
            C9_PROBE_PROTOCOL_ID or \
            probes.get("phase") != "scout_grpo" or \
            probes.get("expected_global_steps") != expected_probe_steps or \
            probes.get("row_count") != len(expected_probe_steps) or \
            not isinstance(probes.get("prompt_matrix_sha256"), str) or \
            len(probes["prompt_matrix_sha256"]) != 64 or \
            not isinstance(probes.get("canonical_rows_sha256"), str) or \
            len(probes["canonical_rows_sha256"]) != 64:
        raise ValueError("C9 Scout evidence lacks the exact probe history")
    manifest = evidence.get("model_file_manifest")
    if not isinstance(manifest, dict) or manifest.get("version") != \
            common.DIRECTORY_FILE_MANIFEST_VERSION:
        raise ValueError("C9 Scout evidence lacks a model-file manifest")
    runtime = evidence.get("scout_runtime_contract")
    if not isinstance(runtime, dict):
        raise ValueError("C9 Scout evidence lacks a runtime contract")
    from tmx_gpu import grpo_gpu

    grpo_gpu.require_canonical_training_execution_environment_contract(runtime)
    novelty = runtime.get("scout_novelty_contract", {})
    if runtime.get("phase") != "scout" or \
            runtime.get("expected_steps") != SCOUT_STEPS or \
            not math.isclose(float(runtime.get("lambda_novelty", -1.0)), 0.5) or \
            novelty.get("rnd_lifecycle") != C9_RND_LIFECYCLE or \
            novelty.get("novelty_normalization") != C9_NOVELTY_NORMALIZATION or \
            runtime.get("probe_contract") != C9_PROBE_CONTRACT:
        raise ValueError("C9 Scout runtime contract mismatch")
    if expected_runtime_contract is not None and runtime != expected_runtime_contract:
        raise ValueError("C9 Scout evidence differs from resolved runtime args")
    code = evidence.get("code_artifact")
    if not isinstance(code, dict) or not isinstance(
            code.get("file_manifest_sha256"), str) or len(
                code["file_manifest_sha256"]) != 64:
        raise ValueError("C9 Scout evidence lacks immutable code identity")
    if expected_code_artifact is not None and code != expected_code_artifact:
        raise ValueError("C9 Scout evidence code identity mismatch")
    return evidence


def validate_behavior_gate_evidence(
    value: Any, *, expected_stage: Optional[str] = None,
    expected_model_manifest: Optional[Dict[str, Any]] = None,
    expected_accepted_digest: Optional[Dict[str, Any]] = None,
    expected_code_artifact: Optional[Dict[str, Any]] = None,
    expected_selection_sha256: Optional[str] = None,
    expected_profile: Optional[str] = None,
    expected_cell: Optional[str] = None,
) -> Dict[str, Any]:
    evidence = validate_signed_payload(
        value, BEHAVIOR_GATE_EVIDENCE_VERSION)
    profile = evidence.get("c9_execution_profile")
    cell = evidence.get("stabilization_cell")
    if profile not in ("canary", C9_PROFILE) or cell not in (
            "C1", "C2", "C3", "C4", "C5"):
        raise ValueError("C9 behavior gate profile/cell mismatch")
    expected_variant = (
        C9_SCIENTIFIC_VARIANT if cell == "C5" else C9_SCREEN_VARIANT)
    if evidence.get("scientific_variant") != expected_variant:
        raise ValueError("C9 behavior gate scientific variant mismatch")
    if expected_profile is not None and profile != expected_profile:
        raise ValueError("C9 behavior gate execution profile mismatch")
    if expected_cell is not None and cell != expected_cell:
        raise ValueError("C9 behavior gate stabilization cell mismatch")
    stage = evidence.get("gate_stage")
    if stage not in ("scout_step50", "scout_final", "sft_fixed_endpoint"):
        raise ValueError("C9 behavior gate has an invalid stage")
    if expected_stage is not None and stage != expected_stage:
        raise ValueError("C9 behavior gate stage mismatch")
    sampling = evidence.get("sampling")
    expected_sampling = {
        "num_prompts": GATE_NUM_PROMPTS,
        "samples_per_prompt": GATE_SAMPLES_PER_PROMPT,
        "selection_seed": GATE_SELECTION_SEED,
        "generation_seed": GATE_GENERATION_SEED,
        "temperature": GATE_TEMPERATURE,
        "top_p": GATE_TOP_P,
        "top_k": GATE_TOP_K,
        "min_p": GATE_MIN_P,
        "max_completion_tokens": GATE_MAX_COMPLETION_TOKENS,
        "paired_seeds_across_prompt_modes": True,
    }
    if sampling != expected_sampling:
        raise ValueError("C9 behavior-gate sampling contract mismatch")
    if evidence.get("thresholds") != GATE_THRESHOLDS:
        raise ValueError("C9 behavior-gate threshold contract mismatch")
    if evidence.get("selection_uses_correctness") is not False:
        raise ValueError("C9 behavior gate must be benchmark/correctness blind")
    if evidence.get("panel_role") != GATE_PANEL_ROLE or \
            evidence.get("panel_held_out_from_scout_training") is not \
            GATE_PANEL_HELD_OUT_FROM_SCOUT_TRAINING:
        raise ValueError(
            "C9 behavior gate must identify its fixed diagnostic panel "
            "honestly; it is not held out from Scout training")
    if evidence.get("passed") is not True:
        raise ValueError("C9 behavior gate did not pass")
    if evidence.get("raw_rollout_count") != (
            2 * GATE_NUM_PROMPTS * GATE_SAMPLES_PER_PROMPT):
        raise ValueError("C9 behavior gate raw-rollout geometry mismatch")
    runtime = evidence.get("model_runtime_contract")
    if not isinstance(runtime, dict):
        raise ValueError("C9 behavior gate lacks the gated model runtime contract")
    expected_step = (
        50 if stage == "scout_step50" else
        SCOUT_STEPS if stage == "scout_final" else SFT_OPTIMIZER_STEPS)
    if evidence.get("model_step") != expected_step:
        raise ValueError("C9 behavior gate model step/endpoint mismatch")
    if stage.startswith("scout") and (
            runtime.get("phase") != "scout"
            or runtime.get("expected_steps") != expected_step
            or not math.isclose(
                float(runtime.get("lambda_novelty", -1.0)), 0.5)
            or runtime.get("c9_execution_profile") != profile
            or runtime.get("stabilization_cell") != cell
            or runtime.get("scientific_variant") != expected_variant):
        raise ValueError("C9 Scout behavior gate runtime mismatch")
    if stage == "sft_fixed_endpoint" and (
            runtime.get("model_init") != "Qwen/Qwen3-1.7B"
            or runtime.get("epochs") != SFT_EPOCHS
            or runtime.get("grad_accum") != 1):
        raise ValueError("C9 SFT behavior gate runtime mismatch")
    for key in (
        "heldout_selection_file_sha256", "raw_rollouts_file_sha256",
        "aggregate_file_sha256", "heldout_selection_sha256",
    ):
        if not isinstance(evidence.get(key), str) or len(evidence[key]) != 64:
            raise ValueError(f"C9 behavior gate lacks {key}")
    manifest = evidence.get("model_file_manifest")
    if not isinstance(manifest, dict) or manifest.get("version") != \
            common.DIRECTORY_FILE_MANIFEST_VERSION:
        raise ValueError("C9 behavior gate lacks a model-file manifest")
    accepted = evidence.get("accepted_library_digest")
    if not isinstance(accepted, dict) or not isinstance(
            accepted.get("sha256"), str) or len(accepted["sha256"]) != 64:
        raise ValueError("C9 behavior gate lacks an accepted-library digest")
    if profile == "canary":
        if stage != "scout_step50" or accepted.get("row_count") != 0 or \
                evidence.get("accepted_file_sha256") is not None:
            raise ValueError("C9 canary gate must not bind an SFT library")
    else:
        if stage == "scout_step50" or accepted.get("row_count") != \
                ACCEPTED_ROWS or not isinstance(
                    evidence.get("accepted_file_sha256"), str) or len(
                        evidence["accepted_file_sha256"]) != 64:
            raise ValueError("C9 full gate must bind the exact 500-row library")
    if not isinstance(evidence.get("code_artifact"), dict):
        raise ValueError("C9 behavior gate lacks immutable code identity")
    if expected_selection_sha256 is not None and \
            evidence.get("heldout_selection_sha256") != expected_selection_sha256:
        raise ValueError("C9 behavior gates do not use the same fixed panel")
    if expected_model_manifest is not None and \
            evidence.get("model_file_manifest") != expected_model_manifest:
        raise ValueError("C9 behavior gate model bytes mismatch")
    if expected_accepted_digest is not None and \
            evidence.get("accepted_library_digest") != expected_accepted_digest:
        raise ValueError("C9 behavior gate accepted-library binding mismatch")
    if expected_code_artifact is not None and \
            evidence.get("code_artifact") != expected_code_artifact:
        raise ValueError("C9 behavior gate code identity mismatch")
    return evidence


def validate_single_scout_filter_replay(value: Any) -> Dict[str, Any]:
    """Validate the independent raw->coverage_pool_c8 replay certificate.

    The pipeline creates this only after the generic MR-ME replay routine has
    reconstructed the exact ordered accepted JSONL from the raw Scout file and
    byte-compared every funnel counter.  Binding both endpoint files and the
    replayed rows prevents a merely self-consistent 500-row subset from being
    presented as the preregistered deterministic selector output.
    """
    evidence = validate_signed_payload(value, SINGLESCOUT_FILTER_REPLAY_VERSION)
    exact = {
        "policy": "coverage_pool_c8",
        "num_sources": 1,
        "source_row_count": SCOUT_TRAJECTORY_ROWS,
        "accepted_row_count": ACCEPTED_ROWS,
        "accepted_unique_problems": ACCEPTED_ROWS,
        "source_balance": False,
        "selector_replay": "raw_to_serialized_accepted_exact_order",
    }
    if any(evidence.get(key) != expected for key, expected in exact.items()):
        raise ValueError("C9 SingleScout filter replay contract mismatch")
    for key in (
            "source_file_sha256", "accepted_file_sha256",
            "accepted_rows_sha256", "funnel_sha256"):
        digest = evidence.get(key)
        if not isinstance(digest, str) or len(digest) != 64 or any(
                char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"C9 SingleScout filter replay lacks {key}")
    return evidence


def validate_sft_training_evidence(value: Any) -> Dict[str, Any]:
    evidence = validate_signed_payload(value, SFT_TRAINING_EVIDENCE_VERSION)
    _require_c9_identity(evidence)
    completion = validate_scout_completion_evidence(
        evidence.get("scout_completion_evidence"))
    accepted = evidence.get("accepted_library")
    if not isinstance(accepted, dict) or \
            accepted.get("semantic_digest", {}).get("row_count") != ACCEPTED_ROWS or \
            not isinstance(accepted.get("file_sha256"), str) or \
            len(accepted["file_sha256"]) != 64:
        raise ValueError("C9 SFT evidence must bind the exact 500-row library")
    sequence = accepted.get("training_sequence_digest")
    if not isinstance(sequence, dict) or \
            sequence.get("version") != \
            common.SFT_TRAINING_SEQUENCE_DIGEST_VERSION or \
            sequence.get("seed") != 0 or sequence.get("epochs") != SFT_EPOCHS or \
            sequence.get("input_row_count") != ACCEPTED_ROWS or \
            sequence.get("sequence_length") != ACCEPTED_ROWS * SFT_EPOCHS or \
            not isinstance(sequence.get("sha256"), str) or \
            len(sequence["sha256"]) != 64:
        raise ValueError(
            "C9 SFT evidence lacks the exact order-sensitive training sequence")
    if evidence.get("code_artifact") != completion.get("code_artifact"):
        raise ValueError("C9 SFT evidence code differs from its Scout producer")
    scout_gate = validate_behavior_gate_evidence(
        evidence.get("scout_behavior_gate"), expected_stage="scout_final",
        expected_model_manifest=completion["model_file_manifest"],
        expected_accepted_digest=accepted["semantic_digest"],
        expected_code_artifact=completion["code_artifact"],
        expected_profile=C9_PROFILE, expected_cell=C9_CELL)
    if scout_gate.get("model_runtime_contract") != completion.get(
            "scout_runtime_contract"):
        raise ValueError("C9 Scout gate/runtime differs from Scout completion")
    funnel = evidence.get("filter_funnel")
    if not isinstance(funnel, dict) or funnel.get("policy") != "coverage_pool_c8" or \
            int(funnel.get("raw", -1)) != SCOUT_TRAJECTORY_ROWS or \
            int(funnel.get("accepted", -1)) != ACCEPTED_ROWS or \
            int(funnel.get("accepted_unique_problems", -1)) != ACCEPTED_ROWS:
        raise ValueError("C9 SFT evidence filter/funnel contract mismatch")
    replay = validate_single_scout_filter_replay(
        evidence.get("single_scout_filter_replay"))
    if replay.get("source_file_sha256") != completion["trajectory"][
            "file_sha256"] or replay.get("accepted_file_sha256") != accepted[
                "file_sha256"] or replay.get("funnel_sha256") != \
                common.canonical_json_sha256(funnel):
        raise ValueError("C9 SingleScout filter replay is not bound to Scout/SFT")
    runtime = evidence.get("sft_runtime_contract")
    from tmx_gpu import grpo_gpu

    if not isinstance(runtime, dict) or runtime.get("model_init") != \
            "Qwen/Qwen3-1.7B" or runtime.get("optimizer") != "adamw" or \
            not math.isclose(float(runtime.get("learning_rate", -1.0)), 5e-6) or \
            runtime.get("epochs") != SFT_EPOCHS or runtime.get("batch_size") != 1 or \
            runtime.get("grad_accum") != 1 or runtime.get("termination_mode") != \
            "append_eos" or runtime.get("parameter_dtype") != "float32" or \
            runtime.get("loss") != "completion_only_token_mean_cross_entropy":
        raise ValueError("C9 SFT runtime contract mismatch")
    grpo_gpu.require_canonical_training_execution_environment_contract(runtime)
    return evidence


def validate_central_lineage(
    value: Any, *, expected_runtime_contract: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    lineage = validate_signed_payload(value, CENTRAL_LINEAGE_VERSION)
    if lineage.get("lineage_contract_version") != CENTRAL_LINEAGE_VERSION:
        raise ValueError("C9 Central lineage contract marker mismatch")
    _require_c9_identity(lineage)
    training = validate_sft_training_evidence(lineage.get("sft_training_evidence"))
    sft_gate = validate_behavior_gate_evidence(
        lineage.get("sft_behavior_gate"), expected_stage="sft_fixed_endpoint",
        expected_accepted_digest=training["accepted_library"]["semantic_digest"],
        expected_code_artifact=training["code_artifact"],
        expected_selection_sha256=training["scout_behavior_gate"][
            "heldout_selection_sha256"],
        expected_profile=C9_PROFILE, expected_cell=C9_CELL,
    )
    if lineage.get("fixed_sft_endpoint") is not True or \
            lineage.get("sft_optimizer_steps") != SFT_OPTIMIZER_STEPS:
        raise ValueError("C9 Central lineage is not the fixed 1,000-update SFT endpoint")
    if sft_gate.get("model_file_manifest") != lineage.get("sft_model_file_manifest"):
        raise ValueError("C9 Central lineage SFT model/gate bytes mismatch")
    artifact = lineage.get("sft_model_artifact")
    if not isinstance(artifact, dict) or artifact.get("version") != \
            "pipeline_sft_model_artifact_v1" or \
            artifact.get("durable_wandb_upload") is not True or \
            artifact.get("policy_file_manifest") != lineage.get(
                "sft_model_file_manifest"):
        raise ValueError(
            "C9 Central lineage lacks the durable, exact SFT model artifact")
    unsigned_artifact = dict(artifact)
    artifact_sha = unsigned_artifact.pop("binding_sha256", None)
    source = artifact.get("source")
    if artifact_sha != common.canonical_json_sha256(unsigned_artifact) or \
            not isinstance(source, dict) or not source.get("artifact_digest"):
        raise ValueError("C9 SFT model artifact binding is invalid")
    common.require_immutable_wandb_artifact_ref(
        str(source.get("artifact_ref", "")), field="C9 SFT model artifact")
    runtime = lineage.get("central_runtime_contract")
    from tmx_gpu import grpo_gpu

    if not isinstance(runtime, dict) or runtime.get("phase") != "central" or \
            runtime.get("expected_steps") != CENTRAL_STEPS or \
            not math.isclose(float(runtime.get("resolved_learning_rate", -1.0)), 1e-6) or \
            not math.isclose(float(runtime.get("lambda_novelty", -1.0)), 0.0) or \
            runtime.get("c9_execution_profile") != C9_PROFILE or \
            runtime.get("stabilization_cell") != C9_CELL or \
            runtime.get("probe_contract") != C9_PROBE_CONTRACT:
        raise ValueError("C9 Central runtime is not 100-step correctness-only GRPO")
    if "scout_novelty_contract" in runtime:
        raise ValueError("C9 Central runtime unexpectedly contains Scout novelty")
    grpo_gpu.require_canonical_training_execution_environment_contract(runtime)
    if expected_runtime_contract is not None and runtime != expected_runtime_contract:
        raise ValueError("C9 Central lineage differs from resolved runtime args")
    return lineage
