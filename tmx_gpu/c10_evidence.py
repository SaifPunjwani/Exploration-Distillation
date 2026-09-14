#!/usr/bin/env python3
"""Fail-closed evidence contract for the isolated C10 Scout canary."""

from __future__ import annotations

import json
import math
import numbers
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from tmx_gpu import c9_evidence, common, sft_gate_gpu


C10_PROFILE = "canary"
C10_SCIENTIFIC_VARIANT = "expdis_c10_survivor_consistent_overlong_filter"
C10_PREREGISTRATION_FILE = \
    "baseline_runs_gpu/C10_OVERLONG_FILTER_PREREGISTRATION.md"
C10_PREREGISTRATION_SHA256 = \
    "96c9ec79d16af5c86e83f33d7651b693782f93631776d020a5f66ea688100dd9"
C10_BEHAVIOR_GATE_EVIDENCE_VERSION = "c10_behavior_gate_evidence_v1"
C10_ENDPOINT_REPLAY_VERSION = "c10_endpoint_replay_v1"
C10_TRAINING_FAILURE_VERSION = "c10_training_terminal_failure_v1"
C10_TRAINING_TRANSPORT_VERSION = "c10_training_transport_invalid_v1"
C10_DYNAMIC_FAILURE_JOURNAL_VERSION = \
    "c10_dynamic_sampling_failure_journal_v1"
C10_PIPELINE_EXECUTION_CONTRACT_VERSION = \
    "c10_canary_pipeline_execution_contract_v1"
GATE_OUTCOMES = ("passed", "semantic_failed", "transport_invalid")
C10_CANONICAL_RUN_NAME = "tmx-c10-survivor-canary-0e3448b184c5"
C10_TERMINAL_REGISTRY_ARTIFACT = \
    "tmx-c10-survivor-canary-terminal-0e3448b184c5"
C10_EXCLUSIVE_LEASE_ARTIFACT = \
    "tmx-c10-survivor-canary-lease-0e3448b184c5"
C10_RESULTS_ARTIFACT = f"{C10_CANONICAL_RUN_NAME}-results"
C10_LEASE_CLAIM_VERSION = "c10_exclusive_lease_claim_v1"
C10_LEASE_BINDING_VERSION = "c10_exclusive_lease_binding_v1"
C10_FORBIDDEN_DATASET_ENV = (
    "TMX_TRAIN_DATASET_JSONL",
    "TMX_DATASET_SHARD_REQUIRED",
    "TMX_DATASET_SHARD_INDEX",
    "TMX_DATASET_SHARD_COUNT",
    "TMX_DATASET_SHARD_POLICY",
    "TMX_DATASET_SHARD_SEED",
    "TMX_DATASET_SHARD_MANIFEST_PATH",
)


def _observed_equal(left: Any, right: Any) -> bool:
    """NaN-aware, scalar-portable equality for signed failure evidence."""
    if isinstance(left, dict) or isinstance(right, dict):
        return isinstance(left, dict) and isinstance(right, dict) and \
            set(left) == set(right) and all(
                _observed_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return isinstance(left, (list, tuple)) and \
            isinstance(right, (list, tuple)) and len(left) == len(right) and \
            all(_observed_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, numbers.Real) and isinstance(right, numbers.Real):
        left_float, right_float = float(left), float(right)
        if math.isnan(left_float) or math.isnan(right_float):
            return math.isnan(left_float) and math.isnan(right_float)
        return left_float == right_float
    return type(left) is type(right) and left == right


def validate_canonical_dataset_environment(environment: Any) -> None:
    """Forbid ambient local/shard overrides for the canonical C10 dataset."""
    if not isinstance(environment, dict):
        environment = dict(environment)
    present = sorted(
        key for key in C10_FORBIDDEN_DATASET_ENV
        if str(environment.get(key, "") or "").strip())
    if present:
        raise ValueError(
            "C10 canonical dataset forbids ambient overrides: "
            + ", ".join(present))


def validate_prelease_environment(environment: Any) -> Dict[str, str]:
    """Reject launcher drift before consuming the one-shot C10 lease.

    The authoritative argparse validation necessarily runs after the lease is
    materialized because the signed lease binding is one of its inputs.  This
    smaller mirror covers every env-derived scientific/identity field before
    that irreversible remote write.  Parser-only fields are protected from
    ``EXTRA_ARGS`` by the launcher and retain their canonical defaults.
    """
    if not isinstance(environment, dict):
        environment = dict(environment)
    validate_canonical_dataset_environment(environment)
    observed = {str(key): str(value) for key, value in environment.items()}
    exact = {
        "RUN_NAME": C10_CANONICAL_RUN_NAME,
        "MODEL": common.CANONICAL_TRAINING_MODEL_ID,
        "MODEL_REVISION": common.CANONICAL_TRAINING_MODEL_REVISION,
        "C10_EXECUTION_PROFILE": C10_PROFILE,
        "C9_EXECUTION_PROFILE": "",
        "C9_MULTIEXPLORER_PROFILE": "",
        "SINGLE_SCOUT_MULTIROUND_PROFILE": "",
        "STABILIZATION_CELL": "",
        "SCIENTIFIC_VARIANT": C10_SCIENTIFIC_VARIANT,
        "ROUNDS": "1",
        "SCOUTS_PER_ROUND": "1",
        "SCOUT_STEPS": "50",
        "SCOUT_ONLY": "1",
        "SEED": "0",
        "DATASET": "dapo_math_17k",
        "MAX_TRAIN_EXAMPLES": "20000",
        "PROMPTS_PER_STEP": "4",
        "NUM_GENERATIONS": "16",
        "RL_VALIDITY_MODE": "c8",
        "RELOAD_EVERY": "1",
        "SAVE_EVERY": "25",
        "KEEP_CHECKPOINTS": "1",
        "CKPT_ARTIFACT_EVERY": "25",
        "PROBE_EVERY": "0",
        "COMPLETION_BUDGET": "0",
        "MASK_TRUNCATED": "1",
        "TRUNCATION_BASELINE_MODE": "unmasked_survivors",
        "PPO_MODE": "0",
        "UPDATES_PER_ROLLOUT": "1",
        "EXPDIS_STABILIZATION_MODE": "c9_selected_batch_update",
        "NOVELTY_NORMALIZATION": "selected_batch_zscore",
        "POLICY": "coverage_pool_c8",
        "MAX_MODEL_LEN": "36864",
        "OPTIMIZER": "adamw",
        "SFT_OPTIMIZER": "adamw",
        "SKIP_SCOUT": "0",
        "TRAJECTORIES": "",
        "RESUME": "1",
        "STOP_AFTER_SFT": "0",
        "FRESH_C8_SFT_GATE": "0",
        "FRESH_C8_MULTISCOUT_COMPONENT": "0",
    }
    drift = [
        f"{key}={observed.get(key)!r} (expected {wanted!r})"
        for key, wanted in exact.items()
        if observed.get(key, "") != wanted
    ]
    try:
        if not math.isclose(
                float(observed.get("LAMBDA", "nan")), 0.5,
                rel_tol=0.0, abs_tol=0.0):
            drift.append("LAMBDA must equal 0.5")
    except ValueError:
        drift.append("LAMBDA must equal 0.5")
    try:
        if not math.isclose(
                float(observed.get("SCOUT_LR", "nan")), 1e-6,
                rel_tol=0.0, abs_tol=1e-15):
            drift.append("SCOUT_LR must equal 1e-6")
    except ValueError:
        drift.append("SCOUT_LR must equal 1e-6")
    if observed.get("DYNAMIC_MAX_ATTEMPTS", "") not in {"0", "64"}:
        drift.append("DYNAMIC_MAX_ATTEMPTS must resolve to 64")
    if observed.get("LAMBDA_SCHEDULE_EXPLICIT", ""):
        drift.append("C10 forbids an explicit lambda schedule")
    if drift:
        raise ValueError("C10 prelease environment drift: " + "; ".join(drift))
    return observed

# The behavior panel and thresholds are intentionally byte-identical to C5.
GATE_NUM_PROMPTS = c9_evidence.GATE_NUM_PROMPTS
GATE_SAMPLES_PER_PROMPT = c9_evidence.GATE_SAMPLES_PER_PROMPT
GATE_SELECTION_SEED = c9_evidence.GATE_SELECTION_SEED
GATE_GENERATION_SEED = c9_evidence.GATE_GENERATION_SEED
GATE_MAX_COMPLETION_TOKENS = c9_evidence.GATE_MAX_COMPLETION_TOKENS
GATE_TEMPERATURE = c9_evidence.GATE_TEMPERATURE
GATE_TOP_P = c9_evidence.GATE_TOP_P
GATE_TOP_K = c9_evidence.GATE_TOP_K
GATE_MIN_P = c9_evidence.GATE_MIN_P
GATE_PANEL_ROLE = c9_evidence.GATE_PANEL_ROLE
GATE_PANEL_HELD_OUT_FROM_SCOUT_TRAINING = \
    c9_evidence.GATE_PANEL_HELD_OUT_FROM_SCOUT_TRAINING
GATE_THRESHOLDS = dict(c9_evidence.GATE_THRESHOLDS)
# Frozen from the completed C5 artifact
# gpu-jrl-c9-c5-canary-v34-20260715-r1-results:v1.  C10's preregistration
# requires the exact same pre-hashed panel, not merely the same selector code.
GATE_HELDOUT_SELECTION_SHA256 = \
    "538ac149652f752f3c2cb073c8f147fd885fb46da9e76f18c8c2e910da52ac07"
GATE_HELDOUT_SELECTION_FILE_SHA256 = \
    "899fd2351e19cc038a89be10df8b565129fea2bbe0ec9553203b0aeea83f820b"


def classify_gate_outcome(
    metrics_by_prompt_mode: Any, health_gate: Any,
) -> str:
    """Separate rerunnable transport invalidity from terminal behavior.

    The preregistration permits a byte-identical rerun only when the fixed
    panel is transport-invalid.  Every other failed criterion is a durable
    semantic failure and must not be sampled again on resume.
    """
    if not isinstance(metrics_by_prompt_mode, dict) or not isinstance(
            health_gate, dict):
        raise ValueError("C10 behavior-gate aggregate is malformed")
    transport_threshold = float(GATE_THRESHOLDS[
        "max_generation_error_rate"])
    for mode in sft_gate_gpu.PROMPT_MODES:
        metrics = metrics_by_prompt_mode.get(mode)
        if not isinstance(metrics, dict):
            raise ValueError(f"C10 behavior gate lacks prompt mode {mode!r}")
        try:
            error_rate = float(metrics["generation_error_rate"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"C10 behavior gate has invalid transport metric for {mode!r}") \
                from exc
        if error_rate > transport_threshold:
            return "transport_invalid"
    passed = health_gate.get("passed")
    if type(passed) is not bool:
        raise ValueError("C10 behavior gate lacks a Boolean verdict")
    return "passed" if passed else "semantic_failed"


def validate_preregistration_file(
    repo_root: Optional[Path] = None,
) -> Path:
    """Require the executing source tree to contain the preregistered bytes."""
    root = Path(repo_root) if repo_root is not None else Path(
        __file__).resolve().parents[1]
    path = root / C10_PREREGISTRATION_FILE
    if not path.is_file():
        raise ValueError(f"C10 preregistration file is missing: {path}")
    if common.file_sha256(str(path)) != C10_PREREGISTRATION_SHA256:
        raise ValueError("C10 preregistration file SHA256 mismatch")
    return path


def signed_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if "version" in payload or "evidence_sha256" in payload:
        raise ValueError("C10 payload must not pre-populate signature fields")
    value = {"version": C10_BEHAVIOR_GATE_EVIDENCE_VERSION, **payload}
    value["evidence_sha256"] = common.canonical_json_sha256(value)
    return value


def signed_endpoint_replay(payload: Dict[str, Any]) -> Dict[str, Any]:
    if "version" in payload or "evidence_sha256" in payload:
        raise ValueError("C10 endpoint payload must not pre-populate signatures")
    value = {"version": C10_ENDPOINT_REPLAY_VERSION, **payload}
    value["evidence_sha256"] = common.canonical_json_sha256(value)
    return value


def signed_training_failure(payload: Dict[str, Any]) -> Dict[str, Any]:
    if "version" in payload or "evidence_sha256" in payload:
        raise ValueError("C10 training-failure payload is pre-signed")
    value = {"version": C10_TRAINING_FAILURE_VERSION, **payload}
    value["evidence_sha256"] = common.canonical_json_sha256(value)
    return value


def signed_training_transport(payload: Dict[str, Any]) -> Dict[str, Any]:
    if "version" in payload or "evidence_sha256" in payload:
        raise ValueError("C10 training-transport payload is pre-signed")
    value = {"version": C10_TRAINING_TRANSPORT_VERSION, **payload}
    value["evidence_sha256"] = common.canonical_json_sha256(value)
    return value


def signed_lease_claim(payload: Dict[str, Any]) -> Dict[str, Any]:
    if "version" in payload or "evidence_sha256" in payload:
        raise ValueError("C10 lease claim is pre-signed")
    value = {"version": C10_LEASE_CLAIM_VERSION, **payload}
    value["evidence_sha256"] = common.canonical_json_sha256(value)
    return value


def validate_lease_claim(value: Any, *, expected_state: Optional[str] = None,
                         expected_holder: Optional[str] = None) -> Dict[str, Any]:
    if not isinstance(value, dict) or value.get("version") != \
            C10_LEASE_CLAIM_VERSION:
        raise ValueError("C10 lease-claim version mismatch")
    unsigned = dict(value)
    recorded = unsigned.pop("evidence_sha256", None)
    if recorded != common.canonical_json_sha256(unsigned):
        raise ValueError("C10 lease-claim signature mismatch")
    exact = {
        "run_name": C10_CANONICAL_RUN_NAME,
        "preregistration_sha256": C10_PREREGISTRATION_SHA256,
    }
    if any(value.get(key) != wanted for key, wanted in exact.items()):
        raise ValueError("C10 lease-claim identity mismatch")
    state = value.get("lease_state")
    if state not in {
            "active", "released_transport", "released_claim_race"}:
        raise ValueError("C10 lease-claim state is invalid")
    if expected_state is not None and state != expected_state:
        raise ValueError("C10 lease-claim state mismatch")
    holder = value.get("holder")
    if not isinstance(holder, dict) or not isinstance(
            holder.get("orchestrator_binding_sha256"), str) or len(
                holder["orchestrator_binding_sha256"]) != 64 or not \
            isinstance(holder.get("receipt_artifact_ref"), str) or not \
            isinstance(holder.get("receipt_artifact_digest"), str) or not \
            holder["receipt_artifact_digest"]:
        raise ValueError("C10 lease holder identity is invalid")
    common.require_immutable_wandb_artifact_ref(
        holder["receipt_artifact_ref"],
        field="C10 lease-holder receipt artifact")
    if expected_holder is not None and holder[
            "orchestrator_binding_sha256"] != expected_holder:
        raise ValueError("C10 lease holder mismatch")
    release = value.get("release")
    if state == "active":
        if release is not None:
            raise ValueError("active C10 lease claim cannot contain a release")
        return value
    if not isinstance(release, dict):
        raise ValueError("released C10 lease claim lacks its binding")
    active = release.get("active_claim")
    if not isinstance(active, dict):
        raise ValueError("C10 lease release lacks its active-claim source")
    _validate_lease_artifact_source(active)
    if not isinstance(active.get("claim_sha256"), str) or len(
            active["claim_sha256"]) != 64:
        raise ValueError("C10 lease release lacks its active-claim digest")
    authorization = release.get("authorization")
    if not isinstance(authorization, dict):
        raise ValueError("C10 lease release lacks authorization")
    if state == "released_transport":
        if authorization.get("kind") != "transport_invalid_result" or \
                authorization.get("result_outcome") != "transport_invalid" \
                or authorization.get("transport_scope") not in {
                    "training", "behavior_gate"}:
            raise ValueError("C10 transport release authorization is invalid")
        result_ref = str(authorization.get("result_artifact_ref", ""))
        common.require_immutable_wandb_artifact_ref(
            result_ref, field="C10 transport result artifact")
        if result_ref.rsplit("/", 1)[-1].rsplit(":", 1)[0] != \
                C10_RESULTS_ARTIFACT or not isinstance(
                    authorization.get("result_artifact_digest"), str) or not \
                authorization["result_artifact_digest"] or not isinstance(
                    authorization.get("transport_evidence_sha256"), str) or \
                len(authorization["transport_evidence_sha256"]) != 64:
            raise ValueError("C10 transport result binding is invalid")
        if result_ref.rsplit("/", 1)[0] != active[
                "artifact_ref"].rsplit("/", 1)[0]:
            raise ValueError("C10 lease release artifacts cross projects")
    elif authorization.get("kind") != "claim_race":
        raise ValueError("C10 claim-race release authorization is invalid")
    return value


def _validate_lease_artifact_source(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(
            value.get("artifact_digest"), str) or not value[
                "artifact_digest"]:
        raise ValueError("C10 lease artifact source is invalid")
    artifact_ref = str(value.get("artifact_ref", ""))
    common.require_immutable_wandb_artifact_ref(
        artifact_ref, field="C10 exclusive lease artifact")
    collection = artifact_ref.rsplit("/", 1)[-1].rsplit(":", 1)[0]
    if collection != C10_EXCLUSIVE_LEASE_ARTIFACT:
        raise ValueError("C10 exclusive-lease collection name mismatch")
    return value


def lease_claim_metadata(claim: Dict[str, Any]) -> Dict[str, Any]:
    """Canonical W&B metadata for an append-only lease record."""
    validate_lease_claim(claim)
    metadata = {
        "run_name": claim["run_name"],
        "preregistration_sha256": claim["preregistration_sha256"],
        "lease_state": claim["lease_state"],
        "holder_binding_sha256":
            claim["holder"]["orchestrator_binding_sha256"],
        "receipt_artifact_ref": claim["holder"]["receipt_artifact_ref"],
        "receipt_artifact_digest":
            claim["holder"]["receipt_artifact_digest"],
        "lease_claim_sha256": claim["evidence_sha256"],
    }
    if claim["lease_state"] != "active":
        active = claim["release"]["active_claim"]
        metadata.update({
            "released_active_claim_ref": active["artifact_ref"],
            "released_active_claim_digest": active["artifact_digest"],
            "released_active_claim_sha256": active["claim_sha256"],
            "release_authorization_kind":
                claim["release"]["authorization"]["kind"],
        })
    if claim["lease_state"] == "released_transport":
        authorization = claim["release"]["authorization"]
        metadata.update({
            "result_artifact_ref": authorization["result_artifact_ref"],
            "result_artifact_digest":
                authorization["result_artifact_digest"],
            "transport_scope": authorization["transport_scope"],
            "transport_evidence_sha256":
                authorization["transport_evidence_sha256"],
        })
    return metadata


def build_transport_release_claim(
    *, active_binding: Dict[str, Any], result_artifact_ref: str,
    result_artifact_digest: str, transport_scope: str,
    transport_evidence_sha256: str,
) -> Dict[str, Any]:
    """Bind a transport release to one lease epoch and one durable result."""
    binding = validate_exclusive_lease_binding(active_binding)
    source = binding["source"]
    claim = binding["claim"]
    release = signed_lease_claim({
        "run_name": C10_CANONICAL_RUN_NAME,
        "preregistration_sha256": C10_PREREGISTRATION_SHA256,
        "lease_state": "released_transport",
        "holder": claim["holder"],
        "release": {
            "active_claim": {
                "artifact_ref": source["artifact_ref"],
                "artifact_digest": source["artifact_digest"],
                "claim_sha256": claim["evidence_sha256"],
            },
            "authorization": {
                "kind": "transport_invalid_result",
                "result_artifact_ref": result_artifact_ref,
                "result_artifact_digest": str(result_artifact_digest),
                "result_outcome": "transport_invalid",
                "transport_scope": str(transport_scope),
                "transport_evidence_sha256":
                    str(transport_evidence_sha256),
            },
        },
    })
    return validate_lease_claim(release, expected_state="released_transport")


def build_exclusive_lease_binding(
    *, claim: Dict[str, Any], artifact_ref: str, artifact_digest: str,
) -> Dict[str, Any]:
    validate_lease_claim(claim, expected_state="active")
    source = {
        "artifact_ref": artifact_ref,
        "artifact_digest": str(artifact_digest),
        "artifact_type": "c10-exclusive-lease",
    }
    _validate_lease_artifact_source(source)
    payload = {
        "version": C10_LEASE_BINDING_VERSION,
        "source": source,
        "claim": claim,
    }
    payload["binding_sha256"] = common.canonical_json_sha256(payload)
    return payload


def validate_exclusive_lease_binding(
    value: Any, *, expected_orchestrator_binding: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not isinstance(value, dict) or value.get("version") != \
            C10_LEASE_BINDING_VERSION:
        raise ValueError("C10 exclusive-lease binding version mismatch")
    unsigned = dict(value)
    recorded = unsigned.pop("binding_sha256", None)
    if recorded != common.canonical_json_sha256(unsigned):
        raise ValueError("C10 exclusive-lease binding signature mismatch")
    source = value.get("source")
    if not isinstance(source, dict) or source.get("artifact_type") != \
            "c10-exclusive-lease":
        raise ValueError("C10 exclusive-lease source is invalid")
    _validate_lease_artifact_source(source)
    lease_ref = str(source["artifact_ref"])
    claim = validate_lease_claim(value.get("claim"), expected_state="active")
    receipt_ref = claim["holder"]["receipt_artifact_ref"]
    if lease_ref.rsplit("/", 1)[0] != receipt_ref.rsplit("/", 1)[0]:
        raise ValueError("C10 lease and receipt must share entity/project")
    if expected_orchestrator_binding is not None:
        source_receipt = expected_orchestrator_binding.get("source")
        if not isinstance(source_receipt, dict) or claim["holder"] != {
                "orchestrator_binding_sha256":
                    expected_orchestrator_binding.get("binding_sha256"),
                "receipt_artifact_ref": source_receipt.get("artifact_ref"),
                "receipt_artifact_digest": source_receipt.get(
                    "artifact_digest"),
        }:
            raise ValueError(
                "C10 exclusive lease is held by a different orchestrator")
    return value


def load_exclusive_lease_binding(
    path: str | Path, *,
    expected_orchestrator_binding: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return validate_exclusive_lease_binding(
        _load_json_object(Path(path), "C10 exclusive lease binding"),
        expected_orchestrator_binding=expected_orchestrator_binding)


def replay_exclusive_lease_records(
    records: list[Dict[str, Any]], *,
    authorize_transport_release: Optional[
        Callable[[Dict[str, Any]], None]] = None,
) -> Optional[Dict[str, Any]]:
    """Replay append-only lease epochs with exact release targeting.

    Each record binds one signed claim to the immutable W&B artifact version
    that contains it.  Only a ``released_transport`` claim targeting the
    *exact* winning active version may close an epoch.  Delayed duplicate
    releases for a previously closed epoch are harmless and can never close a
    newer same-holder epoch.
    """
    winner: Optional[Dict[str, Any]] = None
    active_by_ref: Dict[str, Dict[str, Any]] = {}
    closed_refs: set[str] = set()
    seen_refs: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("C10 lease history record is malformed")
        source = {
            "artifact_ref": record.get("artifact_ref"),
            "artifact_digest": record.get("artifact_digest"),
        }
        _validate_lease_artifact_source(source)
        artifact_ref = str(source["artifact_ref"])
        if artifact_ref in seen_refs:
            raise ValueError("C10 lease history repeats an artifact version")
        seen_refs.add(artifact_ref)
        claim = validate_lease_claim(record.get("claim"))
        if artifact_ref.rsplit("/", 1)[0] != claim["holder"][
                "receipt_artifact_ref"].rsplit("/", 1)[0]:
            raise ValueError("C10 lease claim crosses entity/project")
        normalized = {
            "artifact_ref": artifact_ref,
            "artifact_digest": source["artifact_digest"],
            "claim": claim,
        }
        state = claim["lease_state"]
        if state == "active":
            active_by_ref[artifact_ref] = normalized
            if winner is None:
                winner = normalized
            continue

        target = claim["release"]["active_claim"]
        target_ref = target["artifact_ref"]
        target_record = active_by_ref.get(target_ref)
        if target_record is None or target_record["artifact_digest"] != \
                target["artifact_digest"] or target_record["claim"][
                    "evidence_sha256"] != target["claim_sha256"] or \
                target_record["claim"]["holder"] != claim["holder"]:
            raise ValueError("C10 lease release target is not in its history")
        if state == "released_claim_race":
            # A losing contender never owns the epoch and therefore cannot
            # close it, even when it happens to share the winner's holder.
            continue
        if target_ref in closed_refs:
            if authorize_transport_release is not None:
                authorize_transport_release(claim)
            continue
        if winner is None or winner["artifact_ref"] != target_ref:
            raise ValueError("C10 transport release does not target the winner")
        if authorize_transport_release is not None:
            authorize_transport_release(claim)
        closed_refs.add(target_ref)
        winner = None
    return winner


def validate_published_lease_winner(
    winner: Any, *, published_claim: Dict[str, Any],
    published_artifact_ref: str, published_artifact_digest: str,
) -> Dict[str, Any]:
    """Require the winning epoch to be this bootstrap's exact publication."""
    claim = validate_lease_claim(published_claim, expected_state="active")
    source = {
        "artifact_ref": published_artifact_ref,
        "artifact_digest": published_artifact_digest,
    }
    _validate_lease_artifact_source(source)
    if not isinstance(winner, dict) or winner.get("artifact_ref") != \
            source["artifact_ref"] or winner.get("artifact_digest") != \
            source["artifact_digest"] or winner.get("claim") != claim:
        raise ValueError(
            "C10 published lease lost to another active artifact version")
    return winner


def validate_transport_release_result(
    claim: Dict[str, Any], *, result_artifact_ref: str,
    result_artifact_digest: str, result_metadata: Dict[str, Any],
    result_root: str | Path,
) -> Dict[str, Any]:
    """Revalidate the >5% transport proof that authorizes a lease release."""
    release = validate_lease_claim(
        claim, expected_state="released_transport")["release"]
    authorization = release["authorization"]
    if authorization["result_artifact_ref"] != result_artifact_ref or \
            authorization["result_artifact_digest"] != result_artifact_digest:
        raise ValueError("C10 transport release/result artifact mismatch")
    if not isinstance(result_metadata, dict) or result_metadata.get(
            "c10_outcome") != "transport_invalid" or result_metadata.get(
                "c10_terminal_scientific_failure") is not False or \
            result_metadata.get("c10_transport_scope") != authorization[
                "transport_scope"] or result_metadata.get(
                    "c10_preregistration_sha256") != \
            C10_PREREGISTRATION_SHA256:
        raise ValueError("C10 transport result metadata is invalid")
    if authorization["transport_scope"] == "training":
        validated = validate_training_transport_bundle(
            result_root, portable=True)
        evidence = validated["training_transport"]
        metadata_sha = result_metadata.get(
            "c10_training_transport_evidence_sha256")
    else:
        validated = validate_behavior_transport_bundle(
            result_root, portable_model=True)
        evidence = validated["behavior_gate"]
        metadata_sha = result_metadata.get(
            "c10_canary_behavior_gate_evidence_sha256")
    if evidence["evidence_sha256"] != authorization[
            "transport_evidence_sha256"] or metadata_sha != evidence[
                "evidence_sha256"]:
        raise ValueError("C10 transport release evidence digest mismatch")
    result_lease = validated["execution_contract"].get("exclusive_lease")
    result_lease = validate_exclusive_lease_binding(result_lease)
    released_active = release["active_claim"]
    if result_lease["source"]["artifact_ref"] != released_active[
            "artifact_ref"] or result_lease["source"][
                "artifact_digest"] != released_active[
                    "artifact_digest"] or result_lease["claim"][
                        "evidence_sha256"] != released_active[
                            "claim_sha256"] or result_lease["claim"][
                                "holder"] != claim["holder"]:
        raise ValueError(
            "C10 transport result belongs to a different lease epoch")
    return validated


def validate_signed_payload(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict) or value.get("version") != \
            C10_BEHAVIOR_GATE_EVIDENCE_VERSION:
        raise ValueError("C10 evidence version mismatch")
    unsigned = dict(value)
    recorded = unsigned.pop("evidence_sha256", None)
    if recorded != common.canonical_json_sha256(unsigned):
        raise ValueError("C10 evidence SHA256 is missing or invalid")
    return value


def validate_runtime_contract(value: Any) -> Dict[str, Any]:
    """Validate the complete C10 mechanism identity embedded by GRPO."""
    if not isinstance(value, dict):
        raise ValueError("C10 evidence lacks a runtime contract")
    from tmx_gpu import grpo_gpu

    grpo_gpu.require_canonical_training_execution_environment_contract(value)
    from tmx_jax.data import canonical_dapo_training_dataset_identity
    from tmx_jax.prompting import prompt_metadata

    nested = value.get("survivor_overlong_filter_contract")
    expected_nested = {
        "version": grpo_gpu.C10_OVERLONG_FILTER_CONTRACT_VERSION,
        "c10_execution_profile": C10_PROFILE,
        "truncation_baseline_mode": "unmasked_survivors",
        "dynamic_sampling_eligibility_population": "unmasked_survivors",
        "minimum_dynamic_sampling_survivors": 2,
        "masked_row_advantage": 0.0,
        "masked_rows_in_policy_forward": False,
        "masked_rows_in_policy_backward": False,
        "loss_denominator_population": "full_sampled_rows",
        "rnd_update_population": "all_selected_raw_rows",
        "novelty_normalization_population": "all_selected_raw_scores",
    }
    expected_novelty = {
        "novelty_layers": "7,14,21",
        "rnd_learning_rate": 1e-4,
        "rnd_hidden_dim": 512,
        "novelty_feature_max_length": 16384,
        "incorrect_novelty_scale": 0.0,
        "rnd_lifecycle": grpo_gpu.C9_RND_LIFECYCLE,
        "rnd_scoring_predictor_state": "frozen_for_entire_learner_step",
        "rnd_update_scope": "final_selected_learner_rows",
        "rnd_updates_per_completed_step": 1,
        "novelty_normalization": "selected_batch_zscore",
        "novelty_normalization_population": "all_selected_raw_scores",
        "novelty_gate_order": "normalize_then_correct_only_gate",
        "novelty_normalization_epsilon":
            grpo_gpu.C9_NORMALIZATION_EPSILON,
        "dynamic_sampling_eligibility_novelty": "raw_correct_gated",
        "stabilization_cell": "",
    }
    expected = {
        "version": grpo_gpu.GRPO_RUNTIME_CONTRACT_VERSION,
        "phase": "scout",
        "scientific_variant": C10_SCIENTIFIC_VARIANT,
        "c10_execution_profile": C10_PROFILE,
        "config_tier": (
            "gpu-max:expdis_stabilization_mode,mask_truncated,"
            "novelty_normalization,truncation_baseline_mode+"
            "scientific-variant:"
            "expdis_c10_survivor_consistent_overlong_filter"),
        "model_init": common.CANONICAL_TRAINING_MODEL_ID,
        "base_model_revision": common.CANONICAL_TRAINING_MODEL_REVISION,
        "expected_steps": 50,
        "resolved_learning_rate": 1e-6,
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
        "trajectory_record_schema_version":
            grpo_gpu.TRAJECTORY_RECORD_SCHEMA_VERSION,
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
        "mask_truncated": True,
        "clip_epsilon_low": 0.2,
        "clip_epsilon_high": 0.28,
        "ppo_mode": False,
        "updates_per_rollout": 1,
        "incorrect_novelty_scale": 0.0,
        "lambda_novelty": 0.5,
        "dataset": "dapo_math_17k",
        "training_dataset_identity":
            canonical_dapo_training_dataset_identity(),
        "max_train_examples": 20000,
        "seed": 0,
        "round": 1,
        "generation_failure_policy": {
            "owner": "c10_preregistered_generation_error_row_rate",
            "legacy_consecutive_failure_limit_enforced": False,
            "maximum_generation_error_row_rate": 0.05,
            "excess_classification": "transport_invalid",
            "dynamic_sampling_attempt_limit": 64,
        },
        "generation_timeout_seconds": 3600.0,
        "generation_concurrency_per_server": 8,
        "chunk_tokens": 256,
    }
    drift = [
        f"{key}={value.get(key)!r} (expected {wanted!r})"
        for key, wanted in expected.items()
        if not _observed_equal(value.get(key), wanted)
    ]
    if not _observed_equal(nested, expected_nested):
        drift.append("survivor_overlong_filter_contract mismatch")
    if not _observed_equal(
            value.get("scout_novelty_contract"), expected_novelty):
        drift.append("C5-held-fixed Scout novelty contract mismatch")
    for forbidden in (
            "probe_contract", "multi_explorer_contract",
            "external_lineage_contract"):
        if forbidden in value:
            drift.append(f"unexpected {forbidden}")
    if drift:
        raise ValueError("C10 runtime contract drift: " + "; ".join(drift))
    return value


def validate_endpoint_replay(
    value: Any, *,
    expected_runtime_contract: Optional[Dict[str, Any]] = None,
    expected_model_manifest: Optional[Dict[str, Any]] = None,
    expected_code_artifact: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate the signed, independently recomputed step-50 endpoint."""
    validate_preregistration_file()
    if not isinstance(value, dict) or value.get("version") != \
            C10_ENDPOINT_REPLAY_VERSION:
        raise ValueError("C10 endpoint replay version mismatch")
    unsigned = dict(value)
    recorded = unsigned.pop("evidence_sha256", None)
    if recorded != common.canonical_json_sha256(unsigned):
        raise ValueError("C10 endpoint replay SHA256 is missing or invalid")
    exact = {
        "c10_execution_profile": C10_PROFILE,
        "scientific_variant": C10_SCIENTIFIC_VARIANT,
        "round": 1,
        "model_step": 50,
        "preregistration_file": C10_PREREGISTRATION_FILE,
        "preregistration_sha256": C10_PREREGISTRATION_SHA256,
    }
    if any(value.get(key) != expected for key, expected in exact.items()):
        raise ValueError("C10 endpoint replay identity mismatch")
    runtime = validate_runtime_contract(value.get("runtime_contract"))
    if value.get("runtime_contract_sha256") != \
            common.canonical_json_sha256(runtime):
        raise ValueError("C10 endpoint replay runtime digest mismatch")
    if expected_runtime_contract is not None and runtime != \
            expected_runtime_contract:
        raise ValueError("C10 endpoint replay runtime mismatch")
    manifest = value.get("model_file_manifest")
    if not isinstance(manifest, dict) or manifest.get("version") != \
            common.DIRECTORY_FILE_MANIFEST_VERSION or not isinstance(
                manifest.get("sha256"), str):
        raise ValueError("C10 endpoint replay lacks a full model manifest")
    if expected_model_manifest is not None and manifest != \
            expected_model_manifest:
        raise ValueError("C10 endpoint replay model manifest mismatch")
    code = value.get("code_artifact")
    if not isinstance(code, dict) or not isinstance(
            code.get("file_manifest_sha256"), str) or len(
                code["file_manifest_sha256"]) != 64:
        raise ValueError("C10 endpoint replay lacks immutable code identity")
    if expected_code_artifact is not None and code != expected_code_artifact:
        raise ValueError("C10 endpoint replay code identity mismatch")
    trajectory = value.get("trajectory")
    required_trajectory = {
        "row_count": 50 * 64,
        "step_count": 50,
        "rows_per_step": 64,
        "group_count": 50 * 4,
    }
    if not isinstance(trajectory, dict) or any(
            trajectory.get(key) != expected
            for key, expected in required_trajectory.items()):
        raise ValueError("C10 endpoint trajectory geometry mismatch")
    for key in ("file_sha256", "canonical_rows_sha256"):
        if not isinstance(trajectory.get(key), str) or len(
                trajectory[key]) != 64:
            raise ValueError(f"C10 endpoint trajectory lacks {key}")
    metrics = value.get("metrics")
    if not isinstance(metrics, dict) or metrics.get("row_count") != 50:
        raise ValueError("C10 endpoint metrics geometry mismatch")
    for key in ("file_sha256", "canonical_rows_sha256"):
        if not isinstance(metrics.get(key), str) or len(metrics[key]) != 64:
            raise ValueError(f"C10 endpoint metrics lacks {key}")
    replay = value.get("survivor_replay")
    if not isinstance(replay, dict) or replay.get("raw_selected_rows") != \
            50 * 64 or replay.get("group_count") != 50 * 4 or \
            int(replay.get("survivor_rows", -1)) + int(
                replay.get("masked_rows", -1)) != 50 * 64 or \
            int(replay.get("minimum_survivors_per_group", -1)) < 2 or \
            int(replay.get("maximum_survivors_per_group", -1)) > 16 or \
            float(replay.get(
                "max_abs_survivor_advantage_sum", float("inf"))) > 1e-5:
        raise ValueError("C10 endpoint survivor replay mismatch")
    if not isinstance(replay.get("per_step_sha256"), str) or len(
            replay["per_step_sha256"]) != 64:
        raise ValueError("C10 endpoint survivor replay lacks step digest")
    health = value.get("rolling_health")
    if not isinstance(health, dict) or health.get("completed_step") != 50 or \
            int(health.get("consecutive_bad_windows", 3)) >= 3 or \
            not isinstance(health.get("recent_clip_valid"), list) or len(
                health["recent_clip_valid"]) != 8:
        raise ValueError("C10 endpoint replay health mismatch")
    return value


def validate_training_failure(
    value: Any, *,
    expected_runtime_contract: Optional[Dict[str, Any]] = None,
    expected_code_artifact: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate a signed terminal failure before the step-50 panel."""
    validate_preregistration_file()
    if not isinstance(value, dict) or value.get("version") != \
            C10_TRAINING_FAILURE_VERSION:
        raise ValueError("C10 training-failure version mismatch")
    unsigned = dict(value)
    recorded = unsigned.pop("evidence_sha256", None)
    if recorded != common.canonical_json_sha256(unsigned):
        raise ValueError("C10 training-failure SHA256 is invalid")
    exact = {
        "run_name": C10_CANONICAL_RUN_NAME,
        "c10_execution_profile": C10_PROFILE,
        "scientific_variant": C10_SCIENTIFIC_VARIANT,
        "semantic_failure": True,
        "preregistration_file": C10_PREREGISTRATION_FILE,
        "preregistration_sha256": C10_PREREGISTRATION_SHA256,
    }
    if any(value.get(key) != expected for key, expected in exact.items()):
        raise ValueError("C10 training-failure identity mismatch")
    structural_failure_kinds = {
        "selected_group_geometry_failure",
        "survivor_filter_contract_failure",
    }
    step_health_failure_kinds = {
        "learner_health_nonfinite",
        "rnd_update_contract_failure",
        "incorrect_novelty_credit_failure",
        "novelty_normalization_failure",
    }
    if value.get("failure_kind") not in {
            "rolling_behavior_collapse", "dynamic_sampling_exhaustion",
            *structural_failure_kinds,
            *step_health_failure_kinds}:
        raise ValueError("C10 training-failure kind is not preregistered")
    completed_step = value.get("completed_step")
    failed_step = value.get("failed_step")
    if type(completed_step) is not int or type(failed_step) is not int or \
            completed_step < 0 or completed_step > 50:
        raise ValueError("C10 training-failure step boundary is invalid")
    runtime = validate_runtime_contract(value.get("runtime_contract"))
    if value.get("runtime_contract_sha256") != \
            common.canonical_json_sha256(runtime):
        raise ValueError("C10 training-failure runtime digest mismatch")
    if expected_runtime_contract is not None and runtime != \
            expected_runtime_contract:
        raise ValueError("C10 training-failure runtime mismatch")
    code = value.get("code_artifact")
    if not isinstance(code, dict) or not isinstance(
            code.get("file_manifest_sha256"), str) or len(
                code["file_manifest_sha256"]) != 64:
        raise ValueError("C10 training-failure code identity is invalid")
    if expected_code_artifact is not None and code != expected_code_artifact:
        raise ValueError("C10 training-failure code mismatch")
    for section_name in ("trajectory", "metrics"):
        section = value.get(section_name)
        if not isinstance(section, dict) or type(section.get("row_count")) \
                is not int or section["row_count"] < 0:
            raise ValueError(
                f"C10 training-failure {section_name} identity is invalid")
        for key in ("file_sha256", "canonical_rows_sha256"):
            if not isinstance(section.get(key), str) or len(
                    section[key]) != 64:
                raise ValueError(
                    f"C10 training-failure {section_name} lacks {key}")
    details = value.get("failure_details")
    if not isinstance(details, dict) or not details:
        raise ValueError("C10 training-failure details are missing")
    if value["failure_kind"] == "rolling_behavior_collapse":
        if failed_step != completed_step:
            raise ValueError("C10 rolling-collapse step boundary is invalid")
        health = details.get("rolling_health")
        if not isinstance(health, dict) or health.get(
                "completed_step") != completed_step or int(health.get(
                    "consecutive_bad_windows", 0)) < 3:
            raise ValueError("C10 rolling-collapse evidence is invalid")
    elif value["failure_kind"] == "dynamic_sampling_exhaustion":
        _validate_failed_candidate_identity(value)
        if failed_step != completed_step + 1:
            raise ValueError("C10 sampling-exhaustion step boundary is invalid")
        requested = int(details.get("generation_requested_row_count", 0))
        error_rows = int(details.get("generation_error_row_count", -1))
        error_rate = (
            float(error_rows) / float(requested)
            if requested > 0 else math.inf)
        if not math.isfinite(error_rate) or error_rate > 0.05 or int(
                details.get("dynamic_max_attempts", -1)) != 64 or int(
                details.get("selected_group_count", -1)) >= 4 or int(
                    details.get("needed_group_count", -1)) != 4:
            raise ValueError("C10 sampling-exhaustion evidence is invalid")
    elif value["failure_kind"] in step_health_failure_kinds:
        if failed_step != completed_step or completed_step < 1:
            raise ValueError("C10 completed-step health boundary is invalid")
        health = details.get("health_gate")
        if not isinstance(health, dict) or health.get("failure_kind") != \
                value["failure_kind"] or not isinstance(
                    health.get("observed"), dict):
            raise ValueError("C10 completed-step health evidence is invalid")
    else:
        if failed_step != completed_step + 1 or completed_step >= 50:
            raise ValueError("C10 pre-update health boundary is invalid")
        health = details.get("health_gate")
        if not isinstance(health, dict) or health.get("failure_kind") != \
                value["failure_kind"] or health.get("failure_stage") != \
                "selected_batch_finalization_pre_policy_update" or not \
                isinstance(health.get("observed"), dict):
            raise ValueError("C10 pre-update health evidence is invalid")
        if value["failure_kind"] in structural_failure_kinds:
            selected = value.get("failed_selected_batch")
            if not isinstance(selected, dict) or type(selected.get(
                    "row_count")) is not int or not 0 <= selected[
                        "row_count"] <= 4096 or any(
                            not isinstance(selected.get(key), str) or len(
                                selected[key]) != 64
                            for key in (
                                "file_sha256", "canonical_rows_sha256",
                                "manifest_file_sha256",
                                "manifest_evidence_sha256")):
                raise ValueError(
                    "C10 structural failure lacks failed selected batch")
    return value


def _replay_failed_selected_batch(
    rows: Any, failure: Dict[str, Any], manifest: Any,
) -> None:
    """Independently replay a structural pre-update C10 failure batch."""
    from tmx_gpu import grpo_gpu

    if not isinstance(rows, list) or any(
            not isinstance(row, dict) for row in rows):
        raise ValueError("C10 failed selected batch is malformed")
    failed_step = int(failure["failed_step"])
    if not isinstance(manifest, dict) or manifest.get("version") != \
            "c10_failed_selected_batch_manifest_v1":
        raise ValueError("C10 failed selected manifest is malformed")
    unsigned_manifest = dict(manifest)
    manifest_sha = unsigned_manifest.pop("evidence_sha256", None)
    if manifest_sha != common.canonical_json_sha256(unsigned_manifest) or \
            manifest.get("failed_step") != failed_step or manifest.get(
                "selected_row_count") != len(rows) or manifest.get(
                    "selected_rows_sha256") != \
            common.canonical_json_sha256(rows):
        raise ValueError("C10 failed selected manifest does not bind rows")
    dynamic = manifest.get("dynamic_finalization")
    if not isinstance(dynamic, dict) or set(dynamic) != {
            "kept_group_count", "fallback_group_count",
            "selected_raw_row_count"}:
        raise ValueError("C10 failed selected finalization is malformed")
    indexes = [row.get("c10_failed_selected_row_index") for row in rows]
    if any(type(index) is not int for index in indexes) or \
            indexes != list(range(len(rows))) or any(
                type(row.get("c10_failed_step")) is not int or
                row.get("c10_failed_step") != failed_step for row in rows):
        raise ValueError("C10 failed selected batch order/step mismatch")
    from tmx_jax.rewarding import (
        analyze_completion_text, is_correct_completion)

    for row in rows:
        exact_row_identity = {
            "scientific_variant": C10_SCIENTIFIC_VARIANT,
            "c10_execution_profile": C10_PROFILE,
            "dynamic_sampling_eligibility_population":
                "unmasked_survivors",
            "novelty_normalization": "selected_batch_zscore",
            "rnd_update_scope": "final_selected_learner_rows",
        }
        if any(row.get(key) != wanted
               for key, wanted in exact_row_identity.items()):
            raise ValueError(
                "C10 failed selected row contract identity mismatch")
        completion = row.get("completion_text")
        truth = row.get("ground_truth")
        if not isinstance(completion, str) or not isinstance(truth, str):
            raise ValueError("C10 failed selected row lacks raw grading input")
        true_len_raw = row.get("completion_token_length_true")
        scored_len_raw = row.get("completion_token_length")
        if not grpo_gpu.c10_integral_numeric(true_len_raw) or not \
                grpo_gpu.c10_integral_numeric(scored_len_raw):
            raise ValueError("C10 failed selected row lacks lengths")
        true_len = int(float(true_len_raw))
        scored_len = int(float(scored_len_raw))
        clipped = (
            str(row.get("finish_reason", "")).lower() == "length"
            or true_len >= 16384)
        valid = bool(analyze_completion_text(completion).get(
            "valid_answer_presentation", False))
        correct = is_correct_completion(completion, truth)
        if true_len < 0 or scored_len != min(true_len, 16384) or \
                type(row.get("clipped")) is not bool or type(row.get(
                    "valid_answer")) is not bool or type(row.get(
                        "is_correct")) is not bool or bool(row.get(
                            "clipped")) != clipped or bool(row.get(
                                "valid_answer")) != valid or bool(row.get(
                                    "is_correct")) != correct:
            raise ValueError(
                "C10 failed selected raw completion does not replay")
    for start in range(0, len(rows), 16):
        group = rows[start:start + 16]
        problem_ids = {row.get("problem_id") for row in group}
        prompts = {row.get("prompt_text") for row in group}
        if any(not isinstance(value, str) or not value
               for value in problem_ids) or len(problem_ids) != 1 or \
                any(not isinstance(value, str) or not value
                    for value in prompts) or len(prompts) != 1:
            raise ValueError(
                "C10 failed selected group prompt identity mismatch")
    gate = failure["failure_details"]["health_gate"]
    observed = gate["observed"]
    kind = failure["failure_kind"]
    groups = [rows[start:start + 16] for start in range(0, len(rows), 16)]
    if kind == "selected_group_geometry_failure":
        if not _observed_equal(observed, {
                "kept_group_count": dynamic["kept_group_count"],
                "fallback_group_count": dynamic[
                    "fallback_group_count"]}) or (
                    grpo_gpu.c10_exact_numeric(
                        dynamic["kept_group_count"], 4)
                    and grpo_gpu.c10_exact_numeric(
                        dynamic["fallback_group_count"], 0)):
            raise ValueError(
                "C10 failed selected group geometry does not replay")
        return
    if "selected_raw_row_count" in observed:
        if not grpo_gpu.c10_exact_numeric(
                dynamic["kept_group_count"], 4) or not \
                grpo_gpu.c10_exact_numeric(
                    dynamic["fallback_group_count"], 0):
            raise ValueError(
                "C10 survivor failure violates live gate precedence")
        if not _observed_equal(observed, {
                "selected_raw_row_count": dynamic[
                    "selected_raw_row_count"],
                "materialized_selected_row_count": len(rows)}) or (
                grpo_gpu.c10_exact_numeric(
                    observed["selected_raw_row_count"], 64)
                 and len(rows) == 64):
            raise ValueError("C10 failed selected row count does not replay")
        return
    try:
        start = int(observed["group_start"])
        group = rows[start:start + 16]
        survivors = sum(not bool(row["clipped"]) for row in group)
        populations = sorted(set(str(row.get(
            "dynamic_sampling_eligibility_population")) for row in group))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("C10 failed survivor claim is malformed") from exc
    if not grpo_gpu.c10_exact_numeric(
            dynamic["kept_group_count"], 4) or not \
            grpo_gpu.c10_exact_numeric(
                dynamic["fallback_group_count"], 0) or not \
            grpo_gpu.c10_exact_numeric(
                dynamic["selected_raw_row_count"], 64) or len(rows) != 64:
        raise ValueError(
            "C10 survivor group violates live gate precedence")
    for prior_start in range(0, start, 16):
        prior = rows[prior_start:prior_start + 16]
        prior_survivors = sum(
            not bool(row["clipped"]) for row in prior)
        prior_populations = sorted(set(str(row.get(
            "dynamic_sampling_eligibility_population")) for row in prior))
        if prior_survivors < 2 or prior_populations != [
                "unmasked_survivors"]:
            raise ValueError(
                "C10 survivor claim is not the first live-gate violation")
    if start not in range(0, len(rows), 16) or len(group) != 16 or \
            not _observed_equal(observed, {
                "group_start": start, "survivors": survivors,
                "populations": populations}) or (
                    survivors >= 2 and populations == [
                        "unmasked_survivors"]):
        raise ValueError("C10 failed survivor group does not replay")


def validate_training_transport_invalid(
    value: Any, *,
    expected_runtime_contract: Optional[Dict[str, Any]] = None,
    expected_code_artifact: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate a rerunnable >5% training-generation transport failure."""
    validate_preregistration_file()
    if not isinstance(value, dict) or value.get("version") != \
            C10_TRAINING_TRANSPORT_VERSION:
        raise ValueError("C10 training-transport version mismatch")
    unsigned = dict(value)
    recorded = unsigned.pop("evidence_sha256", None)
    if recorded != common.canonical_json_sha256(unsigned):
        raise ValueError("C10 training-transport SHA256 is invalid")
    exact = {
        "run_name": C10_CANONICAL_RUN_NAME,
        "c10_execution_profile": C10_PROFILE,
        "scientific_variant": C10_SCIENTIFIC_VARIANT,
        "transport_invalid": True,
        "preregistration_file": C10_PREREGISTRATION_FILE,
        "preregistration_sha256": C10_PREREGISTRATION_SHA256,
    }
    if any(value.get(key) != expected for key, expected in exact.items()):
        raise ValueError("C10 training-transport identity mismatch")
    failure_kind = value.get("failure_kind")
    if failure_kind not in {
            "dynamic_sampling_transport_invalid",
            "completed_step_transport_invalid"}:
        raise ValueError("C10 training-transport kind is invalid")
    completed_step = value.get("completed_step")
    failed_step = value.get("failed_step")
    if type(completed_step) is not int or type(failed_step) is not int or \
            completed_step < 0 or completed_step > 50:
        raise ValueError("C10 training-transport step boundary is invalid")
    if failure_kind == "dynamic_sampling_transport_invalid":
        if completed_step >= 50 or failed_step != completed_step + 1:
            raise ValueError("C10 pre-step transport boundary is invalid")
    elif completed_step < 1 or failed_step != completed_step:
        raise ValueError("C10 completed-step transport boundary is invalid")
    runtime = validate_runtime_contract(value.get("runtime_contract"))
    if value.get("runtime_contract_sha256") != \
            common.canonical_json_sha256(runtime):
        raise ValueError("C10 training-transport runtime digest mismatch")
    if expected_runtime_contract is not None and runtime != \
            expected_runtime_contract:
        raise ValueError("C10 training-transport runtime mismatch")
    code = value.get("code_artifact")
    if not isinstance(code, dict) or not isinstance(
            code.get("file_manifest_sha256"), str) or len(
                code["file_manifest_sha256"]) != 64:
        raise ValueError("C10 training-transport code identity is invalid")
    if expected_code_artifact is not None and code != expected_code_artifact:
        raise ValueError("C10 training-transport code mismatch")
    details = value.get("failure_details")
    if not isinstance(details, dict):
        raise ValueError("C10 training-transport details are missing")
    requested = int(details.get("generation_requested_row_count", 0))
    errors = int(details.get("generation_error_row_count", -1))
    rate = float(errors) / float(requested) if requested > 0 else math.inf
    if not math.isfinite(rate) or rate <= 0.05:
        raise ValueError("C10 training transport did not exceed 5%")
    if value["failure_kind"] == "dynamic_sampling_transport_invalid":
        _validate_failed_candidate_identity(
            value,
            allow_empty=int(details.get("candidate_group_count", -1)) == 0,
        )
    for section_name in ("trajectory", "metrics"):
        section = value.get(section_name)
        if not isinstance(section, dict) or type(section.get("row_count")) \
                is not int or section["row_count"] < 0:
            raise ValueError(
                f"C10 training-transport {section_name} identity invalid")
        for key in ("file_sha256", "canonical_rows_sha256"):
            if not isinstance(section.get(key), str) or len(
                    section[key]) != 64:
                raise ValueError(
                    f"C10 training-transport {section_name} lacks {key}")
    return value


def validate_behavior_gate_evidence(
    value: Any, *, expected_model_manifest: Optional[Dict[str, Any]] = None,
    expected_code_artifact: Optional[Dict[str, Any]] = None,
    expected_selection_sha256: Optional[str] = None,
    expected_endpoint_replay_sha256: Optional[str] = None,
    require_passed: Optional[bool] = True,
) -> Dict[str, Any]:
    validate_preregistration_file()
    evidence = validate_signed_payload(value)
    exact = {
        "c10_execution_profile": C10_PROFILE,
        "scientific_variant": C10_SCIENTIFIC_VARIANT,
        "gate_stage": "scout_step50",
        "selection_uses_correctness": False,
        "panel_role": GATE_PANEL_ROLE,
        "panel_held_out_from_scout_training":
            GATE_PANEL_HELD_OUT_FROM_SCOUT_TRAINING,
        "model_step": 50,
        "raw_rollout_count": 2 * GATE_NUM_PROMPTS * GATE_SAMPLES_PER_PROMPT,
        "preregistration_file": C10_PREREGISTRATION_FILE,
        "preregistration_sha256": C10_PREREGISTRATION_SHA256,
        "heldout_selection_sha256": GATE_HELDOUT_SELECTION_SHA256,
        "heldout_selection_file_sha256":
            GATE_HELDOUT_SELECTION_FILE_SHA256,
    }
    if any(evidence.get(key) != expected for key, expected in exact.items()):
        raise ValueError("C10 behavior-gate identity or endpoint mismatch")
    passed = evidence.get("passed")
    outcome = evidence.get("gate_outcome")
    if type(passed) is not bool or outcome not in GATE_OUTCOMES:
        raise ValueError("C10 behavior-gate verdict is malformed")
    if (outcome == "passed") is not passed:
        raise ValueError("C10 behavior-gate outcome/pass mismatch")
    if require_passed is True and not passed:
        raise ValueError("C10 behavior gate did not pass")
    if require_passed is False and passed:
        raise ValueError("C10 negative behavior evidence unexpectedly passed")
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
    if evidence.get("sampling") != expected_sampling or \
            evidence.get("thresholds") != GATE_THRESHOLDS:
        raise ValueError("C10 behavior-gate sampling/threshold contract drift")
    runtime = validate_runtime_contract(evidence.get("model_runtime_contract"))
    if evidence.get("model_runtime_contract_sha256") != \
            common.canonical_json_sha256(runtime):
        raise ValueError("C10 runtime contract digest mismatch")
    accepted = evidence.get("accepted_library_digest")
    if not isinstance(accepted, dict) or accepted.get("row_count") != 0 or \
            evidence.get("accepted_file_sha256") is not None:
        raise ValueError("C10 canary must not bind an SFT library")
    for key in (
            "heldout_selection_file_sha256", "raw_rollouts_file_sha256",
            "aggregate_file_sha256", "heldout_selection_sha256",
            "endpoint_replay_evidence_sha256"):
        digest = evidence.get(key)
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"C10 evidence lacks {key}")
    manifest = evidence.get("model_file_manifest")
    if not isinstance(manifest, dict) or manifest.get("version") != \
            common.DIRECTORY_FILE_MANIFEST_VERSION:
        raise ValueError("C10 evidence lacks a model-file manifest")
    code = evidence.get("code_artifact")
    if not isinstance(code, dict) or not isinstance(
            code.get("file_manifest_sha256"), str):
        raise ValueError("C10 evidence lacks immutable code identity")
    if expected_model_manifest is not None and manifest != expected_model_manifest:
        raise ValueError("C10 gated model bytes mismatch")
    if expected_code_artifact is not None and code != expected_code_artifact:
        raise ValueError("C10 code identity mismatch")
    if expected_selection_sha256 is not None and evidence.get(
            "heldout_selection_sha256") != expected_selection_sha256:
        raise ValueError("C10 fixed-panel identity mismatch")
    if expected_endpoint_replay_sha256 is not None and evidence.get(
            "endpoint_replay_evidence_sha256") != \
            expected_endpoint_replay_sha256:
        raise ValueError("C10 endpoint-to-gate binding mismatch")
    return evidence


def _load_json_object(path: Path, label: str) -> Dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _load_jsonl_objects(path: Path, label: str) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    continue
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise ValueError(
                        f"{label} row {line_number} is not an object")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load {label}: {path}") from exc
    return rows


def _validate_failed_candidate_identity(
    value: Dict[str, Any], *, allow_empty: bool = False,
) -> Dict[str, Any]:
    section = value.get("failed_step_candidates")
    if not isinstance(section, dict) or type(section.get("row_count")) is not \
            int or section["row_count"] < 0 or section["row_count"] % 16 or \
            (section["row_count"] == 0 and not allow_empty):
        raise ValueError("C10 failed-step candidate geometry is invalid")
    for key in ("file_sha256", "canonical_rows_sha256"):
        if not isinstance(section.get(key), str) or len(section[key]) != 64:
            raise ValueError(f"C10 failed-step candidates lack {key}")
    return section


def _validate_dynamic_failure_journal(
    root: Path, *, evidence: Dict[str, Any],
    candidates: list[Dict[str, Any]], transport: bool,
) -> Dict[str, Any]:
    path = root / "round_01" / "scout" / \
        "c10_dynamic_sampling_failure_journal.json"
    journal = _load_json_object(path, "C10 dynamic failure journal")
    if journal.get("version") != C10_DYNAMIC_FAILURE_JOURNAL_VERSION:
        raise ValueError("C10 dynamic journal version mismatch")
    unsigned = dict(journal)
    recorded = unsigned.pop("evidence_sha256", None)
    if recorded != common.canonical_json_sha256(unsigned):
        raise ValueError("C10 dynamic journal signature mismatch")
    details = dict(evidence["failure_details"])
    if transport:
        details.pop("observed_generation_error_rate", None)
    if journal.get("run_name") != C10_CANONICAL_RUN_NAME or journal.get(
            "scientific_variant") != C10_SCIENTIFIC_VARIANT or journal.get(
                "completed_step") != evidence["completed_step"] or \
            journal.get("failed_step") != evidence["failed_step"] or \
            journal.get("runtime_contract_sha256") != evidence[
                "runtime_contract_sha256"] or journal.get(
                    "code_artifact") != evidence["code_artifact"] or \
            journal.get("failure_details") != details or journal.get(
                "candidate_rows") != candidates or journal.get(
                    "candidate_rows_sha256") != \
            common.canonical_json_sha256(candidates):
        raise ValueError("C10 dynamic journal/evidence binding mismatch")
    return journal


def _replay_failed_dynamic_candidates(
    rows: list[Dict[str, Any]], details: Dict[str, Any],
) -> None:
    """Replay the survivor-only eligibility decision for an exhausted step."""
    from tmx_gpu import grpo_gpu

    # A total rollout-server outage legitimately has no scored candidate
    # groups.  The signed 64-attempt summaries still account for all 4,096
    # requested rows and are replayed below; semantic exhaustion continues to
    # require at least one scored group via _validate_failed_candidate_identity.
    if len(rows) % 16:
        raise ValueError("C10 failed candidate rows are not complete groups")
    groups: Dict[int, list[Dict[str, Any]]] = {}
    for row in rows:
        ordinal = row.get("candidate_group_ordinal")
        index = row.get("row_index")
        if type(ordinal) is not int or type(index) is not int or not \
                0 <= index < 16:
            raise ValueError("C10 failed candidate row identity is invalid")
        groups.setdefault(ordinal, []).append(row)
    if sorted(groups) != list(range(len(groups))) or any(
            sorted(int(row["row_index"]) for row in group) != list(range(16))
            for group in groups.values()):
        raise ValueError("C10 failed candidate group ordering is invalid")

    summaries = []
    selected_count = 0
    for ordinal in range(len(groups)):
        group = sorted(groups[ordinal], key=lambda row: int(row["row_index"]))
        common_fields = (
            "problem_id", "dynamic_sampling_attempt",
            "dynamic_sampling_group_index", "dynamic_sampling_reason",
            "dynamic_sampling_group_selected",
            "dynamic_sampling_survivor_count",
            "dynamic_sampling_masked_count",
            "dynamic_sampling_raw_correct_count",
            "dynamic_sampling_survivor_correct_count",
            "dynamic_sampling_eligibility_population",
            "dynamic_sampling_eligibility_population_size",
            "dynamic_sampling_reward_std",
            "dynamic_sampling_novelty_contribution_std",
        )
        if any(any(row.get(field) != group[0].get(field) for row in group)
               for field in common_fields):
            raise ValueError("C10 failed candidate group fields disagree")
        reason, eligible, facts = grpo_gpu.group_eligibility_for_rows(
            group, lambda_novelty=0.5, validity_mode="c8",
            survivor_only=True)
        recorded_selected = bool(
            group[0]["dynamic_sampling_group_selected"])
        if reason != group[0]["dynamic_sampling_reason"] or \
                eligible is not recorded_selected:
            raise ValueError("C10 failed candidate eligibility does not replay")
        exact_facts = {
            "survivor_count": "dynamic_sampling_survivor_count",
            "masked_count": "dynamic_sampling_masked_count",
            "raw_correct_count": "dynamic_sampling_raw_correct_count",
            "survivor_correct_count":
                "dynamic_sampling_survivor_correct_count",
            "eligibility_population":
                "dynamic_sampling_eligibility_population",
            "eligibility_population_size":
                "dynamic_sampling_eligibility_population_size",
        }
        if any(facts[key] != group[0][field]
               for key, field in exact_facts.items()):
            raise ValueError("C10 failed candidate counts do not replay")
        for key, field in (
                ("eligibility_reward_std", "dynamic_sampling_reward_std"),
                ("eligibility_novelty_contribution_std",
                 "dynamic_sampling_novelty_contribution_std")):
            if not math.isclose(
                    float(facts[key]), float(group[0][field]),
                    rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError("C10 failed candidate variance does not replay")
        selected_count += int(recorded_selected)
        summaries.append({
            "problem_id": str(group[0]["problem_id"]),
            "attempt": int(group[0]["dynamic_sampling_attempt"]),
            "group_index": int(group[0][
                "dynamic_sampling_group_index"]),
            "reason": str(group[0]["dynamic_sampling_reason"]),
            "survivor_count": int(group[0][
                "dynamic_sampling_survivor_count"]),
            "masked_count": int(group[0]["dynamic_sampling_masked_count"]),
            "survivor_correct_count": int(group[0][
                "dynamic_sampling_survivor_correct_count"]),
            "eligibility_reward_std": float(group[0][
                "dynamic_sampling_reward_std"]),
            "eligibility_novelty_contribution_std": float(group[0][
                "dynamic_sampling_novelty_contribution_std"]),
            "selected": recorded_selected,
        })
    candidate_count = len(groups)
    error_groups = int(details.get("generation_error_group_count", -1))
    error_rows = int(details.get("generation_error_row_count", -1))
    requested = int(details.get("generation_requested_row_count", -1))
    max_attempts = int(details.get("dynamic_max_attempts", -1))
    attempts = details.get("generation_attempt_summaries")
    if max_attempts != 64 or not isinstance(attempts, list) or len(
            attempts) != max_attempts or [item.get("attempt") for item in
                                          attempts if isinstance(item, dict)] \
            != list(range(1, max_attempts + 1)) or \
            common.canonical_json_sha256(attempts) != details.get(
                "generation_attempt_summaries_sha256"):
        raise ValueError("C10 failed candidate retry boundary does not replay")
    groups_by_attempt: Dict[int, list[Dict[str, Any]]] = {}
    for group in summaries:
        groups_by_attempt.setdefault(int(group["attempt"]), []).append(group)
    cumulative_selected = 0
    replay_requested_groups = 0
    replay_error_groups = 0
    replay_error_rows = 0
    for expected_attempt, attempt_summary in enumerate(attempts, 1):
        if not isinstance(attempt_summary, dict):
            raise ValueError("C10 failed candidate attempt is malformed")
        attempt_groups = groups_by_attempt.get(expected_attempt, [])
        group_indexes = sorted(int(group["group_index"])
                               for group in attempt_groups)
        candidate_groups_this_attempt = int(
            attempt_summary.get("candidate_group_count", -1))
        requested_groups_this_attempt = int(
            attempt_summary.get("requested_group_count", -1))
        error_groups_this_attempt = int(
            attempt_summary.get("generation_error_group_count", -1))
        error_rows_this_attempt = int(
            attempt_summary.get("generation_error_row_count", -1))
        required_groups_this_attempt = 4 - cumulative_selected
        cumulative_selected += sum(
            int(group["selected"]) for group in attempt_groups)
        if group_indexes != list(range(candidate_groups_this_attempt)) or \
                requested_groups_this_attempt != \
                required_groups_this_attempt or \
                candidate_groups_this_attempt + error_groups_this_attempt != \
                requested_groups_this_attempt or not \
                error_groups_this_attempt <= error_rows_this_attempt <= \
                error_groups_this_attempt * 16 or int(attempt_summary.get(
                    "selected_group_count_after_attempt", -1)) != \
                cumulative_selected:
            raise ValueError("C10 failed candidate attempt does not replay")
        replay_requested_groups += requested_groups_this_attempt
        replay_error_groups += error_groups_this_attempt
        replay_error_rows += error_rows_this_attempt
    if set(groups_by_attempt) - set(range(1, max_attempts + 1)) or \
            candidate_count != int(details.get(
                "candidate_group_count", -1)) or \
            selected_count != int(details.get("selected_group_count", -1)) or \
            selected_count >= int(details.get("needed_group_count", -1)) or \
            int(details.get("discarded_group_count", -1)) != \
            candidate_count - selected_count or \
            requested != (candidate_count + error_groups) * 16 or \
            requested != replay_requested_groups * 16 or \
            error_groups != replay_error_groups or \
            error_rows != replay_error_rows or \
            not error_groups <= error_rows <= max(0, error_groups) * 16 or \
            common.canonical_json_sha256(summaries) != details.get(
                "candidate_group_summaries_sha256") or \
            summaries != details.get("candidate_group_summaries"):
        raise ValueError("C10 failed candidate aggregate does not replay")


def _reject_mutually_exclusive_outcomes(
    root: Path, *, allowed: str,
) -> None:
    """Fail closed when one canary directory contains competing outcomes.

    A C10 output directory represents one preregistered stochastic draw.  A
    pre-gate scientific failure, a transport-invalid training attempt, and a
    completed behavior-gate outcome are mutually exclusive histories.  The
    portable validators enforce this independently of pipeline control flow so
    a later resume or artifact uploader cannot reinterpret an earlier result.
    """
    markers = {
        "training_terminal": root / "round_01" / "scout" /
            "c10_training_terminal_outcome.json",
        "training_transport": root / "round_01" / "scout" /
            "c10_training_transport_invalid.json",
        "endpoint": root / "round_01" / "c10_endpoint_replay.json",
        "behavior_gate": root / "round_01" / "c10_behavior_gates" /
            "scout_step50" / "evidence.json",
        "summary": root / "run_summary.json",
    }
    if allowed not in markers:
        raise ValueError(f"unknown C10 outcome marker {allowed!r}")
    present = sorted(
        name for name, path in markers.items()
        if name != allowed and path.exists())
    if present:
        raise ValueError(
            f"C10 {allowed} outcome conflicts with existing evidence: "
            + ", ".join(present))


def validate_execution_contract(
    value: Any, *, expected_output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate the signed, unique, preregistered C10 launch identity."""
    validate_preregistration_file()
    if not isinstance(value, dict):
        raise ValueError("C10 execution contract is missing")
    unsigned = dict(value)
    recorded = unsigned.pop("contract_sha256", None)
    if recorded != common.canonical_json_sha256(unsigned):
        raise ValueError("C10 execution-contract SHA256 is invalid")
    exact = {
        "version": C10_PIPELINE_EXECUTION_CONTRACT_VERSION,
        "run_name": C10_CANONICAL_RUN_NAME,
        "c10_execution_profile": C10_PROFILE,
        "scientific_variant": C10_SCIENTIFIC_VARIANT,
        "preregistration_file": C10_PREREGISTRATION_FILE,
        "preregistration_sha256": C10_PREREGISTRATION_SHA256,
        "model": common.CANONICAL_TRAINING_MODEL_ID,
        "model_revision": common.CANONICAL_TRAINING_MODEL_REVISION,
        "dataset": "dapo_math_17k",
        "seed": 0,
        "rounds": 1,
        "lambda_schedule": [0.5],
        "scout_steps": 50,
        "scout_only": True,
    }
    drift = [
        f"{key}={value.get(key)!r} (expected {wanted!r})"
        for key, wanted in exact.items()
        if not _observed_equal(value.get(key), wanted)
    ]
    if expected_output_dir is not None and os.path.realpath(str(
            value.get("output_dir", ""))) != os.path.realpath(
                expected_output_dir):
        drift.append("output_dir mismatch")
    runtime = validate_runtime_contract(value.get("scout_runtime_contract"))
    gate_contract = value.get("behavior_gate_contract")
    expected_gate = {
        "metric_contract_version": sft_gate_gpu.METRIC_CONTRACT_VERSION,
        "gate_stage": "scout_step50",
        "sampling": {
            "num_prompts": GATE_NUM_PROMPTS,
            "samples_per_prompt": GATE_SAMPLES_PER_PROMPT,
            "selection_seed": GATE_SELECTION_SEED,
            "generation_seed": GATE_GENERATION_SEED,
            "temperature": GATE_TEMPERATURE,
            "top_p": GATE_TOP_P,
            "top_k": GATE_TOP_K,
            "min_p": GATE_MIN_P,
            "max_completion_tokens": GATE_MAX_COMPLETION_TOKENS,
        },
        "thresholds": GATE_THRESHOLDS,
    }
    if not _observed_equal(gate_contract, expected_gate):
        drift.append("behavior_gate_contract mismatch")
    code = value.get("code_artifact")
    if not isinstance(code, dict) or not isinstance(
            code.get("file_manifest_sha256"), str) or len(
                code["file_manifest_sha256"]) != 64:
        drift.append("code_artifact mismatch")
    try:
        expected_receipt = None
        if expected_output_dir is not None:
            from tmx_gpu import orchestrator_receipt

            expected_receipt = orchestrator_receipt.load_binding(
                str(Path(expected_output_dir) /
                    "orchestrator_receipt_binding.json"),
                expected_run_name=C10_CANONICAL_RUN_NAME)
        validate_exclusive_lease_binding(
            value.get("exclusive_lease"),
            expected_orchestrator_binding=expected_receipt)
    except ValueError as exc:
        drift.append(f"exclusive_lease mismatch ({exc})")
    if drift:
        raise ValueError("C10 execution contract drift: " + "; ".join(drift))
    # Keep the validated object reachable for callers without reconstructing
    # it from human-facing launch flags.
    if runtime is not value["scout_runtime_contract"]:
        raise AssertionError("runtime validator unexpectedly replaced object")
    return value


def validate_behavior_gate_bundle(
    gate_dir: str | Path, *,
    expected_endpoint_replay_sha256: str,
    expected_model_manifest: Optional[Dict[str, Any]] = None,
    expected_code_artifact: Optional[Dict[str, Any]] = None,
    require_passed: Optional[bool] = None,
) -> Dict[str, Any]:
    """Rebuild the fixed panel verdict from its portable raw files."""
    root = Path(gate_dir)
    evidence = _load_json_object(root / "evidence.json", "C10 gate evidence")
    selection_path = root / "heldout_selection.jsonl"
    rollout_path = root / "rollouts.jsonl"
    aggregate_path = root / "aggregate_metrics.json"
    for path, key in (
            (selection_path, "heldout_selection_file_sha256"),
            (rollout_path, "raw_rollouts_file_sha256"),
            (aggregate_path, "aggregate_file_sha256")):
        if common.file_sha256(str(path)) != evidence.get(key):
            raise ValueError(f"C10 gate bundle file hash mismatch: {path.name}")
    selection = _load_jsonl_objects(selection_path, "C10 fixed panel")
    if common.canonical_json_sha256(selection) != evidence.get(
            "heldout_selection_sha256"):
        raise ValueError("C10 fixed-panel semantic hash mismatch")
    rows = _load_jsonl_objects(rollout_path, "C10 gate rollouts")
    aggregate = _load_json_object(
        aggregate_path, "C10 aggregate metrics")
    metrics, gate = sft_gate_gpu.recompute_gate_from_rollouts(
        rows,
        num_prompts=GATE_NUM_PROMPTS,
        samples_per_prompt=GATE_SAMPLES_PER_PROMPT,
        generation_seed=GATE_GENERATION_SEED,
        max_completion_tokens=GATE_MAX_COMPLETION_TOKENS,
        ordered_problem_ids=[str(row["problem_id"]) for row in selection],
        thresholds=GATE_THRESHOLDS,
    )
    if metrics != aggregate.get("metrics_by_prompt_mode") or \
            gate != aggregate.get("health_gate"):
        raise ValueError("C10 gate aggregate does not replay from raw rows")
    outcome = classify_gate_outcome(metrics, gate)
    if outcome != evidence.get("gate_outcome") or bool(gate["passed"]) is not \
            evidence.get("passed"):
        raise ValueError("C10 gate outcome does not replay from raw rows")
    return validate_behavior_gate_evidence(
        evidence,
        expected_model_manifest=expected_model_manifest,
        expected_code_artifact=expected_code_artifact,
        expected_selection_sha256=common.canonical_json_sha256(selection),
        expected_endpoint_replay_sha256=
            expected_endpoint_replay_sha256,
        require_passed=require_passed,
    )


def _validate_terminal_bundle(
    out_dir: str | Path, *, require_passed: Optional[bool],
    require_gate_stage: bool = True, portable_model: bool = False,
) -> Dict[str, Any]:
    """Cross-bind every portable C10 object at a terminal gate outcome."""
    root = Path(out_dir).resolve()
    # A completed behavior-gate history cannot coexist with either pre-gate
    # outcome.  ``summary`` is intentionally allowed here and checked by the
    # passed/failed wrapper below.
    for conflict in (
            root / "round_01" / "scout" /
                "c10_training_terminal_outcome.json",
            root / "round_01" / "scout" /
                "c10_training_transport_invalid.json",
            root / "round_01" / "scout" /
                "c10_dynamic_sampling_failure_journal.json"):
        if conflict.exists():
            raise ValueError(
                "C10 behavior-gate outcome conflicts with pre-gate evidence: "
                f"{conflict}")
    state = _load_json_object(root / "pipeline_state.json", "C10 state")
    stages = state.get("stages")
    required_stages = {"r01/scout"}
    if require_gate_stage:
        required_stages.add("r01/c10_canary_behavior_gate")
    if not isinstance(stages, dict) or set(stages) != required_stages:
        raise ValueError(
            "C10 terminal state must contain exactly Scout and behavior gate")
    execution = validate_execution_contract(
        state.get("c10_execution_contract"),
        expected_output_dir=None if portable_model else str(root))
    runtime = execution["scout_runtime_contract"]
    code = execution["code_artifact"]
    scout_stage = stages["r01/scout"]
    gate_stage = stages.get("r01/c10_canary_behavior_gate", {})
    if not isinstance(scout_stage, dict) or not isinstance(gate_stage, dict):
        raise ValueError("C10 terminal stages are malformed")

    endpoint_path = root / "round_01" / "c10_endpoint_replay.json"
    if require_gate_stage and not portable_model and os.path.realpath(str(
            gate_stage.get("endpoint_replay_path", ""))) != \
            os.path.realpath(str(endpoint_path)):
        raise ValueError("C10 state endpoint path mismatch")
    endpoint = _load_json_object(endpoint_path, "C10 endpoint replay")
    endpoint = validate_endpoint_replay(
        endpoint, expected_runtime_contract=runtime,
        expected_code_artifact=code)
    endpoint_sha = endpoint["evidence_sha256"]
    if require_gate_stage and (
            gate_stage.get("canary_endpoint_sha256") != endpoint_sha or
            gate_stage.get("endpoint_replay_file_sha256") !=
            common.file_sha256(str(endpoint_path))):
        raise ValueError("C10 state endpoint digest mismatch")

    expected_final = root / "round_01" / "scout" / "final"
    recorded_final = Path(str(scout_stage.get("final", ""))).resolve()
    final_model = expected_final.resolve() if portable_model else recorded_final
    if (not portable_model and recorded_final != expected_final.resolve()) or \
            not final_model.is_dir():
        raise ValueError("C10 Scout final-model path mismatch")
    trainer = _load_json_object(
        final_model / "trainer_state.json", "C10 trainer state")
    manifest = trainer.get("checkpoint_file_manifest")
    if trainer.get("phase") != "scout" or trainer.get("step") != 50 or \
            trainer.get("runtime_contract") != runtime or \
            trainer.get("code_artifact_binding") != code or \
            trainer.get("rnd_selected_batch_update_count") != 50 or \
            trainer.get("c10_canary_health_state") != \
            endpoint.get("rolling_health"):
        raise ValueError("C10 trainer/runtime/RND/health binding mismatch")
    if portable_model:
        if not isinstance(manifest, dict) or manifest.get("version") != \
                common.DIRECTORY_FILE_MANIFEST_VERSION:
            raise ValueError("portable C10 trainer lacks its model manifest")
    else:
        common.verify_directory_file_manifest(
            str(final_model), manifest,
            exclude_relative_paths=("trainer_state.json",))
    if endpoint.get("model_file_manifest") != manifest or \
            (require_gate_stage and gate_stage.get(
                "model_file_manifest") != manifest):
        raise ValueError("C10 endpoint/state model manifest mismatch")

    trajectory_paths = scout_stage.get("traj_paths")
    expected_trajectory = root / "round_01" / "scout" / \
        "explorer_trajectories.jsonl"
    if not isinstance(trajectory_paths, list) or len(trajectory_paths) != 1 \
            or (not portable_model and os.path.realpath(str(
                trajectory_paths[0])) != os.path.realpath(str(
                    expected_trajectory))):
        raise ValueError("C10 Scout trajectory path mismatch")
    trajectory_rows = _load_jsonl_objects(
        expected_trajectory, "C10 trajectories")
    trajectory_identity = endpoint["trajectory"]
    if common.file_sha256(str(expected_trajectory)) != trajectory_identity[
            "file_sha256"] or common.canonical_json_sha256(
                trajectory_rows) != trajectory_identity[
                    "canonical_rows_sha256"]:
        raise ValueError("C10 endpoint trajectory bytes/rows mismatch")

    metrics_path = root / "metrics.jsonl"
    all_metrics = _load_jsonl_objects(metrics_path, "C10 metrics")
    learner_metrics = [
        row for row in all_metrics
        if row.get("phase") == common.PARITY_PHASES["scout"]
        and type(row.get("grpo/step")) is int
        and "train/grpo_loss" in row
    ]
    metrics_identity = endpoint["metrics"]
    if common.file_sha256(str(metrics_path)) != metrics_identity[
            "file_sha256"] or common.canonical_json_sha256(
                learner_metrics) != metrics_identity[
                    "canonical_rows_sha256"] or len(learner_metrics) != 50:
        raise ValueError("C10 endpoint metrics bytes/rows mismatch")

    # The endpoint JSON is authored by the training process, so a signature
    # and matching raw-file hashes alone are not an independent proof.  Replay
    # the portable 50x64 evidence again at publication/recovery time and bind
    # both its survivor arithmetic and rolling health to the signed endpoint.
    from tmx_gpu import pipeline_gpu

    try:
        portable_replay = pipeline_gpu._replay_c10_survivor_contract(
            trajectory_rows, learner_metrics)
    except RuntimeError as exc:
        raise ValueError(
            "C10 portable endpoint evidence does not replay") from exc
    portable_health = portable_replay.pop("rolling_health")
    if portable_replay != endpoint.get("survivor_replay") or \
            portable_health != endpoint.get("rolling_health"):
        raise ValueError(
            "C10 portable endpoint replay differs from signed endpoint")

    gate_dir = root / "round_01" / "c10_behavior_gates" / "scout_step50"
    expected_gate_path = gate_dir / "evidence.json"
    if require_gate_stage and not portable_model and os.path.realpath(str(
            gate_stage.get("evidence_path", ""))) != os.path.realpath(str(
                expected_gate_path)):
        raise ValueError("C10 state behavior-gate path mismatch")
    gate = validate_behavior_gate_bundle(
        gate_dir,
        expected_endpoint_replay_sha256=endpoint_sha,
        expected_model_manifest=manifest,
        expected_code_artifact=code,
        require_passed=require_passed,
    )
    if require_gate_stage and (
            gate_stage.get("evidence_sha256") != gate["evidence_sha256"] or
            gate_stage.get("gate_outcome") != gate["gate_outcome"] or
            gate_stage.get("passed") is not gate["passed"]):
        raise ValueError("C10 state behavior-gate verdict binding mismatch")
    return {
        "root": str(root),
        "state": state,
        "execution_contract": execution,
        "endpoint_replay": endpoint,
        "behavior_gate": gate,
        "trainer_state": trainer,
        "final_model": str(final_model),
    }


def validate_terminal_behavior_pass(
    out_dir: str | Path,
) -> Dict[str, Any]:
    """Validate an observed passing gate even if final summary upload failed.

    The passing fixed-panel draw is terminal and may never be replaced.  This
    validator intentionally does not authorize model publication; that still
    requires :func:`validate_successful_canary_publication` and its complete
    run summary.
    """
    return _validate_terminal_bundle(out_dir, require_passed=True)


def validate_behavior_transport_bundle(
    out_dir: str | Path, *, portable_model: bool = False,
) -> Dict[str, Any]:
    """Validate the raw fixed-panel bundle for a rerunnable transport fault."""
    result = _validate_terminal_bundle(
        out_dir, require_passed=False, require_gate_stage=False,
        portable_model=portable_model)
    if result["behavior_gate"].get("gate_outcome") != "transport_invalid":
        raise ValueError("C10 unstaged behavior outcome is not transport-invalid")
    root = Path(result["root"])
    if (root / "run_summary.json").exists():
        raise ValueError("C10 behavior transport cannot publish a summary")
    return result


def validate_unstaged_behavior_bundle(
    out_dir: str | Path,
) -> Dict[str, Any]:
    """Replay a gate written before its atomic pipeline-stage commit."""
    result = _validate_terminal_bundle(
        out_dir, require_passed=None, require_gate_stage=False)
    root = Path(result["root"])
    if (root / "run_summary.json").exists():
        raise ValueError("unstaged C10 behavior bundle cannot have a summary")
    return result


def validate_successful_canary_publication(
    out_dir: str | Path,
) -> Dict[str, Any]:
    """Authorize a model artifact only after the complete C10 proof chain."""
    result = _validate_terminal_bundle(out_dir, require_passed=True)
    root = Path(result["root"])
    summary = _load_json_object(root / "run_summary.json", "C10 summary")
    rounds = summary.get("rounds")
    if not isinstance(rounds, list) or len(rounds) != 1 or not isinstance(
            rounds[0], dict):
        raise ValueError("C10 successful summary lacks exactly one round")
    round_summary = rounds[0]
    endpoint = result["endpoint_replay"]
    gate = result["behavior_gate"]
    if Path(str(summary.get("final_model", ""))).resolve() != Path(
            result["final_model"]).resolve() or \
            Path(str(round_summary.get("central_final", ""))).resolve() != \
            Path(result["final_model"]).resolve() or \
            round_summary.get("c10_canary_endpoint_sha256") != \
            endpoint["evidence_sha256"] or \
            round_summary.get("c10_behavior_gate_evidence_sha256") != \
            gate["evidence_sha256"] or \
            round_summary.get("c10_behavior_gate_outcome") != "passed" or \
            round_summary.get("c10_behavior_gate_passed") is not True:
        raise ValueError("C10 successful run summary is not proof-bound")
    result["run_summary"] = summary
    return result


def validate_terminal_semantic_failure(
    out_dir: str | Path, *, portable_model: bool = False,
) -> Dict[str, Any]:
    """Validate a durable, non-rerunnable negative scientific outcome."""
    result = _validate_terminal_bundle(
        out_dir, require_passed=False, portable_model=portable_model)
    if result["behavior_gate"].get("gate_outcome") != "semantic_failed":
        raise ValueError("C10 negative terminal outcome is not semantic")
    if (Path(result["root"]) / "run_summary.json").exists():
        raise ValueError("terminal C10 semantic failure must not publish a model")
    return result


def validate_terminal_training_failure(
    out_dir: str | Path, *, portable: bool = False,
) -> Dict[str, Any]:
    """Validate a pre-gate negative result, locally or after artifact move."""
    root = Path(out_dir).resolve()
    _reject_mutually_exclusive_outcomes(root, allowed="training_terminal")
    state = _load_json_object(root / "pipeline_state.json", "C10 state")
    if state.get("stages") != {}:
        raise ValueError(
            "pre-gate C10 training failure cannot contain completed stages")
    execution = validate_execution_contract(
        state.get("c10_execution_contract"),
        expected_output_dir=None if portable else str(root))
    failure_path = root / "round_01" / "scout" / \
        "c10_training_terminal_outcome.json"
    failure = validate_training_failure(
        _load_json_object(failure_path, "C10 training terminal outcome"),
        expected_runtime_contract=execution["scout_runtime_contract"],
        expected_code_artifact=execution["code_artifact"],
    )
    trajectory_path = root / "round_01" / "scout" / \
        "explorer_trajectories.jsonl"
    metrics_path = root / "metrics.jsonl"
    trajectory_rows = _load_jsonl_objects(
        trajectory_path, "C10 partial trajectories")
    all_metrics = _load_jsonl_objects(metrics_path, "C10 partial metrics")
    learner_metrics = [
        row for row in all_metrics
        if row.get("phase") == common.PARITY_PHASES["scout"]
        and row.get("round") == 1
        and type(row.get("grpo/step")) is int
        and "train/grpo_loss" in row
        and int(row["grpo/step"]) <= failure["completed_step"]
    ]
    if len(trajectory_rows) != failure["completed_step"] * 64 or len(
            learner_metrics) != failure["completed_step"]:
        raise ValueError("C10 training-failure completed geometry mismatch")
    for path, rows, section_name in (
            (trajectory_path, trajectory_rows, "trajectory"),
            (metrics_path, learner_metrics, "metrics")):
        section = failure[section_name]
        if common.file_sha256(str(path)) != section["file_sha256"] or \
                common.canonical_json_sha256(rows) != section[
                    "canonical_rows_sha256"] or len(rows) != section[
                        "row_count"]:
            raise ValueError(
                f"C10 training-failure {section_name} bytes/rows mismatch")
    if failure["completed_step"]:
        from tmx_gpu import pipeline_gpu

        try:
            if failure["failed_step"] == failure["completed_step"] and \
                    failure["failure_kind"] in \
                    pipeline_gpu._C10_STEP_HEALTH_FAILURE_KINDS:
                pipeline_gpu._replay_c10_terminal_health_failure(
                    trajectory_rows, learner_metrics,
                    completed_step=failure["completed_step"],
                    failure_kind=failure["failure_kind"],
                    observed=failure["failure_details"][
                        "health_gate"]["observed"])
            else:
                replay = pipeline_gpu._replay_c10_survivor_contract(
                    trajectory_rows, learner_metrics,
                    expected_completed_steps=failure["completed_step"])
                if failure["failure_kind"] == \
                        "rolling_behavior_collapse" and replay[
                            "rolling_health"] != failure[
                                "failure_details"]["rolling_health"]:
                    raise RuntimeError(
                        "C10 rolling-collapse health does not replay")
        except RuntimeError as exc:
            raise ValueError(
                "C10 terminal failure completed-step prefix does not replay") \
                from exc
    if failure["failure_kind"] == "dynamic_sampling_exhaustion":
        candidate_path = root / "round_01" / "scout" / \
            "c10_failed_step_candidates.jsonl"
        candidates = _load_jsonl_objects(
            candidate_path, "C10 failed-step candidates")
        identity = failure["failed_step_candidates"]
        if common.file_sha256(str(candidate_path)) != identity[
                "file_sha256"] or common.canonical_json_sha256(
                    candidates) != identity["canonical_rows_sha256"] or len(
                        candidates) != identity["row_count"]:
            raise ValueError("C10 failed-step candidate bytes/rows mismatch")
        _replay_failed_dynamic_candidates(
            candidates, failure["failure_details"])
        _validate_dynamic_failure_journal(
            root, evidence=failure, candidates=candidates,
            transport=False)
    if failure["failed_step"] == failure["completed_step"] + 1 and \
            failure["failure_kind"] in {
                "selected_group_geometry_failure",
                "survivor_filter_contract_failure"}:
        selected_path = root / "round_01" / "scout" / \
            "c10_failed_selected_batch.jsonl"
        manifest_path = root / "round_01" / "scout" / \
            "c10_failed_selected_batch_manifest.json"
        selected_rows = _load_jsonl_objects(
            selected_path, "C10 failed selected batch")
        identity = failure["failed_selected_batch"]
        if common.file_sha256(str(selected_path)) != identity[
                "file_sha256"] or common.canonical_json_sha256(
                    selected_rows) != identity[
                        "canonical_rows_sha256"] or len(selected_rows) != \
                identity["row_count"]:
            raise ValueError(
                "C10 failed selected-batch bytes/rows mismatch")
        manifest = _load_json_object(
            manifest_path, "C10 failed selected-batch manifest")
        if common.file_sha256(str(manifest_path)) != identity[
                "manifest_file_sha256"] or manifest.get(
                    "evidence_sha256") != identity[
                        "manifest_evidence_sha256"]:
            raise ValueError("C10 failed selected manifest bytes mismatch")
        _replay_failed_selected_batch(selected_rows, failure, manifest)
    if (root / "run_summary.json").exists():
        raise ValueError("terminal C10 training failure cannot publish a model")
    return {
        "root": str(root),
        "state": state,
        "execution_contract": execution,
        "training_failure": failure,
    }


def validate_training_transport_bundle(
    out_dir: str | Path, *, portable: bool = False,
) -> Dict[str, Any]:
    """Validate a nonterminal training transport attempt before rerun."""
    root = Path(out_dir).resolve()
    _reject_mutually_exclusive_outcomes(root, allowed="training_transport")
    state = _load_json_object(root / "pipeline_state.json", "C10 state")
    if state.get("stages") != {}:
        raise ValueError("C10 training transport cannot complete a stage")
    execution = validate_execution_contract(
        state.get("c10_execution_contract"),
        expected_output_dir=None if portable else str(root))
    evidence_path = root / "round_01" / "scout" / \
        "c10_training_transport_invalid.json"
    evidence = validate_training_transport_invalid(
        _load_json_object(evidence_path, "C10 training transport"),
        expected_runtime_contract=execution["scout_runtime_contract"],
        expected_code_artifact=execution["code_artifact"],
    )
    trajectory_path = root / "round_01" / "scout" / \
        "explorer_trajectories.jsonl"
    metrics_path = root / "metrics.jsonl"
    trajectory_rows = _load_jsonl_objects(
        trajectory_path, "C10 partial trajectories")
    all_metrics = _load_jsonl_objects(metrics_path, "C10 partial metrics")
    learner_metrics = [
        row for row in all_metrics
        if row.get("phase") == common.PARITY_PHASES["scout"]
        and row.get("round") == 1
        and type(row.get("grpo/step")) is int
        and "train/grpo_loss" in row
        and int(row["grpo/step"]) <= evidence["completed_step"]
    ]
    if len(trajectory_rows) != evidence["completed_step"] * 64 or len(
            learner_metrics) != evidence["completed_step"]:
        raise ValueError("C10 transport completed geometry mismatch")
    for path, rows, section_name in (
            (trajectory_path, trajectory_rows, "trajectory"),
            (metrics_path, learner_metrics, "metrics")):
        section = evidence[section_name]
        if common.file_sha256(str(path)) != section["file_sha256"] or \
                common.canonical_json_sha256(rows) != section[
                    "canonical_rows_sha256"] or len(rows) != section[
                        "row_count"]:
            raise ValueError(
                f"C10 training-transport {section_name} bytes/rows mismatch")
    if evidence["completed_step"]:
        from tmx_gpu import pipeline_gpu

        try:
            pipeline_gpu._replay_c10_survivor_contract(
                trajectory_rows, learner_metrics,
                expected_completed_steps=evidence["completed_step"],
                transport_invalid_step=(
                    evidence["completed_step"]
                    if evidence["failure_kind"] ==
                    "completed_step_transport_invalid" else None),
            )
        except RuntimeError as exc:
            raise ValueError(
                "C10 training-transport completed-step prefix does not "
                "replay") from exc
    if evidence["failure_kind"] == "dynamic_sampling_transport_invalid":
        candidate_path = root / "round_01" / "scout" / \
            "c10_failed_step_candidates.jsonl"
        candidates = _load_jsonl_objects(
            candidate_path, "C10 transport candidates")
        identity = evidence["failed_step_candidates"]
        if common.file_sha256(str(candidate_path)) != identity[
                "file_sha256"] or common.canonical_json_sha256(
                    candidates) != identity["canonical_rows_sha256"] or len(
                        candidates) != identity["row_count"]:
            raise ValueError("C10 transport candidate bytes/rows mismatch")
        _replay_failed_dynamic_candidates(
            candidates, evidence["failure_details"])
        _validate_dynamic_failure_journal(
            root, evidence=evidence, candidates=candidates,
            transport=True)
    return {
        "root": str(root),
        "state": state,
        "execution_contract": execution,
        "training_transport": evidence,
    }
