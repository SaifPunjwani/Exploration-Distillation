#!/usr/bin/env python3
"""Raw-evidence, correctness-blind selection of an SFT exposure checkpoint.

Exactly four preregistered optimizer-step checkpoints are eligible.  For each
candidate this module reads ``rollouts.jsonl`` and re-derives health from raw
completion text, finish reason, completion-token count, and error fields.  It
never trusts serialized rates or pass flags.  The selected checkpoint is the
latest passing candidate; correctness is diagnostic-only and is not read by
the selection rule.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

from tmx_gpu import common, fresh_c8_sft, sft_gate_gpu, sft_gpu

SELECTION_CONTRACT_VERSION = "sft_latest_health_passing_raw_v2"
PREREGISTERED_STEPS = (16, 32, 64, 126)
DIAGNOSTIC_PROFILE = "c7_reuse_ga8"
SELECTION_PROFILES = (DIAGNOSTIC_PROFILE, fresh_c8_sft.PROFILE)
DIAGNOSTIC_MODEL_INIT = "Qwen/Qwen3-1.7B"
DIAGNOSTIC_SFT_RUNTIME_CONTRACT = {
    "version": sft_gpu.SFT_RUNTIME_CONTRACT_VERSION,
    "config_tier": "gpu-max:grad_accum",
    "model_init": DIAGNOSTIC_MODEL_INIT,
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
    "grad_accum": 8,
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
DIAGNOSTIC_CONTRACT = {
    "model_init": DIAGNOSTIC_MODEL_INIT,
    "optimizer": "adamw",
    "learning_rate": 5e-6,
    "epochs_planned": 2,
    "grad_accum": 8,
    "accepted_examples": 500,
    "trainable_examples": 500,
    "planned_optimizer_steps": 126,
    "optimizer_steps_executed": 126,
    "micro_steps_executed": 1000,
    "training_completed": True,
    "termination_mode": "append_eos",
    "sft_contract_version": "c8-eos-aware",
    "requested_save_optimizer_steps": list(PREREGISTERED_STEPS),
    "saved_optimizer_steps": list(PREREGISTERED_STEPS),
    "sft_runtime_contract": DIAGNOSTIC_SFT_RUNTIME_CONTRACT,
}
FROZEN_SAMPLING = {
    "paired_seeds_across_prompt_modes": True,
    "generation_seed": 20260713,
    "num_prompts": 32,
    "samples_per_prompt": 2,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "max_completion_tokens": 32768,
}
FROZEN_THRESHOLDS = {
    "min_termination_rate": sft_gate_gpu.DEFAULT_MIN_TERMINATION_RATE,
    "max_clipped_rate": sft_gate_gpu.DEFAULT_MAX_CLIPPED_RATE,
    "min_valid_boxed_answer_rate": sft_gate_gpu.DEFAULT_MIN_VALID_BOXED_RATE,
    "max_median_completion_tokens": (
        32768 * sft_gate_gpu.DEFAULT_MAX_MEDIAN_TOKEN_FRACTION
    ),
    # Calibrated to the frozen Base control before checkpoint inspection.
    "max_repetition_rate": 7 / 64,
    "max_unclosed_think_rate": sft_gate_gpu.DEFAULT_MAX_UNCLOSED_THINK_RATE,
    "max_generation_error_rate": sft_gate_gpu.DEFAULT_MAX_GENERATION_ERROR_RATE,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _profile_steps(selection_profile: str) -> tuple[int, ...]:
    if selection_profile == DIAGNOSTIC_PROFILE:
        return PREREGISTERED_STEPS
    if selection_profile == fresh_c8_sft.PROFILE:
        return fresh_c8_sft.PREREGISTERED_STEPS
    raise ValueError(f"unknown SFT selection profile: {selection_profile!r}")


def _profile_contract(selection_profile: str) -> dict[str, Any]:
    if selection_profile == DIAGNOSTIC_PROFILE:
        return DIAGNOSTIC_CONTRACT
    if selection_profile == fresh_c8_sft.PROFILE:
        return fresh_c8_sft.CHECKPOINT_CONTRACT
    raise ValueError(f"unknown SFT selection profile: {selection_profile!r}")


def _parse_step_map(
    values: Sequence[str], flag: str, *, expected_steps: tuple[int, ...],
) -> dict[int, str]:
    parsed: dict[int, str] = {}
    for value in values:
        try:
            raw_step, item = value.split("=", 1)
            step = int(raw_step)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{flag} entries must be STEP=VALUE, got {value!r}") from exc
        _require(step not in parsed, f"duplicate {flag} step {step}")
        _require(bool(item), f"empty {flag} value for step {step}")
        parsed[step] = item
    _require(
        tuple(sorted(parsed)) == expected_steps,
        f"{flag} must contain exactly preregistered steps {expected_steps}; "
        f"got {tuple(sorted(parsed))}",
    )
    return parsed


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"{label} missing: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{label}:{line_number} is not a JSON object")
        rows.append(row)
    if not rows:
        raise ValueError(f"{label} contains no rows: {path}")
    return rows


def _exact_source(label: str, step: int) -> dict[str, str]:
    marker = "#digest="
    _require(marker in label, f"step {step}: gate source lacks exact artifact digest")
    ref, digest = label.rsplit(marker, 1)
    _require(bool(ref) and bool(digest), f"step {step}: empty gate ref/digest")
    return {"artifact_ref": ref, "artifact_digest": digest}


def _validate_checkpoint_contract(
    metadata: dict[str, Any], step: int,
    selection_profile: str = DIAGNOSTIC_PROFILE,
) -> None:
    _require(metadata.get("checkpoint_kind") == "post_optimizer_step_hf_model",
             f"step {step}: not a post-optimizer SFT checkpoint")
    _require(int(metadata.get("optimizer_step", -1)) == step,
             f"step {step}: checkpoint optimizer_step mismatch")
    for key, expected in _profile_contract(selection_profile).items():
        actual = metadata.get(key)
        if isinstance(expected, float):
            ok = isinstance(actual, (float, int)) and math.isclose(
                float(actual), expected, rel_tol=0.0, abs_tol=1e-15)
        else:
            ok = actual == expected
        contract_label = (
            "diagnostic contract" if selection_profile == DIAGNOSTIC_PROFILE
            else f"{selection_profile} contract")
        _require(ok, f"step {step}: {contract_label} {key}={actual!r}, "
                     f"expected {expected!r}")
    completed = metadata.get("completed_run_evidence")
    if selection_profile == fresh_c8_sft.PROFILE:
        training = fresh_c8_sft.validate_training_evidence(
            metadata.get("fresh_c8_training_evidence"))
        fresh_c8_sft.validate_completed_run_evidence(
            completed,
            expected_training_evidence_sha256=training["evidence_sha256"])


def _normalized_model_manifest(value: Any, model_source: dict[str, str], step: int) \
        -> dict[str, Any]:
    _require(isinstance(value, dict), f"step {step}: model file manifest missing")
    _require(value.get("version") == common.ARTIFACT_DOWNLOAD_MANIFEST_VERSION,
             f"step {step}: model file manifest version mismatch")
    _require(value.get("source") == model_source,
             f"step {step}: model file manifest source mismatch")
    files = value.get("files")
    _require(isinstance(files, list) and bool(files),
             f"step {step}: model file manifest has no files")
    names = {str(item.get("relative_path", "")) for item in files if isinstance(item, dict)}
    _require(any(name.endswith("config.json") for name in names),
             f"step {step}: model artifact lacks config.json")
    _require(any(name.endswith("sft_checkpoint.json") for name in names),
             f"step {step}: model artifact lacks sft_checkpoint.json")
    _require(any(name.endswith((".safetensors", ".bin", ".pt")) for name in names),
             f"step {step}: model artifact lacks weights")
    normalized = {
        "version": value["version"],
        "source": model_source,
        "files": sorted(files, key=lambda item: str(item.get("relative_path", ""))),
    }
    _require(value.get("file_manifest_sha256") == common.canonical_json_sha256(normalized),
             f"step {step}: model file-manifest SHA mismatch")
    return {**normalized, "file_manifest_sha256": value["file_manifest_sha256"]}


def _validate_accepted_sources(value: Any, step: int) -> dict[str, Any]:
    _require(isinstance(value, dict),
             f"step {step}: accepted-library artifact provenance missing")
    _require(value.get("version") == "wandb_accepted_sources_v2",
             f"step {step}: accepted-library provenance version mismatch")
    sources = value.get("sources")
    _require(isinstance(sources, list) and bool(sources),
             f"step {step}: accepted-library artifact list is empty")
    for source in sources:
        _require(isinstance(source, dict) and source.get("artifact_ref")
                 and source.get("artifact_digest") and source.get("file_manifest_sha256"),
                 f"step {step}: accepted artifact lacks exact membership evidence")
    normalized = {"version": value["version"], "sources": sources}
    digest = common.canonical_json_sha256(normalized)
    _require(value.get("sha256") == digest,
             f"step {step}: accepted artifact provenance digest mismatch")
    return value


def _normalized_artifact_identity(value: Any, field: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{field} is missing")
    _require(value.get("version") == common.ARTIFACT_DOWNLOAD_MANIFEST_VERSION,
             f"{field} version mismatch")
    source = value.get("source")
    _require(isinstance(source, dict) and source.get("artifact_ref")
             and source.get("artifact_digest"), f"{field} source missing")
    files = value.get("files")
    _require(isinstance(files, list) and bool(files), f"{field} files missing")
    normalized = {
        "version": value["version"],
        "source": source,
        "files": sorted(files, key=lambda item: str(item.get("relative_path", ""))),
    }
    _require(value.get("file_manifest_sha256")
             == common.canonical_json_sha256(normalized),
             f"{field} file-manifest SHA mismatch")
    return {**normalized, "file_manifest_sha256": value["file_manifest_sha256"]}


def _accepted_identity_from_download_manifest(
    manifest_path: str, accepted_jsonl: str,
) -> dict[str, Any]:
    manifest = _read_json_object(
        Path(manifest_path), "accepted artifact download manifest")
    validated = common.validate_artifact_download_manifest(manifest)
    root = Path(validated["download_root"]).resolve()
    target = Path(accepted_jsonl).expanduser().resolve()
    try:
        relative = target.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(
            "--accepted-jsonl is outside its artifact download root") from exc
    members = {str(item["relative_path"]) for item in validated["files"]}
    _require(relative in members,
             "--accepted-jsonl is not an attributed artifact member")
    return {
        key: validated[key]
        for key in ("version", "source", "files", "file_manifest_sha256")
    }


def _fresh_accepted_input(
    accepted_jsonl: str, accepted_artifact_identity: Any,
    training_evidence: dict[str, Any], accepted_sources: dict[str, Any], step: int,
) -> tuple[sft_gate_gpu.AcceptedExclusions, dict[str, Any]]:
    """Re-read the exact artifact-bound accepted bytes for disjointness."""
    path = Path(accepted_jsonl).expanduser().resolve()
    _require(path.is_file() and not path.is_symlink(),
             f"step {step}: exact accepted JSONL is missing or unsafe")
    identity = _normalized_artifact_identity(
        accepted_artifact_identity, f"step {step}: accepted artifact identity")
    byte_sha = common.file_sha256(str(path))
    semantic = common.semantic_sft_jsonl_digest(str(path))
    expected = training_evidence["accepted_library"]
    _require(byte_sha == expected["file_sha256"],
             f"step {step}: accepted JSONL bytes differ from fresh-c8 training")
    _require(semantic == expected["semantic_digest"],
             f"step {step}: accepted JSONL semantics differ from fresh-c8 training")
    members = [
        item for item in identity["files"]
        if str(item.get("sha256", "")) == byte_sha
        and str(item.get("relative_path", "")).endswith("accepted.jsonl")
    ]
    _require(len(members) == 1,
             f"step {step}: accepted JSONL is not uniquely artifact-bound")
    source_matches = [
        source for source in accepted_sources["sources"]
        if source.get("artifact_ref") == identity["source"]["artifact_ref"]
        and source.get("artifact_digest") == identity["source"]["artifact_digest"]
        and source.get("file_manifest_sha256") == identity["file_manifest_sha256"]
    ]
    _require(bool(source_matches),
             f"step {step}: accepted artifact differs from gate provenance")
    exclusions = sft_gate_gpu.load_accepted_exclusions([str(path)])
    _require(exclusions.row_count == 500,
             f"step {step}: fresh-c8 accepted JSONL must contain 500 rows")
    return exclusions, identity


def _revalidate_heldout_selection(
    heldout_rows: list[dict[str, Any]], *, exclusions: sft_gate_gpu.AcceptedExclusions,
    selection_seed: int, step: int,
) -> tuple[list[str], str, dict[str, Any]]:
    """Recompute identities and reject either form of train/gate overlap."""
    ordered_ids: list[str] = []
    payload: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in heldout_rows:
        problem_id = str(row.get("problem_id", "") or "").strip()
        problem = str(row.get("problem", "") or "")
        training_prompt = str(row.get("training_prompt", "") or "")
        _require(bool(problem_id and problem and training_prompt),
                 f"step {step}: heldout row lacks problem/prompt identity fields")
        _require(problem_id not in seen,
                 f"step {step}: heldout selection repeats problem_id {problem_id!r}")
        seen.add(problem_id)
        prompt_sha = sft_gate_gpu._sha256(training_prompt)
        selection_digest = sft_gate_gpu._sha256(
            f"{int(selection_seed)}\0{problem_id}\0{problem}")
        _require(row.get("training_prompt_sha256") == prompt_sha,
                 f"step {step}: persisted training-prompt hash mismatch")
        _require(row.get("selection_digest") == selection_digest,
                 f"step {step}: persisted seed-selection digest mismatch")
        _require(problem_id not in exclusions.problem_ids,
                 f"step {step}: heldout problem-ID overlaps accepted training data")
        _require(prompt_sha not in exclusions.prompt_hashes,
                 f"step {step}: heldout training-prompt hash overlaps accepted data")
        ordered_ids.append(problem_id)
        payload.append({
            "problem_id": problem_id,
            "selection_digest": selection_digest,
            "training_prompt_sha256": prompt_sha,
        })
    proof = {
        "accepted_row_count": exclusions.row_count,
        "accepted_problem_id_count": len(exclusions.problem_ids),
        "accepted_prompt_hash_count": len(exclusions.prompt_hashes),
        "heldout_problem_ids": ordered_ids,
        "heldout_training_prompt_hashes": [
            row["training_prompt_sha256"] for row in payload],
    }
    proof["sha256"] = common.canonical_json_sha256(proof)
    return ordered_ids, common.canonical_json_sha256(payload), proof


def validate_gate_bundle(
    bundle_dir: str, step: int, gate_source: str,
    selection_profile: str = DIAGNOSTIC_PROFILE,
    *, accepted_jsonl: str = "", accepted_artifact_identity: Any = None,
) -> dict[str, Any]:
    """Recompute one candidate verdict from its raw on-disk gate bundle."""
    root = Path(bundle_dir).expanduser().resolve()
    _require(root.is_dir(), f"step {step}: gate bundle is not a directory: {root}")
    aggregate_path = root / "aggregate_metrics.json"
    rollouts_path = root / "rollouts.jsonl"
    selection_path = root / "heldout_selection.jsonl"
    aggregate = _read_json_object(aggregate_path, f"step {step} aggregate")
    raw_rollouts = _read_jsonl(rollouts_path, f"step {step} raw rollouts")
    heldout_rows = _read_jsonl(selection_path, f"step {step} heldout selection")

    version = sft_gate_gpu.METRIC_CONTRACT_VERSION
    _require(aggregate.get("metric_contract_version") == version,
             f"step {step}: wrong metric contract")
    gate_source_identity = _exact_source(gate_source, step)
    checkpoint = aggregate.get("model_checkpoint", {})
    _require(checkpoint.get("metadata_file") == "sft_checkpoint.json",
             f"step {step}: gate did not evaluate an optimizer-step checkpoint")
    metadata = checkpoint.get("metadata", {})
    _require(isinstance(metadata, dict), f"step {step}: checkpoint metadata missing")
    _require(step in _profile_steps(selection_profile),
             f"step {step}: not preregistered for {selection_profile}")
    _validate_checkpoint_contract(metadata, step, selection_profile)

    accepted_digest = metadata.get("accepted_library_digest")
    _require(isinstance(accepted_digest, dict) and accepted_digest.get("sha256"),
             f"step {step}: accepted-library semantic digest missing")
    _require(aggregate.get("accepted_library_digest") == accepted_digest,
             f"step {step}: gate accepted library differs from checkpoint")
    accepted_sources = _validate_accepted_sources(
        aggregate.get("accepted_source_manifest"), step)
    training_evidence = None
    if selection_profile == fresh_c8_sft.PROFILE:
        training_evidence = fresh_c8_sft.validate_training_evidence(
            metadata.get("fresh_c8_training_evidence"))
        _require(accepted_digest == training_evidence["accepted_library"]["semantic_digest"],
                 f"step {step}: accepted library differs from fresh-c8 training evidence")
        trajectory_digest = {
            "version": "fresh_c8_training_trajectories_v1",
            "row_count": training_evidence["trajectories"]["row_count"],
            "sha256": training_evidence["trajectories"]["file_sha256"],
        }
        exclusions, accepted_identity = _fresh_accepted_input(
            accepted_jsonl, accepted_artifact_identity, training_evidence,
            accepted_sources, step)
    else:
        trajectory_source = metadata.get("trajectory_source")
        _require(isinstance(trajectory_source, dict)
                 and trajectory_source.get("digest", {}).get("sha256"),
                 f"step {step}: trajectory-source provenance missing")
        trajectory_digest = trajectory_source["digest"]

    model_source = aggregate.get("model_source")
    _require(isinstance(model_source, dict) and model_source.get("artifact_ref")
             and model_source.get("artifact_digest"),
             f"step {step}: exact model source missing")
    model_file_manifest = _normalized_model_manifest(
        aggregate.get("model_file_manifest"), model_source, step)

    selection = aggregate.get("selection", {})
    _require(selection.get("uses_model_outputs") is False,
             f"step {step}: held-out selection used model outputs")
    _require(selection.get("uses_correctness") is False,
             f"step {step}: held-out selection used correctness")
    _require(int(selection.get("selection_seed", -1)) == 20260713,
             f"step {step}: wrong selection seed")
    if selection_profile == fresh_c8_sft.PROFILE:
        ordered_ids, heldout_digest, disjointness_proof = \
            _revalidate_heldout_selection(
                heldout_rows, exclusions=exclusions,
                selection_seed=int(selection["selection_seed"]), step=step)
    else:
        ordered_ids = [str(row.get("problem_id", "")) for row in heldout_rows]
        heldout_payload = []
        for row in heldout_rows:
            _require(row.get("problem_id") and row.get("selection_digest")
                     and row.get("training_prompt_sha256"),
                     f"step {step}: heldout selection row lacks identity digest")
            heldout_payload.append({
                "problem_id": str(row["problem_id"]),
                "selection_digest": str(row["selection_digest"]),
                "training_prompt_sha256": str(row["training_prompt_sha256"]),
            })
        heldout_digest = common.canonical_json_sha256(heldout_payload)
    _require(selection.get("heldout_selection_sha256") == heldout_digest,
             f"step {step}: heldout selection digest mismatch")

    sampling = aggregate.get("sampling", {})
    for key, expected in FROZEN_SAMPLING.items():
        _require(sampling.get(key) == expected,
                 f"step {step}: sampling {key}={sampling.get(key)!r}, "
                 f"expected {expected!r}")
    _require(set(aggregate.get("prompt_modes", {})) == set(sft_gate_gpu.PROMPT_MODES),
             f"step {step}: both frozen prompt modes are required")
    serialized_gate = aggregate.get("health_gate", {})
    _require(serialized_gate.get("selection_uses_correctness") is False,
             f"step {step}: serialized gate used correctness")
    _require(serialized_gate.get("thresholds") == FROZEN_THRESHOLDS,
             f"step {step}: thresholds differ from preregistration")
    recomputed_metrics, recomputed_gate = sft_gate_gpu.recompute_gate_from_rollouts(
        raw_rollouts,
        num_prompts=FROZEN_SAMPLING["num_prompts"],
        samples_per_prompt=FROZEN_SAMPLING["samples_per_prompt"],
        generation_seed=FROZEN_SAMPLING["generation_seed"],
        max_completion_tokens=FROZEN_SAMPLING["max_completion_tokens"],
        ordered_problem_ids=ordered_ids,
        thresholds=FROZEN_THRESHOLDS,
    )
    _require(aggregate.get("metrics_by_prompt_mode") == recomputed_metrics,
             f"step {step}: serialized metrics differ from raw-rollout recomputation")
    _require(serialized_gate == recomputed_gate,
             f"step {step}: serialized verdict differs from raw-rollout recomputation")

    candidate = {
        "optimizer_step": step,
        "passed": bool(recomputed_gate["passed"]),
        "model_source": model_source,
        "model_file_manifest": model_file_manifest,
        "accepted_library_digest": accepted_digest,
        "accepted_source_manifest": accepted_sources,
        "trajectory_source_digest": trajectory_digest,
        "heldout_selection_sha256": heldout_digest,
        "thresholds": FROZEN_THRESHOLDS,
        "sampling": dict(FROZEN_SAMPLING),
        "gate_source": gate_source,
        "gate_artifact_identity": gate_source_identity,
        "aggregate_sha256": common.file_sha256(str(aggregate_path)),
        "raw_rollouts_sha256": common.file_sha256(str(rollouts_path)),
        "heldout_selection_file_sha256": common.file_sha256(str(selection_path)),
    }
    if selection_profile == fresh_c8_sft.PROFILE:
        candidate["selection_profile"] = selection_profile
        candidate["fresh_c8_training_evidence"] = training_evidence
        candidate["accepted_artifact_identity"] = accepted_identity
        candidate["heldout_disjointness_proof"] = disjointness_proof
    return candidate


def build_selection_manifest_from_bundles(
    bundle_dirs: dict[int, str], gate_sources: dict[int, str],
    selection_profile: str = DIAGNOSTIC_PROFILE,
    *, accepted_jsonl: str = "", accepted_artifact_identity: Any = None,
) -> dict[str, Any]:
    preregistered_steps = _profile_steps(selection_profile)
    _require(tuple(sorted(bundle_dirs)) == preregistered_steps,
             "gate bundles must cover exactly the preregistered steps")
    _require(tuple(sorted(gate_sources)) == preregistered_steps,
             "gate sources must cover exactly the preregistered steps")
    candidates = [
        validate_gate_bundle(
            bundle_dirs[step], step, gate_sources[step], selection_profile,
            accepted_jsonl=accepted_jsonl,
            accepted_artifact_identity=accepted_artifact_identity)
        for step in preregistered_steps
    ]
    shared_keys = (
        "accepted_library_digest", "trajectory_source_digest",
        "accepted_source_manifest", "heldout_selection_sha256",
        "thresholds", "sampling",
    )
    if selection_profile == fresh_c8_sft.PROFILE:
        shared_keys += (
            "fresh_c8_training_evidence", "accepted_artifact_identity",
            "heldout_disjointness_proof",
        )
    for key in shared_keys:
        _require(all(item[key] == candidates[0][key] for item in candidates[1:]),
                 f"gate candidates disagree on {key}")
    passing = [item for item in candidates if item["passed"]]
    _require(bool(passing), "none of the preregistered SFT checkpoints passed")
    selected = max(passing, key=lambda item: item["optimizer_step"])
    manifest = {
        "selection_contract_version": SELECTION_CONTRACT_VERSION,
        "preregistered_optimizer_steps": list(preregistered_steps),
        "selection_rule": "latest_passing_raw_health_gate",
        "selection_uses_correctness": False,
        "diagnostic_contract": _profile_contract(selection_profile),
        "candidates": candidates,
        "selected_optimizer_step": selected["optimizer_step"],
        "selected_model_source": selected["model_source"],
        "selected_model_file_manifest": selected["model_file_manifest"],
        "selected_gate_source": selected["gate_source"],
        "accepted_library_digest": selected["accepted_library_digest"],
        "accepted_source_manifest": selected["accepted_source_manifest"],
        "trajectory_source_digest": selected["trajectory_source_digest"],
        "heldout_selection_sha256": selected["heldout_selection_sha256"],
    }
    if selection_profile == fresh_c8_sft.PROFILE:
        manifest["selection_profile"] = selection_profile
        manifest["fresh_c8_training_evidence"] = selected[
            "fresh_c8_training_evidence"]
        manifest["accepted_artifact_identity"] = selected[
            "accepted_artifact_identity"]
        manifest["heldout_disjointness_proof"] = selected[
            "heldout_disjointness_proof"]
    manifest["manifest_sha256"] = common.canonical_json_sha256(manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-bundle", action="append", required=True,
                        metavar="STEP=DIRECTORY")
    parser.add_argument("--gate-source", action="append", required=True,
                        metavar="STEP=ARTIFACT_REF#digest=SHA")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--selection-profile", default=DIAGNOSTIC_PROFILE,
        choices=list(SELECTION_PROFILES),
    )
    parser.add_argument(
        "--accepted-jsonl", default="",
        help="fresh-c8 only: exact artifact-bound accepted.jsonl used to "
             "independently revalidate held-out disjointness",
    )
    parser.add_argument(
        "--accepted-download-manifest", default="",
        help="fresh-c8 only: validated artifact download manifest containing "
             "--accepted-jsonl",
    )
    args = parser.parse_args(argv)
    expected_steps = _profile_steps(args.selection_profile)
    bundles = _parse_step_map(
        args.gate_bundle, "--gate-bundle", expected_steps=expected_steps)
    sources = _parse_step_map(
        args.gate_source, "--gate-source", expected_steps=expected_steps)
    accepted_identity = None
    if args.selection_profile == fresh_c8_sft.PROFILE:
        _require(bool(args.accepted_jsonl and args.accepted_download_manifest),
                 "fresh-c8 selection requires --accepted-jsonl and "
                 "--accepted-download-manifest")
        accepted_identity = _accepted_identity_from_download_manifest(
            args.accepted_download_manifest, args.accepted_jsonl)
    manifest = build_selection_manifest_from_bundles(
        bundles, sources, args.selection_profile,
        accepted_jsonl=args.accepted_jsonl,
        accepted_artifact_identity=accepted_identity)
    Path(args.output).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
