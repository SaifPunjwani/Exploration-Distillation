#!/usr/bin/env python3
"""Fail-closed boundaries for primary P1 and secondary Qwen-report P2 evals.

``eval_gpu.py`` produces one provenance-bound exact sample matrix per
benchmark.  This module is the *campaign* boundary: P1 accepts exactly the six
project benchmarks and emits the headline-eligible ``publication_matrix.json``;
P2 accepts exactly AIME24/AIME25 and emits a separately labelled, non-headline
Qwen-report anchor matrix.  Both revalidate frozen manifests and aggregate
accounting and require one shared model/code/runtime/verifier identity. Training
probes and partial runs satisfy neither contract.

The signature is a SHA-256 over the canonical JSON payload (everything except
the ``signature`` member).  It is not a public-key attestation; it makes the
assembled matrix and its source-file hashes tamper evident.  The launcher
adds the separate durable W&B artifact receipt after upload.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from tmx_gpu import common, dapo_evidence
from tmx_gpu.eval_gpu import (
    BENCHMARKS,
    CANONICAL_MODEL_SOURCE_NAMES,
    EVAL_PROTOCOL_VERSION,
    EXECUTING_CODE_FILES,
    MODEL_REVISIONS,
    PUBLICATION_EVAL_CONTAINER_IMAGE_IDENTITY,
    PUBLICATION_RUNTIME_STACK_VERSIONS,
    PUBLICATION_VLLM_VERSION,
    SAMPLE_SEED_RULE,
    atomic_write_json,
    avg_at,
    benchmark_sample_count,
    build_per_problem,
    canonical_rollout_row_identity,
    canonical_json_bytes,
    grade,
    load_benchmark,
    model_hash,
    render_prompt,
    validate_publication_evaluation_execution_environment,
    validate_loaded_benchmark,
)


PUBLICATION_BENCHMARKS = (
    "AIME24",
    "AIME25",
    "MATH500",
    "AMC23",
    "Minerva-Math",
    "GSM8K",
)
QWEN_REPORT_P2_BENCHMARKS = (
    "AIME24",
    "AIME25",
)
# The headline aggregate intentionally excludes the two near-ceiling
# diagnostics.  Keep this list explicit and signed into every matrix: silently
# averaging ``PUBLICATION_BENCHMARKS`` would change the paper's estimand.
HARD_MEAN_BENCHMARKS = (
    "AIME24",
    "AIME25",
    "MATH500",
    "Minerva-Math",
)
MATRIX_SCHEMA_VERSION = "tmx_gpu.publication_matrix_v2"
QWEN_REPORT_P2_MATRIX_SCHEMA_VERSION = "tmx_gpu.qwen_report_anchor_matrix_v1"
TRAINING_CAMPAIGN_BINDING_VERSION = (
    "tmx_gpu.training_campaign_publication_binding_v1")
EXPDIS_SINGLESCOUT_METHOD = "expdis_single_scout"
EXPDIS_SINGLESCOUT_MULTIROUND_METHOD = (
    "disjoint_shard_annealed_single_scout_multiround")
EXPDIS_SINGLESCOUT_MULTIROUND_METHOD_VERSION = (
    "gpu_disjoint_shard_annealed_single_scout_multiround_v1")
TRAINED_EXPDIS_METHODS = frozenset({
    EXPDIS_SINGLESCOUT_METHOD,
    EXPDIS_SINGLESCOUT_MULTIROUND_METHOD,
    "integrated_multiscout",
    "disjoint_shard_annealed_mrme",
})
DAPO_METHOD = dapo_evidence.DAPO_METHOD
_METRIC_KS = (1, 2, 4, 8, 16, 32, 64)
_IMMUTABLE_WANDB_ARTIFACT_REF = re.compile(
    r"^[^/\s:]+/[^/\s:]+/[^/\s:]+:v[0-9]+$")


class MatrixValidationError(ValueError):
    """A source result is not eligible for six-benchmark publication."""


def _fail(message: str) -> None:
    raise MatrixValidationError(message)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        _fail(f"source must be a regular non-symlink file: {path}")
    try:
        value = json.loads(path.read_text(), object_pairs_hook=_strict_object)
    except MatrixValidationError:
        raise
    except Exception as exc:  # noqa: BLE001
        _fail(f"cannot read {path}: {exc}")
    if not isinstance(value, dict):
        _fail(f"source is not a JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _signed_canonical_payload(
    version: str, payload: dict[str, Any],
) -> dict[str, Any]:
    value = {"schema_version": version, **payload}
    value["binding_sha256"] = _canonical_sha256(value)
    return value


def _verify_canonical_binding(
    value: dict[str, Any], *, expected_version: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail("training campaign binding must be a JSON object")
    unsigned = dict(value)
    recorded = unsigned.pop("binding_sha256", None)
    if unsigned.get("schema_version") != expected_version:
        _fail("training campaign binding schema mismatch")
    if recorded != _canonical_sha256(unsigned):
        _fail("training campaign binding SHA256 mismatch")
    return dict(value)


def is_trained_expdis_method(value: Any) -> bool:
    method = str(value or "")
    return method in TRAINED_EXPDIS_METHODS or "expdis" in method.lower()


def is_trained_publication_method(value: Any) -> bool:
    return is_trained_expdis_method(value) or str(value or "") == DAPO_METHOD


def _immutable_artifact_ref(value: Any, where: str) -> str:
    ref = _expect_nonempty_string(value, where)
    if _IMMUTABLE_WANDB_ARTIFACT_REF.fullmatch(ref) is None:
        _fail(f"{where} must use an immutable entity/project/name:vN ref")
    return ref


def infer_expdis_method_from_artifact_metadata(
    metadata: Any, *, where: str,
) -> str | None:
    """Derive an ExpDis method from immutable artifact metadata.

    A caller-supplied display label is never used here.  Partial ExpDis metadata
    is rejected instead of being treated as an ordinary Base checkpoint.
    """
    if not isinstance(metadata, dict):
        _fail(f"{where} metadata must be an object")
    method = metadata.get("method")
    if method == EXPDIS_SINGLESCOUT_MULTIROUND_METHOD:
        if metadata.get("method_version") != \
                EXPDIS_SINGLESCOUT_MULTIROUND_METHOD_VERSION:
            _fail(
                f"{where} single-Scout multi-round method-version mismatch")
        return str(method)
    if method in {"integrated_multiscout", "disjoint_shard_annealed_mrme"}:
        return str(method)
    if method is not None and (
            "multiscout" in str(method).lower()
            or "mrme" in str(method).lower()
            or "expdis" in str(method).lower()):
        _fail(f"{where} contains an unrecognized trained-ExpDis method {method!r}")

    c9_fields = {
        "expdis_stabilization_mode", "c9_execution_profile", "stabilization_cell",
        "scientific_variant", "c9_pipeline_execution_contract_sha256",
        "c9_central_lineage_sha256", "mrme_contract_sha256",
        "mrme_completion_evidence_sha256", "method_version",
    }
    present = c9_fields.intersection(metadata)
    if not present:
        return None
    if metadata.get("c9_execution_profile") == "full" and \
            metadata.get("stabilization_cell") == "C5" and \
            metadata.get("scientific_variant") == \
            "expdis_c9_selected_batch_rnd" and \
            isinstance(metadata.get("c9_pipeline_execution_contract_sha256"), str) and \
            isinstance(metadata.get("c9_central_lineage_sha256"), str):
        return EXPDIS_SINGLESCOUT_METHOD
    if method in {
            "integrated_multiscout", "disjoint_shard_annealed_mrme",
            EXPDIS_SINGLESCOUT_MULTIROUND_METHOD}:
        return str(method)
    _fail(
        f"{where} has partial/noncanonical trained-ExpDis metadata: "
        f"{sorted(present)}")


def _artifact_identity(
    *, ref: str, digest: str, artifact_type: str, where: str,
) -> dict[str, str]:
    return {
        "artifact_ref": _immutable_artifact_ref(ref, f"{where} artifact ref"),
        "artifact_digest": _expect_nonempty_string(
            digest, f"{where} artifact digest"),
        "artifact_type": _expect_nonempty_string(
            artifact_type, f"{where} artifact type"),
    }


def _expect_metadata_fields(
    metadata: dict[str, Any], expected: dict[str, Any], *, where: str,
) -> None:
    for key, value in expected.items():
        _expect_equal(metadata.get(key), value, f"{where} metadata.{key}")


def _code_source_identity(code: Any, *, where: str) -> dict[str, str]:
    if not isinstance(code, dict) or not isinstance(code.get("source"), dict):
        _fail(f"{where} lacks immutable code-artifact identity")
    source = code["source"]
    return {
        "artifact_ref": _immutable_artifact_ref(
            source.get("artifact_ref"), f"{where} code artifact ref"),
        "artifact_digest": _expect_nonempty_string(
            source.get("artifact_digest"), f"{where} code artifact digest"),
    }


def _recompute_c9_single_central_probe_facts(path: Path) -> dict[str, Any]:
    """Replay the exact four SingleScout Central probe records.

    The durable metrics file is authoritative; the pipeline-state summary is
    only accepted after this independent replay reproduces its canonical row
    digest.  The shared MR-ME helper owns the probe field/decode contract, and
    this wrapper additionally rejects out-of-schedule rows and checks the
    SingleScout C9 campaign identity on every record.
    """
    from tmx_gpu import c9_evidence

    if path.is_symlink() or not path.is_file():
        _fail("C9 campaign lacks a regular durable metrics.jsonl")
    expected_steps = [225, 250, 275, 300]
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line, object_pairs_hook=_strict_object)
                if not isinstance(value, dict):
                    _fail(f"C9 metrics line {line_number} is not an object")
                if value.get("phase") == "central_grpo" and \
                        "eval/probe_protocol" in value:
                    rows.append(value)
    except MatrixValidationError:
        raise
    except Exception as exc:  # noqa: BLE001
        _fail(f"cannot replay C9 durable metrics: {exc}")
    observed_steps = [row.get("global_step") for row in rows]
    if len(rows) != 4 or sorted(observed_steps) != expected_steps or \
            len(set(observed_steps)) != 4:
        _fail(
            "C9 Central metrics must contain exactly one probe at each of "
            f"{expected_steps}; observed={observed_steps}")
    expected_identity = {
        "phase": "central_grpo",
        "round": 1,
        "lambda": 0.0,
        "lambda_novelty": 0.0,
        "num_scouts": 1,
        "expdis_stabilization_mode": c9_evidence.C9_RND_LIFECYCLE,
        "scientific_variant": c9_evidence.C9_SCIENTIFIC_VARIANT,
        "c9_execution_profile": "full",
        "stabilization_cell": "C5",
        "novelty_normalization": c9_evidence.C9_NOVELTY_NORMALIZATION,
        "eval/probe_error_rows": 0.0,
    }
    for row in rows:
        drift = [
            key for key, expected in expected_identity.items()
            if row.get(key) != expected
        ]
        if drift:
            _fail(
                f"C9 Central probe {row.get('global_step')!r} identity/error "
                f"contract drift: {drift}")
    exact_rows: list[dict[str, Any]] = []
    for step in expected_steps:
        row = next(item for item in rows if item["global_step"] == step)
        exact = {
            "global_step": step,
            "phase": "central_grpo",
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
            "protocol": c9_evidence.C9_PROBE_PROTOCOL_ID,
            "headline_eligible": 0.0,
            "metric_scale": "fraction",
            "sample_seed_rule": c9_evidence.C9_PROBE_CONTRACT[
                "sample_seed_rule"],
            "benchmark_revision": c9_evidence.C9_PROBE_CONTRACT[
                "benchmark_revision"],
            "benchmark_rows_sha256": c9_evidence.C9_PROBE_CONTRACT[
                "benchmark_rows_sha256"],
            "protocol_complete": 1.0,
            "graded_rows": 120.0,
            "expected_rows": 120.0,
            "complete_problem_count": 30.0,
            "expected_problem_count": 30.0,
            "samples_per_problem": 4.0,
            "max_completion_tokens": 32768.0,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20.0,
            "min_p": 0.0,
            "seed_base": float(c9_evidence.C9_PROBE_CONTRACT["seed_base"]),
            "policy_step": step - 200,
            "staleness_updates": 0,
            "fresh_weights": 1.0,
        }
        drift = [key for key, value in expected.items()
                 if exact.get(key) != value]
        try:
            avg_at_4 = float(exact["avg_at_4"])
        except (TypeError, ValueError):
            avg_at_4 = math.nan
        prompt_sha = exact.get("prompt_matrix_sha256")
        if drift or not isinstance(prompt_sha, str) or len(prompt_sha) != 64 or \
                not math.isfinite(avg_at_4) or not 0.0 <= avg_at_4 <= 1.0:
            _fail(f"C9 Central probe replay failed at {step}: {drift}")
        exact_rows.append(exact)
    if len({row["prompt_matrix_sha256"] for row in exact_rows}) != 1:
        _fail("C9 Central probe prompt matrix changed across steps")
    return {
        "probe_protocol_id": c9_evidence.C9_PROBE_PROTOCOL_ID,
        "phase": "central_grpo",
        "expected_global_steps": expected_steps,
        "row_count": len(exact_rows),
        "prompt_matrix_sha256": exact_rows[0]["prompt_matrix_sha256"],
        "canonical_rows_sha256": _canonical_sha256(exact_rows),
    }


def _stream_canonical_json_array_digest(rows: list[dict[str, Any]]) -> str:
    """Return the canonical digest used by ``common.canonical_json_sha256``.

    This small helper is intentionally exposed for tests.  The production
    trajectory replay below performs the same framing incrementally so a
    32K-token Scout bank does not need a second serialized copy in memory.
    """
    digest = hashlib.sha256(b"[")
    for index, row in enumerate(rows):
        if index:
            digest.update(b",")
        digest.update(canonical_json_bytes(row))
    digest.update(b"]")
    return digest.hexdigest()


def _read_jsonl_strict(path: Path, *, where: str) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        _fail(f"{where} must be a regular non-symlink file")
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    _fail(f"{where} contains blank line {line_number}")
                value = json.loads(line, object_pairs_hook=_strict_object)
                if not isinstance(value, dict):
                    _fail(f"{where} line {line_number} is not an object")
                rows.append(value)
    except MatrixValidationError:
        raise
    except Exception as exc:  # noqa: BLE001
        _fail(f"cannot replay {where}: {exc}")
    return rows


def _require_original_path_suffix(value: Any, suffix: str, *, where: str) -> str:
    """Validate a producer-host path without dereferencing that stale host.

    Results artifacts preserve absolute paths from the training worker.  A
    publication worker must open the downloaded relative member, while still
    proving that the recorded stage graph named the canonical producer path.
    """
    path = _expect_nonempty_string(value, where).replace("\\", "/")
    normalized_suffix = suffix.strip("/")
    if not path.startswith("/") or "//" in path or "/../" in path or \
            not path.endswith("/" + normalized_suffix):
        _fail(f"{where} is not a canonical absolute *{normalized_suffix!r} path")
    return path


def _recompute_c9_single_scout_probe_facts(path: Path) -> dict[str, Any]:
    """Replay exactly one canonical Scout probe at steps 25..200."""
    from tmx_gpu import c9_evidence

    rows = _read_jsonl_strict(path, where="C9 durable metrics")
    probes = [
        row for row in rows
        if row.get("phase") == "scout_grpo" and
        "eval/probe_protocol" in row
    ]
    expected_steps = list(range(25, 201, 25))
    observed_steps = [row.get("global_step") for row in probes]
    if len(probes) != len(expected_steps) or \
            sorted(observed_steps) != expected_steps or \
            len(set(observed_steps)) != len(expected_steps):
        _fail(
            "C9 Scout metrics must contain exactly one probe at each of "
            f"{expected_steps}; observed={observed_steps}")
    identity = {
        "phase": "scout_grpo",
        "round": 1,
        "lambda": 0.5,
        "lambda_novelty": 0.5,
        "num_scouts": 1,
        "expdis_stabilization_mode": c9_evidence.C9_RND_LIFECYCLE,
        "scientific_variant": c9_evidence.C9_SCIENTIFIC_VARIANT,
        "c9_execution_profile": c9_evidence.C9_PROFILE,
        "stabilization_cell": c9_evidence.C9_CELL,
        "novelty_normalization": c9_evidence.C9_NOVELTY_NORMALIZATION,
        "eval/probe_error_rows": 0.0,
    }
    exact_rows: list[dict[str, Any]] = []
    for step in expected_steps:
        row = next(item for item in probes if item["global_step"] == step)
        drift = [key for key, expected in identity.items()
                 if row.get(key) != expected]
        exact = {
            "global_step": step,
            "phase": "scout_grpo",
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
            "protocol": c9_evidence.C9_PROBE_PROTOCOL_ID,
            "headline_eligible": 0.0,
            "metric_scale": "fraction",
            "sample_seed_rule": c9_evidence.C9_PROBE_CONTRACT[
                "sample_seed_rule"],
            "benchmark_revision": c9_evidence.C9_PROBE_CONTRACT[
                "benchmark_revision"],
            "benchmark_rows_sha256": c9_evidence.C9_PROBE_CONTRACT[
                "benchmark_rows_sha256"],
            "protocol_complete": 1.0,
            "graded_rows": 120.0,
            "expected_rows": 120.0,
            "complete_problem_count": 30.0,
            "expected_problem_count": 30.0,
            "samples_per_problem": 4.0,
            "max_completion_tokens": 32768.0,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20.0,
            "min_p": 0.0,
            "seed_base": float(c9_evidence.C9_PROBE_CONTRACT["seed_base"]),
            "policy_step": step,
            "staleness_updates": 0,
            "fresh_weights": 1.0,
        }
        drift.extend(key for key, value in expected.items()
                     if exact.get(key) != value)
        try:
            avg_at_4 = float(exact["avg_at_4"])
        except (TypeError, ValueError):
            avg_at_4 = math.nan
        prompt_sha = exact.get("prompt_matrix_sha256")
        if drift or not isinstance(prompt_sha, str) or len(prompt_sha) != 64 or \
                not math.isfinite(avg_at_4) or not 0.0 <= avg_at_4 <= 1.0:
            _fail(f"C9 Scout probe {step} is not exact/fresh: {sorted(set(drift))}")
        exact_rows.append(exact)
    if len({row["prompt_matrix_sha256"] for row in exact_rows}) != 1:
        _fail("C9 Scout probe prompt matrix changed across steps")
    return {
        "probe_protocol_id": c9_evidence.C9_PROBE_PROTOCOL_ID,
        "phase": "scout_grpo",
        "expected_global_steps": expected_steps,
        "row_count": len(exact_rows),
        "prompt_matrix_sha256": exact_rows[0]["prompt_matrix_sha256"],
        "canonical_rows_sha256": _canonical_sha256(exact_rows),
    }


def _replay_c9_coverage_pool(
    trajectory_path: Path,
    accepted_path: Path,
    *,
    original_trajectory_path: str,
    original_accepted_path: str,
) -> dict[str, Any]:
    """Replay the 12,800-row Scout bank and deterministic c8 filter."""
    from tmx_gpu import c9_evidence, common, filter_pool

    if trajectory_path.is_symlink() or not trajectory_path.is_file():
        _fail("C9 Scout trajectory bank is missing")
    canonical = hashlib.sha256(b"[")
    per_step: dict[int, int] = {}
    correct = 0
    clean = 0
    winners: dict[tuple[str, str], dict[str, Any]] = {}
    row_count = 0
    expected_identity = {
        "expdis_stabilization_mode": c9_evidence.C9_RND_LIFECYCLE,
        "scientific_variant": c9_evidence.C9_SCIENTIFIC_VARIANT,
        "c9_execution_profile": c9_evidence.C9_PROFILE,
        "stabilization_cell": c9_evidence.C9_CELL,
        "novelty_normalization": c9_evidence.C9_NOVELTY_NORMALIZATION,
        "rnd_update_scope": "final_selected_learner_rows",
    }
    try:
        with trajectory_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    _fail(f"C9 Scout trajectories contain blank line {line_number}")
                row = json.loads(line, object_pairs_hook=_strict_object)
                if not isinstance(row, dict):
                    _fail(f"C9 trajectory row {line_number} is not an object")
                if row_count:
                    canonical.update(b",")
                canonical.update(canonical_json_bytes(row))
                row_count += 1
                step = row.get("step")
                if type(step) is not int or not 1 <= step <= 200:
                    _fail(f"C9 trajectory row {line_number} has invalid step")
                per_step[step] = per_step.get(step, 0) + 1
                if any(row.get(key) != value
                       for key, value in expected_identity.items()):
                    _fail(f"C9 trajectory row {line_number} identity drift")
                for key in (
                    "novelty_reward", "blended_reward", "correctness_reward",
                    "novelty_normalization_center",
                    "novelty_normalization_scale",
                    "novelty_normalization_effective_scale",
                ):
                    try:
                        finite = math.isfinite(float(row.get(key, math.nan)))
                    except (TypeError, ValueError):
                        finite = False
                    if not finite:
                        _fail(f"C9 trajectory row {line_number} has nonfinite {key}")
                if not bool(row.get("is_correct", False)) and not math.isclose(
                        float(row["novelty_reward"]), 0.0, abs_tol=1e-7):
                    _fail("C9 incorrect trajectory received novelty credit")
                if bool(row.get("is_correct", False)):
                    correct += 1
                    candidate = dict(row)
                    candidate["source_file"] = original_trajectory_path
                    if filter_pool.passes_c8_contract_gates(candidate):
                        clean += 1
                        key = filter_pool.c8_problem_key(candidate)
                        prior = winners.get(key)
                        if prior is None or filter_pool.c8_within_problem_key(
                                candidate) < filter_pool.c8_within_problem_key(prior):
                            winners[key] = candidate
    except MatrixValidationError:
        raise
    except Exception as exc:  # noqa: BLE001
        _fail(f"cannot replay C9 Scout/filter: {exc}")
    canonical.update(b"]")
    if row_count != c9_evidence.SCOUT_TRAJECTORY_ROWS or \
            set(per_step) != set(range(1, 201)) or \
            set(per_step.values()) != {c9_evidence.SCOUT_ROWS_PER_STEP}:
        _fail("C9 Scout trajectories are not exactly 200 x 64")

    selected = filter_pool.select_coverage_pool_c8(
        list(winners.values()), max_examples=c9_evidence.ACCEPTED_ROWS,
        source_balance=False)
    predicted_rows = []
    for row in selected:
        out = {
            "prompt_text": row.get("prompt_text", ""),
            "completion_text": row.get("completion_text", ""),
            **{key: row[key] for key in filter_pool.METADATA_KEYS if key in row},
            "source_file": row.get("source_file", ""),
            "selection_policy": filter_pool.C8_POLICY,
            "selection_source_balance": False,
        }
        predicted_rows.append(out)
    accepted_rows = _read_jsonl_strict(
        accepted_path, where="C9 accepted SFT library")
    if accepted_rows != predicted_rows or len(accepted_rows) != \
            c9_evidence.ACCEPTED_ROWS:
        _fail("C9 accepted library differs from deterministic coverage_pool_c8 replay")
    funnel = {
        "policy": filter_pool.C8_POLICY,
        "num_sources": 1,
        "raw": row_count,
        "correct": correct,
        "clean_correct": clean,
        "accepted": len(selected),
        "max_examples": c9_evidence.ACCEPTED_ROWS,
        "max_completion_tokens": filter_pool.MAX_COMPLETION_TOKENS,
        "output": original_accepted_path,
        "unique_clean_problems": len(winners),
        "accepted_unique_problems": len(selected),
        "accepted_per_source": {original_trajectory_path: len(selected)},
        "source_balance": False,
    }
    return {
        "trajectory": {
            "row_count": row_count,
            "step_count": len(per_step),
            "rows_per_step": c9_evidence.SCOUT_ROWS_PER_STEP,
            "file_sha256": _file_sha256(trajectory_path),
            "canonical_rows_sha256": canonical.hexdigest(),
        },
        "accepted_rows": accepted_rows,
        "accepted_file_sha256": _file_sha256(accepted_path),
        "accepted_semantic_digest": common.semantic_sft_jsonl_digest(
            str(accepted_path)),
        "training_sequence_digest": common.sft_training_sequence_digest(
            accepted_rows, epochs=2, seed=0),
        "filter_funnel": funnel,
    }


def _replay_c9_behavior_gate(
    gate_dir: Path,
    *,
    stage: str,
    accepted_path: Path,
    accepted_digest: dict[str, Any],
    expected_model_manifest: dict[str, Any],
    expected_code_artifact: dict[str, Any],
    expected_selection_sha256: str | None = None,
) -> dict[str, Any]:
    """Recompute a fixed-panel gate from its selection and raw rollouts."""
    from tmx_gpu import c9_evidence, sft_gate_gpu

    selection_path = gate_dir / "heldout_selection.jsonl"
    rollout_path = gate_dir / "rollouts.jsonl"
    aggregate_path = gate_dir / "aggregate_metrics.json"
    evidence_path = gate_dir / "evidence.json"
    evidence = _read_json(evidence_path)
    selection = _read_jsonl_strict(
        selection_path, where=f"C9 {stage} gate selection")
    rollouts = _read_jsonl_strict(
        rollout_path, where=f"C9 {stage} gate rollouts")
    aggregate = _read_json(aggregate_path)
    file_bindings = {
        "heldout_selection_file_sha256": _file_sha256(selection_path),
        "raw_rollouts_file_sha256": _file_sha256(rollout_path),
        "aggregate_file_sha256": _file_sha256(aggregate_path),
        "heldout_selection_sha256": _c9_gate_selection_digest(selection),
    }
    try:
        canonical_selection = list(_canonical_c9_gate_selection())
    except Exception as exc:  # noqa: BLE001
        _fail(f"cannot reconstruct canonical C9 diagnostic panel: {exc}")
    if selection != canonical_selection:
        _fail(
            f"C9 {stage} gate selection is not the pinned hash-ranked DAPO panel")
    if any(evidence.get(key) != value for key, value in file_bindings.items()):
        _fail(f"C9 {stage} gate support-file binding drift")
    expected_sampling = {
        "num_prompts": 32,
        "samples_per_prompt": 2,
        "selection_seed": 20260713,
        "generation_seed": 20260713,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "max_completion_tokens": 32768,
        "paired_seeds_across_prompt_modes": True,
    }
    selection_summary = aggregate.get("selection")
    if aggregate.get("metric_contract_version") != \
            sft_gate_gpu.METRIC_CONTRACT_VERSION or \
            aggregate.get("gate_stage") != stage or \
            aggregate.get("sampling") != expected_sampling or \
            aggregate.get("correctness_policy") != \
            "diagnostic only; excluded from selection and gate" or \
            not isinstance(selection_summary, dict) or \
            selection_summary.get("uses_model_outputs") is not False or \
            selection_summary.get("uses_correctness") is not False or \
            selection_summary.get("panel_role") != c9_evidence.GATE_PANEL_ROLE or \
            selection_summary.get("held_out_from_scout_training") is not False or \
            selection_summary.get("heldout_selection_sha256") != \
            file_bindings["heldout_selection_sha256"]:
        _fail(f"C9 {stage} gate aggregate contract drift")
    try:
        metrics, verdict = sft_gate_gpu.recompute_gate_from_rollouts(
            rollouts,
            num_prompts=c9_evidence.GATE_NUM_PROMPTS,
            samples_per_prompt=c9_evidence.GATE_SAMPLES_PER_PROMPT,
            generation_seed=c9_evidence.GATE_GENERATION_SEED,
            max_completion_tokens=c9_evidence.GATE_MAX_COMPLETION_TOKENS,
            ordered_problem_ids=[str(row["problem_id"]) for row in selection],
            thresholds=dict(c9_evidence.GATE_THRESHOLDS),
        )
    except Exception as exc:  # noqa: BLE001
        _fail(f"C9 {stage} gate raw-rollout replay failed: {exc}")
    if metrics != aggregate.get("metrics_by_prompt_mode") or \
            verdict != aggregate.get("health_gate") or \
            verdict.get("passed") is not True:
        _fail(f"C9 {stage} gate verdict differs from raw-rollout replay")
    try:
        c9_evidence.validate_behavior_gate_evidence(
            evidence,
            expected_stage=stage,
            expected_model_manifest=expected_model_manifest,
            expected_accepted_digest=accepted_digest,
            expected_code_artifact=expected_code_artifact,
            expected_selection_sha256=expected_selection_sha256,
            expected_profile=c9_evidence.C9_PROFILE,
            expected_cell=c9_evidence.C9_CELL,
        )
    except Exception as exc:  # noqa: BLE001
        _fail(f"C9 {stage} gate evidence is invalid: {exc}")
    if evidence.get("accepted_file_sha256") != _file_sha256(accepted_path):
        _fail(f"C9 {stage} gate accepted-library byte binding drift")
    return evidence


def _canonical_dapo_training_identity() -> dict[str, Any]:
    from tmx_jax import data

    core = {
        "version": data.TRAINING_DATASET_IDENTITY_VERSION,
        "kind": "huggingface_dataset",
        "dataset_label": "dapo_math_17k",
        "repo_id": data.DAPO_DATASET_ID,
        "config": data.DAPO_DATASET_CONFIG,
        "split": "train",
        "revision": data.DAPO_DATASET_REVISION,
    }
    return {**core, "identity_sha256": _canonical_sha256(core)}


@functools.lru_cache(maxsize=1)
def _canonical_c9_gate_selection() -> tuple[dict[str, Any], ...]:
    """Reconstruct the preregistered panel from pinned data/tokenizer bytes."""
    from transformers import AutoTokenizer

    from tmx_gpu import c9_evidence, common, sft_gate_gpu
    from tmx_jax import data

    tokenizer = AutoTokenizer.from_pretrained(
        common.CANONICAL_TRAINING_MODEL_ID,
        revision=common.CANONICAL_TRAINING_MODEL_REVISION,
        trust_remote_code=False,
    )
    examples, _stats = sft_gate_gpu.select_heldout_examples(
        data._load_raw(), tokenizer,
        sft_gate_gpu.AcceptedExclusions(
            problem_ids=frozenset(), prompt_hashes=frozenset(), row_count=0),
        num_prompts=c9_evidence.GATE_NUM_PROMPTS,
        selection_seed=c9_evidence.GATE_SELECTION_SEED,
    )
    # Match the exact three-field rows persisted by pipeline_gpu's C9 gate.
    # The problem itself is committed by selection_digest and the rendered
    # prompt by training_prompt_sha256; correctness is deliberately absent.
    return tuple(_c9_gate_selection_row(example) for example in examples)


def _c9_gate_selection_row(example: Any) -> dict[str, Any]:
    """Project one selected example to the production C9 support-file schema."""
    from tmx_gpu import sft_gate_gpu

    return {
        "problem_id": example.problem_id,
        "selection_digest": example.selection_digest,
        "training_prompt_sha256": sft_gate_gpu._sha256(
            example.training_prompt),
    }


def _c9_gate_selection_digest(selection: list[dict[str, Any]]) -> str:
    """Return the frozen identity digest emitted by ``sft_gate_gpu``."""
    return _canonical_sha256([{
        "problem_id": row.get("problem_id"),
        "selection_digest": row.get("selection_digest"),
        "training_prompt_sha256": row.get("training_prompt_sha256"),
    } for row in selection])


def _validate_c9_execution_contract(
    execution: dict[str, Any], lineage: dict[str, Any],
) -> None:
    """Require the preregistered execution, not merely its C5 label."""
    from tmx_gpu import c9_evidence, common, sft_gate_gpu

    exact = {
        "c9_execution_profile": c9_evidence.C9_PROFILE,
        "stabilization_cell": c9_evidence.C9_CELL,
        "scientific_variant": c9_evidence.C9_SCIENTIFIC_VARIANT,
        "model": common.CANONICAL_TRAINING_MODEL_ID,
        "model_revision": common.CANONICAL_TRAINING_MODEL_REVISION,
        "dataset": "dapo_math_17k",
        "seed": 0,
        "rounds": 1,
        "lambda_schedule": [0.5],
        "scout_steps": c9_evidence.SCOUT_STEPS,
        "central_steps": c9_evidence.CENTRAL_STEPS,
        "policy": "coverage_pool_c8",
        "min_accepted": c9_evidence.ACCEPTED_ROWS,
        "max_accepted": c9_evidence.ACCEPTED_ROWS,
        "harvest_max_passes": 0,
    }
    drift = [key for key, expected in exact.items()
             if execution.get(key) != expected]
    if drift:
        _fail(f"C9 execution scalar contract drift: {drift}")
    expected_gate = {
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
        "thresholds": c9_evidence.GATE_THRESHOLDS,
    }
    if execution.get("behavior_gate_contract") != expected_gate:
        _fail("C9 execution behavior-gate contract drift")
    durability = execution.get("artifact_durability_contract")
    if not isinstance(durability, dict) or \
            durability.get("online_wandb_required") is not True or \
            durability.get("durable_scout_sft_central_required") is not True or \
            type(durability.get("grpo_final_artifact_every_steps")) is not int or \
            durability["grpo_final_artifact_every_steps"] <= 0:
        _fail("C9 execution lacks durable Scout/SFT/Central artifact contract")
    training = lineage["sft_training_evidence"]
    if execution.get("scout_runtime_contract") != \
            training["scout_completion_evidence"]["scout_runtime_contract"] or \
            execution.get("sft_runtime_contract") != \
            training["sft_runtime_contract"]:
        _fail("C9 execution runtime contracts differ from durable lineage")
    dataset_identity = _canonical_dapo_training_identity()
    scout_runtime = execution["scout_runtime_contract"]
    central_runtime = lineage["central_runtime_contract"]
    for label, runtime, phase, steps, novelty in (
        ("Scout", scout_runtime, "scout", 200, 0.5),
        ("Central", central_runtime, "central", 100, 0.0),
    ):
        if runtime.get("phase") != phase or \
                runtime.get("base_model_revision") != \
                common.CANONICAL_TRAINING_MODEL_REVISION or \
                runtime.get("training_dataset_identity") != dataset_identity or \
                runtime.get("max_train_examples") != 20000 or \
                runtime.get("seed") != 0 or runtime.get("round") != 1 or \
                runtime.get("expected_steps") != steps or \
                not math.isclose(float(runtime.get("lambda_novelty", -1.0)), novelty) or \
                runtime.get("probe_contract") != c9_evidence.C9_PROBE_CONTRACT:
            _fail(f"C9 {label} runtime identity drift")
    if scout_runtime.get("model_init") != common.CANONICAL_TRAINING_MODEL_ID:
        _fail("C9 Scout was not initialized from canonical Qwen3-1.7B")
    _require_original_path_suffix(
        central_runtime.get("model_init"),
        "round_01/sft/actual_sft_final", where="C9 Central model_init")
    sft_runtime = execution["sft_runtime_contract"]
    if sft_runtime.get("model_init") != common.CANONICAL_TRAINING_MODEL_ID or \
            sft_runtime.get("base_model_revision") != \
            common.CANONICAL_TRAINING_MODEL_REVISION or \
            sft_runtime.get("seed") != 0:
        _fail("C9 SFT base-model identity drift")


def _validate_c9_sft_artifact(
    campaign_root: Path,
    sft_artifact_root: Path,
    *,
    lineage: dict[str, Any],
    accepted_facts: dict[str, Any],
    sft_artifact_ref: str,
    sft_artifact_digest: str,
    sft_artifact_type: str,
    sft_artifact_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Verify the immutable SFT handoff, including all downloaded bytes."""
    from tmx_gpu import c9_evidence, common

    binding_path = campaign_root / "round_01" / "sft_model_artifact.json"
    binding = _read_json(binding_path)
    if binding != lineage.get("sft_model_artifact"):
        _fail("C9 SFT artifact binding differs from Central lineage")
    unsigned = dict(binding)
    binding_sha = unsigned.pop("binding_sha256", None)
    if binding.get("version") != "pipeline_sft_model_artifact_v1" or \
            binding_sha != _canonical_sha256(unsigned) or \
            binding.get("durable_wandb_upload") is not True or \
            binding.get("artifact_subdir") != "model" or \
            binding.get("round") != 1 or \
            binding.get("stage_mode") != "c9_fixed_endpoint_trained":
        _fail("C9 SFT artifact binding contract drift")
    identity = _artifact_identity(
        ref=sft_artifact_ref, digest=sft_artifact_digest,
        artifact_type=sft_artifact_type, where="SFT model")
    if identity["artifact_type"] != "sft-model" or binding.get("source") != {
            "artifact_ref": identity["artifact_ref"],
            "artifact_digest": identity["artifact_digest"],
    } or binding.get("artifact_type") != "sft-model" or \
            binding.get("artifact_metadata") != sft_artifact_metadata:
        _fail("C9 SFT artifact resolution differs from signed source")
    if sft_artifact_root.is_symlink() or not sft_artifact_root.is_dir():
        _fail("C9 SFT artifact download is not a regular directory")
    model_root = sft_artifact_root / "model"
    try:
        common.verify_directory_file_manifest(
            str(model_root), binding["model_file_manifest"])
    except Exception as exc:  # noqa: BLE001
        _fail(f"C9 SFT artifact byte manifest is invalid: {exc}")
    provenance_path = model_root / "sft_provenance.json"
    campaign_provenance_path = (
        campaign_root / "round_01" / "sft" / "actual_sft_final" /
        "sft_provenance.json")
    if _file_sha256(provenance_path) != _file_sha256(campaign_provenance_path):
        _fail("C9 SFT provenance differs between results and SFT artifacts")
    provenance = _read_json(provenance_path)
    training = lineage["sft_training_evidence"]
    exact = {
        "sft_contract_version": "c8-eos-aware",
        "termination_mode": "append_eos",
        "accepted_examples": c9_evidence.ACCEPTED_ROWS,
        "trainable_examples": c9_evidence.ACCEPTED_ROWS,
        "eos_supervised_examples": c9_evidence.ACCEPTED_ROWS,
        "epochs": c9_evidence.SFT_EPOCHS,
        "model_init": common.CANONICAL_TRAINING_MODEL_ID,
        "optimizer": "adamw",
        "grad_accum": 1,
        "micro_steps": c9_evidence.SFT_OPTIMIZER_STEPS,
        "optimizer_steps": c9_evidence.SFT_OPTIMIZER_STEPS,
        "planned_optimizer_steps": c9_evidence.SFT_OPTIMIZER_STEPS,
        "requested_save_optimizer_steps": [],
        "saved_optimizer_steps": [],
        "c9_training_evidence": training,
        "fresh_c8_training_evidence": None,
        "accepted_library_digest": accepted_facts["accepted_semantic_digest"],
        "accepted_file_sha256": accepted_facts["accepted_file_sha256"],
        "training_sequence_digest": accepted_facts["training_sequence_digest"],
        "sft_runtime_contract": training["sft_runtime_contract"],
    }
    drift = [key for key, expected in exact.items()
             if provenance.get(key) != expected]
    if not math.isclose(float(provenance.get("learning_rate", -1.0)), 5e-6,
                        rel_tol=0.0, abs_tol=1e-15):
        drift.append("learning_rate")
    completed = provenance.get("completed_run_evidence")
    if not isinstance(completed, dict):
        drift.append("completed_run_evidence")
    else:
        unsigned_completed = dict(completed)
        completed_sha = unsigned_completed.pop("evidence_sha256", None)
        if completed_sha != _canonical_sha256(unsigned_completed) or \
                completed.get("training_completed") is not True or \
                completed.get("optimizer_steps_executed") != 1000 or \
                completed.get("micro_steps_executed") != 1000 or \
                completed.get("training_evidence_sha256") != \
                training["evidence_sha256"]:
            drift.append("completed_run_evidence")
    policy_manifest = provenance.get("checkpoint_file_manifest")
    try:
        common.verify_directory_file_manifest(
            str(model_root), policy_manifest,
            exclude_relative_paths=("sft_provenance.json",))
    except Exception as exc:  # noqa: BLE001
        drift.append(f"policy_manifest:{exc}")
    if binding.get("policy_file_manifest") != policy_manifest or \
            binding.get("accepted_library_digest") != \
            accepted_facts["accepted_semantic_digest"] or \
            binding.get("accepted_file_sha256") != \
            accepted_facts["accepted_file_sha256"] or \
            binding.get("training_sequence_digest") != \
            accepted_facts["training_sequence_digest"]:
        drift.append("artifact_handoff_bindings")
    if drift:
        _fail(f"C9 SFT provenance/artifact contract drift: {drift}")
    expected_metadata = {
        "round": 1,
        "stage_mode": "c9_fixed_endpoint_trained",
        "model_file_manifest_sha256": binding["model_file_manifest"]["sha256"],
        "accepted_file_sha256": accepted_facts["accepted_file_sha256"],
        "training_sequence_sha256": accepted_facts[
            "training_sequence_digest"]["sha256"],
        "expdis_stabilization_mode": c9_evidence.C9_RND_LIFECYCLE,
        "scientific_variant": c9_evidence.C9_SCIENTIFIC_VARIANT,
        "c9_execution_profile": c9_evidence.C9_PROFILE,
        "stabilization_cell": c9_evidence.C9_CELL,
        "novelty_normalization": c9_evidence.C9_NOVELTY_NORMALIZATION,
        "novelty_active_phase": "scout_only",
        "rnd_scoring_predictor_state": "frozen_for_entire_learner_step",
        "rnd_update_scope": "final_selected_learner_rows",
        "rnd_updates_per_completed_step": 1,
    }
    _expect_metadata_fields(
        sft_artifact_metadata, expected_metadata, where="SFT model artifact")
    return {
        "artifact": identity,
        "binding_sha256": binding["binding_sha256"],
        "provenance_sha256": _file_sha256(provenance_path),
        "model_file_manifest_sha256": _canonical_sha256(
            binding["model_file_manifest"]),
    }


def _validate_c9_grpo_artifact(
    binding: dict[str, Any],
    artifact_root: Path,
    *,
    phase: str,
    step: int,
    expected_policy_manifest: dict[str, Any],
    artifact_ref: str,
    artifact_digest: str,
    artifact_type: str,
    artifact_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Resolve and rehash one immutable Scout/Central GRPO endpoint."""
    from tmx_gpu import common, mrme_contract

    try:
        validated = mrme_contract.validate_grpo_final_artifact_binding(
            binding, expected_phase=phase, expected_round=1,
            expected_step=step,
            expected_policy_manifest=expected_policy_manifest,
            expected_mrme_provenance=None)
    except Exception as exc:  # noqa: BLE001
        _fail(f"C9 {phase} final artifact binding is invalid: {exc}")
    identity = _artifact_identity(
        ref=artifact_ref, digest=artifact_digest,
        artifact_type=artifact_type, where=f"C9 {phase} model")
    if identity["artifact_type"] != "model" or binding.get("source") != {
            **identity}:
        _fail(f"C9 {phase} artifact resolution differs from signed source")
    if binding.get("artifact_metadata") != artifact_metadata:
        _fail(f"C9 {phase} artifact metadata differs from signed binding")
    if artifact_root.is_symlink() or not artifact_root.is_dir():
        _fail(f"C9 {phase} model artifact download is not a regular directory")
    try:
        common.verify_directory_file_manifest(
            str(artifact_root), binding["model_file_manifest"])
        common.verify_directory_file_manifest(
            str(artifact_root), binding["policy_file_manifest"],
            exclude_relative_paths=("trainer_state.json",))
    except Exception as exc:  # noqa: BLE001
        _fail(f"C9 {phase} downloaded artifact byte manifest is invalid: {exc}")
    if validated != binding:
        _fail(f"C9 {phase} artifact validator changed the signed binding")
    return identity


def _validate_c9_single_campaign(
    campaign_root: Path,
    model_root: Path,
    *,
    campaign_metadata: dict[str, Any],
    model_metadata: dict[str, Any],
    sft_artifact_root: Path | None,
    sft_artifact_ref: str | None,
    sft_artifact_digest: str | None,
    sft_artifact_type: str | None,
    sft_artifact_metadata: dict[str, Any] | None,
    scout_artifact_root: Path | None,
    scout_artifact_ref: str | None,
    scout_artifact_digest: str | None,
    scout_artifact_type: str | None,
    scout_artifact_metadata: dict[str, Any] | None,
    central_training_artifact_root: Path | None,
    central_training_artifact_ref: str | None,
    central_training_artifact_digest: str | None,
    central_training_artifact_type: str | None,
    central_training_artifact_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    from tmx_gpu import c9_evidence, common

    lineage_path = campaign_root / "round_01" / "c9_central_lineage.json"
    state_path = campaign_root / "pipeline_state.json"
    summary_path = campaign_root / "run_summary.json"
    lineage = c9_evidence.validate_central_lineage(_read_json(lineage_path))
    state = _read_json(state_path)
    summary = _read_json(summary_path)
    execution = state.get("c9_execution_contract")
    if not isinstance(execution, dict):
        _fail("C9 campaign lacks its signed execution contract")
    unsigned_execution = dict(execution)
    execution_sha = unsigned_execution.pop("contract_sha256", None)
    if execution.get("version") != c9_evidence.PIPELINE_EXECUTION_CONTRACT_VERSION \
            or execution_sha != common.canonical_json_sha256(unsigned_execution):
        _fail("C9 campaign execution contract signature mismatch")
    if execution.get("c9_execution_profile") != "full" or \
            execution.get("stabilization_cell") != "C5" or \
            execution.get("scientific_variant") != c9_evidence.C9_SCIENTIFIC_VARIANT:
        _fail("C9 campaign is not the canonical full/C5 execution")
    _validate_c9_execution_contract(execution, lineage)

    required_stages = {
        "r01/scout", "r01/harvest", "r01/filter",
        "r01/scout_behavior_gate", "r01/sft", "r01/sft_behavior_gate",
        "r01/central",
    }
    stages = state.get("stages")
    if not isinstance(stages, dict) or not required_stages.issubset(stages):
        _fail("C9 campaign lacks a complete Scout-filter-SFT-Central chain")
    if set(stages) != required_stages:
        _fail("C9 SingleScout campaign contains unexpected/missing stages")
    scout_stage = stages["r01/scout"]
    harvest_stage = stages["r01/harvest"]
    filter_stage = stages["r01/filter"]
    scout_gate_stage = stages["r01/scout_behavior_gate"]
    sft_stage = stages["r01/sft"]
    sft_gate_stage = stages["r01/sft_behavior_gate"]
    if any(not isinstance(item, dict) for item in (
            scout_stage, harvest_stage, filter_stage, scout_gate_stage,
            sft_stage, sft_gate_stage)):
        _fail("C9 SingleScout campaign stage record is malformed")

    _require_original_path_suffix(
        scout_stage.get("final"), "round_01/scout/final",
        where="C9 Scout final")
    trajectory_paths = scout_stage.get("traj_paths")
    if not isinstance(trajectory_paths, list) or len(trajectory_paths) != 1:
        _fail("C9 SingleScout stage must bind exactly one trajectory bank")
    original_trajectory = _require_original_path_suffix(
        trajectory_paths[0], "round_01/scout/explorer_trajectories.jsonl",
        where="C9 Scout trajectory path")
    _require_original_path_suffix(
        scout_stage.get("c9_scout_completion_evidence_path"),
        "round_01/scout/c9_scout_completion_evidence.json",
        where="C9 Scout completion path")
    if harvest_stage.get("traj_paths") != [original_trajectory] or \
            set(harvest_stage) - {"traj_paths", "completed_at"}:
        _fail("C9 zero-harvest stage does not preserve the one Scout bank")
    original_accepted = _require_original_path_suffix(
        filter_stage.get("accepted"), "round_01/accepted.jsonl",
        where="C9 accepted-library path")
    if filter_stage.get("stage_mode") != "trained_pipeline_filter":
        _fail("C9 filter stage is not the trained-pipeline coverage filter")

    completion_path = (
        campaign_root / "round_01" / "scout" /
        "c9_scout_completion_evidence.json")
    completion = _read_json(completion_path)
    try:
        c9_evidence.validate_scout_completion_evidence(completion)
    except Exception as exc:  # noqa: BLE001
        _fail(f"C9 Scout completion evidence is invalid: {exc}")
    training = lineage["sft_training_evidence"]
    training_path = campaign_root / "round_01" / "c9_sft_training_evidence.json"
    if _read_json(training_path) != training:
        _fail("C9 durable SFT evidence differs from Central lineage")
    if training.get("scout_completion_evidence") != completion or \
            scout_stage.get("c9_scout_completion_evidence_sha256") != \
            completion.get("evidence_sha256") or \
            filter_stage.get("scout_completion_evidence_sha256") != \
            completion.get("evidence_sha256"):
        _fail("C9 Scout completion is not cross-bound through filter/SFT")
    scout_trainer_path = campaign_root / "round_01" / "scout" / "final" / \
        "trainer_state.json"
    scout_trainer = _read_json(scout_trainer_path)
    if scout_trainer.get("phase") != "scout" or \
            scout_trainer.get("checkpoint_kind") != "deployable_model" or \
            scout_trainer.get("step") != c9_evidence.SCOUT_STEPS or \
            scout_trainer.get("runtime_contract") != \
            completion["scout_runtime_contract"] or \
            scout_trainer.get("code_artifact_binding") != completion["code_artifact"] or \
            scout_trainer.get("rnd_selected_batch_update_count") != 200 or \
            scout_trainer.get("checkpoint_file_manifest") != \
            completion["model_file_manifest"]:
        _fail("C9 Scout trainer state differs from signed completion")
    scout_binding = _read_json(
        campaign_root / "round_01" / "scout" /
        "final_artifact_binding.json")
    if scout_stage.get("model_artifact") != scout_binding:
        _fail("C9 Scout pipeline state does not bind its immutable model artifact")
    if scout_artifact_root is None or scout_artifact_metadata is None or \
            scout_artifact_ref is None or scout_artifact_digest is None or \
            scout_artifact_type is None:
        _fail("C9 publication requires the immutable Scout model artifact")
    scout_artifact_identity = _validate_c9_grpo_artifact(
        scout_binding, scout_artifact_root, phase="scout", step=200,
        expected_policy_manifest=completion["model_file_manifest"],
        artifact_ref=scout_artifact_ref,
        artifact_digest=scout_artifact_digest,
        artifact_type=scout_artifact_type,
        artifact_metadata=scout_artifact_metadata,
    )

    accepted_path = campaign_root / "round_01" / "accepted.jsonl"
    replay = _replay_c9_coverage_pool(
        campaign_root / "round_01" / "scout" /
        "explorer_trajectories.jsonl",
        accepted_path,
        original_trajectory_path=original_trajectory,
        original_accepted_path=original_accepted,
    )
    if replay["trajectory"] != completion["trajectory"] or \
            replay["filter_funnel"] != filter_stage.get("funnel") or \
            replay["filter_funnel"] != training.get("filter_funnel") or \
            filter_stage.get("accepted_library_digest") != \
            replay["accepted_semantic_digest"] or \
            filter_stage.get("accepted_file_sha256") != \
            replay["accepted_file_sha256"]:
        _fail("C9 Scout/filter durable replay differs from signed summaries")
    accepted_evidence = training.get("accepted_library")
    if accepted_evidence != {
        "semantic_digest": replay["accepted_semantic_digest"],
        "file_sha256": replay["accepted_file_sha256"],
        "training_sequence_digest": replay["training_sequence_digest"],
    }:
        _fail("C9 SFT evidence does not bind exact 500x2 training sequence")
    scout_probe_facts = _recompute_c9_single_scout_probe_facts(
        campaign_root / "metrics.jsonl")
    if scout_probe_facts != completion.get("probe_history"):
        _fail("C9 Scout probe summary differs from durable metrics replay")

    scout_gate_dir = (
        campaign_root / "round_01" / "c9_behavior_gates" / "scout_final")
    _require_original_path_suffix(
        scout_gate_stage.get("evidence_path"),
        "round_01/c9_behavior_gates/scout_final/evidence.json",
        where="C9 Scout gate evidence path")
    scout_gate = _replay_c9_behavior_gate(
        scout_gate_dir, stage="scout_final", accepted_path=accepted_path,
        accepted_digest=replay["accepted_semantic_digest"],
        expected_model_manifest=completion["model_file_manifest"],
        expected_code_artifact=completion["code_artifact"],
    )
    if scout_gate != training.get("scout_behavior_gate") or \
            scout_gate_stage.get("evidence_sha256") != \
            scout_gate["evidence_sha256"] or \
            scout_gate_stage.get("model_file_manifest") != \
            completion["model_file_manifest"]:
        _fail("C9 Scout behavior gate is not cross-bound to pipeline state")

    if sft_artifact_root is None or sft_artifact_metadata is None or \
            sft_artifact_ref is None or sft_artifact_digest is None or \
            sft_artifact_type is None:
        _fail("C9 publication requires the immutable SFT model artifact")
    sft_facts = _validate_c9_sft_artifact(
        campaign_root, sft_artifact_root, lineage=lineage,
        accepted_facts=replay,
        sft_artifact_ref=sft_artifact_ref,
        sft_artifact_digest=sft_artifact_digest,
        sft_artifact_type=sft_artifact_type,
        sft_artifact_metadata=sft_artifact_metadata,
    )
    original_sft_final = _require_original_path_suffix(
        sft_stage.get("final"), "round_01/sft/actual_sft_final",
        where="C9 SFT final")
    if sft_stage.get("stage_mode") != "c9_fixed_endpoint_trained" or \
            sft_stage.get("termination_mode") != "append_eos" or \
            sft_stage.get("c9_training_evidence_sha256") != \
            training["evidence_sha256"] or \
            sft_stage.get("sft_model_artifact") != \
            lineage["sft_model_artifact"] or \
            sft_stage.get("canonical_pending_gate") is not True or \
            sft_stage.get("scientific_variant") != \
            c9_evidence.C9_SCIENTIFIC_VARIANT:
        _fail("C9 SFT pipeline-state handoff drift")
    if lineage["central_runtime_contract"].get("model_init") != original_sft_final:
        _fail("C9 Central runtime does not consume the signed SFT endpoint")
    sft_gate_dir = (
        campaign_root / "round_01" / "c9_behavior_gates" /
        "sft_fixed_endpoint")
    _require_original_path_suffix(
        sft_gate_stage.get("evidence_path"),
        "round_01/c9_behavior_gates/sft_fixed_endpoint/evidence.json",
        where="C9 SFT gate evidence path")
    sft_gate = _replay_c9_behavior_gate(
        sft_gate_dir, stage="sft_fixed_endpoint", accepted_path=accepted_path,
        accepted_digest=replay["accepted_semantic_digest"],
        expected_model_manifest=lineage["sft_model_file_manifest"],
        expected_code_artifact=training["code_artifact"],
        expected_selection_sha256=scout_gate["heldout_selection_sha256"],
    )
    if sft_gate != lineage.get("sft_behavior_gate") or \
            sft_gate_stage.get("evidence_sha256") != sft_gate["evidence_sha256"] or \
            sft_gate_stage.get("model_file_manifest") != \
            lineage["sft_model_file_manifest"]:
        _fail("C9 SFT behavior gate is not cross-bound to pipeline state")
    central_stage = stages["r01/central"]
    if not isinstance(central_stage, dict):
        _fail("C9 campaign Central stage is malformed")
    central_probes = central_stage.get("c9_probe_history")
    if not isinstance(central_probes, dict) or \
            central_probes.get("probe_protocol_id") != \
            c9_evidence.C9_PROBE_PROTOCOL_ID or \
            central_probes.get("phase") != "central_grpo" or \
            central_probes.get("expected_global_steps") != [225, 250, 275, 300] or \
            central_probes.get("row_count") != 4 or \
            not isinstance(central_probes.get("canonical_rows_sha256"), str) or \
            len(central_probes["canonical_rows_sha256"]) != 64:
        _fail("C9 campaign lacks the exact signed-contract Central probe history")
    replayed_central_probes = _recompute_c9_single_central_probe_facts(
        campaign_root / "metrics.jsonl")
    if replayed_central_probes != central_probes:
        _fail(
            "C9 campaign Central probe summary differs from durable metrics replay")
    try:
        c9_evidence.require_shared_probe_prompt_matrix(
            [scout_probe_facts, replayed_central_probes],
            context="C9 SingleScout publication campaign")
    except ValueError as exc:
        _fail(str(exc))
    trainer_path = model_root / "trainer_state.json"
    trainer = _read_json(trainer_path)
    if trainer.get("phase") != "central" or trainer.get("step") != 100 or \
            trainer.get("external_lineage") != lineage or \
            trainer.get("external_lineage_sha256") != lineage.get("evidence_sha256") or \
            trainer.get("runtime_contract") != lineage.get("central_runtime_contract") or \
            float(trainer.get("lambda", -1.0)) != 0.0 or \
            float(trainer.get("runtime_contract", {}).get(
                "lambda_novelty", -1.0)) != 0.0 or \
            (model_root / "rnd_state.pt").exists():
        _fail("evaluated model is not the signed correctness-only C9 Central")
    manifest = trainer.get("checkpoint_file_manifest")
    if not isinstance(manifest, dict) or \
            central_stage.get("checkpoint_file_manifest") != manifest:
        _fail("C9 campaign/model checkpoint manifest mismatch")
    try:
        common.verify_directory_file_manifest(
            str(model_root), manifest,
            exclude_relative_paths=("trainer_state.json",))
    except Exception as exc:  # noqa: BLE001
        _fail(f"C9 final-model byte manifest is invalid: {exc}")
    if central_stage.get("external_lineage_sha256") != lineage.get(
            "evidence_sha256"):
        _fail("C9 campaign Central stage has unrelated lineage")
    _require_original_path_suffix(
        central_stage.get("final"), "round_01/central/final",
        where="C9 Central final")
    central_binding = _read_json(
        campaign_root / "round_01" / "central" /
        "final_artifact_binding.json")
    if central_stage.get("model_artifact") != central_binding:
        _fail("C9 Central pipeline state does not bind its immutable model artifact")
    if central_training_artifact_root is None or \
            central_training_artifact_metadata is None or \
            central_training_artifact_ref is None or \
            central_training_artifact_digest is None or \
            central_training_artifact_type is None:
        _fail("C9 publication requires the immutable training-time Central artifact")
    central_artifact_identity = _validate_c9_grpo_artifact(
        central_binding, central_training_artifact_root,
        phase="central", step=100,
        expected_policy_manifest=manifest,
        artifact_ref=central_training_artifact_ref,
        artifact_digest=central_training_artifact_digest,
        artifact_type=central_training_artifact_type,
        artifact_metadata=central_training_artifact_metadata,
    )
    try:
        common.verify_directory_file_manifest(
            str(model_root), central_binding["model_file_manifest"])
    except Exception as exc:  # noqa: BLE001
        _fail(
            "evaluated final-model artifact is not byte-identical to the "
            f"training-time Central artifact: {exc}")
    if summary.get("final_model") != central_stage.get("final"):
        _fail("C9 campaign summary does not select its completed Central")
    summary_args = summary.get("args")
    if not isinstance(summary_args, dict) or \
            summary_args.get("c9_execution_profile") != "full" or \
            summary_args.get("stabilization_cell") != "C5" or \
            summary_args.get("scientific_variant") != \
            c9_evidence.C9_SCIENTIFIC_VARIANT:
        _fail("C9 campaign summary identity mismatch")

    code = lineage["sft_training_evidence"]["code_artifact"]
    if execution.get("code_artifact") != code or \
            trainer.get("code_artifact_binding") != code:
        _fail("C9 execution/lineage/model code identities differ")
    expected_metadata = {
        "c9_execution_profile": "full",
        "stabilization_cell": "C5",
        "scientific_variant": c9_evidence.C9_SCIENTIFIC_VARIANT,
        "c9_pipeline_execution_contract_sha256": execution_sha,
        "c9_central_lineage_sha256": lineage["evidence_sha256"],
    }
    _expect_metadata_fields(
        campaign_metadata, expected_metadata, where="campaign artifact")
    _expect_metadata_fields(model_metadata, expected_metadata, where="model artifact")
    return {
        "method": EXPDIS_SINGLESCOUT_METHOD,
        "completion_kind": "c9_signed_central_lineage",
        "completion_evidence_sha256": lineage["evidence_sha256"],
        "execution_contract_sha256": execution_sha,
        "final_model_identity_sha256": None,
        "code_artifact": _code_source_identity(code, where="C9 lineage"),
        "trainer_state_sha256": _file_sha256(trainer_path),
        "checkpoint_file_manifest_sha256": _canonical_sha256(manifest),
        "sft_artifact": sft_facts["artifact"],
        "sft_artifact_binding_sha256": sft_facts["binding_sha256"],
        "sft_provenance_sha256": sft_facts["provenance_sha256"],
        "sft_model_file_manifest_sha256": sft_facts[
            "model_file_manifest_sha256"],
        "scout_artifact": scout_artifact_identity,
        "scout_artifact_binding_sha256": scout_binding["binding_sha256"],
        "central_artifact_binding_sha256": central_binding["binding_sha256"],
        "central_training_artifact": central_artifact_identity,
    }


def _validate_mrme_campaign(
    campaign_root: Path,
    model_root: Path,
    *,
    campaign_metadata: dict[str, Any],
    model_metadata: dict[str, Any],
) -> dict[str, Any]:
    from tmx_gpu import common, mrme_contract

    completion = _read_json(campaign_root / "mrme_completed_campaign.json")
    unsigned_completion = dict(completion)
    completion_sha = unsigned_completion.pop("evidence_sha256", None)
    if completion.get("version") != "gpu_multi_explorer_completed_campaign_v2" \
            or completion_sha != common.canonical_json_sha256(unsigned_completion):
        _fail("multi-explorer completed-campaign signature mismatch")
    contract = mrme_contract.validate_contract(
        _read_json(campaign_root / "mrme_contract.json"))
    try:
        independently_completed = \
            mrme_contract.validate_portable_completed_campaign(
                str(campaign_root), allowed_output_parent="/workspace/runs")
    except Exception as exc:  # noqa: BLE001
        _fail(
            "multi-explorer portable campaign artifact is not independently "
            f"rehydratable and replay-validatable: {exc}")
    if independently_completed != completion:
        _fail(
            "stored multi-explorer completion differs from independent "
            "portable campaign replay")
    portable = _read_json(campaign_root / "mrme_portable_campaign.json")
    portable_sha = portable.get("evidence_sha256")
    unsigned_portable = dict(portable)
    unsigned_portable.pop("evidence_sha256", None)
    if portable.get("version") != \
            mrme_contract.MRME_PORTABLE_CAMPAIGN_VERSION or \
            portable_sha != common.canonical_json_sha256(unsigned_portable) or \
            portable.get("completion") != completion:
        _fail("multi-explorer portable campaign signature/binding mismatch")
    method = completion.get("method")
    if method not in {
        mrme_contract.INTEGRATED_MULTISCOUT_METHOD,
        mrme_contract.DISJOINT_SHARD_MRME_METHOD,
        mrme_contract.DISJOINT_SHARD_SINGLESCOUT_MULTIROUND_METHOD,
    }:
        _fail("multi-explorer campaign has an unrecognized method")
    single_scout_multiround = method == \
        mrme_contract.DISJOINT_SHARD_SINGLESCOUT_MULTIROUND_METHOD
    if single_scout_multiround:
        _expect_equal(
            completion.get("method_version"),
            mrme_contract.DISJOINT_SHARD_SINGLESCOUT_MULTIROUND_METHOD_VERSION,
            "single-Scout multi-round completion method version")
        _expect_equal(
            completion.get("single_scout_multiround"), True,
            "single-Scout multi-round completion identity")
        _expect_equal(
            contract.get("method_version"),
            mrme_contract.DISJOINT_SHARD_SINGLESCOUT_MULTIROUND_METHOD_VERSION,
            "single-Scout multi-round contract method version")
        _expect_equal(
            contract.get("single_scout_multiround"), True,
            "single-Scout multi-round contract identity")
        _expect_equal(
            contract.get("scouts_per_round"), 1,
            "single-Scout multi-round K")
        if not 1 <= int(contract.get("rounds", 0)) <= 4:
            _fail("single-Scout multi-round contract requires 1 <= R <= 4")
    for key, expected in (
        ("method", contract["method"]),
        ("stabilization_profile", contract["stabilization_profile"]),
        ("scientific_variant", contract["scientific_variant"]),
        ("contract_sha256", contract["contract_sha256"]),
        ("rounds", contract["rounds"]),
        ("scouts_per_round", contract["scouts_per_round"]),
        ("external_p1_required", True),
        ("headline_eligible", False),
    ):
        _expect_equal(completion.get(key), expected, f"MR-ME completion.{key}")
    _expect_equal(
        completion.get("external_publication_evaluation"),
        mrme_contract.STRICT_EXTERNAL_P1_REQUIREMENT,
        "MR-ME completion external P1 contract",
    )

    # The local MR-ME replay proves that each gate is self-consistent, but the
    # publication worker must additionally prove that every Scout and pooled
    # SFT used the one preregistered, correctness-blind DAPO panel.  Do this
    # against the downloaded thin evidence, independently of its own hashes.
    if mrme_contract.is_c9_multi_explorer(contract):
        try:
            canonical_panel = list(_canonical_c9_gate_selection())
        except Exception as exc:  # noqa: BLE001
            _fail(f"cannot reconstruct canonical C9 diagnostic panel: {exc}")
        canonical_panel_sha = _c9_gate_selection_digest(canonical_panel)
        for round_index in range(1, int(contract["rounds"]) + 1):
            round_dir = campaign_root / f"round_{round_index:02d}"
            gate_paths = [
                round_dir / "scouts" / f"scout_{scout_index:02d}" /
                "c9_behavior_gates" / "scout_final" /
                "heldout_selection.jsonl"
                for scout_index in range(
                    1, int(contract["scouts_per_round"]) + 1)
            ]
            gate_paths.append(
                round_dir / "c9_behavior_gates" / "sft_fixed_endpoint" /
                "heldout_selection.jsonl")
            for selection_path in gate_paths:
                selection = _read_jsonl_strict(
                    selection_path,
                    where="MR-ME C9 behavior-gate selection")
                if selection != canonical_panel or \
                        _c9_gate_selection_digest(selection) != \
                        canonical_panel_sha:
                    _fail(
                        "MR-ME C9 behavior gate does not use the pinned "
                        "hash-ranked DAPO panel")

    trainer_path = model_root / "trainer_state.json"
    trainer = _read_json(trainer_path)
    try:
        local_identity = mrme_contract.model_handoff_identity(
            str(model_root), expected_phase="central")
    except Exception as exc:  # noqa: BLE001
        _fail(f"multi-explorer final-model identity is invalid: {exc}")
    portable_identity = dict(local_identity)
    portable_identity.pop("identity_sha256", None)
    portable_identity["path"] = completion.get("final_model")
    if common.canonical_json_sha256(portable_identity) != completion.get(
            "final_model_identity_sha256"):
        _fail("multi-explorer campaign does not bind the evaluated final model")
    provenance = trainer.get("mrme_provenance")
    if not isinstance(provenance, dict) or \
            provenance.get("method") != method or \
            provenance.get("mrme_contract_sha256") != contract["contract_sha256"] or \
            provenance.get("phase") != "central" or \
            int(provenance.get("mrme_scouts_per_round", 0)) != int(
                contract["scouts_per_round"]):
        _fail("multi-explorer model provenance differs from its campaign")
    code = contract.get("code_identity")
    if trainer.get("code_artifact_binding") != code:
        _fail("multi-explorer model code identity differs from its campaign")
    expected_metadata = {
        "method": method,
        "stabilization_profile": completion["stabilization_profile"],
        "scientific_variant": completion["scientific_variant"],
        "external_p1_required": True,
        "headline_eligible": False,
        "mrme_contract_sha256": contract["contract_sha256"],
        "mrme_partition_sha256": completion["partition_sha256"],
        "mrme_completion_evidence_sha256": completion_sha,
        "mrme_portable_campaign_evidence_sha256": portable_sha,
        "mrme_final_model_identity_sha256": completion[
            "final_model_identity_sha256"],
    }
    if single_scout_multiround:
        expected_metadata["method_version"] = \
            mrme_contract.DISJOINT_SHARD_SINGLESCOUT_MULTIROUND_METHOD_VERSION
        expected_metadata["single_scout_multiround"] = True
    _expect_metadata_fields(
        campaign_metadata, expected_metadata, where="campaign artifact")
    _expect_metadata_fields(model_metadata, expected_metadata, where="model artifact")
    facts = {
        "method": str(method),
        "completion_kind": "signed_multi_explorer_completed_campaign",
        "completion_evidence_sha256": str(completion_sha),
        "execution_contract_sha256": contract["contract_sha256"],
        "final_model_identity_sha256": completion[
            "final_model_identity_sha256"],
        "code_artifact": _code_source_identity(
            code, where="multi-explorer contract"),
        "trainer_state_sha256": _file_sha256(trainer_path),
        "checkpoint_file_manifest_sha256": _canonical_sha256(
            trainer["checkpoint_file_manifest"]),
    }
    if single_scout_multiround:
        facts["method_version"] = \
            mrme_contract.DISJOINT_SHARD_SINGLESCOUT_MULTIROUND_METHOD_VERSION
    return facts


def validate_training_campaign_evidence(
    campaign_root: Path,
    model_root: Path,
    *,
    campaign_artifact_ref: str,
    campaign_artifact_digest: str,
    campaign_artifact_type: str,
    campaign_artifact_metadata: dict[str, Any],
    model_artifact_ref: str,
    model_artifact_digest: str,
    model_artifact_type: str,
    model_artifact_metadata: dict[str, Any],
    code_artifact_ref: str,
    code_artifact_digest: str,
    sft_artifact_root: Path | None = None,
    sft_artifact_ref: str | None = None,
    sft_artifact_digest: str | None = None,
    sft_artifact_type: str | None = None,
    sft_artifact_metadata: dict[str, Any] | None = None,
    scout_artifact_root: Path | None = None,
    scout_artifact_ref: str | None = None,
    scout_artifact_digest: str | None = None,
    scout_artifact_type: str | None = None,
    scout_artifact_metadata: dict[str, Any] | None = None,
    central_training_artifact_root: Path | None = None,
    central_training_artifact_ref: str | None = None,
    central_training_artifact_digest: str | None = None,
    central_training_artifact_type: str | None = None,
    central_training_artifact_metadata: dict[str, Any] | None = None,
    dapo_results_artifact_root: Path | None = None,
    dapo_results_artifact_ref: str | None = None,
    dapo_results_artifact_digest: str | None = None,
    dapo_results_artifact_type: str | None = None,
    dapo_results_artifact_metadata: dict[str, Any] | None = None,
    dapo_checkpoint_artifact_root: Path | None = None,
    dapo_checkpoint_artifact_ref: str | None = None,
    dapo_checkpoint_artifact_digest: str | None = None,
    dapo_checkpoint_artifact_type: str | None = None,
    dapo_checkpoint_artifact_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and cross-bind immutable trained campaign/model/code artifacts."""
    campaign_root = campaign_root.resolve()
    model_root = model_root.resolve()
    if campaign_root.is_symlink() or not campaign_root.is_dir():
        _fail("training campaign artifact download is not a regular directory")
    if model_root.is_symlink() or not model_root.is_dir():
        _fail("trained model artifact download is not a regular directory")
    campaign_artifact = _artifact_identity(
        ref=campaign_artifact_ref, digest=campaign_artifact_digest,
        artifact_type=campaign_artifact_type, where="training campaign")
    model_artifact = _artifact_identity(
        ref=model_artifact_ref, digest=model_artifact_digest,
        artifact_type=model_artifact_type, where="final model")
    if model_artifact["artifact_type"] != "model":
        _fail("final model artifact must have type 'model'")
    evaluation_code = {
        "artifact_ref": _immutable_artifact_ref(
            code_artifact_ref, "evaluation code artifact ref"),
        "artifact_digest": _expect_nonempty_string(
            code_artifact_digest, "evaluation code artifact digest"),
    }
    is_dapo = campaign_artifact["artifact_type"] == \
        dapo_evidence.DAPO_CAMPAIGN_ARTIFACT_TYPE or \
        campaign_artifact_metadata.get("method") == DAPO_METHOD
    if is_dapo:
        if campaign_artifact["artifact_type"] != \
                dapo_evidence.DAPO_CAMPAIGN_ARTIFACT_TYPE:
            _fail("DAPO campaign artifact has the wrong type")
        dapo_values = (
            dapo_results_artifact_root, dapo_results_artifact_ref,
            dapo_results_artifact_digest, dapo_results_artifact_type,
            dapo_results_artifact_metadata,
            dapo_checkpoint_artifact_root, dapo_checkpoint_artifact_ref,
            dapo_checkpoint_artifact_digest, dapo_checkpoint_artifact_type,
            dapo_checkpoint_artifact_metadata,
        )
        if any(value is None for value in dapo_values):
            _fail("DAPO publication requires its exact results artifact")
        assert dapo_results_artifact_root is not None
        assert dapo_results_artifact_ref is not None
        assert dapo_results_artifact_digest is not None
        assert dapo_results_artifact_type is not None
        assert dapo_results_artifact_metadata is not None
        assert dapo_checkpoint_artifact_root is not None
        assert dapo_checkpoint_artifact_ref is not None
        assert dapo_checkpoint_artifact_digest is not None
        assert dapo_checkpoint_artifact_type is not None
        assert dapo_checkpoint_artifact_metadata is not None
        results_artifact = _artifact_identity(
            ref=dapo_results_artifact_ref,
            digest=dapo_results_artifact_digest,
            artifact_type=dapo_results_artifact_type,
            where="DAPO results")
        if results_artifact["artifact_type"] != \
                dapo_evidence.DAPO_RESULTS_ARTIFACT_TYPE:
            _fail("DAPO results artifact has the wrong type")
        checkpoint_artifact = _artifact_identity(
            ref=dapo_checkpoint_artifact_ref,
            digest=dapo_checkpoint_artifact_digest,
            artifact_type=dapo_checkpoint_artifact_type,
            where="DAPO step-100 checkpoint")
        if checkpoint_artifact["artifact_type"] != \
                dapo_evidence.DAPO_CHECKPOINT_ARTIFACT_TYPE:
            _fail("DAPO checkpoint artifact has the wrong type")
        try:
            completion = dapo_evidence.validate_completion_campaign(
                campaign_root=str(campaign_root), model_root=str(model_root),
                checkpoint_root=str(dapo_checkpoint_artifact_root.resolve()),
                results_root=str(dapo_results_artifact_root.resolve()),
                campaign_artifact_type=campaign_artifact_type,
                campaign_artifact_metadata=campaign_artifact_metadata,
                model_artifact_ref=model_artifact_ref,
                model_artifact_digest=model_artifact_digest,
                model_artifact_type=model_artifact_type,
                model_artifact_metadata=model_artifact_metadata,
                checkpoint_artifact_ref=dapo_checkpoint_artifact_ref,
                checkpoint_artifact_digest=dapo_checkpoint_artifact_digest,
                checkpoint_artifact_type=dapo_checkpoint_artifact_type,
                checkpoint_artifact_metadata=dapo_checkpoint_artifact_metadata,
                results_artifact_ref=dapo_results_artifact_ref,
                results_artifact_digest=dapo_results_artifact_digest,
                results_artifact_type=dapo_results_artifact_type,
                results_artifact_metadata=dapo_results_artifact_metadata,
            )
        except ValueError as exc:
            _fail(f"invalid DAPO completion campaign: {exc}")
        _expect_equal(
            completion["code_artifact_binding"]["source"], evaluation_code,
            "DAPO training/evaluation code artifact identity")
        checkpoint_sha256 = model_hash(str(model_root))
        _expect_sha256(checkpoint_sha256, "evaluated model checkpoint SHA256")
        return _signed_canonical_payload(
            TRAINING_CAMPAIGN_BINDING_VERSION,
            {
                "method": DAPO_METHOD,
                "campaign_artifact": campaign_artifact,
                "model_artifact": model_artifact,
                "checkpoint_artifact": checkpoint_artifact,
                "results_artifact": results_artifact,
                "code_artifact": evaluation_code,
                "evaluated_checkpoint_sha256": checkpoint_sha256,
                "completion_kind": "signed_dapo_100_completed_campaign",
                "completion_evidence_sha256": completion["binding_sha256"],
                "execution_contract_sha256": completion[
                    "training_runtime_contract_sha256"],
                "final_model_identity_sha256": completion[
                    "final_model_artifact_binding_sha256"],
                "trainer_state_sha256": completion[
                    "final_trainer_state_sha256"],
                "checkpoint_file_manifest_sha256": completion[
                    "step_100_checkpoint_file_manifest_sha256"],
                "model_file_manifest_sha256": _canonical_sha256(
                    completion["model_artifact"]["file_manifest"]),
                "checkpoint_artifact_file_manifest_sha256": _canonical_sha256(
                    completion["checkpoint_artifact"]["file_manifest"]),
                "results_file_manifest_sha256": _canonical_sha256(
                    completion["results_artifact"]["file_manifest"]),
                "history_sha256": _canonical_sha256(completion["history"]),
                "base_model_revision": completion["base_model"]["revision"],
                "training_dataset_identity_sha256": completion[
                    "training_dataset_identity"]["identity_sha256"],
            },
        )
    if campaign_artifact["artifact_type"] != "expdis-results":
        _fail("ExpDis training campaign artifact must have type 'expdis-results'")
    campaign_method = infer_expdis_method_from_artifact_metadata(
        campaign_artifact_metadata, where="campaign artifact")
    model_method = infer_expdis_method_from_artifact_metadata(
        model_artifact_metadata, where="model artifact")
    if campaign_method is None or model_method is None or \
            campaign_method != model_method:
        _fail("campaign and model artifacts do not share one canonical ExpDis method")
    if campaign_method == EXPDIS_SINGLESCOUT_METHOD:
        facts = _validate_c9_single_campaign(
            campaign_root, model_root,
            campaign_metadata=campaign_artifact_metadata,
            model_metadata=model_artifact_metadata,
            sft_artifact_root=(
                sft_artifact_root.resolve()
                if sft_artifact_root is not None else None),
            sft_artifact_ref=sft_artifact_ref,
            sft_artifact_digest=sft_artifact_digest,
            sft_artifact_type=sft_artifact_type,
            sft_artifact_metadata=sft_artifact_metadata,
            scout_artifact_root=(
                scout_artifact_root.resolve()
                if scout_artifact_root is not None else None),
            scout_artifact_ref=scout_artifact_ref,
            scout_artifact_digest=scout_artifact_digest,
            scout_artifact_type=scout_artifact_type,
            scout_artifact_metadata=scout_artifact_metadata,
            central_training_artifact_root=(
                central_training_artifact_root.resolve()
                if central_training_artifact_root is not None else None),
            central_training_artifact_ref=central_training_artifact_ref,
            central_training_artifact_digest=central_training_artifact_digest,
            central_training_artifact_type=central_training_artifact_type,
            central_training_artifact_metadata=central_training_artifact_metadata)
    else:
        facts = _validate_mrme_campaign(
            campaign_root, model_root,
            campaign_metadata=campaign_artifact_metadata,
            model_metadata=model_artifact_metadata)
    _expect_equal(facts["method"], campaign_method, "derived campaign method")
    _expect_equal(
        facts["code_artifact"], evaluation_code,
        "training/evaluation code artifact identity")
    checkpoint_sha256 = model_hash(str(model_root))
    _expect_sha256(checkpoint_sha256, "evaluated model checkpoint SHA256")
    return _signed_canonical_payload(
        TRAINING_CAMPAIGN_BINDING_VERSION,
        {
            "method": campaign_method,
            "campaign_artifact": campaign_artifact,
            "model_artifact": model_artifact,
            "code_artifact": evaluation_code,
            "evaluated_checkpoint_sha256": checkpoint_sha256,
            **{key: facts[key] for key in (
                "completion_kind", "completion_evidence_sha256",
                "execution_contract_sha256", "final_model_identity_sha256",
                "trainer_state_sha256", "checkpoint_file_manifest_sha256",
            )},
            **({
                "sft_artifact": facts["sft_artifact"],
                "sft_artifact_binding_sha256": facts[
                    "sft_artifact_binding_sha256"],
                "sft_provenance_sha256": facts["sft_provenance_sha256"],
                "sft_model_file_manifest_sha256": facts[
                    "sft_model_file_manifest_sha256"],
                "scout_artifact": facts["scout_artifact"],
                "scout_artifact_binding_sha256": facts[
                    "scout_artifact_binding_sha256"],
                "central_artifact_binding_sha256": facts[
                    "central_artifact_binding_sha256"],
                "central_training_artifact": facts[
                    "central_training_artifact"],
            } if campaign_method == EXPDIS_SINGLESCOUT_METHOD else {}),
            **({
                "method_version": facts["method_version"],
            } if campaign_method == EXPDIS_SINGLESCOUT_MULTIROUND_METHOD else {}),
        },
    )


def validate_training_campaign_binding(value: dict[str, Any]) -> dict[str, Any]:
    binding = _verify_canonical_binding(
        value, expected_version=TRAINING_CAMPAIGN_BINDING_VERSION)
    method = binding.get("method")
    if method not in TRAINED_EXPDIS_METHODS | {DAPO_METHOD}:
        _fail("training campaign binding has an unrecognized trained method")
    campaign_type = (
        dapo_evidence.DAPO_CAMPAIGN_ARTIFACT_TYPE
        if method == DAPO_METHOD else "expdis-results")
    for key, expected_type in (
        ("campaign_artifact", campaign_type),
        ("model_artifact", "model"),
    ):
        artifact = binding.get(key)
        if not isinstance(artifact, dict):
            _fail(f"training campaign binding lacks {key}")
        _immutable_artifact_ref(artifact.get("artifact_ref"), f"{key} ref")
        _expect_nonempty_string(artifact.get("artifact_digest"), f"{key} digest")
        _expect_equal(artifact.get("artifact_type"), expected_type, f"{key} type")
    code = binding.get("code_artifact")
    if not isinstance(code, dict):
        _fail("training campaign binding lacks code_artifact")
    _immutable_artifact_ref(code.get("artifact_ref"), "binding code artifact ref")
    _expect_nonempty_string(code.get("artifact_digest"), "binding code artifact digest")
    for key in (
        "evaluated_checkpoint_sha256", "completion_evidence_sha256",
        "execution_contract_sha256", "trainer_state_sha256",
        "checkpoint_file_manifest_sha256",
    ):
        _expect_sha256(binding.get(key), f"training campaign binding.{key}")
    final_identity = binding.get("final_model_identity_sha256")
    if final_identity is not None:
        _expect_sha256(final_identity, "training campaign binding final model identity")
    if method == DAPO_METHOD:
        checkpoint = binding.get("checkpoint_artifact")
        if not isinstance(checkpoint, dict):
            _fail("DAPO training binding lacks immutable step-100 checkpoint")
        _immutable_artifact_ref(
            checkpoint.get("artifact_ref"),
            "binding DAPO checkpoint artifact ref")
        _expect_nonempty_string(
            checkpoint.get("artifact_digest"),
            "binding DAPO checkpoint artifact digest")
        _expect_equal(
            checkpoint.get("artifact_type"),
            dapo_evidence.DAPO_CHECKPOINT_ARTIFACT_TYPE,
            "binding DAPO checkpoint artifact type")
        results = binding.get("results_artifact")
        if not isinstance(results, dict):
            _fail("DAPO training binding lacks immutable results artifact")
        _immutable_artifact_ref(
            results.get("artifact_ref"), "binding DAPO results artifact ref")
        _expect_nonempty_string(
            results.get("artifact_digest"), "binding DAPO results artifact digest")
        _expect_equal(
            results.get("artifact_type"), dapo_evidence.DAPO_RESULTS_ARTIFACT_TYPE,
            "binding DAPO results artifact type")
        _expect_equal(
            binding.get("completion_kind"),
            "signed_dapo_100_completed_campaign", "DAPO completion kind")
        _expect_equal(
            binding.get("base_model_revision"),
            common.CANONICAL_TRAINING_MODEL_REVISION,
            "DAPO Base model revision")
        for key in (
            "model_file_manifest_sha256",
            "checkpoint_artifact_file_manifest_sha256",
            "results_file_manifest_sha256", "history_sha256",
            "training_dataset_identity_sha256",
        ):
            _expect_sha256(binding.get(key), f"training campaign binding.{key}")
    if binding.get("method") == EXPDIS_SINGLESCOUT_METHOD:
        artifact = binding.get("sft_artifact")
        if not isinstance(artifact, dict):
            _fail("SingleScout training binding lacks immutable SFT artifact")
        _immutable_artifact_ref(
            artifact.get("artifact_ref"), "binding SFT artifact ref")
        _expect_nonempty_string(
            artifact.get("artifact_digest"), "binding SFT artifact digest")
        _expect_equal(
            artifact.get("artifact_type"), "sft-model", "binding SFT artifact type")
        for key in (
            "sft_artifact_binding_sha256", "sft_provenance_sha256",
            "sft_model_file_manifest_sha256",
            "scout_artifact_binding_sha256",
            "central_artifact_binding_sha256",
        ):
            _expect_sha256(binding.get(key), f"training campaign binding.{key}")
        scout_artifact = binding.get("scout_artifact")
        if not isinstance(scout_artifact, dict):
            _fail("SingleScout training binding lacks immutable Scout artifact")
        _immutable_artifact_ref(
            scout_artifact.get("artifact_ref"), "binding Scout artifact ref")
        _expect_nonempty_string(
            scout_artifact.get("artifact_digest"),
            "binding Scout artifact digest")
        _expect_equal(
            scout_artifact.get("artifact_type"), "model",
            "binding Scout artifact type")
        central_artifact = binding.get("central_training_artifact")
        if not isinstance(central_artifact, dict):
            _fail("SingleScout binding lacks training-time Central artifact")
        _immutable_artifact_ref(
            central_artifact.get("artifact_ref"),
            "binding training-time Central artifact ref")
        _expect_nonempty_string(
            central_artifact.get("artifact_digest"),
            "binding training-time Central artifact digest")
        _expect_equal(
            central_artifact.get("artifact_type"), "model",
            "binding training-time Central artifact type")
    if binding.get("method") == EXPDIS_SINGLESCOUT_MULTIROUND_METHOD:
        _expect_equal(
            binding.get("method_version"),
            EXPDIS_SINGLESCOUT_MULTIROUND_METHOD_VERSION,
            "single-Scout multi-round binding method version")
    return binding


@functools.lru_cache(maxsize=None)
def _canonical_base_checkpoint_sha256(model_family: str) -> str:
    source = CANONICAL_MODEL_SOURCE_NAMES.get(str(model_family))
    revision = MODEL_REVISIONS.get(str(model_family))
    if source is None or revision is None:
        _fail(f"Base has unsupported model family {model_family!r}")
    digest = model_hash(source, revision=revision)
    return _expect_sha256(digest, f"canonical Base checkpoint {model_family}")


def _validate_publication_method_identity(identity: dict[str, Any]) -> None:
    method = str(identity.get("method_name") or "")
    if method == "Base":
        family = str(identity.get("model_family") or "")
        source = CANONICAL_MODEL_SOURCE_NAMES.get(family)
        revision = MODEL_REVISIONS.get(family)
        if source is None or revision is None:
            _fail(f"Base has unsupported model family {family!r}")
        for field in (
            "model_source_name", "model_local_or_hf_path", "served_model_name",
        ):
            _expect_equal(identity.get(field), source, f"Base canonical {field}")
        _expect_equal(
            identity.get("model_source_revision"), revision,
            "Base canonical model revision")
        _expect_equal(
            identity.get("model_artifact_ref"), None,
            "Base model artifact ref")
        _expect_equal(
            identity.get("model_artifact_digest"), None,
            "Base model artifact digest")
        _expect_equal(
            identity.get("training_campaign_binding_sha256"), None,
            "Base training campaign binding")
        _expect_equal(
            identity.get("checkpoint_sha256"),
            _canonical_base_checkpoint_sha256(family),
            "Base canonical checkpoint SHA256")
        return
    if not is_trained_publication_method(method):
        _fail(
            f"unsupported publication method {method!r}; only Base or a signed "
            "canonical trained method is eligible")


def _verify_matrix_payload_signature(value: dict[str, Any]) -> str:
    if not isinstance(value, dict):
        _fail("evaluation matrix must be a JSON object")
    signature = value.get("signature")
    if not isinstance(signature, dict):
        _fail("evaluation matrix lacks a signature")
    if signature.get("algorithm") != "sha256-canonical-json":
        _fail("evaluation matrix signature algorithm is not supported")
    payload = {key: item for key, item in value.items() if key != "signature"}
    observed = _canonical_sha256(payload)
    if signature.get("payload_sha256") != observed:
        _fail("evaluation matrix signature mismatch")
    return observed


def _validate_matrix_shared_identity(value: dict[str, Any], *, label: str) -> None:
    shared_identity = value.get("shared_identity")
    if not isinstance(shared_identity, dict):
        _fail(f"{label} shared identity is missing")
    _validate_publication_method_identity(shared_identity)
    _expect_equal(
        shared_identity.get("container_image_identity"),
        PUBLICATION_EVAL_CONTAINER_IMAGE_IDENTITY,
        f"{label} evaluation container image identity",
    )
    _expect_equal(
        shared_identity.get("runtime_stack_versions"),
        PUBLICATION_RUNTIME_STACK_VERSIONS,
        f"{label} evaluation runtime stack",
    )
    environment = shared_identity.get("evaluation_execution_environment")
    try:
        validate_publication_evaluation_execution_environment(
            environment,
            expected_run_name=value.get("run_name"),
            expected_code_artifact_ref=shared_identity.get("code_artifact_ref"),
            expected_code_artifact_digest=shared_identity.get(
                "code_artifact_digest"),
        )
    except ValueError as exc:
        _fail(f"{label} evaluation execution environment is invalid: {exc}")
    environment_sha = _expect_sha256(
        shared_identity.get("evaluation_execution_environment_sha256"),
        f"{label} evaluation execution environment SHA256",
    )
    _expect_equal(
        environment_sha,
        _canonical_sha256(environment),
        f"{label} evaluation execution environment SHA256",
    )


def verify_publication_matrix_signature(value: dict[str, Any]) -> str:
    """Verify the primary six-benchmark P1 publication matrix.

    A valid checksum only proves self-consistency.  It must not allow an old
    six-way aggregate (or a newly signed wrong aggregate) to masquerade as the
    canonical four-benchmark hard mean.
    """
    observed = _verify_matrix_payload_signature(value)
    _expect_equal(
        value.get("schema_version"), MATRIX_SCHEMA_VERSION,
        "publication matrix schema_version")
    _expect_equal(
        value.get("publication_status"), "complete",
        "publication matrix status")
    _expect_equal(
        value.get("headline_eligible"), True,
        "publication matrix headline eligibility")
    _expect_equal(value.get("protocol"), "P1", "publication matrix protocol")
    _expect_equal(
        value.get("matrix_role"), "primary_project_protocol",
        "publication matrix role")
    _expect_equal(
        value.get("benchmark_order"), list(PUBLICATION_BENCHMARKS),
        "publication matrix benchmark order")
    _expect_equal(
        value.get("benchmark_count"), len(PUBLICATION_BENCHMARKS),
        "publication matrix benchmark count")
    _expect_equal(
        value.get("hard_mean_benchmarks"), list(HARD_MEAN_BENCHMARKS),
        "publication matrix hard-mean benchmarks")
    _expect_equal(
        value.get("hard_mean_scale"), "fraction",
        "publication matrix hard-mean scale")
    _validate_matrix_shared_identity(value, label="publication matrix")
    per_benchmark = value.get("per_benchmark")
    if not isinstance(per_benchmark, dict) or \
            set(per_benchmark) != set(PUBLICATION_BENCHMARKS):
        _fail("publication matrix must contain exactly six benchmark results")
    primary_values: dict[str, float] = {}
    for benchmark in PUBLICATION_BENCHMARKS:
        item = per_benchmark.get(benchmark)
        if not isinstance(item, dict):
            _fail(f"publication matrix {benchmark} result must be an object")
        primary_values[benchmark] = _expect_metric(
            item.get("primary_value"), f"{benchmark} primary value")
    expected_hard_mean = math.fsum(
        primary_values[name] for name in HARD_MEAN_BENCHMARKS
    ) / len(HARD_MEAN_BENCHMARKS)
    observed_hard_mean = _expect_metric(
        value.get("hard_mean"), "publication matrix hard mean")
    _expect_equal(
        observed_hard_mean, expected_hard_mean,
        "publication matrix recomputed four-benchmark hard mean")
    return observed


def verify_qwen_report_p2_matrix_signature(value: dict[str, Any]) -> str:
    """Verify the secondary AIME-only Qwen-report P2 anchor matrix.

    P2 is deliberately not a headline/publication estimand.  Its separate
    schema and mandatory ``headline_eligible=False`` prevent the larger Qwen
    output budget from being mixed into the project's canonical P1 table.
    """
    observed = _verify_matrix_payload_signature(value)
    _expect_equal(
        value.get("schema_version"), QWEN_REPORT_P2_MATRIX_SCHEMA_VERSION,
        "Qwen-report P2 matrix schema_version")
    _expect_equal(
        value.get("publication_status"), "complete",
        "Qwen-report P2 matrix status")
    _expect_equal(
        value.get("headline_eligible"), False,
        "Qwen-report P2 matrix headline eligibility")
    _expect_equal(value.get("protocol"), "P2", "Qwen-report P2 matrix protocol")
    _expect_equal(
        value.get("matrix_role"), "secondary_qwen_report_anchor",
        "Qwen-report P2 matrix role")
    _expect_equal(
        value.get("benchmark_order"), list(QWEN_REPORT_P2_BENCHMARKS),
        "Qwen-report P2 matrix benchmark order")
    _expect_equal(
        value.get("benchmark_count"), len(QWEN_REPORT_P2_BENCHMARKS),
        "Qwen-report P2 matrix benchmark count")
    _expect_equal(
        value.get("anchor_mean_benchmarks"), list(QWEN_REPORT_P2_BENCHMARKS),
        "Qwen-report P2 anchor-mean benchmarks")
    _expect_equal(
        value.get("anchor_mean_scale"), "fraction",
        "Qwen-report P2 anchor-mean scale")
    _validate_matrix_shared_identity(value, label="Qwen-report P2 matrix")
    per_benchmark = value.get("per_benchmark")
    if not isinstance(per_benchmark, dict) or \
            set(per_benchmark) != set(QWEN_REPORT_P2_BENCHMARKS):
        _fail("Qwen-report P2 matrix must contain exactly the two AIME results")
    primary_values: dict[str, float] = {}
    for benchmark in QWEN_REPORT_P2_BENCHMARKS:
        item = per_benchmark.get(benchmark)
        if not isinstance(item, dict):
            _fail(f"Qwen-report P2 matrix {benchmark} result must be an object")
        primary_values[benchmark] = _expect_metric(
            item.get("primary_value"), f"{benchmark} primary value")
    expected_anchor_mean = math.fsum(primary_values.values()) / len(primary_values)
    observed_anchor_mean = _expect_metric(
        value.get("anchor_mean"), "Qwen-report P2 anchor mean")
    _expect_equal(
        observed_anchor_mean, expected_anchor_mean,
        "Qwen-report P2 recomputed two-benchmark anchor mean")
    return observed


def verify_evaluation_matrix_signature(value: dict[str, Any]) -> str:
    """Verify either the primary P1 matrix or secondary Qwen-report P2 matrix."""
    protocol = value.get("protocol") if isinstance(value, dict) else None
    if protocol == "P1":
        return verify_publication_matrix_signature(value)
    if protocol == "P2":
        return verify_qwen_report_p2_matrix_signature(value)
    _fail(f"evaluation matrix has unsupported protocol {protocol!r}")


def verify_remote_eval_artifacts(
    matrix: dict[str, Any],
    resolve_digest: Callable[[str], str],
    materialize_artifact: Callable[[str, Path], str | Path] | None = None,
) -> dict[str, str]:
    """Resolve and independently re-hash every matrix eval artifact.

    ``resolve_digest`` is injected so the scientific validation is unit
    testable without a network.  The publication launcher supplies
    ``lambda ref: wandb.Api().artifact(ref).digest``.  ``materialize_artifact``
    may likewise be injected by tests; production defaults to a fresh W&B
    download.  A matching server digest is necessary but not sufficient: exact
    member names and member bytes must also match the hashes signed in the
    publication matrix.
    """
    verify_evaluation_matrix_signature(matrix)
    benchmark_order = tuple(matrix.get("benchmark_order") or ())
    per_benchmark = matrix.get("per_benchmark")
    if not isinstance(per_benchmark, dict) or \
            set(per_benchmark) != set(benchmark_order):
        _fail("evaluation matrix artifact bindings do not match benchmark order")
    if materialize_artifact is None:
        def materialize_artifact(ref: str, destination: Path) -> str | Path:
            try:
                import wandb

                artifact = wandb.Api().artifact(ref)
                if str(getattr(artifact, "qualified_name", "") or "") != ref:
                    _fail(f"remote eval artifact resolved identity differs from {ref!r}")
                return artifact.download(root=str(destination))
            except MatrixValidationError:
                raise
            except Exception as exc:  # noqa: BLE001
                _fail(f"cannot freshly download remote eval artifact {ref!r}: {exc}")

    expected_members = {
        "aggregate_metrics.json": "aggregate_file_sha256",
        "evaluation_manifest.json": "manifest_file_sha256",
        "rollouts.jsonl": "rollouts_file_sha256",
    }
    resolved: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="tmx-eval-promotion-") as tmp:
        download_parent = Path(tmp)
        for benchmark in benchmark_order:
            item = per_benchmark.get(benchmark)
            evidence = (
                item.get("eval_results_artifact")
                if isinstance(item, dict) else None)
            if not isinstance(evidence, dict):
                _fail(f"{benchmark} lacks signed eval artifact evidence")
            ref = _expect_nonempty_string(
                evidence.get("artifact_ref"), f"{benchmark} eval artifact ref")
            expected = _expect_nonempty_string(
                evidence.get("artifact_digest"),
                f"{benchmark} eval artifact digest")
            if _IMMUTABLE_WANDB_ARTIFACT_REF.fullmatch(ref) is None:
                _fail(
                    f"{benchmark} eval artifact ref must use an immutable :vN version")
            observed = resolve_digest(ref)
            _expect_equal(
                observed, expected, f"{benchmark} remote eval artifact digest")

            destination = download_parent / benchmark
            destination.mkdir()
            try:
                materialized = Path(materialize_artifact(ref, destination)).resolve()
            except MatrixValidationError:
                raise
            except Exception as exc:  # noqa: BLE001
                _fail(f"cannot materialize {benchmark} eval artifact: {exc}")
            destination_resolved = destination.resolve()
            try:
                materialized.relative_to(destination_resolved)
            except ValueError:
                _fail(
                    f"{benchmark} eval artifact download root escapes its fresh "
                    "promotion directory")
            if not materialized.is_dir() or materialized.is_symlink():
                _fail(f"{benchmark} eval artifact download root is not a directory")
            observed_members: dict[str, Path] = {}
            for path in materialized.rglob("*"):
                if path.is_dir():
                    continue
                relative = path.relative_to(materialized).as_posix()
                if path.is_symlink() or not path.is_file():
                    _fail(
                        f"{benchmark} eval artifact member {relative!r} is not a "
                        "regular file")
                observed_members[relative] = path
            if set(observed_members) != set(expected_members):
                _fail(
                    f"{benchmark} remote eval artifact membership differs from "
                    f"the exact expected members: observed "
                    f"{sorted(observed_members)!r}")
            for member, matrix_hash_field in expected_members.items():
                expected_hash = _expect_sha256(
                    item.get(matrix_hash_field),
                    f"{benchmark} signed {member} SHA256",
                )
                observed_hash = _file_sha256(observed_members[member])
                _expect_equal(
                    observed_hash, expected_hash,
                    f"{benchmark} remote eval artifact member {member} SHA256",
                )
            resolved[benchmark] = str(observed)
    return resolved


def _expect_equal(actual: Any, expected: Any, where: str) -> None:
    if actual != expected:
        _fail(f"{where}: observed {actual!r}, expected {expected!r}")


def _expect_nonempty_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{where} must be a nonempty string")
    return value


def _expect_sha256(value: Any, where: str) -> str:
    result = _expect_nonempty_string(value, where)
    if len(result) != 64 or any(ch not in "0123456789abcdef" for ch in result):
        _fail(f"{where} is not a lowercase SHA-256 digest")
    return result


def _expect_metric(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        _fail(f"{where} must be finite and in [0, 1]")
    return result


def _benchmarks_for_protocol(protocol: str) -> tuple[str, ...]:
    if protocol == "P1":
        return PUBLICATION_BENCHMARKS
    if protocol == "P2":
        return QWEN_REPORT_P2_BENCHMARKS
    _fail(f"unsupported evaluation matrix protocol {protocol!r}")


def _source_paths(
    root: Path, *, protocol: str = "P1",
) -> dict[str, tuple[Path, Path, Path, Path]]:
    if root.is_symlink() or not root.is_dir():
        _fail(f"evaluation root must be a regular directory: {root}")
    benchmark_order = _benchmarks_for_protocol(protocol)
    expected: dict[str, tuple[Path, Path, Path, Path]] = {}
    for benchmark in benchmark_order:
        bench_dir = root / benchmark
        expected[benchmark] = (
            bench_dir / "aggregate_metrics.json",
            bench_dir / "evaluation_manifest.json",
            bench_dir / "rollouts.jsonl",
            bench_dir / "evaluation_artifact_receipt.json",
        )
    expected_aggregates = {paths[0].resolve() for paths in expected.values()}
    expected_manifests = {paths[1].resolve() for paths in expected.values()}
    expected_rollouts = {paths[2].resolve() for paths in expected.values()}
    expected_receipts = {paths[3].resolve() for paths in expected.values()}
    observed_aggregates = {path.resolve() for path in root.rglob("aggregate_metrics.json")}
    observed_manifests = {path.resolve() for path in root.rglob("evaluation_manifest.json")}
    observed_rollouts = {path.resolve() for path in root.rglob("rollouts.jsonl")}
    observed_receipts = {
        path.resolve() for path in root.rglob("evaluation_artifact_receipt.json")
    }
    membership_label = (
        "the six project P1 benchmarks" if protocol == "P1"
        else "the two AIME-only Qwen-report P2 benchmarks")
    if observed_aggregates != expected_aggregates:
        _fail(f"aggregate file membership is not exactly {membership_label}")
    if observed_manifests != expected_manifests:
        _fail(f"manifest file membership is not exactly {membership_label}")
    if observed_rollouts != expected_rollouts:
        _fail(f"rollout file membership is not exactly {membership_label}")
    if observed_receipts != expected_receipts:
        _fail(f"artifact receipt membership is not exactly {membership_label}")
    return expected


def _dataset_spec(benchmark: str) -> dict[str, Any]:
    repo, config, split = BENCHMARKS[benchmark]["hf"]
    return {
        "repository": repo,
        "config": config,
        "split": split,
        "revision": BENCHMARKS[benchmark]["revision"],
    }


def _shared_identity(manifest: dict[str, Any], aggregate: dict[str, Any]) -> dict[str, Any]:
    model = manifest.get("model")
    code = manifest.get("code")
    if not isinstance(model, dict) or not isinstance(code, dict):
        _fail("manifest model/code identity is missing")
    runtime = manifest.get("runtime_stack_versions")
    if not isinstance(runtime, dict) or not runtime:
        _fail("manifest runtime stack is missing")
    _expect_equal(
        runtime, PUBLICATION_RUNTIME_STACK_VERSIONS,
        "frozen evaluation runtime stack")
    container_image_identity = manifest.get("container_image_identity")
    _expect_equal(
        container_image_identity, PUBLICATION_EVAL_CONTAINER_IMAGE_IDENTITY,
        "frozen evaluation container image identity")

    checkpoint_hash = _expect_sha256(
        model.get("checkpoint_sha256"), "model.checkpoint_sha256")
    code_ref = _expect_nonempty_string(code.get("artifact_ref"), "code.artifact_ref")
    code_digest = _expect_nonempty_string(
        code.get("artifact_digest"), "code.artifact_digest")
    environment = manifest.get("evaluation_execution_environment")
    try:
        environment = validate_publication_evaluation_execution_environment(
            environment,
            expected_code_artifact_ref=code_ref,
            expected_code_artifact_digest=code_digest,
        )
    except ValueError as exc:
        _fail(f"manifest evaluation execution environment is invalid: {exc}")
    environment_sha = _expect_sha256(
        manifest.get("evaluation_execution_environment_sha256"),
        "evaluation_execution_environment_sha256",
    )
    _expect_equal(
        environment_sha,
        _canonical_sha256(environment),
        "evaluation execution environment SHA256",
    )
    frozen_environment = environment["frozen_invariants"]
    _expect_equal(
        container_image_identity,
        frozen_environment["container_image_identity"],
        "receipt-bound evaluation container image identity",
    )
    _expect_equal(
        runtime,
        frozen_environment["runtime_stack_versions"],
        "receipt-bound evaluation runtime stack",
    )
    executing = code.get("executing_file_sha256")
    required_code = set(EXECUTING_CODE_FILES)
    if not isinstance(executing, dict) or set(executing) != required_code:
        _fail("executing-code hash membership is not the frozen evaluator set")
    for relative, digest in executing.items():
        _expect_sha256(digest, f"code.executing_file_sha256[{relative!r}]")

    artifact_ref = model.get("artifact_ref")
    artifact_digest = model.get("artifact_digest")
    if bool(artifact_ref) != bool(artifact_digest):
        _fail("model artifact ref and digest must be both set or both null")
    if artifact_ref:
        _expect_nonempty_string(artifact_ref, "model.artifact_ref")
        _expect_nonempty_string(artifact_digest, "model.artifact_digest")

    identity = {
        "method_name": aggregate.get("method_name"),
        "model_name": aggregate.get("model_name"),
        "model_family": model.get("family"),
        "model_local_or_hf_path": model.get("local_or_hf_path"),
        "model_source_name": model.get("source_name"),
        "model_source_revision": model.get("source_revision"),
        "served_model_name": model.get("served_model_name"),
        "checkpoint_sha256": checkpoint_hash,
        "model_config_signature_sha256": model.get("config_signature_sha256"),
        "model_artifact_ref": artifact_ref,
        "model_artifact_digest": artifact_digest,
        "code_artifact_ref": code_ref,
        "code_artifact_digest": code_digest,
        "executing_file_sha256": executing,
        "runtime_stack_versions": runtime,
        "container_image_identity": container_image_identity,
        "evaluation_execution_environment": environment,
        "evaluation_execution_environment_sha256": environment_sha,
        "expected_vllm_version": manifest.get("expected_vllm_version"),
        "prompt_template_sha256": manifest.get("prompt_template_sha256"),
        "answer_extractor_version": manifest.get("answer_extractor_version"),
        "verifier_version": manifest.get("verifier_version"),
        "sample_seed_rule": manifest.get("sample_seed_rule"),
        "decoding": manifest.get("decoding"),
        "training_campaign_binding_sha256": manifest.get(
            "training_campaign_binding_sha256"),
    }
    for key in (
        "method_name", "model_name", "model_family", "model_local_or_hf_path",
        "model_source_name",
        "model_source_revision", "served_model_name",
        "model_config_signature_sha256", "container_image_identity",
        "expected_vllm_version",
        "prompt_template_sha256", "answer_extractor_version",
        "verifier_version", "sample_seed_rule",
    ):
        _expect_nonempty_string(identity[key], key)
    _expect_sha256(
        identity["model_config_signature_sha256"],
        "model_config_signature_sha256")
    _expect_sha256(identity["prompt_template_sha256"], "prompt_template_sha256")
    expected_revision = MODEL_REVISIONS.get(str(identity["model_family"]))
    _expect_equal(identity["model_source_revision"], expected_revision,
                  "model source revision")
    _expect_equal(identity["expected_vllm_version"], PUBLICATION_VLLM_VERSION,
                  "expected vLLM version")
    _expect_equal(identity["sample_seed_rule"], SAMPLE_SEED_RULE, "sample seed rule")
    _validate_publication_method_identity(identity)
    return identity


def _load_canonical_evaluation_evidence(
    benchmark: str, *, model_family: str, thinking_mode: str,
) -> dict[str, Any]:
    """Independently reconstruct benchmark and prompt identities for replay.

    The per-benchmark manifest is only a claim.  Promotion therefore reloads
    the preregistered dataset revision, revalidates its normalized row hash, and
    renders each prompt with the pinned *base* tokenizer for the declared model
    family.  Rollout-supplied gold answers and hashes are compared against this
    reconstruction rather than against one another.
    """
    source_name = CANONICAL_MODEL_SOURCE_NAMES.get(str(model_family))
    source_revision = MODEL_REVISIONS.get(str(model_family))
    if source_name is None or source_revision is None:
        _fail(f"unsupported publication model family {model_family!r}")
    try:
        rows = load_benchmark(benchmark)
        rows_sha256 = validate_loaded_benchmark(
            benchmark, rows, publication=True)
    except SystemExit as exc:
        _fail(f"{benchmark} canonical dataset reload failed: {exc}")
    except Exception as exc:  # noqa: BLE001
        _fail(f"{benchmark} canonical dataset reload failed: {exc}")

    expected_ids = set(range(int(BENCHMARKS[benchmark]["canonical_count"])))
    observed_ids = {int(row["problem_idx"]) for row in rows}
    if observed_ids != expected_ids or len(rows) != len(expected_ids):
        _fail(
            f"{benchmark} canonical row ordering/coverage differs from "
            "contiguous problem_idx order")

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            source_name, revision=source_revision)
        template_prompt = render_prompt(tokenizer, "{PROBLEM}", thinking_mode)
        identity_by_problem: dict[int, dict[str, Any]] = {}
        gold_by_problem: dict[int, str] = {}
        for row in rows:
            problem_idx = int(row["problem_idx"])
            prompt = render_prompt(tokenizer, str(row["problem"]), thinking_mode)
            identity_by_problem[problem_idx] = canonical_rollout_row_identity(
                row, prompt)
            gold_by_problem[problem_idx] = str(row["ground_truth"])
    except Exception as exc:  # noqa: BLE001
        _fail(f"{benchmark} canonical prompt reconstruction failed: {exc}")
    return {
        "rows_sha256": rows_sha256,
        "prompt_template_sha256": hashlib.sha256(
            template_prompt.encode("utf-8")).hexdigest(),
        "identity_by_problem": identity_by_problem,
        "gold_by_problem": gold_by_problem,
    }


def _validate_rollouts(
    benchmark: str,
    path: Path,
    *,
    manifest_sha256: str,
    expected_problem_count: int,
    expected_samples: int,
    aggregate: dict[str, Any],
    canonical_evidence: dict[str, Any],
) -> dict[str, Any]:
    """Re-grade and reconstruct the exact problem-by-sample matrix."""
    if path.is_symlink() or not path.is_file():
        _fail(f"{benchmark} rollout source must be a regular file")
    results: dict[int, dict[int, dict[str, Any]]] = {
        problem_idx: {} for problem_idx in range(expected_problem_count)
    }
    gold_by_problem: dict[int, str] = {}
    row_count = 0
    try:
        source = path.open()
    except Exception as exc:  # noqa: BLE001
        _fail(f"cannot open {benchmark} rollouts: {exc}")
    with source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                _fail(f"{benchmark} rollouts contain blank line {line_number}")
            try:
                row = json.loads(line, object_pairs_hook=_strict_object)
            except MatrixValidationError:
                raise
            except Exception as exc:  # noqa: BLE001
                _fail(f"{benchmark} malformed rollout line {line_number}: {exc}")
            if not isinstance(row, dict):
                _fail(f"{benchmark} rollout line {line_number} is not an object")
            row_count += 1
            if row.get("evaluation_manifest_sha256") != manifest_sha256:
                _fail(f"{benchmark} rollout line {line_number} has wrong manifest digest")
            pi, si = row.get("problem_idx"), row.get("sample_idx")
            if isinstance(pi, bool) or not isinstance(pi, int) or \
                    pi not in results:
                _fail(f"{benchmark} rollout line {line_number} has invalid problem_idx")
            if isinstance(si, bool) or not isinstance(si, int) or \
                    not 0 <= si < expected_samples:
                _fail(f"{benchmark} rollout line {line_number} has invalid sample_idx")
            if si in results[pi]:
                _fail(f"{benchmark} has duplicate rollout cell ({pi}, {si})")
            if row.get("finish_reason") not in {"stop", "length"}:
                _fail(f"{benchmark} rollout cell ({pi}, {si}) is not terminal-success")
            text = row.get("text")
            gold = row.get("gold")
            if not isinstance(text, str) or not isinstance(gold, str):
                _fail(f"{benchmark} rollout cell ({pi}, {si}) lacks text/gold")
            expected_identity = canonical_evidence["identity_by_problem"].get(pi)
            expected_gold = canonical_evidence["gold_by_problem"].get(pi)
            if not isinstance(expected_identity, dict) or expected_gold is None:
                _fail(f"{benchmark} rollout problem {pi} lacks canonical evidence")
            if gold != expected_gold:
                _fail(
                    f"{benchmark} rollout problem {pi} ground truth differs from "
                    "the pinned canonical row")
            for field in (
                "canonical_row_sha256", "problem_sha256", "prompt_sha256",
            ):
                if row.get(field) != expected_identity.get(field):
                    _fail(
                        f"{benchmark} rollout cell ({pi}, {si}) {field} differs "
                        "from the pinned canonical problem/prompt identity")
            expected_seed = pi * 100003 + si
            if row.get("sample_seed") != expected_seed:
                _fail(
                    f"{benchmark} rollout cell ({pi}, {si}) sample seed differs "
                    "from the frozen seed rule")
            if pi in gold_by_problem and gold_by_problem[pi] != gold:
                _fail(f"{benchmark} problem {pi} has inconsistent ground truth")
            gold_by_problem[pi] = gold
            pred, correct = grade(text, gold)
            if row.get("pred") != pred or row.get("correct") is not bool(correct):
                _fail(f"{benchmark} rollout cell ({pi}, {si}) fails verifier replay")
            tokens = row.get("completion_tokens")
            if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
                _fail(f"{benchmark} rollout cell ({pi}, {si}) has invalid token count")
            results[pi][si] = row

    expected_rows = expected_problem_count * expected_samples
    if row_count != expected_rows:
        _fail(
            f"{benchmark} rollout row count {row_count}, expected {expected_rows}")
    for pi, by_sample in results.items():
        if set(by_sample) != set(range(expected_samples)):
            _fail(f"{benchmark} problem {pi} does not have the exact sample matrix")
    if set(gold_by_problem) != set(range(expected_problem_count)):
        _fail(f"{benchmark} rollout problem membership is incomplete")

    per_problem, error_count, total_count = build_per_problem(results)
    _expect_equal(error_count, 0, f"{benchmark} replay generation errors")
    _expect_equal(total_count, expected_rows, f"{benchmark} replay row count")
    recomputed_avg: dict[str, float] = {}
    for k in (1, 4, 8, 16, 32, 64):
        name = f"avg@{k}"
        value = avg_at(per_problem, k)
        if k <= expected_samples:
            if value is None:
                _fail(f"{benchmark} replay did not produce {name}")
            published = _expect_metric(aggregate.get(name), f"{benchmark} {name}")
            if not math.isclose(float(value), published, rel_tol=0.0, abs_tol=1e-12):
                _fail(
                    f"{benchmark} {name} mismatch: replay={value}, "
                    f"aggregate={published}")
            recomputed_avg[name] = float(value)
        elif name in aggregate:
            _fail(f"{benchmark} aggregate contains ineligible {name}")
    problem_ids_sha = hashlib.sha256(json.dumps(
        list(range(expected_problem_count)), separators=(",", ":"),
    ).encode()).hexdigest()
    _expect_equal(
        aggregate.get("eligible_problem_ids_sha256"), problem_ids_sha,
        f"{benchmark} eligible problem membership",
    )
    return {
        "rollouts_file_sha256": _file_sha256(path),
        "rollout_row_count": row_count,
        "recomputed_avg": recomputed_avg,
    }


def _validate_artifact_receipt(
    benchmark: str,
    path: Path,
    *,
    aggregate_path: Path,
    manifest_path: Path,
    rollouts_path: Path,
    manifest_sha256: str,
) -> dict[str, Any]:
    receipt = _read_json(path)
    _expect_equal(
        receipt.get("schema_version"),
        "tmx_gpu.evaluation_artifact_receipt_v1",
        f"{benchmark} artifact receipt schema",
    )
    artifact_ref = _expect_nonempty_string(
        receipt.get("artifact_ref"), f"{benchmark} eval artifact ref")
    artifact_digest = _expect_nonempty_string(
        receipt.get("artifact_digest"), f"{benchmark} eval artifact digest")
    _expect_equal(receipt.get("evaluation_manifest_sha256"), manifest_sha256,
                  f"{benchmark} receipt manifest digest")
    _expect_equal(receipt.get("upload_verified"), True,
                  f"{benchmark} receipt upload verification")
    expected_hashes = {
        "aggregate_metrics.json": _file_sha256(aggregate_path),
        "rollouts.jsonl": _file_sha256(rollouts_path),
        "evaluation_manifest.json": _file_sha256(manifest_path),
    }
    _expect_equal(receipt.get("source_file_sha256"), expected_hashes,
                  f"{benchmark} artifact source hashes")
    return {
        "artifact_ref": artifact_ref,
        "artifact_digest": artifact_digest,
        "receipt_file_sha256": _file_sha256(path),
    }


def _validate_benchmark(
    benchmark: str,
    aggregate_path: Path,
    manifest_path: Path,
    rollouts_path: Path,
    artifact_receipt_path: Path,
    *,
    protocol: str = "P1",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    aggregate = _read_json(aggregate_path)
    manifest = _read_json(manifest_path)
    manifest_sha = _canonical_sha256(manifest)
    aggregate_sha = _file_sha256(aggregate_path)
    manifest_file_sha = _file_sha256(manifest_path)

    _expect_equal(manifest.get("manifest_version"),
                  "tmx_gpu.evaluation_manifest_v1", f"{benchmark} manifest version")
    _expect_equal(manifest.get("eval_protocol_version"), EVAL_PROTOCOL_VERSION,
                  f"{benchmark} eval protocol version")
    expected_headline_eligible = protocol == "P1"
    expected_budget = 32768 if protocol == "P1" else 38912
    minimum_server_len = 34816 if protocol == "P1" else 40960
    _expect_equal(manifest.get("protocol"), protocol, f"{benchmark} protocol")
    _expect_equal(manifest.get("headline_eligible_requested"),
                  expected_headline_eligible,
                  f"{benchmark} publication request")
    _expect_equal(manifest.get("benchmark"), benchmark, f"{benchmark} manifest name")

    model = manifest.get("model")
    if not isinstance(model, dict):
        _fail(f"{benchmark} model identity is missing")
    family = model.get("family")
    expected_samples = benchmark_sample_count(benchmark, family)
    expected_count = int(BENCHMARKS[benchmark]["canonical_count"])
    expected_source_name = CANONICAL_MODEL_SOURCE_NAMES.get(str(family))
    if expected_source_name is None:
        _fail(f"{benchmark} has unsupported model family {family!r}")
    _expect_equal(
        model.get("source_name"), expected_source_name,
        f"{benchmark} canonical model source name",
    )
    dataset = manifest.get("benchmark_dataset")
    expected_dataset = {
        **_dataset_spec(benchmark),
        "rows_sha256": BENCHMARKS[benchmark]["canonical_rows_sha256"],
        "problem_count": expected_count,
    }
    _expect_equal(dataset, expected_dataset, f"{benchmark} frozen dataset")
    _expect_equal(manifest.get("num_samples_per_problem"), expected_samples,
                  f"{benchmark} manifest sample count")
    _expect_equal(manifest.get("sample_seed_rule"), SAMPLE_SEED_RULE,
                  f"{benchmark} seed rule")
    decoding = manifest.get("decoding")
    if not isinstance(decoding, dict):
        _fail(f"{benchmark} decoding contract is missing")
    for field, expected in (
        ("temperature", 0.6), ("top_p", 0.95), ("top_k", 20),
        ("min_p", 0.0), ("thinking_mode", "auto"),
        ("max_completion_tokens", expected_budget),
    ):
        _expect_equal(decoding.get(field), expected, f"{benchmark} decoding.{field}")
    canonical_evidence = _load_canonical_evaluation_evidence(
        benchmark,
        model_family=str(family),
        thinking_mode=str(decoding.get("thinking_mode")),
    )
    _expect_equal(
        canonical_evidence["rows_sha256"],
        BENCHMARKS[benchmark]["canonical_rows_sha256"],
        f"{benchmark} independently reloaded canonical rows",
    )
    _expect_equal(
        manifest.get("prompt_template_sha256"),
        canonical_evidence["prompt_template_sha256"],
        f"{benchmark} mixed/noncanonical prompt template",
    )
    server_len = decoding.get("server_max_model_len")
    if isinstance(server_len, bool) or not isinstance(server_len, int) or \
            server_len < minimum_server_len:
        _fail(f"{benchmark} server context cannot fit the {protocol} budget")

    for field, expected in (
        ("eval_protocol_version", EVAL_PROTOCOL_VERSION),
        ("protocol", protocol),
        ("protocol_id", f"{EVAL_PROTOCOL_VERSION}:{protocol}"),
        ("evaluation_manifest_sha256", manifest_sha),
        ("headline_eligible", expected_headline_eligible),
        ("evaluation_status", "complete"),
        ("strict_exact_sample_matrix_required", True),
        ("publication_protocol_required", True),
        ("exact_sample_matrix", True),
        ("benchmark", benchmark),
        ("num_problems", expected_count),
        ("num_samples_per_problem", expected_samples),
        ("observed_num_samples_per_problem_max", expected_samples),
        ("requested_problem_count", expected_count),
        ("observed_problem_count", expected_count),
        ("requested_samples_per_problem", expected_samples),
        ("successful_sample_count_min", expected_samples),
        ("successful_sample_count_max", expected_samples),
        ("complete_problem_count", expected_count),
        ("generation_error_count", 0),
        ("generation_error_rate", 0.0),
        ("missing_problem_ids", []),
        ("unexpected_problem_ids", []),
        ("wrong_sample_counts", {}),
        ("sample_seed_rule", SAMPLE_SEED_RULE),
        ("benchmark_rows_sha256", BENCHMARKS[benchmark]["canonical_rows_sha256"]),
        ("benchmark_dataset_spec", _dataset_spec(benchmark)),
        ("temperature", 0.6), ("top_p", 0.95), ("top_k", 20),
        ("min_p", 0.0), ("thinking_mode", "auto"),
        ("max_completion_tokens", expected_budget),
    ):
        _expect_equal(aggregate.get(field), expected, f"{benchmark} aggregate.{field}")
    eligible = aggregate.get("eligible_problem_count_by_k")
    expected_eligible = {
        str(k): expected_count if k <= expected_samples else 0 for k in _METRIC_KS
    }
    _expect_equal(eligible, expected_eligible, f"{benchmark} eligible counts")

    # The aggregate duplicates key provenance so downstream readers cannot
    # accidentally combine a manifest for one checkpoint with another result.
    aggregate_manifest_pairs = (
        ("answer_extractor_version", "answer_extractor_version"),
        ("verifier_version", "verifier_version"),
        ("model_family", ("model", "family")),
        ("model_name", ("model", "source_name")),
        ("model_revision", ("model", "source_revision")),
        ("served_model_name", ("model", "served_model_name")),
        ("checkpoint_hash", ("model", "checkpoint_sha256")),
        ("model_config_signature_sha256", ("model", "config_signature_sha256")),
        ("model_artifact_ref", ("model", "artifact_ref")),
        ("model_artifact_digest", ("model", "artifact_digest")),
        ("code_artifact_ref", ("code", "artifact_ref")),
        ("code_artifact_digest", ("code", "artifact_digest")),
        ("prompt_template_hash", "prompt_template_sha256"),
        ("runtime_stack_versions", "runtime_stack_versions"),
        ("container_image_identity", "container_image_identity"),
        ("evaluation_execution_environment",
         "evaluation_execution_environment"),
        ("evaluation_execution_environment_sha256",
         "evaluation_execution_environment_sha256"),
        ("server_max_model_len", ("decoding", "server_max_model_len")),
        ("training_campaign_binding_sha256",
         "training_campaign_binding_sha256"),
    )
    for aggregate_key, manifest_key in aggregate_manifest_pairs:
        if isinstance(manifest_key, tuple):
            expected = manifest[manifest_key[0]].get(manifest_key[1])
        else:
            expected = manifest.get(manifest_key)
        _expect_equal(aggregate.get(aggregate_key), expected,
                      f"{benchmark} aggregate/manifest {aggregate_key}")

    primary_name = f"avg@{expected_samples}"
    primary_value = _expect_metric(
        aggregate.get(primary_name), f"{benchmark} {primary_name}")
    rollout_evidence = _validate_rollouts(
        benchmark,
        rollouts_path,
        manifest_sha256=manifest_sha,
        expected_problem_count=expected_count,
        expected_samples=expected_samples,
        aggregate=aggregate,
        canonical_evidence=canonical_evidence,
    )
    artifact_evidence = _validate_artifact_receipt(
        benchmark,
        artifact_receipt_path,
        aggregate_path=aggregate_path,
        manifest_path=manifest_path,
        rollouts_path=rollouts_path,
        manifest_sha256=manifest_sha,
    )
    identity = _shared_identity(manifest, aggregate)
    result = {
        "benchmark": benchmark,
        "problem_count": expected_count,
        "samples_per_problem": expected_samples,
        "generation_count": expected_count * expected_samples,
        "primary_metric": primary_name,
        "primary_value": primary_value,
        "aggregate_file_sha256": aggregate_sha,
        "manifest_file_sha256": manifest_file_sha,
        "evaluation_manifest_sha256": manifest_sha,
        **rollout_evidence,
        "eval_results_artifact": artifact_evidence,
    }
    return result, identity, aggregate


def assemble_publication_matrix(
    eval_root: Path,
    *,
    protocol: str = "P1",
    run_name: str | None = None,
    method_name: str | None = None,
    model_family: str | None = None,
    upstream_training_campaign_artifact_ref: str | None = None,
    upstream_training_campaign_artifact_digest: str | None = None,
    upstream_final_model_artifact_digest: str | None = None,
    require_upstream_training_evidence: bool = False,
    training_campaign_binding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    benchmark_order = _benchmarks_for_protocol(protocol)
    paths = _source_paths(eval_root, protocol=protocol)
    results: dict[str, dict[str, Any]] = {}
    shared: dict[str, Any] | None = None
    for benchmark in benchmark_order:
        result, identity, _aggregate = _validate_benchmark(
            benchmark, *paths[benchmark], protocol=protocol)
        if shared is None:
            shared = identity
        elif canonical_json_bytes(identity) != canonical_json_bytes(shared):
            _fail(f"{benchmark} has mixed model/code/runtime/prompt/verifier identity")
        results[benchmark] = result
    assert shared is not None
    eval_artifact_refs = [
        results[name]["eval_results_artifact"]["artifact_ref"]
        for name in benchmark_order
    ]
    eval_artifact_digests = [
        results[name]["eval_results_artifact"]["artifact_digest"]
        for name in benchmark_order
    ]
    if len(set(eval_artifact_refs)) != len(benchmark_order) or \
            len(set(eval_artifact_digests)) != len(benchmark_order):
        _fail("per-benchmark eval artifact receipts are duplicated")

    if run_name is not None:
        _expect_nonempty_string(run_name, "run_name")
    if method_name is not None:
        _expect_equal(shared["method_name"], method_name, "method_name")
    if model_family is not None:
        _expect_equal(shared["model_family"], model_family, "model_family")

    evidence_values = (
        upstream_training_campaign_artifact_ref,
        upstream_training_campaign_artifact_digest,
        upstream_final_model_artifact_digest,
    )
    if any(evidence_values) and not all(evidence_values):
        _fail("upstream training evidence ref/digest/final-model digest are all-or-none")
    is_trained = is_trained_publication_method(shared["method_name"])
    if (require_upstream_training_evidence or is_trained) and \
            training_campaign_binding is None:
        _fail(
            "trained-method publication requires a downloaded, validated, signed "
            "training campaign binding")
    upstream = None
    if training_campaign_binding is not None:
        upstream = validate_training_campaign_binding(training_campaign_binding)
        _expect_equal(
            shared["training_campaign_binding_sha256"],
            upstream["binding_sha256"],
            "evaluation/training campaign binding SHA256",
        )
        _expect_equal(upstream["method"], shared["method_name"],
                      "training campaign/evaluation method")
        _expect_equal(
            upstream["model_artifact"]["artifact_ref"],
            shared["model_artifact_ref"],
            "training campaign/evaluation model artifact ref",
        )
        _expect_equal(
            upstream["model_artifact"]["artifact_digest"],
            shared["model_artifact_digest"],
            "training campaign/evaluation model artifact digest",
        )
        _expect_equal(
            upstream["code_artifact"]["artifact_ref"],
            shared["code_artifact_ref"],
            "training campaign/evaluation code artifact ref",
        )
        _expect_equal(
            upstream["code_artifact"]["artifact_digest"],
            shared["code_artifact_digest"],
            "training campaign/evaluation code artifact digest",
        )
        _expect_equal(
            upstream["evaluated_checkpoint_sha256"],
            shared["checkpoint_sha256"],
            "training campaign/evaluation checkpoint SHA256",
        )
        if all(evidence_values):
            _expect_equal(
                upstream_training_campaign_artifact_ref,
                upstream["campaign_artifact"]["artifact_ref"],
                "legacy campaign artifact ref",
            )
            _expect_equal(
                upstream_training_campaign_artifact_digest,
                upstream["campaign_artifact"]["artifact_digest"],
                "legacy campaign artifact digest",
            )
            _expect_equal(
                upstream_final_model_artifact_digest,
                upstream["model_artifact"]["artifact_digest"],
                "legacy final-model artifact digest",
            )
    elif all(evidence_values):
        _fail(
            "scalar upstream artifact identifiers are not signed campaign "
            "completion evidence")
    elif shared["training_campaign_binding_sha256"] is not None:
        _fail("evaluation names an absent training campaign binding")
    if is_trained and shared["training_campaign_binding_sha256"] is None:
        _fail("trained ExpDis evaluation lacks its campaign-binding SHA256")
    if not is_trained and shared["training_campaign_binding_sha256"] is not None:
        _fail("non-ExpDis evaluation cannot carry an ExpDis campaign binding")

    if protocol == "P1":
        hard_mean = math.fsum(
            results[name]["primary_value"] for name in HARD_MEAN_BENCHMARKS
        ) / len(HARD_MEAN_BENCHMARKS)
        payload: dict[str, Any] = {
            "schema_version": MATRIX_SCHEMA_VERSION,
            "publication_status": "complete",
            "headline_eligible": True,
            "authority": "six_benchmark_exact_p1_matrix",
            "matrix_role": "primary_project_protocol",
            "run_name": run_name or eval_root.name,
            "protocol": "P1",
            "benchmark_order": list(benchmark_order),
            "benchmark_count": len(benchmark_order),
            "hard_mean_benchmarks": list(HARD_MEAN_BENCHMARKS),
            "hard_mean": hard_mean,
            "hard_mean_scale": "fraction",
            "per_benchmark": results,
            "shared_identity": shared,
            "upstream_training_evidence": upstream,
            "created_at_unix": time.time(),
        }
    else:
        anchor_mean = math.fsum(
            results[name]["primary_value"] for name in benchmark_order
        ) / len(benchmark_order)
        payload = {
            "schema_version": QWEN_REPORT_P2_MATRIX_SCHEMA_VERSION,
            "publication_status": "complete",
            "headline_eligible": False,
            "authority": "two_benchmark_exact_qwen_report_p2_secondary_anchor",
            "matrix_role": "secondary_qwen_report_anchor",
            "run_name": run_name or eval_root.name,
            "protocol": "P2",
            "benchmark_order": list(benchmark_order),
            "benchmark_count": len(benchmark_order),
            "anchor_mean_benchmarks": list(benchmark_order),
            "anchor_mean": anchor_mean,
            "anchor_mean_scale": "fraction",
            "per_benchmark": results,
            "shared_identity": shared,
            "upstream_training_evidence": upstream,
            "created_at_unix": time.time(),
        }
    payload["signature"] = {
        "algorithm": "sha256-canonical-json",
        "payload_sha256": _canonical_sha256(payload),
    }
    verify_evaluation_matrix_signature(payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--protocol", choices=("P1", "P2"), default="P1")
    parser.add_argument("--output", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--method-name", default=None)
    parser.add_argument("--model-family", default=None)
    parser.add_argument("--upstream-training-campaign-artifact-ref", default=None)
    parser.add_argument("--upstream-training-campaign-artifact-digest", default=None)
    parser.add_argument("--upstream-final-model-artifact-digest", default=None)
    parser.add_argument("--require-upstream-training-evidence", action="store_true")
    parser.add_argument("--training-campaign-binding", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_root = Path(args.eval_root)
    default_name = (
        "publication_matrix.json" if args.protocol == "P1"
        else "qwen_report_p2_anchor_matrix.json")
    output = Path(args.output) if args.output else eval_root / default_name
    output.unlink(missing_ok=True)
    try:
        matrix = assemble_publication_matrix(
            eval_root,
            protocol=args.protocol,
            run_name=args.run_name,
            method_name=args.method_name,
            model_family=args.model_family,
            upstream_training_campaign_artifact_ref=(
                args.upstream_training_campaign_artifact_ref),
            upstream_training_campaign_artifact_digest=(
                args.upstream_training_campaign_artifact_digest),
            upstream_final_model_artifact_digest=(
                args.upstream_final_model_artifact_digest),
            require_upstream_training_evidence=(
                args.require_upstream_training_evidence),
            training_campaign_binding=(
                _read_json(Path(args.training_campaign_binding))
                if args.training_campaign_binding else None),
        )
        atomic_write_json(output, matrix)
    except Exception:
        output.unlink(missing_ok=True)
        raise
    print(json.dumps({
        "evaluation_matrix": str(output),
        "protocol": matrix["protocol"],
        "primary_matrix_metric": (
            matrix["hard_mean"] if matrix["protocol"] == "P1"
            else matrix["anchor_mean"]),
        "signature": matrix["signature"]["payload_sha256"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
