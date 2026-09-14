from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tmx_gpu import (
    c9_evidence, common, eval_gpu, eval_matrix, filter_pool, sft_gate_gpu,
)
from tmx_gpu.tests import orchestrator_receipt_fixtures as receipt_fixtures


SHA_A = "a" * 64
SHA_B = "b" * 64
CODE_DIGEST = "code-artifact-digest"
_REAL_CANONICAL_EVIDENCE_LOADER = (
    eval_matrix._load_canonical_evaluation_evidence)


def _synthetic_canonical_row(benchmark: str, problem_idx: int) -> dict:
    return {
        "problem_idx": problem_idx,
        "problem": f"canonical {benchmark} problem {problem_idx}",
        "ground_truth": "1",
    }


def _synthetic_canonical_prompt(benchmark: str, problem_idx: int) -> str:
    return f"canonical prompt::{benchmark}::{problem_idx}"


@pytest.fixture(autouse=True)
def _avoid_network_for_publication_matrix_rows(monkeypatch):
    """Unit matrices use deterministic rows; production reloads pinned HF rows."""
    def evidence(benchmark, *, model_family, thinking_mode):
        del model_family, thinking_mode
        count = int(eval_gpu.BENCHMARKS[benchmark]["canonical_count"])
        identities = {}
        gold = {}
        for problem_idx in range(count):
            row = _synthetic_canonical_row(benchmark, problem_idx)
            identities[problem_idx] = eval_gpu.canonical_rollout_row_identity(
                row, _synthetic_canonical_prompt(benchmark, problem_idx))
            gold[problem_idx] = row["ground_truth"]
        return {
            "rows_sha256": eval_gpu.BENCHMARKS[benchmark][
                "canonical_rows_sha256"],
            "prompt_template_sha256": "5" * 64,
            "identity_by_problem": identities,
            "gold_by_problem": gold,
        }

    monkeypatch.setattr(
        eval_matrix, "_load_canonical_evaluation_evidence", evidence)
    monkeypatch.setattr(
        eval_matrix, "_canonical_base_checkpoint_sha256",
        lambda _model_family: SHA_A,
    )


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _manifest(
    benchmark: str, *, family: str = "qwen3-1.7b", protocol: str = "P1",
    run_name: str = "run",
) -> dict:
    repo, config, split = eval_gpu.BENCHMARKS[benchmark]["hf"]
    count = eval_gpu.BENCHMARKS[benchmark]["canonical_count"]
    samples = eval_gpu.benchmark_sample_count(benchmark, family)
    evaluation_environment = receipt_fixtures.evaluation_environment(
        run_name=run_name,
        code_ref="entity/project/code:v1",
        code_digest=CODE_DIGEST,
    )
    return {
        "manifest_version": "tmx_gpu.evaluation_manifest_v1",
        "eval_protocol_version": eval_gpu.EVAL_PROTOCOL_VERSION,
        "protocol": protocol,
        "headline_eligible_requested": protocol == "P1",
        "benchmark": benchmark,
        "benchmark_dataset": {
            "repository": repo,
            "config": config,
            "split": split,
            "revision": eval_gpu.BENCHMARKS[benchmark]["revision"],
            "rows_sha256": eval_gpu.BENCHMARKS[benchmark]["canonical_rows_sha256"],
            "problem_count": count,
        },
        "num_samples_per_problem": samples,
        "sample_seed_rule": eval_gpu.SAMPLE_SEED_RULE,
        "decoding": {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "thinking_mode": "auto",
            "max_completion_tokens": 32768 if protocol == "P1" else 38912,
            "server_max_model_len": 36864 if protocol == "P1" else 40960,
        },
        "model": {
            "local_or_hf_path": "Qwen/Qwen3-1.7B",
            "source_name": "Qwen/Qwen3-1.7B",
            "source_revision": eval_gpu.MODEL_REVISIONS[family],
            "served_model_name": "Qwen/Qwen3-1.7B",
            "family": family,
            "checkpoint_sha256": SHA_A,
            "config_signature_sha256": SHA_B,
            "artifact_ref": None,
            "artifact_digest": None,
        },
        "code": {
            "artifact_ref": "entity/project/code:v1",
            "artifact_digest": CODE_DIGEST,
            "executing_file_sha256": {
                "tmx_gpu/eval_gpu.py": "1" * 64,
                "tmx_gpu/eval_matrix.py": "2" * 64,
                "tmx_gpu/common.py": "3" * 64,
                "tmx_gpu/orchestrator_receipt.py": "d" * 64,
                "tmx_gpu/dapo_evidence.py": "c" * 64,
                "tmx_gpu/c9_evidence.py": "4" * 64,
                "tmx_gpu/mrme_contract.py": "5" * 64,
                "tmx_gpu/filter_pool.py": "6" * 64,
                "tmx_gpu/sft_gate_gpu.py": "a" * 64,
                "tmx_gpu/vllm_pool.py": "b" * 64,
                "tmx_jax/data.py": "7" * 64,
                "tmx_jax/prompting.py": "c" * 64,
                "tmx_jax/rewarding.py": "8" * 64,
                "tmx_gpu/launch/base_eval_job.sh": "9" * 64,
            },
        },
        "runtime_stack_versions": dict(
            eval_gpu.PUBLICATION_RUNTIME_STACK_VERSIONS),
        "expected_vllm_version": eval_gpu.PUBLICATION_VLLM_VERSION,
        "container_image_identity": (
            eval_gpu.PUBLICATION_EVAL_CONTAINER_IMAGE_IDENTITY),
        "evaluation_execution_environment": evaluation_environment,
        "evaluation_execution_environment_sha256": (
            eval_gpu.payload_sha256(evaluation_environment)),
        "prompt_template_sha256": "5" * 64,
        "answer_extractor_version": eval_gpu.REWARDING_CONTRACT_VERSION,
        "verifier_version": eval_gpu.REWARDING_CONTRACT_VERSION,
        "training_campaign_binding_sha256": None,
    }


def _aggregate(benchmark: str, manifest: dict, value: float) -> dict:
    count = eval_gpu.BENCHMARKS[benchmark]["canonical_count"]
    samples = manifest["num_samples_per_problem"]
    model = manifest["model"]
    code = manifest["code"]
    dataset = manifest["benchmark_dataset"]
    primary = f"avg@{samples}"
    protocol = manifest["protocol"]
    aggregate = {
        "eval_protocol_version": eval_gpu.EVAL_PROTOCOL_VERSION,
        "protocol": protocol,
        "protocol_id": f"{eval_gpu.EVAL_PROTOCOL_VERSION}:{protocol}",
        "evaluation_manifest_sha256": eval_matrix._canonical_sha256(manifest),
        "headline_eligible": protocol == "P1",
        "evaluation_status": "complete",
        "strict_exact_sample_matrix_required": True,
        "publication_protocol_required": True,
        "exact_sample_matrix": True,
        "answer_extractor_version": manifest["answer_extractor_version"],
        "verifier_version": manifest["verifier_version"],
        "benchmark": benchmark,
        "method_name": "Base",
        "model_name": model["source_name"],
        "model_family": model["family"],
        "model_revision": model["source_revision"],
        "served_model_name": model["served_model_name"],
        "checkpoint_hash": model["checkpoint_sha256"],
        "model_config_signature_sha256": model["config_signature_sha256"],
        "model_artifact_ref": model["artifact_ref"],
        "model_artifact_digest": model["artifact_digest"],
        "code_artifact_ref": code["artifact_ref"],
        "code_artifact_digest": code["artifact_digest"],
        "training_campaign_binding_sha256": manifest.get(
            "training_campaign_binding_sha256"),
        "prompt_template_hash": manifest["prompt_template_sha256"],
        "num_problems": count,
        "num_samples_per_problem": samples,
        "observed_num_samples_per_problem_max": samples,
        "requested_problem_count": count,
        "observed_problem_count": count,
        "requested_samples_per_problem": samples,
        "successful_sample_count_min": samples,
        "successful_sample_count_max": samples,
        "complete_problem_count": count,
        "generation_error_count": 0,
        "generation_error_rate": 0.0,
        "missing_problem_ids": [],
        "unexpected_problem_ids": [],
        "wrong_sample_counts": {},
        "sample_seed_rule": manifest["sample_seed_rule"],
        "server_max_model_len": manifest["decoding"]["server_max_model_len"],
        "runtime_stack_versions": manifest["runtime_stack_versions"],
        "container_image_identity": manifest["container_image_identity"],
        "evaluation_execution_environment": manifest[
            "evaluation_execution_environment"],
        "evaluation_execution_environment_sha256": manifest[
            "evaluation_execution_environment_sha256"],
        "benchmark_dataset_spec": {
            "repository": dataset["repository"],
            "config": dataset["config"],
            "split": dataset["split"],
            "revision": dataset["revision"],
        },
        "benchmark_rows_sha256": dataset["rows_sha256"],
        "eligible_problem_ids_sha256": hashlib.sha256(json.dumps(
            list(range(count)), separators=(",", ":"),
        ).encode()).hexdigest(),
        "eligible_problem_count_by_k": {
            str(k): count if k <= samples else 0
            for k in (1, 2, 4, 8, 16, 32, 64)
        },
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "thinking_mode": "auto",
        "max_completion_tokens": manifest["decoding"]["max_completion_tokens"],
        primary: value,
    }
    for k in (1, 4, 8, 16, 32, 64):
        if k <= samples:
            aggregate[f"avg@{k}"] = value
    return aggregate


def _make_matrix_root(
    tmp_path: Path, *, method: str = "Base", artifact=False,
    training_binding_sha256: str | None = None,
    benchmark_values: dict[str, float] | None = None,
    protocol: str = "P1",
    run_name: str = "run",
) -> Path:
    root = tmp_path / "results" / "run"
    benchmark_order = (
        eval_matrix.PUBLICATION_BENCHMARKS if protocol == "P1"
        else eval_matrix.QWEN_REPORT_P2_BENCHMARKS)
    for benchmark in benchmark_order:
        manifest = _manifest(
            benchmark, protocol=protocol, run_name=run_name)
        if artifact:
            manifest["model"]["local_or_hf_path"] = "/workspace/model"
            manifest["model"]["served_model_name"] = "/workspace/model"
            manifest["model"]["artifact_ref"] = "entity/project/final-model:v3"
            manifest["model"]["artifact_digest"] = "model-artifact-digest"
        manifest["training_campaign_binding_sha256"] = training_binding_sha256
        value = float((benchmark_values or {}).get(benchmark, 1.0))
        if value not in {0.0, 1.0}:
            raise ValueError("fixture benchmark values must be binary")
        aggregate = _aggregate(benchmark, manifest, value)
        aggregate["method_name"] = method
        if artifact:
            aggregate["served_model_name"] = "/workspace/model"
            aggregate["model_artifact_ref"] = manifest["model"]["artifact_ref"]
            aggregate["model_artifact_digest"] = manifest["model"]["artifact_digest"]
        bench_dir = root / benchmark
        _write_json(bench_dir / "evaluation_manifest.json", manifest)
        _write_json(bench_dir / "aggregate_metrics.json", aggregate)
        manifest_sha = eval_matrix._canonical_sha256(manifest)
        with (bench_dir / "rollouts.jsonl").open("w") as rollouts:
            for problem_idx in range(eval_gpu.BENCHMARKS[benchmark]["canonical_count"]):
                for sample_idx in range(manifest["num_samples_per_problem"]):
                    row_identity = eval_gpu.canonical_rollout_row_identity(
                        _synthetic_canonical_row(benchmark, problem_idx),
                        _synthetic_canonical_prompt(benchmark, problem_idx),
                    )
                    rollouts.write(json.dumps({
                        "problem_idx": problem_idx,
                        "sample_idx": sample_idx,
                        "sample_seed": problem_idx * 100003 + sample_idx,
                        "evaluation_manifest_sha256": manifest_sha,
                        "gold": "1",
                        **row_identity,
                        "pred": "1" if value == 1.0 else "0",
                        "correct": value == 1.0,
                        "finish_reason": "stop",
                        "completion_tokens": 4,
                        "text": (
                            "reasoning \\boxed{1}" if value == 1.0
                            else "reasoning \\boxed{0}"
                        ),
                    }, sort_keys=True) + "\n")
        source_hashes = {
            name: eval_matrix._file_sha256(bench_dir / name)
            for name in (
                "aggregate_metrics.json", "rollouts.jsonl",
                "evaluation_manifest.json",
            )
        }
        _write_json(bench_dir / "evaluation_artifact_receipt.json", {
            "schema_version": "tmx_gpu.evaluation_artifact_receipt_v1",
            "artifact_ref": f"entity/project/eval-{benchmark}:v1",
            "artifact_digest": f"digest-{benchmark}",
            "evaluation_manifest_sha256": manifest_sha,
            "source_file_sha256": source_hashes,
            "upload_verified": True,
        })
    return root


def _training_binding(
    method: str, *, model_digest: str = "model-artifact-digest",
    checkpoint_sha256: str = SHA_A,
) -> dict:
    payload = {
            "method": method,
            "campaign_artifact": {
                "artifact_ref": "entity/project/campaign:v1",
                "artifact_digest": "campaign-digest",
                "artifact_type": "expdis-results",
            },
            "model_artifact": {
                "artifact_ref": "entity/project/final-model:v3",
                "artifact_digest": model_digest,
                "artifact_type": "model",
            },
            "code_artifact": {
                "artifact_ref": "entity/project/code:v1",
                "artifact_digest": CODE_DIGEST,
            },
            "evaluated_checkpoint_sha256": checkpoint_sha256,
            "completion_kind": "test-signed-completion",
            "completion_evidence_sha256": "6" * 64,
            "execution_contract_sha256": "7" * 64,
            "final_model_identity_sha256": "8" * 64,
            "trainer_state_sha256": "9" * 64,
            "checkpoint_file_manifest_sha256": "a" * 64,
    }
    if method == eval_matrix.EXPDIS_SINGLESCOUT_METHOD:
        payload.update({
            "sft_artifact": {
                "artifact_ref": "entity/project/sft-model:v4",
                "artifact_digest": "sft-digest",
                "artifact_type": "sft-model",
            },
            "scout_artifact": {
                "artifact_ref": "entity/project/scout-model:v5",
                "artifact_digest": "scout-digest",
                "artifact_type": "model",
            },
            "central_training_artifact": {
                "artifact_ref": "entity/project/central-training:v6",
                "artifact_digest": "central-training-digest",
                "artifact_type": "model",
            },
            "sft_artifact_binding_sha256": "b" * 64,
            "sft_provenance_sha256": "c" * 64,
            "sft_model_file_manifest_sha256": "d" * 64,
            "scout_artifact_binding_sha256": "e" * 64,
            "central_artifact_binding_sha256": "f" * 64,
        })
    if method == eval_matrix.EXPDIS_SINGLESCOUT_MULTIROUND_METHOD:
        payload["method_version"] = \
            eval_matrix.EXPDIS_SINGLESCOUT_MULTIROUND_METHOD_VERSION
    return eval_matrix._signed_canonical_payload(
        eval_matrix.TRAINING_CAMPAIGN_BINDING_VERSION, payload)


def _c9_central_probe_row(global_step: int) -> dict:
    return {
        "global_step": global_step,
        "phase": "central_grpo",
        "round": 1,
        "lambda": 0.0,
        "lambda_novelty": 0.0,
        "num_scouts": 1,
        "expdis_stabilization_mode": "c9_selected_batch_update",
        "scientific_variant": "expdis_c9_selected_batch_rnd",
        "c9_execution_profile": "full",
        "stabilization_cell": "C5",
        "novelty_normalization": "selected_batch_zscore",
        "eval/probe_protocol": (
            c9_evidence.C9_PROBE_PROTOCOL_ID),
        "eval/probe_headline_eligible": 0.0,
        "eval/probe_metric_scale": "fraction",
        "eval/probe_sample_seed_rule": c9_evidence.C9_PROBE_CONTRACT[
            "sample_seed_rule"],
        "eval/probe_benchmark_revision": c9_evidence.C9_PROBE_CONTRACT[
            "benchmark_revision"],
        "eval/probe_benchmark_rows_sha256": c9_evidence.C9_PROBE_CONTRACT[
            "benchmark_rows_sha256"],
        "eval/probe_prompt_matrix_sha256": "5" * 64,
        "eval/probe_protocol_complete": 1.0,
        "eval/probe_error_rows": 0.0,
        "eval/probe_graded_rows": 120.0,
        "eval/probe_expected_rows": 120.0,
        "eval/probe_complete_problem_count": 30.0,
        "eval/probe_expected_problem_count": 30.0,
        "eval/probe_samples_per_problem": 4.0,
        "eval/probe_max_completion_tokens": 32768.0,
        "eval/probe_temperature": 0.6,
        "eval/probe_top_p": 0.95,
        "eval/probe_top_k": 20.0,
        "eval/probe_min_p": 0.0,
        "eval/probe_seed_base": 0.0,
        "eval/probe_policy_step": global_step - 200,
        "eval/probe_staleness_updates": 0,
        "eval/probe_fresh_weights": 1.0,
        "eval/AIME24_avg_at_4": 0.5,
    }


def _write_c9_probe_metrics(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _c9_scout_probe_row(global_step: int) -> dict:
    row = _c9_central_probe_row(global_step + 200)
    row.update({
        "global_step": global_step,
        "phase": "scout_grpo",
        "lambda": 0.5,
        "lambda_novelty": 0.5,
        "eval/probe_policy_step": global_step,
    })
    return row


def test_c9_single_central_probe_facts_are_replayed_from_metrics(tmp_path):
    path = tmp_path / "metrics.jsonl"
    rows = [_c9_central_probe_row(step) for step in (225, 250, 275, 300)]
    _write_c9_probe_metrics(path, rows)
    facts = eval_matrix._recompute_c9_single_central_probe_facts(path)
    assert facts == {
        "probe_protocol_id": c9_evidence.C9_PROBE_PROTOCOL_ID,
        "phase": "central_grpo",
        "expected_global_steps": [225, 250, 275, 300],
        "row_count": 4,
        "prompt_matrix_sha256": "5" * 64,
        "canonical_rows_sha256": facts["canonical_rows_sha256"],
    }
    assert len(facts["canonical_rows_sha256"]) == 64


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("missing", "exactly one probe"),
        ("extra", "exactly one probe"),
        ("duplicate", "exactly one probe"),
        ("decode", "probe replay failed"),
        ("stale", "probe replay failed"),
        ("errors", "identity/error contract drift"),
        ("wrong_campaign", "identity/error contract drift"),
    ],
)
def test_c9_single_central_probe_replay_rejects_bad_evidence(
    tmp_path, mutation, match,
):
    rows = [_c9_central_probe_row(step) for step in (225, 250, 275, 300)]
    if mutation == "missing":
        rows.pop()
    elif mutation == "extra":
        rows.append(_c9_central_probe_row(325))
    elif mutation == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif mutation == "decode":
        rows[1]["eval/probe_top_p"] = 0.9
    elif mutation == "stale":
        rows[2]["eval/probe_fresh_weights"] = 0.0
    elif mutation == "errors":
        rows[0]["eval/probe_error_rows"] = 1.0
    else:
        rows[0]["num_scouts"] = 3
    path = tmp_path / "metrics.jsonl"
    _write_c9_probe_metrics(path, rows)
    with pytest.raises(eval_matrix.MatrixValidationError, match=match):
        eval_matrix._recompute_c9_single_central_probe_facts(path)


def test_c9_single_probe_summary_digest_cannot_be_substituted(tmp_path):
    path = tmp_path / "metrics.jsonl"
    rows = [_c9_central_probe_row(step) for step in (225, 250, 275, 300)]
    _write_c9_probe_metrics(path, rows)
    replayed = eval_matrix._recompute_c9_single_central_probe_facts(path)
    stored = copy.deepcopy(replayed)
    stored["canonical_rows_sha256"] = "f" * 64
    assert replayed != stored


def test_c9_single_scout_probes_are_exact_and_fresh(tmp_path):
    path = tmp_path / "metrics.jsonl"
    rows = [_c9_scout_probe_row(step) for step in range(25, 201, 25)]
    _write_c9_probe_metrics(path, rows)
    facts = eval_matrix._recompute_c9_single_scout_probe_facts(path)
    assert facts["expected_global_steps"] == list(range(25, 201, 25))
    assert facts["row_count"] == 8
    drifted = copy.deepcopy(rows)
    drifted[3]["eval/probe_staleness_updates"] = 1
    _write_c9_probe_metrics(path, drifted)
    with pytest.raises(eval_matrix.MatrixValidationError, match="not exact/fresh"):
        eval_matrix._recompute_c9_single_scout_probe_facts(path)


def test_c9_single_rejects_consistent_cross_phase_prompt_drift(tmp_path):
    rows = [_c9_scout_probe_row(step) for step in range(25, 201, 25)]
    central = [_c9_central_probe_row(step) for step in (225, 250, 275, 300)]
    for row in central:
        # Every Central boundary is internally consistent.  This specifically
        # exercises the cross-phase case that per-phase checks cannot detect.
        row["eval/probe_prompt_matrix_sha256"] = "6" * 64
    path = tmp_path / "metrics.jsonl"
    _write_c9_probe_metrics(path, rows + central)
    scout_facts = eval_matrix._recompute_c9_single_scout_probe_facts(path)
    central_facts = eval_matrix._recompute_c9_single_central_probe_facts(path)
    with pytest.raises(ValueError, match="prompt matrices differ"):
        c9_evidence.require_shared_probe_prompt_matrix(
            [scout_facts, central_facts], context="test SingleScout")


def _make_c9_filter_fixture(tmp_path: Path):
    trajectory = tmp_path / "round_01" / "scout" / "explorer_trajectories.jsonl"
    accepted = tmp_path / "round_01" / "accepted.jsonl"
    trajectory.parent.mkdir(parents=True)
    with trajectory.open("w") as handle:
        for index in range(12_800):
            correct = index < 500
            row = {
                "step": index // 64 + 1,
                "problem_id": f"p{index}" if correct else "wrong",
                "prompt_text": f"problem {index}" if correct else "wrong prompt",
                "completion_text": "work \\boxed{1}" if correct else "wrong",
                "ground_truth": "1",
                "is_correct": correct,
                "valid_answer": correct,
                "terminated": True,
                "clipped": False,
                "completion_token_length": 128,
                "correctness_reward": 1.0 if correct else -1.0,
                "novelty_reward": 0.25 if correct else 0.0,
                "blended_reward": 1.25 if correct else -1.0,
                "novelty_normalization_center": 0.0,
                "novelty_normalization_scale": 1.0,
                "novelty_normalization_effective_scale": 1.0,
                "expdis_stabilization_mode": c9_evidence.C9_RND_LIFECYCLE,
                "scientific_variant": c9_evidence.C9_SCIENTIFIC_VARIANT,
                "c9_execution_profile": c9_evidence.C9_PROFILE,
                "stabilization_cell": c9_evidence.C9_CELL,
                "novelty_normalization": c9_evidence.C9_NOVELTY_NORMALIZATION,
                "rnd_update_scope": "final_selected_learner_rows",
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    _, funnel = filter_pool.pool_trajectories(
        [str(trajectory)], policy=filter_pool.C8_POLICY,
        max_examples=500, output=str(accepted),
        max_completion_tokens=filter_pool.MAX_COMPLETION_TOKENS,
        source_balance=False)
    return trajectory, accepted, funnel


def test_c9_single_replays_exact_12800_to_500_coverage_pool(tmp_path):
    trajectory, accepted, funnel = _make_c9_filter_fixture(tmp_path)
    replay = eval_matrix._replay_c9_coverage_pool(
        trajectory, accepted,
        original_trajectory_path=str(trajectory),
        original_accepted_path=str(accepted))
    assert replay["trajectory"]["row_count"] == 12_800
    assert replay["trajectory"]["step_count"] == 200
    assert replay["filter_funnel"] == funnel
    assert replay["accepted_semantic_digest"]["row_count"] == 500
    assert replay["training_sequence_digest"]["sequence_length"] == 1000


def test_c9_single_filter_replay_rejects_accepted_order_substitution(tmp_path):
    trajectory, accepted, _funnel = _make_c9_filter_fixture(tmp_path)
    rows = accepted.read_text().splitlines()
    rows[0], rows[1] = rows[1], rows[0]
    accepted.write_text("\n".join(rows) + "\n")
    with pytest.raises(eval_matrix.MatrixValidationError, match="deterministic"):
        eval_matrix._replay_c9_coverage_pool(
            trajectory, accepted,
            original_trajectory_path=str(trajectory),
            original_accepted_path=str(accepted))


def test_streamed_canonical_trajectory_digest_matches_common_contract():
    rows = [{"z": 1, "a": "α"}, {"nested": {"b": 2}}]
    assert eval_matrix._stream_canonical_json_array_digest(rows) == \
        common.canonical_json_sha256(rows)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _make_c9_gate_fixture(tmp_path: Path, monkeypatch):
    accepted = tmp_path / "accepted.jsonl"
    accepted_rows = [
        {"prompt_text": f"p{i}", "completion_text": "work \\boxed{1}"}
        for i in range(500)
    ]
    _write_jsonl(accepted, accepted_rows)
    accepted_digest = common.semantic_sft_jsonl_digest(str(accepted))
    selection = [{
        "problem_id": f"gate-{index}",
        "selection_digest": hashlib.sha256(
            f"selection-{index}".encode()).hexdigest(),
        "training_prompt_sha256": hashlib.sha256(
            f"prompt-{index}".encode()).hexdigest(),
    } for index in range(32)]
    monkeypatch.setattr(
        eval_matrix, "_canonical_c9_gate_selection",
        lambda: tuple(copy.deepcopy(selection)))
    gate_dir = tmp_path / "gate"
    gate_dir.mkdir()
    selection_path = gate_dir / "heldout_selection.jsonl"
    rollout_path = gate_dir / "rollouts.jsonl"
    aggregate_path = gate_dir / "aggregate_metrics.json"
    evidence_path = gate_dir / "evidence.json"
    _write_jsonl(selection_path, selection)
    rollouts = []
    for mode in sft_gate_gpu.PROMPT_MODES:
        for prompt_index, item in enumerate(selection):
            for sample_index in range(2):
                rollouts.append({
                    "metric_contract_version": sft_gate_gpu.METRIC_CONTRACT_VERSION,
                    "problem_id": item["problem_id"],
                    "prompt_mode": mode,
                    "sample_idx": sample_index,
                    "sample_seed": 20260713 + prompt_index * 100003 + sample_index,
                    "finish_reason": "stop",
                    "generation_error_detail": None,
                    "completion_tokens": 10,
                    "completion_text": "<think>ok</think> \\boxed{1}",
                    "correctness_diagnostic_only": True,
                })
    _write_jsonl(rollout_path, rollouts)
    metrics, verdict = sft_gate_gpu.recompute_gate_from_rollouts(
        rollouts, num_prompts=32, samples_per_prompt=2,
        generation_seed=20260713, max_completion_tokens=32768,
        ordered_problem_ids=[row["problem_id"] for row in selection],
        thresholds=dict(c9_evidence.GATE_THRESHOLDS))
    sampling = {
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
    aggregate = {
        "metric_contract_version": sft_gate_gpu.METRIC_CONTRACT_VERSION,
        "gate_stage": "scout_final",
        "selection": {
            "policy": "sha256(selection_seed, problem_id, problem)",
            "uses_model_outputs": False,
            "uses_correctness": False,
            "panel_role": c9_evidence.GATE_PANEL_ROLE,
            "held_out_from_scout_training": False,
            "heldout_selection_sha256": common.canonical_json_sha256(selection),
        },
        "sampling": sampling,
        "metrics_by_prompt_mode": metrics,
        "health_gate": verdict,
        "correctness_policy": "diagnostic only; excluded from selection and gate",
    }
    _write_json(aggregate_path, aggregate)
    manifest = {
        "version": common.DIRECTORY_FILE_MANIFEST_VERSION,
        "files": [{"relative_path": "config.json", "sha256": "a" * 64,
                   "size_bytes": 2}],
        "sha256": "b" * 64,
    }
    code = {"file_manifest_sha256": "c" * 64}
    evidence = c9_evidence.signed_payload(
        c9_evidence.BEHAVIOR_GATE_EVIDENCE_VERSION,
        {
            "c9_execution_profile": "full",
            "stabilization_cell": "C5",
            "scientific_variant": c9_evidence.C9_SCIENTIFIC_VARIANT,
            "gate_stage": "scout_final",
            "selection_uses_correctness": False,
            "panel_role": c9_evidence.GATE_PANEL_ROLE,
            "panel_held_out_from_scout_training": False,
            "passed": True,
            "sampling": sampling,
            "thresholds": dict(c9_evidence.GATE_THRESHOLDS),
            "heldout_selection_sha256": common.canonical_json_sha256(selection),
            "heldout_selection_file_sha256": common.file_sha256(
                str(selection_path)),
            "raw_rollout_count": 128,
            "raw_rollouts_file_sha256": common.file_sha256(str(rollout_path)),
            "aggregate_file_sha256": common.file_sha256(str(aggregate_path)),
            "model_file_manifest": manifest,
            "model_runtime_contract": {
                "phase": "scout", "expected_steps": 200,
                "lambda_novelty": 0.5, "c9_execution_profile": "full",
                "stabilization_cell": "C5",
                "scientific_variant": c9_evidence.C9_SCIENTIFIC_VARIANT,
            },
            "model_step": 200,
            "accepted_library_digest": accepted_digest,
            "accepted_file_sha256": common.file_sha256(str(accepted)),
            "code_artifact": code,
        })
    _write_json(evidence_path, evidence)
    return {
        "gate_dir": gate_dir,
        "accepted": accepted,
        "accepted_digest": accepted_digest,
        "selection": selection,
        "manifest": manifest,
        "code": code,
        "rollout_path": rollout_path,
        "evidence_path": evidence_path,
    }


def test_c9_canonical_panel_projection_matches_pipeline_support_schema():
    example = SimpleNamespace(
        problem_id="dapo-17",
        problem="not persisted",
        ground_truth="not persisted",
        training_prompt="rendered prompt",
        selection_digest="a" * 64,
    )
    assert eval_matrix._c9_gate_selection_row(example) == {
        "problem_id": "dapo-17",
        "selection_digest": "a" * 64,
        "training_prompt_sha256": hashlib.sha256(
            b"rendered prompt").hexdigest(),
    }


def test_c9_gate_replays_raw_verdict_and_pinned_panel(tmp_path, monkeypatch):
    fixture = _make_c9_gate_fixture(tmp_path, monkeypatch)
    evidence = eval_matrix._replay_c9_behavior_gate(
        fixture["gate_dir"], stage="scout_final",
        accepted_path=fixture["accepted"],
        accepted_digest=fixture["accepted_digest"],
        expected_model_manifest=fixture["manifest"],
        expected_code_artifact=fixture["code"])
    assert evidence["passed"] is True


def test_c9_gate_rejects_self_hashed_noncanonical_panel(tmp_path, monkeypatch):
    fixture = _make_c9_gate_fixture(tmp_path, monkeypatch)
    selection_path = fixture["gate_dir"] / "heldout_selection.jsonl"
    rows = copy.deepcopy(fixture["selection"])
    rows[0]["problem_id"] = "substituted"
    _write_jsonl(selection_path, rows)
    with pytest.raises(eval_matrix.MatrixValidationError, match="pinned hash-ranked"):
        eval_matrix._replay_c9_behavior_gate(
            fixture["gate_dir"], stage="scout_final",
            accepted_path=fixture["accepted"],
            accepted_digest=fixture["accepted_digest"],
            expected_model_manifest=fixture["manifest"],
            expected_code_artifact=fixture["code"])


def test_c9_gate_rejects_raw_rollout_verdict_substitution(tmp_path, monkeypatch):
    fixture = _make_c9_gate_fixture(tmp_path, monkeypatch)
    rows = [json.loads(line) for line in fixture["rollout_path"].read_text().splitlines()]
    rows[0]["finish_reason"] = "length"
    rows[0]["completion_tokens"] = 32768
    _write_jsonl(fixture["rollout_path"], rows)
    evidence = json.loads(fixture["evidence_path"].read_text())
    evidence["raw_rollouts_file_sha256"] = common.file_sha256(
        str(fixture["rollout_path"]))
    unsigned = dict(evidence)
    unsigned.pop("evidence_sha256")
    _write_json(
        fixture["evidence_path"],
        c9_evidence.signed_payload(
            c9_evidence.BEHAVIOR_GATE_EVIDENCE_VERSION,
            {key: value for key, value in unsigned.items() if key != "version"}))
    with pytest.raises(eval_matrix.MatrixValidationError, match="verdict differs"):
        eval_matrix._replay_c9_behavior_gate(
            fixture["gate_dir"], stage="scout_final",
            accepted_path=fixture["accepted"],
            accepted_digest=fixture["accepted_digest"],
            expected_model_manifest=fixture["manifest"],
            expected_code_artifact=fixture["code"])


def test_mrme_eval_uses_portable_replay_for_thin_download(
    tmp_path, monkeypatch,
):
    """A thin results artifact must never enter the local-path validator first."""
    from tmx_gpu import mrme_contract

    campaign_root = tmp_path / "downloaded-thin-results"
    model_root = tmp_path / "downloaded-final-model"
    campaign_root.mkdir()
    model_root.mkdir()
    manifest = {"sha256": "9" * 64, "files": []}
    code = {
        "source": {
            "artifact_ref": "entity/project/code:v7",
            "artifact_digest": "code-digest",
        },
    }
    contract = {
        "method": mrme_contract.INTEGRATED_MULTISCOUT_METHOD,
        "stabilization_profile": "fixture-portable-profile",
        "scientific_variant": "fixture-portable-variant",
        "contract_sha256": "1" * 64,
        "rounds": 1,
        "scouts_per_round": 3,
        "code_identity": code,
    }
    original_final = "/workspace/runs/fixture/round_01/central/final"
    local_identity = {
        "path": str(model_root),
        "phase": "central",
        "checkpoint_file_manifest": manifest,
        "identity_sha256": "ignored-after-local-validation",
    }
    portable_identity = dict(local_identity)
    portable_identity.pop("identity_sha256")
    portable_identity["path"] = original_final
    completion_payload = {
        "version": "gpu_multi_explorer_completed_campaign_v2",
        "method": contract["method"],
        "stabilization_profile": contract["stabilization_profile"],
        "scientific_variant": contract["scientific_variant"],
        "contract_sha256": contract["contract_sha256"],
        "partition_sha256": "2" * 64,
        "rounds": 1,
        "scouts_per_round": 3,
        "external_p1_required": True,
        "headline_eligible": False,
        "external_publication_evaluation":
            mrme_contract.STRICT_EXTERNAL_P1_REQUIREMENT,
        "final_model": original_final,
        "final_model_identity_sha256": common.canonical_json_sha256(
            portable_identity),
    }
    completion = {
        **completion_payload,
        "evidence_sha256": common.canonical_json_sha256(completion_payload),
    }
    portable_payload = {
        "version": mrme_contract.MRME_PORTABLE_CAMPAIGN_VERSION,
        "completion": completion,
    }
    portable = {
        **portable_payload,
        "evidence_sha256": common.canonical_json_sha256(portable_payload),
    }
    _write_json(campaign_root / "mrme_completed_campaign.json", completion)
    _write_json(campaign_root / "mrme_contract.json", {"thin": True})
    _write_json(campaign_root / "mrme_portable_campaign.json", portable)
    _write_json(model_root / "trainer_state.json", {
        "checkpoint_file_manifest": manifest,
        "mrme_provenance": {
            "method": contract["method"],
            "mrme_contract_sha256": contract["contract_sha256"],
            "phase": "central",
            "mrme_scouts_per_round": 3,
        },
        "code_artifact_binding": code,
    })

    def local_only_validator(_root):
        raise AssertionError("thin download has no original local model paths")

    portable_calls = []

    def portable_validator(root, *, allowed_output_parent):
        portable_calls.append((root, allowed_output_parent))
        return copy.deepcopy(completion)

    monkeypatch.setattr(mrme_contract, "validate_contract", lambda _value: contract)
    monkeypatch.setattr(
        mrme_contract, "validate_completed_campaign", local_only_validator)
    monkeypatch.setattr(
        mrme_contract, "validate_portable_completed_campaign",
        portable_validator)
    monkeypatch.setattr(
        mrme_contract, "is_c9_multi_explorer", lambda _contract: False)
    monkeypatch.setattr(
        mrme_contract, "model_handoff_identity",
        lambda _root, expected_phase: copy.deepcopy(local_identity))

    with pytest.raises(AssertionError, match="thin download"):
        mrme_contract.validate_completed_campaign(str(campaign_root))
    metadata = {
        "method": contract["method"],
        "stabilization_profile": contract["stabilization_profile"],
        "scientific_variant": contract["scientific_variant"],
        "external_p1_required": True,
        "headline_eligible": False,
        "mrme_contract_sha256": contract["contract_sha256"],
        "mrme_partition_sha256": completion["partition_sha256"],
        "mrme_completion_evidence_sha256": completion["evidence_sha256"],
        "mrme_portable_campaign_evidence_sha256": portable["evidence_sha256"],
        "mrme_final_model_identity_sha256": completion[
            "final_model_identity_sha256"],
    }
    facts = eval_matrix._validate_mrme_campaign(
        campaign_root, model_root,
        campaign_metadata=metadata, model_metadata=metadata)
    assert portable_calls == [(str(campaign_root), "/workspace/runs")]
    assert facts["completion_evidence_sha256"] == completion["evidence_sha256"]


def _mutate_pair(root: Path, benchmark: str, mutate_manifest=None, mutate_aggregate=None):
    bench = root / benchmark
    manifest_path = bench / "evaluation_manifest.json"
    aggregate_path = bench / "aggregate_metrics.json"
    manifest = json.loads(manifest_path.read_text())
    aggregate = json.loads(aggregate_path.read_text())
    if mutate_manifest:
        mutate_manifest(manifest)
    if mutate_aggregate:
        mutate_aggregate(aggregate)
    aggregate["evaluation_manifest_sha256"] = eval_matrix._canonical_sha256(manifest)
    _write_json(manifest_path, manifest)
    _write_json(aggregate_path, aggregate)
    new_manifest_sha = aggregate["evaluation_manifest_sha256"]
    rollouts_path = bench / "rollouts.jsonl"
    rows = [json.loads(line) for line in rollouts_path.read_text().splitlines()]
    for row in rows:
        row["evaluation_manifest_sha256"] = new_manifest_sha
    rollouts_path.write_text("".join(
        json.dumps(row, sort_keys=True) + "\n" for row in rows))
    receipt_path = bench / "evaluation_artifact_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["evaluation_manifest_sha256"] = new_manifest_sha
    receipt["source_file_sha256"] = {
        name: eval_matrix._file_sha256(bench / name)
        for name in (
            "aggregate_metrics.json", "rollouts.jsonl",
            "evaluation_manifest.json",
        )
    }
    _write_json(receipt_path, receipt)


def test_production_canonical_replay_reloads_pinned_rows_and_base_tokenizer(
    monkeypatch,
):
    benchmark = "AIME24"
    rows = [
        _synthetic_canonical_row(benchmark, problem_idx)
        for problem_idx in range(
            eval_gpu.BENCHMARKS[benchmark]["canonical_count"])
    ]
    calls = []

    monkeypatch.setattr(
        eval_matrix, "load_benchmark",
        lambda name: calls.append(("dataset", name)) or copy.deepcopy(rows),
    )

    def validate(name, observed, *, publication):
        calls.append(("validate", name, publication, len(observed)))
        return eval_gpu.BENCHMARKS[name]["canonical_rows_sha256"]

    monkeypatch.setattr(eval_matrix, "validate_loaded_benchmark", validate)

    class Tokenizer:
        def apply_chat_template(
            self, messages, *, tokenize, add_generation_prompt, **kwargs,
        ):
            assert tokenize is False and add_generation_prompt is True
            return json.dumps(
                {"messages": messages, "kwargs": kwargs},
                sort_keys=True, separators=(",", ":"),
            )

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(name, *, revision):
            calls.append(("tokenizer", name, revision))
            return Tokenizer()

    monkeypatch.setitem(
        sys.modules, "transformers",
        SimpleNamespace(AutoTokenizer=AutoTokenizer),
    )
    evidence = _REAL_CANONICAL_EVIDENCE_LOADER(
        benchmark, model_family="qwen3-1.7b", thinking_mode="auto")

    assert calls[:2] == [
        ("dataset", benchmark),
        ("validate", benchmark, True, len(rows)),
    ]
    assert calls[2] == (
        "tokenizer", "Qwen/Qwen3-1.7B",
        eval_gpu.MODEL_REVISIONS["qwen3-1.7b"],
    )
    assert evidence["gold_by_problem"][0] == "1"
    assert set(evidence["identity_by_problem"]) == set(range(len(rows)))
    assert len(evidence["prompt_template_sha256"]) == 64


def test_valid_exact_p1_matrix_is_signed_and_hard_mean_is_canonical_four_way(
    tmp_path,
):
    values = {
        "AIME24": 0.0,
        "AIME25": 0.0,
        "MATH500": 1.0,
        "Minerva-Math": 1.0,
        # These appendix diagnostics must not move the headline aggregate.
        "AMC23": 1.0,
        "GSM8K": 1.0,
    }
    root = _make_matrix_root(tmp_path, benchmark_values=values)
    matrix = eval_matrix.assemble_publication_matrix(
        root, run_name="run", method_name="Base", model_family="qwen3-1.7b")
    assert matrix["headline_eligible"] is True
    assert matrix["benchmark_order"] == list(eval_matrix.PUBLICATION_BENCHMARKS)
    assert matrix["hard_mean_benchmarks"] == list(
        eval_matrix.HARD_MEAN_BENCHMARKS)
    assert matrix["hard_mean"] == pytest.approx(0.5)
    assert matrix["hard_mean"] != pytest.approx(sum(values.values()) / 6)
    assert eval_matrix.verify_publication_matrix_signature(matrix) == (
        matrix["signature"]["payload_sha256"])
    tampered = copy.deepcopy(matrix)
    tampered["hard_mean"] += 0.01
    with pytest.raises(eval_matrix.MatrixValidationError, match="signature mismatch"):
        eval_matrix.verify_publication_matrix_signature(tampered)


def test_valid_exact_p2_matrix_is_aime_only_secondary_qwen_anchor(tmp_path):
    root = _make_matrix_root(
        tmp_path,
        protocol="P2",
        run_name="p2-run",
        benchmark_values={"AIME24": 0.0, "AIME25": 1.0},
    )
    matrix = eval_matrix.assemble_publication_matrix(
        root,
        protocol="P2",
        run_name="p2-run",
        method_name="Base",
        model_family="qwen3-1.7b",
    )
    assert matrix["schema_version"] == (
        eval_matrix.QWEN_REPORT_P2_MATRIX_SCHEMA_VERSION)
    assert matrix["protocol"] == "P2"
    assert matrix["matrix_role"] == "secondary_qwen_report_anchor"
    assert matrix["headline_eligible"] is False
    assert matrix["benchmark_order"] == list(
        eval_matrix.QWEN_REPORT_P2_BENCHMARKS)
    assert matrix["anchor_mean"] == pytest.approx(0.5)
    assert "hard_mean" not in matrix
    assert eval_matrix.verify_qwen_report_p2_matrix_signature(matrix) == (
        matrix["signature"]["payload_sha256"])
    assert eval_matrix.verify_evaluation_matrix_signature(matrix) == (
        matrix["signature"]["payload_sha256"])
    with pytest.raises(eval_matrix.MatrixValidationError, match="schema_version"):
        eval_matrix.verify_publication_matrix_signature(matrix)


def test_p2_matrix_rejects_non_aime_file_membership(tmp_path):
    root = _make_matrix_root(tmp_path, protocol="P2")
    _write_json(root / "MATH500" / "aggregate_metrics.json", {})
    with pytest.raises(
        eval_matrix.MatrixValidationError,
        match="two AIME-only Qwen-report P2 benchmarks",
    ):
        eval_matrix.assemble_publication_matrix(root, protocol="P2")


def test_base_matrix_rejects_local_model_artifact_even_with_base_label(tmp_path):
    root = _make_matrix_root(tmp_path, method="Base", artifact=True)
    with pytest.raises(
        eval_matrix.MatrixValidationError,
        match="Base canonical model_local_or_hf_path|Base model artifact",
    ):
        eval_matrix.assemble_publication_matrix(root)


def test_matrix_rejects_arbitrary_untrained_publication_method(tmp_path):
    root = _make_matrix_root(tmp_path, method="arbitrary-untrained")
    with pytest.raises(
        eval_matrix.MatrixValidationError,
        match="unsupported publication method",
    ):
        eval_matrix.assemble_publication_matrix(root)


def test_base_matrix_rejects_noncanonical_checkpoint_hash(tmp_path):
    root = _make_matrix_root(tmp_path)
    _mutate_pair(
        root,
        "AIME24",
        mutate_manifest=lambda value: value["model"].__setitem__(
            "checkpoint_sha256", "c" * 64),
        mutate_aggregate=lambda value: value.__setitem__(
            "checkpoint_hash", "c" * 64),
    )
    with pytest.raises(
        eval_matrix.MatrixValidationError,
        match="Base canonical checkpoint SHA256",
    ):
        eval_matrix.assemble_publication_matrix(root)


def test_resigned_matrix_cannot_relabel_base_as_untrained_method(tmp_path):
    matrix = eval_matrix.assemble_publication_matrix(_make_matrix_root(tmp_path))
    matrix["shared_identity"]["method_name"] = "arbitrary-untrained"
    _resign(matrix)
    with pytest.raises(
        eval_matrix.MatrixValidationError,
        match="unsupported publication method",
    ):
        eval_matrix.verify_publication_matrix_signature(matrix)


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("schema_version", "tmx_gpu.publication_matrix_v1", "schema_version"),
        ("hard_mean_benchmarks", list(eval_matrix.PUBLICATION_BENCHMARKS),
         "hard-mean benchmarks"),
        ("hard_mean_scale", "percent", "hard-mean scale"),
        ("hard_mean", 1.0, "recomputed four-benchmark hard mean"),
    ],
)
def test_resigned_noncanonical_matrix_contract_is_rejected(
    tmp_path, field, bad_value, message,
):
    values = {
        "AIME24": 0.0,
        "AIME25": 0.0,
        "MATH500": 1.0,
        "Minerva-Math": 1.0,
        "AMC23": 1.0,
        "GSM8K": 1.0,
    }
    matrix = eval_matrix.assemble_publication_matrix(
        _make_matrix_root(tmp_path, benchmark_values=values))
    matrix[field] = bad_value
    _resign(matrix)
    with pytest.raises(eval_matrix.MatrixValidationError, match=message):
        eval_matrix.verify_publication_matrix_signature(matrix)


def test_publication_identity_hashes_every_dynamic_campaign_validator():
    manifest = _manifest("AIME24")
    assert set(manifest["code"]["executing_file_sha256"]) == set(
        eval_gpu.EXECUTING_CODE_FILES)
    assert {
        "tmx_gpu/common.py",
        "tmx_gpu/dapo_evidence.py",
        "tmx_gpu/c9_evidence.py",
        "tmx_gpu/mrme_contract.py",
        "tmx_gpu/filter_pool.py",
        "tmx_gpu/sft_gate_gpu.py",
        "tmx_gpu/vllm_pool.py",
        "tmx_jax/data.py",
    }.issubset(eval_gpu.EXECUTING_CODE_FILES)
    executing = eval_gpu.executing_code_hashes()
    assert executing["tmx_gpu/dapo_evidence.py"] == hashlib.sha256(
        (eval_gpu.REPO_ROOT / "tmx_gpu/dapo_evidence.py").read_bytes()).hexdigest()
    missing = copy.deepcopy(manifest)
    del missing["code"]["executing_file_sha256"]["tmx_gpu/dapo_evidence.py"]
    with pytest.raises(
            eval_matrix.MatrixValidationError, match="executing-code hash membership"):
        eval_matrix._shared_identity(
            missing, _aggregate("AIME24", missing, 0.0))


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("container_image_identity", "vllm/vllm-openai:v0.24.0",
         "container image identity"),
        ("runtime_stack_versions", {
            **eval_gpu.PUBLICATION_RUNTIME_STACK_VERSIONS,
            "transformers": "5.12.2",
        }, "runtime stack"),
    ],
)
def test_matrix_rejects_noncanonical_evaluation_environment(
    field, bad_value, message,
):
    manifest = _manifest("AIME24")
    manifest[field] = bad_value
    aggregate = _aggregate("AIME24", manifest, 1.0)
    with pytest.raises(eval_matrix.MatrixValidationError, match=message):
        eval_matrix._shared_identity(manifest, aggregate)


def test_matrix_rejects_manifest_aggregate_container_mismatch(tmp_path):
    root = _make_matrix_root(tmp_path)
    _mutate_pair(
        root, "AIME24",
        mutate_aggregate=lambda value: value.__setitem__(
            "container_image_identity", "different-image"),
    )
    with pytest.raises(
            eval_matrix.MatrixValidationError,
            match="aggregate/manifest container_image_identity"):
        eval_matrix.assemble_publication_matrix(root)


def test_resigned_matrix_cannot_change_shared_evaluation_environment(tmp_path):
    matrix = eval_matrix.assemble_publication_matrix(_make_matrix_root(tmp_path))
    matrix["shared_identity"]["container_image_identity"] = "different-image"
    _resign(matrix)
    with pytest.raises(
            eval_matrix.MatrixValidationError,
            match="evaluation container image identity"):
        eval_matrix.verify_publication_matrix_signature(matrix)


def _with_different_valid_receipt_digest(environment: dict) -> dict:
    value = copy.deepcopy(environment)
    binding = value["frozen_invariants"]["orchestrator_receipt_binding"]
    binding["source"]["artifact_digest"] = "different-receipt-digest"
    binding["artifact_files"]["source"][
        "artifact_digest"] = "different-receipt-digest"
    file_unsigned = {
        "version": binding["artifact_files"]["version"],
        "source": binding["artifact_files"]["source"],
        "files": binding["artifact_files"]["files"],
    }
    binding["artifact_files"]["file_manifest_sha256"] = \
        common.canonical_json_sha256(file_unsigned)
    binding_unsigned = {
        key: item for key, item in binding.items() if key != "binding_sha256"
    }
    binding["binding_sha256"] = common.canonical_json_sha256(binding_unsigned)
    return value


@pytest.mark.parametrize("variant", ["receipt_source", "receipt_digest", "native"])
@pytest.mark.parametrize("protocol", ["P1", "P2"])
def test_matrix_rejects_mixed_receipt_or_native_facts(
    tmp_path, variant, protocol,
):
    root = _make_matrix_root(tmp_path, protocol=protocol)
    if variant == "receipt_source":
        environment = receipt_fixtures.evaluation_environment(
            run_name="different-run",
            code_ref="entity/project/code:v1",
            code_digest=CODE_DIGEST,
        )
    elif variant == "receipt_digest":
        environment = _with_different_valid_receipt_digest(
            receipt_fixtures.evaluation_environment(
                run_name="run",
                code_ref="entity/project/code:v1",
                code_digest=CODE_DIGEST,
            )
        )
    else:
        environment = receipt_fixtures.evaluation_environment(
            run_name="run",
            code_ref="entity/project/code:v1",
            code_digest=CODE_DIGEST,
            family="A100",
        )
    environment_sha = eval_gpu.payload_sha256(environment)

    def mutate_manifest(manifest):
        manifest["evaluation_execution_environment"] = environment
        manifest["evaluation_execution_environment_sha256"] = environment_sha

    def mutate_aggregate(aggregate):
        aggregate["evaluation_execution_environment"] = environment
        aggregate["evaluation_execution_environment_sha256"] = environment_sha

    _mutate_pair(root, "AIME25", mutate_manifest, mutate_aggregate)
    with pytest.raises(eval_matrix.MatrixValidationError, match="mixed"):
        eval_matrix.assemble_publication_matrix(root, protocol=protocol)


def test_resigned_matrix_rejects_tampered_native_receipt_facts(tmp_path):
    matrix = eval_matrix.assemble_publication_matrix(_make_matrix_root(tmp_path))
    environment = matrix["shared_identity"][
        "evaluation_execution_environment"]
    environment["compatibility_facts"]["native_runtime"][
        "driver_versions"].pop()
    matrix["shared_identity"][
        "evaluation_execution_environment_sha256"] = \
        eval_gpu.payload_sha256(environment)
    _resign(matrix)
    with pytest.raises(
        eval_matrix.MatrixValidationError,
        match=r"execution environment.*driver",
    ):
        eval_matrix.verify_publication_matrix_signature(matrix)


def _resign(matrix: dict) -> None:
    payload = {key: value for key, value in matrix.items() if key != "signature"}
    matrix["signature"] = {
        "algorithm": "sha256-canonical-json",
        "payload_sha256": eval_matrix._canonical_sha256(payload),
    }


def _eval_artifact_materializer(root: Path, matrix: dict, *, mutation=None):
    by_ref = {
        matrix["per_benchmark"][benchmark]["eval_results_artifact"][
            "artifact_ref"]: benchmark
        for benchmark in matrix["benchmark_order"]
    }

    def materialize(ref: str, destination: Path) -> Path:
        benchmark = by_ref[ref]
        for name in (
            "aggregate_metrics.json", "evaluation_manifest.json", "rollouts.jsonl",
        ):
            shutil.copy2(root / benchmark / name, destination / name)
        if mutation is not None:
            mutation(benchmark, destination)
        return destination

    return materialize


def test_remote_eval_artifact_verification_resolves_all_six_immutable_refs(tmp_path):
    root = _make_matrix_root(tmp_path)
    matrix = eval_matrix.assemble_publication_matrix(root)
    expected = {
        evidence["eval_results_artifact"]["artifact_ref"]:
            evidence["eval_results_artifact"]["artifact_digest"]
        for evidence in matrix["per_benchmark"].values()
    }
    calls = []

    def resolve(ref):
        calls.append(ref)
        return expected[ref]

    resolved = eval_matrix.verify_remote_eval_artifacts(
        matrix, resolve, _eval_artifact_materializer(root, matrix))
    assert list(resolved) == list(eval_matrix.PUBLICATION_BENCHMARKS)
    assert calls == [
        matrix["per_benchmark"][benchmark]["eval_results_artifact"]["artifact_ref"]
        for benchmark in eval_matrix.PUBLICATION_BENCHMARKS
    ]


def test_remote_eval_artifact_verification_supports_two_benchmark_p2(tmp_path):
    root = _make_matrix_root(tmp_path, protocol="P2")
    matrix = eval_matrix.assemble_publication_matrix(root, protocol="P2")
    expected = {
        evidence["eval_results_artifact"]["artifact_ref"]:
            evidence["eval_results_artifact"]["artifact_digest"]
        for evidence in matrix["per_benchmark"].values()
    }
    resolved = eval_matrix.verify_remote_eval_artifacts(
        matrix,
        lambda ref: expected[ref],
        _eval_artifact_materializer(root, matrix),
    )
    assert list(resolved) == list(eval_matrix.QWEN_REPORT_P2_BENCHMARKS)


def test_remote_eval_artifact_digest_mismatch_is_rejected(tmp_path):
    matrix = eval_matrix.assemble_publication_matrix(_make_matrix_root(tmp_path))
    with pytest.raises(eval_matrix.MatrixValidationError, match="remote eval artifact digest"):
        eval_matrix.verify_remote_eval_artifacts(
            matrix, lambda _ref: "remote-drift",
            lambda _ref, _destination: pytest.fail(
                "member download must not run after digest mismatch"),
        )


def test_remote_eval_artifact_mutable_alias_is_rejected_before_resolution(tmp_path):
    matrix = eval_matrix.assemble_publication_matrix(_make_matrix_root(tmp_path))
    matrix["per_benchmark"]["AIME24"]["eval_results_artifact"][
        "artifact_ref"] = "entity/project/eval-AIME24:latest"
    _resign(matrix)
    calls = []
    with pytest.raises(eval_matrix.MatrixValidationError, match="immutable :vN"):
        eval_matrix.verify_remote_eval_artifacts(
            matrix, lambda ref: calls.append(ref) or "unused",
            lambda _ref, _destination: pytest.fail(
                "member download must not run for mutable aliases"),
        )
    assert calls == []


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("wrong", "remote eval artifact member rollouts.jsonl SHA256"),
        ("missing", "remote eval artifact membership"),
        ("extra", "remote eval artifact membership"),
    ],
)
def test_matching_remote_digest_cannot_hide_wrong_missing_or_extra_member_bytes(
    tmp_path, case, message,
):
    root = _make_matrix_root(tmp_path)
    matrix = eval_matrix.assemble_publication_matrix(root)
    expected_digests = {
        item["eval_results_artifact"]["artifact_ref"]:
            item["eval_results_artifact"]["artifact_digest"]
        for item in matrix["per_benchmark"].values()
    }

    def mutate(benchmark, destination):
        if benchmark != "AIME24":
            return
        if case == "wrong":
            (destination / "rollouts.jsonl").write_text("substituted bytes\n")
        elif case == "missing":
            (destination / "evaluation_manifest.json").unlink()
        else:
            (destination / "untracked.txt").write_text("extra")

    with pytest.raises(eval_matrix.MatrixValidationError, match=message):
        eval_matrix.verify_remote_eval_artifacts(
            matrix,
            lambda ref: expected_digests[ref],
            _eval_artifact_materializer(root, matrix, mutation=mutate),
        )


@pytest.mark.parametrize("missing", ["aggregate_metrics.json", "evaluation_manifest.json"])
def test_missing_source_is_rejected(tmp_path, missing):
    root = _make_matrix_root(tmp_path)
    (root / "AIME24" / missing).unlink()
    with pytest.raises(eval_matrix.MatrixValidationError, match="membership"):
        eval_matrix.assemble_publication_matrix(root)


def test_duplicate_or_extra_benchmark_source_is_rejected(tmp_path):
    root = _make_matrix_root(tmp_path)
    extra = root / "duplicate-AIME24"
    extra.mkdir()
    for name in ("aggregate_metrics.json", "evaluation_manifest.json"):
        (extra / name).write_bytes((root / "AIME24" / name).read_bytes())
    with pytest.raises(eval_matrix.MatrixValidationError, match="membership"):
        eval_matrix.assemble_publication_matrix(root)


@pytest.mark.parametrize(
    "case,match",
    [
        ("duplicate", "duplicate rollout cell"),
        ("nonterminal", "not terminal-success"),
        ("manifest", "wrong manifest digest"),
        ("verifier", "fails verifier replay"),
        ("metric", "avg@64 mismatch"),
    ],
)
def test_rollout_matrix_is_replayed_not_trusted(tmp_path, case, match):
    root = _make_matrix_root(tmp_path)
    bench = root / "AIME24"
    rollouts_path = bench / "rollouts.jsonl"
    lines = rollouts_path.read_text().splitlines()
    first = json.loads(lines[0])
    if case == "duplicate":
        lines.append(lines[0])
    elif case == "nonterminal":
        first["finish_reason"] = "error"
        lines[0] = json.dumps(first)
    elif case == "manifest":
        first["evaluation_manifest_sha256"] = "0" * 64
        lines[0] = json.dumps(first)
    elif case == "verifier":
        first["correct"] = False
        lines[0] = json.dumps(first)
    elif case == "metric":
        aggregate_path = bench / "aggregate_metrics.json"
        aggregate = json.loads(aggregate_path.read_text())
        aggregate["avg@64"] = 0.5
        _write_json(aggregate_path, aggregate)
    rollouts_path.write_text("\n".join(lines) + "\n")
    with pytest.raises(eval_matrix.MatrixValidationError, match=match):
        eval_matrix.assemble_publication_matrix(root)


def _rehash_local_eval_receipt(bench: Path) -> None:
    receipt_path = bench / "evaluation_artifact_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["source_file_sha256"] = {
        name: eval_matrix._file_sha256(bench / name)
        for name in (
            "aggregate_metrics.json", "rollouts.jsonl",
            "evaluation_manifest.json",
        )
    }
    _write_json(receipt_path, receipt)


def test_locally_rehashed_ground_truth_substitution_is_rejected(tmp_path):
    """Self-consistent fake labels cannot be promoted as canonical results."""
    root = _make_matrix_root(tmp_path)
    bench = root / "AIME24"
    rollouts_path = bench / "rollouts.jsonl"
    rows = [json.loads(line) for line in rollouts_path.read_text().splitlines()]
    for row in rows:
        row["gold"] = "2"
        row["pred"] = "2"
        row["correct"] = True
        row["text"] = "fabricated reasoning \\boxed{2}"
    rollouts_path.write_text("".join(
        json.dumps(row, sort_keys=True) + "\n" for row in rows))
    _rehash_local_eval_receipt(bench)

    with pytest.raises(
        eval_matrix.MatrixValidationError,
        match="ground truth differs from the pinned canonical row",
    ):
        eval_matrix.assemble_publication_matrix(root)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("canonical_row_sha256", "canonical_row_sha256"),
        ("problem_sha256", "problem_sha256"),
        ("prompt_sha256", "prompt_sha256"),
        ("sample_seed", "sample seed"),
    ],
)
def test_locally_rehashed_problem_prompt_order_or_seed_substitution_is_rejected(
    tmp_path, field, message,
):
    root = _make_matrix_root(tmp_path)
    bench = root / "AIME24"
    rollouts_path = bench / "rollouts.jsonl"
    rows = [json.loads(line) for line in rollouts_path.read_text().splitlines()]
    rows[0][field] = (999999 if field == "sample_seed" else "f" * 64)
    rollouts_path.write_text("".join(
        json.dumps(row, sort_keys=True) + "\n" for row in rows))
    _rehash_local_eval_receipt(bench)

    with pytest.raises(eval_matrix.MatrixValidationError, match=message):
        eval_matrix.assemble_publication_matrix(root)


def test_duplicate_eval_artifact_receipt_is_rejected(tmp_path):
    root = _make_matrix_root(tmp_path)
    source = json.loads(
        (root / "AIME24" / "evaluation_artifact_receipt.json").read_text())
    target_path = root / "AIME25" / "evaluation_artifact_receipt.json"
    target = json.loads(target_path.read_text())
    target["artifact_ref"] = source["artifact_ref"]
    target["artifact_digest"] = source["artifact_digest"]
    _write_json(target_path, target)
    with pytest.raises(eval_matrix.MatrixValidationError, match="duplicated"):
        eval_matrix.assemble_publication_matrix(root)


@pytest.mark.parametrize(
    "case,match",
    [
        ("p2", "protocol"),
        ("inexact", "exact_sample_matrix"),
        ("wrong_count", "sample_count_min"),
        ("dataset", "frozen dataset"),
    ],
)
def test_p2_inexact_wrong_count_and_dataset_drift_are_rejected(
    tmp_path, case, match,
):
    root = _make_matrix_root(tmp_path)

    def mutate_manifest(manifest):
        if case == "p2":
            manifest["protocol"] = "P2"
        elif case == "dataset":
            manifest["benchmark_dataset"]["revision"] = "moving"

    def mutate_aggregate(aggregate):
        if case == "inexact":
            aggregate["exact_sample_matrix"] = False
        elif case == "wrong_count":
            aggregate["successful_sample_count_min"] = 63

    _mutate_pair(
        root, "AIME24",
        mutate_manifest=mutate_manifest,
        mutate_aggregate=mutate_aggregate,
    )
    with pytest.raises(eval_matrix.MatrixValidationError, match=match):
        eval_matrix.assemble_publication_matrix(root)


@pytest.mark.parametrize(
    "field,new_value,match",
    [
        ("checkpoint_sha256", "c" * 64, "mixed|Base canonical checkpoint"),
        ("code_digest", "different-code", "mixed|execution environment"),
        ("prompt", "d" * 64, "mixed"),
        ("verifier", "other-verifier", "mixed"),
    ],
)
def test_mixed_checkpoint_code_prompt_or_verifier_is_rejected(
    tmp_path, field, new_value, match,
):
    root = _make_matrix_root(tmp_path)

    def mutate_manifest(manifest):
        if field == "checkpoint_sha256":
            manifest["model"]["checkpoint_sha256"] = new_value
        elif field == "code_digest":
            manifest["code"]["artifact_digest"] = new_value
        elif field == "prompt":
            manifest["prompt_template_sha256"] = new_value
        else:
            manifest["verifier_version"] = new_value

    def mutate_aggregate(aggregate):
        if field == "checkpoint_sha256":
            aggregate["checkpoint_hash"] = new_value
        elif field == "code_digest":
            aggregate["code_artifact_digest"] = new_value
        elif field == "prompt":
            aggregate["prompt_template_hash"] = new_value
        else:
            aggregate["verifier_version"] = new_value

    _mutate_pair(root, "AIME25", mutate_manifest, mutate_aggregate)
    with pytest.raises(eval_matrix.MatrixValidationError, match=match):
        eval_matrix.assemble_publication_matrix(root)


@pytest.mark.parametrize(
    "method",
    [
        "integrated_multiscout",
        "disjoint_shard_annealed_mrme",
        eval_matrix.EXPDIS_SINGLESCOUT_MULTIROUND_METHOD,
    ],
)
def test_canonical_multiexplorer_methods_require_exact_signed_binding(
    tmp_path, method,
):
    binding = _training_binding(method)
    root = _make_matrix_root(
        tmp_path, method=method, artifact=True,
        training_binding_sha256=binding["binding_sha256"],
    )
    with pytest.raises(eval_matrix.MatrixValidationError, match="campaign binding"):
        eval_matrix.assemble_publication_matrix(root)
    matrix = eval_matrix.assemble_publication_matrix(
        root, training_campaign_binding=binding)
    assert matrix["shared_identity"]["method_name"] == method
    assert matrix["upstream_training_evidence"] == binding


@pytest.mark.parametrize(
    "missing,match",
    [
        ("sft_artifact", "immutable SFT artifact"),
        ("scout_artifact", "immutable Scout artifact"),
        ("central_training_artifact", "training-time Central artifact"),
    ],
)
def test_single_scout_binding_requires_all_three_durable_stage_artifacts(
    missing, match,
):
    valid = _training_binding(eval_matrix.EXPDIS_SINGLESCOUT_METHOD)
    assert eval_matrix.validate_training_campaign_binding(valid) == valid
    unsigned = {
        key: copy.deepcopy(value) for key, value in valid.items()
        if key not in {"version", "binding_sha256", missing}
    }
    tampered = eval_matrix._signed_canonical_payload(
        eval_matrix.TRAINING_CAMPAIGN_BINDING_VERSION, unsigned)
    with pytest.raises(eval_matrix.MatrixValidationError, match=match):
        eval_matrix.validate_training_campaign_binding(tampered)


def test_default_base_label_cannot_launder_a_trained_expdis_artifact(tmp_path):
    binding = _training_binding("integrated_multiscout")
    root = _make_matrix_root(
        tmp_path, method="Base", artifact=True,
        training_binding_sha256=binding["binding_sha256"],
    )
    with pytest.raises(
        eval_matrix.MatrixValidationError,
        match=(
            "campaign/evaluation method|non-ExpDis evaluation|"
            "Base canonical model_local_or_hf_path"),
    ):
        eval_matrix.assemble_publication_matrix(
            root, training_campaign_binding=binding)


def test_unrelated_campaign_artifact_cannot_bind_evaluated_model(tmp_path):
    binding = _training_binding(
        "disjoint_shard_annealed_mrme", model_digest="unrelated-model-digest")
    root = _make_matrix_root(
        tmp_path, method="disjoint_shard_annealed_mrme", artifact=True,
        training_binding_sha256=binding["binding_sha256"],
    )
    with pytest.raises(eval_matrix.MatrixValidationError, match="model artifact digest"):
        eval_matrix.assemble_publication_matrix(
            root, training_campaign_binding=binding)


def test_scalar_upstream_identifiers_are_not_completion_evidence(tmp_path):
    root = _make_matrix_root(tmp_path, method="ExpDis", artifact=True)
    with pytest.raises(eval_matrix.MatrixValidationError, match="campaign binding"):
        eval_matrix.assemble_publication_matrix(
            root,
            upstream_training_campaign_artifact_ref="entity/project/campaign:v1",
            upstream_training_campaign_artifact_digest="campaign-digest",
            upstream_final_model_artifact_digest="model-artifact-digest",
        )


@pytest.mark.parametrize(
    "metadata,expected",
    [
        ({"method": "integrated_multiscout"}, "integrated_multiscout"),
        ({"method": "disjoint_shard_annealed_mrme"},
         "disjoint_shard_annealed_mrme"),
        ({
            "method": eval_matrix.EXPDIS_SINGLESCOUT_MULTIROUND_METHOD,
            "method_version":
                eval_matrix.EXPDIS_SINGLESCOUT_MULTIROUND_METHOD_VERSION,
        }, eval_matrix.EXPDIS_SINGLESCOUT_MULTIROUND_METHOD),
        ({
            "expdis_stabilization_mode": "selected_batch_fixed_predictor_then_one_update",
            "c9_execution_profile": "full",
            "stabilization_cell": "C5",
            "scientific_variant": "expdis_c9_selected_batch_rnd",
            "c9_pipeline_execution_contract_sha256": "1" * 64,
            "c9_central_lineage_sha256": "2" * 64,
        }, "expdis_single_scout"),
    ],
)
def test_trained_method_is_derived_from_immutable_artifact_metadata(
    metadata, expected,
):
    assert eval_matrix.infer_expdis_method_from_artifact_metadata(
        metadata, where="test artifact") == expected


def test_partial_trained_metadata_fails_closed_instead_of_becoming_base():
    with pytest.raises(eval_matrix.MatrixValidationError, match="partial"):
        eval_matrix.infer_expdis_method_from_artifact_metadata(
            {"c9_execution_profile": "full"}, where="test artifact")


def test_cli_removes_stale_matrix_on_failure(tmp_path, monkeypatch):
    root = tmp_path / "missing"
    root.mkdir()
    output = root / "publication_matrix.json"
    output.write_text('{"stale": true}\n')
    monkeypatch.setattr(sys, "argv", [
        "eval_matrix.py", "--eval-root", str(root), "--output", str(output),
    ])
    with pytest.raises(eval_matrix.MatrixValidationError):
        eval_matrix.main()
    assert not output.exists()


def test_launcher_waits_verifies_upload_before_headline_promotion():
    launcher = Path(__file__).resolve().parents[1] / "launch" / "base_eval_job.sh"
    text = launcher.read_text()
    subprocess.run(["bash", "-n", str(launcher)], check=True)
    assert 'rm -f "$PUBLICATION_MATRIX"' in text
    assert "python3 -m tmx_gpu.eval_matrix" in text
    wait = text.index("logged.wait()")
    download = text.index("logged.download(root=str(verify_root))")
    headline = text.index('run.summary["headline_eligible"] = is_headline')
    assert wait < download < headline
    remote_resolution = text.index("verify_remote_eval_artifacts(")
    matrix_wandb_init = text.index("run = wandb.init(", remote_resolution)
    assert remote_resolution < matrix_wandb_init < headline
    assert 'is_headline = protocol == "P1"' in text
    assert 'artifact_role = (' in text
    assert '"eval-publication-matrix" if is_headline' in text
    assert '"eval-qwen-report-p2-anchor"' in text
    assert 'BENCHMARKS="AIME24 AIME25"' in text
    assert '--protocol "$PROTOCOL"' in text
    assert 'verify_evaluation_matrix_signature' in text
    derive = text.index("validate_training_campaign_evidence(")
    server = text.index("# ---- start one vLLM server per GPU")
    benchmark = text.index("python3 tmx_gpu/eval_gpu.py")
    assert derive < server < benchmark
    sft_resolution = text.index(
        'campaign_root / "round_01" / "sft_model_artifact.json"')
    scout_resolution = text.index(
        '"final_artifact_binding.json"', sft_resolution)
    central_resolution = text.index(
        '"final_artifact_binding.json"', scout_resolution + 1)
    assert sft_resolution < scout_resolution < central_resolution < derive < server
    for artifact_path in (
        "/workspace/sft_model_artifact",
        "/workspace/scout_model_artifact",
        "/workspace/central_training_artifact",
    ):
        assert artifact_path in text
    assert 'METHOD_NAME_INPUT="${METHOD_NAME:-}"' in text
    assert "model_method != derived_method" in text
    assert "training_campaign_binding_sha256" in text
    assert "validate_training_campaign_binding(upstream)" in text
    independent_replay = text.index(
        "independently_revalidated = validate_training_campaign_evidence(")
    assert independent_replay < matrix_wandb_init < headline
    promotion_bindings = text.index("artifact_bindings = [")
    promotion_downloads = text.index(
        'for label in ("campaign", "model", "sft", "scout", "central_training")')
    assert promotion_bindings < promotion_downloads < independent_replay
    assert '("sft", upstream["sft_artifact"])' in text
    assert '("scout", upstream["scout_artifact"])' in text
    assert '("central_training", upstream["central_training_artifact"])' in text
    assert 'root = pathlib.Path(f"/workspace/promotion_{label}_artifact")' in text
    assert 'promotion_roots["campaign"]' in text
    assert 'promotion_roots["model"]' in text
    assert eval_gpu.PUBLICATION_EVAL_CONTAINER_IMAGE_IDENTITY in text
    assert '--container-image-identity "$EVAL_CONTAINER_IMAGE_IDENTITY"' in text
    for package, version in (
        ("wandb", "0.28.0"),
        ("datasets", "5.0.0"),
        ("hf-transfer", "0.1.9"),
        ("requests", "2.34.2"),
        ("huggingface-hub", "1.21.0"),
        ("numpy", "2.2.6"),
        ("safetensors", "0.8.0"),
    ):
        assert f'"{package}=={version}"' in text
    assert "missing_required = sorted(set(EXECUTING_CODE_FILES)" in text
    assert 'relative.startswith(("tmx_gpu/", "tmx_jax/"))' in text
