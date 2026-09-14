"""Adversarial tests for C9 MR-ME row and behavior-gate provenance."""

from __future__ import annotations

import json
import inspect
from argparse import Namespace
from pathlib import Path

import pytest

from tmx_gpu import (c9_evidence, common, grpo_gpu, mrme_contract,
                     pipeline_gpu, sft_gate_gpu)
from tmx_gpu.tests import orchestrator_receipt_fixtures as receipt_fixtures
from tmx_jax import data as data_mod


def _contract(*, rounds: int, scouts: int):
    schedule = (0.5,) if rounds == 1 else tuple(
        0.5 if index == 0 else 0.25 for index in range(rounds))
    return mrme_contract.build_contract(
        model="Qwen/Qwen3-1.7B",
        model_revision=common.CANONICAL_TRAINING_MODEL_REVISION,
        dataset="dapo_math_17k",
        dataset_identity=data_mod.training_dataset_identity("dapo_math_17k"),
        max_train_examples=20_000,
        rounds=rounds,
        scouts_per_round=scouts,
        lambda_schedule=schedule,
        base_seed=0,
        scout_steps=200,
        central_steps=100,
        prompts_per_step=4,
        num_generations=16,
        pooling_policy=mrme_contract.MRME_POOLING_POLICY,
        max_accepted=500,
        harvest_max_passes=0,
        stabilization_profile=mrme_contract.C9_MULTIEXPLORER_PROFILE,
        training_contract={"test_fixture": True},
        code_identity={"artifact_ref": "code:v1", "digest": "test"},
    )


def _selected_row():
    return {
        "problem_id": "p",
        "prompt_text": "problem",
        "completion_text": "reasoning \\boxed{0}",
        "ground_truth": "0",
        "correctness_reward": 1.0,
        "novelty_reward": 0.1,
        "blended_reward": 1.05,
        "is_correct": True,
        "finish_reason": "stop",
        "completion_token_length": 4,
        "clipped": False,
        "terminated": True,
        "valid_answer": True,
        "expdis_stabilization_mode": "c9_selected_batch_update",
        "scientific_variant":
            mrme_contract.C9_MULTIEXPLORER_SCIENTIFIC_VARIANT,
        "c9_execution_profile": "full",
        "stabilization_cell": "C5",
        "rnd_lifecycle": "c9_selected_batch_update",
        "novelty_normalization": "selected_batch_zscore",
        "novelty_normalization_center": 0.5,
        "novelty_normalization_scale": 0.25,
        "novelty_normalization_effective_scale": 0.25,
        "rnd_update_scope": "final_selected_learner_rows",
    }


def _slot_args(contract, *, round_index: int, scout_index: int, shard_sha: str):
    scout = contract["rounds_spec"][round_index - 1]["scouts"][scout_index - 1]
    return Namespace(
        phase="scout",
        round=round_index,
        seed=int(scout["seed"]),
        scientific_variant=contract["scientific_variant"],
        mrme_contract_sha256=contract["contract_sha256"],
        mrme_method=contract["method"],
        mrme_stabilization_profile=contract["stabilization_profile"],
        mrme_shard_manifest_sha256=shard_sha,
        mrme_scout_index=scout_index,
        mrme_scouts_per_round=int(contract["scouts_per_round"]),
    )


def test_trajectory_record_binds_only_mrme_scout_slots():
    contract = _contract(rounds=1, scouts=3)
    args = _slot_args(
        contract, round_index=1, scout_index=2, shard_sha="a" * 64)
    provenance = grpo_gpu._mrme_trajectory_provenance(args)
    record = grpo_gpu.trajectory_record(
        _selected_row(), 1, mrme_trajectory_provenance=provenance)
    assert record["mrme_contract_sha256"] == contract["contract_sha256"]
    assert record["mrme_scout_index"] == 2
    assert record["mrme_scout_seed"] == 1

    args.phase = "central"
    assert grpo_gpu._mrme_trajectory_provenance(args) is None
    single = Namespace(phase="scout")
    assert grpo_gpu._mrme_trajectory_provenance(single) is None
    ordinary = grpo_gpu.trajectory_record(_selected_row(), 1)
    assert not any(key.startswith("mrme_") for key in ordinary)


@pytest.mark.parametrize(
    ("rounds", "scouts", "round_index", "scout_index", "wrong_index"),
    [(1, 3, 1, 1, 2), (2, 2, 2, 2, 1)],
)
def test_c9_trajectory_facts_reject_copied_scout_slot(
    tmp_path, rounds, scouts, round_index, scout_index, wrong_index,
):
    contract = _contract(rounds=rounds, scouts=scouts)
    shard_sha = "b" * 64
    provenance = grpo_gpu._mrme_trajectory_provenance(_slot_args(
        contract, round_index=round_index, scout_index=scout_index,
        shard_sha=shard_sha))
    path = tmp_path / "explorer_trajectories.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for step in range(1, 201):
            for index in range(64):
                row = grpo_gpu.trajectory_record(
                    {**_selected_row(), "problem_id": f"p-{step}-{index}"},
                    step, mrme_trajectory_provenance=provenance)
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    facts = mrme_contract._c9_multi_explorer_trajectory_facts(
        str(path), contract=contract, round_index=round_index,
        scout_index=scout_index, shard_manifest_sha256=shard_sha)
    assert facts["row_count"] == 12_800
    with pytest.raises(ValueError, match="campaign-slot provenance drift"):
        mrme_contract._c9_multi_explorer_trajectory_facts(
            str(path), contract=contract, round_index=round_index,
            scout_index=wrong_index, shard_manifest_sha256=shard_sha)


def _gate_fixture(root: Path):
    root.mkdir(parents=True)
    selection = [{"problem_id": f"p{index}"}
                 for index in range(c9_evidence.GATE_NUM_PROMPTS)]
    rollouts = []
    for mode in sft_gate_gpu.PROMPT_MODES:
        for prompt_idx, selected in enumerate(selection):
            for sample_idx in range(c9_evidence.GATE_SAMPLES_PER_PROMPT):
                rollouts.append({
                    "metric_contract_version":
                        sft_gate_gpu.METRIC_CONTRACT_VERSION,
                    "problem_id": selected["problem_id"],
                    "prompt_mode": mode,
                    "sample_idx": sample_idx,
                    "sample_seed": (
                        c9_evidence.GATE_GENERATION_SEED
                        + prompt_idx * 100003 + sample_idx),
                    "finish_reason": "stop",
                    "generation_error_detail": None,
                    "completion_tokens": 5,
                    "completion_text": "reasoning \\boxed{1}",
                    "correctness_diagnostic_only": False,
                })
    selection_path = root / "heldout_selection.jsonl"
    rollout_path = root / "rollouts.jsonl"
    aggregate_path = root / "aggregate_metrics.json"
    with selection_path.open("w", encoding="utf-8") as handle:
        for row in selection:
            handle.write(json.dumps(row) + "\n")
    with rollout_path.open("w", encoding="utf-8") as handle:
        for row in rollouts:
            handle.write(json.dumps(row) + "\n")
    selection_sha = common.canonical_json_sha256(selection)
    sampling = {"fixture": "sampling"}
    metrics, gate = sft_gate_gpu.recompute_gate_from_rollouts(
        rollouts, num_prompts=c9_evidence.GATE_NUM_PROMPTS,
        samples_per_prompt=c9_evidence.GATE_SAMPLES_PER_PROMPT,
        generation_seed=c9_evidence.GATE_GENERATION_SEED,
        max_completion_tokens=c9_evidence.GATE_MAX_COMPLETION_TOKENS,
        ordered_problem_ids=[row["problem_id"] for row in selection],
        thresholds=dict(c9_evidence.GATE_THRESHOLDS))
    common.atomic_write_json(str(aggregate_path), {
        "selection": {"heldout_selection_sha256": selection_sha},
        "sampling": sampling,
        "metrics_by_prompt_mode": metrics,
        "health_gate": gate,
    })
    evidence = {
        "heldout_selection_sha256": selection_sha,
        "heldout_selection_file_sha256": common.file_sha256(
            str(selection_path)),
        "raw_rollout_count": len(rollouts),
        "raw_rollouts_file_sha256": common.file_sha256(str(rollout_path)),
        "aggregate_file_sha256": common.file_sha256(str(aggregate_path)),
        "sampling": sampling,
    }
    evidence_path = root / "evidence.json"
    common.atomic_write_json(str(evidence_path), evidence)
    return evidence_path, evidence


def test_behavior_gate_support_files_are_rehashed(tmp_path):
    evidence_path, evidence = _gate_fixture(tmp_path / "gate")
    result = mrme_contract.validate_c9_behavior_gate_support_files(
        str(evidence_path), evidence)
    assert result["selection_row_count"] == 32
    assert result["rollout_row_count"] == 128


def test_behavior_gate_support_missing_and_drift_fail_closed(tmp_path):
    missing_path, missing_evidence = _gate_fixture(tmp_path / "missing")
    (missing_path.parent / "rollouts.jsonl").unlink()
    with pytest.raises(ValueError, match="supporting file is missing"):
        mrme_contract.validate_c9_behavior_gate_support_files(
            str(missing_path), missing_evidence)

    drift_path, drift_evidence = _gate_fixture(tmp_path / "drift")
    with (drift_path.parent / "aggregate_metrics.json").open(
            "a", encoding="utf-8") as handle:
        handle.write(" \n")
    with pytest.raises(ValueError, match="supporting file drifted"):
        mrme_contract.validate_c9_behavior_gate_support_files(
            str(drift_path), drift_evidence)


def test_behavior_gate_recomputes_verdict_from_raw_rollouts(tmp_path):
    evidence_path, evidence = _gate_fixture(tmp_path / "resigned-wrong")
    aggregate_path = evidence_path.parent / "aggregate_metrics.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate["metrics_by_prompt_mode"]["training"][
        "termination_rate"] = 0.5
    common.atomic_write_json(str(aggregate_path), aggregate)
    evidence["aggregate_file_sha256"] = common.file_sha256(
        str(aggregate_path))
    common.atomic_write_json(str(evidence_path), evidence)
    with pytest.raises(ValueError, match="raw-rollout replay"):
        mrme_contract.validate_c9_behavior_gate_support_files(
            str(evidence_path), evidence)


def test_round_source_validation_invokes_support_rehash(
    tmp_path, monkeypatch,
):
    contract = _contract(rounds=1, scouts=3)
    init_identity = mrme_contract.model_handoff_identity(
        contract["model"], model_revision=contract["model_revision"])
    calls = []
    sources = []
    for scout_index in (1, 2, 3):
        trajectory = tmp_path / f"scout-{scout_index}.jsonl"
        trajectory.write_text("{}\n", encoding="utf-8")
        gate = {"slot": scout_index}
        gate_path = tmp_path / f"gate-{scout_index}.json"
        common.atomic_write_json(str(gate_path), gate)
        final_identity = {"slot": scout_index}
        record = {
            "version": "gpu_mrme_scout_source_v1",
            "contract_sha256": contract["contract_sha256"],
            "round_index": 1,
            "scout_index": scout_index,
            "seed": scout_index - 1,
            "lambda_novelty": 0.5,
            "trajectory_path": str(trajectory),
            "trajectory_sha256": common.file_sha256(str(trajectory)),
            "trajectory_rows": 12_800,
            "rows_per_step": 64,
            "shard_manifest_sha256": "c" * 64,
            "scout_init_identity": init_identity,
            "scout_final": str(tmp_path / f"model-{scout_index}"),
            "scout_final_identity": final_identity,
            "scout_final_artifact": {
                "slot": scout_index, "model_file_manifest": {}},
            "source_kind": "raw_scout_trajectory_pool",
            "per_scout_preselection": False,
            "c9_multi_explorer_scout_evidence": {"behavior_gate": gate},
            "c9_behavior_gate_path": str(gate_path),
        }
        record["record_sha256"] = common.canonical_json_sha256(record)
        sources.append(record)

    monkeypatch.setattr(
        mrme_contract, "validate_scout_final",
        lambda path, **kwargs: {"slot": int(path.rsplit("-", 1)[1])})
    monkeypatch.setattr(
        mrme_contract, "validate_scout_runtime_contract",
        lambda *args, **kwargs: None)
    monkeypatch.setattr(
        mrme_contract, "validate_c9_multi_explorer_scout_evidence",
        lambda value, **kwargs: value)
    monkeypatch.setattr(
        mrme_contract, "validate_grpo_final_artifact_binding",
        lambda value, **kwargs: value)
    monkeypatch.setattr(
        common, "verify_directory_file_manifest",
        lambda *args, **kwargs: {})
    monkeypatch.setattr(
        mrme_contract, "validate_c9_behavior_gate_support_files",
        lambda path, evidence: calls.append((path, evidence)))

    validated = mrme_contract.validate_round_scout_sources(
        contract, 1, sources,
        expected_shard_manifest_sha256="c" * 64,
        expected_scout_init_identity=init_identity)
    assert len(validated) == 3
    assert [evidence["slot"] for _, evidence in calls] == [1, 2, 3]


def _pooled_sft_gate_fixture(contract):
    accepted = {
        "version": "semantic_sft_jsonl_digest_v1",
        "row_count": 500,
        "sha256": "d" * 64,
    }
    policy_manifest = {
        "version": common.DIRECTORY_FILE_MANIFEST_VERSION,
        "files": [{
            "relative_path": "model.safetensors",
            "size": 1,
            "sha256": "e" * 64,
        }],
        "sha256": "f" * 64,
    }
    runtime = {
        "model_init": contract["model"],
        "base_model_revision": contract["model_revision"],
        "optimizer": "adamw",
        "learning_rate": 5e-6,
        "epochs": 2,
        "batch_size": 1,
        "grad_accum": 1,
        "seed": 0,
        "termination_mode": "append_eos",
        "parameter_dtype": "float32",
        "loss": "completion_only_token_mean_cross_entropy",
        "execution_environment": receipt_fixtures.environment(),
    }
    sampling = {
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
    }
    gate = mrme_contract.sign_c9_multi_explorer_sft_gate_evidence({
        "contract_sha256": contract["contract_sha256"],
        "method": contract["method"],
        "stabilization_profile": contract["stabilization_profile"],
        "scientific_variant": contract["scientific_variant"],
        "round_index": 1,
        "scout_index": 0,
        "seed": 0,
        "c9_execution_profile": "full",
        "stabilization_cell": "C5",
        "gate_stage": "sft_fixed_endpoint",
        "selection_uses_correctness": False,
        "panel_role": c9_evidence.GATE_PANEL_ROLE,
        "panel_held_out_from_scout_training":
            c9_evidence.GATE_PANEL_HELD_OUT_FROM_SCOUT_TRAINING,
        "passed": True,
        "sampling": sampling,
        "thresholds": dict(c9_evidence.GATE_THRESHOLDS),
        "heldout_selection_sha256": "a" * 64,
        "heldout_selection_file_sha256": "b" * 64,
        "raw_rollout_count": (
            2 * c9_evidence.GATE_NUM_PROMPTS
            * c9_evidence.GATE_SAMPLES_PER_PROMPT),
        "raw_rollouts_file_sha256": "c" * 64,
        "aggregate_file_sha256": "d" * 64,
        "model_file_manifest": policy_manifest,
        "model_runtime_contract": runtime,
        "model_step": 1000,
        "accepted_library_digest": accepted,
        "accepted_file_sha256": "e" * 64,
        "code_artifact": contract["code_identity"],
    })
    return gate, runtime, policy_manifest, accepted


def _resign_pooled_gate(value):
    payload = dict(value)
    payload.pop("version", None)
    payload.pop("evidence_sha256", None)
    return mrme_contract.sign_c9_multi_explorer_sft_gate_evidence(payload)


def _pooled_central_lineage(contract):
    gate, runtime, policy_manifest, accepted = _pooled_sft_gate_fixture(
        contract)
    full_manifest = {
        **policy_manifest,
        "files": [
            *policy_manifest["files"],
            {
                "relative_path": "sft_provenance.json",
                "size": 1,
                "sha256": "1" * 64,
            },
        ],
        "sha256": "2" * 64,
    }
    artifact = {
        "version": "pipeline_sft_model_artifact_v1",
        "durable_wandb_upload": True,
        "source": {
            "artifact_ref": "entity/project/c9-pooled-sft:v3",
            "artifact_digest": "remote-digest",
        },
        "artifact_type": "sft-model",
        "artifact_metadata": {
            "round": 1,
            "stage_mode": "c9_multi_explorer_fixed_endpoint_trained",
            "model_file_manifest_sha256": full_manifest["sha256"],
            "accepted_file_sha256": "e" * 64,
            "training_sequence_sha256": "3" * 64,
        },
        "artifact_subdir": "model",
        "model_file_manifest": full_manifest,
        "policy_file_manifest": policy_manifest,
        "accepted_library_digest": accepted,
        "accepted_file_sha256": "e" * 64,
        "training_sequence_digest": {"sha256": "3" * 64},
        "round": 1,
        "stage_mode": "c9_multi_explorer_fixed_endpoint_trained",
    }
    artifact["binding_sha256"] = common.canonical_json_sha256(artifact)
    central_runtime = {
        "phase": "central", "round": 1, "lambda_novelty": 0.0,
        "expected_steps": 100,
        "execution_environment": receipt_fixtures.environment(),
    }
    trajectory = {"artifact_count": 3, "sha256": "4" * 64}
    lineage = mrme_contract.build_c9_multi_explorer_central_lineage(
        contract=contract, round_index=1,
        shard_manifest_sha256="5" * 64,
        trajectory_source_digest=trajectory,
        accepted_library_digest=accepted,
        sft_behavior_gate=gate,
        sft_model_file_manifest=policy_manifest,
        sft_model_artifact=artifact,
        central_runtime_contract=central_runtime)
    return lineage, gate, runtime, policy_manifest, accepted, trajectory


def test_pooled_sft_gate_and_durable_central_lineage_validate():
    contract = _contract(rounds=1, scouts=3)
    lineage, gate, runtime, manifest, accepted, trajectory = \
        _pooled_central_lineage(contract)
    assert mrme_contract.validate_c9_multi_explorer_sft_gate_evidence(
        gate, contract=contract, round_index=1,
        expected_runtime_contract=runtime,
        expected_code_identity=contract["code_identity"],
        expected_model_manifest=manifest,
        expected_accepted_digest=accepted,
        expected_selection_sha256="a" * 64) == gate
    assert mrme_contract.validate_c9_multi_explorer_central_lineage(
        lineage, contract=contract, round_index=1,
        expected_shard_manifest_sha256="5" * 64,
        expected_trajectory_source_digest=trajectory,
        expected_accepted_digest=accepted,
        expected_central_runtime_contract=lineage[
            "central_runtime_contract"]) == lineage


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda gate: gate.pop("aggregate_file_sha256"), "aggregate_file"),
        (lambda gate: gate.__setitem__("passed", False), "passed=False"),
    ],
)
def test_pooled_sft_gate_missing_or_failed_evidence_fails_closed(
    mutation, message,
):
    contract = _contract(rounds=1, scouts=3)
    gate, _runtime, _manifest, _accepted = _pooled_sft_gate_fixture(contract)
    mutation(gate)
    gate = _resign_pooled_gate(gate)
    with pytest.raises(ValueError, match=message):
        mrme_contract.validate_c9_multi_explorer_sft_gate_evidence(
            gate, contract=contract, round_index=1)


def test_pooled_sft_gate_and_lineage_reject_unsigned_tampering():
    contract = _contract(rounds=1, scouts=3)
    lineage, gate, *_rest = _pooled_central_lineage(contract)
    gate["heldout_selection_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="SHA256"):
        mrme_contract.validate_c9_multi_explorer_sft_gate_evidence(
            gate, contract=contract, round_index=1)
    lineage["central_runtime_contract"] = {
        **lineage["central_runtime_contract"], "lambda_novelty": 0.5}
    with pytest.raises(ValueError, match="lineage SHA256"):
        mrme_contract.validate_c9_multi_explorer_central_lineage(
            lineage, contract=contract, round_index=1,
            expected_shard_manifest_sha256="5" * 64)


def test_pooled_sft_gate_failure_is_ordered_before_any_central_launch():
    source = inspect.getsource(pipeline_gpu.run_round)
    gate = source.index("_ensure_c9_multi_pooled_sft_gate")
    central_stage = source.index("# ---- Stage 5: Central GRPO")
    launch = source.index("grpo_gpu.run_grpo", central_stage)
    assert gate < central_stage < launch


def _central_probe_row(
    global_step, local_step, contract, *, round_index=1, shard_sha="d" * 64,
):
    return {
        "global_step": global_step,
        "phase": "central_grpo",
        "round": round_index,
        "lambda": 0.0,
        "lambda_novelty": 0.0,
        "num_scouts": int(contract["scouts_per_round"]),
        "rl_validity_mode": "c8",
        "expdis_stabilization_mode": "c9_selected_batch_update",
        "scientific_variant":
            mrme_contract.C9_MULTIEXPLORER_SCIENTIFIC_VARIANT,
        "c9_execution_profile": "full",
        "stabilization_cell": "C5",
        "novelty_normalization": "selected_batch_zscore",
        "mrme_contract_sha256": contract["contract_sha256"],
        "mrme_method": contract["method"],
        "mrme_stabilization_profile": contract["stabilization_profile"],
        "mrme_round_index": round_index,
        "mrme_scout_index": 0,
        "mrme_scouts_per_round": int(contract["scouts_per_round"]),
        "mrme_shard_manifest_sha256": shard_sha,
        "mrme_stage_seed": int(contract["base_seed"]) + round_index - 1,
        "eval/probe_error_rows": 0.0,
        "eval/probe_protocol": c9_evidence.C9_PROBE_PROTOCOL_ID,
        "eval/probe_headline_eligible": 0.0,
        "eval/probe_metric_scale": "fraction",
        "eval/probe_sample_seed_rule": c9_evidence.C9_PROBE_CONTRACT[
            "sample_seed_rule"],
        "eval/probe_benchmark_revision": c9_evidence.C9_PROBE_CONTRACT[
            "benchmark_revision"],
        "eval/probe_benchmark_rows_sha256": c9_evidence.C9_PROBE_CONTRACT[
            "benchmark_rows_sha256"],
        "eval/probe_prompt_matrix_sha256": "6" * 64,
        "eval/probe_protocol_complete": 1.0,
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
        "eval/probe_policy_step": local_step,
        "eval/probe_staleness_updates": 0,
        "eval/probe_fresh_weights": 1.0,
        "eval/AIME24_avg_at_4": 0.5,
    }


def _scout_probe_row(
    global_step, local_step, contract, *, round_index, scout_index,
    shard_sha="d" * 64,
):
    spec = contract["rounds_spec"][round_index - 1]
    scout = spec["scouts"][scout_index - 1]
    return {
        "global_step": global_step,
        "phase": "scout_grpo",
        "round": round_index,
        "lambda": float(spec["lambda_novelty"]),
        "lambda_novelty": float(spec["lambda_novelty"]),
        "num_scouts": int(contract["scouts_per_round"]),
        "rl_validity_mode": "c8",
        "expdis_stabilization_mode": "c9_selected_batch_update",
        "scientific_variant":
            mrme_contract.C9_MULTIEXPLORER_SCIENTIFIC_VARIANT,
        "c9_execution_profile": "full",
        "stabilization_cell": "C5",
        "novelty_normalization": "selected_batch_zscore",
        "mrme_contract_sha256": contract["contract_sha256"],
        "mrme_method": contract["method"],
        "mrme_stabilization_profile": contract["stabilization_profile"],
        "mrme_round_index": round_index,
        "mrme_scout_index": scout_index,
        "mrme_scouts_per_round": int(contract["scouts_per_round"]),
        "mrme_shard_manifest_sha256": shard_sha,
        "mrme_stage_seed": int(scout["seed"]),
        "eval/probe_error_rows": 0.0,
        "eval/probe_protocol": c9_evidence.C9_PROBE_PROTOCOL_ID,
        "eval/probe_headline_eligible": 0.0,
        "eval/probe_metric_scale": "fraction",
        "eval/probe_sample_seed_rule": c9_evidence.C9_PROBE_CONTRACT[
            "sample_seed_rule"],
        "eval/probe_benchmark_revision": c9_evidence.C9_PROBE_CONTRACT[
            "benchmark_revision"],
        "eval/probe_benchmark_rows_sha256": c9_evidence.C9_PROBE_CONTRACT[
            "benchmark_rows_sha256"],
        "eval/probe_prompt_matrix_sha256": "6" * 64,
        "eval/probe_protocol_complete": 1.0,
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
        "eval/probe_policy_step": local_step,
        "eval/probe_staleness_updates": 0,
        "eval/probe_fresh_weights": 1.0,
        "eval/AIME24_avg_at_4": 0.5,
    }


@pytest.mark.parametrize(
    ("rounds", "scouts", "round_index"),
    [(1, 3, 1), (2, 2, 2)],
)
def test_completed_campaign_replays_every_scout_probe_slot(
    tmp_path, rounds, scouts, round_index,
):
    contract = _contract(rounds=rounds, scouts=scouts)
    shard_sha = "d" * 64
    cycle_steps = scouts * 200 + 100
    round_offset = (round_index - 1) * cycle_steps
    rows = []
    sources = []
    for scout_index in range(1, scouts + 1):
        scout_offset = round_offset + (scout_index - 1) * 200
        rows.extend(_scout_probe_row(
            scout_offset + local_step, local_step, contract,
            round_index=round_index, scout_index=scout_index,
            shard_sha=shard_sha)
            for local_step in range(25, 201, 25))
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    for scout_index in range(1, scouts + 1):
        scout_offset = round_offset + (scout_index - 1) * 200
        probe_history = mrme_contract._c9_multi_explorer_probe_facts(
            str(metrics), global_step_offset=scout_offset, contract=contract,
            round_index=round_index, scout_index=scout_index,
            shard_manifest_sha256=shard_sha)
        sources.append({
            "scout_index": scout_index,
            "c9_multi_explorer_scout_evidence": {
                "probe_history": probe_history,
            },
        })

    mrme_contract._validate_c9_multi_explorer_scout_probe_replays(
        str(metrics), contract=contract, round_index=round_index,
        sources=sources, shard_manifest_sha256=shard_sha)

    # Preserve a structurally valid probe row but change its measured value
    # after the Scout source record was signed. Completion must replay the raw
    # metrics and reject the now-stale stored digest.
    rows[-1]["eval/AIME24_avg_at_4"] = 0.25
    metrics.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match=(
            rf"Scout {scouts} probe history differs from durable metrics replay")):
        mrme_contract._validate_c9_multi_explorer_scout_probe_replays(
            str(metrics), contract=contract, round_index=round_index,
            sources=sources, shard_manifest_sha256=shard_sha)


def test_c9_mrme_rejects_consistent_prompt_drift_across_k_and_rounds(tmp_path):
    contract = _contract(rounds=2, scouts=2)
    shard_sha = "d" * 64
    rows = []
    histories = []
    cycle_steps = 2 * 200 + 100
    for round_index in (1, 2):
        round_offset = (round_index - 1) * cycle_steps
        for scout_index in (1, 2):
            scout_offset = round_offset + (scout_index - 1) * 200
            rows.extend(_scout_probe_row(
                scout_offset + local_step, local_step, contract,
                round_index=round_index, scout_index=scout_index,
                shard_sha=shard_sha)
                for local_step in range(25, 201, 25))
        central_offset = round_offset + 2 * 200
        central_rows = [_central_probe_row(
            central_offset + local_step, local_step, contract,
            round_index=round_index, shard_sha=shard_sha)
            for local_step in range(25, 101, 25)]
        if round_index == 2:
            # Rebuild the whole R2-Central summary around a different but
            # internally stable prompt matrix.  Every local phase check passes;
            # only the campaign-wide K x R binding should reject it.
            for row in central_rows:
                row["eval/probe_prompt_matrix_sha256"] = "7" * 64
        rows.extend(central_rows)
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    for round_index in (1, 2):
        round_offset = (round_index - 1) * cycle_steps
        for scout_index in (1, 2):
            histories.append(mrme_contract._c9_multi_explorer_probe_facts(
                str(metrics),
                global_step_offset=round_offset + (scout_index - 1) * 200,
                contract=contract, round_index=round_index,
                scout_index=scout_index,
                shard_manifest_sha256=shard_sha))
        histories.append(mrme_contract._c9_multi_explorer_central_probe_facts(
            str(metrics), global_step_offset=round_offset + 2 * 200,
            contract=contract, round_index=round_index,
            shard_manifest_sha256=shard_sha))

    with pytest.raises(ValueError, match="prompt matrices differ"):
        c9_evidence.require_shared_probe_prompt_matrix(
            histories, context="test C9 MR-ME campaign")


def test_multi_central_probe_history_requires_all_exact_fresh_boundaries(
    tmp_path,
):
    metrics = tmp_path / "metrics.jsonl"
    offset = 600
    contract = _contract(rounds=1, scouts=3)
    shard_sha = "d" * 64
    rows = [_central_probe_row(offset + step, step, contract)
            for step in range(25, 101, 25)]
    metrics.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    facts = mrme_contract._c9_multi_explorer_central_probe_facts(
        str(metrics), global_step_offset=offset, contract=contract,
        round_index=1, shard_manifest_sha256=shard_sha)
    assert facts["row_count"] == 4
    assert facts["expected_global_steps"] == [625, 650, 675, 700]

    metrics.write_text(
        "".join(json.dumps(row) + "\n" for row in rows[:-1]),
        encoding="utf-8")
    with pytest.raises(ValueError, match="lacks every exact probe boundary"):
        mrme_contract._c9_multi_explorer_central_probe_facts(
            str(metrics), global_step_offset=offset, contract=contract,
            round_index=1, shard_manifest_sha256=shard_sha)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mrme_scout_index", 2),
        ("mrme_shard_manifest_sha256", "e" * 64),
        ("mrme_stage_seed", 9),
    ],
)
def test_multi_central_probe_history_rejects_wrong_campaign_slot(
    tmp_path, field, value,
):
    contract = _contract(rounds=1, scouts=3)
    shard_sha = "d" * 64
    offset = 600
    rows = [_central_probe_row(offset + step, step, contract)
            for step in range(25, 101, 25)]
    rows[0][field] = value
    metrics = tmp_path / f"wrong-{field}.jsonl"
    metrics.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="probe"):
        mrme_contract._c9_multi_explorer_central_probe_facts(
            str(metrics), global_step_offset=offset, contract=contract,
            round_index=1, shard_manifest_sha256=shard_sha)
