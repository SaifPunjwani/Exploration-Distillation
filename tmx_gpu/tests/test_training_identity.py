"""Immutable base-model and training-dataset identity boundaries."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import requests

from tmx_gpu import common, grpo_gpu, mrme_contract, pipeline_gpu, sft_gpu
from tmx_gpu.vllm_pool import VLLMPool
from tmx_jax import data as data_mod


def _grpo_args(*extra: str):
    return grpo_gpu.build_arg_parser().parse_args([
        "--output-dir", "/tmp/training-identity-test", *extra,
    ])


def _c9_args(*extra: str):
    return _grpo_args(
        "--model", common.CANONICAL_TRAINING_MODEL_ID,
        "--model-revision", common.CANONICAL_TRAINING_MODEL_REVISION,
        "--container-image-identity",
        grpo_gpu.CANONICAL_TRAINING_CONTAINER_IMAGE,
        "--phase", "scout", "--rl-validity-mode", "c8",
        "--lambda-novelty", "0.5",
        "--expdis-stabilization-mode", grpo_gpu.C9_RND_LIFECYCLE,
        "--novelty-normalization", "selected_batch_zscore",
        "--scientific-variant", grpo_gpu.C9_SCIENTIFIC_VARIANT,
        "--code-source-manifest", "/tmp/code.json",
        *extra,
    )


def test_remote_revision_kwargs_never_touch_local_handoffs(tmp_path):
    assert common.remote_revision_kwargs(
        "Qwen/Qwen3-1.7B", "abc123") == {"revision": "abc123"}
    local = tmp_path / "sft-final"
    local.mkdir()
    assert common.remote_revision_kwargs(str(local), "abc123") == {}
    assert common.remote_revision_kwargs("./future-checkpoint", "abc123") == {}


def test_managed_vllm_initial_remote_gets_revision_and_reload_clears_it(
    tmp_path, monkeypatch,
):
    commands = []

    def no_stale(*_args, **_kwargs):
        raise requests.RequestException("no server")

    class Proc:
        pid = 123
        returncode = None

        def poll(self):
            return None

    monkeypatch.setattr(requests, "get", no_stale)
    monkeypatch.setattr(
        subprocess, "Popen",
        lambda cmd, **_kwargs: commands.append(list(cmd)) or Proc())
    pool = VLLMPool(
        model_dir=common.CANONICAL_TRAINING_MODEL_ID,
        model_revision=common.CANONICAL_TRAINING_MODEL_REVISION,
        gpu_ids=[1], served_model_name="base", log_dir=str(tmp_path),
        settle_seconds=0.0,
    )
    pool.procs = [None]
    pool.log_files = [None]
    pool._launch_one(0)
    revision_index = commands[0].index("--revision")
    assert commands[0][revision_index + 1] == \
        common.CANONICAL_TRAINING_MODEL_REVISION

    monkeypatch.setattr(pool, "_kill", lambda: None)
    monkeypatch.setattr(pool, "launch", lambda: None)
    monkeypatch.setattr(pool, "wait_healthy", lambda timeout=1200.0: None)
    local = tmp_path / "central-final"
    local.mkdir()
    pool.reload(str(local))
    assert pool.model_revision == ""


@pytest.mark.parametrize("revision", ["", "wrong-commit"])
def test_c9_rejects_missing_or_wrong_base_revision(revision):
    with pytest.raises(ValueError, match="immutable canonical training checkpoint"):
        grpo_gpu.validate_expdis_stabilization(
            _c9_args("--model-revision", revision),
            phase="scout", lambda_novelty=0.5)


def test_c9_mrme_round2_scout_requires_exact_prior_central(tmp_path):
    previous_shard_sha = "b" * 64
    scouts = 2
    contract = mrme_contract.build_contract(
        model=common.CANONICAL_TRAINING_MODEL_ID,
        model_revision=common.CANONICAL_TRAINING_MODEL_REVISION,
        dataset="dapo_math_17k",
        dataset_identity=data_mod.training_dataset_identity("dapo_math_17k"),
        max_train_examples=20_000, rounds=2, scouts_per_round=scouts,
        lambda_schedule=(0.5, 0.25), base_seed=0,
        scout_steps=200, central_steps=100, prompts_per_step=4,
        num_generations=16, pooling_policy="coverage_pool_c8",
        max_accepted=500, harvest_max_passes=0,
        stabilization_profile=mrme_contract.C9_MULTIEXPLORER_PROFILE,
        training_contract={"fixture": True},
        code_identity={"artifact_ref": "code:v1", "digest": "abc"},
    )
    contract_path = tmp_path / "mrme_contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    contract_sha = contract["contract_sha256"]
    method = contract["method"]
    profile = contract["stabilization_profile"]
    central = tmp_path / "round_01_central"
    central.mkdir()
    (central / "config.json").write_text("{}\n", encoding="utf-8")
    model_manifest = common.directory_file_manifest(str(central))
    runtime = {
        "phase": "central",
        "base_model_revision": common.CANONICAL_TRAINING_MODEL_REVISION,
        "scientific_variant":
            mrme_contract.C9_MULTIEXPLORER_SCIENTIFIC_VARIANT,
        "lambda_novelty": 0.0,
        "seed": 0,
        "multi_explorer_contract": {
            "contract_sha256": contract_sha,
            "method": method,
            "stabilization_profile": profile,
            "scout_index": 0,
            "scouts_per_round": scouts,
            "shard_manifest_sha256": previous_shard_sha,
        },
    }
    provenance = {
        "method": method,
        "stabilization_profile": profile,
        "scientific_variant":
            mrme_contract.C9_MULTIEXPLORER_SCIENTIFIC_VARIANT,
        "mrme_contract_sha256": contract_sha,
        "mrme_shard_manifest_sha256": previous_shard_sha,
        "mrme_scout_index": 0,
        "mrme_scouts_per_round": scouts,
        "stage_seed": 0,
        "phase": "central",
    }
    state = {
        "phase": "central",
        "round": 1,
        "step": 100,
        "runtime_contract": runtime,
        "mrme_provenance": provenance,
        "checkpoint_file_manifest": model_manifest,
    }
    state_path = central / "trainer_state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    args = _c9_args(
        "--model", str(central), "--phase", "scout", "--round", "2",
        "--c9-execution-profile", "full", "--stabilization-cell", "C5",
        "--scientific-variant",
        mrme_contract.C9_MULTIEXPLORER_SCIENTIFIC_VARIANT,
    )
    args.mrme_contract_sha256 = contract_sha
    args.mrme_contract_json = str(contract_path)
    args.mrme_method = method
    args.mrme_stabilization_profile = profile
    args.mrme_scout_index = 1
    args.mrme_scouts_per_round = scouts
    args.mrme_shard_manifest_sha256 = "c" * 64
    grpo_gpu.validate_expdis_stabilization(
        args, phase="scout", lambda_novelty=0.5)

    state["round"] = 0
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match="immediately prior Central"):
        grpo_gpu.validate_expdis_stabilization(
            args, phase="scout", lambda_novelty=0.5)


def test_grpo_and_sft_contracts_bind_model_revision_and_dataset_identity(tmp_path):
    args = _grpo_args(
        "--model-revision", common.CANONICAL_TRAINING_MODEL_REVISION)
    runtime = grpo_gpu.build_runtime_contract(
        args, resolved_lr=5e-6, resolved_steps=200)
    assert runtime["base_model_revision"] == \
        common.CANONICAL_TRAINING_MODEL_REVISION
    assert runtime["training_dataset_identity"] == \
        data_mod.training_dataset_identity("dapo_math_17k")

    accepted = tmp_path / "accepted.jsonl"
    accepted.write_text("", encoding="utf-8")
    sargs = sft_gpu.build_arg_parser().parse_args([
        "--accepted", str(accepted), "--output-dir", str(tmp_path / "sft"),
        "--model-revision", common.CANONICAL_TRAINING_MODEL_REVISION,
    ])
    sft_runtime = sft_gpu.build_sft_runtime_contract(
        sargs, resolved_max_total_len=18432)
    assert sft_runtime["base_model_revision"] == \
        common.CANONICAL_TRAINING_MODEL_REVISION


def test_resume_rejects_model_revision_runtime_drift():
    old_args = _grpo_args("--model-revision", "old-commit")
    new_args = _grpo_args("--model-revision", "new-commit")
    old = grpo_gpu.build_runtime_contract(
        old_args, resolved_lr=5e-6, resolved_steps=200)
    new = grpo_gpu.build_runtime_contract(
        new_args, resolved_lr=5e-6, resolved_steps=200)
    recorded = {
        "phase": "scout", "model": old_args.model,
        "config_tier": grpo_gpu.resolved_config_tier(old_args),
        "rl_validity_mode": "legacy", "external_lineage": None,
        "external_lineage_sha256": None, "runtime_contract": old,
        "code_artifact_binding": None,
    }
    with pytest.raises(RuntimeError, match="runtime_contract"):
        grpo_gpu.check_resume_compat(
            recorded, phase="scout", model=new_args.model,
            config_tier=grpo_gpu.resolved_config_tier(new_args),
            runtime_contract_current=new)


def test_dataset_identities_pin_hub_sources_and_hash_local_bytes(
    tmp_path, monkeypatch,
):
    dapo = data_mod.training_dataset_identity("dapo_math_17k")
    assert dapo["repo_id"] == data_mod.DAPO_DATASET_ID
    assert dapo["config"] == data_mod.DAPO_DATASET_CONFIG
    assert dapo["split"] == "train"
    assert dapo["revision"] == data_mod.DAPO_DATASET_REVISION

    local = tmp_path / "train.jsonl"
    local.write_bytes(b'{"problem":"p","answer":"1"}\n')
    monkeypatch.setenv("TMX_TRAIN_DATASET_JSONL", str(local))
    first = data_mod.training_dataset_identity("jsonl:clean")
    assert first["resolved_path"] == str(local.resolve())
    assert first["content_sha256"] == hashlib.sha256(local.read_bytes()).hexdigest()
    local.write_bytes(local.read_bytes() + b'{"problem":"q","answer":"2"}\n')
    second = data_mod.training_dataset_identity("jsonl:clean")
    assert second["content_sha256"] != first["content_sha256"]
    assert second["identity_sha256"] != first["identity_sha256"]


def test_mrme_identity_binds_model_revision_and_dataset_source():
    dataset_identity = data_mod.training_dataset_identity("dapo_math_17k")
    contract = mrme_contract.build_contract(
        model=common.CANONICAL_TRAINING_MODEL_ID,
        model_revision=common.CANONICAL_TRAINING_MODEL_REVISION,
        dataset="dapo_math_17k", dataset_identity=dataset_identity,
        max_train_examples=20_000, rounds=1, scouts_per_round=3,
        lambda_schedule=(0.5,), base_seed=0, scout_steps=200,
        central_steps=100, prompts_per_step=4, num_generations=16,
        pooling_policy="coverage_pool_c8", max_accepted=500,
        harvest_max_passes=0,
        stabilization_profile=mrme_contract.C9_MULTIEXPLORER_PROFILE,
        training_contract={"fixture": True},
        code_identity={"artifact_ref": "code:v1", "digest": "abc"},
    )
    assert contract["model_revision"] == \
        common.CANONICAL_TRAINING_MODEL_REVISION
    assert contract["dataset_identity"] == dataset_identity
    tampered = dict(dataset_identity)
    tampered["revision"] = "moving-head"
    with pytest.raises(ValueError, match="identity SHA256 mismatch"):
        mrme_contract.build_contract(
            model=common.CANONICAL_TRAINING_MODEL_ID,
            model_revision=common.CANONICAL_TRAINING_MODEL_REVISION,
            dataset="dapo_math_17k", dataset_identity=tampered,
            max_train_examples=20_000, rounds=1, scouts_per_round=3,
            lambda_schedule=(0.5,), base_seed=0, scout_steps=200,
            central_steps=100, prompts_per_step=4, num_generations=16,
            pooling_policy="coverage_pool_c8", max_accepted=500,
            harvest_max_passes=0,
            stabilization_profile=mrme_contract.C9_MULTIEXPLORER_PROFILE,
            training_contract={"fixture": True},
            code_identity={"artifact_ref": "code:v1", "digest": "abc"},
        )


def test_pipeline_propagates_revision_to_child_trainers(tmp_path):
    args = pipeline_gpu.build_arg_parser().parse_args([
        "--model-revision", common.CANONICAL_TRAINING_MODEL_REVISION,
        "--output-dir", str(tmp_path), "--run-name", "identity",
    ])
    gargs = pipeline_gpu._grpo_args(
        args, phase="scout", model=args.model,
        output_dir=str(tmp_path / "scout"), lam=0.5, steps=1,
        round_idx=1, global_step_offset=0)
    sargs = pipeline_gpu._sft_args(
        args, accepted=str(tmp_path / "accepted.jsonl"),
        output_dir=str(tmp_path / "sft"), round_idx=1, lam=0.5,
        global_step_offset=1)
    assert gargs.model_revision == common.CANONICAL_TRAINING_MODEL_REVISION
    assert sargs.model_revision == common.CANONICAL_TRAINING_MODEL_REVISION


def test_launcher_protects_and_passes_model_revision():
    text = Path("tmx_gpu/launch/expdis_job.sh").read_text(encoding="utf-8")
    assert "--model|--model-revision|" in text
    assert '--model-revision "$MODEL_REVISION"' in text
    assert common.CANONICAL_TRAINING_MODEL_REVISION in text
