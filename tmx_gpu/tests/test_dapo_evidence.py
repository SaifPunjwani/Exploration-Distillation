from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tmx_gpu import common, dapo_evidence, eval_gpu, eval_matrix, grpo_gpu
from tmx_gpu.tests import orchestrator_receipt_fixtures as receipt_fixtures


def _binding():
    value = {
        "version": common.ARTIFACT_DOWNLOAD_MANIFEST_VERSION,
        "source": {
            "artifact_ref": "entity/project/tmx-gpu-code:v29",
            "artifact_digest": "code-digest",
        },
        "files": [{
            "relative_path": "tmx_gpu/grpo_gpu.py",
            "sha256": "a" * 64,
            "size_bytes": 1,
        }],
    }
    value["file_manifest_sha256"] = common.canonical_json_sha256(value)
    return value


def _runtime():
    args = grpo_gpu.build_arg_parser().parse_args([
        "--output-dir", "/tmp/dapo-test",
        "--model", dapo_evidence.MODEL,
        "--container-image-identity",
        grpo_gpu.CANONICAL_TRAINING_CONTAINER_IMAGE,
        "--phase", "central",
        "--scientific-variant", dapo_evidence.SCIENTIFIC_VARIANT,
        "--steps", "100",
        "--lr", "0.000001",
        "--lambda-novelty", "0",
        "--prompts-per-step", "4",
        "--num-generations", "16",
        "--max-prompt-len", "2048",
        "--max-completion-len", "16384",
        "--loss-denominator-len", "16384",
        "--soft-overlong-expected-len", "13107",
        "--soft-overlong-cache-len", "3277",
        "--temperature", "1.0",
        "--top-p", "0.95",
        "--top-k", "20",
        "--updates-per-rollout", "1",
        "--clip-epsilon-low", "0.2",
        "--clip-epsilon-high", "0.28",
        "--rl-validity-mode", "c8",
        "--dynamic-max-attempts", "64",
        "--optimizer", "adamw",
        "--incorrect-novelty-scale", "0.0",
        "--dataset", "dapo_math_17k",
        "--max-train-examples", "20000",
        "--seed", "0",
        "--round", "1",
        "--param-dtype", "float32",
        "--vllm-max-model-len", "36864",
    ])
    args._training_execution_environment = receipt_fixtures.environment(
        run_name=args.run_name)
    return grpo_gpu.build_runtime_contract(
        args, resolved_lr=1e-6, resolved_steps=100)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _download_manifest(root: Path, *, version: str, ref: str, type_: str,
                       metadata: dict) -> dict:
    files = [{
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": common.file_sha256(str(path)),
        "size_bytes": path.stat().st_size,
    } for path in sorted(item for item in root.rglob("*") if item.is_file())]
    return {
        "version": version,
        "source": {"artifact_ref": ref, "artifact_digest": f"digest-{type_}"},
        "artifact_type": type_,
        "artifact_metadata": metadata,
        "files": files,
    }


def _resume_fixture(tmp_path: Path, step: int = 25):
    binding = _binding()
    binding_path = tmp_path / "current-code-binding.json"
    _write_json(binding_path, binding)

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}\n")
    (checkpoint / "optimizer_state.pt").write_bytes(b"optimizer")
    file_manifest = common.directory_file_manifest(
        str(checkpoint), exclude_relative_paths=("trainer_state.json",))
    state = {
        "step": step,
        "phase": "central",
        "model": dapo_evidence.MODEL,
        "rl_validity_mode": "c8",
        "optimizer_state_saved": True,
        "checkpoint_parameter_dtype": "float32",
        "policy_attention_implementation": "sdpa",
        "resolved_lr": 1e-6,
        "config_tier": (
            f"contract+scientific-variant:{dapo_evidence.SCIENTIFIC_VARIANT}"),
        "external_lineage": None,
        "external_lineage_sha256": None,
        "runtime_contract": _runtime(),
        "code_artifact_binding": binding,
        "checkpoint_file_manifest": file_manifest,
    }
    _write_json(checkpoint / "trainer_state.json", state)

    results = tmp_path / "results"
    _write_json(results / "dapo_code_binding.json", binding)
    _write_json(results / f"step_{step:06d}" / "trainer_state.json", state)
    _write_jsonl(results / "metrics.jsonl", [{
        "global_step": learner_step,
        "phase": "central_grpo",
        "train/grpo_loss": -0.01,
    } for learner_step in range(1, step + 1)])
    history_rows = [
        {"step": learner_step, "problem_id": f"p{learner_step}-{i}"}
        for learner_step in range(1, step + 1) for i in range(64)
    ]
    _write_jsonl(results / "explorer_trajectories.jsonl", history_rows)
    _write_jsonl(results / "rollouts_all.jsonl", history_rows)

    checkpoint_manifest = _download_manifest(
        checkpoint,
        version="wandb_checkpoint_download_v1",
        ref="entity/project/dapo-ckpt:v0",
        type_="checkpoint",
        metadata={"step": step, "round": 1, "phase": "central"},
    )
    results_manifest = _download_manifest(
        results,
        version="wandb_results_download_v1",
        ref="entity/project/dapo-progress:v0",
        type_="grpo-progress",
        metadata={
            "step": step,
            "round": 1,
            "phase": "central",
            "model": dapo_evidence.MODEL,
            "scientific_variant": dapo_evidence.SCIENTIFIC_VARIANT,
        },
    )
    checkpoint_manifest_path = tmp_path / "checkpoint-download.json"
    results_manifest_path = tmp_path / "results-download.json"
    _write_json(checkpoint_manifest_path, checkpoint_manifest)
    _write_json(results_manifest_path, results_manifest)
    return SimpleNamespace(
        binding=binding,
        binding_path=binding_path,
        checkpoint=checkpoint,
        checkpoint_manifest_path=checkpoint_manifest_path,
        results=results,
        results_manifest_path=results_manifest_path,
        output=tmp_path / "output",
        step=step,
    )


def _validate(fixture):
    return dapo_evidence.validate_and_restore_resume_pair(
        code_binding_path=str(fixture.binding_path),
        checkpoint_root=str(fixture.checkpoint),
        checkpoint_manifest_path=str(fixture.checkpoint_manifest_path),
        results_root=str(fixture.results),
        results_manifest_path=str(fixture.results_manifest_path),
        output_dir=str(fixture.output),
    )


def test_dapo_resume_pair_validates_before_restoring_jsonls(tmp_path):
    fixture = _resume_fixture(tmp_path)
    record = _validate(fixture)
    assert record["checkpoint_step"] == 25
    for name in ("metrics.jsonl", "explorer_trajectories.jsonl", "rollouts_all.jsonl"):
        assert (fixture.output / name).read_bytes() == (fixture.results / name).read_bytes()


def test_dapo_resume_pair_allows_step100_finalization_only(tmp_path):
    fixture = _resume_fixture(tmp_path, step=100)
    assert _validate(fixture)["checkpoint_step"] == 100


@pytest.mark.parametrize("mutation,match", [
    ("state", "trainer states are not identical"),
    ("metadata", "progress artifact metadata"),
    ("metrics", "metrics must contain exactly one"),
    ("trajectories", "exactly 64 selected"),
    ("alias", "immutable"),
    ("type", "wrong type"),
])
def test_dapo_resume_pair_rejects_unpaired_evidence_before_copy(
        tmp_path, mutation, match):
    fixture = _resume_fixture(tmp_path)
    if mutation == "state":
        state_path = fixture.results / f"step_{fixture.step:06d}" / "trainer_state.json"
        state = json.loads(state_path.read_text())
        state["step"] += 1
        _write_json(state_path, state)
    elif mutation == "metadata":
        manifest = json.loads(fixture.results_manifest_path.read_text())
        manifest["artifact_metadata"]["step"] += 1
        _write_json(fixture.results_manifest_path, manifest)
    elif mutation == "metrics":
        rows = [
            row for row in (
                json.loads(line)
                for line in (fixture.results / "metrics.jsonl").read_text().splitlines()
            )
            if row["global_step"] != fixture.step
        ]
        _write_jsonl(fixture.results / "metrics.jsonl", rows)
    elif mutation == "trajectories":
        rows = [
            json.loads(line)
            for line in (fixture.results / "explorer_trajectories.jsonl")
            .read_text().splitlines()
        ][:-1]
        _write_jsonl(fixture.results / "explorer_trajectories.jsonl", rows)
    else:
        manifest = json.loads(fixture.results_manifest_path.read_text())
        if mutation == "alias":
            manifest["source"]["artifact_ref"] = "entity/project/dapo-progress:latest"
        else:
            manifest["artifact_type"] = "dataset"
        _write_json(fixture.results_manifest_path, manifest)
    # Refresh byte entries only for mutations to artifact members. This lets
    # the semantic pair check—not a stale synthetic SHA—be the rejection site.
    if mutation in {"state", "metrics", "trajectories"}:
        manifest = json.loads(fixture.results_manifest_path.read_text())
        refreshed = _download_manifest(
            fixture.results,
            version="wandb_results_download_v1",
            ref=manifest["source"]["artifact_ref"],
            type_=manifest["artifact_type"],
            metadata=manifest["artifact_metadata"],
        )
        _write_json(fixture.results_manifest_path, refreshed)
    with pytest.raises(ValueError, match=match):
        _validate(fixture)
    assert not (fixture.output / "metrics.jsonl").exists()


def test_dapo_resume_pair_rejects_unproven_local_checkpoint(tmp_path):
    fixture = _resume_fixture(tmp_path)
    (fixture.output / "step_000025").mkdir(parents=True)
    with pytest.raises(ValueError, match="unproven local checkpoints"):
        _validate(fixture)
    assert not (fixture.output / "metrics.jsonl").exists()


def _completion_fixture(tmp_path: Path):
    import torch
    from safetensors.torch import save_file

    code = _binding()
    runtime = _runtime()
    metadata = {
        "step": 100,
        "round": 1,
        "phase": "central",
        "model_revision": common.CANONICAL_TRAINING_MODEL_REVISION,
        "runtime_contract_sha256": common.canonical_json_sha256(runtime),
        "training_dataset_identity_sha256": runtime[
            "training_dataset_identity"]["identity_sha256"],
    }
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}\n")
    (checkpoint / "tokenizer.json").write_text('{"version":"tiny"}\n')
    fp32_tensors = {
        "linear.weight": torch.tensor(
            [[1.00390625, -2.015625], [3.1415927, 0.33333334]],
            dtype=torch.float32),
        "step_counter": torch.tensor([100], dtype=torch.int64),
    }
    save_file(fp32_tensors, checkpoint / "model.safetensors")
    (checkpoint / "optimizer_state.pt").write_bytes(b"optimizer-state")
    checkpoint_manifest = common.directory_file_manifest(
        str(checkpoint), exclude_relative_paths=("trainer_state.json",))
    checkpoint_state = {
        "step": 100,
        "phase": "central",
        "round": 1,
        "model": dapo_evidence.MODEL,
        "model_revision": common.CANONICAL_TRAINING_MODEL_REVISION,
        "rl_validity_mode": "c8",
        "optimizer_state_saved": True,
        "checkpoint_parameter_dtype": "float32",
        "policy_attention_implementation": "sdpa",
        "resolved_lr": 1e-6,
        "config_tier": (
            f"contract+scientific-variant:{dapo_evidence.SCIENTIFIC_VARIANT}"),
        "external_lineage": None,
        "external_lineage_sha256": None,
        "runtime_contract": runtime,
        "code_artifact_binding": code,
        "checkpoint_file_manifest": checkpoint_manifest,
    }
    _write_json(checkpoint / "trainer_state.json", checkpoint_state)

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n")
    (model / "tokenizer.json").write_text('{"version":"tiny"}\n')
    save_file(
        {"linear.weight": fp32_tensors["linear.weight"].to(torch.bfloat16)},
        model / "model-00001-of-00002.safetensors")
    save_file(
        {"step_counter": fp32_tensors["step_counter"]},
        model / "model-00002-of-00002.safetensors")
    _write_json(model / "model.safetensors.index.json", {
        "metadata": {},
        "weight_map": {
            "linear.weight": "model-00001-of-00002.safetensors",
            "step_counter": "model-00002-of-00002.safetensors",
        },
    })
    policy_manifest = common.directory_file_manifest(
        str(model), exclude_relative_paths=("trainer_state.json",))
    final_state = {
        **checkpoint_state,
        "optimizer_state_saved": False,
        "checkpoint_kind": "deployable_model",
        "checkpoint_file_manifest": policy_manifest,
    }
    _write_json(model / "trainer_state.json", final_state)
    model_source = {
        "artifact_ref": "entity/project/dapo-final:v0",
        "artifact_digest": "model-digest",
        "artifact_type": "model",
    }
    final_binding = {
        "version": dapo_evidence.GRPO_FINAL_MODEL_ARTIFACT_BINDING_VERSION,
        "source": model_source,
        "artifact_subdir": ".",
        "model_file_manifest": common.directory_file_manifest(str(model)),
        "policy_file_manifest": policy_manifest,
        "artifact_metadata": metadata,
        "phase": "central",
        "round": 1,
        "step": 100,
        "runtime_contract_sha256": common.canonical_json_sha256(runtime),
        "mrme_provenance": None,
    }
    final_binding["binding_sha256"] = common.canonical_json_sha256(final_binding)

    results = tmp_path / "results"
    _write_json(results / "provenance" / "dapo_code_binding.json", code)
    _write_json(results / "step_000100" / "trainer_state.json", checkpoint_state)
    _write_json(results / "final" / "trainer_state.json", final_state)
    _write_json(results / "final_artifact_binding.json", final_binding)
    _write_jsonl(results / "metrics.jsonl", [{
        "global_step": step,
        "phase": "central_grpo",
        "train/grpo_loss": -0.01,
    } for step in range(1, 101)])
    trajectory_rows = [
        {"step": step, "problem_id": f"p{step}-{index}"}
        for step in range(1, 101) for index in range(64)
    ]
    _write_jsonl(results / "explorer_trajectories.jsonl", trajectory_rows)
    _write_jsonl(results / "rollouts_all.jsonl", trajectory_rows)
    results_metadata = {
        "exit_status": 0,
        "completion_status": "complete",
        "method": dapo_evidence.DAPO_METHOD,
        "scientific_variant": dapo_evidence.SCIENTIFIC_VARIANT,
        "code_artifact_ref": code["source"]["artifact_ref"],
        "code_artifact_digest": code["source"]["artifact_digest"],
        "resume_checkpoint_artifact": "",
        "resume_results_artifact": "",
    }
    kwargs = {
        "model_root": str(model),
        "checkpoint_root": str(checkpoint),
        "results_root": str(results),
        "model_artifact_ref": model_source["artifact_ref"],
        "model_artifact_digest": model_source["artifact_digest"],
        "model_artifact_type": "model",
        "model_artifact_metadata": metadata,
        "checkpoint_artifact_ref": "entity/project/dapo-checkpoint:v3",
        "checkpoint_artifact_digest": "checkpoint-digest",
        "checkpoint_artifact_type": "checkpoint",
        "checkpoint_artifact_metadata": metadata,
        "results_artifact_ref": "entity/project/dapo-results:v1",
        "results_artifact_digest": "results-digest",
        "results_artifact_type": "dapo-results",
        "results_artifact_metadata": results_metadata,
    }
    completion = dapo_evidence.build_completion_binding(**kwargs)
    campaign = tmp_path / "campaign"
    _write_json(
        campaign / dapo_evidence.DAPO_COMPLETION_BINDING_FILENAME, completion)
    campaign_metadata = {
        "completion_status": "complete",
        "method": dapo_evidence.DAPO_METHOD,
        "scientific_variant": dapo_evidence.SCIENTIFIC_VARIANT,
        "completion_binding_sha256": completion["binding_sha256"],
        "model_artifact_ref": kwargs["model_artifact_ref"],
        "model_artifact_digest": kwargs["model_artifact_digest"],
        "checkpoint_artifact_ref": kwargs["checkpoint_artifact_ref"],
        "checkpoint_artifact_digest": kwargs["checkpoint_artifact_digest"],
        "results_artifact_ref": kwargs["results_artifact_ref"],
        "results_artifact_digest": kwargs["results_artifact_digest"],
        "code_artifact_ref": code["source"]["artifact_ref"],
        "code_artifact_digest": code["source"]["artifact_digest"],
    }
    return SimpleNamespace(
        model=model, checkpoint=checkpoint, results=results, campaign=campaign,
        kwargs=kwargs, completion=completion,
        campaign_metadata=campaign_metadata,
    )


def _refresh_final_model_evidence(fixture) -> None:
    policy_manifest = common.directory_file_manifest(
        str(fixture.model), exclude_relative_paths=("trainer_state.json",))
    final_state_path = fixture.model / "trainer_state.json"
    final_state = json.loads(final_state_path.read_text())
    final_state["checkpoint_file_manifest"] = policy_manifest
    _write_json(final_state_path, final_state)
    _write_json(fixture.results / "final" / "trainer_state.json", final_state)
    binding_path = fixture.results / "final_artifact_binding.json"
    binding = json.loads(binding_path.read_text())
    binding["policy_file_manifest"] = policy_manifest
    binding["model_file_manifest"] = common.directory_file_manifest(
        str(fixture.model))
    binding.pop("binding_sha256", None)
    binding["binding_sha256"] = common.canonical_json_sha256(binding)
    _write_json(binding_path, binding)


def _refresh_checkpoint_evidence(fixture) -> None:
    manifest = common.directory_file_manifest(
        str(fixture.checkpoint), exclude_relative_paths=("trainer_state.json",))
    state_path = fixture.checkpoint / "trainer_state.json"
    state = json.loads(state_path.read_text())
    state["checkpoint_file_manifest"] = manifest
    _write_json(state_path, state)
    _write_json(fixture.results / "step_000100" / "trainer_state.json", state)


def test_dapo_completion_campaign_replays_exact_remote_bytes(tmp_path):
    fixture = _completion_fixture(tmp_path)
    validated = dapo_evidence.validate_completion_campaign(
        campaign_root=str(fixture.campaign),
        campaign_artifact_type=dapo_evidence.DAPO_CAMPAIGN_ARTIFACT_TYPE,
        campaign_artifact_metadata=fixture.campaign_metadata,
        **fixture.kwargs,
    )
    assert validated == fixture.completion
    assert validated["history"]["step_count"] == 100
    assert validated["checkpoint_artifact"]["source"]["artifact_type"] == \
        "checkpoint"
    cast = validated["checkpoint_artifact"]["fp32_to_bf16_tensor_cast"]
    assert cast["version"] == "dapo_fp32_to_bf16_tensor_cast_v1"
    assert cast["tensor_count"] == 2


def test_eval_matrix_builds_signed_dapo_publication_lineage(tmp_path):
    fixture = _completion_fixture(tmp_path)
    binding = eval_matrix.validate_training_campaign_evidence(
        fixture.campaign, fixture.model,
        campaign_artifact_ref="entity/project/dapo-campaign:v0",
        campaign_artifact_digest="campaign-digest",
        campaign_artifact_type=dapo_evidence.DAPO_CAMPAIGN_ARTIFACT_TYPE,
        campaign_artifact_metadata=fixture.campaign_metadata,
        model_artifact_ref=fixture.kwargs["model_artifact_ref"],
        model_artifact_digest=fixture.kwargs["model_artifact_digest"],
        model_artifact_type=fixture.kwargs["model_artifact_type"],
        model_artifact_metadata=fixture.kwargs["model_artifact_metadata"],
        code_artifact_ref=_binding()["source"]["artifact_ref"],
        code_artifact_digest=_binding()["source"]["artifact_digest"],
        dapo_results_artifact_root=fixture.results,
        dapo_results_artifact_ref=fixture.kwargs["results_artifact_ref"],
        dapo_results_artifact_digest=fixture.kwargs["results_artifact_digest"],
        dapo_results_artifact_type=fixture.kwargs["results_artifact_type"],
        dapo_results_artifact_metadata=fixture.kwargs[
            "results_artifact_metadata"],
        dapo_checkpoint_artifact_root=fixture.checkpoint,
        dapo_checkpoint_artifact_ref=fixture.kwargs[
            "checkpoint_artifact_ref"],
        dapo_checkpoint_artifact_digest=fixture.kwargs[
            "checkpoint_artifact_digest"],
        dapo_checkpoint_artifact_type=fixture.kwargs[
            "checkpoint_artifact_type"],
        dapo_checkpoint_artifact_metadata=fixture.kwargs[
            "checkpoint_artifact_metadata"],
    )
    assert eval_matrix.validate_training_campaign_binding(binding) == binding
    assert binding["method"] == dapo_evidence.DAPO_METHOD
    assert binding["results_artifact"]["artifact_type"] == "dapo-results"
    assert binding["checkpoint_artifact"]["artifact_type"] == "checkpoint"


@pytest.mark.parametrize("mutation,match", [
    ("tensor_drift", "differs from exact checkpoint cast"),
    ("tensor_dtype", "dtype contract mismatch"),
    ("tensor_key", "tensor-name mismatch"),
    ("checkpoint", "directory file manifest"),
    ("metrics", "metrics must contain exactly one"),
    ("dataset", "dataset|runtime"),
])
def test_dapo_completion_rejects_tampered_model_checkpoint_history_or_dataset(
        tmp_path, mutation, match):
    import torch
    from safetensors.torch import load_file, save_file

    fixture = _completion_fixture(tmp_path)
    final_shard = fixture.model / "model-00001-of-00002.safetensors"
    if mutation == "tensor_drift":
        tensors = load_file(final_shard)
        tensors["linear.weight"][0, 0] += torch.tensor(
            1.0, dtype=torch.bfloat16)
        save_file(tensors, final_shard)
        _refresh_final_model_evidence(fixture)
    elif mutation == "tensor_dtype":
        tensors = load_file(final_shard)
        save_file({
            "linear.weight": tensors["linear.weight"].to(torch.float32),
        }, final_shard)
        _refresh_final_model_evidence(fixture)
    elif mutation == "tensor_key":
        tensors = load_file(final_shard)
        save_file({"renamed.weight": tensors["linear.weight"]}, final_shard)
        index_path = fixture.model / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        del index["weight_map"]["linear.weight"]
        index["weight_map"]["renamed.weight"] = final_shard.name
        _write_json(index_path, index)
        _refresh_final_model_evidence(fixture)
    elif mutation == "checkpoint":
        (fixture.checkpoint / "optimizer_state.pt").write_bytes(b"tampered")
    elif mutation == "metrics":
        rows = (fixture.results / "metrics.jsonl").read_text().splitlines()[:-1]
        (fixture.results / "metrics.jsonl").write_text("\n".join(rows) + "\n")
    else:
        state_path = fixture.checkpoint / "trainer_state.json"
        state = json.loads(state_path.read_text())
        state["runtime_contract"]["training_dataset_identity"]["revision"] = \
            "moving-main"
        _write_json(state_path, state)
    with pytest.raises(ValueError, match=match):
        dapo_evidence.build_completion_binding(**fixture.kwargs)


def test_dapo_completion_rejects_duplicate_or_non_safetensor_policy(tmp_path):
    from safetensors.torch import load_file, save_file

    duplicate = _completion_fixture(tmp_path / "duplicate")
    first = load_file(
        duplicate.model / "model-00001-of-00002.safetensors")
    second_path = duplicate.model / "model-00002-of-00002.safetensors"
    second = load_file(second_path)
    save_file({**second, **first}, second_path)
    _refresh_final_model_evidence(duplicate)
    with pytest.raises(ValueError, match="duplicate"):
        dapo_evidence.build_completion_binding(**duplicate.kwargs)

    unsafe = _completion_fixture(tmp_path / "non_safetensor")
    for path in unsafe.model.glob("*.safetensors*"):
        path.unlink()
    (unsafe.model / "pytorch_model.bin").write_bytes(b"not-safetensors")
    _refresh_final_model_evidence(unsafe)
    with pytest.raises(ValueError, match="non-safetensor"):
        dapo_evidence.build_completion_binding(**unsafe.kwargs)


@pytest.mark.parametrize("mutation,match", [
    ("source_nonfinite", "non-finite"),
    ("final_nonfinite", "non-finite"),
    ("config_drift", "config/tokenizer bytes differ"),
    ("missing_tokenizer", "config/tokenizer bytes differ"),
])
def test_dapo_completion_rejects_nonfinite_or_nonweight_drift(
        tmp_path, mutation, match):
    import torch
    from safetensors.torch import load_file, save_file

    fixture = _completion_fixture(tmp_path)
    if mutation == "source_nonfinite":
        path = fixture.checkpoint / "model.safetensors"
        tensors = load_file(path)
        tensors["linear.weight"][0, 0] = torch.inf
        save_file(tensors, path)
        _refresh_checkpoint_evidence(fixture)
    elif mutation == "final_nonfinite":
        path = fixture.model / "model-00001-of-00002.safetensors"
        tensors = load_file(path)
        tensors["linear.weight"][0, 0] = torch.nan
        save_file(tensors, path)
        _refresh_final_model_evidence(fixture)
    elif mutation == "config_drift":
        (fixture.model / "config.json").write_text('{"drift":true}\n')
        _refresh_final_model_evidence(fixture)
    else:
        (fixture.model / "tokenizer.json").unlink()
        _refresh_final_model_evidence(fixture)
    with pytest.raises(ValueError, match=match):
        dapo_evidence.build_completion_binding(**fixture.kwargs)


def test_dapo_completion_rejects_resumed_results_lineage(tmp_path):
    fixture = _completion_fixture(tmp_path)
    fixture.kwargs["results_artifact_metadata"] = {
        **fixture.kwargs["results_artifact_metadata"],
        "resume_checkpoint_artifact": "entity/project/checkpoint:v0",
    }
    with pytest.raises(ValueError, match="fresh|resumed"):
        dapo_evidence.build_completion_binding(**fixture.kwargs)


def test_dapo_publication_protocol_requires_signed_campaign(monkeypatch):
    monkeypatch.setattr(
        eval_gpu, "runtime_stack_versions",
        lambda: dict(eval_gpu.PUBLICATION_RUNTIME_STACK_VERSIONS))
    args = SimpleNamespace(
        require_publication_protocol=True, protocol="P1", benchmark="AIME24",
        max_problems=None, model_family="qwen3-1.7b",
        server_max_model_len=36864, model=dapo_evidence.MODEL,
        model_source_name=dapo_evidence.MODEL,
        model_revision=common.CANONICAL_TRAINING_MODEL_REVISION,
        expected_vllm_version="0.24.0",
        container_image_identity=(
            eval_gpu.PUBLICATION_EVAL_CONTAINER_IMAGE_IDENTITY),
        model_artifact_ref="entity/project/dapo-final:v0",
        model_artifact_digest="model-digest",
        code_artifact_ref="entity/project/code:v1",
        code_artifact_digest="code-digest",
        run_name="unit-test",
        _evaluation_execution_environment=(
            receipt_fixtures.evaluation_environment(
                run_name="unit-test",
                code_ref="entity/project/code:v1",
                code_digest="code-digest",
            )
        ),
        method_name=dapo_evidence.DAPO_METHOD,
        training_campaign_binding_sha256=None,
        temperature=0.6, top_p=0.95, top_k=20, min_p=0.0,
        thinking_mode="auto", max_completion_tokens=32768,
    )
    with pytest.raises(SystemExit, match="signed training campaign"):
        eval_gpu.validate_publication_protocol(
            args, benchmark_default_samples=64, resolved_samples=64)
    args.training_campaign_binding_sha256 = "a" * 64
    eval_gpu.validate_publication_protocol(
        args, benchmark_default_samples=64, resolved_samples=64)


def test_dapo_launchers_bind_completion_and_resolve_before_server():
    launch_dir = Path(__file__).resolve().parents[1] / "launch"
    dapo_job = launch_dir / "dapo_job.sh"
    eval_job = launch_dir / "base_eval_job.sh"
    subprocess.run(["bash", "-n", str(dapo_job)], check=True)
    subprocess.run(["bash", "-n", str(eval_job)], check=True)
    training = dapo_job.read_text()
    assert "build_completion_binding(" in training
    assert "dapo-checkpoint-completion-verify" in training
    assert "DAPO_CAMPAIGN_ARTIFACT_TYPE" in training
    assert "validate_completion_campaign(" in training
    evaluation = eval_job.read_text()
    resolve = evaluation.index(
        'campaign_resolution["metadata"].get("method") == DAPO_METHOD')
    results = evaluation.index("/workspace/dapo_results_artifact", resolve)
    checkpoint = evaluation.index("/workspace/dapo_checkpoint_artifact", resolve)
    validate = evaluation.index("validate_training_campaign_evidence(", checkpoint)
    server = evaluation.index("# ---- start one vLLM server per GPU")
    assert resolve < results < checkpoint < validate < server
    assert "dapo_checkpoint_artifact_root=" in evaluation
    assert '("checkpoint", upstream["checkpoint_artifact"])' in evaluation


def test_checkpoint_progress_artifact_contains_complete_resume_evidence(
        monkeypatch, tmp_path):
    output = tmp_path / "run"
    checkpoint = output / "step_000025"
    provenance = output / "provenance"
    checkpoint.mkdir(parents=True)
    provenance.mkdir()
    for path, value in (
        (output / "metrics.jsonl", "{}\n"),
        (output / "explorer_trajectories.jsonl", "{}\n"),
        (output / "rollouts_all.jsonl", "{}\n"),
        (checkpoint / "trainer_state.json", "{}\n"),
        (provenance / "dapo_code_binding.json", "{}\n"),
        (provenance / "dapo_code_manifest.json", "{}\n"),
    ):
        path.write_text(value)

    class Artifact:
        def __init__(self, name, type, metadata):  # noqa: A002
            self.name = name
            self.type = type
            self.metadata = metadata
            self.files = {}

        def add_file(self, path, name):
            self.files[name] = path

    class Logged:
        waited = False

        def wait(self):
            self.waited = True

    class Run:
        def __init__(self):
            self.artifact = None
            self.aliases = None
            self.logged = Logged()

        def log_artifact(self, artifact, aliases):
            self.artifact = artifact
            self.aliases = aliases
            return self.logged

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Artifact=Artifact))
    run = Run()
    args = SimpleNamespace(
        run_name="gpu-dapo-original",
        phase="central",
        round=1,
        model=dapo_evidence.MODEL,
        scientific_variant=dapo_evidence.SCIENTIFIC_VARIANT,
        output_dir=str(output),
        metrics_jsonl=str(output / "metrics.jsonl"),
        code_source_manifest=str(provenance / "dapo_code_manifest.json"),
        _code_artifact_binding=_binding(),
    )
    grpo_gpu._log_checkpoint_progress_artifact(
        run, args=args, step=25, checkpoint_dir=str(checkpoint))
    assert set(run.artifact.files) == {
        "metrics.jsonl",
        "explorer_trajectories.jsonl",
        "rollouts_all.jsonl",
        "dapo_code_binding.json",
        "dapo_code_manifest.json",
        "step_000025/trainer_state.json",
    }
    assert run.artifact.type == "grpo-progress"
    assert run.artifact.metadata["step"] == 25
    assert run.aliases == ["latest", "step-25"]
    assert run.logged.waited is True
