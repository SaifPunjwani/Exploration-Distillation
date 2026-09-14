"""In-memory immutable execution-receipt fixtures for unit tests."""

from __future__ import annotations

from tmx_gpu import common, eval_gpu, grpo_gpu, orchestrator_receipt


def native_facts(family: str = "H100") -> dict:
    name = f"NVIDIA {family}-SXM-80GB"
    capability = [9, 0] if family == "H100" else [8, 0]
    return {
        "torch_cuda_version": "13.0",
        "cudnn_version": 91002,
        "nccl_version": [2, 27, 3],
        "visible_gpu_count": 8,
        "gpu_names": [name] * 8,
        "gpu_family": family,
        "gpu_capabilities": [capability] * 8,
        "driver_versions": ["580.65.06"] * 8,
    }


def _binding(
    *, run_name: str = "unit-test", code_ref: str = "entity/project/code:v1",
    code_digest: str = "code-digest", family: str = "H100",
    profile: str, image: str, image_environment_name: str,
    command_value: list[str],
) -> dict:
    polling_ref = (
        f"entity/project/{common.sanitize_wandb_id(run_name + '-orchestrator-receipt')}:latest")
    command = {"kind": "argv", "value": list(command_value)}
    command_sha = common.canonical_json_sha256(command)
    raw = {
        "container": {
            "image": image,
            "command": command["value"],
        },
        "resource_shape": (
            "gpu.8xh100-sxm" if family == "H100" else "gpu.8xa100-80gb"),
        "envs": [{"name": name, "value": value} for name, value in {
            "RUN_NAME": run_name,
            "CODE_ARTIFACT": code_ref,
            "CODE_ARTIFACT_DIGEST": code_digest,
            image_environment_name: image,
            "ORCHESTRATOR_COMMAND_SHA256": command_sha,
            "ORCHESTRATOR_RECEIPT_ARTIFACT": polling_ref,
        }.items()],
    }
    receipt = orchestrator_receipt.build_receipt(raw, profile=profile)
    source = {
        "artifact_ref":
            f"entity/project/{common.sanitize_wandb_id(run_name + '-orchestrator-receipt')}:v1",
        "artifact_digest": "receipt-digest",
        "artifact_type": orchestrator_receipt.ARTIFACT_TYPE,
    }
    files = [{
        "relative_path": name,
        "sha256": char * 64,
        "size_bytes": 1,
    } for name, char in zip(
        sorted(orchestrator_receipt.EXPECTED_ARTIFACT_MEMBERS), ("a", "b"),
        strict=True,
    )]
    file_unsigned = {
        "version": common.ARTIFACT_DOWNLOAD_MANIFEST_VERSION,
        "source": {
            "artifact_ref": source["artifact_ref"],
            "artifact_digest": source["artifact_digest"],
        },
        "files": files,
    }
    unsigned = {
        "version": orchestrator_receipt.BINDING_VERSION,
        "source": source,
        "artifact_metadata":
            orchestrator_receipt.receipt_artifact_metadata(receipt),
        "artifact_files": {
            **file_unsigned,
            "file_manifest_sha256": common.canonical_json_sha256(file_unsigned),
        },
        "receipt": receipt,
    }
    return {**unsigned, "binding_sha256": common.canonical_json_sha256(unsigned)}


def binding(
    *, run_name: str = "unit-test", code_ref: str = "entity/project/code:v1",
    code_digest: str = "code-digest", family: str = "H100",
) -> dict:
    return _binding(
        run_name=run_name,
        code_ref=code_ref,
        code_digest=code_digest,
        family=family,
        profile=orchestrator_receipt.TRAINING_PROFILE,
        image=grpo_gpu.CANONICAL_TRAINING_CONTAINER_IMAGE,
        image_environment_name="TRAINING_CONTAINER_IMAGE_IDENTITY",
        command_value=["/bin/bash", "-lc", "run-unit-test"],
    )


def evaluation_binding(
    *, run_name: str = "unit-test", code_ref: str = "entity/project/code:v1",
    code_digest: str = "code-digest", family: str = "H100",
) -> dict:
    return _binding(
        run_name=run_name,
        code_ref=code_ref,
        code_digest=code_digest,
        family=family,
        profile=orchestrator_receipt.EVALUATION_PROFILE,
        image=eval_gpu.PUBLICATION_EVAL_CONTAINER_IMAGE_IDENTITY,
        image_environment_name="EVAL_CONTAINER_IMAGE_IDENTITY",
        command_value=["/bin/bash", "-lc", "bash tmx_gpu/launch/base_eval_job.sh"],
    )


def environment(**kwargs) -> dict:
    family = str(kwargs.get("family", "H100"))
    return grpo_gpu.canonical_training_execution_environment(
        grpo_gpu.CANONICAL_TRAINING_CONTAINER_IMAGE,
        binding(**kwargs),
        native_facts(family),
    )


def evaluation_environment(**kwargs) -> dict:
    family = str(kwargs.get("family", "H100"))
    return eval_gpu.build_publication_evaluation_execution_environment(
        evaluation_binding(**kwargs),
        native_facts(family),
        observed_runtime_stack=dict(eval_gpu.PUBLICATION_RUNTIME_STACK_VERSIONS),
    )
