"""Adversarial tests for immutable remote-job execution receipts."""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tmx_gpu import common, eval_gpu, grpo_gpu, orchestrator_receipt
from tmx_gpu.tests import orchestrator_receipt_fixtures as receipt_fixtures

RUN_NAME = "c9-receipt-test"
CODE_REF = "entity/project/tmx-code:v28"
CODE_DIGEST = "code-digest-123"
POLL_REF = f"entity/project/{RUN_NAME}-orchestrator-receipt:latest"


def _command() -> list[str]:
    return ["/bin/bash", "-lc", "bash tmx_gpu/launch/expdis_job.sh"]


def _command_sha256() -> str:
    return common.canonical_json_sha256({"kind": "argv", "value": _command()})


def _spec(*, family: str = "h100") -> dict:
    shape = "gpu.8xh100-sxm" if family == "h100" else "gpu.8xa100-80gb"
    env = {
        "RUN_NAME": RUN_NAME,
        "CODE_ARTIFACT": CODE_REF,
        "CODE_ARTIFACT_DIGEST": CODE_DIGEST,
        "TRAINING_CONTAINER_IMAGE_IDENTITY":
            grpo_gpu.CANONICAL_TRAINING_CONTAINER_IMAGE,
        "ORCHESTRATOR_COMMAND_SHA256": _command_sha256(),
        "ORCHESTRATOR_RECEIPT_ARTIFACT": POLL_REF,
    }
    return {
        "container": {
            "image": grpo_gpu.CANONICAL_TRAINING_CONTAINER_IMAGE,
            "command": _command(),
        },
        "resource_shape": shape,
        "envs": [{"name": key, "value": value} for key, value in env.items()],
    }


def _write_artifact(root: Path, raw_spec: dict) -> tuple[dict, dict]:
    root.mkdir(parents=True)
    receipt = orchestrator_receipt.build_receipt(raw_spec)
    common.atomic_write_json(
        str(root / orchestrator_receipt.RAW_SPEC_FILENAME), raw_spec)
    common.atomic_write_json(
        str(root / orchestrator_receipt.RECEIPT_FILENAME), receipt)
    source = {
        "artifact_ref":
            f"entity/project/{RUN_NAME}-orchestrator-receipt:v7",
        "artifact_digest": "receipt-artifact-digest",
    }
    files = [{
        "relative_path": name,
        "sha256": common.file_sha256(str(root / name)),
        "size_bytes": (root / name).stat().st_size,
    } for name in sorted(orchestrator_receipt.EXPECTED_ARTIFACT_MEMBERS)]
    normalized = {
        "version": common.ARTIFACT_DOWNLOAD_MANIFEST_VERSION,
        "source": source,
        "files": files,
    }
    manifest = {
        **normalized,
        "file_manifest_sha256": common.canonical_json_sha256(normalized),
        "download_root": str(root),
    }
    return receipt, manifest


def _binding(tmp_path: Path, *, family: str = "h100") -> dict:
    receipt, manifest = _write_artifact(tmp_path / "artifact", _spec(family=family))
    return orchestrator_receipt.build_binding(
        source_ref=manifest["source"]["artifact_ref"],
        source_digest=manifest["source"]["artifact_digest"],
        artifact_type=orchestrator_receipt.ARTIFACT_TYPE,
        artifact_metadata=orchestrator_receipt.receipt_artifact_metadata(receipt),
        download_manifest=manifest,
    )


def _native(family: str) -> dict:
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


def test_receipt_binds_raw_spec_and_accepts_either_8_gpu_family(tmp_path):
    for family in ("A100", "H100"):
        binding = _binding(tmp_path / family, family=family.lower())
        orchestrator_receipt.validate_binding(
            binding,
            expected_run_name=RUN_NAME,
            expected_container_image=grpo_gpu.CANONICAL_TRAINING_CONTAINER_IMAGE,
            expected_code_artifact_ref=CODE_REF,
            expected_code_artifact_digest=CODE_DIGEST,
            expected_command_sha256=_command_sha256(),
            expected_polling_artifact_ref=POLL_REF,
        )
        environment = grpo_gpu.canonical_training_execution_environment(
            grpo_gpu.CANONICAL_TRAINING_CONTAINER_IMAGE,
            binding,
            _native(family),
        )
        assert environment["frozen_invariants"][
            "orchestrator_receipt_binding"]["source"][
                "artifact_ref"].endswith(":v7")
        assert environment["compatibility_facts"]["native_runtime"][
            "gpu_family"] == family
        grpo_gpu.require_canonical_training_execution_environment_contract({
            "execution_environment": environment,
        })


@pytest.mark.parametrize("mutation,match", [
    (lambda spec: spec["container"].update({"command": []}), "command"),
    (lambda spec: spec.update({"resource_shape": "gpu.4xh100-sxm"}), "8x"),
    (lambda spec: next(item for item in spec["envs"]
                       if item["name"] == "CODE_ARTIFACT").update(
                           {"value": "entity/project/tmx-code:latest"}),
     "immutable"),
    (lambda spec: next(item for item in spec["envs"]
                       if item["name"] == "TRAINING_CONTAINER_IMAGE_IDENTITY").update(
                           {"value": "spoofed@sha256:bad"}),
     "disagree"),
    (lambda spec: next(item for item in spec["envs"]
                       if item["name"] == "ORCHESTRATOR_COMMAND_SHA256").update(
                           {"value": "0" * 64}),
     "container.command"),
    (lambda spec: next(item for item in spec["envs"]
                       if item["name"] == "ORCHESTRATOR_RECEIPT_ARTIFACT").update(
                           {"value": "entity/project/shared:latest"}),
     "RUN_NAME-unique"),
])
def test_raw_spec_spoofs_fail_closed(mutation, match):
    spec = _spec()
    mutation(spec)
    with pytest.raises(ValueError, match=match):
        orchestrator_receipt.build_receipt(spec)


def test_raw_spec_or_normalized_receipt_mutation_breaks_signature(tmp_path):
    spec = _spec()
    receipt = orchestrator_receipt.build_receipt(spec)
    mutated_spec = copy.deepcopy(spec)
    mutated_spec["container"]["image"] = "other@sha256:deadbeef"
    with pytest.raises(ValueError, match=r"raw-spec SHA256|disagree"):
        orchestrator_receipt.validate_receipt(receipt, mutated_spec)

    mutated_receipt = copy.deepcopy(receipt)
    mutated_receipt["job"]["command"]["value"][-1] = "bash other.sh"
    with pytest.raises(ValueError, match="signature"):
        orchestrator_receipt.validate_receipt(mutated_receipt, spec)


def test_binding_rejects_spoofed_expected_identity_and_signature(tmp_path):
    binding = _binding(tmp_path)
    for keyword, value, match in (
        ("expected_run_name", "other", "run name"),
        ("expected_container_image", "other", "container image"),
        ("expected_code_artifact_ref", "entity/project/other:v1", "code artifact ref"),
        ("expected_code_artifact_digest", "other", "code artifact digest"),
        ("expected_command_sha256", "0" * 64, "command SHA256"),
        ("expected_polling_artifact_ref",
         "entity/project/other-orchestrator-receipt:latest", "polling artifact"),
    ):
        with pytest.raises(ValueError, match=match):
            orchestrator_receipt.validate_binding(binding, **{keyword: value})
    tampered = copy.deepcopy(binding)
    tampered["source"]["artifact_digest"] = "spoof"
    with pytest.raises(ValueError, match="signature"):
        orchestrator_receipt.validate_binding(tampered)


def test_binding_rejects_unexpected_artifact_member(tmp_path):
    root = tmp_path / "artifact"
    receipt, manifest = _write_artifact(root, _spec())
    extra = root / "untrusted.txt"
    extra.write_text("x", encoding="utf-8")
    manifest["files"].append({
        "relative_path": "untrusted.txt",
        "sha256": common.file_sha256(str(extra)),
        "size_bytes": 1,
    })
    normalized = {
        "version": manifest["version"],
        "source": manifest["source"],
        "files": manifest["files"],
    }
    manifest["file_manifest_sha256"] = common.canonical_json_sha256(normalized)
    with pytest.raises(ValueError, match="exactly"):
        orchestrator_receipt.build_binding(
            source_ref=manifest["source"]["artifact_ref"],
            source_digest=manifest["source"]["artifact_digest"],
            artifact_type=orchestrator_receipt.ARTIFACT_TYPE,
            artifact_metadata=orchestrator_receipt.receipt_artifact_metadata(receipt),
            download_manifest=manifest,
        )


def test_native_facts_require_all_eight_homogeneous_devices():
    facts = _native("H100")
    facts["gpu_names"][-1] = "NVIDIA A100-SXM4-80GB"
    with pytest.raises(ValueError, match="homogeneous"):
        orchestrator_receipt.validate_native_compatibility_facts(facts)
    facts = _native("H100")
    facts["driver_versions"].pop()
    with pytest.raises(ValueError, match="driver"):
        orchestrator_receipt.validate_native_compatibility_facts(facts)
    facts = _native("H100")
    facts["nccl_version"] = None
    with pytest.raises(ValueError, match="NCCL"):
        orchestrator_receipt.validate_native_compatibility_facts(facts)


class _File:
    def __init__(self, name: str):
        self.name = name


class _Artifact:
    # W&B 0.28 preserves the lookup alias here.  The immutable version is
    # available separately through source_qualified_name and version.
    qualified_name = POLL_REF
    source_qualified_name = \
        f"entity/project/{RUN_NAME}-orchestrator-receipt:v9"
    version = "v9"
    source_version = "v9"
    digest = "remote-digest"
    type = orchestrator_receipt.ARTIFACT_TYPE

    def __init__(self, source: Path, metadata: dict):
        self.source = source
        self.metadata = metadata

    def files(self):
        return [_File(name) for name in orchestrator_receipt.EXPECTED_ARTIFACT_MEMBERS]

    def download(self, *, root: str):
        for name in orchestrator_receipt.EXPECTED_ARTIFACT_MEMBERS:
            shutil.copy2(self.source / name, Path(root) / name)
        return root


class _Api:
    def __init__(self, artifact, expected_ref=POLL_REF):
        self.value = artifact
        self.expected_ref = expected_ref

    def artifact(self, ref, type=None):
        assert ref == self.expected_ref
        assert type == orchestrator_receipt.ARTIFACT_TYPE
        return self.value


def test_latest_is_only_polled_then_binding_records_immutable_ref(tmp_path):
    source = tmp_path / "source"
    receipt, _ = _write_artifact(source, _spec())
    artifact = _Artifact(
        source, orchestrator_receipt.receipt_artifact_metadata(receipt))
    binding = orchestrator_receipt.resolve_artifact_binding(
        POLL_REF,
        output_binding=str(tmp_path / "binding.json"),
        download_root=str(tmp_path / "downloads"),
        expected_run_name=RUN_NAME,
        expected_container_image=grpo_gpu.CANONICAL_TRAINING_CONTAINER_IMAGE,
        expected_code_artifact_ref=CODE_REF,
        expected_code_artifact_digest=CODE_DIGEST,
        expected_command_sha256=_command_sha256(),
        api=_Api(artifact),
    )
    assert binding["source"]["artifact_ref"].endswith(":v9")
    assert ":latest" not in json.dumps(binding["source"])
    with pytest.raises(ValueError, match="vN or the deterministic"):
        orchestrator_receipt.require_pollable_ref(
            "entity/project/name:production", field="receipt")


@pytest.mark.parametrize(
    ("attribute", "value", "match"),
    (
        ("source_qualified_name",
         "entity/project/other-orchestrator-receipt:v9", "sequence mismatch"),
        ("version", "v8", "version mismatch"),
        ("source_version", "v8", "version mismatch"),
        ("version", "latest", "version is invalid"),
        ("qualified_name",
         "entity/project/other-orchestrator-receipt:latest",
         "lookup identity mismatch"),
    ),
)
def test_latest_resolution_rejects_inconsistent_immutable_identity(
        tmp_path, attribute, value, match):
    source = tmp_path / "source"
    receipt, _ = _write_artifact(source, _spec())
    artifact = _Artifact(
        source, orchestrator_receipt.receipt_artifact_metadata(receipt))
    setattr(artifact, attribute, value)
    with pytest.raises(ValueError, match=match):
        orchestrator_receipt.resolve_artifact_binding(
            POLL_REF,
            output_binding=str(tmp_path / "binding.json"),
            download_root=str(tmp_path / "downloads"),
            expected_run_name=RUN_NAME,
            expected_container_image=grpo_gpu.CANONICAL_TRAINING_CONTAINER_IMAGE,
            expected_code_artifact_ref=CODE_REF,
            expected_code_artifact_digest=CODE_DIGEST,
            expected_command_sha256=_command_sha256(),
            api=_Api(artifact),
        )


def test_latest_resolution_requires_api_immutable_source(tmp_path):
    source = tmp_path / "source"
    receipt, _ = _write_artifact(source, _spec())
    artifact = _Artifact(
        source, orchestrator_receipt.receipt_artifact_metadata(receipt))
    artifact.source_qualified_name = ""
    with pytest.raises(ValueError, match="must be a nonempty string"):
        orchestrator_receipt.resolve_artifact_binding(
            POLL_REF,
            output_binding=str(tmp_path / "binding.json"),
            download_root=str(tmp_path / "downloads"),
            expected_run_name=RUN_NAME,
            expected_container_image=grpo_gpu.CANONICAL_TRAINING_CONTAINER_IMAGE,
            expected_code_artifact_ref=CODE_REF,
            expected_code_artifact_digest=CODE_DIGEST,
            expected_command_sha256=_command_sha256(),
            api=_Api(artifact),
        )


@pytest.mark.parametrize("attribute", ("version", "source_version"))
def test_latest_resolution_requires_reported_versions(tmp_path, attribute):
    source = tmp_path / "source"
    receipt, _ = _write_artifact(source, _spec())
    artifact = _Artifact(
        source, orchestrator_receipt.receipt_artifact_metadata(receipt))
    setattr(artifact, attribute, None)
    with pytest.raises(ValueError, match="must be a nonempty string"):
        orchestrator_receipt.resolve_artifact_binding(
            POLL_REF,
            output_binding=str(tmp_path / "binding.json"),
            download_root=str(tmp_path / "downloads"),
            expected_run_name=RUN_NAME,
            expected_container_image=grpo_gpu.CANONICAL_TRAINING_CONTAINER_IMAGE,
            expected_code_artifact_ref=CODE_REF,
            expected_code_artifact_digest=CODE_DIGEST,
            expected_command_sha256=_command_sha256(),
            api=_Api(artifact),
        )


def test_direct_immutable_resolution_succeeds(tmp_path):
    source = tmp_path / "source"
    receipt, _ = _write_artifact(source, _spec())
    direct_ref = f"entity/project/{RUN_NAME}-orchestrator-receipt:v9"
    artifact = _Artifact(
        source, orchestrator_receipt.receipt_artifact_metadata(receipt))
    artifact.qualified_name = direct_ref
    binding = orchestrator_receipt.resolve_artifact_binding(
        direct_ref,
        output_binding=str(tmp_path / "binding.json"),
        download_root=str(tmp_path / "downloads"),
        expected_run_name=RUN_NAME,
        expected_container_image=grpo_gpu.CANONICAL_TRAINING_CONTAINER_IMAGE,
        expected_code_artifact_ref=CODE_REF,
        expected_code_artifact_digest=CODE_DIGEST,
        expected_command_sha256=_command_sha256(),
        api=_Api(artifact, expected_ref=direct_ref),
    )
    assert binding["source"]["artifact_ref"] == direct_ref


def test_direct_immutable_resolution_rejects_different_version(tmp_path):
    source = tmp_path / "source"
    receipt, _ = _write_artifact(source, _spec())
    direct_ref = f"entity/project/{RUN_NAME}-orchestrator-receipt:v9"
    artifact = _Artifact(
        source, orchestrator_receipt.receipt_artifact_metadata(receipt))
    artifact.qualified_name = direct_ref
    artifact.source_qualified_name = \
        f"entity/project/{RUN_NAME}-orchestrator-receipt:v8"
    artifact.version = "v8"
    artifact.source_version = "v8"
    with pytest.raises(ValueError, match="differs from immutable request"):
        orchestrator_receipt.resolve_artifact_binding(
            direct_ref,
            output_binding=str(tmp_path / "binding.json"),
            download_root=str(tmp_path / "downloads"),
            expected_run_name=RUN_NAME,
            expected_container_image=grpo_gpu.CANONICAL_TRAINING_CONTAINER_IMAGE,
            expected_code_artifact_ref=CODE_REF,
            expected_code_artifact_digest=CODE_DIGEST,
            expected_command_sha256=_command_sha256(),
            api=_Api(artifact, expected_ref=direct_ref),
        )


def test_evaluation_profile_is_distinct_and_binds_frozen_runtime():
    binding = receipt_fixtures.evaluation_binding(
        run_name="evaluation-profile-test",
        code_ref=CODE_REF,
        code_digest=CODE_DIGEST,
    )
    orchestrator_receipt.validate_binding(
        binding,
        expected_run_name="evaluation-profile-test",
        expected_container_image=(
            eval_gpu.PUBLICATION_EVAL_CONTAINER_IMAGE_IDENTITY),
        expected_code_artifact_ref=CODE_REF,
        expected_code_artifact_digest=CODE_DIGEST,
        profile=orchestrator_receipt.EVALUATION_PROFILE,
    )
    with pytest.raises(ValueError, match="profile"):
        orchestrator_receipt.validate_binding(binding)

    environment = receipt_fixtures.evaluation_environment(
        run_name="evaluation-profile-test",
        code_ref=CODE_REF,
        code_digest=CODE_DIGEST,
    )
    validated = eval_gpu.validate_publication_evaluation_execution_environment(
        environment,
        expected_run_name="evaluation-profile-test",
        expected_code_artifact_ref=CODE_REF,
        expected_code_artifact_digest=CODE_DIGEST,
    )
    assert validated["frozen_invariants"]["container_image_identity"] == \
        eval_gpu.PUBLICATION_EVAL_CONTAINER_IMAGE_IDENTITY
    assert validated["compatibility_facts"]["native_runtime"][
        "visible_gpu_count"] == 8


def test_evaluation_environment_rejects_tampered_native_facts():
    environment = receipt_fixtures.evaluation_environment()
    tampered = copy.deepcopy(environment)
    tampered["compatibility_facts"]["native_runtime"][
        "driver_versions"].pop()
    with pytest.raises(ValueError, match="driver"):
        eval_gpu.validate_publication_evaluation_execution_environment(tampered)


def test_evaluation_launcher_binds_receipt_before_vllm_start():
    launcher = Path(__file__).resolve().parents[1] / "launch" / "base_eval_job.sh"
    subprocess.run(["bash", "-n", str(launcher)], check=True)
    source = launcher.read_text(encoding="utf-8")
    resolve_at = source.index("python3 -m tmx_gpu.orchestrator_receipt resolve")
    native_at = source.index(
        "orchestrator_receipt.observed_native_compatibility_facts()")
    server_at = source.index("python3 -m vllm.entrypoints.openai.api_server")
    assert resolve_at < native_at < server_at
    assert "--profile strict_publication_evaluation_v1" in source
    assert "--evaluation-execution-environment" in source
