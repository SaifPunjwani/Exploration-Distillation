"""Immutable Lepton job-spec receipts for publication execution jobs.

The launcher process cannot establish which OCI image or accelerator shape the
orchestrator actually scheduled by inspecting a caller-controlled environment
variable.  A controller therefore captures the authoritative output of
``lep job get --path`` (the raw job ``spec`` object), places it beside a signed
normalization record in a W&B artifact, and passes only that artifact reference
to the job. Strict launchers resolve the artifact to ``:vN`` and bind its exact
bytes into every training or evaluation runtime contract before scientific
work begins.

The SHA256 signatures below are tamper-evident content signatures, not claims
of public-key authorship.  Trust comes from the immutable W&B artifact identity
supplied by the controller; the signatures prevent fields from being detached
from the raw orchestrator spec inside that artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tmx_gpu import common

RECEIPT_VERSION = "lepton_job_orchestrator_receipt_v1"
BINDING_VERSION = "lepton_job_orchestrator_binding_v1"
ARTIFACT_TYPE = "orchestrator-receipt"
RAW_SPEC_FILENAME = "lep_job_get.json"
RECEIPT_FILENAME = "orchestrator_receipt.json"
EXPECTED_ARTIFACT_MEMBERS = (RAW_SPEC_FILENAME, RECEIPT_FILENAME)

TRAINING_PROFILE = "strict_publication_training_v1"
EVALUATION_PROFILE = "strict_publication_evaluation_v1"
RECEIPT_PROFILES = (TRAINING_PROFILE, EVALUATION_PROFILE)

_PROFILE_IMAGE_ENV = {
    TRAINING_PROFILE: "TRAINING_CONTAINER_IMAGE_IDENTITY",
    EVALUATION_PROFILE: "EVAL_CONTAINER_IMAGE_IDENTITY",
}
_PROFILE_LABEL = {
    TRAINING_PROFILE: "training",
    EVALUATION_PROFILE: "evaluation",
}

_QUALIFIED_IMMUTABLE_REF_RE = re.compile(
    r"^[^/\s]+/[^/\s]+/[^/:\s]+:v[0-9]+$")
_QUALIFIED_LATEST_REF_RE = re.compile(
    r"^[^/\s]+/[^/\s]+/[^/:\s]+:latest$")
_GPU_SHAPE_RE = re.compile(
    r"^gpu\.8x(?P<family>a100|h100)(?:-(?:40gb|80gb|sxm))?$",
    re.IGNORECASE,
)

_COMMON_PROTECTED_ENV_NAMES = (
    "RUN_NAME",
    "CODE_ARTIFACT",
    "CODE_ARTIFACT_DIGEST",
    "ORCHESTRATOR_COMMAND_SHA256",
    "ORCHESTRATOR_RECEIPT_ARTIFACT",
)


def _require_profile(value: Any) -> str:
    profile = _require_nonempty_string(value, "orchestrator receipt profile")
    if profile not in RECEIPT_PROFILES:
        raise ValueError(
            f"unsupported orchestrator receipt profile {profile!r}; expected "
            f"one of {list(RECEIPT_PROFILES)!r}")
    return profile


def _protected_env_names(profile: str) -> tuple[str, ...]:
    profile = _require_profile(profile)
    return (*_COMMON_PROTECTED_ENV_NAMES, _PROFILE_IMAGE_ENV[profile])


def _require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def require_qualified_immutable_ref(value: Any, *, field: str) -> str:
    ref = _require_nonempty_string(value, field)
    if _QUALIFIED_IMMUTABLE_REF_RE.fullmatch(ref) is None:
        raise ValueError(
            f"{field} must be a fully qualified immutable W&B artifact "
            f"reference entity/project/name:vN; got {ref!r}")
    return ref


def require_pollable_ref(value: Any, *, field: str) -> str:
    """Allow an immutable ref or the one deterministic polling alias.

    No other mutable W&B alias is accepted.  ``:latest`` is never persisted as
    provenance: ``resolve_artifact_binding`` replaces it with the server's
    qualified ``:vN`` identity before returning.
    """
    ref = _require_nonempty_string(value, field)
    if _QUALIFIED_IMMUTABLE_REF_RE.fullmatch(ref) is None and \
            _QUALIFIED_LATEST_REF_RE.fullmatch(ref) is None:
        raise ValueError(
            f"{field} must be entity/project/name:vN or the deterministic "
            f"polling alias entity/project/name:latest; got {ref!r}")
    return ref


def _raw_command(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("Lepton job spec container.command is empty")
        return {"kind": "string", "value": value}
    if isinstance(value, list) and value and all(
            isinstance(part, str) and part.strip() for part in value):
        return {"kind": "argv", "value": list(value)}
    raise ValueError(
        "Lepton job spec container.command must be a nonempty string or "
        "nonempty string array")


def _protected_environment(
    spec: Mapping[str, Any], *, profile: str,
) -> dict[str, str]:
    protected_names = _protected_env_names(profile)
    raw_envs = spec.get("envs")
    if not isinstance(raw_envs, list):
        raise ValueError("Lepton job spec envs must be an array")
    values: dict[str, str] = {}
    for entry in raw_envs:
        if not isinstance(entry, dict):
            raise ValueError("Lepton job spec env entry must be an object")
        name = entry.get("name")
        if name not in protected_names:
            continue
        if name in values:
            raise ValueError(f"Lepton job spec duplicates protected env {name}")
        # Secret/valueFrom references are not suitable for scientific identity.
        values[str(name)] = _require_nonempty_string(
            entry.get("value"), f"Lepton job spec env {name}")
    missing = sorted(set(protected_names) - set(values))
    if missing:
        raise ValueError(
            "Lepton job spec lacks protected environment values: "
            + ", ".join(missing))
    return values


def extract_job_spec_facts(
    raw_spec: Any, *, profile: str = TRAINING_PROFILE,
) -> dict[str, Any]:
    """Extract and validate publication-critical facts from raw ``lep`` spec."""
    profile = _require_profile(profile)
    profile_label = _PROFILE_LABEL[profile]
    if not isinstance(raw_spec, dict):
        raise ValueError("authoritative Lepton job spec must be one JSON object")
    container = raw_spec.get("container")
    if not isinstance(container, dict):
        raise ValueError("Lepton job spec lacks container object")
    image = _require_nonempty_string(
        container.get("image"), "Lepton job spec container.image")
    command = _raw_command(container.get("command"))

    direct_shape = raw_spec.get("resource_shape")
    legacy_requirement = raw_spec.get("resource_requirement")
    legacy_shape = (
        legacy_requirement.get("resource_shape")
        if isinstance(legacy_requirement, dict) else None
    )
    if direct_shape is not None and legacy_shape is not None and \
            direct_shape != legacy_shape:
        raise ValueError("Lepton job spec resource-shape fields disagree")
    resource_shape = _require_nonempty_string(
        direct_shape if direct_shape is not None else legacy_shape,
        "Lepton job spec resource_shape",
    )
    match = _GPU_SHAPE_RE.fullmatch(resource_shape)
    if match is None:
        raise ValueError(
            f"strict {profile_label} requires one 8x A100/H100 Lepton "
            "resource shape; "
            f"got {resource_shape!r}")

    environment = _protected_environment(raw_spec, profile=profile)
    code_ref = require_qualified_immutable_ref(
        environment["CODE_ARTIFACT"], field="Lepton CODE_ARTIFACT")
    code_digest = _require_nonempty_string(
        environment["CODE_ARTIFACT_DIGEST"], "Lepton CODE_ARTIFACT_DIGEST")
    image_env_name = _PROFILE_IMAGE_ENV[profile]
    image_env = environment[image_env_name]
    if image_env != image:
        raise ValueError(
            f"Lepton container.image and {image_env_name} environment value "
            "disagree")
    command_sha256 = common.canonical_json_sha256(command)
    if environment["ORCHESTRATOR_COMMAND_SHA256"] != command_sha256:
        raise ValueError(
            "Lepton ORCHESTRATOR_COMMAND_SHA256 does not match the exact "
            "container.command")
    receipt_ref = require_pollable_ref(
        environment["ORCHESTRATOR_RECEIPT_ARTIFACT"],
        field="Lepton ORCHESTRATOR_RECEIPT_ARTIFACT",
    )
    if not receipt_ref.endswith(":latest"):
        raise ValueError(
            "Lepton ORCHESTRATOR_RECEIPT_ARTIFACT must use only the :latest "
            "polling alias")
    receipt_name = receipt_ref.rsplit("/", 1)[-1].rsplit(":", 1)[0]
    expected_receipt_name = common.sanitize_wandb_id(
        f"{environment['RUN_NAME']}-orchestrator-receipt")
    if receipt_name != expected_receipt_name:
        raise ValueError(
            "Lepton orchestrator receipt artifact is not RUN_NAME-unique")
    family = match.group("family").upper()
    return {
        "receipt_profile": profile,
        "run_name": environment["RUN_NAME"],
        "container_image_identity": image,
        "command": command,
        "command_sha256": command_sha256,
        "resource_shape": resource_shape,
        "accelerator_family": family,
        "gpu_count": 8,
        "environment": {
            name: environment[name] for name in _protected_env_names(profile)},
        "code_artifact": {
            "artifact_ref": code_ref,
            "artifact_digest": code_digest,
        },
    }


def build_receipt(
    raw_spec: Any, *, profile: str = TRAINING_PROFILE,
) -> dict[str, Any]:
    profile = _require_profile(profile)
    facts = extract_job_spec_facts(raw_spec, profile=profile)
    unsigned = {
        "version": RECEIPT_VERSION,
        "profile": profile,
        "raw_spec_filename": RAW_SPEC_FILENAME,
        "raw_spec_sha256": common.canonical_json_sha256(raw_spec),
        "job": facts,
    }
    return {**unsigned, "receipt_sha256": common.canonical_json_sha256(unsigned)}


def validate_receipt(
    receipt: Any, raw_spec: Any, *, profile: str = TRAINING_PROFILE,
) -> dict[str, Any]:
    profile = _require_profile(profile)
    if not isinstance(receipt, dict):
        raise ValueError("orchestrator receipt must be one JSON object")
    if receipt.get("version") != RECEIPT_VERSION:
        raise ValueError("orchestrator receipt version mismatch")
    if receipt.get("profile") != profile:
        raise ValueError("orchestrator receipt profile mismatch")
    if receipt.get("raw_spec_filename") != RAW_SPEC_FILENAME:
        raise ValueError("orchestrator receipt raw-spec filename mismatch")
    unsigned = dict(receipt)
    signature = unsigned.pop("receipt_sha256", None)
    if signature != common.canonical_json_sha256(unsigned):
        raise ValueError("orchestrator receipt signature mismatch")
    if receipt.get("raw_spec_sha256") != common.canonical_json_sha256(raw_spec):
        raise ValueError("orchestrator receipt raw-spec SHA256 mismatch")
    facts = extract_job_spec_facts(raw_spec, profile=profile)
    if receipt.get("job") != facts:
        raise ValueError("orchestrator receipt normalized job facts mismatch")
    return receipt


def receipt_artifact_metadata(receipt: Mapping[str, Any]) -> dict[str, Any]:
    job = receipt["job"]
    code = job["code_artifact"]
    return {
        "receipt_version": RECEIPT_VERSION,
        "receipt_profile": receipt["profile"],
        "receipt_sha256": receipt["receipt_sha256"],
        "raw_spec_sha256": receipt["raw_spec_sha256"],
        "run_name": job["run_name"],
        "container_image_identity": job["container_image_identity"],
        "command_sha256": job["command_sha256"],
        "resource_shape": job["resource_shape"],
        "accelerator_family": job["accelerator_family"],
        "gpu_count": 8,
        "code_artifact_ref": code["artifact_ref"],
        "code_artifact_digest": code["artifact_digest"],
    }


def _stable_download_manifest(download: Mapping[str, Any]) -> dict[str, Any]:
    return common.stable_artifact_download_identity(dict(download))


def build_binding(
    *, source_ref: str, source_digest: str, artifact_type: str,
    artifact_metadata: Any, download_manifest: Mapping[str, Any],
    profile: str = TRAINING_PROFILE,
) -> dict[str, Any]:
    profile = _require_profile(profile)
    source_ref = require_qualified_immutable_ref(
        source_ref, field="resolved orchestrator receipt artifact")
    source_digest = _require_nonempty_string(
        source_digest, "resolved orchestrator receipt artifact digest")
    if artifact_type != ARTIFACT_TYPE:
        raise ValueError(
            f"orchestrator receipt artifact type must be {ARTIFACT_TYPE!r}")
    validated_download = common.validate_artifact_download_manifest(
        dict(download_manifest), expected_source={
            "artifact_ref": source_ref,
            "artifact_digest": source_digest,
        })
    members = [item["relative_path"] for item in validated_download["files"]]
    if members != sorted(EXPECTED_ARTIFACT_MEMBERS):
        raise ValueError(
            "orchestrator receipt artifact must contain exactly "
            f"{list(EXPECTED_ARTIFACT_MEMBERS)!r}; got {members!r}")
    root = validated_download["download_root"]
    with open(os.path.join(root, RAW_SPEC_FILENAME), encoding="utf-8") as handle:
        raw_spec = json.load(handle)
    with open(os.path.join(root, RECEIPT_FILENAME), encoding="utf-8") as handle:
        receipt = json.load(handle)
    validate_receipt(receipt, raw_spec, profile=profile)
    expected_metadata = receipt_artifact_metadata(receipt)
    if artifact_metadata != expected_metadata:
        raise ValueError("orchestrator receipt artifact metadata mismatch")
    unsigned = {
        "version": BINDING_VERSION,
        "source": {
            "artifact_ref": source_ref,
            "artifact_digest": source_digest,
            "artifact_type": ARTIFACT_TYPE,
        },
        "artifact_metadata": expected_metadata,
        "artifact_files": _stable_download_manifest(validated_download),
        "receipt": receipt,
    }
    return {**unsigned, "binding_sha256": common.canonical_json_sha256(unsigned)}


def validate_binding(
    binding: Any, *, expected_run_name: str | None = None,
    expected_container_image: str | None = None,
    expected_code_artifact_ref: str | None = None,
    expected_code_artifact_digest: str | None = None,
    expected_command_sha256: str | None = None,
    expected_polling_artifact_ref: str | None = None,
    profile: str = TRAINING_PROFILE,
) -> dict[str, Any]:
    profile = _require_profile(profile)
    if not isinstance(binding, dict):
        raise ValueError("orchestrator receipt binding must be one JSON object")
    if binding.get("version") != BINDING_VERSION:
        raise ValueError("orchestrator receipt binding version mismatch")
    unsigned = dict(binding)
    signature = unsigned.pop("binding_sha256", None)
    if signature != common.canonical_json_sha256(unsigned):
        raise ValueError("orchestrator receipt binding signature mismatch")
    source = binding.get("source")
    if not isinstance(source, dict):
        raise ValueError("orchestrator receipt binding lacks source")
    require_qualified_immutable_ref(
        source.get("artifact_ref"), field="orchestrator receipt source")
    _require_nonempty_string(
        source.get("artifact_digest"), "orchestrator receipt source digest")
    if source.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError("orchestrator receipt source artifact type mismatch")
    receipt = binding.get("receipt")
    if not isinstance(receipt, dict) or receipt.get("version") != RECEIPT_VERSION:
        raise ValueError("orchestrator receipt binding lacks validated receipt")
    if receipt.get("profile") != profile:
        raise ValueError("orchestrator receipt binding profile mismatch")
    receipt_unsigned = dict(receipt)
    receipt_signature = receipt_unsigned.pop("receipt_sha256", None)
    if receipt_signature != common.canonical_json_sha256(receipt_unsigned):
        raise ValueError("embedded orchestrator receipt signature mismatch")
    if binding.get("artifact_metadata") != receipt_artifact_metadata(receipt):
        raise ValueError("orchestrator binding metadata/receipt mismatch")
    files = binding.get("artifact_files")
    if not isinstance(files, dict) or files.get("version") != \
            common.ARTIFACT_DOWNLOAD_MANIFEST_VERSION:
        raise ValueError("orchestrator binding lacks artifact file manifest")
    if files.get("source") != {
        "artifact_ref": source["artifact_ref"],
        "artifact_digest": source["artifact_digest"],
    }:
        raise ValueError("orchestrator binding artifact-file source mismatch")
    member_names = [
        item.get("relative_path") for item in files.get("files", [])
        if isinstance(item, dict)
    ]
    if member_names != sorted(EXPECTED_ARTIFACT_MEMBERS):
        raise ValueError("orchestrator binding artifact membership mismatch")
    normalized_files = {
        "version": files["version"],
        "source": files["source"],
        "files": files["files"],
    }
    if files.get("file_manifest_sha256") != \
            common.canonical_json_sha256(normalized_files):
        raise ValueError("orchestrator binding file-manifest signature mismatch")

    job = receipt.get("job")
    if not isinstance(job, dict):
        raise ValueError("orchestrator receipt binding lacks job facts")
    expected_pairs = (
        ("run name", expected_run_name, job.get("run_name")),
        ("container image", expected_container_image,
         job.get("container_image_identity")),
        ("code artifact ref", expected_code_artifact_ref,
         job.get("code_artifact", {}).get("artifact_ref")
         if isinstance(job.get("code_artifact"), dict) else None),
        ("code artifact digest", expected_code_artifact_digest,
         job.get("code_artifact", {}).get("artifact_digest")
         if isinstance(job.get("code_artifact"), dict) else None),
        ("command SHA256", expected_command_sha256,
         job.get("command_sha256")),
        ("polling artifact ref", expected_polling_artifact_ref,
         job.get("environment", {}).get("ORCHESTRATOR_RECEIPT_ARTIFACT")
         if isinstance(job.get("environment"), dict) else None),
    )
    for label, expected, observed in expected_pairs:
        if expected is not None and str(expected) != str(observed):
            raise ValueError(
                f"orchestrator receipt {label} mismatch: expected "
                f"{expected!r}, got {observed!r}")
    if job.get("gpu_count") != 8 or job.get("accelerator_family") not in {
            "A100", "H100"}:
        raise ValueError("orchestrator receipt hardware facts are not 8x A100/H100")
    return binding


def load_binding(
    path: str, *, expected_run_name: str | None = None,
    expected_container_image: str | None = None,
    expected_code_artifact_ref: str | None = None,
    expected_code_artifact_digest: str | None = None,
    expected_command_sha256: str | None = None,
    expected_polling_artifact_ref: str | None = None,
    profile: str = TRAINING_PROFILE,
) -> dict[str, Any]:
    if not path or not os.path.isfile(path):
        raise ValueError(
            f"--orchestrator-receipt-binding is not a file: {path!r}")
    with open(path, encoding="utf-8") as handle:
        binding = json.load(handle)
    return validate_binding(
        binding,
        expected_run_name=expected_run_name,
        expected_container_image=expected_container_image,
        expected_code_artifact_ref=expected_code_artifact_ref,
        expected_code_artifact_digest=expected_code_artifact_digest,
        expected_command_sha256=expected_command_sha256,
        expected_polling_artifact_ref=expected_polling_artifact_ref,
        profile=profile,
    )


def observed_native_compatibility_facts() -> dict[str, Any]:
    """Capture hardware/native facts without freezing A100 to H100 values."""
    import torch

    cuda_available = bool(torch.cuda.is_available())
    count = int(torch.cuda.device_count()) if cuda_available else 0
    gpu_names = [str(torch.cuda.get_device_name(index)) for index in range(count)]
    capabilities = [
        list(torch.cuda.get_device_capability(index)) for index in range(count)
    ]
    try:
        cudnn = torch.backends.cudnn.version()
    except Exception:  # pragma: no cover - depends on native build
        cudnn = None
    try:
        nccl_raw = torch.cuda.nccl.version() if count else None
        nccl = list(nccl_raw) if isinstance(nccl_raw, tuple) else nccl_raw
    except Exception:  # pragma: no cover - depends on native build
        nccl = None
    driver_versions: list[str] = []
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=10,
        )
        driver_versions = [
            line.strip() for line in proc.stdout.splitlines() if line.strip()
        ]
    except (FileNotFoundError, subprocess.SubprocessError):
        driver_versions = []
    families = []
    for name in gpu_names:
        upper = name.upper()
        families.append(
            "A100" if "A100" in upper else
            "H100" if "H100" in upper else None)
    family = families[0] if families and len(set(families)) == 1 else None
    return {
        "torch_cuda_version": str(torch.version.cuda)
        if torch.version.cuda is not None else None,
        "cudnn_version": int(cudnn) if cudnn is not None else None,
        "nccl_version": nccl,
        "visible_gpu_count": count,
        "gpu_names": gpu_names,
        "gpu_family": family,
        "gpu_capabilities": capabilities,
        "driver_versions": driver_versions,
    }


def validate_native_compatibility_facts(
    facts: Any, *, expected_accelerator_family: str | None = None,
) -> dict[str, Any]:
    if not isinstance(facts, dict):
        raise ValueError("native compatibility facts must be one object")
    expected_keys = {
        "torch_cuda_version", "cudnn_version", "nccl_version",
        "visible_gpu_count", "gpu_names", "gpu_family", "gpu_capabilities",
        "driver_versions",
    }
    if set(facts) != expected_keys:
        raise ValueError("native compatibility fact fields mismatch")
    if facts.get("visible_gpu_count") != 8:
        raise ValueError("strict execution requires exactly 8 visible GPUs")
    family = facts.get("gpu_family")
    if family not in {"A100", "H100"}:
        raise ValueError("strict execution requires observed A100 or H100 GPUs")
    if expected_accelerator_family is not None and family != \
            expected_accelerator_family:
        raise ValueError(
            "observed GPU family disagrees with Lepton resource shape")
    names = facts.get("gpu_names")
    if not isinstance(names, list) or len(names) != 8 or not all(
            isinstance(name, str) and name for name in names):
        raise ValueError("strict execution lacks all 8 observed GPU names")
    expected_token = str(family)
    if not all(expected_token in name.upper() for name in names):
        raise ValueError("strict execution GPU names are not one homogeneous family")
    capabilities = facts.get("gpu_capabilities")
    if not isinstance(capabilities, list) or len(capabilities) != 8 or not all(
            isinstance(capability, list) and len(capability) == 2 and all(
                isinstance(value, int) for value in capability)
            for capability in capabilities):
        raise ValueError("strict execution lacks all 8 GPU compute capabilities")
    drivers = facts.get("driver_versions")
    if not isinstance(drivers, list) or len(drivers) != 8 or not all(
            isinstance(value, str) and value for value in drivers):
        raise ValueError("strict execution lacks all 8 GPU driver observations")
    if not facts.get("torch_cuda_version"):
        raise ValueError("strict execution lacks torch CUDA version")
    if not isinstance(facts.get("cudnn_version"), int) or \
            int(facts["cudnn_version"]) <= 0:
        raise ValueError("strict execution lacks cuDNN version")
    nccl = facts.get("nccl_version")
    if nccl is None or nccl == "" or nccl == []:
        raise ValueError("strict execution lacks NCCL version")
    return facts


def write_receipt_bundle(
    raw_spec_path: str, output_dir: str, *, profile: str = TRAINING_PROFILE,
) -> dict[str, Any]:
    with open(raw_spec_path, encoding="utf-8") as handle:
        raw_spec = json.load(handle)
    receipt = build_receipt(raw_spec, profile=profile)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    existing = sorted(path.name for path in root.iterdir())
    if existing:
        raise ValueError(
            f"receipt output directory must be empty; found {existing!r}")
    common.atomic_write_json(str(root / RAW_SPEC_FILENAME), raw_spec)
    common.atomic_write_json(str(root / RECEIPT_FILENAME), receipt)
    return receipt


def _artifact_download_manifest(
    *, artifact: Any, source_ref: str, source_digest: str, root: str,
) -> dict[str, Any]:
    remote_names = sorted(str(item.name) for item in artifact.files())
    if remote_names != sorted(EXPECTED_ARTIFACT_MEMBERS):
        raise ValueError(
            "orchestrator receipt artifact remote membership mismatch: "
            f"{remote_names!r}")
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root, exist_ok=False)
    artifact.download(root=root)
    files = []
    for relative in remote_names:
        path = os.path.join(root, relative)
        if os.path.islink(path) or not os.path.isfile(path):
            raise ValueError(
                f"orchestrator receipt artifact member is not regular: {relative}")
        files.append({
            "relative_path": relative,
            "sha256": common.file_sha256(path),
            "size_bytes": os.path.getsize(path),
        })
    normalized = {
        "version": common.ARTIFACT_DOWNLOAD_MANIFEST_VERSION,
        "source": {
            "artifact_ref": source_ref,
            "artifact_digest": source_digest,
        },
        "files": files,
    }
    manifest = {
        **normalized,
        "file_manifest_sha256": common.canonical_json_sha256(normalized),
        "download_root": os.path.realpath(root),
    }
    common.validate_artifact_download_manifest(manifest)
    return manifest


def _resolved_immutable_artifact_ref(
    artifact: Any, *, requested_ref: str,
) -> str:
    """Return the immutable identity behind one W&B artifact lookup.

    W&B 0.28 retains the lookup alias in ``qualified_name`` when an artifact
    is fetched through ``:latest``.  The immutable identity is exposed through
    ``source_qualified_name`` instead.  Never manufacture a versioned identity
    from the mutable alias: require W&B to provide an immutable source and
    bind it to the exact artifact sequence that was requested.
    """
    requested_ref = require_pollable_ref(
        requested_ref, field="orchestrator receipt artifact")
    requested_sequence, requested_alias = requested_ref.rsplit(":", 1)
    qualified_ref = require_pollable_ref(
        getattr(artifact, "qualified_name", ""),
        field="resolved orchestrator receipt lookup",
    )
    if qualified_ref != requested_ref:
        raise ValueError(
            "resolved orchestrator receipt lookup identity mismatch")

    source_ref = getattr(artifact, "source_qualified_name", "")
    resolved_ref = require_qualified_immutable_ref(
        source_ref, field="resolved orchestrator receipt artifact")
    resolved_sequence, resolved_version = resolved_ref.rsplit(":", 1)
    if resolved_sequence != requested_sequence:
        raise ValueError(
            "resolved orchestrator receipt source sequence mismatch")
    if requested_alias != "latest" and resolved_ref != requested_ref:
        raise ValueError(
            "resolved orchestrator receipt source differs from immutable "
            "request")

    reported_version = _require_nonempty_string(
        getattr(artifact, "version", None),
        "resolved orchestrator receipt artifact version",
    )
    source_version = _require_nonempty_string(
        getattr(artifact, "source_version", None),
        "resolved orchestrator receipt artifact source version",
    )
    if re.fullmatch(r"v[0-9]+", reported_version) is None or re.fullmatch(
            r"v[0-9]+", source_version) is None:
        raise ValueError(
            "resolved orchestrator receipt artifact version is invalid")
    if reported_version != resolved_version or source_version != \
            resolved_version:
        raise ValueError(
            "resolved orchestrator receipt artifact version mismatch")
    return resolved_ref


def resolve_artifact_binding(
    artifact_ref: str, *, output_binding: str, download_root: str,
    expected_run_name: str, expected_container_image: str,
    expected_code_artifact_ref: str, expected_code_artifact_digest: str,
    expected_command_sha256: str,
    profile: str = TRAINING_PROFILE,
    poll_seconds: float = 10.0, timeout_seconds: float = 1800.0,
    api: Any = None,
) -> dict[str, Any]:
    profile = _require_profile(profile)
    requested_ref = require_pollable_ref(
        artifact_ref, field="orchestrator receipt artifact")
    requested_name = requested_ref.rsplit("/", 1)[-1].rsplit(":", 1)[0]
    expected_name = common.sanitize_wandb_id(
        f"{expected_run_name}-orchestrator-receipt")
    if requested_name != expected_name:
        raise ValueError(
            "orchestrator receipt artifact must be uniquely named for RUN_NAME: "
            f"expected {expected_name!r}, got {requested_name!r}")
    if api is None:
        import wandb
        api = wandb.Api()
    deadline = time.monotonic() + float(timeout_seconds)
    last_error: Exception | None = None
    artifact = None
    while artifact is None:
        try:
            artifact = api.artifact(requested_ref, type=ARTIFACT_TYPE)
        except Exception as exc:
            last_error = exc
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "timed out waiting for orchestrator receipt artifact "
                    f"{requested_ref!r}: {last_error}") from exc
            time.sleep(max(0.05, float(poll_seconds)))
    resolved_ref = _resolved_immutable_artifact_ref(
        artifact, requested_ref=requested_ref)
    source_digest = _require_nonempty_string(
        getattr(artifact, "digest", ""),
        "resolved orchestrator receipt artifact digest",
    )
    if getattr(artifact, "type", None) != ARTIFACT_TYPE:
        raise ValueError("resolved orchestrator receipt artifact type mismatch")
    digest_root = os.path.join(
        download_root, re.sub(r"[^A-Za-z0-9_.-]", "_", source_digest))
    manifest = _artifact_download_manifest(
        artifact=artifact, source_ref=resolved_ref,
        source_digest=source_digest, root=digest_root)
    binding = build_binding(
        source_ref=resolved_ref,
        source_digest=source_digest,
        artifact_type=ARTIFACT_TYPE,
        artifact_metadata=dict(getattr(artifact, "metadata", {}) or {}),
        download_manifest=manifest,
        profile=profile,
    )
    validate_binding(
        binding,
        expected_run_name=expected_run_name,
        expected_container_image=expected_container_image,
        expected_code_artifact_ref=expected_code_artifact_ref,
        expected_code_artifact_digest=expected_code_artifact_digest,
        expected_command_sha256=expected_command_sha256,
        expected_polling_artifact_ref=(
            requested_ref if requested_ref.endswith(":latest") else
            f"{requested_ref.rsplit(':', 1)[0]}:latest"
        ),
        profile=profile,
    )
    common.atomic_write_json(output_binding, binding)
    return binding


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    bundle = sub.add_parser(
        "build", help="build the exact two-file receipt artifact payload")
    bundle.add_argument("--raw-spec", required=True)
    bundle.add_argument("--output-dir", required=True)
    bundle.add_argument(
        "--profile", choices=RECEIPT_PROFILES, default=TRAINING_PROFILE)

    command_digest = sub.add_parser(
        "command-digest",
        help="compute the command SHA256 to preregister in a job spec",
    )
    command_digest.add_argument("--raw-spec", required=True)

    resolve = sub.add_parser(
        "resolve", help="poll, resolve, verify, and bind a receipt artifact")
    resolve.add_argument("--artifact-ref", required=True)
    resolve.add_argument("--output-binding", required=True)
    resolve.add_argument("--download-root", required=True)
    resolve.add_argument("--expected-run-name", required=True)
    resolve.add_argument("--expected-container-image", required=True)
    resolve.add_argument("--expected-code-artifact-ref", required=True)
    resolve.add_argument("--expected-code-artifact-digest", required=True)
    resolve.add_argument("--expected-command-sha256", required=True)
    resolve.add_argument(
        "--profile", choices=RECEIPT_PROFILES, default=TRAINING_PROFILE)
    resolve.add_argument("--poll-seconds", type=float, default=10.0)
    resolve.add_argument("--timeout-seconds", type=float, default=1800.0)

    publish = sub.add_parser(
        "publish", help="build and upload one RUN_NAME-unique receipt artifact")
    publish.add_argument("--raw-spec", required=True)
    publish.add_argument("--entity", required=True)
    publish.add_argument("--project", required=True)
    publish.add_argument("--run-name", required=True)
    publish.add_argument(
        "--profile", choices=RECEIPT_PROFILES, default=TRAINING_PROFILE)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    if args.command == "build":
        receipt = write_receipt_bundle(
            args.raw_spec, args.output_dir, profile=args.profile)
        print(json.dumps({
            "receipt_sha256": receipt["receipt_sha256"],
            "artifact_metadata": receipt_artifact_metadata(receipt),
        }, sort_keys=True))
        return
    if args.command == "command-digest":
        with open(args.raw_spec, encoding="utf-8") as handle:
            raw_spec = json.load(handle)
        if not isinstance(raw_spec, dict) or not isinstance(
                raw_spec.get("container"), dict):
            raise ValueError("job spec lacks container object")
        normalized = _raw_command(raw_spec["container"].get("command"))
        print(json.dumps({
            "normalized_command": normalized,
            "command_sha256": common.canonical_json_sha256(normalized),
        }, sort_keys=True))
        return
    if args.command == "publish":
        import wandb

        with tempfile.TemporaryDirectory(
                prefix="tmx-orchestrator-receipt-") as tmp:
            receipt = write_receipt_bundle(
                args.raw_spec, tmp, profile=args.profile)
            if receipt["job"]["run_name"] != args.run_name:
                raise ValueError("publish --run-name differs from raw job spec")
            artifact_name = common.sanitize_wandb_id(
                f"{args.run_name}-orchestrator-receipt")
            expected_polling_ref = (
                f"{args.entity}/{args.project}/{artifact_name}:latest")
            if receipt["job"]["environment"][
                    "ORCHESTRATOR_RECEIPT_ARTIFACT"] != expected_polling_ref:
                raise ValueError(
                    "raw job spec receipt alias differs from publish destination")
            run = wandb.init(
                entity=args.entity,
                project=args.project,
                name=common.sanitize_wandb_id(
                    f"{args.run_name}-orchestrator-receipt-publish"),
                job_type="orchestrator-receipt",
                reinit=True,
            )
            try:
                artifact = wandb.Artifact(
                    artifact_name,
                    type=ARTIFACT_TYPE,
                    metadata=receipt_artifact_metadata(receipt),
                )
                for filename in EXPECTED_ARTIFACT_MEMBERS:
                    artifact.add_file(os.path.join(tmp, filename), name=filename)
                logged = run.log_artifact(artifact, aliases=["latest"])
                logged.wait()
                resolved_ref = require_qualified_immutable_ref(
                    getattr(logged, "qualified_name", ""),
                    field="published orchestrator receipt artifact",
                )
                digest = _require_nonempty_string(
                    getattr(logged, "digest", ""),
                    "published orchestrator receipt digest",
                )
            finally:
                run.finish()
        print(json.dumps({
            "artifact_ref": resolved_ref,
            "artifact_digest": digest,
            "polling_ref": f"{args.entity}/{args.project}/{artifact_name}:latest",
            "command_sha256": receipt["job"]["command_sha256"],
        }, sort_keys=True))
        return
    binding = resolve_artifact_binding(
        args.artifact_ref,
        output_binding=args.output_binding,
        download_root=args.download_root,
        expected_run_name=args.expected_run_name,
        expected_container_image=args.expected_container_image,
        expected_code_artifact_ref=args.expected_code_artifact_ref,
        expected_code_artifact_digest=args.expected_code_artifact_digest,
        expected_command_sha256=args.expected_command_sha256,
        profile=args.profile,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps({
        "artifact_ref": binding["source"]["artifact_ref"],
        "artifact_digest": binding["source"]["artifact_digest"],
        "binding_sha256": binding["binding_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
