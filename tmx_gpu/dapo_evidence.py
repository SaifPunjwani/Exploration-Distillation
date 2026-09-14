"""Fail-closed pairing for original-paper DAPO crash-resume evidence."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

from tmx_gpu import common

SCIENTIFIC_VARIANT = "original_paper_dapo_100_central_lr"
MODEL = "Qwen/Qwen3-1.7B"
DAPO_METHOD = "DAPO-style correctness"
DAPO_COMPLETION_BINDING_VERSION = "dapo_completion_binding_v1"
DAPO_COMPLETION_BINDING_FILENAME = "dapo_completion_binding.json"
DAPO_CAMPAIGN_ARTIFACT_TYPE = "dapo-training-campaign"
DAPO_RESULTS_ARTIFACT_TYPE = "dapo-results"
DAPO_CHECKPOINT_ARTIFACT_TYPE = "checkpoint"
GRPO_FINAL_MODEL_ARTIFACT_BINDING_VERSION = "grpo_final_model_artifact_v1"
EXPECTED_STEPS = 100


def _signed_payload(version: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if "version" in payload or "binding_sha256" in payload:
        raise ValueError("DAPO binding payload pre-populates signature fields")
    value = {"version": str(version), **payload}
    value["binding_sha256"] = common.canonical_json_sha256(value)
    return value


def _validate_signed_payload(
    value: Any, *, expected_version: str,
) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("DAPO completion binding must be one JSON object")
    unsigned = dict(value)
    recorded = unsigned.pop("binding_sha256", None)
    if unsigned.get("version") != expected_version:
        raise ValueError("DAPO completion binding version mismatch")
    if recorded != common.canonical_json_sha256(unsigned):
        raise ValueError("DAPO completion binding SHA256 is missing or invalid")
    return dict(value)


def _artifact_identity(
    *, ref: str, digest: str, artifact_type: str, expected_type: str,
    field: str,
) -> Dict[str, str]:
    ref = common.require_immutable_wandb_artifact_ref(ref, field=field)
    if not str(digest):
        raise ValueError(f"{field} lacks an artifact digest")
    if str(artifact_type) != expected_type:
        raise ValueError(
            f"{field} type {artifact_type!r} differs from {expected_type!r}")
    return {
        "artifact_ref": ref,
        "artifact_digest": str(digest),
        "artifact_type": expected_type,
    }


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"malformed JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(row)
    return rows


def _validate_download(root: Path, manifest: Dict[str, Any], version: str) -> None:
    if manifest.get("version") != version:
        raise ValueError(f"download manifest version mismatch: {manifest.get('version')!r}")
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise ValueError("download manifest lacks source")
    common.require_immutable_wandb_artifact_ref(
        str(source.get("artifact_ref", "")), field="resume artifact")
    if not source.get("artifact_digest"):
        raise ValueError("download manifest lacks artifact digest")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("download manifest has no files")
    expected = set()
    for entry in entries:
        relative = str(entry.get("relative_path", ""))
        if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"unsafe artifact member: {relative!r}")
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing/non-regular artifact member: {relative}")
        if common.file_sha256(str(path)) != str(entry.get("sha256", "")):
            raise ValueError(f"artifact member SHA256 mismatch: {relative}")
        if path.stat().st_size != int(entry.get("size_bytes", -1)):
            raise ValueError(f"artifact member size mismatch: {relative}")
        if relative in expected:
            raise ValueError(f"duplicate artifact member: {relative}")
        expected.add(relative)
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*") if path.is_file()
    }
    if actual != expected:
        raise ValueError("artifact-root membership differs from download manifest")


def _single_state(root: Path) -> tuple[Path, Dict[str, Any]]:
    paths = sorted(root.rglob("trainer_state.json"))
    if len(paths) != 1:
        raise ValueError(
            f"artifact must contain exactly one trainer_state.json; got {len(paths)}")
    return paths[0], _load_json(paths[0])


def _require_runtime(state: Dict[str, Any]) -> int:
    step = int(state.get("step", -1))
    if not 1 <= step <= 100:
        raise ValueError(f"checkpoint step must be in [1,100], got {step}")
    if state.get("phase") != "central" or state.get("model") != MODEL:
        raise ValueError("checkpoint phase/model differs from DAPO contract")
    if state.get("rl_validity_mode") != "c8" or \
            state.get("optimizer_state_saved") is not True:
        raise ValueError("checkpoint lacks c8 optimizer-complete evidence")
    if state.get("checkpoint_parameter_dtype") != "float32":
        raise ValueError("checkpoint does not preserve FP32 learner parameters")
    if state.get("policy_attention_implementation") not in {
            "flash_attention_2", "sdpa"}:
        raise ValueError("checkpoint lacks a pinned policy attention implementation")
    if not math.isclose(float(state.get("resolved_lr", -1)), 1e-6,
                        rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("checkpoint learning rate differs from 1e-6")
    expected_tier = f"contract+scientific-variant:{SCIENTIFIC_VARIANT}"
    if state.get("config_tier") != expected_tier:
        raise ValueError("checkpoint config tier differs from DAPO contract")
    if state.get("external_lineage") is not None or \
            state.get("external_lineage_sha256") is not None:
        raise ValueError("single-model DAPO checkpoint unexpectedly has external lineage")
    runtime = state.get("runtime_contract")
    if not isinstance(runtime, dict):
        raise ValueError("checkpoint lacks full runtime contract")
    from tmx_jax.data import training_dataset_identity
    from tmx_jax.prompting import prompt_metadata
    from tmx_gpu import grpo_gpu

    expected = {
        "version": grpo_gpu.GRPO_RUNTIME_CONTRACT_VERSION,
        "phase": "central",
        "scientific_variant": SCIENTIFIC_VARIANT,
        "config_tier": expected_tier,
        "model_init": MODEL,
        "base_model_revision": common.CANONICAL_TRAINING_MODEL_REVISION,
        "expected_steps": 100,
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
        "trajectory_record_schema_version": (
            grpo_gpu.TRAJECTORY_RECORD_SCHEMA_VERSION),
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
        "mask_truncated": False,
        "clip_epsilon_low": 0.2,
        "clip_epsilon_high": 0.28,
        "ppo_mode": False,
        "updates_per_rollout": 1,
        "incorrect_novelty_scale": 0.0,
        "lambda_novelty": 0.0,
        "dataset": "dapo_math_17k",
        "training_dataset_identity": training_dataset_identity(
            "dapo_math_17k"),
        "max_train_examples": 20000,
        "seed": 0,
        "round": 1,
        "max_consecutive_generation_failures": 3,
        "generation_timeout_seconds": 3600.0,
        "generation_concurrency_per_server": 8,
        "chunk_tokens": 256,
        # Native CUDA/GPU facts are compatibility evidence, so accept either
        # allowed 8x A100/H100 observation after validating the immutable
        # image/runtime/orchestrator-receipt invariants below.
        "execution_environment": runtime.get("execution_environment"),
    }
    grpo_gpu.require_canonical_training_execution_environment_contract(runtime)
    if runtime != expected:
        keys = sorted(set(runtime) | set(expected))
        mismatches = {
            key: (runtime.get(key), expected.get(key))
            for key in keys if runtime.get(key) != expected.get(key)
        }
        raise ValueError(f"checkpoint runtime contract mismatch: {mismatches}")
    return step


def _require_metadata(
    checkpoint_manifest: Dict[str, Any], results_manifest: Dict[str, Any], step: int,
) -> None:
    if checkpoint_manifest.get("artifact_type") != "checkpoint":
        raise ValueError("resume model artifact type must be checkpoint")
    ckpt_meta = checkpoint_manifest.get("artifact_metadata") or {}
    if (int(ckpt_meta.get("step", -1)) != step
            or int(ckpt_meta.get("round", -1)) != 1
            or ckpt_meta.get("phase") != "central"):
        raise ValueError("checkpoint artifact metadata differs from trainer state")
    result_type = results_manifest.get("artifact_type")
    if result_type not in {"dapo-results", "grpo-progress"}:
        raise ValueError("resume evidence artifact has the wrong type")
    result_meta = results_manifest.get("artifact_metadata") or {}
    if result_meta.get("scientific_variant") != SCIENTIFIC_VARIANT:
        raise ValueError("resume evidence scientific variant mismatch")
    if result_type == "grpo-progress" and (
        int(result_meta.get("step", -1)) != step
        or int(result_meta.get("round", -1)) != 1
        or result_meta.get("phase") != "central"
        or result_meta.get("model") != MODEL
    ):
        raise ValueError("progress artifact metadata differs from selected checkpoint")


def _step_counts(rows: Iterable[Dict[str, Any]], *, key: str) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for row in rows:
        step = int(row.get(key, -1))
        counts[step] = counts.get(step, 0) + 1
    return counts


def _require_complete_history(
    result_root: Path, checkpoint_step: int,
) -> Dict[str, Any]:
    expected_steps = set(range(1, checkpoint_step + 1))
    metrics = _jsonl(result_root / "metrics.jsonl")
    completed = [
        row for row in metrics
        if row.get("phase") == "central_grpo" and "train/grpo_loss" in row
    ]
    metric_counts = _step_counts(completed, key="global_step")
    for step in sorted(expected_steps):
        if metric_counts.get(step, 0) != 1:
            raise ValueError(
                "metrics must contain exactly one completed Central-route training row "
                f"for every step 1..{checkpoint_step}; step {step} has "
                f"{metric_counts.get(step, 0)}")
    if set(metric_counts) != expected_steps:
        raise ValueError(
            "metrics completed-step membership differs from the exact DAPO range")

    explorer = _jsonl(result_root / "explorer_trajectories.jsonl")
    explorer_counts = _step_counts(explorer, key="step")
    for step in sorted(expected_steps):
        if explorer_counts.get(step, 0) != 64:
            raise ValueError(
                "explorer_trajectories.jsonl must contain exactly 64 selected "
                f"rows for every step 1..{checkpoint_step}; step {step} has "
                f"{explorer_counts.get(step, 0)}")
    if set(explorer_counts) != expected_steps:
        raise ValueError(
            "explorer trajectory step membership differs from the exact DAPO range")

    rollouts = _jsonl(result_root / "rollouts_all.jsonl")
    rollout_counts = _step_counts(rollouts, key="step")
    for step in sorted(expected_steps):
        if rollout_counts.get(step, 0) < 64:
            raise ValueError(
                "rollouts_all.jsonl must contain at least 64 candidate rows "
                f"for every step 1..{checkpoint_step}; step {step} has "
                f"{rollout_counts.get(step, 0)}")
    if set(rollout_counts) != expected_steps:
        raise ValueError(
            "candidate rollout step membership differs from the exact DAPO range")
    return {
        "step_count": int(checkpoint_step),
        "completed_metric_row_count": len(completed),
        "selected_trajectory_row_count": len(explorer),
        "candidate_rollout_row_count": len(rollouts),
        "metrics_file_sha256": common.file_sha256(
            str(result_root / "metrics.jsonl")),
        "selected_trajectories_file_sha256": common.file_sha256(
            str(result_root / "explorer_trajectories.jsonl")),
        "candidate_rollouts_file_sha256": common.file_sha256(
            str(result_root / "rollouts_all.jsonl")),
        "completed_metrics_sha256": common.canonical_json_sha256(completed),
    }


def _atomic_copy_lines(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".tmp")
    with source.open("rb") as src, tmp.open("wb") as dst:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
        dst.flush()
        os.fsync(dst.fileno())
    os.replace(tmp, destination)


def validate_and_restore_resume_pair(
    *, code_binding_path: str, checkpoint_root: str,
    checkpoint_manifest_path: str, results_root: str,
    results_manifest_path: str, output_dir: str,
) -> Dict[str, Any]:
    """Validate two immutable artifacts as one checkpoint/evidence pair.

    No result JSONL is copied into ``output_dir`` until every state, runtime,
    artifact-metadata, byte-manifest, and learner-step check succeeds.
    """
    code_binding = _load_json(Path(code_binding_path))
    ckpt_root = Path(checkpoint_root).resolve()
    result_root = Path(results_root).resolve()
    checkpoint_manifest = _load_json(Path(checkpoint_manifest_path))
    results_manifest = _load_json(Path(results_manifest_path))
    _validate_download(
        ckpt_root, checkpoint_manifest, "wandb_checkpoint_download_v1")
    _validate_download(
        result_root, results_manifest, "wandb_results_download_v1")
    checkpoint_state_path, checkpoint_state = _single_state(ckpt_root)
    _result_state_path, result_state = _single_state(result_root)
    if result_state != checkpoint_state:
        raise ValueError("results and checkpoint trainer states are not identical")
    if checkpoint_state.get("code_artifact_binding") != code_binding:
        raise ValueError("checkpoint code binding differs from current exact code")
    step = _require_runtime(checkpoint_state)
    _require_metadata(checkpoint_manifest, results_manifest, step)
    recorded_manifest = checkpoint_state.get("checkpoint_file_manifest")
    if not isinstance(recorded_manifest, dict):
        raise ValueError("checkpoint lacks its internal file manifest")
    common.verify_directory_file_manifest(
        str(checkpoint_state_path.parent), recorded_manifest,
        exclude_relative_paths=("trainer_state.json",))

    result_binding = _load_json(result_root / "dapo_code_binding.json")
    if result_binding != code_binding:
        raise ValueError("results code binding differs from current exact code")
    _require_complete_history(result_root, step)

    output = Path(output_dir)
    conflicts = sorted(output.glob("step_[0-9]*"))
    if conflicts:
        raise ValueError(f"unproven local checkpoints conflict with resume: {conflicts}")
    for name in ("metrics.jsonl", "explorer_trajectories.jsonl", "rollouts_all.jsonl"):
        _atomic_copy_lines(result_root / name, output / name)
    return {
        "checkpoint_step": step,
        "checkpoint_artifact": checkpoint_manifest["source"],
        "results_artifact": results_manifest["source"],
        "code_file_manifest_sha256": code_binding["file_manifest_sha256"],
    }


def _validate_stable_code_binding(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict) or value.get("version") != \
            common.ARTIFACT_DOWNLOAD_MANIFEST_VERSION:
        raise ValueError("DAPO completion lacks a stable code-artifact binding")
    source = value.get("source")
    if not isinstance(source, dict):
        raise ValueError("DAPO code binding lacks its immutable source")
    common.require_immutable_wandb_artifact_ref(
        str(source.get("artifact_ref", "")), field="DAPO training code artifact")
    if not source.get("artifact_digest"):
        raise ValueError("DAPO training code artifact lacks a digest")
    files = value.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("DAPO code binding has no file manifest")
    normalized_files = []
    seen = set()
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("DAPO code binding contains a non-object file entry")
        relative = str(item.get("relative_path", ""))
        path = Path(relative)
        if not relative or path.is_absolute() or ".." in path.parts or relative in seen:
            raise ValueError("DAPO code binding contains an unsafe/duplicate path")
        sha256 = str(item.get("sha256", ""))
        size = item.get("size_bytes")
        if len(sha256) != 64 or isinstance(size, bool) or not isinstance(size, int) \
                or size < 0:
            raise ValueError("DAPO code binding contains an invalid file identity")
        seen.add(relative)
        normalized_files.append({
            "relative_path": relative,
            "sha256": sha256,
            "size_bytes": size,
        })
    normalized = {
        "version": common.ARTIFACT_DOWNLOAD_MANIFEST_VERSION,
        "source": {
            "artifact_ref": str(source["artifact_ref"]),
            "artifact_digest": str(source["artifact_digest"]),
        },
        "files": sorted(normalized_files, key=lambda item: item["relative_path"]),
    }
    expected_sha = common.canonical_json_sha256(normalized)
    if value.get("file_manifest_sha256") != expected_sha:
        raise ValueError("DAPO code binding file-manifest SHA256 mismatch")
    expected_keys = set(normalized) | {"file_manifest_sha256"}
    if set(value) != expected_keys:
        raise ValueError("DAPO stable code binding has unexpected/missing fields")
    return {**normalized, "file_manifest_sha256": expected_sha}


def _validate_final_model_binding(
    value: Any, *, model_root: Path, expected_model_artifact: Dict[str, str],
    expected_model_metadata: Dict[str, Any], expected_runtime: Dict[str, Any],
    expected_code_binding: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("DAPO completion lacks its final-model artifact binding")
    unsigned = dict(value)
    recorded = unsigned.pop("binding_sha256", None)
    if recorded != common.canonical_json_sha256(unsigned):
        raise ValueError("DAPO final-model artifact binding SHA256 mismatch")
    if value.get("version") != GRPO_FINAL_MODEL_ARTIFACT_BINDING_VERSION:
        raise ValueError("DAPO final-model artifact binding version mismatch")
    if value.get("source") != expected_model_artifact:
        raise ValueError("DAPO final-model source differs from the resolved artifact")
    if value.get("artifact_subdir") != "." or value.get("phase") != "central" \
            or value.get("round") != 1 or value.get("step") != EXPECTED_STEPS:
        raise ValueError("DAPO final-model artifact binding has wrong stage identity")
    runtime_sha = common.canonical_json_sha256(expected_runtime)
    if value.get("runtime_contract_sha256") != runtime_sha:
        raise ValueError("DAPO final-model binding has the wrong runtime contract")
    if value.get("artifact_metadata") != expected_model_metadata:
        raise ValueError("DAPO final-model binding metadata differs from W&B")
    model_manifest = value.get("model_file_manifest")
    if not isinstance(model_manifest, dict):
        raise ValueError("DAPO final-model binding lacks a model-file manifest")
    common.verify_directory_file_manifest(str(model_root), model_manifest)
    state = _load_json(model_root / "trainer_state.json")
    if state.get("step") != EXPECTED_STEPS or state.get("phase") != "central" or \
            state.get("model") != MODEL or state.get("round") != 1:
        raise ValueError("DAPO final model trainer state has wrong stage identity")
    if state.get("runtime_contract") != expected_runtime or \
            state.get("code_artifact_binding") != expected_code_binding:
        raise ValueError("DAPO final model runtime/code differs from training evidence")
    if state.get("optimizer_state_saved") is not False:
        raise ValueError("DAPO deployable final model must not claim optimizer state")
    policy_manifest = value.get("policy_file_manifest")
    if policy_manifest != state.get("checkpoint_file_manifest"):
        raise ValueError("DAPO final model policy manifest differs from trainer state")
    common.verify_directory_file_manifest(
        str(model_root), policy_manifest, exclude_relative_paths=("trainer_state.json",))
    expected_dataset_sha = expected_runtime["training_dataset_identity"][
        "identity_sha256"]
    expected_metadata = {
        "step": EXPECTED_STEPS,
        "round": 1,
        "phase": "central",
        "model_revision": common.CANONICAL_TRAINING_MODEL_REVISION,
        "runtime_contract_sha256": runtime_sha,
        "training_dataset_identity_sha256": expected_dataset_sha,
    }
    if expected_model_metadata != expected_metadata:
        raise ValueError("DAPO final model artifact metadata is not the exact contract")
    return dict(value)


def _artifact_file_binding(
    root: Path, *, artifact: Dict[str, str], metadata: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "source": artifact,
        "artifact_metadata": dict(metadata),
        "file_manifest": common.directory_file_manifest(str(root)),
    }


def _policy_safetensor_index(root: Path) -> Dict[str, Dict[str, Any]]:
    """Index every policy tensor without relying on HF shard boundaries."""
    from safetensors import safe_open

    files = sorted(root.glob("*.safetensors"))
    nested = sorted(root.rglob("*.safetensors"))
    if nested != files:
        raise ValueError("DAPO policy safetensors must be regular top-level files")
    unsafe_weight_suffixes = (
        ".bin", ".pt", ".pth", ".ckpt", ".onnx", ".h5", ".msgpack",
        ".npz", ".gguf",
    )
    non_safetensor = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.name != "optimizer_state.pt"
        and path.name.lower().endswith(unsafe_weight_suffixes))
    if non_safetensor:
        raise ValueError("DAPO policy contains non-safetensor weight files")
    if not files:
        raise ValueError("DAPO policy contains no safetensor weights")
    tensors: Dict[str, Dict[str, Any]] = {}
    for path in files:
        if path.is_symlink() or not path.is_file():
            raise ValueError("DAPO policy contains a non-regular safetensor shard")
        try:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                keys = list(handle.keys())
                if not keys:
                    raise ValueError("DAPO policy contains an empty safetensor shard")
                for name in keys:
                    if not name or name in tensors:
                        raise ValueError(
                            f"DAPO policy contains duplicate/empty tensor name {name!r}")
                    value = handle.get_slice(name)
                    tensors[name] = {
                        "path": path,
                        "relative_path": path.relative_to(root).as_posix(),
                        "shape": [int(item) for item in value.get_shape()],
                        "dtype": str(value.get_dtype()),
                    }
        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"cannot index DAPO safetensor shard {path.name}") from exc
    index_paths = sorted(root.glob("*.safetensors.index.json"))
    if len(index_paths) > 1 or (len(files) > 1 and len(index_paths) != 1):
        raise ValueError("DAPO sharded policy lacks one canonical safetensor index")
    if index_paths:
        index = _load_json(index_paths[0])
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or set(weight_map) != set(tensors):
            raise ValueError("DAPO safetensor index tensor membership mismatch")
        for name, relative in weight_map.items():
            if not isinstance(relative, str) or Path(relative).name != relative or \
                    tensors[name]["relative_path"] != relative:
                raise ValueError("DAPO safetensor index shard mapping mismatch")
    return tensors


def _validate_fp32_to_bf16_policy_cast(
    *, checkpoint_root: Path, final_root: Path,
) -> Dict[str, Any]:
    """Prove final tensors equal exact BF16 casts of step-100 FP32 tensors.

    Safetensor slices are compared in bounded row chunks so validation does not
    materialize both full 1.7B-parameter state dictionaries at once.  Tensor
    names, shapes, and dtypes—not shard filenames—define the correspondence.
    """
    import torch
    from safetensors import safe_open

    checkpoint = _policy_safetensor_index(checkpoint_root)
    final = _policy_safetensor_index(final_root)
    if set(checkpoint) != set(final):
        missing = sorted(set(checkpoint) - set(final))
        extra = sorted(set(final) - set(checkpoint))
        raise ValueError(
            "DAPO final/checkpoint tensor-name mismatch: "
            f"missing={missing[:5]} extra={extra[:5]}")

    def non_weight_files(root: Path) -> Dict[str, str]:
        values = {}
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            if relative in {"trainer_state.json", "optimizer_state.pt"} or \
                    relative.endswith(".safetensors") or \
                    relative.endswith(".safetensors.index.json"):
                continue
            if path.is_symlink():
                raise ValueError("DAPO policy contains a symlinked non-weight file")
            values[relative] = common.file_sha256(str(path))
        return values

    checkpoint_non_weight = non_weight_files(checkpoint_root)
    final_non_weight = non_weight_files(final_root)
    if checkpoint_non_weight != final_non_weight:
        raise ValueError(
            "DAPO final/checkpoint non-weight config/tokenizer bytes differ")
    schema = []
    target_chunk_bytes = 64 * 1024 * 1024
    floating_codes = frozenset({
        "F64", "F32", "F16", "BF16", "F8_E4M3", "F8_E5M2",
    })
    nonfloating_sizes = {
        "BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2,
        "I32": 4, "U32": 4, "I64": 8, "U64": 8,
    }
    for name in sorted(checkpoint):
        source_info = checkpoint[name]
        final_info = final[name]
        if source_info["shape"] != final_info["shape"]:
            raise ValueError(f"DAPO tensor shape mismatch for {name!r}")
        source_dtype = source_info["dtype"]
        final_dtype = final_info["dtype"]
        floating = source_dtype in floating_codes
        if floating and (source_dtype != "F32" or final_dtype != "BF16"):
            raise ValueError(
                f"DAPO tensor dtype contract mismatch for {name!r}: "
                f"checkpoint={source_dtype}, final={final_dtype}")
        if not floating and final_dtype != source_dtype:
            raise ValueError(
                f"DAPO non-floating tensor dtype mismatch for {name!r}")
        shape = source_info["shape"]
        if any(size <= 0 for size in shape):
            raise ValueError(f"DAPO tensor has an empty/invalid shape: {name!r}")
        if not floating and source_dtype not in nonfloating_sizes:
            raise ValueError(
                f"DAPO tensor has unsupported dtype {source_dtype!r}: {name!r}")
        trailing_values = math.prod(shape[1:]) if shape else 1
        source_bytes = 4 if floating else nonfloating_sizes[source_dtype]
        rows_per_chunk = max(
            1, target_chunk_bytes // max(1, trailing_values * source_bytes))
        with safe_open(
                str(source_info["path"]), framework="pt", device="cpu") as source_file, \
                safe_open(
                    str(final_info["path"]), framework="pt",
                    device="cpu") as final_file:
            source_slice = source_file.get_slice(name)
            final_slice = final_file.get_slice(name)
            ranges = [(0, 1)] if not shape else [
                (start, min(shape[0], start + rows_per_chunk))
                for start in range(0, shape[0], rows_per_chunk)
            ]
            for start, stop in ranges:
                if shape:
                    source_tensor = source_slice[start:stop]
                    final_tensor = final_slice[start:stop]
                else:
                    source_tensor = source_file.get_tensor(name)
                    final_tensor = final_file.get_tensor(name)
                if floating:
                    if not bool(torch.isfinite(source_tensor).all()) or \
                            not bool(torch.isfinite(final_tensor).all()):
                        raise ValueError(
                            f"DAPO floating tensor contains non-finite values: {name!r}")
                    expected = source_tensor.to(torch.bfloat16)
                    equal = torch.equal(
                        expected.contiguous().view(torch.uint16),
                        final_tensor.contiguous().view(torch.uint16))
                else:
                    equal = torch.equal(source_tensor, final_tensor)
                if not equal:
                    raise ValueError(
                        f"DAPO final tensor differs from exact checkpoint cast: {name!r}")
        schema.append({
            "name": name,
            "shape": shape,
            "checkpoint_dtype": source_dtype,
            "final_dtype": final_dtype,
        })
    return {
        "version": "dapo_fp32_to_bf16_tensor_cast_v1",
        "relation": "final_tensor == checkpoint_tensor.to(torch.bfloat16)",
        "tensor_count": len(schema),
        "tensor_schema_sha256": common.canonical_json_sha256(schema),
        "non_weight_file_count": len(checkpoint_non_weight),
        "non_weight_files_sha256": common.canonical_json_sha256(
            checkpoint_non_weight),
        "max_source_chunk_bytes": target_chunk_bytes,
    }


def _validate_step_100_checkpoint(
    *, checkpoint_root: Path, checkpoint_artifact: Dict[str, str],
    checkpoint_metadata: Dict[str, Any], expected_runtime: Dict[str, Any],
    expected_code_binding: Dict[str, Any], final_root: Path,
) -> Dict[str, Any]:
    state = _load_json(checkpoint_root / "trainer_state.json")
    if _require_runtime(state) != EXPECTED_STEPS:
        raise ValueError("DAPO checkpoint artifact is not exact step 100")
    if state.get("runtime_contract") != expected_runtime or \
            state.get("code_artifact_binding") != expected_code_binding:
        raise ValueError("DAPO checkpoint runtime/code differs from completion")
    manifest = state.get("checkpoint_file_manifest")
    if not isinstance(manifest, dict):
        raise ValueError("DAPO checkpoint lacks its file manifest")
    common.verify_directory_file_manifest(
        str(checkpoint_root), manifest,
        exclude_relative_paths=("trainer_state.json",))
    expected_metadata = {
        "step": EXPECTED_STEPS,
        "round": 1,
        "phase": "central",
        "model_revision": common.CANONICAL_TRAINING_MODEL_REVISION,
        "runtime_contract_sha256": common.canonical_json_sha256(expected_runtime),
        "training_dataset_identity_sha256": expected_runtime[
            "training_dataset_identity"]["identity_sha256"],
    }
    if checkpoint_metadata != expected_metadata:
        raise ValueError("DAPO checkpoint artifact metadata is not the exact contract")
    cast_evidence = _validate_fp32_to_bf16_policy_cast(
        checkpoint_root=checkpoint_root, final_root=final_root)
    return {
        "source": checkpoint_artifact,
        "artifact_metadata": dict(checkpoint_metadata),
        "file_manifest": common.directory_file_manifest(str(checkpoint_root)),
        "fp32_to_bf16_tensor_cast": cast_evidence,
        "trainer_state": state,
    }


def build_completion_binding(
    *, model_root: str, checkpoint_root: str, results_root: str,
    model_artifact_ref: str, model_artifact_digest: str,
    model_artifact_type: str, model_artifact_metadata: Dict[str, Any],
    checkpoint_artifact_ref: str, checkpoint_artifact_digest: str,
    checkpoint_artifact_type: str,
    checkpoint_artifact_metadata: Dict[str, Any],
    results_artifact_ref: str, results_artifact_digest: str,
    results_artifact_type: str, results_artifact_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Rebuild and sign the exact successful DAPO-100 completion package.

    Both roots must be downloads of the committed W&B artifacts.  This avoids
    signing mutable training-pod files which differ from the durable objects
    consumed later by publication evaluation.
    """
    model_path = Path(model_root).resolve()
    checkpoint_artifact_path = Path(checkpoint_root).resolve()
    results_path = Path(results_root).resolve()
    for label, root in (
        ("model", model_path), ("checkpoint", checkpoint_artifact_path),
        ("results", results_path),
    ):
        if root.is_symlink() or not root.is_dir():
            raise ValueError(f"DAPO {label} artifact root is not a regular directory")
    model_artifact = _artifact_identity(
        ref=model_artifact_ref, digest=model_artifact_digest,
        artifact_type=model_artifact_type, expected_type="model",
        field="DAPO final model artifact")
    results_artifact = _artifact_identity(
        ref=results_artifact_ref, digest=results_artifact_digest,
        artifact_type=results_artifact_type,
        expected_type=DAPO_RESULTS_ARTIFACT_TYPE,
        field="DAPO results artifact")
    checkpoint_artifact = _artifact_identity(
        ref=checkpoint_artifact_ref, digest=checkpoint_artifact_digest,
        artifact_type=checkpoint_artifact_type,
        expected_type=DAPO_CHECKPOINT_ARTIFACT_TYPE,
        field="DAPO step-100 checkpoint artifact")
    expected_results_metadata = {
        "exit_status": 0,
        "completion_status": "complete",
        "method": DAPO_METHOD,
        "scientific_variant": SCIENTIFIC_VARIANT,
        "code_artifact_ref": str(results_artifact_metadata.get(
            "code_artifact_ref", "")),
        "code_artifact_digest": str(results_artifact_metadata.get(
            "code_artifact_digest", "")),
        "resume_checkpoint_artifact": str(results_artifact_metadata.get(
            "resume_checkpoint_artifact", "")),
        "resume_results_artifact": str(results_artifact_metadata.get(
            "resume_results_artifact", "")),
    }
    if results_artifact_metadata != expected_results_metadata:
        raise ValueError("DAPO results artifact metadata is not completion-exact")
    if expected_results_metadata["resume_checkpoint_artifact"] or \
            expected_results_metadata["resume_results_artifact"]:
        raise ValueError("fresh DAPO completion binding forbids resumed lineage")

    code_path = results_path / "provenance" / "dapo_code_binding.json"
    code_binding = _validate_stable_code_binding(_load_json(code_path))
    if code_binding["source"] != {
            "artifact_ref": expected_results_metadata["code_artifact_ref"],
            "artifact_digest": expected_results_metadata["code_artifact_digest"]}:
        raise ValueError("DAPO results metadata/code binding mismatch")

    checkpoint_path = results_path / f"step_{EXPECTED_STEPS:06d}"
    checkpoint_state = _load_json(checkpoint_path / "trainer_state.json")
    step = _require_runtime(checkpoint_state)
    if step != EXPECTED_STEPS:
        raise ValueError("DAPO completion must contain exactly 100 learner steps")
    runtime = checkpoint_state["runtime_contract"]
    if checkpoint_state.get("code_artifact_binding") != code_binding:
        raise ValueError("DAPO step-100 checkpoint code binding mismatch")
    recorded_manifest = checkpoint_state.get("checkpoint_file_manifest")
    if not isinstance(recorded_manifest, dict):
        raise ValueError("DAPO step-100 checkpoint lacks its byte manifest")

    final_binding_path = results_path / "final_artifact_binding.json"
    final_binding = _validate_final_model_binding(
        _load_json(final_binding_path), model_root=model_path,
        expected_model_artifact=model_artifact,
        expected_model_metadata=dict(model_artifact_metadata),
        expected_runtime=runtime, expected_code_binding=code_binding)
    checkpoint_binding = _validate_step_100_checkpoint(
        checkpoint_root=checkpoint_artifact_path,
        checkpoint_artifact=checkpoint_artifact,
        checkpoint_metadata=dict(checkpoint_artifact_metadata),
        expected_runtime=runtime, expected_code_binding=code_binding,
        final_root=model_path)
    if checkpoint_state != checkpoint_binding["trainer_state"]:
        raise ValueError(
            "DAPO results evidence differs from the committed step-100 checkpoint")
    result_final_state = _load_json(results_path / "final" / "trainer_state.json")
    model_final_state = _load_json(model_path / "trainer_state.json")
    if result_final_state != model_final_state:
        raise ValueError("DAPO results/model artifacts contain different final states")
    if result_final_state.get("runtime_contract") != runtime or \
            result_final_state.get("code_artifact_binding") != code_binding:
        raise ValueError("DAPO final and step-100 training contracts differ")

    history = _require_complete_history(results_path, EXPECTED_STEPS)
    dataset_identity = runtime.get("training_dataset_identity")
    from tmx_jax.data import training_dataset_identity

    expected_dataset_identity = training_dataset_identity("dapo_math_17k")
    if dataset_identity != expected_dataset_identity:
        raise ValueError("DAPO completion does not bind the pinned training dataset")
    if runtime.get("base_model_revision") != \
            common.CANONICAL_TRAINING_MODEL_REVISION:
        raise ValueError("DAPO completion does not bind the pinned Base revision")
    payload = {
        "method": DAPO_METHOD,
        "scientific_variant": SCIENTIFIC_VARIANT,
        "fresh_rerun_required": True,
        "base_model": {
            "model_id": MODEL,
            "revision": common.CANONICAL_TRAINING_MODEL_REVISION,
        },
        "training_dataset_identity": dataset_identity,
        "training_runtime_contract": runtime,
        "training_runtime_contract_sha256": common.canonical_json_sha256(runtime),
        "code_artifact_binding": code_binding,
        "model_artifact": _artifact_file_binding(
            model_path, artifact=model_artifact,
            metadata=dict(model_artifact_metadata)),
        "checkpoint_artifact": {
            key: value for key, value in checkpoint_binding.items()
            if key != "trainer_state"
        },
        "results_artifact": _artifact_file_binding(
            results_path, artifact=results_artifact,
            metadata=dict(results_artifact_metadata)),
        "final_model_artifact_binding": final_binding,
        "final_model_artifact_binding_sha256": final_binding["binding_sha256"],
        "step_100_trainer_state_sha256": common.file_sha256(
            str(checkpoint_path / "trainer_state.json")),
        "final_trainer_state_sha256": common.file_sha256(
            str(model_path / "trainer_state.json")),
        "step_100_checkpoint_file_manifest_sha256": common.canonical_json_sha256(
            recorded_manifest),
        "history": history,
    }
    return _signed_payload(DAPO_COMPLETION_BINDING_VERSION, payload)


def validate_completion_binding(value: Any) -> Dict[str, Any]:
    binding = _validate_signed_payload(
        value, expected_version=DAPO_COMPLETION_BINDING_VERSION)
    exact = {
        "method": DAPO_METHOD,
        "scientific_variant": SCIENTIFIC_VARIANT,
        "fresh_rerun_required": True,
    }
    for key, expected in exact.items():
        if binding.get(key) != expected:
            raise ValueError(f"DAPO completion binding has wrong {key}")
    if binding.get("base_model") != {
            "model_id": MODEL,
            "revision": common.CANONICAL_TRAINING_MODEL_REVISION}:
        raise ValueError("DAPO completion binding has the wrong Base identity")
    from tmx_jax.data import training_dataset_identity

    if binding.get("training_dataset_identity") != training_dataset_identity(
            "dapo_math_17k"):
        raise ValueError("DAPO completion binding has the wrong dataset identity")
    runtime = binding.get("training_runtime_contract")
    if not isinstance(runtime, dict) or binding.get(
            "training_runtime_contract_sha256") != common.canonical_json_sha256(runtime):
        raise ValueError("DAPO completion runtime contract SHA256 mismatch")
    from tmx_gpu import grpo_gpu

    grpo_gpu.require_canonical_training_execution_environment_contract(runtime)
    for name, expected_type in (
        ("model_artifact", "model"),
        ("checkpoint_artifact", DAPO_CHECKPOINT_ARTIFACT_TYPE),
        ("results_artifact", DAPO_RESULTS_ARTIFACT_TYPE),
    ):
        artifact = binding.get(name)
        if not isinstance(artifact, dict) or not isinstance(
                artifact.get("source"), dict):
            raise ValueError(f"DAPO completion binding lacks {name}")
        _artifact_identity(
            ref=str(artifact["source"].get("artifact_ref", "")),
            digest=str(artifact["source"].get("artifact_digest", "")),
            artifact_type=str(artifact["source"].get("artifact_type", "")),
            expected_type=expected_type, field=f"DAPO {name}")
        manifest = artifact.get("file_manifest")
        if not isinstance(manifest, dict) or manifest.get("version") != \
                common.DIRECTORY_FILE_MANIFEST_VERSION:
            raise ValueError(f"DAPO completion {name} lacks a byte manifest")
        if name == "checkpoint_artifact":
            cast = artifact.get("fp32_to_bf16_tensor_cast")
            if not isinstance(cast, dict) or cast.get("version") != \
                    "dapo_fp32_to_bf16_tensor_cast_v1" or \
                    cast.get("relation") != \
                    "final_tensor == checkpoint_tensor.to(torch.bfloat16)" or \
                    not isinstance(cast.get("tensor_count"), int) or \
                    cast.get("tensor_count", 0) < 1 or \
                    not isinstance(cast.get("tensor_schema_sha256"), str) or \
                    len(cast["tensor_schema_sha256"]) != 64 or \
                    not isinstance(cast.get("non_weight_file_count"), int) or \
                    cast.get("non_weight_file_count", -1) < 1 or \
                    not isinstance(cast.get("non_weight_files_sha256"), str) or \
                    len(cast["non_weight_files_sha256"]) != 64:
                raise ValueError(
                    "DAPO completion lacks exact FP32-to-BF16 tensor evidence")
    code = _validate_stable_code_binding(binding.get("code_artifact_binding"))
    if code["source"] != {
            key: binding["results_artifact"]["artifact_metadata"].get(
                f"code_artifact_{key.split('_')[-1]}")
            for key in ("artifact_ref", "artifact_digest")}:
        raise ValueError("DAPO completion results/code identities differ")
    for key in (
        "final_model_artifact_binding_sha256",
        "step_100_trainer_state_sha256",
        "final_trainer_state_sha256",
        "step_100_checkpoint_file_manifest_sha256",
    ):
        value = binding.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"DAPO completion binding lacks {key}")
    history = binding.get("history")
    if not isinstance(history, dict) or history.get("step_count") != EXPECTED_STEPS \
            or history.get("completed_metric_row_count") != EXPECTED_STEPS \
            or history.get("selected_trajectory_row_count") != EXPECTED_STEPS * 64:
        raise ValueError("DAPO completion history is not exactly 100 steps")
    return binding


def validate_completion_campaign(
    *, campaign_root: str, model_root: str, checkpoint_root: str,
    results_root: str,
    campaign_artifact_type: str, campaign_artifact_metadata: Dict[str, Any],
    model_artifact_ref: str, model_artifact_digest: str,
    model_artifact_type: str, model_artifact_metadata: Dict[str, Any],
    checkpoint_artifact_ref: str, checkpoint_artifact_digest: str,
    checkpoint_artifact_type: str,
    checkpoint_artifact_metadata: Dict[str, Any],
    results_artifact_ref: str, results_artifact_digest: str,
    results_artifact_type: str, results_artifact_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Replay a committed DAPO campaign from remote model/results bytes."""
    if campaign_artifact_type != DAPO_CAMPAIGN_ARTIFACT_TYPE:
        raise ValueError("DAPO campaign artifact has the wrong type")
    campaign_path = Path(campaign_root).resolve()
    if campaign_path.is_symlink() or not campaign_path.is_dir():
        raise ValueError("DAPO campaign artifact root is not a regular directory")
    binding_path = campaign_path / DAPO_COMPLETION_BINDING_FILENAME
    recorded = validate_completion_binding(_load_json(binding_path))
    rebuilt = build_completion_binding(
        model_root=model_root, checkpoint_root=checkpoint_root,
        results_root=results_root,
        model_artifact_ref=model_artifact_ref,
        model_artifact_digest=model_artifact_digest,
        model_artifact_type=model_artifact_type,
        model_artifact_metadata=model_artifact_metadata,
        checkpoint_artifact_ref=checkpoint_artifact_ref,
        checkpoint_artifact_digest=checkpoint_artifact_digest,
        checkpoint_artifact_type=checkpoint_artifact_type,
        checkpoint_artifact_metadata=checkpoint_artifact_metadata,
        results_artifact_ref=results_artifact_ref,
        results_artifact_digest=results_artifact_digest,
        results_artifact_type=results_artifact_type,
        results_artifact_metadata=results_artifact_metadata)
    if recorded != rebuilt:
        raise ValueError("DAPO campaign binding differs from replayed remote bytes")
    expected_campaign_metadata = {
        "completion_status": "complete",
        "method": DAPO_METHOD,
        "scientific_variant": SCIENTIFIC_VARIANT,
        "completion_binding_sha256": recorded["binding_sha256"],
        "model_artifact_ref": model_artifact_ref,
        "model_artifact_digest": model_artifact_digest,
        "checkpoint_artifact_ref": checkpoint_artifact_ref,
        "checkpoint_artifact_digest": checkpoint_artifact_digest,
        "results_artifact_ref": results_artifact_ref,
        "results_artifact_digest": results_artifact_digest,
        "code_artifact_ref": recorded["code_artifact_binding"]["source"][
            "artifact_ref"],
        "code_artifact_digest": recorded["code_artifact_binding"]["source"][
            "artifact_digest"],
    }
    if campaign_artifact_metadata != expected_campaign_metadata:
        raise ValueError("DAPO campaign artifact metadata differs from its binding")
    return recorded


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-binding", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--results-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pair-record", required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    record = validate_and_restore_resume_pair(
        code_binding_path=args.code_binding,
        checkpoint_root=args.checkpoint_root,
        checkpoint_manifest_path=args.checkpoint_manifest,
        results_root=args.results_root,
        results_manifest_path=args.results_manifest,
        output_dir=args.output_dir,
    )
    common.atomic_write_json(args.pair_record, record)
    print(f"[dapo] exact resume pair validated at step {record['checkpoint_step']}",
          flush=True)


if __name__ == "__main__":
    main()
