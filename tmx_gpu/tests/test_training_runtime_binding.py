"""Immutable training-container and package-stack provenance boundaries."""

from __future__ import annotations

from pathlib import Path

import pytest

from tmx_gpu import grpo_gpu, sft_gpu
from tmx_gpu.tests import orchestrator_receipt_fixtures as receipt_fixtures


def _strict_c9_args(*extra: str):
    args = grpo_gpu.build_arg_parser().parse_args([
        "--output-dir", "/tmp/runtime-binding-c9",
        "--phase", "scout",
        "--lambda-novelty", "0.5",
        "--rl-validity-mode", "c8",
        "--expdis-stabilization-mode", grpo_gpu.C9_RND_LIFECYCLE,
        "--novelty-normalization", "selected_batch_zscore",
        "--scientific-variant", grpo_gpu.C9_SCIENTIFIC_VARIANT,
        "--code-source-manifest", "/tmp/code.json",
        *extra,
    ])
    args._training_execution_environment = receipt_fixtures.environment(
        run_name=args.run_name)
    return args


def test_strict_grpo_contract_rejects_wrong_or_missing_image():
    for value in (None, "vllm/vllm-openai:v0.25.0", "wrong@sha256:deadbeef"):
        argv = [] if value is None else ["--container-image-identity", value]
        with pytest.raises(ValueError, match="container-image-identity exactly"):
            grpo_gpu.build_runtime_contract(
                _strict_c9_args(*argv), resolved_lr=1e-6,
                resolved_steps=200)


def test_strict_grpo_contract_binds_exact_image_and_full_stack():
    args = _strict_c9_args(
        "--container-image-identity",
        grpo_gpu.CANONICAL_TRAINING_CONTAINER_IMAGE)
    runtime = grpo_gpu.build_runtime_contract(
        args, resolved_lr=1e-6, resolved_steps=200)
    environment = runtime["execution_environment"]
    assert environment["version"] == \
        grpo_gpu.TRAINING_EXECUTION_ENVIRONMENT_VERSION
    assert environment["frozen_invariants"]["container_image_identity"] == \
        grpo_gpu.CANONICAL_TRAINING_CONTAINER_IMAGE
    assert environment["frozen_invariants"]["runtime_stack"] == \
        grpo_gpu.CANONICAL_TRAINING_RUNTIME_STACK
    assert environment["frozen_invariants"][
        "orchestrator_receipt_binding"]["source"]["artifact_ref"].endswith(":v1")


def test_live_preflight_rejects_one_package_version_drift(monkeypatch):
    drifted = dict(grpo_gpu.CANONICAL_TRAINING_RUNTIME_STACK)
    drifted["transformers"] = "5.13.1"
    monkeypatch.setattr(
        grpo_gpu, "observed_training_runtime_stack", lambda: drifted)
    with pytest.raises(RuntimeError, match="transformers.*5.13.1"):
        grpo_gpu.validate_canonical_training_execution_environment(
            grpo_gpu.CANONICAL_TRAINING_CONTAINER_IMAGE,
            receipt_fixtures.binding())


def test_generic_grpo_and_sft_remain_environment_agnostic(tmp_path):
    generic = grpo_gpu.build_arg_parser().parse_args([
        "--output-dir", str(tmp_path / "grpo")])
    assert "execution_environment" not in grpo_gpu.build_runtime_contract(
        generic, resolved_lr=5e-6, resolved_steps=200)

    accepted = tmp_path / "accepted.jsonl"
    accepted.write_text("", encoding="utf-8")
    sargs = sft_gpu.build_arg_parser().parse_args([
        "--accepted", str(accepted), "--output-dir", str(tmp_path / "sft")])
    assert "execution_environment" not in sft_gpu.build_sft_runtime_contract(
        sargs, resolved_max_total_len=18432)


def test_strict_sft_contract_binds_same_environment(tmp_path):
    accepted = tmp_path / "accepted.jsonl"
    accepted.write_text("", encoding="utf-8")
    args = sft_gpu.build_arg_parser().parse_args([
        "--accepted", str(accepted), "--output-dir", str(tmp_path / "sft"),
        "--container-image-identity",
        grpo_gpu.CANONICAL_TRAINING_CONTAINER_IMAGE,
        "--require-canonical-training-environment",
    ])
    args._training_execution_environment = receipt_fixtures.environment(
        run_name=args.run_name)
    runtime = sft_gpu.build_sft_runtime_contract(
        args, resolved_max_total_len=18432)
    grpo_gpu.require_canonical_training_execution_environment_contract(runtime)


def test_launchers_pin_identity_and_bootstrap_versions():
    launch_dir = Path(__file__).resolve().parents[1] / "launch"
    for name in ("expdis_job.sh", "dapo_job.sh"):
        source = (launch_dir / name).read_text(encoding="utf-8")
        assert grpo_gpu.CANONICAL_TRAINING_CONTAINER_IMAGE in source
        assert '--container-image-identity "$TRAINING_CONTAINER_IMAGE_IDENTITY"' \
            in source
        assert "tmx_gpu.orchestrator_receipt resolve" in source
        assert "--orchestrator-receipt-binding" in source
        assert "ORCHESTRATOR_COMMAND_SHA256" in source
        assert "CODE_ARTIFACT_DIGEST" in source
        for package, version in {
            "wandb": "0.28.0",
            "datasets": "5.0.0",
            "requests": "2.34.2",
            "huggingface_hub": "1.21.0",
            "hf-transfer": "0.1.9",
            "numpy": "2.2.6",
            "safetensors": "0.8.0",
        }.items():
            assert f'{package}=={version}' in source
