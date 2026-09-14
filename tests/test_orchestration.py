from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataclasses import replace

from tmx.orchestration.backends import build_submission_plan
from tmx.orchestration.contract import validate_dapo_drgrpo_contract
from tmx.orchestration.safety import assert_trc_external_ip_policy
from tmx.orchestration.spec import (
    BackendSpec,
    ExperimentSpec,
    ResumeSpec,
    StorageSpec,
    TopologySpec,
    load_experiment_spec,
)
from tmx.orchestration.stages import stage_preset
from tmx.orchestration.topologies import network_preset, topology_preset


def test_trc_external_ip_guard_rejects_internal_ip() -> None:
    with pytest.raises(ValueError):
        assert_trc_external_ip_policy(["gcloud", "tpu-vm", "ssh", "--internal-ips"])


def test_production_novelty_spec_validates() -> None:
    spec = load_experiment_spec("configs/production_trc2_v5lite256_explorer_novelty075.json")
    result = spec.validate()
    assert result.ok, result.errors


def test_raw_plan_is_hf_only_and_external_ip_safe() -> None:
    spec = load_experiment_spec("configs/production_trc2_v5lite256_explorer_novelty075.json")
    plan = build_submission_plan(spec, backend="raw-tpu-tmx-jax")
    encoded = json.dumps(plan.env)
    assert plan.executable
    assert plan.env["TMX_PIPELINE_STAGE"] == "explorer_novelty"
    assert plan.env["TMX_PIPELINE_STOP_AFTER_EXPLORER"] == "1"
    assert "TMX_ALLOW_GCS_ARTIFACTS" in plan.env
    assert plan.env["TMX_ALLOW_GCS_ARTIFACTS"] == "0"
    assert plan.env["INCORRECT_NOVELTY_SCALE"] == "1.0"
    assert plan.env["REWARD_CORRECT"] == "1.0"
    assert plan.env["REWARD_INCORRECT"] == "-1.0"
    assert plan.env["FORMAT_PENALTY_WEIGHT"] == "0.0"
    assert plan.env["TMX_APPLY_FORMAT_PENALTY"] == "0"
    assert plan.env["TMX_REQUIRE_HF_ARTIFACT_UPLOAD"] == "1"
    assert "--internal-ips" not in encoded


def test_raw_plan_blocks_actual_without_hf_trajectories() -> None:
    spec = ExperimentSpec(
        name="actual_grpo_smoke",
        algorithm=stage_preset("actual_grpo_16k"),
    )
    plan = build_submission_plan(spec, backend="raw-tpu-tmx-jax")
    assert not plan.executable
    assert any("requires HF Explorer trajectories" in blocker for blocker in plan.blockers)


def test_raw_plan_actual_grpo_uses_hf_trajectories_without_explorer_rerun() -> None:
    spec = load_experiment_spec("configs/production_trc2_v5lite256_actual_grpo.json")
    plan = build_submission_plan(spec, backend="raw-tpu-tmx-jax")
    assert plan.executable, plan.blockers
    assert plan.env["TMX_PIPELINE_STAGE"] == "actual_grpo"
    assert plan.env["TMX_PIPELINE_SKIP_EXPLORER"] == "1"
    assert plan.env["TMX_PIPELINE_STOP_AFTER_SFT"] == "0"
    assert plan.env["TMX_EXISTING_TRAJECTORIES"].startswith("hf://SaifPunjwani/")
    assert plan.env["SOURCE_RUN"] == ""
    assert plan.env["INIT_CHECKPOINT"] == ""
    assert plan.env["TMX_ALLOW_GCS_ARTIFACTS"] == "0"


def test_raw_plan_actual_sft_stops_before_actual_grpo() -> None:
    spec = load_experiment_spec("configs/production_trc2_v5lite256_actual_sft.json")
    plan = build_submission_plan(spec, backend="raw-tpu-tmx-jax")
    assert plan.executable, plan.blockers
    assert plan.env["TMX_PIPELINE_STAGE"] == "actual_sft"
    assert plan.env["TMX_PIPELINE_SKIP_EXPLORER"] == "1"
    assert plan.env["TMX_PIPELINE_STOP_AFTER_SFT"] == "1"
    assert plan.env["TMX_EXISTING_TRAJECTORIES"].startswith("hf://SaifPunjwani/")


def test_raw_launcher_exports_phase_controls_to_tmx_jax() -> None:
    restart_script = Path("scripts/restart_dapo_drgrpo_twoslice.sh").read_text()
    multihost_script = Path("tmx_jax/scripts/launch_jax_pipeline_multihost_v5lite.sh").read_text()
    pipeline_script = Path("tmx_jax/scripts/launch_jax_pipeline.sh").read_text()
    pipeline_py = Path("tmx_jax/pipeline.py").read_text()
    assert "TMX_PIPELINE_STAGE" in restart_script
    assert "TMX_PIPELINE_STOP_AFTER_EXPLORER" in restart_script
    assert "TMX_PIPELINE_STOP_AFTER_SFT" in restart_script
    assert "TMX_PIPELINE_STOP_AFTER_EXPLORER" in multihost_script
    assert "TMX_PIPELINE_STOP_AFTER_SFT" in multihost_script
    assert "TMX_PIPELINE_STOP_AFTER_EXPLORER" in pipeline_script
    assert "TMX_PIPELINE_STOP_AFTER_SFT" in pipeline_script
    assert "stopping after Explorer stage" in pipeline_py
    assert "stopping after SFT stage" in pipeline_py


def test_xpk_plan_is_blocked_without_container_image() -> None:
    spec = load_experiment_spec(Path("configs/production_trc2_v5lite256_explorer_novelty075.json"))
    plan = build_submission_plan(spec, backend="xpk-gke-maxtext-tunix")
    assert not plan.executable
    assert any("container_image" in blocker for blocker in plan.blockers)
    assert "NOVELTY_LAYERS=7,14,21" in plan.commands[0]
    assert "create-pathways" in plan.commands[0]
    assert "--internal-ips" not in plan.commands[0]
    assert "compute routers" not in plan.commands[0]


def test_tunix_direct_plan_runs_existing_tpu_hf_only_all_workers() -> None:
    spec = load_experiment_spec("configs/tunix/tmx_dapo_drgrpo_resume_step100.json")
    plan = build_submission_plan(spec)
    assert plan.executable, plan.blockers
    assert plan.backend == "tunix-cli-direct"
    assert plan.env["TPU_PROJECT"] == spec.network.project
    assert plan.env["TPU_ZONE"] == "europe-west4-b"
    assert plan.env["TPU_NAME"] == "trc2-v5lite256-ew4b-r1"
    assert plan.env["TPU_WORKER"] == "all"
    assert plan.env["TMX_ALLOW_GCS_ARTIFACTS"] == "0"
    assert plan.env["GRPO_NUM_GENERATIONS"] == "16"
    assert plan.env["MAX_COMPLETION_LEN"] == "16384"
    assert plan.env["LAMBDA_NOVELTY"] == "0.5"
    command = plan.commands[0]
    assert "infra/xpk/run_paper_pipeline.sh" in command
    assert "TMX_EXPERIMENT_SPEC_JSON_B64=" in command
    assert "--internal-ips" not in command


def test_xpk_plan_can_be_unblocked_after_container_smoke() -> None:
    spec = load_experiment_spec(Path("configs/production_trc2_v5lite256_explorer_novelty075.json"))
    spec = replace(
        spec,
        backend=replace(
            spec.backend,
            container_image="europe-west4-docker.pkg.dev/example-project/tmx/maxtext-tunix:smoke",
            maxtext_model_name="qwen3-1.7b",
            maxtext_load_parameters_path="hf://SaifPunjwani/two-model-exploration-models/base/qwen3-1.7b",
            experimental_adapter_ack=True,
        ),
    )
    plan = build_submission_plan(spec, backend="xpk-gke-maxtext-tunix")
    assert plan.executable
    assert "xpk workload create-pathways" in plan.commands[0]
    assert "--tpu-type v5litepod-256" in plan.commands[0]
    assert "--num-slices 1" in plan.commands[0]
    assert "TMX_ALLOW_GCS_ARTIFACTS=0" in plan.commands[0]
    assert "TMX_REWARD_PLUGIN=tmx.orchestration.reward_plugins" in plan.commands[0]
    assert "--internal" not in plan.commands[0]


def test_production_spec_rejects_gcs_artifacts() -> None:
    spec = ExperimentSpec(
        name="bad_gcs",
        storage=StorageSpec(allow_gcs=True),
    )
    result = spec.validate()
    assert not result.ok
    assert any("GCS artifact storage" in error for error in result.errors)


def test_production_spec_rejects_overlapping_workers() -> None:
    spec = ExperimentSpec(
        name="bad_topology",
        topology=TopologySpec(
            total_workers=64,
            chips_per_worker=4,
            train_workers=(0, 1),
            rollout_workers=(1, 2),
        ),
    )
    result = spec.validate()
    assert not result.ok
    assert any("overlap" in error for error in result.errors)


def test_resume_spec_requires_source_when_not_fresh() -> None:
    spec = ExperimentSpec(name="bad_resume", resume=ResumeSpec(fresh_run=False))
    result = spec.validate()
    assert not result.ok
    assert any("resume.fresh_run=false" in error for error in result.errors)


def test_presets_cover_trc256_novelty_contract() -> None:
    network = network_preset("trc2-v5litepod-256-ew4b")
    topology = topology_preset("v5litepod-256-64train-192rollout")
    stage = stage_preset("explorer_novelty075_16k")
    assert network.external_ips_required
    assert not network.allow_nat
    assert topology.train_chip_count == 64
    assert topology.rollout_chip_count == 192
    assert stage.lambda_novelty == 0.75
    assert stage.incorrect_novelty_scale == 1.0
    assert not validate_dapo_drgrpo_contract(stage).errors


def test_raw_plan_supports_all_known_tpu_target_classes() -> None:
    targets = (
        ("trc2-v5litepod-256-ew4b", "v5litepod-256-64train-192rollout", "v5litepod-256"),
        ("v5litepod-64-ew4b", "v5litepod-64-16train-48rollout", "v5litepod-64"),
        ("v5litepod-64-uc1a", "v5litepod-64-16train-48rollout", "v5litepod-64"),
        ("v6e-64-ew4a", "v6e-64-16train-48rollout", "v6e-64"),
        ("v6e-64-use1d", "v6e-64-16train-48rollout", "v6e-64"),
    )
    for network_name, topology_name, accelerator_type in targets:
        network = network_preset(network_name)
        topology = topology_preset(topology_name)
        spec = ExperimentSpec(
            name=f"smoke_{network_name}",
            network=network,
            topology=topology,
            algorithm=stage_preset("explorer_novelty075_16k"),
            backend=BackendSpec(xpk_tpu_type=network.accelerator_type),
        )
        result = spec.validate()
        assert result.ok, (network_name, result.errors)
        plan = build_submission_plan(spec, backend="raw-tpu-tmx-jax")
        assert plan.executable, (network_name, plan.blockers)
        assert plan.env["TPU_ACCELERATOR_TYPE"] == accelerator_type
        assert plan.env["TRC2_EXPECTED_ACCELERATOR"] == accelerator_type
        assert plan.env["TMX_ALLOW_GCS_ARTIFACTS"] == "0"
        assert "--internal-ips" not in " ".join(plan.commands)


def test_dapo_contract_rejects_missing_clip_higher() -> None:
    stage = replace(stage_preset("explorer_novelty075_16k"), grpo_clip_epsilon_high=0.2)
    result = validate_dapo_drgrpo_contract(stage)
    assert result.errors
    assert any("clip-high" in error or "clip-higher" in error for error in result.errors)


def test_dapo_contract_rejects_novelty_without_incorrect_reward() -> None:
    stage = replace(stage_preset("explorer_novelty075_16k"), incorrect_novelty_scale=0.0)
    result = validate_dapo_drgrpo_contract(stage)
    assert result.errors
    assert any("incorrect_novelty_scale" in error for error in result.errors)


def test_dapo_contract_rejects_nonnegative_incorrect_reward_for_novelty() -> None:
    stage = replace(stage_preset("explorer_novelty075_16k"), reward_incorrect=0.0)
    result = validate_dapo_drgrpo_contract(stage)
    assert result.errors
    assert any("negative reward_incorrect" in error for error in result.errors)


def test_dapo_contract_rejects_static_format_penalty() -> None:
    stage = replace(stage_preset("explorer_novelty075_16k"), format_penalty_weight=0.2)
    result = validate_dapo_drgrpo_contract(stage)
    assert result.errors
    assert any("static format penalty" in error for error in result.errors)


def test_trc256_direct_launch_defaults_match_contract() -> None:
    script = Path("scripts/launch_trc2_v5lite256_tmx_jax_resume.sh").read_text()
    assert 'LAMBDA_NOVELTY="${LAMBDA_NOVELTY:-0.75}"' in script
    assert 'TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-v5litepod-256}"' in script
    assert "enableExternalIps" not in script
    assert "--internal-ips" not in script


def test_tmx_jax_multihost_gcs_is_opt_in() -> None:
    train_py = Path("tmx_jax/train.py")
    text = train_py.read_text()
    assert '_env_flag("TMX_ALLOW_GCS_ARTIFACTS", "1")' not in text
    assert 'and _env_flag("TMX_ALLOW_GCS_ARTIFACTS", "0")' in text
    assert '_env_flag("TMX_ORBAX_DIRECT_GCS_CHECKPOINT", "0")' in text
