from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from .contract import validate_dapo_drgrpo_contract
from .safety import SafetyResult, is_trc_project, validate_network_policy


@dataclass(frozen=True)
class NetworkSpec:
    # Public specs carry no GCP project id or TPU name. Empty values are
    # filled from TPU_PROJECT / TPU_ZONE / TPU_NAME when a spec is loaded
    # (see load_experiment_spec); launch commands refuse to run without them.
    project: str = ""
    zone: str = "us-central1-a"
    tpu_name: str = ""
    accelerator_type: str = "v5litepod-256"
    external_ips_required: bool = True
    create_tpu: bool = False
    allow_nat: bool = False
    allow_internal_ips: bool = False


@dataclass(frozen=True)
class TopologySpec:
    total_workers: int = 64
    chips_per_worker: int = 4
    train_workers: tuple[int, ...] = tuple(range(16))
    rollout_workers: tuple[int, ...] = tuple(range(16, 64))
    train_chips: tuple[int, ...] = (0, 1, 2, 3)
    rollout_chips: tuple[int, ...] = (0, 1, 2, 3)

    @property
    def train_chip_count(self) -> int:
        return len(self.train_workers) * len(self.train_chips)

    @property
    def rollout_chip_count(self) -> int:
        return len(self.rollout_workers) * len(self.rollout_chips)


@dataclass(frozen=True)
class StorageSpec:
    hf_model_repo: str = "SaifPunjwani/two-model-exploration-models"
    hf_dataset_repo: str = "SaifPunjwani/two-model-exploration-checkpoints"
    hf_repo_type_models: str = "model"
    hf_repo_type_data: str = "dataset"
    artifact_prefix: str = ""
    allow_gcs: bool = False
    checkpoint_every_steps: int = 25
    rollout_upload_policy: str = "sharded"
    process_zero_uploads_only: bool = True


@dataclass(frozen=True)
class LoggingSpec:
    wandb_project: str = "two-model-explore"
    wandb_entity: str = os.environ.get("WANDB_ENTITY", "")  # empty = the W&B default entity
    minimal_wandb: bool = True
    process_zero_only: bool = True


@dataclass(frozen=True)
class ResumeSpec:
    fresh_run: bool = True
    source_run: str = ""
    source_step: str = "latest"
    init_checkpoint: str = ""
    existing_trajectories: str = ""


@dataclass(frozen=True)
class AlgorithmSpec:
    phase: str = "explorer_novelty"
    model_name: str = "Qwen/Qwen3-1.7B"
    dataset_name: str = "dapo_math_17k"
    max_prompt_len: int = 2048
    max_completion_len: int = 16384
    max_total_len: int = 18432
    grpo_batch_size: int = 1
    grpo_grad_accum: int = 1
    grpo_num_generations: int = 16
    grpo_max_steps: int = 100
    grpo_lr: float = 5e-6
    grpo_clip_epsilon: float = 0.2
    grpo_clip_epsilon_high: float = 0.28
    grpo_kl_beta: float = 0.0
    grpo_advantage_normalization: str = "none"
    dynamic_sampling: bool = True
    dynamic_sampling_max_attempts: int = 8
    dynamic_sampling_groups_per_attempt: int = 1
    dynamic_sampling_accept_reward_variance: bool = True
    reward_correct: float = 1.0
    reward_incorrect: float = -1.0
    format_penalty_weight: float = 0.0
    apply_format_penalty: bool = False
    lambda_novelty: float = 0.75
    novelty_layers: str = "7,14,21"
    novelty_metric: str = "sqrt_mse"
    incorrect_novelty_scale: float = 1.0
    soft_overlong_expected_len: int = 13107
    soft_overlong_cache_len: int = 3277
    distill_epochs: int = 2
    distill_batch_size: int = 1
    max_distill_examples: int = 20000
    min_accepted_trajectories: int = 500
    max_accepted_trajectories: int = 500
    accepted_selection_policy: str = "quality_first"
    actual_online_grpo_max_steps: int = 250
    actual_grpo_lr: float = 1e-6
    actual_grpo_kl_coef: float = 0.0
    benchmark_num_rollouts: int = 32
    benchmark_max_completion_len: int = 32768


@dataclass(frozen=True)
class BackendSpec:
    preferred: str = "xpk-gke-maxtext-tunix"
    fallback: str = "raw-tpu-tmx-jax"
    engine: str = "maxtext_tunix"
    container_image: str = ""
    xpk_cluster_name: str = "tmx-trc2-v5lite256"
    xpk_workload_name: str = ""
    xpk_tpu_type: str = "v5litepod-256"
    xpk_num_slices: int = 1
    xpk_priority: str = "high"
    xpk_use_pathways: bool = True
    maxtext_root: str = "/workspace/MaxText"
    tunix_root: str = "/workspace/tunix"
    maxtext_model_name: str = ""
    maxtext_load_parameters_path: str = ""
    maxtext_base_output_directory: str = "/tmp/tmx-output"
    maxtext_entrypoint: str = "maxtext.trainers.post_train.rl.train_rl"
    maxtext_loss_algo: str = "grpo"
    reward_plugin: str = "tmx.orchestration.reward_plugins"
    rollout_engine: str = "vllm_tpu"
    experimental_adapter_ack: bool = False


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    network: NetworkSpec = field(default_factory=NetworkSpec)
    topology: TopologySpec = field(default_factory=TopologySpec)
    storage: StorageSpec = field(default_factory=StorageSpec)
    logging: LoggingSpec = field(default_factory=LoggingSpec)
    resume: ResumeSpec = field(default_factory=ResumeSpec)
    algorithm: AlgorithmSpec = field(default_factory=AlgorithmSpec)
    backend: BackendSpec = field(default_factory=BackendSpec)

    def validate(self) -> SafetyResult:
        errors: list[str] = []
        warnings: list[str] = []
        net = validate_network_policy(
            self.network.project,
            self.network.accelerator_type,
            self.network.external_ips_required,
        )
        errors.extend(net.errors)
        warnings.extend(net.warnings)
        if self.network.allow_nat:
            errors.append("NAT/router creation is forbidden for this pipeline")
        if self.network.allow_internal_ips:
            errors.append("internal-IP TPU access is forbidden for TRC runs")
        if self.network.create_tpu and is_trc_project(self.network.project):
            errors.append("TRC TPUs should be created by tpunanny, not this pipeline")
        if self.storage.allow_gcs:
            errors.append("GCS artifact storage is disabled; Hugging Face is canonical storage")
        if self.backend.xpk_num_slices < 1:
            errors.append("backend.xpk_num_slices must be >= 1")
        if self.backend.xpk_tpu_type and self.backend.xpk_tpu_type != self.network.accelerator_type:
            errors.append("backend.xpk_tpu_type must match network.accelerator_type")
        if self.backend.rollout_engine not in {"vllm_tpu", "vllm", "sglang_jax"}:
            errors.append("backend.rollout_engine must be one of: vllm_tpu, vllm, sglang_jax")
        phase = str(self.algorithm.phase)
        if not self.resume.fresh_run and not (
            self.resume.init_checkpoint
            or self.resume.source_run
            or self.resume.existing_trajectories
        ):
            errors.append("resume.fresh_run=false requires resume.init_checkpoint, resume.source_run, or resume.existing_trajectories")
        if self.resume.fresh_run and (self.resume.init_checkpoint or self.resume.source_run):
            warnings.append("resume checkpoint fields are ignored because resume.fresh_run=true")
        if phase in {"actual_sft", "actual_grpo"} and not (
            self.resume.existing_trajectories or self.resume.source_run
        ):
            errors.append(f"{phase} requires resume.existing_trajectories or resume.source_run")
        if self.resume.existing_trajectories.startswith("gs://") and not self.storage.allow_gcs:
            errors.append("resume.existing_trajectories cannot be gs:// when storage.allow_gcs=false")
        if self.resume.existing_trajectories and not (
            self.resume.existing_trajectories.startswith("hf://")
            or self.resume.existing_trajectories.startswith("/")
            or self.resume.existing_trajectories.startswith(".")
        ):
            errors.append("resume.existing_trajectories must be hf://, absolute local path, or relative local path")
        valid_workers = set(range(self.topology.total_workers))
        train_workers = set(self.topology.train_workers)
        rollout_workers = set(self.topology.rollout_workers)
        invalid_train_workers = sorted(train_workers - valid_workers)
        invalid_rollout_workers = sorted(rollout_workers - valid_workers)
        if invalid_train_workers:
            errors.append(f"train_workers contains out-of-range workers: {invalid_train_workers}")
        if invalid_rollout_workers:
            errors.append(f"rollout_workers contains out-of-range workers: {invalid_rollout_workers}")
        overlap_workers = sorted(train_workers & rollout_workers)
        if overlap_workers:
            errors.append(f"train_workers and rollout_workers overlap: {overlap_workers}")
        valid_chips = set(range(self.topology.chips_per_worker))
        invalid_train_chips = sorted(set(self.topology.train_chips) - valid_chips)
        invalid_rollout_chips = sorted(set(self.topology.rollout_chips) - valid_chips)
        if invalid_train_chips:
            errors.append(f"train_chips contains out-of-range chips: {invalid_train_chips}")
        if invalid_rollout_chips:
            errors.append(f"rollout_chips contains out-of-range chips: {invalid_rollout_chips}")
        if self.topology.train_chip_count + self.topology.rollout_chip_count > self.topology.total_workers * self.topology.chips_per_worker:
            errors.append("topology over-allocates TPU chips")
        if self.algorithm.grpo_kl_beta != 0.0 and self.algorithm.phase.startswith("explorer"):
            warnings.append("Explorer DAPO contract normally uses KL=0")
        contract = validate_dapo_drgrpo_contract(self.algorithm)
        errors.extend(contract.errors)
        warnings.extend(contract.warnings)
        return SafetyResult(ok=not errors, errors=tuple(errors), warnings=tuple(warnings))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce_tuple_ints(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    return tuple(int(x) for x in value)


def _dataclass_from_mapping(cls, data: Mapping[str, Any]):
    if cls is TopologySpec:
        converted = dict(data)
        for key in ("train_workers", "rollout_workers", "train_chips", "rollout_chips"):
            if key in converted:
                converted[key] = _coerce_tuple_ints(converted[key])
        return cls(**converted)
    return cls(**dict(data))


def network_with_env_defaults(network: NetworkSpec) -> NetworkSpec:
    """Fill empty TPU coordinates from TPU_PROJECT / TPU_ZONE / TPU_NAME."""
    return replace(
        network,
        project=network.project or os.environ.get("TPU_PROJECT", ""),
        zone=network.zone or os.environ.get("TPU_ZONE", ""),
        tpu_name=network.tpu_name or os.environ.get("TPU_NAME", ""),
    )


def missing_tpu_coordinates(network: NetworkSpec) -> list[str]:
    """Names of the network fields that must be set before touching a TPU."""
    return [
        name
        for name, value in (
            ("network.project", network.project),
            ("network.zone", network.zone),
            ("network.tpu_name", network.tpu_name),
        )
        if not value
    ]


def load_experiment_spec(path: str | Path) -> ExperimentSpec:
    data = json.loads(Path(path).read_text())
    return ExperimentSpec(
        name=data["name"],
        network=network_with_env_defaults(_dataclass_from_mapping(NetworkSpec, data.get("network", {}))),
        topology=_dataclass_from_mapping(TopologySpec, data.get("topology", {})),
        storage=_dataclass_from_mapping(StorageSpec, data.get("storage", {})),
        logging=_dataclass_from_mapping(LoggingSpec, data.get("logging", {})),
        resume=_dataclass_from_mapping(ResumeSpec, data.get("resume", {})),
        algorithm=_dataclass_from_mapping(AlgorithmSpec, data.get("algorithm", {})),
        backend=_dataclass_from_mapping(BackendSpec, data.get("backend", {})),
    )
