from __future__ import annotations

import base64
import json
import os
import shlex
from dataclasses import dataclass, field

from .spec import ExperimentSpec


@dataclass(frozen=True)
class SubmissionPlan:
    backend: str
    commands: tuple[str, ...]
    env: dict[str, str] = field(default_factory=dict)
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def executable(self) -> bool:
        return not self.blockers


def _quote_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(p)) for p in parts if str(p))


def _env_command(prefix_env: dict[str, str], command: str) -> str:
    assignments = " ".join(f"{key}={shlex.quote(str(value))}" for key, value in sorted(prefix_env.items()))
    return f"{assignments} {command}".strip()


def _common_env(spec: ExperimentSpec) -> dict[str, str]:
    alg = spec.algorithm
    storage = spec.storage
    resume = spec.resume
    return {
        "RUN_NAME": spec.name,
        "TMX_PIPELINE_STAGE": alg.phase,
        "MODEL_NAME": alg.model_name,
        "MAX_PROMPT_LEN": str(alg.max_prompt_len),
        "MAX_COMPLETION_LEN": str(alg.max_completion_len),
        "MAX_TOTAL_LEN": str(alg.max_total_len),
        "GRPO_BATCH_SIZE": str(alg.grpo_batch_size),
        "GRPO_GRAD_ACCUM": str(alg.grpo_grad_accum),
        "GRPO_NUM_GENERATIONS": str(alg.grpo_num_generations),
        "GRPO_MAX_STEPS": str(alg.grpo_max_steps),
        "GRPO_LR": str(alg.grpo_lr),
        "GRPO_CLIP_EPSILON": str(alg.grpo_clip_epsilon),
        "GRPO_CLIP_EPSILON_HIGH": str(alg.grpo_clip_epsilon_high),
        "GRPO_KL_BETA": str(alg.grpo_kl_beta),
        "GRPO_ADVANTAGE_NORMALIZATION": alg.grpo_advantage_normalization,
        "DYNAMIC_SAMPLING": str(alg.dynamic_sampling).lower(),
        "DYNAMIC_SAMPLING_MAX_ATTEMPTS": str(alg.dynamic_sampling_max_attempts),
        "DYNAMIC_SAMPLING_GROUPS_PER_ATTEMPT": str(alg.dynamic_sampling_groups_per_attempt),
        "TMX_DYNAMIC_SAMPLING_GROUPS_PER_ATTEMPT": str(alg.dynamic_sampling_groups_per_attempt),
        "TMX_DYNAMIC_SAMPLING_ACCEPT_REWARD_VARIANCE": "1" if alg.dynamic_sampling_accept_reward_variance else "0",
        "REWARD_CORRECT": str(alg.reward_correct),
        "REWARD_INCORRECT": str(alg.reward_incorrect),
        "FORMAT_PENALTY_WEIGHT": str(alg.format_penalty_weight),
        "TMX_APPLY_FORMAT_PENALTY": "1" if alg.apply_format_penalty else "0",
        "LAMBDA_NOVELTY": str(alg.lambda_novelty),
        "NOVELTY_LAYERS": alg.novelty_layers,
        "NOVELTY_METRIC": alg.novelty_metric,
        "INCORRECT_NOVELTY_SCALE": str(alg.incorrect_novelty_scale),
        "SOFT_OVERLONG_EXPECTED_LEN": str(alg.soft_overlong_expected_len),
        "SOFT_OVERLONG_CACHE_LEN": str(alg.soft_overlong_cache_len),
        "DISTILL_EPOCHS": str(alg.distill_epochs),
        "DISTILL_BATCH_SIZE": str(alg.distill_batch_size),
        "MAX_DISTILL_EXAMPLES": str(alg.max_distill_examples),
        "MIN_ACCEPTED_TRAJECTORIES": str(alg.min_accepted_trajectories),
        "MAX_ACCEPTED_TRAJECTORIES": str(alg.max_accepted_trajectories),
        "ACCEPTED_SELECTION_POLICY": alg.accepted_selection_policy,
        "ACTUAL_ONLINE_GRPO_MAX_STEPS": str(alg.actual_online_grpo_max_steps),
        "ACTUAL_GRPO_LR": str(alg.actual_grpo_lr),
        "ACTUAL_GRPO_KL_COEF": str(alg.actual_grpo_kl_coef),
        "BENCHMARK_NUM_ROLLOUTS": str(alg.benchmark_num_rollouts),
        "BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN": str(alg.benchmark_max_completion_len),
        "TMX_ARTIFACT_BACKEND": "hf",
        "TMX_ALLOW_GCS_ARTIFACTS": "1" if storage.allow_gcs else "0",
        "TMX_GCS_CHECKPOINT_BASE": "",
        "TMX_HF_CHECKPOINT_REPO": storage.hf_dataset_repo,
        "TMX_HF_MODEL_REPO": storage.hf_model_repo,
        "TMX_HF_DATASET_REPO": storage.hf_dataset_repo,
        "TMX_HF_REPO_TYPE": storage.hf_repo_type_data,
        "TMX_HF_ARTIFACT_PREFIX": storage.artifact_prefix or spec.name,
        "TMX_HF_CHECKPOINT_EVERY_STEPS": str(storage.checkpoint_every_steps),
        "TMX_REQUIRE_HF_ARTIFACT_UPLOAD": "1",
        # hf_transfer is optional and not guaranteed on fresh TPU worker envs.
        # Disable it by default so HF-only storage does not fail before
        # training starts because of a missing acceleration package.
        "HF_HUB_ENABLE_HF_TRANSFER": os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0"),
        "SAVE_EVERY_STEPS": str(storage.checkpoint_every_steps),
        "TMX_WANDB_MINIMAL": "1" if spec.logging.minimal_wandb else "0",
        "TMX_WANDB_SINGLE_RUN": "1",
        "TMX_FRESH_RUN": "1" if resume.fresh_run else "0",
        "SOURCE_RUN": resume.source_run,
        "SOURCE_STEP": resume.source_step,
        "INIT_CHECKPOINT": resume.init_checkpoint,
    }


def _hf_trajectory_uri_from_source_run(spec: ExperimentSpec) -> str:
    run = spec.resume.source_run.strip()
    if not run:
        return ""
    step = spec.resume.source_step.strip() or "latest"
    if step == "latest":
        filename = "explorer_trajectories_latest.jsonl"
    else:
        padded = step if step.startswith("step_") else f"rollouts_step_{int(step):06d}.jsonl"
        filename = padded if padded.endswith(".jsonl") else f"{padded}.jsonl"
    return f"hf://{spec.storage.hf_dataset_repo}/{run}/explorer/trajectories/{filename}"


def _actual_trajectory_uri(spec: ExperimentSpec) -> str:
    explicit = spec.resume.existing_trajectories.strip()
    if explicit:
        return explicit
    return _hf_trajectory_uri_from_source_run(spec)


def _workload_env(spec: ExperimentSpec, *, engine: str) -> dict[str, str]:
    backend = spec.backend
    env = _common_env(spec)
    spec_json = json.dumps(spec.to_dict(), sort_keys=True, separators=(",", ":"))
    env.update(
        {
            "TMX_BACKEND_ENGINE": engine,
            "TMX_EXPERIMENT_SPEC_JSON_B64": base64.b64encode(spec_json.encode("utf-8")).decode("ascii"),
            "TMX_MAXTEXT_ROOT": backend.maxtext_root,
            "TMX_TUNIX_ROOT": backend.tunix_root,
            "TMX_MAXTEXT_MODEL_NAME": os.environ.get("TMX_MAXTEXT_MODEL_NAME", backend.maxtext_model_name),
            "TMX_MAXTEXT_LOAD_PARAMETERS_PATH": os.environ.get(
                "TMX_MAXTEXT_LOAD_PARAMETERS_PATH", backend.maxtext_load_parameters_path
            ),
            "TMX_MAXTEXT_BASE_OUTPUT_DIRECTORY": backend.maxtext_base_output_directory,
            "TMX_MAXTEXT_ENTRYPOINT": backend.maxtext_entrypoint,
            "TMX_MAXTEXT_LOSS_ALGO": backend.maxtext_loss_algo,
            "TMX_REWARD_PLUGIN": backend.reward_plugin,
            "TMX_ROLLOUT_ENGINE": backend.rollout_engine,
            "TMX_XPK_CLUSTER": backend.xpk_cluster_name,
            "TMX_XPK_WORKLOAD": backend.xpk_workload_name or spec.name,
            "TMX_XPK_TPU_TYPE": backend.xpk_tpu_type,
            "TMX_XPK_NUM_SLICES": str(backend.xpk_num_slices),
            "TMX_TRAIN_CHIP_COUNT": str(spec.topology.train_chip_count),
            "TMX_ROLLOUT_CHIP_COUNT": str(spec.topology.rollout_chip_count),
            "TMX_TRAIN_WORKERS": ",".join(str(x) for x in spec.topology.train_workers),
            "TMX_ROLLOUT_WORKERS": ",".join(str(x) for x in spec.topology.rollout_workers),
        }
    )
    if backend.experimental_adapter_ack:
        env["TMX_MAXTEXT_TUNIX_EXPERIMENTAL_ACK"] = "1"
    return env


def build_raw_tpu_plan(spec: ExperimentSpec) -> SubmissionPlan:
    env = _common_env(spec)
    topo = spec.topology
    net = spec.network
    blockers: list[str] = []
    warnings: list[str] = ["raw TPU path is the proven Explorer backend; XPK/MaxText/Tunix remains experimental"]
    if spec.algorithm.phase.startswith("explorer"):
        # The JAX pipeline can continue into SFT/Actual by default. Production
        # Explorer specs are phase-scoped, so stop after Explorer unless a
        # caller intentionally exports a different value.
        env["TMX_PIPELINE_STOP_AFTER_EXPLORER"] = "1"
    elif spec.algorithm.phase in {"actual_sft", "actual_grpo"}:
        trajectory_uri = _actual_trajectory_uri(spec)
        if not trajectory_uri:
            blockers.append(
                f"{spec.algorithm.phase} requires HF Explorer trajectories via "
                "resume.existing_trajectories or resume.source_run"
            )
        else:
            env.update(
                {
                    "TMX_PIPELINE_SKIP_EXPLORER": "1",
                    "TMX_EXISTING_TRAJECTORIES": trajectory_uri,
                    "TMX_PIPELINE_CLEAR_MEMORY_BEFORE_ACTUAL": "1",
                    # Avoid the raw launcher interpreting source_run as an
                    # Explorer checkpoint resume. Actual stages consume the
                    # trajectories instead and build SFT/Actual locally.
                    "SOURCE_RUN": "",
                    "SOURCE_STEP": "",
                    "INIT_CHECKPOINT": "",
                }
            )
            if spec.algorithm.phase == "actual_sft":
                env["TMX_PIPELINE_STOP_AFTER_SFT"] = "1"
            else:
                env["TMX_PIPELINE_STOP_AFTER_SFT"] = "0"
            if spec.resume.init_checkpoint:
                warnings.append(
                    "raw Actual path ignores resume.init_checkpoint and runs SFT+Actual from HF trajectories; "
                    "use a dedicated SFT-checkpoint runner for true Actual-only resume"
                )
    rollout_worker_count = len(topo.rollout_workers)
    min_healthy = max(1, int(rollout_worker_count * 0.75))
    env.update({
        "TRC2_PROJECT": net.project,
        "TRC2_ZONE": net.zone,
        "TRC2_TPU_NAME": net.tpu_name,
        "TRC2_EXPECTED_ACCELERATOR": net.accelerator_type,
        "TPU_ACCELERATOR_TYPE": net.accelerator_type,
        "TRAIN_WORKERS": " ".join(str(x) for x in topo.train_workers),
        "SERVE_WORKERS": " ".join(str(x) for x in topo.rollout_workers),
        "VLLM_CONCURRENCY": str(max(1, topo.rollout_chip_count)),
        "VLLM_FANOUT_PER_PROMPT": str(min(16, max(1, topo.rollout_chip_count))),
        "MIN_HEALTHY_SERVE_WORKERS": str(min_healthy),
        "TMX_VLLM_RELOAD_MIN_HEALTHY_WORKERS": str(min_healthy),
        # TRC pods are created with external IPs and must not rely on IAP/NAT.
        # Per-worker fanout is slower than gcloud --worker=all, but it avoids
        # one flaky worker aborting a full 256-chip launch.
        "TMX_GCLOUD_TPU_USE_IAP": "0",
        "TMX_FAST_PACKAGE_FANOUT": "0",
        "TMX_FAST_WORKER_SCRIPT_FANOUT": "0",
        "JAX_HEARTBEAT_TIMEOUT_SECONDS": "900",
        "TMX_TRAIN_LAUNCH_STABILITY_SECONDS": "3",
    })
    env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
    guard_cmd = f"{env_prefix} scripts/trc2_v5lite256_guard.sh"
    cmd = f"{env_prefix} scripts/launch_trc2_v5lite256_tmx_jax_resume.sh"
    return SubmissionPlan(
        backend="raw-tpu-tmx-jax",
        commands=(
            guard_cmd,
            cmd,
        ),
        env=env,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
    )


def build_xpk_maxtext_plan(spec: ExperimentSpec) -> SubmissionPlan:
    backend = spec.backend
    blockers: list[str] = []
    warnings: list[str] = []
    image = os.environ.get("TMX_XPK_CONTAINER_IMAGE", "").strip() or backend.container_image
    engine = os.environ.get("TMX_BACKEND_ENGINE", "").strip() or backend.engine
    maxtext_model_name = os.environ.get("TMX_MAXTEXT_MODEL_NAME", "").strip() or backend.maxtext_model_name
    maxtext_load_path = os.environ.get("TMX_MAXTEXT_LOAD_PARAMETERS_PATH", "").strip() or backend.maxtext_load_parameters_path
    if not image:
        blockers.append("backend.container_image is required for XPK/GKE workload submission")
    if engine != "maxtext_tunix":
        blockers.append("XPK/GKE production backend currently supports engine='maxtext_tunix' only")
    if not maxtext_model_name:
        blockers.append("backend.maxtext_model_name or TMX_MAXTEXT_MODEL_NAME is required for MaxText")
    if not maxtext_load_path:
        blockers.append("backend.maxtext_load_parameters_path or TMX_MAXTEXT_LOAD_PARAMETERS_PATH is required")
    if not backend.experimental_adapter_ack and os.environ.get("TMX_MAXTEXT_TUNIX_EXPERIMENTAL_ACK", "") != "1":
        blockers.append("set backend.experimental_adapter_ack=true or TMX_MAXTEXT_TUNIX_EXPERIMENTAL_ACK=1 after container smoke test")
    if spec.network.create_tpu:
        blockers.append("TRC TPU creation must remain outside this pipeline; use tpunanny-created TPU")
    env = _workload_env(spec, engine=engine)
    if maxtext_model_name:
        env["TMX_MAXTEXT_MODEL_NAME"] = maxtext_model_name
    if maxtext_load_path:
        env["TMX_MAXTEXT_LOAD_PARAMETERS_PATH"] = maxtext_load_path
    if image:
        env["TMX_XPK_CONTAINER_IMAGE"] = image
    inner_command = _env_command(env, "python -m tmx.orchestration.workload run")
    workload_mode = "create-pathways" if backend.xpk_use_pathways else "create"
    command = _quote_join(
        [
            "xpk",
            "workload",
            workload_mode,
            "--workload",
            backend.xpk_workload_name or spec.name,
            "--docker-image",
            image or "<container_image_required>",
            "--cluster",
            backend.xpk_cluster_name,
            "--tpu-type",
            backend.xpk_tpu_type,
            "--num-slices",
            backend.xpk_num_slices,
            "--project",
            spec.network.project,
            "--priority",
            backend.xpk_priority,
            "--zone",
            spec.network.zone,
            "--command",
            inner_command,
        ]
    )
    warnings.append("do not pass private/internal-IP flags; TRC external IPs must stay enabled")
    warnings.append("XPK creates/submits workloads only; TRC TPU creation remains managed by tpunanny")
    warnings.append("MaxText/Tunix adapter must pass a smoke test before removing experimental_adapter_ack blocker")
    return SubmissionPlan(
        backend="xpk-gke-maxtext-tunix",
        commands=(command,),
        env=env,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
    )


def build_tunix_direct_plan(spec: ExperimentSpec) -> SubmissionPlan:
    """Plan the direct TPU-VM Tunix/MaxText/vLLM path.

    This is the no-XPK/TRC-friendly production path used for the current
    v5litepod-256 resume. It does not create TPU resources, NATs, or routers;
    it only runs the already-existing TPU VM through the gate scripts.
    """
    env = _workload_env(spec, engine="tunix-cli-direct")
    env.update(
        {
            "TPU_PROJECT": spec.network.project,
            "TPU_ZONE": spec.network.zone,
            "TPU_NAME": spec.network.tpu_name,
            "TPU_WORKER": "all",
            "TPU_BATCH_SIZE": str(spec.topology.total_workers),
            "TMX_USE_TUNIX_EXPERIMENTAL": "1",
            "GATES": os.environ.get("GATES", "1 3 4 6"),
            "TIERS": os.environ.get("TIERS", "1 2 3"),
            "SKIP_PREFLIGHT": "1",
        }
    )
    blockers: list[str] = []
    if spec.network.create_tpu:
        blockers.append("direct Tunix path only uses an existing TPU; create_tpu must be false")
    if spec.network.allow_nat or spec.network.allow_internal_ips:
        blockers.append("direct Tunix path requires external-IP TPU access with no NAT/internal-IP flags")
    if spec.storage.allow_gcs:
        blockers.append("direct Tunix path is HF-only; storage.allow_gcs must be false")

    spec_json = json.dumps(spec.to_dict(), sort_keys=True, separators=(",", ":"))
    spec_b64 = base64.b64encode(spec_json.encode("utf-8")).decode("ascii")
    direct_env = {
        "TPU_PROJECT": spec.network.project,
        "TPU_ZONE": spec.network.zone,
        "TPU_NAME": spec.network.tpu_name,
        "TPU_WORKER": "all",
        "TPU_BATCH_SIZE": str(spec.topology.total_workers),
        "TMX_EXPERIMENT_SPEC_JSON_B64": spec_b64,
        "TMX_USE_TUNIX_EXPERIMENTAL": "1",
        "SKIP_PREFLIGHT": "1",
        "GATES": env["GATES"],
        "TIERS": env["TIERS"],
    }
    command = _env_command(direct_env, "bash infra/xpk/run_paper_pipeline.sh")
    warnings = (
        "direct path assumes bootstrap_tunix_env.sh can install Tunix/MaxText on every selected TPU worker",
        "λ>0 runs use tmx.orchestration.tunix_main and require the hidden-state novelty smoke to pass",
        "do not add private-only networking, NAT gateway, router, or TPU creation flags",
    )
    return SubmissionPlan(
        backend="tunix-cli-direct",
        commands=(command,),
        env=env,
        blockers=tuple(blockers),
        warnings=warnings,
    )


def build_submission_plan(spec: ExperimentSpec, backend: str | None = None) -> SubmissionPlan:
    selected = backend or spec.backend.preferred
    if selected == "raw-tpu-tmx-jax":
        return build_raw_tpu_plan(spec)
    if selected == "tunix-cli-direct":
        return build_tunix_direct_plan(spec)
    if selected in {"xpk-gke-maxtext", "xpk-gke-maxtext-tunix", "maxtext-tunix"}:
        return build_xpk_maxtext_plan(spec)
    raise ValueError(f"unknown backend {selected!r}")
