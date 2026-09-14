"""Translate ExperimentSpec into Tunix CLI overrides.

Tunix's CLI takes a base YAML plus `key=value` overrides::

    python -m tunix.cli.grpo_main <base.yaml> key1=v1 key2=v2 ...

This module emits the override list from a TMX ExperimentSpec so we don't
maintain a parallel YAML for every spec variant. It is intentionally limited
to the correctness-only Tunix CLI path. Novelty/RND Explorer runs use the
proven raw ``tmx-jax`` backend until the programmatic Tunix adapter passes a
real TPU smoke; the CLI cannot inject the custom rollout needed for RND.

  - GRPO with KL_beta = 0 for correctness-only baselines
  - reward functions: tmx.orchestration.tunix_rewards
  - rollout via vLLM with TPU JAX backend

Note on key naming: the override keys follow the schema in
``tunix/cli/base_agentic_config.yaml``. Bumping the Tunix pin may rename
keys; any unknown key will be rejected at startup and surfaced in Gate 1.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

from .spec import ExperimentSpec, load_experiment_spec

_DEFAULT_BASE_YAML = "/opt/tunix/tunix/cli/base_agentic_config.yaml"


def _mesh_shape(device_count: int) -> str:
    """Return a conservative 2D fsdp/tp mesh string for Tunix/MaxText.

    We keep TP at 4 when possible because the rollout mesh maps directly to
    vLLM's TP dimension in Tunix's vLLM integration. For the actor, this is
    also a reasonable Qwen3-1.7B sharding default on v5e/v6e.
    """
    if device_count <= 0:
        raise ValueError("mesh device_count must be positive")
    if device_count % 4 == 0:
        return f"({device_count // 4},4)"
    return f"({device_count},1)"


def _emit_common_overrides(spec: ExperimentSpec, *, allow_novelty: bool) -> list[str]:
    """Return Tunix override strings shared by CLI and programmatic paths."""
    alg = spec.algorithm
    backend = spec.backend
    storage = spec.storage
    resume = spec.resume
    if float(alg.lambda_novelty) > 0.0 and not allow_novelty:
        raise ValueError(
            "Tunix CLI cannot run TMX novelty/RND Explorer jobs because it cannot "
            "inject the custom novelty rollout. Use the raw tmx-jax backend for "
            "production novelty runs, or the explicitly experimental "
            "tmx.orchestration.tunix_main adapter after TPU smoke."
        )

    model_id = alg.model_name
    model_short = model_id.split("/")[-1]
    train_devices = spec.topology.train_chip_count
    rollout_devices = spec.topology.rollout_chip_count

    overrides: list[str] = [
        # Model — load via MaxText so we can use sharded JAX training.
        "model_config.model_source=maxtext",
        f"model_config.model_name={model_short}",
        f"model_config.model_id={model_id}",
        "actor_model_config.model_source=maxtext",
        f"actor_model_config.model_name={model_short}",
        f"actor_model_config.model_id={model_id}",
        "reference_model_config.model_source=maxtext",
        f"reference_model_config.model_name={model_short}",
        f"reference_model_config.model_id={model_id}",
        "rollout_model_config.model_source=maxtext",
        f"rollout_model_config.model_name={model_short}",
        f"rollout_model_config.model_id={model_id}",
        f"tokenizer_config.tokenizer_path={model_id}",
        "tokenizer_config.tokenizer_type=huggingface",
        "actor_model_config.mesh.axis_names=('fsdp','tp')",
        f"actor_model_config.mesh.shape={_mesh_shape(train_devices)}",
        "reference_model_config.same_mesh_as=actor",
        "reference_model_config.mesh={}",
        "rollout_model_config.mesh.axis_names=('fsdp','tp')",
        f"rollout_model_config.mesh.shape={_mesh_shape(rollout_devices)}",

        # Dataset and training.
        f"dataset_name={alg.dataset_name}",
        f"batch_size={alg.grpo_batch_size}",
        f"rl_training_config.max_steps={alg.grpo_max_steps}",
        f"rl_training_config.gradient_accumulation_steps={alg.grpo_grad_accum}",
        f"rl_training_config.actor_optimizer_config.value={alg.grpo_lr}",
        f"rl_training_config.actor_optimizer_config.peak_value={alg.grpo_lr}",
        "rl_training_config.actor_optimizer_config.init_value=0.0",
        "rl_training_config.actor_optimizer_config.end_value=0.0",
        f"rl_training_config.checkpoint_root_directory={backend.maxtext_base_output_directory}",
        f"rl_training_config.checkpointing_options.save_interval_steps={storage.checkpoint_every_steps}",

        # Correctness-only GRPO config for the CLI path. The programmatic path
        # adds DAPO/Dr.GRPO-specific keys below and constructs DAPOConfig itself.
        f"grpo_config.num_generations={alg.grpo_num_generations}",
        "grpo_config.num_iterations=1",
        f"grpo_config.epsilon={alg.grpo_clip_epsilon}",

        # Rollout — vLLM-on-TPU with JAX backend (Tunix's pinned commit).
        f"rollout_config.max_prompt_length={alg.max_prompt_len}",
        f"rollout_config.total_generation_steps={alg.max_completion_len}",
        f"rollout_config.rollout_vllm_model_version={model_id}",
        "rollout_config.rollout_vllm_tpu_backend_type=jax",
        f"rollout_config.rollout_vllm_max_num_batched_tokens={alg.max_total_len}",
        f"rollout_config.rollout_vllm_max_num_seqs={max(1, alg.grpo_batch_size * alg.grpo_num_generations)}",
        "rollout_config.rollout_vllm_async_scheduling=true",
        f"rollout_engine={backend.rollout_engine}",
        "training_mode=grpo",

        # Reward plugin: TMX correctness + incorrectness_penalty (sums to +1/-1).
        'reward_functions=["tmx.orchestration.tunix_rewards"]',
    ]

    if allow_novelty:
        overrides.extend(
            [
                "grpo_config.advantage_estimator=drgrpo",
                "grpo_config.loss_agg_mode=sequence-mean-token-scale",
                f"grpo_config.epsilon_high={alg.grpo_clip_epsilon_high}",
                f"grpo_config.dynamic_sampling={str(alg.dynamic_sampling).lower()}",
                "grpo_config.beta=null",
                "grpo_config.overlong_buffer.enable=true",
                f"grpo_config.overlong_buffer.overlong_buffer_length={alg.soft_overlong_cache_len}",
                "grpo_config.overlong_buffer.overlong_buffer_penalty=1.0",
                f"grpo_config.overlong_buffer.max_response_length={alg.max_completion_len}",
            ]
        )
    else:
        overrides.append(f"grpo_config.beta={alg.grpo_kl_beta}")

    # Resume from HF or local checkpoint.
    init_path = resume.init_checkpoint or backend.maxtext_load_parameters_path
    if init_path:
        overrides.append(f"model_config.model_path={init_path}")
        overrides.append(f"actor_model_config.model_path={init_path}")
        overrides.append(f"reference_model_config.model_path={init_path}")
        overrides.append(f"rollout_model_config.model_path={init_path}")

    # Pathways toggle (multi-host single-controller).
    if backend.xpk_use_pathways:
        overrides.append("rollout_config.rollout_sglang_jax_enable_single_process=true")

    # Tunix CLI does not currently own our HF artifact mirror. Production
    # HF upload remains process-zero in the raw TMX pipeline.

    return overrides


def emit_overrides(spec: ExperimentSpec) -> list[str]:
    """Return CLI-safe overrides for correctness-only Tunix GRPO."""
    return _emit_common_overrides(spec, allow_novelty=False)


def emit_programmatic_overrides(spec: ExperimentSpec) -> list[str]:
    """Return overrides for the TMX programmatic Tunix DAPO/Dr.GRPO path."""
    return _emit_common_overrides(spec, allow_novelty=True)


def emit_command(spec: ExperimentSpec, *, base_yaml: str = _DEFAULT_BASE_YAML) -> list[str]:
    """Full CLI invocation for ``python -m tunix.cli.grpo_main``."""
    return [
        sys.executable, "-m", "tunix.cli.grpo_main",
        base_yaml,
        *emit_overrides(spec),
    ]


def emit_programmatic_argv(
    spec: ExperimentSpec, *, base_yaml: str = _DEFAULT_BASE_YAML
) -> list[str]:
    """Argv list suitable for constructing ``TMXGrpoPipeline``."""
    return [
        "tmx-tunix-main",
        base_yaml,
        *emit_programmatic_overrides(spec),
    ]


def _cmd_emit(args: argparse.Namespace) -> int:
    spec = load_experiment_spec(args.spec)
    if args.format == "shell":
        print(" ".join(shlex.quote(s) for s in emit_overrides(spec)))
    elif args.format == "command":
        print(" ".join(shlex.quote(s) for s in emit_command(spec, base_yaml=args.base_yaml)))
    else:
        print(json.dumps(emit_overrides(spec), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Emit Tunix CLI overrides from a TMX ExperimentSpec")
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser("emit")
    p.add_argument("--spec", required=True, type=Path)
    p.add_argument("--format", choices=("shell", "command", "json"), default="shell")
    p.add_argument("--base-yaml", default=_DEFAULT_BASE_YAML)
    p.set_defaults(func=_cmd_emit)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
