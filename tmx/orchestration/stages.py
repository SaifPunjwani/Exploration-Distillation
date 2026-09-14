from __future__ import annotations

from dataclasses import replace

from .spec import AlgorithmSpec


BASE_DAPO_16K = AlgorithmSpec(
    phase="explorer_novelty",
    model_name="Qwen/Qwen3-1.7B",
    dataset_name="dapo_math_17k",
    max_prompt_len=2048,
    max_completion_len=16384,
    max_total_len=18432,
    grpo_batch_size=1,
    grpo_grad_accum=1,
    grpo_num_generations=16,
    grpo_max_steps=100,
    grpo_lr=5e-6,
    grpo_clip_epsilon=0.2,
    grpo_clip_epsilon_high=0.28,
    grpo_kl_beta=0.0,
    grpo_advantage_normalization="none",
    dynamic_sampling=True,
    dynamic_sampling_max_attempts=8,
    dynamic_sampling_groups_per_attempt=1,
    soft_overlong_expected_len=13107,
    soft_overlong_cache_len=3277,
)


STAGE_PRESETS: dict[str, AlgorithmSpec] = {
    "explorer_correctness_16k": replace(
        BASE_DAPO_16K,
        phase="explorer_correctness",
        lambda_novelty=0.0,
        incorrect_novelty_scale=0.0,
        dynamic_sampling_accept_reward_variance=False,
    ),
    "explorer_novelty075_16k": replace(
        BASE_DAPO_16K,
        phase="explorer_novelty",
        lambda_novelty=0.75,
        incorrect_novelty_scale=1.0,
        dynamic_sampling_accept_reward_variance=True,
    ),
    "explorer_novelty05_16k": replace(
        BASE_DAPO_16K,
        phase="explorer_novelty",
        lambda_novelty=0.5,
        incorrect_novelty_scale=1.0,
        dynamic_sampling_accept_reward_variance=True,
    ),
    "actual_grpo_16k": replace(
        BASE_DAPO_16K,
        phase="actual_grpo",
        lambda_novelty=0.0,
        incorrect_novelty_scale=0.0,
        grpo_num_generations=16,
        grpo_max_steps=250,
        grpo_lr=1e-6,
        dynamic_sampling=True,
        dynamic_sampling_accept_reward_variance=False,
    ),
    "actual_sft_16k": replace(
        BASE_DAPO_16K,
        phase="actual_sft",
        lambda_novelty=0.0,
        incorrect_novelty_scale=0.0,
        dynamic_sampling=False,
        grpo_max_steps=0,
        distill_epochs=2,
        distill_batch_size=1,
        max_distill_examples=20000,
    ),
    "eval_aime24_avg32_32k": replace(
        BASE_DAPO_16K,
        phase="eval",
        lambda_novelty=0.0,
        incorrect_novelty_scale=0.0,
        dynamic_sampling=False,
        grpo_max_steps=0,
        benchmark_num_rollouts=32,
        benchmark_max_completion_len=32768,
    ),
}


def stage_preset(name: str) -> AlgorithmSpec:
    try:
        return STAGE_PRESETS[name]
    except KeyError as exc:
        available = ", ".join(sorted(STAGE_PRESETS))
        raise ValueError(f"unknown stage preset {name!r}; available: {available}") from exc
