"""Config for the JAX TPU two-model pipeline.

Loads Qwen3 config from HF Hub; overlays training hyperparams from CLI / env.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

from .model import Qwen3Config


@dataclass
class TrainConfig:
    # Model / tokenizer
    model_name: str = "Qwen/Qwen3-1.7B"
    # Context contract: 16k completion matches Qwen3 generation length used in eval
    # so train and eval distributions are aligned (no train/eval gap from clipping).
    max_prompt_len: int = 2048
    max_completion_len: int = 16384
    max_total_len: int = 18432
    # GRPO contract
    grpo_batch_size: int = 1
    grpo_grad_accum: int = 1
    grpo_num_generations: int = 16
    grpo_max_steps: int = 500
    grpo_loss_type: str = "dr_grpo"
    # Loss aggregation. "dr_grpo" -> Liu 2025 fixed-denominator (default for our method).
    # "sequence-mean-token-mean" -> vanilla GRPO / Shao 2024 / tunix reference (use for vanilla baseline).
    # "token-mean" -> DAPO Eq. 12 token-level.
    grpo_loss_aggregation: str = "dr_grpo"
    optimizer: str = "adamw"  # Contract path is AdamW-only; non-adamw values are rejected.
    grpo_lr: float = 5e-6
    grpo_clip_epsilon: float = 0.2
    grpo_clip_epsilon_high: float = 0.28
    grpo_kl_beta: float = 0.0
    grpo_updates_per_rollout: int = 1
    grpo_advantage_normalization: str = "none"  # none | std; production DAPO/Dr.GRPO uses none.
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.0
    # Rollout/generation
    trajectory_batch_size: int = 2    # total prompts per optimizer step; should equal batch_size * grad_accum
    vllm_server_urls: str = ""         # comma-separated
    vllm_concurrency: int = 16
    vllm_fanout_per_prompt: int = 1
    dynamic_sampling: bool = True
    dynamic_sampling_max_attempts: int = 8
    mask_truncated_completions: bool = False
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    enable_thinking: str = "auto"      # auto | always | never
    # Reward
    reward_correct: float = 1.0
    reward_incorrect: float = -1.0
    format_penalty_weight: float = 0.0
    nontermination_penalty: float = 0.0
    invalid_answer_penalty: float = 0.0
    soft_overlong_expected_len: int = 13107
    soft_overlong_cache_len: int = 3277
    # Novelty
    lambda_novelty: float = 0.0
    novelty_metric: str = "sqrt_mse"
    novelty_clip: float = 0.0  # 0 = disabled
    novelty_feature_source: str = "multilayer"
    novelty_layers: List[int] = field(default_factory=lambda: [7, 14, 21])
    novelty_layer_pool: str = "mean"
    novelty_layer_agg: str = "mean"
    novelty_text_contract: str = "full_raw"
    novelty_feature_max_length: int = 10240
    novelty_zscore: str = "off"
    correctness_zscore: str = "off"
    rnd_hidden: int = 512
    rnd_lr: float = 1e-4
    incorrect_novelty_scale: float = 0.0
    gate_novelty_by_quality: bool = False
    # Dataset
    dataset_name: str = "dapo_math_17k"
    max_train_examples: int = 20000
    max_eval_examples: int = 32
    # SFT
    distill_lr: float = 5e-6
    distill_epochs: int = 2
    distill_batch_size: int = 1
    max_distill_examples: int = 20000
    min_accepted_trajectories: int = 500
    max_accepted_trajectories: int = 500
    accepted_selection_policy: str = "quality_first"  # quality_first | chronological
    # Actual GRPO
    actual_online_grpo_max_steps: int = 250
    actual_grpo_lr: float = 1e-6
    actual_grpo_kl_coef: float = 0.0
    # Multi-round: run Explorer→SFT→Actual repeatedly, where round N+1's Explorer
    # init_checkpoint = round N's Actual final checkpoint. Each round trains on a
    # disjoint shard of the dataset and gets a fresh RND novelty signal.
    # pipeline_mode="two_model" (default) is the single-pass behavior unchanged.
    pipeline_mode: str = "two_model"  # two_model | multi_round
    num_rounds: int = 1
    # split/constant_total = divide total budgets across rounds.
    # constant/full_per_round = spend the full budget every round.
    round_budget_mode: str = "split"
    # uniform = deterministic shuffled disjoint shards; sequential = ordered contiguous shards.
    round_shard_policy: str = "uniform"
    multi_round_reset_rnd: bool = True  # True = fresh RND each round (recommended); False = restore from prev
    # Eval — 32k matches Qwen3 baseline contract (paper-comparable). Training
    # runs at 16k completion (memory-bound) but eval is uncapped to allow full
    # thinking traces.
    benchmark_num_rollouts: int = 32
    benchmark_rollout_max_completion_len: int = 32768
    # Output
    output_dir: str = "runs/jax_run"
    wandb_project: str = "two-model-explore"
    wandb_entity: str = os.environ.get("WANDB_ENTITY", "")  # empty = the W&B default entity
    wandb_run_name: str = ""
    seed: int = 0
    # Fault tolerance
    save_every_steps: int = 50
    init_checkpoint: str = ""
    rescore_chunk_size: int = 1
    logprob_chunk_tokens: int = 64
    # Mesh auto-detected from jax.devices()


def validate_contract(cfg: TrainConfig, *, require_eval_contract: bool = False) -> None:
    """Fail fast on accidental deviations from the current Qwen3/AIME contract.

    Step counts and accepted-trajectory thresholds are intentionally configurable
    for smoke runs. The sequence/group/loss/eval shape is not.
    """
    long_completion_train = os.environ.get("TMX_ALLOW_LONG_COMPLETION_TRAIN", "0") in {
        "1", "true", "True", "yes", "on"
    }
    full_dapo_drgrpo = os.environ.get("TMX_DAPO_DRGRPO_FULL_CONTRACT", "0") in {
        "1", "true", "True", "yes", "on"
    }
    contract_completion_len = int(
        os.environ.get("TMX_DAPO_DRGRPO_CONTRACT_COMPLETION_LEN", "20480") or "20480"
    )
    contract_soft_expected = int(
        os.environ.get("TMX_DAPO_DRGRPO_CONTRACT_SOFT_OVERLONG_EXPECTED_LEN", "16384")
        or "16384"
    )
    contract_soft_cache = int(
        os.environ.get("TMX_DAPO_DRGRPO_CONTRACT_SOFT_OVERLONG_CACHE_LEN", "4096")
        or "4096"
    )
    expected_num_generations = int(
        os.environ.get(
            "TMX_EXPECTED_GRPO_NUM_GENERATIONS",
            "16",
        )
        or "16"
    )
    expected_batch_size = int(os.environ.get("TMX_EXPECTED_GRPO_BATCH_SIZE", "1") or "1")
    expected = {
        "grpo_batch_size": expected_batch_size,
        "grpo_num_generations": expected_num_generations,
        "grpo_loss_type": "dr_grpo",
        "optimizer": "adamw",
        "novelty_zscore": "off",
        "correctness_zscore": "off",
        "format_penalty_weight": 0.0,
        "gate_novelty_by_quality": False,
    }
    mismatches = []
    if cfg.max_prompt_len != 2048:
        mismatches.append(f"max_prompt_len={cfg.max_prompt_len!r} expected 2048")
    if long_completion_train:
        allowed_completion_lens = (
            (contract_completion_len,)
            if full_dapo_drgrpo
            else (8192, 12288, 16384, 20480)
        )
        if cfg.max_completion_len not in allowed_completion_lens:
            mismatches.append(
                "max_completion_len="
                f"{cfg.max_completion_len!r} expected one of "
                f"{'/'.join(str(x) for x in allowed_completion_lens)}"
            )
        if cfg.max_total_len != cfg.max_prompt_len + cfg.max_completion_len:
            mismatches.append(
                f"max_total_len={cfg.max_total_len!r} expected prompt+completion "
                f"({cfg.max_prompt_len + cfg.max_completion_len})"
            )
    else:
        # Training contract: 16k completion (memory-bound on v5lite/v6e).
        # Eval generation runs at 32k via benchmark_rollout_max_completion_len.
        if cfg.max_completion_len != 16384:
            mismatches.append(f"max_completion_len={cfg.max_completion_len!r} expected 16384")
        if cfg.max_total_len != 18432:
            mismatches.append(f"max_total_len={cfg.max_total_len!r} expected 18432")
    for field_name, expected_value in expected.items():
        actual = getattr(cfg, field_name)
        if isinstance(expected_value, str):
            ok = str(actual).lower() == expected_value
        else:
            ok = actual == expected_value
        if not ok:
            mismatches.append(f"{field_name}={actual!r} expected {expected_value!r}")
    expected_grad_accum = int(os.environ.get("TMX_EXPECTED_GRPO_GRAD_ACCUM", "1") or "1")
    if int(cfg.grpo_grad_accum) != expected_grad_accum:
        mismatches.append(f"grpo_grad_accum={cfg.grpo_grad_accum!r} expected {expected_grad_accum}")
    adv_norm = str(cfg.grpo_advantage_normalization or "none").lower()
    if float(cfg.lambda_novelty or 0.0) != 0.0 and adv_norm not in ("none", "off"):
        allow_novelty_adv_norm = os.environ.get(
            "TMX_ALLOW_NOVELTY_ADVANTAGE_NORMALIZATION", "0"
        ) in {"1", "true", "True", "yes", "on"}
        if not allow_novelty_adv_norm:
            mismatches.append(
                "grpo_advantage_normalization must stay none/off for lambda_novelty>0 "
                "unless TMX_ALLOW_NOVELTY_ADVANTAGE_NORMALIZATION=1"
            )
    if cfg.max_train_examples < 17000:
        mismatches.append(
            f"max_train_examples={cfg.max_train_examples!r} expected full DAPO scale (>=17000)"
        )
    if full_dapo_drgrpo:
        if not bool(cfg.dynamic_sampling):
            mismatches.append("dynamic_sampling=False expected True for DAPO dynamic sampling")
        expected_updates = int(os.environ.get("TMX_EXPECTED_GRPO_UPDATES_PER_ROLLOUT", "16") or "16")
        if int(cfg.grpo_updates_per_rollout) != expected_updates:
            mismatches.append(
                f"grpo_updates_per_rollout={cfg.grpo_updates_per_rollout!r} expected {expected_updates}"
            )
        if str(cfg.grpo_advantage_normalization or "none").lower() not in ("none", "off"):
            mismatches.append(
                "grpo_advantage_normalization must be none/off for true Dr.GRPO+DAPO "
                "(advantage = reward - group_mean, no std division per Dr.GRPO paper)"
            )
        if abs(float(cfg.grpo_clip_epsilon) - 0.2) > 1e-9:
            mismatches.append(f"grpo_clip_epsilon={cfg.grpo_clip_epsilon!r} expected 0.2")
        if abs(float(cfg.grpo_clip_epsilon_high) - 0.28) > 1e-9:
            mismatches.append(f"grpo_clip_epsilon_high={cfg.grpo_clip_epsilon_high!r} expected 0.28")
        if abs(float(cfg.grpo_kl_beta or 0.0)) > 1e-12:
            mismatches.append(f"grpo_kl_beta={cfg.grpo_kl_beta!r} expected 0.0")
        if abs(float(cfg.reward_correct) - 1.0) > 1e-9:
            mismatches.append(f"reward_correct={cfg.reward_correct!r} expected 1.0")
        if abs(float(cfg.reward_incorrect) - (-1.0)) > 1e-9:
            mismatches.append(f"reward_incorrect={cfg.reward_incorrect!r} expected -1.0")
        if int(cfg.soft_overlong_expected_len or 0) != contract_soft_expected:
            mismatches.append(
                "soft_overlong_expected_len="
                f"{cfg.soft_overlong_expected_len!r} expected {contract_soft_expected}"
            )
        if int(cfg.soft_overlong_cache_len or 0) != contract_soft_cache:
            mismatches.append(
                f"soft_overlong_cache_len={cfg.soft_overlong_cache_len!r} expected {contract_soft_cache}"
            )
    if require_eval_contract:
        if cfg.benchmark_num_rollouts != 32:
            mismatches.append(f"benchmark_num_rollouts={cfg.benchmark_num_rollouts!r} expected 32")
        if cfg.benchmark_rollout_max_completion_len != 32768:
            mismatches.append(
                "benchmark_rollout_max_completion_len="
                f"{cfg.benchmark_rollout_max_completion_len!r} expected 32768"
            )
    if mismatches:
        raise ValueError("Contract violation: " + "; ".join(mismatches))


def parse_args() -> TrainConfig:
    import argparse
    p = argparse.ArgumentParser()
    # Simple direct mapping of TrainConfig fields → --kebab-case flags.
    cfg = TrainConfig()
    for fname, fval in vars(cfg).items():
        flag = "--" + fname.replace("_", "-")
        kind = type(fval)
        if kind is list:
            p.add_argument(flag, type=lambda s: [int(x) for x in s.split(",")], default=fval)
        elif kind is bool:
            p.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"), default=fval)
        else:
            p.add_argument(flag, type=kind, default=fval)
    ns = p.parse_args()
    return TrainConfig(**vars(ns))
