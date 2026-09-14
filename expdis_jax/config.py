"""Config for the Exploration-Distillation JAX TPU pipeline.

TrainConfig (paper defaults), CLI parsing, and the fail-closed contract validator.
Model dimensions are read from the HF checkpoint by ``weights.load_hf_config`` via
``train._make_qwen3_config``, not here.
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainConfig:
    # Model / tokenizer
    model_name: str = "Qwen/Qwen3-1.7B"
    # Context contract: 32k completions in training and evaluation.  The model
    # sequence length includes the 2k prompt allowance.
    max_prompt_len: int = 2048
    max_completion_len: int = 32768
    max_total_len: int = 34816
    # GRPO contract
    # Four prompt groups x 16 generations = 64 selected rollout rows/update.
    grpo_batch_size: int = 4
    grpo_grad_accum: int = 1
    grpo_num_generations: int = 16
    grpo_max_steps: int = 200
    grpo_loss_type: str = "dr_grpo"
    # Loss aggregation. "dr_grpo" -> Liu 2025 fixed-denominator (default for our method).
    # "sequence-mean-token-mean" -> vanilla GRPO / Shao 2024 / tunix reference (use for vanilla baseline).
    # "token-mean" -> DAPO Eq. 12 token-level.
    grpo_loss_aggregation: str = "dr_grpo"
    # Dr.GRPO fixed per-row loss denominator: the 32,768-token completion
    # budget, independent of the realized completion length.  Kept as a field
    # so an ablation can change it explicitly; the full contract pins it to the
    # completion cap.
    dr_grpo_loss_normalizer: int = 32768
    optimizer: str = "adamw"  # Contract path is AdamW-only; non-adamw values are rejected.
    grpo_lr: float = 5e-6
    # Per-stage learning-rate schedules (opt-in; "constant" defaults preserve the
    # frozen constant-LR contracts byte-for-byte). Modes: "constant" |
    # "linear_warmup_constant" (DAPO: update n of warmup w uses peak*min(n/w,1),
    # matching tmx_gpu.learning_rate_for_update) | "cosine" | "warmup_cosine".
    # Schedules are pure functions of the successful-optimizer-update count —
    # never microbatches or discarded rollout groups — and reset per stage/round
    # because each stage builds a fresh optimizer.
    grpo_lr_schedule: str = "constant"          # Scout / Explorer GRPO
    grpo_lr_warmup_steps: int = 0
    grpo_lr_decay_steps: int = 0
    grpo_lr_min_ratio: float = 0.0
    actual_grpo_lr_schedule: str = "constant"   # Central (mapped onto grpo_* for stage 4)
    actual_grpo_lr_warmup_steps: int = 0
    actual_grpo_lr_decay_steps: int = 0
    actual_grpo_lr_min_ratio: float = 0.0
    distill_lr_schedule: str = "constant"       # SFT distillation
    distill_lr_warmup_steps: int = 0
    distill_lr_decay_steps: int = 0
    distill_lr_min_ratio: float = 0.0
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
    trajectory_batch_size: int = 4    # total prompts per optimizer step; equals batch_size * grad_accum
    vllm_server_urls: str = ""         # comma-separated
    vllm_concurrency: int = 16
    vllm_fanout_per_prompt: int = 1
    dynamic_sampling: bool = True
    dynamic_sampling_max_attempts: int = 8
    mask_truncated_completions: bool = True
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
    soft_overlong_expected_len: int = 26214
    soft_overlong_cache_len: int = 6554
    # Novelty
    lambda_novelty: float = 0.0
    novelty_metric: str = "sqrt_mse"
    novelty_clip: float = 0.0  # 0 = disabled
    novelty_feature_source: str = "multilayer"
    # Empty/auto resolves the quarter-depth block indices from the loaded model.
    novelty_layers: List[int] = field(default_factory=list)
    novelty_layer_pool: str = "mean"
    novelty_layer_agg: str = "mean"
    novelty_text_contract: str = "full_raw"
    novelty_feature_max_length: int = 34816
    novelty_zscore: str = "off"
    correctness_zscore: str = "off"
    rnd_hidden: int = 512
    rnd_lr: float = 1e-4
    incorrect_novelty_scale: float = 0.0
    gate_novelty_by_quality: bool = False
    # Dataset
    dataset_name: str = "dapo_math_17k"
    max_train_examples: int = 20000
    max_eval_examples: int = 32  # unused in expdis_jax; kept for launcher CLI compatibility
    # SFT
    distill_lr: float = 5e-6
    distill_epochs: int = 2
    distill_batch_size: int = 1
    distill_max_total_len: int = 6548
    max_distill_examples: int = 500
    # The paper permits any non-empty accepted pool up to the cap.
    min_accepted_trajectories: int = 1
    max_accepted_trajectories: int = 500
    accepted_selection_policy: str = "coverage_pool_c8"  # coverage_pool_c8 (frozen contract) | quality_first | chronological
    accepted_min_completion_tokens: int = 128
    accepted_max_completion_tokens: int = 4500
    # Legacy alternative-policy option. The paper's coverage_pool_c8 policy
    # always chooses exactly one winner per problem regardless of this field.
    accepted_per_problem_cap: int = 0
    # Actual GRPO
    actual_online_grpo_max_steps: int = 100
    actual_grpo_lr: float = 1e-6
    actual_grpo_kl_coef: float = 0.0
    # Multi-round: run Explorer→SFT→Actual repeatedly, where round N+1's Explorer
    # init_checkpoint = round N's Actual final checkpoint. Each round trains on a
    # disjoint shard of the dataset and gets a fresh RND novelty signal.
    # pipeline_mode="two_model" (default) is the single-pass behavior unchanged.
    pipeline_mode: str = "two_model"  # two_model | multi_round
    num_rounds: int = 1
    scouts_per_round: int = 1
    # Empty keeps lambda_novelty fixed.  Multi-round MR-ME supplies one
    # comma-separated value per round, e.g. 0.75,0.50,0.35,0.25.
    round_novelty_schedule: str = ""
    round_index: int = 1
    scout_index: int = 0
    central_init_policy: str = "carryover"  # carryover | base_reset (ablation)
    # split/constant_total = divide total budgets across rounds.
    # constant/full_per_round = spend the full budget every round.
    round_budget_mode: str = "split"
    # uniform = deterministic shuffled disjoint shards; sequential = ordered contiguous shards.
    round_shard_policy: str = "uniform"
    multi_round_reset_rnd: bool = True  # True = fresh RND each round (recommended); False = restore from prev
    # Eval uses the same 32k completion budget as training.
    benchmark_num_rollouts: int = 64
    benchmark_rollout_max_completion_len: int = 32768
    # Output
    output_dir: str = "runs/jax_run"
    checkpoint_root: str = ""  # shared filesystem directory or gs:// prefix on multiple hosts
    pipeline_root_dir: str = ""  # populated by the driver; defines stage checkpoint namespaces
    wandb_project: str = "expdis"
    wandb_entity: str = os.environ.get("WANDB_ENTITY", "")  # empty = the W&B default entity
    wandb_run_name: str = ""
    seed: int = 0
    # Fault tolerance
    save_every_steps: int = 50
    # Cross-stage/round handoff and same-stage resume are intentionally
    # separate.  The former restores parameters only with a fresh optimizer
    # and local step zero; the latter restores complete same-run state.
    init_weights_checkpoint: str = ""
    resume_checkpoint: str = ""
    central_parent_checkpoint: str = ""
    # Legacy ambiguous field retained for existing launchers.  New MR-ME code
    # must use one of the two explicit fields above.
    init_checkpoint: str = ""
    rescore_chunk_size: int = 1  # unused in training; kept for debug-script compatibility
    logprob_chunk_tokens: int = 64
    # Mesh auto-detected from jax.devices()


def validate_contract(cfg: TrainConfig, *, require_eval_contract: bool = False) -> None:
    """Fail fast on accidental deviations from the current Qwen3/AIME contract.

    Step counts and accepted-trajectory thresholds are intentionally configurable
    for smoke runs. The sequence/group/loss/eval shape is not.
    """
    long_completion_train = os.environ.get("EXPDIS_ALLOW_LONG_COMPLETION_TRAIN", "0") in {
        "1", "true", "True", "yes", "on"
    }
    full_dapo_drgrpo = os.environ.get("EXPDIS_DAPO_DRGRPO_FULL_CONTRACT", "1") in {
        "1", "true", "True", "yes", "on"
    }
    contract_completion_len = int(
        os.environ.get("EXPDIS_DAPO_DRGRPO_CONTRACT_COMPLETION_LEN", "32768") or "32768"
    )
    contract_soft_expected = int(
        os.environ.get("EXPDIS_DAPO_DRGRPO_CONTRACT_SOFT_OVERLONG_EXPECTED_LEN", "26214")
        or "26214"
    )
    contract_soft_cache = int(
        os.environ.get("EXPDIS_DAPO_DRGRPO_CONTRACT_SOFT_OVERLONG_CACHE_LEN", "6554")
        or "6554"
    )
    contract_loss_normalizer = int(
        os.environ.get("EXPDIS_DAPO_DRGRPO_CONTRACT_LOSS_NORMALIZER", "32768") or "32768"
    )
    expected_num_generations = int(
        os.environ.get(
            "EXPDIS_EXPECTED_GRPO_NUM_GENERATIONS",
            "16",
        )
        or "16"
    )
    expected_batch_size = int(os.environ.get("EXPDIS_EXPECTED_GRPO_BATCH_SIZE", "4") or "4")
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
    if os.environ.get("EXPDIS_NOVELTY_TEXT_HASH_FALLBACK", "0").lower() not in ("0", "false", "no", "off", ""):
        mismatches.append("text-hash novelty was removed; ExpDis requires live hidden-state RND")
    if int(cfg.num_rounds) < 1:
        mismatches.append(f"num_rounds={cfg.num_rounds!r} expected >=1")
    if int(cfg.scouts_per_round) < 1:
        mismatches.append(f"scouts_per_round={cfg.scouts_per_round!r} expected >=1")
    if int(cfg.round_index) < 1 or int(cfg.round_index) > int(cfg.num_rounds):
        mismatches.append(
            f"round_index={cfg.round_index!r} expected within 1..{int(cfg.num_rounds)}"
        )
    if str(cfg.central_init_policy).lower() not in ("carryover", "base_reset"):
        mismatches.append(
            f"central_init_policy={cfg.central_init_policy!r} expected carryover/base_reset"
        )
    if int(cfg.accepted_min_completion_tokens) < 0:
        mismatches.append("accepted_min_completion_tokens must be non-negative")
    if int(cfg.accepted_max_completion_tokens) < int(cfg.accepted_min_completion_tokens):
        mismatches.append(
            "accepted_max_completion_tokens must be >= accepted_min_completion_tokens"
        )
    if int(cfg.distill_max_total_len) != int(cfg.max_prompt_len) + int(
        cfg.accepted_max_completion_tokens
    ):
        mismatches.append(
            "distill_max_total_len must equal max_prompt_len + "
            "accepted_max_completion_tokens"
        )
    if int(cfg.max_accepted_trajectories) > 500 or int(cfg.max_distill_examples) > 500:
        mismatches.append("per-round accepted/SFT cap cannot exceed 500")
    explicit_sources = [
        bool(str(cfg.init_weights_checkpoint or "").strip()),
        bool(str(cfg.resume_checkpoint or "").strip()),
    ]
    if sum(explicit_sources) > 1:
        mismatches.append(
            "init_weights_checkpoint and resume_checkpoint are mutually exclusive"
        )
    try:
        from .lineage import fixed_total_geometry, parse_round_schedule

        if str(cfg.round_budget_mode).lower() in ("split", "constant_total", "total"):
            geometry = fixed_total_geometry(
                scouts_per_round=int(cfg.scouts_per_round),
                rounds=int(cfg.num_rounds),
                total_scout_updates=int(cfg.grpo_max_steps),
                total_central_updates=int(cfg.actual_online_grpo_max_steps),
                prompts_per_update=int(cfg.grpo_batch_size) * int(cfg.grpo_grad_accum),
                generations_per_prompt=int(cfg.grpo_num_generations),
            )
            if any(
                allocation < 1
                for per_round in geometry.scout_updates_by_round
                for allocation in per_round
            ):
                mismatches.append(
                    "fixed-total Scout budget gives at least one Scout zero updates"
                )
            if any(allocation < 1 for allocation in geometry.central_updates_by_round):
                mismatches.append(
                    "fixed-total Central budget gives at least one round zero updates"
                )
        parse_round_schedule(cfg.round_novelty_schedule, int(cfg.num_rounds))
    except (ValueError, IndexError) as exc:
        mismatches.append(str(exc))
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
        # Training contract: 32k completion, matching the pilot budget.
        # Eval generation also runs at 32k via benchmark_rollout_max_completion_len.
        if cfg.max_completion_len != 32768:
            mismatches.append(f"max_completion_len={cfg.max_completion_len!r} expected 32768")
        if cfg.max_total_len != 34816:
            mismatches.append(f"max_total_len={cfg.max_total_len!r} expected 34816")
    for field_name, expected_value in expected.items():
        actual = getattr(cfg, field_name)
        if isinstance(expected_value, str):
            ok = str(actual).lower() == expected_value
        else:
            ok = actual == expected_value
        if not ok:
            mismatches.append(f"{field_name}={actual!r} expected {expected_value!r}")
    expected_grad_accum = int(os.environ.get("EXPDIS_EXPECTED_GRPO_GRAD_ACCUM", "1") or "1")
    if int(cfg.grpo_grad_accum) != expected_grad_accum:
        mismatches.append(f"grpo_grad_accum={cfg.grpo_grad_accum!r} expected {expected_grad_accum}")
    if int(cfg.dr_grpo_loss_normalizer) <= 0:
        mismatches.append(
            f"dr_grpo_loss_normalizer={cfg.dr_grpo_loss_normalizer!r} expected >0"
        )
    expected_prompts_per_update = expected_batch_size * expected_grad_accum
    if int(cfg.trajectory_batch_size) != expected_prompts_per_update:
        mismatches.append(
            "trajectory_batch_size="
            f"{cfg.trajectory_batch_size!r} expected {expected_prompts_per_update} "
            "(grpo_batch_size * grpo_grad_accum)"
        )
    adv_norm = str(cfg.grpo_advantage_normalization or "none").lower()
    if float(cfg.lambda_novelty or 0.0) != 0.0 and adv_norm not in ("none", "off"):
        allow_novelty_adv_norm = os.environ.get(
            "EXPDIS_ALLOW_NOVELTY_ADVANTAGE_NORMALIZATION", "0"
        ) in {"1", "true", "True", "yes", "on"}
        if not allow_novelty_adv_norm:
            mismatches.append(
                "grpo_advantage_normalization must stay none/off for lambda_novelty>0 "
                "unless EXPDIS_ALLOW_NOVELTY_ADVANTAGE_NORMALIZATION=1"
            )
    if cfg.max_train_examples < 17000:
        mismatches.append(
            f"max_train_examples={cfg.max_train_examples!r} expected full DAPO scale (>=17000)"
        )
    if not math.isfinite(float(cfg.lambda_novelty)) or float(cfg.lambda_novelty) < 0:
        mismatches.append("lambda_novelty must be finite and non-negative")
    if full_dapo_drgrpo:
        pinned = {
            "grpo_loss_aggregation": "dr_grpo", "dynamic_sampling_max_attempts": 8,
            "novelty_metric": "sqrt_mse", "novelty_clip": 0.0,
            "novelty_feature_source": "multilayer", "novelty_layer_pool": "mean",
            "novelty_layer_agg": "mean", "novelty_text_contract": "full_raw",
            "rnd_hidden": 512, "rnd_lr": 1e-4,
            "adam_beta1": 0.9, "adam_beta2": 0.95, "adam_epsilon": 1e-8,
            "weight_decay": 0.0, "nontermination_penalty": 0.0,
            "invalid_answer_penalty": 0.0,
            "temperature": 1.0, "top_p": 0.95, "top_k": 20,
            "distill_epochs": 2, "accepted_min_completion_tokens": 128,
            "accepted_max_completion_tokens": 4500,
            "actual_grpo_kl_coef": 0.0, "actual_grpo_lr": 1e-6, "distill_lr": 5e-6,
            "grpo_lr_schedule": "constant", "actual_grpo_lr_schedule": "constant",
            "distill_lr_schedule": "constant", "enable_thinking": "auto",
            "central_init_policy": "carryover", "round_budget_mode": "split",
            "round_shard_policy": "uniform", "multi_round_reset_rnd": True,
        }
        for name, value in pinned.items():
            if getattr(cfg, name) != value:
                mismatches.append(f"{name}={getattr(cfg, name)!r} expected {value!r}")
        if int(cfg.novelty_feature_max_length) < int(cfg.max_completion_len):
            mismatches.append("novelty_feature_max_length must cover the full completion budget")
        for name, default in {
            "EXPDIS_TRAIN_VLLM_RELOAD_EVERY_STEPS": "1",
            "EXPDIS_TRAIN_VLLM_RELOAD_FIRST_STEP": "0",
            "EXPDIS_DYNAMIC_SAMPLING_ELIGIBILITY": "blended_reward",
            "EXPDIS_DYNAMIC_SAMPLING_ACCEPT_REWARD_VARIANCE": "1",
            "EXPDIS_EXTERNAL_VLLM_RELOAD_REQUIRED": "1",
            "EXPDIS_EXTERNAL_VLLM_RELOAD_WAIT": "1",
        }.items():
            if os.environ.get(name, default) != default:
                mismatches.append(f"{name} must be {default} under the paper contract")
        effective_prompts = os.environ.get("EXPDIS_EFFECTIVE_PROMPTS_PER_STEP", "").strip()
        if effective_prompts and effective_prompts != "4":
            mismatches.append("EXPDIS_EFFECTIVE_PROMPTS_PER_STEP must be 4 under the paper contract")
        if not bool(cfg.dynamic_sampling):
            mismatches.append("dynamic_sampling=False expected True for DAPO dynamic sampling")
        if not bool(cfg.mask_truncated_completions):
            mismatches.append(
                "mask_truncated_completions=False expected True for DAPO Overlong Filtering"
            )
        expected_updates = 1
        if int(cfg.grpo_updates_per_rollout) != expected_updates:
            mismatches.append(
                f"grpo_updates_per_rollout={cfg.grpo_updates_per_rollout!r} expected {expected_updates}"
            )
        if (contract_completion_len, contract_loss_normalizer, contract_soft_expected, contract_soft_cache) != (32768, 32768, 26214, 6554):
            mismatches.append("paper length constants cannot be changed through environment overrides")
        if int(cfg.grpo_batch_size) * int(cfg.grpo_grad_accum) != 4 or cfg.grpo_num_generations != 16:
            mismatches.append("paper updates require B=4 prompt groups and G=16 completions")
        if int(cfg.dr_grpo_loss_normalizer) != contract_loss_normalizer:
            mismatches.append(
                "dr_grpo_loss_normalizer="
                f"{cfg.dr_grpo_loss_normalizer!r} expected {contract_loss_normalizer}"
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
        if abs(float(cfg.incorrect_novelty_scale or 0.0)) > 1e-12:
            mismatches.append(
                f"incorrect_novelty_scale={cfg.incorrect_novelty_scale!r} expected 0.0 "
                "(novelty is credited only on verifier-correct completions)"
            )
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
        if int(cfg.soft_overlong_expected_len or 0) + int(cfg.soft_overlong_cache_len or 0) != int(
            cfg.max_completion_len
        ):
            mismatches.append(
                "soft-overlong window must end at max_completion_len: "
                f"{cfg.soft_overlong_expected_len}+{cfg.soft_overlong_cache_len}!="
                f"{cfg.max_completion_len}"
            )
    if require_eval_contract:
        if cfg.benchmark_num_rollouts != 64:
            mismatches.append(f"benchmark_num_rollouts={cfg.benchmark_num_rollouts!r} expected 64")
        if cfg.benchmark_rollout_max_completion_len != 32768:
            mismatches.append(
                "benchmark_rollout_max_completion_len="
                f"{cfg.benchmark_rollout_max_completion_len!r} expected 32768"
            )
    if mismatches:
        raise ValueError("Contract violation: " + "; ".join(mismatches))


def public_config_name(name: str) -> str:
    """Paper role names at the CLI; serialized fields retain compatibility."""
    return (name.replace("actual_online_grpo", "main_rl")
            .replace("actual_grpo", "main_rl")
            .replace("scout", "explorer").replace("central", "main"))


def config_from_dict(values: dict) -> TrainConfig:
    aliases = {public_config_name(name): name for name in TrainConfig.__dataclass_fields__}
    normalized = {}
    for name, value in values.items():
        field = aliases.get(name, name)
        if field in normalized:
            raise ValueError(f"configuration supplies both names for {field}")
        normalized[field] = value
    return TrainConfig(**normalized)


def parse_args() -> TrainConfig:
    import argparse
    p = argparse.ArgumentParser()
    # Simple direct mapping of TrainConfig fields → --kebab-case flags.
    cfg = TrainConfig()
    for fname, fval in vars(cfg).items():
        flag = "--" + fname.replace("_", "-")
        public_flag = "--" + public_config_name(fname).replace("_", "-")
        flags = [public_flag] if public_flag == flag else [public_flag, flag]
        kind = type(fval)
        if kind is list:
            kind = lambda s: [] if s.strip().lower() in ("", "auto") else [int(x) for x in s.split(",")]
        elif kind is bool:
            kind = lambda s: s.lower() in ("1", "true", "yes")
        p.add_argument(*flags, dest=fname, type=kind, default=fval)
    ns = p.parse_args()
    return TrainConfig(**vars(ns))
