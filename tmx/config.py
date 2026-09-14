import argparse
import itertools
import os
from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence, Tuple


WANDB_PROJECT = "two-model-explore"
WANDB_ENTITY: Optional[str] = os.environ.get("WANDB_ENTITY") or None  # None = the W&B default entity
WANDB_RUN_NAME = os.environ.get("WANDB_RUN_NAME") or None


@dataclass
class RunConfig:
    # Model + data
    model_name: str = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    dataset_name: str = "gsm8k"  # gsm8k | dapo_math_17k | deepscaler_preview | aime_2024 | aime_2025
    dataset_config_name: Optional[str] = None
    dataset_split_train: str = "train"
    dataset_split_eval: str = "test"
    dataset_split_policy: str = "auto"  # auto | native | derived_80_10_10
    dataset_split_seed: int = 42
    dataset_manifest_path: Optional[str] = None
    benchmark_dataset_name: Optional[str] = None  # gsm8k | dapo_math_17k | deepscaler_preview | aime_2024 | aime_2025
    benchmark_dataset_config_name: Optional[str] = None
    benchmark_dataset_split_eval: str = "test"
    benchmark_dataset_split_policy: str = "auto"  # auto | native | first_available
    benchmark_manifest_path: Optional[str] = None
    benchmark_eval_mode: str = "auto"  # auto | greedy | sampled
    benchmark_num_rollouts: int = 1
    benchmark_rollout_temperature: float = 1.0
    benchmark_rollout_top_p: float = 1.0
    benchmark_rollout_pass_k: Sequence[int] = field(default_factory=lambda: [1, 4, 8, 16, 32])
    benchmark_rollout_max_completion_len: Optional[int] = None
    benchmark_eval_batch_size: Optional[int] = None
    benchmark_rollout_output_path: Optional[str] = None
    topic_split_mode: str = "none"  # none | heuristic_math_topics
    topic_partition_policy: str = "explicit_or_heuristic"  # explicit_or_heuristic | largest_holdout
    train_selection_policy: str = "random"  # random | base_solvable
    train_probe_pool_size: Optional[int] = None
    train_probe_samples_per_prompt: int = 2
    train_probe_temperature: float = 0.8
    train_probe_batch_size: Optional[int] = None
    train_selection_manifest_path: Optional[str] = None
    prepare_train_selection_only: bool = False
    max_train_examples: int = 4
    max_eval_examples: int = 4
    max_benchmark_examples: Optional[int] = None
    eval_split: str = "test"  # test | train

    # GRPO (Explorer)
    grpo_max_steps: int = 20
    grpo_lr: float = 5e-6
    grpo_batch_size: int = 1          # per-device; global batch = batch_size * world_size * grad_accum
    grpo_grad_accum: int = 1
    grpo_loss_type: str = "dr_grpo"  # dr_grpo | grpo | dapo
    grpo_num_generations: int = 2
    grpo_generation_batch_size: int = 4  # divisible by global batch (4)

    # Novelty
    lambda_novelty: float = 0.1
    novelty_metric: str = "mse"      # mse | sqrt_mse | l2
    novelty_clip: Optional[float] = None
    novelty_feature_source: str = "multilayer"  # last | multilayer
    novelty_layers: Sequence[int] = field(default_factory=list)  # empty => auto
    novelty_layer_pool: str = "mean"  # mean | last_token
    novelty_layer_agg: str = "mean"   # mean | weighted
    novelty_text_contract: str = "full_raw"  # scored | first_line | full_raw
    novelty_feature_max_length: Optional[int] = 512
    novelty_zscore: str = "off"       # off | batch | ema
    correctness_zscore: str = "off"   # off | batch | ema
    rnd_hidden: int = 512
    rnd_lr: float = 1e-4
    novelty_study_tag: str = ""
    novelty_study_mode: str = "off"  # off | full

    # Distillation (Actual model)
    distill_batch_size: int = 2
    distill_epochs: int = 2
    distill_lr: float = 5e-5
    max_distill_examples: int = 256

    # Actual model (offline GRPO from Explorer trajectories)
    actual_train_mode: str = "grpo"  # grpo | distill | sft_grpo | sft_online_grpo
    actual_grpo_epochs: int = 2
    actual_grpo_lr: float = 5e-6
    actual_grpo_batch_size: int = 2
    actual_grpo_grad_accum: int = 1
    actual_online_grpo_max_steps: Optional[int] = None
    actual_grpo_kl_coef: float = 0.05
    actual_grpo_reward_scale: float = 1.0
    actual_grpo_reward_norm: bool = True
    actual_grpo_reward_clip: Optional[float] = None
    actual_grpo_adv_clip: Optional[float] = None
    max_actual_examples: Optional[int] = None
    actual_reward_source: str = "correctness"  # correctness | combined
    actual_target_correct_fraction: float = 0.0
    actual_target_contract: str = "full_trace"  # canonical | full_trace | compare
    actual_init_source: str = "base"  # base | baseline_grpo
    actual_min_accepted_trajectories: int = 8
    actual_dataset_policy: str = "accepted_balanced"  # accepted_balanced | accepted_only | correct_balanced | correct_only | all
    actual_max_incorrect_ratio: float = 1.0

    # Sequence lengths
    max_prompt_len: int = 192
    max_completion_len: int = 96
    exploration_max_completion_len: Optional[int] = None
    eval_max_completion_len: Optional[int] = None
    max_total_len: int = 288

    # Sampling
    temperature_schedule: Sequence[float] = field(default_factory=lambda: [1.0])
    num_samples_per_prompt: int = 2
    answer_constraint_mode: str = "auto"  # auto | none | integer | expression
    exploration_response_style: str = "reasoning_final"  # answer_only | reasoning_final
    eval_response_style: str = "answer_only"  # answer_only | reasoning_final
    exploration_answer_constraint_mode: str = "none"  # auto | none | integer | expression
    eval_answer_constraint_mode: str = "auto"  # auto | none | integer | expression
    exploration_thinking_mode: str = "auto"  # auto | think | no_think
    eval_thinking_mode: str = "auto"  # auto | think | no_think
    actual_thinking_mode: str = "think"  # auto | think | no_think

    # Logging + IO
    use_wandb: bool = True
    wandb_project: str = WANDB_PROJECT
    wandb_entity: str = WANDB_ENTITY
    wandb_run_name: Optional[str] = WANDB_RUN_NAME
    explorer_traj_path: str = "explorer_trajectories.jsonl"
    explorer_output_dir: Optional[str] = None
    plots_dir: str = "plots"
    base_eval_mode: str = "raw"  # raw | grpo
    pipeline_mode: str = "two_model"  # base_only | two_model | multi_round
    wall_clock_budget_minutes: Optional[float] = None
    accuracy_probe_every_steps: int = 0
    accuracy_probe_max_examples: int = 64
    eval_batch_size: int = 4
    trajectory_batch_size: int = 4
    save_base_model: bool = False
    save_actual_model: bool = True
    saved_model_root: Optional[str] = None
    log_model_artifact_to_wandb: bool = True
    promote_best_actual_model: bool = True

    # Misc
    seed: int = 42
    device_preference: str = "auto"  # auto | cuda | cpu | tpu
    decoupling_mode: str = "decoupled"  # decoupled | coupled | frozen
    traj_novelty_train: bool = False
    reward_correct: float = 1.0
    reward_incorrect: float = -1.0
    reward_shape: str = "linear"  # linear | piecewise
    incorrect_novelty_scale: float = 0.0
    reward_piecewise_bounds: Sequence[float] = field(default_factory=lambda: [-1.0, -0.5, 0.0, 0.5, 1.0])
    format_penalty_weight: float = 0.2
    gate_novelty_by_quality: bool = True
    id_ood_holdout_topics: Sequence[str] = field(default_factory=list)
    num_rounds: int = 4
    round_shard_policy: str = "topic_balanced"  # topic_balanced | sequential
    round_budget_mode: str = "constant_total"  # constant_total | per_round_full
    anchor_metric: str = "none"  # none | output_kl | param_l2
    anchor_weight: float = 0.0
    anchor_trainable_only: bool = True

    # External embedding-based novelty/diversity analysis
    embedding_novelty_eval: bool = False
    embedding_model_name: str = "Qwen/Qwen3-Embedding-4B"
    embedding_batch_size: int = 8
    embedding_max_texts: int = 768
    embedding_max_length: int = 256
    embedding_cluster_k: int = 8
    embedding_run_umap: bool = True
    embedding_run_tsne: bool = True
    embedding_permutation_iters: int = 1000

def parse_temperature_schedule(s: Sequence[float] | str) -> List[float]:
    if isinstance(s, str):
        parts = [p.strip() for p in s.split(",") if p.strip()]
        return [float(p) for p in parts] if parts else [1.0]
    return list(s)


def schedule_value(schedule: Sequence[float], fraction: float) -> float:
    """Linear interpolation over a list of floats."""
    if not schedule:
        return 1.0
    if len(schedule) == 1:
        return schedule[0]
    clamped = max(0.0, min(1.0, fraction))
    pos = clamped * (len(schedule) - 1)
    idx = int(pos)
    if idx >= len(schedule) - 1:
        return schedule[-1]
    weight = pos - idx
    return schedule[idx] * (1 - weight) + schedule[idx + 1] * weight


def parse_args():
    parser = argparse.ArgumentParser(description="Two-model exploration with GRPO + RND (TPU-friendly).")
    parser.add_argument("--model-name", type=str, help="HF model name.")
    parser.add_argument(
        "--dataset-name",
        choices=["gsm8k", "dapo_math_17k", "deepscaler_preview", "aime_2024", "aime_2025"],
        help="Dataset to use for training/eval.",
    )
    parser.add_argument("--dataset-config-name", type=str, help="Optional HF dataset config name.")
    parser.add_argument("--dataset-split-train", type=str, help="Dataset train split name.")
    parser.add_argument("--dataset-split-eval", type=str, help="Dataset eval split name.")
    parser.add_argument(
        "--dataset-split-policy",
        choices=["auto", "native", "derived_80_10_10"],
        help="How to choose train/eval splits when datasets may not ship an eval split.",
    )
    parser.add_argument("--dataset-split-seed", type=int, help="Seed used for derived dataset splits and round shards.")
    parser.add_argument("--dataset-manifest-path", type=str, help="Optional path for the resolved dataset manifest JSON.")
    parser.add_argument(
        "--benchmark-dataset-name",
        choices=["gsm8k", "dapo_math_17k", "deepscaler_preview", "aime_2024", "aime_2025"],
        help="Optional benchmark dataset to evaluate after training. Use this for train-on-one / eval-on-another workflows.",
    )
    parser.add_argument("--benchmark-dataset-config-name", type=str, help="Optional HF config name for the benchmark dataset.")
    parser.add_argument("--benchmark-dataset-split-eval", type=str, help="Benchmark dataset eval split name.")
    parser.add_argument(
        "--benchmark-dataset-split-policy",
        choices=["auto", "native", "first_available"],
        help="How to resolve the benchmark dataset eval split.",
    )
    parser.add_argument("--benchmark-manifest-path", type=str, help="Optional path for the resolved benchmark dataset manifest JSON.")
    parser.add_argument(
        "--benchmark-eval-mode",
        choices=["auto", "greedy", "sampled"],
        help="Benchmark eval mode. 'sampled' generates multiple stochastic rollouts per problem.",
    )
    parser.add_argument("--benchmark-num-rollouts", type=int, help="Number of benchmark generations per problem.")
    parser.add_argument("--benchmark-rollout-temperature", type=float, help="Sampling temperature for benchmark rollouts.")
    parser.add_argument("--benchmark-rollout-top-p", type=float, help="Sampling top-p for benchmark rollouts.")
    parser.add_argument(
        "--benchmark-rollout-pass-k",
        type=str,
        help="Comma-separated pass@k values to estimate from the benchmark rollout pool.",
    )
    parser.add_argument(
        "--benchmark-rollout-max-completion-len",
        type=int,
        help="Max completion tokens for sampled benchmark evaluation.",
    )
    parser.add_argument("--benchmark-eval-batch-size", type=int, help="Batch size for benchmark evaluation.")
    parser.add_argument("--benchmark-rollout-output-path", type=str, help="Optional path for benchmark rollout JSON outputs.")
    parser.add_argument(
        "--topic-split-mode",
        choices=["none", "heuristic_math_topics"],
        help="Optional topic tagging mode for novelty generalization analysis.",
    )
    parser.add_argument(
        "--topic-partition-policy",
        choices=["explicit_or_heuristic", "largest_holdout"],
        help="How to choose ID/OOD topic holdouts when none are supplied explicitly.",
    )
    parser.add_argument(
        "--train-selection-policy",
        choices=["random", "base_solvable"],
        help="How to choose the final training subset from the candidate pool.",
    )
    parser.add_argument("--train-probe-pool-size", type=int, help="Candidate-pool size used for train-selection probing.")
    parser.add_argument("--train-probe-samples-per-prompt", type=int, help="How many samples to try per prompt in base_solvable selection.")
    parser.add_argument("--train-probe-temperature", type=float, help="Sampling temperature used in base_solvable selection.")
    parser.add_argument("--train-probe-batch-size", type=int, help="Prompt batch size used during base_solvable probing.")
    parser.add_argument("--train-selection-manifest-path", type=str, help="Optional JSON path to load/save the probed train-selection manifest.")
    parser.add_argument("--prepare-train-selection-only", action="store_true", help="Prepare the train-selection manifest and exit before training.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "tpu"], default="auto")
    parser.add_argument("--decoupling-mode", choices=["decoupled", "coupled", "frozen"], help="Decoupling study mode.")
    parser.add_argument(
        "--pipeline-mode",
        choices=["base_only", "two_model", "multi_round"],
        help="Run only the baseline, the default two-model pipeline, or the reset-per-round multi-round pipeline.",
    )
    parser.add_argument("--max-train-examples", type=int, help="Number of GSM8K train examples.")
    parser.add_argument("--max-eval-examples", type=int, help="Number of GSM8K eval examples.")
    parser.add_argument("--max-benchmark-examples", type=int, help="Number of benchmark eval examples.")
    parser.add_argument(
        "--eval-split",
        choices=["test", "train"],
        help="Which GSM8K split to use for evaluation.",
    )
    parser.add_argument("--grpo-max-steps", type=int, help="GRPO max steps.")
    parser.add_argument("--grpo-lr", type=float, help="GRPO learning rate.")
    parser.add_argument("--grpo-batch-size", type=int, help="GRPO per-device batch size.")
    parser.add_argument("--grpo-grad-accum", type=int, help="GRPO gradient accumulation steps.")
    parser.add_argument("--grpo-loss-type", choices=["dr_grpo", "grpo", "dapo"], help="Loss: dr_grpo (default), grpo, dapo.")
    parser.add_argument("--loss-type-sweep", type=str, help="Comma-separated GRPO loss types to sweep.")
    parser.add_argument("--grpo-num-generations", type=int, help="Number of generations per prompt.")
    parser.add_argument("--grpo-generation-batch-size", type=int, help="Batch size during generation.")
    parser.add_argument("--lambda-novelty", type=float, help="Novelty weight.")
    parser.add_argument("--lambda-sweep", type=str, help="Comma-separated novelty weights to sweep sequentially.")
    parser.add_argument("--novelty-metric", choices=["mse", "sqrt_mse", "l2", "l1", "cosine"], help="Novelty bonus metric.")
    parser.add_argument("--novelty-metric-sweep", type=str, help="Comma-separated novelty metrics to sweep.")
    parser.add_argument("--novelty-clip", type=float, help="Clip novelty bonus at this max value.")
    parser.add_argument("--novelty-clip-sweep", type=str, help="Comma-separated novelty clip values to sweep (use 'none' to disable).")
    parser.add_argument(
        "--novelty-feature-source",
        choices=["last", "multilayer"],
        help="Novelty feature source: last hidden state or selected intermediate layers.",
    )
    parser.add_argument(
        "--novelty-layers",
        type=str,
        help="Comma-separated transformer layer indices (0-based) for multilayer novelty (empty => auto).",
    )
    parser.add_argument(
        "--novelty-layer-pool",
        choices=["mean", "last_token"],
        help="Pooling function over completion tokens for each selected layer.",
    )
    parser.add_argument(
        "--novelty-layer-agg",
        choices=["mean", "weighted"],
        help="How to combine per-layer novelty scores.",
    )
    parser.add_argument(
        "--novelty-text-contract",
        choices=["scored", "first_line", "full_raw"],
        help="Which normalized completion view to score for novelty.",
    )
    parser.add_argument(
        "--novelty-feature-max-length",
        type=int,
        help="Max token length used when re-encoding completion text for novelty features.",
    )
    parser.add_argument(
        "--novelty-zscore",
        choices=["off", "batch", "ema"],
        help="Normalization mode for novelty before reward shaping.",
    )
    parser.add_argument(
        "--correctness-zscore",
        choices=["off", "batch", "ema"],
        help="Normalization mode for correctness before reward shaping.",
    )
    parser.add_argument(
        "--rnd-hidden",
        type=int,
        help="Hidden width for the RND target/predictor MLPs.",
    )
    parser.add_argument(
        "--novelty-study-tag",
        type=str,
        help="Optional tag value for grouping novelty study runs.",
    )
    parser.add_argument(
        "--novelty-study-mode",
        choices=["off", "full"],
        help="Set to 'full' to enforce novelty-study defaults (linear reward, embedding eval).",
    )
    parser.add_argument("--novelty-feature-source-sweep", type=str, help="Comma-separated novelty feature sources to sweep.")
    parser.add_argument("--reward-shape-sweep", type=str, help="Comma-separated reward shapes to sweep.")
    parser.add_argument("--decoupling-sweep", type=str, help="Comma-separated decoupling modes to sweep.")
    parser.add_argument("--temperature-schedule", type=str, help="Comma-separated temps, linearly interpolated over training.")
    parser.add_argument("--num-samples-per-prompt", type=int, help="Explorer samples per prompt for trajectories.")
    parser.add_argument(
        "--answer-constraint-mode",
        choices=["auto", "none", "integer", "expression"],
        help="Optional generation-time answer constraint for trajectory/eval decoding.",
    )
    parser.add_argument(
        "--exploration-response-style",
        choices=["answer_only", "reasoning_final"],
        help="Prompt/decoding style for Explorer GRPO training and trajectory collection.",
    )
    parser.add_argument(
        "--eval-response-style",
        choices=["answer_only", "reasoning_final"],
        help="Prompt/decoding style for Base/Explorer/Actual evaluation.",
    )
    parser.add_argument(
        "--exploration-answer-constraint-mode",
        choices=["auto", "none", "integer", "expression"],
        help="Generation-time answer constraint for Explorer training and trajectory collection.",
    )
    parser.add_argument(
        "--eval-answer-constraint-mode",
        choices=["auto", "none", "integer", "expression"],
        help="Generation-time answer constraint for evaluation decoding.",
    )
    parser.add_argument(
        "--exploration-thinking-mode",
        choices=["auto", "think", "no_think"],
        help="Force Qwen-style /think or /no_think control for Explorer generation.",
    )
    parser.add_argument(
        "--eval-thinking-mode",
        choices=["auto", "think", "no_think"],
        help="Force Qwen-style /think or /no_think control for evaluation generation.",
    )
    parser.add_argument(
        "--actual-thinking-mode",
        choices=["auto", "think", "no_think"],
        help="Force Qwen-style /think or /no_think control for Actual model SFT training.",
    )
    parser.add_argument("--explorer-traj-path", type=str, help="Where to save Explorer trajectories jsonl.")
    parser.add_argument("--explorer-output-dir", type=str, help="Where to write Explorer GRPO outputs/checkpoints.")
    parser.add_argument("--plots-dir", type=str, help="Where to write matplotlib plots.")
    parser.add_argument(
        "--base-eval-mode",
        choices=["raw", "grpo"],
        help="Baseline evaluation model: raw base or GRPO-trained base.",
    )
    parser.add_argument("--wall-clock-budget-minutes", type=float, help="Optional target wall-clock budget for a run.")
    parser.add_argument("--accuracy-probe-every-steps", type=int, help="Optional periodic GRPO accuracy probe interval.")
    parser.add_argument("--accuracy-probe-max-examples", type=int, help="Max eval examples for periodic GRPO probes.")
    parser.add_argument("--eval-batch-size", type=int, help="Batch size for generation-based evaluation.")
    parser.add_argument("--trajectory-batch-size", type=int, help="Batch size for explorer trajectory generation.")
    parser.add_argument("--saved-model-root", type=str, help="Optional directory where the final Actual model bundle is written.")
    parser.add_argument("--save-base-model", action="store_true", help="Save the evaluated Base model bundle for later benchmark replay.")
    parser.add_argument("--no-save-actual-model", action="store_true", help="Disable saving the final Actual model bundle.")
    parser.add_argument("--no-log-model-artifact", action="store_true", help="Disable logging the saved Actual model bundle as a W&B artifact.")
    parser.add_argument("--no-promote-best-model", action="store_true", help="Disable promoted aliases for best-performing Actual model artifacts.")
    parser.add_argument("--max-prompt-len", type=int, help="Max prompt tokens.")
    parser.add_argument("--max-completion-len", type=int, help="Max completion tokens.")
    parser.add_argument("--exploration-max-completion-len", type=int, help="Max completion tokens for Explorer GRPO and trajectory collection.")
    parser.add_argument("--eval-max-completion-len", type=int, help="Max completion tokens for Base/Explorer/Actual evaluation.")
    parser.add_argument("--max-total-len", type=int, help="Max total tokens for distillation.")
    parser.add_argument("--distill-epochs", type=int, help="Distillation epochs.")
    parser.add_argument("--distill-lr", type=float, help="Distillation learning rate.")
    parser.add_argument("--distill-batch-size", type=int, help="Distillation batch size.")
    parser.add_argument("--max-distill-examples", type=int, help="Max distill examples.")
    parser.add_argument(
        "--actual-train-mode",
        choices=["grpo", "distill", "sft_grpo", "sft_online_grpo"],
        help="How to train the Actual model (offline grpo, distill, sft->offline grpo, or sft->online grpo on dataset prompts).",
    )
    parser.add_argument(
        "--actual-target-contract",
        choices=["canonical", "full_trace", "compare"],
        help="Which explorer target text to use for Actual training.",
    )
    parser.add_argument(
        "--actual-init-source",
        choices=["base", "baseline_grpo"],
        help="Initialize Actual from the raw base model or the GRPO-trained baseline model when available.",
    )
    parser.add_argument("--actual-grpo-epochs", type=int, help="Offline GRPO epochs for Actual.")
    parser.add_argument("--actual-grpo-lr", type=float, help="Offline GRPO learning rate for Actual.")
    parser.add_argument("--actual-grpo-batch-size", type=int, help="Offline GRPO batch size for Actual.")
    parser.add_argument("--actual-grpo-grad-accum", type=int, help="Offline GRPO grad accumulation for Actual.")
    parser.add_argument("--actual-online-grpo-max-steps", type=int, help="Optional explicit max-step budget for Actual online GRPO over dataset prompts.")
    parser.add_argument("--actual-grpo-kl-coef", type=float, help="KL coef to base model during Actual GRPO.")
    parser.add_argument("--actual-grpo-reward-scale", type=float, help="Scale factor for offline rewards.")
    parser.add_argument("--actual-grpo-reward-clip", type=float, help="Clip offline rewards at this max abs value.")
    parser.add_argument("--actual-grpo-adv-clip", type=float, help="Clip normalized advantages at this max abs value.")
    parser.add_argument(
        "--actual-reward-source",
        choices=["correctness", "combined"],
        help="Reward source for Actual offline GRPO: correctness-only or Explorer combined reward.",
    )
    parser.add_argument(
        "--actual-target-correct-fraction",
        type=float,
        help="If > 0, oversample correct trajectories so Actual's offline dataset reaches this fraction.",
    )
    parser.add_argument(
        "--actual-min-accepted-trajectories",
        type=int,
        help="Minimum accepted Explorer trajectories required before training the Actual model.",
    )
    parser.add_argument(
        "--actual-dataset-policy",
        choices=["accepted_balanced", "accepted_only", "correct_balanced", "correct_only", "all"],
        help="Which Explorer trajectory pool to use for Actual offline GRPO.",
    )
    parser.add_argument(
        "--actual-max-incorrect-ratio",
        type=float,
        help="For balanced Actual datasets, cap incorrect examples to this ratio times the positive pool size.",
    )
    parser.add_argument("--no-actual-reward-norm", action="store_true", help="Disable reward normalization for Actual GRPO.")
    parser.add_argument("--max-actual-examples", type=int, help="Cap trajectory examples for Actual GRPO (random subset).")
    parser.add_argument("--traj-novelty-train", action="store_true", help="Update RND predictor during trajectory collection.")
    parser.add_argument("--reward-correct", type=float, help="Reward for correct completion (default 1.0).")
    parser.add_argument("--reward-incorrect", type=float, help="Reward for incorrect completion (default 0.0).")
    parser.add_argument("--reward-shape", choices=["linear", "piecewise"], help="Reward shaping function.")
    parser.add_argument(
        "--incorrect-novelty-scale",
        type=float,
        help="Multiplier on novelty reward for incorrect completions in linear reward mode.",
    )
    parser.add_argument(
        "--reward-piecewise-bounds",
        type=str,
        help="Five comma-separated bounds for piecewise reward buckets (a,b,c,d,e).",
    )
    parser.add_argument("--format-penalty-weight", type=float, help="Penalty weight for malformed explorer completions.")
    parser.add_argument("--no-gate-novelty-by-quality", action="store_true", help="Disable quality-gating of novelty reward.")
    parser.add_argument("--num-rounds", type=int, help="Number of reset-per-round training rounds.")
    parser.add_argument(
        "--round-shard-policy",
        choices=["topic_balanced", "sequential"],
        help="How to shard the train split for multi-round execution.",
    )
    parser.add_argument(
        "--round-budget-mode",
        choices=["constant_total", "per_round_full"],
        help="Whether multi-round splits the total training budget across rounds or reuses the full budget every round.",
    )
    parser.add_argument(
        "--anchor-metric",
        choices=["none", "output_kl", "param_l2"],
        help="Optional anchor penalty that keeps Explorer close to the frozen Actual/Base model.",
    )
    parser.add_argument("--anchor-weight", type=float, help="Strength of the optional explorer anchor penalty.")
    parser.add_argument("--anchor-all-params", action="store_true", help="Apply parameter L2 anchor over all params, not just trainable ones.")
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument("--id-ood-holdout-topics", type=str, help="Comma-separated topic names to hold out from train and treat as OOD in eval.")
    parser.add_argument("--embedding-novelty-eval", action="store_true", help="Run embedding-based novelty/diversity analysis after eval.")
    parser.add_argument("--embedding-model-name", type=str, help="Embedding model name for external novelty analysis.")
    parser.add_argument("--embedding-batch-size", type=int, help="Batch size for embedding extraction.")
    parser.add_argument("--embedding-max-texts", type=int, help="Max eval samples per model used for embedding analysis.")
    parser.add_argument("--embedding-max-length", type=int, help="Max token length per completion for embedding analysis.")
    parser.add_argument("--embedding-cluster-k", type=int, help="Cluster count for embedding analysis.")
    parser.add_argument("--embedding-permutation-iters", type=int, help="Permutation-test iterations.")
    parser.add_argument("--no-embedding-umap", action="store_true", help="Disable UMAP projection plotting.")
    parser.add_argument("--no-embedding-tsne", action="store_true", help="Disable t-SNE projection plotting.")
    parser.add_argument("--wandb-run-name", type=str, help="Override W&B run name.")
    parser.add_argument("--wandb-project", type=str, help="Override W&B project.")
    parser.add_argument("--wandb-entity", type=str, help="Override W&B entity.")
    parser.add_argument("--use-wandb", action="store_true", help="Force-enable W&B even if default is False.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging.")
    return parser.parse_args()

def _parse_csv(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [p.strip() for p in s.split(",") if p.strip()]


def _parse_float_or_none(value: str) -> Optional[float]:
    if value.lower() in ("none", "null"):
        return None
    return float(value)


def _parse_int_csv(s: Optional[str]) -> List[int]:
    if not s:
        return []
    return [int(v) for v in _parse_csv(s)]


def _parse_positive_int_csv(s: Optional[str]) -> List[int]:
    return [value for value in _parse_int_csv(s) if value > 0]


def _build_sweep_suffix(items: List[Tuple[str, object]]) -> str:
    parts = []
    for key, value in items:
        if value is None:
            val = "none"
        else:
            val = str(value)
        if key == "lambda_novelty":
            parts.append(f"lambda{val}")
        elif key == "grpo_loss_type":
            parts.append(f"loss{val}")
        elif key == "novelty_metric":
            parts.append(f"metric{val}")
        elif key == "novelty_clip":
            parts.append(f"clip{val}")
        elif key == "decoupling_mode":
            parts.append(f"mode{val}")
        else:
            parts.append(f"{key}{val}")
    return "_".join(parts)


def build_configs_from_args(args) -> List[RunConfig]:
    base = RunConfig()

    if args.model_name:
        base.model_name = args.model_name
    if args.dataset_name:
        base.dataset_name = args.dataset_name
    if args.dataset_config_name:
        base.dataset_config_name = args.dataset_config_name
    if args.dataset_split_train:
        base.dataset_split_train = args.dataset_split_train
    if args.dataset_split_eval:
        base.dataset_split_eval = args.dataset_split_eval
    if args.dataset_split_policy:
        base.dataset_split_policy = args.dataset_split_policy
    if args.dataset_split_seed is not None:
        base.dataset_split_seed = args.dataset_split_seed
    if args.train_selection_policy is not None:
        base.train_selection_policy = args.train_selection_policy
    if args.train_probe_pool_size is not None:
        base.train_probe_pool_size = args.train_probe_pool_size
    if args.train_probe_samples_per_prompt is not None:
        base.train_probe_samples_per_prompt = args.train_probe_samples_per_prompt
    if args.train_probe_temperature is not None:
        base.train_probe_temperature = args.train_probe_temperature
    if args.train_selection_manifest_path:
        base.train_selection_manifest_path = args.train_selection_manifest_path
    if args.prepare_train_selection_only:
        base.prepare_train_selection_only = True
    if args.dataset_manifest_path:
        base.dataset_manifest_path = args.dataset_manifest_path
    if args.benchmark_dataset_name:
        base.benchmark_dataset_name = args.benchmark_dataset_name
    if args.benchmark_dataset_config_name:
        base.benchmark_dataset_config_name = args.benchmark_dataset_config_name
    if args.benchmark_dataset_split_eval:
        base.benchmark_dataset_split_eval = args.benchmark_dataset_split_eval
    if args.benchmark_dataset_split_policy:
        base.benchmark_dataset_split_policy = args.benchmark_dataset_split_policy
    if args.benchmark_manifest_path:
        base.benchmark_manifest_path = args.benchmark_manifest_path
    if args.benchmark_eval_mode is not None:
        base.benchmark_eval_mode = args.benchmark_eval_mode
    if args.benchmark_num_rollouts is not None:
        base.benchmark_num_rollouts = max(1, int(args.benchmark_num_rollouts))
    if args.benchmark_rollout_temperature is not None:
        base.benchmark_rollout_temperature = float(args.benchmark_rollout_temperature)
    if args.benchmark_rollout_top_p is not None:
        base.benchmark_rollout_top_p = float(args.benchmark_rollout_top_p)
    if args.benchmark_rollout_pass_k is not None:
        parsed_pass_k = _parse_positive_int_csv(args.benchmark_rollout_pass_k)
        if not parsed_pass_k:
            raise ValueError("--benchmark-rollout-pass-k must provide at least one positive integer.")
        base.benchmark_rollout_pass_k = parsed_pass_k
    if args.benchmark_rollout_max_completion_len is not None:
        base.benchmark_rollout_max_completion_len = int(args.benchmark_rollout_max_completion_len)
    if args.benchmark_eval_batch_size is not None:
        base.benchmark_eval_batch_size = int(args.benchmark_eval_batch_size)
    if args.benchmark_rollout_output_path:
        base.benchmark_rollout_output_path = args.benchmark_rollout_output_path
    if args.topic_split_mode:
        base.topic_split_mode = args.topic_split_mode
    if args.topic_partition_policy:
        base.topic_partition_policy = args.topic_partition_policy
    if args.pipeline_mode:
        base.pipeline_mode = args.pipeline_mode
    if args.max_train_examples is not None:
        base.max_train_examples = args.max_train_examples
    if args.max_eval_examples is not None:
        base.max_eval_examples = args.max_eval_examples
    if args.max_benchmark_examples is not None:
        base.max_benchmark_examples = args.max_benchmark_examples
    if args.eval_split:
        base.eval_split = args.eval_split
    if args.grpo_max_steps is not None:
        base.grpo_max_steps = args.grpo_max_steps
    if args.train_probe_batch_size is not None:
        base.train_probe_batch_size = args.train_probe_batch_size
    if args.grpo_lr is not None:
        base.grpo_lr = args.grpo_lr
    if args.grpo_batch_size is not None:
        base.grpo_batch_size = args.grpo_batch_size
    if args.grpo_grad_accum is not None:
        base.grpo_grad_accum = args.grpo_grad_accum
    if args.grpo_loss_type is not None:
        base.grpo_loss_type = args.grpo_loss_type
    if args.grpo_num_generations is not None:
        base.grpo_num_generations = args.grpo_num_generations
    if args.grpo_generation_batch_size is not None:
        base.grpo_generation_batch_size = args.grpo_generation_batch_size
    if args.lambda_novelty is not None:
        base.lambda_novelty = args.lambda_novelty
    if args.novelty_metric is not None:
        base.novelty_metric = args.novelty_metric
    if args.novelty_clip is not None:
        base.novelty_clip = args.novelty_clip
    if args.novelty_feature_source is not None:
        base.novelty_feature_source = args.novelty_feature_source
    if args.novelty_layers is not None:
        base.novelty_layers = _parse_int_csv(args.novelty_layers)
    if args.novelty_layer_pool is not None:
        base.novelty_layer_pool = args.novelty_layer_pool
    if args.novelty_layer_agg is not None:
        base.novelty_layer_agg = args.novelty_layer_agg
    if args.novelty_text_contract is not None:
        base.novelty_text_contract = args.novelty_text_contract
    if args.novelty_feature_max_length is not None:
        base.novelty_feature_max_length = args.novelty_feature_max_length
    if args.novelty_zscore is not None:
        base.novelty_zscore = args.novelty_zscore
    if args.correctness_zscore is not None:
        base.correctness_zscore = args.correctness_zscore
    if args.rnd_hidden is not None:
        base.rnd_hidden = args.rnd_hidden
    if args.novelty_study_tag is not None:
        base.novelty_study_tag = args.novelty_study_tag
    if args.novelty_study_mode is not None:
        base.novelty_study_mode = args.novelty_study_mode
    if args.decoupling_mode is not None:
        base.decoupling_mode = args.decoupling_mode
    if args.temperature_schedule:
        base.temperature_schedule = parse_temperature_schedule(args.temperature_schedule)
    if args.num_samples_per_prompt is not None:
        base.num_samples_per_prompt = args.num_samples_per_prompt
    if args.answer_constraint_mode is not None:
        base.answer_constraint_mode = args.answer_constraint_mode
    if args.exploration_response_style is not None:
        base.exploration_response_style = args.exploration_response_style
    if args.eval_response_style is not None:
        base.eval_response_style = args.eval_response_style
    if args.exploration_answer_constraint_mode is not None:
        base.exploration_answer_constraint_mode = args.exploration_answer_constraint_mode
    if args.eval_answer_constraint_mode is not None:
        base.eval_answer_constraint_mode = args.eval_answer_constraint_mode
    if args.exploration_thinking_mode is not None:
        base.exploration_thinking_mode = args.exploration_thinking_mode
    if args.eval_thinking_mode is not None:
        base.eval_thinking_mode = args.eval_thinking_mode
    if getattr(args, "actual_thinking_mode", None) is not None:
        base.actual_thinking_mode = args.actual_thinking_mode
    if args.explorer_traj_path:
        base.explorer_traj_path = args.explorer_traj_path
    if args.explorer_output_dir:
        base.explorer_output_dir = args.explorer_output_dir
    if args.plots_dir:
        base.plots_dir = args.plots_dir
    if args.base_eval_mode:
        base.base_eval_mode = args.base_eval_mode
    if args.wall_clock_budget_minutes is not None:
        base.wall_clock_budget_minutes = args.wall_clock_budget_minutes
    if args.accuracy_probe_every_steps is not None:
        base.accuracy_probe_every_steps = args.accuracy_probe_every_steps
    if args.accuracy_probe_max_examples is not None:
        base.accuracy_probe_max_examples = args.accuracy_probe_max_examples
    if args.eval_batch_size is not None:
        base.eval_batch_size = args.eval_batch_size
    if args.trajectory_batch_size is not None:
        base.trajectory_batch_size = args.trajectory_batch_size
    if args.saved_model_root:
        base.saved_model_root = args.saved_model_root
    if args.save_base_model:
        base.save_base_model = True
    if args.no_save_actual_model:
        base.save_actual_model = False
    if args.no_log_model_artifact:
        base.log_model_artifact_to_wandb = False
    if args.no_promote_best_model:
        base.promote_best_actual_model = False
    if args.max_prompt_len is not None:
        base.max_prompt_len = args.max_prompt_len
    if args.max_completion_len is not None:
        base.max_completion_len = args.max_completion_len
    if args.exploration_max_completion_len is not None:
        base.exploration_max_completion_len = args.exploration_max_completion_len
    if args.eval_max_completion_len is not None:
        base.eval_max_completion_len = args.eval_max_completion_len
    if args.max_total_len is not None:
        base.max_total_len = args.max_total_len
    if args.distill_epochs is not None:
        base.distill_epochs = args.distill_epochs
    if args.distill_lr is not None:
        base.distill_lr = args.distill_lr
    if args.distill_batch_size is not None:
        base.distill_batch_size = args.distill_batch_size
    if args.max_distill_examples is not None:
        base.max_distill_examples = args.max_distill_examples
    if args.actual_train_mode is not None:
        base.actual_train_mode = args.actual_train_mode
    if args.actual_target_contract is not None:
        base.actual_target_contract = args.actual_target_contract
    if args.actual_init_source is not None:
        base.actual_init_source = args.actual_init_source
    if args.actual_grpo_epochs is not None:
        base.actual_grpo_epochs = args.actual_grpo_epochs
    if args.actual_grpo_lr is not None:
        base.actual_grpo_lr = args.actual_grpo_lr
    if args.actual_grpo_batch_size is not None:
        base.actual_grpo_batch_size = args.actual_grpo_batch_size
    if args.actual_grpo_grad_accum is not None:
        base.actual_grpo_grad_accum = args.actual_grpo_grad_accum
    if args.actual_online_grpo_max_steps is not None:
        base.actual_online_grpo_max_steps = args.actual_online_grpo_max_steps
    if args.actual_grpo_kl_coef is not None:
        base.actual_grpo_kl_coef = args.actual_grpo_kl_coef
    if args.actual_grpo_reward_scale is not None:
        base.actual_grpo_reward_scale = args.actual_grpo_reward_scale
    if args.actual_grpo_reward_clip is not None:
        base.actual_grpo_reward_clip = args.actual_grpo_reward_clip
    if args.actual_grpo_adv_clip is not None:
        base.actual_grpo_adv_clip = args.actual_grpo_adv_clip
    if args.actual_reward_source is not None:
        base.actual_reward_source = args.actual_reward_source
    if args.actual_target_correct_fraction is not None:
        base.actual_target_correct_fraction = args.actual_target_correct_fraction
    if args.actual_min_accepted_trajectories is not None:
        base.actual_min_accepted_trajectories = args.actual_min_accepted_trajectories
    if args.actual_dataset_policy is not None:
        base.actual_dataset_policy = args.actual_dataset_policy
    if args.actual_max_incorrect_ratio is not None:
        base.actual_max_incorrect_ratio = args.actual_max_incorrect_ratio
    if args.no_actual_reward_norm:
        base.actual_grpo_reward_norm = False
    if args.max_actual_examples is not None:
        base.max_actual_examples = args.max_actual_examples
    if args.traj_novelty_train:
        base.traj_novelty_train = True
    if args.reward_correct is not None:
        base.reward_correct = args.reward_correct
    if args.reward_incorrect is not None:
        base.reward_incorrect = args.reward_incorrect
    if args.reward_shape is not None:
        base.reward_shape = args.reward_shape
    if args.incorrect_novelty_scale is not None:
        base.incorrect_novelty_scale = args.incorrect_novelty_scale
    if args.reward_piecewise_bounds is not None:
        bounds = [float(v) for v in _parse_csv(args.reward_piecewise_bounds)]
        if len(bounds) != 5:
            raise ValueError("--reward-piecewise-bounds must provide exactly 5 comma-separated numbers.")
        base.reward_piecewise_bounds = bounds
    if args.format_penalty_weight is not None:
        base.format_penalty_weight = args.format_penalty_weight
    if args.no_gate_novelty_by_quality:
        base.gate_novelty_by_quality = False
    if args.num_rounds is not None:
        base.num_rounds = args.num_rounds
    if args.round_shard_policy is not None:
        base.round_shard_policy = args.round_shard_policy
    if args.round_budget_mode is not None:
        base.round_budget_mode = args.round_budget_mode
    if args.anchor_metric is not None:
        base.anchor_metric = args.anchor_metric
    if args.anchor_weight is not None:
        base.anchor_weight = args.anchor_weight
    if args.anchor_all_params:
        base.anchor_trainable_only = False
    if args.seed is not None:
        base.seed = args.seed
    if args.id_ood_holdout_topics is not None:
        base.id_ood_holdout_topics = _parse_csv(args.id_ood_holdout_topics)
    if args.embedding_novelty_eval:
        base.embedding_novelty_eval = True
    if args.embedding_model_name:
        base.embedding_model_name = args.embedding_model_name
    if args.embedding_batch_size is not None:
        base.embedding_batch_size = args.embedding_batch_size
    if args.embedding_max_texts is not None:
        base.embedding_max_texts = args.embedding_max_texts
    if args.embedding_max_length is not None:
        base.embedding_max_length = args.embedding_max_length
    if args.embedding_cluster_k is not None:
        base.embedding_cluster_k = args.embedding_cluster_k
    if args.embedding_permutation_iters is not None:
        base.embedding_permutation_iters = args.embedding_permutation_iters
    if args.no_embedding_umap:
        base.embedding_run_umap = False
    if args.no_embedding_tsne:
        base.embedding_run_tsne = False
    if args.wandb_run_name:
        base.wandb_run_name = args.wandb_run_name
    if args.wandb_project:
        base.wandb_project = args.wandb_project
    if args.wandb_entity:
        base.wandb_entity = args.wandb_entity

    if args.no_wandb:
        base.use_wandb = False
    if args.use_wandb:
        base.use_wandb = True

    if base.id_ood_holdout_topics and base.topic_split_mode == "none":
        base.topic_split_mode = "heuristic_math_topics"

    if base.novelty_study_mode == "full":
        base.reward_shape = "linear"
        base.embedding_novelty_eval = True
        if base.topic_split_mode == "none":
            base.topic_split_mode = "heuristic_math_topics"

    base.device_preference = args.device

    sweeps = {}
    if args.lambda_sweep:
        sweeps["lambda_novelty"] = [float(v) for v in _parse_csv(args.lambda_sweep)]
    if args.loss_type_sweep:
        sweeps["grpo_loss_type"] = _parse_csv(args.loss_type_sweep)
    if args.novelty_metric_sweep:
        sweeps["novelty_metric"] = _parse_csv(args.novelty_metric_sweep)
    if args.novelty_clip_sweep:
        sweeps["novelty_clip"] = [_parse_float_or_none(v) for v in _parse_csv(args.novelty_clip_sweep)]
    if args.novelty_feature_source_sweep:
        sweeps["novelty_feature_source"] = _parse_csv(args.novelty_feature_source_sweep)
    if args.reward_shape_sweep:
        sweeps["reward_shape"] = _parse_csv(args.reward_shape_sweep)
    if args.decoupling_sweep:
        sweeps["decoupling_mode"] = _parse_csv(args.decoupling_sweep)

    if sweeps:
        keys = list(sweeps.keys())
        value_lists = [sweeps[k] for k in keys]
        cfgs = []
        for combo in itertools.product(*value_lists):
            updates = dict(zip(keys, combo, strict=True))
            run_name = base.wandb_run_name
            if run_name:
                run_name = f"{run_name}_{_build_sweep_suffix(list(updates.items()))}"
            cfgs.append(replace(base, **updates, wandb_run_name=run_name))
    else:
        cfgs = [base]

    expanded_cfgs = []
    for cfg in cfgs:
        if cfg.actual_target_contract != "compare":
            expanded_cfgs.append(cfg)
            continue
        for target_contract in ("canonical", "full_trace"):
            run_name = cfg.wandb_run_name
            if run_name:
                run_name = f"{run_name}_target{target_contract}"
            expanded_cfgs.append(
                replace(
                    cfg,
                    actual_target_contract=target_contract,
                    wandb_run_name=run_name,
                )
            )

    return expanded_cfgs
