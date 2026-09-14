import os
from typing import Any, Dict


def _env_flag(name: str, default: str = "0") -> bool:
    value = str(os.environ.get(name, default) or default).strip().lower()
    return value not in ("", "0", "false", "no", "off")


def wandb_minimal_mode() -> bool:
    return _env_flag("EXPDIS_WANDB_MINIMAL", "1")


def wandb_verbose_mode() -> bool:
    return _env_flag("EXPDIS_WANDB_VERBOSE", "0")


def _is_media_key(key: str) -> bool:
    if key.startswith("plots/"):
        return True
    for suffix in ("_hist", "_table", "_bar", "_scatter", "_image"):
        if key.endswith(suffix):
            return True
    return False


def _allow_train_metric(name: str) -> bool:
    return name in {
        "step",
        "loss",
        "reward",
        "grad_norm",
        "learning_rate",
        "wall_time_seconds",
        "tokens_per_sec",
    }


def _allow_grpo_metric(name: str) -> bool:
    return name in {
        "step",
        "loss",
        "reward",
        "reward_std",
        "kl",
        "entropy",
        "learning_rate",
        "tokens_per_sec",
        "health_gate_triggered",
        "health_gate_reason/degenerate_terminal",
        "health_gate_terminal_step",
        "health_gate_terminal_loss",
        "health_gate_terminal_mean_length",
        "health_gate_terminal_clipped_ratio",
        "health_gate_terminal_reward",
    }


def _allow_grpo_reward_metric(name: str) -> bool:
    return name in {
        "correctness_used_mean",
        "novelty_used_mean",
        "novelty_reward_mean",
        "final_mean",
        "format_quality_mean",
        "clipped_fraction",
        "terminated_fraction",
        "completion_token_length_mean",
        "novelty_feature_truncated_fraction",
        "novelty_feature_mean_input_tokens",
        "terminated_final_mean",
        "terminated_correctness_mean",
        "terminated_novelty_mean",
        "terminated_count",
    }


def _allow_traj_metric(name: str) -> bool:
    return name in {
        "count",
        "num_trajectories",
        "accepted_count",
        "accepted_fraction",
        "correct_fraction",
        "reward_mean",
        "mean_reward",
        "novelty_mean",
        "mean_novelty",
        "format_quality_mean",
        "terminated_fraction",
        "clipped_fraction",
        "valid_answer_fraction",
    }


def _allow_distill_metric(name: str) -> bool:
    return name in {
        "step",
        "epoch",
        "batch_loss",
        "loss",
        "wall_time_seconds",
    }


def _allow_actual_grpo_metric(name: str) -> bool:
    return name in {
        "step",
        "epoch",
        "loss",
        "reward",
        "kl",
        "entropy",
        "grad_norm",
        "learning_rate",
        "tokens_per_sec",
        "loss_pg",
        "loss_kl",
        "reward_mean",
        "reward_std",
        "reward_norm_active",
        "kl_mean",
        "mean_logp",
        "mean_len",
        "batch_correct_frac",
        "batch_r_correct",
        "batch_r_novel",
        "wall_time_seconds",
    }


def _allow_eval_metric(key: str) -> bool:
    suffix = key.split("/", 1)[1]
    if "_pass@" in suffix or "_avg@" in suffix:
        return True
    allowed_suffixes = (
        "_accuracy",
        "_valid_answer_fraction",
        "_clipped_fraction",
        "_mean_latency_seconds",
        "_problem_count",
        "_rollout_count",
    )
    return any(suffix.endswith(item) for item in allowed_suffixes)


def _allow_actual_metric(name: str) -> bool:
    return name in {
        "skipped",
        "skip_accepted_count",
        "skip_positive_count",
        "policy_correct_available",
        "policy_high_quality_incorrect_available",
        "policy_incorrect_available",
        "policy_used_fallback",
        "init_from_baseline_grpo",
    }


def _allow_final_metric(name: str) -> bool:
    return name in {
        "base_accuracy",
        "explorer_accuracy",
        "actual_accuracy",
        "base_valid_answer_fraction",
        "explorer_valid_answer_fraction",
        "actual_valid_answer_fraction",
    }


def filter_wandb_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not payload:
        return {}

    if not wandb_minimal_mode():
        if wandb_verbose_mode():
            return dict(payload)
        return {k: v for k, v in payload.items() if not _is_media_key(k)}

    filtered: Dict[str, Any] = {}
    for key, value in payload.items():
        if _is_media_key(key):
            continue
        if key.startswith("train/"):
            if _allow_train_metric(key.split("/", 1)[1]):
                filtered[key] = value
            continue
        if key.startswith("grpo/reward/"):
            if _allow_grpo_reward_metric(key.split("/", 2)[2]):
                filtered[key] = value
            continue
        if key.startswith("actual_grpo/reward/"):
            if _allow_grpo_reward_metric(key.split("/", 2)[2]):
                filtered[key] = value
            continue
        if key.startswith("grpo/"):
            suffix = key.split("/", 1)[1]
            if _allow_grpo_metric(suffix) or suffix.startswith("diversity/"):
                filtered[key] = value
            continue
        if key.startswith("traj/"):
            if _allow_traj_metric(key.split("/", 1)[1]):
                filtered[key] = value
            continue
        if key.startswith("distill/"):
            if _allow_distill_metric(key.split("/", 1)[1]):
                filtered[key] = value
            continue
        if key.startswith("actual_grpo/"):
            suffix = key.split("/", 1)[1]
            if suffix.startswith("diversity/") or _allow_actual_grpo_metric(suffix):
                filtered[key] = value
            continue
        if key.startswith("eval/"):
            if _allow_eval_metric(key):
                filtered[key] = value
            continue
        if key.startswith("actual/"):
            if _allow_actual_metric(key.split("/", 1)[1]):
                filtered[key] = value
            continue
        if key.startswith("final/"):
            if _allow_final_metric(key.split("/", 1)[1]):
                filtered[key] = value
            continue
        if key.startswith("anchor/") and key.endswith("param_l2"):
            filtered[key] = value
            continue
    return filtered
