from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ContractResult:
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


class AlgorithmLike(Protocol):
    phase: str
    max_prompt_len: int
    max_completion_len: int
    max_total_len: int
    grpo_num_generations: int
    grpo_clip_epsilon: float
    grpo_clip_epsilon_high: float
    grpo_kl_beta: float
    grpo_advantage_normalization: str
    dynamic_sampling: bool
    dynamic_sampling_accept_reward_variance: bool
    reward_correct: float
    reward_incorrect: float
    format_penalty_weight: float
    apply_format_penalty: bool
    lambda_novelty: float
    incorrect_novelty_scale: float
    soft_overlong_expected_len: int
    soft_overlong_cache_len: int


DAPO_CLIP_EPSILON = 0.2
DAPO_CLIP_EPSILON_HIGH = 0.28
DAPO_KL_BETA = 0.0
DR_GRPO_ADVANTAGE_NORMALIZATION = "none"


def _is_rl_phase(phase: str) -> bool:
    return phase.startswith("explorer") or phase == "actual_grpo"


def _is_novelty_phase(alg: AlgorithmLike) -> bool:
    return alg.phase.startswith("explorer") and float(alg.lambda_novelty) > 0.0


def validate_dapo_drgrpo_contract(alg: AlgorithmLike) -> ContractResult:
    """Validate the production RL contract before launch.

    This is intentionally strict for RL phases. SFT/eval stages can share the
    same config schema without requiring dynamic sampling or optimizer settings.
    """
    errors: list[str] = []
    warnings: list[str] = []
    phase = str(alg.phase)

    if int(alg.max_total_len) != int(alg.max_prompt_len) + int(alg.max_completion_len):
        errors.append("max_total_len must equal max_prompt_len + max_completion_len")

    if not _is_rl_phase(phase):
        return ContractResult(errors=tuple(errors), warnings=tuple(warnings))

    if int(alg.grpo_num_generations) < 2:
        errors.append("GRPO/DAPO requires grouped rollouts; grpo_num_generations must be >= 2")
    if abs(float(alg.grpo_clip_epsilon) - DAPO_CLIP_EPSILON) > 1e-9:
        errors.append(f"DAPO clip-low must be {DAPO_CLIP_EPSILON}")
    if abs(float(alg.grpo_clip_epsilon_high) - DAPO_CLIP_EPSILON_HIGH) > 1e-9:
        errors.append(f"DAPO clip-high must be {DAPO_CLIP_EPSILON_HIGH}")
    if float(alg.grpo_clip_epsilon_high) <= float(alg.grpo_clip_epsilon):
        errors.append("DAPO clip-higher requires grpo_clip_epsilon_high > grpo_clip_epsilon")
    if abs(float(alg.grpo_kl_beta) - DAPO_KL_BETA) > 1e-12:
        errors.append("DAPO production contract uses KL beta = 0")
    if str(alg.grpo_advantage_normalization).lower() != DR_GRPO_ADVANTAGE_NORMALIZATION:
        errors.append("Dr.GRPO production contract requires advantage_normalization='none'")
    if not bool(alg.dynamic_sampling):
        errors.append("DAPO production RL phases require dynamic_sampling=true")
    if int(alg.soft_overlong_expected_len) + int(alg.soft_overlong_cache_len) != int(alg.max_completion_len):
        errors.append("soft_overlong_expected_len + soft_overlong_cache_len must equal max_completion_len")
    if int(alg.soft_overlong_expected_len) <= int(alg.soft_overlong_cache_len):
        warnings.append("soft-overlong expected length is not larger than cache length")
    if float(alg.reward_correct) <= float(alg.reward_incorrect):
        errors.append("reward_correct must be greater than reward_incorrect")
    if bool(alg.apply_format_penalty) or abs(float(alg.format_penalty_weight)) > 1e-12:
        errors.append("production DAPO/Dr.GRPO contract logs format quality but does not apply a static format penalty")

    if _is_novelty_phase(alg):
        if not bool(alg.dynamic_sampling_accept_reward_variance):
            errors.append("novelty Explorer requires reward-variance dynamic sampling acceptance")
        if float(alg.incorrect_novelty_scale) <= 0.0:
            errors.append("novelty Explorer requires incorrect_novelty_scale > 0")
        if float(alg.reward_incorrect) >= 0.0:
            errors.append("novelty Explorer requires negative reward_incorrect so wrong-but-novel completions are not trivially positive")
    else:
        if float(alg.lambda_novelty) == 0.0 and float(alg.incorrect_novelty_scale) != 0.0:
            warnings.append("incorrect_novelty_scale is ignored when lambda_novelty=0")

    return ContractResult(errors=tuple(errors), warnings=tuple(warnings))
