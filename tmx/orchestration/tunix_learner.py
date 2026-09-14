"""Tunix learner adapter for TMX DAPO + Dr.GRPO + RND novelty.

The important integration point is the learner, not the rollout object.
Tunix's ``GrpoLearner`` computes rewards after it receives ``RolloutOutput``
from ``rl_cluster.generate(...)``. The reward manager only sees
``training_input`` kwargs, so arbitrary attributes attached to a vLLM rollout
output are not forwarded to reward functions. This module intercepts the
generated ``RolloutOutput`` and injects a per-sample ``novelty`` kwarg before
reward computation.

The rollout engine remains Tunix/vLLM-on-TPU, so generation still uses the
production KV-cache path. This adapter only adds the TMX novelty reward path
and swaps the algorithm config to DAPO with Dr.GRPO advantages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .spec import AlgorithmSpec, ExperimentSpec


@dataclass(frozen=True)
class TunixNoveltyConfig:
    lambda_novelty: float
    incorrect_novelty_scale: float
    layers: tuple[int, ...] = (7, 14, 21)
    metric: str = "sqrt_mse"
    required: bool = True


class RolloutNoveltyScorer(Protocol):
    def score_rollout(
        self,
        rollout_output: Any,
        *,
        prompts: list[str],
        completions: list[str],
    ) -> list[float]:
        """Return one novelty score per completion."""


def novelty_config_from_algorithm(alg: AlgorithmSpec) -> TunixNoveltyConfig:
    layers = tuple(int(x.strip()) for x in str(alg.novelty_layers).split(",") if x.strip())
    return TunixNoveltyConfig(
        lambda_novelty=float(alg.lambda_novelty),
        incorrect_novelty_scale=float(alg.incorrect_novelty_scale),
        layers=layers,
        metric=str(alg.novelty_metric),
        required=float(alg.lambda_novelty) > 0.0,
    )


def build_dapo_drgrpo_config(spec: ExperimentSpec) -> Any:
    """Build Tunix DAPOConfig with Dr.GRPO advantage estimator.

    Tunix separates DAPO and Dr.GRPO in two modules. Dr.GRPO is registered by
    importing ``drgrpo_learner``; DAPOConfig then points
    ``advantage_estimator`` at that registered estimator while preserving
    DAPO's clip-higher, dynamic sampling, and soft-overlong reward shaping.
    """
    # Import for side effect: registers the "drgrpo" advantage estimator.
    from tunix.rl.grpo import drgrpo_learner as _drgrpo_learner  # noqa: F401
    from tunix.rl.grpo.dapo_learner import DAPOConfig

    alg = spec.algorithm
    return DAPOConfig(
        advantage_estimator="drgrpo",
        loss_agg_mode="sequence-mean-token-scale",
        num_generations=int(alg.grpo_num_generations),
        num_iterations=1,
        beta=None,
        epsilon=float(alg.grpo_clip_epsilon),
        epsilon_high=float(alg.grpo_clip_epsilon_high),
        dynamic_sampling=bool(alg.dynamic_sampling),
        overlong_buffer={
            "enable": True,
            "overlong_buffer_length": int(alg.soft_overlong_cache_len),
            "overlong_buffer_penalty": 1.0,
            "max_response_length": int(alg.max_completion_len),
        },
    )


class HiddenStateNoveltyScorer:
    """RND scorer placeholder for the real hidden-state path.

    This class deliberately fails if no actor feature extractor is provided.
    That is safer than silently switching novelty to a token-hash proxy: the
    paper run requires the original TMX hidden-state RND contract.
    """

    def __init__(
        self,
        *,
        feature_extractor: Any | None,
        novelty_state: Any | None,
        config: TunixNoveltyConfig,
    ) -> None:
        self.feature_extractor = feature_extractor
        self.novelty_state = novelty_state
        self.config = config

    def score_rollout(
        self,
        rollout_output: Any,
        *,
        prompts: list[str],
        completions: list[str],
    ) -> list[float]:
        if self.config.lambda_novelty <= 0.0:
            return [0.0] * len(completions)
        if self.feature_extractor is None or self.novelty_state is None:
            raise RuntimeError(
                "TMX Tunix novelty requires hidden-state feature extraction from "
                "the live actor model. The MaxText/Tunix adapter did not expose "
                "the requested layers; patch the actor adapter or disable novelty "
                "only for a correctness baseline."
            )
        features = self.feature_extractor(rollout_output)
        from .novelty_state import score_and_update

        self.novelty_state, novelty = score_and_update(self.novelty_state, features)
        return [float(x) for x in np.asarray(novelty)]


def _zero_novelty(count: int) -> list[float]:
    return [0.0] * int(count)


def build_tmx_dapo_learner_class(base_cls: Any | None = None) -> type:
    """Return a DAPOLearner subclass that injects novelty into rewards."""
    if base_cls is None:
        from tunix.rl.grpo.dapo_learner import DAPOLearner as base_cls

    class TMXNoveltyDAPOLearner(base_cls):  # type: ignore[misc, valid-type]
        """DAPO learner with Dr.GRPO config and TMX novelty reward injection."""

        def __init__(
            self,
            *args: Any,
            novelty_scorer: RolloutNoveltyScorer | None = None,
            novelty_config: TunixNoveltyConfig | None = None,
            **kwargs: Any,
        ) -> None:
            super().__init__(*args, **kwargs)
            self._tmx_novelty_scorer = novelty_scorer
            self._tmx_novelty_config = novelty_config or TunixNoveltyConfig(
                lambda_novelty=0.0,
                incorrect_novelty_scale=0.0,
                required=False,
            )

        def _generate_and_compute_advantage(self, training_input: Any, mode: Any = None) -> Any:
            captured: dict[str, Any] = {}
            original_generate = self.rl_cluster.generate
            original_compute_rewards = self._compute_rewards

            def generate_and_capture(*args: Any, **kwargs: Any) -> Any:
                output = original_generate(*args, **kwargs)
                captured["rollout_output"] = output
                return output

            def compute_rewards_with_novelty(
                *,
                prompts: list[str],
                completions: list[str],
                mode: Any,
                **kwargs: Any,
            ) -> Any:
                if self._tmx_novelty_config.lambda_novelty > 0.0:
                    scorer = self._tmx_novelty_scorer
                    if scorer is None:
                        if self._tmx_novelty_config.required:
                            raise RuntimeError("TMX novelty scorer is required but missing")
                        kwargs["novelty"] = _zero_novelty(len(completions))
                    else:
                        rollout_output = captured.get("rollout_output")
                        if rollout_output is None:
                            raise RuntimeError(
                                "Tunix generated completions but no RolloutOutput was captured"
                            )
                        kwargs["novelty"] = scorer.score_rollout(
                            rollout_output,
                            prompts=list(prompts),
                            completions=list(completions),
                        )
                return original_compute_rewards(
                    prompts=prompts,
                    completions=completions,
                    mode=mode,
                    **kwargs,
                )

            self.rl_cluster.generate = generate_and_capture
            self._compute_rewards = compute_rewards_with_novelty
            try:
                return super()._generate_and_compute_advantage(training_input, mode)
            finally:
                self.rl_cluster.generate = original_generate
                self._compute_rewards = original_compute_rewards

    return TMXNoveltyDAPOLearner


__all__ = [
    "HiddenStateNoveltyScorer",
    "RolloutNoveltyScorer",
    "TunixNoveltyConfig",
    "build_dapo_drgrpo_config",
    "build_tmx_dapo_learner_class",
    "novelty_config_from_algorithm",
]
