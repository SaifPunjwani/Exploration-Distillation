from __future__ import annotations

from types import SimpleNamespace

import pytest

from tmx.orchestration.spec import load_experiment_spec
from tmx.orchestration.tunix_learner import (
    TunixNoveltyConfig,
    build_tmx_dapo_learner_class,
    novelty_config_from_algorithm,
)


class _FakeBaseLearner:
    def __init__(self, *, rl_cluster, reward_fns, algo_config) -> None:
        self.rl_cluster = rl_cluster
        self.reward_fns = reward_fns
        self.algo_config = algo_config
        self.seen_rewards_kwargs = None

    def _compute_rewards(self, *, prompts, completions, mode, **kwargs):
        self.seen_rewards_kwargs = kwargs
        return [0.0] * len(completions)

    def _generate_and_compute_advantage(self, training_input, mode=None):
        out = self.rl_cluster.generate(prompts=list(training_input["prompts"]), mode=mode)
        return self._compute_rewards(
            prompts=list(training_input["prompts"]),
            completions=out.text,
            mode=mode,
            answer=training_input["answer"],
        )


class _FakeCluster:
    def __init__(self) -> None:
        self.generated = SimpleNamespace(
            text=[r"\boxed{1}", r"\boxed{2}"],
            tokens=[[11, 12], [21, 22]],
            left_padded_prompt_tokens=[[0, 1], [0, 2]],
        )

    def generate(self, **_kwargs):
        return self.generated


class _FakeScorer:
    def __init__(self) -> None:
        self.seen_rollout = None

    def score_rollout(self, rollout_output, *, prompts, completions):
        self.seen_rollout = rollout_output
        assert prompts == ["p1", "p2"]
        assert completions == [r"\boxed{1}", r"\boxed{2}"]
        return [0.25, 0.75]


def test_learner_injects_novelty_from_captured_rollout() -> None:
    cls = build_tmx_dapo_learner_class(_FakeBaseLearner)
    scorer = _FakeScorer()
    learner = cls(
        rl_cluster=_FakeCluster(),
        reward_fns=[],
        algo_config=object(),
        novelty_scorer=scorer,
        novelty_config=TunixNoveltyConfig(lambda_novelty=0.5, incorrect_novelty_scale=1.0),
    )
    rewards = learner._generate_and_compute_advantage(
        {"prompts": ["p1", "p2"], "answer": ["1", "2"]},
        mode="train",
    )
    assert rewards == [0.0, 0.0]
    assert learner.seen_rewards_kwargs["novelty"] == [0.25, 0.75]
    assert scorer.seen_rollout is learner.rl_cluster.generated


def test_learner_fails_loudly_when_required_novelty_missing() -> None:
    cls = build_tmx_dapo_learner_class(_FakeBaseLearner)
    learner = cls(
        rl_cluster=_FakeCluster(),
        reward_fns=[],
        algo_config=object(),
        novelty_scorer=None,
        novelty_config=TunixNoveltyConfig(lambda_novelty=0.5, incorrect_novelty_scale=1.0),
    )
    with pytest.raises(RuntimeError, match="novelty scorer"):
        learner._generate_and_compute_advantage(
            {"prompts": ["p1"], "answer": ["1"]},
            mode="train",
        )


def test_novelty_config_matches_resume_spec_contract() -> None:
    spec = load_experiment_spec("configs/tunix/tmx_dapo_drgrpo_resume_step100.json")
    cfg = novelty_config_from_algorithm(spec.algorithm)
    assert cfg.lambda_novelty == 0.5
    assert cfg.incorrect_novelty_scale == 1.0
    assert cfg.layers == (7, 14, 21)
    assert cfg.metric == "sqrt_mse"
    assert cfg.required is True
