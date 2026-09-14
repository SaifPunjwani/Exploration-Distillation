from __future__ import annotations

import pytest

from tmx.orchestration.tunix_rewards import (
    correctness,
    incorrectness_penalty,
    novelty_reward,
)


@pytest.fixture(autouse=True)
def _clear_lambda_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TMX_LAMBDA_NOVELTY", raising=False)
    monkeypatch.delenv("TMX_INCORRECT_NOVELTY_SCALE", raising=False)


def test_novelty_reward_zero_when_lambda_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TMX_LAMBDA_NOVELTY", "0.0")
    out = novelty_reward(["p"], [r"\boxed{1}"], novelty=[0.5], answer=["1"])
    assert out == [0.0]


def test_novelty_reward_zero_when_novelty_missing() -> None:
    out = novelty_reward(["p"], ["c"], novelty=None)
    assert out == [0.0]


def test_novelty_reward_applies_lambda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TMX_LAMBDA_NOVELTY", "0.5")
    out = novelty_reward(
        ["p1", "p2"],
        [r"\boxed{1}", r"\boxed{2}"],
        novelty=[0.4, 0.8],
        answer=["1", "2"],
    )
    # Both correct, scale=1 by default → λ·novelty
    assert out == [0.2, 0.4]


def test_novelty_reward_applies_incorrect_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TMX_LAMBDA_NOVELTY", "1.0")
    monkeypatch.setenv("TMX_INCORRECT_NOVELTY_SCALE", "0.5")
    out = novelty_reward(
        ["p1", "p2"],
        [r"\boxed{1}", r"\boxed{99}"],  # second is wrong
        novelty=[0.4, 0.4],
        answer=["1", "2"],
    )
    # Correct: λ·1·0.4 = 0.4. Incorrect: λ·0.5·0.4 = 0.2.
    assert out == [0.4, 0.2]


def test_novelty_reward_falls_back_when_no_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TMX_LAMBDA_NOVELTY", "0.5")
    out = novelty_reward(
        ["p1", "p2"],
        ["irrelevant", "irrelevant"],
        novelty=[0.2, 0.6],
        answer=None,
    )
    assert out == [0.1, 0.3]


def test_novelty_reward_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="novelty length"):
        novelty_reward(["p"], ["c"], novelty=[0.1, 0.2])


def test_full_dapo_contract_with_novelty_sums_correctly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The three reward fns sum to the DAPO+novelty contract per sample.

    For correct sample: +1 (correctness) + 0 (penalty) + λ·1·novelty
    For incorrect:      +0          + (-1)             + λ·scale·novelty
    """
    monkeypatch.setenv("TMX_LAMBDA_NOVELTY", "0.5")
    monkeypatch.setenv("TMX_INCORRECT_NOVELTY_SCALE", "1.0")
    prompts = ["p1", "p2"]
    completions = [r"\boxed{42}", r"\boxed{99}"]
    answer = ["42", "1"]
    novelty = [0.4, 0.6]
    a = correctness(prompts, completions, answer=answer)
    b = incorrectness_penalty(prompts, completions, answer=answer)
    c = novelty_reward(prompts, completions, novelty=novelty, answer=answer)
    total = [x + y + z for x, y, z in zip(a, b, c, strict=True)]
    # correct: 1 + 0 + 0.5*1*0.4 = 1.20
    # wrong:   0 + (-1) + 0.5*1*0.6 = -0.70
    assert total == [1.2, -0.7]


def test_only_three_top_level_public_fns() -> None:
    """Tunix's plugin loader collects every public top-level fn whose
    __module__ matches. Three fns by design:
      correctness, incorrectness_penalty, novelty_reward."""
    import inspect

    from tmx.orchestration import tunix_rewards as mod

    fns = sorted(
        name
        for name, m in inspect.getmembers(mod, inspect.isfunction)
        if m.__module__ == mod.__name__ and not name.startswith("_")
    )
    assert fns == ["correctness", "incorrectness_penalty", "novelty_reward"]
