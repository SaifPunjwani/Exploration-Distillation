from __future__ import annotations

import inspect

import pytest

from tmx.orchestration import tunix_rewards


def _public_top_level_fns(module) -> list[str]:
    return [
        name
        for name, m in inspect.getmembers(module, inspect.isfunction)
        if m.__module__ == module.__name__ and not name.startswith("_")
    ]


def test_only_three_public_top_level_fns() -> None:
    """Tunix's plugin loader collects every public top-level fn whose
    __module__ matches. We expose exactly three:
      correctness, incorrectness_penalty, novelty_reward."""
    fns = _public_top_level_fns(tunix_rewards)
    assert set(fns) == {"correctness", "incorrectness_penalty", "novelty_reward"}, fns


def test_correctness_signature_matches_tunix_contract() -> None:
    sig = inspect.signature(tunix_rewards.correctness)
    params = list(sig.parameters)
    assert params[0] == "prompts"
    assert params[1] == "completions"
    # accepts arbitrary kwargs (Tunix forwards dataset columns by name)
    assert any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def test_correctness_returns_floats_one_per_completion() -> None:
    out = tunix_rewards.correctness(
        prompts=["p1", "p2", "p3"],
        completions=[r"\boxed{42}", r"\boxed{0}", "no answer here"],
        answer=["42", "1", "5"],
    )
    assert out == [1.0, 0.0, 0.0]
    assert all(isinstance(x, float) for x in out)


def test_incorrectness_penalty_returns_neg_or_zero() -> None:
    out = tunix_rewards.incorrectness_penalty(
        prompts=["p1", "p2"],
        completions=[r"\boxed{42}", r"\boxed{99}"],
        answer=["42", "1"],
    )
    assert out == [0.0, -1.0]


def test_correctness_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        tunix_rewards.correctness(
            prompts=["p"], completions=[r"\boxed{1}", r"\boxed{2}"], answer=["1", "2"]
        )


def test_correctness_requires_answer() -> None:
    with pytest.raises(ValueError, match="answer"):
        tunix_rewards.correctness(prompts=["p"], completions=["c"])


def test_combined_reward_is_plus_one_or_minus_one() -> None:
    """correctness + incorrectness_penalty == DAPO contract +1/-1 per sample."""
    prompts = ["a", "b"]
    completions = [r"\boxed{42}", r"\boxed{0}"]
    answer = ["42", "1"]
    plus = tunix_rewards.correctness(prompts, completions, answer=answer)
    minus = tunix_rewards.incorrectness_penalty(prompts, completions, answer=answer)
    combined = [a + b for a, b in zip(plus, minus, strict=True)]
    assert combined == [1.0, -1.0]
