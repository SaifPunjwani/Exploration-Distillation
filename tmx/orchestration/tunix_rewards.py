"""Tunix-compatible reward functions for the TMX paper pipeline.

Tunix's plugin loader (``tunix.cli.config.obtain_reward_fn``) imports a module
by dotted path and collects every top-level function whose ``__module__``
matches the import. Each function must match the contract::

    def reward(prompts, completions, **kwargs) -> list[float]

with ``prompts`` and ``completions`` of equal length and the return a
``list[float]`` of the same length. Extra dataset columns (e.g. ``answer``)
and rollout extras (e.g. ``novelty``) are forwarded as kwargs by
``tunix/rl/reward_manager.py``.

We delegate the correctness check to ``reward_plugins.is_correct`` so the
TMX-internal scoring path and the Tunix path share the same answer
extractor. We avoid re-exposing ``is_correct`` here at the top level
because Tunix would try to call it as a reward fn with the wrong signature.

Three reward fns are exposed (Tunix sums them):
  - ``correctness``           +1 if correct, 0 otherwise
  - ``incorrectness_penalty`` -1 if incorrect, 0 otherwise
  - ``novelty_reward``        λ · novelty per sample (full DAPO+novelty contract)
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from .reward_plugins import is_correct as _is_correct


def correctness(
    prompts: Sequence[str],
    completions: Sequence[str],
    answer: Sequence[str] | None = None,
    **_: object,
) -> list[float]:
    """+1 if the boxed/final answer matches ground truth, 0 otherwise.

    Tunix sums reward fns. This is the +1 component; the matching -1 for
    incorrect lives in ``incorrectness_penalty`` so an operator can ablate
    the asymmetric penalty without rewriting the extractor.

    ``answer`` comes from the dataset's ground-truth column. The Tunix
    ``SequenceRewardManager`` forwards extra dataset columns by name.
    """
    if answer is None:
        raise ValueError("correctness requires the dataset's `answer` column")
    if len(prompts) != len(completions) or len(completions) != len(answer):
        raise ValueError(
            f"length mismatch: prompts={len(prompts)} completions={len(completions)} "
            f"answer={len(answer)}"
        )
    return [1.0 if _is_correct(c, a) else 0.0 for c, a in zip(completions, answer, strict=True)]


def incorrectness_penalty(
    prompts: Sequence[str],
    completions: Sequence[str],
    answer: Sequence[str] | None = None,
    **_: object,
) -> list[float]:
    """-1 if incorrect / unparseable, 0 otherwise.

    Combined with ``correctness``, the per-sample reward is +1 / -1, the
    DAPO contract value (``reward_correct=1.0``, ``reward_incorrect=-1.0``).

    DAPO's overlong reward shaping is built into ``DAPOConfig`` itself
    (``tunix.rl.grpo.dapo_learner``), so this module deliberately does NOT
    add a length penalty.
    """
    if answer is None:
        raise ValueError("incorrectness_penalty requires the dataset's `answer` column")
    if len(prompts) != len(completions) or len(completions) != len(answer):
        raise ValueError(
            f"length mismatch: prompts={len(prompts)} completions={len(completions)} "
            f"answer={len(answer)}"
        )
    return [0.0 if _is_correct(c, a) else -1.0 for c, a in zip(completions, answer, strict=True)]


def novelty_reward(
    prompts: Sequence[str],
    completions: Sequence[str],
    novelty: Sequence[float] | None = None,
    answer: Sequence[str] | None = None,
    **_: object,
) -> list[float]:
    """λ · novelty per sample, with optional ``incorrect_novelty_scale``.

    ``novelty`` is forwarded by ``NoveltyAwareVllmRollout`` as a per-sample
    list[float] in the rollout extras. If novelty is None (e.g., running
    the CLI path without our custom rollout), this returns zeros — the
    correctness component still computes correctly without it.

    The blending coefficients ``λ`` and ``incorrect_novelty_scale`` come
    from env (set by ``tunix_workload.py`` from the spec at startup):

      TMX_LAMBDA_NOVELTY            default 0.0
      TMX_INCORRECT_NOVELTY_SCALE   default 1.0

    Contract per ``tmx_jax/rnd.blend_rewards``:
      novelty_used = novelty           if correct
                   = scale · novelty   otherwise
      reward      += λ · novelty_used
    """
    if novelty is None:
        return [0.0] * len(completions)
    if len(novelty) != len(completions):
        raise ValueError(
            f"novelty length {len(novelty)} != completions length {len(completions)}"
        )

    lam = float(os.environ.get("TMX_LAMBDA_NOVELTY", "0.0"))
    if lam == 0.0:
        return [0.0] * len(completions)

    scale = float(os.environ.get("TMX_INCORRECT_NOVELTY_SCALE", "1.0"))

    if answer is not None and len(answer) == len(completions):
        out: list[float] = []
        for c, n, a in zip(completions, novelty, answer, strict=True):
            n_used = float(n) if _is_correct(c, a) else scale * float(n)
            out.append(lam * n_used)
        return out

    # No answer column → can't apply incorrect_novelty_scale per-sample;
    # fall back to plain λ·novelty (equivalent to scale=1 for all samples).
    return [lam * float(n) for n in novelty]
