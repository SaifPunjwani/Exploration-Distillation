"""Tunix custom rollout that attaches RND novelty per trajectory.

Tunix's reward callback contract is ``(prompts, completions, **kwargs) -> list[float]``
— no hidden states, no model params. To get RND novelty into the reward
signal we wrap Tunix's ``VllmRollout`` and attach a per-trajectory ``novelty``
field that the reward fn reads from ``**kwargs``.

Flow per rollout batch:
  1. Run the parent vLLM rollout to get completions.
  2. Concatenate prompt + completion token IDs, build attention masks.
  3. Forward the actor model with ``return_pooled_layers=(7,14,21)`` to get
     per-sample features (mean-pooled within each layer, averaged across
     layers).
  4. Run RND state's ``score_and_update``: compute novelty per sample, take
     an Adam step on the predictor.
  5. Attach ``novelty`` (B,) to the rollout output's metadata so the Tunix
     reward manager forwards it to ``novelty_reward`` as a kwarg.
  6. Periodically mirror RND state to HF (so resuming an Explorer run
     continues from the same predictor weights).

This module is the Tunix-internal hook. It depends on Tunix's exact rollout
class hierarchy (``tunix.rl.rollout.vllm_rollout.VllmRollout``) which is
pinned in our docker image. If Tunix bumps and the parent class signature
changes, the smoke gate will fail loudly.

Construction is deferred to ``build_novelty_rollout(...)`` so the module is
importable without Tunix installed (matters for offline tests / CI).
"""

from __future__ import annotations

from typing import Any

from .novelty_state import NoveltyConfig, NoveltyState, create_novelty_state, score_and_update


def build_novelty_rollout(
    *,
    actor_apply: Any,
    actor_params_getter: Any,
    novelty_config: NoveltyConfig,
    base_rollout_cls: Any | None = None,
    rng_seed: int = 0,
):
    """Construct a Tunix-compatible novelty-aware rollout instance.

    Imports of ``tunix.rl.rollout`` are deferred to call time so this module
    stays importable in CPU/CI environments where Tunix isn't installed.

    Parameters
    ----------
    actor_apply:
        Callable matching ``model.apply`` for the actor. Decoupled from the
        module so we can shard via ``jax.jit`` / ``shard_map`` outside.
    actor_params_getter:
        ``Callable[[], dict]`` that returns the latest actor params. Wrapped
        rather than passed by value because Tunix updates the params each
        train step and we want fresh params on every rollout.
    novelty_config:
        Static config for the RND scorer.
    base_rollout_cls:
        Optional override of Tunix's ``VllmRollout`` (e.g. for testing). If
        None, the live Tunix class is imported.
    rng_seed:
        Seed for the initial RND target/predictor MLP weights.
    """
    import jax

    from .novelty_features import extract_features, pad_token_ids

    if base_rollout_cls is None:
        # Deferred import — only required at construction time.
        from tunix.rl.rollout.vllm_rollout import VllmRollout as _VllmRollout
        base_rollout_cls = _VllmRollout

    state = create_novelty_state(jax.random.PRNGKey(rng_seed), novelty_config)

    class _NoveltyAwareVllmRollout(base_rollout_cls):  # type: ignore[misc, valid-type]
        """vLLM rollout with RND novelty attached per trajectory."""

        # Mutable carry — Tunix wraps the rollout in jit-free code, so we can
        # safely mutate this in __call__. State persists across rollout
        # batches (online RND).
        _novelty_state: NoveltyState = state
        _batch_counter: int = 0

        def _attach_novelty(
            self,
            outputs: Any,
            prompt_token_ids: list[list[int]],
            completion_token_ids: list[list[int]],
        ) -> Any:
            """Compute novelty for the batch and stash it on `outputs`."""
            cls = type(self)

            # Concatenate prompt + completion per sample for the actor forward.
            joined = [
                list(p) + list(c)
                for p, c in zip(prompt_token_ids, completion_token_ids, strict=True)
            ]
            token_ids, mask = pad_token_ids(joined, pad_id=0)
            features = extract_features(
                actor_apply,
                actor_params_getter(),
                token_ids,
                mask,
                layers=tuple(novelty_config.layers),
            )
            new_state, novelty = score_and_update(cls._novelty_state, features)
            cls._novelty_state = new_state
            cls._batch_counter += 1

            # Attach to whatever container Tunix gave us.
            if hasattr(outputs, "extras") and isinstance(outputs.extras, dict):
                outputs.extras["novelty"] = list(map(float, novelty.tolist()))
            elif hasattr(outputs, "metadata") and isinstance(outputs.metadata, dict):
                outputs.metadata["novelty"] = list(map(float, novelty.tolist()))
            else:
                # Fall back to setting an attribute; Tunix will surface unknown
                # attributes as kwargs to reward fns per reward_manager auto-injection.
                outputs.novelty = list(map(float, novelty.tolist()))
            return outputs

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            outputs = super().__call__(*args, **kwargs)
            try:
                prompt_ids = self._extract_prompt_token_ids(outputs, args, kwargs)
                completion_ids = self._extract_completion_token_ids(outputs)
            except (AttributeError, KeyError, TypeError) as exc:
                # Fail loudly: if we can't find token IDs, novelty is 0,
                # which silently drops a contract component. Better to crash.
                raise RuntimeError(
                    "novelty_rollout could not locate prompt/completion token IDs "
                    "on the parent VllmRollout output. Tunix may have bumped its "
                    "rollout API; update _extract_*_token_ids to match."
                ) from exc
            return self._attach_novelty(outputs, prompt_ids, completion_ids)

        @staticmethod
        def _extract_prompt_token_ids(
            outputs: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
        ) -> list[list[int]]:
            for source in (kwargs.get("prompt_token_ids"), kwargs.get("prompts")):
                if source is not None:
                    return [list(seq) for seq in source]
            for arg in args:
                if isinstance(arg, list | tuple) and arg and isinstance(arg[0], list | tuple):
                    return [list(seq) for seq in arg]
            if hasattr(outputs, "prompt_token_ids"):
                return [list(seq) for seq in outputs.prompt_token_ids]
            raise KeyError("prompt_token_ids not found in args/kwargs/outputs")

        @staticmethod
        def _extract_completion_token_ids(outputs: Any) -> list[list[int]]:
            for attr in ("completion_token_ids", "output_token_ids", "token_ids"):
                if hasattr(outputs, attr):
                    return [list(seq) for seq in getattr(outputs, attr)]
            raise KeyError("completion_token_ids not found on outputs")

        def current_novelty_state(self) -> NoveltyState:
            return type(self)._novelty_state

    return _NoveltyAwareVllmRollout


__all__ = ["build_novelty_rollout"]
