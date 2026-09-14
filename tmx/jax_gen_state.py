"""Shared registry for JAX-native in-process generation.

Train.py registers the trainer's model + a callable returning current params +
tokenizer at startup. generate.batched_generate reads from this registry when
TMX_USE_JAX_GEN=1 is set, dispatching to jax_generate_batched (pad-to-max,
no KV cache) or jax_generate_kv (KV cache, faster).

This decouples generate.py (which doesn't know about the model) from train.py
(which owns it) without forcing a hard import dependency.
"""
from typing import Any, Callable, Optional

_REGISTERED: dict = {}


def register(*, cfg: Any, model: Any, params_ref: Callable[[], Any], tokenizer: Any, mesh: Any = None) -> None:
    """Register the trainer state for in-process JAX generation.

    Args:
      cfg: Qwen3Config
      model: Flax Qwen3Model instance (no params)
      params_ref: zero-arg callable that returns current params each rollout
                  (must reflect updates from training steps)
      tokenizer: HF AutoTokenizer
      mesh: jax.sharding.Mesh — used to shard rollout inputs across the FSDP
            axis so the per-chip hidden state stays within HBM.
    """
    _REGISTERED["cfg"] = cfg
    _REGISTERED["model"] = model
    _REGISTERED["params_ref"] = params_ref
    _REGISTERED["tokenizer"] = tokenizer
    _REGISTERED["mesh"] = mesh


def get():
    """Return (cfg, model, params, tokenizer, mesh). Raises if not registered."""
    if not _REGISTERED:
        raise RuntimeError(
            "jax_gen_state not registered; train.py must call "
            "tmx_jax.jax_gen_state.register(...) before generation"
        )
    return (
        _REGISTERED["cfg"],
        _REGISTERED["model"],
        _REGISTERED["params_ref"](),
        _REGISTERED["tokenizer"],
        _REGISTERED.get("mesh"),
    )


def is_registered() -> bool:
    return bool(_REGISTERED)


__all__ = ["register", "get", "is_registered"]
