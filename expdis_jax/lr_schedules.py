"""Pure learning-rate schedule math shared by train/distill/pipeline and tests.

Dependency-free (stdlib ``math`` only) so exact LR sequences are testable
without JAX installed, and multihost consistency holds by construction: the
LR is a pure function of the 1-indexed optimizer-update count, never of
mutable scheduler state, microbatch counts, or discarded rollout groups.

``linear_warmup_constant`` follows the DAPO/verl convention and matches
``tmx_gpu.grpo_gpu.learning_rate_for_update`` exactly: 1-indexed update ``n``
uses ``peak * min(n / warmup_steps, 1)`` (update 1 gets ``peak/warmup``,
update ``warmup`` reaches the peak, later updates hold it).

The cosine modes mirror ``optax.warmup_cosine_decay_schedule`` evaluated at
``count = n - 1`` (optax counts completed updates, 0-based): linear 0 -> peak
over ``warmup`` counts, then cosine peak -> ``min_ratio * peak`` with
``decay_steps`` as the TOTAL horizon including warmup.
"""

from __future__ import annotations

import math

LR_SCHEDULE_MODES = ("constant", "linear_warmup_constant", "cosine", "warmup_cosine")

_STAGE_PREFIXES = {
    "grpo": "grpo",            # Scout / Explorer GRPO (pipeline maps the Central's
    "actual_grpo": "actual_grpo",  # actual_grpo_* fields onto grpo_* for stage 4)
    "distill": "distill",      # SFT distillation
}


def stage_lr_fields(cfg, stage: str = "grpo"):
    """Resolve (mode, warmup_steps, decay_steps, min_ratio) for a stage."""
    try:
        prefix = _STAGE_PREFIXES[stage]
    except KeyError:
        raise ValueError(
            f"unknown LR-schedule stage {stage!r}; expected one of {tuple(_STAGE_PREFIXES)}"
        ) from None
    kind = str(getattr(cfg, f"{prefix}_lr_schedule", "constant") or "constant").strip().lower()
    warmup = max(0, int(getattr(cfg, f"{prefix}_lr_warmup_steps", 0) or 0))
    decay = max(0, int(getattr(cfg, f"{prefix}_lr_decay_steps", 0) or 0))
    min_ratio = float(getattr(cfg, f"{prefix}_lr_min_ratio", 0.0) or 0.0)
    return kind, warmup, decay, min_ratio


def validate_lr_schedule(kind: str, warmup: int, decay: int, *, stage: str = "grpo") -> None:
    """Fail closed before any model/optimizer work on an incoherent schedule."""
    if kind not in LR_SCHEDULE_MODES:
        raise ValueError(
            f"[{stage}] unknown lr_schedule {kind!r}; expected " + " | ".join(LR_SCHEDULE_MODES)
        )
    if kind == "constant" and warmup != 0:
        raise ValueError(f"[{stage}] constant LR schedule requires warmup_steps=0; got {warmup}")
    if kind == "linear_warmup_constant" and warmup <= 0:
        raise ValueError(f"[{stage}] linear_warmup_constant requires lr_warmup_steps > 0")
    if kind in ("cosine", "warmup_cosine") and decay <= 0:
        raise ValueError(f"[{stage}] {kind} requires lr_decay_steps > 0 (total horizon)")
    if kind == "warmup_cosine" and warmup <= 0:
        raise ValueError(f"[{stage}] warmup_cosine requires lr_warmup_steps > 0")
    if kind == "warmup_cosine" and decay <= warmup:
        raise ValueError(
            f"[{stage}] warmup_cosine requires lr_decay_steps > lr_warmup_steps; "
            f"got decay={decay}, warmup={warmup}"
        )


def lr_at_update(cfg, peak_lr: float, update_step: int, *, stage: str = "grpo") -> float:
    """LR applied on the 1-indexed successful optimizer update ``update_step``.

    Pure reference used for logging, manifests, and tests; the optax schedule
    built by ``expdis_jax.train._lr_schedule`` evaluates to exactly these values
    at ``count = update_step - 1``.
    """
    kind, warmup, decay, min_ratio = stage_lr_fields(cfg, stage)
    validate_lr_schedule(kind, warmup, decay, stage=stage)
    peak = float(peak_lr)
    if not math.isfinite(peak) or peak <= 0.0:
        raise ValueError(f"[{stage}] peak learning rate must be finite and > 0; got {peak_lr!r}")
    n = int(update_step)
    if n < 1:
        raise ValueError(f"[{stage}] optimizer update_step must be >= 1; got {update_step!r}")
    if kind == "constant":
        return peak
    if kind == "linear_warmup_constant":
        return peak * min(float(n) / float(warmup), 1.0)
    # cosine modes: mirror optax.warmup_cosine_decay_schedule at count = n - 1.
    count = n - 1
    w = warmup if kind == "warmup_cosine" else 0
    if w > 0 and count < w:
        return peak * (float(count) / float(w))
    floor = min_ratio * peak
    denom = max(decay - w, 1)
    progress = min(max(float(count - w) / float(denom), 0.0), 1.0)
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))
