"""Per-stage LR schedule tests for expdis_jax.

The pure-math tests import only ``expdis_jax.lr_schedules`` (no JAX required) and
pin the exact LR sequences: the DAPO ``linear_warmup_constant`` convention
(1-indexed update n uses peak * min(n / warmup, 1), matching
``tmx_gpu.grpo_gpu.learning_rate_for_update`` on the `gpu` branch), warmup
boundaries and off-by-one behavior, per-stage independence, and fail-closed
validation. The optax-parity tests (skip-gated on jax/optax) check that the
schedule callable built by ``expdis_jax.train._lr_schedule`` evaluates to exactly
``lr_at_update`` at count = n - 1, so logging/manifests and the optimizer can
never disagree.
"""

import unittest
from types import SimpleNamespace

from expdis_jax.lr_schedules import (
    LR_SCHEDULE_MODES,
    lr_at_update,
    stage_lr_fields,
    validate_lr_schedule,
)

try:
    import jax  # noqa: F401
    import optax  # noqa: F401
    from expdis_jax.train import _lr_schedule
    _HAVE_JAX = True
except Exception:  # pragma: no cover - environment without JAX deps
    _HAVE_JAX = False


def _cfg(**kw):
    base = {}
    for prefix in ("grpo", "actual_grpo", "distill"):
        base[f"{prefix}_lr_schedule"] = "constant"
        base[f"{prefix}_lr_warmup_steps"] = 0
        base[f"{prefix}_lr_decay_steps"] = 0
        base[f"{prefix}_lr_min_ratio"] = 0.0
    base.update(kw)
    return SimpleNamespace(**base)


def _gpu_reference_lr(peak_lr, update_step, *, schedule, warmup_steps):
    """tmx_gpu.grpo_gpu.learning_rate_for_update, replicated verbatim as the
    cross-branch reference (the gpu branch is not importable from main)."""
    if schedule == "constant":
        return float(peak_lr)
    return float(peak_lr) * min(float(update_step) / float(warmup_steps), 1.0)


class PureScheduleMathTests(unittest.TestCase):
    PEAK = 1e-6

    def test_constant_is_peak_at_every_update(self):
        cfg = _cfg()
        for n in (1, 2, 20, 200, 10_000):
            self.assertEqual(lr_at_update(cfg, self.PEAK, n, stage="grpo"), self.PEAK)

    def test_dapo_warmup_exact_sequence_scout_profile(self):
        # Codex's proposed Scout profile: warmup 20 of 200 steps, peak 1e-6.
        cfg = _cfg(grpo_lr_schedule="linear_warmup_constant", grpo_lr_warmup_steps=20)
        self.assertAlmostEqual(lr_at_update(cfg, self.PEAK, 1, stage="grpo"), 5e-8, places=18)
        self.assertAlmostEqual(lr_at_update(cfg, self.PEAK, 10, stage="grpo"), 5e-7, places=18)
        self.assertAlmostEqual(
            lr_at_update(cfg, self.PEAK, 19, stage="grpo"), 0.95 * self.PEAK, places=18
        )
        self.assertEqual(lr_at_update(cfg, self.PEAK, 20, stage="grpo"), self.PEAK)
        self.assertEqual(lr_at_update(cfg, self.PEAK, 21, stage="grpo"), self.PEAK)
        self.assertEqual(lr_at_update(cfg, self.PEAK, 200, stage="grpo"), self.PEAK)

    def test_dapo_warmup_off_by_one_first_update_is_never_zero(self):
        # Update 1 must get peak/warmup (verl convention), NOT 0 — a zero first
        # LR would silently waste the first rollout batch.
        cfg = _cfg(grpo_lr_schedule="linear_warmup_constant", grpo_lr_warmup_steps=20)
        first = lr_at_update(cfg, self.PEAK, 1, stage="grpo")
        self.assertGreater(first, 0.0)
        self.assertAlmostEqual(first, self.PEAK / 20.0, places=18)

    def test_matches_gpu_implementation_convention_exactly(self):
        cfg = _cfg(grpo_lr_schedule="linear_warmup_constant", grpo_lr_warmup_steps=20)
        for n in range(1, 60):
            self.assertEqual(
                lr_at_update(cfg, self.PEAK, n, stage="grpo"),
                _gpu_reference_lr(
                    self.PEAK, n, schedule="linear_warmup_constant", warmup_steps=20
                ),
                f"divergence from tmx_gpu convention at update {n}",
            )

    def test_central_short_warmup_profile(self):
        # Codex's Central profile: 2-3 warmup steps of a 25-step stage.
        cfg = _cfg(
            actual_grpo_lr_schedule="linear_warmup_constant",
            actual_grpo_lr_warmup_steps=3,
        )
        self.assertAlmostEqual(
            lr_at_update(cfg, self.PEAK, 1, stage="actual_grpo"), self.PEAK / 3.0, places=18
        )
        self.assertEqual(lr_at_update(cfg, self.PEAK, 3, stage="actual_grpo"), self.PEAK)
        self.assertEqual(lr_at_update(cfg, self.PEAK, 25, stage="actual_grpo"), self.PEAK)

    def test_stages_are_independent(self):
        # Scout warmed up; Central and SFT stay constant — reading a stage must
        # never leak another stage's settings.
        cfg = _cfg(grpo_lr_schedule="linear_warmup_constant", grpo_lr_warmup_steps=20)
        self.assertAlmostEqual(lr_at_update(cfg, self.PEAK, 1, stage="grpo"), 5e-8, places=18)
        self.assertEqual(lr_at_update(cfg, self.PEAK, 1, stage="actual_grpo"), self.PEAK)
        self.assertEqual(lr_at_update(cfg, 5e-6, 1, stage="distill"), 5e-6)
        self.assertEqual(
            stage_lr_fields(cfg, "actual_grpo"), ("constant", 0, 0, 0.0)
        )

    def test_purity_gives_resume_parity(self):
        # Crash-resume correctness: the LR at update n is a pure function of n,
        # so a run restored at update 12 reproduces the uninterrupted sequence.
        cfg = _cfg(grpo_lr_schedule="linear_warmup_constant", grpo_lr_warmup_steps=20)
        uninterrupted = [lr_at_update(cfg, self.PEAK, n, stage="grpo") for n in range(1, 41)]
        resumed = [lr_at_update(cfg, self.PEAK, n, stage="grpo") for n in range(12, 41)]
        self.assertEqual(uninterrupted[11:], resumed)

    def test_warmup_cosine_boundaries(self):
        cfg = _cfg(
            grpo_lr_schedule="warmup_cosine",
            grpo_lr_warmup_steps=5,
            grpo_lr_decay_steps=100,
            grpo_lr_min_ratio=0.1,
        )
        # optax convention: count = n - 1 → peak lands on update 6 (count 5).
        self.assertAlmostEqual(
            lr_at_update(cfg, self.PEAK, 6, stage="grpo"), self.PEAK, places=15
        )
        self.assertAlmostEqual(
            lr_at_update(cfg, self.PEAK, 101, stage="grpo"), 0.1 * self.PEAK, places=15
        )
        self.assertAlmostEqual(
            lr_at_update(cfg, self.PEAK, 10_000, stage="grpo"), 0.1 * self.PEAK, places=15
        )

    def test_fail_closed_validation(self):
        cases = [
            dict(grpo_lr_schedule="cyclical"),
            dict(grpo_lr_schedule="linear_warmup_constant"),  # warmup missing
            dict(grpo_lr_schedule="constant", grpo_lr_warmup_steps=10),
            dict(grpo_lr_schedule="cosine"),  # decay missing
            dict(grpo_lr_schedule="warmup_cosine", grpo_lr_decay_steps=50),  # warmup missing
            dict(
                grpo_lr_schedule="warmup_cosine",
                grpo_lr_warmup_steps=50,
                grpo_lr_decay_steps=50,  # decay must exceed warmup
            ),
        ]
        for kw in cases:
            with self.assertRaises(ValueError, msg=f"case {kw} must fail closed"):
                lr_at_update(_cfg(**kw), self.PEAK, 1, stage="grpo")
        with self.assertRaises(ValueError):
            lr_at_update(_cfg(), self.PEAK, 0, stage="grpo")  # updates are 1-indexed
        with self.assertRaises(ValueError):
            lr_at_update(_cfg(), 0.0, 1, stage="grpo")  # peak must be > 0
        with self.assertRaises(ValueError):
            stage_lr_fields(_cfg(), "rnd")  # unknown stage

    def test_mode_registry_is_frozen(self):
        self.assertEqual(
            LR_SCHEDULE_MODES,
            ("constant", "linear_warmup_constant", "cosine", "warmup_cosine"),
        )
        for kind in LR_SCHEDULE_MODES:
            validate_lr_schedule(kind, 10 if kind != "constant" else 0, 100, stage="grpo")


@unittest.skipUnless(_HAVE_JAX, "jax/optax not installed")
class OptaxCallableParityTests(unittest.TestCase):
    """The optimizer's schedule (evaluated at optax count = n - 1) must equal
    the pure reference used for logging and manifests at update n."""

    PEAK = 1e-6

    def test_constant_returns_plain_float(self):
        # Frozen-contract behavior: default must be the untouched float, not a
        # callable, so existing optimizer construction is byte-identical.
        out = _lr_schedule(_cfg(), self.PEAK, stage="grpo")
        self.assertIsInstance(out, float)
        self.assertEqual(out, self.PEAK)

    def test_dapo_warmup_callable_matches_pure_reference(self):
        cfg = _cfg(grpo_lr_schedule="linear_warmup_constant", grpo_lr_warmup_steps=20)
        sched = _lr_schedule(cfg, self.PEAK, stage="grpo")
        for n in (1, 2, 10, 19, 20, 21, 50, 200):
            self.assertAlmostEqual(
                float(sched(n - 1)),
                lr_at_update(cfg, self.PEAK, n, stage="grpo"),
                places=12,
                msg=f"optax/logging divergence at update {n}",
            )

    def test_distill_stage_callable_matches_pure_reference(self):
        cfg = _cfg(distill_lr_schedule="linear_warmup_constant", distill_lr_warmup_steps=4)
        sched = _lr_schedule(cfg, 5e-6, stage="distill")
        for n in (1, 2, 4, 5, 8):
            self.assertAlmostEqual(
                float(sched(n - 1)),
                lr_at_update(cfg, 5e-6, n, stage="distill"),
                places=12,
            )

    def test_warmup_cosine_callable_matches_pure_reference(self):
        cfg = _cfg(
            grpo_lr_schedule="warmup_cosine",
            grpo_lr_warmup_steps=5,
            grpo_lr_decay_steps=100,
            grpo_lr_min_ratio=0.1,
        )
        sched = _lr_schedule(cfg, self.PEAK, stage="grpo")
        for n in (1, 3, 5, 6, 30, 100, 101, 500):
            self.assertAlmostEqual(
                float(sched(n - 1)),
                lr_at_update(cfg, self.PEAK, n, stage="grpo"),
                places=12,
                msg=f"cosine divergence at update {n}",
            )


if __name__ == "__main__":
    unittest.main()
