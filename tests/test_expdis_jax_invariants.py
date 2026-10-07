"""Tests for the frozen Dr.GRPO + DAPO method invariants in expdis_jax.

Three tiers by dependency weight:
- PureRewardingTests: expdis_jax.rewarding is deliberately pure Python — runs anywhere.
- ContractValidationTests: expdis_jax.config imports the Flax model module — gated on flax.
- GrpoKernelTests: exact advantage/loss math — gated on jax.
"""

import os
import unittest

from expdis_jax.rewarding import (
    REWARDING_CONTRACT_VERSION,
    extract_presented_answer,
    is_correct_completion,
)

try:
    import jax  # noqa: F401
    import jax.numpy as jnp
    from expdis_jax.grpo import GrpoConfig, compute_group_advantages, dr_grpo_loss_from_logps
    _HAVE_JAX = True
except Exception:  # pragma: no cover - environment without JAX deps
    _HAVE_JAX = False

try:
    from expdis_jax.config import TrainConfig, validate_contract
    from expdis_jax.pipeline import (
        _validate_experimental_mrme_gate,
        _validate_multihost_stage_boundary,
    )
    _HAVE_CONFIG = True
except Exception:  # pragma: no cover - config pulls in the Flax model module
    _HAVE_CONFIG = False


class PureRewardingTests(unittest.TestCase):
    """The exact-rational verifier behind every correctness reward."""

    def test_contract_version_pinned(self):
        self.assertEqual(REWARDING_CONTRACT_VERSION, "expdis_jax.rewarding@gpu_repro_v2_exact_rational")

    def test_boxed_integer_correct(self):
        self.assertTrue(is_correct_completion(r"Thus \boxed{204}.", "204"))

    def test_boxed_wrong_answer(self):
        self.assertFalse(is_correct_completion(r"Thus \boxed{203}.", "204"))

    def test_missing_answer_never_correct(self):
        self.assertFalse(is_correct_completion("I could not finish the solution.", "204"))

    def test_exact_rational_equality_fraction_vs_decimal(self):
        # gpu_repro_v2_exact_rational: 1/2 and 0.5 are the same rational.
        self.assertTrue(is_correct_completion(r"\boxed{1/2}", "0.5"))
        self.assertTrue(is_correct_completion(r"\boxed{0.5}", "1/2"))
        self.assertFalse(is_correct_completion(r"\boxed{1/3}", "0.3333"))

    def test_comma_grouping(self):
        self.assertTrue(is_correct_completion(r"\boxed{1,024}", "1024"))

    def test_latex_fraction(self):
        self.assertTrue(is_correct_completion(r"\boxed{\frac{3}{4}}", "0.75"))

    def test_extract_presented_answer_takes_boxed(self):
        self.assertEqual(extract_presented_answer(r"maybe 7? no: \boxed{42}"), "42")


@unittest.skipUnless(_HAVE_CONFIG, "flax/jax not installed (expdis_jax.config imports the model module)")
class ContractValidationTests(unittest.TestCase):
    """validate_contract must fail closed on every frozen-kernel deviation."""

    def setUp(self):
        self._env = dict(os.environ)
        os.environ["EXPDIS_DAPO_DRGRPO_FULL_CONTRACT"] = "1"

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)

    def _assert_rejects(self, pattern: str, **overrides):
        from dataclasses import replace
        cfg = replace(TrainConfig(), **overrides)
        with self.assertRaisesRegex(Exception, pattern):
            validate_contract(cfg)

    def test_default_32k_contract_is_self_consistent(self):
        cfg = TrainConfig()
        validate_contract(cfg, require_eval_contract=True)
        self.assertEqual(cfg.max_prompt_len, 2048)
        self.assertEqual(cfg.max_completion_len, 32768)
        self.assertEqual(cfg.max_total_len, 34816)
        self.assertEqual(
            cfg.soft_overlong_expected_len + cfg.soft_overlong_cache_len,
            cfg.max_completion_len,
        )
        self.assertEqual(cfg.grpo_batch_size * cfg.grpo_grad_accum, 4)
        self.assertEqual(
            cfg.grpo_batch_size * cfg.grpo_grad_accum * cfg.grpo_num_generations,
            64,
        )
        # Dr.GRPO denominator = the completion budget (paper), never the realized length.
        self.assertEqual(cfg.dr_grpo_loss_normalizer, 32768)
        self.assertEqual(cfg.dr_grpo_loss_normalizer, cfg.max_completion_len)
        self.assertEqual(cfg.incorrect_novelty_scale, 0.0)
        self.assertEqual(cfg.save_every_steps, 50)
        self.assertEqual(cfg.grpo_max_steps, 200)
        self.assertEqual(cfg.actual_online_grpo_max_steps, 100)
        self.assertEqual(cfg.scouts_per_round, 1)
        self.assertEqual(cfg.num_rounds, 1)
        self.assertEqual(cfg.round_budget_mode, "split")

    def test_rejects_wrong_clip_epsilon(self):
        self._assert_rejects("grpo_clip_epsilon=0.3", grpo_clip_epsilon=0.3)

    def test_rejects_wrong_clip_epsilon_high(self):
        self._assert_rejects("grpo_clip_epsilon_high=0.5", grpo_clip_epsilon_high=0.5)

    def test_rejects_nonzero_kl(self):
        self._assert_rejects("grpo_kl_beta=0.1", grpo_kl_beta=0.1)

    def test_rejects_std_advantage_normalization(self):
        # Dr.GRPO: advantage = reward - group_mean, NO std division.
        self._assert_rejects("no std division", grpo_advantage_normalization="std")

    def test_rejects_disabled_dynamic_sampling(self):
        self._assert_rejects("dynamic_sampling", dynamic_sampling=False)

    def test_rejects_disabled_overlong_filtering(self):
        self._assert_rejects("Overlong Filtering", mask_truncated_completions=False)

    def test_rejects_stale_16k_total_length(self):
        self._assert_rejects("max_total_len=18432", max_total_len=18432)

    def test_rejects_loss_normalizer_drift(self):
        # The historical 16,384 constant is rejected under the paper contract.
        self._assert_rejects(
            "dr_grpo_loss_normalizer=16384",
            dr_grpo_loss_normalizer=16384,
        )

    def test_rejects_ungated_novelty(self):
        # Novelty is credited only on verifier-correct completions.
        self._assert_rejects("incorrect_novelty_scale=1.0", incorrect_novelty_scale=1.0)

    def test_rejects_wrong_reward_values(self):
        self._assert_rejects("reward_correct=0.5", reward_correct=0.5)
        self._assert_rejects("reward_incorrect=0.0", reward_incorrect=0.0)

    def test_single_round_single_scout_is_not_experimental(self):
        _validate_experimental_mrme_gate(TrainConfig())

    def test_mrme_supports_fixed_budget_rounds(self):
        from dataclasses import replace

        cfg = replace(
            TrainConfig(),
            pipeline_mode="multi_round",
            num_rounds=4,
            scouts_per_round=3,
            round_novelty_schedule="0.75,0.50,0.35,0.25",
        )
        _validate_experimental_mrme_gate(cfg)

    def test_experimental_mrme_requires_every_update_reload(self):
        from dataclasses import replace

        cfg = replace(TrainConfig(), scouts_per_round=3)
        os.environ["EXPDIS_ENABLE_EXPERIMENTAL_TPU_MRME"] = "1"
        os.environ["EXPDIS_TRAIN_VLLM_RELOAD_EVERY_STEPS"] = "50"
        with self.assertRaisesRegex(RuntimeError, "RELOAD_EVERY_STEPS=1"):
            _validate_experimental_mrme_gate(cfg)
        # Every-batch synchronization is the default, so an unset variable passes.
        os.environ.pop("EXPDIS_TRAIN_VLLM_RELOAD_EVERY_STEPS", None)
        _validate_experimental_mrme_gate(cfg)
        os.environ["EXPDIS_TRAIN_VLLM_RELOAD_EVERY_STEPS"] = "1"
        _validate_experimental_mrme_gate(cfg)

    def test_multihost_full_pipeline_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "STOP_AFTER_EXPLORER=1"):
            _validate_multihost_stage_boundary(4)
        os.environ["EXPDIS_PIPELINE_STOP_AFTER_EXPLORER"] = "1"
        _validate_multihost_stage_boundary(4)


@unittest.skipUnless(_HAVE_JAX, "jax not installed")
class GrpoKernelTests(unittest.TestCase):
    """Exact math of the frozen kernel."""

    def test_advantages_are_mean_centered_without_std_division(self):
        rewards = jnp.asarray([1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0], dtype=jnp.float32)
        adv = compute_group_advantages(rewards, num_generations=4, normalization="none")
        # group 1 mean=-0.5 -> [1.5,-0.5,-0.5,-0.5]; group 2 mean=0.5 -> [0.5,0.5,0.5,-1.5]
        expected = [1.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, -1.5]
        for got, want in zip(adv.tolist(), expected):
            self.assertAlmostEqual(got, want, places=6)

    def test_std_normalization_is_a_distinct_opt_in_branch(self):
        # Pick a group whose sample standard deviation is not 1.0; the prior
        # [1,-1,-1,-1] fixture made the two branches nearly equal by accident.
        rewards = jnp.asarray([2.0, -1.0, -1.0, -1.0], dtype=jnp.float32)
        plain = compute_group_advantages(rewards, num_generations=4, normalization="none")
        scaled = compute_group_advantages(rewards, num_generations=4, normalization="std")
        # If the default silently divided by std these would match; they must not.
        self.assertNotAlmostEqual(float(plain[0]), float(scaled[0]), places=3)

    def test_dr_grpo_fixed_denominator_ignores_token_count(self):
        cfg = GrpoConfig(max_completion_len=8, loss_aggregation="dr_grpo")
        logps = jnp.zeros((2, 4), dtype=jnp.float32)  # new == old -> ratio == 1
        adv = jnp.asarray([1.0, 0.5], dtype=jnp.float32)
        for mask_tokens in (1, 4):
            mask = jnp.zeros((2, 4), dtype=jnp.float32).at[:, :mask_tokens].set(1.0)
            loss, _ = dr_grpo_loss_from_logps(logps, mask, logps, adv, cfg)
            manual = -(1.0 + 0.5) * mask_tokens / (2 * 8)  # sum(adv*mask) / (B * max_completion_len)
            self.assertAlmostEqual(float(loss), manual, places=6,
                                   msg=f"denominator must be B*max_completion_len (mask_tokens={mask_tokens})")

    def test_dr_grpo_normalizer_is_independent_of_generation_cap(self):
        cfg = GrpoConfig(
            max_completion_len=16,
            loss_normalizer=8,
            loss_aggregation="dr_grpo",
        )
        logps = jnp.zeros((2, 4), dtype=jnp.float32)
        adv = jnp.asarray([1.0, 0.5], dtype=jnp.float32)
        mask = jnp.ones((2, 4), dtype=jnp.float32)
        loss, _ = dr_grpo_loss_from_logps(logps, mask, logps, adv, cfg)
        self.assertAlmostEqual(float(loss), -(1.0 + 0.5) * 4 / (2 * 8), places=6)


if __name__ == "__main__":
    unittest.main()
