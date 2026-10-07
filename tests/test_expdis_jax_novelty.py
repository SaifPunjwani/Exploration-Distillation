"""RND novelty scorer invariants: novelty before the predictor update, and the
predictor's Adam state persisting across updates within one Explorer while a
fresh ``init_rnd_map`` (new Explorer or round) resets it."""

import unittest

try:
    import jax  # noqa: F401
    import jax.numpy as jnp
    import numpy as np
    from expdis_jax.novelty import (
        RNDBlock,
        init_rnd_map,
        is_adam_opt_state,
        novelty_reward,
        score_and_update_rnd_map,
    )
    _HAVE_JAX = True
except Exception:  # pragma: no cover - environment without JAX deps
    _HAVE_JAX = False


@unittest.skipUnless(_HAVE_JAX, "jax/flax/optax not installed")
class RndPredictorStateTests(unittest.TestCase):
    LAYERS = (7,)
    INPUT_DIM = 8
    HIDDEN = 16

    def _features(self, seed: int):
        rng = np.random.default_rng(seed)
        return {"layer_7": jnp.asarray(rng.normal(size=(4, self.INPUT_DIM)).astype(np.float32))}

    def _update(self, rnd_map, seed: int):
        return score_and_update_rnd_map(
            rnd_map,
            self._features(seed),
            hidden_dim=self.HIDDEN,
            lr=1e-4,
            metric="sqrt_mse",
        )

    def test_initial_state_is_canonical_adam(self):
        rnd_map = init_rnd_map(rng=0, layers=self.LAYERS, input_dim=self.INPUT_DIM, hidden_dim=self.HIDDEN)
        state = rnd_map["layer_7"]["opt_state"]
        self.assertTrue(is_adam_opt_state(state))
        self.assertEqual(int(state[0].count), 0)
        self.assertFalse(is_adam_opt_state(None))
        self.assertFalse(is_adam_opt_state(()))

    def test_adam_moments_persist_across_updates_within_an_explorer(self):
        rnd_map = init_rnd_map(rng=0, layers=self.LAYERS, input_dim=self.INPUT_DIM, hidden_dim=self.HIDDEN)
        after_one, _ = self._update(rnd_map, seed=1)
        state_one = after_one["layer_7"]["opt_state"]
        self.assertTrue(is_adam_opt_state(state_one))
        self.assertEqual(int(state_one[0].count), 1)
        mu_one = np.asarray(state_one[0].mu["fc1"]["kernel"])
        self.assertGreater(float(np.abs(mu_one).max()), 0.0, "first update must populate the Adam moments")

        after_two, _ = self._update(after_one, seed=2)
        state_two = after_two["layer_7"]["opt_state"]
        self.assertEqual(int(state_two[0].count), 2, "the second update must continue the same Adam state")
        mu_two = np.asarray(state_two[0].mu["fc1"]["kernel"])
        self.assertFalse(np.allclose(mu_one, mu_two), "moments must accumulate, not restart from zero")
        # The frozen target never changes.
        np.testing.assert_array_equal(
            np.asarray(rnd_map["layer_7"]["target"]["fc1"]["kernel"]),
            np.asarray(after_two["layer_7"]["target"]["fc1"]["kernel"]),
        )

    def test_new_explorer_or_round_resets_predictor_and_optimizer(self):
        rnd_map = init_rnd_map(rng=0, layers=self.LAYERS, input_dim=self.INPUT_DIM, hidden_dim=self.HIDDEN)
        after_two, _ = self._update(*self._update(rnd_map, seed=1)[:1], seed=2)
        fresh = init_rnd_map(rng=0, layers=self.LAYERS, input_dim=self.INPUT_DIM, hidden_dim=self.HIDDEN)
        self.assertEqual(int(fresh["layer_7"]["opt_state"][0].count), 0)
        self.assertEqual(float(np.abs(np.asarray(fresh["layer_7"]["opt_state"][0].mu["fc1"]["kernel"])).max()), 0.0)
        self.assertGreater(int(after_two["layer_7"]["opt_state"][0].count), 0)

    def test_novelty_is_computed_before_the_predictor_update(self):
        rnd_map = init_rnd_map(rng=0, layers=self.LAYERS, input_dim=self.INPUT_DIM, hidden_dim=self.HIDDEN)
        feats = self._features(seed=3)
        block = RNDBlock(hidden_dim=self.HIDDEN, out_dim=self.HIDDEN)
        expected = novelty_reward(
            rnd_map["layer_7"]["target"],
            rnd_map["layer_7"]["predictor"],
            block.apply,
            block.apply,
            feats["layer_7"],
            "sqrt_mse",
        )
        updated, per_layer = score_and_update_rnd_map(
            rnd_map, feats, hidden_dim=self.HIDDEN, lr=1e-4, metric="sqrt_mse"
        )
        np.testing.assert_allclose(per_layer["layer_7"], np.asarray(expected), rtol=1e-5, atol=1e-6)
        self.assertFalse(
            np.allclose(
                np.asarray(updated["layer_7"]["predictor"]["fc1"]["kernel"]),
                np.asarray(rnd_map["layer_7"]["predictor"]["fc1"]["kernel"]),
            ),
            "the predictor must be updated after scoring",
        )


if __name__ == "__main__":
    unittest.main()
